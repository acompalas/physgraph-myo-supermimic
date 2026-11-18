"""B2: roll out a trained PPO checkpoint in the real mujoco-warp env and
write a SuperMimic replay buffer to disk, for the offline world-model fit.

Row t of env e (official _step_and_store convention):
  state      (142)  full pre-step state s_t  (lib/sm/state.py)
  frame      long   demo frame of s_t (progress_buf before the step)
  action     (48)   raw policy action a_t (mu + noise, before soft clamp)
  u          (45)   what the sim actually received: root force 3, root
                    torque 3 (post PID+EMA), muscle ctrl 39
  terminated bool   step t failed (failure_buf)
  truncated  bool   step t hit the episode end (success_buf)
  reward     float  r_t from the env
  rterms     (K)    env reward_dict terms, names in meta["rterm_names"]
  next_tip_force  (20)  PhysGraph tip_force obs after the step: cfrc_ext[distal, 3:6] + norm (world)
  next_pad_force  (20)  pad-side part of it, in each distal body frame + norm (lib/sm/contact.py)
  next_act_len/vel/force (na each)  mujoco_warp actuator_length/velocity/force after the step
  next_joints_pos (60)  MuJoCo positions of the 20 tracked points after the
                    step (label for later FK / world-model checks)

Action path matches training (lib/rl/base.py preprocess_actions):
soft clamp -1 + 2*sigmoid(2a) = tanh(a), so muscle ctrl = sigmoid(2a).
Use --hard-clamp to match live_eval instead.

The env must see the same MYOHAND_* flags as the checkpoint's run; they
are printed and saved in meta. Termination tightening starts loose in a
fresh env: set MYOHAND_TIGHTEN_METHOD=const to use the final factor.

Example:
  MYOHAND_...=... python scripts/sm_collect.py --checkpoint eval_checkpoints/X.pth \
      --num-envs 256 --steps 1000 --out sm_data/collect_X.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import warp as wp

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
from envs.mjwarp_myohand_env import MyoHandPourEnv  # noqa: E402
from lib.rl.network_builder_myohand_transformer import MyoHandTransformerNetwork  # noqa: E402
from lib.sm.buffer import FieldSpec, ReplayBuffer  # noqa: E402
from lib.sm.contact import PadForce  # noqa: E402
from lib.sm.state import PID_FIELDS, get_state, layout_of  # noqa: E402

RMS = "running_mean_std.running_mean_std"


def norm(x, sd, key, dev, eps=1e-5, clip=5.0):
    m = sd[f"{RMS}.{key}.running_mean"].to(dev).float()
    v = sd[f"{RMS}.{key}.running_var"].to(dev).float()
    return torch.clamp((x - m) / torch.sqrt(v + eps), -clip, clip)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=1000, help="rows per env (= buffer size)")
    p.add_argument("--noise", type=float, default=0.0, help="std of Gaussian noise added to mu")
    p.add_argument("--ckpt-sigma-scale", type=float, default=0.0,
                   help="if > 0, per-dim noise std = scale * exp(checkpoint a2c_network.sigma), i.e. PPO's own action std")
    p.add_argument("--hard-clamp", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = args.device

    flags = {k: v for k, v in sorted(os.environ.items()) if k.startswith("MYOHAND_")}
    print("MYOHAND flags:", flags if flags else "(none set, env defaults)")

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=dev)
    sd = ckpt["model"]
    env = MyoHandPourEnv(num_envs=args.num_envs, device=dev)
    A = env.action_space.shape[0]
    net = MyoHandTransformerNetwork(params=None, actions_num=A).to(dev)
    pre = "a2c_network."
    missing, unexpected = net.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=False)
    print(f"net loaded: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing (first 5):", missing[:5])
    net.eval()
    noise_std = torch.full((A,), args.noise, device=dev)
    if args.ckpt_sigma_scale > 0:
        noise_std = args.ckpt_sigma_scale * torch.exp(sd["a2c_network.sigma"].float().to(dev))
    print(f"collection noise std: mean {noise_std.mean():.3f} min {noise_std.min():.3f} max {noise_std.max():.3f}")

    L = layout_of(env)
    sl = L.slices()
    pid0 = sl["pid"].start
    force_sl = slice(pid0 + 3 * PID_FIELDS.index("_applied_force"), pid0 + 3 * PID_FIELDS.index("_applied_force") + 3)
    print(f"state layout: nq={L.nq} nv={L.nv} na={L.na} npid={L.npid} dim={L.dim}")

    obs = env.reset()
    nu = int(env.mj_model.nu)
    pad_force = PadForce(env)
    tip_ids = torch.as_tensor(env.tip_body_ids, device=dev, dtype=torch.long)
    cons_err, cons_ref = 0.0, 0.0
    buf, schema, rterm_names = None, None, None  # built on the first step, once reward_dict names are known

    ep_lens, ep_term = [], []
    ctrl_err = 0.0
    t0 = time.time()
    for t in range(args.steps):
        s, f = get_state(env)
        with torch.no_grad():
            o = obs["obs"]
            n_obs = {k: norm(o[k], sd, k, dev) for k in ("proprioception", "privileged", "target")}
            mu, _, _, _ = net({"obs": n_obs})
            a = mu + noise_std * torch.randn_like(mu) if bool((noise_std > 0).any()) else mu
            a_env = torch.clamp(a, -1, 1) if args.hard_clamp else torch.tanh(a)
            obs, r, d, info = env.step(a_env)
        ctrl = wp.to_torch(env.data.ctrl).float()
        u = torch.cat([env._wrist_force, env._wrist_torque, ctrl], -1)
        expect = (a_env[:, 9:] + 1.0) / 2.0
        ctrl_err = max(ctrl_err, (ctrl - expect).abs().max().item())
        if buf is None:
            rterm_names = list(info["reward_dict"].keys())
            schema = {
                "state": FieldSpec(L.dim),
                "frame": FieldSpec(dtype=torch.long),
                "action": FieldSpec(A),
                "u": FieldSpec(6 + nu),
                "terminated": FieldSpec(dtype=torch.bool),
                "truncated": FieldSpec(dtype=torch.bool),
                "reward": FieldSpec(),
                "rterms": FieldSpec(len(rterm_names)),
                "next_joints_pos": FieldSpec(60),
                "next_tip_force": FieldSpec(20),
                "next_pad_force": FieldSpec(20),
                "next_act_len": FieldSpec(nu),
                "next_act_vel": FieldSpec(nu),
                "next_act_force": FieldSpec(nu),
            }
            buf = ReplayBuffer(schema, args.steps, args.num_envs, device=dev)
        pad_obs, tot_world = pad_force(env)
        cf = wp.to_torch(env.data.cfrc_ext)[:, tip_ids, 3:6].float()
        tip_force = torch.cat([cf, cf.norm(dim=-1, keepdim=True)], -1).reshape(args.num_envs, 20)
        cons_err = max(cons_err, (tot_world - cf).norm(dim=-1).max().item())
        cons_ref = max(cons_ref, cf.norm(dim=-1).max().item())
        term = env.failure_buf_.clone()
        trunc = env.success_buf_.clone()
        buf.insert({
            "state": s, "frame": f, "action": a, "u": u,
            "terminated": term, "truncated": trunc, "reward": r.float(),
            "rterms": torch.stack([info["reward_dict"][k].float() for k in rterm_names], -1),
            "next_joints_pos": env.current_joints_pos_.reshape(args.num_envs, -1).float(),
            "next_tip_force": tip_force, "next_pad_force": pad_obs,
            "next_act_len": wp.to_torch(env.data.actuator_length).float(),
            "next_act_vel": wp.to_torch(env.data.actuator_velocity).float(),
            "next_act_force": wp.to_torch(env.data.actuator_force).float(),
        })
        done = term | trunc
        if done.any():
            ep_lens += env.running_progress_buf[done].tolist()
            ep_term += term[done].tolist()
        obs, _ = env.reset_done()
        if (t + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  step {t + 1}/{args.steps}  {(t + 1) * args.num_envs / el:.0f} rows/s  episodes done {len(ep_lens)}", flush=True)
    el = time.time() - t0

    # ---------- report ----------
    S = buf["state"]
    fr = buf["frame"]
    dn = buf["terminated"] | buf["truncated"]
    print(f"\ncollected {args.steps} x {args.num_envs} = {args.steps * args.num_envs} rows in {el:.1f} s "
          f"({args.steps * args.num_envs / el:.0f} rows/s)")
    print(f"max |ctrl - (a_env+1)/2| = {ctrl_err:.2e}  (expect 0)")
    # u root force at row t must equal _applied_force in state at row t+1 (same episode)
    nd = ~dn[:, :-1]
    fdiff = (buf["u"][:, :-1, 0:3] - S[:, 1:, force_sl]).abs()[nd]
    print(f"max |u_force[t] - state_pid_force[t+1]| (non-done rows) = {fdiff.max().item():.2e}  (expect 0)")
    print(f"contact extraction: max |sum of contacts - cfrc_ext| = {cons_err:.2e} N  (max |cfrc_ext| {cons_ref:.1f} N, expect ~0)")
    tf = buf["next_tip_force"].view(-1, 5, 4)[..., 3]
    pf = buf["next_pad_force"].view(-1, 5, 4)[..., 3]
    print("per finger (thumb..pinky): contact rate " + " ".join(f"{x:.2f}" for x in (tf > 1e-6).float().mean(0).tolist())
          + " | mean |tip| N " + " ".join(f"{x:.1f}" for x in tf.mean(0).tolist())
          + " | mean |pad| N " + " ".join(f"{x:.1f}" for x in pf.mean(0).tolist()))
    nan = {k: int((~torch.isfinite(buf[k].float())).sum()) for k in schema}
    print("non-finite counts:", {k: v for k, v in nan.items() if v} or "none")

    if ep_lens:
        el_t = torch.tensor(ep_lens, dtype=torch.float)
        n_term = int(sum(ep_term))
        print(f"episodes finished: {len(ep_lens)}  terminated {n_term}  truncated {len(ep_lens) - n_term}")
        q = torch.quantile(el_t, torch.tensor([0.1, 0.5, 0.9]))
        print(f"episode length: mean {el_t.mean():.0f}  p10 {q[0]:.0f}  p50 {q[1]:.0f}  p90 {q[2]:.0f}  max {el_t.max():.0f}  (demo {env.seq_len})")
    else:
        print("no episode finished")
    print(f"reward per step: mean {buf['reward'].mean():.3f}  std {buf['reward'].std():.3f}")

    edges = list(range(0, env.seq_len + 50, 50))
    h = torch.histc(fr.float(), bins=len(edges) - 1, min=0, max=edges[-1]).long().tolist()
    print("rows per demo-frame bin (frame coverage):")
    for i, c in enumerate(h):
        print(f"  [{edges[i]:3d},{edges[i + 1]:3d})  {c:8d}  {100 * c / fr.numel():5.1f}%")

    Lw = 8  # official wm.horizon: windows of 8 rows = 7 predicted transitions
    print(f"valid world-model windows ({Lw} rows): {int(buf.valid_starts(Lw).sum())}")

    out = args.out or os.path.join(repo_root, "sm_data",
                                   f"collect_{os.path.splitext(os.path.basename(args.checkpoint))[0]}_E{args.num_envs}_T{args.steps}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({
        "buffer": buf.state_dict(),
        "meta": {
            "checkpoint": args.checkpoint, "flags": flags, "args": vars(args),
            "layout": {"nq": L.nq, "nv": L.nv, "na": L.na, "npid": L.npid},
            "rterm_names": rterm_names, "seq_len": env.seq_len, "dt": env.dt,
        },
    }, out)
    print(f"saved {out}  ({os.path.getsize(out) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
