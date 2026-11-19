"""C1 test: the imagined env (lib/sm/wm_env.py) against the real env.

  1. open loop: from held-out buffer start rows, replay the LOGGED raw
     actions for H steps through imagination; compare the imagined reward
     per step and summed with the logged real reward (minus the dropped
     power term), and the imagined mug / joint state with the logged state.
  2. closed loop: Run G's policy acts in BOTH the real env (set to the same
     start states via set_state) and imagination, each from its own
     observations, for H steps; compare trajectories, rewards, terminations.
     This is how imagination is used in training.
  3. BPTT preview: gradient of the H-step imagined return w.r.t. the actions,
     per step (does it vanish or explode with depth?), plus time and peak
     memory at a few batch sizes.

World model: --wm (default the detached-head fit). Run with Run G's flags,
cap-free, tightening off (scale factor 1 on both sides).
"""
import argparse
import os
import sys
import time

import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
from envs.mjwarp_myohand_env import MyoHandPourEnv  # noqa: E402
from lib.rl.network_builder_myohand_transformer import MyoHandTransformerNetwork  # noqa: E402
from lib.sm.actuation import Actuation, params_from_env  # noqa: E402
from lib.sm.buffer import ReplayBuffer  # noqa: E402
from lib.sm.state import get_state, set_state  # noqa: E402
from lib.sm.task import TorchTask  # noqa: E402
from lib.sm.wm_env import ImaginedEnv  # noqa: E402
from lib.sm.world_model import Dynamics, MyoPourRep, ObsHead, quat_angle  # noqa: E402

RMS = "running_mean_std.running_mean_std"
KEYS = ("proprioception", "privileged", "target")


def norm(x, sd, key, dev, eps=1e-5, clip=5.0):
    m = sd[f"{RMS}.{key}.running_mean"].to(dev).float()
    v = sd[f"{RMS}.{key}.running_var"].to(dev).float()
    return torch.clamp((x - m) / torch.sqrt(v + eps), -clip, clip)


def start_from_window(buf, e, idx):
    """Start tuple for the first row of each sampled window (same rule as ImaginedEnv.sample_start)."""
    i = idx[:, 0]
    T = buf.buffer_size
    j = (i - 1) % T
    has_prev = (((i - buf._base()) % T) > 0) & ~(buf["terminated"] | buf["truncated"])[e, j]
    full = buf["state"][e, i]
    prev = torch.where(has_prev[:, None], buf["state"][e, j], full)
    z = torch.zeros(len(e), 20, device=full.device)
    tip = torch.where(has_prev[:, None], buf["next_tip_force"][e, j], z)
    pad = torch.where(has_prev[:, None], buf["next_pad_force"][e, j], z)
    return full, prev, buf["frame"][e, i], tip, pad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", default="sm_data/wm_b2b_detach.pt")
    p.add_argument("--data", default="sm_data/b2b_test.pt")
    p.add_argument("--checkpoint", default="../physgraph-myosuite-mjwarp/eval_checkpoints/runG_latest.pth")
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--n-open", type=int, default=2048)
    p.add_argument("--n-closed", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda:0"
    H = args.H

    env = MyoHandPourEnv(num_envs=args.n_closed, device=dev)
    A = env.action_space.shape[0]
    env.reset()
    env.step(torch.zeros(args.n_closed, A, device=dev))  # lazy reward tables
    env.reset()
    task = TorchTask(env)
    actm = Actuation(params_from_env(env), dev)

    ck = torch.load(args.wm, weights_only=False, map_location=dev)
    lay = ck["layout"]
    rep = MyoPourRep(lay, dev)
    hidden = tuple(int(h) for h in ck["args"]["hidden"].split(","))
    dyn = Dynamics(rep, 45, hidden).to(dev)
    dyn.load_state_dict(ck["model"])
    head = ObsHead(rep, u_dim=45 if ck["args"].get("obs_mode") == "head_detach" else 0).to(dev)
    head.load_state_dict(ck["head"])
    dyn.eval(), head.eval()
    for prm in list(dyn.parameters()) + list(head.parameters()):
        prm.requires_grad_(False)
    print(f"world model {args.wm} (obs_mode {ck['args'].get('obs_mode')}, step {ck['step']})")

    d = torch.load(args.data, weights_only=False)
    buf = ReplayBuffer.from_state_dict(d["buffer"], device=dev)
    names = d["meta"]["rterm_names"]
    i_pow = names.index("reward_power")

    # ---------- 1. open loop on logged actions ----------
    gen = torch.Generator(device=dev).manual_seed(1)
    e, idx = buf.sample_indices(H + 1, args.n_open, generator=gen)
    W = {k: buf[k][e[:, None], idx] for k in ("state", "action", "reward", "rterms", "frame", "terminated")}
    ienv = ImaginedEnv(dyn, head, actm, task, rep, buf, args.n_open)
    with torch.no_grad():
        ienv.reset(start_from_window(buf, e, idx))
        r_im, mug_err, joint_err = [], [], []
        for k in range(H):
            _, r, dsoft, info = ienv.step(W["action"][:, k])
            r_im.append(r)
            g = rep.from_full(W["state"][:, k + 1])
            mug_err.append((ienv.s["mp"] - g["mp"]).norm(dim=-1) * 100)
            joint_err.append(torch.rad2deg((ienv.s["qj"] - g["qj"]).abs()).mean(-1))
    r_im = torch.stack(r_im, 1)
    r_real = W["reward"][:, :H] - 0.5 * W["rterms"][:, :H, i_pow]
    f0 = W["frame"][:, 0]
    print(f"\n1. open loop, logged actions, {args.n_open} windows, real reward/step mean {r_real.mean():.2f}")
    for k in (1, 8, 16, H):
        rk = (r_im[:, k - 1] - r_real[:, k - 1]).abs()
        print(f"   step {k:2d}: |reward err| median {rk.median():.3f} p90 {torch.quantile(rk, 0.9):.3f}   "
              f"mug {mug_err[k - 1].median():.2f} cm   joints {joint_err[k - 1].median():.2f} deg")
    ret_err = (r_im.sum(1) - r_real.sum(1)).abs() / r_real.sum(1).abs()
    for name, lo, hi in (("approach", 0, 50), ("grasp+carry", 50, 373), ("pour", 373, 10_000)):
        m = (f0 >= lo) & (f0 < hi)
        if m.any():
            print(f"   {H}-step return rel. error, {name:12s}: median {ret_err[m].median():.3f}  p90 {torch.quantile(ret_err[m], 0.9):.3f}")

    # ---------- 2. closed loop with Run G ----------
    sd = torch.load(args.checkpoint, weights_only=False, map_location=dev)["model"]
    net = MyoHandTransformerNetwork(params=None, actions_num=A).to(dev)
    net.load_state_dict({k[12:]: v for k, v in sd.items() if k.startswith("a2c_network.")}, strict=False)
    net.eval()
    act = lambda o: net({"obs": {k: norm(o[k], sd, k, dev) for k in KEYS}})[0]  # noqa: E731
    N = args.n_closed
    gen = torch.Generator(device=dev).manual_seed(2)
    ienv_c = ImaginedEnv(dyn, head, actm, task, rep, buf, N, generator=gen)
    cand = ienv_c.sample_start(16 * N)
    keep = ((cand[2] >= 30) & (cand[2] < 560)).nonzero().squeeze(-1)[:N]  # mid-episode starts
    assert keep.numel() == N, "not enough mid-episode start rows"
    start = tuple(x[keep] for x in cand)
    full, prev, frame, tip, pad = start
    with torch.no_grad():
        o_im = ienv_c.reset(start)
        set_state(env, full, frame)
        env.running_progress_buf[:] = frame
        kp, kc = task.kin(prev), task.kin(full)
        env.prev_body_xpos = kc["joints_pos"].clone()
        env.current_joints_pos_ = kc["joints_pos"].clone()
        env.current_joints_vel_ = (kc["joints_pos"] - kp["joints_pos"]) / task.dt
        o_re = {k: v.clone() for k, v in o_im.items()}
        alive_im = torch.ones(N, device=dev)
        alive_re = torch.ones(N, dtype=torch.bool, device=dev)
        R_im = torch.zeros(N, device=dev)
        R_re = torch.zeros(N, device=dev)
        print(f"\n2. closed loop, Run G acting in both, {N} starts at frames {int(frame.min())}..{int(frame.max())}")
        for k in range(H):
            _, r_i, d_i, info = ienv_c.step(act(o_im))
            o_im = ienv_c.obs()
            o_re, r_r, _, inf_r = env.step(torch.tanh(act(o_re["obs"] if "obs" in o_re else o_re)))
            R_im += alive_im * r_i
            alive_im = alive_im * (1 - d_i)
            R_re += alive_re.float() * (r_r - 0.5 * inf_r["reward_dict"]["reward_power"])
            alive_re &= ~env.failure_buf_
            if k + 1 in (1, 8, 16, H):
                sr, _ = get_state(env)
                g = rep.from_full(sr)
                mug = (ienv_c.s["mp"] - g["mp"]).norm(dim=-1) * 100
                jd = torch.rad2deg((ienv_c.s["qj"] - g["qj"]).abs()).mean(-1)
                print(f"   step {k + 1:2d}: imagined vs real  mug {mug.median():.2f} cm (p90 {torch.quantile(mug, 0.9):.2f})  "
                      f"joints {jd.median():.2f} deg   alive imag(soft) {alive_im.mean():.2f} real {alive_re.float().mean():.2f}")
        rel = (R_im - R_re).abs() / R_re.abs().clamp(min=1e-6)
        print(f"   {H}-step return (survival-weighted): imag mean {R_im.mean():.1f}  real mean {R_re.mean():.1f}  "
              f"rel err median {rel.median():.3f}  corr {torch.corrcoef(torch.stack([R_im, R_re]))[0, 1]:.2f}")

    # ---------- 3. BPTT preview ----------
    # actions = Run G's own imagined actions (zero actions drop the mug at once,
    # the soft survival weight goes to 0 and every later gradient is trivially 0)
    print(f"\n3. BPTT through {H} imagined steps, actions = Run G's imagined actions as leaf tensors")
    for rep_i, B in enumerate((256, 256, 1024)):
        g3 = torch.Generator(device=dev).manual_seed(3)
        ie = ImaginedEnv(dyn, head, actm, task, rep, buf, B, generator=g3)
        cand = ie.sample_start(16 * B)
        keep = ((cand[2] >= 30) & (cand[2] < 560)).nonzero().squeeze(-1)[:B]
        start = tuple(x[keep] for x in cand)
        with torch.no_grad():
            o = ie.reset(start)
            acts = []
            for k in range(H):
                ak = act(o)
                acts.append(ak)
                o, _, _, _ = ie.step(ak)
        a = torch.stack(acts, 1).clone().requires_grad_(True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        ie.reset(start)
        c = torch.ones(B, device=dev)
        Rt = torch.zeros(B, device=dev)
        for k in range(H):
            _, r, dk, _ = ie.step(a[:, k])
            c = c * (1 - dk)
            Rt = Rt + c * r
        Rt.sum().backward()
        torch.cuda.synchronize()
        g = a.grad.norm(dim=-1).mean(0)
        tag = " (warm-up)" if rep_i == 0 else ""
        print(f"   batch {B}{tag}: {time.time() - t0:.2f} s fwd+bwd, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB, "
              f"grad finite {bool(torch.isfinite(a.grad).all())}, final survival c mean {c.mean():.2f}")
    print("   mean |dR/da_k| by step k: " + " ".join(f"k{k}:{g[k]:.2e}" for k in (0, 1, 4, 8, 16, 24, H - 1)))
    print(f"   by action group at k=0: wrist pos {a.grad[:, 0, 0:3].norm(dim=-1).mean():.2e}  "
          f"wrist rot {a.grad[:, 0, 3:6].norm(dim=-1).mean():.2e}  muscles {a.grad[:, 0, 9:].norm(dim=-1).mean():.2e}")

    # ---------- 4. termination margin in real data ----------
    with torch.no_grad():
        Sb = buf["state"][:, 1:buf.num_transitions]
        Sp = buf["state"][:, :buf.num_transitions - 1]
        fr = buf["frame"][:, 1:buf.num_transitions]
        okb = ~(buf["terminated"] | buf["truncated"])[:, :buf.num_transitions - 1] & (fr >= 8)
        Sb, Sp, fr = Sb[okb], Sp[okb], fr[okb]
        sel = torch.randperm(Sb.shape[0], device=dev)[:8192]
        _, _, zt, ze = task.reward(Sb[sel], Sp[sel], torch.zeros(len(sel), 45, device=dev), fr[sel], 1.0)
        z = torch.maximum(zt, ze)
        q = torch.quantile(z, torch.tensor([0.5, 0.9, 0.99], device=dev))
        print(f"\n4. normalized termination error z on real continuing steps (scale 1): median {q[0]:.2f} p90 {q[1]:.2f} p99 {q[2]:.2f}")
        for sft in (0.05, 0.02, 0.01):
            dd = torch.sigmoid((z - 1) / sft)
            print(f"   softness {sft}: mean per-step d {dd.mean():.4f} -> survival after {H} steps {(1 - dd).pow(H).mean():.2f}")

if __name__ == "__main__":
    main()
