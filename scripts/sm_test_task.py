"""A4 test: lib/sm/task.py against the live env, step by step.

Runs Run G (deterministic + a little of its PPO noise, so states vary) in
the real env. After every env.step, rebuilds from get_state (post-step state
and the previous state), the applied u, the frame and the real contact forces:
  - every reward term and the total (env total minus 0.5 * reward_power,
    the dropped term) vs info["reward_dict"]
  - the actor observation, block by block, vs the obs env.step returned
  - hard termination vs env.failure_buf_, plus the soft d on both sides
Then gradient sanity and cost at batch 1024.

Run with Run G's flags (cap-free); keep the termination tightening
(MYOHAND_TIGHTEN_METHOD unset = exp_decay) so termination is exercised.
"""
import argparse
import os
import sys
import time

import torch
import warp as wp

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
from envs.mjwarp_myohand_env import MyoHandPourEnv  # noqa: E402
from lib.rl.network_builder_myohand_transformer import MyoHandTransformerNetwork  # noqa: E402
from lib.sm.state import get_state  # noqa: E402
from lib.sm.task import TorchTask, scale_factor_at  # noqa: E402

RMS = "running_mean_std.running_mean_std"


def norm(x, sd, key, dev, eps=1e-5, clip=5.0):
    m = sd[f"{RMS}.{key}.running_mean"].to(dev).float()
    v = sd[f"{RMS}.{key}.running_var"].to(dev).float()
    return torch.clamp((x - m) / torch.sqrt(v + eps), -clip, clip)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="../physgraph-myosuite-mjwarp/eval_checkpoints/runG_latest.pth")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=700)
    p.add_argument("--sigma-scale", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda:0"
    E = args.num_envs

    env = MyoHandPourEnv(num_envs=E, device=dev)
    sd = torch.load(args.checkpoint, weights_only=False, map_location=dev)["model"]
    A = env.action_space.shape[0]
    net = MyoHandTransformerNetwork(params=None, actions_num=A).to(dev)
    net.load_state_dict({k[12:]: v for k, v in sd.items() if k.startswith("a2c_network.")}, strict=False)
    net.eval()
    sig = args.sigma_scale * torch.exp(sd["a2c_network.sigma"].float())

    obs = env.reset()
    env.step(torch.zeros(E, A, device=dev))  # builds the env's lazy reward tables
    obs = env.reset()
    task = TorchTask(env)
    tip_ids = torch.as_tensor(env.tip_body_ids, device=dev, dtype=torch.long)

    term_err, tot_err, n_rows = {}, 0.0, 0
    blk_err = {}
    agree = disagree = 0
    d_on, d_off = [], []
    ep_done = 0
    for t in range(args.steps):
        s_prev, _ = get_state(env)
        with torch.no_grad():
            o = obs["obs"]
            mu, _, _, _ = net({"obs": {k: norm(o[k], sd, k, dev) for k in ("proprioception", "privileged", "target")}})
            obs, r, d, info = env.step(torch.tanh(mu + sig * torch.randn_like(mu)))
            s_next, _ = get_state(env)
            frame = env.progress_buf.clone()
            u = torch.cat([env._wrist_force, env._wrist_torque, wp.to_torch(env.data.ctrl).float()], -1)
            cf = wp.to_torch(env.data.cfrc_ext)[:, tip_ids, 3:6].float()
            tip_force = torch.cat([cf, cf.norm(dim=-1, keepdim=True)], -1).reshape(E, 20)

            k_next, k_prev = task.kin(s_next), task.kin(s_prev)
            sf = scale_factor_at(env._tight_step)
            rr, T, zt, ze = task.reward(s_next, s_prev, u, frame, sf, k_next, k_prev)
            for key, v in T.items():
                term_err[key] = max(term_err.get(key, 0.0), (v - info["reward_dict"][key].float()).abs().max().item())
            env_tot = r.float() - 0.5 * info["reward_dict"]["reward_power"].float()
            tot_err = max(tot_err, (rr - env_tot).abs().max().item())
            soft, hard = task.termination(zt, ze, env.running_progress_buf)
            fb = env.failure_buf_
            agree += int((hard == fb).sum())
            disagree += int((hard != fb).sum())
            if fb.any():
                d_on.append(soft[fb])
            if (~fb).any():
                d_off.append(soft[~fb])

            ob = task.obs(s_next, s_prev, frame, tip_force, k_next, k_prev)
            for key in ob:
                ref = obs["obs"][key].float()
                assert ob[key].shape == ref.shape, f"{key}: {tuple(ob[key].shape)} vs env {tuple(ref.shape)}"
                err = (ob[key] - ref).abs()
                m, i = err.max(0)[0].max(0)
                if m.item() > blk_err.get(key, (0.0, 0))[0]:
                    blk_err[key] = (m.item(), int(i))
            n_rows += E
        done = env.success_buf_ | env.failure_buf_
        ep_done += int(done.sum())
        obs, _ = env.reset_done()

    print(f"\nchecked {n_rows} transitions ({ep_done} episode ends), scale_factor now {scale_factor_at(env._tight_step):.3f}")
    print("reward terms, max |torch - env|:")
    for key, v in term_err.items():
        print(f"  {key:24s} {v:.2e}")
    print(f"total reward (env minus 0.5*power) max |diff| {tot_err:.2e}")
    print("observation blocks, max |torch - env| (worst dim):")
    for key, (m, i) in blk_err.items():
        print(f"  {key:15s} {m:.2e}  (dim {i} of {obs['obs'][key].shape[-1]})")
    print(f"termination: hard agrees {agree}/{agree + disagree}")
    if d_on:
        print(f"  soft d where env failed: mean {torch.cat(d_on).mean():.3f} min {torch.cat(d_on).min():.3f}")
    if d_off:
        x = torch.cat(d_off)
        print(f"  soft d where env continued: mean {x.mean():.4f} max {x.max():.3f}  (>0.5: {(x > 0.5).float().mean():.4f})")

    # gradients and cost
    s_prev, _ = get_state(env)
    B = 1024
    idx = torch.randint(0, E, (B,), device=dev)
    sn = s_prev[idx].clone().requires_grad_(True)
    sp = s_prev[idx].clone()
    fr = env.progress_buf[idx].clamp(min=1)
    u = torch.zeros(B, 45, device=dev)
    rr, T, zt, ze = task.reward(sn, sp, u, fr, 1.0)
    soft, _ = task.termination(zt, ze, fr)
    ob = task.obs(sn, sp, fr, torch.zeros(B, 20, device=dev))
    loss = rr.sum() + soft.sum() + sum(v.sum() for v in ob.values())
    g, = torch.autograd.grad(loss, sn)
    hq = torch.as_tensor(env.dof_adrs, device=dev)
    print(f"gradient: finite {bool(torch.isfinite(g).all())}  hand-joint grads nonzero {(g[:, hq].abs() > 0).float().mean():.3f}")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        sn = s_prev[idx].clone().requires_grad_(True)
        rr, T, zt, ze = task.reward(sn, sp, u, fr, 1.0)
        ob = task.obs(sn, sp, fr, torch.zeros(B, 20, device=dev))
        (rr.sum() + sum(v.sum() for v in ob.values())).backward()
    torch.cuda.synchronize()
    print(f"cost: reward + obs forward+backward at batch {B}: {(time.time() - t0) / 5 * 1000:.0f} ms (includes 2 FK calls)")


if __name__ == "__main__":
    main()
