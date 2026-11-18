"""A3 test: lib/sm/fk.py against MuJoCo.

  1. CPU MuJoCo mj_kinematics on sampled logged states: every body's xpos and
     xquat, every site's site_xpos.
  2. The 20 tracked points the reward uses (env body_ids / site_ids, as in
     _compute_reward) on every logged transition vs the next_joints_pos labels
     that mujoco_warp produced during collection (row t label = state t+1).
  3. Gradient finiteness and the cost of forward+backward at batch 1024.

Needs the same MYOHAND_* flags as collection (builds a 1-env env).
"""
import argparse
import os
import sys
import time

import mujoco
import numpy as np
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
from lib.sm.buffer import ReplayBuffer  # noqa: E402
from lib.sm.fk import TorchFK  # noqa: E402
from lib.sm.world_model import quat_conj, quat_mul  # noqa: E402


def quat_angle(q1, q2):
    """Float32-accurate angle: 2 atan2(|vec(q1 q2*)|, |w|) (acos loses ~0.03 deg near 0)."""
    dq = quat_mul(q1, quat_conj(q2))
    return 2.0 * torch.atan2(dq[..., 1:].norm(dim=-1), dq[..., 0].abs())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="sm_data/collect_runG_noise1.0_E64_T1254.pt,sm_data/collect_runG_latest_E64_T1000.pt")
    p.add_argument("--n-cpu", type=int, default=500)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    dev = args.device

    from envs.mjwarp_myohand_env import MyoHandPourEnv
    env = MyoHandPourEnv(num_envs=1, device=dev)
    m = env.mj_model
    fk = TorchFK(m, dev)
    pt_body = torch.tensor(env.body_ids[1:], device=dev)
    pt_site = torch.tensor(env.site_ids[1:].clip(min=0), device=dev)
    pt_is_site = env.is_site_mask.to(dev)
    print(f"model: nbody {m.nbody} nsite {m.nsite} njnt {m.njnt} nq {m.nq}; joint types {sorted(set(m.jnt_type.tolist()))}; "
          f"tracked points {len(pt_body)} ({int(pt_is_site.sum())} sites)")

    d = mujoco.MjData(m)
    ok = True
    for path in args.data.split(","):
        sd = torch.load(path, weights_only=False)
        buf = ReplayBuffer.from_state_dict(sd["buffer"], device=dev)
        nq = sd["meta"]["layout"]["nq"]
        n = buf.num_transitions
        S = buf["state"][:, :n]
        print(f"{os.path.basename(path)}")

        # 1. vs CPU MuJoCo on sampled states
        flat = S.reshape(-1, S.shape[-1])
        idx = torch.randperm(flat.shape[0], device=dev)[:args.n_cpu]
        q = flat[idx, :nq]
        with torch.no_grad():
            xp, xq = fk(q)
            sp = fk.site_xpos(xp, xq, torch.arange(m.nsite, device=dev))
        e_pos, e_ang, e_site = 0.0, 0.0, 0.0
        for i in range(q.shape[0]):
            d.qpos[:] = q[i].double().cpu().numpy()
            mujoco.mj_kinematics(m, d)
            rp = torch.tensor(d.xpos, device=dev, dtype=torch.float32)
            rq = torch.tensor(d.xquat, device=dev, dtype=torch.float32)
            rs = torch.tensor(d.site_xpos, device=dev, dtype=torch.float32)
            e_pos = max(e_pos, (xp[i, 1:] - rp[1:]).norm(dim=-1).max().item())
            e_ang = max(e_ang, quat_angle(xq[i, 1:], rq[1:]).max().item())
            e_site = max(e_site, (sp[i] - rs).norm(dim=-1).max().item() if m.nsite else 0.0)
        print(f"  vs mj_kinematics ({q.shape[0]} states, all bodies/sites): max xpos err {e_pos:.2e} m  "
              f"max xquat err {np.degrees(e_ang):.2e} deg  max site_xpos err {e_site:.2e} m")
        ok &= e_pos < 1e-5 and e_site < 1e-5 and np.degrees(e_ang) < 1e-3

        # 2. tracked points vs mujoco_warp labels (row t label = joints at state t+1)
        done = (buf["terminated"] | buf["truncated"])[:, :n]
        keep = ~done[:, :-1]
        q1 = S[:, 1:][keep][:, :nq]
        lab = buf["next_joints_pos"][:, :n - 1][keep].reshape(-1, 20, 3)
        errs = []
        with torch.no_grad():
            for c in range(0, q1.shape[0], 8192):
                xp, xq = fk(q1[c:c + 8192])
                pts = torch.where(pt_is_site[None, :, None], fk.site_xpos(xp, xq, pt_site), xp[:, pt_body])
                errs.append((pts - lab[c:c + 8192]).norm(dim=-1))
        e = torch.cat(errs)
        print(f"  tracked points vs mujoco_warp labels ({e.shape[0]} transitions x 20): "
              f"max {e.max():.2e} m  p99 {torch.quantile(e.flatten()[:1_000_000], 0.99):.2e}  mean {e.mean():.2e}")
        ok &= e.max().item() < 1e-5

    # 3. gradients and cost
    q = flat[:1024, :nq].clone().requires_grad_(True)
    xp, xq = fk(q)
    pts = torch.where(pt_is_site[None, :, None], fk.site_xpos(xp, xq, pt_site), xp[:, pt_body])
    g, = torch.autograd.grad(pts.sum() + xq.sum(), q)
    hand = torch.tensor(env.dof_adrs, device=dev)
    print(f"grad: finite {bool(torch.isfinite(g).all())}  hand-joint grads nonzero {(g[:, hand].abs() > 0).float().mean():.3f}")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        q = flat[:1024, :nq].clone().requires_grad_(True)
        xp, xq = fk(q)
        pts = torch.where(pt_is_site[None, :, None], fk.site_xpos(xp, xq, pt_site), xp[:, pt_body])
        (pts.sum() + xq.sum()).backward()
    torch.cuda.synchronize()
    print(f"cost: forward+backward at batch 1024: {(time.time() - t0) / 10 * 1000:.1f} ms")
    print("ALL PASS" if ok else "SOME FAILED")


if __name__ == "__main__":
    main()
