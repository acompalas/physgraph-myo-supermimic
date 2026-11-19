"""C2 prep: where does one imagined step spend its time?

Times forward+backward of each component of ImaginedEnv.step at a given
batch (CUDA-synchronized, median of repeats, after warm-up):
  actuation (A2) | dynamics (B3) | force head | FK (A3), one call |
  reward (A4, FK given) | obs (A4, FK given) | full step
then the same full step with torch.compile on the FK, the reward and the
obs functions (--compile), to see how much is launch overhead.

Run with Run G's flags, cap-free.
"""
import argparse
import os
import sys
import time

import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
from envs.mjwarp_myohand_env import MyoHandPourEnv  # noqa: E402
from lib.sm.actuation import Actuation, params_from_env  # noqa: E402
from lib.sm.buffer import ReplayBuffer  # noqa: E402
from lib.sm.task import TorchTask  # noqa: E402
from lib.sm.wm_env import ImaginedEnv  # noqa: E402
from lib.sm.world_model import Dynamics, MyoPourRep, ObsHead  # noqa: E402


def timed(fn, reps=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", default="sm_data/wm_b2b_detach.pt")
    p.add_argument("--data", default="sm_data/b2b_test.pt")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()
    dev = "cuda:0"
    B = args.batch

    env = MyoHandPourEnv(num_envs=4, device=dev)
    env.reset()
    env.step(torch.zeros(4, env.action_space.shape[0], device=dev))
    task = TorchTask(env)
    actm = Actuation(params_from_env(env), dev)
    ck = torch.load(args.wm, weights_only=False, map_location=dev)
    rep = MyoPourRep(ck["layout"], dev)
    dyn = Dynamics(rep, 45, tuple(int(h) for h in ck["args"]["hidden"].split(","))).to(dev)
    dyn.load_state_dict(ck["model"])
    head = ObsHead(rep, u_dim=45).to(dev)
    head.load_state_dict(ck["head"])
    for m in (dyn, head):
        m.eval()
        for prm in m.parameters():
            prm.requires_grad_(False)
    buf = ReplayBuffer.from_state_dict(torch.load(args.data, weights_only=False)["buffer"], device=dev)
    ie = ImaginedEnv(dyn, head, actm, task, rep, buf, B, generator=torch.Generator(device=dev).manual_seed(0))
    start = ie.sample_start(B)
    full, prev, frame = start[0], start[1], start[2]
    a0 = torch.zeros(B, 48, device=dev)

    def f_act():
        a = a0.clone().requires_grad_(True)
        u, pid, act = actm(a, full[:, -18:], rep.from_full(full)["act"])
        (u.sum() + act.sum()).backward()

    s0 = rep.from_full(full)
    u0 = torch.zeros(B, 45, device=dev)

    def f_dyn():
        s = {k: v.clone().requires_grad_(True) for k, v in s0.items()}
        nxt = dyn(s, u0, s["act"])
        sum(v.sum() for v in nxt.values()).backward()

    def f_head():
        s = {k: v.clone().requires_grad_(True) for k, v in s0.items()}
        head(s, u0).sum().backward()

    def f_fk():
        q = full[:, :task.nq].clone().requires_grad_(True)
        xp, xq = task.fk(q)
        (xp.sum() + xq.sum()).backward()

    def f_rew():
        sn = full.clone().requires_grad_(True)
        k1, k0 = task.kin(sn), task.kin(prev)
        r, T, zt, ze = task.reward(sn, prev, u0, frame, 1.0, k1, k0)
        (r.sum() + zt.sum()).backward()

    def f_obs():
        sn = full.clone().requires_grad_(True)
        k1, k0 = task.kin(sn), task.kin(prev)
        o = task.obs(sn, prev, frame, torch.zeros(B, 20, device=dev), k1, k0)
        sum(v.sum() for v in o.values()).backward()

    def f_step():
        ie.reset(start)
        a = a0.clone().requires_grad_(True)
        o, r, d, _ = ie.step(a)
        (r.sum() + d.sum() + sum(v.sum() for v in o.values())).backward()

    print(f"batch {B}, forward+backward, median ms:")
    res = {}
    for name, fn in (("actuation", f_act), ("dynamics", f_dyn), ("force head", f_head), ("FK (1 call)", f_fk),
                     ("reward (+2 FK)", f_rew), ("obs (+2 FK)", f_obs), ("full step", f_step)):
        res[name] = timed(fn)
        print(f"  {name:16s} {res[name]:8.1f}")
    print(f"  -> 32-step rollout estimate: {32 * res['full step'] / 1000:.2f} s")

    if args.compile:
        print("\ncompiling FK, reward and obs with torch.compile (first calls are slow)...")
        t0 = time.time()
        fk_c = torch.compile(TorchTaskFK(task.fk), dynamic=False)
        task.fk = fk_c
        task.reward = torch.compile(task.reward, dynamic=False)
        task.obs = torch.compile(task.obs, dynamic=False)
        f_step()
        print(f"  compile + first step: {time.time() - t0:.1f} s")
        for name, fn in (("FK (1 call)", f_fk), ("full step", f_step)):
            t = timed(fn)
            print(f"  {name:16s} {t:8.1f}  (eager {res[name]:.1f})")
        print(f"  -> 32-step rollout estimate: {32 * timed(f_step) / 1000:.2f} s")


class TorchTaskFK(torch.nn.Module):
    """Module wrapper so torch.compile sees FK as one function; keeps site_xpos."""

    def __init__(self, fk):
        super().__init__()
        self.fk = fk
        self.site_body, self.site_pos = fk.site_body, fk.site_pos

    def forward(self, qpos):
        return self.fk(qpos)

    def site_xpos(self, xpos, xquat, site_ids):
        return self.fk.site_xpos(xpos, xquat, site_ids)


if __name__ == "__main__":
    main()
