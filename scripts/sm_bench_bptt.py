"""C2 benchmark: one BPTT actor update through H imagined steps.

The deterministic graph-transformer actor (lib/sm/actor_critic.py) acts in
the compiled imagined env from replay-buffer start states for H steps; loss
= -mean survival-weighted return (critic bootstrap left out here); backward
into the actor. Reports per batch size: time of forward+backward, of the
actor alone, peak GPU memory, gradient norm, or OOM.

torch.compile specializes on the batch size, so each new --batches entry
compiles again (minutes each); the first full rollout per size is the
compile run and is reported separately.

Run with the training flags: Run N's obs (MYOHAND_OBS_ACT=1) plus the pad
force (MYOHAND_OBS_PADF=1), cap-free.
"""
import argparse
import os
import sys
import time

import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
from envs.mjwarp_myohand_env import MyoHandPourEnv  # noqa: E402
from lib.sm.actor_critic import OBS_KEYS, SMActor  # noqa: E402
from lib.sm.actuation import Actuation, params_from_env  # noqa: E402
from lib.sm.buffer import ReplayBuffer  # noqa: E402
from lib.sm.task import TorchTask  # noqa: E402
from lib.sm.wm_env import ImaginedEnv  # noqa: E402
from lib.sm.world_model import Dynamics, MyoPourRep, ObsHead  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", default="sm_data/wm_b2b_detach.pt")
    p.add_argument("--data", default="sm_data/b2b_train.pt")
    p.add_argument("--batches", default="1024")
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--softness", type=float, default=0.05)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--compile-actor", action="store_true")
    args = p.parse_args()
    dev = "cuda:0"
    H = args.H
    assert os.environ.get("MYOHAND_OBS_PADF") == "1", "run with MYOHAND_OBS_PADF=1 (training obs)"

    env = MyoHandPourEnv(num_envs=4, device=dev)
    env.reset()
    env.step(torch.zeros(4, env.action_space.shape[0], device=dev))
    task = TorchTask(env)
    if not args.no_compile:
        task.compile()
    actm = Actuation(params_from_env(env), dev)
    ck = torch.load(args.wm, weights_only=False, map_location=dev)
    rep = MyoPourRep(ck["layout"], dev)
    dyn = Dynamics(rep, 45, tuple(int(h) for h in ck["args"]["hidden"].split(","))).to(dev)
    dyn.load_state_dict(ck["model"])
    head = ObsHead(rep, u_dim=45).to(dev)
    head.load_state_dict(ck["head"])
    for m in (dyn, head):
        m.eval()
        m.requires_grad_(False)
    buf = ReplayBuffer.from_state_dict(torch.load(args.data, weights_only=False)["buffer"], device=dev)
    obs_dims = {k: v.shape[0] for k, v in env.observation_space.spaces.items()}
    print(f"obs dims {obs_dims}")
    actor = SMActor(obs_dims).to(dev)
    actor.grad_ckpt = args.grad_ckpt
    if args.compile_actor:
        actor._forward = torch.compile(actor._forward, dynamic=False)
    print(f"actor compiled {args.compile_actor}")
    print(f"actor grad checkpointing {actor.grad_ckpt}")
    print(f"actor params {sum(p.numel() for p in actor.parameters()) / 1e6:.2f} M")

    for B in (int(b) for b in args.batches.split(",")):
        ie = ImaginedEnv(dyn, head, actm, task, rep, buf, B, softness=args.softness,
                         generator=torch.Generator(device=dev).manual_seed(0))
        try:
            with torch.no_grad():
                o = ie.reset()
                actor.update_norm(o)
                for k in OBS_KEYS:
                    assert o[k].shape[-1] == obs_dims[k], f"{k}: imagined {o[k].shape[-1]} vs env {obs_dims[k]}"

            def update():
                start = ie.sample_start(B)
                o = ie.reset(start)
                c = torch.ones(B, device=dev)
                R = torch.zeros(B, device=dev)
                for _ in range(H):
                    o, r, d, _ = ie.step(actor(o))
                    c = c * (1 - d)
                    R = R + c * r
                actor.zero_grad(set_to_none=True)
                (-R.mean()).backward()
                return R.mean().item(), c.mean().item()

            torch.cuda.synchronize()
            t0 = time.time()
            update()
            torch.cuda.synchronize()
            t_first = time.time() - t0
            torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(3):
                t0 = time.time()
                Rm, cm = update()
                torch.cuda.synchronize()
                ts.append(time.time() - t0)
            gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in actor.parameters() if p.grad is not None))
            # actor alone, same batch, H calls
            o = ie.reset()
            torch.cuda.synchronize()
            t0 = time.time()
            out = sum(actor(o).sum() for _ in range(H))
            out.backward()
            torch.cuda.synchronize()
            ta = time.time() - t0
            print(f"batch {B:5d}: update {sorted(ts)[1]:.2f} s (first call incl. compile {t_first:.0f} s), actor alone x{H} {ta:.2f} s, "
                  f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB, grad norm {gn:.2e} finite {bool(torch.isfinite(gn))}, "
                  f"return {Rm:.1f}, survival {cm:.2f}")
        except torch.cuda.OutOfMemoryError:
            print(f"batch {B:5d}: OUT OF MEMORY")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
