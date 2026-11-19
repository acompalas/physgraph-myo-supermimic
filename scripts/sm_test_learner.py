"""C3 test: the actor-critic learner (lib/sm/learner.py) on the imagined env.

Offline, with the fixed b2b world model and buffer (no online data), from a
fresh actor. Checks:
  1. max_step_reward bounds every imagined reward (V_max is valid)
  2. N updates run with finite losses and gradients; per-update time
  3. on a FIXED set of start states (same B, so no recompiles), the
     imagined per-step reward and end survival before vs after training
  4. the critic's value loss falls (first vs last quarter of updates)

What this does not show: anything about the real simulator. A policy that
improves in imagination can exploit world-model errors; that is checked in
D1/D2 with online data and real-env evaluation.

Run with the training flags (MYOHAND_OBS_ACT=1, MYOHAND_OBS_PADF=1, cap-free).
"""
import argparse
import os
import sys
import time

import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
from envs.mjwarp_myohand_env import MyoHandPourEnv  # noqa: E402
from lib.sm.actor_critic import SMActor  # noqa: E402
from lib.sm.actuation import Actuation, params_from_env  # noqa: E402
from lib.sm.buffer import ReplayBuffer  # noqa: E402
from lib.sm.learner import ActorCriticLearner, LearnerCfg, SMCriticWM, critic_input  # noqa: E402
from lib.sm.task import TorchTask  # noqa: E402
from lib.sm.wm_env import ImaginedEnv  # noqa: E402
from lib.sm.world_model import Dynamics, MyoPourRep, ObsHead  # noqa: E402


@torch.no_grad()
def evaluate(actor, ie, start, H):
    o = ie.reset(start)
    surv = torch.ones(ie.B, device=o["target"].device)
    R = torch.zeros_like(surv)
    for _ in range(H):
        o, r, d, _ = ie.step(actor(o))
        R = R + surv * r
        surv = surv * (1 - d)
    return (R / H).mean().item(), surv.mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", default="sm_data/wm_b2b_detach.pt")
    p.add_argument("--data", default="sm_data/b2b_train.pt")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--H", type=int, default=32)
    p.add_argument("--softness", type=float, default=0.05)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--compile-actor", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    dev = "cuda:0"
    assert os.environ.get("MYOHAND_OBS_PADF") == "1", "run with MYOHAND_OBS_PADF=1 (training obs)"
    torch.manual_seed(args.seed)

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
    buf = ReplayBuffer.from_state_dict(torch.load(args.data, weights_only=False)["buffer"], device=dev)
    obs_dims = {k: v.shape[0] for k, v in env.observation_space.spaces.items()}

    B, H = args.batch, args.H
    ie = ImaginedEnv(dyn, head, actm, task, rep, buf, B, softness=args.softness,
                     generator=torch.Generator(device=dev).manual_seed(args.seed))
    actor = SMActor(obs_dims).to(dev)
    actor.grad_ckpt = args.grad_ckpt
    if args.compile_actor:
        actor._forward = torch.compile(actor._forward, dynamic=False)

    # fixed eval starts (own generator), actor obs normalizer from buffer starts
    gen_eval = torch.Generator(device=dev).manual_seed(12345)
    ie.gen, gen_train = gen_eval, ie.gen
    eval_start = ie.sample_start(B)
    ie.gen = gen_train
    with torch.no_grad():
        o = ie.reset(eval_start)
        actor.update_norm(o)
        rep_dim = rep.features(ie.s).shape[-1]
        x = critic_input(ie, o)
    critic = SMCriticWM(rep_dim, obs_dims["target"], task.seq_len).to(dev)
    assert x.shape[-1] == rep_dim + obs_dims["target"] + 1
    cfg = LearnerCfg(batch_size=B, horizon=H)
    r_max = task.max_step_reward
    learner = ActorCriticLearner(actor, critic, ie, cfg, r_max)
    print(f"obs dims {obs_dims} | critic input {x.shape[-1]} (rep {rep_dim} + goal {obs_dims['target']} + phase)")
    print(f"actor {sum(p.numel() for p in actor.parameters()) / 1e6:.2f} M, critic {sum(p.numel() for p in critic.parameters()) / 1e3:.0f} k params | "
          f"max_step_reward {r_max:.2f}, V_max {learner.v_max:.0f} | B {B} H {H} softness {args.softness}")

    # 1. reward bound on imagined rewards (random-init actor)
    with torch.no_grad():
        o = ie.reset(eval_start)
        rmax_seen = 0.0
        for _ in range(H):
            o, r, d, _ = ie.step(actor(o))
            rmax_seen = max(rmax_seen, r.max().item())
    print(f"[1] max imagined reward {rmax_seen:.2f} <= max_step_reward {r_max:.2f}: {'PASS' if rmax_seen <= r_max + 1e-4 else 'FAIL'}")

    r0, s0 = evaluate(actor, ie, eval_start, H)
    print(f"[3] before: eval reward/step {r0:.3f}, end survival {s0:.3f}")

    hist, times = [], []
    for i in range(1, args.updates + 1):
        torch.cuda.synchronize()
        t0 = time.time()
        st = learner.update()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
        st = {k: float(v) for k, v in st.items()}
        hist.append(st)
        bad = [k for k, v in st.items() if not torch.isfinite(torch.tensor(v))]
        if bad:
            print(f"update {i}: NON-FINITE {bad} -> FAIL")
            return
        if i % args.log_every == 0 or i == 1:
            print(f"update {i:4d}: return {st['return']:8.2f} | r/step {st['reward_per_step']:.3f} | surv_end {st['survival_end']:.3f} | "
                  f"grad {st['grad_norm']:.2e} | V loss {st['value_loss']:.0f} V {st['value_mean']:.1f} / G {st['gae_return_mean']:.1f} EV {st['explained_var']:+.2f} | {times[-1]:.2f} s", flush=True)

    r1, s1 = evaluate(actor, ie, eval_start, H)
    q = max(1, len(hist) // 4)
    vl0 = sum(h["value_loss"] for h in hist[:q]) / q
    vl1 = sum(h["value_loss"] for h in hist[-q:]) / q
    steady = sorted(times[1:])[len(times[1:]) // 2] if len(times) > 1 else times[0]
    print(f"[2] {args.updates} updates finite: PASS | median update {steady:.2f} s (first {times[0]:.0f} s incl. compile) | "
          f"peak GPU {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"[3] after : eval reward/step {r1:.3f} ({r1 - r0:+.3f}), end survival {s1:.3f} ({s1 - s0:+.3f}): "
          f"{'IMPROVED' if r1 > r0 else 'NOT IMPROVED'}")
    ev0 = sum(h['explained_var'] for h in hist[:q]) / q
    ev1 = sum(h['explained_var'] for h in hist[-q:]) / q
    print(f"[4] critic explained variance first quarter {ev0:+.2f} -> last quarter {ev1:+.2f} (value loss {vl0:.0f} -> {vl1:.0f}): "
          f"{'TRACKING' if ev1 > max(ev0, 0.0) else 'NOT TRACKING'}")


if __name__ == "__main__":
    main()
