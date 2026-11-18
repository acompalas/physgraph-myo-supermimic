"""A1 check: is our state definition complete (Markov)?

Copies the full state of envs [0, N/2) into envs [N/2, N), applies identical
actions to both halves for K steps, and reports how far the halves drift
apart, per state group. A complete state gives drift at float-noise level.

Negative controls (--omit pid | act | qvel) leave one group uncopied, to show
the check detects a missing piece.

Run from the repo root with the MYOHAND_* flags of the run being studied, e.g.
  MYOHAND_JQ_MODE=g MYOHAND_OBS_ACT=1 MYOHAND_MUG_PRESET=small \
    python scripts/sm_check_markov.py
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.mjwarp_myohand_env import MyoHandPourEnv  # noqa: E402
from lib.sm.state import copy_warmstart, get_state, layout_of, set_state  # noqa: E402


def state_groups(env, L):
    sl = L.slices()
    q0, v0 = sl["qpos"].start, sl["qvel"].start
    rng = lambda a, b: list(range(a, b))  # noqa: E731
    hand_q = rng(env.root_adr, env.root_adr + 7) + [int(i) for i in env.dof_adrs]
    hand_v = rng(env.root_dof_adr, env.root_dof_adr + 6) + [int(i) for i in env.dof_veladrs]
    g = {
        "hand_qpos": [q0 + i for i in hand_q],
        "src_mug_qpos": [q0 + i for i in rng(env.src_adr, env.src_adr + 7)],
        "dst_mug_qpos": [q0 + i for i in rng(env.dst_adr, env.dst_adr + 7)],
        "hand_qvel": [v0 + i for i in hand_v],
        "src_mug_qvel": [v0 + i for i in rng(env.src_dof_adr, env.src_dof_adr + 6)],
        "dst_mug_qvel": [v0 + i for i in rng(env.dst_dof_adr, env.dst_dof_adr + 6)],
        "act": rng(sl["act"].start, sl["act"].stop),
        "pid": rng(sl["pid"].start, sl["pid"].stop),
    }
    covered = sorted(i for v in g.values() for i in v)
    if covered != list(range(L.dim)):
        print(f"[WARN] groups cover {len(set(covered))} of {L.dim} state entries")
    return {k: torch.tensor(v, device=env.device) for k, v in g.items() if v}


def report(tag, full, src, dst, groups):
    diff = (full[dst] - full[src]).abs()
    cells = [f"{k}={diff[:, idx].max().item():.2e}" for k, idx in groups.items()]
    print(f"{tag:>8} | " + "  ".join(cells))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--warmup", type=int, default=30, help="random-action steps before copying")
    p.add_argument("--steps", type=int, default=20, help="identical-action steps after copying")
    p.add_argument("--noise", type=float, default=0.3, help="action std (actions are clamped to [-1,1])")
    p.add_argument("--omit", choices=["none", "pid", "act", "qvel"], default="none")
    p.add_argument("--no-warmstart", action="store_true", help="do not copy the solver warm start")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    N = args.num_envs
    assert N % 2 == 0, "--num-envs must be even"
    h = N // 2

    env = MyoHandPourEnv(num_envs=N, device="cuda:0", headless=True)
    A = env.action_space.shape[0]
    L = layout_of(env)
    groups = state_groups(env, L)
    print(f"[INFO] state layout: nq={L.nq} nv={L.nv} na={L.na} pid={L.npid} -> dim={L.dim}, actions={A}")

    env.reset()
    for _ in range(args.warmup):
        env.step((args.noise * torch.randn(N, A, device=env.device)).clamp(-1, 1))

    full, frame = get_state(env)
    src = torch.arange(h, device=env.device)
    dst = src + h

    new = full[src].clone()
    if args.omit != "none":
        sl = L.slices()[args.omit]
        new[:, sl] = full[dst][:, sl]  # leave this group as the dst envs had it
    set_state(env, new, frame[src], env_ids=dst)
    ws = False if args.no_warmstart else copy_warmstart(env, src, dst)
    print(f"[INFO] omit={args.omit}  warm start copied={ws}")

    scale = full[:, :].std(dim=0)
    print("[INFO] typical magnitude (std across envs) per group: " + "  ".join(
        f"{k}={scale[idx].mean().item():.2e}" for k, idx in groups.items()))

    report("copied", get_state(env)[0], src, dst, groups)
    for k in range(1, args.steps + 1):
        a = (args.noise * torch.randn(h, A, device=env.device)).clamp(-1, 1)
        env.step(torch.cat([a, a], dim=0))
        if k in (1, 2, 5, 10, args.steps):
            report(f"step {k}", get_state(env)[0], src, dst, groups)


if __name__ == "__main__":
    main()
