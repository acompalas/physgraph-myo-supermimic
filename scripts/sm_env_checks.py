"""Three env checks (read-only; builds a 4-env env, steps it briefly).

1. Activation reset: does reset_idx zero data.act? (code zeroes it only
   when MYOHAND_RANDOM_START=1). Step with random actions, reset, print act.
2. Eval clamp: training feeds the env tanh(mu) (rl_games soft clamp),
   live_eval feeds clamp(mu, -1, 1). Size of the difference on logged
   Run G actions, for wrist dims and for muscle ctrl u = (a + 1) / 2.
3. Finger self-collision: which pairs of hand bodies can generate contacts
   at all under MuJoCo's filter (contype/conaffinity bitmask, parent-child
   filter, <exclude> pairs), using the model the env builds.

Run with the same MYOHAND_* flags as collection.
"""
import os
import sys

import mujoco
import numpy as np
import torch
import warp as wp

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)


def main():
    from envs.mjwarp_myohand_env import MyoHandPourEnv
    env = MyoHandPourEnv(num_envs=4, device="cuda:0")
    m = env.mj_model

    # ---- 1. activation reset ----
    print("1. activation reset (MYOHAND_RANDOM_START =", os.environ.get("MYOHAND_RANDOM_START", "0") + ")")
    env.reset()
    act = wp.to_torch(env.data.act)
    print(f"   after first reset: max|act| {act.abs().max():.3f}")
    torch.manual_seed(0)
    for _ in range(20):
        env.step(torch.rand(4, env.action_space.shape[0], device="cuda:0") * 2 - 1)
    print(f"   after 20 random steps: mean act {act.mean():.3f}  max {act.max():.3f}")
    env.reset_idx(torch.arange(4, device="cuda:0"))
    print(f"   after reset_idx: mean act {act.mean():.3f}  max {act.max():.3f}  -> "
          f"{'NOT zeroed (bug)' if act.abs().max() > 1e-6 else 'zeroed'}")
    print(f"   PID EMA force after reset: {env._applied_force.abs().max():.3f} (zeroed in reset_idx)")

    # ---- 2. eval clamp ----
    print("\n2. soft (training) vs hard (live_eval) clamp on logged Run G actions")
    path = os.path.join(repo_root, "sm_data/b2b_test.pt")
    if os.path.exists(path):
        a = torch.load(path, weights_only=False)["buffer"]["storage"]["action"].reshape(-1, 48)
        soft, hard = torch.tanh(a), a.clamp(-1, 1)
        d = (soft - hard).abs()
        print(f"   |a| > 1 (saturated under hard clamp): wrist {(a[:, :6].abs() > 1).float().mean():.2f}  "
              f"muscles {(a[:, 9:].abs() > 1).float().mean():.2f}")
        print(f"   mean |tanh(a) - clamp(a)|: wrist {d[:, :6].mean():.3f}  muscles {d[:, 9:].mean():.3f}  "
              f"(muscle ctrl diff = half of that: {d[:, 9:].mean() / 2:.3f}, max {d[:, 9:].max() / 2:.3f})")
    else:
        print(f"   {path} not found, skipped")

    # ---- 3. finger self-collision ----
    print("\n3. hand body pairs that can collide (MuJoCo contact filter)")
    hand_root = int(m.body_rootid[int(env.tip_body_ids[0])])
    hand_bodies = [b for b in range(m.nbody) if int(m.body_rootid[b]) == hand_root]
    excl = {(int(min(a, b)), int(max(a, b))) for a, b in zip(m.exclude_signature >> 16, m.exclude_signature & 0xFFFF)} \
        if m.nexclude else set()
    filterparent = not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    geoms = {b: [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b] for b in hand_bodies}
    n_coll_geoms = sum(1 for b in hand_bodies for g in geoms[b] if m.geom_contype[g] or m.geom_conaffinity[g])
    print(f"   hand bodies {len(hand_bodies)}, hand geoms {sum(len(v) for v in geoms.values())}, "
          f"with nonzero contype/conaffinity {n_coll_geoms}; exclude pairs {m.nexclude}; filterparent {filterparent}")
    can = []
    for i, b1 in enumerate(hand_bodies):
        for b2 in hand_bodies[i + 1:]:
            if (b1, b2) in excl:
                continue
            p1, p2 = int(m.body_parentid[b1]), int(m.body_parentid[b2])
            if filterparent and (p1 == b2 or p2 == b1):
                continue
            ok = any((m.geom_contype[g1] & m.geom_conaffinity[g2]) or (m.geom_contype[g2] & m.geom_conaffinity[g1])
                     for g1 in geoms[b1] for g2 in geoms[b2])
            if ok:
                can.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2)))
    print(f"   collidable hand-hand body pairs: {len(can)}")
    for pr in can[:40]:
        print("     ", pr)
    tips = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)) for b in env.tip_body_ids]
    mug_ok = []
    for t in tips:
        gt = geoms[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, t)]
        mug_ok.append(sum(1 for g in gt if m.geom_contype[g] or m.geom_conaffinity[g]))
    print(f"   distal phalanx collision geoms (thumb..pinky): {mug_ok}")
    sample = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}", int(m.geom_contype[g]), int(m.geom_conaffinity[g]))
              for b in hand_bodies[:6] for g in geoms[b]][:12]
    print("   sample hand geoms (name, contype, conaffinity):", sample)


if __name__ == "__main__":
    main()
