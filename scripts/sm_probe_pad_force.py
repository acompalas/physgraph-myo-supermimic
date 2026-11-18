"""Probe the pad-side contact force before adding it to collection (B2b).

On logged states from the deterministic Run G set, CPU MuJoCo mj_forward
(same model the env builds) with the logged qpos/qvel/act, muscle ctrl and
root force/torque (minus the env's root damping, as in its first substep).
For each distal phalanx (env.tip_body_ids, PhysGraph's tip_force bodies):

  1. consistency: sum of per-contact linear forces on the body (contact
     frame -> world, sign by which geom the body owns) vs MuJoCo's own
     cfrc_ext[body, 3:6] after mj_rnePostConstraint
  2. split: contact point p in the body frame; bone axis a = body origin ->
     tip site; pad direction n = pad offset made perpendicular to a
     (assets/retargeted/pad_sites_local.json); pad contact iff p . n > 0.
     Per finger and phase: total, pad and non-pad force, pad fraction.
  3. figure: contact points in each body frame (along a vs along n), pad vs
     non-pad, with the pad site; saved to sm_data/pad_probe.png

Needs the same MYOHAND_* flags as collection (builds a 1-env env).
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

PHASES = (("approach", 0, 50), ("grasp+carry", 50, 373), ("pour", 373, 10_000))
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="sm_data/collect_runG_latest_E64_T1000.pt")
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--out", default="sm_data/pad_probe")
    args = p.parse_args()

    from envs.mjwarp_myohand_env import MyoHandPourEnv
    env = MyoHandPourEnv(num_envs=1, device="cuda:0")
    m = env.mj_model
    d = mujoco.MjData(m)
    tip_bodies = [int(b) for b in env.tip_body_ids]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in tip_bodies]
    hand_root = int(m.body_rootid[tip_bodies[0]])
    src_body = int(m.jnt_bodyid[[j for j in range(m.njnt) if int(m.jnt_qposadr[j]) == int(env.src_adr)][0]])
    pj = json.load(open(os.path.join(repo_root, "assets/retargeted/pad_sites_local.json")))["bodies"]
    axes, normals, pads = [], [], []
    for nm in names:
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, pj[nm]["tip_site"])
        assert int(m.site_bodyid[sid]) == mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm), f"{nm}: tip site not on body"
        a = np.array(m.site_pos[sid], dtype=np.float64)
        a /= np.linalg.norm(a)
        pad = np.array(pj[nm]["pad_local_m"], dtype=np.float64)
        n = pad - pad.dot(a) * a
        n /= np.linalg.norm(n)
        axes.append(a), normals.append(n), pads.append(pad)
        print(f"{nm:16s} axis {np.round(a, 3)}  pad dir {np.round(n, 3)}  pad site {np.round(pad * 1000, 1)} mm")

    sd = torch.load(args.data, weights_only=False)
    st = sd["buffer"]["storage"]
    L = sd["meta"]["layout"]
    nq, nv, na = L["nq"], L["nv"], L["na"]
    S, U, F = st["state"].numpy(), st["u"].numpy(), st["frame"].numpy()
    rd = env.root_dof_adr
    lin = float(env._root_damp_lin) if env._root_damp is not None else 0.0
    ang = env._root_damp_ang.cpu().numpy() if env._root_damp is not None else np.zeros((3, 3))

    f6 = np.zeros(6)
    rows = []          # (finger, phase, |total|, |pad|, |nonpad|, n_contacts)
    pts = []           # (finger, along_a, along_n, is_pad, |f|)
    max_cons, max_rel = 0.0, 0.0
    sign_votes = {+1: 0, -1: 0}
    for e in range(min(args.envs, S.shape[0])):
        for t in range(0, S.shape[1], args.stride):
            s, u, fr = S[e, t], U[e, t], int(F[e, t])
            d.qpos[:] = s[:nq]
            d.qvel[:] = s[nq:nq + nv]
            d.act[:] = s[nq + nv:nq + nv + na]
            d.ctrl[:] = u[6:]
            d.qfrc_applied[:] = 0.0
            qv = s[nq + rd:nq + rd + 6]
            d.qfrc_applied[rd:rd + 3] = u[0:3] - lin * qv[0:3]
            d.qfrc_applied[rd + 3:rd + 6] = u[3:6] - qv[3:6] @ ang
            mujoco.mj_forward(m, d)
            mujoco.mj_rnePostConstraint(m, d)
            tot = np.zeros((5, 3)); padf = np.zeros((5, 3)); npadf = np.zeros((5, 3)); cnt = np.zeros(5, int)
            part = np.zeros((5, 2, 3, 3))  # [pad, non-pad] x [mug, hand, other]
            for ci in range(d.ncon):
                c = d.contact[ci]
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                for k, b in enumerate(tip_bodies):
                    if b not in (b1, b2) or b1 == b2:
                        continue
                    mujoco.mj_contactForce(m, d, ci, f6)
                    fw = np.asarray(c.frame).reshape(3, 3).T @ f6[:3]   # world, as applied to geom2's body
                    sgn = 1.0 if b == b2 else -1.0
                    fb = sgn * fw
                    R = d.xmat[b].reshape(3, 3)
                    pl = R.T @ (c.pos - d.xpos[b])
                    is_pad = pl.dot(normals[k]) > 0.0
                    tot[k] += fb
                    (padf if is_pad else npadf)[k] += fb
                    other = b1 if b == b2 else b2
                    cat = 0 if other == src_body else (1 if int(m.body_rootid[other]) == hand_root else 2)
                    part[k, 0 if is_pad else 1, cat] += fb
                    cnt[k] += 1
                    pts.append((k, pl.dot(axes[k]), pl.dot(normals[k]), is_pad, np.linalg.norm(fb)))
            for k, b in enumerate(tip_bodies):
                ref = d.cfrc_ext[b, 3:6]
                err = np.linalg.norm(tot[k] - ref)
                max_cons = max(max_cons, err)
                if np.linalg.norm(ref) > 1e-3:
                    max_rel = max(max_rel, err / np.linalg.norm(ref))
                    sign_votes[+1] += int(np.linalg.norm(tot[k] - ref) < np.linalg.norm(-tot[k] - ref))
                    sign_votes[-1] += int(np.linalg.norm(-tot[k] - ref) < np.linalg.norm(tot[k] - ref))
                ph = next(i for i, (_, lo, hi) in enumerate(PHASES) if lo <= fr < hi)
                rows.append((k, ph, np.linalg.norm(tot[k]), np.linalg.norm(padf[k]), np.linalg.norm(npadf[k]), cnt[k],
                             *np.linalg.norm(part[k], axis=-1).flatten()))

    print(f"\n1. consistency vs cfrc_ext[3:6]: max abs err {max_cons:.2e} N, max rel err {max_rel:.2e} "
          f"(sign convention votes: as-coded {sign_votes[+1]}, flipped {sign_votes[-1]})")
    R_ = np.array(rows, dtype=float)
    print("\n2. per finger and phase (mean over sampled states; force in N):")
    print(f"   {'finger':7s} {'phase':12s} {'states':>6s} {'in contact':>10s} {'|total|':>8s} {'|pad|':>7s} {'|non-pad|':>9s} {'pad frac':>8s}")
    for k, fn in enumerate(FINGERS):
        for ph, (pn, _, _) in enumerate(PHASES):
            sel = R_[(R_[:, 0] == k) & (R_[:, 1] == ph)]
            if len(sel) == 0:
                continue
            inc = sel[:, 5] > 0
            frac = sel[inc, 3].sum() / max(sel[inc, 3].sum() + sel[inc, 4].sum(), 1e-9)
            print(f"   {fn:7s} {pn:12s} {len(sel):6d} {inc.mean():10.2f} {sel[:, 2].mean():8.2f} {sel[:, 3].mean():7.2f} "
                  f"{sel[:, 4].mean():9.2f} {frac:8.2f}")

    print("\n2b. by contact partner (mean |force| N): pad mug / pad hand / pad other | non-pad mug / non-pad hand / non-pad other")
    for k, fn in enumerate(FINGERS):
        for ph, (pn, _, _) in enumerate(PHASES):
            sel = R_[(R_[:, 0] == k) & (R_[:, 1] == ph)]
            if len(sel) == 0:
                continue
            v = sel[:, 6:12].mean(0)
            print(f"   {fn:7s} {pn:12s} {v[0]:6.2f} {v[1]:6.2f} {v[2]:6.2f} | {v[3]:6.2f} {v[4]:6.2f} {v[5]:6.2f}")

    P = np.array(pts, dtype=float)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out + ".npz", points=P, axes=np.array(axes), normals=np.array(normals), pads=np.array(pads), names=np.array(names))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 5, figsize=(20, 4.2))
        for k, ax in enumerate(axs):
            q = P[P[:, 0] == k]
            for flag, col, lab in ((1, "tab:blue", "pad"), (0, "tab:red", "non-pad")):
                z = q[q[:, 3] == flag]
                ax.scatter(z[:, 1] * 1000, z[:, 2] * 1000, s=4 + 2 * np.clip(z[:, 4], 0, 10), c=col, alpha=0.4, label=lab)
            ax.scatter([pads[k].dot(axes[k]) * 1000], [pads[k].dot(normals[k]) * 1000], marker="*", s=200, c="k", label="pad site")
            ax.axhline(0, c="gray", lw=0.8)
            ax.set_title(f"{FINGERS[k]} ({len(q)} contacts)")
            ax.set_xlabel("along bone axis (mm)")
            ax.set_ylabel("toward pad (mm)")
            ax.set_aspect("equal", adjustable="datalim")
        axs[0].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out + ".png", dpi=110)
        print(f"\n3. figure saved {args.out}.png  (points: {args.out}.npz)")
    except ImportError:
        print(f"\n3. matplotlib not available; points saved {args.out}.npz")


if __name__ == "__main__":
    main()
