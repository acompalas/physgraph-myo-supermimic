"""B3: fit the SuperMimic world model offline and check it (feasibility gate).

Training follows the official WorldModelLearner (wm_learner.py): windows of
--horizon rows (official 8) from the replay buffer, autoregressive rollout on
the logged actions from the logged first state (horizon - 1 predictions),
Bryson-weighted state loss, Adam + Muon at lr 2e-4, global grad clip 1.0,
batch 2048. Normalizer stats come once from the whole training buffer
(official updates them online from collected data).

Offline differences (this is a diagnostic, not the training loop): a fixed
dataset from Run G, many gradient steps, and evaluation on a held-out set.

Reports, on the test buffer, open-loop errors after H steps (default 7 and
32) per demo phase, for three predictors:
  zero    velocity deltas = 0 (coasting), a floor any model must beat
  oracle  true velocity deltas; tests only our integration vs the sim
  model   the trained world model
Gate (Anderson's call, defaults are a proposal): at H=32, median mug position
error < --gate-mug-cm and median joint error < --gate-joint-deg, separately in
the grasp/carry and pour phases.

Needs the same MYOHAND_* flags as collection (it builds a 1-env env only to
read the qpos/qvel layout).
"""
import argparse
import os
import sys
import time

import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
from lib.sm.buffer import ReplayBuffer  # noqa: E402
import torch.nn as nn  # noqa: E402
from lib.sm.world_model import (Dynamics, MyoPourRep, ObsHead, build_optimizers, forces30,  # noqa: E402
                                layout_from_env, quat_angle, rollout, state_loss)

PHASES = (("approach", 0, 50), ("grasp+carry", 50, 373), ("pour", 373, 10_000))  # pour_frame = 373


def load_buffer(path, dev):
    d = torch.load(path, weights_only=False)
    return ReplayBuffer.from_state_dict(d["buffer"], device=dev), d["meta"]


def window(buf, L, n, gen):
    e, idx = buf.sample_indices(L, n, generator=gen)
    keys = [k for k in ("state", "u", "frame", "next_tip_force", "next_pad_force") if k in buf.schema]
    return {k: buf[k][e[:, None], idx] for k in keys}


@torch.no_grad()
def eval_rollouts(model, rep, buf, H, n, seed, mode, head=None, mean30=None):
    gen = torch.Generator(device=buf.device).manual_seed(seed)
    w = window(buf, H + 1, n, gen)
    S = w["state"]
    gt = rep.from_full(S[:, 1:])
    s = rep.from_full(S[:, 0])
    u = w["u"][:, :H]
    o_last = None
    for k in range(H):
        gk = {key: v[:, k] for key, v in gt.items()}
        if mode == "model":
            s, o_last = model.step(s, u[:, k], gk["act"])
        elif mode == "zero":
            s = rep.apply_prediction(s, torch.zeros(S.shape[0], rep.pred_dim, device=S.device), gk["act"])
        elif mode == "oracle":
            s = rep.apply_prediction(s, rep.oracle_prediction(s, gk), gk["act"])
    g = {key: v[:, -1] for key, v in gt.items()}
    err = {
        "mug_cm": (s["mp"] - g["mp"]).norm(dim=-1) * 100,
        "mug_deg": torch.rad2deg(quat_angle(s["mq"], g["mq"])),
        "root_cm": (s["rp"] - g["rp"]).norm(dim=-1) * 100,
        "root_deg": torch.rad2deg(quat_angle(s["rq"], g["rq"])),
        "joint_deg": torch.rad2deg((s["qj"] - g["qj"]).abs()).mean(-1),
    }
    if mean30 is not None and mode == "model":
        tgt = forces30(w["next_tip_force"][:, H - 1], w["next_pad_force"][:, H - 1]).view(-1, 2, 5, 3).norm(dim=-1)
        def mag_err(f30):
            e = (f30.view(-1, 2, 5, 3).norm(dim=-1) - tgt).abs().mean(-1)  # (n, 2) mean over fingers
            return e[:, 0], e[:, 1]
        if head is not None:
            ul = u[:, H - 1] if head.u_dim else None
            err["tip_N"], err["pad_N"] = mag_err(head(s, ul))
            _, err["pad_N_truestate"] = mag_err(head(g, ul))
        else:  # forces from the dynamics MLP: rollout output, and one step from the true previous state
            err["tip_N"], err["pad_N"] = mag_err(model.obs_denorm(o_last))
            prev = rep.from_full(S[:, H - 1])
            _, o_true = model.step(prev, u[:, H - 1], g["act"])
            _, err["pad_N_truestate"] = mag_err(model.obs_denorm(o_true))
        _, err["pad_N_meanpred"] = mag_err(mean30.expand(tgt.shape[0], -1))
    return err, w["frame"][:, 0]


def report(tag, err, f0):
    print(f"  {tag}")
    for name, lo, hi in PHASES:
        m = (f0 >= lo) & (f0 < hi)
        if m.sum() == 0:
            continue
        cells = []
        for k, v in err.items():
            x = v[m]
            cells.append(f"{k} {x.median():.2f}/{torch.quantile(x, 0.9):.2f}")
        print(f"    {name:12s} n={int(m.sum()):5d}  " + "  ".join(cells))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="sm_data/b2b_train.pt")
    p.add_argument("--test", default="sm_data/b2b_test.pt")
    p.add_argument("--obs-weight", type=float, default=1.0, help="w_o for the force observation loss; 0 disables")
    p.add_argument("--obs-mode", choices=("head", "head_detach", "wm"), default="head",
                   help="head: separate g(s_next) (SuperMimic eq. 1); wm: extra outputs of the dynamics MLP from (s, u) (FreeMusco-style)")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--horizon", type=int, default=8, help="rows per window (official wm.horizon)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--hidden", default="2048,2048,2048")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-n", type=int, default=4096)
    p.add_argument("--eval-h", default="7,32")
    p.add_argument("--gate-mug-cm", type=float, default=1.0)
    p.add_argument("--gate-joint-deg", type=float, default=5.0)
    p.add_argument("--out", default="sm_data/wm_b2b.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    dev = args.device
    Hs = [int(h) for h in args.eval_h.split(",")]

    from envs.mjwarp_myohand_env import MyoHandPourEnv
    env = MyoHandPourEnv(num_envs=1, device=dev)
    lay = layout_from_env(env)
    del env
    print(f"layout: root q{lay['root_q']} v{lay['root_v']}  src q{lay['src_q']} v{lay['src_v']}  "
          f"hand joints {len(lay['hand_q'])} (q {lay['hand_q'][0]}..{lay['hand_q'][-1]})  "
          f"decimation {lay['decimation']} dt {lay['dt']:.5f}")

    train, mtr = load_buffer(args.train, dev)
    test, _ = load_buffer(args.test, dev)
    assert mtr["layout"]["nq"] == lay["nq"] and mtr["layout"]["na"] == lay["na"], "buffer/env layout mismatch"
    rep = MyoPourRep(lay, dev)
    print(f"substep_coeff c = {rep.c:.4f}  features {rep.feat_dim}  u {train['u'].shape[-1]}  prediction {rep.pred_dim}")

    use_obs = args.obs_weight > 0 and "next_pad_force" in train.schema
    wm_obs = use_obs and args.obs_mode == "wm"
    model = Dynamics(rep, train["u"].shape[-1], tuple(int(h) for h in args.hidden.split(",")),
                     n_obs=30 if wm_obs else 0).to(dev)
    n = train.num_transitions
    with torch.no_grad():
        model.state_normalizer.fit(rep.features(rep.from_full(train["state"][:, :n])))
        model.action_normalizer.fit(train["u"][:, :n])
    head, mean30 = None, None
    if use_obs:
        head = ObsHead(rep, u_dim=train["u"].shape[-1] if args.obs_mode == "head_detach" else 0).to(dev) if not wm_obs else None
        with torch.no_grad():
            # row t's next_* label belongs to state t+1
            fs = rep.features(rep.from_full(train["state"][:, 1:n]))
            tg = forces30(train["next_tip_force"][:, :n - 1], train["next_pad_force"][:, :n - 1])
            if wm_obs:
                model.fit_obs_stats(tg)
                mean30 = model.obs_mean[None]
                print(f"obs: forces as extra dynamics outputs (from s, u), w_o {args.obs_weight}")
            else:
                head.fit_stats(fs, tg, train["u"][:, :n - 1])
                mean30 = head.out_mean[None]
                print(f"obs head: {sum(p.numel() for p in head.parameters()) / 1e6:.2f} M params, w_o {args.obs_weight}")
    opts = build_optimizers(nn.ModuleList([model] + ([head] if head is not None else [])), args.lr)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M, optimizers "
          + ", ".join(type(o).__name__ for o in opts))

    print("\nbaselines on test (median/p90 at step H):")
    for H in Hs:
        for mode in ("zero", "oracle"):
            err, f0 = eval_rollouts(model, rep, test, H, args.eval_n, 123 + H, mode)
            report(f"{mode} H={H}", err, f0)

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    best = float("inf")
    t0 = time.time()
    model.train()
    if head is not None:
        head.train()
    for step in range(1, args.steps + 1):
        w = window(train, args.horizon, args.batch, gen)
        S = w["state"]
        s0 = rep.from_full(S[:, 0])
        gt = rep.from_full(S[:, 1:])
        pred, o_pred = rollout(model, s0, w["u"][:, :-1], gt["act"], with_obs=True)
        loss, groups = state_loss(rep, pred, gt)
        if use_obs:
            tg30 = forces30(w["next_tip_force"][:, :-1], w["next_pad_force"][:, :-1])
            if wm_obs:
                o_loss = ((o_pred - (tg30 - model.obs_mean) / model.obs_std) ** 2).mean()
            else:
                if args.obs_mode == "head_detach":  # no force gradient into the dynamics
                    o_loss = head.loss({k: v.detach() for k, v in pred.items()}, tg30, w["u"][:, :-1])
                else:
                    o_loss = head.loss(pred, tg30)
            loss = loss + args.obs_weight * o_loss
            groups["obs"] = o_loss.detach()
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        params = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
        gn = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        for o in opts:
            o.step()
        if step % 200 == 0 or step == 1:
            g = "  ".join(f"{k} {v.item():.2e}" for k, v in groups.items())
            print(f"step {step:6d}  loss {loss.item():.4f}  gnorm {gn.item():.2f}  {g}  "
                  f"({(time.time() - t0) / step:.3f} s/step)", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            if head is not None:
                head.eval()
            print(f"eval at step {step} (model, median/p90 at step H):")
            for H in Hs:
                err, f0 = eval_rollouts(model, rep, test, H, args.eval_n, 123 + H, "model", head, mean30)
                report(f"model H={H}", err, f0)
            m = f0 >= 50
            score = err["mug_cm"][m].median().item()  # last H in Hs
            if score < best:
                best = score
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                torch.save({"model": model.state_dict(), "head": None if head is None else head.state_dict(),
                            "layout": lay, "args": vars(args),
                            "step": step, "score_mug_cm": score}, args.out)
                print(f"  saved {args.out} (median mug err {score:.3f} cm, frames >= 50, H={Hs[-1]})")
            model.train()
            if head is not None:
                head.train()

    # gate at the longest H, on the final model
    model.eval()
    H = Hs[-1]
    if head is not None:
        head.eval()
    err, f0 = eval_rollouts(model, rep, test, H, args.eval_n, 999, "model", head, mean30)
    print(f"\nGATE at H={H}: median mug < {args.gate_mug_cm} cm and median joint < {args.gate_joint_deg} deg")
    ok = True
    for name, lo, hi in PHASES[1:]:
        m = (f0 >= lo) & (f0 < hi)
        mc, jd = err["mug_cm"][m].median().item(), err["joint_deg"][m].median().item()
        pas = mc < args.gate_mug_cm and jd < args.gate_joint_deg
        ok &= pas
        print(f"  {name:12s} mug {mc:.2f} cm  joint {jd:.2f} deg  -> {'PASS' if pas else 'FAIL'}")
    print("GATE PASS" if ok else "GATE FAIL")
    if mean30 is not None:
        print(f"force head at H={H} (median N, mean over fingers): pad on rollout / pad on true state / mean predictor | tip on rollout")
        for name, lo, hi in PHASES:
            m = (f0 >= lo) & (f0 < hi)
            if m.sum() == 0:
                continue
            print(f"  {name:12s} {err['pad_N'][m].median():6.2f} / {err['pad_N_truestate'][m].median():6.2f} / "
                  f"{err['pad_N_meanpred'][m].median():6.2f} | {err['tip_N'][m].median():6.2f}")
    print(f"total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
