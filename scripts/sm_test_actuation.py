"""A2 test: lib/sm/actuation.py against logged simulator data (no sim steps).

For every in-episode row t of the B2 buffers (row t not done, so row t+1 is
its successor), feed the stored raw action, PID state and activations and
compare to what the env actually produced:
  u[t]              root force/torque + muscle ctrl the sim received
  pid in state[t+1] next PID / EMA state
  act in state[t+1] next muscle activations (MuJoCo integrated them)
Then check gradients d(u, act_next)/d(a_raw) are finite and nonzero.

Needs the same MYOHAND_* flags as collection (builds a 1-env env for params).
"""
import argparse
import os
import sys

import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
from lib.sm.actuation import Actuation, params_from_env  # noqa: E402
from lib.sm.buffer import ReplayBuffer  # noqa: E402


def check(path, act_model, dev):
    d = torch.load(path, weights_only=False)
    buf = ReplayBuffer.from_state_dict(d["buffer"], device=dev)
    L = d["meta"]["layout"]
    nq, nv, na = L["nq"], L["nv"], L["na"]
    a0, p0 = nq + nv, nq + nv + na
    n = buf.num_transitions
    S, A, U = buf["state"][:, :n], buf["action"][:, :n], buf["u"][:, :n]
    done = (buf["terminated"] | buf["truncated"])[:, :n]
    keep = ~done[:, :-1]
    s, s1 = S[:, :-1][keep], S[:, 1:][keep]
    a, u_log = A[:, :-1][keep], U[:, :-1][keep]
    with torch.no_grad():
        u, pid1, act1 = act_model(a, s[:, p0:p0 + 18], s[:, a0:a0 + na])
    act_log = s1[:, a0:a0 + na]
    pid_log = s1[:, p0:p0 + 18]

    def stat(name, x, y):
        e = (x - y).abs()
        print(f"    {name:12s} max {e.max():.2e}  p99 {torch.quantile(e.flatten()[:1_000_000], 0.99):.2e}  "
              f"mean {e.mean():.2e}  (scale {y.abs().mean():.2e})")
    print(f"  {os.path.basename(path)}: {s.shape[0]} transitions")
    stat("u_force", u[:, 0:3], u_log[:, 0:3])
    stat("u_torque", u[:, 3:6], u_log[:, 3:6])
    stat("u_ctrl", u[:, 6:], u_log[:, 6:])
    stat("pid_next", pid1, pid_log)
    stat("act_next", act1, act_log)
    # coasting reference: how far act moves in one step
    stat("act_change", s[:, a0:a0 + na], act_log)
    return a, s, a0, p0, na


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="sm_data/collect_runG_noise1.0_E64_T1254.pt,sm_data/collect_runG_latest_E64_T1000.pt")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    dev = args.device

    from envs.mjwarp_myohand_env import MyoHandPourEnv
    env = MyoHandPourEnv(num_envs=1, device=dev)
    prm = params_from_env(env)
    del env
    dp = torch.tensor(prm["dynprm"])
    print(f"params: dt {prm['dt']:.5f} sim_dt {prm['sim_dt']:.5f} substeps {prm['substeps']} ema {prm['ema']}  "
          f"Kp_pos {prm['Kp_pos']} Kp_rot {prm['Kp_rot']}")
    print(f"muscles: tau_act {sorted(set(dp[:, 0].tolist()))}  tau_deact {sorted(set(dp[:, 1].tolist()))}  "
          f"smooth {sorted(set(dp[:, 2].tolist()))}  actlimited {sum(prm['actlimited'])}/{len(prm['actlimited'])}")
    am = Actuation(prm, dev)

    out = None
    for path in args.data.split(","):
        out = check(path, am, dev)

    a, s, a0, p0, na = out
    idx = torch.randperm(a.shape[0], device=dev)[:4096]
    ar = a[idx].clone().requires_grad_(True)
    u, pid1, act1 = am(ar, s[idx, p0:p0 + 18], s[idx, a0:a0 + na])
    g_act, = torch.autograd.grad(act1.sum(), ar, retain_graph=True)
    g_root, = torch.autograd.grad(u[:, :6].sum(), ar)
    print("gradients (4096 rows):")
    print(f"  d act_next / d a[9:]   finite {bool(torch.isfinite(g_act).all())}  "
          f"nonzero {(g_act[:, 9:].abs() > 0).float().mean():.3f}  mean|g| {g_act[:, 9:].abs().mean():.2e}  "
          f"leak to a[0:9] {g_act[:, :9].abs().max():.1e}")
    print(f"  d root u / d a[0:6]    finite {bool(torch.isfinite(g_root).all())}  "
          f"nonzero {(g_root[:, :6].abs() > 0).float().mean():.3f}  a[6:9] (dead dims) {g_root[:, 6:9].abs().max():.1e}")


if __name__ == "__main__":
    main()
