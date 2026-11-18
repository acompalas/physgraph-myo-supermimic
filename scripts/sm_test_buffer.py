"""B1 unit tests for lib/sm/buffer.py on synthetic episodes (CPU, no sim).

Each row stores state = [env, episode_id, t_in_episode]. Episodes have
random lengths and end with terminated or truncated. Checks:
  1. every sampled window lies in one episode with consecutive t
  2. no done flag in rows 0..L-2 of any window
  3. number of valid starts == brute force (partial fill and after wrap)
  4. every valid (env, start) gets sampled, per-env counts roughly uniform
  5. errors: L > buffer_size, and L longer than every episode
  6. state_dict round trip
Run: python scripts/sm_test_buffer.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.sm.buffer import FieldSpec, ReplayBuffer  # noqa: E402


def make(E, T, n_insert, seed, min_len=3, max_len=40):
    g = torch.Generator().manual_seed(seed)
    schema = {
        "state": FieldSpec(3),
        "frame": FieldSpec(dtype=torch.long),
        "terminated": FieldSpec(dtype=torch.bool),
        "truncated": FieldSpec(dtype=torch.bool),
    }
    buf = ReplayBuffer(schema, T, E)
    ep_id = torch.zeros(E, dtype=torch.long)
    t_in = torch.zeros(E, dtype=torch.long)
    ep_len = torch.randint(min_len, max_len + 1, (E,), generator=g)
    rows = [[] for _ in range(E)]  # ground truth (ep, t, done)
    for _ in range(n_insert):
        done = t_in == ep_len - 1
        term = done & (torch.rand(E, generator=g) < 0.5)
        trunc = done & ~term
        st = torch.stack([torch.arange(E), ep_id, t_in], 1).float()
        buf.insert({"state": st, "frame": t_in.clone(), "terminated": term, "truncated": trunc})
        for e in range(E):
            rows[e].append((int(ep_id[e]), int(t_in[e]), bool(done[e])))
            rows[e] = rows[e][-T:]
        t_in = torch.where(done, torch.zeros_like(t_in), t_in + 1)
        ep_id = ep_id + done.long()
        new_len = torch.randint(min_len, max_len + 1, (E,), generator=g)
        ep_len = torch.where(done, new_len, ep_len)
    return buf, rows


def brute_valid(rows, L):
    out = []
    for e, r in enumerate(rows):
        for a in range(len(r) - L + 1):
            if not any(r[i][2] for i in range(a, a + L - 1)):
                out.append((e, a))
    return out


def check(name, cond, msg=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {msg}")
    return bool(cond)


def run_case(E, T, n_insert, L, seed):
    print(f"case E={E} T={T} inserted={n_insert} L={L}")
    buf, rows = make(E, T, n_insert, seed)
    ok = True
    gt = brute_valid(rows, L)
    v = buf.valid_starts(L)
    ok &= check("valid count == brute force", int(v.sum()) == len(gt), f"({int(v.sum())} vs {len(gt)})")
    gt_set = set(gt)
    got_set = {(int(e), int(a)) for e, a in v.nonzero().tolist()}
    ok &= check("valid set == brute force", got_set == gt_set)

    B = 20000
    g = torch.Generator().manual_seed(seed + 1)
    s = buf.sample(L, B, generator=g)
    st = s["state"]
    same_env = (st[:, :, 0] == st[:, :1, 0]).all()
    same_ep = (st[:, :, 1] == st[:, :1, 1]).all()
    consec = (st[:, 1:, 2] - st[:, :-1, 2] == 1).all() if L > 1 else torch.tensor(True)
    ok &= check("windows in one env", same_env)
    ok &= check("windows in one episode", same_ep)
    ok &= check("t consecutive", consec)
    if L > 1:
        d = (s["terminated"] | s["truncated"])[:, :-1]
        ok &= check("no done in rows 0..L-2", not d.any())
    ok &= check("frame field matches state t", (s["frame"].float() == st[:, :, 2]).all())

    # coverage and per-env uniformity via storage indices
    g = torch.Generator().manual_seed(seed + 2)
    env_idx, sidx = buf.sample_indices(L, B, generator=g)
    base = buf._base()
    age = (sidx[:, 0] - base) % T
    hit = {(int(e), int(a)) for e, a in zip(env_idx.tolist(), age.tolist())}
    ok &= check("samples only valid starts", hit <= gt_set)
    if len(gt) <= B // 20:
        ok &= check("every valid start sampled", hit == gt_set, f"({len(hit)}/{len(gt)})")
    n_e = torch.bincount(torch.tensor([e for e, _ in gt]), minlength=E)
    has = n_e > 0
    cnt = torch.bincount(env_idx, minlength=E).float()
    exp = B / int(has.sum())
    dev = ((cnt[has] - exp).abs() / exp).max().item()
    ok &= check("env draw uniform over envs with windows", dev < 0.25, f"(max rel dev {dev:.3f})")
    return ok


def main():
    ok = True
    ok &= run_case(E=8, T=200, n_insert=120, L=8, seed=0)    # partial fill
    ok &= run_case(E=8, T=200, n_insert=750, L=8, seed=1)    # wrapped
    ok &= run_case(E=4, T=64, n_insert=300, L=1, seed=2)     # start states
    ok &= run_case(E=16, T=100, n_insert=333, L=33, seed=3)  # BPTT-length windows

    print("errors")
    buf, _ = make(4, 50, 60, 4, min_len=3, max_len=5)
    try:
        buf.sample(51, 10)
        ok &= check("L > buffer_size raises", False)
    except ValueError:
        ok &= check("L > buffer_size raises", True)
    try:
        buf.sample(10, 10)  # all episodes <= 5 rows
        ok &= check("L > every episode raises", False)
    except RuntimeError:
        ok &= check("L > every episode raises", True)

    print("round trip")
    buf, _ = make(4, 50, 77, 5)
    buf2 = ReplayBuffer.from_state_dict(buf.state_dict())
    same = all(torch.equal(buf[k], buf2[k]) for k in buf.schema) and buf._step == buf2._step \
        and buf.num_transitions == buf2.num_transitions
    ok &= check("state_dict round trip", same)
    s1 = buf.sample(5, 100, generator=torch.Generator().manual_seed(9))
    s2 = buf2.sample(5, 100, generator=torch.Generator().manual_seed(9))
    ok &= check("same samples after round trip", all(torch.equal(s1[k], s2[k]) for k in s1))
    st = buf.sample_starts(32)
    ok &= check("sample_starts shapes", st["state"].shape == (32, 3) and st["frame"].shape == (32,))

    print("ALL PASS" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
