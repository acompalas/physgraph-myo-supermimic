"""Gradient checkpointing check for SMActor: same actor, same inputs, 32-step
action-dependent chain; gradients with grad_ckpt off vs on must match."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from lib.sm.actor_critic import OBS_KEYS, SMActor

torch.manual_seed(0)
dev = "cuda"
dims = {"privileged": 120, "proprioception": 223, "target": 760}
actor = SMActor(dims).to(dev)
B, H = 64, 32
o0 = {k: torch.randn(B, dims[k], device=dev) for k in OBS_KEYS}
actor.update_norm(o0)
with torch.no_grad():
    a_dim = actor(o0).shape[-1]
W = {k: torch.randn(a_dim, dims[k], device=dev) * 0.1 for k in OBS_KEYS}

def run(ckpt):
    actor.grad_ckpt = ckpt
    actor.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    o, loss = {k: v.clone() for k, v in o0.items()}, 0.0
    for _ in range(H):
        a = actor(o)
        loss = loss + (a ** 2).mean()
        o = {k: o[k] + torch.tanh(a @ W[k]) for k in OBS_KEYS}
    loss.backward()
    g = torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None])
    return loss.item(), g, torch.cuda.max_memory_allocated() / 1e9

l0, g0, m0 = run(False)
l1, g1, m1 = run(True)
rel = ((g0 - g1).norm() / g0.norm()).item()
print(f"loss   off {l0:.6f}  on {l1:.6f}")
print(f"|grad| off {g0.norm():.4e}  on {g1.norm():.4e}  rel diff {rel:.2e}  max abs diff {(g0-g1).abs().max():.2e}")
print(f"peak   off {m0:.2f} GB  on {m1:.2f} GB")
print("PASS" if l0 == l1 and rel < 1e-5 else "FAIL")
