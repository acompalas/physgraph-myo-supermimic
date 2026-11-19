"""C2: SuperMimic actor and critic for MyoHandPourEnv.

Actor: our graph transformer (MyoHandLatentEncoder + MyoHandMLPAction, the
same modules Run N's PPO network uses), made deterministic: it outputs mu
only, the raw action before the env's tanh soft clamp (A2 applies tanh in
imagination, the collector applies it in the real env). No sigma, no value
head. Official SuperMimic's actor is an MLP autoencoder; using our
transformer is a deliberate choice (lab research), an MLP actor comes later
only as a comparison.

Critic: separate MLP, official critic_hidden_dims (256, 128) with the
official build_mlp (Linear-LayerNorm-GELU). Input: the actor observation
(all three blocks) plus the episode phase frame / seq_len, since the task is
time-limited (episodes truncate at the demo's end).

Normalization: running per-feature mean/std on every input, updated from
real-env data by the runner, applied as (x - mean) / (std + 1e-2): the
rsl_rl EmpiricalNormalization the official code uses.
"""
import torch
import torch.nn as nn

from lib.rl.network_builder_myohand_transformer import MyoHandLatentEncoder, MyoHandMLPAction
from lib.sm.world_model import build_mlp

OBS_KEYS = ("proprioception", "privileged", "target")


class RunningNorm(nn.Module):
    """rsl_rl EmpiricalNormalization: count-weighted running mean/var, (x - mean) / (std + eps)."""

    def __init__(self, dim, eps=1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).float()
        n = x.shape[0]
        if n == 0:
            return
        bm, bv = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        rate = n / tot.float()
        delta = bm - self.mean
        self.mean += rate * delta
        self.var += rate * (bv - self.var + delta * (bm - self.mean))
        self.count.copy_(tot)

    def forward(self, x):
        return (x - self.mean) / (self.var.clamp(min=0).sqrt() + self.eps)


class SMActor(nn.Module):
    def __init__(self, obs_dims, action_dim=48):
        super().__init__()
        self.encoder = MyoHandLatentEncoder(action_dim=action_dim)
        self.head = MyoHandMLPAction(action_dim=action_dim)
        self.norm = nn.ModuleDict({k: RunningNorm(obs_dims[k]) for k in OBS_KEYS})

    @torch.no_grad()
    def update_norm(self, obs):
        for k in OBS_KEYS:
            self.norm[k].update(obs[k])

    def forward(self, obs):
        return self.head(self.encoder({k: self.norm[k](obs[k]) for k in OBS_KEYS}))


class SMCritic(nn.Module):
    def __init__(self, obs_dims, seq_len, hidden_dims=(256, 128)):
        super().__init__()
        self.seq_len = float(seq_len)
        d = sum(obs_dims[k] for k in OBS_KEYS) + 1
        self.norm = RunningNorm(d)
        self.mlp = build_mlp(d, hidden_dims, 1)

    def features(self, obs, frame):
        return torch.cat([obs[k] for k in OBS_KEYS] + [(frame.float() / self.seq_len).unsqueeze(-1)], -1)

    @torch.no_grad()
    def update_norm(self, obs, frame):
        self.norm.update(self.features(obs, frame))

    def forward(self, obs, frame):
        return self.mlp(self.norm(self.features(obs, frame))).squeeze(-1)
