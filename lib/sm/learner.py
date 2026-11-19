"""C3: actor-critic policy learner through the imagined env.

Port of the official ActorCriticPolicyLearner (supermimic @7e94b07,
learners/policy_learner.py + learner_utils.py), on our ImaginedEnv:

  actor step : one imagined rollout of H steps from buffer start states;
               return = sum_t gamma^t * survival_t * r_t, survival_t =
               prod_{k<t} (1 - d_k) (survival masking: the reward of the
               transition that terminates still counts), plus the bootstrap
               gamma^H * clamp(V(s_H), max=V_max) * survival_H (no grad into
               the critic's weights, grad flows through s_H and survival);
               loss = -mean(return); grad-norm clip 1.0, then element clip
               0.1; Adam (beta2 0.95) + Muon on hidden Linear weights, lr 2e-4
  critic fit : GAE(gamma, lam) returns of the same rollout (soft done p_t
               discounts the next value by 1 - p_t), clamped at V_max; one
               random permutation of the T*B samples, 5 epochs x 4
               minibatches, MSE, grad-norm clip 1.0; Adam (default betas) +
               Muon, lr 2e-4
  V_max      : max_step_reward * reward_scale / (1 - gamma)
  rewards    : scaled by reward_scale = dt, as the official rewards are dt-scaled;
               logged return / reward_per_step are unscaled

Critic input (official: privileged WM state + goal, no input normalizer):
our world-model state features (MyoPourRep.features, 124) + the actor's
target block (the demo goal, 760) + phase frame / seq_len, through the
official build_mlp (Linear-LayerNorm-GELU, 256/128).

Not ported: CUDA-graph capture of the update (CapturedUpdate) and
num_batches > 1 (official default is 1).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn

from lib.sm.world_model import build_mlp, build_optimizers


@dataclass
class LearnerCfg:
    lr: float = 2e-4
    batch_size: int = 4096
    horizon: int = 32
    gamma: float = 0.99
    lam: float = 0.95
    beta2: float = 0.95
    grad_clip_value: float = 0.1
    survival_masking: bool = True
    reward_scale: float = 1.0  # official rewards are dt-scaled (weight * step_dt); set to env dt


class SMCriticWM(nn.Module):
    """Official critic: MLP on the privileged state + goal, no input normalizer."""

    def __init__(self, rep_dim, goal_dim, seq_len, hidden_dims=(256, 128)):
        super().__init__()
        self.seq_len = float(seq_len)
        self.mlp = build_mlp(rep_dim + goal_dim + 1, hidden_dims, 1)

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def critic_input(ienv, obs):
    """(B, 124 + 760 + 1) from the imagined env's current state."""
    phase = (ienv.frame.float() / float(ienv.seq_len)).unsqueeze(-1)
    return torch.cat([ienv.rep.features(ienv.s), obs["target"], phase], -1)


def gae_returns(values, last_value, rewards, dones, gamma, lam):
    """Official gae_returns: A_t + V(s_t) of a time-major rollout."""
    returns = torch.empty_like(values)
    advantage = 0.0
    next_value = last_value
    for t in reversed(range(values.shape[0])):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + not_done * gamma * next_value - values[t]
        advantage = delta + not_done * gamma * lam * advantage
        returns[t] = advantage + values[t]
        next_value = values[t]
    return returns


def _optimize_step(optimizers, loss, params=None, max_norm=None, clip_value=0.0):
    """Official _optimize_step: zero_grad -> backward -> norm clip -> element clip -> step."""
    for o in optimizers:
        o.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = None
    if params is not None and max_norm is not None:
        params = list(params)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
        if clip_value > 0:
            torch.nn.utils.clip_grad_value_(params, clip_value)
    for o in optimizers:
        o.step()
    return grad_norm


class ActorCriticLearner:
    def __init__(self, actor, critic, ienv, cfg: LearnerCfg, max_step_reward):
        self.actor, self.critic, self.ienv, self.cfg = actor, critic, ienv, cfg
        self.actor_params = list(actor.parameters())
        self.actor_opts = build_optimizers(actor, cfg.lr, betas=(0.9, cfg.beta2))
        self.critic_opts = build_optimizers(critic, cfg.lr)
        self.v_max = max_step_reward * cfg.reward_scale / (1.0 - cfg.gamma)

    def update(self, start=None):
        cfg, ie = self.cfg, self.ienv
        for m in (ie.dyn, ie.head):
            m.eval()
            m.requires_grad_(False)
        self.actor.train()
        obs = ie.reset(start if start is not None else ie.sample_start(cfg.batch_size))
        ret, survival = 0.0, 1.0
        crit_x, rews, dones, surv_end = [], [], [], []
        for t in range(cfg.horizon):
            crit_x.append(critic_input(ie, obs).detach())
            obs, r, d, _ = ie.step(self.actor(obs))
            r = r * cfg.reward_scale
            rews.append(r.detach())
            dones.append(d.detach())
            ret = ret + cfg.gamma ** t * survival * r
            if cfg.survival_masking:
                survival = survival * (1 - d)
        x_last = critic_input(ie, obs)
        self.critic.requires_grad_(False)
        alive = survival if cfg.survival_masking else 1.0 - d
        ret = ret + cfg.gamma ** cfg.horizon * self.critic(x_last).clamp(max=self.v_max) * alive
        loss = -ret.mean()
        grad_norm = _optimize_step(self.actor_opts, loss, self.actor_params, 1.0, cfg.grad_clip_value)
        stats = {
            "return": ret.mean().detach() / cfg.reward_scale,
            "reward_per_step": torch.stack(rews).mean() / cfg.reward_scale,
            "survival_end": (survival.mean() if torch.is_tensor(survival) else torch.tensor(survival)).detach(),
            "grad_norm": grad_norm.detach(),
        }
        stats.update(self._fit_critic(torch.stack(crit_x), x_last.detach(), torch.stack(rews), torch.stack(dones)))
        return stats

    def _fit_critic(self, x, x_last, rewards, dones):
        cfg = self.cfg
        with torch.no_grad():
            values = self.critic(x)
            last_value = self.critic(x_last)
        returns = gae_returns(values, last_value, rewards, dones, cfg.gamma, cfg.lam).clamp(max=self.v_max)
        x, returns = x.flatten(0, 1), returns.flatten(0, 1)
        perm = torch.randperm(len(returns) // 4 * 4, device=returns.device)
        self.critic.requires_grad_(True)
        losses = []
        for _ in range(5):
            for idx in perm.view(4, -1):
                loss = (returns[idx] - self.critic(x[idx])).pow(2).mean()
                _optimize_step(self.critic_opts, loss, self.critic.parameters(), max_norm=1.0)
                losses.append(loss.detach())
        v_pre = values.flatten(0, 1)
        ev = 1.0 - (returns - v_pre).var() / returns.var().clamp(min=1e-8)
        return {"value_loss": torch.stack(losses).mean(), "value_mean": values.mean(),
                "gae_return_mean": returns.mean(), "explained_var": ev}
