"""C1: the imagined env. A differentiable stand-in for MyoHandPourEnv built
from the tested pieces, following the official WorldModelEnv
(supermimic @7e94b07, envs/world_model_env.py):

  reset : start states sampled from the replay buffer (official
          sample_initial_states: buffer.sample(1, B))
  step  : raw action a
          -> A2  Actuation: tanh, wrist PID+EMA, muscle ctrl, activations (exact)
          -> B3  Dynamics: next hand/mug state (learned)
          -> B3d ObsHead: tip and pad contact forces of the next state (learned)
          -> A4  TorchTask: reward, soft termination, actor obs (exact)
          demo clock advances one frame, clamped at the clip end (official
          _compute_commands); returns (obs, reward, term_prob, info)

Our additions vs official: the analytic actuation in front of the world
model, the force head, and keeping the previous full state (the env's
joint velocity is a finite difference over consecutive states).

Start states: the observation at a start row needs the previous row (joint
velocity) and that row's contact forces (logged as next_* labels of the
previous row). Rows that open an episode have no predecessor: there the
previous state is the start state itself and forces are zero, which is what
the fixed env produces at a reset (hand at the rest pose, no contact).
"""
import torch

from lib.sm.world_model import forces_to_obs


class ImaginedEnv:
    def __init__(self, dynamics, head, actuation, task, rep, buffer, batch_size,
                 scale_factor=1.0, softness=0.05, generator=None):
        self.dyn, self.head, self.act_model = dynamics, head, actuation
        self.task, self.rep, self.buf = task, rep, buffer
        self.B = batch_size
        self.scale_factor = scale_factor
        self.softness = softness
        self.gen = generator
        self.seq_len = task.seq_len
        n = rep.a0 + rep.na
        self.p0 = n  # PID block start in the full state

    # ---------- reset ----------
    def sample_start(self, B=None):
        """(full (B,142), prev_full, frame (B,), tip20, pad20) from the buffer."""
        B = B or self.B
        buf = self.buf
        e, idx = buf.sample_indices(1, B, generator=self.gen)
        i = idx[:, 0]
        T = buf.buffer_size
        j = (i - 1) % T
        done_prev = (buf["terminated"] | buf["truncated"])[e, j]
        age = (i - buf._base()) % T
        has_prev = (age > 0) & ~done_prev
        full = buf["state"][e, i]
        prev = torch.where(has_prev[:, None], buf["state"][e, j], full)
        z = torch.zeros(B, 20, device=full.device)
        tip = torch.where(has_prev[:, None], buf["next_tip_force"][e, j], z) if "next_tip_force" in buf.schema else z
        pad = torch.where(has_prev[:, None], buf["next_pad_force"][e, j], z) if "next_pad_force" in buf.schema else z
        return full, prev, buf["frame"][e, i], tip, pad

    def reset(self, start=None):
        full, prev, frame, tip, pad = start if start is not None else self.sample_start()
        self.full, self.prev_full = full.detach(), prev.detach()
        self.frame = frame.clone()
        self.steps = frame.clone()  # steps since reset == frame (no random start)
        self.s = self.rep.from_full(self.full)
        self.tip, self.pad = tip, pad
        self.k, self.k_prev = self.task.kin(self.full), self.task.kin(self.prev_full)
        return self.obs()

    def obs(self):
        return self.task.obs(self.full, self.prev_full, self.frame, self.tip, self.k, self.k_prev, self.pad)

    # ---------- step ----------
    def step(self, a_raw):
        pid = self.full[..., self.p0:self.p0 + 18]
        u, pid_next, act_next = self.act_model(a_raw, pid, self.s["act"])
        s_next = self.dyn(self.s, u, act_next)
        full_next = self.rep.to_full(s_next, self.full, pid_next)
        f30 = self.head(s_next, u) if self.head.u_dim else self.head(s_next)
        tip, pad = forces_to_obs(f30)
        frame_next = torch.clamp(self.frame + 1, max=self.seq_len - 1)
        steps_next = self.steps + 1
        k_next = self.task.kin(full_next)  # the only FK call per step; the previous one is reused
        r, terms, zt, ze = self.task.reward(full_next, self.full, u, frame_next, self.scale_factor, k_next, self.k)
        d_soft, d_hard = self.task.termination(zt, ze, steps_next, self.softness)
        self.prev_full, self.full, self.s = self.full, full_next, s_next
        self.k_prev, self.k = self.k, k_next
        self.frame, self.steps, self.tip, self.pad = frame_next, steps_next, tip, pad
        info = {"terms": terms, "hard": d_hard, "u": u, "z_track": zt, "z_err": ze}
        return self.obs(), r, d_soft, info
