"""Full simulator state of MyoHandPourEnv, for the SuperMimic world model.

The full state is everything the env carries from one control step to the
next that changes the next physical state:

    [ qpos (nq) | qvel (nv) | act (na) | wrist PID state (18) ]

The demo frame index (env.progress_buf) is kept separately as a long tensor:
it selects the demo target but does not affect the physics.

Left out on purpose:
  - prev_body_xpos: memory used only by the reward's finite-difference joint
    velocity, not by the dynamics.
  - MuJoCo's solver warm start (data.qacc_warmstart): an initial guess for the
    constraint solver. scripts/sm_check_markov.py measures what leaving it out
    costs.
"""

from dataclasses import dataclass

import mujoco_warp as mjw
import torch
import warp as wp

# Wrist PID memory carried between steps (see MyoHandPourEnv.step), 3 values each.
PID_FIELDS = (
    "pos_error_integral",
    "prev_pos_error",
    "rot_error_integral",
    "prev_rot_error",
    "_applied_force",
    "_applied_torque",
)


@dataclass(frozen=True)
class StateLayout:
    nq: int
    nv: int
    na: int
    npid: int = 3 * len(PID_FIELDS)

    @property
    def dim(self) -> int:
        return self.nq + self.nv + self.na + self.npid

    def slices(self) -> dict:
        out, o = {}, 0
        for name, n in (("qpos", self.nq), ("qvel", self.nv), ("act", self.na), ("pid", self.npid)):
            out[name] = slice(o, o + n)
            o += n
        return out


def _act(env):
    act = getattr(env.data, "act", None)
    if act is None:
        return None
    t = wp.to_torch(act)
    return t if t.numel() > 0 else None


def layout_of(env) -> StateLayout:
    nq = wp.to_torch(env.data.qpos).shape[1]
    nv = wp.to_torch(env.data.qvel).shape[1]
    act = _act(env)
    na = 0 if act is None else act.shape[1]
    return StateLayout(nq=nq, nv=nv, na=na)


def get_state(env):
    """Return (full [N, D] float32, frame [N] long), copies of the current state."""
    parts = [wp.to_torch(env.data.qpos), wp.to_torch(env.data.qvel)]
    act = _act(env)
    if act is not None:
        parts.append(act)
    parts += [getattr(env, name) for name in PID_FIELDS]
    full = torch.cat([p.float() for p in parts], dim=-1).clone()
    return full, env.progress_buf.clone()


def set_state(env, full, frame, env_ids=None, forward=True):
    """Write a full state (and demo frame index) into the given envs."""
    L = layout_of(env)
    sl = L.slices()
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=full.device)
    env_ids = env_ids.to(full.device)
    wp.to_torch(env.data.qpos)[env_ids] = full[:, sl["qpos"]]
    wp.to_torch(env.data.qvel)[env_ids] = full[:, sl["qvel"]]
    act = _act(env)
    if act is not None:
        act[env_ids] = full[:, sl["act"]]
    p0 = sl["pid"].start
    for i, name in enumerate(PID_FIELDS):
        getattr(env, name)[env_ids] = full[:, p0 + 3 * i: p0 + 3 * i + 3]
    env.progress_buf[env_ids] = frame.to(env.progress_buf.dtype)
    if forward:
        mjw.forward(env.model, env.data)


def copy_warmstart(env, src_ids, dst_ids):
    """Copy MuJoCo's constraint-solver warm start between envs, if exposed."""
    ws = getattr(env.data, "qacc_warmstart", None)
    if ws is None:
        return False
    t = wp.to_torch(ws)
    t[dst_ids] = t[src_ids].clone()
    return True
