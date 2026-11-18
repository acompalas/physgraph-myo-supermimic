"""A2: analytic, differentiable actuation for MyoHandPourEnv, in torch.

Maps the actor's raw action and the carried PID / activation state to what
the simulator receives and to the next PID / activation state:

    a_env = tanh(a)                     training's soft clamp (lib/rl/base.py:
                                        -1 + 2*sigmoid(2a) == tanh(a))
    wrist PID + EMA   (copy of MyoHandPourEnv.step)
        e_p = a_env[0:3],  I_p = clip(I_p + e_p*dt, -1, 1),  D_p = (e_p - e_p_prev)/dt
        F   = Kp e_p + Ki I_p + Kd D_p,  F = ema*F + (1-ema)*F_prev     (same for torque)
        a_env[6:9] is unused, as in the env
    muscle ctrl  u = (a_env[9:] + 1) / 2
    activations: MuJoCo muscle dynamics (mujoco_warp util_misc.muscle_dynamics,
        forward._next_activation / support.next_act), explicit Euler per
        physics substep with ctrl held, clamped to actrange if actlimited:
        tau_act   = prm0 * (0.5 + 1.5*clip(act,0,1))
        tau_deact = prm1 / (0.5 + 1.5*clip(act,0,1))
        tau       = hard switch on sign(clip(u,0,1) - act), or quintic-smoothed if prm2 > 0
        act      += dt_sim * (clip(u,0,1) - act) / tau

This replaces what the offline world-model fit (B3) read from logged data.
"""
import mujoco
import torch

MJ_DYN_MUSCLE = int(mujoco.mjtDyn.mjDYN_MUSCLE)
MJ_MINVAL = 1e-15


def params_from_env(env):
    m = env.mj_model
    assert (m.actuator_dyntype == MJ_DYN_MUSCLE).all(), \
        f"expected muscle actuators only, got dyntypes {sorted(set(m.actuator_dyntype.tolist()))}, muscle={MJ_DYN_MUSCLE}"
    assert (m.actuator_actnum == 1).all(), "expected one activation per actuator"
    return {
        "dt": float(env.dt), "sim_dt": float(m.opt.timestep), "substeps": int(env.physics_substeps),
        "Kp_pos": float(env.Kp_pos), "Ki_pos": float(env.Ki_pos), "Kd_pos": float(env.Kd_pos),
        "Kp_rot": env.Kp_rot.tolist(), "Ki_rot": env.Ki_rot.tolist(), "Kd_rot": env.Kd_rot.tolist(),
        "ema": float(env.wrist_ema),
        "dynprm": m.actuator_dynprm[:, :3].tolist(),
        "actadr": m.actuator_actadr.tolist(),
        "actlimited": m.actuator_actlimited.astype(bool).tolist(),
        "actrange": m.actuator_actrange.tolist(),
    }


def _quintic_sigmoid(x):
    x = x.clamp(0.0, 1.0)
    return x * x * x * (3.0 * x * (2.0 * x - 5.0) + 10.0)


class Actuation:
    def __init__(self, p, device):
        t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)  # noqa: E731
        self.p = p
        self.dt, self.h, self.n_sub, self.ema = p["dt"], p["sim_dt"], p["substeps"], p["ema"]
        self.Kp_p, self.Ki_p, self.Kd_p = p["Kp_pos"], p["Ki_pos"], p["Kd_pos"]
        self.Kp_r, self.Ki_r, self.Kd_r = t(p["Kp_rot"]), t(p["Ki_rot"]), t(p["Kd_rot"])
        prm = t(p["dynprm"])
        self.tau_a, self.tau_d, self.smooth = prm[:, 0], prm[:, 1], prm[:, 2]
        self.actadr = torch.tensor(p["actadr"], device=device)
        self.inv = torch.argsort(self.actadr)  # act index -> actuator index
        self.lim = torch.tensor(p["actlimited"], device=device)
        rng = t(p["actrange"])
        self.lo = torch.where(self.lim, rng[:, 0], torch.full_like(rng[:, 0], -float("inf")))
        self.hi = torch.where(self.lim, rng[:, 1], torch.full_like(rng[:, 1], float("inf")))

    @staticmethod
    def soft_clamp(a):
        return torch.tanh(a)

    def wrist(self, a_env, pid):
        I_p, ep_prev, I_r, er_prev, F_prev, T_prev = pid.split(3, dim=-1)
        e_p = a_env[..., 0:3]
        I_p = (I_p + e_p * self.dt).clamp(-1.0, 1.0)
        F = self.Kp_p * e_p + self.Ki_p * I_p + self.Kd_p * (e_p - ep_prev) / self.dt
        e_r = a_env[..., 3:6]
        I_r = (I_r + e_r * self.dt).clamp(-1.0, 1.0)
        T = self.Kp_r * e_r + self.Ki_r * I_r + self.Kd_r * (e_r - er_prev) / self.dt
        F = self.ema * F + (1.0 - self.ema) * F_prev
        T = self.ema * T + (1.0 - self.ema) * T_prev
        return torch.cat([F, T], -1), torch.cat([I_p, e_p, I_r, e_r, F, T], -1)

    def act_dot(self, ctrl, act_u):
        """muscle_dynamics per actuator; ctrl, act_u indexed by actuator."""
        c = ctrl.clamp(0.0, 1.0)
        a = act_u.clamp(0.0, 1.0)
        tau_a = self.tau_a * (0.5 + 1.5 * a)
        tau_d = self.tau_d / (0.5 + 1.5 * a)
        d = c - act_u
        hard = torch.where(d > 0, tau_a, tau_d)
        width = self.smooth.clamp(min=MJ_MINVAL)
        smooth = tau_d + (tau_a - tau_d) * _quintic_sigmoid(d / width + 0.5)
        tau = torch.where(self.smooth < MJ_MINVAL, hard, smooth)
        return d / tau.clamp(min=MJ_MINVAL)

    def act_step(self, act, ctrl):
        a = act[..., self.actadr]  # actuator order
        for _ in range(self.n_sub):
            a = a + self.h * self.act_dot(ctrl, a)
            a = torch.maximum(torch.minimum(a, self.hi), self.lo)
        return a[..., self.inv]  # back to act order

    def __call__(self, a_raw, pid, act, soft=True):
        """Returns (u = [F 3, T 3, ctrl nu], pid_next 18, act_next na)."""
        a_env = self.soft_clamp(a_raw) if soft else a_raw.clamp(-1.0, 1.0)
        root, pid_next = self.wrist(a_env, pid)
        ctrl = (a_env[..., 9:] + 1.0) / 2.0
        return torch.cat([root, ctrl], -1), pid_next, self.act_step(act, ctrl)
