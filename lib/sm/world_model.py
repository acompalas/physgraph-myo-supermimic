"""SuperMimic world model for MyoHandPourEnv.

Port of the official Dynamics + GenCoordRepresentation
(HybridRobotics/supermimic @7e94b07: models.py, representations.py,
utils/gencoord_helpers.py, utils/helpers.py), extended to our scene.

Same as official:
  - MLP (Linear -> LayerNorm -> GELU) x hidden, then Linear; inputs are the
    normalized state features and normalized action (no sim context here).
  - prediction = accel-level velocity deltas in the CURRENT body frame;
    poses integrate analytically, semi-implicit, with
    substep_coeff c = (decimation + 1) / (2 * decimation)
    (decimation = physics substeps per control step, 4 here, so c = 0.625).
  - quaternion update quat_integrate(q, w_world * dt), left-multiplied.
  - state loss: Bryson's rule, weight 1/nominal^2 per channel group,
    width-weighted mean; root rotation scored as sqrt(1/2) * R.
  - normalizers: running mean/std with (x - mean) / (std + 1e-2), as
    rsl_rl EmpiricalNormalization.

Different, because our scene differs:
  - two free bodies, the hand root and the source mug; each gets
    [dv_b, dw_b] and the same integration. The destination mug is held
    fixed (A1: it moves < 0.1 mm).
  - features use world root pose (pos + rot6d) instead of only z and g_b:
    the table, the fixed destination mug and the world-frame wrist force
    break the translation invariance the official z/g_b input relies on.
  - the action input is u = [root force 3 (world), root torque 3 (body),
    muscle ctrl 39], what the sim receives after our PID+EMA. Official
    feeds proc(action), its actuator command; ours has PID memory, so u
    is the actuator-level equivalent.
  - muscle activations are an input feature but not predicted: they
    follow MuJoCo's activation ODE from u and are passed in (logged in the
    offline fit, computed analytically by A2 later).
"""
import math

import torch
import torch.nn as nn


# ---------------- quaternion helpers, (w, x, y, z) ----------------
def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], -1)


def quat_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], -1)


def quat_apply(q, v):
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def quat_apply_inverse(q, v):
    return quat_apply(quat_conj(q), v)


def matrix_from_quat(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q.shape[:-1], 3, 3)


def rot6d(R):
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def quat_integrate(quat, ang_vel_world, dt):
    """Official utils/helpers.py quat_integrate."""
    omega = ang_vel_world * dt
    norm_safe = torch.sqrt((omega * omega).sum(-1, keepdim=True) + 1e-8)
    half = norm_safe / 2
    dq = torch.cat([torch.cos(half), torch.sin(half) * omega / norm_safe], -1)
    q = quat_mul(dq, quat)
    return q / torch.norm(q, dim=-1, keepdim=True)


def quat_angle(q1, q2):
    d = (q1 * q2).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(d)


# ---------------- state layout ----------------
def layout_from_env(env):
    """Index layout of the 142-d full state (lib/sm/state.py) from the env."""
    nq = int(env.mj_model.nq)
    nv = int(env.mj_model.nv)
    na = int(env.mj_model.na)
    lay = {
        "nq": nq, "nv": nv, "na": na,
        "root_q": int(env.root_adr), "root_v": int(env.root_dof_adr),
        "src_q": int(env.src_adr), "src_v": int(env.src_dof_adr),
        "dst_q": int(env.dst_adr), "dst_v": int(env.dst_dof_adr),
        "hand_q": [int(i) for i in env.dof_adrs],
        "hand_v": [int(i) for i in env.dof_veladrs],
        "decimation": int(env.physics_substeps), "dt": float(env.dt),
    }
    return lay


class MyoPourRep:
    """Split the full state into the carried world-model state and back.

    Carried state s (dict):
      rp, rq        root pos (world), quat
      rv, rw        root lin / ang velocity in the root body frame
      qj, vj        hand joint pos / vel (23)
      mp, mq, mv, mw  source mug, same convention as the root
      act           muscle activations (39), passed through, not predicted
    """
    LOSS_NOMINAL = {  # official WorldModelLearnerCfg values; mug uses the root's
        "root_pos": 0.01, "root_ori": 0.02,
        "joint_pos": 0.1, "joint_vel": 3.14,
        "mug_pos": 0.01, "mug_ori": 0.02,
    }

    def __init__(self, lay, device):
        self.lay = lay
        self.dt = lay["dt"]
        d = lay["decimation"]
        self.c = (d + 1) / (2 * d)
        nq, nv = lay["nq"], lay["nv"]
        self.hq = torch.tensor(lay["hand_q"], device=device)
        self.hv = torch.tensor([nq + i for i in lay["hand_v"]], device=device)
        self.rq0, self.rv0 = lay["root_q"], nq + lay["root_v"]
        self.mq0, self.mv0 = lay["src_q"], nq + lay["src_v"]
        self.a0 = nq + nv
        self.na = lay["na"]
        self.nj = len(lay["hand_q"])
        self.pred_dim = 6 + self.nj + 6
        self.feat_dim = 3 + 6 + 3 + 3 + 2 * self.nj + 3 + 6 + 3 + 3 + 3 + 6 + self.na

    def from_full(self, full):
        rp = full[..., self.rq0:self.rq0 + 3]
        rq = full[..., self.rq0 + 3:self.rq0 + 7]
        mp = full[..., self.mq0:self.mq0 + 3]
        mq = full[..., self.mq0 + 3:self.mq0 + 7]
        # MuJoCo free joint: qvel[0:3] world linear, qvel[3:6] body-frame angular
        rv = quat_apply_inverse(rq, full[..., self.rv0:self.rv0 + 3])
        rw = full[..., self.rv0 + 3:self.rv0 + 6]
        mv = quat_apply_inverse(mq, full[..., self.mv0:self.mv0 + 3])
        mw = full[..., self.mv0 + 3:self.mv0 + 6]
        return {
            "rp": rp, "rq": rq, "rv": rv, "rw": rw,
            "qj": full[..., self.hq], "vj": full[..., self.hv],
            "mp": mp, "mq": mq, "mv": mv, "mw": mw,
            "act": full[..., self.a0:self.a0 + self.na],
        }

    def to_full(self, s, template, pid=None):
        """Inverse of from_full: write the carried state into a copy of `template`
        (…, 142) (destination mug and anything not carried are kept from it).
        Velocities go back to MuJoCo's free-joint convention (world linear,
        body angular). Differentiable."""
        full = template.clone()
        full[..., self.rq0:self.rq0 + 3] = s["rp"]
        full[..., self.rq0 + 3:self.rq0 + 7] = s["rq"]
        full[..., self.rv0:self.rv0 + 3] = quat_apply(s["rq"], s["rv"])
        full[..., self.rv0 + 3:self.rv0 + 6] = s["rw"]
        full[..., self.hq] = s["qj"]
        full[..., self.hv] = s["vj"]
        full[..., self.mq0:self.mq0 + 3] = s["mp"]
        full[..., self.mq0 + 3:self.mq0 + 7] = s["mq"]
        full[..., self.mv0:self.mv0 + 3] = quat_apply(s["mq"], s["mv"])
        full[..., self.mv0 + 3:self.mv0 + 6] = s["mw"]
        full[..., self.a0:self.a0 + self.na] = s["act"]
        if pid is not None:
            p0 = self.a0 + self.na
            full[..., p0:p0 + pid.shape[-1]] = pid
        return full

    def features(self, s):
        Rr = matrix_from_quat(s["rq"])
        Rm = matrix_from_quat(s["mq"])
        rel_p = (Rr.transpose(-1, -2) @ (s["mp"] - s["rp"]).unsqueeze(-1)).squeeze(-1)
        rel_R = Rr.transpose(-1, -2) @ Rm
        return torch.cat([
            s["rp"], rot6d(Rr), s["rv"], s["rw"], s["qj"], s["vj"],
            s["mp"], rot6d(Rm), s["mv"], s["mw"], rel_p, rot6d(rel_R), s["act"],
        ], -1)

    def _integrate_body(self, p, q, v_b, w_b, dv, dw):
        """Official GenCoordRepresentation.apply_prediction, root part."""
        c, dt = self.c, self.dt
        v_global = quat_apply(q, v_b + dv)
        w_global = quat_apply(q, w_b + dw)
        v_pose = quat_apply(q, v_b + c * dv)
        w_pose = quat_apply(q, w_b + c * dw)
        p2 = p + v_pose * dt
        q2 = quat_integrate(q, w_pose, dt)
        return p2, q2, quat_apply_inverse(q2, v_global), quat_apply_inverse(q2, w_global)

    def apply_prediction(self, s, pred, act_next):
        nj = self.nj
        dv_r, dw_r = pred[..., 0:3], pred[..., 3:6]
        dvj = pred[..., 6:6 + nj]
        dv_m, dw_m = pred[..., 6 + nj:9 + nj], pred[..., 9 + nj:12 + nj]
        rp, rq, rv, rw = self._integrate_body(s["rp"], s["rq"], s["rv"], s["rw"], dv_r, dw_r)
        mp, mq, mv, mw = self._integrate_body(s["mp"], s["mq"], s["mv"], s["mw"], dv_m, dw_m)
        return {
            "rp": rp, "rq": rq, "rv": rv, "rw": rw,
            "qj": s["qj"] + (s["vj"] + self.c * dvj) * self.dt, "vj": s["vj"] + dvj,
            "mp": mp, "mq": mq, "mv": mv, "mw": mw, "act": act_next,
        }

    def oracle_prediction(self, s, s_next):
        """Velocity deltas that make apply_prediction hit s_next's velocities
        exactly (expressed in the current body frame). Used only to test the
        integration against the simulator."""
        def body(q, v_b, w_b, q2, v2, w2):
            return (quat_apply_inverse(q, quat_apply(q2, v2)) - v_b,
                    quat_apply_inverse(q, quat_apply(q2, w2)) - w_b)
        dv_r, dw_r = body(s["rq"], s["rv"], s["rw"], s_next["rq"], s_next["rv"], s_next["rw"])
        dv_m, dw_m = body(s["mq"], s["mv"], s["mw"], s_next["mq"], s_next["mv"], s_next["mw"])
        return torch.cat([dv_r, dw_r, s_next["vj"] - s["vj"], dv_m, dw_m], -1)

    def loss_blocks(self, s):
        h = math.sqrt(0.5)
        return {
            "root_pos": s["rp"], "root_ori": matrix_from_quat(s["rq"]).flatten(-2) * h,
            "joint_pos": s["qj"], "joint_vel": s["vj"],
            "mug_pos": s["mp"], "mug_ori": matrix_from_quat(s["mq"]).flatten(-2) * h,
        }


# ---------------- model ----------------
class Normalizer(nn.Module):
    """rsl_rl EmpiricalNormalization forward: (x - mean) / (std + eps)."""

    def __init__(self, dim, eps=1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    @torch.no_grad()
    def fit(self, x):
        x = x.reshape(-1, x.shape[-1]).double()
        self.mean.copy_(x.mean(0).float())
        self.std.copy_(x.std(0, unbiased=False).float())

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)


def build_mlp(input_dim, hidden_dims, output_dim):
    """Official models.build_mlp."""
    layers, cur = [], input_dim
    for h in hidden_dims:
        layers += [nn.Linear(cur, h), nn.LayerNorm(h), nn.GELU()]
        cur = h
    layers.append(nn.Linear(cur, output_dim))
    return nn.Sequential(*layers)


class Dynamics(nn.Module):
    """n_obs > 0: the same MLP also outputs n_obs standardized observations of
    the NEXT state (FreeMusco-style extra outputs, predicted from s_t and u_t),
    used for contact forces that depend on the applied action, not on the state only."""

    def __init__(self, rep, u_dim, hidden_dims=(2048, 2048, 2048), n_obs=0):
        super().__init__()
        self.rep = rep
        self.n_obs = n_obs
        self.state_normalizer = Normalizer(rep.feat_dim)
        self.action_normalizer = Normalizer(u_dim)
        self.register_buffer("obs_mean", torch.zeros(max(n_obs, 1)))
        self.register_buffer("obs_std", torch.ones(max(n_obs, 1)))
        self.mlp = build_mlp(rep.feat_dim + u_dim, hidden_dims, rep.pred_dim + n_obs)

    @torch.no_grad()
    def fit_obs_stats(self, targets):
        t = targets.reshape(-1, targets.shape[-1]).double()
        self.obs_mean.copy_(t.mean(0).float())
        self.obs_std.copy_(t.std(0, unbiased=False).float().clamp(min=1e-3))

    def step(self, s, u, act_next):
        """-> (next state, standardized obs prediction or None)"""
        x = torch.cat([self.state_normalizer(self.rep.features(s)), self.action_normalizer(u)], -1)
        out = self.mlp(x)
        nxt = self.rep.apply_prediction(s, out[..., :self.rep.pred_dim], act_next)
        return nxt, (out[..., self.rep.pred_dim:] if self.n_obs else None)

    def forward(self, s, u, act_next):
        return self.step(s, u, act_next)[0]

    def obs_denorm(self, o):
        return o * self.obs_std + self.obs_mean


def build_optimizers(model, lr, betas=(0.9, 0.999)):
    """Official learner_utils.build_optimizers: Muon on hidden Linear weights
    (between the first and last Linear of each Sequential), Adam on the rest."""
    hidden = [lin.weight for seq in model.modules() if isinstance(seq, nn.Sequential)
              for lin in [m for m in seq if isinstance(m, nn.Linear)][1:-1]]
    ids = {id(p) for p in hidden}
    rest = [p for p in model.parameters() if id(p) not in ids]
    opts = [torch.optim.Adam(rest, lr=lr, betas=betas)]
    if hidden:
        opts.append(torch.optim.Muon(hidden, lr=lr, adjust_lr_fn="match_rms_adamw"))
    return opts


def state_loss(rep, pred_seq, gt_seq):
    """Official WorldModelLearner._compute_losses over stacked states."""
    pb = rep.loss_blocks(pred_seq)
    gb = rep.loss_blocks(gt_seq)
    names = list(pb)
    group_mse = torch.stack([((pb[k] - gb[k]) ** 2).mean() for k in names])
    widths = torch.tensor([float(pb[k].shape[-1]) for k in names], device=group_mse.device)
    weights = torch.tensor([1.0 / rep.LOSS_NOMINAL[k] ** 2 for k in names], device=group_mse.device)
    return (group_mse * weights * widths).sum() / widths.sum(), dict(zip(names, group_mse))


def stack_states(states):
    return {k: torch.stack([s[k] for s in states], 1) for k in states[0]}


def rollout(model, s0, u, act_next, with_obs=False):
    """Autoregressive rollout over u (B, K, U); act_next (B, K, na).
    with_obs: also return the stacked standardized obs outputs (B, K, n_obs)."""
    s, out, obs = s0, [], []
    for k in range(u.shape[1]):
        s, o = model.step(s, u[:, k], act_next[:, k])
        out.append(s)
        obs.append(o)
    states = stack_states(out)
    if not with_obs:
        return states
    return states, (torch.stack(obs, 1) if obs[0] is not None else None)


# ---------------- learned observation head ----------------
def forces30(tip20, pad20):
    """(…, 20) tip + (…, 20) pad in PhysGraph layout (5 x [f3, |f|]) -> (…, 30) vectors only."""
    t = tip20.reshape(*tip20.shape[:-1], 5, 4)[..., :3]
    p = pad20.reshape(*pad20.shape[:-1], 5, 4)[..., :3]
    return torch.cat([t.flatten(-2), p.flatten(-2)], -1)


def forces_to_obs(f30):
    """(…, 30) -> (tip20, pad20) with magnitudes, PhysGraph layout."""
    f = f30.reshape(*f30.shape[:-1], 2, 5, 3)
    o = torch.cat([f, f.norm(dim=-1, keepdim=True)], -1)
    return o[..., 0, :, :].flatten(-2), o[..., 1, :, :].flatten(-2)


class ObsHead(nn.Module):
    """Learned observation o_hat = g(s_hat) of SuperMimic eq. (1), for the
    quantities our analytic observation cannot produce: distal-phalanx tip
    force (world) and pad-side force (distal body frame), 5 x 3 each.
    Trained on rolled-out states with a per-component standardized MSE."""

    def __init__(self, rep, hidden_dims=(512, 512, 512), out_dim=30, u_dim=0):
        super().__init__()
        self.rep = rep
        self.u_dim = u_dim  # > 0: also condition on the action u_t that produced s_{t+1}
        self.in_normalizer = Normalizer(rep.feat_dim)
        self.u_normalizer = Normalizer(max(u_dim, 1))
        self.register_buffer("out_mean", torch.zeros(out_dim))
        self.register_buffer("out_std", torch.ones(out_dim))
        self.mlp = build_mlp(rep.feat_dim + u_dim, hidden_dims, out_dim)

    @torch.no_grad()
    def fit_stats(self, feats, targets, u=None):
        self.in_normalizer.fit(feats)
        if self.u_dim:
            self.u_normalizer.fit(u)
        t = targets.reshape(-1, targets.shape[-1]).double()
        self.out_mean.copy_(t.mean(0).float())
        self.out_std.copy_(t.std(0, unbiased=False).float().clamp(min=1e-3))

    def forward_norm(self, s, u=None):
        x = self.in_normalizer(self.rep.features(s))
        if self.u_dim:
            x = torch.cat([x, self.u_normalizer(u)], -1)
        return self.mlp(x)

    def forward(self, s, u=None):
        return self.forward_norm(s, u) * self.out_std + self.out_mean

    def loss(self, s, targets, u=None):
        return ((self.forward_norm(s, u) - (targets - self.out_mean) / self.out_std) ** 2).mean()
