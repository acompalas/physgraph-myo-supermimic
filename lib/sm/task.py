"""A4: the task in torch. Reward, actor observation and soft termination of
MyoHandPourEnv, computed from full states (lib/sm/state.py layout) and the
demo frame, differentiably (FK from lib/sm/fk.py replaces MuJoCo's xpos).

Faithful copy of MyoHandPourEnv._compute_reward / _compute_obs (same terms,
weights, env flags and demo tables, read from the live env), except:
  - power (finger, needs muscle forces) is not computed: dropped by decision
    (saturated at ~0 for MyoHand); finger_tip_force reward and contact_fail
    are asserted off (weight 0 / flag off), as in Run N.
  - angles use 2*atan2(|v|, |w|) instead of 2*acos(|w|): same value, finite
    gradient at zero angle.
  - termination is also returned in SuperMimic's soft form
    d = sigmoid((z - 1) / softness), z = max over normalized errors
    (tracking errors gated by the 8-step grace, velocity blow-ups not),
    so hard termination is exactly z > 1.
  - contact forces for the observation are an input (real in the env,
    predicted by the world model's force head in imagination).

Construct after the env has taken one step (it computes some tables lazily).
Frames: reward and termination at a transition use the frame of the NEXT
state (env cur_idx = progress_buf after the step); the observation of a
state uses that state's own frame.
"""
import math
import os

import numpy as np
import torch

from envs.mjwarp_myohand_env import rotmat_to_quat
from lib.sm.fk import TorchFK
from lib.sm.world_model import quat_apply, quat_conj, quat_mul


def _angle(q):
    return 2.0 * torch.atan2(q[..., 1:].norm(dim=-1), q[..., 0].abs())


def scale_factor_at(step):
    """Env termination tightening (MYOHAND_TIGHTEN_*), at env step counter `step`."""
    tm = os.environ.get("MYOHAND_TIGHTEN_METHOD", "exp_decay")
    tf = float(os.environ.get("MYOHAND_TIGHTEN_FACTOR", "0.7"))
    ts = float(os.environ.get("MYOHAND_TIGHTEN_STEPS", "3200"))
    ls = step + int(os.environ.get("MYOHAND_TIGHTEN_OFFSET", "0"))
    if tm == "None":
        return 1.0
    if tm == "const":
        return tf
    if tm == "linear_decay":
        return 1 - (1 - tf) / ts * min(ls, ts)
    if tm == "exp_decay":
        return (np.e * 2) ** (-1 * ls / ts) * (1 - tf) + tf
    raise NotImplementedError(tm)


class TorchTask:
    def __init__(self, env):
        assert hasattr(env, "_tip_q_tgt") and hasattr(env, "_jq_M"), "step the env once before building TorchTask"
        assert float(os.environ.get("MYOHAND_FORCE_REWARD_W", "0.0")) == 0.0, "tip-force reward not supported"
        assert os.environ.get("MYOHAND_CONTACT_FAIL", "0") != "1", "contact_fail not supported"
        self.env = env
        e = env
        self.dev = e.device
        self.dt = e.dt
        L = e.mj_model
        self.nq, self.nv, self.na = int(L.nq), int(L.nv), int(L.na)
        self.fk = TorchFK(L, self.dev)
        t = lambda x: torch.as_tensor(x, device=self.dev)  # noqa: E731
        self.body_ids = t(e.body_ids[1:]).long()
        self.site_ids = t(e.site_ids[1:]).long().clamp(min=0)
        self.is_site = e.is_site_mask.to(self.dev)
        self.tip_ids = t(e.tip_body_ids).long()
        self.hq, self.hv = t(e.dof_adrs).long(), t(e.dof_veladrs).long()
        self.rq, self.rv = int(e.root_adr), int(e.root_dof_adr)
        self.mq, self.mv = int(e.src_adr), int(e.src_dof_adr)
        self.dq, self.dv = int(e.dst_adr), int(e.dst_dof_adr)
        self.weight_idx = {k: [i - 1 for i in v] for k, v in e.weight_idx.items()}
        self.seq_len = e.seq_len
        # reward constants (read, not re-derived)
        self.tr_w = float(os.environ.get("MYOHAND_TIPROT_W", "1.0"))
        self.tr_k = float(os.environ.get("MYOHAND_TIPROT_K", "2.0"))
        self.jq_w = float(os.environ.get("MYOHAND_JQ_W", "1.0"))
        self.jq_lam = float(os.environ.get("MYOHAND_JQ_LAMBDA", "10.0"))
        self.jq_lamw = float(os.environ.get("MYOHAND_JQ_LAMBDA_WIDE", "2.0"))
        self.jq_mode_g = os.environ.get("MYOHAND_JQ_MODE", "") == "g"
        self.jq_gdz = math.radians(float(os.environ.get("MYOHAND_JQ_DEADZONE_DEG", "6.0")))
        self.lim_w = float(os.environ.get("MYOHAND_LIM_W", "2.0"))
        self.lim_cap = float(os.environ.get("MYOHAND_LIM_CAP", "0.75"))
        self.ffs_w = float(os.environ.get("MYOHAND_FFSP_W", "2.0"))
        self.ffs_cap = float(os.environ.get("MYOHAND_FFSP_CAP_MM", "30")) / 1000.0
        self.pad_w = float(e._pad_w)
        self.obs_act = bool(e._obs_act)
        self.obs_pad = bool(e._obs_pad)
        self.obs_padf = bool(getattr(e, "_obs_padf", False))
        self._fk_call = self.fk
        self.pad_body_ids = None
        if e.demo_pad_pos is not None:
            import mujoco
            self.pad_body_ids = t([mujoco.mj_name2id(L, mujoco.mjtObj.mjOBJ_BODY, n) for n in e._pad_body_names]).long()

    # ---------- state helpers ----------
    def _split(self, s):
        nq, nv = self.nq, self.nv
        qpos, qvel = s[..., :nq], s[..., nq:nq + nv]
        act = s[..., nq + nv:nq + nv + self.na]
        return qpos, qvel, act

    def kin(self, s):
        """FK quantities of a full state: tracked points (…,20,3), body xpos/xquat."""
        qpos = s[..., :self.nq]
        xpos, xquat = self._fk_call(qpos)
        sites = self.fk.site_xpos(xpos, xquat, self.site_ids)
        jp = torch.where(self.is_site[:, None], sites, xpos[..., self.body_ids, :])
        return {"joints_pos": jp, "xpos": xpos, "xquat": xquat}

    def _bodies(self, qpos, qvel):
        rp, rquat = qpos[..., self.rq:self.rq + 3], qpos[..., self.rq + 3:self.rq + 7]
        mp, mquat = qpos[..., self.mq:self.mq + 3], qpos[..., self.mq + 3:self.mq + 7]
        return {
            "rp": rp, "rquat": rquat, "rv": qvel[..., self.rv:self.rv + 3],
            "rw": quat_apply(rquat, qvel[..., self.rv + 3:self.rv + 6]),
            "mp": mp, "mquat": mquat, "mv": qvel[..., self.mv:self.mv + 3],
            "mw": quat_apply(mquat, qvel[..., self.mv + 3:self.mv + 6]),
        }

    def _pad_world(self, k, frame_shape):
        e = self.env
        q = k["xquat"][..., self.pad_body_ids, :]
        p = k["xpos"][..., self.pad_body_ids, :]
        return p + quat_apply(q, e._pad_local.expand(*q.shape[:-1], 3))

    # ---------- reward + termination ----------
    def reward(self, s_next, s_prev, u, frame, scale_factor, k_next=None, k_prev=None):
        """Transition reward at s_next (frame = s_next's frame, env cur_idx).
        Returns (reward, terms, z_track, z_err)."""
        e = self.env
        qpos, qvel, _ = self._split(s_next)
        b = self._bodies(qpos, qvel)
        k = k_next if k_next is not None else self.kin(s_next)
        kp = k_prev if k_prev is not None else self.kin(s_prev)
        jp = k["joints_pos"]
        jv = (jp - kp["joints_pos"]) / self.dt
        idx = frame
        nxt = torch.clamp(idx + 1, max=self.seq_len - 1)
        t_ep, t_eq = e.demo_opt_wrist_pos[idx], e.demo_opt_wrist_quat[idx]
        t_jp = e.demo_target_joints_pos[idx]
        t_op = e.demo_src_obj_traj[idx, :3, 3]
        t_oq = rotmat_to_quat(e.demo_src_obj_traj[idx, :3, :3])
        t_ev = (e.demo_opt_wrist_pos[nxt] - t_ep) / self.dt
        t_ew = e.demo_wrist_ang_vel[idx]
        t_jv = (e.demo_target_joints_pos[nxt] - t_jp) / self.dt
        t_ov = (e.demo_src_obj_traj[nxt, :3, 3] - t_op) / self.dt
        t_ow = e.demo_src_obj_ang_vel[idx]

        d_ep = (t_ep - b["rp"]).norm(dim=-1)
        d_jp = (t_jp - jp).norm(dim=-1)
        dist = {key: d_jp[..., self.weight_idx[key]].mean(-1) for key in
                ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip", "level_1_joints", "level_2_joints")}
        T = {}
        T["reward_eef_pos"] = torch.exp(-40 * d_ep)
        T["reward_eef_rot"] = torch.exp(-_angle(quat_mul(t_eq, quat_conj(b["rquat"]))))
        T["reward_thumb_tip_pos"] = torch.exp(-100 * dist["thumb_tip"])
        T["reward_index_tip_pos"] = torch.exp(-90 * dist["index_tip"])
        T["reward_middle_tip_pos"] = torch.exp(-80 * dist["middle_tip"])
        T["reward_pinky_tip_pos"] = torch.exp(-60 * dist["pinky_tip"])
        T["reward_ring_tip_pos"] = torch.exp(-60 * dist["ring_tip"])
        T["reward_level_1_pos"] = torch.exp(-50 * dist["level_1_joints"])
        T["reward_level_2_pos"] = torch.exp(-40 * dist["level_2_joints"])
        d_op = (t_op - b["mp"]).norm(dim=-1)
        a_or = _angle(quat_mul(t_oq, quat_conj(b["mquat"])))
        T["reward_obj_pos"] = torch.exp(-80 * d_op)
        T["reward_obj_rot"] = torch.exp(-3 * a_or)
        T["reward_eef_vel"] = torch.exp(-(t_ev - b["rv"]).abs().mean(-1))
        T["reward_eef_ang_vel"] = torch.exp(-(t_ew - b["rw"]).abs().mean(-1))
        T["reward_joints_vel"] = torch.exp(-(t_jv - jv).abs().mean(-1).mean(-1))
        T["reward_obj_vel"] = torch.exp(-(t_ov - b["mv"]).abs().mean(-1))
        T["reward_obj_ang_vel"] = torch.exp(-(t_ow - b["mw"]).abs().mean(-1))
        # fingertip orientation
        tq = k["xquat"][..., self.tip_ids, :]
        tt = e._tip_q_tgt[torch.clamp(idx, max=e._tip_q_tgt.shape[0] - 1)]
        T["reward_tip_rot"] = torch.exp(-self.tr_k * _angle(quat_mul(tt, quat_conj(tq)))).mean(-1)
        # pad position
        if self.pad_body_ids is not None:
            pd = (e.demo_pad_pos[idx] - self._pad_world(k, idx.shape)).norm(dim=-1)
            T["reward_pad"] = torch.exp(-e._pad_k * pd).mean(-1)
        else:
            T["reward_pad"] = torch.zeros_like(d_ep)
        # joint angles
        q_cur = qpos[..., self.hq]
        q_tgt = e.demo_opt_dof_pos[torch.clamp(idx, max=e.demo_opt_dof_pos.shape[0] - 1)]
        if self.jq_mode_g:
            gex = torch.clamp((q_cur - q_tgt).abs() - self.jq_gdz, min=0.0)
            T["reward_joint_q"] = torch.exp(-self.jq_lam * (gex * e._jq_gwvec).sum(-1))
        else:
            ex = torch.clamp((q_cur - q_tgt).abs() - e._jq_dz, min=0.0) @ e._jq_M
            T["reward_joint_q"] = ((0.5 * torch.exp(-self.jq_lam * ex) + 0.5 * torch.exp(-self.jq_lamw * ex)) * e._jq_gw).sum(-1)
        over = torch.clamp(e._lim_lo - q_cur, min=0.0) + torch.clamp(q_cur - e._lim_hi, min=0.0)
        T["reward_limit"] = 1.0 - torch.clamp(over.sum(-1) / self.lim_cap, 0.0, 1.0)
        # finger spacing / crossover
        J, Tg = jp, t_jp
        short = torch.zeros_like(d_ep)
        for a in (1, 2, 3):
            for l1 in (1, 2, 3):
                for l2 in (1, 2, 3):
                    d = (J[..., 4 * a + l1, :] - J[..., 4 * (a + 1) + l2, :]).norm(dim=-1)
                    dd = (Tg[..., 4 * a + l1, :] - Tg[..., 4 * (a + 1) + l2, :]).norm(dim=-1)
                    mg = torch.clamp(e._ffs_dfrac * dd, max=min(e._ffs_margin[l1 - 1], e._ffs_margin[l2 - 1]))
                    short = short + torch.clamp(mg - d, min=0.0)
        uc = J[..., 16, :] - J[..., 4, :]
        uc = uc / uc.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ut = Tg[..., 16, :] - Tg[..., 4, :]
        ut = ut / ut.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        for a in (1, 2, 3):
            for l in (1, 2, 3):  # noqa: E741
                g = ((J[..., 4 * (a + 1) + l, :] - J[..., 4 * a + l, :]) * uc).sum(-1)
                gd = ((Tg[..., 4 * (a + 1) + l, :] - Tg[..., 4 * a + l, :]) * ut).sum(-1)
                short = short + torch.clamp(e._ffs_gfrac * gd - g, min=0.0)
        T["reward_ffspace"] = 1.0 - torch.clamp(short / self.ffs_cap, 0.0, 1.0)
        # wrist power (u: world force, body torque; qvel root: world linear, body angular)
        wpow = ((u[..., 0:3] * qvel[..., self.rv:self.rv + 3]).sum(-1).abs()
                + (u[..., 3:6] * qvel[..., self.rv + 3:self.rv + 6]).sum(-1).abs())
        T["reward_wrist_power"] = torch.exp(-2 * wpow)

        r = (0.1 * T["reward_eef_pos"] + 0.6 * T["reward_eef_rot"]
             + 0.9 * T["reward_thumb_tip_pos"] + 0.8 * T["reward_index_tip_pos"]
             + 0.75 * T["reward_middle_tip_pos"] + 0.6 * T["reward_pinky_tip_pos"]
             + 0.6 * T["reward_ring_tip_pos"] + 0.5 * T["reward_level_1_pos"]
             + 0.3 * T["reward_level_2_pos"] + 5.0 * T["reward_obj_pos"]
             + 1.0 * T["reward_obj_rot"] + 0.1 * T["reward_eef_vel"]
             + 0.05 * T["reward_eef_ang_vel"] + 0.1 * T["reward_joints_vel"]
             + 0.1 * T["reward_obj_vel"] + 0.1 * T["reward_obj_ang_vel"]
             + self.tr_w * T["reward_tip_rot"] + self.pad_w * T["reward_pad"]
             + self.jq_w * T["reward_joint_q"] + self.lim_w * T["reward_limit"]
             + self.ffs_w * T["reward_ffspace"] + 0.5 * T["reward_wrist_power"])

        sf = scale_factor
        z_track = torch.stack([
            d_op / (0.02 / 0.343 * sf ** 3),
            dist["thumb_tip"] / (0.04 / 0.7 * sf), dist["index_tip"] / (0.045 / 0.7 * sf),
            dist["middle_tip"] / (0.05 / 0.7 * sf), dist["pinky_tip"] / (0.06 / 0.7 * sf),
            dist["ring_tip"] / (0.06 / 0.7 * sf), dist["level_1_joints"] / (0.07 / 0.7 * sf),
            dist["level_2_joints"] / (0.08 / 0.7 * sf),
            (a_or / math.pi * 180) / (30 / 0.343 * sf ** 3),
        ], -1).amax(-1)
        z_err = torch.stack([
            b["rv"].norm(dim=-1) / 100, b["rw"].norm(dim=-1) / 200, jv.norm(dim=-1).mean(-1) / 100,
            qvel[..., self.hv].abs().mean(-1) / 200, b["mv"].norm(dim=-1) / 100, b["mw"].norm(dim=-1) / 200,
        ], -1).amax(-1)
        return r, T, z_track, z_err

    @staticmethod
    def termination(z_track, z_err, steps_since_reset, softness=0.05):
        """(soft d in [0,1], hard bool). Hard equals the env's failure_buf.
        The env gates tracking failures by running_progress_buf >= 8."""
        z = torch.maximum(torch.where(steps_since_reset >= 8, z_track, torch.zeros_like(z_track)), z_err)
        return torch.sigmoid((z - 1.0) / softness), z > 1.0

    # ---------- observation ----------
    def compile(self):
        """torch.compile the FK, reward and obs (C2 profile: FK 73 -> 3 ms, imagined
        step 253 -> 35 ms at batch 1024 on the 2060; ~6 min to compile once)."""
        self._fk_call = torch.compile(self.fk.__call__, dynamic=False)
        self.reward = torch.compile(self.reward, dynamic=False)
        self.obs = torch.compile(self.obs, dynamic=False)
        return self

    def obs(self, s, s_prev, frame, tip_force, k=None, k_prev=None, pad_force=None):
        """Actor observation of state s (its own frame), PhysGraph layout.
        tip_force: (…, 20) cfrc-style distal force + norm (real or predicted)."""
        e = self.env
        qpos, qvel, act = self._split(s)
        b = self._bodies(qpos, qvel)
        k = k if k is not None else self.kin(s)
        kp = k_prev if k_prev is not None else self.kin(s_prev)
        jp = k["joints_pos"]
        jv = (jp - kp["joints_pos"]) / self.dt
        lead = qpos.shape[:-1]
        z = lambda n: qpos.new_zeros(*lead, n)  # noqa: E731
        dof = qpos[..., self.hq]
        rh_proprio = torch.cat([dof, torch.cos(dof), torch.sin(dof), z(3), b["rquat"], b["rv"], b["rw"]], -1)
        com = quat_apply(b["mquat"], e.src_com.expand(*lead, 3)) + b["mp"] - b["rp"]
        rh_priv = torch.cat([qvel[..., self.hv], b["mp"] - b["rp"], b["mquat"], b["mv"], b["mw"], com,
                             e.src_weight.reshape(1).expand(*lead, 1), tip_force], -1)
        dp, dquat = qpos[..., self.dq:self.dq + 3], qpos[..., self.dq + 3:self.dq + 7]
        dvel = qvel[..., self.dv:self.dv + 3]
        dw = quat_apply(dquat, qvel[..., self.dv + 3:self.dv + 6])
        dcom = quat_apply(dquat, e.dst_com.expand(*lead, 3)) + dp - b["rp"]
        lh_priv = torch.cat([z(23), dp - b["rp"], dquat, dvel, dw, dcom,
                             e.dst_weight.reshape(1).expand(*lead, 1), z(20)], -1)

        idx = frame
        t_wp, t_wq = e.demo_opt_wrist_pos[idx], e.demo_opt_wrist_quat[idx]
        t_wv, t_ww = e.demo_wrist_vel[idx], e.demo_wrist_ang_vel[idx]
        t_jp, t_jv = e.demo_target_joints_pos[idx], e.demo_joints_vel[idx]
        t_op = e.demo_src_obj_traj[idx, :3, 3]
        t_oq = rotmat_to_quat(e.demo_src_obj_traj[idx, :3, :3])
        t_ov, t_ow = e.demo_src_obj_vel[idx], e.demo_src_obj_ang_vel[idx]
        all_j = torch.cat([b["rp"].unsqueeze(-2), jp], -2)
        o2j = (b["mp"].unsqueeze(-2) - all_j).norm(dim=-1)
        td = e.rh_demo["tips_distance"].to(self.dev)[torch.clamp(idx, max=e.rh_demo["tips_distance"].shape[0] - 1)]
        rh_target = torch.cat([
            t_wp - b["rp"], t_wv, t_wv - b["rv"], t_wq, quat_mul(b["rquat"], quat_conj(t_wq)), t_ww, t_ww - b["rw"],
            (t_jp - jp).flatten(-2), t_jv.flatten(-2), (t_jv - jv).flatten(-2),
            t_op - b["mp"], t_ov, t_ov - b["mv"], t_oq, quat_mul(b["mquat"], quat_conj(t_oq)), t_ow, t_ow - b["mw"],
            o2j, td, e.src_mug_bps.expand(*lead, -1),
        ], -1)
        lh_target = torch.cat([
            z(3), z(3), z(3), z(4), z(4), z(3), z(3), z(60), z(60), z(60),
            e.lh_target_pos.expand(*lead, 3) - dp, e.demo_dst_obj_vel.expand(*lead, 3), e.demo_dst_obj_vel - dvel,
            e.lh_target_quat.expand(*lead, 4), quat_mul(dquat, quat_conj(e.lh_target_quat.expand(*lead, 4))),
            e.demo_dst_obj_ang_vel.expand(*lead, 3), e.demo_dst_obj_ang_vel - dw,
            z(21), z(5), e.dst_mug_bps.expand(*lead, -1),
        ], -1)
        proprio = torch.cat([rh_proprio, torch.zeros_like(rh_proprio)], -1)
        if self.obs_act:
            proprio = torch.cat([proprio, act], -1)
        if self.obs_pad:
            proprio = torch.cat([proprio, (e.demo_pad_pos[idx] - self._pad_world(k, idx.shape)).flatten(-2)], -1)
        if self.obs_padf:
            proprio = torch.cat([proprio, pad_force], -1)
        return {"proprioception": proprio, "privileged": torch.cat([rh_priv, lh_priv], -1),
                "target": torch.cat([rh_target, lh_target], -1)}
