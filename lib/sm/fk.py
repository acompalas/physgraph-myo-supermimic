"""A3: differentiable forward kinematics of the MuJoCo model, in torch.

A copy of MuJoCo's mj_kinematics (engine_core_smooth.c) for the joint types
in our scene (free, hinge, slide, ball), batched over the leading dim:

  for each body b (parents before children):
    pos  = xpos[parent] + R(xquat[parent]) @ body_pos[b]
    quat = xquat[parent] * body_quat[b]
    for each joint j of b, in order:
      free : pos = qpos[a:a+3], quat = normalize(qpos[a+3:a+7])
      slide: pos += R(quat) @ axis_j * (qpos[a] - qpos0[a])
      hinge: anchor = R(quat) @ jnt_pos_j + pos
             quat   = quat * axis_angle(axis_j, qpos[a] - qpos0[a])
             pos    = anchor - R(quat) @ jnt_pos_j
      ball : same as hinge with the local quat = normalize(qpos[a:a+4])
    xquat[b] = normalize(quat), xpos[b] = pos
  site_xpos = xpos[site_body] + R(xquat[site_body]) @ site_pos

Everything the reward and termination read (tracked points from xpos /
site_xpos, tip and pad body orientations from xquat) follows from these.
"""
import torch

from lib.sm.world_model import quat_apply, quat_mul

JNT_FREE, JNT_BALL, JNT_SLIDE, JNT_HINGE = 0, 1, 2, 3


def _normalize(q):
    return q / torch.norm(q, dim=-1, keepdim=True)


def _axis_angle_quat(axis, angle):
    h = 0.5 * angle.unsqueeze(-1)
    return torch.cat([torch.cos(h), torch.sin(h) * axis], -1)


class TorchFK:
    def __init__(self, mj_model, device, bodies=None):
        """bodies: optional list of body ids to keep (their ancestors are
        computed too); default all bodies."""
        m = mj_model
        t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)  # noqa: E731
        self.nbody = int(m.nbody)
        need = set(range(1, self.nbody)) if bodies is None else set()
        if bodies is not None:
            for b in bodies:
                while b > 0:
                    need.add(int(b))
                    b = int(m.body_parentid[b])
        self.order = sorted(need)  # MuJoCo body ids are parent-before-child
        self.parent = {b: int(m.body_parentid[b]) for b in self.order}
        self.body_pos = {b: t(m.body_pos[b]) for b in self.order}
        self.body_quat = {b: t(m.body_quat[b]) for b in self.order}
        self.joints = {}
        for b in self.order:
            js = []
            for j in range(int(m.body_jntadr[b]), int(m.body_jntadr[b]) + int(m.body_jntnum[b])):
                js.append((int(m.jnt_type[j]), int(m.jnt_qposadr[j]), t(m.jnt_axis[j]), t(m.jnt_pos[j]),
                           float(m.qpos0[m.jnt_qposadr[j]])))
            self.joints[b] = js
        self.site_body = torch.tensor(m.site_bodyid, device=device)
        self.site_pos = t(m.site_pos)
        self.device = device

    def __call__(self, qpos):
        """qpos (..., nq) -> xpos (..., nbody, 3), xquat (..., nbody, 4).
        Bodies not computed are left at the origin / identity."""
        lead = qpos.shape[:-1]
        zero = qpos.new_zeros(*lead, 3)
        ident = qpos.new_zeros(*lead, 4)
        ident[..., 0] = 1.0
        xpos = [zero] * self.nbody
        xquat = [ident] * self.nbody
        for b in self.order:
            p = self.parent[b]
            pos = xpos[p] + quat_apply(xquat[p], self.body_pos[b].expand(*lead, 3))
            quat = quat_mul(xquat[p], self.body_quat[b].expand(*lead, 4))
            for jtype, a, axis, jpos, q0 in self.joints[b]:
                if jtype == JNT_FREE:
                    pos = qpos[..., a:a + 3]
                    quat = _normalize(qpos[..., a + 3:a + 7])
                elif jtype == JNT_SLIDE:
                    pos = pos + quat_apply(quat, axis.expand(*lead, 3)) * (qpos[..., a:a + 1] - q0)
                else:
                    jp = jpos.expand(*lead, 3)
                    anchor = quat_apply(quat, jp) + pos
                    if jtype == JNT_HINGE:
                        qloc = _axis_angle_quat(axis.expand(*lead, 3), qpos[..., a] - q0)
                    else:
                        qloc = _normalize(qpos[..., a:a + 4])
                    quat = quat_mul(quat, qloc)
                    pos = anchor - quat_apply(quat, jp)
            xpos[b] = pos
            xquat[b] = _normalize(quat)
        return torch.stack(xpos, -2), torch.stack(xquat, -2)

    def site_xpos(self, xpos, xquat, site_ids):
        sb = self.site_body[site_ids]
        q = xquat[..., sb, :]
        return xpos[..., sb, :] + quat_apply(q, self.site_pos[site_ids].expand(*q.shape[:-1], 3))
