"""Pad-side contact force on the distal phalanges, from mujoco_warp contacts.

PhysGraph's tip_force (env obs) is cfrc_ext[body, 3:6] of each distal
phalanx: the net force of every contact on that body, in world axes. This
module splits it by where each contact sits on the body:

  bone axis a  = body origin -> tip site (body frame)
  pad dir   n  = pad site offset (assets/retargeted/pad_sites_local.json)
                 made perpendicular to a
  pad contact iff (contact point in body frame) . n > 0

Output per finger: pad-side force in the DISTAL BODY frame + its magnitude
(20 dims, PhysGraph's tip_force layout), plus the world-frame total of all
contacts per finger, used to check the extraction against cfrc_ext.
Validated on CPU MuJoCo by scripts/sm_probe_pad_force.py (sum of per-contact
forces == cfrc_ext exactly; force on geom[1]'s body is +f, geom[0]'s is -f).
Must be called right after env.step: the env's reward runs mjw.forward on the
post-step state, so contacts and efc forces belong to that state.
"""
import json
import os

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from lib.sm.world_model import quat_apply_inverse

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PadForce:
    def __init__(self, env):
        m = env.mj_model
        dev = env.device
        self.tip_bodies = [int(b) for b in env.tip_body_ids]
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in self.tip_bodies]
        pj = json.load(open(os.path.join(_REPO, "assets/retargeted/pad_sites_local.json")))["bodies"]
        normals = []
        for nm in names:
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, pj[nm]["tip_site"])
            a = np.array(m.site_pos[sid], dtype=np.float64)
            a /= np.linalg.norm(a)
            pad = np.array(pj[nm]["pad_local_m"], dtype=np.float64)
            n = pad - pad.dot(a) * a
            normals.append(n / np.linalg.norm(n))
        self.normal = torch.tensor(np.array(normals), dtype=torch.float32, device=dev)  # (5, 3)
        self.geom_body = torch.tensor(m.geom_bodyid, dtype=torch.long, device=dev)
        b2f = torch.full((int(m.nbody),), -1, dtype=torch.long, device=dev)
        for k, b in enumerate(self.tip_bodies):
            b2f[b] = k
        self.body2finger = b2f
        self.dev = dev
        assert hasattr(mjw, "contact_force"), "this mujoco_warp has no contact_force; update or use the efc fallback"

    def __call__(self, env):
        """Returns (pad_obs (E, 20) in distal body frames, total_world (E, 5, 3))."""
        E, dev = env.num_envs, self.dev
        pad = torch.zeros(E * 5, 3, device=dev)
        tot = torch.zeros(E * 5, 3, device=dev)
        d = env.data
        nacon = int(wp.to_torch(d.nacon)[0].item())
        if nacon > 0:
            ids = wp.from_torch(torch.arange(nacon, dtype=torch.int32, device=dev))
            out = wp.zeros(nacon, dtype=wp.spatial_vector, device=str(dev))
            mjw.contact_force(env.model, d, ids, True, out)
            f = wp.to_torch(out)[:, :3]                         # world, on geom[1]'s body
            geom = wp.to_torch(d.contact.geom)[:nacon].long()
            wid = wp.to_torch(d.contact.worldid)[:nacon].long()
            pos = wp.to_torch(d.contact.pos)[:nacon]
            xpos = wp.to_torch(d.xpos)
            xquat = wp.to_torch(d.xquat)
            b = self.geom_body[geom]                            # (nacon, 2)
            for side, sgn in ((0, -1.0), (1, 1.0)):
                k = self.body2finger[b[:, side]]
                sel = (k >= 0) & (b[:, 0] != b[:, 1])
                if not bool(sel.any()):
                    continue
                bb, w, kk = b[sel, side], wid[sel], k[sel]
                q = xquat[w, bb]
                p_loc = quat_apply_inverse(q, pos[sel] - xpos[w, bb])
                fw = sgn * f[sel]
                f_loc = quat_apply_inverse(q, fw)
                is_pad = (p_loc * self.normal[kk]).sum(-1) > 0
                idx = w * 5 + kk
                tot.index_add_(0, idx, fw)
                pad.index_add_(0, idx[is_pad], f_loc[is_pad])
        pad = pad.view(E, 5, 3)
        pad_obs = torch.cat([pad, pad.norm(dim=-1, keepdim=True)], -1).reshape(E, 20)
        return pad_obs, tot.view(E, 5, 3)
