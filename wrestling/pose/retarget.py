"""Video keypoints -> Unitree G1 reference motion.

Stages:
  lift        pinhole lift of the MediaPipe 2D keypoints, root depth from apparent body size, per-joint
              depth from the MediaPipe world landmarks; left/right label swaps fixed over time
  ground      robust ground-plane fit to the lowest contact candidates, z = 0 on the mat, facing +x
  scale       uniform human -> G1 scale from leg length
  ik          per-frame keypoint and bone-direction IK with joint limits and a free root, soft contact
              targets from the 2D contact labels, soles flat where planted and the CoM inside the
              support polygon on feet-only frames; low-confidence limbs follow the mirrored visible limb
  ground fix  lowest point on the mat whenever the video shows a contact, no penetration
  gate        2D reprojection error through a fitted camera, contact agreement, limb directions,
              sustained knee/shank errors, floor penetration and joint acceleration

All thresholds are global defaults; nothing is tuned per clip. If the gate fails, a few
variants (camera FOV, smoothing, depth gain) are tried and the best one is kept.
"""

from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from ..robot import build_model

NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 11, 12, 13, 14, 15, 16
L_IDX, R_IDX = 19, 20
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN, L_HEEL, R_HEEL, L_TOE, R_TOE = 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
LEG_L, LEG_R = [L_HIP, L_KN, L_AN, L_HEEL, L_TOE], [R_HIP, R_KN, R_AN, R_HEEL, R_TOE]
ARM_L, ARM_R = [L_SH, L_EL, L_WR, 17, L_IDX, 21], [R_SH, R_EL, R_WR, 18, R_IDX, 22]
BONES = [
    (L_HIP, L_KN),
    (L_KN, L_AN),
    (R_HIP, R_KN),
    (R_KN, R_AN),
    (L_SH, L_EL),
    (L_EL, L_WR),
    (R_SH, R_EL),
    (R_EL, R_WR),
    (L_SH, L_HIP),
    (R_SH, R_HIP),
]
GROUPS = {
    "feet": [L_HEEL, L_TOE, R_HEEL, R_TOE],
    "knees": [L_KN, R_KN],
    "hands": [L_WR, R_WR, L_IDX, R_IDX],
    "hips": [L_HIP, R_HIP],
}
DEFAULTS = dict(
    w_depth=0.3,
    foot_fs_lo=0.25,
    foot_fs_span=0.30,
    foot_fs_floor=0.05,
    kp_sigma=0.5,
    q_sigma=0.5,
    depth_gain=1.0,
    w_pos=1.0,
    w_dir=0.3,
    w_reg=0.05,
    deswap=True,
    w_contact=2.0,
    w_hand=6.0,
    w_pen=30.0,
    w_waist=0.3,
    w_sym=1.5,
    occ_sigma=0.0,
    w_foot=4.0,
    mirror_full_conf=0.15,
    mirror_none_conf=0.6,
    far_leg_vis_scale=0.5,
    contact_frac=0.05,
    hip_contact_frac=0.20,
    w_flat=10.0,
    w_com=25.0,
    com_ik_margin=0.02,
    fps_out=50.0,
    max_nfev=80,
    fov_deg=65.0,
    depth_sigma=6.0,
    max_ground_tilt_deg=30.0,
    still_sp_frac=0.12,
    still_h_frac=0.16,
    feas_sigma=2.0,
    joint_acc_rms_max=40.0,
)
VARIANTS = [
    {},
    {"fov_deg": 50.0},
    {"fov_deg": 80.0},
    {"q_sigma": 0.5, "w_reg": 0.02},
    {"depth_gain": 0.5},
    {"w_dir": 0.8, "kp_sigma": 1.5},
    {"deswap": False},
    {"ev_on": 0.0},
    {"ev_on": 0.0, "fov_deg": 50.0},
]
GATE = dict(
    reproj_mean_pct=6.0,
    reproj_p95_pct=12.0,
    contact_agree_min=0.85,
    floor_penetration_m=0.01,
    leg_angle_p95_deg=20.0,
    arm_angle_p95_deg=30.0,
    leg_sym_dev_p95_deg=35.0,
    knee_angle_sustained_deg=25.0,
    elbow_angle_sustained_deg=35.0,
    shank_dir_sustained_deg=30.0,
    foot_contact_height_m=0.06,
    foot_contact_miss_frac=0.15,
    limit_margin_deg=2.0,
    robot_limited_frac_max=0.25,
)


def _fill(a):
    a = a.copy()
    T = len(a)
    f = a.reshape(T, -1)
    idx = np.arange(T)
    for k in range(f.shape[1]):
        bad = ~np.isfinite(f[:, k])
        if bad.all():
            f[:, k] = 0
        elif bad.any():
            f[bad, k] = np.interp(idx[bad], idx[~bad], f[~bad, k])
    return f.reshape(a.shape)


def deswap(kp, world):
    """MediaPipe flips L/R limbs in side views. Per frame and per limb pair, keep the labelling whose 2D
    positions are closest to the previous (already fixed) frame."""
    kp, world = kp.copy(), world.copy()
    n = 0
    for A, B in ((LEG_L, LEG_R), (ARM_L, ARM_R)):
        for t in range(1, len(kp)):
            keep = np.nansum(np.linalg.norm(kp[t, A, :2] - kp[t - 1, A, :2], axis=1)) + np.nansum(
                np.linalg.norm(kp[t, B, :2] - kp[t - 1, B, :2], axis=1)
            )
            swap = np.nansum(np.linalg.norm(kp[t, B, :2] - kp[t - 1, A, :2], axis=1)) + np.nansum(
                np.linalg.norm(kp[t, A, :2] - kp[t - 1, B, :2], axis=1)
            )
            if swap < 0.7 * keep:
                for arr in (kp, world):
                    arr[t, A + B] = arr[t, B + A]
                n += 1
    return kp, world, n


def far_l_mask(w):
    return gaussian_filter1d(((w[:, L_HIP, 2] - w[:, R_HIP, 2]) > 0).astype(float), 3.0, mode="nearest") > 0.5


def far_leg_evidence(kp, far_l, p):
    """[0,1] per frame: how strongly the far leg's own 2D is self-consistent evidence that it is visible."""
    k2 = _fill(kp[..., :2])
    F = {
        n: np.where(far_l[:, None], k2[:, l], k2[:, r])
        for n, l, r in (("hip", L_HIP, R_HIP), ("kn", L_KN, R_KN), ("an", L_AN, R_AN))
    }
    N = {
        n: np.where(far_l[:, None], k2[:, r], k2[:, l])
        for n, l, r in (("hip", L_HIP, R_HIP), ("kn", L_KN, R_KN), ("an", L_AN, R_AN))
    }
    nrm = lambda a, b: np.linalg.norm(a - b, axis=-1) + 1e-6
    ll = np.maximum(
        nrm(N["hip"], N["kn"]) + nrm(N["kn"], N["an"]), nrm(F["hip"], F["kn"]) + nrm(F["kn"], F["an"])
    )
    ls = p.get("ev_len_sigma", 0.25)
    lenok = np.minimum(
        *(
            np.exp(-(np.log(nrm(F[a], F[b]) / nrm(N[a], N[b])) ** 2) / (2 * ls**2))
            for a, b in (("hip", "kn"), ("kn", "an"))
        )
    )
    sep = 0.5 * (nrm(F["kn"], N["kn"]) + nrm(F["an"], N["an"])) / ll
    sepok = np.clip((sep - p.get("ev_sep_lo", 0.08)) / p.get("ev_sep_span", 0.10), 0, 1)
    acc = lambda x: np.linalg.norm(np.diff(x, 2, axis=0), axis=-1)
    af, an = (gaussian_filter1d(np.r_[0, acc(x), 0], 4.0, mode="nearest") for x in (F["an"], N["an"]))
    smooth = np.clip(1 - (af / (an + 0.002 * ll.mean()) - p.get("ev_jerk_ratio", 2.0)) / 2.0, 0, 1)
    ev = lenok * sepok * smooth
    sepstrong = np.clip((sep - p.get("ev_sep_strong", 0.28)) / p.get("ev_sep_strong_span", 0.10), 0, 1)
    ev = np.maximum(ev, sepstrong * smooth)
    ev = gaussian_filter1d(ev, 3.0, mode="nearest")
    return np.clip((ev - p.get("ev_lo", 0.55)) / p.get("ev_span", 0.15), 0, 1) * p.get("ev_on", 1.0)


def far_leg_discount(kp, world, p):
    """MediaPipe reports the far leg of a side-on body as visible even when it is hidden behind the near leg
    and its 2D points are guesses. Side-on-ness = |hip depth difference| / hip width from the world landmarks;
    unless the far leg's own 2D is self-consistent (see far_leg_evidence), its visibilities are scaled down by up
    to far_leg_vis_scale, so it follows the mirrored near leg instead of the guessed 2D."""
    kp = kp.copy()
    w = _fill(world)
    dz = w[:, L_HIP, 2] - w[:, R_HIP, 2]
    hw = np.linalg.norm(w[:, L_HIP] - w[:, R_HIP], axis=1) + 1e-6
    side = gaussian_filter1d(np.clip((np.abs(dz) / hw - 0.6) / 0.3, 0, 1), 3.0, mode="nearest")
    ev = far_leg_evidence(kp, far_l_mask(w), p)
    side = side * (1 - ev)
    sc = 1 - (1 - p["far_leg_vis_scale"]) * side
    far_l = far_l_mask(w)
    for leg, isfar in ((LEG_L, far_l), (LEG_R, ~far_l)):
        kp[:, leg, 2] = np.where(
            isfar[:, None],
            np.maximum(kp[:, leg, 2] * sc[:, None], (np.where(ev > 0.0, 0.3 + 0.6 * ev, 0.0))[:, None]),
            kp[:, leg, 2],
        )
    return kp, side


def mirror_fill_legs(H, kpc):
    """3D bilateral prior for the TARGETS: a low-confidence leg keypoint is blended toward the other leg's keypoint
    reflected across the body's own sagittal plane (normal = hip-to-hip axis through the hip midpoint), so the
    mirror is taken in the body frame and holds for prone/side-on poses, not in the camera or world frame."""
    H = H.copy()
    mid = 0.5 * (H[:, L_HIP] + H[:, R_HIP])
    nrm = H[:, L_HIP] - H[:, R_HIP]
    nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9)
    src = H.copy()
    for a, b in zip(LEG_L[1:], LEG_R[1:]):
        for me, ot in ((a, b), (b, a)):
            ref = src[:, ot] - 2 * np.sum((src[:, ot] - mid) * nrm, 1, keepdims=True) * nrm
            c = kpc[:, me][:, None]
            use = (kpc[:, ot] > kpc[:, me])[:, None]
            H[:, me] = np.where(use, c * src[:, me] + (1 - c) * ref, src[:, me])
    return H


def lift(kp2d, world, p, size):
    """Pinhole lift. Root depth per frame from apparent body size (metres/px from metric vs 2D bone lengths,
    heavily smoothed) and an assumed horizontal FOV; per-joint depth offsets from the world landmarks. Keeping
    the root depth preserves motion toward and away from the camera."""
    kp = _fill(kp2d)
    w = _fill(world)
    T = len(kp)
    W, H = float(size[0]), float(size[1])
    f = 0.5 * W / np.tan(np.radians(p["fov_deg"]) / 2)
    cx, cy = W / 2, H / 2
    R = np.full((T, len(BONES)), np.nan)
    for i, (a, b) in enumerate(BONES):
        l2 = np.linalg.norm(kp[:, a, :2] - kp[:, b, :2], axis=1)
        lw = np.linalg.norm(w[:, a, :2] - w[:, b, :2], axis=1)
        ok = (lw > 0.08) & (kp[:, a, 2] > 0.5) & (kp[:, b, 2] > 0.5)
        R[ok, i] = lw[ok] / l2[ok]
    s_t = np.nanmedian(R, 1)
    s_med = float(np.nanmedian(s_t))
    s_t = np.where(np.isfinite(s_t), s_t, s_med)
    s_t = gaussian_filter1d(s_t, p["depth_sigma"], mode="nearest")
    D = f * s_t
    u = (
        gaussian_filter1d(kp[..., :2], p["kp_sigma"], axis=0, mode="nearest")
        if p["kp_sigma"] > 0
        else kp[..., :2]
    )
    dz = gaussian_filter1d(w[..., 2], max(p["kp_sigma"], 1.5), axis=0, mode="nearest") * p["depth_gain"]
    Z = D[:, None] + dz
    C = np.stack([(u[..., 0] - cx) / f * Z, Z, -(u[..., 1] - cy) / f * Z], -1)
    return C, s_med


def ground_and_face(C, max_tilt=30.0):
    cand = [L_HEEL, L_TOE, R_HEEL, R_TOE, L_KN, R_KN, L_WR, R_WR, L_IDX, R_IDX]
    low = C[np.arange(len(C)), np.array(cand)[np.argmin(C[:, cand, 2], 1)]]
    pts = [low]
    for j in (L_HEEL, L_TOE, R_HEEL, R_TOE):
        near = C[:, j, 2] - low[:, 2] < 0.03
        pts.append(C[near, j])
    P = np.concatenate(pts)
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    keep = np.ones(len(P), bool)
    for _ in range(5):
        coef = np.linalg.lstsq(A[keep], P[keep, 2], rcond=None)[0]
        r = P[:, 2] - A @ coef
        keep = r < np.percentile(r[keep], 70) + 0.02
    n = np.array([-coef[0], -coef[1], 1.0])
    n /= np.linalg.norm(n)
    if np.degrees(np.arccos(n[2])) > max_tilt:
        st = slice(0, max(5, len(C) // 10))
        up = np.median(C[st, [L_SH, R_SH]].mean(1) - C[st, [L_AN, R_AN]].mean(1), 0)
        n = up / np.linalg.norm(up)
        if np.degrees(np.arccos(n[2])) > max_tilt:
            n = np.array([0.0, 0.0, 1.0])
    Rl = Rot.align_vectors([[0, 0, 1]], [n])[0].as_matrix()
    G = C @ Rl.T
    G[..., 2] -= np.percentile((G[:, cand, 2]).min(1), 5)
    f = np.nanmedian(np.r_[G[:, [L_TOE, R_TOE], :2].mean(1) - G[:, [L_HEEL, R_HEEL], :2].mean(1)], 0)
    yaw = float(np.arctan2(f[1], f[0]))
    Ry = Rot.from_euler("z", -yaw).as_matrix()
    G = G @ Ry.T
    st = slice(0, max(5, len(G) // 15))
    shear = 0.0
    ank = G[st, [L_AN, R_AN]].mean(1)
    sh = G[st, [L_SH, R_SH]].mean(1)
    dv = (sh - ank).mean(0)
    if np.degrees(np.arctan2(abs(dv[0]), dv[2])) < 20 and dv[2] > 0.8:
        shear = float(dv[1] / dv[2])
        G[..., 1] -= shear * G[..., 2]
    o = G[:10, [L_HIP, R_HIP], :2].mean((0, 1))
    G[..., :2] -= o
    tilt = float(np.degrees(np.arccos(n[2])))
    return G, dict(
        R=(Ry @ Rl),
        origin=o,
        lateral_shear=shear,
        ground_tilt_deg=tilt,
        facing_yaw_deg=float(np.degrees(yaw)),
    )


def human_contacts(G, vis, p):
    stature = np.percentile(G[:, NOSE, 2], 90) + 0.1
    out = {}
    for g, idx in GROUPS.items():
        thr = (p["hip_contact_frac"] if g == "hips" else p["contact_frac"]) * stature + (
            0.05 if g == "knees" else 0
        )
        out[g] = G[:, idx, 2].min(1) < thr
    return out, float(stature)


class G1:
    def __init__(self):
        self.m = build_model()
        self.d = mujoco.MjData(self.m)
        m = self.m
        B = lambda n: m.body(n).id
        self.b = {
            k: B(v)
            for k, v in dict(
                pel="pelvis",
                lhp="left_hip_pitch_link",
                rhp="right_hip_pitch_link",
                lkn="left_knee_link",
                rkn="right_knee_link",
                lan="left_ankle_roll_link",
                ran="right_ankle_roll_link",
                lsh="left_shoulder_pitch_link",
                rsh="right_shoulder_pitch_link",
                lel="left_elbow_link",
                rel="right_elbow_link",
                lwr="left_wrist_yaw_link",
                rwr="right_wrist_yaw_link",
                tor="torso_link",
            ).items()
        }
        self.jlo = m.jnt_range[1:, 0] + 1e-3
        self.jhi = m.jnt_range[1:, 1] - 1e-3
        self.colgeo = []
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]) or m.geom_bodyid[g] == 0:
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
                v = m.mesh_vert[a : a + n]
                v = v[np.random.default_rng(0).choice(len(v), min(len(v), 60), False)]
            else:
                v = np.zeros((1, 3))
            self.colgeo.append(
                (g, v, m.geom_size[g][0] if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH else 0.0)
            )
        self.allgeo = []
        rng = np.random.default_rng(1)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == 0:
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
                v = m.mesh_vert[a : a + n]
                v = v[rng.choice(len(v), min(len(v), 400), False)]
                self.allgeo.append((g, v, 0.0))
            else:
                self.allgeo.append((g, np.zeros((1, 3)), float(m.geom_size[g][0])))
        self.ikgeo = [(g, v[:30], r) for g, v, r in self.allgeo]
        gb = np.array([m.geom_bodyid[g] for g, _, _ in self.ikgeo])
        self.group_geoms = {
            k: np.flatnonzero(np.isin(gb, [m.body(n).id for n in names])) for k, names in GROUP_BODIES.items()
        }
        self.foot_geoms = {
            k: np.flatnonzero(np.isin(gb, [m.body(n).id for n in names])) for k, names in FOOT_BODIES.items()
        }
        self.hand_geoms = {
            k: np.flatnonzero(np.isin(gb, [m.body(n).id for n in names])) for k, names in HAND_BODIES.items()
        }
        self.sole = {}
        for sd, bn in (("l", "left_ankle_roll_link"), ("r", "right_ankle_roll_link")):
            bid = m.body(bn).id
            gs = [
                g
                for g in range(m.ngeom)
                if m.geom_bodyid[g] == bid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
            ]
            xs = np.array([m.geom_pos[g][0] for g in gs])
            mid = np.median(xs)
            self.sole[sd] = dict(
                heel=[g for g, x in zip(gs, xs) if x <= mid],
                toe=[g for g, x in zip(gs, xs) if x > mid],
                r=float(m.geom_size[gs[0], 0]) if gs else 0.0,
            )
        d = self.d
        d.qpos[:] = m.qpos0
        d.qpos[2] = 0.8
        mujoco.mj_kinematics(m, d)
        x = d.xpos
        self.leg = float(
            np.linalg.norm(x[self.b["lhp"]] - x[self.b["lkn"]])
            + np.linalg.norm(x[self.b["lkn"]] - x[self.b["lan"]])
        )
        self.ankle_h = float(
            x[self.b["lan"]][2]
            - (d.xpos[self.b["lan"]] + d.xmat[self.b["lan"]].reshape(3, 3) @ [0, 0, -0.03])[2]
            + 0.005
        )

    def fk(self, q):
        self.d.qpos[:] = q
        mujoco.mj_kinematics(self.m, self.d)

    def keypoints(self):
        """robot points in MediaPipe index order (dict mp_index -> xyz)."""
        d, b = self.d, self.b
        x = d.xpos
        R = lambda k: d.xmat[b[k]].reshape(3, 3)
        loc = lambda k, v: x[b[k]] + R(k) @ np.asarray(v)
        return {
            L_HIP: x[b["lhp"]],
            R_HIP: x[b["rhp"]],
            L_KN: x[b["lkn"]],
            R_KN: x[b["rkn"]],
            L_AN: x[b["lan"]],
            R_AN: x[b["ran"]],
            L_HEEL: loc("lan", [-0.05, 0, -0.03]),
            R_HEEL: loc("ran", [-0.05, 0, -0.03]),
            L_TOE: loc("lan", [0.12, 0, -0.03]),
            R_TOE: loc("ran", [0.12, 0, -0.03]),
            L_SH: x[b["lsh"]],
            R_SH: x[b["rsh"]],
            L_EL: x[b["lel"]],
            R_EL: x[b["rel"]],
            L_WR: x[b["lwr"]],
            R_WR: x[b["rwr"]],
            L_IDX: loc("lwr", [0.09, 0, 0]),
            R_IDX: loc("rwr", [0.09, 0, 0]),
            NOSE: loc("tor", [0.08, 0, 0.42]),
        }

    def geom_minz(self):
        """per-geom lowest z (coarse 30-vertex sets; fast enough inside the IK residual)."""
        d = self.d
        return np.array(
            [
                float((d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min() - r)
                for g, v, r in self.ikgeo
            ]
        )

    def floor_min(self):
        """(min z over every robot geom vertex, body name) for the current FK state."""
        d = self.d
        best = (9.0, -1)
        for g, v, r in self.allgeo:
            z = float((d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min() - r)
            if z < best[0]:
                best = (z, self.m.geom_bodyid[g])
        return best

    def lowest(self, bodies=None):
        d = self.d
        zs = {}
        for g, v, r in self.colgeo:
            bid = self.m.geom_bodyid[g]
            z = (d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min() - r
            zs[bid] = min(zs.get(bid, 9.0), z)
        return zs


IKPTS = [
    L_HIP,
    R_HIP,
    L_KN,
    R_KN,
    L_AN,
    R_AN,
    L_HEEL,
    R_HEEL,
    L_TOE,
    R_TOE,
    L_SH,
    R_SH,
    L_EL,
    R_EL,
    L_WR,
    R_WR,
    L_IDX,
    R_IDX,
    NOSE,
]
WPOS = {
    L_HIP: 2,
    R_HIP: 2,
    L_KN: 2,
    R_KN: 2,
    L_AN: 2.5,
    R_AN: 2.5,
    L_HEEL: 1.5,
    R_HEEL: 1.5,
    L_TOE: 2,
    R_TOE: 2,
    L_SH: 1.5,
    R_SH: 1.5,
    L_EL: 1,
    R_EL: 1,
    L_WR: 2.5,
    R_WR: 2.5,
    L_IDX: 1.5,
    R_IDX: 1.5,
    NOSE: 0.7,
}
DIRS = [
    (L_HIP, L_KN),
    (L_KN, L_AN),
    (R_HIP, R_KN),
    (R_KN, R_AN),
    (L_HEEL, L_TOE),
    (R_HEEL, R_TOE),
    (L_SH, L_EL),
    (L_EL, L_WR),
    (R_SH, R_EL),
    (R_EL, R_WR),
    ("mh", "ms"),
    (R_HIP, L_HIP),
    (R_SH, L_SH),
]


def _pt(K, k):
    if k == "mh":
        return 0.5 * (K[L_HIP] + K[R_HIP])
    if k == "ms":
        return 0.5 * (K[L_SH] + K[R_SH])
    return K[k]


DIR_LIMB = {
    (L_HIP, L_KN): "leg_l",
    (L_KN, L_AN): "leg_l",
    (L_HEEL, L_TOE): "leg_l",
    (R_HIP, R_KN): "leg_r",
    (R_KN, R_AN): "leg_r",
    (R_HEEL, R_TOE): "leg_r",
    (L_SH, L_EL): "arm_l",
    (L_EL, L_WR): "arm_l",
    (R_SH, R_EL): "arm_r",
    (R_EL, R_WR): "arm_r",
}
LIMB_KP = {"leg_l": (L_KN, L_AN), "leg_r": (R_KN, R_AN), "arm_l": (L_EL, L_WR), "arm_r": (R_EL, R_WR)}
OTHER = {"leg_l": "leg_r", "leg_r": "leg_l", "arm_l": "arm_r", "arm_r": "arm_l"}
MIRROR = {
    "leg": [
        ("hip_pitch", 1),
        ("hip_roll", -1),
        ("hip_yaw", -1),
        ("knee", 1),
        ("ankle_pitch", 1),
        ("ankle_roll", -1),
    ],
    "arm": [
        ("shoulder_pitch", 1),
        ("shoulder_roll", -1),
        ("shoulder_yaw", -1),
        ("elbow", 1),
        ("wrist_roll", -1),
        ("wrist_pitch", 1),
        ("wrist_yaw", -1),
    ],
}


def vis_conf(v):
    """visibility -> confidence in [0,1] (0.3 -> 0, 0.8 -> 1)."""
    return np.clip((np.nan_to_num(v, nan=0.0) - 0.3) / 0.5, 0.0, 1.0)


def limb_conf(vis, sigma=3.0):
    """per-limb (T,) confidence = worse of its two distal keypoints, temporally smoothed."""
    return {
        l: np.clip(gaussian_filter1d(vis_conf(np.minimum(vis[:, a], vis[:, b])), sigma, mode="nearest"), 0, 1)
        for l, (a, b) in LIMB_KP.items()
    }


def limb_qidx(m):
    """limb -> (qpos idx array, mirror sign array)."""
    out = {}
    for l in LIMB_KP:
        kind, side = l.split("_")
        nm = "left" if side == "l" else "right"
        out[l] = (
            np.array([m.joint(f"{nm}_{j}_joint").qposadr[0] for j, _ in MIRROR[kind]]),
            np.array([sg for _, sg in MIRROR[kind]], float),
        )
    return out


CONTACT_Z = {"knees": ([L_KN, R_KN], 0.05), "hands": ([L_WR, R_WR], 0.035), "hips": ([L_HIP, R_HIP], 0.13)}


def _com_outside(c, pts, margin):
    """2-vector from the closest point of the support polygon (shrunk by margin toward its centroid) to c; 0 inside."""
    cen = pts.mean(0)
    try:
        from scipy.spatial import ConvexHull

        hv = pts[ConvexHull(pts).vertices]
    except Exception:
        dvec = pts - cen
        ax = dvec[np.argmax(np.linalg.norm(dvec, axis=1))]
        ax = ax / (np.linalg.norm(ax) + 1e-9)
        proj = (pts - cen) @ ax
        hv = np.array([cen + ax * proj.min(), cen + ax * proj.max()])
    sh = cen + (hv - cen) * np.maximum(
        0.0, 1.0 - margin / (np.linalg.norm(hv - cen, axis=1, keepdims=True) + 1e-9)
    )
    if len(sh) >= 3:
        e = np.roll(sh, -1, 0) - sh
        nrm = np.c_[e[:, 1], -e[:, 0]]
        sgn = np.sign(np.sum(nrm * (cen - sh), 1, keepdims=True))
        nrm = nrm * sgn
        if np.all(np.sum(nrm * (c - sh), 1) >= 0):
            return np.zeros(2)
    best = None
    for a, b in zip(sh, np.roll(sh, -1, 0)):
        ab = b - a
        u = np.clip(np.dot(c - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
        q = a + u * ab
        if best is None or np.linalg.norm(c - q) < np.linalg.norm(c - best):
            best = q
    return c - best


def ik(robot, H, p, hc=None, conf=None, kpc=None, fc=None, view=None, footc=None, htc=None, handc=None):
    global WAIST
    WAIST = np.array(
        [
            robot.m.joint(n).qposadr[0] - 7
            for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
        ]
    )
    T = len(H)
    m = robot.m
    Q = np.zeros((T, m.nq))
    cost = np.zeros(T)
    q = m.qpos0.copy()
    q[:2] = 0.5 * (H[0, L_HIP, :2] + H[0, R_HIP, :2])
    q[2] = 0.79
    Rprev = Rot.identity()
    LO = np.r_[[-5] * 3, [-np.pi] * 3, robot.jlo]
    HI = np.r_[[5] * 3, [np.pi] * 3, robot.jhi]
    x_prev = np.r_[q[:3], 0, 0, 0, q[7:]]
    LQ = limb_qidx(m)
    for t in range(T):
        kw = {k: (0.25 + 0.75 * kpc[t, k]) if kpc is not None else 1.0 for k in IKPTS}
        fw = {L_HEEL: 1.0, L_TOE: 1.0, R_HEEL: 1.0, R_TOE: 1.0}
        if footc is not None:
            for sd, (he, to) in (("l", (L_HEEL, L_TOE)), ("r", (R_HEEL, R_TOE))):
                fw[he] = fw[to] = max(float(footc[sd][t]), p["foot_fs_floor"])
                kw[he] *= fw[he]
                kw[to] *= fw[to]
        lw = {l: (0.25 + 0.75 * conf[l][t]) if conf is not None else 1.0 for l in LIMB_KP}
        sym = [(l, max(0.0, conf[OTHER[l]][t] - conf[l][t])) for l in LIMB_KP] if conf is not None else []
        h = H[t]

        def build(x):
            qq = np.empty(m.nq)
            qq[:3] = x[:3]
            w = (Rot.from_rotvec(x[3:6]) * Rprev).as_quat()
            qq[3:7] = [w[3], w[0], w[1], w[2]]
            qq[7:] = x[6:]
            return qq

        def res(x):
            robot.fk(build(x))
            K = robot.keypoints()
            r = []
            for k in IKPTS:
                e = K[k] - h[k]
                if view is not None:
                    e = e - (1.0 - p["w_depth"]) * np.dot(e, view) * view
                r.append(p["w_pos"] * WPOS[k] * kw[k] * e)
            gz = robot.geom_minz()
            if hc is not None:
                for g in ("knees", "hips"):
                    if hc[g][t]:
                        r.append([p["w_contact"] * gz[robot.group_geoms[g]].min()])
            if (
                hc is not None
                and hc.get("hands") is not None
                and p.get("w_hand", 0) > 0
                and handc is not None
            ):
                for sd in ("l", "r"):
                    if handc[sd][t] > 0.05:
                        r.append([p["w_hand"] * float(handc[sd][t]) * gz[robot.hand_geoms[sd]].min()])
            elif hc is not None and hc.get("hands") is not None and p.get("w_hand", 0) > 0:
                if hc["hands"][t]:
                    r.append([p["w_hand"] * gz[robot.group_geoms["hands"]].min()])
            if fc is not None:
                for sd in ("l", "r"):
                    if fc[sd][t] > 0:
                        r.append([p["w_foot"] * fc[sd][t] * gz[robot.foot_geoms[sd]].min()])
            if htc is not None and fc is not None and (p.get("w_flat", 0) > 0 or p.get("w_com", 0) > 0):
                d_ = robot.d
                sup = []
                for sd in ("l", "r"):
                    for part in ("heel", "toe"):
                        wgt = float(fc[sd][t]) * float(htc[sd + part][t])
                        if wgt <= 0.05:
                            continue
                        zs = [d_.geom_xpos[g][2] - robot.sole[sd]["r"] for g in robot.sole[sd][part]]
                        if p.get("w_flat", 0) > 0:
                            r.append(p["w_flat"] * wgt * np.array(zs))
                        if wgt > 0.5:
                            sup += [d_.geom_xpos[g][:2].copy() for g in robot.sole[sd][part]]
                feet_only = hc is None or not any(hc[g][t] for g in ("knees", "hands", "hips"))
                if p.get("w_com", 0) > 0 and feet_only and len(sup) >= 2:
                    mujoco.mj_comPos(robot.m, d_)
                    r.append(
                        p["w_com"]
                        * _com_outside(d_.subtree_com[1][:2], np.array(sup), p.get("com_ik_margin", 0.02))
                    )
            r.append(p["w_pen"] * np.minimum(gz, 0.0))
            for a, b in DIRS:
                hv = _pt(h, b) - _pt(h, a)
                rv = _pt(K, b) - _pt(K, a)
                e = rv / (np.linalg.norm(rv) + 1e-9) - hv / (np.linalg.norm(hv) + 1e-9)
                if view is not None:
                    e = e - (1.0 - p["w_depth"]) * np.dot(e, view) * view
                r.append(p["w_dir"] * lw.get(DIR_LIMB.get((a, b)), 1.0) * fw.get(b, 1.0) * e)
            for l, ws in sym:
                if ws <= 0:
                    continue
                ia, sg = LQ[l]
                ib, _ = LQ[OTHER[l]]
                r.append(p["w_sym"] * ws * (x[ia - 1] - sg * x[ib - 1]))
            reg = p["w_reg"] * (x - x_prev)
            reg[:3] *= 3
            waist = p["w_waist"] * x[6 + WAIST]
            return np.concatenate(r + [reg, waist])

        x0 = np.clip(np.r_[x_prev[:3], 0, 0, 0, x_prev[6:]], LO + 1e-6, HI - 1e-6)
        sol = least_squares(res, x0, bounds=(LO, HI), max_nfev=p["max_nfev"], x_scale=1.0)
        Q[t] = build(sol.x)
        cost[t] = float(np.sqrt(np.mean(sol.fun**2)))
        w = Q[t, 3:7]
        Rprev = Rot.from_quat([w[1], w[2], w[3], w[0]])
        x_prev = np.r_[sol.x[:3], 0, 0, 0, sol.x[6:]]
    return Q, cost


def smooth_occluded(robot, Q, conf, sigma):
    """extra temporal smoothing of limb joints, blended in by (1 - confidence): occluded limbs cannot pop."""
    if sigma <= 0:
        return Q
    Q = Q.copy()
    for l, (ia, _) in limb_qidx(robot.m).items():
        c = conf[l][:, None]
        Qs = gaussian_filter1d(Q[:, ia], sigma, axis=0, mode="nearest")
        Q[:, ia] = c * Q[:, ia] + (1 - c) * Qs
    return Q


def mirror_occluded(robot, Q, conf, p):
    """Bilateral rule: a fully occluded limb (confidence <= mirror_full_conf) whose partner is
    more visible gets EXACTLY the partner's joint angles mirrored across the body's sagittal plane (joint space is
    the body frame: pitch/knee/elbow copied, roll/yaw negated). Partly visible limbs blend proportionally to
    visibility up to mirror_none_conf, above which the limb is its own IK solution. Legs and arms, any move."""
    Q = Q.copy()
    Q0 = Q.copy()
    LQ = limb_qidx(robot.m)
    lo, hi = p["mirror_full_conf"], p["mirror_none_conf"]
    for l in LIMB_KP:
        ia, sg = LQ[l]
        ib, _ = LQ[OTHER[l]]
        a = np.clip((conf[l] - lo) / (hi - lo), 0, 1)[:, None]
        a = np.where((conf[OTHER[l]] > conf[l])[:, None], a, 1.0)
        Q[:, ia] = np.clip(a * Q0[:, ia] + (1 - a) * sg * Q0[:, ib], robot.jlo[ia - 7], robot.jhi[ia - 7])
    return Q


def smooth_q(Q, sigma):
    if sigma <= 0:
        return Q
    Q = Q.copy()
    Q[:, :3] = gaussian_filter1d(Q[:, :3], sigma, axis=0, mode="nearest")
    Q[:, 7:] = gaussian_filter1d(Q[:, 7:], sigma, axis=0, mode="nearest")
    qq = Q[:, 3:7] * np.sign(np.sum(Q[:, 3:7] * Q[:1, 3:7], 1, keepdims=True))
    qq = gaussian_filter1d(qq, sigma, axis=0, mode="nearest")
    Q[:, 3:7] = qq / np.linalg.norm(qq, axis=1, keepdims=True)
    return Q


def depenetrate(robot, Q, tol=0.0):
    """lift the root so no robot geom vertex (visual or collision) is below the mat; returns worst before."""
    lift = np.zeros(len(Q))
    for t in range(len(Q)):
        robot.fk(Q[t])
        lift[t] = max(0.0, -robot.floor_min()[0] - tol)
    from scipy.ndimage import maximum_filter1d

    L = gaussian_filter1d(maximum_filter1d(lift, 5), 1.0)
    L = np.maximum(L, lift)
    Q = Q.copy()
    Q[:, 2] += L
    return Q, float(lift.max())


def ground_robot(robot, Q, hc):
    T = len(Q)
    zmin = np.zeros(T)
    for t in range(T):
        robot.fk(Q[t])
        zmin[t] = min(robot.lowest().values())
    any_c = np.any(np.stack(list(hc.values()), 1), 1)
    shift = np.where(any_c, -zmin, np.nan)
    idx = np.arange(T)
    if np.isfinite(shift).any():
        shift = np.interp(idx, idx[np.isfinite(shift)], shift[np.isfinite(shift)])
    else:
        shift = -zmin
    shift = gaussian_filter1d(shift, 1.5, mode="nearest")
    shift = np.maximum(shift, -zmin)
    Q = Q.copy()
    Q[:, 2] += shift
    return Q, float(np.abs(shift).max())


GROUP_BODIES = {
    "feet": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "knees": ["left_knee_link", "right_knee_link"],
    "hands": [
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_wrist_pitch_link",
        "right_wrist_pitch_link",
    ],
    "hips": ["pelvis", "left_hip_pitch_link", "right_hip_pitch_link"],
}
FOOT_BODIES = {
    "l": ["left_ankle_pitch_link", "left_ankle_roll_link"],
    "r": ["right_ankle_pitch_link", "right_ankle_roll_link"],
}
HAND_BODIES = {
    "l": ["left_wrist_yaw_link", "left_wrist_pitch_link", "left_wrist_roll_link"],
    "r": ["right_wrist_yaw_link", "right_wrist_pitch_link", "right_wrist_roll_link"],
}


def robot_tracks(robot, Q):
    K = np.zeros((len(Q), 33, 3))
    low = {g: np.zeros(len(Q)) for g in GROUP_BODIES}
    ids = {g: [robot.m.body(n).id for n in v] for g, v in GROUP_BODIES.items()}
    for t in range(len(Q)):
        robot.fk(Q[t])
        k = robot.keypoints()
        for j, v in k.items():
            K[t, j] = v
        gz = robot.geom_minz()
        for g in ids:
            low[g][t] = float(gz[robot.group_geoms[g]].min())
    return K, low


def fit_camera(K3, kp2d):
    """Global weak-perspective camera (yaw, pitch, roll, scale, tx, ty) mapping robot keypoints to pixels,
    fitted per clip by least squares; then per-frame translation residual is NOT refit (root drift counts)."""
    vis = kp2d[..., 2] > 0.5
    J = np.array(IKPTS)

    def proj(x, P):
        R = Rot.from_euler("zyx", x[:3]).as_matrix()
        Pc = P @ R.T
        return np.stack([x[3] * Pc[..., 1] + x[4], -x[3] * Pc[..., 2] + x[5]], -1)

    def res(x):
        e = proj(x, K3[:, J]) - kp2d[:, J, :2]
        return (e * vis[:, J, None]).ravel()

    u0 = np.nanmean(kp2d[:, J, 0])
    v0 = np.nanmean(kp2d[:, J, 1])
    best = None
    for yaw0 in np.radians([0, 90, 180, 270]):
        x0 = np.array([yaw0, 0, 0, 300.0, u0, v0])
        s = least_squares(res, x0, loss="soft_l1", f_scale=10.0, max_nfev=400)
        if best is None or s.cost < best.cost:
            best = s
    return proj(best.x, K3), best.x


def gate(npz_path, kp2d_npz, hc=None, p=DEFAULTS):
    z = np.load(npz_path, allow_pickle=True)
    k = np.load(kp2d_npz)
    Q = z["qpos"]
    tq = z["times"] if "times" in z.files else np.arange(len(Q)) / 30.0
    kp = k["kp2d"]
    tk = k["t"]
    n = min(len(tk), int(np.searchsorted(tk, tq[-1] + 1e-6)))
    idx = np.clip(np.round(np.interp(tk[:n], tq, np.arange(len(tq)))).astype(int), 0, len(Q) - 1)
    robot = G1()
    K3, low = robot_tracks(robot, Q[idx])
    kp = kp[:n]
    kpf, wf, _ = deswap(kp, k["world"][:n] if "world" in k.files else np.zeros_like(kp))
    if "world" in k.files:
        kpf, _ = far_leg_discount(kpf, wf, dict(DEFAULTS, **p))
    uv, cam = fit_camera(K3, kpf)
    J = np.array(IKPTS)
    vis = kpf[:, J, 2] > 0.5
    stature_px = np.percentile(np.nanmax(kpf[:, :, 1], 1) - np.nanmin(kpf[:, :, 1], 1), 90)
    err = np.linalg.norm(uv[:, J] - kpf[:, J, :2], axis=-1)[vis] / stature_px * 100
    if hc is None:
        hc = video_contacts_2d(kpf, stature_px, p)
    agree = {}
    for g in GROUP_BODIES:
        v = hc[g][:n]
        r = robot_contact(g, K3, low)
        if v.sum() == 0 and r.sum() == 0:
            continue
        agree[g] = dict(
            agree=float((v == r).mean()),
            video_frames=int(v.sum()),
            g1_frames=int(r.sum()),
            recall=float((v & r).sum() / max(1, v.sum())),
        )
    fz = []
    fb = []
    for q in Q:
        robot.fk(q)
        z, b = robot.floor_min()
        fz.append(z)
        fb.append(b)
    fz = np.array(fz)
    wi = int(np.argmin(fz))
    floor = dict(
        max_penetration_m=float(max(0.0, -fz.min())),
        worst_body=robot.m.body(int(fb[wi])).name,
        worst_t=float(tq[wi]),
        frames_over_1cm=int((fz < -0.01).sum()),
    )
    limbs = {
        "l_thigh": (L_HIP, L_KN),
        "l_shin": (L_KN, L_AN),
        "r_thigh": (R_HIP, R_KN),
        "r_shin": (R_KN, R_AN),
        "l_upper": (L_SH, L_EL),
        "l_fore": (L_EL, L_WR),
        "r_upper": (R_SH, R_EL),
        "r_fore": (R_EL, R_WR),
    }

    def unit(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)

    side = {
        "l_thigh": "r_thigh",
        "l_shin": "r_shin",
        "r_thigh": "l_thigh",
        "r_shin": "l_shin",
        "l_upper": "r_upper",
        "l_fore": "r_fore",
        "r_upper": "l_upper",
        "r_fore": "l_fore",
    }
    cf = {nm: vis_conf(np.minimum(kpf[:, a, 2], kpf[:, b, 2])) for nm, (a, b) in limbs.items()}
    la, la_vis = {}, {}
    Qi_ = Q[idx]
    mg = np.deg2rad(GATE["limit_margin_deg"])
    leg_lim = {}
    for sd, nmj in (("l", "left"), ("r", "right")):
        js = [robot.m.joint(f"{nmj}_{j}_joint") for j in ("hip_pitch", "knee", "ankle_pitch")]
        qa = np.array([j.qposadr[0] for j in js])
        rg = np.array([robot.m.jnt_range[j.id] for j in js])
        leg_lim[sd] = ((Qi_[:, qa] - rg[:, 0] < mg) | (rg[:, 1] - Qi_[:, qa] < mg)).any(1)
    nolim = np.zeros(n, bool)
    lim_mask = {
        "l_thigh": leg_lim["l"],
        "l_shin": leg_lim["l"],
        "r_thigh": leg_lim["r"],
        "r_shin": leg_lim["r"],
        "l_upper": nolim,
        "l_fore": nolim,
        "r_upper": nolim,
        "r_fore": nolim,
    }
    robot_limited_frac = float((leg_lim["l"] | leg_lim["r"]).mean()) if n else 0.0
    seg = {nm: np.linalg.norm(kpf[:, b, :2] - kpf[:, a, :2], axis=-1) for nm, (a, b) in limbs.items()}
    rel = {nm: seg[nm] / (np.nanpercentile(seg[nm], 90) + 1e-9) for nm in limbs}
    short = {nm: np.where(cf[nm] >= 0.5, rel[nm], rel[side[nm]]) < p.get("fs_min", 0.35) for nm in limbs}
    for nm, (a, b) in limbs.items():
        oa, ob = limbs[side[nm]]
        dv = unit(kpf[:, b, :2] - kpf[:, a, :2])
        dn = unit(kpf[:, ob, :2] - kpf[:, oa, :2])
        c = cf[nm][:, None]
        tgt = unit(c * dv + (1 - c) * np.where(cf[side[nm]][:, None] > c, dn, dv))
        ang = np.degrees(np.arccos(np.clip(np.sum(tgt * unit(uv[:, b] - uv[:, a]), 1), -1, 1)))
        la[nm] = np.where(np.isfinite(tgt).all(1) & ~short[nm] & ~lim_mask[nm], ang, np.nan)
        la_vis[nm] = np.where(cf[nm] > 0.4, ang, np.nan)

    def p95(v):
        return float(np.nanpercentile(v, 95)) if np.isfinite(v).any() else 0.0

    per_side = {
        "leg_l": p95(np.fmax(la["l_thigh"], la["l_shin"])),
        "leg_r": p95(np.fmax(la["r_thigh"], la["r_shin"])),
        "arm_l": p95(np.fmax(la["l_upper"], la["l_fore"])),
        "arm_r": p95(np.fmax(la["r_upper"], la["r_fore"])),
    }
    LQ = limb_qidx(robot.m)
    lc = limb_conf(kpf[..., 2])
    Qi = Q[idx]
    sym_dev = {}
    for l in LIMB_KP:
        ia, sg = LQ[l]
        ib, _ = LQ[OTHER[l]]
        occ = (lc[l] < 0.4) & (lc[OTHER[l]] > 0.6)
        dev = np.degrees(np.abs(Qi[:, ia] - sg * Qi[:, ib])).max(1)
        sym_dev[l] = p95(np.where(occ, dev, np.nan))
    win = max(3, int(round(0.2 / max(1e-3, np.median(np.diff(tk[:n])) if n > 1 else 1 / 30))))
    from scipy.ndimage import median_filter

    def sustained(v):
        v = np.where(np.isfinite(v), v, 0.0)
        return float(median_filter(v, size=win, mode="nearest").max()) if len(v) else 0.0

    def bend(P, a, b, c):
        u = unit(P[:, a, :2] - P[:, b, :2])
        w = unit(P[:, c, :2] - P[:, b, :2])
        return 180 - np.degrees(np.arccos(np.clip(np.sum(u * w, 1), -1, 1)))

    knee_pts = {
        "leg_l": (L_HIP, L_KN, L_AN),
        "leg_r": (R_HIP, R_KN, R_AN),
        "arm_l": (L_SH, L_EL, L_WR),
        "arm_r": (R_SH, R_EL, R_WR),
    }
    vb = {l: bend(kpf, *pts) for l, pts in knee_pts.items()}
    rb = {l: bend(uv, *pts) for l, pts in knee_pts.items()}
    knee_dev, shank_dev, knee_series = {}, {}, {}
    for l, o in (("leg_l", "leg_r"), ("leg_r", "leg_l"), ("arm_l", "arm_r"), ("arm_r", "arm_l")):
        c = lc[l]
        tgt = np.where(lc[o] > c, c * vb[l] + (1 - c) * vb[o], vb[l])
        s1, s2 = {
            "leg_l": ("l_thigh", "l_shin"),
            "leg_r": ("r_thigh", "r_shin"),
            "arm_l": ("l_upper", "l_fore"),
            "arm_r": ("r_upper", "r_fore"),
        }[l]
        tgt = np.where(short[s1] | short[s2] | lim_mask[s1], np.nan, tgt)
        knee_series[l] = np.abs(rb[l] - tgt)
        knee_dev[l] = sustained(knee_series[l])
        shank_dev[l] = sustained(
            la[{"leg_l": "l_shin", "leg_r": "r_shin", "arm_l": "l_fore", "arm_r": "r_fore"}[l]]
        )
    fcv, _ = foot_contacts(_fill(kpf), stature_px, p)
    fz_side = {}
    for sd in ("l", "r"):
        zz = []
        for q in Q[idx]:
            robot.fk(q)
            zz.append(robot.geom_minz()[robot.foot_geoms[sd]].min())
        fz_side[sd] = np.array(zz)
    foot = {}
    for sd in ("l", "r"):
        from scipy.ndimage import binary_erosion

        on = binary_erosion(fcv[sd][:n] > 0.5, np.ones(max(1, win // 2) * 2 + 1, bool), border_value=1)
        hgt = np.where(on, fz_side[sd], np.nan)
        foot[sd] = dict(
            video_contact_frames=int(on.sum()),
            miss_frac=float((fz_side[sd][on] > GATE["foot_contact_height_m"]).mean()) if on.any() else 0.0,
            sustained_height_m=sustained(np.where(on, fz_side[sd], 0.0)),
            max_height_m=float(np.nanmax(hgt)) if on.any() else 0.0,
        )
    sust = dict(
        window_frames=win,
        knee_angle_dev_deg=knee_dev,
        shank_dir_dev_deg=shank_dev,
        foot=foot,
        worst_knee_t={l: float(tk[int(np.argmax(np.nan_to_num(v)))] - tk[0]) for l, v in knee_series.items()},
        knee_angle_max_deg=max(knee_dev["leg_l"], knee_dev["leg_r"]),
        shank_dir_max_deg=max(shank_dev["leg_l"], shank_dev["leg_r"]),
        elbow_angle_max_deg=max(knee_dev["arm_l"], knee_dev["arm_r"]),
        forearm_dir_max_deg=max(shank_dev["arm_l"], shank_dev["arm_r"]),
        foot_height_max_m=max(f["sustained_height_m"] for f in foot.values()),
        foot_miss_frac_max=max(f["miss_frac"] for f in foot.values()),
    )
    LA = np.stack(list(la.values()), 1)
    legs = LA[:, :4]
    limb = dict(
        occluded_sym_dev_p95_deg=sym_dev,
        leg_sym_dev_p95_deg=max(sym_dev["leg_l"], sym_dev["leg_r"]),
        leg_angle_p95_deg=max(per_side["leg_l"], per_side["leg_r"]),
        arm_angle_p95_deg=max(per_side["arm_l"], per_side["arm_r"]),
        per_side_p95_deg=per_side,
        leg_angle_max_deg=float(np.nanmax(legs)) if np.isfinite(legs).any() else 0.0,
        per_limb_p95_deg={k: p95(v) for k, v in la.items()},
        per_limb_visible_only_p95_deg={k: p95(v) for k, v in la_vis.items()},
        occluded_frac={k: float((v < 0.4).mean()) for k, v in cf.items()},
        robot_limited_frac=robot_limited_frac,
        robot_limited_frac_by_leg={sd: float(v.mean()) for sd, v in leg_lim.items()},
        worst_leg_t=float(tk[int(np.nanargmax(np.nanmax(legs, 1)))] - tk[0])
        if np.isfinite(legs).any()
        else None,
    )
    import hashlib

    qmd5 = hashlib.md5(np.ascontiguousarray(Q, dtype=np.float64).tobytes()).hexdigest()
    out = dict(
        npz=str(npz_path),
        qpos_md5=qmd5,
        frames=int(n),
        floor=floor,
        limb=limb,
        sustained=sust,
        reproj_mean_pct=float(err.mean()),
        reproj_p95_pct=float(np.percentile(err, 95)),
        reproj_by_point_pct={
            str(j): float(
                np.nanmean(np.linalg.norm(uv[:, j] - kpf[:, j, :2], axis=-1)[kpf[:, j, 2] > 0.5])
                / stature_px
                * 100
            )
            for j in (L_WR, R_WR, L_HIP, L_AN, R_AN, L_KN, NOSE)
        },
        contacts=agree,
        camera=[float(c) for c in cam],
        g1_min_z={g: float(v.min()) for g, v in low.items()},
        g1_pelvis_z_min=float(Q[:, 2].min()),
        g1_pelvis_z_start=float(Q[0, 2]),
    )
    ca = min([a["agree"] for a in agree.values()] or [1.0])
    out["contact_agree_min"] = ca
    checks = dict(
        reproj_mean=out["reproj_mean_pct"] < GATE["reproj_mean_pct"],
        reproj_p95=out["reproj_p95_pct"] < GATE["reproj_p95_pct"],
        contacts=ca >= GATE["contact_agree_min"],
        floor=floor["max_penetration_m"] < GATE["floor_penetration_m"],
        leg_angle=limb["leg_angle_p95_deg"] < GATE["leg_angle_p95_deg"],
        arm_angle=limb["arm_angle_p95_deg"] < GATE["arm_angle_p95_deg"],
        leg_symmetry=limb["leg_sym_dev_p95_deg"] < GATE["leg_sym_dev_p95_deg"],
        knee_angle=sust["knee_angle_max_deg"] < GATE["knee_angle_sustained_deg"],
        shank_dir=sust["shank_dir_max_deg"] < GATE["shank_dir_sustained_deg"],
        elbow_angle=sust["elbow_angle_max_deg"] < GATE["elbow_angle_sustained_deg"],
        robot_limited=robot_limited_frac <= GATE["robot_limited_frac_max"],
        foot_contact=(sust["foot_height_max_m"] < GATE["foot_contact_height_m"])
        and (sust["foot_miss_frac_max"] < GATE["foot_contact_miss_frac"]),
    )
    out["checks"] = checks
    out["pass"] = bool(all(checks.values()))
    out["score"] = (
        out["reproj_mean_pct"]
        + 0.5 * out["reproj_p95_pct"]
        + 20 * (1 - ca)
        + 0.3 * limb["leg_angle_p95_deg"]
        + 500 * floor["max_penetration_m"]
        + 0.1 * limb["leg_sym_dev_p95_deg"]
        + 0.2 * sust["knee_angle_max_deg"]
        + 0.1 * sust["shank_dir_max_deg"]
        + 100 * sust["foot_height_max_m"]
        + 10 * sust["foot_miss_frac_max"]
    )
    return out, uv[:, :], kpf


def robot_contact(g, K3, low):
    """G1 contact per group = lowest geom of that group's bodies within 3 cm of the mat (hips 5 cm)."""
    return low[g] < (0.05 if g == "hips" else 0.03)


def video_contacts_2d(kp, stature_px, p):
    """2D-only contact labels: robust ground line in image (lower envelope of soles/knees/hands)."""
    return _contacts_for(kp, stature_px, p, GROUPS)


def _contacts_for(kp, stature_px, p, groups):
    cand = [L_HEEL, L_TOE, R_HEEL, R_TOE, L_KN, R_KN, L_WR, R_WR, L_IDX, R_IDX]
    P = kp[:, cand, :2].reshape(-1, 2)
    P = P[np.isfinite(P).all(1)]
    A = np.c_[P[:, 0], np.ones(len(P))]
    keep = np.ones(len(P), bool)
    for _ in range(6):
        c = np.linalg.lstsq(A[keep], P[keep, 1], rcond=None)[0]
        r = A @ c - P[:, 1]
        keep = r < np.percentile(r[keep], 25) + 0.01 * stature_px
    out = {}
    for g, idx in groups.items():
        h = (c[0] * kp[:, idx, 0] + c[1]) - kp[:, idx, 1]
        thr = (p["hip_contact_frac"] if g == "hips" else p["contact_frac"]) * stature_px + (
            0.05 * stature_px if g == "knees" else 0
        )
        out[g] = np.nanmin(h, 1) < thr
        if g == "feet":
            fps = float(p.get("fps_in", 30.0))
            sp = (
                np.linalg.norm(
                    np.gradient(gaussian_filter1d(_fill(kp[:, idx, :2]), 1.0, axis=0), axis=0), axis=-1
                )
                * fps
            )
            if len(sp) > 6:
                sp[:2] = sp[2]
                sp[-2:] = sp[-3]
            lowest = np.nanmax(kp[:, :, 1], 1)[:, None]
            planted = (sp < 0.25 * stature_px) & (lowest - kp[:, idx, 1] < 0.06 * stature_px)
            from scipy.ndimage import median_filter as _mf

            win = max(3, int(round(0.27 * fps)))
            still = _mf(sp, size=(win, 1), mode="nearest") < p.get("still_sp_frac", 0.12) * stature_px
            still = still & (lowest - kp[:, idx, 1] < p.get("still_h_frac", 0.16) * stature_px)
            planted = planted | still
            out[g] = out[g] | planted.reshape(len(kp), -1, 2).any(2).any(1)
    return out


def _deflicker(lab):
    """Drop 1-2 frame label flickers (incl. the clip's first/last frames, where keypoints are least stable)."""
    out = lab.copy()
    for i in range(len(lab)):
        nb = lab[max(0, i - 2) : i + 3]
        if (
            lab[i] != np.round(np.median(np.r_[lab[max(0, i - 3) : i], lab[i + 1 : i + 4]]))
            and (nb == lab[i]).sum() <= 2
        ):
            out[i] = 1 - lab[i]
    return out


def hand_contacts(kp, stature_px, p):
    """per-hand video contact in [0,1] for 'l'/'r'. Same 2D ground-line rule as video_contacts_2d hands;
    an occluded hand inherits the visible hand's label (bilateral prior), then deflickered."""
    out = {}
    for sd, idx in (("l", [L_WR, L_IDX]), ("r", [R_WR, R_IDX])):
        out[sd] = _deflicker(_contacts_for(kp, stature_px, p, {"hands": idx})["hands"].astype(float))
    cf = {
        sd: gaussian_filter1d(vis_conf(np.nanmin(kp[:, idx, 2], 1)), 2.0, mode="nearest")
        for sd, idx in (("l", [L_WR, L_IDX, L_EL]), ("r", [R_WR, R_IDX, R_EL]))
    }
    res = {}
    for sd, o in (("l", "r"), ("r", "l")):
        c = cf[sd]
        inherit = cf[o] > c
        res[sd] = np.where(inherit, c * out[sd] + (1 - c) * out[o], out[sd])
    return res, cf


def foot_contacts(kp, stature_px, p):
    """per-foot video contact in [0,1] for 'l'/'r'. Each foot is labelled from its own soles vs the 2D ground line
    (same rule as video_contacts_2d feet). A foot whose sole keypoints are occluded inherits the visible foot's
    label (bilateral prior), blended by visibility confidence: MediaPipe guesses an occluded far foot."""
    out = {}
    for sd, idx in (("l", [L_HEEL, L_TOE]), ("r", [R_HEEL, R_TOE])):
        g = {"feet": idx}
        out[sd] = _deflicker(_contacts_for(kp, stature_px, p, g)["feet"].astype(float))
    cf = {
        sd: gaussian_filter1d(vis_conf(np.nanmin(kp[:, idx, 2], 1)), 2.0, mode="nearest")
        for sd, idx in (("l", [L_HEEL, L_TOE, L_AN]), ("r", [R_HEEL, R_TOE, R_AN]))
    }
    res = {}
    for sd, o in (("l", "r"), ("r", "l")):
        c = cf[sd]
        inherit = cf[o] > c
        res[sd] = np.where(inherit, c * out[sd] + (1 - c) * out[o], out[sd])
    return res, cf


def retarget(kp2d_npz, outdir, p):
    k = np.load(kp2d_npz)
    kp, world, t = k["kp2d"], k["world"], k["t"]
    nsw = 0
    if p["deswap"]:
        kp, world, nsw = deswap(kp, world)
    kp, sideon = far_leg_discount(kp, world, p)
    C, mpp = lift(kp, world, p, k["size"])
    G, gi = ground_and_face(C, p["max_ground_tilt_deg"])
    hc3, stature = human_contacts(G, kp[..., 2], p)
    kf = _fill(kp)
    stature_px = np.percentile(np.nanmax(kf[:, :, 1], 1) - np.nanmin(kf[:, :, 1], 1), 90)
    hc = video_contacts_2d(kf, stature_px, p)
    robot = G1()
    hum_leg = np.percentile(
        np.linalg.norm(G[:, L_HIP] - G[:, L_KN], axis=-1) + np.linalg.norm(G[:, L_KN] - G[:, L_AN], axis=-1),
        75,
    )
    S = robot.leg / hum_leg
    H = G * S
    kpc = vis_conf(kp[..., 2])
    conf = limb_conf(kp[..., 2])
    H = mirror_fill_legs(H, gaussian_filter1d(kpc, 2.0, axis=0, mode="nearest"))
    fc, fcf = foot_contacts(kf, stature_px, p)
    handc, _handcf = hand_contacts(kf, stature_px, p)
    view = gi["R"] @ np.array([0.0, 1.0, 0.0])
    view = view / np.linalg.norm(view)
    footc = {}
    for sd, (he, to) in (("l", (L_HEEL, L_TOE)), ("r", (R_HEEL, R_TOE))):
        fl = np.linalg.norm(kf[:, to, :2] - kf[:, he, :2], axis=1)
        rel = fl / (np.nanpercentile(fl, 90) + 1e-9)
        footc[sd] = gaussian_filter1d(
            np.clip((rel - p["foot_fs_lo"]) / p["foot_fs_span"], 0, 1), 2.0, mode="nearest"
        )
    htc = {}
    for sd, (he, to) in (("l", (L_HEEL, L_TOE)), ("r", (R_HEEL, R_TOE))):
        htc[sd + "heel"] = _deflicker(
            _contacts_for(kf, stature_px, p, {"feet": [he, he]})["feet"].astype(float)
        )
        htc[sd + "toe"] = _deflicker(
            _contacts_for(kf, stature_px, p, {"feet": [to, to]})["feet"].astype(float)
        )
    raw = dict(htc)
    for sd, o in (("l", "r"), ("r", "l")):
        c = fcf[sd]
        inh = fcf[o] > c
        for part in ("heel", "toe"):
            htc[sd + part] = np.where(inh, c * raw[sd + part] + (1 - c) * raw[o + part], raw[sd + part])
    fs = p.get("feas_sigma", 2.0)
    htc = {k: gaussian_filter1d(v, fs, mode="nearest") for k, v in htc.items()} if fs > 0 else htc
    Q, cost = ik(robot, H, p, hc, conf=conf, kpc=kpc, fc=fc, view=view, footc=footc, htc=htc, handc=handc)
    Q = smooth_q(Q, p["q_sigma"])
    Q = smooth_occluded(robot, Q, conf, p["occ_sigma"])
    Q = mirror_occluded(robot, Q, conf, p)
    Q, shift = ground_robot(robot, Q, hc)
    to = np.arange(0, t[-1] - t[0] + 1e-9, 1.0 / p["fps_out"])
    ti = t - t[0]
    Qo = np.stack([np.interp(to, ti, Q[:, j]) for j in range(Q.shape[1])], 1)
    Qo[:, 3:7] /= np.linalg.norm(Qo[:, 3:7], axis=1, keepdims=True)
    Qo[:, 7:] = np.clip(Qo[:, 7:], robot.jlo, robot.jhi)
    Qo, pen = depenetrate(robot, Qo)
    m = robot.m
    V = np.zeros((len(Qo), m.nv))
    dt = 1.0 / p["fps_out"]
    for i in range(len(Qo) - 1):
        mujoco.mj_differentiatePos(m, V[i], dt, Qo[i], Qo[i + 1])
    V[-1] = V[-2]
    Qo[:, 7:] = np.clip(Qo[:, 7:], robot.jlo, robot.jhi)
    diag = dict(
        mpp=mpp,
        lr_swaps_fixed=nsw,
        human_stature_m=stature,
        human_to_g1_scale=float(S),
        ik_rms_mean=float(cost.mean()),
        ik_rms_max=float(cost.max()),
        ground_shift_max_m=shift,
        depenetration_max_m=pen,
        ground_tilt_deg=gi["ground_tilt_deg"],
        lateral_shear=gi["lateral_shear"],
        facing_yaw_deg=gi["facing_yaw_deg"],
        video2d_contact_frames={g: int(v.sum()) for g, v in hc.items()},
        lift3d_contact_frames={g: int(v.sum()) for g, v in hc3.items()},
        human_min_z_m={
            "pelvis": float(G[:, [L_HIP, R_HIP], 2].mean(1).min()),
            "wrists": float(G[:, [L_WR, R_WR], 2].min()),
            "kickback_max_m": float(np.max(G[:, [L_HIP, R_HIP], 0].mean(1) - G[:, [L_AN, R_AN], 0].mean(1))),
        },
    )
    np.savez(
        outdir / "human3d.npz",
        cam_lift=C,
        ground=G,
        g1_scaled=H,
        t=t,
        **{f"contact_{g}": v for g, v in hc.items()},
    )
    np.savez(outdir / "g1_raw.npz", qpos=Q, ik_cost=cost, t=t)
    return Qo, V, to, hc, diag


def run(video, name, outdir, npz_out, mp_python=None, model=None):
    """Keypoints (in the MediaPipe interpreter), then retarget variants until one passes the gate."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    kpf = outdir / "kp2d.npz"
    if not kpf.exists():
        py = mp_python or os.environ.get("MP_PYTHON", sys.executable)
        cmd = [py, str(Path(__file__).with_name("keypoints.py")), str(video), str(kpf)] + (
            ["--model", str(model)] if model else []
        )
        subprocess.run(cmd, check=True)
    tries = []
    best = None
    for i, var in enumerate(VARIANTS):
        p = dict(DEFAULTS, **var)
        Qo, V, to, hc, diag = retarget(kpf, outdir, p)
        cand = outdir / f"try{i}.npz"
        meta = dict(
            name=name,
            source=f"video: {Path(video).name}",
            fps=p["fps_out"],
            dt=1.0 / p["fps_out"],
            n_frames=len(Qo),
        )
        np.savez_compressed(cand, qpos=Qo, qvel=V, ctrl=Qo[:, 7:], times=to, meta_json=json.dumps(meta))
        g, _, _ = gate(cand, kpf, p=p)
        dt = 1.0 / p["fps_out"]
        jacc = float(np.sqrt(((np.diff(Qo[:, 7:], 2, axis=0) / dt**2) ** 2).mean())) if len(Qo) > 3 else 0.0
        g["joint_acc_rms"] = jacc
        g["checks"]["accel_ok"] = jacc <= p["joint_acc_rms_max"]
        g["pass"] = bool(g["pass"]) and g["checks"]["accel_ok"]
        g.update(variant=var, diag=diag)
        tries.append(g)
        failed = [k for k, v in g["checks"].items() if not v]
        print(
            f"try {i} {var}: reprojection {g['reproj_mean_pct']:.2f}% (p95 {g['reproj_p95_pct']:.2f}%), "
            f"contact agreement {g['contact_agree_min']:.2f}, pass={g['pass']} {failed}",
            flush=True,
        )
        if g["pass"] or best is None or (not best[1]["pass"] and g["score"] < best[1]["score"]):
            best = (cand, g)
        if g["pass"]:
            break
    cand, g = best
    Path(npz_out).parent.mkdir(parents=True, exist_ok=True)
    os.replace(cand, npz_out)
    for f in outdir.glob("try*.npz"):
        f.unlink()
    rep = dict(
        name=name,
        npz=str(npz_out),
        passed=g["pass"],
        gate_thresholds=GATE,
        chosen=g,
        tries=tries,
        wall_s=time.time() - t0,
    )
    (outdir / "gate.json").write_text(json.dumps(rep, indent=2, default=float))
    return rep


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Retarget a video to a G1 reference motion.")
    ap.add_argument("video")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True, help="output motion .npz")
    ap.add_argument("--workdir", required=True, help="keypoints, intermediates and gate.json")
    ap.add_argument("--mp-python", default=None, help="Python interpreter with mediapipe installed")
    ap.add_argument("--model", default=None, help="pose_landmarker_heavy.task")
    a = ap.parse_args()
    rep = run(a.video, a.name, a.workdir, a.out, a.mp_python, a.model)
    print(json.dumps({k: rep[k] for k in ("name", "npz", "passed")}))
    sys.exit(0 if rep["passed"] else 2)


if __name__ == "__main__":
    main()
