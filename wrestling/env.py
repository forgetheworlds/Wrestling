"""Single-robot motion imitation environment.

DeepMimic-style tracking of a reference clip: reference state initialisation (RSI), early
termination on falls, and a residual action on top of the reference joint targets. The
reward, costs and terminations are configured by a skill config (see `wrestling.skills`).
"""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .costs import ConstraintTerms
from .feedforward import ff_ctrl_offset, reference_torque
from .robot import (
    FREE_QPOS,
    FREE_QVEL,
    N_JOINT,
    ROBOT_QPOS,
    ROBOT_QVEL,
    apply_pd_gains,
    build_model,
    default_joint_qpos,
)


@dataclass
class EnvConfig:
    timestep: float = 0.002
    frame_skip: int = 10
    episode_seconds: float = 5.0
    # actuation
    action_scale: float = 0.25  # used when the gains file has no per-joint action scale
    act_scale_mult: float = 1.0
    pd_gains: str = ""
    ff_torque: bool = False  # reference inverse-dynamics torque, fed through the PD target
    ff_vel: bool = True  # with ff_torque, damp toward the reference joint velocity
    hand_collision: bool = False
    action_filter: float = 0.0  # a_f = f * a_f_prev + (1 - f) * a
    obs_prev_action: bool = False
    obs_yaw: bool = False
    # reference state initialisation
    p_rsi: float = 0.8
    rsi_joint_noise: float = 0.02
    rsi_root_noise: float = 0.005
    rsi_max_back: int = 10
    rsi_focus_frac: float = 0.0  # share of RSI starts drawn from [rsi_focus_lo, rsi_focus_hi]
    rsi_focus_lo: float = 0.0
    rsi_focus_hi: float = 1.0
    late_rsi_frac: float = 0.0  # share of RSI starts drawn from the last 40% of the clip
    # terminations
    fall_height: float = 0.18
    et_fall_height_rel: float = 0.0  # > 0: fall when the pelvis is this far below the reference pelvis
    fly_height: float = 1.4
    et_fall_hold: int = 3
    et_root_dev: float = 0.45
    et_torso_align: float = 0.75  # cosine between robot and reference torso z axes
    et_torso_hdev: float = 0.20
    et_upright_rel_deg: float = 60.0  # angle between robot and reference pelvis z axes
    root_drop: float = 0.20
    ground_contacts: dict = None
    # rewards and costs
    stages: list = None
    stage_weights: dict = None
    kernel: dict = None
    cost_spec: dict = None
    costs_off: tuple = ()
    penalty_cap: float = 0.5
    lagrange: tuple = ()
    tq_min_nm: float = 10.0
    cmd_k: float = 4.0
    cmd_yaw_k: float = 2.0
    foot_kp_w: float = 1.0
    foot_kp_tol: float = 0.0
    knee_flex_deg: float = 0.0
    knee_flex_sigma_deg: float = 10.0
    knee_flex_stages: list = None
    support_act_m: float = 0.04
    support_scale_m: float = 0.04
    w_upright_bonus: float = 0.0
    upright_bonus_k: float = 4.0
    clip_end_bonus: float = 0.0
    w_action_rate: float = 0.0
    w_action_jerk: float = 0.0
    w_dq_smooth: float = 0.0
    # training perturbations
    push_prob: float = 0.0
    push_force: float = 40.0
    push_steps: int = 5


FIELDS = {f.name for f in dataclasses.fields(EnvConfig)}
RUNTIME_FIELDS = ("p_rsi", "rsi_joint_noise", "rsi_root_noise", "push_prob", "lagrange")
EE_SITES = ("left_foot", "right_foot")
EE_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link", "left_knee_link", "right_knee_link")


class MotionClip:
    def __init__(self, qpos, qvel, ctrl, dt, meta):
        self.qpos, self.qvel, self.ctrl = qpos, qvel, ctrl
        self.n = len(qpos)
        self.dt = dt
        self.times = np.arange(self.n) * dt
        self.meta = meta

    @classmethod
    def load(cls, path: str | Path) -> "MotionClip":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        clip = cls(
            np.asarray(z["qpos"], np.float64),
            np.asarray(z["qvel"], np.float64),
            np.asarray(z["ctrl"], np.float64),
            float(meta.get("dt", 0.02)),
            meta,
        )
        clip.times = np.asarray(z["times"], np.float64)
        return clip

    def frame_at(self, t: float) -> int:
        return int(np.clip(int(round(t / self.dt)), 0, self.n - 1))


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _yaw(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def transform_clip(c: MotionClip, yaw: float, scale: float) -> MotionClip:
    """Rotate a clip about z through its first root position and play it `scale` times faster."""
    n2 = max(4, int(round((c.n - 1) / scale)) + 1)
    src = np.clip(np.arange(n2) * scale, 0, c.n - 1)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, c.n - 1)
    f = (src - i0)[:, None]
    lerp = lambda A: A[i0] * (1 - f) + A[i1] * f
    q, dq, u = lerp(c.qpos), lerp(c.qvel) * scale, lerp(c.ctrl)
    qa, qb = c.qpos[i0, 3:7], c.qpos[i1, 3:7].copy()
    qb[np.sum(qa * qb, 1) < 0] *= -1
    qq = qa * (1 - f) + qb * f
    q[:, 3:7] = qq / np.linalg.norm(qq, axis=1, keepdims=True)
    if yaw:
        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        p0 = q[0, :2].copy()
        q[:, :2] = (q[:, :2] - p0) @ R.T + p0
        dq[:, :2] = dq[:, :2] @ R.T
        qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        q[:, 3:7] = np.array([_qmul(qz, x) for x in q[:, 3:7]])
    nearest = np.clip(np.round(src).astype(int), 0, c.n - 1)

    def resample(x):
        if isinstance(x, list) and len(x) == c.n:
            return [x[j] for j in nearest]
        if isinstance(x, dict):
            return {k: resample(v) for k, v in x.items()}
        return x

    meta = resample(copy.deepcopy(c.meta))
    meta["dt"] = c.dt
    return MotionClip(q, dq, u, c.dt, meta)


class ImitationEnv:
    """One policy over the clips of a compiled skill config (see `wrestling.skills.load_skill`)."""

    def __init__(self, skill: dict, overrides: dict | None = None, seed: int = 0):
        self.skill = skill
        self.clip_specs = skill["clips"]
        self.skills = skill["skills"]
        self._overrides = dict(overrides or {})
        self.rng = np.random.default_rng(seed)
        self.cfg = self._make_cfg(0)
        self.model = build_model(self.cfg.timestep, hand_collision=self.cfg.hand_collision)
        self._act_scale = apply_pd_gains(self.model, self.cfg.pd_gains)
        if self.cfg.ff_torque:
            # The PD target may leave the joint range to produce the feedforward torque; force stays clamped.
            self.model.actuator_ctrllimited[:] = 0
        self.data = mujoco.MjData(self.model)
        self.default_j = default_joint_qpos(self.model)
        self.pelvis = self.model.body("pelvis").id
        self.torso = self.model.body("torso_link").id
        self.ee_site_ids = [self.model.site(n).id for n in EE_SITES]
        self.ee_body_ids = [self.model.body(n).id for n in EE_BODIES]
        self.hand_ids = [self.model.body(n).id for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
        self.base_clips = [MotionClip.load(c["path"]) for c in self.clip_specs]
        w = np.array([c.get("weight", 1.0) for c in self.clip_specs], float)
        self.w = w / w.sum()
        self._cache = {}
        self.cmd_win = 0.5
        self.act_dim = N_JOINT
        self.obs_dim = (
            3
            + 3
            + 3
            + 3 * N_JOINT
            + 3
            + 3
            + 2
            + (N_JOINT if self.cfg.obs_prev_action else 0)
            + (2 if self.cfg.obs_yaw else 0)
            + len(self.skills)
        )
        self.step_count = 0
        self._fall_streak = 0
        self._push_left = 0
        self._clear_action_history()
        self.select_clip(0, 0.0, 1.0)

    def _make_cfg(self, k: int) -> EnvConfig:
        c = {**self.skill["clip_cfg"][k], **self._overrides}
        return EnvConfig(**{f: copy.deepcopy(v) for f, v in c.items() if f in FIELDS})

    def _clear_action_history(self):
        z = lambda: np.zeros(N_JOINT)
        self._a_f, self._a_prev, self._a_prev2, self._af_prev, self._af_prev2 = z(), z(), z(), z(), z()

    # Clip selection, yaw/time augmentation and per-clip reference data.

    def _aug_grid(self, k):
        a = {**self.skill.get("augment", {}), **self.clip_specs[k].get("augment", {})}
        yl, yh = a.get("yaw_deg", [0, 0])
        sl, sh = a.get("time_scale", [1, 1])
        ys = (
            np.linspace(yl, yh, int(a.get("yaw_bins", 12)), endpoint=(yh - yl) < 360)
            if yh > yl
            else np.array([yl])
        )
        ss = np.linspace(sl, sh, int(a.get("scale_bins", 5))) if sh > sl else np.array([sl])
        return np.deg2rad(ys), ss

    def select_clip(self, k: int, yaw: float | None = None, scale: float | None = None):
        if yaw is None or scale is None:
            ys, ss = self._aug_grid(k)
            yaw = float(self.rng.choice(ys))
            scale = float(self.rng.choice(ss))
        key = (k, round(yaw, 4), round(scale, 4))
        if key not in self._cache:
            if len(self._cache) > 256:
                self._cache.pop(next(iter(self._cache)))
            self.clip = (
                transform_clip(self.base_clips[k], yaw, scale)
                if (yaw or scale != 1.0)
                else self.base_clips[k]
            )
            self._apply_cfg(k)
            self._precompute_ref()
            self._cache[key] = (self.clip, self.ref, ConstraintTerms(self))
        self.clip, self.ref, self.terms = self._cache[key]
        self._apply_cfg(k)
        (
            self.ref_ee,
            self.ref_com,
            self.ref_torso_z,
            self.ref_torso_h,
            self.ref_torso_yaw,
            self.ref_pel_z,
            self.ff_off,
        ) = self.ref
        self._forbidden_geoms = None
        self.max_steps = int(self.cfg.episode_seconds / (self.cfg.timestep * self.cfg.frame_skip))
        self.clip_idx = k
        self._ref_command()
        return self.clip

    def _apply_cfg(self, k):
        new = self._make_cfg(k)
        for f in FIELDS - set(RUNTIME_FIELDS):
            setattr(self.cfg, f, getattr(new, f))

    def _precompute_ref(self):
        d = mujoco.MjData(self.model)
        c = self.clip
        ee, com, tz, th, ty, pz = [], [], [], [], [], []
        for i in range(c.n):
            d.qpos[:] = c.qpos[i]
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            ee.append(
                [d.site_xpos[j].copy() for j in self.ee_site_ids]
                + [d.xpos[j].copy() for j in self.ee_body_ids]
            )
            com.append(d.subtree_com[1].copy())
            R = d.xmat[self.torso].reshape(3, 3)
            tz.append(R[:, 2].copy())
            th.append(float(d.xpos[self.torso][2]))
            ty.append(float(np.arctan2(R[1, 0], R[0, 0])))
            pz.append(d.xmat[self.pelvis].reshape(3, 3)[:, 2].copy())
        ff = None
        if self.cfg.ff_torque:
            tau = reference_torque(self.model, c.qpos, c.qvel, c.dt)
            ff = ff_ctrl_offset(self.model, tau, c.qvel, vel=self.cfg.ff_vel)
        self.ref = (np.array(ee), np.array(com), np.array(tz), np.array(th), np.array(ty), np.array(pz), ff)

    def _ref_command(self):
        """Reference root velocity and yaw rate over the next 0.5 s, in the reference heading frame."""
        q = self.clip.qpos
        n = self.clip.n
        h = max(1, int(round(self.cmd_win / self.clip.dt)))
        j1 = np.minimum(np.arange(n) + h, n - 1)
        span = np.maximum(1, j1 - np.arange(n)) * self.clip.dt
        yaw = np.unwrap([_yaw(x) for x in q[:, 3:7]])
        v = (q[j1, :2] - q[:, :2]) / span[:, None]
        c, s = np.cos(yaw), np.sin(yaw)
        self.ref_cmd = np.c_[c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], (yaw[j1] - yaw) / span]
        tail = n - 1 - h
        if tail > 0:
            self.ref_cmd[tail:] = self.ref_cmd[tail]

    # Observation.

    def _ee(self) -> np.ndarray:
        return np.array(
            [self.data.site_xpos[i] for i in self.ee_site_ids] + [self.data.xpos[i] for i in self.ee_body_ids]
        )

    def _dyaw(self, i: int) -> float:
        R = self.data.xmat[self.torso].reshape(3, 3)
        e = float(np.arctan2(R[1, 0], R[0, 0])) - self.ref_torso_yaw[i]
        return float((e + np.pi) % (2 * np.pi) - np.pi)

    @staticmethod
    def _gravity(quat):
        qw, qx, qy, qz = quat
        return np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])

    def _obs(self) -> np.ndarray:
        d = self.data
        qj = d.qpos[FREE_QPOS:ROBOT_QPOS]
        nxt = min(self.frame_i + 1, self.clip.n - 1)
        phase = self.frame_i / max(1, self.clip.n - 1)
        parts = [
            d.qvel[3:6] * 0.25,
            self._gravity(d.qpos[3:7]),
            d.qvel[0:3] * 0.5,
            qj,
            d.qvel[FREE_QVEL:ROBOT_QVEL] * 0.05,
            self.clip.qpos[nxt, FREE_QPOS:] - qj,
            (self.ref_com[nxt] - d.subtree_com[1]) * 2.0,
            (self.clip.qpos[nxt, :3] - d.qpos[:3]) * 2.0,
            [np.sin(np.pi * phase), np.cos(np.pi * phase)],
        ]
        if self.cfg.obs_prev_action:
            parts.append(self._a_f)
        if self.cfg.obs_yaw:
            dy = self._dyaw(nxt)
            parts.append([np.sin(dy), np.cos(dy) - 1.0])
        onehot = np.zeros(len(self.skills))
        onehot[self.skills.index(self.clip_specs[self.clip_idx].get("skill", self.skills[0]))] = 1
        parts.append(onehot)
        o = np.concatenate(parts).astype(np.float32)
        # Root velocity, CoM error and root error are expressed in the robot heading frame.
        yw = _yaw(d.qpos[3:7])
        c, s = np.cos(yw), np.sin(yw)
        for k in (6, 9 + 3 * N_JOINT, 12 + 3 * N_JOINT):
            v = o[k : k + 3]
            o[k : k + 3] = [c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]]
        return o

    # Episode control.

    def start(self, frame: int = 0, noise: bool = True) -> np.ndarray:
        """Set the robot to a reference frame (with optional start noise) and begin an episode."""
        self.step_count = 0
        self._fall_streak = 0
        self._push_left = 0
        self.data.xfrc_applied[:] = 0
        self.terms.reset()
        self._set_state_from_frame(frame, noise)
        return self._obs()

    def _set_state_from_frame(self, i: int, noise: bool = True) -> None:
        i = int(np.clip(i, 0, self.clip.n - 1))
        q = self.clip.qpos[i].copy()
        if noise:
            q[FREE_QPOS:] += self.rng.normal(0, self.cfg.rsi_joint_noise, N_JOINT)
            q[:2] += self.rng.normal(0, self.cfg.rsi_root_noise, 2)
        d = self.data
        d.qpos[:] = q
        d.qvel[:] = self.clip.qvel[i]
        d.ctrl[:] = self.clip.ctrl[i] + (self.ff_off[i] if self.ff_off is not None else 0.0)
        mujoco.mj_forward(self.model, d)
        lowest = min(d.site_xpos[j][2] for j in self.ee_site_ids)
        if lowest < 0.015:
            d.qpos[2] += 0.015 - lowest
            mujoco.mj_forward(self.model, d)
        self.frame_i = i
        self.phase_t = float(self.clip.times[i])
        self._dq_prev = d.qvel[FREE_QVEL:ROBOT_QVEL].copy()
        self._clear_action_history()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.select_clip(int(self.rng.choice(len(self.w), p=self.w)))
        self.step_count = 0
        self._fall_streak = 0
        self._push_left = 0
        self._clear_action_history()
        self.terms.reset()
        frame = 0
        if self.rng.random() < self.cfg.p_rsi:
            n = self.clip.n
            hi = max(1, n - self.cfg.rsi_max_back)
            if self.rng.random() < self.cfg.rsi_focus_frac:
                lo = int(self.cfg.rsi_focus_lo * n)
                frame = int(self.rng.integers(lo, max(lo + 1, min(hi, int(self.cfg.rsi_focus_hi * n)))))
            elif self.rng.random() < self.cfg.late_rsi_frac:
                frame = int(self.rng.integers(int(0.6 * n), hi))
            else:
                frame = int(self.rng.integers(0, hi))
        self._set_state_from_frame(frame, noise=True)
        return self._obs()

    # Dynamics, reward and termination.

    def _tracking(self) -> dict:
        d, i = self.data, self.frame_i
        qj = d.qpos[FREE_QPOS:ROBOT_QPOS]
        de = np.linalg.norm(self._ee() - self.ref_ee[i], axis=1)
        if self.cfg.foot_kp_tol > 0 or self.cfg.foot_kp_w != 1.0:
            de[:2] = np.sqrt(self.cfg.foot_kp_w) * np.maximum(0.0, de[:2] - self.cfg.foot_kp_tol)
        root_err = float(np.sum((d.qpos[:3] - self.clip.qpos[i, :3]) ** 2))
        info = {
            "joint_mse": float(np.mean((qj - self.clip.qpos[i, FREE_QPOS:]) ** 2)),
            "ee_err": float(np.sum(de**2)),
            "root_mse": root_err,
            "root_dev": float(np.sqrt(root_err)),
            "torso_align": float(d.xmat[self.torso].reshape(3, 3)[:, 2] @ self.ref_torso_z[i]),
            "torso_hdev": float(abs(d.xpos[self.torso][2] - self.ref_torso_h[i])),
            "height": float(d.qpos[2]),
            "frame": i,
        }
        yw = _yaw(d.qpos[3:7])
        c, s = np.cos(yw), np.sin(yw)
        v = d.qvel[0:2]
        vh = np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])
        cmd = self.ref_cmd[i]
        info["r_cmd_vel"] = float(np.exp(-self.cfg.cmd_k * np.sum((vh - cmd[:2]) ** 2)))
        info["r_cmd_yaw"] = float(np.exp(-self.cfg.cmd_yaw_k * (d.qvel[5] - cmd[2]) ** 2))
        return info

    def _forbidden_contact(self) -> bool:
        gc = self.cfg.ground_contacts
        if not gc or not gc.get("bodies"):
            return False
        m = self.model
        if self._forbidden_geoms is None:
            names = {n.split(":")[0] for k in gc.get("forbidden", []) for n in gc["bodies"].get(k, [])}
            self._forbidden_geoms = {
                g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and m.body(m.geom_bodyid[g]).name in names
            }
        for k in range(self.data.ncon):
            c = self.data.contact[k]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 == 0 and c.geom2 in self._forbidden_geoms) or (
                b2 == 0 and c.geom1 in self._forbidden_geoms
            ):
                return True
        return False

    def _fallen(self, info) -> bool:
        cfg = self.cfg
        h = info["height"]
        if cfg.et_fall_height_rel > 0:
            low = h < self.clip.qpos[min(self.frame_i, self.clip.n - 1), 2] - cfg.et_fall_height_rel
        else:
            low = h < cfg.fall_height
        bad = (
            low
            or h > cfg.fly_height
            or info["pelvis_ref_ang_deg"] > cfg.et_upright_rel_deg
            or info["torso_align"] < cfg.et_torso_align
            or info["torso_hdev"] > cfg.et_torso_hdev
            or info["constraint_term"]
            or self._forbidden_contact()
        )
        self._fall_streak = self._fall_streak + 1 if bad else 0
        return self._fall_streak >= cfg.et_fall_hold or info["root_dev"] > cfg.et_root_dev

    def step(self, action):
        cfg, d = self.cfg, self.data
        raw = np.clip(np.asarray(action, np.float64).reshape(N_JOINT), -1, 1)
        self._a_f = (
            cfg.action_filter * self._a_f + (1 - cfg.action_filter) * raw if cfg.action_filter > 0 else raw
        )
        nf = min(self.frame_i + 1, self.clip.n - 1)
        scale = self._act_scale if self._act_scale is not None else cfg.action_scale
        d.ctrl[:] = (
            self.clip.ctrl[nf]
            + (self.ff_off[nf] if self.ff_off is not None else 0.0)
            + cfg.act_scale_mult * scale * self._a_f
        )
        if cfg.push_prob > 0:
            if self._push_left <= 0 and self.rng.random() < cfg.push_prob:
                ang = self.rng.uniform(0, 2 * np.pi)
                mag = self.rng.uniform(0.3, 1.0) * cfg.push_force
                self._push_vec = np.array([np.cos(ang) * mag, np.sin(ang) * mag, 0.0])
                self._push_left = cfg.push_steps
            if self._push_left > 0:
                d.xfrc_applied[self.pelvis, :3] = self._push_vec
                self._push_left -= 1
            else:
                d.xfrc_applied[self.pelvis, :3] = 0.0
        for _ in range(cfg.frame_skip):
            mujoco.mj_step(self.model, d)
        self.phase_t += cfg.timestep * cfg.frame_skip
        self.frame_i = self.clip.frame_at(self.phase_t)
        self.step_count += 1

        info = self._tracking()
        if len(cfg.lagrange):
            self.terms.lam = np.asarray(cfg.lagrange, float)
        rew, ci = self.terms.step(self._a_f, self._af_prev, self._af_prev2, info)
        info.update(ci)
        if cfg.w_action_rate > 0 or cfg.w_action_jerk > 0:
            da = raw - self._a_prev
            dda = raw - 2 * self._a_prev + self._a_prev2
            pen = cfg.w_action_rate * float(np.mean(da**2)) + cfg.w_action_jerk * float(np.mean(dda**2))
            rew -= pen
        pz = d.xmat[self.pelvis].reshape(3, 3)[:, 2]
        ang = float(np.arccos(np.clip(pz @ self.ref_pel_z[self.frame_i], -1, 1)))
        info["pelvis_ref_ang_deg"] = float(np.degrees(ang))
        if cfg.w_upright_bonus > 0:
            rew += cfg.w_upright_bonus * float(np.exp(-cfg.upright_bonus_k * ang * ang))
        self._a_prev2, self._a_prev = self._a_prev, raw.copy()
        self._af_prev2, self._af_prev = self._af_prev, self._a_f.copy()
        dq = d.qvel[FREE_QVEL:ROBOT_QVEL].copy()
        if cfg.w_dq_smooth > 0:
            fi = self.frame_i
            dref = self.clip.qvel[fi, FREE_QVEL:] - self.clip.qvel[max(fi - 1, 0), FREE_QVEL:]
            rew -= cfg.w_dq_smooth * float(np.mean(((dq - self._dq_prev) - dref) ** 2))
        self._dq_prev = dq

        done = True
        if self._fallen(info):
            info["term"] = "fall"
        elif self.frame_i >= self.clip.n - 1:
            info["term"] = "clip_end"
            rew += cfg.clip_end_bonus
        elif self.step_count >= self.max_steps:
            info["term"] = "timeout"
        else:
            done = False
        info["clip"] = self.clip_specs[self.clip_idx]["id"]
        return self._obs(), rew, done, info
