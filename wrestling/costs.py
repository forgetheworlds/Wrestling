"""Constrained imitation terms: bounded rewards, hinge costs with Lagrange multipliers, stages.

- Reward: every term is in [0, 1] and the weights of each stage are normalised, so r_task is in [0, 1].
- Costs: c_j = clip((x_j - d_j) / s_j, 0, 1), zero while the constraint holds. The penalty
  sum_j lambda_j * c_j uses multipliers that the trainer updates from batch cost means and budgets.
- Termination: forbidden body groups touching the mat, or the pelvis dropping below the reference.

Anything the reference itself does (joint-limit proximity, torque, self-contact, foot skate,
support margin) is never counted as a violation.
"""

from __future__ import annotations

import numpy as np
import mujoco

COST_NAMES = (
    "body_contact",
    "self_collision",
    "joint_limit",
    "torque_spike",
    "action_jerk",
    "contact_slip",
    "hold_jitter",
)
COST_SPEC = {
    "body_contact": (0.0, 1.0),
    "self_collision": (0.0, 1.0),
    "joint_limit": (0.0, 1.0),
    "torque_spike": (0.9, 0.1),
    "action_jerk": (0.6, 0.6),
    "contact_slip": (0.10, 0.20),
    "hold_jitter": (0.5, 1.0),
}
# pose, keypoints, root, contact, hand keypoints, hold smoothness, cmd vel, cmd yaw, knee flex, support margin
DEFAULT_WEIGHTS = {
    "plain": (0.35, 0.25, 0.30, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "hands": (0.20, 0.15, 0.15, 0.20, 0.20, 0.10, 0.0, 0.0, 0.0, 0.0),
}
SMOOTH_WINDOW = 3


def polygon_margin(pts: np.ndarray, p: np.ndarray) -> float:
    """Signed distance from p to the convex hull of 2D points (positive inside)."""
    P = np.unique(np.round(pts, 4), axis=0)
    if len(P) >= 3:
        P = P[np.lexsort((P[:, 1], P[:, 0]))]
        cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lo, up = [], []
        for q in P:
            while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
                lo.pop()
            lo.append(q)
        for q in P[::-1]:
            while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0:
                up.pop()
            up.append(q)
        H = np.array(lo[:-1] + up[:-1])
        if len(H) >= 3:
            e = np.roll(H, -1, 0) - H
            n = np.c_[e[:, 1], -e[:, 0]] / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
            sd = -((p - H) * n).sum(1)
            if np.all(sd >= 0):
                return float(sd.min())
    Q = P if len(P) < 3 else H
    best = np.inf
    for a, b in zip(Q, np.roll(Q, -1, 0) if len(Q) > 1 else Q):
        ab = b - a
        t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return -best


def _window_max(x: np.ndarray, w: int) -> np.ndarray:
    return np.array([x[max(0, i - w) : i + w + 1].max() for i in range(len(x))])


class ConstraintTerms:
    def __init__(self, env):
        self.env = env
        m = self.m = env.model
        cfg = env.cfg
        clip = env.clip
        d = mujoco.MjData(m)
        self.floor = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0]
        robot = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0]
        body_name = lambda g: m.body(m.geom_bodyid[g]).name or ""
        gc = cfg.ground_contacts or {}
        groups = gc.get("bodies", {})
        names = lambda keys: {n.split(":")[0] for k in keys for n in groups.get(k, [])}
        allowed, forbidden = names(gc.get("allowed", ["feet"])), names(gc.get("forbidden", []))
        self.allowed = {g for g in robot if body_name(g) in allowed}
        self.term = {g for g in robot if body_name(g) in forbidden}
        self.feet = [m.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.hands = list(env.hand_ids)
        self.foot_geoms = [
            g for g in robot if m.geom_bodyid[g] in self.feet and m.geom_contype[g] | m.geom_conaffinity[g]
        ]

        # Reference contacts: feet and wrists close to the mat, or the hand schedule stored with the motion.
        cb = self.feet + self.hands
        hz = np.zeros((clip.n, 4))
        for i in range(clip.n):
            d.qpos[:] = clip.qpos[i]
            mujoco.mj_kinematics(m, d)
            hz[i] = [d.xpos[b][2] for b in cb]
        self.ref_c = np.c_[hz[:, :2] < 0.06, hz[:, 2:] < 0.13]
        self.ref_hz = hz[:, 2:]
        hc = clip.meta.get("hand_contact")
        if hc:
            self.ref_c[:, 2] = np.array(hc["left"][: clip.n], bool)
            self.ref_c[:, 3] = np.array(hc["right"][: clip.n], bool)

        # Stages: [{name, lo, hi, hands_down_frac}] from the config or the motion file; one stage otherwise.
        stages = cfg.stages or clip.meta.get("cmo_stages") or [{"name": "s0", "lo": 0.0, "hi": 1.0}]
        self.stage = np.zeros(clip.n, int)
        self.stage_names, self.kind, self.hold = [], [], set()
        for k, s in enumerate(stages):
            self.stage[int(round(s["lo"] * (clip.n - 1))) :] = k
            self.stage_names.append(s["name"])
            hands = s.get("hands_down_frac", 0.0) >= 0.4
            self.kind.append("hands" if hands else "plain")
            if hands:
                self.hold.add(k)
        self.ref_pz = clip.qpos[:, 2]

        sw = cfg.stage_weights or {}
        self.stage_w = {}
        for kind, base in DEFAULT_WEIGHTS.items():
            w = np.array(sw.get(kind, sw.get("*", base)), float)
            if len(w) < 10:
                w = np.r_[w, np.zeros(10 - len(w))]
            self.stage_w[kind] = w / max(w.sum(), 1e-9)
        self.k = {"pose": 8.0, "keypoints": 10.0, "root": 20.0} | (cfg.kernel or {})
        self.cspec = dict(COST_SPEC) | {n: tuple(v) for n, v in (cfg.cost_spec or {}).items()}
        self.off = np.array([0.0 if n in (cfg.costs_off or ()) else 1.0 for n in COST_NAMES])
        self.lam = np.zeros(len(COST_NAMES))

        lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
        qr = clip.qpos[:, 7:]
        self.ref_prox = np.max(np.maximum(0.0, 1.0 - np.minimum(qr - lo, hi - qr) / 0.03), axis=1)
        self.tqmax = np.where(m.jnt_actfrcrange[1:, 1] > 0, m.jnt_actfrcrange[1:, 1], 100.0)[: m.nu]
        # The 5 Nm wrist motors saturate on exploration noise alone, so torque spikes ignore them.
        self.tq_mask = self.tqmax >= cfg.tq_min_nm

        # Self-collision ignores bodies at most 3 joints apart: the G1 arm and torso capsules overlap at rest.
        parent = m.body_parentid

        def chain(b):
            out = [b]
            while out[-1] != 0:
                out.append(int(parent[out[-1]]))
            return out

        def hops(b1, b2):
            a, b = chain(b1), chain(b2)
            common = next(x for x in a if x in b)
            return a.index(common) + b.index(common)

        self.near = np.array([[hops(a, b) <= 3 for b in range(m.nbody)] for a in range(m.nbody)])
        self._prev_cpos = None

        ref_self, self.ref_tq = [], np.zeros(clip.n)
        for i in range(clip.n):
            d.qpos[:] = clip.qpos[i]
            d.qvel[:] = clip.qvel[i]
            d.ctrl[:] = clip.ctrl[i]
            mujoco.mj_forward(m, d)
            self.ref_tq[i] = float(np.max((np.abs(d.actuator_force) / self.tqmax)[self.tq_mask]))
            ref_self.append(
                {
                    self._pair(m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])
                    for c in d.contact[: d.ncon]
                    if m.geom_bodyid[c.geom1] != 0 and m.geom_bodyid[c.geom2] != 0
                }
            )
        self.ref_slip = np.zeros(clip.n)
        self.ref_margin = np.full(clip.n, np.inf)
        prev = None
        dt = m.opt.timestep * cfg.frame_skip
        for i in range(clip.n):
            d.qpos[:] = clip.qpos[i]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            cp = np.array([d.xpos[b] for b in self.feet + self.hands])
            if prev is not None:
                v = np.linalg.norm(cp[:, :2] - prev[:, :2], axis=1) / max(dt, clip.dt)
                down = self.ref_c[i]
                self.ref_slip[i] = float(v[down].max()) if down.any() else 0.0
            prev = cp
            mg = self._margin_of(d)
            if mg is not None:
                self.ref_margin[i] = mg
        W = SMOOTH_WINDOW
        self.ref_slip = _window_max(self.ref_slip, W)
        self.ref_tq = _window_max(self.ref_tq, W)
        self.ref_self = [set().union(*ref_self[max(0, i - W) : i + W + 1]) for i in range(clip.n)]

        self.knee_adr = [m.joint(n).qposadr[0] for n in ("left_knee_joint", "right_knee_joint")]
        ks = cfg.knee_flex_stages
        self.knee_on = [
            any(x == "*" or x in name for x in ks) if ks else kind != "hands"
            for name, kind in zip(self.stage_names, self.kind)
        ]

    @staticmethod
    def _pair(b1, b2):
        return (int(min(b1, b2)), int(max(b1, b2)))

    def _margin_of(self, d):
        """CoM distance to the edge of the support polygon of the feet on the mat (None in flight)."""
        on = [g for g in self.foot_geoms if d.geom_xpos[g][2] < self.m.geom_size[g][0] + 0.02]
        if not on:
            return None
        # A foot that touches at any corner contributes its whole sole.
        bodies = {self.m.geom_bodyid[g] for g in on}
        pts = [d.geom_xpos[g][:2].copy() for g in self.foot_geoms if self.m.geom_bodyid[g] in bodies]
        return polygon_margin(np.array(pts), d.subtree_com[1][:2])

    def _contacts(self):
        d = self.env.data
        body_hit = term_hit = self_hit = 0
        for k in range(d.ncon):
            c = d.contact[k]
            f1, f2 = c.geom1 in self.floor, c.geom2 in self.floor
            if f1 ^ f2:
                g = c.geom2 if f1 else c.geom1
                body_hit += g not in self.allowed
                term_hit += g in self.term
            elif not f1 and not f2:
                b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
                if not self.near[b1, b2] and self._pair(b1, b2) not in self.ref_self[self.env.frame_i]:
                    self_hit += 1
        return body_hit, term_hit, self_hit

    def step(self, a, a_prev, a_prev2, base):
        env, d, m, cfg = self.env, self.env.data, self.m, self.env.cfg
        i = env.frame_i
        stage = self.stage[i]
        kind = self.kind[stage]

        cpos = np.array([d.xpos[b] for b in self.feet + self.hands])
        ref_hands = env.ref_ee[i][2:4]
        hxy = np.linalg.norm(cpos[2:, :2] - ref_hands[:, :2], axis=1)
        rc = self.ref_c[i, 2:]
        posted = np.where(
            rc, (np.abs(cpos[2:, 2] - self.ref_hz[i]) < 0.04) & (hxy < 0.10), cpos[2:, 2] < 0.07
        )
        sim_c = np.r_[cpos[:2, 2] < 0.06, posted]
        match = sim_c == self.ref_c[i]
        contact = float(np.mean(match[:2])) * float(min(match[2], match[3]))
        hand_kp = float(
            min(
                np.exp(-20.0 * np.sum((cpos[2] - ref_hands[0]) ** 2)),
                np.exp(-20.0 * np.sum((cpos[3] - ref_hands[1]) ** 2)),
            )
        )
        hold = 0.0
        if stage in self.hold:
            hold = float(np.linalg.norm(d.qvel[3:6])) + float(
                np.sqrt(np.mean((d.qvel[6:] - env.clip.qvel[i, 6:]) ** 2))
            )

        r_knee = 0.0
        if cfg.knee_flex_deg > 0:
            r_knee = 1.0
            if self.knee_on[stage]:
                deficit = np.maximum(0.0, cfg.knee_flex_deg - np.degrees(d.qpos[self.knee_adr]))
                r_knee = float(np.mean(np.exp(-((deficit / cfg.knee_flex_sigma_deg) ** 2))))
        r_sup = 1.0
        if self.stage_w[kind][9] > 0:
            mg = self._margin_of(d)
            act = min(cfg.support_act_m, self.ref_margin[i])
            if mg is not None and mg < act:
                r_sup = float(np.exp(-(((act - mg) / cfg.support_scale_m) ** 2)))

        K = self.k
        terms = np.array(
            [
                np.exp(-K["pose"] * base["joint_mse"]),
                np.exp(-K["keypoints"] * base["ee_err"]),
                np.exp(-K["root"] * base["root_mse"]),
                contact,
                hand_kp,
                np.exp(-hold),
                base.get("r_cmd_vel", 0.0),
                base.get("r_cmd_yaw", 0.0),
                r_knee,
                r_sup,
            ]
        )
        r_task = float(terms @ self.stage_w[kind])

        body_hit, term_hit, self_hit = self._contacts()
        q = d.qpos[7:]
        lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
        prox = max(
            0.0, float(np.max(np.maximum(0.0, 1.0 - np.minimum(q - lo, hi - q) / 0.03))) - self.ref_prox[i]
        )
        tq = float(np.max((np.abs(d.actuator_force) / self.tqmax)[self.tq_mask]))
        tq = max(0.0, tq - max(0.0, self.ref_tq[i] - self.cspec["torque_spike"][0]))
        jerk = float(np.mean((a - 2 * a_prev + a_prev2) ** 2))
        slip = 0.0
        if self._prev_cpos is not None:
            v = np.linalg.norm(cpos[:, :2] - self._prev_cpos[:, :2], axis=1) / (
                m.opt.timestep * cfg.frame_skip
            )
            both = sim_c & self.ref_c[i]
            slip = max(0.0, (float(v[both].max()) if both.any() else 0.0) - self.ref_slip[i])
        self._prev_cpos = cpos
        x = {
            "hold_jitter": hold,
            "body_contact": float(body_hit > 0),
            "self_collision": float(self_hit > 0),
            "joint_limit": prox,
            "torque_spike": tq,
            "action_jerk": jerk,
            "contact_slip": slip,
        }
        c = (
            np.array([np.clip((x[n] - self.cspec[n][0]) / self.cspec[n][1], 0.0, 1.0) for n in COST_NAMES])
            * self.off
        )
        r = r_task - min(float(self.lam @ c), cfg.penalty_cap)
        bad = bool(term_hit) or d.qpos[2] < self.ref_pz[i] - cfg.root_drop or not np.isfinite(d.qpos).all()
        info = dict(
            hand_post=sim_c[2:].copy(),
            ref_hand_post=self.ref_c[i, 2:].copy(),
            r_task=r_task,
            costs=c,
            stage=stage,
            constraint_term=bad,
            terms=terms,
        )
        return r, info

    def reset(self):
        self._prev_cpos = None
