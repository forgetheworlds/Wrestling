"""Reference torque feedforward (inverse dynamics with a contact-force split).

Position-PD alone sags under load: steady-state error = tau_gravity / kp, and a crouched G1 knee needs ~60-90 Nm,
i.e. 0.4-0.6 rad of error at kp=150. The real G1 low-level command is (q, dq, kp, kd, tau), so we compute tau from
the reference and feed it forward:
  1. mj_inverse (contacts disabled) on the reference (q, dq, ddq) -> generalised force the whole body needs.
  2. Its 6 floating-base rows can only come from ground contact: solve min ||J_root^T f - b||^2 + reg||f||^2 over the
     lowest points of the geoms touching the floor in the reference (feet, knees, hands), with fz >= 0.
  3. tau_joint = qfrc[joints] - J_joint^T f, clipped to the G1 motor limits (jnt_actfrcrange).
With position actuators the torque is applied by shifting the PD target: ctrl = q_ref + (tau + kd*dq_ref) / kp.
The actuator force (incl. the ff) is still clamped to the real motor limit by MuJoCo.
"""

from __future__ import annotations
import numpy as np, mujoco
from scipy.optimize import lsq_linear

CONTACT_BODIES = ("ankle_roll_link", "knee_link", "wrist_yaw_link", "wrist_pitch_link", "wrist_roll_link")


def _contact_points(m, d, tol):
    """Lowest point of every contact-capable foot/knee/hand geom within tol of the mat (exact mj_geomDistance)."""
    floor = [g for g in range(m.ngeom) if m.geom_bodyid[g] == 0]
    pts = []
    ft = np.zeros(6)
    for g in range(m.ngeom):
        if not (m.geom_contype[g] | m.geom_conaffinity[g]) or m.geom_bodyid[g] == 0:
            continue
        if not any(s in m.body(m.geom_bodyid[g]).name for s in CONTACT_BODIES):
            continue
        if d.geom_xpos[g][2] - m.geom_rbound[g] > tol:
            continue
        for f in floor:
            dist = mujoco.mj_geomDistance(m, d, g, f, tol + 0.05, ft)
            if dist < tol:
                pts.append((m.geom_bodyid[g], np.array([ft[0], ft[1], 0.0])))
    return pts


def reference_torque(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    dt: float,
    static: bool = False,
    contact_tol: float = 0.03,
    reg: float = 1e-4,
    limit_frac: float = 0.9,
) -> np.ndarray:
    """Per-frame actuator feedforward torque (n, nu). static=True: zero velocity/acceleration (pure load holding)."""
    m = model
    flags = m.opt.disableflags
    m.opt.disableflags = flags | mujoco.mjtDisableBit.mjDSBL_CONTACT
    d = mujoco.MjData(m)
    n = len(qpos)
    qacc = np.zeros_like(qvel) if static else np.gradient(qvel, dt, axis=0)
    if not static:  # light smoothing of the finite-difference acceleration
        k = np.array([1, 2, 3, 2, 1], float)
        k /= k.sum()
        qacc = np.stack(
            [np.convolve(np.pad(qacc[:, j], 2, mode="edge"), k, "valid") for j in range(qacc.shape[1])], 1
        )
    dof = m.jnt_dofadr[m.actuator_trnid[:, 0]]
    lim = m.jnt_actfrcrange[m.actuator_trnid[:, 0], 1] * limit_frac
    tau = np.zeros((n, m.nu))
    jp = np.zeros((3, m.nv))
    try:
        for i in range(n):
            d.qpos[:] = qpos[i]
            d.qvel[:] = 0 if static else qvel[i]
            d.qacc[:] = qacc[i]
            mujoco.mj_inverse(m, d)
            f_all = d.qfrc_inverse.copy()
            pts = _contact_points(m, d, contact_tol)
            if pts:
                J = []
                for b, p in pts:
                    mujoco.mj_jac(m, d, jp, None, p, b)
                    J.append(jp.copy())
                J = np.concatenate(J, 0)  # (3k, nv)
                A = np.vstack([J[:, :6].T, np.sqrt(reg) * np.eye(J.shape[0])])
                bvec = np.concatenate([f_all[:6], np.zeros(J.shape[0])])
                lo = np.tile([-np.inf, -np.inf, 0.0], len(pts))
                hi = np.full(J.shape[0], np.inf)
                f = lsq_linear(A, bvec, bounds=(lo, hi), method="bvls").x
                f_all = f_all - J.T @ f
            tau[i] = np.clip(f_all[dof], -lim, lim)
    finally:
        m.opt.disableflags = flags
    return tau


def ff_ctrl_offset(
    model: mujoco.MjModel, tau: np.ndarray, qvel: np.ndarray | None = None, vel: bool = True
) -> np.ndarray:
    """PD-target shift that makes a position actuator output tau (+ kd*dq_ref damping toward the reference velocity)."""
    kp = model.actuator_gainprm[:, 0]
    kd = -model.actuator_biasprm[:, 2]
    off = tau / kp
    if vel and qvel is not None:
        dof = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        off = off + kd * qvel[:, dof] / kp
    return off
