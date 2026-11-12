"""Check a reference motion, or import one from a CSV of G1 poses.

    python -m wrestling.clip check motions/sprawl.npz
    python -m wrestling.clip import my_move.csv --fps 30 --name my_move

A motion file is an .npz at 50 Hz (see docs/EXTENDING.md). CSV rows are one frame each:
root x y z (m), root quaternion (xyzw by default), then the 29 G1 joint angles (rad) in model order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from . import PHYSICS, ROOT
from .robot import FREE_QPOS, N_JOINT, ROBOT_QPOS, ROBOT_QVEL, build_model

DT = PHYSICS["timestep"] * PHYSICS["frame_skip"]


def joint_names(m: mujoco.MjModel) -> list[str]:
    return [m.joint(i).name for i in range(1, m.njnt)]


def check(path: str | Path) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings). Errors stop training from working; warnings usually hurt it."""
    err, warn = [], []
    z = np.load(path, allow_pickle=False)
    missing = {"qpos", "qvel", "ctrl", "times", "meta_json"} - set(z.files)
    if missing:
        return [f"missing arrays: {sorted(missing)}"], warn
    q, v, u, t = (np.asarray(z[k], float) for k in ("qpos", "qvel", "ctrl", "times"))
    meta = json.loads(str(z["meta_json"]))
    n = len(q)
    for name, a, w in (("qpos", q, ROBOT_QPOS), ("qvel", v, ROBOT_QVEL), ("ctrl", u, N_JOINT)):
        if a.shape != (n, w):
            err.append(f"{name} shape {a.shape}, expected ({n}, {w})")
        elif not np.isfinite(a).all():
            err.append(f"{name} has NaN or inf")
    if t.shape != (n,):
        err.append(f"times shape {t.shape}, expected ({n},)")
    if n < 25:
        err.append(f"only {n} frames, need at least half a second")
    if abs(float(meta.get("dt", 0)) - DT) > 1e-9 or (n > 1 and np.abs(np.diff(t) - DT).max() > 1e-6):
        err.append(f"must be sampled at {1 / DT:g} Hz (meta dt and times spacing {DT})")
    if meta.get("n_frames") not in (None, n):
        err.append(f"meta n_frames {meta['n_frames']} != {n} frames")
    if err:
        return err, warn

    norm = np.linalg.norm(q[:, 3:7], axis=1)
    if np.abs(norm - 1).max() > 1e-6:
        err.append(f"root quaternion not unit length (max error {np.abs(norm - 1).max():.2e})")
    m = build_model()
    names = joint_names(m)
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    over = np.maximum(lo - q[:, FREE_QPOS:], q[:, FREE_QPOS:] - hi).max(0)
    for j in np.flatnonzero(over > 0.02):
        warn.append(f"{names[j]} outside its limits by up to {over[j]:.3f} rad")
    if np.abs(u - q[:, FREE_QPOS:]).max() > 1e-3:
        warn.append("ctrl differs from the joint angles in qpos (it should be the same targets)")

    d = mujoco.MjData(m)
    dv = np.zeros(m.nv)
    vel_err, low = [], []
    for i in range(n):
        if i + 1 < n:
            mujoco.mj_differentiatePos(m, dv, DT, q[i], q[i + 1])
            vel_err.append(np.abs(dv - v[i]))
        d.qpos[:] = q[i]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        floor = [c.dist for c in d.contact[: d.ncon] if 0 in (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])]
        low.append(min(floor, default=np.inf))
    vel_err = float(np.percentile(vel_err, 95))
    if vel_err > 1.0:
        warn.append(f"qvel disagrees with finite differences of qpos (95th percentile {vel_err:.2f})")
    low = np.array(low)
    if low.min() < -0.03:
        warn.append(f"robot sinks {-low.min() * 100:.1f} cm into the floor (frame {int(low.argmin())})")
    if not np.isfinite(low[0]) and q[0, 2] > 0.9:
        warn.append(f"first frame floats above the floor (root height {q[0, 2]:.2f} m)")
    jacc = float(np.sqrt(((np.diff(q[:, FREE_QPOS:], 2, axis=0) / DT**2) ** 2).mean()))
    if jacc > 150:
        warn.append(f"joint acceleration RMS {jacc:.0f} rad/s^2 is high; smooth the clip")
    return err, warn


def slerp_rows(qa: np.ndarray, qb: np.ndarray, f: np.ndarray) -> np.ndarray:
    qb = np.where((np.sum(qa * qb, 1) < 0)[:, None], -qb, qb)
    out = qa * (1 - f) + qb * f
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def import_csv(csv: str | Path, fps: float, name: str, quat: str = "xyzw", out: str | Path = "") -> Path:
    raw = np.loadtxt(csv, delimiter=",", ndmin=2)
    if raw.shape[1] != ROBOT_QPOS:
        raise SystemExit(f"{csv}: {raw.shape[1]} columns, expected {ROBOT_QPOS} (xyz, quaternion, 29 joints)")
    if quat == "xyzw":
        raw[:, 3:7] = raw[:, [6, 3, 4, 5]]
    raw[:, 3:7] /= np.linalg.norm(raw[:, 3:7], axis=1, keepdims=True)
    src = np.arange(0, (len(raw) - 1) / fps + 1e-9, DT) * fps
    i0 = np.minimum(np.floor(src).astype(int), len(raw) - 1)
    i1 = np.minimum(i0 + 1, len(raw) - 1)
    f = (src - i0)[:, None]
    q = raw[i0] * (1 - f) + raw[i1] * f
    q[:, 3:7] = slerp_rows(raw[i0, 3:7], raw[i1, 3:7], f)
    m = build_model()
    v = np.zeros((len(q), m.nv))
    for i in range(len(q) - 1):
        mujoco.mj_differentiatePos(m, v[i], DT, q[i], q[i + 1])
    v[-1] = v[-2]
    meta = dict(name=name, source=f"csv: {Path(csv).name} at {fps:g} fps", fps=1 / DT, dt=DT, n_frames=len(q))
    out = Path(out or ROOT / "motions" / f"{name}.npz")
    t = np.arange(len(q)) * DT
    np.savez_compressed(out, qpos=q, qvel=v, ctrl=q[:, FREE_QPOS:], times=t, meta_json=json.dumps(meta))
    from .pose.annotate import annotate

    annotate(out)
    return out


def report(path) -> int:
    err, warn = check(path)
    for e in err:
        print(f"ERROR {e}")
    for w in warn:
        print(f"WARN  {w}")
    print(f"{path}: {'FAIL' if err else 'ok'}" + (f", {len(warn)} warning(s)" if warn and not err else ""))
    return 1 if err else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="report problems with motion files")
    c.add_argument("motions", nargs="+")
    i = sub.add_parser("import", help="CSV of G1 poses -> motions/NAME.npz")
    i.add_argument("csv")
    i.add_argument("--fps", type=float, required=True, help="frame rate of the CSV")
    i.add_argument("--name", required=True)
    i.add_argument("--quat", choices=("xyzw", "wxyz"), default="xyzw")
    i.add_argument("--out", default="", help="default motions/NAME.npz")
    sub.add_parser("joints", help="print the G1 joint order used in motion files")
    a = ap.parse_args()
    if a.cmd == "check":
        sys.exit(max(report(p) for p in a.motions))
    if a.cmd == "joints":
        print("\n".join(joint_names(build_model())))
        return
    sys.exit(report(import_csv(a.csv, a.fps, a.name, a.quat, a.out)))


if __name__ == "__main__":
    main()
