"""Export the legacy Wrestling G1 motion clips for SuperMimic/mjlab_research.

Usage:
    python -m wrestling.supermimic_export motions/*.npz --out-dir work/supermimic

SuperMimic's MotionLoader reads joint names and world-frame root twists. The
legacy clips have MuJoCo-ordered joints and a body-frame root angular velocity.
This module converts both explicitly, instead of reusing the older control
targets or silently assuming that the two robot assets have the same ordering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Same 29-DoF G1 ordering as mjlab_research/scripts/csv_to_npz.py.
MJLAB_G1_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def legacy_joint_names() -> list[str]:
    """Read actual MuJoCo joint order; it need not match mjlab's order."""
    from .clip import joint_names
    from .robot import build_model

    return joint_names(build_model())


def export_motion(
    source: str | Path,
    destination: str | Path,
    *,
    source_joint_names: list[str] | tuple[str, ...] | None = None,
) -> Path:
    """Convert a validated 50 Hz legacy clip into SuperMimic's six-field format.

    A source_joint_names override enables tests without installing G1 assets.
    It must describe the exact order of the source file's 29 joint columns.
    """
    source = Path(source)
    destination = Path(destination)
    with np.load(source, allow_pickle=False) as z:
        required = {"qpos", "qvel", "times", "meta_json"}
        missing = required - set(z.files)
        if missing:
            raise ValueError(f"{source}: missing {sorted(missing)}")
        qpos = np.asarray(z["qpos"], dtype=np.float64)
        qvel = np.asarray(z["qvel"], dtype=np.float64)
        times = np.asarray(z["times"], dtype=np.float64)
        meta = json.loads(str(z["meta_json"]))

    if qpos.ndim != 2 or qpos.shape[1] != 36 or qpos.shape[0] < 2:
        raise ValueError(f"{source}: qpos must have shape (T >= 2, 36), got {qpos.shape}")
    n = len(qpos)
    if qvel.shape != (n, 35) or times.shape != (n,):
        raise ValueError(f"{source}: expected qvel ({n}, 35) and times ({n},)")
    if not (np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(times).all()):
        raise ValueError(f"{source}: non-finite motion values")

    dt = float(meta.get("dt", 0))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"{source}: invalid motion dt {dt}")
    if not np.allclose(np.diff(times), dt, rtol=0, atol=1e-5):
        raise ValueError(f"{source}: inconsistent frame timestamps")
    if abs(dt - 0.02) > 1e-5:
        raise ValueError(f"{source}: expected 50 Hz references; got {1.0 / dt:.3f} Hz")

    quat_wxyz = qpos[:, 3:7]
    quat_norm_error = np.abs(np.linalg.norm(quat_wxyz, axis=1) - 1.0).max()
    if quat_norm_error > 1e-3:
        raise ValueError(f"{source}: non-unit root quaternions (max error {quat_norm_error:.4f})")

    names = list(source_joint_names) if source_joint_names is not None else legacy_joint_names()
    if len(names) != 29 or len(set(names)) != 29 or set(names) != set(MJLAB_G1_JOINTS):
        raise ValueError(
            f"{source}: legacy G1 joint names do not match mjlab's 29-joint model; "
            "review the model version and retargeting order"
        )
    order = np.array([names.index(name) for name in MJLAB_G1_JOINTS], dtype=int)

    # Legacy qvel[3:6] is a body-frame angular velocity. mjlab's
    # root_ang_vel_w is in world coordinates. MuJoCo uses wxyz for quats;
    # scipy Rotation accepts xyzw.
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    omega_world = Rotation.from_quat(quat_xyzw).apply(qvel[:, 3:6])

    output = {
        "fps": np.asarray([1.0 / dt], dtype=np.float32),
        "joint_names": np.asarray(MJLAB_G1_JOINTS),
        "joint_pos": qpos[:, 7:][:, order].astype(np.float32),
        "joint_vel": qvel[:, 6:][:, order].astype(np.float32),
        "root_pos_w": qpos[:, :3].astype(np.float32),
        "root_quat_w": quat_wxyz.astype(np.float32),
        "root_lin_vel_w": qvel[:, :3].astype(np.float32),
        "root_ang_vel_w": omega_world.astype(np.float32),
        "wrestling_source_meta_json": np.asarray(json.dumps(meta)),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **output)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motions", nargs="+", type=Path, help="legacy motions/*.npz files")
    parser.add_argument("--out-dir", type=Path, default=Path("work/supermimic"))
    args = parser.parse_args()

    # Resolve model once so that all clips use the same measured joint order.
    source_names = legacy_joint_names()
    for motion in args.motions:
        out = export_motion(
            motion, args.out_dir / motion.name, source_joint_names=source_names
        )
        print(f"{motion} -> {out}")


if __name__ == "__main__":
    main()
