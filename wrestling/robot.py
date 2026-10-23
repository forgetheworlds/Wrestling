"""Unitree G1 scene and actuator gains."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import mujoco
import numpy as np

from . import ROOT

G1_DIR = Path(os.environ.get("WRESTLING_G1_DIR", ROOT / "third_party/unitree_g1"))
G1_XML = G1_DIR / "g1.xml"

FREE_QPOS = 7
FREE_QVEL = 6
N_JOINT = 29
ROBOT_QPOS = FREE_QPOS + N_JOINT
ROBOT_QVEL = FREE_QVEL + N_JOINT


@contextmanager
def _in_dir(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def build_model(timestep: float = 0.002, hand_collision: bool = False) -> mujoco.MjModel:
    """G1 on a flat mat.

    Menagerie only gives the wrist links collision geoms, so the rubber hands float above the
    mat in poses where the hands are posted. `hand_collision` makes the hand meshes collide.
    """
    if not G1_XML.exists():
        raise FileNotFoundError(f"{G1_XML} not found. Run scripts/setup.sh to fetch the Unitree G1 model.")
    with _in_dir(G1_DIR):
        spec = mujoco.MjSpec()
        spec.modelname = "g1_wrestling"
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.timestep = timestep
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size[:] = [0, 0, 0.05]
        floor.friction[:] = [1.2, 0.005, 0.0001]
        floor.rgba[:] = [0.2, 0.25, 0.3, 1.0]
        robot = mujoco.MjSpec.from_file(str(G1_XML))
        robot.meshdir = str(G1_DIR / "assets")
        if hand_collision:
            for g in robot.geoms:
                if "rubber_hand" in (g.meshname or ""):
                    g.contype = 1
                    g.conaffinity = 1
                    g.condim = 3
        frame = spec.worldbody.add_frame()
        frame.name = "robot_frame"
        spec.attach(robot, prefix="", frame=frame)
        model = spec.compile()
    if model is None:
        raise RuntimeError("failed to compile the G1 scene")
    return model


def apply_pd_gains(model: mujoco.MjModel, path: str | Path) -> np.ndarray | None:
    """Set per-joint kp/kd from a gains JSON. Returns the per-actuator action scale, if given."""
    if not path:
        return None
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    joints = json.loads(p.read_text())["joints"]
    scale = np.full(model.nu, np.nan)
    for i in range(model.nu):
        name = model.actuator(i).name
        g = next((v for k, v in joints.items() if k in name), None)
        if g is None:
            raise KeyError(f"{p.name}: no gains for actuator {name}")
        model.actuator_gainprm[i, 0] = g["kp"]
        model.actuator_biasprm[i, 1] = -g["kp"]
        model.actuator_biasprm[i, 2] = -g["kd"]
        if "action_scale" in g:
            scale[i] = g["action_scale"]
    return None if np.isnan(scale).any() else scale


def default_joint_qpos(model: mujoco.MjModel) -> np.ndarray:
    return np.array(model.qpos0[FREE_QPOS:ROBOT_QPOS], dtype=np.float64)
