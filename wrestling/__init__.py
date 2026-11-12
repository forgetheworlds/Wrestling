"""Video-to-humanoid wrestling motion imitation on the Unitree G1."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHYSICS = json.loads((ROOT / "configs/mujoco.json").read_text())

# Headless Linux has no window system for MuJoCo's default GLFW backend.
if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    os.environ.setdefault("MUJOCO_GL", PHYSICS["headless_gl"])
