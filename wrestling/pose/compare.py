"""Side-by-side check video: the source video with the MediaPipe 2D skeleton (green) and the G1's
reprojected keypoints (red), next to the G1 rendered from the gate's fitted camera direction.

Needs opencv, so run it in the pose environment:
    python -m wrestling.pose.compare VIDEO MOTION.npz KP2D.npz OUT.mp4
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot

from ..render import matte_floor
from . import retarget

EDGES = [
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 12),
    (23, 24),
    (11, 23),
    (12, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (29, 31),
    (28, 30),
    (30, 32),
    (15, 19),
    (16, 20),
]
GREEN = (0, 220, 0)
RED = (40, 40, 255)


def render(video, motion, kp2d, out_mp4, height=540, cam_distance=1.9):
    gate, uv, kp = retarget.gate(motion, kp2d)
    z = np.load(motion)
    Q, tq = z["qpos"], z["times"]
    k = np.load(kp2d)
    tk = k["t"][: len(uv)]
    W0, H0 = (int(v) for v in k["size"])
    robot = retarget.G1()
    m = robot.m
    matte_floor(m)
    width = int(height * W0 / H0)
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, height)
    ren = mujoco.Renderer(m, height, width)

    cam = mujoco.MjvCamera()
    view = Rot.from_euler("zyx", gate["camera"][:3]).as_matrix().T @ np.array([1.0, 0, 0])
    cam.azimuth = float(np.degrees(np.arctan2(-view[1], -view[0])))
    cam.elevation = float(min(-np.degrees(np.arcsin(view[2])), -8.0))
    cam.distance = cam_distance
    track = gaussian_filter1d(Q[:, :3], 8.0, axis=0, mode="nearest")

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    tmp = str(out_mp4) + ".avi"
    writer = None
    i = fi = 0
    while i < len(tk):
        ok, fr = cap.read()
        if not ok:
            break
        tf = fi / fps
        fi += 1
        if tf + 1e-6 < tk[i]:
            continue
        fr = cv2.resize(fr, (W0, H0))
        for a, b in EDGES:
            for P, col in ((kp[i], GREEN), (uv[i], RED)):
                pa, pb = P[a, :2], P[b, :2]
                if np.isfinite(pa).all() and np.isfinite(pb).all():
                    cv2.line(fr, tuple(int(x) for x in pa), tuple(int(x) for x in pb), col, 3)
        fr = cv2.resize(fr, (width, height))
        j = int(np.argmin(np.abs(tq - (tk[i] - tk[0]))))
        cam.lookat[:] = [track[j, 0], track[j, 1], 0.15 + 0.6 * track[j, 2]]
        robot.d.qpos[:] = Q[j]
        mujoco.mj_forward(m, robot.d)
        ren.update_scene(robot.d, cam)
        both = np.concatenate([fr, cv2.cvtColor(ren.render(), cv2.COLOR_RGB2BGR)], 1)
        cv2.putText(
            both, "green: video keypoints   red: G1 reprojected", (10, 28), 0, 0.5, (255, 255, 255), 1
        )
        if writer is None:
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"MJPG"), 30, (both.shape[1], both.shape[0]))
        writer.write(both)
        i += 1
    writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            tmp,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "22",
            str(out_mp4),
        ],
        check=True,
    )
    Path(tmp).unlink()
    return gate


def main():
    ap = argparse.ArgumentParser(description="Video vs retargeted G1, side by side.")
    ap.add_argument("video")
    ap.add_argument("motion")
    ap.add_argument("kp2d")
    ap.add_argument("out")
    a = ap.parse_args()
    g = render(a.video, a.motion, a.kp2d, a.out)
    print(f"reprojection {g['reproj_mean_pct']:.2f}% of stature, gate pass={g['pass']}")


if __name__ == "__main__":
    main()
