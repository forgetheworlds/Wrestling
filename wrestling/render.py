"""Render the policy (physics) next to the reference motion (kinematic playback).

    python -m wrestling.render configs/skills/double_leg_smooth.json checkpoints/double_leg.npz --out double_leg.mp4
    python -m wrestling.render configs/skills/double_leg.json --reference-only --out reference.mp4

The rollout is deterministic from frame 0 and stops when the episode ends (end of the clip or
a fall). A .gif output is written through ffmpeg with a generated palette.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d

from .env import ImitationEnv
from .evaluate import EVAL_OVERRIDES, rollout
from .policy import load_checkpoint
from .skills import load_skill

MAT_RGBA = [0.75, 0.80, 0.85, 1.0]


def matte_floor(model: mujoco.MjModel) -> None:
    """Visual only: remove reflections and mute the checker floor."""
    if model.nmat:
        model.mat_reflectance[:] = 0.0
        model.mat_specular[:] = np.minimum(model.mat_specular, 0.2)
        model.mat_shininess[:] = np.minimum(model.mat_shininess, 0.1)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if gid >= 0:
        mid = model.geom_matid[gid]
        if mid >= 0:
            model.mat_rgba[mid] = MAT_RGBA
        else:
            model.geom_rgba[gid] = MAT_RGBA


class SideBySide:
    def __init__(
        self,
        env: ImitationEnv,
        width: int,
        height: int,
        azimuth: float,
        panels: tuple[str, ...],
        distance: float = 2.2,
    ):
        self.env, self.panels = env, panels
        m = env.model
        matte_floor(m)
        self.renderer = mujoco.Renderer(m, height=height, width=width)
        self.ref = mujoco.MjData(m)
        self.track = gaussian_filter1d(env.clip.qpos[:, :2], 10.0, axis=0, mode="nearest")
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance, self.cam.elevation, self.cam.azimuth = distance, -10.0, azimuth
        self.font = ImageFont.load_default(size=max(14, height // 26))
        self.frames = []

    def __call__(self, env: ImitationEnv) -> None:
        i = min(env.frame_i, env.clip.n - 1)
        self.ref.qpos[:] = env.clip.qpos[i]
        mujoco.mj_forward(env.model, self.ref)
        self.cam.lookat[:] = [*self.track[i], 0.5]
        imgs = []
        for name in self.panels:
            self.renderer.update_scene(env.data if name == "policy" else self.ref, camera=self.cam)
            im = Image.fromarray(self.renderer.render())
            ImageDraw.Draw(im).text((12, 10), name, fill=(235, 235, 235), font=self.font)
            imgs.append(np.asarray(im))
        self.frames.append(np.hstack(imgs))


def write_video(frames: list[np.ndarray], out: Path, fps: float, gif_width: int = 640) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix != ".gif":
        with imageio.get_writer(out, fps=fps, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(f)
        return
    import imageio_ffmpeg

    with tempfile.TemporaryDirectory() as tmp:
        mp4 = Path(tmp) / "clip.mp4"
        write_video(frames, mp4, fps)
        vf = f"fps=25,scale={gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp4),
                "-vf",
                vf,
                str(out),
            ],
            check=True,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("checkpoint", nargs="?")
    ap.add_argument("--out", required=True, help=".mp4 or .gif")
    ap.add_argument("--clip", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference-only", action="store_true")
    ap.add_argument("--azimuth", type=float, default=90.0)
    ap.add_argument("--distance", type=float, default=2.2, help="camera distance in metres")
    ap.add_argument("--size", type=int, nargs=2, default=(640, 480), metavar=("W", "H"))
    ap.add_argument("--slowmo", type=float, default=1.0, help="playback speed factor, e.g. 0.5")
    ap.add_argument("--gif-width", type=int, default=640)
    a = ap.parse_args()

    skill = load_skill(a.config)
    env = ImitationEnv(skill, EVAL_OVERRIDES, seed=0)
    env.select_clip(a.clip, 0.0, 1.0)
    if a.reference_only:
        rec = SideBySide(env, *a.size, a.azimuth, ("reference",), a.distance)
        for i in range(env.clip.n):
            env.frame_i = i
            rec(env)
        print(f"reference: {env.clip.n} frames")
    else:
        if not a.checkpoint:
            ap.error("checkpoint required unless --reference-only")
        params, _ = load_checkpoint(a.checkpoint)
        rec = SideBySide(env, *a.size, a.azimuth, ("policy", "reference"), a.distance)
        env.rng = np.random.default_rng(a.seed)
        env.start(0, noise=False)
        rec(env)
        r = rollout(env, params, a.clip, a.seed, on_step=rec)
        print(
            f"{r['clip']}: {r['term']}, progress {r['progress']:.3f}, joint mse {r['joint_mse']:.4f}, {len(rec.frames)} frames"
        )
    fps = 1.0 / (env.cfg.timestep * env.cfg.frame_skip) * a.slowmo
    write_video(rec.frames, Path(a.out), fps, a.gif_width)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
