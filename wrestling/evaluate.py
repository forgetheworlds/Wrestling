"""Evaluate a checkpoint: deterministic policy from frame 0 of every clip, over several seeds.

Seed 0 starts exactly on the reference; other seeds add small joint and root noise to the
start state. A rollout succeeds when it reaches the end of the clip without a fall.

    python -m wrestling.evaluate configs/skills/double_leg.json checkpoints/double_leg.npz
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .env import ImitationEnv
from .policy import act, load_checkpoint
from .skills import load_skill

EVAL_NOISE = (0.02, 0.005)
EVAL_OVERRIDES = {"p_rsi": 0.0, "push_prob": 0.0}


def rollout(env: ImitationEnv, params: dict, clip: int, seed: int, on_step=None) -> dict:
    env.rng = np.random.default_rng(seed)
    env.select_clip(clip, 0.0, 1.0)
    env.cfg.rsi_joint_noise, env.cfg.rsi_root_noise = EVAL_NOISE if seed > 0 else (0.0, 0.0)
    obs = env.start(0, noise=seed > 0)
    done, mse, qs, info = False, [], [], {}
    while not done:
        obs, _, done, info = env.step(act(params, obs))
        mse.append(info["joint_mse"])
        qs.append(env.data.qpos[7:].copy())
        if on_step is not None:
            on_step(env)
    q = np.array(qs)
    jitter = float(np.sqrt(((np.diff(q, 2, axis=0) / 0.02**2) ** 2).mean())) if len(q) > 3 else 0.0
    return {
        "clip": env.clip_specs[clip]["id"],
        "seed": seed,
        "success": info["term"] == "clip_end",
        "term": info["term"],
        "progress": info["frame"] / (env.clip.n - 1),
        "joint_mse": float(np.mean(mse)),
        "jitter": jitter,
    }


def evaluate(skill: dict, params: dict, seeds: int = 16) -> list[dict]:
    env = ImitationEnv(skill, EVAL_OVERRIDES, seed=0)
    return [rollout(env, params, k, s) for k in range(len(skill["clips"])) for s in range(seeds)]


def summarize(rows: list[dict]) -> dict:
    out = {}
    for cid in dict.fromkeys(r["clip"] for r in rows):
        rs = [r for r in rows if r["clip"] == cid]
        out[cid] = {
            "success": f"{sum(r['success'] for r in rs)}/{len(rs)}",
            "progress": round(float(np.mean([r["progress"] for r in rs])), 3),
            "joint_mse": round(float(np.mean([r["joint_mse"] for r in rs])), 4),
            "jitter": round(float(np.mean([r["jitter"] for r in rs])), 2),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="skill config JSON")
    ap.add_argument("checkpoint")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--json", default="", help="also write per-rollout results here")
    a = ap.parse_args()
    params, _ = load_checkpoint(a.checkpoint)
    rows = evaluate(load_skill(a.config), params, a.seeds)
    for r in rows:
        print(
            f"{r['clip']:>10} seed {r['seed']:2d}  {r['term']:9s} progress {r['progress']:.3f}  joint mse {r['joint_mse']:.4f}"
        )
    print(json.dumps(summarize(rows), indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
