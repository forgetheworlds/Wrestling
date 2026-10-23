"""Skill configs: which clips to imitate, which reward spec to use, and how to train.

A skill config names a reward spec (configs/rewards) and may override any part of it, either
for the whole skill ("reward") or per clip. A config can extend another with "extends", which
is how fine-tuning stages are written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import ROOT
from .rewards import compile_spec, deep_merge, load_spec

TRAIN_DEFAULTS = {
    "minutes": 60.0,
    "workers": 8,
    "horizon": 256,
    "minibatch": 512,
    "epochs": 5,
    "lr": 3e-4,
    "lr_final_frac": 1.0,
    "target_kl": 0.0,
    "max_log_std": None,
    "init_log_std": -1.2,
    "gamma": 0.98,
    "gae_lambda": 0.95,
    "eval_every": 10,
    "n_eval": 16,
    "save_every": 150,
    "seed": 0,
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.suffix:
        p = ROOT / "configs/skills" / f"{p}.json"
    return p if p.is_absolute() else ROOT / p


def read_config(path: str | Path) -> dict:
    p = _resolve(path)
    cfg = json.loads(p.read_text())
    if cfg.get("extends"):
        base = read_config(p.parent / cfg["extends"])
        cfg = deep_merge(base, {k: v for k, v in cfg.items() if k != "extends"})
    return cfg


def load_skill(path: str | Path) -> dict:
    """Compile a skill config into per-clip environment settings and training settings."""
    cfg = read_config(path)
    spec = deep_merge(load_spec(cfg["reward_spec"]), cfg.get("reward", {}))
    clips, clip_cfg, lagrange = [], [], None
    scale_min = min(cfg.get("augment", {}).get("time_scale", [1, 1]))
    for c in cfg["clips"]:
        env_cfg, lag = compile_spec(deep_merge(spec, c.get("reward", {})))
        lagrange = lagrange or lag
        motion = ROOT / c["motion"]
        if "episode_seconds" not in env_cfg:
            n = np.load(motion)["qpos"].shape[0]
            env_cfg["episode_seconds"] = round(n * 0.02 / scale_min + 1.0, 2)
        clips.append(dict(c, path=str(motion)))
        clip_cfg.append(env_cfg)
    return {
        "name": cfg["name"],
        "skills": cfg.get("skills") or sorted({c.get("skill", c["id"]) for c in cfg["clips"]}),
        "clips": clips,
        "augment": cfg.get("augment", {}),
        "clip_cfg": clip_cfg,
        "lagrange": lagrange,
        "train": {**TRAIN_DEFAULTS, **cfg.get("train", {})},
    }
