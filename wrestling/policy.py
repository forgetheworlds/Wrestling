"""Checkpoint I/O and the NumPy forward pass used for rollouts, evaluation and rendering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PARAM_KEYS = ("w1", "b1", "w2", "b2", "w_mu", "b_mu", "w_v", "b_v", "log_std")


def load_checkpoint(path: str | Path) -> tuple[dict, dict]:
    z = np.load(path)
    params = {k: np.asarray(z[k]) for k in PARAM_KEYS}
    meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    return params, meta


def save_checkpoint(path: str | Path, params: dict, meta: dict) -> None:
    np.savez(path, **{k: np.asarray(params[k]) for k in PARAM_KEYS}, meta_json=json.dumps(meta))


def forward(p: dict, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Action mean, log std and value for one observation."""
    x = np.tanh(obs @ p["w1"] + p["b1"])
    x = np.tanh(x @ p["w2"] + p["b2"])
    v = float((x @ p["w_v"] + p["b_v"]).reshape(-1)[0])
    return np.tanh(x @ p["w_mu"] + p["b_mu"]), p["log_std"], v


def act(p: dict, obs: np.ndarray) -> np.ndarray:
    return forward(p, obs)[0]
