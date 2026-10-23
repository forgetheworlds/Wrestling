"""Reward specs: JSON files that compile into environment settings.

A spec lists costs (behaviour that should never happen), terminations and rewards, in that
order of priority. Specs can extend another spec with "base"; dictionaries are deep-merged.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from . import ROOT

TERMS = (
    "pose",
    "keypoints",
    "root",
    "contact",
    "hand_keypoints",
    "hold_smooth",
    "cmd_vel",
    "cmd_yaw",
    "knee_flex",
    "support_margin",
)
COSTS = (
    "body_contact",
    "self_collision",
    "joint_limit",
    "torque_spike",
    "action_jerk",
    "contact_slip",
    "hold_jitter",
)
BODY_GROUPS = {
    "feet": ["left_ankle_roll_link", "right_ankle_roll_link"],
    "hands": [
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_wrist_roll_link",
        "right_wrist_roll_link",
        "left_wrist_pitch_link",
        "right_wrist_pitch_link",
        "left_rubber_hand",
        "right_rubber_hand",
    ],
    "forearms": ["left_elbow_link", "right_elbow_link"],
    "knees": ["left_knee_link", "right_knee_link"],
    "shins": ["left_ankle_pitch_link", "right_ankle_pitch_link"],
    "hips": [
        "pelvis",
        "left_hip_pitch_link",
        "right_hip_pitch_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
    ],
    "torso": ["torso_link", "waist_yaw_link", "waist_roll_link"],
    "head": ["head_link"],
}


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = (
            deep_merge(out[k], v)
            if isinstance(v, dict) and isinstance(out.get(k), dict)
            else copy.deepcopy(v)
        )
    return out


def load_spec(name_or_path: str | dict) -> dict:
    if isinstance(name_or_path, dict):
        spec = name_or_path
    else:
        p = Path(name_or_path)
        if not p.suffix:
            p = ROOT / "configs/rewards" / f"{name_or_path}.json"
        elif not p.is_absolute():
            p = ROOT / p
        spec = json.loads(p.read_text())
    if spec.get("base"):
        return deep_merge(load_spec(spec["base"]), {k: v for k, v in spec.items() if k != "base"})
    return spec


def compile_spec(spec: dict) -> tuple[dict, dict]:
    """Returns (environment config overrides, Lagrangian settings)."""
    cfg = {}

    # Costs are hinge-normalised, c = clip((x - threshold) / scale, 0, 1), with Lagrangian budgets.
    costs = spec.get("costs", {})
    off, hinge, budgets = [], {}, []
    for n in COSTS:
        c = costs.get(n, {})
        if c.get("enabled", True) is False:
            off.append(n)
        if "threshold" in c or "scale" in c:
            hinge[n] = [c.get("threshold", 0.0), c.get("scale", 1.0)]
        budgets.append(float(c.get("budget", spec.get("cost_budget_default", 0.05))))
    if spec.get("sliding_allowed") and "contact_slip" not in off:
        off.append("contact_slip")
    cfg["costs_off"] = off
    if hinge:
        cfg["cost_spec"] = hinge
    lag = spec.get("lagrange", {})
    if "penalty_cap" in lag:
        cfg["penalty_cap"] = lag["penalty_cap"]
    lagrange = {
        "budgets": budgets,
        "lambda_lr": lag.get("lambda_lr", 0.01),
        "eta_max": lag.get("eta_max", 0.25),
    }

    # Terminations and which body groups may touch the mat.
    t = spec.get("terminations", {})
    for k in (
        "fall_height",
        "fly_height",
        "et_fall_hold",
        "et_root_dev",
        "et_torso_align",
        "et_torso_hdev",
        "et_upright_rel_deg",
        "et_fall_height_rel",
        "root_drop",
    ):
        if k in t:
            cfg[k] = t[k]
    gc = spec.get("ground_contacts")
    if gc:
        groups = deep_merge(BODY_GROUPS, gc.get("bodies", {}))
        allowed = gc.get("allowed", ["feet"])
        forbidden = gc.get("forbidden") or [g for g in groups if g not in allowed]
        cfg["ground_contacts"] = {
            "allowed": allowed,
            "forbidden": [g for g in forbidden if g not in allowed],
            "bodies": groups,
        }

    # Rewards: every term is in [0, 1] and the weights of each stage kind are normalised to sum to 1.
    rw = spec.get("rewards", {})
    kern = {k: rw[k]["k"] for k in ("pose", "keypoints", "root") if "k" in rw.get(k, {})}
    if kern:
        cfg["kernel"] = kern
    if "k" in rw.get("cmd_vel", {}):
        cfg["cmd_k"] = rw["cmd_vel"]["k"]
    if "k" in rw.get("cmd_yaw", {}):
        cfg["cmd_yaw_k"] = rw["cmd_yaw"]["k"]
    kp = rw.get("keypoints", {})
    if "foot_weight" in kp:
        cfg["foot_kp_w"] = kp["foot_weight"]
    if "foot_tol_m" in kp:
        cfg["foot_kp_tol"] = kp["foot_tol_m"]
    kf = rw.get("knee_flex", {})
    if kf.get("weight", 0) > 0:
        cfg["knee_flex_deg"] = kf.get("min_deg", 40.0)
        cfg["knee_flex_sigma_deg"] = kf.get("sigma_deg", 10.0)
        if kf.get("stages"):
            cfg["knee_flex_stages"] = kf["stages"]
    sm = rw.get("support_margin", {})
    if "active_below_m" in sm:
        cfg["support_act_m"] = sm["active_below_m"]
    if "scale_m" in sm:
        cfg["support_scale_m"] = sm["scale_m"]
    if any("weight" in rw.get(n, {}) for n in TERMS) or spec.get("stage_weights"):
        default = [rw.get(n, {}).get("weight", 0.0) for n in TERMS]
        sw = {"*": default}
        for kind, wd in (spec.get("stage_weights") or {}).items():
            sw[kind] = [wd.get(n, default[i]) if isinstance(wd, dict) else wd[i] for i, n in enumerate(TERMS)]
        cfg["stage_weights"] = sw
    ub = rw.get("upright_bonus", {})
    if "weight" in ub:
        cfg["w_upright_bonus"] = ub["weight"]
    if "k" in ub:
        cfg["upright_bonus_k"] = ub["k"]

    # Actuation: PD gains plus reference torque feedforward.
    ac = spec.get("actuation", {})
    if "pd_gains" in ac:
        cfg["pd_gains"] = ac["pd_gains"]
    if "ff_torque" in ac:
        cfg["ff_torque"] = bool(ac["ff_torque"])
    if "ff_vel" in ac:
        cfg["ff_vel"] = bool(ac["ff_vel"])

    # Random horizontal pushes on the pelvis during training.
    push = spec.get("robustness", {}).get("push", {})
    if "prob" in push:
        cfg["push_prob"] = push["prob"]
    if "force_n" in push:
        cfg["push_force"] = push["force_n"]
    if "steps" in push:
        cfg["push_steps"] = push["steps"]

    # Raw environment fields.
    cfg.update({k: v for k, v in spec.get("env", {}).items() if not k.startswith("_")})
    return cfg, lagrange
