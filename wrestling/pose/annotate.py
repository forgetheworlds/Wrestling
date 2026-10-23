"""Contact schedule and coarse stages for a reference motion, written into its metadata.

Feet are in contact when the sole site is within 3 cm of its own lowest height in the clip, hands
when the wrist is below 12 cm. Stages are runs of (height level, hands down, moving) merged until
each lasts at least `min_s` seconds. The environment uses the hand schedule for the hand-post term
and the stages for per-stage reward weights.

    python -m wrestling.pose.annotate motions/sprawl.npz
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..robot import build_model


def stages_and_contacts(qpos, times, min_s=0.3, max_stages=6):
    m = build_model()
    d = mujoco.MjData(m)
    dt = float(np.median(np.diff(times)))
    feet = [m.site(n).id for n in ("left_foot", "right_foot")]
    hands = [m.body(n).id for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
    F = np.zeros((len(qpos), 2))
    HS = np.zeros((len(qpos), 2))
    for i, q in enumerate(qpos):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        F[i] = [d.site_xpos[s][2] for s in feet]
        HS[i] = [d.xpos[b][2] for b in hands]
    pz = qpos[:, 2]
    speed = np.r_[0, np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1) / dt]
    speed = gaussian_filter1d(speed, 3)
    fz = F - F.min(0, keepdims=True)
    left_foot, right_foot = fz[:, 0] < 0.03, fz[:, 1] < 0.03
    hand_down = HS.min(1) < 0.12
    h0 = np.median(pz[: max(5, len(pz) // 10)])
    level = np.where(pz < 0.55 * h0, "ground", np.where(pz < 0.85 * h0, "low", "stand"))
    moving = np.where(speed > 0.25, "move", "still")
    key = np.array([f"{a}{'_hands' if h else ''}_{c}" for a, h, c in zip(level, hand_down, moving)])

    cuts = [0] + [i for i in range(1, len(key)) if key[i] != key[i - 1]] + [len(key)]
    segs = [[cuts[k], cuts[k + 1]] for k in range(len(cuts) - 1)]
    min_len = max(1, int(round(min_s / dt)))
    while len(segs) > 1 and (min(b - a for a, b in segs) < min_len or len(segs) > max_stages):
        k = int(np.argmin([b - a for a, b in segs]))
        if k > 0 and (
            k == len(segs) - 1 or segs[k - 1][1] - segs[k - 1][0] <= segs[k + 1][1] - segs[k + 1][0]
        ):
            j = k - 1
        else:
            j = k + 1
        merged = [min(segs[k][0], segs[j][0]), max(segs[k][1], segs[j][1])]
        segs = sorted([s for i, s in enumerate(segs) if i not in (j, k)] + [merged])

    stages = []
    for i, (a, b) in enumerate(segs):
        names, counts = np.unique(key[a:b], return_counts=True)
        stages.append(
            dict(
                name=f"s{i}_{names[np.argmax(counts)]}",
                lo=round(a / len(qpos), 3),
                hi=round(b / len(qpos), 3),
                hands_down_frac=round(float(hand_down[a:b].mean()), 2),
            )
        )
    schedule = dict(
        left_foot=left_foot.astype(int).tolist(),
        right_foot=right_foot.astype(int).tolist(),
        left_hand=(HS[:, 0] < 0.12).astype(int).tolist(),
        right_hand=(HS[:, 1] < 0.12).astype(int).tolist(),
    )
    return stages, schedule


def annotate(path):
    z = np.load(path)
    data = {k: z[k] for k in z.files if k != "meta_json"}
    meta = json.loads(str(z["meta_json"]))
    stages, schedule = stages_and_contacts(data["qpos"], data["times"])
    meta["contact_schedule"] = schedule
    meta["hand_contact"] = dict(left=schedule["left_hand"], right=schedule["right_hand"])
    meta["cmo_stages"] = stages
    np.savez_compressed(path, **data, meta_json=json.dumps(meta))
    return stages


def main():
    ap = argparse.ArgumentParser(description="Add contact schedule and stages to a motion file.")
    ap.add_argument("motion")
    for s in annotate(ap.parse_args().motion):
        print(s)


if __name__ == "__main__":
    main()
