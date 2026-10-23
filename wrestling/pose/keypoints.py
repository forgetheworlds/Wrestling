"""MediaPipe PoseLandmarker (heavy): 2D image keypoints and metric world landmarks per frame.

Two passes. Pass 1 runs on the downscaled frame to find the person; pass 2 re-runs on a smoothed,
padded, person-centred crop of the full-resolution frame, so a small or prone subject gets more pixels.
Coordinates are returned in the downscaled frame (`size`).

Runs in its own environment with mediapipe and opencv (scripts/setup_pose.sh).
Output npz: kp2d (T,33,3) pixel x, y and visibility; world (T,33,3) metric, hip-centred; t; fps; size.
"""

from __future__ import annotations
import argparse, os, numpy as np, cv2
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
from scipy.ndimage import gaussian_filter1d


def _landmarker(model):
    return vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=model),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    )


def _frames(video, fps):
    cap = cv2.VideoCapture(video)
    src = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    nxt = 0.0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        t = i / src
        i += 1
        if t + 1e-6 < nxt:
            continue
        nxt += 1.0 / fps
        yield t, fr


def _detect(lm, img, ts_ms):
    res = lm.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), ts_ms
    )
    if not res.pose_landmarks:
        return None, None
    return (
        np.array([[a.x, a.y, a.visibility] for a in res.pose_landmarks[0]]),
        np.array([[a.x, a.y, a.z] for a in res.pose_world_landmarks[0]]),
    )


def run(video, out, model, fps=30.0, max_w=1280, crop_side=1024, pad=0.25):
    lm = _landmarker(model)
    kp, wd, ts = [], [], []
    for t, fr in _frames(video, fps):
        h, w = fr.shape[:2]
        s = min(1.0, max_w / w)
        W, H = int(w * s), int(h * s)
        p, q = _detect(lm, cv2.resize(fr, (W, H)) if s < 1 else fr, int(t * 1000))
        kp.append(p * [W, H, 1] if p is not None else np.full((33, 3), np.nan))
        wd.append(q if q is not None else np.full((33, 3), np.nan))
        ts.append(t)
    kp1 = np.array(kp)
    wd1 = np.array(wd)
    S = w / W
    lo = np.nanmin(kp1[..., :2], 1) * S
    hi = np.nanmax(kp1[..., :2], 1) * S
    for a in (lo, hi):
        for k in range(2):
            c = a[:, k]
            bad = ~np.isfinite(c)
            if bad.all():
                c[:] = 0
            elif bad.any():
                c[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), c[~bad])
    cen = gaussian_filter1d((lo + hi) / 2, 3, axis=0)
    side = gaussian_filter1d(np.max(hi - lo, 1), 3) * (1 + 2 * pad)
    side = np.maximum(side, 0.2 * min(w, h))
    lm2 = _landmarker(model)
    kp2 = kp1.copy()
    wd2 = wd1.copy()
    boxes = []
    for j, (t, fr) in enumerate(_frames(video, fps)):
        if j >= len(kp1):
            break
        x0, y0 = cen[j] - side[j] / 2
        sd = side[j]
        M = np.array([[crop_side / sd, 0, -x0 * crop_side / sd], [0, crop_side / sd, -y0 * crop_side / sd]])
        crop = cv2.warpAffine(
            fr, M, (crop_side, crop_side), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT
        )
        p, q = _detect(lm2, crop, int(t * 1000))
        boxes.append([x0, y0, sd])
        if p is None:
            continue
        full = np.c_[p[:, 0] * sd + x0, p[:, 1] * sd + y0]
        kp2[j] = np.c_[full / S, p[:, 2]]
        wd2[j] = q
    np.savez(
        out,
        kp2d=kp2,
        world=wd2,
        kp2d_pass1=kp1,
        world_pass1=wd1,
        t=np.array(ts),
        fps=fps,
        size=np.array([W, H]),
        crop_boxes_fullres=np.array(boxes),
    )
    return out


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("video")
    a.add_argument("out")
    a.add_argument("--model", default=os.environ.get("MP_MODEL", "third_party/pose_landmarker_heavy.task"))
    a.add_argument("--fps", type=float, default=30.0)
    r = a.parse_args()
    run(r.video, r.out, r.model, r.fps)
    print("wrote", r.out)
