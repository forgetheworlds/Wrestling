#!/usr/bin/env bash
# Video -> G1 reference motion, contact annotations and a side-by-side check video.
# Usage: scripts/video_to_motion.sh VIDEO NAME
# Writes motions/NAME.npz and work/NAME/{kp2d.npz, gate.json, NAME_vs_video.mp4}.
# Exits 2 if no retarget variant passed the accuracy gate (the best attempt is still written).
set -euo pipefail
cd "$(dirname "$0")/.."

VIDEO="$1"
NAME="$2"
WORK="work/$NAME"
MOTION="motions/$NAME.npz"

rc=0
.venv/bin/python -m wrestling.pose.retarget "$VIDEO" --name "$NAME" --out "$MOTION" --workdir "$WORK" \
  --mp-python .venv-pose/bin/python --model third_party/pose_landmarker_heavy.task || rc=$?
[[ $rc -eq 0 || $rc -eq 2 ]] || exit $rc
.venv/bin/python -m wrestling.pose.annotate "$MOTION"
.venv-pose/bin/python -m wrestling.pose.compare "$VIDEO" "$MOTION" "$WORK/kp2d.npz" "$WORK/${NAME}_vs_video.mp4"
exit $rc
