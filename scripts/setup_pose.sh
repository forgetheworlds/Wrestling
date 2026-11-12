#!/usr/bin/env bash
# Separate environment for the video pipeline (mediapipe needs Python 3.12 and numpy 1.x).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3.12}"
if command -v "$PYTHON" >/dev/null; then
  "$PYTHON" -m venv .venv-pose
  .venv-pose/bin/pip install -q --upgrade pip
  .venv-pose/bin/pip install -q -r requirements-pose.txt
elif command -v uv >/dev/null; then
  uv venv -q -p 3.12 .venv-pose
  VIRTUAL_ENV=.venv-pose uv pip install -q -r requirements-pose.txt
else
  echo "need python3.12 (or uv to fetch it)"; exit 1
fi

mkdir -p third_party
MODEL=third_party/pose_landmarker_heavy.task
[[ -f "$MODEL" ]] || curl -sSLo "$MODEL" \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

echo "ready: .venv-pose"
