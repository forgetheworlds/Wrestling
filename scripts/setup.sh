#!/usr/bin/env bash
# Creates .venv with pinned dependencies and fetches the Unitree G1 model from MuJoCo Menagerie.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 12))' \
  || { echo "need Python 3.12 or newer (set PYTHON=python3.12)"; exit 1; }
MENAGERIE_COMMIT=$("$PYTHON" -c 'import json; print(json.load(open("configs/mujoco.json"))["menagerie_commit"])')

"$PYTHON" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

if [[ ! -f third_party/unitree_g1/g1.xml ]]; then
  tmp="$(mktemp -d)"
  git -C "$tmp" init -q
  git -C "$tmp" remote add origin https://github.com/google-deepmind/mujoco_menagerie.git
  git -C "$tmp" sparse-checkout set unitree_g1
  git -C "$tmp" fetch -q --depth 1 --filter=blob:none origin "$MENAGERIE_COMMIT"
  git -C "$tmp" checkout -q FETCH_HEAD
  mkdir -p third_party
  rm -rf third_party/unitree_g1
  mv "$tmp/unitree_g1" third_party/unitree_g1
  rm -rf "$tmp"
fi

.venv/bin/python -c 'import mujoco; from wrestling.robot import build_model; build_model(); print("mujoco", mujoco.__version__, "+ G1 ok")'
echo "ready: source .venv/bin/activate"
