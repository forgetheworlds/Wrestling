#!/usr/bin/env bash
# Creates .venv with pinned dependencies and fetches the Unitree G1 model from MuJoCo Menagerie.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
MENAGERIE_COMMIT=f054586a8e90465d49ee5be15335c4a0c7f57caf

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

echo "ready: source .venv/bin/activate"
