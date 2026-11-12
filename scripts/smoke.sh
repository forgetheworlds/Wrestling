#!/usr/bin/env bash
# Install check on CPU: tests, a two-iteration training run, evaluation and a headless render.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
out=runs/smoke

python -m wrestling.clip check motions/*.npz
python -m unittest
python -m wrestling.train double_leg --out "$out" --iters 2 --workers 2
python -m wrestling.evaluate double_leg "$out/best.npz" --seeds 2
python -m wrestling.render double_leg_smooth checkpoints/double_leg.npz --size 320 240 --out "$out/double_leg.mp4"
echo "smoke ok: $out/double_leg.mp4"
