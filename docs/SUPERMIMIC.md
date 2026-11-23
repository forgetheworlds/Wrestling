# SuperMimic wrestling controller

## Target

Train one motion-conditioned Unitree G1 controller to reproduce wrestling
references from a shared library. The motion selector is a reference clip and
time index. The actor produces joint actions every 20 ms. Its job is execution:
balance, stepping, level changes, mat contact, and clean recovery.

**Initial motion set:** stance and stance hold, forward/backward/side shuffle,
level change, double-leg penetration step, sprawl, sit-out, and return to stance.

## Implementation base

Use the official [SuperMimic](https://github.com/HybridRobotics/supermimic)
with its sibling [mjlab_research](https://github.com/HybridRobotics/mjlab_research)
checkout as the new GPU trainer. Prefer extending the existing
`SuperMimic-GenCoord-G1` tracking task over maintaining a second world-model
training loop here.

Training loop:

1. MuJoCo Warp collects transitions of the G1 executing motion targets.
2. A differentiable dynamics model fits those transitions.
3. The actor tracks reference motion windows by backpropagating imagined
   short-horizon rewards through the learned world model; a critic estimates
   the remaining return.
4. Repeat data collection, world-model fitting, and policy improvement.
5. Evaluate the actor back in the original physics simulator.

At inference the actor uses joint/IMU observations and a short future reference
window. The world model and critic are training components.

Use the default `{"motion": 1.0, "velocity": 0.0}` task allocation initially.
Velocity-only control can come later; this milestone is reference tracking.

## Motion ingestion

Keep `wrestling/pose/` for camera keypoints, contact annotation, retargeting,
and motion inspection. The original `motions/*.npz` clips remain intact.

Convert them to the current official `mjlab_research` format:

```bash
scripts/setup.sh
source .venv/bin/activate
python -m wrestling.supermimic_export motions/*.npz --out-dir work/supermimic
python -m unittest tests.test_supermimic_export
```

Exports contain `fps`, `joint_names`, `joint_pos`, `joint_vel`,
`root_pos_w`, `root_quat_w`, `root_lin_vel_w`, and `root_ang_vel_w`.
The converter reorders joints by name and rotates legacy body-frame angular
velocity to the world frame. It also carries source metadata for traceability.

Always render a few frames in **mjlab's G1** before training: a valid file does
not prove the two versions of G1 have identical proportions, contact geometry,
joint limits, or control tuning.

## First GPU experiment

Set up `supermimic` and `mjlab_research` as sibling folders using their
official installation instructions (the official trainer uses Python 3.13).

Start by reproducing the official LAFAN1 tracking example. Then train with a
single exported wrestling clip:

```bash
cd /path/to/supermimic_ws/supermimic
uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --motion-file /absolute/path/to/Wrestling/work/supermimic/double_leg.npz \
  --num-envs 512 --runner.run_name wrestling_double_leg
```

The exact `--num-envs` and training config should be selected for the GPU.
Confirm that the motion plays through mjlab before launching an extended run.

When single-motion tracking works, supply a quoted glob:

```bash
uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --motion-file '/absolute/path/to/Wrestling/work/supermimic/*.npz' \
  --num-envs 512 --runner.run_name wrestling_multimotion
```

The upstream loader accepts multiple NPZ clips. Each clip retains its own
boundaries; reference windows and reset sampling must respect those boundaries.

## Data progression

**A. Motion quality:** For every motion, inspect root/limb trajectories and
contacts before training. Review foot/hand/knee placement, motion timing,
penetration, joint angular velocity and acceleration, start/end poses, and
feasibility with the target G1 joint limits. For camera clips, compare both
image reprojection and 3-D physical plausibility. Fix impossible references
upstream instead of asking the policy to imitate them.

**B. General movement data:** Add a compatible, licensed G1-retargeted subset
of BONES-SEED or LAFAN1 to cover balance, walking, turning, stepping, and
getting up. Validate joint order and compatibility with the same mjlab asset.
Wrestling motions should be sampled deliberately so that the large general
dataset does not drown out rare transitions.

**C. Wrestling-specific data:** Record several clear repetitions at different
speeds and starting configurations, including an actual sprawl with hand
posting and recovery. One clean repetition is enough to test tracking; multiple
variations are needed to test robustness and avoid overfitting a single clip.

**D. Contact:** SuperMimic's reference format is kinematic. It does not need
measured force-plate trajectories. Ground reaction forces arise in MuJoCo
during training. Preserve contact labels for evaluation, use non-penetrating
contact-consistent references, and inspect normal forces/impulses at feet,
knees and hands in rollouts. For sprawls and sit-outs, verify that the
simulator's G1 has physical hand and knee collision geometry and does not
treat intended floor contacts as fatal falls.

## Evaluation contract

Compare the new shared policy against the existing per-skill PPO baseline.
For every motion, report:

- Full-clip completion on 16 repeatable seeds from frame zero, plus random
  starting frames (RSI), and recovery from small external perturbations.
- Joint/body/root tracking error, including error around foot touchdown,
  hand posting and knee contact.
- Velocity and acceleration smoothness, action jerk, slip distance, and
  contact/impact impulses. Compare smoothness only at comparable tracking
  accuracy.
- A policy-vs-reference video, with contact points and failure timestamps.

A checkpoint passes the initial milestone when it reliably completes forward
and backward shuffles and double-leg entries with one set of weights.
A sprawl is a separate milestone because the current video-derived reference
and floor contact behavior need specific validation.

## Code migration

Preserve `wrestling/train.py`, `wrestling/ppo.py`, historical checkpoints,
and results as a baseline. The new training source is upstream SuperMimic.
Keep the local package focused on motion capture, retargeting, dataset
validation, motion exports, and evaluation reports.

Next implementation tasks:

1. Replay converted motions in the official mjlab G1 and verify mapping.
2. Build a clip-quality report including feet/hand/knee contact timelines.
3. Add a balanced motion-library manifest and dataset split by performance/
   recording, not random frames.
4. Train one general-movement baseline, then add curated wrestling references.
5. Port evaluation metrics to the SuperMimic policy and publish shared-policy
   rollouts alongside the legacy checkpoint comparisons.
