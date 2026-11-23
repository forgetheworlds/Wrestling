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

## Dataset strategy: original wrestling footage first

The primary data source is the project owner's **self-recorded wrestling-drill videos**.
Use external motion capture **selectively** for falling, rolling, stabilizing,
getting up from the mat, and returning to a standing wrestling stance. General
walking and balancing examples are supplemental, not the dataset's purpose.

Three data families:

1. **Wrestling:** stance, stance hold, forward/backward/side shuffles, level
   changes, penetration steps, double-leg entries, sprawling, sit-outs,
   technical stand-ups, and recovery to stance. Record multiple trials per
   movement, including distinct speeds, stances, and approach directions.
2. **Recovery mocap:** supine/prone/left-side/right-side to upright, kneeling
   or seated to upright, roll-to-kneel, stumble recovery, and fall-to-stand
   trajectories. Keep full transitions with meaningful beginning and ending.
3. **Simulator-generated recovery conditions:** randomized orientation, joint
   pose, small disturbances, friction and COM perturbations, and imperfect
   starts, applied **only within physically feasible and mechanically safe
   states**. This is robustness training, not a source of human mocap.

Prioritize data quality over clip count. Keep a traceable manifest with source,
permission/license, subject/session, original fps, movement, retargeting
version, contact labels, quality score, and whether a clip is allowed to be
redistributed. Do not mix train and test frames from the same recording.

### Verified external sources and permissions

- [Ubisoft La Forge LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset):
  6 fall-and-get-up, 2 push-and-fall, 3 push-and-stumble, and 5 ground
  movement sequences. An [existing G1 retarget](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
  is available; check kinematic feasibility in our G1 model. Source motion
  license is CC BY-NC-ND 4.0. Do not assume that processed references may be
  redistributed.
- [CMU Mocap Subject 140](https://mocap.cs.cmu.edu/search.php?subjectnumber=140):
  9 getting-up-from-ground trials. CMU's original database permits broad
  reuse but forbids direct resale of the dataset. Retarget to G1 and verify.
- [SFU Mocap](https://mocap.cs.sfu.ca/):
  research-only floor/parkour rolls; cannot use in commercial products.
- [BABEL/AMASS](https://babel.is.tue.mpg.de/):
  semantic labels and third-party mocap for locating additional recovery
  motions, subject to their non-commercial and redistribution restrictions.

**Do not presume BONES-SEED is a freely distributable open dataset.** Bones
Studio's official data is gated under a proprietary license and should not be
used or redistributed without permission. It is not needed for the initial plan.

### Capture protocol for primary wrestling motion

Use a fixed camera with the entire body and mat visible, consistent fps and
lighting, known scale/calibration, and 1–2 seconds of stable start and end.
Prefer synchronized side and front/oblique views for floor maneuvers with
self-occlusion, recorded at 60 fps or higher when available. Capture multiple
natural repetitions, not edited copies. Tag likely contact periods for each
foot, hand and knee, with confidence rather than fabricated force values.
Retarget to G1, filter jitter without erasing timing, validate contacts and
joint limits, then replay in mjlab before accepting the motion.

### Training / sampling

Start with one clip per skill for pipeline verification. Then build a balanced
multimotion library using *episode/clip sampling*, not raw frame count:
consider an initial 60% wrestling / 30% recovery / 10% other locomotion as a
**tunable experiment**, not a proven optimum. Within wrestling, oversample
difficult, underrepresented transitions such as hand-posting sprawls.
Compare held-out repetitions and new recording sessions to training results.

A reference-conditioned tracker performs the supplied clip; by itself it does
**not** decide how to get up after an arbitrary unexpected fall. Once tracked
get-up trajectories work, add a narrow recovery-mode selector that routes
measured robot state (supine, prone, side, kneeling, etc.) to an achievable
recovery reference. Train perturbation recovery within the simulator and
measure success under a range of starts, rather than claiming that literally
every fall is recoverable.

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
