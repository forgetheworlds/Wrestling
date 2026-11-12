# Adding a move

A move is a reference motion in `motions/` plus a skill config in `configs/skills/`. No code changes.

## 1. Get a motion

**From video.** One person, whole body in frame, fixed camera.

```bash
scripts/setup_pose.sh                              # once
scripts/video_to_motion.sh my_clip.mp4 my_move     # -> motions/my_move.npz
```

The script retargets the clip, gates it against the video and annotates contacts. It exits with 2 if no
retarget passed the gate; `work/my_move/gate.json` says which check failed and
`work/my_move/my_move_vs_video.mp4` shows the robot over the video.

**From another retargeter.** A CSV with one frame per row: root `x y z` (m), root quaternion, then the
29 joint angles (rad) in G1 model order (`python -m wrestling.clip joints`). This is the layout of the
public G1 retargets of LAFAN1.

```bash
python -m wrestling.clip import my_move.csv --fps 30 --name my_move   # --quat wxyz if needed
```

## 2. Check it

```bash
python -m wrestling.clip check motions/my_move.npz
```

Errors (wrong shapes, wrong rate, NaNs, bad quaternions) stop training from working. Warnings (joints past
their limits, feet through the floor, mismatched velocities, very jerky joints) make it harder.

## Motion format

`.npz`, 50 Hz (`1 / (timestep * frame_skip)` in `configs/mujoco.json`), z up, mat at z = 0.

| Array | Shape | Contents |
|---|---|---|
| `qpos` | N × 36 | root position (m), root quaternion (w x y z), 29 joint angles (rad) |
| `qvel` | N × 35 | root linear velocity (world, m/s), root angular velocity (body, rad/s), joint velocities (rad/s) |
| `ctrl` | N × 29 | joint targets, the same as `qpos[:, 7:]` |
| `times` | N | seconds, 0.02 apart |
| `meta_json` | | JSON: `name`, `source`, `fps`, `dt`, `n_frames`; `contact_schedule`, `hand_contact` and `cmo_stages` from `python -m wrestling.pose.annotate` |

## 3. Add a config

`configs/skills/my_move.json`. The shortest version reuses an existing skill's rewards and training settings:

```json
{
 "name": "my_move",
 "extends": "shuffle_forward.json",
 "clips": [{"id": "my_move", "motion": "motions/my_move.npz", "skill": "shuffle"}]
}
```

| Field | |
|---|---|
| `reward_spec` | `shuffle` for footwork, `shot_sprawl` for level changes and mat contact (`configs/rewards/`) |
| `clips` | one or more motions; per clip `reward` overrides, e.g. `ground_contacts.allowed: ["feet", "knees"]` |
| `skill` | the label given to the policy; same labels let you start from an existing checkpoint |
| `augment` | `yaw_deg`, `time_scale` ranges for randomised heading and speed |
| `reward` | overrides of the reward spec and environment (`env`) |
| `train` | `minutes`, `workers`, `lr`, ... (defaults in `wrestling/skills.py`) |
| `extends` | start from another config; this is how fine-tuning stages are written |

## 4. Train, evaluate, render

```bash
python -m wrestling.train my_move                    # runs/my_move/best.npz
python -m wrestling.evaluate my_move runs/my_move/best.npz
python -m wrestling.render my_move runs/my_move/best.npz --out my_move.mp4
```

`--init checkpoints/shuffle_forward.npz` starts from a finished skill instead of from scratch.

## Worked example: side shuffle

`motions/shuffle_left.npz` came from a phone clip of a left side shuffle through
`scripts/video_to_motion.sh` (passed the gate on the first try, 5.0% mean reprojection error).
`configs/skills/shuffle_left.json` is the three-line config above. Starting from the forward shuffle,
12 minutes of training on 7 CPU workers took it from 0/16 to 16/16 completed rollouts:

```bash
python -m wrestling.clip check motions/shuffle_left.npz
python -m wrestling.train shuffle_left --init checkpoints/shuffle_forward.npz --minutes 12 --workers 7
python -m wrestling.evaluate shuffle_left runs/shuffle_left/best.npz
python -m wrestling.render shuffle_left runs/shuffle_left/best.npz --out shuffle_left.mp4
```

The pose is still looser than the reference at this point; a full run and the smoothing stage tighten it.
