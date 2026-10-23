"""Regression: the shipped checkpoints still complete their clips, and the reference motions are well formed."""

import json
import unittest
from pathlib import Path

import numpy as np

from wrestling.evaluate import evaluate
from wrestling.policy import load_checkpoint
from wrestling.pose.annotate import stages_and_contacts
from wrestling.skills import load_skill

ROOT = Path(__file__).resolve().parents[1]
FINISHED = {
    "double_leg": "configs/skills/double_leg_smooth.json",
    "shuffle_forward": "configs/skills/shuffle_forward_smooth.json",
}


class CheckpointTest(unittest.TestCase):
    def test_finished_skills_complete_the_clip(self):
        for name, cfg in FINISHED.items():
            params, _ = load_checkpoint(ROOT / "checkpoints" / f"{name}.npz")
            rows = evaluate(load_skill(ROOT / cfg), params, seeds=3)
            with self.subTest(skill=name):
                self.assertTrue(all(r["success"] for r in rows), rows)


class MotionTest(unittest.TestCase):
    def test_motions(self):
        for path in sorted((ROOT / "motions").glob("*.npz")):
            z = np.load(path)
            meta = json.loads(str(z["meta_json"]))
            with self.subTest(motion=path.stem):
                self.assertEqual(z["qpos"].shape[1], 36)
                self.assertEqual(len(z["qpos"]), meta["n_frames"])
                np.testing.assert_allclose(np.linalg.norm(z["qpos"][:, 3:7], axis=1), 1.0, atol=1e-6)
                if "cmo_stages" in meta:
                    stages, schedule = stages_and_contacts(z["qpos"], z["times"])
                    self.assertEqual(stages, meta["cmo_stages"])
                    self.assertEqual(schedule, meta["contact_schedule"])


if __name__ == "__main__":
    unittest.main()
