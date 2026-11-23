"""Contract tests for the SuperMimic motion adapter (no G1 asset needed)."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wrestling.supermimic_export import MJLAB_G1_JOINTS, export_motion


class SuperMimicExportTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.source = self.root / "legacy.npz"
        self.output = self.root / "out.npz"
        self.names = list(reversed(MJLAB_G1_JOINTS))
        self.qpos = np.zeros((4, 36))
        self.qvel = np.zeros((4, 35))
        # +90deg yaw: local angular x points toward world y.
        self.qpos[:, 3:7] = [2**-0.5, 0, 0, 2**-0.5]
        self.qpos[:, 7:] = np.arange(29)
        self.qvel[:, 6:] = np.arange(29) * 0.1
        self.qvel[:, 3] = 1.0
        self.qvel[:, 0] = 0.25

    def write_source(self):
        np.savez(
            self.source,
            qpos=self.qpos,
            qvel=self.qvel,
            times=np.arange(4) * 0.02,
            meta_json=json.dumps({"dt": 0.02, "name": "test"}),
        )

    def test_joint_reordering_and_world_angular_velocity(self):
        self.write_source()
        result = export_motion(
            self.source, self.output, source_joint_names=self.names
        )
        self.assertEqual(result, self.output)
        with np.load(self.output, allow_pickle=False) as data:
            self.assertEqual(data["joint_names"].tolist(), list(MJLAB_G1_JOINTS))
            np.testing.assert_allclose(data["joint_pos"][0], np.arange(28, -1, -1))
            np.testing.assert_allclose(
                data["joint_vel"][0], np.arange(28, -1, -1) * 0.1, atol=1e-6
            )
            np.testing.assert_allclose(data["root_ang_vel_w"][0], [0, 1, 0], atol=1e-6)
            np.testing.assert_allclose(data["root_lin_vel_w"][0], [0.25, 0, 0])
            self.assertEqual(float(data["fps"][0]), 50.0)
            self.assertIn("wrestling_source_meta_json", data.files)

    def test_bad_quaternion_rejected(self):
        self.qpos[:, 3:7] = [2, 0, 0, 0]
        self.write_source()
        with self.assertRaisesRegex(ValueError, "non-unit root quaternions"):
            export_motion(self.source, self.output, source_joint_names=self.names)

    def test_wrong_joint_mapping_rejected(self):
        self.write_source()
        with self.assertRaisesRegex(ValueError, "joint names"):
            export_motion(
                self.source,
                self.output,
                source_joint_names=["wrong_joint"] * 29,
            )

    def test_bad_timebase_rejected(self):
        self.write_source()
        with np.load(self.source, allow_pickle=False) as data:
            contents = {k: data[k] for k in data.files}
        contents["times"] = np.arange(4) * 0.03
        np.savez(self.source, **contents)
        with self.assertRaisesRegex(ValueError, "inconsistent frame timestamps"):
            export_motion(self.source, self.output, source_joint_names=self.names)


if __name__ == "__main__":
    unittest.main()
