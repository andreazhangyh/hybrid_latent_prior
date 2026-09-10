import itertools
import sys
import unittest
from pathlib import Path

import isaacgym
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'isaacgymenvs/tasks/amp/poselib'))
from poselib.core.rotation3d import quat_from_rotation_matrix
from scripts.evaluate_imitation_suite import nearest_transition
from scripts.retarget_raw_lafan import raw_tpose
from poselib.skeleton.skeleton3d import SkeletonTree


class EvaluationTests(unittest.TestCase):
    def test_matrix_conversion_signed_permutations(self):
        matrices = []
        for permutation in itertools.permutations(range(3)):
            for signs in itertools.product((-1, 1), repeat=3):
                matrix = np.eye(3)[list(permutation)] * np.array(signs)[:, None]
                if np.linalg.det(matrix) > 0:
                    matrices.append(matrix)
        q = quat_from_rotation_matrix(torch.tensor(np.array(matrices), dtype=torch.float64))
        np.testing.assert_allclose(Rotation.from_quat(q.numpy()).as_matrix(), matrices, atol=1e-12)

    def test_chunked_matching_equals_definition(self):
        torch.manual_seed(1)
        query, bank = torch.randn(11, 20), torch.randn(37, 20)
        direct = ((query[:, None, :10] - bank[None, :, :10]).square().mean(-1)
                  + (query[:, None, 10:] - bank[None, :, 10:]).square().mean(-1))
        value, index = nearest_transition(query, bank, chunk=7)
        expected, expected_index = direct.min(-1)
        self.assertTrue(torch.allclose(value, expected, atol=1e-6))
        self.assertTrue(torch.equal(index, expected_index))

    def test_raw_rest_pose_geometry(self):
        path = ROOT / 'isaacgymenvs/tasks/amp/poselib/data/LAFAN/lafan1_npz_2026-Sep-07/train/walk1_subject1.npz'
        if not path.exists():
            self.skipTest('Local source dataset not installed')
        data = np.load(str(path), allow_pickle=True)['arr_0'].item()
        tree = SkeletonTree(data['node_names'], torch.tensor(data['parent_indices']), torch.tensor(data['joint_offsets']))
        pose = raw_tpose(tree)
        pos = pose.global_translation
        ids = tree._node_indices
        # Rest skeleton after Y-up -> Z-up: legs down, trunk up, arms lateral.
        for side, sign in [('Left', 1), ('Right', -1)]:
            leg = pos[ids[side+'Leg']] - pos[ids[side+'UpLeg']]
            arm = pos[ids[side+'ForeArm']] - pos[ids[side+'Arm']]
            self.assertLess(float(leg[2]), -40)
            self.assertGreater(float(arm[1]) * sign, 30)
            self.assertLess(float(arm[[0, 2]].abs().max()), .01)
        self.assertGreater(float(pos[ids['Neck'], 2] - pos[ids['Hips'], 2]), 40)


if __name__ == '__main__':
    unittest.main()
