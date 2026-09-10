"""Retarget the original Ubisoft centimeter/+X-bone LaFAN convention.

Writes a separate dataset; never changes active motion links or source files.
The legacy converter assumed an already reoriented, meter-scale skeleton.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
POSELIB = ROOT / 'isaacgymenvs/tasks/amp/poselib'
sys.path.insert(0, str(POSELIB))
from poselib.core.rotation3d import quat_from_rotation_matrix, quat_mul
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from retarget_motion import project_joints


def raw_tpose(tree):
    # Global rest frames in the original Y-up BVH coordinates. Bone +X is
    # up for the trunk, down for legs, and lateral for outstretched arms.
    trunk = [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]
    leg = [[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]]
    left = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
    right = [[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]]
    frames = []
    for name in tree.node_names:
        if any(x in name for x in ('UpLeg', 'Leg', 'Foot', 'Toe')):
            frame = leg
        elif any(x in name for x in ('Shoulder', 'Arm', 'Hand')):
            frame = left if name.startswith('Left') else right
        else:
            frame = trunk
        frames.append(frame)
    # The legacy poselib matrix conversion applies overlapping sign branches
    # at tied quaternion magnitudes (these signed permutation frames have
    # exactly such ties). Use scipy's stable matrix conversion here.
    q = torch.tensor(Rotation.from_matrix(frames).as_quat(), dtype=torch.float32)
    # Same world-coordinate conversion as SkeletonMotion.from_lafan_npz.
    world = torch.tensor([0.5, 0.5, 0.5, 0.5])
    q = quat_mul(world.expand_as(q), q)
    return SkeletonState.from_rotation_and_root_translation(
        tree, q, torch.zeros(3), is_local=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new output directory: %s' % args.output)
    config = json.loads((POSELIB / 'data/configs/retarget_lafan_to_amp.json').read_text())
    target = SkeletonState.from_file(str(POSELIB / config['target_tpose']))
    records = []
    for path in sorted(args.source.rglob('*.npz')):
        motion = SkeletonMotion.from_lafan_npz(str(path))
        # Validate raw BVH convention instead of silently treating arbitrary NPZ
        # data as centimeters or using these frames for a modified skeleton.
        shin = motion.skeleton_tree.local_translation[motion.skeleton_tree.index('LeftLeg')]
        if not (30 < shin[0] < 60 and shin[1:].abs().max() < 0.1):
            raise ValueError('Expected original LaFAN centimeter/+X bones: %s' % path)
        corrected = motion.retarget_to_by_tpose(
            joint_mapping=config['joint_mapping'], source_tpose=raw_tpose(motion.skeleton_tree),
            target_tpose=target, rotation_to_target_skeleton=torch.tensor(config['rotation']),
            scale_to_target_skeleton=0.01)
        corrected = project_joints(corrected)
        root = corrected.root_translation.clone()
        root[:, 2] += 0.05 - corrected.global_translation[..., 2].min()
        state = SkeletonState.from_rotation_and_root_translation(
            corrected.skeleton_tree, corrected.local_rotation, root, is_local=True)
        corrected = SkeletonMotion.from_skeleton_state(state, fps=corrected.fps)
        dest = args.output / path.relative_to(args.source).with_suffix('.npy')
        dest.parent.mkdir(parents=True, exist_ok=True)
        corrected.to_file(str(dest))
        records.append(dict(path=str(dest), frames=len(root), fps=corrected.fps,
                            root_height_min=float(root[:, 2].min()),
                            root_height_max=float(root[:, 2].max()),
                            root_speed_max=float(corrected.global_root_velocity.norm(dim=-1).max())))
        print(dest, flush=True)
    (args.output / 'conversion_manifest.json').write_text(json.dumps(dict(
        source=str(args.source.resolve()), scale=0.01, convention='original_ubisoft_bvh',
        records=records), indent=2))


if __name__ == '__main__':
    main()
