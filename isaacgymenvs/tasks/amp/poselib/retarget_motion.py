import torch

from poselib.core.rotation3d import (
    quat_from_angle_axis,
    quat_mul,
    quat_rotate,
)
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState


def _unit(vec):
    return vec / torch.norm(vec, dim=-1, keepdim=True).clamp(min=1e-8)


def _expanded_local_dir(motion, child_id, reference_quat):
    local_dir = _unit(motion.skeleton_tree.local_translation[child_id])
    shape = (1,) * (reference_quat.dim() - 1) + (3,)
    return local_dir.view(shape).expand(reference_quat.shape[:-1] + (3,))


def _hinge_axis(device, dtype):
    return torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=dtype)


def _identity_quat(shape, device, dtype):
    quat = torch.zeros(shape + (4,), device=device, dtype=dtype)
    quat[..., 3] = 1.0
    return quat


def _project_limb(motion, parent_name, hinge_name, child_name, hinge_sign, positive_compensation):
    ids = motion.skeleton_tree._node_indices
    parent_id = ids[parent_name]
    hinge_id = ids[hinge_name]
    child_id = ids[child_name]

    parent_pos = motion.global_translation[..., parent_id, :]
    hinge_pos = motion.global_translation[..., hinge_id, :]
    child_pos = motion.global_translation[..., child_id, :]
    parent_rot = motion.local_rotation[..., parent_id, :]
    hinge_rot = motion.local_rotation[..., hinge_id, :]

    upper_delta = _unit(parent_pos - hinge_pos)
    lower_delta = _unit(child_pos - hinge_pos)
    hinge_dot = torch.sum(-upper_delta * lower_delta, dim=-1).clamp(-1.0, 1.0)
    hinge_theta = torch.acos(hinge_dot)
    hinge_quat = quat_from_angle_axis(
        hinge_sign * torch.abs(hinge_theta),
        _hinge_axis(hinge_theta.device, hinge_theta.dtype),
    )

    local_dir = _expanded_local_dir(motion, child_id, hinge_rot)
    projected_dir = quat_rotate(hinge_quat, local_dir)
    original_dir = quat_rotate(hinge_rot, local_dir)
    comp_dot = torch.sum(original_dir * projected_dir, dim=-1).clamp(-1.0, 1.0)
    comp_theta = torch.acos(comp_dot)

    if positive_compensation:
        comp_theta = torch.where(original_dir[..., 1] >= 0, comp_theta, -comp_theta)
    else:
        comp_theta = torch.where(original_dir[..., 1] <= 0, comp_theta, -comp_theta)

    local_axis = local_dir[(0,) * (local_dir.dim() - 1)]
    comp_quat = quat_from_angle_axis(comp_theta, local_axis.unsqueeze(0))
    return parent_id, hinge_id, quat_mul(parent_rot, comp_quat), hinge_quat


def project_joints(motion):
    """Project AMP humanoid elbows and knees to their 1-DOF hinge axes."""
    new_local_rotation = motion.local_rotation.clone()

    limb_specs = [
        ("right_upper_arm", "right_lower_arm", "right_hand", -1.0, False),
        ("left_upper_arm", "left_lower_arm", "left_hand", -1.0, False),
        ("right_thigh", "right_shin", "right_foot", 1.0, True),
        ("left_thigh", "left_shin", "left_foot", 1.0, True),
    ]
    for spec in limb_specs:
        parent_id, hinge_id, parent_rot, hinge_quat = _project_limb(motion, *spec)
        new_local_rotation[..., parent_id, :] = parent_rot
        new_local_rotation[..., hinge_id, :] = hinge_quat

    ids = motion.skeleton_tree._node_indices
    hand_shape = new_local_rotation[..., ids["left_hand"], :].shape[:-1]
    identity = _identity_quat(hand_shape, new_local_rotation.device, new_local_rotation.dtype)
    new_local_rotation[..., ids["left_hand"], :] = identity
    new_local_rotation[..., ids["right_hand"], :] = identity

    new_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree,
        new_local_rotation,
        motion.root_translation,
        is_local=True,
    )
    return SkeletonMotion.from_skeleton_state(new_state, fps=motion.fps)
