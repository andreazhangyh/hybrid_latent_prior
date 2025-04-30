# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE..

import os
from enum import Enum

# for motion library
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import gym
import numpy as np
import omegaconf
import torch
import torch_scatter
from gym import spaces
from isaacgym import gymapi, gymtorch

from isaacgymenvs.tasks.amp.humanoid_amp_base import (
    DOF_BODY_IDS,
    DOF_OFFSETS,
    HumanoidAMPBase,
    compute_humanoid_reset,
    dof_to_obs,
)
from isaacgymenvs.tasks.amp.utils_amp import gym_util
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib
from isaacgymenvs.utils.torch_jit_utils import (
    calc_heading_quat,
    calc_heading_quat_inv,
    my_quat_rotate,
    quat_conjugate,
    quat_diff_rad,
    quat_mul,
    quat_to_rotation_6d,
    slerp,
    to_torch,
    normalize
)
from isaacgymenvs.utils.smpl_sim.smpl_eval import compute_metrics_lite

PLANE_CONTACT_THRESHOLD = 0.05  # at the initial frame, foot must higher than this value
# Obses
# NUM_OBS - Proprioceptive states (defined in humanoid_amp_base.py)
NUM_NEXT_OBS = 15 * (3 + 6 + 3 + 3)  # right next frame

ET_THRESHOLD = 0.1
PREPARATION_TIME = 0
ET_TIMEWINDOW = 30

class HumanoidImitate(HumanoidAMPBase):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = cfg

        # render
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidImitate.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        # goal
        self._num_next_obs_steps = cfg["env"].get("numNextObsSteps", 1)
        self.goal_obs_space = spaces.Box(np.ones(self.num_goal_obs) * -np.Inf, np.ones(self.num_goal_obs) * np.Inf)
        self.clip_goal_obs = self.cfg["env"].get("clipGoalObservations", np.Inf)

        # reward style
        self.imitation_rew_style = self.cfg["env"]["imitationRewStyle"]
        self.imitation_reset_style = self.cfg["env"]["imitationResetStyle"]
        if not self.cfg["env"]["test"]: # train
            try:
                assert not (self.imitation_rew_style == "deepmimic" and not self.cfg["env"]["enableEarlyTermination"])
            except:
                raise ValueError("deepmimic reward is only allowable for training where early termination is enabled!")

        self._energy_rew_coef = self.cfg["env"].get("energyRewCoef", 0.0)

        # visualize reference motion
        self._display_reference = self.cfg["env"]["displayReference"] and self.cfg["env"]["test"] and not headless and not self.cfg["env"].get("prior_rollout", False)

        # motion sampling related
        fps = round(1 / (self.cfg["sim"]["dt"] + 1e-7))  # assume fps is always integer
        control_freq_inv = self.cfg["env"]["controlFrequencyInv"]
        self.max_episode_length_in_time = max_episode_length_in_time = self.cfg["env"]["episodeLength"] / (
            fps / control_freq_inv
        )  # in seconds (60 / ((1/60) / 2))
        self.truncate_time = self.cfg["env"].get("truncateTime", max_episode_length_in_time)  # use for sampling reference motion

        # render
        self.cfg["env"]["renderFPS"] = 60
        if self.cfg["env"]["kinematic"]:
            self.ori_control_freq_inv = self.cfg["env"]["controlFrequencyInv"]
            self.cfg["env"]["controlFrequencyInv"] = 1
            self.cfg["env"]["renderFPS"] = 30
            self._enable_early_termination = False
        
        self.render_every = max(fps // self.cfg["env"]["renderFPS"], 1)

        # eval - prior rollout
        self.cfg["env"]["prior_rollout"] = False if not self.cfg["env"]["test"] else self.cfg["env"].get("prior_rollout", False)
        if self.cfg["env"]["prior_rollout"]:
            self.cfg["env"]["envSpacing"] = 0
            self.cfg["env"]["enableEarlyTermination"] = False

        # eval - jittering test
        self.eval_jitter = cfg["env"].get("eval_jitter", False) if cfg["env"]["test"] else False
        self.eval_metric = self.cfg["env"].get("eval_metric", False)

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )
        motion_file = cfg["env"].get("motion_file", "amp_humanoid_backflip.npy")
        motion_file_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/amp/motions/")

        if not isinstance(motion_file, omegaconf.listconfig.ListConfig):
            motion_file_list = [motion_file,]
        else:
            motion_file_list = motion_file

        self._low_memory_load = cfg["env"].get("low_memory_load", False)
        self._noise_scale = cfg["env"].get("noise_scale", 0.0)
        self._add_noise = True if self._noise_scale > 0 else False

        motion_file_path = []
        for motion_file in motion_file_list:
            if isinstance(motion_file, str):
                motion_file = os.path.join(motion_file_root_path, motion_file)
                # Case 1 : if it is single motion file
                if motion_file.split(".")[-1] == "npy":
                    motion_file_path.append(motion_file)
                # Case 2 : if it is directory
                elif os.path.isdir(motion_file):
                    if motion_file.split("/")[-1] in ["LAFAN_LOCO"]:
                        subset = "test" if self.cfg["env"]["test"] else "train"
                        motion_file = os.path.join(motion_file, subset)
                    temp_motion_file_path = list(Path(motion_file).rglob("*.npy"))
                    if "LAFAN_ALL" in motion_file:
                        if hasattr(self, "exclude_motion_keys"):
                            for exclude_key in self.exclude_motion_keys:
                                temp_motion_file_path = [path for path in temp_motion_file_path if exclude_key not in str(path)]
                        else:
                            temp_motion_file_path = [path for path in temp_motion_file_path if 'obstacles' not in str(path)]
                    elif "ASSASSIN_MOVES" in motion_file:
                        excludes = ['level-to-land', 'jump-wall', 'jump-to-level']
                        temp = []
                        for path in temp_motion_file_path:
                            flag = False
                            for exclude in excludes:
                                if exclude in str(path):
                                    flag = True
                                    break
                            if not flag:
                                temp.append(path)
                        temp_motion_file_path = temp
                    motion_file_path.extend(temp_motion_file_path)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

        self.motion_file_name = [str(m).split('/')[-1].split('.')[0] for m in motion_file_path]
        if self._low_memory_load and self.cfg["env"]["test"]:
            self.motion_file_path = motion_file_path
            idx = torch.randint(low=0, high=len(self.motion_file_path), size=(1,)).item()
            self._load_motion(self.motion_file_path[idx:idx+1])
        else:
            self._load_motion(motion_file_path)

        # foot shape
        foot_offset = torch.tensor(
            [0.045, 0, -0.0255], device=self.device, dtype=torch.float32, requires_grad=False
        )  # offset from foot joint to the rigid body com (box)
        query_points = torch.tensor(
            [
                [0.0895, 0.045, 0.0285],
                [0.0895, 0.045, -0.0285],
                [0.0895, -0.045, 0.0285],
                [0.0895, -0.045, -0.0285],
                [-0.0895, 0.045, 0.0285],
                [-0.0895, 0.045, -0.0285],
                [-0.0895, -0.045, 0.0285],
                [-0.0895, -0.045, -0.0285],
            ],
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.query_points = foot_offset[None] + query_points  # (offset from foot joint for each query points)

        # Return Buffer
        self.return_buf = torch.zeros(
            self.num_envs, self.max_episode_length + 1, device=self.device, dtype=torch.float32
        )

        # Reference Motion Buffer
        self.all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._ref_buf_length = self.max_episode_length + self._num_next_obs_steps
        self._ref_root_states_buf = torch.zeros(
            self.num_envs, self._ref_buf_length, 13, device=self.device, dtype=torch.float32
        )
        self._ref_dof_pos_buf = torch.zeros(
            self.num_envs,
            self._ref_buf_length,
            self.humanoid_num_dof,
            device=self.device,
            dtype=torch.float32,
        )
        self._ref_dof_vel_buf = torch.zeros(
            self.num_envs,
            self._ref_buf_length,
            self.humanoid_num_dof,
            device=self.device,
            dtype=torch.float32,
        )
        self._ref_key_pos_buf = torch.zeros(
            self.num_envs,
            self._ref_buf_length,
            self._key_body_ids.shape[0],
            3,
            device=self.device,
            dtype=torch.float32,
        )
        self._ref_rigid_body_pos_buf = torch.zeros(
            self.num_envs,
            self._ref_buf_length,
            self.humanoid_num_bodies,
            3,
            device=self.device,
            dtype=torch.float32,
        )
        self._ref_rigid_body_rot_buf = torch.zeros(
            self.num_envs,
            self._ref_buf_length,
            self.humanoid_num_bodies,
            4,
            device=self.device,
            dtype=torch.float32,
        )
        self._ref_rigid_body_vel_buf = torch.zeros(
            self.num_envs,
            self._ref_buf_length,
            self.humanoid_num_bodies,
            3,
            device=self.device,
            dtype=torch.float32,
        )
        self._ref_rigid_body_ang_vel_buf = torch.zeros(
            self.num_envs,
            self._ref_buf_length,
            self.humanoid_num_bodies,
            3,
            device=self.device,
            dtype=torch.float32,
        )
        self._curr_motion_ids = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.int64,
        )

        self._updating_motion_weight = False

        if self._display_reference and hasattr(self, "_ref_actor_idx"):
            if self.cfg["env"]["kinematic"]:
                self.ref_actor_indices = self.global_actor_indices[:, 0].clone()
                self.ref_dof_actor_indices = self.global_dof_actor_indices[:, 0].clone()
            else:
                self.ref_actor_indices = self.global_actor_indices[:, 1 + self._ref_actor_idx].clone()
                self.ref_dof_actor_indices = self.global_dof_actor_indices[:, 1 + self._ref_dof_start_idx].clone()
        return

    def _create_ground_plane(self):
        if not self._display_reference:
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            plane_params.static_friction = self.plane_static_friction
            plane_params.dynamic_friction = self.plane_dynamic_friction
            plane_params.restitution = self.plane_restitution
            self.gym.add_ground(self.sim, plane_params)
        else: # use box instead of ground plane (to disable collision to reference bodies)
            BOX_X, BOX_Y, BOX_Z = 100, 100, 0.2
            box_dims = gymapi.Vec3(BOX_X, BOX_Y, BOX_Z)
            box_asset_options = gymapi.AssetOptions()
            box_asset_options.fix_base_link = True
            # box_asset_options.collapse_fixed_joints = True
            box_asset_options.disable_gravity = True
            box_asset = self.gym.create_box(self.sim, box_dims.x, box_dims.y, box_dims.z, box_asset_options)
            box_pose = gymapi.Transform()
            box_pose.p = gymapi.Vec3(0, 0, -BOX_Z / 2)
            box_prop = self.gym.get_asset_rigid_shape_properties(box_asset)
            box_prop[0].friction = 1.0
            box_prop[0].rolling_friction = box_prop[0].friction / 100.0
            self.gym.set_asset_rigid_shape_properties(box_asset, box_prop)
            self.ground_asset = box_asset
            self.ground_start_pose = box_pose
        return

    def _prepare_else_assets(self):
        # NOTE THAT HUMANOID has priority
        # Ex) MUST DEFINE HUMANOID HANDLE AND THAN OBJECTS
        else_assets, else_start_poses, else_num_bodies, else_num_shapes = [], [], [], []
        self.num_else_actor = 0
        # ground plane
        if hasattr(self, "ground_asset"):
            else_assets.append(self.ground_asset)
            else_start_poses.append(self.ground_start_pose)
            else_num_bodies.append(1)
            else_num_shapes.append(1)
            self.num_else_actor += 1

        if self._display_reference and not self.cfg["env"]["kinematic"]:
            humanoid_asset_options = gymapi.AssetOptions()
            humanoid_asset_file = self.humanoid_asset_file
            # motion reference actor
            humanoid_asset_options.fix_base_link = False
            humanoid_asset_options.disable_gravity = True
            ref_humanoid_asset = self.gym.load_asset(
                self.sim, self.humanoid_asset_root, humanoid_asset_file, humanoid_asset_options
            )
            else_assets.append(ref_humanoid_asset)
            else_start_poses.append(self.humanoid_start_pose)
            else_num_bodies.append(self.gym.get_asset_rigid_body_count(ref_humanoid_asset))
            else_num_shapes.append(self.gym.get_asset_rigid_shape_count(ref_humanoid_asset))
            self.num_else_actor += 1

        return else_assets, else_start_poses, else_num_bodies, else_num_shapes

    def _create_else_actors(self, env_ptr, env_idx, else_assets, else_start_poses):
        else_handles = []
        # ground plane
        if hasattr(self, "ground_asset"):
            contact_filter = 0
            segmentation_id = 1
            handle = self.gym.create_actor(
                env_ptr,
                else_assets[0],
                else_start_poses[0],
                "ground",
                env_idx,
                contact_filter,
                segmentation_id,
            )
            ground_color = gymapi.Vec3(0.3, 0.3, 0.3)
            self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL, ground_color)
            else_handles.append(handle)

        # agent
        if self._display_reference and not self.cfg["env"]["kinematic"]:
            offset = 1 if hasattr(self, "ground_asset") else 0
            contact_filter = (
                1  # >1 : ignore all the collision, 0 : enable all collision, -1 : collision defined by robot file
            )
            self._ref_actor_idx = 1  # index among else actors
            self._ref_dof_start_idx = 0

            # handle for ref agent
            segmentation_id = 2 # segmentation ID used in segmentation camera sensors
            handle = self.gym.create_actor(
                env_ptr,
                else_assets[offset],
                else_start_poses[offset],
                "reference",
                self.num_envs + env_idx,
                contact_filter,
                segmentation_id,
            )
            humanoid_color = gymapi.Vec3(0.8, 0.8, 0.8)
            for j in range(self.humanoid_num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL, humanoid_color)
            else_handles.append(handle)

        return else_handles

    def set_else_actors_dof_properties(self, env_ptr, else_handles, else_assets):
        if self._display_reference:
            if self._pd_control:
                for handle, asset in zip(else_handles, else_assets):
                    dof_prop = self.gym.get_asset_dof_properties(asset)
                    dof_prop["driveMode"] = gymapi.DOF_MODE_NONE
                    dof_prop["friction"][:] = 1000000
                    dof_prop["damping"][:] = 0
                    dof_prop["stiffness"][:] = 0
                    self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)
        return

    # NCP style action to pd target
    def _action_to_pd_targets(self, action):
        pd_tar = self._humanoid_dof_pos + self._pd_action_scale * action
        # clamp
        pd_tar = torch.maximum(pd_tar, self._pd_action_limit_lower)
        pd_tar = torch.minimum(pd_tar, self._pd_action_limit_upper)
        return pd_tar

    def post_physics_step(self):
        self.progress_buf += 1

        if not self.cfg["env"]["kinematic"]:
            self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()
        
        # eval metric
        if self.eval_metric:
            self.eval_tracking_metric()
        return

    def _compute_ref_diff_observations(self, env_ids=None):
        if env_ids is None:
            B = self.num_envs
            time_indices = torch.stack([self.progress_buf + i for i in range(1, self._num_next_obs_steps + 1)], dim=-1)
            ref_rigid_body_pos = self._ref_rigid_body_pos_buf[self.all_env_ids[:, None], time_indices]
            ref_rigid_body_rot = self._ref_rigid_body_rot_buf[self.all_env_ids[:, None], time_indices]
            ref_rigid_body_vel = self._ref_rigid_body_vel_buf[self.all_env_ids[:, None], time_indices]
            ref_rigid_body_ang_vel = self._ref_rigid_body_ang_vel_buf[self.all_env_ids[:, None], time_indices]

            sim_rigid_body_pos = self._humanoid_rigid_body_pos[:, None].repeat(1, self._num_next_obs_steps, 1, 1)
            sim_rigid_body_rot = self._humanoid_rigid_body_rot[:, None].repeat(1, self._num_next_obs_steps, 1, 1)
            sim_rigid_body_vel = self._humanoid_rigid_body_vel[:, None].repeat(1, self._num_next_obs_steps, 1, 1)
            sim_rigid_body_ang_vel = self._humanoid_rigid_body_ang_vel[:, None].repeat(1, self._num_next_obs_steps, 1, 1)
            sim_root_rot = self._humanoid_root_states[:, None, 3:7].repeat(1, self._num_next_obs_steps, 1)
        else:
            B = env_ids.shape[0]
            time_indices = torch.stack([self.progress_buf[env_ids] + i for i in range(1, self._num_next_obs_steps + 1)], dim=-1)
            ref_rigid_body_pos = self._ref_rigid_body_pos_buf[env_ids[:, None], time_indices]
            ref_rigid_body_rot = self._ref_rigid_body_rot_buf[env_ids[:, None], time_indices]
            ref_rigid_body_vel = self._ref_rigid_body_vel_buf[env_ids[:, None], time_indices]
            ref_rigid_body_ang_vel = self._ref_rigid_body_ang_vel_buf[env_ids[:, None], time_indices]

            sim_rigid_body_pos = self._humanoid_rigid_body_pos[env_ids, None].repeat(1, self._num_next_obs_steps, 1, 1)
            sim_rigid_body_rot = self._humanoid_rigid_body_rot[env_ids, None].repeat(1, self._num_next_obs_steps, 1, 1)
            sim_rigid_body_vel = self._humanoid_rigid_body_vel[env_ids, None].repeat(1, self._num_next_obs_steps, 1, 1)
            sim_rigid_body_ang_vel = self._humanoid_rigid_body_ang_vel[env_ids, None].repeat(1, self._num_next_obs_steps, 1, 1)
            sim_root_rot = self._humanoid_root_states[env_ids, None, 3:7].repeat(1, self._num_next_obs_steps, 1)

        if self._add_noise:
            scale = self._noise_scale
            ref_rigid_body_pos += torch.empty_like(ref_rigid_body_pos).normal_(generator=self.env_rng) * scale
            # ref_rigid_body_rot += torch.empty_like(ref_rigid_body_rot).normal_(generator=self.env_rng) * scale
            # ref_rigid_body_rot = normalize(ref_rigid_body_rot)
            ref_rigid_body_vel += torch.empty_like(ref_rigid_body_vel).normal_(generator=self.env_rng) * scale
            # ref_rigid_body_ang_vel += torch.empty_like(ref_rigid_body_ang_vel).normal_(generator=self.env_rng) * scale

        J = self.humanoid_num_bodies
        ref_diff_obs = compute_ref_diff_observations(
            ref_rigid_body_pos.view(-1, J, 3),
            ref_rigid_body_rot.view(-1, J, 4),
            ref_rigid_body_vel.view(-1, J, 3),
            ref_rigid_body_ang_vel.view(-1, J, 3),
            sim_rigid_body_pos.view(-1, J, 3),
            sim_rigid_body_rot.view(-1, J, 4),
            sim_rigid_body_vel.view(-1, J, 3),
            sim_rigid_body_ang_vel.view(-1, J, 3),
            sim_root_rot.view(-1, 4),
        )
        ref_diff_obs = ref_diff_obs.view(B, -1)
        return ref_diff_obs

    def _compute_observations(self, env_ids=None):
        # default simulation states (defined in humanoid_amp_base.py)
        sim_obs = super()._compute_observations(env_ids=env_ids)
        goal_obs = self._compute_goal_observations(env_ids=env_ids)

        if env_ids is None:
            self.obs_buf[:] = sim_obs
            if goal_obs != None:
                self.goal_obs_buf[:] = goal_obs
        else:
            self.obs_buf[env_ids] = sim_obs
            if goal_obs != None:
                self.goal_obs_buf[env_ids] = goal_obs

    def _compute_goal_observations(self, env_ids=None):
        goal_obs = self._compute_ref_diff_observations(env_ids)
        return goal_obs

    def _compute_reward(self, actions):
        # joint positions (global)
        curr_rigid_body_pos = self._humanoid_rigid_body_pos
        goal_rigid_body_pos = self._ref_rigid_body_pos_buf[self.all_env_ids, self.progress_buf]

        # joint rotations (global)
        curr_rigid_body_rot = self._humanoid_rigid_body_rot
        goal_rigid_body_rot = self._ref_rigid_body_rot_buf[self.all_env_ids, self.progress_buf]

        # joint velocities (global)
        curr_rigid_body_vel = self._humanoid_rigid_body_vel
        goal_rigid_body_vel = self._ref_rigid_body_vel_buf[self.all_env_ids, self.progress_buf]

        # joint angular velocities (global)
        curr_rigid_body_ang_vel = self._humanoid_rigid_body_ang_vel
        goal_rigid_body_ang_vel = self._ref_rigid_body_ang_vel_buf[self.all_env_ids, self.progress_buf]

        # ee pos (global)
        curr_ee_pos = self._humanoid_rigid_body_pos[:, self._key_body_ids]
        goal_ee_pos = self._ref_key_pos_buf[self.all_env_ids, self.progress_buf]

        # joint rotations (local)
        curr_dof_pos = self._humanoid_dof_pos
        goal_dof_pos = self._ref_dof_pos_buf[self.all_env_ids, self.progress_buf]

        # joint angular velocities (local)
        curr_dof_vel = self._humanoid_dof_vel
        goal_dof_vel = self._ref_dof_vel_buf[self.all_env_ids, self.progress_buf]

        # com pos
        curr_root_pos = self._humanoid_root_states[:, :3]
        goal_root_pos = self._ref_root_states_buf[self.all_env_ids, self.progress_buf, :3]
        
        # com rot
        curr_root_rot = self._humanoid_root_states[:, 3:7]
        goal_root_rot = self._ref_root_states_buf[self.all_env_ids, self.progress_buf, 3:7]

        # com vel
        curr_root_vel = self._humanoid_root_states[:, 7:10]
        goal_root_vel = self._ref_root_states_buf[self.all_env_ids, self.progress_buf, 7:10]

        # com ang vel
        curr_root_ang_vel = self._humanoid_root_states[:, 10:]
        goal_root_ang_vel = self._ref_root_states_buf[self.all_env_ids, self.progress_buf, 10:]

        if self.imitation_rew_style in ["deepmimic", "deepmimic_mul"]:
            if self.imitation_rew_style == "deepmimic_mul":
                multiplication = True
            else:
                multiplication = False
            self.rew_buf[:] = compute_deepmimic_reward(
                curr_dof_pos,
                goal_dof_pos,
                curr_dof_vel,
                goal_dof_vel,
                curr_ee_pos,
                goal_ee_pos,
                curr_root_pos,
                goal_root_pos,
                multiplication,
            )
        elif self.imitation_rew_style == "phc":
            self.rew_buf[:] = compute_phc_reward(
                curr_rigid_body_pos,
                goal_rigid_body_pos,
                curr_rigid_body_rot,
                goal_rigid_body_rot,
                curr_rigid_body_vel,
                goal_rigid_body_vel,
                curr_rigid_body_ang_vel,
                goal_rigid_body_ang_vel,
            )
        
        elif self.imitation_rew_style == "ncp":
            self.rew_buf[:] = compute_ncp_reward(
                curr_dof_pos,
                goal_dof_pos,
                curr_dof_vel,
                goal_dof_vel,
                curr_ee_pos,
                goal_ee_pos,
                curr_root_pos,
                goal_root_pos,
                curr_root_rot,
                goal_root_rot,
                curr_root_vel,
                goal_root_vel,
                curr_root_ang_vel,
                goal_root_ang_vel,
            )
        
        if self._energy_rew_coef > 0:
            energy_rew = compute_energy_reward(
                actions=actions,
                dof_vel=curr_dof_vel,
                coef=self._energy_rew_coef
            )
            self.rew_buf = 0.95 * self.rew_buf + 0.05 * energy_rew

        self.return_buf[self.all_env_ids, self.progress_buf] = self.rew_buf

        return

    def _compute_reset(self):
        curr_rigid_body_pos = self._humanoid_rigid_body_pos
        goal_rigid_body_pos = self._ref_rigid_body_pos_buf[self.all_env_ids, self.progress_buf]
        if self.imitation_reset_style == "normal":
            self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
                self.reset_buf,
                self.progress_buf,
                self._humanoid_contact_forces,
                self._contact_body_ids,
                self._humanoid_rigid_body_pos,
                self.max_episode_length,
                self._enable_early_termination,
                self._termination_height,
            )
        elif self.imitation_reset_style == "reward":
            self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
                self.reset_buf,
                self.progress_buf,
                self._humanoid_contact_forces,
                self._contact_body_ids,
                self._humanoid_rigid_body_pos,
                self.max_episode_length,
                False,
                self._termination_height,
            )
            env_ids = (1 - self.reset_buf).nonzero(as_tuple=False).squeeze(-1)
            self.return_buf[:, : PREPARATION_TIME + 1] = 0
            if len(env_ids) > 0 and self._enable_early_termination:
                time_window = torch.stack([self.progress_buf[env_ids] - i for i in range(ET_TIMEWINDOW)], dim=-1)
                windowed_return = self.return_buf[env_ids[:, None], time_window].sum(dim=-1)
                windowed_time = torch.minimum(
                    self.progress_buf[env_ids] - PREPARATION_TIME,
                    torch.ones_like(self.progress_buf[env_ids]) * ET_TIMEWINDOW,
                )
                terminated = windowed_return < ET_THRESHOLD * windowed_time
                terminated = torch.logical_and(terminated, self.progress_buf[env_ids] > PREPARATION_TIME)
                self._terminate_buf[env_ids] = terminated.long()
                self.reset_buf[env_ids] = torch.where(
                    terminated, torch.ones_like(self.reset_buf[env_ids]), self.reset_buf[env_ids]
                )

        elif self.imitation_reset_style == "error_max":
            self.reset_buf[:], self._terminate_buf[:] = compute_imitation_reset_max(
                self.reset_buf,
                self.progress_buf,
                curr_rigid_body_pos,
                goal_rigid_body_pos,
                self._contact_body_ids,
                self.max_episode_length,
                self._enable_early_termination,
            )
        elif self.imitation_reset_style == "error_mean":
            self.reset_buf[:], self._terminate_buf[:] = compute_imitation_reset_mean(
                self.reset_buf,
                self.progress_buf,
                curr_rigid_body_pos,
                goal_rigid_body_pos,
                self._contact_body_ids,
                self.max_episode_length,
                self._enable_early_termination,
            )
        elif self.imitation_reset_style == "root_dist":
            self.reset_buf[:], self._terminate_buf[:] = compute_imitation_reset_root_dist(
                self.reset_buf,
                self.progress_buf,
                curr_rigid_body_pos[:, 0],
                goal_rigid_body_pos[:, 0],
                self.max_episode_length,
                self._enable_early_termination,
            )
        return

    @property
    def num_goal_obs(self) -> int:
        return NUM_NEXT_OBS * self._num_next_obs_steps

    @property
    def goal_observation_space(self) -> gym.Space:
        return self.goal_obs_space

    def allocate_buffers(self):
        super().allocate_buffers()
        self.goal_obs_buf = torch.zeros((self.num_envs, self.num_goal_obs), device=self.device, dtype=torch.float)

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get("actions", None):
            actions = self.dr_randomizations["actions"]["noise_lambda"](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            if self.force_render:
                if (self._display_reference or self.cfg["env"]["kinematic"]) and i == 0:
                    self._motion_sync()
                if i % self.render_every == 0:
                    self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        if self.force_render and (self._display_reference or self.cfg["env"]["kinematic"]):
            self.reset_call_flag = False

        self.control_steps += 1

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get("observations", None):
            self.obs_buf = self.dr_randomizations["observations"]["noise_lambda"](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        # goal related
        self.obs_dict["goal_obs"] = torch.clamp(
            self.goal_obs_buf,
            -self.clip_goal_obs,
            self.clip_goal_obs,
        )
        return (
            self.obs_dict,
            self.rew_buf.to(self.rl_device),
            self.reset_buf.to(self.rl_device),
            self.extras,
        )

    def reset(self):
        _ = super().reset()

        self.obs_dict["goal_obs"] = torch.clamp(
            self.goal_obs_buf,
            -self.clip_goal_obs,
            self.clip_goal_obs,
        )
        return self.obs_dict

    def reset_done(self):
        _, done_env_ids = super().reset_done()
        self.obs_dict["goal_obs"] = torch.clamp(
            self.goal_obs_buf,
            -self.clip_goal_obs,
            self.clip_goal_obs,
        )
        return self.obs_dict, done_env_ids

    def _load_motion(self, motion_file):
        if self.cfg["env"]["test"]: # to faithfully eval
            self.env_rng = torch.Generator(device=self.device)
            self.env_rng.manual_seed(self.cfg["env"]["seed"])
        else:
            self.env_rng = None
        self._motion_lib = MotionLib(
            motion_file=motion_file,
            dof_body_ids=DOF_BODY_IDS,
            dof_offsets=DOF_OFFSETS,
            key_body_ids=self._key_body_ids.cpu().numpy(),
            device=self.device,
            min_len=self.max_episode_length_in_time,
            motion_matching=self.cfg["env"].get("motion_matching", False), # eval
            generator=self.env_rng
        )
        return

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.return_buf[env_ids] = 0
        if self.force_render and (self._display_reference or self.cfg["env"]["kinematic"]):
            self.reset_call_flag = True
        return

    def _reset_actors(self, env_ids):
        if self._state_init == HumanoidImitate.StateInit.Default:
            self._reset_default(env_ids)
        elif (
            self._state_init == HumanoidImitate.StateInit.Start or self._state_init == HumanoidImitate.StateInit.Random
        ):
            self._reset_ref_state_init(env_ids)
        elif self._state_init == HumanoidImitate.StateInit.Hybrid:
            self._reset_hybrid_state_init(env_ids)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

        return

    def _reset_default(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._initial_root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self._reset_default_env_ids = env_ids
        return

    def _sample_ref_states(self, motion_ids, motion_times):
        if len(motion_ids.shape) == 1:
            motion_ids = motion_ids[:, None]
            motion_times = motion_times[:, None]
        num_envs, num_ref = motion_ids.shape
        motion_ids = motion_ids.reshape(-1)
        motion_times = motion_times.reshape(-1)

        # sample
        output_global = True
        (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            key_pos,
            global_pos,
            global_rot,
            global_vel,
            global_ang_vel,
        ) = self._motion_lib.get_motion_state(motion_ids, motion_times, output_global=output_global)
        root_pos = root_pos.view(num_envs, num_ref, 3)
        root_rot = root_rot.view(num_envs, num_ref, 4)
        root_vel = root_vel.view(num_envs, num_ref, 3)
        root_ang_vel = root_ang_vel.view(num_envs, num_ref, 3)
        dof_pos = dof_pos.view(num_envs, num_ref, self.humanoid_num_dof)
        dof_vel = dof_vel.view(num_envs, num_ref, self.humanoid_num_dof)
        key_pos = key_pos.view(num_envs, num_ref, self._key_body_ids.shape[0], 3)
        if output_global:
            global_pos = global_pos.view(num_envs, num_ref, self.humanoid_num_bodies, 3)
            global_rot = global_rot.view(num_envs, num_ref, self.humanoid_num_bodies, 4)
            global_vel = global_vel.view(num_envs, num_ref, self.humanoid_num_bodies, 3)
            global_ang_vel = global_ang_vel.view(num_envs, num_ref, self.humanoid_num_bodies, 3)

        return (
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_pos,
            global_pos,
            global_rot,
            global_vel,
            global_ang_vel,
        )

    def update_motion_weights(self, motion_values, returns, decay=0.95):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        new_motion_values = motion_values
        if len(env_ids) > 0:
            new_motion_values = motion_values.clone()
            # assume maximum reward per each step is 1
            update_normalized_returns = torch.clamp(returns[env_ids, 0] / self.max_episode_length, min=0, max=1)
            update_motion_ids = self._curr_motion_ids[env_ids]
            out = torch_scatter.scatter(
                src=update_normalized_returns,
                index=update_motion_ids,
                out=torch.zeros_like(motion_values),
                reduce="mean",
            )
            update_ids = out.nonzero(as_tuple=False).squeeze(-1)
            # EMA update
            new_motion_values[update_ids] = motion_values[update_ids].mul_(decay).add_(out[update_ids] * (1 - decay))
            if not self._updating_motion_weight:
                self._updating_motion_weight = (
                    new_motion_values.nonzero(as_tuple=False).squeeze(-1).shape[0] == new_motion_values.shape[0]
                )
            else:
                self._motion_lib._motion_weights[:] = (1 - new_motion_values).pow(3)
                self._motion_lib._motion_weights[:] /= self._motion_lib._motion_weights.sum() + 1e-6
        return new_motion_values

    def _sample_motion_ids_and_times(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        self._curr_motion_ids[env_ids] = motion_ids

        if self._state_init == HumanoidImitate.StateInit.Random or self._state_init == HumanoidImitate.StateInit.Hybrid:
            motion_times = self._motion_lib.sample_time(motion_ids, truncate_time=self.truncate_time)
            if not self.cfg["env"]["kinematic"]:
                # if some of motion clips are shorter than truncate time, then we need to set them 0
                _modify_ids = torch.where(motion_times < 0)
                motion_times[_modify_ids] = 0
            else:
                motion_times[:] = 0

            # modify
            motion_ids = torch.stack(
                [motion_ids] * (self._ref_buf_length), axis=1
            )  # (num_envs, self.max_episode_length)
            if self.cfg["env"]["kinematic"]:
                interval = self.dt * self.ori_control_freq_inv
            else:
                interval = self.dt
            motion_times = torch.stack(
                [motion_times + k * interval for k in range(self._ref_buf_length)],
                axis=1,
            )  # (num_envs, self.max_episode_length)
        elif self._state_init == HumanoidImitate.StateInit.Start:
            motion_times = torch.zeros(num_envs)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return motion_ids, motion_times

    def _reset_ref_state_init(self, env_ids):
        motion_ids, motion_times = self._sample_motion_ids_and_times(env_ids)
        if self.eval_jitter:
            motion_ids[:] = 0
            motion_times[:] = 0

        # # (prior_viz)
        if self.cfg["env"]["prior_rollout"] and not self.cfg["env"]["motion_matching"]:
            motion_ids = motion_ids[:1].expand(env_ids.shape[0], -1)
            motion_times = motion_times[:1].expand(env_ids.shape[0], -1)
        (
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_pos,
            global_pos,
            global_rot,
            global_vel,
            global_ang_vel,
        ) = self._sample_ref_states(
            motion_ids,
            motion_times,
        )

        if not self.eval_jitter:
            # # eliminate root offset - global joints
            min_foot_pos = global_pos[:, :, [11, 14], 2].min(dim=-1).values.min(dim=-1).values # (B, )
            foot_offset = min_foot_pos - PLANE_CONTACT_THRESHOLD
            foot_offset = torch.maximum(foot_offset, torch.zeros_like(foot_offset))
            root_pos[..., 2] -= foot_offset[:, None]
            global_pos[..., 2] -= foot_offset[:, None, None]
            key_pos[..., 2] -= foot_offset[:, None, None]

            # avoid contact for the first frame
            min_body_pos = global_pos[:, 0, :, 2].min(dim=-1).values # (B,)
            body_offset = min_body_pos - PLANE_CONTACT_THRESHOLD
            body_offset = torch.minimum(body_offset, torch.zeros_like(body_offset))
            root_pos[..., 2] -= body_offset[:, None]
            global_pos[..., 2] -= body_offset[:, None, None]
            key_pos[..., 2] -= body_offset[:, None, None]
        else:
            # zero velocity
            root_vel[:] = 0
            root_ang_vel[:] = 0
            dof_vel[:] = 0
            global_vel[:] = 0
            global_ang_vel[:] = 0
        
        if self.cfg["env"]["test"] and self.cfg["env"]["white_mode"]:
            global_pos[..., :2] -= root_pos[:, 0, None, None, :2]
            key_pos[..., :2] -= root_pos[:, 0, None, None, :2]
            root_pos[..., :2] -= root_pos[:, 0, None, :2]

        self._set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
            key_pos=key_pos,
            global_pos=global_pos,
            global_rot=global_rot,
            global_vel=global_vel,
            global_ang_vel=global_ang_vel,
        )

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids[:, 0]
        self._reset_ref_motion_times = motion_times[:, 0]

        if self._low_memory_load and self.cfg["env"]["test"]:
            idx = torch.randint(low=0, high=len(self.motion_file_path), size=(1,), generator=self.env_rng_cpu).item()
            self._load_motion(self.motion_file_path[idx:idx+1])
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if len(ref_reset_ids) > 0:
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if len(default_reset_ids) > 0:
            self._reset_default(default_reset_ids)

        return

    def _set_humanoid_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rigid_body_pos, rigid_body_rot, rigid_body_vel, rigid_body_ang_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel

        self._humanoid_dof_pos[env_ids] = dof_pos
        self._humanoid_dof_vel[env_ids] = dof_vel

        self._humanoid_rigid_body_pos[env_ids] = rigid_body_pos
        self._humanoid_rigid_body_rot[env_ids] = rigid_body_rot
        self._humanoid_rigid_body_vel[env_ids] = rigid_body_vel
        self._humanoid_rigid_body_ang_vel[env_ids] = rigid_body_ang_vel
        return

    def _set_else_states(
        self,
        env_ids,
        root_pos,
        root_rot,
        dof_pos,
        root_vel,
        root_ang_vel,
        dof_vel,
        key_pos,
        global_pos,
        global_rot,
        global_vel,
        global_ang_vel,
    ):
         ## set reward buf
        self._ref_root_states_buf[env_ids] = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        self._ref_dof_pos_buf[env_ids] = dof_pos
        self._ref_dof_vel_buf[env_ids] = dof_vel
        self._ref_key_pos_buf[env_ids] = key_pos
        self._ref_rigid_body_pos_buf[env_ids] = global_pos
        self._ref_rigid_body_rot_buf[env_ids] = global_rot
        self._ref_rigid_body_vel_buf[env_ids] = global_vel
        self._ref_rigid_body_ang_vel_buf[env_ids] = global_ang_vel

        # for rendering
        if self._display_reference and not self.cfg["env"]["kinematic"]:  # only for rendering
            # first set reference motion agent
            as_idx = self._ref_actor_idx
            ae_idx = as_idx + 1
            rs_idx = self._ref_actor_idx
            re_idx = rs_idx + self.humanoid_num_bodies
            ds_idx = self._ref_dof_start_idx
            de_idx = ds_idx + self.humanoid_num_dof

            # set poses
            self._else_root_states[env_ids, as_idx: ae_idx, 0:3] = root_pos[:, :1]
            self._else_root_states[env_ids, as_idx: ae_idx, 3:7] = root_rot[:, :1]
            self._else_rigid_body_pos[env_ids, rs_idx: re_idx] = global_pos[:, 0]
            self._else_rigid_body_rot[env_ids, rs_idx: re_idx] = global_rot[:, 0]
            self._else_dof_pos[
                env_ids, ds_idx: de_idx
            ] = dof_pos[:, 0]
            # zero velocities
            self._else_root_states[env_ids, as_idx: ae_idx, 7:10] = root_vel[:, :1]
            self._else_root_states[env_ids, as_idx: ae_idx, 10:13] = root_ang_vel[:, :1]
            self._else_rigid_body_vel[env_ids, rs_idx: re_idx] = global_vel[:, 0]
            self._else_rigid_body_ang_vel[env_ids, rs_idx: re_idx] = global_ang_vel[:, 0]
            self._else_dof_vel[
                env_ids, ds_idx: de_idx
            ] = dof_vel[:, 0]
            if self._pd_control:
                self._target_actions[
                    env_ids, ds_idx: de_idx
                ] = self._else_dof_pos[
                    env_ids, ds_idx: de_idx
                ]
        return

    def _set_env_state(
        self,
        env_ids,
        root_pos,
        root_rot,
        dof_pos,
        root_vel,
        root_ang_vel,
        dof_vel,
        key_pos,
        global_pos,
        global_rot,
        global_vel,
        global_ang_vel,
    ):
        self._set_humanoid_state(
            env_ids=env_ids,
            root_pos=root_pos[:, 0],
            root_rot=root_rot[:, 0],
            dof_pos=dof_pos[:, 0],
            root_vel=root_vel[:, 0],
            root_ang_vel=root_ang_vel[:, 0],
            dof_vel=dof_vel[:, 0],
            rigid_body_pos=global_pos[:, 0],
            rigid_body_rot=global_rot[:, 0],
            rigid_body_vel=global_vel[:, 0],
            rigid_body_ang_vel=global_ang_vel[:, 0],
        )
        self._set_else_states(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
            key_pos=key_pos,
            global_pos=global_pos,
            global_rot=global_rot,
            global_vel=global_vel,
            global_ang_vel=global_ang_vel,
        )
        global_actor_indices = self.global_actor_indices[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(global_actor_indices),
            len(global_actor_indices),
        )

        global_dof_actor_indices = self.global_dof_actor_indices[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(global_dof_actor_indices),
            len(global_dof_actor_indices),
        )
        return

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        
        if self.cfg["env"]["prior_rollout"]:
            # upper view (prior_viz)
            cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 8.0, 5)
        else:
            # default
            cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 3.0, 1.0)

        # static
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        if self.cfg["env"]["prior_rollout"]:
            char_root_pos = self._humanoid_root_states[:, 0:3].mean(dim=0).cpu().numpy()
        else:
            char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _motion_sync(self):
        frame_idx = self.progress_buf + 1
        goal_root_states = self._ref_root_states_buf[self.all_env_ids, frame_idx]
        goal_root_pos = goal_root_states[:, :3]
        goal_root_rot = goal_root_states[:, 3:7]
        goal_root_vel = goal_root_states[:, 7:10]
        goal_root_ang_vel = goal_root_states[:, 10:]
        goal_dof_pos = self._ref_dof_pos_buf[self.all_env_ids, frame_idx]
        goal_dof_vel = self._ref_dof_vel_buf[self.all_env_ids, frame_idx]

        # # offset for motion
        global_pos = self._ref_rigid_body_pos_buf[self.all_env_ids, frame_idx]
        global_rot = self._ref_rigid_body_rot_buf[self.all_env_ids, frame_idx]
        global_vel = self._ref_rigid_body_vel_buf[self.all_env_ids, frame_idx]
        global_ang_vel = self._ref_rigid_body_ang_vel_buf[self.all_env_ids, frame_idx]

        env_ids = self.all_env_ids
        if self.cfg["env"]["kinematic"]:  # kinematic-only
            self._set_humanoid_state(
                env_ids=env_ids,
                root_pos=goal_root_pos,
                root_rot=goal_root_rot,
                dof_pos=goal_dof_pos,
                root_vel=goal_root_vel,
                root_ang_vel=goal_root_ang_vel,
                dof_vel=goal_dof_vel,
                rigid_body_pos=global_pos,
                rigid_body_rot=global_rot,
                rigid_body_vel=global_vel,
                rigid_body_ang_vel=global_ang_vel,
            )
        elif self._display_reference:  # agent & kinematic
            rs_idx = self._ref_actor_idx
            re_idx = rs_idx + self.humanoid_num_bodies
            ds_idx = self._ref_dof_start_idx
            de_idx = ds_idx + self.humanoid_num_dof 
            self._else_root_states[env_ids, self._ref_actor_idx, 0:3] = goal_root_pos
            self._else_root_states[env_ids, self._ref_actor_idx, 3:7] = goal_root_rot
            self._else_root_states[env_ids, self._ref_actor_idx, 7:10] = goal_root_vel
            self._else_root_states[env_ids, self._ref_actor_idx, 10:13] = goal_root_ang_vel
            self._else_rigid_body_pos[env_ids, rs_idx: re_idx] = global_pos
            self._else_rigid_body_rot[env_ids, rs_idx: re_idx] = global_rot
            self._else_rigid_body_vel[env_ids, rs_idx: re_idx] = global_vel
            self._else_rigid_body_ang_vel[env_ids, rs_idx: re_idx] = global_ang_vel

            self._else_dof_pos[
                env_ids, ds_idx: de_idx
            ] = goal_dof_pos
            self._else_dof_vel[
                env_ids, ds_idx: de_idx
            ] = goal_dof_vel

        if not self.reset_call_flag:
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(self.ref_actor_indices),
                len(self.ref_actor_indices),
            )

            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._dof_state),
                gymtorch.unwrap_tensor(self.ref_dof_actor_indices),
                len(self.ref_dof_actor_indices),
            )

    def eval_tracking_metric(self,):
        assert not self._enable_early_termination
        if not hasattr(self, "sim_rigid_body_pos"):
            self.sim_rigid_body_pos = torch.zeros_like(self._ref_rigid_body_pos_buf)
            # initialize first one
            self.sim_rigid_body_pos[self.all_env_ids, 0] = self._ref_rigid_body_pos_buf[self.all_env_ids, 0]
        # update
        self.sim_rigid_body_pos[self.all_env_ids, self.progress_buf] = self._humanoid_rigid_body_pos

        # last
        if self.progress_buf[0] == self.max_episode_length - 1:
            sim_rigid_body_pos = self.sim_rigid_body_pos[:, :self.max_episode_length]
            goal_rigid_body_pos = self._ref_rigid_body_pos_buf[:, :self.max_episode_length]
            sim_rigid_body_pos = sim_rigid_body_pos.detach().cpu().numpy()
            goal_rigid_body_pos = goal_rigid_body_pos.detach().cpu().numpy()
            self.metric = compute_metrics_lite(sim_rigid_body_pos, goal_rigid_body_pos)
            delattr(self, "sim_rigid_body_pos")
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_ref_diff_observations(
    ref_rigid_body_pos,
    ref_rigid_body_rot,
    ref_rigid_body_vel,
    ref_rigid_body_ang_vel,
    sim_rigid_body_pos,
    sim_rigid_body_rot,
    sim_rigid_body_vel,
    sim_rigid_body_ang_vel,
    sim_root_rot,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    N, J, _ = ref_rigid_body_pos.shape
    # heading rots
    sim_heading_rot = calc_heading_quat_inv(sim_root_rot)
    sim_heading_rot_expand = sim_heading_rot[:, None].repeat(1, J, 1)
    sim_heading_rot_flat = sim_heading_rot_expand.reshape(N * J, -1)

    # rigid body pos
    ref_rigid_body_pos_flat = ref_rigid_body_pos.reshape(N * J, -1)
    sim_rigid_body_pos_flat = sim_rigid_body_pos.reshape(N * J, -1)
    ref_rigid_body_pos_obs = my_quat_rotate(
        sim_heading_rot_flat, ref_rigid_body_pos_flat - sim_rigid_body_pos_flat
    ).view(N, -1)

    # rigid body rot
    ref_rigid_body_rot_flat = ref_rigid_body_rot.reshape(N * J, -1)
    sim_rigid_body_rot_flat = sim_rigid_body_rot.reshape(N * J, -1)
    ref_rigid_body_rot_obs = quat_mul(
        sim_heading_rot_flat, quat_mul(ref_rigid_body_rot_flat, quat_conjugate(sim_rigid_body_rot_flat))
    )
    ref_rigid_body_rot_obs = quat_to_rotation_6d(ref_rigid_body_rot_obs).view(N, -1)

    # rigid body vel
    ref_rigid_body_vel_flat = ref_rigid_body_vel.reshape(N * J, -1)
    sim_rigid_body_vel_flat = sim_rigid_body_vel.reshape(N * J, -1)
    ref_rigid_body_vel_obs = my_quat_rotate(
        sim_heading_rot_flat, ref_rigid_body_vel_flat - sim_rigid_body_vel_flat
    ).view(N, -1)

    # rigid body ang vel
    ref_rigid_body_ang_vel_obs = (ref_rigid_body_ang_vel - sim_rigid_body_ang_vel).view(N, -1)

    obs = torch.cat(
        (
            ref_rigid_body_pos_obs,
            ref_rigid_body_rot_obs,
            ref_rigid_body_vel_obs,
            ref_rigid_body_ang_vel_obs,
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
# inspired by reference - deepmimic (Peng, 2018)
def compute_deepmimic_reward(
    curr_dof_pos,
    goal_dof_pos,
    curr_dof_vel,
    goal_dof_vel,
    curr_ee_pos,
    goal_ee_pos,
    curr_root_pos,
    goal_root_pos,
    multiplication,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    """
    obs = {
        root_states (13),
        dof_pos (28),
        dof_vel (28),
        key_pos (12)
    }
    """
    # compute errors
    dof_pos_error = (curr_dof_pos - goal_dof_pos).pow(2).mean(dim=-1)

    dof_vel_error = (curr_dof_vel - goal_dof_vel).pow(2).mean(dim=-1)

    ee_pos_error = (curr_ee_pos - goal_ee_pos).pow(2).sum(dim=-1).mean(dim=-1)

    root_pos_error = (curr_root_pos - goal_root_pos).pow(2).sum(dim=-1)

    # compute reward
    body_rot_reward = torch.exp(-2 * dof_pos_error)
    body_ang_vel_reward = torch.exp(-0.1 * dof_vel_error)
    ee_pos_reward = torch.exp(-40 * ee_pos_error)
    root_pos_reward = torch.exp(-10 * root_pos_error)

    # deepmimic version
    if multiplication:
        reward = body_rot_reward * body_ang_vel_reward * ee_pos_reward * root_pos_reward
    else:
        reward = 0.65 * body_rot_reward + 0.1 * body_ang_vel_reward + 0.15 * ee_pos_reward + 0.1 * root_pos_reward
    # print(body_rot_reward[0].item(), body_ang_vel_reward[0].item(), ee_pos_reward[0].item(), root_pos_reward[0].item())

    return reward


@torch.jit.script
# inspired by reference - PHC (Luo et al., 2023)
def compute_phc_reward(
    curr_rigid_body_pos,
    goal_rigid_body_pos,
    curr_rigid_body_rot,
    goal_rigid_body_rot,
    curr_rigid_body_vel,
    goal_rigid_body_vel,
    curr_rigid_body_ang_vel,
    goal_rigid_body_ang_vel,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    # compute errors
    N, J, _ = curr_rigid_body_rot.shape

    body_pos_error = (curr_rigid_body_pos - goal_rigid_body_pos).pow(2).sum(dim=(-1, -2))

    curr_rigid_body_rot = curr_rigid_body_rot.reshape(-1, 4)
    goal_rigid_body_rot = goal_rigid_body_rot.reshape(-1, 4)
    body_rot_error = quat_diff_rad(curr_rigid_body_rot, goal_rigid_body_rot).view(N, J).pow(2).mean(dim=-1)

    body_vel_error = (curr_rigid_body_vel - goal_rigid_body_vel).pow(2).sum(dim=-1).mean(dim=-1)

    body_ang_vel_error = (curr_rigid_body_ang_vel - goal_rigid_body_ang_vel).pow(2).sum(dim=-1).mean(dim=-1)

    # compute reward
    body_pos_reward = torch.exp(-100 * body_pos_error)
    body_rot_reward = torch.exp(-10 * body_rot_error)
    body_vel_reward = torch.exp(-0.1 * body_vel_error)
    body_ang_vel_reward = torch.exp(-0.1 * body_ang_vel_error)

    # deepmimic version
    reward = 0.5 * body_pos_reward + 0.3 * body_rot_reward + 0.1 * body_vel_reward + 0.1 * body_ang_vel_reward
    # print(body_pos_reward[0].item(), body_rot_reward[0].item(), body_vel_reward[0].item(), body_ang_vel_reward[0].item())

    return reward

@torch.jit.script
def compute_energy_reward(
    actions,
    dof_vel,
    coef
):
    # type: (Tensor, Tensor, float) -> Tensor
    mult = (actions * dof_vel).pow(2).sum(dim=-1)
    energy_rew = torch.exp(-coef * mult)
    return energy_rew


@torch.jit.script
def compute_imitation_reset_mean(
    reset_buf,
    progress_buf,
    curr_rigid_body_pos,
    goal_rigid_body_pos,
    contact_body_ids,
    max_episode_length,
    enable_early_termination,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool) -> Tuple[Tensor, Tensor]
    # inspired by "Perpetual Humanoid Control for Real-time Simulated Avatars" (2023)
    terminated = torch.zeros_like(reset_buf)

    # rigid_body_pos -> B, K, 3
    if enable_early_termination:
        pos_dist = (curr_rigid_body_pos - goal_rigid_body_pos).pow(2).sum(dim=-1).sqrt()
        pos_dist[:, contact_body_ids] = 0
        mean_pos_dist = pos_dist.sum(dim=-1) / (pos_dist.shape[-1] - contact_body_ids.shape[0])
        has_deviated = mean_pos_dist > 0.5
        terminated = torch.where(has_deviated, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

@torch.jit.script
def compute_imitation_reset_root_dist(
    reset_buf,
    progress_buf,
    curr_root_pos,
    goal_root_pos,
    max_episode_length,
    enable_early_termination,
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, bool) -> Tuple[Tensor, Tensor]
    # inspired by "Perpetual Humanoid Control for Real-time Simulated Avatars" (2023)
    terminated = torch.zeros_like(reset_buf)

    # rigid_body_pos -> B, K, 3
    if enable_early_termination:
        root_dist = (curr_root_pos - goal_root_pos).pow(2).sum(dim=-1).sqrt()
        has_deviated = root_dist > 1.0
        terminated = torch.where(has_deviated, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

@torch.jit.script
def compute_imitation_reset_max(
    reset_buf,
    progress_buf,
    curr_rigid_body_pos,
    goal_rigid_body_pos,
    contact_body_ids,
    max_episode_length,
    enable_early_termination,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool) -> Tuple[Tensor, Tensor]
    # inspired by "Perpetual Humanoid Control for Real-time Simulated Avatars" (2023)
    terminated = torch.zeros_like(reset_buf)

    # rigid_body_pos -> B, K, 3
    if enable_early_termination:
        pos_dist = (curr_rigid_body_pos - goal_rigid_body_pos).pow(2).sum(dim=-1).sqrt()
        pos_dist[:, contact_body_ids] = 0
        max_pos_dist = torch.max(pos_dist, dim=-1).values
        has_deviated = max_pos_dist > 0.5
        terminated = torch.where(has_deviated, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated


@torch.jit.script
def compute_ncp_reward(
    curr_dof_pos,
    goal_dof_pos,
    curr_dof_vel,
    goal_dof_vel,
    curr_ee_pos,
    goal_ee_pos,
    curr_root_pos,
    goal_root_pos,
    curr_root_rot,
    goal_root_rot,
    curr_root_vel,
    goal_root_vel,
    curr_root_ang_vel,
    goal_root_ang_vel,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    # compute errors
    # dof
    dof_pos_error = (curr_dof_pos - goal_dof_pos).abs().sum(dim=-1)
    dof_vel_error = (curr_dof_vel - goal_dof_vel).abs().sum(dim=-1)

    # end-effector
    # curr_local_ee_pos = compute_local_key_body_pos(curr_root_pos, curr_root_rot, curr_ee_pos)
    # goal_local_ee_pos = compute_local_key_body_pos(goal_root_pos, goal_root_rot, goal_ee_pos)
    curr_local_ee_pos = curr_ee_pos - curr_root_pos[:, None]
    goal_local_ee_pos = goal_ee_pos - goal_root_pos[:, None]
    ee_pos_error = (curr_local_ee_pos - goal_local_ee_pos).norm(2, dim=-1).sum(dim=-1)

    # root pos
    root_pos_error = (curr_root_pos - goal_root_pos).pow(2).sum(dim=-1)
    # root rot
    root_rot_error = quat_diff_rad(curr_root_rot, goal_root_rot).pow(2)
    # root vel
    root_vel_error = (curr_root_vel - goal_root_vel).pow(2).sum(dim=-1)
    # root ang vel
    root_ang_vel_error = (curr_root_ang_vel - goal_root_ang_vel).pow(2).sum(dim=-1)

    # compute reward
    joint_pos_reward = torch.exp(-2 * dof_pos_error)
    joint_vel_reward = torch.exp(-0.1 * dof_vel_error)
    key_pos_reward = torch.exp(-10 * ee_pos_error)
    root_pos_reward = torch.exp(-20 * (root_pos_error + 0.5 * root_rot_error))
    root_vel_reward = torch.exp(-2 * (root_vel_error + 0.1 * root_ang_vel_error))

    reward = 0.3 * joint_pos_reward + 0.1 * joint_vel_reward + 0.3 * key_pos_reward + 0.2 * root_pos_reward + 0.1 * root_vel_reward
    # print(joint_pos_reward[0].item(), joint_vel_reward[0].item(), key_pos_reward[0].item(), root_pos_reward[0].item(), root_vel_reward[0].item())

    return reward

# @torch.jit.script
# def compute_local_key_body_pos(root_pos, root_rot, key_body_pos):
#     # type: (Tensor, Tensor, Tensor) -> Tensor

#     heading_rot = calc_heading_quat_inv(root_rot)

#     root_pos_expand = root_pos.unsqueeze(-2)
#     centered_key_body_pos = key_body_pos - root_pos_expand

#     heading_rot_expand = heading_rot.unsqueeze(-2)
#     heading_rot_expand = heading_rot_expand.repeat((1, centered_key_body_pos.shape[1], 1))
#     flat_key_pos = centered_key_body_pos.view(
#         centered_key_body_pos.shape[0] * centered_key_body_pos.shape[1],
#         centered_key_body_pos.shape[2],
#     )
#     flat_heading_rot = heading_rot_expand.view(
#         heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
#         heading_rot_expand.shape[2],
#     )
#     local_key_body_pos = my_quat_rotate(flat_heading_rot, flat_key_pos)
#     local_key_body_pos = local_key_body_pos.view(
#         centered_key_body_pos.shape[0],
#         centered_key_body_pos.shape[1],
#         centered_key_body_pos.shape[2],
#     )
#     return local_key_body_pos