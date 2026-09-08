# Copyright (c) 2018-2023, NVIDIA Corporation
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
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import copy
import time
from datetime import datetime

import numpy as np
import torch
from rl_games.algos_torch import torch_ext, model_builder
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common, schedulers, vecenv
from tensorboardX import SummaryWriter
from torch import nn, optim
from omegaconf import ListConfig, OmegaConf
from pathlib import Path

import isaacgymenvs.learning.common_agent as common_agent
import isaacgymenvs.learning.replay_buffer as replay_buffer
from isaacgymenvs.utils.torch_jit_utils import to_torch


class cVAEAgent(common_agent.CommonAgent):
    def __init__(self, base_name, params):
        # Basic Settings
        self.distill = params.get("load_expert", False) # expert for distillation
        self.use_latent_regularize = params["config"].get("latent_regularize", False) # whether using latent regulairzation loss or not
        if self.use_latent_regularize:
            self.latent_reg_loss_coef = params["config"]["latent_reg_loss_coef"] # latent regularization loss weight

        super().__init__(base_name, params)

        # Loss weights
        # Actor loss (RL only)
        self.actor_coef = params["config"].get("actor_coef", 1) # actor loss weight for RL

        # KL divergence loss (cVAE, ours)
        self.vae_kl_loss_coef = params["config"].get("vae_kl_loss_coef", None) # KL divergence loss weight of cVAE / Ours
        vae_kl_schedule = params["config"].get("vae_kl_schedule", False) # whether using 10 times upscale throughout training or not
        self.vae_kl_schedule = False if self.vae_kl_loss_coef == None else vae_kl_schedule # VQ model doesn't need this loss / schedule
        if self.vae_kl_schedule:
            self.init_vae_kl_loss_coef = self.vae_kl_loss_coef
            self.kl_schedule_end_epochs = int(self.max_epochs * 0.125) # upscale throughout a portion of training

        # Commitment loss (VQ, ours) - similar to KL's explanation
        self.vae_commit_loss_coef = params["config"].get("vae_commit_loss_coef", None)
        vae_commit_schedule = params["config"].get("vae_commit_schedule", False) # 3 times upscale
        self.vae_commit_schedule = False if self.vae_commit_loss_coef == None else vae_commit_schedule # (Aborted)
        if self.vae_commit_schedule:
            self.init_vae_commit_loss_coef = self.vae_commit_loss_coef
            self.commit_schedule_end_epochs = int(self.max_epochs * 0.125)

        # Running mean standard deviation for goal observation (state of next frame)
        # Not that RMS for proprioceptive observation is already included in the self.model
        if self._normalize_goal_input:
            self._goal_input_mean_std = RunningMeanStd(self._goal_observation_space.shape).to(self.ppo_device)
        
        # Prioritized sampling : sample more for motion clips that have small returns
        if self._prioritized_sampling:
            self._normalized_motion_returns = torch.zeros_like(self.vec_env.env._motion_lib._motion_lengths)
        
        # Setup for Online Distillation
        if self.distill:
            expert_path = params["load_expert_path"]
            self.load_expert_network(expert_path, params["config"]["device"])
            self.expert_loss_coef = params["config"].get("expert_loss_coef", 0.0) # to balance other losses while online distillation
        
            # We only update actor part for Online Distillation
            actor_lr = params["config"].get("actor_learning_rate", float(self.last_lr)) # we use higher lr for online distillation
            param_name_filter = ['encoder', 'decoder', 'quantizer', 'mu', 'sigma']
            def _name_filter(name, qs):
                for q in qs:
                    if q in name:
                        print("\033[34m \033[106m" + "%s is registered to actor optimizer" % name + "\033[0m")
                        return True
                print("\033[31m \033[102m" + "%s is not registered to actor optimizer" % name + "\033[0m")
                return False
            actor_parameters = [param for name, param in self.model.named_parameters() if _name_filter(name, param_name_filter)]
            # override optimizer
            self.optimizer = optim.Adam(
                actor_parameters,
                actor_lr,
                eps=1e-08,
                weight_decay=self.weight_decay,
            )
        return

    # Load expert policy for Online Distillation
    def load_expert_network(self, path, device):
        assert os.path.isdir(path)
        params = OmegaConf.load(os.path.join(path, "config.yaml"))
        net_config = dict(OmegaConf.load(os.path.join(path, "net_config.yaml")))
        for cfg_key in net_config.keys():
            if type(net_config[cfg_key]) == ListConfig:
                net_config[cfg_key] = tuple(net_config[cfg_key])

        builder = model_builder.ModelBuilder()
        network = builder.load(params["train"]["params"])
        ckpt_list = list(Path(os.path.join(path, "nn")).rglob("*.pth"))
        try:
            pretrained_ckpt = sorted(ckpt_list, key=lambda x: int(str(x).split("_")[-1][:-4]))[-1]
        except:
            pretrained_ckpt = sorted(ckpt_list)[0]
        print("\033[30m \033[106m" + str(pretrained_ckpt) + "\033[0m")
        weights = torch_ext.load_checkpoint(pretrained_ckpt)
        self.expert = network.build(net_config).to(device)
        self.expert.load_state_dict(weights["model"])
        self.expert.eval()
        expert_goal_input_mean_std = RunningMeanStd(self._goal_observation_space.shape).to(self.ppo_device)
        expert_goal_input_mean_std.load_state_dict(weights["goal_input_mean_std"])
        self._expert_goal_input_mean_std = expert_goal_input_mean_std
        self._expert_goal_input_mean_std.eval()
        return

    # call once at start point for each epoch
    def update_epoch(self):
        super().update_epoch()
        # vae kl schedule : 10 times larger
        if self.vae_kl_schedule:
            mult = 1 + min(self.epoch_num / self.kl_schedule_end_epochs, 1) * 9
            self.vae_kl_loss_coef = self.init_vae_kl_loss_coef * mult
        # (Aborted) vae commit schedule : 3 times larger
        if self.vae_commit_schedule:
            mult = 1 + min(self.epoch_num / self.commit_schedule_end_epochs, 1) * 2
            self.vae_commit_loss_coef = self.init_vae_commit_loss_coef * mult
        return self.epoch_num

    # allocate additional buffers to experience buffer for trajecotry collection
    def init_tensors(self):
        super().init_tensors()
        self._build_goal_buffers()
        if self.distill:
            self._build_expert_mu_buffers()
        if self.use_latent_regularize:
            self._build_prev_post_mu_buffers()
        return

    # setting eval mode for trajectory collection
    def set_eval(self):
        super().set_eval()
        if self._normalize_goal_input:
            self._goal_input_mean_std.eval()
        return

    # setting training mode for parameter update
    def set_train(self):
        super().set_train()
        if self._normalize_goal_input:
            self._goal_input_mean_std.train()
        return

    # helper function to save checkpoint
    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_goal_input:
            state["goal_input_mean_std"] = self._goal_input_mean_std.state_dict()
        if self._prioritized_sampling:
            state["normalized_motion_returns"] = self._normalized_motion_returns
        return state

    # helper function to load checkpoint
    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self._normalize_goal_input:
            self._goal_input_mean_std.load_state_dict(weights["goal_input_mean_std"])
        if self._prioritized_sampling:
            if self._normalized_motion_returns.shape[0] == weights["normalized_motion_returns"].shape[0]:
                self._normalized_motion_returns = weights["normalized_motion_returns"]
            else:
                # it means dataset has just changed - start new training
                weights['epoch'] = 0
                weights['optimizer'] = None
                weights['frame'] = 0
        return

    # helper function to load checkpoint
    def set_full_state_weights(self, weights):
        self.set_weights(weights)
        self.epoch_num = weights['epoch'] # frames as well?
        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])

        if weights['optimizer'] != None:
            self.optimizer.load_state_dict(weights['optimizer'])
        self.frame = weights.get('frame', 0)
        self.last_mean_rewards = weights.get('last_mean_rewards', -100500)

        env_state = weights.get('env_state', None)

        if self.vec_env is not None:
            self.vec_env.set_env_state(env_state)

    # Infer the action during trajectory collection
    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs["obs"])
        processed_goal_obs = self._preproc_goal_obs(obs["goal_obs"])
        self.model.eval()
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": processed_obs,
            "goal_obs": processed_goal_obs,
            "rnn_states": self.rnn_states,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value: # (Aborted) not actually used
                states = obs["states"]
                input_dict = {
                    "is_train": False,
                    "states": states,
                }
                value = self.get_central_value(input_dict)
                res_dict["values"] = value
        return res_dict
    
    # extract expert action (deterministic) for online distillation
    def get_expert_mu_values(self, obs):
        processed_obs = self._preproc_obs(obs["obs"])
        # note that goal state should be processed outside from forward function of model
        # equivalent to self._preproc_goal_obs() but RMS is set for expert
        processed_goal_obs = self._expert_goal_input_mean_std(obs["goal_obs"])
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": processed_obs,
            "goal_obs": processed_goal_obs,
            "rnn_states": self.rnn_states,
        }

        with torch.no_grad():
            res_dict = self.expert(input_dict)
        return res_dict["mus"]

    # eval value from the critic
    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs = obs_dict["obs"]
        goal_obs = obs_dict["goal_obs"]

        processed_obs = self._preproc_obs(obs)
        processed_goal_obs = self._preproc_goal_obs(goal_obs)

        if self.normalize_input:
            processed_obs = self.model.norm_obs(processed_obs)

        value = self.model.a2c_network.eval_critic(processed_obs, processed_goal_obs)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    # trajectory collection for each epoch
    def play_steps(self):
        self.set_eval()

        epinfos = []
        update_list = self.update_list


        # rollouts
        for n in range(self.horizon_length):
            self.obs, done_env_ids = self._env_reset_done()
            self.experience_buffer.update_data("obses", n, self.obs["obs"])
            self.experience_buffer.update_data("goal_obses", n, self.obs["goal_obs"])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data("states", n, self.obs["states"])
            
            # get expert action and store it to experience buffer
            if self.distill:
                expert_mus = self.get_expert_mu_values(self.obs)
                self.experience_buffer.update_data("expert_mus", n, expert_mus)

            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data("rewards", n, shaped_rewards)
            self.experience_buffer.update_data("next_obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones)

            # for latent regularize loss
            if self.use_latent_regularize:
                self.experience_buffer.update_data("prev_post_mu", n, self.prev_post_mu_buffer)
                self.experience_buffer.update_data("prev_dones", n, self.prev_dones_buffer)
                # update prev posterior mu
                self.prev_post_mu_buffer[:] = res_dict["post_mu"]
                # update prev_dones
                self.prev_dones_buffer[:] = self.dones

                # for hybrid - quantcond
                if self._collect_prior_mu:
                    self.experience_buffer.update_data("prev_prior_mu", n, self.prev_prior_mu_buffer)
                    self.prev_prior_mu_buffer[:] = res_dict["prior_mu"]
            terminated = infos["terminate"].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= 1.0 - terminated
            self.experience_buffer.update_data("next_values", n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[:: self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            # update motion clip sampling weights for environment
            if hasattr(self.vec_env.env, "update_motion_weights") and self._prioritized_sampling:
                self._normalized_motion_returns[:] = self.vec_env.env.update_motion_weights(self._normalized_motion_returns, self.current_rewards)
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_next_values = self.experience_buffer.tensor_dict["next_values"]

        mb_rewards = self.experience_buffer.tensor_dict["rewards"]

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict["returns"] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size

        return batch_dict

    # helper function to process batch from the trajectory collector to make as dataset for training
    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict["goal_obs"] = batch_dict["goal_obses"]
        if self.distill:
            self.dataset.values_dict["expert_mu"] = batch_dict["expert_mus"]
        if self.use_latent_regularize:
            self.dataset.values_dict["prev_post_mu"] = batch_dict["prev_post_mu"]
            if self._collect_prior_mu:
                self.dataset.values_dict["prev_prior_mu"] = batch_dict["prev_prior_mu"]
            self.dataset.values_dict["prev_dones"] = batch_dict["prev_dones"]
        return

    # trainer
    def train_epoch(self):
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get("rnn_masks", None)

        self.set_train()

        self.curr_frames = batch_dict.pop("played_frames")
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                if self.schedule_type == "legacy" and not self.distill:
                    self.last_lr, self.entropy_coef = self.scheduler.update(
                        self.last_lr,
                        self.entropy_coef,
                        self.epoch_num,
                        0,
                        curr_train_info["kl"].item(),
                    )
                    self.update_lr(self.last_lr)

                if train_info is None:
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

            if not self.distill:
                av_kls = torch_ext.mean_list(train_info["kl"])

                if self.schedule_type == "standard":
                    self.last_lr, self.entropy_coef = self.scheduler.update(
                        self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item()
                    )
                    self.update_lr(self.last_lr)

        if not self.distill:
            if self.schedule_type == "standard_epoch":
                self.last_lr, self.entropy_coef = self.scheduler.update(
                    self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item()
                )
                self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        train_info["play_time"] = play_time
        train_info["update_time"] = update_time
        train_info["total_time"] = total_time
        self._record_train_batch_info(batch_dict, train_info)

        return train_info

    def calc_gradients(self, input_dict):
        self.train_result = dict()
        if self.distill:
            self.calc_gradients_non_rl(input_dict)
        else:
            self.calc_gradients_rl(input_dict)
        return

    def calc_gradients_non_rl(self, input_dict):
        self.set_train()
        obs_batch = input_dict["obs"]
        obs_batch = self._preproc_obs(obs_batch)
        goal_obs_batch = input_dict["goal_obs"]
        goal_obs_batch = self._preproc_goal_obs(goal_obs_batch)
        expert_mu_batch = input_dict["expert_mu"]
        actions_batch = input_dict["actions"] # to calculate negative log likelihood, actually not using

        # when using latent regularization loss
        if self.use_latent_regularize:
            prev_post_mu = input_dict["prev_post_mu"]
            if self._collect_prior_mu:
                prev_prior_mu = input_dict["prev_prior_mu"]
                prev_gap = (prev_post_mu - prev_prior_mu).detach()
            prev_dones = input_dict["prev_dones"]

        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch, # to calculate negative log likelihood, actually not using
            "obs": obs_batch,
            "goal_obs": goal_obs_batch,
        }

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            mu = res_dict["mus"]
            expert_loss = (mu - expert_mu_batch).pow(2).sum(dim=-1).mean()
            loss = self.expert_loss_coef * expert_loss
            # for cVAE, ours
            if "vae_kl_loss" in res_dict:
                vae_kl_loss = res_dict["vae_kl_loss"]
                loss = loss + self.vae_kl_loss_coef * vae_kl_loss
            # for VQ, ours
            if "vae_commit_loss" in res_dict:
                vae_commit_loss = res_dict["vae_commit_loss"]
                loss = loss + self.vae_commit_loss_coef * vae_commit_loss
            # for latent regularization loss
            if self.use_latent_regularize:
                post_mu = res_dict["post_mu"]
                if self._collect_prior_mu:
                    prior_mu = res_dict["prior_mu"]
                    # mask out discontinuous trjectory
                    current_gap = post_mu - prior_mu.detch()
                    gap_diff = (current_gap - prev_gap).pow(2).sum(dim=-1)
                    prior_diff = (prior_mu - prev_prior_mu).pow(2).sum(dim=-1)
                    latent_reg_loss = (gap_diff + prior_diff) * (1 - prev_dones.float())
                    latent_reg_loss = latent_reg_loss.mean()
                    loss = loss + self.latent_reg_loss_coef * latent_reg_loss
                else:
                    # mask out discontinuous trjectory
                    latent_reg_loss = (post_mu - prev_post_mu).pow(2).sum(dim=-1) * (1 - prev_dones.float())
                    latent_reg_loss = latent_reg_loss.mean()
                    loss = loss + self.latent_reg_loss_coef * latent_reg_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        # TODO: Refactor this ugliest code of the year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        # update training stats for logging
        train_result = {
            "expert_loss": expert_loss
        }
        # for cVAE, ours
        if "vae_kl_loss" in res_dict:
            train_result["vae_kl_loss_non_rl"] = vae_kl_loss
        # for VQ, ours
        if "vae_commit_loss" in res_dict:
            train_result["vae_commit_loss_non_rl"] = vae_commit_loss
        self._record_latent_align_result(res_dict, train_result)
        # for latent regularization loss
        if self.use_latent_regularize:
            train_result["latent_reg_loss_non_rl"] = latent_reg_loss
        self.train_result.update(train_result)
        return

    def calc_gradients_rl(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]
        obs_batch = self._preproc_obs(obs_batch)
        goal_obs_batch = input_dict["goal_obs"]
        goal_obs_batch = self._preproc_goal_obs(goal_obs_batch)

        # when using latent regularization while training pure RL (from scratch)
        # we don't use this since we train expert with 0-VAE (no prior net, variational inference)
        # when using latent regularization loss
        if self.use_latent_regularize:
            prev_post_mu = input_dict["prev_post_mu"]
            if self._collect_prior_mu:
                prev_prior_mu = input_dict["prev_prior_mu"]
                prev_gap = (prev_post_mu - prev_prior_mu).detach()
            prev_dones = input_dict["prev_dones"]

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
            "goal_obs": goal_obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info["actor_loss"]

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info["critic_loss"]

            b_loss = self.bound_loss(mu)

            losses, sum_mask = torch_ext.apply_masks(
                [
                    a_loss.unsqueeze(1),
                    c_loss,
                    entropy.unsqueeze(1),
                    b_loss.unsqueeze(1),
                ],
                rnn_masks,
            )
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = (
                self.actor_coef * a_loss
                + self.critic_coef * c_loss
                - self.entropy_coef * entropy
                + self.bounds_loss_coef * b_loss
            )
            # for cVAE, ours
            if "vae_kl_loss" in res_dict:
                vae_kl_loss = res_dict["vae_kl_loss"]
                loss = loss + self.vae_kl_loss_coef * vae_kl_loss
            # for VQ, ours
            if "vae_commit_loss" in res_dict:
                vae_commit_loss = res_dict["vae_commit_loss"]
                loss = loss + self.vae_commit_loss_coef * vae_commit_loss
            # for latent regularization loss
            if self.use_latent_regularize:
                post_mu = res_dict["post_mu"]
                if self._collect_prior_mu:
                    prior_mu = res_dict["prior_mu"]
                    # mask out discontinuous trjectory
                    current_gap = post_mu - prior_mu.detch()
                    gap_diff = (current_gap - prev_gap).pow(2).sum(dim=-1)
                    prior_diff = (prior_mu - prev_prior_mu).pow(2).sum(dim=-1)
                    latent_reg_loss = (gap_diff + prior_diff) * (1 - prev_dones.float())
                    latent_reg_loss = latent_reg_loss.mean()
                    loss = loss + self.latent_reg_loss_coef * latent_reg_loss
                else:
                    # mask out discontinuous trjectory
                    latent_reg_loss = (post_mu - prev_post_mu).pow(2).sum(dim=-1) * (1 - prev_dones.float())
                    latent_reg_loss = latent_reg_loss.mean()
                    loss = loss + self.latent_reg_loss_coef * latent_reg_loss

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        # TODO: Refactor this ugliest code of the year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        rl_info = {
            "entropy": entropy,
            "kl": kl_dist,
            "last_lr": self.last_lr,
            "lr_mul": lr_mul,
            "b_loss": b_loss,
        }
        if "vae_kl_loss" in res_dict:
            rl_info["vae_kl_loss"] = vae_kl_loss
        if "vae_commit_loss" in res_dict:
            rl_info["vae_commit_loss"] = vae_commit_loss
        self._record_latent_align_result(res_dict, rl_info)
        if self.use_latent_regularize:
            rl_info["latent_reg_loss"] = latent_reg_loss
        self.train_result.update(rl_info)
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        return

    # helper function to load training configs
    def _load_config_params(self, config):
        super()._load_config_params(config)
        self._goal_observation_space = self.env_info["goal_observation_space"]
        self._no_goal = config.get("no_goal", False)
        self._normalize_goal_input = config.get("normalize_goal_input", True)
        self._prioritized_sampling = config.get("prioritized_sampling", False)

        self._latent_dim = config.get("latent_dim", 64) if not self._no_goal else 0
        self._enc_type = config.get("enc_type", "continuous")
        if self._enc_type in ["continuous", "hybrid"]:
            self._enc_scale = config.get("enc_scale", 0.3)
            self._continuous_enc_style = config.get("continuous_enc_style", "standard")
            self._latent_align_config = config.get("latent_align", None)
        if self._enc_type in ["discrete", "hybrid"]:
            self._code_num = config.get("code_num", 512)
            self._quant_type = config.get("quant_type", "basic")
            self._num_quants = config.get("num_quants", 4)
        # transfer learning (i.e. seed from dataset A, reuse this for training dataset B)
        self._transfer_type = config.get("transfer_type", None)
        self._transfer = True if self._transfer_type != None else False
        if self._transfer:
            assert self._transfer_type in ["full", "decoder", "decoder_last"]
        return

    # helper function to build network configs
    def _build_net_config(self):
        config = super()._build_net_config()
        config["goal_input_shape"] = self._goal_observation_space.shape
        config["latent_shape"] = (self._latent_dim,)
        config["no_goal"] = self._no_goal
        print("Latent Dimension : ", self._latent_dim)
        config["enc_type"] = self._enc_type
        if self._enc_type in ["continuous", "hybrid"]:
            config["enc_scale"] = self._enc_scale
            config["continuous_enc_style"] = self._continuous_enc_style
            config["latent_align"] = self._latent_align_config
        if self._enc_type in ["discrete", "hybrid"]:
            config["code_num"] = self._code_num
            config["quant_type"] = self._quant_type
            config["num_quants"] = self._num_quants
        return config

    def _init_train(self):
        super()._init_train()
        return

    # building experience buffer for goal observations
    def _build_goal_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict["goal_obses"] = torch.zeros(
            batch_shape + self._goal_observation_space.shape, device=self.ppo_device
        )
        self.tensor_list += ["goal_obses"]
        return

    # building experience buffer for expert actions (deterministic)
    def _build_expert_mu_buffers(self):
        self.experience_buffer.tensor_dict["expert_mus"] = torch.zeros_like(self.experience_buffer.tensor_dict["mus"])
        self.tensor_list += ["expert_mus"]
        return

    # (only for latent regularization loss) building experience buffer for previous latent from policy
    def _build_prev_post_mu_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict["prev_post_mu"] = torch.zeros(
            batch_shape + (self._latent_dim,), device=self.ppo_device
        )
        self.experience_buffer.tensor_dict["prev_dones"] = torch.zeros_like(self.experience_buffer.tensor_dict["dones"])
        self.tensor_list += ["prev_post_mu", "prev_dones"]
        if self._enc_type == "hybrid" and self._continuous_enc_style == "quant_cond":
            self.experience_buffer.tensor_dict["prev_prior_mu"] = torch.zeros(
                batch_shape + (self._latent_dim,), device=self.ppo_device
            )
            self.tensor_list += ["prev_prior_mu"]
            self._collect_prior_mu = True
        else:
            self._collect_prior_mu = False

        # for trajectory collection
        self.prev_post_mu_buffer = torch.zeros(self.vec_env.env.num_envs, self._latent_dim, device=self.ppo_device)

        # prev_dones acts as mask (if prev latent made episode done, than we need to mask it since the new episode begins)
        self.prev_dones_buffer = torch.ones(self.vec_env.env.num_envs, dtype=torch.uint8, device=self.ppo_device)

        # additional
        if self._collect_prior_mu:
            self.prev_prior_mu_buffer = torch.zeros(self.vec_env.env.num_envs, self._latent_dim, device=self.ppo_device)
        return

    # RMS process for goal observation
    def _preproc_goal_obs(self, goal_obs):
        if self._normalize_goal_input:
            goal_obs = self._goal_input_mean_std(goal_obs)
        return goal_obs

    def _record_train_batch_info(self, batch_dict, train_info):
        return

    def _record_latent_align_result(self, res_dict, train_result):
        for key, value in res_dict.items():
            if key.startswith("latent_align_"):
                train_result[key] = value.detach()
        return

    def _log_latent_align_info(self, train_info, frame):
        for key, values in train_info.items():
            if key.startswith("latent_align_"):
                tag = "latent_align/" + key[len("latent_align_"):]
                self.writer.add_scalar(tag, torch_ext.mean_list(values).item(), frame)
        return

    # log info for wandb
    def _log_train_info(self, train_info, frame):
        if not self.distill: # for RL
            super()._log_train_info(train_info, frame)
            if "vae_kl_loss" in train_info:
                self.writer.add_scalar(
                    "losses/vae_kl_loss",
                    torch_ext.mean_list(train_info["vae_kl_loss"]).item(),
                    frame,
                )
            if "vae_commit_loss" in train_info:
                self.writer.add_scalar(
                    "losses/vae_commit_loss",
                    torch_ext.mean_list(train_info["vae_commit_loss"]).item(),
                    frame,
                )
            if "latent_reg_loss" in train_info:
                self.writer.add_scalar(
                    "losses/latent_reg_loss",
                    torch_ext.mean_list(train_info["latent_reg_loss"]).item(),
                    frame,
                )
            self._log_latent_align_info(train_info, frame)

        else: # for distill
            self.writer.add_scalar(
                "losses/expert_loss",
                torch_ext.mean_list(train_info["expert_loss"]).item(),
                frame,
            )
            if "vae_kl_loss_non_rl" in train_info:
                self.writer.add_scalar(
                    "losses/vae_kl_loss_non_rl",
                    torch_ext.mean_list(train_info["vae_kl_loss_non_rl"]).item(),
                    frame,
                )
            if "vae_commit_loss_non_rl" in train_info:
                self.writer.add_scalar(
                    "losses/vae_commit_loss_non_rl",
                    torch_ext.mean_list(train_info["vae_commit_loss_non_rl"]).item(),
                    frame,
                )
            if "latent_reg_loss_non_rl" in train_info:
                self.writer.add_scalar(
                    "losses/latent_reg_loss_non_rl",
                    torch_ext.mean_list(train_info["latent_reg_loss_non_rl"]).item(),
                    frame,
                )
            self._log_latent_align_info(train_info, frame)
        return
