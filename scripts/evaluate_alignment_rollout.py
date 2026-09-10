"""Bounded deterministic rollout on fixed motion IDs; evaluation only."""
import argparse
import json
import runpy
import sys
import types
from pathlib import Path

import isaacgym
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaacgymenvs.learning.cvae_player import cVAEPlayerContinuous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("prior", "posterior"), required=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--zero-action", action="store_true",
                        help="Diagnostic control: hold the current PD pose instead of policy actions")
    args, overrides = parser.parse_known_args()
    args.output = args.output.resolve()

    def run(player):
        env = player.env
        def fixed_motion(task, env_ids):
            ids = env_ids % task._motion_lib.num_motions()
            task._curr_motion_ids[env_ids] = ids
            times = torch.arange(task._ref_buf_length, device=env_ids.device) * task.dt
            return ids[:, None].expand(-1, task._ref_buf_length), times[None].expand(len(env_ids), -1)
        env._sample_motion_ids_and_times = types.MethodType(fixed_motion, env)
        player.prior_rollout = args.mode == "prior"
        obs = player.env_reset(env)
        player.get_batch_size(obs["obs"], 1)
        if player.is_rnn:
            player.init_rnn()
        obs, _ = player._env_reset_done()
        def state_diagnostics():
            reference = env._ref_rigid_body_pos_buf[env.all_env_ids, env.progress_buf]
            return dict(progress=env.progress_buf.tolist(),
                        body_error=(env._humanoid_rigid_body_pos - reference).norm(dim=-1).tolist(),
                        root_position=env._humanoid_root_states[:, :3].tolist(),
                        reference_root=reference[:, 0].tolist(),
                        dof_limit_excess=torch.maximum(
                            env.dof_limits_lower[:env.humanoid_num_dof] - env._humanoid_dof_pos,
                            env._humanoid_dof_pos - env.dof_limits_upper[:env.humanoid_num_dof]
                        ).clamp_min(0).tolist(),
                        dof_position=env._humanoid_dof_pos.tolist())
        initial_state = state_diagnostics()
        first_step = None
        count = env.num_envs
        alive = torch.ones(count, device=env.device, dtype=torch.bool)
        life = torch.zeros(count, device=env.device)
        rewards = torch.zeros_like(life)
        position = torch.zeros_like(life)
        failure = torch.zeros_like(alive)
        for step in range(min(args.steps, env.max_episode_length - 1)):
            action = player.get_action(obs, is_deterministic=True)
            if args.zero_action:
                action = torch.zeros_like(action)
            if not torch.isfinite(action).all():
                raise RuntimeError("Nonfinite rollout action")
            obs, reward, done, info = player.env_step(env, action)
            if step == 0:
                first_step = state_diagnostics()
                first_step["action"] = action.tolist()
            reference = env._ref_rigid_body_pos_buf[env.all_env_ids, env.progress_buf]
            error = (env._humanoid_rigid_body_pos - reference).norm(dim=-1).mean(-1)
            position += error * alive
            rewards += reward.reshape(-1).to(env.device) * alive
            life += alive
            failure |= alive & info["terminate"].to(env.device).bool()
            alive &= ~done.reshape(-1).to(env.device).bool()
            if not alive.any():
                break
        result = dict(mode=args.mode, checkpoint=player.checkpoint_fn,
                      motion_ids=list(range(count)), start_time_seconds=0, dt=env.dt,
                      steps=life.tolist(), survival_seconds=(life * env.dt).tolist(),
                      reward_sum=rewards.tolist(), position_error=(position / life.clamp_min(1)).tolist(),
                      failure=failure.tolist(), evaluation_seed=42)
        result.update(initial_state=initial_state, first_step=first_step,
                      reset_style=env.imitation_reset_style, zero_action=args.zero_action)
        assert all(torch.isfinite(x).all() for x in (life, rewards, position))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("FIXED_ROLLOUT", json.dumps(result), flush=True)

    cVAEPlayerContinuous.run = run
    sys.argv = [str(ROOT / "isaacgymenvs/train.py")] + [x for x in overrides if x != "--"]
    runpy.run_path(str(ROOT / "isaacgymenvs/train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
