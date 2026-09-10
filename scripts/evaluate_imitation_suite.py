"""Reproducible full-library evaluation, without changing training behavior.

All eligible motions are evaluated at evenly spaced start times. No auto-reset
within a trajectory; failures and full-horizon errors are reported separately.
Prior matching uses bounded matrix products and sample-weighted aggregation.
"""
import argparse
import fcntl
import hashlib
import json
import math
import runpy
import sys
import types
from pathlib import Path

import isaacgym
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaacgymenvs.learning.cvae_player import cVAEPlayerContinuous


def nearest_transition(query, bank, chunk=8192):
    best = torch.full((len(query),), float('inf'), device=query.device)
    indices = torch.zeros(len(query), dtype=torch.long, device=query.device)
    qnorm = query.square().sum(-1)
    for start in range(0, len(bank), chunk):
        ref = bank[start:start + chunk]
        distances = (qnorm[:, None] + ref.square().sum(-1)[None]
                     - 2 * query.matmul(ref.t())).clamp_min(0) / (query.shape[-1] // 2)
        values, ids = distances.min(-1)
        better = values < best
        indices = torch.where(better, ids + start, indices)
        best = torch.minimum(best, values)
    return best, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['posterior', 'prior', 'hold'], required=True)
    parser.add_argument('--starts', type=int, default=3)
    parser.add_argument('--prior-samples', type=int, default=800000)
    parser.add_argument('--matching', action='store_true')
    args, overrides = parser.parse_known_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Multiple launcher workers may reach the same job. Hold a process-owned
    # filesystem lock until completion, and reuse the completed result only.
    job_lock = args.output.with_suffix('.lock').open('a')
    fcntl.flock(job_lock.fileno(), fcntl.LOCK_EX)
    if args.output.exists():
        print('Completed by another worker:', args.output, flush=True)
        return

    def run(player):
        env = player.env
        lib = env._motion_lib
        n = lib.num_motions()
        if len(env.motion_file_name) != n:
            raise ValueError('Motion library filtered clips; explicit filename mapping is required')
        horizon = env.max_episode_length - 1
        ids_all = torch.arange(n, device=env.device).repeat(args.starts)
        fractions = torch.linspace(0, 1, args.starts, device=env.device).repeat_interleave(n)
        if args.mode == 'prior':
            repeats = math.ceil(args.prior_samples / (len(ids_all) * horizon))
            ids_all = ids_all.repeat(repeats)
            fractions = fractions.repeat(repeats)
        cursor = [0]
        def sample(task, env_ids):
            idx = torch.arange(len(env_ids), device=env.device) + cursor[0]
            idx = idx.clamp_max(len(ids_all) - 1)
            ids = ids_all[idx]
            starts = fractions[idx] * (lib._motion_lengths[ids] - task._ref_buf_length * task.dt).clamp_min(0)
            task._curr_motion_ids[env_ids] = ids
            times = starts[:, None] + torch.arange(task._ref_buf_length, device=env.device)[None] * task.dt
            return ids[:, None].expand_as(times), times
        env._sample_motion_ids_and_times = types.MethodType(sample, env)
        player.prior_rollout = args.mode == 'prior'
        # Build reference features only, then disable the unbounded native
        # B x reference x feature distance allocation during every env step.
        bank = counts = None
        if args.matching:
            if not hasattr(lib, 'curr_feats'):
                raise ValueError('Pass motion_matching=True to construct the reference bank')
            bank = torch.cat([lib.curr_feats, lib.hist_feats], -1)
            if not torch.isfinite(bank).all():
                raise ValueError('Nonfinite reference features')
            counts = torch.zeros(len(bank), dtype=torch.long, device=env.device)
        if args.matching:
            original_post = env.post_physics_step
            def post_without_dense_matching(task):
                # Keep motion_matching=True during reset: the native prior
                # visualization reset otherwise copies environment 0's motion
                # into all actors. Suppress only the dense matching operation.
                task.cfg['env']['motion_matching'] = False
                try:
                    return original_post()
                finally:
                    task.cfg['env']['motion_matching'] = True
            env.post_physics_step = types.MethodType(post_without_dense_matching, env)
        records, match_queries = [], []
        match_sum = 0.
        valid_samples = total_samples = 0
        def flush():
            nonlocal match_sum, valid_samples, total_samples
            if not match_queries:
                return
            query = torch.cat(match_queries)
            dist, idx = nearest_transition(query, bank)
            valid = torch.isfinite(dist) & (dist < 10)
            counts.scatter_add_(0, idx[valid], torch.ones_like(idx[valid]))
            match_sum += dist[valid].double().sum().item()
            valid_samples += int(valid.sum())
            total_samples += len(query)
            match_queries.clear()
        for start in range(0, len(ids_all), env.num_envs):
            cursor[0] = start
            size = min(env.num_envs, len(ids_all) - start)
            env.reset_idx(env.all_env_ids)
            obs = player.env_reset(env)
            player.get_batch_size(obs['obs'], 1)
            obs, _ = player._env_reset_done()
            if player.is_rnn:
                player.init_rnn()
            reference = env._ref_rigid_body_pos_buf[:, :env.max_episode_length].clone()
            predicted = [reference[:, 0].clone()]
            excess = torch.maximum(env.dof_limits_lower[:env.humanoid_num_dof] - env._humanoid_dof_pos,
                                   env._humanoid_dof_pos - env.dof_limits_upper[:env.humanoid_num_dof]).clamp_min(0)
            initial_excess = excess.max(-1).values
            failure_step = torch.full((env.num_envs,), horizon + 1, device=env.device, dtype=torch.long)
            fall_step = failure_step.clone()
            rewards = torch.zeros(env.num_envs, device=env.device)
            first_error = None
            hold_action = ((env._humanoid_dof_pos - env._pd_action_offset) / env._pd_action_scale).clone()
            used_steps = horizon
            for step in range(1, horizon + 1):
                action = hold_action if args.mode == 'hold' else player.get_action(obs, is_deterministic=True)
                if not torch.isfinite(action).all():
                    raise ValueError('Nonfinite action')
                obs, reward, done, info = player.env_step(env, action)
                pos = env._humanoid_rigid_body_pos.clone()
                if not torch.isfinite(pos).all():
                    raise ValueError('Nonfinite simulation')
                predicted.append(pos)
                error = (pos - reference[:, step]).norm(dim=-1).max(-1).values
                if first_error is None:
                    first_error = error
                failure_step = torch.where((error > .5) & (failure_step > horizon), step, failure_step)
                fall_step = torch.where((env._humanoid_root_states[:, 2] < .5) & (fall_step > horizon), step, fall_step)
                rewards += reward.reshape(-1).to(env.device)
                if bank is not None:
                    q = torch.cat([(env._curr_amp_obs_buf - lib.feats_mean) / lib.feats_std,
                                   (env._hist_amp_obs_buf[:, 0] - lib.feats_mean) / lib.feats_std], -1)[:size]
                    remaining = args.prior_samples - total_samples - sum(len(x) for x in match_queries)
                    match_queries.append(q[:max(0, remaining)].clone())
                    if sum(len(x) for x in match_queries) >= 256:
                        flush()
                    if total_samples + sum(len(x) for x in match_queries) >= args.prior_samples:
                        used_steps = step
                        break
            pred = torch.stack(predicted, 1)[:, 1:]
            ref = reference[:, 1:1 + used_steps]
            gerr = (pred - ref).norm(dim=-1).mean((1, 2)) * 1000
            lerr = ((pred - pred[:, :, :1]) - (ref - ref[:, :, :1])).norm(dim=-1).mean((1, 2)) * 1000
            vel = ((pred[:, 1:] - pred[:, :-1]) - (ref[:, 1:] - ref[:, :-1])).norm(dim=-1).mean((1, 2)) * 1000 / env.dt
            acc = ((pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]) -
                   (ref[:, 2:] - 2 * ref[:, 1:-1] + ref[:, :-2])).norm(dim=-1).mean((1, 2)) * 1000 / env.dt ** 2
            for i in range(size):
                motion_id = int(ids_all[start+i])
                motion_name = env.motion_file_name[motion_id]
                records.append(dict(motion_id=motion_id, motion_name=motion_name,
                    split='test' if motion_name.endswith('subject5') else 'train', start_fraction=float(fractions[start+i]),
                    steps=used_steps, mpjpe_mm=float(lerr[i]), gmpjpe_mm=float(gerr[i]),
                    velocity_error_mm_s=float(vel[i]) if used_steps > 1 else None,
                    acceleration_error_mm_s2=float(acc[i]) if used_steps > 2 else None,
                    first_step_max_body_error_m=float(first_error[i]), initial_max_limit_excess_rad=float(initial_excess[i]),
                    tracking_failure_step=int(failure_step[i]), fall_step=int(fall_step[i]),
                    reward_sum=float(rewards[i])))
            print('EVAL_PROGRESS', len(records), '/', len(ids_all), 'samples', total_samples, flush=True)
            if bank is not None and total_samples + sum(len(x) for x in match_queries) >= args.prior_samples:
                break
        flush()
        summary = {}
        for key in records[0]:
            if key in ('motion_id', 'motion_name', 'split', 'start_fraction'):
                continue
            values = [r[key] for r in records if r[key] is not None]
            summary[key] = sum(values) / len(values) if values else None
        summary['tracking_success_fraction'] = sum(r['tracking_failure_step'] > horizon for r in records) / len(records)
        summary['no_low_root_fraction'] = sum(r['fall_step'] > horizon for r in records) / len(records)
        if counts is not None:
            summary.update(samples=total_samples, valid_samples=valid_samples,
                coverage_percent=100 * int((counts > 0).sum()) / len(counts), reference_transitions=len(counts),
                filtering_percent=100 * (1 - valid_samples / total_samples),
                matching_distance=match_sum / valid_samples if valid_samples else None)
        result = dict(checkpoint=player.checkpoint_fn, mode=args.mode, seed=player.random_seed,
            dt=env.dt, motion_dataset=env.cfg['env']['motion_file'], motions=n,
            num_inference_quants=getattr(player, '_num_inference_quants', None),
            protocol='all eligible motions, evenly spaced starts, full horizon, initial frame excluded',
            command=sys.argv, summary=summary, episodes=records)
        ckpt = Path(next(x.split('=', 1)[1] for x in sys.argv if x.startswith('checkpoint='))).resolve()
        result.update(checkpoint_path=str(ckpt), checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                      evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      low_root_definition='pelvis z < 0.5m; includes intentional ground motions',
                      tracking_success_definition='all simulated frames: maximum body position error <= 0.5m')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print('EVAL_RESULT', json.dumps(summary, allow_nan=False), flush=True)

    cVAEPlayerContinuous.run = run
    sys.argv = [str(ROOT / 'isaacgymenvs/train.py')] + [x for x in overrides if x != '--']
    runpy.run_path(str(ROOT / 'isaacgymenvs/train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
