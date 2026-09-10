"""Launch isolated evaluation processes and retain exact commands and logs."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'isaacgymenvs'
OUT = ROOT / 'artifacts/imitation_evaluation'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', default='4')
    p.add_argument('--phase', choices=['diagnostic', 'full', 'validation'], default='diagnostic')
    p.add_argument('--dataset', choices=['original', 'corrected_v2'])
    p.add_argument('--models', help='Comma-separated subset for an additional GPU worker')
    args = p.parse_args()
    checkpoints = {
        'local20000': ('imitation/HybridDistill', 'runs/imitation_hybrid_b6offnv3/nn/imitation_hybrid_b6offnv3_20000.pth'),
        'local10000': ('imitation/HybridDistill', 'runs/imitation_hybrid_b6offnv3/nn/imitation_hybrid_b6offnv3_10000.pth'),
        'local2500': ('imitation/HybridDistill', 'runs/imitation_hybrid_b6offnv3/nn/imitation_hybrid_b6offnv3_2500.pth'),
        'official_hybrid': ('imitation/HybridDistill', 'pretrained_weights/imitation/imitation_hybrid/nn/imitation_hybrid_weight.pth'),
        'expert': ('imitation/ExpertPPO', 'pretrained_weights/imitation/imitation_expert/nn/imitation_expert_weight.pth')}
    datasets = {'original': 'LAFAN_ALL', 'corrected_v2': 'LAFAN_ALL_corrected_v2'}
    if args.phase == 'validation':
        checkpoints = {'corrected100': ('imitation/HybridDistill',
                       str(OUT / 'corrected_training_check/nn/validation100.pth'))}
        datasets = {'corrected_v2': 'LAFAN_ALL_corrected_v2'}
    if args.dataset:
        datasets = {args.dataset: datasets[args.dataset]}
    if args.models:
        checkpoints = {name: checkpoints[name] for name in args.models.split(',')}
    jobs = []
    if args.phase == 'diagnostic':
        for data in datasets:
            for model in checkpoints:
                jobs.append((data, model, 'posterior', 42, 1, False))
            jobs.append((data, 'local10000', 'hold', 42, 1, False))
    else:
        for data in datasets:
            for model in checkpoints:
                for seed in (42, 43, 44):
                    jobs.append((data, model, 'posterior', seed, 3, False))
            prior_models = ('corrected100',) if args.phase == 'validation' else ('local20000', 'local10000', 'official_hybrid')
            for model in prior_models:
                if model in checkpoints:
                    jobs.append((data, model, 'prior', 42, 3, True))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTORCH_JIT='0',
               LD_LIBRARY_PATH=str(Path(sys.executable).resolve().parents[1] / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
               OMP_NUM_THREADS='2', HYDRA_FULL_ERROR='1')
    OUT.mkdir(parents=True, exist_ok=True)
    if not (BASE / 'tasks/amp/poselib/data/AMP/LAFAN_ALL_corrected_v2_2026-Sep-09/conversion_manifest.json').exists():
        raise RuntimeError('Corrected dataset conversion is not yet complete')
    for data, model, mode, seed, starts, matching in jobs:
        name = '%s_v3_%s_%s_%s_s%d' % (args.phase, data, model, mode, seed)
        output = OUT / (name + '.json')
        if output.exists():
            print('EXISTS', output, flush=True)
            continue
        config, checkpoint = checkpoints[model]
        cmd = [sys.executable, str(ROOT / 'scripts/evaluate_imitation_suite.py'), '--output', str(output),
               '--mode', mode, '--starts', str(starts)]
        if matching:
            cmd += ['--matching']
        cmd += ['test=True', 'headless=True', 'num_envs=60', 'task=LafanImitation', 'train='+config,
                'checkpoint='+checkpoint, 'motion_dataset='+datasets[data], 'seed='+str(seed),
                'enable_et=False', 'eval_metric=False', 'num_inference_quants='+str(1 if mode == 'prior' else 0),
                'prior_rollout='+str(mode == 'prior'), 'motion_matching='+str(matching)]
        (OUT / (name + '.command.json')).write_text(json.dumps(cmd, indent=2))
        print('START', name, flush=True)
        with (OUT / (name + '.log')).open('w') as log:
            proc = subprocess.run(cmd, cwd=str(BASE), env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            print('FAILED', name, proc.returncode, flush=True)
            raise SystemExit(proc.returncode)
        print('DONE', name, json.loads(output.read_text())['summary'], flush=True)


if __name__ == '__main__':
    main()
