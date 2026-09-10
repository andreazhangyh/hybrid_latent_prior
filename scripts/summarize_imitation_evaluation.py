"""Validate and aggregate the completed v3 evaluation matrix."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/imitation_evaluation'


def main():
    table = []
    for data in ('original', 'corrected_v2'):
        models = ['local2500', 'local10000', 'local20000', 'official_hybrid', 'expert']
        if data == 'corrected_v2':
            models.append('corrected100')
        for model in models:
            phase = 'validation' if model == 'corrected100' else 'full'
            files = [OUT / ('%s_v3_%s_%s_posterior_s%d.json' % (phase, data, model, seed)) for seed in (42,43,44)]
            loaded = [json.loads(p.read_text()) for p in files]
            assert all(x['motions'] == 60 and len(x['episodes']) == 180 for x in loaded)
            assert all(x['num_inference_quants'] == 0 or model == 'expert' for x in loaded)
            episodes = [r for x in loaded for r in x['episodes']]
            for split in ('all', 'train', 'test'):
                rows = [r for r in episodes if split == 'all' or r['split'] == split]
                record = dict(data=data, model=model, mode='posterior', split=split, episodes=len(rows))
                for key in ('mpjpe_mm', 'gmpjpe_mm', 'velocity_error_mm_s', 'acceleration_error_mm_s2',
                            'first_step_max_body_error_m', 'initial_max_limit_excess_rad', 'reward_sum'):
                    record[key] = sum(r[key] for r in rows)/len(rows)
                    assert math.isfinite(record[key])
                record['tracking_success_percent'] = 100*sum(r['tracking_failure_step'] > 149 for r in rows)/len(rows)
                record['no_low_root_percent'] = 100*sum(r['fall_step'] > 149 for r in rows)/len(rows)
                table.append(record)
        prior_models = ['local10000', 'local20000', 'official_hybrid']
        if data == 'corrected_v2':
            prior_models.append('corrected100')
        for model in prior_models:
            phase = 'validation' if model == 'corrected100' else 'full'
            path = OUT / ('%s_v3_%s_%s_prior_s42.json' % (phase, data, model))
            result = json.loads(path.read_text())
            assert result['summary']['samples'] == 800000
            assert result['num_inference_quants'] == 1
            row = dict(data=data, model=model, mode='prior', split='all')
            for key in ('samples','valid_samples','reference_transitions','coverage_percent','filtering_percent','matching_distance'):
                row[key] = result['summary'][key]
            table.append(row)
    (OUT/'summary.json').write_text(json.dumps(table,indent=2,allow_nan=False))
    fields = list(dict.fromkeys(k for r in table for k in r))
    with (OUT/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(table)
    print('Validated all 40 evaluations; wrote summary.json and summary.csv')
    for r in table:
        if r['split']=='all': print(r)


if __name__ == '__main__':
    main()
