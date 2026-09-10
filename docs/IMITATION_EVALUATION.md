# Hybrid imitation evaluation and LaFAN data diagnosis

Evaluation snapshot: the original `imitation_hybrid_b6offnv3` run through epoch
20000, inspected on 2026-09-09. Full-matrix results and the final interpretation
are recorded in `artifacts/imitation_evaluation/report.md` when the matrix is
complete. Diagnostic-prefixed files are development checks, not final scores.

## Latest corrected-data results

The current A0–A3 snapshot uses the 20,000-epoch checkpoints and the corrected
`LAFAN_ALL_corrected_v2/test` split (10 motions). The prior report is in
`artifacts/imitation_evaluation/corrected_a0_a3_20000_report.md`; the paired
test-set prior/posterior tracking-success table is in
`artifacts/imitation_evaluation/test_tracking_success_official_a0_a3.md`.

The tracking table contains 30 trajectories per weight (three starts for each
of 10 motions) and applies the same 0.5 m maximum-body-error threshold to prior
and posterior. Prior samples RVQ codes using only the current-state prior;
posterior uses the goal-conditioned representation, so their success rates
measure different capabilities. The official checkpoint's earlier full-data
result is a reference only: it uses 60 motions and 383,590 reference
transitions, while this test split has 69,578.

## Metric meanings

`mpjpe_mm` is root-relative mean body-position error; `gmpjpe_mm` preserves
world translation. The evaluator field names are opposite to the convention
used by some papers. Velocity and acceleration errors are finite-difference
errors in mm/s and mm/s². `tracking_success_fraction` counts trajectories that
never exceed 0.5 m maximum body error. `fall_step` and `no_low_root_fraction`
use the diagnostic condition pelvis/root z < 0.5 m. `coverage_percent` counts
reference AMP transitions reached by at least one valid prior sample;
`filtering_percent` is the fraction rejected by nearest-feature distance 10;
`matching_distance` is the average normalized nearest-feature distance of
accepted samples. `reward_sum` is accumulated simulator reward and is a
monitoring value for the expert-distillation runs.

## Protocol

`scripts/evaluate_imitation_suite.py` uses the repository's model loading,
normalization, action scaling, simulator and reference interpolation. It replaces
only the player's evaluation loop. It does not update model parameters.

- All 60 eligible motions (50 subject1–4, 10 subject5); the repository excludes
  17 obstacle motions. Each motion has three evenly spaced valid segment starts.
- Each segment contains 149 simulated control steps at dt=0.0332 seconds. The
  artificially copied initial reference frame is excluded from error metrics.
- Posterior evaluations use all eight RVQ layers, deterministic decoder actions,
  and seeds 42/43/44. Seeds do not imply different motion coverage: segment starts
  are fixed, and deterministic posterior results may be identical.
- MPJPE removes root translation; G-MPJPE uses world positions. Both use mm.
  Velocity/acceleration errors use dt/dt² and are in mm/s and mm/s², unlike the
  native unscaled frame-difference output. Metrics include failed trajectories.
- Tracking success means maximum body error stays <=0.5 m for the whole segment.
  This is an explicitly defined diagnostic, not a claim of reproducing every
  paper evaluation detail. A separate pelvis-height statistic includes deliberate
  ground motions and must not be interpreted as a universal fall-success metric.
- Prior evaluation uses the first RVQ codebook, seed 42, 800000 state-transition
  samples and all eligible motions as its matching reference bank. Matching is
  performed in chunks, preserving the native normalized-feature squared-distance
  definition and threshold `<10`. Distance averages weight every valid sample
  equally. No valid match is represented as JSON null, rather than NaN.
- No early reset within a segment. Failure timing is recorded independently from
  full-horizon errors. Neither failures nor large errors are dropped from reports.

The original training configuration recursively loaded `LAFAN_ALL`, including
both subdirectories. Its subject5 scores are therefore **not held-out test
generalization**. The corrected 100-epoch validation run explicitly loads only
`LAFAN_ALL_corrected_v2/train`; its subject5 evaluation is separately identifiable.

## Data correction

The original Ubisoft BVH uses centimeters and bones oriented along local +X.
The old pipeline used a meter-scale factor of 1 and a source rest pose that did
not match those local frames. For example, the old walk1_subject1 data has median
root height 17.55 m and median root speed 59.95 m/s. The corrected values are
0.857 m and 0.599 m/s, respectively.

`scripts/retarget_raw_lafan.py` consumes the existing source NPZ data, validates
the original centimeter/+X convention, constructs its actual rest frames, and
retargets with a root translation scale of 0.01. Target body dimensions, the
existing elbow/knee hinge projection, and the 0.05 m floor clearance are retained.
It does not clamp arbitrary invalid poses merely to improve an evaluation score.

The full 77-motion/496672-frame audit, with input/output hashes and joint-limit
statistics, is `artifacts/imitation_evaluation/data_audit.json`. The audit uses
XML joint limits and an excess tolerance of 0.05 rad; residual violations in some
retargeted poses remain visible rather than being hidden. GPU first-step behavior
and official policy controls provide separate checks of physical compatibility.

The new dataset is immutable for this evaluation:

```
isaacgymenvs/tasks/amp/poselib/data/AMP/LAFAN_ALL_corrected_v2_2026-Sep-09
assets/amp/motions/LAFAN_ALL_corrected_v2 -> that directory
```

The original `LAFAN_ALL` link, BVH, NPZ and existing training checkpoints are
preserved. Existing training processes have already loaded their old data; a
data correction cannot retroactively repair their training history.

Two rotation utility fixes accompany the conversion: mutually exclusive
matrix-to-quaternion sign branches for tied magnitudes, and tuple-compatible
identity-quaternion shapes with `PYTORCH_JIT=0`. Regression tests cover all 24
proper signed-permutation rotations, source rest-frame geometry, and chunked
matching against the direct distance formula. Tests pass with JIT on and off.

## Commands

Use `hlr_env`; do not use base Python. All simulator commands require:

```bash
export LD_LIBRARY_PATH=/home/yuhuan.zhang/miniconda3/envs/hlr_env/lib:$LD_LIBRARY_PATH
export PYTORCH_JIT=0
```

From the repository root, the completed evaluation launcher can be reproduced
with the commands saved alongside every result (`*.command.json`). The launcher
skips existing results, so use a fresh artifact location when changing a protocol:

```bash
python scripts/run_imitation_evaluation.py --phase full --gpu 4
python scripts/run_imitation_evaluation.py --phase validation --gpu 5
python scripts/summarize_imitation_evaluation.py
python -m unittest tests.test_imitation_evaluation -v
```

Generate another independent corrected dataset with a **new** output path:

```bash
python scripts/retarget_raw_lafan.py \
  --source isaacgymenvs/tasks/amp/poselib/data/LAFAN/lafan1_npz_2026-Sep-07 \
  --output /path/to/new/LAFAN_ALL_corrected
```

For subsequent formal training, explicitly select the corrected training split
and a new experiment. This is a fresh training command, not a resumption of the
corrupted-data run:

```bash
cd isaacgymenvs
CUDA_VISIBLE_DEVICES=5 python train.py \
  headless=True task=LafanImitation train=imitation/HybridDistill \
  experiment=imitation_hybrid_corrected \
  expert=pretrained_weights/imitation/imitation_expert \
  motion_dataset=LAFAN_ALL_corrected_v2/train
```

The bounded validation actually executed during diagnosis is separate from a
formal run. Its command, run directory, curves and checkpoint are retained under
`artifacts/imitation_evaluation/corrected_training_check*`.
