# Phase 2-4: Gaussian latent alignment

Implemented and verified on 2026-09-08, based on commit
`76eaca2cc1c2318df30172faa4d64f3475253e74`. The smoke manifest records source
hashes and the exact tracked patch used for verification. Existing user setup
changes and the live A0/A1 training processes were preserved.

## Configuration and graph

| Preset (`train=imitation/...`) | Loss | Direction | Prior std | Initial coefficient |
| --- | --- | --- | --- | ---: |
| HybridDistillA0 | MSE | bidirectional | no head | 0.1 |
| HybridDistillA1 | MSE | posterior detached | no head | 0.1 |
| HybridDistillA2 | KL(q\|\|p) | bidirectional | learned | 0.018 |
| HybridDistillA3 | KL(q\|\|p) | posterior detached | learned | 0.018 |

`HybridDistill` remains A0 by default. The existing KL schedule still ramps
the coefficient by 10x over the first 25000 epochs of a 200000-epoch budget.
The Gaussian presets use the plan's initial scale calibration
`2 * 0.3**2 * 0.1 = 0.018`. Other optimizer, sampling, environment, decoder,
RVQ, and training settings are unchanged, verified against the saved A0 config.

Phase 2 adds only a prior logstd linear head from the existing prior features
to the 64-dimensional latent. Its weights start at zero and bias at log(0.3).
Initialization preserves the CPU RNG stream, and the parent builder skips
generic initialization of this head. Existing parameter initialization and
RVQ buffers therefore remain bitwise identical across all four variants.

Posterior std is fixed at 0.3. Prior logstd is clamped to [-5, 1], then
`prior_std = exp(logstd) + 1e-6`, as specified by the plan. Consequently the
effective initial prior std is approximately 0.300001; the exact mathematical
equal-variance test supplies identical std tensors without this epsilon.
Head projection and KL run in FP32 with autocast disabled. The standalone
KL helper preserves FP64 for the formula test.

Phase 3 uses `q_loc=post_loc`; Phase 4 uses `q_loc=post_loc.detach()` only in
alignment. The deterministic action latent, quantization, commitment, and
prior-mean rollout paths are unchanged. No posterior variance head, rollout
variance sampling, residual-radius loss, InfoNCE, or hidden-feature KL exists.

Checkpoint metadata records the alignment configuration. A legacy/MSE model
state can load into A2/A3 by initializing precisely the two new head tensors;
partial head absence and other mismatches still fail strict loading. Gaussian
checkpoints cannot load into A0/A1, even with strict=False. A changed Gaussian
posterior std or clamp configuration is also rejected. Migration covers model
weights, not automatic conversion of an old optimizer's parameter groups.

## T0: numerical and gradient gates

All 17 tests passed, including CUDA autocast, before any smoke test started:

```bash
CUDA_VISIBLE_DEVICES=2 PYTORCH_JIT=0 \
ALIGNMENT_REPORT=artifacts/alignment_phase234/numerical_gradients.json \
  /home/yuhuan.zhang/miniconda3/envs/hlr_env/bin/python -m unittest \
  tests.test_latent_alignment tests.test_latent_alignment_rvq \
  tests.test_gaussian_alignment -v
```

The FP64 identity `KL = MSE / (2*sigma**2)` has maximum absolute error
`1.1368683772161603e-13`, below 1e-6. Tests also verify zero KL at equal means,
increasing KL with mean error, the stationary minimum at
`prior_var=posterior_var+squared_error`, and finite values/gradients at extreme
logstd clamp inputs. CUDA autocast produces an FP32 KL tensor.

Real RVQ gradient L2 norms on the same fixed test batch, alignment-only:

| Variant | Posterior MLP | Posterior loc | Prior MLP | Prior loc | Prior logstd |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 33.960603 | 18.265208 | 2.670581 | 2.201709 | absent |
| A1 | 0 | 0 | 2.670581 | 2.201709 | absent |
| A2 | 188.668765 | 101.472701 | 14.836464 | 12.231634 | 3.293505 |
| A3 | 0 | 0 | 14.836464 | 12.231634 | 3.293505 |

Gradients are cleared before every independent backward. Action and commitment
each update posterior but not prior; the full loss updates posterior in A1/A3.
All eight RVQ layers are active in the initialization/invariance matrix test,
and existing dropout tests remain in the suite. Full-size networks for all
variants strictly load the historical A0 epoch-7500 checkpoint.

Hydra testing caught both an incorrect relative inheritance path and package
placement. Explicit absolute base paths with `@_here_` fixed these issues;
the four fully composed presets then passed validation before simulation.

## T1: actual training smoke tests

Each variant completed 9 epochs / 108 optimizer updates, with 8192 environments,
horizon 16, minibatch 65536, 6 mini-epochs, seed 42, and the original expert.
GPU 2/3/4/5 hosted independent A0/A1/A2/A3 tests, all of which exited normally.
The wrapper stopped at the requested epoch without changing max_epochs or
the coefficient schedule. No pilot or new formal A2/A3 run was started.

```bash
/home/yuhuan.zhang/miniconda3/envs/hlr_env/bin/python -u \
  scripts/smoke_gaussian_alignment.py \
  --output artifacts/alignment_phase234/smoke --gpus 2 3 4 5

/home/yuhuan.zhang/miniconda3/envs/hlr_env/bin/python \
  scripts/verify_alignment_smoke.py artifacts/alignment_phase234/smoke
```

The launcher refuses to reuse an existing manifest to prevent duplicate runs.
Each epoch's first real minibatch records raw and weighted gradient L2 norms
via autograd.grad on the existing forward graph, without an extra forward,
RVQ update, random draw, or change to optimizer gradients. All nine observed
gradient matrices match the expected routes, including the learned head.
The bounded training wrapper also supports optional milestone rollout reviews;
these load saved checkpoints in a separate test process and record fixed-motion
diagnostics without changing optimizer state or training hyperparameters.

| Variant | Updates | Max weighted alignment/action gradient ratio | Last mean prior std |
| --- | ---: | ---: | ---: |
| A0 | 108 | 0.911026 | n/a |
| A1 | 108 | 0.310737 | n/a |
| A2 | 108 | 0.787167 | 0.319681 |
| A3 | 108 | 0.265973 | 0.350559 |

The ratio combines posterior and prior group L2 norms and applies each loss's
actual coefficient. Stop guards check nonfinite values, >=50% prior logstd
clamping, weighted gradient ratio >=100, or three consecutive epochs with
latent norms >10x the first epoch. None fired. Both clamp fractions remained
zero in A2/A3. All TensorBoard scalars are finite, and the trained checkpoints
strictly reload with identical outputs. Learned logstd weights are nonzero.
These are operational checks, not evidence of relative task performance.

Artifacts under `artifacts/alignment_phase234/`:

- `numerical_gradients.json`: equal-variance error and all four loss matrices.
- `smoke/manifest.json`, `worktree.patch`: commands, GPUs, PIDs and tested source.
- `smoke/results.json`, `verification.json`: completion and post-run checks.
- `smoke/A*/epochs.jsonl`, `gradients.jsonl`, `train.log`: real training records.
- `smoke/A*/validation_batch.pt`, `check_checkpoint.pth`: reusable inputs/weights.
- The run directories recorded in verification.json contain TensorBoard events
  and saved Hydra/network configurations, including alignment type and direction.

## Formal launch and rollout diagnosis

On 2026-09-08 at 18:24 server local time, formal A2 and A3 training was
launched at the user's request before resolving rollout diagnostics:

| Variant | GPU | PID | Run directory |
| --- | --- | --- | --- |
| A2 | 2 | 1680561 | `isaacgymenvs/runs/alignment_a2_seed42_pygbilbo` |
| A3 | 3 | 1680563 | `isaacgymenvs/runs/alignment_a3_seed42_3i6ek7c7` |

These are independent single-GPU runs with seed 42 and unchanged training
settings. The wrapper records alignment gradient audits and finite-value
guards. Automatic milestone rollout reviews are disabled pending diagnosis.
Launch commands, PIDs, tracked diffs, and a source archive are stored in
`artifacts/alignment_formal/a{2,3}_seed42/`. Source changes were not committed;
the base commit alone does not identify the launched code. Both processes
were confirmed live; A2 reached epoch 32 (384 updates) and A3 epoch 31
(372 updates). The observed gradient routes match A2/A3 and loss/std are finite.
These observations do not establish long-run convergence or motion quality.

The evaluation script now records initial and first-step state, joint-limit
violations, and an optional `--zero-action` diagnostic. Fixed-motion sampling
also updates the task's current motion IDs. This only affects evaluation.

With the A0 epoch-7500 checkpoint and fixed test motion IDs 0 and 1, both
posterior-policy and zero-action rollouts terminate after one step. Initial
joint-limit violations reach 2.194 and 2.292 radians. Even with zero action,
maximum first-step body errors reach 0.903 and 0.793 meters, triggering the
existing `error_max` threshold of 0.5 meters. Initial body-error tensors are
zero because reset explicitly copies reference body poses into those tensors;
this alone does not demonstrate physical consistency after simulation.

Evidence is in `artifacts/alignment_formal_checks/a0_diagnostic.json` and
`a0_zero_action.json`, with corresponding logs. This rules out policy output
as the sole cause and identifies incompatible initial joint angles as a
concrete issue. It does not yet establish which retargeting/conversion step
introduced those angles or the prevalence across the training dataset.
No motion data, simulator limits, termination thresholds, or training settings
were modified to mask this failure. Rollout quality remains unvalidated.
