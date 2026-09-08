# Phase 1: posterior-detach MSE

Scope: A0-compatible defaults and A1 only. Set
`train.params.config.latent_align.direction=post_to_prior` for A1.
Default `bidirectional` retains A0. No variance head or A2/A3 implementation.

Only the alignment target is detached: `(post_loc.detach() - prior_loc)`.
The decoder's latent is not detached. No parameters or checkpoint keys are
added. Weight and schedule continue to use `vae_kl_loss_coef` and
`vae_kl_schedule`; inactive variance/weight placeholder fields were removed.

Metrics now flow through encoder, actor, model, agent aggregation, and
TensorBoard. Entropy/perplexity are batch-local, computed separately per RVQ
layer, excluding dropout indices (-1). Fully inactive layers report zero
entropy/perplexity and zero active fraction; active-fraction tags distinguish
dropout from single-code collapse. Epoch metrics average minibatch statistics,
not pooled code counts across the entire epoch. Logging tensors are detached.

## Validation

```bash
conda run -n hlr_env python -m unittest \
  tests.test_latent_alignment tests.test_latent_alignment_rvq -v
```

Eight tests pass without importing or starting Isaac Gym. Tests include actual
RVQ with training dropout/EMA, the existing decoder and action head, legacy
encoder forward from the baseline Git commit, exact output/buffer equality,
alignment/action/commitment/full-loss gradients, and detached metric propagation.

The full-size offline audit additionally strictly loads the original epoch
7500 checkpoint and uses the real fixed batch described in ALIGNMENT_PHASE0.md.
A0/A1 raw alignment loss is exactly `0.10698716342449188`; actions, commitment,
and RVQ state also match exactly under identical Python/Torch RNG states.

Alignment-only gradient L1 sums:

| Parameters | A0 | A1 |
| --- | ---: | ---: |
| posterior MLP | 1371.3021161556244 | 0 |
| posterior loc head | 151.30635976791382 | 0 |
| prior MLP | 803.2602894306183 | 803.2602894306183 |
| prior loc head | 50.07440638542175 | 50.07440638542175 |

A1 action-only posterior MLP/head gradients: `749.172884 / 69.326450`.
A1 commitment-only posterior MLP/head gradients: `501.403457 / 59.073365`.
Both give zero prior gradients. The full-loss test (weights 10/0.1/1) gives
nonzero gradients to both posterior groups and both prior groups. These are
L1 sums of parameter gradients, not L2 norms. All values are finite.

Offline reproduction (plotting dependencies are isolated in /tmp):

```bash
PYTHONPATH=/tmp/hlr-alignment-plot MPLCONFIGDIR=/tmp/hlr-alignment-mpl \
  /home/yuhuan.zhang/miniconda3/envs/hlr_env/bin/python scripts/audit_latent_alignment.py \
  --checkpoint isaacgymenvs/runs/imitation_hybrid_b6offnv3/nn/imitation_hybrid_b6offnv3_7500.pth \
  --batch artifacts/alignment_phase01/a0_check/validation_batch.pt \
  --baseline-run isaacgymenvs/runs/imitation_hybrid_b6offnv3 \
  --output artifacts/alignment_phase01/offline
```

## Training gate

After committing, run `scripts/check_alignment_training.py` for 3 epochs on
GPU 1 with the original HybridDistill configuration and only the A1 direction
override. The wrapper preserves max_epochs and all training settings, saves a
batch/checkpoint, checks finite training metrics, and flushes TensorBoard.
Then `scripts/launch_alignment_a1.py` launches a fresh independent seed-42 run,
requiring both offline and three-epoch smoke evidence. It records command,
commit, GPU, PID, initialization, and log location in `launch.json`.

The formal run starts from scratch with the same expert as A0, not from the
partially trained A0 or diagnostic checkpoint. GPU 0's A0 is left running.
Additional seeds and distributed training are not part of this launch.
No A2 work is authorized by these scripts.
