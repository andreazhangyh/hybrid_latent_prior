# Phase 0: quantcond baseline audit

Audited 2026-09-08, before A1 training. Original HEAD:
`2113e72461f254060be72b3b920c888b22afcf8c`.

## Existing run

The existing process is A0: PID 429739 on GPU 0, command below, seed 42,
run `isaacgymenvs/runs/imitation_hybrid_b6offnv3`. Its saved configuration uses
hybrid/quantcond, RVQ, latent dimension 64, 1024 codes, and 8 quantizers.
It predates the alignment configuration and uses bidirectional squared error.
The process was observed live during this audit and was not restarted.

```bash
cd isaacgymenvs
LD_LIBRARY_PATH=/home/yuhuan.zhang/miniconda3/envs/hlr_env/lib \
PYTORCH_JIT=0 HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 \
conda run -n hlr_env python train.py headless=True task=LafanImitation \
  train=imitation/HybridDistill experiment=imitation_hybrid \
  expert=pretrained_weights/imitation/imitation_expert
```

Checkpoint frozen for comparison: `nn/imitation_hybrid_b6offnv3_7500.pth`.
SHA256: `7318e7854f422cf396f10621ac3ce706c9c89d000fa118afc023489b554a74a7`.
The run does not contain a historical Git SHA record; the SHA above is the
checked repository baseline, not a claim that an unrecorded process snapshot
can be recovered. Local setup modifications are preserved and snapshotted.

## Computation graph

Let `u = post_loc_net(post_net(cat(obs, goal)))` and
`p = prior_loc_net(prior_net(obs))`. Quantcond computes
`r = RVQ(u - p.detach())` and `z = p.detach() + r`.
The decoder receives `z`. A0 alignment is `(z - p).square().sum(-1)`;
the model then averages over the batch. The RVQ straight-through path carries
alignment and action gradients into the posterior. Action and commitment
cannot update the prior because the residual input and reconstruction use
detached prior values. A0 alignment updates both prior MLP and prior loc head.

Distillation remains `10 * action + gamma * alignment + 1 * commitment`.
`gamma = 0.1 * (1 + 9 * min(epoch / 25000, 1))`, with max_epochs 200000.
The existing vae_kl fields remain authoritative. No optimizer, sampling,
decoder, RVQ, or training hyperparameter changes were made.

## Evidence and limitations

Local artifacts are under `artifacts/alignment_phase01/`:

- `a0_check/validation_batch.pt`: first 256 real model inputs from the first
  distillation minibatch of a fresh seed-42 A0 check on GPU 1. The stored goal
  inputs are already normalized at capture. This is a fixed diagnostic batch,
  not a held-out motion evaluation dataset.
- `a0_check/epochs.jsonl`: one completed epoch at the original 8192 environments,
  horizon 16, minibatch 65536, 6 mini-epochs, and max_epochs 200000.
- `offline/baseline_scalars.csv` and `baseline_losses.png`: exported historical
  A0 loss curves. The old process cannot emit newly added metrics; historical
  posterior/prior/residual norms and layer entropy curves are unavailable.
- `offline/audit.json`: checkpoint/batch/expert/data hashes, full-network
  strict checkpoint loading, exact legacy/default compatibility, finite-value
  assertions, gradient results, and fixed-batch per-layer statistics.
- `offline/a0_config.yaml`, `a0_net_config.yaml`, `a0_hash_code.txt`,
  `environment.txt`, `user_setup.patch`: configuration and environment snapshot.

The fixed batch SHA256 is
`0677979ffdbdaf13c339598538a366702ef9a35fc723c568f89b7325bfa3c1b1`.
Python 3.7.12, Torch 1.8.1, CUDA 11.1. Plot dependencies were installed only in
`/tmp/hlr-alignment-plot`, leaving the training environment intact.

Default A0 matches the original commit's encoder forward exactly, including
actions, commitment, and RVQ buffers after training-mode forward. Its raw
alignment loss on the fixed batch is `0.10698716342449188`. Alignment-only
gradient L1 sums are posterior MLP `1371.3021161556244`, posterior head
`151.30635976791382`, prior MLP `803.2602894306183`, prior head
`50.07440638542175`. Action and commitment each give zero prior gradients.
All checked outputs and gradients are finite. Evaluation-mode statistics
include all eight RVQ layers; training-mode dropout is retained unchanged.

The one-epoch check stops through a diagnostic wrapper after logging, without
lowering max_epochs or accelerating the KL schedule. The initial wrapper
attempt failed before simulation because Hydra resolved an imported entrypoint
incorrectly; using the original script entrypoint fixed it.
