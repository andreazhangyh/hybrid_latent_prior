# Hybrid Latent Prior Environment Setup

## Current Workspace State

- Project root: `/home/yuhuan.zhang/hybrid/hybrid_latent_prior`
- Isaac Gym Preview 4 is available locally at `/home/yuhuan.zhang/hybrid/.local/isaacgym/isaacgym`
- LaFAN1 source dataset is available at `/home/yuhuan.zhang/hybrid/ubisoft-laforge-animation-dataset`
- LaFAN1 archive contains 77 BVH sequences; the same 77 BVH files are already present under `isaacgymenvs/tasks/amp/poselib/lafan1/data`
- Retargeted motion files have been generated under `isaacgymenvs/tasks/amp/poselib/data/AMP` and linked from `assets/amp/motions`
- Pretrained weights have been downloaded and extracted under `isaacgymenvs/pretrained_weights`
- Current default Python is `base` Python 3.13, which is incompatible with Isaac Gym Preview 4
- The machine has visible NVIDIA GPUs. Current verified setup uses GPU 0.

## Recommended Environment

Use a fresh conda environment named `hlr_env` matching Isaac Gym Preview 4:

- Python 3.7
- PyTorch 1.8.1
- CUDA toolkit 11.1 from conda
- Isaac Gym Preview 4 installed editable from the local `.local/isaacgym` tree
- `hybrid_latent_prior` installed editable from this repo
- `torch-scatter==2.0.8` built for `torch==1.8.1+cu111`

Do not reuse the existing `hssim` or `env_isaaclab*` environments. They use Python 3.10/3.11 and Torch 2.x, while Isaac Gym Preview 4 only ships Python 3.6/3.7/3.8 Linux bindings.

## Install Commands

```bash
cd /home/yuhuan.zhang/hybrid/.local/isaacgym/isaacgym/python
conda env create -n hlr_env -f rlgpu_conda_env.yml

conda activate hlr_env
pip install -e /home/yuhuan.zhang/hybrid/.local/isaacgym/isaacgym/python

cd /home/yuhuan.zhang/hybrid/hybrid_latent_prior
pip install -e .
pip install torch-scatter==2.0.8 -f https://data.pyg.org/whl/torch-1.8.1+cu111.html
pip install tensorboardX
```

Install `matplotlib` only if you enable `VISUALIZE=True` in the poselib scripts.

`human-body-prior` is only needed for SMPL/AMASS motion loading. It is not required for the LaFAN retargeting path after the local optional-import patch.

On this RTX 4090 machine, PyTorch 1.8.1/CUDA 11.1 needs two runtime settings:

```bash
export LD_LIBRARY_PATH=/home/yuhuan.zhang/miniconda3/envs/hlr_env/lib:$LD_LIBRARY_PATH
export PYTORCH_JIT=0
```

`LD_LIBRARY_PATH` lets Isaac Gym find `libpython3.7m.so.1.0`; `PYTORCH_JIT=0` avoids old NVRTC JIT codegen for Ada GPUs.

## Data Preparation

Run from the poselib directory:

```bash
conda activate hlr_env
cd /home/yuhuan.zhang/hybrid/hybrid_latent_prior/isaacgymenvs/tasks/amp/poselib

python generate_amp_humanoid_tpose.py
python lafan_bvh_to_npz.py

python lafan_processing.py

mkdir -p /home/yuhuan.zhang/hybrid/hybrid_latent_prior/isaacgymenvs/assets/amp/motions
ln -sfn /home/yuhuan.zhang/hybrid/hybrid_latent_prior/isaacgymenvs/tasks/amp/poselib/data/AMP/LAFAN_ALL_* /home/yuhuan.zhang/hybrid/hybrid_latent_prior/isaacgymenvs/assets/amp/motions/LAFAN_ALL
mkdir -p /home/yuhuan.zhang/hybrid/hybrid_latent_prior/assets/amp/motions
ln -sfn /home/yuhuan.zhang/hybrid/hybrid_latent_prior/isaacgymenvs/tasks/amp/poselib/data/AMP/LAFAN_ALL_* /home/yuhuan.zhang/hybrid/hybrid_latent_prior/assets/amp/motions/LAFAN_ALL
```

`lafan_processing.py` now also accepts an explicit source directory:

```bash
LAFAN_NPZ_ROOT_DIR=data/LAFAN/lafan1_npz_YYYY-Mon-DD python lafan_processing.py
```

For task configs that use `LAFAN_LOCO`, generate a locomotion subset with:

```bash
LAFAN_MODE=loco python lafan_processing.py
ln -sfn /home/yuhuan.zhang/hybrid/hybrid_latent_prior/isaacgymenvs/tasks/amp/poselib/data/AMP/LAFAN_LOCO_* /home/yuhuan.zhang/hybrid/hybrid_latent_prior/isaacgymenvs/assets/amp/motions/LAFAN_LOCO
ln -sfn /home/yuhuan.zhang/hybrid/hybrid_latent_prior/isaacgymenvs/tasks/amp/poselib/data/AMP/LAFAN_LOCO_* /home/yuhuan.zhang/hybrid/hybrid_latent_prior/assets/amp/motions/LAFAN_LOCO
```

The root-level `assets/amp/motions` links are required because the task code resolves motions relative to `isaacgymenvs/tasks/../../assets/amp/motions`.

## Pretrained Weights

README expects pretrained weights under:

```text
/home/yuhuan.zhang/hybrid/hybrid_latent_prior/isaacgymenvs/pretrained_weights
```

The official README Google Drive bundle was downloaded to:

```text
/data-nfs/yuhuan.zhang/hybrid_downloads/pretrained_weights.zip
```

It contains `pretrained_weights/imitation/imitation_expert/nn/imitation_expert_weight.pth` and the other imitation/task checkpoints. The bundle has been extracted to:

```text
/home/yuhuan.zhang/hybrid/hybrid_latent_prior/isaacgymenvs/pretrained_weights
```

Training from scratch does not require pretrained weights for the first expert imitation stage, but it does require the retargeted motion data.

## Smoke Tests

First verify imports:

```bash
conda activate hlr_env
export LD_LIBRARY_PATH=/home/yuhuan.zhang/miniconda3/envs/hlr_env/lib:$LD_LIBRARY_PATH
python -c "import isaacgym; import torch; import isaacgymenvs; import torch_scatter; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Then verify Isaac Gym itself:

```bash
cd /home/yuhuan.zhang/hybrid/.local/isaacgym/isaacgym/python/examples
python joint_monkey.py --headless
```

Then run a minimal project command from `isaacgymenvs`:

```bash
cd /home/yuhuan.zhang/hybrid/hybrid_latent_prior/isaacgymenvs
PYTORCH_JIT=0 python train.py test=True headless=True num_envs=1 task=LafanImitation train=imitation/ExpertPPO checkpoint=pretrained_weights/imitation/imitation_expert/nn/imitation_expert_weight.pth max_iterations=1
```

For Hybrid imitation training:

```bash
cd /home/yuhuan.zhang/hybrid/hybrid_latent_prior/isaacgymenvs
PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 python train.py headless=True task=LafanImitation train=imitation/HybridDistill experiment=imitation_hybrid expert=pretrained_weights/imitation/imitation_expert
```

As of 2026-09-07 23:00 Asia/Shanghai, this full training has been started in the background:

```text
parent PID: 429721
train PID: 429739
log: /home/yuhuan.zhang/hybrid/hybrid_latent_prior/isaacgymenvs/train_logs/imitation_hybrid_20260907_2305.log
run: /home/yuhuan.zhang/hybrid/hybrid_latent_prior/isaacgymenvs/runs/imitation_hybrid_b6offnv3
```
