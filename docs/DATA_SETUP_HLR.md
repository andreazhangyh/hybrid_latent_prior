# HLR 权重与 LaFAN AMP 数据配置

本文用于在另一台 Linux 机器上准备本项目运行所需的两类外部资源：

- 官方 expert checkpoint；
- LaFAN1 BVH 转换并 retarget 后的 AMP motion 数据。

本文假设项目已克隆到 `${REPO}`，并且已经按照
[`ENV_SETUP_HLR.md`](ENV_SETUP_HLR.md) 创建好了 `hlr_env` 环境和 Isaac Gym
Preview 4。

## 1. 获取代码

```bash
git clone -b codex/current-state \
  https://github.com/andreazhangyh/hybrid_latent_prior.git
cd hybrid_latent_prior
export REPO="$PWD"
```

不要依赖仓库中的本机 motion symlink；这些链接可能指向另一台机器上的绝对路径，
后文会重新创建。

## 2. 下载并安装 expert checkpoint

官方权重包来自项目 README 中的 Google Drive 链接：

<https://drive.google.com/file/d/1gEnpnCDHP5ysDTW3rE_ftUlp8cRcMRmh/view?usp=sharing>

### 方法 A：浏览器下载

1. 在浏览器打开上面的链接并下载 zip 文件。
2. 将文件放到临时目录，例如 `${REPO}/downloads/pretrained_weights.zip`。
3. 解压到项目的 `isaacgymenvs/` 目录：

```bash
mkdir -p "$REPO/downloads"
unzip "$REPO/downloads/pretrained_weights.zip" -d "$REPO/isaacgymenvs"
```

如果 zip 内已经包含 `pretrained_weights/` 顶层目录，上面的命令会直接得到正确结构。

### 方法 B：命令行下载

在 `hlr_env` 中安装兼容 Python 3.7 的 gdown，然后使用文件 ID 下载：

```bash
conda activate hlr_env
python -m pip install 'gdown<5'
mkdir -p "$REPO/downloads"
gdown --fuzzy \
  'https://drive.google.com/file/d/1gEnpnCDHP5ysDTW3rE_ftUlp8cRcMRmh/view?usp=sharing' \
  -O "$REPO/downloads/pretrained_weights.zip"
unzip "$REPO/downloads/pretrained_weights.zip" -d "$REPO/isaacgymenvs"
```

验证 expert checkpoint：

```bash
test -f "$REPO/isaacgymenvs/pretrained_weights/imitation/imitation_expert/nn/imitation_expert_weight.pth"
du -h "$REPO/isaacgymenvs/pretrained_weights/imitation/imitation_expert/nn/imitation_expert_weight.pth"
```

如果只运行 HybridDistill，expert checkpoint 就足够；完整权重包还包含其他 imitation、
tracking、navigation 和 in-betweening 模型。

## 3. 获取 LaFAN1 BVH 数据

从官方仓库获取原始 LaFAN1 数据。官方仓库使用 Git LFS 保存
`lafan1/lafan1.zip`，因此必须安装 Git LFS 并执行 `git lfs pull`；仅普通 `git clone`
可能只得到一个很小的 LFS 指针文件，而不是实际数据。

<https://github.com/ubisoft/ubisoft-laforge-animation-dataset>

```bash
# Ubuntu/Debian 示例；如果系统已有 git-lfs，可跳过安装步骤。
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install

cd "$REPO"
git clone --depth 1 \
  https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git \
  /tmp/ubisoft-laforge-animation-dataset
cd /tmp/ubisoft-laforge-animation-dataset
git lfs pull

mkdir -p /tmp/lafan1-bvh
unzip -q lafan1/lafan1.zip -d /tmp/lafan1-bvh

mkdir -p "$REPO/isaacgymenvs/tasks/amp/poselib/lafan1/data"
find /tmp/lafan1-bvh -type f -name '*.bvh' -exec cp {} \
  "$REPO/isaacgymenvs/tasks/amp/poselib/lafan1/data/" \;
```

确认 BVH 数量：

```bash
find "$REPO/isaacgymenvs/tasks/amp/poselib/lafan1/data" \
  -type f -name '*.bvh' | wc -l
```

当前项目预期 77 个 BVH 文件。若官方仓库的 zip 内部目录结构发生变化，只要将其中所有
LaFAN1 `.bvh` 文件放入上面的 `lafan1/data` 目录即可。

## 4. 生成 LaFAN NPZ 和 AMP motion

以下命令必须在 `hlr_env` 中执行，并从 poselib 目录运行：

```bash
conda activate hlr_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_JIT=0

cd "$REPO/isaacgymenvs/tasks/amp/poselib"

python generate_amp_humanoid_tpose.py

export NPZ_DIR="$PWD/data/LAFAN/lafan1_npz_transfer"
LAFAN_NPZ_OUT_DIR="$NPZ_DIR" python lafan_bvh_to_npz.py
python generate_lafan_tpose.py

LAFAN_NPZ_ROOT_DIR="$NPZ_DIR" LAFAN_MODE=all python lafan_processing.py
LAFAN_NPZ_ROOT_DIR="$NPZ_DIR" LAFAN_MODE=loco python lafan_processing.py
```

生成结果位于：

```text
data/LAFAN/lafan1_npz_transfer/
data/AMP/LAFAN_ALL_<日期>/
data/AMP/LAFAN_LOCO_<日期>/
```

其中 `LAFAN_ALL` 用于普通 imitation，`LAFAN_LOCO` 用于 locomotion/in-betweening
相关任务。

## 5. 创建 motion symlink

任务代码从仓库根目录的 `assets/amp/motions` 解析 motion，因此需要创建两个位置的
链接。下面命令会自动选择本次生成的最新目录：

```bash
ALL_DIR="$(find "$REPO/isaacgymenvs/tasks/amp/poselib/data/AMP" \
  -mindepth 1 -maxdepth 1 -type d -name 'LAFAN_ALL_*' | sort | tail -n 1)"
LOCO_DIR="$(find "$REPO/isaacgymenvs/tasks/amp/poselib/data/AMP" \
  -mindepth 1 -maxdepth 1 -type d -name 'LAFAN_LOCO_*' | sort | tail -n 1)"

test -n "$ALL_DIR" && test -n "$LOCO_DIR"

mkdir -p "$REPO/assets/amp/motions"
mkdir -p "$REPO/isaacgymenvs/assets/amp/motions"

ln -sfn "$ALL_DIR" "$REPO/assets/amp/motions/LAFAN_ALL"
ln -sfn "$LOCO_DIR" "$REPO/assets/amp/motions/LAFAN_LOCO"
ln -sfn "$ALL_DIR" "$REPO/isaacgymenvs/assets/amp/motions/LAFAN_ALL"
ln -sfn "$LOCO_DIR" "$REPO/isaacgymenvs/assets/amp/motions/LAFAN_LOCO"

readlink -f "$REPO/assets/amp/motions/LAFAN_ALL"
readlink -f "$REPO/assets/amp/motions/LAFAN_LOCO"
```

如果只运行 `LafanImitation`，至少需要 `LAFAN_ALL`；使用 `LAFAN_LOCO` 的任务还需要
第二个链接。

## 6. 最小验证

```bash
cd "$REPO"
python -c "import isaacgym; import torch; import isaacgymenvs; print(torch.__version__, torch.version.cuda)"

cd "$REPO/isaacgymenvs"
PYTORCH_JIT=0 python train.py \
  test=True headless=True num_envs=1 \
  task=LafanImitation train=imitation/ExpertPPO \
  checkpoint=pretrained_weights/imitation/imitation_expert/nn/imitation_expert_weight.pth \
  max_iterations=1
```

如果该命令能正常加载环境、checkpoint 和 motion，说明两类外部资源配置完成。

## 7. 资源清单与空间需求

大致空间需求如下，实际大小会随数据版本和权重包变化：

| 资源 | 用途 | 当前参考大小 |
|---|---|---:|
| expert checkpoint | HybridDistill 的教师策略 | 约 87 MB |
| 完整 pretrained_weights 包 | 所有预训练任务 | 约 1.8 GB 解压后 |
| 原始 LaFAN BVH | 重新生成数据 | 约 333 MB（当前仓库副本） |
| 中间 NPZ | retarget 输入 | 约 1.2 GB |
| AMP NPY | 训练/测试 motion | 约 1.9 GB |

数据集和权重不应提交到 Git；请遵守各自项目的许可证和下载条款。
