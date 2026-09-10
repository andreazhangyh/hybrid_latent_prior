# A0–A3 latent alignment 改动与 loss 公式

本文根据当前代码和配置总结 HybridDistill 的 A0、A1、A2、A3 四组实验。
实验配置位于：

```text
isaacgymenvs/cfg/train/imitation/HybridDistillA0.yaml
isaacgymenvs/cfg/train/imitation/HybridDistillA1.yaml
isaacgymenvs/cfg/train/imitation/HybridDistillA2.yaml
isaacgymenvs/cfg/train/imitation/HybridDistillA3.yaml
```

核心实现位于：

```text
isaacgymenvs/learning/cvae_network_builder.py
isaacgymenvs/learning/cvae_model.py
isaacgymenvs/learning/cvae_agent.py
isaacgymenvs/utils/quantizers.py
vector_quantize_pytorch/residual_vq.py
vector_quantize_pytorch/vector_quantize_pytorch.py
```

## 1. 四组实验的差异

| 实验 | 配置继承 | latent 对齐形式 | 对齐方向 | prior 方差 | alignment 初始系数 |
|---|---|---|---|---|---:|
| A0 | `HybridDistill` | MSE | bidirectional | 无 | 0.1 |
| A1 | `HybridDistill` | MSE | posterior → prior | 无 | 0.1 |
| A2 | `HybridDistill` | Gaussian KL | bidirectional | learned | 0.018 |
| A3 | `HybridDistillA2` | Gaussian KL | posterior → prior | learned | 0.018 |

其中 A3 继承 A2，再只增加 `direction: post_to_prior`；A1 只改变 A0 的
`direction`；A2 同时改变 alignment 类型、prior 方差建模和 alignment 系数。

所有实验仍使用相同的基本设置：

- `continuous_enc_style: quantcond`；
- latent dimension 为 64；
- RVQ，8 个 quantizer，每个 codebook 默认 1024 个 code；
- `vae_commit_loss_coef = 1.0`；
- `expert_loss_coef = 10.0`；
- `vae_kl_schedule = True`；
- posterior 标准差常数 `posterior_std = 0.3`；
- prior log-standard-deviation clamp 为 `[-5.0, 1.0]`。

这里的配置键仍然叫 `vae_kl_loss`，但 A0/A1 中实际是 MSE；因此本文统一称为
`alignment loss`。

## 2. 共同的 quantcond latent 路径

令当前状态为 (s)，目标状态为 (g)。代码首先计算：

\[
h_q = f_q([s,g]), \\
\mu_q = W_q h_q + b_q,
\]

\[
h_p = f_p(s), \\
\mu_p = W_p h_p + b_p.
\]

其中 (q) 表示 goal-conditioned posterior 分支，(p) 表示 state-only prior
分支。

`quantcond` 不直接量化 (mu_q)，而是量化 posterior 与 prior 的 residual：

\[
r = \mu_q - \operatorname{sg}(\mu_p),
\]

\[
\hat r = \operatorname{RVQ}(r),
\]

最后使用 straight-through 形式构造 posterior latent mean：

\[
\tilde\mu_q = \operatorname{sg}(\mu_p) + \hat r.
\]

因此 forward 时 posterior latent 仍包含 residual 的梯度路径，而 residual 的
输入不通过显式的 prior 分支反传。alignment loss 则直接比较最终的
(	ilde\mu_q) 与 (mu_p)。

训练时 decoder 使用 posterior latent：

\[
\hat a = D(s, \tilde\mu_q),
\]

prior rollout 使用 state-only 的 prior mean。

## 3. 共同的总 loss

在 `cVAEAgent.calc_gradients_non_rl` 中，HybridDistill 的基础 loss 是：

\[
L = \lambda_e L_{expert}
  + \lambda_a L_{align}
  + \lambda_c L_{commit}.
\]

默认系数为：

\[
\lambda_e = 10, \qquad \lambda_c = 1.
\]

### 3.1 Expert action loss

expert checkpoint 给出目标 action mean (a^*)，当前 decoder 给出
(hat a)：

\[
L_{expert}
= \frac{1}{B}\sum_{b=1}^{B}
\left\|\hat a_b-a^*_b\right\|_2^2.
\]

对应代码：

```python
expert_loss = (mu - expert_mu_batch).pow(2).sum(dim=-1).mean()
```

### 3.2 RVQ commitment loss

对第 (k) 个 RVQ 层，令输入 residual 为 (x_k)，量化 code 为 (e_k)。
当前非 learnable EMA codebook 使用：

\[
C_k = \frac{1}{B}\sum_{b=1}^{B}
\left\|\operatorname{sg}(e_{k,b})-x_{k,b}\right\|_2^2.
\]

ResidualVQ 对实际启用的 quantizer 层返回各自的 commitment loss，模型再对层
维取平均：

\[
L_{commit}=\frac{1}{K}\sum_{k=1}^{K}C_k.
\]

当前 `vae_commit_loss_coef=1.0`。训练模式下 RVQ 还会进行 quantizer dropout，
因此每个 batch 的有效 (K) 可能小于 8。

## 4. A0：原始确定性双向 MSE baseline

### 配置改动

A0 配置没有覆盖 `HybridDistill.yaml` 的 alignment 字段，使用默认值：

```yaml
latent_align:
  loss_type: mse
  direction: bidirectional
  learn_prior_std: false
```

因此 A0 是原始 HybridDistill 的确定性 latent alignment baseline。

### Alignment loss

\[
L_{align}^{A0}
= \left\|\tilde\mu_q-\mu_p\right\|_2^2
= \sum_{i=1}^{d}
  \left(\tilde\mu_{q,i}-\mu_{p,i}\right)^2.
\]

其中 (d=64)。代码为：

```python
align_target = post_loc
return (align_target - prior_loc).pow(2).sum(dim=-1)
```

### 梯度路径

- alignment loss 更新 posterior/residual 路径；
- alignment loss 更新 `_prior_net` 和 `_prior_loc_net`；
- action loss 和 commitment loss 主要更新 posterior、decoder 和 quantizer；
- action/commitment 路径不会通过 quantcond residual 的 detached prior 输入更新 prior。

因此 A0 的 alignment gradient 对 posterior 和 prior 都是非零的。

### 总 loss

\[
L_{A0}=10L_{expert}+\gamma_{A0}(t)L_{align}^{A0}+L_{commit}.
\]

`vae_kl_schedule=True` 时，alignment 系数从 0.1 线性增加到 1.0：

\[
\gamma_{A0}(t)=0.1\left(1+9\cdot
\min\left(\frac{t}{0.125T},1\right)\right),
\]

其中 (T) 是 `max_epochs`，(t) 是当前 epoch。

## 5. A1：只改变 alignment 的梯度方向

### 配置改动

A1 只增加：

```yaml
latent_align:
  direction: post_to_prior
```

其他设置与 A0 完全相同：仍是确定性 latent、MSE、RVQ 和相同 loss 权重。

### Alignment loss

A1 的 forward 数值公式与 A0 完全相同：

\[
L_{align}^{A1}
= \left\|\operatorname{sg}(\tilde\mu_q)-\mu_p\right\|_2^2.
\]

代码只在 target 上增加 detach：

```python
align_target = post_loc.detach()
return (align_target - prior_loc).pow(2).sum(dim=-1)
```

### 梯度路径

- alignment loss 不再更新 posterior、residual 或 RVQ codebook；
- alignment loss 仍更新 `_prior_net` 和 `_prior_loc_net`；
- action loss 和 commitment loss 仍更新 posterior，因此完整总 loss 下 posterior
  仍然有梯度；
- A0 与 A1 的同一 batch forward 输出和 alignment 标量应完全相同，区别只在
  backward 梯度。

### 总 loss

\[
L_{A1}=10L_{expert}+\gamma_{A1}(t)L_{align}^{A1}+L_{commit},
\]

其中 (gamma_{A1}(t)) 与 A0 相同，从 0.1 调度到 1.0。

## 6. A2：learned prior variance 的 Gaussian KL

### 配置改动

A2 覆盖：

```yaml
vae_kl_loss_coef: 0.018
latent_align:
  loss_type: gaussian_kl
  learn_prior_std: true
```

`direction` 没有覆盖，因此仍为 `bidirectional`。

模型新增 prior log-standard-deviation head：

```python
self._prior_logstd_net = nn.Linear(units[-1], latent_dim)
```

其初始权重为 0，bias 为 (log(0.3))，所以初始时每个 latent 维度的 prior
standard deviation 都是 0.3。

### 两个 Gaussian 分布

posterior 方差不预测，固定为：

\[
q(z|s,g)=\mathcal N(\tilde\mu_q,\sigma_q^2I),
\qquad \sigma_q=0.3.
\]

prior mean 仍为 (mu_p)，但 standard deviation 由 prior feature 预测：

\[
u_p=W_\sigma h_p+b_\sigma,
\]

\[
\ell_p=\operatorname{clip}(u_p,-5,1),
\qquad \sigma_{p,i}=\exp(\ell_{p,i})+10^{-6}.
\]

于是：

\[
p(z|s)=\mathcal N(\mu_p,
\operatorname{diag}(\sigma_p^2)).
\]

### Alignment loss

A2 使用 (q\Vert p) 的逐维 diagonal Gaussian KL：

\[
L_{align}^{A2}
=D_{KL}(q\Vert p)
=\sum_{i=1}^{d}\left[
\log\frac{\sigma_{p,i}}{\sigma_q}
+\frac{\sigma_q^2+
      (\tilde\mu_{q,i}-\mu_{p,i})^2}
      {2\sigma_{p,i}^2}
-\frac12
\right].
\]

代码通过 `torch.distributions.kl_divergence` 计算，并强制在 FP32 autocast
关闭的上下文中执行，以避免旧版 CUDA/PyTorch 混合精度下的数值问题。

### 等方差时与 MSE 的关系

如果 (sigma_p=sigma_q=sigma)，则：

\[
D_{KL}(q\Vert p)=
\frac{1}{2\sigma^2}
\left\|\tilde\mu_q-\mu_p\right\|_2^2.
\]

当 (sigma=0.3) 时，KL 相当于 MSE 的 (1/(2\times0.3^2)=5.5556) 倍。
因此 A2/A3 使用初始系数 0.018，使其初始 effective mean-matching 权重大致
对应 A0/A1 的 (0.1)：

\[
0.018\times\frac{1}{2(0.3)^2}\approx0.1.
\]

这只是初始尺度匹配；训练过程中 prior variance 会学习并改变 KL 的实际权重。

### 总 loss

\[
L_{A2}=10L_{expert}+\gamma_{A2}(t)L_{align}^{A2}+L_{commit}.
\]

由于 A2 继承 `vae_kl_schedule=True`，(gamma_{A2}) 从 0.018 调度到 0.18：

\[
\gamma_{A2}(t)=0.018\left(1+9\cdot
\min\left(\frac{t}{0.125T},1\right)\right).
\]

### 梯度路径

在 alignment loss 下，posterior mean、prior mean 和 prior logstd head 都有梯度；
完整 loss 下 posterior 还会从 action 和 commitment 获得梯度。

## 7. A3：Gaussian KL + posterior detach

### 配置改动

A3 继承 A2，并增加：

```yaml
latent_align:
  direction: post_to_prior
```

因此 A3 同时包含：

1. A2 的 learned prior variance；
2. A1 的 posterior-detach alignment 梯度方向。

### Alignment loss

\[
L_{align}^{A3}
=D_{KL}\left(
\mathcal N(\operatorname{sg}(\tilde\mu_q),\sigma_q^2I)
\;\middle\Vert\;
\mathcal N(\mu_p,\operatorname{diag}(\sigma_p^2))
\right).
\]

展开为：

\[
L_{align}^{A3}
=\sum_{i=1}^{d}\left[
\log\frac{\sigma_{p,i}}{\sigma_q}
+\frac{\sigma_q^2+
      (\operatorname{sg}(\tilde\mu_{q,i})-\mu_{p,i})^2}
      {2\sigma_{p,i}^2}
-\frac12
\right].
\]

### 梯度路径

- alignment loss 不更新 posterior mean、residual 或 RVQ codebook；
- alignment loss 更新 prior mean；
- alignment loss 更新 prior logstd head；
- 完整总 loss 下 posterior 仍通过 action 和 commitment 更新；
- prior rollout 仍使用 prior mean，learned std 不直接用于 rollout 采样。

### 总 loss

\[
L_{A3}=10L_{expert}+\gamma_{A3}(t)L_{align}^{A3}+L_{commit},
\]

其中 (gamma_{A3}(t)) 与 A2 相同，从 0.018 调度到 0.18。

## 8. A0–A3 对比总结

| 项目 | A0 | A1 | A2 | A3 |
|---|---:|---:|---:|---:|
| posterior alignment gradient | 有 | 无 | 有 | 无 |
| prior mean alignment gradient | 有 | 有 | 有 | 有 |
| prior logstd head | 无 | 无 | 有 | 有 |
| posterior std | 不参与 alignment | 不参与 alignment | 固定 0.3 | 固定 0.3 |
| alignment 公式 | MSE | MSE | (D_{KL}(q\Vert p)) | (D_{KL}(\operatorname{sg}(q)\Vert p)) |
| 初始 alignment coef | 0.1 | 0.1 | 0.018 | 0.018 |
| 最终 scheduled coef | 1.0 | 1.0 | 0.18 | 0.18 |
| action/commitment posterior gradient | 有 | 有 | 有 | 有 |

## 9. 代码级验证依据

当前测试覆盖以下不变量：

- A0/A1 的 forward 输出和 raw alignment loss 一致；
- A0/A1 的差异只体现在 alignment backward 的 posterior gradient；
- A2/A3 的 equal-variance Gaussian KL 与
  \(\|q-p\|^2/(2\sigma^2)\) 一致；
- A2/A3 的 prior logstd head 初始化为 0 weight、`log(0.3)` bias；
- A2/A3 的 prior logstd clamp 极值下 loss 和梯度保持 finite；
- 四组实验的 alignment/action/commitment/full gradient matrix 满足预期。

相关测试：

```text
tests/test_latent_alignment.py
tests/test_latent_alignment_rvq.py
tests/test_gaussian_alignment.py
scripts/verify_alignment_smoke.py
```

测试中的完整总 loss 形式为：

```python
full_loss = 10 * action_loss + alignment_coef * alignment_loss + commitment_loss
```

其中 A0/A1 的 `alignment_coef=0.1`，A2/A3 的 `alignment_coef=0.018`，对应
四组配置在训练初始时的系数。
