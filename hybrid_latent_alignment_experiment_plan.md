# Hybrid Latent Posterior–Prior 对齐损失实验计划

> 用途：作为 Codex CLI 修改和验证 jinseokbae/hybrid_latent_prior 仓库的执行指导。
>
> 核心原则：每一步只改变一个实验因素。先检验梯度 detach，再检验分布建模，最后组合。

## 1. 研究目标

原始 Hybrid Latent 方法使用：

\[
L_{\mathrm{mm}}=\|\bar z-z_p\|_2^2
\]

需要把计划中的修改拆成两个独立因素：

1. 梯度方向：原始双向对齐 vs. detach posterior 的单向 posterior→prior 蒸馏；
2. 表征形式：确定性 64 维 latent 向量的 MSE vs. 64 维 Gaussian latent 分布的 KL。

本阶段不直接在 512 维 encoder hidden feature 上做 KL。那会同时改变对齐层位置，成为第三个实验因素，并可能使后续的 _prior_loc_net 失去监督。

最终需要回答：

- 改进主要来自 stop-gradient 的位置，还是来自 Gaussian 分布建模？
- learned prior variance 是否表达有效条件不确定性，而不是简单吸收预测误差？
- 单向蒸馏是否保留更多 posterior 目标信息和 RVQ code 多样性？
- residual 是否因此过大，使模型退化为主要依赖 RVQ 的离散模型？

## 2. Codex CLI 执行边界

开始修改前必须：

1. 运行 git status --short，保留用户已有修改；
2. 运行 git rev-parse HEAD，在实验记录中保存实际 commit；本计划分析时的公开仓库 commit 为 2113e72461f254060be72b3b920c888b22afcf8c；
3. 查找并阅读 AGENTS.md；
4. 阅读以下文件：
   - isaacgymenvs/learning/cvae_network_builder.py
   - isaacgymenvs/learning/cvae_model.py
   - isaacgymenvs/learning/cvae_agent.py
   - isaacgymenvs/cfg/train/imitation/HybridDistill.yaml
   - isaacgymenvs/utils/quantizers.py
5. 不修改 expert checkpoint、motion dataset、RVQ 算法和 low-level decoder；
6. 不删除、重置或格式化用户已有修改；
7. 每个阶段独立验证；若用户要求提交，每个阶段独立 commit。

## 3. 当前代码基线

当前 quantcond 路径近似为：

\[
h_q=f_\phi^{\mathrm{feat}}(s,\tilde s),\qquad z_q=W_qh_q
\]

\[
h_p=f_\theta^{\mathrm{feat}}(s),\qquad z_p=W_ph_p
\]

\[
y=z_q-\operatorname{sg}(z_p),\qquad
\bar y=Q_{\mathrm{RVQ}}(y)
\]

\[
\bar z=\bar y+\operatorname{sg}(z_p)
\]

对应的关键代码是：

~~~python
post_prior_loc_gap = post_loc - prior_loc.detach()
post_prior_loc_gap, indices, commit_loss = self._post_quantizer(
    post_prior_loc_gap[:, None]
)
post_loc = prior_loc.detach() + post_prior_loc_gap[:, 0]

kl_loss = (post_loc - prior_loc).pow(2).sum(dim=-1)
~~~

这里的 post_loc 已经是重建后的 \(\bar z\)。虽然变量名叫 kl_loss，Hybrid 配置下它实际是 MSE/SSE。

原始 stop-gradient 只隔离 prior 与 residual/RVQ/action reconstruction 路径。最终 loss 中的 prior_loc 没有 detach；RVQ 使用 straight-through estimator，因此原始 \(L_{\mathrm{mm}}\) 同时更新 posterior 和 prior。

原始总损失近似为：

\[
L=10L_{\mathrm{action}}+L_{\mathrm{commit}}+\gamma L_{\mathrm{align}}
\]

配置中的 vae_kl_loss_coef 从 0.1 调度到 1.0。为兼容旧 checkpoint，可以保留旧键名，但新日志统一使用 latent_align/*，不要把 Hybrid MSE 继续描述为真正的 KL。

## 4. 2×2 实验矩阵

所有实验必须保持相同的：

- commit、expert checkpoint 和 motion 数据；
- 数据采样方式和随机种子；
- 64 维 latent 和相同 RVQ 配置；
- decoder、optimizer、训练预算；
- action、commitment、temporal regularization 权重；
- validation batch、rollout motions 和初始状态。

| ID | 表征 | 对齐梯度 | 损失 | 研究目的 |
|---|---|---|---|---|
| A0 | 确定性向量 | 双向 | 原始 MSE | 官方基线 |
| A1 | 确定性向量 | posterior detach | 单向 MSE | 单独检验梯度路由 |
| A2 | Gaussian 分布 | 双向 | KL(q‖p) | 单独检验 learned uncertainty |
| A3 | Gaussian 分布 | posterior detach | KL(sg(q)‖p) | 完整组合方法 |

建议增加统一配置：

~~~yaml
latent_align:
  loss_type: mse
  direction: bidirectional
  posterior_std: 0.3
  learn_prior_std: false
  prior_logstd_min: -5.0
  prior_logstd_max: 1.0
  reduction: sum
  coef: 0.1
  schedule_multiplier: 10.0
~~~

四组实验分别为：

~~~yaml
# A0
loss_type: mse
direction: bidirectional
learn_prior_std: false

# A1
loss_type: mse
direction: post_to_prior
learn_prior_std: false

# A2
loss_type: gaussian_kl
direction: bidirectional
learn_prior_std: true

# A3
loss_type: gaussian_kl
direction: post_to_prior
learn_prior_std: true
~~~

默认配置必须完全复现 A0，旧命令和旧 checkpoint 不得因新增字段失效。

## 5. Phase 0：冻结并复现 A0

目标：建立可比较基线，不改变数值行为。

任务：

1. 保存训练命令、commit、checkpoint、环境版本和 seed；
2. 记录 expert loss、commit loss、原始 alignment loss；
3. 记录 posterior/prior/residual norm 和各层 RVQ entropy/perplexity；
4. 保存固定 validation batch；
5. 保持当前 sum(dim=-1) reduction；
6. 验证 action loss 不更新 prior，而 alignment loss 同时更新 posterior/prior。

验收条件：

- 默认配置与原代码在相同 seed、相同 batch 下前向数值一致；
- 新增日志不改变 loss；
- 无 NaN/Inf；
- 已保存基线曲线和固定 batch 统计。

## 6. Phase 1：实现 A1，只改变 detach

A0：

~~~python
align_target = post_loc
align_loss = (align_target - prior_loc).pow(2).sum(dim=-1)
~~~

A1：

~~~python
align_target = post_loc.detach()
align_loss = (align_target - prior_loc).pow(2).sum(dim=-1)
~~~

要求：

- A0/A1 的 forward 输出完全相同；
- A0/A1 的 alignment loss 标量完全相同；
- 唯一变化是 alignment loss 不再更新 posterior；
- action loss 和 commitment loss 仍训练 posterior；
- _prior_net 和 _prior_loc_net 必须从 alignment loss 获得非零梯度。

本阶段禁止加入：

- prior variance head；
- residual radius loss；
- target reconstruction 或 InfoNCE；
- hidden-feature KL；
- code entropy regularization。

验收条件：

- 同一 batch 下 A0.align_loss == A1.align_loss；
- 只对 alignment loss backward：
  - A0 posterior grad 非零、prior grad 非零；
  - A1 posterior grad 为零或 None、prior grad 非零；
- 对完整 loss backward，A1 posterior 仍有 action/commitment 梯度。

## 7. Phase 2：增加 Gaussian KL 基础设施

第一版只让 prior 预测方差，posterior 方差固定：

\[
q=\mathcal N(\bar z,\sigma_q^2I),\qquad \sigma_q=0.3
\]

\[
p=\mathcal N(z_p,\operatorname{diag}(\sigma_p^2(s)))
\]

在 ContinuousEncoder 的 quantcond/quantdirect 分支增加：

~~~python
self._prior_logstd_net = nn.Linear(units[-1], latent_dim)
~~~

初始化：

~~~python
nn.init.zeros_(self._prior_logstd_net.weight)
nn.init.constant_(
    self._prior_logstd_net.bias,
    math.log(posterior_std),
)
~~~

使训练初始时 \(\sigma_p=\sigma_q=0.3\)。

方差计算：

~~~python
prior_logstd = torch.clamp(
    self._prior_logstd_net(prior_feat),
    min=prior_logstd_min,
    max=prior_logstd_max,
)
prior_std = torch.exp(prior_logstd) + 1e-6
~~~

KL 必须用 FP32：

~~~python
with torch.cuda.amp.autocast(enabled=False):
    q_loc = post_loc.float()
    q_std = torch.full_like(q_loc, posterior_std)
    p_loc = prior_loc.float()
    p_std = prior_std.float()

    q_dist = torch.distributions.Normal(q_loc, q_std)
    p_dist = torch.distributions.Normal(p_loc, p_std)
    align_loss = torch.distributions.kl_divergence(
        q_dist, p_dist
    ).sum(dim=-1)
~~~

第一版不要让 posterior 预测 variance。当前 action path 使用确定性 post_loc；一个不参与 action reconstruction 的 posterior variance head 缺乏独立监督，会再引入退化解。

## 8. Phase 3：实现 A2，只改变分布建模

A2 使用：

~~~python
q_loc = post_loc
p_loc = prior_loc
align_loss = KL(q || p)
~~~

不 detach posterior。期望梯度：

- posterior mean 分支非零；
- prior mean 分支非零；
- prior logstd head 非零。

必须进行等价性测试。固定：

\[
\sigma_p=\sigma_q=\sigma
\]

则：

\[
D_{\mathrm{KL}}(q\Vert p)
=\frac{1}{2\sigma^2}\|\bar z-z_p\|_2^2
\]

使用 FP64 小张量验证误差小于 1e-6。

为匹配 A0 初始尺度，若 \(\sigma=0.3\)：

\[
\lambda_{KL}=2\sigma^2\lambda_{MSE}=0.18\lambda_{MSE}
\]

因此 0.1→1.0 的 MSE 权重，对应 KL 初始参考约为 0.018→0.18。

这只是初始化参考。必须记录 action、alignment、commitment 分别对 posterior/prior 参数产生的 gradient norm，避免 weighted alignment gradient 压倒主损失。

## 9. Phase 4：实现 A3，组合两项修改

A3：

~~~python
q_loc = post_loc.detach()
q_std = fixed_posterior_std
p_loc = prior_loc
p_std = learned_prior_std
align_loss = KL(q || p)
~~~

对应：

\[
L_{A3}=D_{\mathrm{KL}}\left[
\operatorname{sg}(q_\phi(z|s,\tilde s))
\Vert p_\theta(z|s)
\right]
\]

期望梯度：

- posterior mean 的 alignment 梯度为零；
- prior mean 梯度非零；
- prior logstd 梯度非零；
- 完整 loss 下 posterior 仍从 action/commitment 获得梯度。

不要把 learned prior std 立即用于 prior rollout 采样。第一轮推理仍使用 prior mean；否则会同时改变推理策略和运动平滑性。

## 10. 建议代码结构

把四种损失集中到一个函数，避免逻辑散落：

~~~python
def _compute_latent_alignment(
    self,
    post_loc,
    prior_loc,
    prior_feat,
):
    target_loc = (
        post_loc.detach()
        if self._latent_align_direction == "post_to_prior"
        else post_loc
    )

    if self._latent_align_loss_type == "mse":
        per_sample = (
            target_loc - prior_loc
        ).pow(2).sum(dim=-1)
        info = {}

    elif self._latent_align_loss_type == "gaussian_kl":
        prior_logstd = torch.clamp(
            self._prior_logstd_net(prior_feat),
            self._prior_logstd_min,
            self._prior_logstd_max,
        )
        prior_std = torch.exp(prior_logstd) + 1e-6

        with torch.cuda.amp.autocast(enabled=False):
            q_loc = target_loc.float()
            q_std = torch.full_like(
                q_loc, self._posterior_std
            )
            q_dist = Normal(q_loc, q_std)
            p_dist = Normal(
                prior_loc.float(), prior_std.float()
            )
            per_sample = kl_divergence(
                q_dist, p_dist
            ).sum(dim=-1)

        info = {
            "prior_std": prior_std,
            "prior_logstd": prior_logstd,
        }
    else:
        raise ValueError(
            f"Unknown latent align loss: "
            f"{self._latent_align_loss_type}"
        )

    return per_sample, info
~~~

实现约束：

- 默认 loss_type=mse、direction=bidirectional；
- 不改变 post_z/prior_z 的 forward 值；
- A0/A1 下不要让无用 variance 参数进入 optimizer；
- 加载旧 checkpoint 时，只允许新增 prior logstd head 成为预期 missing key；
- 不可静默吞掉其他 checkpoint mismatch；
- 记录 alignment 类型、方向、系数和 posterior std。

## 11. 单元测试和梯度测试

若仓库没有现成测试框架，增加一个无需启动 Isaac Gym 的纯 PyTorch 测试脚本。

### 数值测试

1. A0 与原始公式一致；
2. A0/A1 forward loss 完全一致；
3. 固定等方差时 Gaussian KL 等于缩放 MSE；
4. prior_loc == post_loc 且方差相同时 KL 接近 0；
5. mean error 增大时 KL 增大；
6. 固定 mean error 时，最优 prior variance 接近 posterior_var + squared_error；
7. 极端输入和 logstd clamp 下无 NaN/Inf。

### 梯度矩阵

只对 alignment loss backward：

| 实验 | post mean grad | prior mean grad | prior logstd grad |
|---|---:|---:|---:|
| A0 | 非零 | 非零 | 不存在 |
| A1 | 0/None | 非零 | 不存在 |
| A2 | 非零 | 非零 | 非零 |
| A3 | 0/None | 非零 | 非零 |

还必须验证：

- A1/A3 中 _prior_loc_net 梯度非零；
- 完整 loss 下 A1/A3 posterior 梯度非零；
- action loss 单独 backward 时 quantcond 路径不更新 prior；
- 每次独立 backward 前清空梯度，避免累积造成误判。

### Checkpoint 测试

- A0/A1 能加载原始 checkpoint；
- A2/A3 加载旧 checkpoint 时，只有 prior logstd head 是预期 missing key；
- A2/A3 保存并重载后输出一致；
- 不得把 A2/A3 checkpoint 静默当成 A0。

## 12. 训练分级

### T0：公式和梯度测试

- 不启动 Isaac Gym；
- 完成数值、梯度、checkpoint 测试；
- 每次修改后运行。

### T1：Smoke test

- 每个实验 100–200 updates；
- 可暂时降低到 256 environments；
- 只检查运行、显存、日志、NaN 和梯度；
- 不用 smoke test 判断最终性能。

终止条件：

- loss 出现 NaN/Inf；
- prior std 大量触及 clamp；
- weighted alignment gradient 比 action gradient 高两个数量级；
- posterior/prior norm 持续爆炸。

### T2：Pilot

- A0–A3 使用相同一个 seed；
- 使用正式预算的约 5%–10%；
- 固定 validation batch 和 rollout motions；
- 只在 pilot 阶段选择 KL 系数和 variance regularization；
- 选定后冻结超参数。

### T3：正式实验

- A0–A3 至少 3 个相同 seeds；
- 使用完全相同训练预算；
- 报告均值、标准差和单 seed 曲线；
- 不得只给某个实验增加训练时间。

## 13. 必须记录的指标

### 优化指标

- raw/weighted action loss；
- raw/weighted commitment loss；
- raw/weighted alignment loss；
- 三种 loss 对 posterior/prior 的 gradient norm；
- learning rate 和 alignment coefficient。

### Latent 指标

- post_loc、prior_loc 的 mean/std/norm；
- residual 的 mean/std/norm；
- posterior/prior covariance effective rank；
- posterior–prior cosine similarity；
- 固定 obs、改变 goal_obs 时的 posterior feature variance。

### Gaussian 指标

- prior std 的 mean、median、p05、p95；
- logstd clamp 上下界占比；
- predicted variance 与实际 squared error 的相关性；
- Gaussian KL/NLL；
- 按 predicted std 分桶后的实际 squared error。

只有“预测 std 越大的样本实际 error 也越大”，才能支持 learned uncertainty 的解释。平均 std 变大不是证据。

### RVQ 指标

对每个 quantizer 分别记录：

- code usage count；
- entropy 和 perplexity；
- active code ratio；
- 每层量化前后的 residual energy；
- quantizer dropout 下的层激活频率。

### 控制质量

- expert action MSE；
- imitation reward；
- tracking position/orientation error；
- prior-only rollout 存活时间；
- jerk/acceleration；
- foot sliding；
- failure/fall rate。

## 14. Posterior collapse 检测

不能只用 residual norm 判断坍缩。

### Goal sensitivity

固定 obs，输入多个合理的 goal_obs：

\[
V_{\mathrm{goal}}
=\operatorname{Var}_j[\mu_q(s,\tilde s_j)]
\]

若接近 0，posterior 对目标不敏感。

### Residual intervention

分别向 decoder 输入：

1. 正常 residual；
2. zero residual；
3. batch-shuffled residual；
4. random valid RVQ code。

比较 action 和 rollout。如果 residual 改变但 action 几乎不变，说明 decoder 功能上忽略 latent。

### Target probe

冻结模型，用轻量 linear/MLP probe 从 residual 预测目标动作、未来 pose 或 motion label，并与 prior feature、shuffled residual 和 constant baseline 对比。

## 15. 结果解释

| 结果 | 解释 |
|---|---|
| A1 优于 A0，A3≈A1 | 主要贡献来自单向梯度 |
| A2 优于 A0，A3≈A2 | 主要贡献来自 uncertainty weighting |
| A3 同时优于 A1/A2 | 两个因素存在协同作用 |
| A1/A3 code entropy 上升但 prior rollout 下降 | residual 承担过多 |
| A2/A3 std 大量触及上限 | variance 在吸收误差，不能宣称学到不确定性 |
| residual 非零但 intervention 不改变 action | 功能性 posterior/decoder collapse |
| KL 下降但 action/rollout 不改善 | 分布对齐不是有效控制代理 |
| A2/A3 与等比例 MSE 一致 | KL 只是重标定 MSE |

正式结论必须依据至少 3 个 seeds 的均值和方差。

## 16. Phase 5：A0–A3 完成后才允许加入的变量

### Residual radius loss

若 A1/A3 residual 持续增大且 prior rollout 下降，可独立增加：

\[
L_{\mathrm{radius}}
=\left[\max(\|\bar y\|_2-r,0)\right]^2
\]

半径 r 先取 A0 收敛后 residual norm 的 90%–95% 分位数。不要直接恢复把 residual 压向零的完整 MSE。

### Target-information loss

如果 posterior/RVQ 出现功能性坍缩，再单独增加 residual→target reconstruction 或 InfoNCE，显式要求：

\[
I(\bar y;\tilde s\mid s)>0
\]

### 512 维 hidden-feature KL

最后再考虑。直接只对 post_feat/prior_feat 做 KL 会训练 _prior_net，但不会训练 _prior_loc_net。必须保留 64 维 supervision、共享 projection head，或者让 feature distribution 参数直接生成实际 post_loc/prior_loc。

## 17. 已知代码注意事项

启用 latent temporal regularization 前检查：

- cvae_agent.py 和 cvae_amp_agent.py 的相关分支存在 detch() 拼写，应为 detach()；
- 某些判断使用 quant_cond，而配置/encoder 使用 quantcond；
- 当前 latent_regularize=False，因此这些分支可能没有执行。

这些属于独立问题。若修复，单独提交，并证明默认 latent_regularize=False 时行为不变。

## 18. 最终交付物

Codex CLI 完成后必须提供：

1. 修改文件清单；
2. A0–A3 配置或可复现命令；
3. 数值等价测试结果；
4. 四种损失的梯度矩阵实测结果；
5. smoke test 结果；
6. checkpoint 兼容性说明；
7. TensorBoard/W&B 指标名称表；
8. 固定 validation batch 的 A0–A3 对比；
9. 未完成项、限制和下一步建议。

不能只报告“运行成功”，必须报告实际梯度是否符合实验矩阵。
