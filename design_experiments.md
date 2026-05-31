# 教学项目实现：梯度反演攻击与 LAUGD 轻量防御（含消融）

本项目面向教学任务：用最小可运行实现复现“联邦学习中的梯度泄漏现象”，并实现一种新的轻量防御策略 LAUGD（Leakage-Aware Unbiased Gradient Dropout），再通过对比与消融展示隐私-效用权衡。

## 1. 目标与范围

### 1.1 目标

- 复现：服务器可见单客户端更新时，DLG/iDLG 风格梯度反演可重构输入（以 MNIST 图像为例）
- 实现：Clipping、DP-light、LAUGD 三类防御，并支持开关式消融
- 对比：在训练效用（test acc）与隐私泄漏（PSNR/MSE、可视化）上做可复现实验

### 1.2 教学化简

- 训练协议采用 FedSGD 风格：每轮客户端上传“梯度列表”（更利于把攻击与防御讲清楚、跑通）
- 只评估“单轮、单客户端更新可见”的泄漏威胁
- 不引入密码学协议（SecAgg/TEE），仅做客户端侧更新处理

## 2. 威胁模型

- 对手：honest-but-curious 服务器（或任何能看到单客户端上传更新的观察者）
- 可见信息：某一轮某个客户端上传的更新向量（本实现默认是梯度列表）
- 攻击能力：在离线环境中运行梯度反演/更新匹配优化，试图重构输入或推断标签
- 防御目标：降低重构质量与标签恢复能力，同时尽量保持全局模型精度

## 3. 系统与数据设置

### 3.1 数据与模型

- 数据：MNIST（灰度 1×28×28）
- 模型：`fl_privacy.models.SimpleCNN`
- 损失：CrossEntropy

### 3.2 联邦划分（非IID）

训练集用 Dirichlet 按标签划分到多个客户端：

- 参数：`dirichlet_alpha`，越小越非IID
- 实现：`fl_privacy.data.dirichlet_partitions`

### 3.3 训练协议（FedSGD）

每一轮：

1. 服务器采样 `clients_per_round` 个客户端
2. 每个客户端取一个本地 mini-batch，计算梯度列表 `g_k = ∇_θ ℓ(θ; x_k, y_k)`
3. 客户端对 `g_k` 做防御处理得到 `ĝ_k` 并上传
4. 服务器做均值聚合 `ĝ = mean_k(ĝ_k)` 并做 SGD 更新 `θ ← θ − lr·ĝ`

说明：该流程是教学版最小实现。若切换为 FedAvg（多步本地训练、上传参数增量）会让攻击实现更复杂，本项目暂不作为默认。

## 4. 攻击实现：DLG / iDLG 梯度反演

### 4.1 核心思想

攻击者拿到观测梯度 `g_obs`，构造可学习的伪输入 `x̂`（以及标签 `ŷ`），通过优化使其梯度与观测梯度匹配：

`min_{x̂} Σ_l || ∇_θ ℓ(θ; x̂, ŷ)_l − g_obs,l ||^2 + λ·TV(x̂)`

### 4.2 标签推断（iDLG 思路）

本实现支持两种模式：

- 不提供真标签：用最后一层 bias 的梯度做 iDLG 式的标签推断（默认）
- 提供真标签：`--use-true-label` 仅用于“攻击上限”演示

实现：`fl_privacy.attack.dlg_reconstruct`

## 5. 防御实现

本项目把“防御对象”统一定义为：客户端上传给服务器聚合的更新向量（本实现默认是梯度列表）。

### 5.1 Clipping（基线）

对整条更新向量做全局 L2 裁剪：

- 输入：`g`
- 输出：`clip(g, C)`

实现：`fl_privacy.defenses.clip_by_global_norm`

### 5.2 DP-light（基线）

教学版的“裁剪 + 高斯噪声”：

1. `ḡ = clip(g, C)`
2. `ĝ = ḡ + N(0, (σ·C)^2 I)`

实现：`fl_privacy.defenses.dp_light`

说明：这不是完整 DP 训练管线（未计算隐私预算、未做采样校准），只作为“加法噪声基线”用于课堂对照。

### 5.3 LAUGD（核心方法）

对每个参数张量（可视作“层”）分别计算尖峰/集中度分数并自适应决定失活率：

#### 分数（集中度）

对该层梯度展平向量 `g_l ∈ R^{d_l}`：

`s_l = sqrt(d_l) * ||g_l||_2 / (||g_l||_1 + ε)`

直觉：越尖峰（少数坐标承担主要能量），`||g||_1` 相对更小，从而 `s_l` 更大。

#### 从分数到失活率

`p_l = clip( α (s_l − τ), 0, p_max )`

#### 随机失活与无偏缩放

逐元素掩码 `m ~ Bernoulli(1 − p_l)`，输出：

- 无偏版本：`ĝ_l = (m ⊙ g_l) / (1 − p_l)`，满足 `E[ĝ_l]=g_l`
- 消融版本：`ĝ_l = (m ⊙ g_l)`（不做无偏缩放）

实现：`fl_privacy.defenses.laugd`

## 6. 消融实验设计

本项目内置的可复现消融设置（`scripts/run_suite.py`）：

1. `baseline`：无防御
2. `clipping`：仅裁剪
3. `dp_light`：裁剪 + 高斯噪声
4. `laugd`：自适应分数 + Bernoulli 掩码 + 无偏缩放
5. `laugd_no_unbias`：移除无偏缩放（观察训练稳定性变化）
6. `laugd_fixed_p`：固定失活率（观察“自适应分数”的贡献）
7. `laugd_fixed_budget`：固定预算掩码（数量确定、位置随机），与 Bernoulli 掩码对照

你也可以通过 `scripts/run_train.py` 的参数自行扩展：

- `--normalize-score`：把分数归一化到 [0,1]（跨层可比）
- `--mask-mode bernoulli|fixed_budget`
- `--alpha/--tau/--p-max/--fixed-p`

## 7. 指标与预期结论

### 7.1 训练效用

- `test_acc`：全局测试集准确率（每 `eval_every` 轮评估一次）

### 7.2 泄漏强度（攻击效果）

- 可视化：`recon.png`（真值 vs 重构）
- 数值指标：
  - `MSE`：越小越接近真值
  - `PSNR`：越大越接近真值
- 标签推断：`label_true` vs `label_recon`

### 7.3 预期现象（教学讲解重点）

- Baseline 下 DLG/iDLG 能重构出明显轮廓（尤其 batch size = 1）
- Clipping 与 DP-light 会降低重构质量，但需要权衡精度
- LAUGD 更偏向破坏“可匹配结构”，在相近精度下往往能让攻击更不稳定（经验性观察）
- 去掉无偏缩放或设置过大 `p_max` 会显著影响收敛

## 8. 复现步骤（建议课堂流程）

### 8.1 单配置：训练 + 捕获 + 攻击

1) 训练并保存攻击观测：

```bash
python scripts/run_train.py --defense none --rounds 30 --capture-attack --attack-round 0 --attack-client 0
```

2) 对保存的更新执行攻击：

```bash
python scripts/run_attack.py --obs runs/<run_name>/attack_obs.pt --out-dir runs/<run_name>/attack --iters 800
```

### 8.2 一键对比 + 消融

```bash
python scripts/run_suite.py --rounds 30 --attack-iters 800
```

输出：

- `runs/suite_*/summary.md`：对比汇总表
- 每个 setting 的 `attack/recon.png`：可用于课堂展示

### 8.3 Sweep（体现“优越性”的推荐方式）

当多种防御在定性可视化上都“看不出来原图”时，建议用 sweep 来画隐私-效用前沿，并统计 label leakage（未知标签时的标签恢复准确率），更容易区分方法差异。

```bash
python scripts/run_sweep.py --rounds 30 --attack-iters 800 --laugd-unbiased
```

输出到 `outputs/<exp_name>/`：

- `privacy_utility_unknown.png`：未知标签攻击下 PSNR vs test_acc（含前沿）
- `privacy_utility_true.png`：已知标签攻击下 PSNR vs test_acc（含前沿）
- `label_leak_unknown.png`：未知标签攻击下标签恢复准确率
- `rows.json`：全量样本记录（配置×seed×round×client×attack_mode）

## 9. 代码对应关系

- 数据与划分：`fl_privacy/data.py`
- 模型：`fl_privacy/models.py`
- 联邦训练（FedSGD 核心函数）：`fl_privacy/fl.py`
- 防御：`fl_privacy/defenses.py`
- 攻击：`fl_privacy/attack.py`
- 脚本：
  - 训练：`scripts/run_train.py`
  - 攻击：`scripts/run_attack.py`
  - 批量对比与消融：`scripts/run_suite.py`
