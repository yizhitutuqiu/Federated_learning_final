# Federated Learning Privacy：梯度反演攻击与轻量防御（教学项目完整说明）

本项目是一个面向教学的“可复现实验小项目”：在联邦学习中假设服务器（或窃听者）能看到**单个客户端在单轮上传的更新**，则可以通过梯度反演（Gradient Inversion / DLG 系列）重建客户端的私有训练样本；我们实现并对比多种轻量防御（含 LAUGD v1/v2），并提供 suite / sweep / 可视化报告脚本，帮助你用一条命令跑完实验、得到“隐私-效用权衡”图和表格。

如果你只想快速跑通流程，先看“快速开始”；如果你要做答辩/课堂讲解，建议按“概念地图 → 复现实验 → 如何解读结果”的顺序阅读。

---

## 1. 你会得到什么

- 一个最小可运行的联邦学习（教学版 FedSGD）训练管线：多客户端、Dirichlet 非IID 划分、服务器聚合更新
- 三种梯度反演攻击（unknown-label/true-label 两种口径）：
  - DLG（MSE 梯度匹配）
  - IG（cosine 梯度匹配）
  - IG + LBFGS（cosine 目标不变，优化器换 LBFGS）
- 六种防御对比（默认 suite 模式 paper6）：
  - baseline（无防御）
  - clipping（全局范数裁剪）
  - dp_light（裁剪 + 高斯噪声，教学基线）
  - svd（低秩投影）
  - laugd_v1（LAUGD：按层泄漏分数 → 自适应 dropout + 无偏缩放）
  - laugd_v2（LAUGD v2：在 v1 上增加 pre-clip、结构化掩码与头部强化等）
- 可复现的评估与可视化：
  - 训练效用：test_acc / loss 曲线
  - 隐私泄漏：重建图 recon.png、PSNR/MSE、label_ok（标签是否被推断正确）
  - 隐私-效用散点图（含 Pareto 前沿表）

---

## 2. 项目结构（从“我该看哪里”出发）

- 核心实现：`fl_privacy/`
  - 攻击：`fl_privacy/attack.py`
  - 防御：`fl_privacy/defenses.py`
  - 训练与联邦：`fl_privacy/fl.py`、`fl_privacy/data.py`、`fl_privacy/models.py`
  - 指标：`fl_privacy/metrics.py`
- 可运行脚本：`scripts/`
  - 训练并捕获攻击观测：`scripts/run_train.py`
  - 运行攻击并产出指标/可视化：`scripts/run_attack.py`
  - 一键跑“多防御×多攻击”的对比表：`scripts/run_suite.py`
  - 网格 sweep + 隐私-效用散点 + Pareto：`scripts/run_sweep.py`
  - 三条 loss 曲线对比：`scripts/run_compare_plot.py`
  - 把结果变成“更直观的一页报告”：`scripts/make_visual_report.py`
  - 生成更丰富的汇总材料：`scripts/make_new_table_report.py`
- 教学材料：`research/`
  - 选题报告：`research/report.md`
  - 演示稿草稿：`research/ppt.md`

输出目录：
- 单次训练/攻击的中间产物：`runs/`
- 汇总图/表/报告：`outputs/`

---

## 3. 概念地图：层、维、组（以及“我们到底在 mask 什么”）

### 3.1 训练时客户端上传的是什么

本项目默认是教学版 FedSGD：每一轮，客户端在一个 mini-batch 上计算梯度列表并上传（更适合讲清楚 DLG）。

- 客户端上传的更新：`update = [g_0, g_1, ..., g_L]`
- 每个 `g_l` 都是一个张量（对应某个参数张量的梯度，如 conv.weight、fc.bias）

### 3.2 “层（layer）”是什么

在本项目里，“层”指 `update` 这个 list 里的**一个张量** `g_l`（严格说是“参数张量”，为了教学简化称为层）。

我们始终是：
- 先按层计算泄漏分数 `score_l`
- 再把分数映射成该层 dropout 概率 `p_l`
- 再在这一层内部生成 mask，并对该层梯度做变换

### 3.3 “维（dimension）/ 0 维”是什么

张量 `g_l` 有形状 shape：
- Conv 权重梯度：`[out_channels, in_channels, kH, kW]`
- Linear 权重梯度：`[out_features, in_features]`

“0 维”就是 shape 的第一个维度，即：
- Conv 的 `out_channels`
- Linear 的 `out_features`

### 3.4 “组（group）”是什么（结构化 mask 的关键）

当我们使用结构化 mask（`mask_mode="channel"`）时：
- 不再对每个元素分别掩码
- 而是把这一层按 0 维切成很多块：第 i 块就是 `g_l[i, ...]`
- 每个块就是一个“组”

换句话说：
- 逐元素 mask：随机性粒度是元素 `g_l[i,j,k,...]`
- 按组 mask：随机性粒度是组 `g_l[i, ...]`（整块一起丢/一起留）

---

## 4. 威胁模型与攻击口径（unknown_label vs true_label）

### 4.1 威胁模型（教学版）

- 攻击者：honest-but-curious 服务器 / 窃听者
- 可见信息：某轮某客户端上传的更新（在本项目中默认是“梯度列表”）
- 攻击目标：从更新中重建输入 `x`，或推断标签 `y`

### 4.2 两种攻击口径

- unknown_label：攻击者**拿不到 label**，需要同时推断 label（默认更贴近隐私讨论）
- true_label：攻击者已知 label（作为攻击上限对照）

在本项目中：
- `scripts/run_attack.py` 默认是 unknown_label；加 `--use-true-label` 切到 true_label
- unknown_label 时会用 iDLG 思路从最后一层梯度推断标签（见 `fl_privacy.attack.infer_label_idlg`）

---

## 5. 攻击实现（3 种）

代码位置：`fl_privacy/attack.py`。

### 5.1 DLG（MSE 梯度匹配）

- 目标：让伪输入产生的梯度 `grads_hat` 在每一层上尽量接近观测梯度 `grads_obs`
- 匹配项：逐层 MSE（`mean((gh - go)^2)`）
- 优化：Adam 直接优化 `x_var`，并加 TV 正则提升视觉平滑

入口：`dlg_reconstruct`（同时在 unknown_label 时进行标签推断）。

### 5.2 IG（cosine 梯度匹配）

- 目标：用余弦相似度对齐梯度方向，避免受尺度影响
- 匹配项：逐层 `1 - cos(gh, go)`
- 优化：Adam；可选 `--l2-reg` 做输入 L2 正则

入口：`ig_reconstruct`。

### 5.3 IG + LBFGS（同目标，不同优化器）

- 目标：仍然是 IG 的 cosine matching
- 不同点：优化器换为 `torch.optim.LBFGS(..., line_search_fn="strong_wolfe")`

入口：`lbfgs_reconstruct`。

---

## 6. 防御实现（6 种对比 + 1 个可选 2024 基线）

代码位置：`fl_privacy/defenses.py`，训练侧入口是 `scripts/run_train.py` 的 `--defense` 参数。

### 6.1 baseline（无防御）

客户端直接上传原始梯度列表。

### 6.2 clipping（全局范数裁剪）

对整条更新（所有层拼起来）做全局 L2 范数裁剪：
- `g <- g * min(1, C / ||g||_2)`

### 6.3 dp_light（裁剪 + 高斯噪声）

教学版 DP 基线（不做隐私会计）：
- 先全局裁剪到 `C`
- 再加噪：`g <- g + N(0, (σC)^2 I)`

### 6.4 svd（低秩投影）

对每个二维及以上张量做 reshape 后 SVD，只保留前 `rank_ratio` 的主成分再还原，降低细粒度信息。

### 6.5 laugd_v1（LAUGD：Leakage-Aware Unbiased Gradient Dropout）

核心思路：尖峰/集中度越强的层越容易泄漏，因此对该层更大概率地随机“打洞”。

1) 逐层集中度分数（越大越尖峰）：

- 展平 `g_l` 得到长度 `d_l` 的向量
- 分数：
  - `s_l = sqrt(d_l) * ||g_l||_2 / (||g_l||_1 + eps)`

2) 分数映射为该层 dropout 概率：
- `p_l = clip(alpha * (s_l - tau), 0, p_max)`

3) 在层内做随机掩码，并做无偏缩放：
- 逐元素掩码（默认）：`m ~ Bernoulli(1 - p_l)`，`g_l <- (m ⊙ g_l) / (1 - p_l)`
- 也支持 fixed_budget：固定保留数量、位置随机，缩放用 `k_keep / d_all`

### 6.6 laugd_v2（v1 + 稳定性与“头部保护”增强）

v2 仍沿用“按层算分数 → 得到 p_l → 掩码 + 无偏缩放”的主框架，但更偏向 unknown_label 场景的攻击：

- pre-clip：在做 mask 前先全局裁剪，提升训练稳定性
- 结构化掩码：`mask_mode="channel"`，对张量第 0 维按“整组”采样（更结构化、方差更低）
- 头部强化：对最后若干层把该层 p 乘以 `head_mult`（以及对最后一层 `last_layer_mult`），优先破坏标签相关信息

### 6.7 （可选）RGM 2024：Random Gradient Masking（固定 p 的随机掩码）

如果你想额外做“2024 及以后”的基线对比，本项目提供了一个教学版 RGM：
- 不算泄漏分数
- 对每层使用同一个固定丢弃率 `p` 随机置零，并做无偏缩放

实现上复用了 `laugd`：通过 `--fixed-p p --p-max p` 把“自适应”退化为“固定 p 的随机 mask”。

---

## 7. 指标怎么读（你最常用的 4 个）

### 7.1 test_acc（效用）

全局测试集准确率，越大越好，代表模型训练质量。

### 7.2 PSNR / MSE（隐私泄漏强度）

用于衡量“重建图”与真值图的距离：

- MSE：越小重建越准，因此越小越不隐私；越大越隐私
- PSNR：和 MSE 单调相反（对图像更直观），越大重建越准，因此越大越不隐私；越小越隐私

### 7.3 label_ok（标签是否被推断正确）

`label_ok = 1[label_recon == label_true]`。

在 unknown_label 口径下：
- label_ok 越高，代表标签越容易从梯度中被推断出来（更不隐私）
- 防御希望 label_ok 越低越好（尤其是头部层相关策略）

### 7.4 recon.png / loss_curve.png（展示用）

`scripts/run_attack.py` 会输出：
- `recon.png`：真值 vs 重建对比图
- `loss_curve.png`：重建优化过程的 loss 曲线（用于展示“攻击是否稳定收敛”）

---

## 8. 快速开始（最常用的几条命令）

### 8.1 安装依赖

```bash
cd /data/litengmo/ml-test-1/Federated_learning_final
pip install -r requirements.txt
```

### 8.2 单次演示：训练 + 捕获 + 攻击

1) 训练并捕获一次攻击观测（默认 batch_size=1 更容易重建）：

```bash
python scripts/run_train.py \
  --defense none \
  --rounds 30 \
  --num-clients 20 \
  --clients-per-round 5 \
  --batch-size 1 \
  --capture-attack \
  --attack-round 0 \
  --attack-client 0
```

2) 对捕获结果执行攻击（unknown_label 默认）：

```bash
python scripts/run_attack.py \
  --obs runs/<run_name>/attack_obs.pt \
  --out-dir runs/<run_name>/attack_dlg/unknown_label \
  --iters 800 \
  --method dlg
```

### 8.3 一键对比（6 防御 × 3 攻击）

```bash
python scripts/run_suite.py \
  --mode paper6 \
  --attack-methods dlg,ig,lbfgs \
  --attack-modes unknown_label \
  --rounds 30 \
  --attack-iters 800
```

输出：
- `runs/<suite_name>/summary.md`（表格）
- 每个配置：`runs/<suite_name>__<setting>/attack_<method>/unknown_label/`（含 recon.png、attack_metrics.json）

如果中途断了，可以在相同 `suite-name` 下加 `--resume` 继续（已完成的配置会跳过）：

```bash
python scripts/run_suite.py --mode paper6 --suite-name <suite_name> --resume
```

### 8.4 sweep：隐私-效用散点 + Pareto 前沿

```bash
python scripts/run_sweep.py \
  --device auto \
  --attack-method ig \
  --attack-modes unknown_label \
  --laugd-unbiased \
  --include-laugd-v2 \
  --include-svd
```

输出到 `outputs/<exp_name>/`：
- `privacy_utility_unknown.png`、`privacy_utility_unknown_mse.png`
- `pareto_unknown_label.md`
- `agg_rows.json`

---

## 9. “suite” vs “sweep” 的区别（什么时候用哪个）

- suite：固定少量配置，跑完直接给一个表；更适合课堂演示、或你已经确定一组超参
- sweep：对超参做网格，并在多个 seed×轮次×客户端采样，最终画散点图与 Pareto；更适合做“我们方法是否存在非支配点”的严谨对比

---

## 10. 常见问题与坑位（遇到报错先看这里）

### 10.1 为什么 baseline+attack 会影响训练曲线

当启用捕获攻击观测时，训练脚本会强制“必须包含某个客户端/轮次”以保证能捕获到指定客户端的更新；这会改变客户端采样，从而影响训练轨迹（loss 曲线会不同）。

### 10.2 batch_size 设成 32 还能攻击吗

可以，但为了保持“单样本重建”的口径，`scripts/run_attack.py` 会自动只取 batch 的第一个样本进行攻击与计算指标。

### 10.3 `.gitignore` 里为什么忽略 `*.pt`

攻击观测 `attack_obs*.pt` 可能很大且包含潜在敏感信息，默认不建议提交到版本库。

---

## 11. 代码导航（你要改哪里）

- 你要改攻击（更强/更快/换目标）：看 `fl_privacy/attack.py`、`scripts/run_attack.py`
- 你要改防御（LAUGD 变体/结构化 mask/头部策略）：看 `fl_privacy/defenses.py`、`scripts/run_train.py`
- 你要改“批量实验怎么跑/怎么汇总”：看 `scripts/run_suite.py`、`scripts/run_sweep.py`

---

## 12. 参考（用于“方法来源/年份”写报告）

- DLG：Zhu et al., “Deep Leakage from Gradients”, NeurIPS 2019
- IG：Geiping et al., “Inverting Gradients – How easy is it to break privacy in federated learning?”, NeurIPS 2020
- RGM（随机梯度掩码，2024）：Joon Kim et al., “Random Gradient Masking as a Defensive Measure to Deep Leakage in Federated Learning”, 2024（arXiv:2408.08430）

