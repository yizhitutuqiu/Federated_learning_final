# 联邦学习隐私安全（教学小项目）：一种新的轻量防御策略

作者：（拟）博士后研究者（隐私安全方向）  
日期：2026-05-13  
目录位置：`Federated_learning_final/research/report.md`

## 0. 摘要

联邦学习（Federated Learning, FL）并不天然“隐私安全”：即便数据不出本地，**客户端上传的梯度/模型更新**仍可能被用于成员推断、属性推断或梯度反演（gradient inversion）等攻击。教学任务中，一个常见难点是：现有防御要么依赖较重的密码协议/系统假设，要么实现复杂，难以在课堂里快速讲清楚并复现实验现象。

本报告将项目规模缩小为“提出并验证一个新的轻量防御策略”，目标面向教学：**不改协议、只改客户端上传更新的处理方式**，让学生能在一个小实验里观察到“攻击显著变难，但训练仍能进行”的现象。

我们提出一种新的防御：**LAUGD（Leakage-Aware Unbiased Gradient Dropout，泄露感知的无偏梯度随机失活）**。核心思想是：梯度反演往往利用更新中“高保真、细粒度”的信息；我们对每一层梯度做一个**由“泄露风险分数”自适应决定强度的随机掩码**，并用无偏缩放保持期望梯度不变，从而把“可用于重构的确定性细节”变成“训练可容忍的随机噪声/缺失”，在不引入复杂加密的情况下显著降低泄露。

---

## 1. 背景与问题定义

### 1.1 典型威胁面

在现代FL系统中，隐私与安全威胁至少来自三类观察者/对手：

1. **好奇但半诚实的服务器（honest-but-curious server）**：遵循协议但试图从可见信息（明文更新、聚合结果、元数据）推断用户数据。
2. **恶意客户端/串谋客户端（malicious clients, collusion）**：通过构造更新实施投毒、后门，或与服务器串谋提升推断攻击成功率。
3. **旁路观察者（eavesdropper）与系统侧信息**：包括通信元数据、参与频率、掉线模式等“非梯度信息”带来的隐私泄露。

### 1.2 教学场景下的简化目标

为了降低工作量与讲解复杂度，本项目做如下教学化简：

- 只考虑最典型的泄露威胁：**服务器或观察者看到单个客户端更新**，并尝试做梯度反演/隐私推断（攻击参考SoK与近期攻击研究）。见：[1,2]
- 防御设计约束：**不依赖SecAgg/密码学/可信硬件**，只允许“客户端侧对上传更新做处理”。
- 输出形式：一个简单防御算法 + 一套最小实验（例如MNIST/CIFAR小网络）展示“防御有效性与效用权衡”。

---

## 2. 新防御策略：LAUGD（泄露感知的无偏梯度随机失活）

### 2.1 威胁模型（明确假设）

- 攻击者能力：攻击者能拿到某一轮某一客户端的完整更新向量（梯度或参数增量），并运行梯度反演/更新匹配类攻击来重构输入或推断标签/属性。见：[1,2]
- 防御目标：降低反演重构质量与标签/属性恢复准确率，同时尽量保持训练收敛与精度。
- 非目标：在“服务器完全恶意 + 可自适应诱导客户端 + 拿到本地训练过程全部中间量”的最强攻击下给出形式化保证（教学任务不追求这一层）。

### 2.2 设计直觉

梯度反演依赖“高约束”的优化：攻击者寻找一个输入，使得其产生的梯度与观测梯度尽可能一致。若观测梯度里有大量**细粒度确定性信息**（例如某些层的尖峰分量、与单样本强相关的通道），匹配问题更容易；反之，如果这些信息被转化为**随机缺失/随机扰动**，攻击就会变得不稳定，需要更多先验与更长优化。

LAUGD的核心点：

1. 用一个简单可计算的“泄露风险分数”衡量某层梯度是否“尖峰/集中”（越集中通常越容易携带可反演信息）。
2. 分数越高，该层就越强地做随机失活（dropout），但用无偏缩放保证期望梯度不变，使训练仍可进行。

### 2.3 泄露风险分数（简单、可讲清楚）

对第 l 层梯度向量 \(g_l \in \mathbb{R}^{d_l}\)，定义“集中度分数”：

\[
s_l \;=\; \frac{\sqrt{d_l}\,\lVert g_l\rVert_2}{\lVert g_l\rVert_1 + \epsilon}
\]

- 当梯度很均匀时，\(\lVert g\rVert_1 \approx \sqrt{d}\lVert g\rVert_2\)，则 \(s_l \approx 1\)
- 当梯度极度尖峰（少量大分量）时，\(\lVert g\rVert_1\) 相对更小，\(s_l > 1\) 更大

然后把 \(s_l\) 映射成失活率 \(p_l \in [0,p_{\max}]\)：

\[
p_l \;=\; \mathrm{clip}\big(\alpha\,(s_l-\tau),\,0,\,p_{\max}\big)
\]

其中 \(\alpha>0\) 控制敏感度，\(\tau\) 是阈值（教学中可用网格搜索/经验设定）。

### 2.4 LAUGD算法（客户端侧）

对每一轮、每一客户端：

1. 计算本地更新（梯度或参数增量），按层拆分为 \(\{g_l\}_{l=1}^L\)
2. 对每层算 \(s_l\)，得到失活率 \(p_l\)
3. 采样掩码 \(m_l \sim \mathrm{Bernoulli}(1-p_l)^{d_l}\)（逐元素独立）
4. 无偏缩放并输出：

\[
\tilde g_l \;=\; \frac{m_l \odot g_l}{1-p_l}
\]

5. 将 \(\tilde g=\{\tilde g_l\}\) 上传到服务器聚合（与常规FedAvg完全兼容）

可选增强（仍保持教学简洁）：

- 在输出前做**全局范数裁剪**（standard clipping），再叠加少量高斯噪声，形成“LAUGD + 轻量DP”的组合，便于对比DP的作用。见：[3]

### 2.5 为什么它算“新的”（与常见做法的区别）

- 与固定比例稀疏化/Top-k不同：LAUGD是**按层自适应**，且依据一个明确可解释的泄露风险分数。
- 与直接加DP噪声不同：LAUGD通过“随机缺失 + 无偏估计”改变观测梯度结构，往往能在相同训练损失下更明显地破坏反演的确定性匹配。
- 与SVD/投影类防御不同：LAUGD不需要矩阵分解或额外公共数据，课堂实现成本更低。对照可参考：[4,5,6]

---

## 3. 教学实验设计（最小可复现）

### 3.1 实验任务与模型（建议）

- 数据：MNIST 或 CIFAR-10（教学更偏MNIST易复现）
- 模型：2-3层CNN + 1个全连接分类头
- 联邦设置：10–50个客户端，Dirichlet划分模拟非IID

### 3.2 攻击（选一个主攻，保证能跑通）

- 梯度反演/更新匹配攻击：先做SGD场景；如果时间充裕，再尝试“超越SGD”的攻击设定作为bonus。见：[2]

### 3.3 指标（既教学又直观）

- 效用：全局测试准确率曲线
- 隐私：反演重构质量（例如PSNR/SSIM/LPIPS，或肉眼可视化），以及标签恢复准确率
- 额外：通信量变化（LAUGD不改变维度，通信量不变；但可扩展为“掩码+压缩”加分）

### 3.4 对比组（最少四组就够）

1. Baseline：无防御
2. Clipping：仅做范数裁剪
3. DP-light：裁剪 + 高斯噪声（弱到中等强度）
4. LAUGD：本文方法（可再加 LAUGD + DP-light）

---

## 4. 局限性与讨论（课堂要强调的点）

1. LAUGD不是形式化隐私保证：它更像“降低泄露信号/增加攻击不确定性”的工程防护。
2. 如果攻击者拥有更强先验（例如知道数据分布、能观测多轮同一客户端、能诱导客户端参与），LAUGD效果可能下降；这正是课堂中“威胁模型很重要”的教学点。
3. LAUGD提高了梯度估计方差：失活率过大可能导致收敛变慢或精度下降；因此需要展示“p 的强度—效用”曲线。

---

## 5. 相关工作定位（为什么需要对照）

本策略位于“轻量防御”谱系中，适合教学对照：

- 梯度反演攻击与系统化整理：帮助学生理解攻击依赖哪些信息。见：[1,2]
- SVD/投影/两阶段更新等防御：可作为“更强但更复杂”的对照。见：[4,5,6]
- DP作为“可解释、可量化”的基线：用来解释形式化保证与工程防护的差异。见：[3]

---

## 6. 课堂交付物（建议）

1. 一页讲义：攻击直觉 + LAUGD公式 + 参数含义
2. 一个最小实验脚本/笔记本：Baseline vs DP-light vs LAUGD（+可选组合）
3. 一张结果图：训练精度曲线 + 反演可视化对比
4. 一段讨论题：威胁模型变化时，LAUGD为什么可能失效？如何与DP/SecAgg组合？

---

## 参考文献（近两年优先）

[1] SoK: On Gradient Leakage in Federated Learning, USENIX Security 2025（页面访问可能受验证限制）. https://www.usenix.org/conference/usenixsecurity25/presentation/du  
[2] Gradient Inversion Attacks Beyond SGD, OpenReview. https://openreview.net/forum?id=uLdGZhxlxV  
[3] Secure Stateful Aggregation: A Practical Protocol with Applications in Differentially-Private Federated Learning, arXiv 2410.11368, 2024. https://arxiv.org/html/2410.11368v1  
[4] SelectiveShield: Lightweight Hybrid Defense Against Gradient Leakage in Federated Learning, arXiv 2508.04265, 2025. https://arxiv.org/html/2508.04265v1/  
[5] SVDefense: Effective Defense against Gradient Inversion Attacks via Singular Value Decomposition, arXiv 2510.03319, 2025. https://arxiv.org/html/2510.03319v1/  
[6] GradientHide: Federated Learning with Two-Stage Local Update for Defending Against Gradient Inversion Attacks, OpenReview. https://openreview.net/forum?id=hcL8wbKFB5  
