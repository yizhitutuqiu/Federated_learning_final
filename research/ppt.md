# PPT草稿：联邦学习隐私安全（教学小项目）——LAUGD轻量防御策略

---

## 1. 标题页

- 题目：联邦学习隐私安全（教学小项目）：LAUGD 轻量防御策略
- 副标题：Leakage-Aware Unbiased Gradient Dropout
- 作者：（拟）博士后研究者（隐私安全方向）
- 日期：2026-05-13

---

## 2. 为什么要做（动机）

- 联邦学习 ≠ 天然隐私安全
- 即使“数据不出本地”，客户端上传的梯度/更新仍可能泄露信息
- 教学痛点
  - 现有防御要么系统/密码学复杂
  - 要么难以用一个小实验快速复现“攻击→防御”的核心现象
- 目标：提出一个“可讲清楚、可复现、改动小”的新防御

---

## 3. 典型攻击现象（想让学生看到什么）

- 梯度反演（gradient inversion / update matching）
  - 攻击者拿到某一轮某客户端的更新
  - 通过优化找到“能产生同样梯度”的输入
  - 可能重构出图像轮廓、标签，甚至敏感属性
- 结论：上传更新里包含“高保真、细粒度”的泄露信号

---

## 4. 威胁模型（教学版）

- 攻击者：服务器或观察者（honest-but-curious）
- 可见信息：单轮、单客户端更新向量（梯度或参数增量）
- 攻击能力：运行梯度反演/更新匹配攻击
- 防御约束：不改协议，不用SecAgg/TEE/复杂加密
- 防御目标
  - 降低反演重构质量、标签/属性恢复准确率
  - 尽量保持训练收敛与精度

---

## 5. 设计直觉（一句话）

- 把“可用于重构的确定性细节”
  - 变成“训练可容忍的随机缺失/随机扰动”
  - 同时保持更新在期望意义上不偏（unbiased）

---

## 6. 我们的方法：LAUGD（概览）

- 全称：Leakage-Aware Unbiased Gradient Dropout
- 做法：客户端侧按层处理梯度
  1. 计算每层“泄露风险分数”（是否尖峰/集中）
  2. 风险越高，随机失活越强（更大 dropout rate）
  3. 用无偏缩放保持期望梯度不变
- 与FedAvg兼容：服务器端无需修改

---

## 7. 泄露风险分数（可解释）

对第 l 层梯度向量 \(g_l \in \mathbb{R}^{d_l}\)，定义集中度分数：

\[
s_l=\frac{\sqrt{d_l}\,\lVert g_l\rVert_2}{\lVert g_l\rVert_1+\epsilon}
\]

- 直觉
  - 梯度越“尖峰”，信息越集中，越可能携带可反演细节
  - 越“均匀”，单个坐标携带的确定性信息更少

---

## 8. 从分数到失活率（自适应强度）

\[
p_l=\mathrm{clip}(\alpha\,(s_l-\tau),\,0,\,p_{\max})
\]

- \(\tau\)：风险阈值（超过才加强失活）
- \(\alpha\)：敏感度（控制增长速度）
- \(p_{\max}\)：上限（防止训练崩）

---

## 9. 核心算法（无偏随机失活）

采样掩码：

\[
m_l \sim \mathrm{Bernoulli}(1-p_l)^{d_l}
\]

输出更新（无偏缩放）：

\[
\tilde g_l=\frac{m_l\odot g_l}{1-p_l}
\]

- 性质：\(\mathbb{E}[\tilde g_l]=g_l\)
- 解释：训练仍在“期望梯度”方向上推进，但攻击看到的是随机化后的观测

---

## 10. 与常见方法对比（教学重点）

- vs 固定稀疏化 / Top-k
  - LAUGD 按层自适应，强度由风险分数驱动
- vs 直接加DP噪声
  - LAUGD 改变“可匹配结构”，不只是幅度扰动
  - 可与轻量DP叠加作为对照
- vs SVD/投影类防御
  - LAUGD 不需要矩阵分解、公共数据或额外训练阶段

---

## 11. 直觉示意（建议配图）

- 原始梯度：某些层/通道出现尖峰 → 攻击可强约束匹配
- LAUGD后：尖峰被随机打散/缺失 → 匹配优化更不稳定
- 建议配图
  - 同一层梯度的直方图：Before / After
  - 掩码比例随 \(s_l\) 变化曲线

---

## 12. 最小可复现实验（设置）

- 数据：MNIST（优先）或 CIFAR-10
- 模型：2–3层CNN + 全连接分类头
- 联邦：10–50客户端，Dirichlet 非IID划分
- 训练：FedAvg，SGD（先跑通）

---

## 13. 攻击与评估指标

- 攻击：梯度反演/更新匹配（先选一个实现跑通）
- 隐私指标（直观）
  - 重构可视化（最重要）
  - PSNR / SSIM / LPIPS（选1–2个即可）
  - 标签恢复准确率
- 效用指标
  - 全局测试准确率曲线

---

## 14. 对比组（最少四组）

1. Baseline：无防御
2. Clipping：仅范数裁剪
3. DP-light：裁剪 + 高斯噪声（弱到中等强度）
4. LAUGD：本文方法

可选加分：
- LAUGD + DP-light（展示可组合性）

---

## 15. 预期结果（要讲清楚的结论）

- 结论1：LAUGD 显著降低反演重构质量（图像更模糊/更难辨识）
- 结论2：相比 DP-light，在相近精度下降下，LAUGD 对“结构匹配型攻击”的破坏更明显（经验性观察）
- 结论3：失活强度太大 → 收敛变慢/精度下降（展示权衡曲线）

---

## 16. 参数选择建议（教学版）

- \(\tau\)：从 1.0 附近开始（因为均匀梯度时 \(s_l\approx 1\)）
- \(\alpha\)：从小到大网格（例如 0.5 / 1 / 2）
- \(p_{\max}\)：建议不超过 0.5（先保证可训练）
- 实操策略：先固定 \(p_{\max}\)，再调 \(\alpha,\tau\) 找到“隐私改善明显且精度可接受”的点

---

## 17. 局限性（课堂讨论点）

- 非形式化保证：LAUGD 是工程防护，不等价于DP的严格隐私定义
- 攻击者更强时可能失效
  - 多轮观测同一客户端
  - 更强先验/更强约束
  - 诱导式/自适应攻击
- 方差增大：本质上引入了梯度噪声/缺失，会影响收敛

---

## 18. 可扩展方向（可作为课程大作业选题）

- 与 DP 组合：LAUGD + 小噪声，比较“无偏随机缺失 vs 纯噪声”
- 与压缩组合：把掩码当作稀疏化策略，评估通信与隐私的共同收益
- 与 SecAgg 组合：讨论“可见性变化”对攻击成立条件的影响
- 超越SGD：在 Adam / FedAdam 等设置下测试攻击与防御

---

## 19. Takeaways（总结）

- 关键观点：攻击依赖“可匹配的确定性细节”
- LAUGD：用泄露风险驱动的无偏随机失活，降低反演可行性
- 教学价值：实现简单、易复现、适合课堂展示隐私-效用权衡

---

## 20. 参考文献（与报告一致）

- [1] SoK: On Gradient Leakage in Federated Learning, USENIX Security 2025. https://www.usenix.org/conference/usenixsecurity25/presentation/du
- [2] Gradient Inversion Attacks Beyond SGD, OpenReview. https://openreview.net/forum?id=uLdGZhxlxV
- [3] Secure Stateful Aggregation: A Practical Protocol with Applications in Differentially-Private Federated Learning, arXiv 2410.11368, 2024. https://arxiv.org/html/2410.11368v1
- [4] SelectiveShield: Lightweight Hybrid Defense Against Gradient Leakage in Federated Learning, arXiv 2508.04265, 2025. https://arxiv.org/html/2508.04265v1/
- [5] SVDefense: Effective Defense against Gradient Inversion Attacks via Singular Value Decomposition, arXiv 2510.03319, 2025. https://arxiv.org/html/2510.03319v1/
- [6] GradientHide: Federated Learning with Two-Stage Local Update for Defending Against Gradient Inversion Attacks, OpenReview. https://openreview.net/forum?id=hcL8wbKFB5

