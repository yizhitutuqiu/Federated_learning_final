# Federated Learning Privacy（教学项目实现）

本目录实现了一个可复现的联邦学习隐私安全教学项目：在“服务器可见单客户端更新”的威胁模型下，复现梯度反演攻击，并对比 Clipping / DP-light / LAUGD（Leakage-Aware Unbiased Gradient Dropout）及若干消融。

更完整的项目说明（含 3 攻击×6 防御、sweep/suite、指标解释与常见问题）见：[PROJECT.md](file:///data/litengmo/ml-test-1/Federated_learning_final/PROJECT.md)。

## 目录

- `fl_privacy/`：核心实现（数据划分、联邦训练、攻击、防御、指标）
- `scripts/`：可直接运行的训练/攻击/批量实验脚本
- `research/`：选题报告与PPT草稿

## 依赖

环境已包含 PyTorch 与 torchvision。若需要显式安装：

```bash
pip install -r requirements.txt
```

## 1) 训练并捕获一次攻击观测

默认是 FedSGD 风格：每轮每个客户端上传“梯度列表”（更适合教学版 DLG 攻击复现）。

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

输出会打印 `runs/<run_name>` 路径，其中包含：

- `metrics.jsonl`：训练/测试指标
- `attack_obs.pt`：被攻击者可见的更新（已经过防御处理）+ 对应真实样本

## 2) 对捕获的更新执行梯度反演（DLG/iDLG）

```bash
python scripts/run_attack.py \
  --obs runs/<run_name>/attack_obs.pt \
  --out-dir runs/<run_name>/attack \
  --iters 800
```

输出目录包含：

- `recon.png`：真值 vs 重构图
- `attack_metrics.json`：PSNR/MSE 与标签推断结果

## 3) 一键跑完对比 + 消融（建议课堂演示）

```bash
python scripts/run_suite.py --rounds 30 --attack-iters 800
```

会在 `runs/suite_*/summary.md` 生成表格汇总，并在每个 setting 目录下保存 `attack/recon.png`。

## 4) 一键生成 loss 曲线（baseline / baseline+attack / baseline+attack+defense）

会在 `outputs/<exp_name>/loss_curves.png` 生成训练损失曲线图，并把每个 run 的目录写入 `outputs/<exp_name>/runs.json`。

```bash
python scripts/run_compare_plot.py --device auto --defense laugd --unbiased
```

### 4.1 把结果变“直观可看”（重构拼图 + 简单指标表）

对已有的 `outputs/<exp_name>/runs.json`，生成：

- `recon_compare.png`：baseline+attack vs baseline+attack+defense 的重构图拼图
- `overview.md`：直接可读的一页报告（内嵌 loss 曲线与重构图）

```bash
python scripts/make_visual_report.py --compare-dir outputs/<exp_name>
```

## 5) Sweep：隐私-效用前沿 + 标签泄漏统计（推荐用来体现方法差异）

会自动对 Clipping / DP-light / LAUGD 做参数网格，并在多个 seed×客户端×轮次上重复采样，输出隐私-效用散点图（含前沿）与 label leakage 统计图到 `outputs/<exp_name>/`。

```bash
python scripts/run_sweep.py --device auto --laugd-unbiased
```

主要输出：

- `outputs/<exp_name>/rows.json`：全部样本记录
- `outputs/<exp_name>/privacy_utility_unknown.png`：未知标签攻击下的 PSNR vs test_acc
- `outputs/<exp_name>/privacy_utility_true.png`：已知标签攻击下的 PSNR vs test_acc
- `outputs/<exp_name>/label_leak_unknown.png`：未知标签攻击下的标签恢复准确率

## 6) 设计与实验方案

详见：[design_experiments.md](file:///data/litengmo/ml-test-1/Federated_learning_final/design_experiments.md)
