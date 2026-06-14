# Sweep 汇总：tiny_old_vs_v2

- 输出文件：
  - rows.json：全部样本（配置×seed×round×client×attack_mode）
  - agg_rows.json：按 (family, config, attack_mode) 聚合后的点
  - privacy_utility_*.png：隐私-效用散点与前沿
  - label_leak_unknown.png：标签恢复准确率对比
  - pareto_*.md：全局 Pareto 前沿表格（严谨判断是否存在非支配点）

建议解读：同等 test_acc 下 PSNR 越低（或 MSE 越高）越隐私；unknown_label 模式下 label recovery 越低越好。
