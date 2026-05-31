# Sweep 汇总：sweep_smoke

- 输出文件：
  - rows.json：全部样本（配置×seed×round×client×attack_mode）
  - privacy_utility_*.png：隐私-效用散点与前沿
  - label_leak_unknown.png：标签恢复准确率对比

建议解读：同等 test_acc 下 PSNR 越低（或 MSE 越高）越隐私；unknown_label 模式下 label recovery 越低越好。
