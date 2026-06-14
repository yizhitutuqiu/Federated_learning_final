# Pareto 前沿（mode=unknown_label，目标：test_acc↑ 且 PSNR↓）

说明：每个点先在 (family, config, mode) 内做平均聚合，再做全局非支配筛选。

## 非支配点数量（按方法）

| family | count |
|---|---:|
| baseline | 1 |
| dp_light | 1 |
| laugd | 1 |

## 前沿点表格

| family | config | n | test_acc | PSNR | MSE | label_acc |
|---|---|---:|---:|---:|---:|---:|
| baseline | baseline | 1 | 0.1378 | 5.59 | 0.275838 | 1.000 |
| dp_light | dp_nm0.1 | 1 | 0.0998 | 5.54 | 0.279342 | 1.000 |
| laugd | laugd_a1.0_t1.0_pmax0.2 | 1 | 0.0979 | 5.53 | 0.280220 | 1.000 |
