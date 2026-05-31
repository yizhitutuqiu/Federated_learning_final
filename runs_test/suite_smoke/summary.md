# 实验套件汇总：suite_smoke

| setting | test_acc | PSNR↑ | MSE↓ | label_true | label_recon | run_dir |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0.1378 | 5.67 | 0.271279 | 0 | 0 | runs_test/suite_smoke__baseline |
| clipping | 0.0900 | 5.56 | 0.277898 | 0 | 0 | runs_test/suite_smoke__clipping |
| dp_light | 0.0998 | 5.55 | 0.278508 | 0 | 0 | runs_test/suite_smoke__dp_light |
| laugd | 0.0972 | 5.49 | 0.282724 | 0 | 0 | runs_test/suite_smoke__laugd |
| laugd_no_unbias | 0.1017 | 5.59 | 0.276017 | 0 | 0 | runs_test/suite_smoke__laugd_no_unbias |
| laugd_fixed_p | 0.1061 | 5.51 | 0.281321 | 0 | 0 | runs_test/suite_smoke__laugd_fixed_p |
| laugd_fixed_budget | 0.0972 | 5.49 | 0.282434 | 0 | 0 | runs_test/suite_smoke__laugd_fixed_budget |
