from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--suite-name", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--clients-per-round", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--attack-round", type=int, default=0)
    p.add_argument("--attack-client", type=int, default=0)
    p.add_argument("--attack-iters", type=int, default=800)
    p.add_argument("--mode", type=str, default="basic", choices=["basic", "paper5", "paper6", "full"])
    p.add_argument("--attack-methods", type=str, default="dlg,ig,lbfgs")
    p.add_argument("--attack-modes", type=str, default="unknown_label,true_label")
    p.add_argument("--attack-l2-reg", type=float, default=0.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--include-rgm-2024", action="store_true")
    p.add_argument("--rgm-p", type=float, default=0.3)
    p.add_argument("--rgm-mask-mode", type=str, default="fixed_budget", choices=["bernoulli", "fixed_budget", "channel"])
    return p.parse_args()


def read_last_test_acc(metrics_jsonl: Path):
    last = None
    with open(metrics_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            last = json.loads(line)
    if last is None:
        return float("nan")
    return float(last.get("test_acc", float("nan")))


def main():
    args = parse_args()
    suite_name = args.suite_name.strip() or time.strftime("suite_%Y%m%d_%H%M%S")
    suite_dir = Path(args.runs_dir) / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    base_train = [
        "python",
        str(Path(__file__).resolve().parent / "run_train.py"),
        "--data-dir",
        args.data_dir,
        "--runs-dir",
        args.runs_dir,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--rounds",
        str(args.rounds),
        "--num-clients",
        str(args.num_clients),
        "--clients-per-round",
        str(args.clients_per_round),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--dirichlet-alpha",
        str(args.dirichlet_alpha),
        "--capture-attack",
        "--attack-round",
        str(args.attack_round),
        "--attack-client",
        str(args.attack_client),
    ]

    base_configs = [
        {"name": "baseline", "extra": ["--defense", "none"]},
        {"name": "clipping", "extra": ["--defense", "clipping", "--clip-norm", "1.0"]},
        {
            "name": "laugd",
            "extra": ["--defense", "laugd", "--alpha", "1.0", "--tau", "1.0", "--p-max", "0.5", "--unbiased"],
        },
    ]
    laugd_opt = {
        "name": "laugd_opt",
        "extra": [
            "--defense",
            "laugd",
            "--alpha",
            "1.0",
            "--tau",
            "1.0",
            "--p-max",
            "0.5",
            "--unbiased",
            "--laugd-preclip",
            "--clip-norm",
            "1.0",
            "--mask-mode",
            "channel",
            "--head-mult",
            "2.0",
            "--head-layers",
            "2",
        ],
    }

    dp_cfg = {"name": "dp_light", "extra": ["--defense", "dp_light", "--clip-norm", "1.0", "--noise-multiplier", "0.1"]}
    svd_cfg = {"name": "svd", "extra": ["--defense", "svd", "--svd-rank-ratio", "0.25"]}

    if args.mode == "basic":
        configs = [base_configs[0], base_configs[1], dp_cfg, base_configs[2], laugd_opt]
    elif args.mode == "paper5":
        configs = [base_configs[0], base_configs[1], svd_cfg, base_configs[2], laugd_opt]
    elif args.mode == "paper6":
        configs = [base_configs[0], base_configs[1], dp_cfg, svd_cfg, base_configs[2], laugd_opt]
    else:
        configs = [base_configs[0], base_configs[1], dp_cfg, svd_cfg, base_configs[2], laugd_opt]

    if args.mode == "full":
        configs.extend(
            [
                {
                    "name": "laugd_no_unbias",
                    "extra": ["--defense", "laugd", "--alpha", "1.0", "--tau", "1.0", "--p-max", "0.5"],
                },
                {
                    "name": "laugd_fixed_p",
                    "extra": ["--defense", "laugd", "--fixed-p", "0.3", "--p-max", "0.5", "--unbiased"],
                },
                {
                    "name": "laugd_fixed_budget",
                    "extra": [
                        "--defense",
                        "laugd",
                        "--alpha",
                        "1.0",
                        "--tau",
                        "1.0",
                        "--p-max",
                        "0.5",
                        "--unbiased",
                        "--mask-mode",
                        "fixed_budget",
                    ],
                },
            ]
        )

    if bool(args.include_rgm_2024):
        p_drop = float(max(0.0, min(0.95, float(args.rgm_p))))
        configs.append(
            {
                "name": f"rgm_2024_p{p_drop}",
                "extra": [
                    "--defense",
                    "laugd",
                    "--fixed-p",
                    str(p_drop),
                    "--p-max",
                    str(p_drop),
                    "--unbiased",
                    "--mask-mode",
                    str(args.rgm_mask_mode),
                ],
            }
        )

    rows = []
    attack_methods = [x.strip() for x in args.attack_methods.split(",") if x.strip()]
    attack_modes = [x.strip() for x in args.attack_modes.split(",") if x.strip()]
    attack_modes = [m for m in attack_modes if m in ("unknown_label", "true_label")]
    if not attack_methods:
        raise ValueError("attack_methods is empty")
    if not attack_modes:
        raise ValueError("attack_modes is empty")

    for cfg in configs:
        run_name = f"{suite_name}__{cfg['name']}"
        run_dir = Path(args.runs_dir) / run_name
        obs = run_dir / "attack_obs.pt"
        metrics_jsonl = run_dir / "metrics.jsonl"
        if not (bool(args.resume) and run_dir.exists() and obs.exists() and metrics_jsonl.exists()):
            cmd_train = base_train + ["--run-name", run_name] + cfg["extra"]
            out = subprocess.check_output(cmd_train, text=True).strip().splitlines()[-1].strip()
            run_dir = Path(out)
            obs = run_dir / "attack_obs.pt"
            metrics_jsonl = run_dir / "metrics.jsonl"

        attack_metrics = {}
        for method in attack_methods:
            for mode in attack_modes:
                out_dir = run_dir / f"attack_{method}" / mode
                out_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = out_dir / "attack_metrics.json"
                if bool(args.resume) and metrics_path.exists():
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        attack_metrics[f"{method}__{mode}"] = json.load(f)
                    continue
                cmd_attack = [
                    "python",
                    str(Path(__file__).resolve().parent / "run_attack.py"),
                    "--obs",
                    str(obs),
                    "--out-dir",
                    str(out_dir),
                    "--device",
                    args.device,
                    "--seed",
                    str(args.seed),
                    "--iters",
                    str(args.attack_iters),
                    "--method",
                    str(method),
                ]
                if float(args.attack_l2_reg) > 0.0:
                    cmd_attack.extend(["--l2-reg", str(float(args.attack_l2_reg))])
                if mode == "true_label":
                    cmd_attack.append("--use-true-label")
                subprocess.check_call(cmd_attack)
                with open(metrics_path, "r", encoding="utf-8") as f:
                    attack_metrics[f"{method}__{mode}"] = json.load(f)

        row = {"name": cfg["name"], "run_dir": str(run_dir), "test_acc": read_last_test_acc(metrics_jsonl)}
        for k, m in attack_metrics.items():
            row[f"psnr__{k}"] = float(m.get("psnr", float("nan")))
            row[f"mse__{k}"] = float(m.get("mse", float("nan")))
            row[f"label_true__{k}"] = int(m.get("label_true", -1))
            row[f"label_recon__{k}"] = int(m.get("label_recon", -1))
        rows.append(row)

    with open(suite_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append(f"# 实验套件汇总：{suite_name}")
    lines.append("")
    cols = ["setting", "test_acc", "run_dir"]
    for method in attack_methods:
        for mode in attack_modes:
            key = f"{method}__{mode}"
            cols.extend([f"psnr({key})", f"mse({key})", f"label_ok({key})"])
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|")
    for r in rows:
        parts = [r["name"], f"{r['test_acc']:.4f}", r["run_dir"]]
        for method in attack_methods:
            for mode in attack_modes:
                key = f"{method}__{mode}"
                ps = r.get(f"psnr__{key}", float("nan"))
                ms = r.get(f"mse__{key}", float("nan"))
                lt = r.get(f"label_true__{key}", -1)
                lr = r.get(f"label_recon__{key}", -2)
                ok = 1 if int(lt) == int(lr) else 0
                parts.extend([f"{ps:.2f}", f"{ms:.6f}", str(ok)])
        lines.append("| " + " | ".join(parts) + " |")
    with open(suite_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(str(suite_dir))


if __name__ == "__main__":
    main()
