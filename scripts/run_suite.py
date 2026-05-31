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

    configs = [
        {"name": "baseline", "extra": ["--defense", "none"]},
        {"name": "clipping", "extra": ["--defense", "clipping", "--clip-norm", "1.0"]},
        {"name": "dp_light", "extra": ["--defense", "dp_light", "--clip-norm", "1.0", "--noise-multiplier", "0.1"]},
        {
            "name": "laugd",
            "extra": ["--defense", "laugd", "--alpha", "1.0", "--tau", "1.0", "--p-max", "0.5", "--unbiased"],
        },
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

    rows = []
    for cfg in configs:
        run_name = f"{suite_name}__{cfg['name']}"
        cmd_train = base_train + ["--run-name", run_name] + cfg["extra"]
        out = subprocess.check_output(cmd_train, text=True).strip().splitlines()[-1].strip()
        run_dir = Path(out)

        obs = run_dir / "attack_obs.pt"
        cmd_attack = [
            "python",
            str(Path(__file__).resolve().parent / "run_attack.py"),
            "--obs",
            str(obs),
            "--out-dir",
            str(run_dir / "attack"),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--iters",
            str(args.attack_iters),
        ]
        subprocess.check_call(cmd_attack)

        with open(run_dir / "attack" / "attack_metrics.json", "r", encoding="utf-8") as f:
            attack_metrics = json.load(f)

        rows.append(
            {
                "name": cfg["name"],
                "run_dir": str(run_dir),
                "test_acc": read_last_test_acc(run_dir / "metrics.jsonl"),
                "psnr": float(attack_metrics.get("psnr", float("nan"))),
                "mse": float(attack_metrics.get("mse", float("nan"))),
                "label_true": int(attack_metrics.get("label_true", -1)),
                "label_recon": int(attack_metrics.get("label_recon", -1)),
            }
        )

    with open(suite_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append(f"# 实验套件汇总：{suite_name}")
    lines.append("")
    lines.append("| setting | test_acc | PSNR↑ | MSE↓ | label_true | label_recon | run_dir |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['test_acc']:.4f} | {r['psnr']:.2f} | {r['mse']:.6f} | {r['label_true']} | {r['label_recon']} | {r['run_dir']} |"
        )
    with open(suite_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(str(suite_dir))


if __name__ == "__main__":
    main()
