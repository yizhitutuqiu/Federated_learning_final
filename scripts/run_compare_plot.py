import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--outputs-dir", type=str, default="outputs")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--exp-name", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--clients-per-round", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=5)

    p.add_argument("--attack-round", type=int, default=0)
    p.add_argument("--attack-client", type=int, default=0)
    p.add_argument("--attack-iters", type=int, default=800)

    p.add_argument("--defense", type=str, default="laugd", choices=["clipping", "dp_light", "laugd"])
    p.add_argument("--clip-norm", type=float, default=1.0)
    p.add_argument("--noise-multiplier", type=float, default=0.1)

    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--p-max", type=float, default=0.5)
    p.add_argument("--normalize-score", action="store_true")
    p.add_argument("--unbiased", action="store_true")
    p.add_argument("--mask-mode", type=str, default="bernoulli", choices=["bernoulli", "fixed_budget"])
    p.add_argument("--fixed-p", type=float, default=-1.0)
    return p.parse_args()


def read_metrics(metrics_path: Path):
    rounds = []
    train_loss = []
    test_loss = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rounds.append(int(row["round"]))
        train_loss.append(float(row["train_loss"]))
        test_loss.append(float(row["test_loss"]))
    return rounds, train_loss, test_loss


def run_train(args, run_name: str, defense: str, capture_attack: bool):
    cmd = [
        "python",
        str(Path(__file__).resolve().parent / "run_train.py"),
        "--data-dir",
        args.data_dir,
        "--runs-dir",
        args.runs_dir,
        "--run-name",
        run_name,
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
        "--eval-every",
        str(args.eval_every),
        "--defense",
        defense,
    ]

    if defense == "clipping":
        cmd += ["--clip-norm", str(args.clip_norm)]
    elif defense == "dp_light":
        cmd += ["--clip-norm", str(args.clip_norm), "--noise-multiplier", str(args.noise_multiplier)]
    elif defense == "laugd":
        cmd += [
            "--alpha",
            str(args.alpha),
            "--tau",
            str(args.tau),
            "--p-max",
            str(args.p_max),
            "--mask-mode",
            args.mask_mode,
        ]
        if args.normalize_score:
            cmd.append("--normalize-score")
        if args.unbiased:
            cmd.append("--unbiased")
        if args.fixed_p >= 0:
            cmd += ["--fixed-p", str(args.fixed_p)]

    if capture_attack:
        cmd += [
            "--capture-attack",
            "--attack-round",
            str(args.attack_round),
            "--attack-client",
            str(args.attack_client),
        ]

    out = subprocess.check_output(cmd, text=True).strip().splitlines()[-1].strip()
    return Path(out)


def run_attack(run_dir: Path, device: str, iters: int, seed: int):
    cmd = [
        "python",
        str(Path(__file__).resolve().parent / "run_attack.py"),
        "--obs",
        str(run_dir / "attack_obs.pt"),
        "--out-dir",
        str(run_dir / "attack"),
        "--device",
        device,
        "--iters",
        str(iters),
        "--seed",
        str(seed),
    ]
    subprocess.check_call(cmd)


def plot_losses(out_dir: Path, curves: list[tuple[str, list[int], list[float]]], title: str):
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for name, xs, ys in curves:
        plt.plot(xs, ys, label=name, linewidth=2)
    plt.xlabel("round")
    plt.ylabel("train_loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves.png", dpi=180)
    plt.close()


def main():
    args = parse_args()
    exp_name = args.exp_name.strip() or time.strftime("compare_%Y%m%d_%H%M%S")
    out_dir = Path(args.outputs_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = run_train(args, run_name=f"{exp_name}__baseline", defense="none", capture_attack=False)
    attack_dir = run_train(args, run_name=f"{exp_name}__baseline_attack", defense="none", capture_attack=True)
    run_attack(attack_dir, device=args.device, iters=args.attack_iters, seed=args.seed)

    defense_dir = run_train(args, run_name=f"{exp_name}__attack_{args.defense}", defense=args.defense, capture_attack=True)
    run_attack(defense_dir, device=args.device, iters=args.attack_iters, seed=args.seed)

    curves = []
    for name, rd in [
        ("baseline", baseline_dir),
        ("baseline+attack", attack_dir),
        (f"baseline+attack+{args.defense}", defense_dir),
    ]:
        xs, train_loss, _ = read_metrics(rd / "metrics.jsonl")
        curves.append((name, xs, train_loss))

    plot_losses(out_dir, curves, title="Train Loss Curves (FedSGD)")

    with open(out_dir / "runs.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline": str(baseline_dir),
                "baseline_attack": str(attack_dir),
                "baseline_attack_defense": str(defense_dir),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(str(out_dir))


if __name__ == "__main__":
    main()
