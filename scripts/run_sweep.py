import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--outputs-dir", type=str, default="outputs")
    p.add_argument("--exp-name", type=str, default="")
    p.add_argument("--device", type=str, default="auto")

    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--num-clients", type=int, default=20)
    p.add_argument("--clients-per-round", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.2)
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=5)

    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--attack-rounds", type=str, default="0,5,10")
    p.add_argument("--attack-clients", type=str, default="0,1,2")
    p.add_argument("--attack-iters", type=int, default=800)
    p.add_argument("--attack-method", type=str, default="dlg", choices=["dlg", "ig", "lbfgs"])
    p.add_argument("--attack-modes", type=str, default="unknown_label", choices=["unknown_label", "unknown_label,true_label"])
    p.add_argument("--attack-l2-reg", type=float, default=0.0)

    p.add_argument("--clipping-grid", type=str, default="0.2,0.5,1.0,2.0")
    p.add_argument("--dp-grid", type=str, default="0.02,0.05,0.1,0.2,0.3")
    p.add_argument("--include-svd", action="store_true")
    p.add_argument("--svd-grid", type=str, default="0.1,0.2,0.3,0.5")
    p.add_argument("--laugd-grid", type=str, default="0.1,0.2,0.3,0.5,0.7")
    p.add_argument("--laugd-alpha-grid", type=str, default="0.5,1.0,2.0")
    p.add_argument("--laugd-tau-grid", type=str, default="0.9,1.0,1.1")
    p.add_argument("--laugd-normalize-score", action="store_true")
    p.add_argument("--laugd-mask-mode", type=str, default="bernoulli", choices=["bernoulli", "fixed_budget"])
    p.add_argument("--laugd-unbiased", action="store_true")
    p.add_argument("--include-clip-laugd", action="store_true")
    p.add_argument("--clip-laugd-clip-grid", type=str, default="0.5,1.0")
    p.add_argument("--include-laugd-v2", action="store_true")
    p.add_argument("--laugd-v2-clip-grid", type=str, default="1.0")
    p.add_argument("--laugd-v2-keep-prob-min", type=float, default=0.25)
    p.add_argument("--laugd-v2-mag-gamma", type=float, default=1.0)
    p.add_argument("--laugd-v2-last-layer-mult", type=float, default=2.0)
    return p.parse_args()


def parse_list(s: str, cast):
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(cast(x))
    return out


def write_attack_spec(path: Path, rounds: list[int], clients: list[int]):
    specs = []
    for r in rounds:
        for c in clients:
            specs.append({"round": int(r), "client_id": int(c), "tag": f"r{int(r)}_c{int(c)}"})
    path.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
    return specs


def read_last_test_acc(metrics_jsonl: Path):
    last = None
    with open(metrics_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            last = json.loads(line)
    if last is None:
        return float("nan")
    return float(last.get("test_acc", float("nan")))


def run_train(base_cmd: list[str]):
    out = subprocess.check_output(base_cmd, text=True).strip().splitlines()[-1].strip()
    return Path(out)


def run_attack(
    obs_path: Path,
    out_dir: Path,
    device: str,
    iters: int,
    seed: int,
    use_true_label: bool,
    method: str,
    l2_reg: float,
):
    cmd = [
        "python",
        str(Path(__file__).resolve().parent / "run_attack.py"),
        "--obs",
        str(obs_path),
        "--out-dir",
        str(out_dir),
        "--device",
        device,
        "--iters",
        str(iters),
        "--seed",
        str(seed),
        "--method",
        str(method),
    ]
    if float(l2_reg) > 0.0:
        cmd.extend(["--l2-reg", str(l2_reg)])
    if use_true_label:
        cmd.append("--use-true-label")
    subprocess.check_call(cmd)


def load_attack_metrics(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pareto_front(points: list[dict], x_key: str, y_key: str, maximize_x: bool, maximize_y: bool):
    def dominates(a, b):
        ax = a[x_key]
        ay = a[y_key]
        bx = b[x_key]
        by = b[y_key]
        ok_x = ax >= bx if maximize_x else ax <= bx
        ok_y = ay >= by if maximize_y else ay <= by
        strict_x = ax > bx if maximize_x else ax < bx
        strict_y = ay > by if maximize_y else ay < by
        return ok_x and ok_y and (strict_x or strict_y)

    front = []
    for p in points:
        dominated = False
        for q in points:
            if q is p:
                continue
            if dominates(q, p):
                dominated = True
                break
        if not dominated:
            front.append(p)
    return front


def mean(xs: list[float]):
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def safe_float(x):
    try:
        v = float(x)
    except Exception:
        return float("nan")
    if v != v:
        return float("nan")
    return v


def plot_scatter(out_path: Path, rows: list[dict], title: str, x_key: str, y_key: str, groups: list[str]):
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))
    import matplotlib.pyplot as plt

    colors = {
        "baseline": "#1f77b4",
        "clipping": "#ff7f0e",
        "dp_light": "#2ca02c",
        "svd": "#17becf",
        "laugd": "#d62728",
        "clip_laugd": "#9467bd",
        "laugd_v2": "#8c564b",
    }

    plt.figure(figsize=(7.5, 5.2))
    for g in groups:
        pts = [r for r in rows if r["family"] == g]
        if not pts:
            continue
        xs = [p[x_key] for p in pts]
        ys = [p[y_key] for p in pts]
        plt.scatter(xs, ys, s=28, alpha=0.8, label=g, c=colors.get(g, None))

        front = pareto_front(pts, x_key=x_key, y_key=y_key, maximize_x=True, maximize_y=False)
        if front:
            front = sorted(front, key=lambda z: z[x_key])
            plt.plot([p[x_key] for p in front], [p[y_key] for p in front], linewidth=2, alpha=0.6, c=colors.get(g, None))

    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_bar(out_path: Path, stats: dict[str, float], title: str, ylabel: str):
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))
    import matplotlib.pyplot as plt

    names = list(stats.keys())
    vals = [stats[k] for k in names]
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(names, vals)
    plt.ylim(0.0, 1.0)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    exp_name = args.exp_name.strip() or time.strftime("sweep_%Y%m%d_%H%M%S")
    out_dir = Path(args.outputs_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_list(args.seeds, int)
    attack_rounds = parse_list(args.attack_rounds, int)
    attack_clients = parse_list(args.attack_clients, int)
    attack_modes = [m.strip() for m in args.attack_modes.split(",") if m.strip()]

    spec_path = out_dir / "attack_spec.json"
    write_attack_spec(spec_path, rounds=attack_rounds, clients=attack_clients)

    clipping_grid = parse_list(args.clipping_grid, float)
    dp_grid = parse_list(args.dp_grid, float)
    svd_grid = parse_list(args.svd_grid, float)
    laugd_grid = parse_list(args.laugd_grid, float)
    laugd_alpha_grid = parse_list(args.laugd_alpha_grid, float)
    laugd_tau_grid = parse_list(args.laugd_tau_grid, float)
    clip_laugd_clip_grid = parse_list(args.clip_laugd_clip_grid, float)
    laugd_v2_clip_grid = parse_list(args.laugd_v2_clip_grid, float)

    configs = []
    configs.append({"family": "baseline", "name": "baseline", "defense": "none", "extra": []})
    for c in clipping_grid:
        configs.append({"family": "clipping", "name": f"clipping_C{c}", "defense": "clipping", "extra": ["--clip-norm", str(c)]})
    for nm in dp_grid:
        configs.append(
            {
                "family": "dp_light",
                "name": f"dp_nm{nm}",
                "defense": "dp_light",
                "extra": ["--clip-norm", "1.0", "--noise-multiplier", str(nm)],
            }
        )
    if args.include_svd:
        for rr in svd_grid:
            configs.append(
                {
                    "family": "svd",
                    "name": f"svd_r{rr}",
                    "defense": "svd",
                    "extra": ["--svd-rank-ratio", str(rr)],
                }
            )
    for alpha in laugd_alpha_grid:
        for tau in laugd_tau_grid:
            for pm in laugd_grid:
                extra = [
                    "--alpha",
                    str(alpha),
                    "--tau",
                    str(tau),
                    "--p-max",
                    str(pm),
                    "--mask-mode",
                    args.laugd_mask_mode,
                ]
                if args.laugd_normalize_score:
                    extra.append("--normalize-score")
                if args.laugd_unbiased:
                    extra.append("--unbiased")
                name = f"laugd_a{alpha}_t{tau}_pmax{pm}"
                configs.append({"family": "laugd", "name": name, "defense": "laugd", "extra": extra})

    if args.include_laugd_v2:
        for clip_c in laugd_v2_clip_grid:
            for alpha in laugd_alpha_grid:
                for tau in laugd_tau_grid:
                    for pm in laugd_grid:
                        extra = [
                            "--clip-norm",
                            str(clip_c),
                            "--alpha",
                            str(alpha),
                            "--tau",
                            str(tau),
                            "--p-max",
                            str(pm),
                            "--mask-mode",
                            args.laugd_mask_mode,
                            "--normalize-score",
                            "--unbiased",
                            "--mag-aware",
                            "--mag-gamma",
                            str(args.laugd_v2_mag_gamma),
                            "--keep-prob-min",
                            str(args.laugd_v2_keep_prob_min),
                            "--last-layer-mult",
                            str(args.laugd_v2_last_layer_mult),
                        ]
                        name = f"laugd_v2_C{clip_c}_a{alpha}_t{tau}_pmax{pm}"
                        configs.append({"family": "laugd_v2", "name": name, "defense": "clip_laugd", "extra": extra})

    if args.include_clip_laugd:
        for clip_c in clip_laugd_clip_grid:
            for alpha in laugd_alpha_grid:
                for tau in laugd_tau_grid:
                    for pm in laugd_grid:
                        extra = [
                            "--clip-norm",
                            str(clip_c),
                            "--alpha",
                            str(alpha),
                            "--tau",
                            str(tau),
                            "--p-max",
                            str(pm),
                            "--mask-mode",
                            args.laugd_mask_mode,
                        ]
                        if args.laugd_normalize_score:
                            extra.append("--normalize-score")
                        if args.laugd_unbiased:
                            extra.append("--unbiased")
                        name = f"clip_laugd_C{clip_c}_a{alpha}_t{tau}_pmax{pm}"
                        configs.append({"family": "clip_laugd", "name": name, "defense": "clip_laugd", "extra": extra})

    all_rows = []
    for seed in seeds:
        for cfg in configs:
            run_name = f"{exp_name}__{cfg['name']}__seed{seed}"
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
                str(seed),
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
                cfg["defense"],
                "--capture-attack",
                "--attack-spec",
                str(spec_path),
            ] + cfg["extra"]

            run_dir = run_train(cmd)
            test_acc = read_last_test_acc(run_dir / "metrics.jsonl")

            for r in attack_rounds:
                for c in attack_clients:
                    tag = f"r{r}_c{c}"
                    obs_path = run_dir / f"attack_obs_{tag}.pt"
                    if not obs_path.exists():
                        continue

                    for mode in attack_modes:
                        use_true_label = mode == "true_label"
                        out_attack_dir = run_dir / "attack_sweep" / tag / mode
                        out_attack_dir.mkdir(parents=True, exist_ok=True)
                        run_attack(
                            obs_path=obs_path,
                            out_dir=out_attack_dir,
                            device=args.device,
                            iters=args.attack_iters,
                            seed=seed,
                            use_true_label=use_true_label,
                            method=str(args.attack_method),
                            l2_reg=float(args.attack_l2_reg),
                        )
                        m = load_attack_metrics(out_attack_dir / "attack_metrics.json")
                        all_rows.append(
                            {
                                "family": cfg["family"],
                                "config": cfg["name"],
                                "seed": seed,
                                "attack_round": r,
                                "attack_client": c,
                                "attack_mode": mode,
                                "test_acc": float(test_acc),
                                "psnr": float(m.get("psnr", float("nan"))),
                                "mse": float(m.get("mse", float("nan"))),
                                "label_true": int(m.get("label_true", -1)),
                                "label_recon": int(m.get("label_recon", -1)),
                                "run_dir": str(run_dir),
                            }
                        )

    with open(out_dir / "rows.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    def aggregate_rows():
        groups = {}
        for r in all_rows:
            fam = r.get("family")
            cfg = r.get("config")
            mode = r.get("attack_mode")
            if fam is None or cfg is None or mode is None:
                continue
            key = (fam, cfg, mode)
            groups.setdefault(key, []).append(r)

        agg_rows = []
        for (fam, cfg, mode), lst in groups.items():
            test_acc = mean([safe_float(x.get("test_acc")) for x in lst])
            psnr = mean([safe_float(x.get("psnr")) for x in lst])
            mse = mean([safe_float(x.get("mse")) for x in lst])
            label_ok = sum(1 for x in lst if int(x.get("label_true", -1)) == int(x.get("label_recon", -2)))
            agg_rows.append(
                {
                    "family": fam,
                    "config": cfg,
                    "attack_mode": mode,
                    "n": len(lst),
                    "test_acc": test_acc,
                    "psnr": psnr,
                    "mse": mse,
                    "label_acc": label_ok / len(lst) if lst else float("nan"),
                }
            )
        return agg_rows

    agg_rows = aggregate_rows()
    with open(out_dir / "agg_rows.json", "w", encoding="utf-8") as f:
        json.dump(agg_rows, f, ensure_ascii=False, indent=2)

    def write_pareto(mode: str):
        pts = [r for r in agg_rows if r["attack_mode"] == mode]
        for r in pts:
            r["_x"] = safe_float(r.get("test_acc"))
            r["_y"] = safe_float(r.get("psnr"))
        pts = [r for r in pts if r["_x"] == r["_x"] and r["_y"] == r["_y"]]
        front = pareto_front(pts, x_key="_x", y_key="_y", maximize_x=True, maximize_y=False)
        front = sorted(front, key=lambda z: (-z["_x"], z["_y"]))

        fam_counts = {}
        for r in front:
            fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1

        lines = []
        lines.append(f"# Pareto 前沿（mode={mode}，目标：test_acc↑ 且 PSNR↓）")
        lines.append("")
        lines.append("说明：每个点先在 (family, config, mode) 内做平均聚合，再做全局非支配筛选。")
        lines.append("")
        lines.append("## 非支配点数量（按方法）")
        lines.append("")
        lines.append("| family | count |")
        lines.append("|---|---:|")
        for fam, cnt in sorted(fam_counts.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"| {fam} | {cnt} |")
        lines.append("")
        lines.append("## 前沿点表格")
        lines.append("")
        lines.append("| family | config | n | test_acc | PSNR | MSE | label_acc |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for r in front:
            lines.append(
                f"| {r['family']} | {r['config']} | {r['n']} | {r['test_acc']:.4f} | {r['psnr']:.2f} | {r['mse']:.6f} | {r['label_acc']:.3f} |"
            )
        (out_dir / f"pareto_{mode}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for mode in attack_modes:
        write_pareto(mode)

    def summarize_label_acc(mode: str):
        families = sorted({r["family"] for r in all_rows})
        stats = {}
        for fam in families:
            rows = [r for r in all_rows if r["family"] == fam and r["attack_mode"] == mode]
            if not rows:
                stats[fam] = float("nan")
                continue
            ok = sum(1 for r in rows if r["label_true"] == r["label_recon"])
            stats[fam] = ok / len(rows)
        return stats

    rows_unknown = [r for r in all_rows if r["attack_mode"] == "unknown_label"]
    rows_true = [r for r in all_rows if r["attack_mode"] == "true_label"]
    groups_all = sorted({r["family"] for r in all_rows})

    if "unknown_label" in attack_modes and rows_unknown:
        plot_scatter(
            out_dir / "privacy_utility_unknown.png",
            rows_unknown,
            title="Privacy-Utility (unknown label attack)",
            x_key="test_acc",
            y_key="psnr",
            groups=groups_all,
        )
        plot_scatter(
            out_dir / "privacy_utility_unknown_mse.png",
            rows_unknown,
            title="Privacy-Utility (unknown label attack)",
            x_key="test_acc",
            y_key="mse",
            groups=groups_all,
        )
        plot_bar(
            out_dir / "label_leak_unknown.png",
            summarize_label_acc("unknown_label"),
            title="Label Leakage (unknown label attack)",
            ylabel="label recovery acc",
        )

    if "true_label" in attack_modes and rows_true:
        plot_scatter(
            out_dir / "privacy_utility_true.png",
            rows_true,
            title="Privacy-Utility (true label attack)",
            x_key="test_acc",
            y_key="psnr",
            groups=groups_all,
        )
        plot_scatter(
            out_dir / "privacy_utility_true_mse.png",
            rows_true,
            title="Privacy-Utility (true label attack)",
            x_key="test_acc",
            y_key="mse",
            groups=groups_all,
        )

    lines = []
    lines.append(f"# Sweep 汇总：{exp_name}")
    lines.append("")
    lines.append("- 输出文件：")
    lines.append(f"  - rows.json：全部样本（配置×seed×round×client×attack_mode）")
    lines.append(f"  - agg_rows.json：按 (family, config, attack_mode) 聚合后的点")
    lines.append(f"  - privacy_utility_*.png：隐私-效用散点与前沿")
    lines.append(f"  - label_leak_unknown.png：标签恢复准确率对比")
    lines.append(f"  - pareto_*.md：全局 Pareto 前沿表格（严谨判断是否存在非支配点）")
    lines.append("")
    lines.append("建议解读：同等 test_acc 下 PSNR 越低（或 MSE 越高）越隐私；unknown_label 模式下 label recovery 越低越好。")
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(str(out_dir))


if __name__ == "__main__":
    main()
