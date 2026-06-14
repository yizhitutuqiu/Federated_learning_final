import argparse
import json
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--suite-dir",
        type=str,
        default="runs/suite_laugd_opt_all_attacks",
    )
    p.add_argument(
        "--out-md",
        type=str,
        default="outputs/new_Table.md",
    )
    p.add_argument(
        "--assets-dir",
        type=str,
        default="outputs/new_Table_assets",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--example-iters", type=int, default=400)
    p.add_argument("--example-lr", type=float, default=0.1)
    p.add_argument("--example-tv-reg", type=float, default=1e-4)
    p.add_argument("--example-l2-reg", type=float, default=1e-4)
    return p.parse_args()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, s: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def label_image(img: Image.Image, text: str, header_h: int = 36, color=(0, 0, 0), border: bool = False):
    w, h = img.size
    canvas = Image.new("RGB", (w, h + header_h), color=(255, 255, 255))
    canvas.paste(img, (0, header_h))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), text, fill=color)
    if border:
        d.rectangle([(1, 1), (w - 2, h + header_h - 2)], outline=(220, 20, 60), width=4)
    return canvas


def grid(images: list[Image.Image], cols: int = 3, gap: int = 16, bg=(255, 255, 255)):
    if not images:
        raise ValueError("empty images")
    cols = max(1, int(cols))
    rows = (len(images) + cols - 1) // cols
    max_w = max(im.size[0] for im in images)
    max_h = max(im.size[1] for im in images)
    W = cols * max_w + (cols - 1) * gap
    H = rows * max_h + (rows - 1) * gap
    out = Image.new("RGB", (W, H), color=bg)
    for i, im in enumerate(images):
        r = i // cols
        c = i % cols
        x = c * (max_w + gap)
        y = r * (max_h + gap)
        out.paste(im, (x, y))
    return out


def plot_bar(
    path: Path,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    colors: list[str] | None = None,
):
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(max(5.0, 0.7 * len(labels)), 3.2))
    xs = list(range(len(labels)))
    ax.bar(xs, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_example_attack(
    root: Path,
    obs_path: Path,
    out_dir: Path,
    device: str,
    method: str,
    iters: int,
    lr: float,
    tv_reg: float,
    l2_reg: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(root / "scripts" / "run_attack.py"),
        "--obs",
        str(obs_path),
        "--out-dir",
        str(out_dir),
        "--device",
        str(device),
        "--seed",
        "0",
        "--iters",
        str(int(iters)),
        "--lr",
        str(float(lr)),
        "--tv-reg",
        str(float(tv_reg)),
        "--method",
        str(method),
    ]
    if method in ("ig", "lbfgs") and float(l2_reg) > 0.0:
        cmd.extend(["--l2-reg", str(float(l2_reg))])
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        if device != "cpu":
            cmd_cpu = [x for x in cmd]
            for i in range(len(cmd_cpu) - 1):
                if cmd_cpu[i] == "--device":
                    cmd_cpu[i + 1] = "cpu"
                    break
            subprocess.check_call(cmd_cpu)
        else:
            raise


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    suite_dir = (root / args.suite_dir).resolve()
    suite_summary = suite_dir / "summary.json"
    rows = read_json(suite_summary)

    assets_dir = (root / args.assets_dir).resolve()
    assets_dir.mkdir(parents=True, exist_ok=True)
    out_md = (root / args.out_md).resolve()

    name_map = {"laugd": "laugd_v1", "laugd_opt": "laugd_v2"}
    highlight_name = "laugd_v2"

    for r in rows:
        r["display_name"] = name_map.get(r["name"], r["name"])

    defenses = [r["display_name"] for r in rows]
    attack_methods = ["dlg", "ig", "lbfgs"]
    mode = "unknown_label"

    colors = ["#d62728" if n == highlight_name else "#1f77b4" for n in defenses]

    plot_bar(
        assets_dir / "test_acc.png",
        defenses,
        [float(r.get("test_acc", float("nan"))) for r in rows],
        title="Utility: test_acc (higher is better)",
        ylabel="test_acc",
        colors=colors,
    )

    for m in attack_methods:
        plot_bar(
            assets_dir / f"psnr_{m}_{mode}.png",
            defenses,
            [float(r.get(f"psnr__{m}__{mode}", float("nan"))) for r in rows],
            title=f"Privacy proxy: PSNR under {m} ({mode}) (lower is better)",
            ylabel="PSNR",
            colors=colors,
        )

    for m in attack_methods:
        ok = []
        for r in rows:
            lt = r.get(f"label_true__{m}__{mode}")
            lr = r.get(f"label_recon__{m}__{mode}")
            ok.append(1.0 if lt == lr else 0.0)
        plot_bar(
            assets_dir / f"label_ok_{m}_{mode}.png",
            defenses,
            ok,
            title=f"Label inference success under {m} ({mode}) (lower is better)",
            ylabel="label_ok",
            colors=colors,
        )

    for m in attack_methods:
        imgs = []
        for r in rows:
            run_dir = (root / r["run_dir"]).resolve()
            p = run_dir / f"attack_{m}" / mode / "recon.png"
            if not p.exists():
                continue
            im = Image.open(p).convert("RGB")
            dn = r["display_name"]
            imgs.append(
                label_image(
                    im,
                    dn,
                    color=(220, 20, 60) if dn == highlight_name else (0, 0, 0),
                    border=dn == highlight_name,
                )
            )
        if imgs:
            out_img = grid(imgs, cols=3)
            out_img.save(assets_dir / f"recon_grid_{m}_{mode}.png")

    example_defenses = ["baseline", "dp_light", "svd", "laugd", "laugd_opt"]
    example_methods = ["ig", "lbfgs"]
    for d in example_defenses:
        r = next((x for x in rows if x["name"] == d), None)
        if r is None:
            continue
        dn = r["display_name"]
        obs = (root / r["run_dir"] / "attack_obs.pt").resolve()
        if not obs.exists():
            continue
        for m in example_methods:
            out_dir = assets_dir / "example_loss_curves" / dn / m
            run_example_attack(
                root=root,
                obs_path=obs,
                out_dir=out_dir,
                device=args.device,
                method=m,
                iters=args.example_iters,
                lr=args.example_lr,
                tv_reg=args.example_tv_reg,
                l2_reg=args.example_l2_reg,
            )

    md = []
    md.append("# 实验汇总（new_Table）")
    md.append("")
    md.append(f"- suite_dir: `{suite_dir}`")
    md.append("- 口径：只看 unknown_label（攻击者不知道真实标签）")
    md.append("- 指标方向：test_acc↑更好；PSNR↓/MSE↑更隐私；label_ok↓更隐私")
    md.append(f"- 标红：{highlight_name}")
    md.append("")

    md.append("## 1) Utility（训练质量）")
    md.append("")
    md.append("![](new_Table_assets/test_acc.png)")
    md.append("")

    md.append("## 2) 总表（unknown_label）")
    md.append("")
    header = [
        "defense",
        "test_acc",
        "DLG PSNR",
        "DLG MSE",
        "DLG label_ok",
        "IG PSNR",
        "IG MSE",
        "IG label_ok",
        "LBFGS PSNR",
        "LBFGS MSE",
        "LBFGS label_ok",
    ]
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        def _ok(method: str):
            lt = r.get(f"label_true__{method}__{mode}")
            lr = r.get(f"label_recon__{method}__{mode}")
            return 1 if lt == lr else 0

        md.append(
            "| "
            + " | ".join(
                [
                    r["display_name"],
                    fmt(r.get("test_acc"), 4),
                    fmt(r.get(f"psnr__dlg__{mode}"), 2),
                    fmt(r.get(f"mse__dlg__{mode}"), 4),
                    str(_ok("dlg")),
                    fmt(r.get(f"psnr__ig__{mode}"), 2),
                    fmt(r.get(f"mse__ig__{mode}"), 4),
                    str(_ok("ig")),
                    fmt(r.get(f"psnr__lbfgs__{mode}"), 2),
                    fmt(r.get(f"mse__lbfgs__{mode}"), 4),
                    str(_ok("lbfgs")),
                ]
            )
            + " |"
        )
    md.append("")

    md.append("## 3) 按攻击类型的可视化（unknown_label）")
    md.append("")
    for m in attack_methods:
        md.append(f"### 3.{attack_methods.index(m)+1}) {m.upper()}")
        md.append("")
        md.append(f"![](new_Table_assets/psnr_{m}_{mode}.png)")
        md.append("")
        md.append(f"![](new_Table_assets/label_ok_{m}_{mode}.png)")
        md.append("")
        grid_path = assets_dir / f"recon_grid_{m}_{mode}.png"
        if grid_path.exists():
            md.append(f"![](new_Table_assets/recon_grid_{m}_{mode}.png)")
            md.append("")

    md.append("## 4) 重建 loss 曲线示例（unknown_label，重新跑攻击得到）")
    md.append("")
    md.append("- 示例方法：IG / LBFGS；示例防御：baseline / dp_light / svd / laugd_v1 / laugd_v2")
    md.append("")
    for m in example_methods:
        md.append(f"### {m.upper()} loss curves")
        md.append("")
        for d in example_defenses:
            dn = name_map.get(d, d)
            p = assets_dir / "example_loss_curves" / dn / m / "loss_curve.png"
            if p.exists():
                md.append(f"#### {dn}")
                md.append("")
                md.append(f"![](new_Table_assets/example_loss_curves/{dn}/{m}/loss_curve.png)")
                md.append("")

    md.append("## 5) 关键解读（unknown_label）")
    md.append("")
    md.append("- laugd_v2 的核心收益：在 unknown_label 下显著破坏标签推断（label_ok=0），并在 IG/LBFGS 下将 PSNR 压到与/优于 dp_light 的水平。")
    md.append("- laugd_v1 的优势：几乎不伤 test_acc，但在 IG/LBFGS 下提升幅度有限。")
    md.append("- dp_light：对 IG 攻击非常强，但效用损失相对更大。")
    md.append("- svd：在更强优化器（LBFGS）下更占优，但对 IG 未必稳定。")
    md.append("")

    write_text(out_md, "\n".join(md) + "\n")
    print(str(out_md))


if __name__ == "__main__":
    main()
