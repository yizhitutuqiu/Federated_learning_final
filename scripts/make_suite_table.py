import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--suite-dir", type=str, required=True)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--mode", type=str, default="unknown_label", choices=["unknown_label", "true_label"])
    return p.parse_args()


def fnum(x, nd=4):
    try:
        v = float(x)
    except Exception:
        return ""
    if v != v:
        return ""
    return f"{v:.{nd}f}"


def main():
    args = parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    src = suite_dir / "summary.json"
    rows = json.loads(src.read_text(encoding="utf-8"))

    methods = ["dlg", "ig", "lbfgs"]
    mode = str(args.mode)

    cols = [
        "defense",
        "test_acc",
        "DLG PSNR",
        "DLG MSE",
        "DLG label_ok",
        "IG PSNR",
        "IG MSE",
        "IG label_ok",
        "IG+LBFGS PSNR",
        "IG+LBFGS MSE",
        "IG+LBFGS label_ok",
    ]

    lines = []
    lines.append(f"# 总实验表（3×N）：{suite_dir.name}（attack_mode={mode}）")
    lines.append("")
    lines.append(f"- 数据来源：{src}")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(cols) - 1)) + "|")

    for r in rows:
        name = str(r.get("name", ""))
        parts = [name, fnum(r.get("test_acc"), nd=4)]
        for m in methods:
            k = f"{m}__{mode}"
            ps = r.get(f"psnr__{k}", float("nan"))
            ms = r.get(f"mse__{k}", float("nan"))
            lt = int(r.get(f"label_true__{k}", -1))
            lr = int(r.get(f"label_recon__{k}", -2))
            ok = 1 if lt == lr else 0
            parts.extend([fnum(ps, nd=2), fnum(ms, nd=6), str(ok) if lt >= 0 else ""])
        lines.append("| " + " | ".join(parts) + " |")

    out = Path(args.out).resolve() if args.out.strip() else (suite_dir / f"total_table_{mode}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(out))


if __name__ == "__main__":
    main()

