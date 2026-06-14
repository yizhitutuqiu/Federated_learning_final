import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--compare-dir", type=str, required=True)
    return p.parse_args()


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def label_image(img: Image.Image, text: str, header_h: int = 36):
    w, h = img.size
    canvas = Image.new("RGB", (w, h + header_h), color=(255, 255, 255))
    canvas.paste(img, (0, header_h))
    d = ImageDraw.Draw(canvas)
    d.text((10, 8), text, fill=(0, 0, 0))
    return canvas


def hstack(images: list[Image.Image], gap: int = 16, bg=(255, 255, 255)):
    if not images:
        raise ValueError("empty images")
    heights = [im.size[1] for im in images]
    widths = [im.size[0] for im in images]
    H = max(heights)
    W = sum(widths) + gap * (len(images) - 1)
    out = Image.new("RGB", (W, H), color=bg)
    x = 0
    for im in images:
        y = (H - im.size[1]) // 2
        out.paste(im, (x, y))
        x += im.size[0] + gap
    return out


def fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


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


def make_compare_report(compare_dir: Path):
    runs_path = compare_dir / "runs.json"
    runs = read_json(runs_path)

    entries = [
        ("baseline+attack", runs.get("baseline_attack")),
        ("baseline+attack+defense", runs.get("baseline_attack_defense")),
    ]

    imgs = []
    rows = []
    for name, run_dir_str in entries:
        if not run_dir_str:
            continue
        run_dir = Path(run_dir_str)
        recon = run_dir / "attack" / "recon.png"
        metrics = run_dir / "attack" / "attack_metrics.json"
        if recon.exists():
            im = Image.open(recon).convert("RGB")
            imgs.append(label_image(im, name))
        if metrics.exists():
            m = read_json(metrics)
            rows.append(
                {
                    "setting": name,
                    "run_dir": str(run_dir),
                    "label_true": m.get("label_true"),
                    "label_recon": m.get("label_recon"),
                    "mse": m.get("mse"),
                    "psnr": m.get("psnr"),
                }
            )

    if imgs:
        out_img = hstack(imgs)
        out_img.save(compare_dir / "recon_compare.png")

    md = []
    md.append("# 直观对比报告（重构 + loss 曲线）")
    md.append("")
    md.append("## 1) Loss 曲线")
    md.append("")
    if (compare_dir / "loss_curves.png").exists():
        md.append("![](loss_curves.png)")
        md.append("")
    else:
        md.append("- 未找到 loss_curves.png")
        md.append("")

    md.append("## 2) 重构对比")
    md.append("")
    if (compare_dir / "recon_compare.png").exists():
        md.append("![](recon_compare.png)")
        md.append("")
    else:
        md.append("- 未找到 recon 图片（请确认 run_dir/attack/recon.png 存在）")
        md.append("")

    if rows:
        md.append("## 3) 攻击指标（参考）")
        md.append("")
        md.append("| setting | label_true | label_recon | MSE↓ | PSNR↑ | run_dir |")
        md.append("|---|---:|---:|---:|---:|---|")
        for r in rows:
            md.append(
                f"| {r['setting']} | {r['label_true']} | {r['label_recon']} | {fmt(r['mse'], 6)} | {fmt(r['psnr'], 2)} | {r['run_dir']} |"
            )
        md.append("")

    (compare_dir / "overview.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _pick_representative(rows: list[dict], family: str, mode: str):
    pts = [r for r in rows if r.get("family") == family and r.get("attack_mode") == mode]
    if not pts:
        return None
    for r in pts:
        r["_x"] = float(r.get("test_acc", float("nan")))
        r["_y"] = float(r.get("psnr", float("nan")))
    pts = [r for r in pts if r["_x"] == r["_x"] and r["_y"] == r["_y"]]
    if not pts:
        return None
    front = pareto_front(pts, x_key="_x", y_key="_y", maximize_x=True, maximize_y=False)
    front = front or pts
    return max(front, key=lambda z: (z["_x"], -z["_y"]))


def _pick_anchor_tuples(rows: list[dict], mode: str, families: list[str]):
    per_family = []
    for fam in families:
        s = set()
        for r in rows:
            if r.get("attack_mode") != mode or r.get("family") != fam:
                continue
            s.add((int(r.get("seed", -1)), int(r.get("attack_round", -1)), int(r.get("attack_client", -1))))
        per_family.append(s)
    if not per_family:
        return []
    common = set.intersection(*per_family) if all(per_family) else set()
    return sorted(common)


def _best_baseline_acc_for_tuple(rows: list[dict], mode: str, tpl: tuple[int, int, int]):
    best = None
    for r in rows:
        if r.get("attack_mode") != mode or r.get("family") != "baseline":
            continue
        t = (int(r.get("seed", -1)), int(r.get("attack_round", -1)), int(r.get("attack_client", -1)))
        if t != tpl:
            continue
        try:
            acc = float(r.get("test_acc", float("nan")))
        except Exception:
            continue
        if best is None or acc > best:
            best = acc
    return best if best is not None else float("nan")


def _pick_representative_fixed_tuple(
    rows: list[dict],
    family: str,
    mode: str,
    fixed_tpl: tuple[int, int, int],
):
    seed, rr, cc = fixed_tpl
    pts = [
        r
        for r in rows
        if r.get("family") == family
        and r.get("attack_mode") == mode
        and int(r.get("seed", -1)) == seed
        and int(r.get("attack_round", -1)) == rr
        and int(r.get("attack_client", -1)) == cc
    ]
    if not pts:
        return None
    for r in pts:
        r["_x"] = float(r.get("test_acc", float("nan")))
        r["_y"] = float(r.get("psnr", float("nan")))
    pts = [r for r in pts if r["_x"] == r["_x"] and r["_y"] == r["_y"]]
    if not pts:
        return None
    front = pareto_front(pts, x_key="_x", y_key="_y", maximize_x=True, maximize_y=False)
    front = front or pts
    return max(front, key=lambda z: (z["_x"], -z["_y"]))


def make_sweep_report(compare_dir: Path):
    rows_path = compare_dir / "rows.json"
    rows = read_json(rows_path)
    if not isinstance(rows, list):
        raise ValueError("rows.json must be a list")

    families = ["baseline", "clipping", "dp_light", "laugd"]
    modes = ["unknown_label", "true_label"]

    selections: dict[str, dict[str, dict]] = {m: {} for m in modes}
    for mode in modes:
        common_tuples = _pick_anchor_tuples(rows, mode=mode, families=families)
        fixed_tpl = None
        if common_tuples:
            fixed_tpl = max(common_tuples, key=lambda t: _best_baseline_acc_for_tuple(rows, mode=mode, tpl=t))

        for fam in families:
            if fixed_tpl is not None:
                sel = _pick_representative_fixed_tuple(rows, family=fam, mode=mode, fixed_tpl=fixed_tpl)
            else:
                sel = _pick_representative(rows, family=fam, mode=mode)
            if sel is not None:
                selections[mode][fam] = sel

    for mode in modes:
        imgs = []
        for fam in families:
            sel = selections[mode].get(fam)
            if sel is None:
                continue
            tag = f"r{int(sel['attack_round'])}_c{int(sel['attack_client'])}"
            run_dir = Path(sel["run_dir"])
            recon = run_dir / "attack_sweep" / tag / mode / "recon.png"
            if not recon.exists():
                continue
            txt = (
                f"{fam} | {sel.get('config')} | seed={sel.get('seed')} {tag} | "
                f"y={sel.get('label_true')}→{sel.get('label_recon')} | "
                f"acc={fmt(sel.get('test_acc'),3)} | psnr={fmt(sel.get('psnr'),2)}"
            )
            im = Image.open(recon).convert("RGB")
            imgs.append(label_image(im, txt))

        if imgs:
            out_img = hstack(imgs)
            out_img.save(compare_dir / f"recon_sweep_{mode}.png")

    md = []
    md.append("# Sweep 可视化总览")
    md.append("")
    md.append("这份报告会尽量在同一组 (seed, round, client) 上对四种方法做重构可视化；如果四种方法没有共同的样本点，才会退化为各自单独挑选代表点。")
    md.append("")

    md.append("## 1) 隐私-效用散点（unknown label）")
    md.append("")
    if (compare_dir / "privacy_utility_unknown.png").exists():
        md.append("![](privacy_utility_unknown.png)")
        md.append("")
    if (compare_dir / "label_leak_unknown.png").exists():
        md.append("![](label_leak_unknown.png)")
        md.append("")
    if (compare_dir / "recon_sweep_unknown_label.png").exists():
        md.append("![](recon_sweep_unknown_label.png)")
        md.append("")

    md.append("## 2) 隐私-效用散点（true label）")
    md.append("")
    if (compare_dir / "privacy_utility_true.png").exists():
        md.append("![](privacy_utility_true.png)")
        md.append("")
    if (compare_dir / "recon_sweep_true_label.png").exists():
        md.append("![](recon_sweep_true_label.png)")
        md.append("")

    md.append("## 3) 代表点表格（便于复现）")
    md.append("")
    md.append("| mode | family | config | test_acc | psnr | mse | seed | r | c | run_dir |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for mode in modes:
        for fam in families:
            sel = selections[mode].get(fam)
            if sel is None:
                continue
            md.append(
                f"| {mode} | {fam} | {sel.get('config')} | {fmt(sel.get('test_acc'),4)} | {fmt(sel.get('psnr'),2)} | {fmt(sel.get('mse'),6)} | {sel.get('seed')} | {sel.get('attack_round')} | {sel.get('attack_client')} | {sel.get('run_dir')} |"
            )
    md.append("")

    (compare_dir / "overview.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    compare_dir = Path(args.compare_dir).resolve()
    runs_path = compare_dir / "runs.json"
    if runs_path.exists():
        make_compare_report(compare_dir)
    else:
        rows_path = compare_dir / "rows.json"
        if rows_path.exists():
            make_sweep_report(compare_dir)
        else:
            raise FileNotFoundError(f"runs.json or rows.json not found in {compare_dir}")
    print(str(compare_dir))


if __name__ == "__main__":
    main()
