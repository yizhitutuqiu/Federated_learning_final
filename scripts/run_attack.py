import argparse
import json
from pathlib import Path

import torch

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fl_privacy.attack import dlg_reconstruct, ig_reconstruct, lbfgs_reconstruct
from fl_privacy.metrics import mse, psnr
from fl_privacy.models import SimpleCNN
from fl_privacy.utils import ensure_dir, get_device, save_image_grid, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--obs", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--tv-reg", type=float, default=1e-4)
    p.add_argument("--method", type=str, default="dlg", choices=["dlg", "ig", "lbfgs"])
    p.add_argument("--l2-reg", type=float, default=0.0)
    p.add_argument("--use-true-label", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    payload = torch.load(args.obs, map_location="cpu")
    model = SimpleCNN()
    model.load_state_dict(payload["model_state"])

    x_true = payload["x"]
    y_true = payload["y"]
    grads_obs = payload["grads_obs"]

    if isinstance(x_true, torch.Tensor) and x_true.dim() >= 1 and x_true.shape[0] > 1:
        x_true = x_true[:1]
    if isinstance(y_true, torch.Tensor) and y_true.dim() >= 1 and y_true.shape[0] > 1:
        y_true = y_true[:1]

    label = int(y_true.item()) if args.use_true_label else None
    if args.method == "dlg":
        res = dlg_reconstruct(
            model=model,
            grads_obs=grads_obs,
            input_shape=tuple(x_true.shape[1:]),
            num_classes=10,
            device=device,
            iters=args.iters,
            lr=args.lr,
            tv_reg=args.tv_reg,
            seed=args.seed,
            label=label,
        )
    else:
        if args.method == "ig":
            res = ig_reconstruct(
                model=model,
                grads_obs=grads_obs,
                input_shape=tuple(x_true.shape[1:]),
                num_classes=10,
                device=device,
                iters=args.iters,
                lr=args.lr,
                tv_reg=args.tv_reg,
                l2_reg=args.l2_reg,
                seed=args.seed,
                label=label,
            )
        else:
            res = lbfgs_reconstruct(
                model=model,
                grads_obs=grads_obs,
                input_shape=tuple(x_true.shape[1:]),
                num_classes=10,
                device=device,
                iters=args.iters,
                lr=args.lr,
                tv_reg=args.tv_reg,
                l2_reg=args.l2_reg,
                seed=args.seed,
                label=label,
            )

    out_dir = Path(args.out_dir.strip()) if args.out_dir.strip() else Path(args.obs).resolve().parent
    ensure_dir(out_dir)

    x_recon = res.x_recon
    losses = [float(x) for x in (res.losses or [])]
    stats = {
        "obs": str(Path(args.obs).resolve()),
        "round": int(payload["round"]),
        "client_id": int(payload["client_id"]),
        "defense": payload.get("defense", "unknown"),
        "attack_method": str(args.method),
        "label_true": int(y_true.item()),
        "label_recon": int(res.y_recon),
        "mse": mse(x_recon, x_true),
        "psnr": psnr(x_recon, x_true),
        "loss_final": losses[-1] if losses else float("nan"),
    }
    with open(out_dir / "attack_metrics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    with open(out_dir / "losses.json", "w", encoding="utf-8") as f:
        json.dump({"losses": losses}, f, ensure_ascii=False)

    if losses:
        import os

        os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.2))
        ax.plot(list(range(len(losses))), losses, linewidth=1.5)
        ax.set_xlabel("iter")
        ax.set_ylabel("recon loss")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "loss_curve.png", dpi=160)
        plt.close(fig)

    imgs = torch.cat([x_true, x_recon], dim=0)
    titles = [f"true y={int(y_true.item())}", f"recon y={int(res.y_recon)}"]
    save_image_grid(out_dir / "recon.png", imgs, titles=titles)

    print(str(out_dir))


if __name__ == "__main__":
    main()
