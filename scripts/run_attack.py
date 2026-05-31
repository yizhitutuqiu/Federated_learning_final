import argparse
import json
from pathlib import Path

import torch

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fl_privacy.attack import dlg_reconstruct
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

    label = int(y_true.item()) if args.use_true_label else None
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

    out_dir = Path(args.out_dir.strip()) if args.out_dir.strip() else Path(args.obs).resolve().parent
    ensure_dir(out_dir)

    x_recon = res.x_recon
    stats = {
        "obs": str(Path(args.obs).resolve()),
        "round": int(payload["round"]),
        "client_id": int(payload["client_id"]),
        "defense": payload.get("defense", "unknown"),
        "label_true": int(y_true.item()),
        "label_recon": int(res.y_recon),
        "mse": mse(x_recon, x_true),
        "psnr": psnr(x_recon, x_true),
        "loss_final": float(res.losses[-1]) if res.losses else float("nan"),
    }
    with open(out_dir / "attack_metrics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    imgs = torch.cat([x_true, x_recon], dim=0)
    titles = [f"true y={int(y_true.item())}", f"recon y={int(res.y_recon)}"]
    save_image_grid(out_dir / "recon.png", imgs, titles=titles)

    print(str(out_dir))


if __name__ == "__main__":
    main()
