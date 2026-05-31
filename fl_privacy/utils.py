from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str | Path, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_image_grid(path: str | Path, images: torch.Tensor, titles: list[str] | None = None):
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))
    import matplotlib.pyplot as plt

    images = images.detach().cpu()
    n = images.shape[0]
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(rows, cols)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= n:
            continue
        img = images[i]
        if img.shape[0] == 1:
            ax.imshow(img[0], cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
