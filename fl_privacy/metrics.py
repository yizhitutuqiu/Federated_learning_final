from __future__ import annotations

import math

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def mse(x: torch.Tensor, y: torch.Tensor):
    return torch.mean((x - y) ** 2).item()


@torch.no_grad()
def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0):
    v = mse(x, y)
    if v == 0.0:
        return float("inf")
    return 10.0 * math.log10((data_range**2) / v)
