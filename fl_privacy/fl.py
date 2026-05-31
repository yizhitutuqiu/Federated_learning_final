from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .metrics import accuracy


@dataclass(frozen=True)
class RoundResult:
    round_idx: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float


def _params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def evaluate(model: nn.Module, loader: Iterable, device: torch.device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    accs = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(loss.item())
        accs.append(accuracy(logits, y))
    return float(sum(losses) / max(1, len(losses))), float(sum(accs) / max(1, len(accs)))


def compute_grads(model: nn.Module, batch: tuple[torch.Tensor, torch.Tensor], device: torch.device):
    model.train()
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    loss_fn = nn.CrossEntropyLoss()
    logits = model(x)
    loss = loss_fn(logits, y)
    grads = torch.autograd.grad(loss, _params(model), create_graph=False, retain_graph=False)
    grads = [g.detach().clone() for g in grads]
    return loss.item(), logits.detach(), grads, x.detach(), y.detach()


@torch.no_grad()
def apply_grads(model: nn.Module, grads: list[torch.Tensor], lr: float):
    for p, g in zip(_params(model), grads, strict=True):
        p.add_(g, alpha=-lr)


def aggregate_mean(updates: list[list[torch.Tensor]]):
    if not updates:
        raise ValueError("empty updates")
    out = []
    for tensors in zip(*updates, strict=True):
        stacked = torch.stack(tensors, dim=0)
        out.append(torch.mean(stacked, dim=0))
    return out
