from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AttackResult:
    x_recon: torch.Tensor
    y_recon: int
    losses: list[float]


def _params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def infer_label_idlg(grads: list[torch.Tensor], num_classes: int):
    last = grads[-1].detach().flatten()
    if last.numel() != num_classes:
        return int(torch.argmin(last).item())
    return int(torch.argmin(last).item())


def total_variation(x: torch.Tensor):
    tv_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    return tv_h + tv_w


def dlg_reconstruct(
    model: nn.Module,
    grads_obs: list[torch.Tensor],
    input_shape: tuple[int, int, int],
    num_classes: int,
    device: torch.device,
    iters: int = 800,
    lr: float = 0.1,
    tv_reg: float = 1e-4,
    seed: int | None = None,
    label: int | None = None,
):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    model.eval()
    grads_obs = [g.detach().to(device) for g in grads_obs]

    if label is None:
        y = infer_label_idlg(grads_obs, num_classes=num_classes)
    else:
        y = int(label)

    y_t = torch.tensor([y], device=device, dtype=torch.long)
    x_var = torch.randn((1, *input_shape), device=device, requires_grad=True)
    opt = torch.optim.Adam([x_var], lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    losses: list[float] = []
    params = _params(model)

    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        x = torch.sigmoid(x_var)
        logits = model(x)
        loss = loss_fn(logits, y_t)
        grads_hat = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

        match = 0.0
        for gh, go in zip(grads_hat, grads_obs, strict=True):
            match = match + torch.mean((gh - go) ** 2)

        reg = tv_reg * total_variation(x)
        obj = match + reg
        obj.backward()
        opt.step()
        losses.append(float(obj.detach().item()))

    x_recon = torch.sigmoid(x_var.detach()).cpu()
    return AttackResult(x_recon=x_recon, y_recon=y, losses=losses)


def ig_reconstruct(
    model: nn.Module,
    grads_obs: list[torch.Tensor],
    input_shape: tuple[int, int, int],
    num_classes: int,
    device: torch.device,
    iters: int = 800,
    lr: float = 0.1,
    tv_reg: float = 1e-4,
    l2_reg: float = 0.0,
    seed: int | None = None,
    label: int | None = None,
):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    model.eval()
    grads_obs = [g.detach().to(device) for g in grads_obs]

    if label is None:
        y = infer_label_idlg(grads_obs, num_classes=num_classes)
    else:
        y = int(label)

    y_t = torch.tensor([y], device=device, dtype=torch.long)
    x_var = torch.randn((1, *input_shape), device=device, requires_grad=True)
    opt = torch.optim.Adam([x_var], lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    losses: list[float] = []
    params = _params(model)

    eps = 1e-12
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        x = torch.sigmoid(x_var)
        logits = model(x)
        loss = loss_fn(logits, y_t)
        grads_hat = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

        match = 0.0
        for gh, go in zip(grads_hat, grads_obs, strict=True):
            gh_f = gh.flatten()
            go_f = go.flatten()
            denom = (torch.linalg.vector_norm(gh_f, ord=2) * torch.linalg.vector_norm(go_f, ord=2)) + eps
            cos = torch.sum(gh_f * go_f) / denom
            match = match + (1.0 - cos)

        reg = tv_reg * total_variation(x)
        if l2_reg > 0.0:
            reg = reg + float(l2_reg) * torch.mean(x**2)

        obj = match + reg
        obj.backward()
        opt.step()
        losses.append(float(obj.detach().item()))

    x_recon = torch.sigmoid(x_var.detach()).cpu()
    return AttackResult(x_recon=x_recon, y_recon=y, losses=losses)


def lbfgs_reconstruct(
    model: nn.Module,
    grads_obs: list[torch.Tensor],
    input_shape: tuple[int, int, int],
    num_classes: int,
    device: torch.device,
    iters: int = 300,
    lr: float = 1.0,
    tv_reg: float = 1e-4,
    l2_reg: float = 0.0,
    seed: int | None = None,
    label: int | None = None,
):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    model.eval()
    grads_obs = [g.detach().to(device) for g in grads_obs]

    if label is None:
        y = infer_label_idlg(grads_obs, num_classes=num_classes)
    else:
        y = int(label)

    y_t = torch.tensor([y], device=device, dtype=torch.long)
    x_var = torch.randn((1, *input_shape), device=device, requires_grad=True)
    loss_fn = nn.CrossEntropyLoss()
    params = _params(model)

    losses: list[float] = []
    eps = 1e-12

    opt = torch.optim.LBFGS(
        [x_var],
        lr=lr,
        max_iter=1,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    for _ in range(iters):
        def closure():
            opt.zero_grad(set_to_none=True)
            x = torch.sigmoid(x_var)
            logits = model(x)
            loss = loss_fn(logits, y_t)
            grads_hat = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

            match = 0.0
            for gh, go in zip(grads_hat, grads_obs, strict=True):
                gh_f = gh.flatten()
                go_f = go.flatten()
                denom = (torch.linalg.vector_norm(gh_f, ord=2) * torch.linalg.vector_norm(go_f, ord=2)) + eps
                cos = torch.sum(gh_f * go_f) / denom
                match = match + (1.0 - cos)

            reg = tv_reg * total_variation(x)
            if l2_reg > 0.0:
                reg = reg + float(l2_reg) * torch.mean(x**2)

            obj = match + reg
            obj.backward()
            return obj

        val = opt.step(closure)
        losses.append(float(val.detach().item()) if isinstance(val, torch.Tensor) else float(val))

    x_recon = torch.sigmoid(x_var.detach()).cpu()
    return AttackResult(x_recon=x_recon, y_recon=y, losses=losses)
