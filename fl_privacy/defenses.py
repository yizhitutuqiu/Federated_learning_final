from __future__ import annotations

from dataclasses import dataclass

import math
import torch


@dataclass(frozen=True)
class DefenseStats:
    global_norm_before: float | None = None
    global_norm_after: float | None = None
    layer_scores: list[float] | None = None
    layer_ps: list[float] | None = None


def _global_l2_norm(tensors: list[torch.Tensor]):
    s = 0.0
    for t in tensors:
        s += torch.sum(t.detach() ** 2).item()
    return math.sqrt(s)


def clip_by_global_norm(
    update: list[torch.Tensor],
    max_norm: float,
):
    norm = _global_l2_norm(update)
    if norm == 0.0:
        return [t.clone() for t in update], DefenseStats(global_norm_before=0.0, global_norm_after=0.0)
    scale = min(1.0, max_norm / (norm + 1e-12))
    clipped = [t * scale for t in update]
    return clipped, DefenseStats(global_norm_before=norm, global_norm_after=norm * scale)


def dp_light(
    update: list[torch.Tensor],
    clip_norm: float,
    noise_multiplier: float,
):
    clipped, stats = clip_by_global_norm(update, max_norm=clip_norm)
    if noise_multiplier <= 0.0:
        return clipped, stats
    sigma = noise_multiplier * clip_norm
    noised = [t + torch.randn_like(t) * sigma for t in clipped]
    return noised, stats


def _concentration_score(g: torch.Tensor, eps: float):
    v = g.detach().flatten()
    d = v.numel()
    l2 = torch.linalg.vector_norm(v, ord=2).item()
    l1 = torch.linalg.vector_norm(v, ord=1).item()
    return (math.sqrt(d) * l2) / (l1 + eps), d


def laugd(
    update: list[torch.Tensor],
    alpha: float,
    tau: float,
    p_max: float,
    eps: float = 1e-12,
    normalize_score: bool = False,
    unbiased: bool = True,
    mask_mode: str = "bernoulli",
    fixed_p: float | None = None,
):
    layer_scores: list[float] = []
    layer_ps: list[float] = []
    out: list[torch.Tensor] = []

    for t in update:
        if t.numel() == 0:
            out.append(t.clone())
            layer_scores.append(0.0)
            layer_ps.append(0.0)
            continue

        score, d = _concentration_score(t, eps=eps)
        if normalize_score:
            denom = max(1.0, math.sqrt(d) - 1.0)
            score = max(0.0, min(1.0, (score - 1.0) / denom))

        if fixed_p is None:
            p = alpha * (score - tau)
            p = float(max(0.0, min(p_max, p)))
        else:
            p = float(max(0.0, min(p_max, fixed_p)))

        if p <= 0.0:
            out.append(t.clone())
            layer_scores.append(float(score))
            layer_ps.append(p)
            continue

        if mask_mode == "bernoulli":
            mask = (torch.rand_like(t) > p).to(dtype=t.dtype)
        elif mask_mode == "fixed_budget":
            d_all = t.numel()
            k_keep = int(round((1.0 - p) * d_all))
            k_keep = max(0, min(d_all, k_keep))
            idx = torch.randperm(d_all, device=t.device)[:k_keep]
            mask = torch.zeros(d_all, device=t.device, dtype=t.dtype)
            mask[idx] = 1.0
            mask = mask.view_as(t)
        else:
            raise ValueError(f"unknown mask_mode={mask_mode}")

        if unbiased:
            keep_prob = 1.0 - p
            scaled = (mask * t) / keep_prob
        else:
            scaled = mask * t

        out.append(scaled)
        layer_scores.append(float(score))
        layer_ps.append(p)

    return out, DefenseStats(layer_scores=layer_scores, layer_ps=layer_ps)
