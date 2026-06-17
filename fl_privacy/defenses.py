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


def svd_project(
    update: list[torch.Tensor],
    rank_ratio: float,
):
    if rank_ratio <= 0.0 or rank_ratio > 1.0:
        raise ValueError("rank_ratio must be in (0, 1]")
    out: list[torch.Tensor] = []
    for t in update:
        if t.numel() == 0:
            out.append(t.clone())
            continue
        if t.dim() < 2:
            out.append(t.clone())
            continue

        m = t.shape[0]
        n = int(t.numel() // m)
        a = t.detach().reshape(m, n)
        r = int(max(1, round(rank_ratio * min(m, n))))
        u, s, v_h = torch.linalg.svd(a, full_matrices=False)
        u_r = u[:, :r]
        s_r = s[:r]
        v_r = v_h[:r, :]
        approx = (u_r * s_r) @ v_r
        out.append(approx.reshape_as(t).to(dtype=t.dtype))
    return out, DefenseStats()


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
    mag_aware: bool = False,
    mag_gamma: float = 1.0,
    keep_prob_min: float = 0.2,
    last_layer_mult: float = 1.0,
    head_mult: float = 1.0,
    head_layers: int = 2,
):
    layer_scores: list[float] = []
    layer_ps: list[float] = []
    out: list[torch.Tensor] = []

    for layer_idx, t in enumerate(update):
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

        if last_layer_mult != 1.0 and layer_idx == (len(update) - 1):
            p = float(max(0.0, min(p_max, p * float(last_layer_mult))))

        if head_mult != 1.0 and head_layers > 0 and layer_idx >= (len(update) - int(head_layers)):
            p = float(max(0.0, min(p_max, p * float(head_mult))))

        if p <= 0.0:
            out.append(t.clone())
            layer_scores.append(float(score))
            layer_ps.append(p)
            continue

        if mask_mode == "bernoulli":
            if mag_aware:
                if keep_prob_min <= 0.0 or keep_prob_min >= 1.0:
                    raise ValueError("keep_prob_min must be in (0, 1)")
                g_abs = t.detach().abs()
                mean_abs = g_abs.mean()
                w = (g_abs / (mean_abs + eps)).pow(float(mag_gamma))
                w_mean = w.mean()
                drop_p = p * w / (w_mean + eps)
                drop_p = torch.clamp(drop_p, 0.0, 1.0 - float(keep_prob_min))
                keep_prob = 1.0 - drop_p
                mask = (torch.rand_like(t) < keep_prob).to(dtype=t.dtype)
            else:
                keep_prob = 1.0 - p
                mask = (torch.rand_like(t) > p).to(dtype=t.dtype)
        elif mask_mode == "channel":
            if mag_aware:
                raise ValueError("mag_aware is only supported with mask_mode=bernoulli")
            keep_prob = 1.0 - p
            if t.dim() >= 2:
                c0 = t.shape[0]
                m = (torch.rand((c0,), device=t.device) < keep_prob).to(dtype=t.dtype)
                view = (c0,) + (1,) * (t.dim() - 1)
                mask = m.view(view).expand_as(t)
            else:
                mask = (torch.rand_like(t) > p).to(dtype=t.dtype)
        elif mask_mode == "fixed_budget":
            if mag_aware:
                raise ValueError("mag_aware is only supported with mask_mode=bernoulli")
            d_all = t.numel()
            k_keep = int(round((1.0 - p) * d_all))
            k_keep = max(0, min(d_all, k_keep))
            keep_prob = float(k_keep / d_all) if d_all > 0 else 0.0
            idx = torch.randperm(d_all, device=t.device)[:k_keep]
            mask = torch.zeros(d_all, device=t.device, dtype=t.dtype)
            mask[idx] = 1.0
            mask = mask.view_as(t)
        else:
            raise ValueError(f"unknown mask_mode={mask_mode}")

        if unbiased:
            if isinstance(keep_prob, float):
                if keep_prob <= 0.0:
                    scaled = torch.zeros_like(t)
                else:
                    scaled = (mask * t) / keep_prob
            else:
                kp = keep_prob.to(dtype=t.dtype)
                if torch.all(kp <= 0.0):
                    scaled = torch.zeros_like(t)
                else:
                    scaled = (mask * t) / kp
        else:
            scaled = mask * t

        out.append(scaled)
        layer_scores.append(float(score))
        layer_ps.append(p)

    return out, DefenseStats(layer_scores=layer_scores, layer_ps=layer_ps)
