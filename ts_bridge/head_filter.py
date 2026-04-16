"""
ts_bridge.head_filter
======================

Classify TensionLM heads by specialisation and select the ones that produce
useful edges for the external graph.

Heuristic (ports the role-guess logic from `visualise.py` mode_stats, with a
more formal API and a typed role enum):

    inactive        : mean τ < 0.15              → no useful signal
    syntactic       : peak_pos < 0.2·W, variance > 0.05
                                                  → recent-token tracking;
                                                    noise at graph level
    long_range      : peak_pos > 0.7·W           → long-distance semantic;
                                                    THE ONES WE WANT
    diffuse         : variance < 0.02            → broad background; weak
                                                    signal, low graph ROI
    mid_range       : otherwise                  → ambiguous; include with
                                                    reduced weight

Only `long_range` and (optionally) `mid_range` heads contribute to exported
edges. This is the lever that decides what the graph "hears" from the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np
import torch


class HeadRole(str, Enum):
    INACTIVE   = "inactive"
    SYNTACTIC  = "syntactic"
    LONG_RANGE = "long_range"
    DIFFUSE    = "diffuse"
    MID_RANGE  = "mid_range"


@dataclass
class HeadStat:
    layer:      int
    head:       int
    mean_tau:   float
    peak_pos:   float   # in window units [0, W)
    variance:   float
    role:       HeadRole

    @property
    def carries_graph_signal(self) -> bool:
        """Whether this head should contribute to exported edges."""
        return self.role in (HeadRole.LONG_RANGE, HeadRole.MID_RANGE)


def _classify(mean_tau: float, peak_pos: float, variance: float,
              W: int) -> HeadRole:
    if mean_tau < 0.15:
        return HeadRole.INACTIVE
    if peak_pos < W * 0.2 and variance > 0.05:
        return HeadRole.SYNTACTIC
    if peak_pos > W * 0.7:
        return HeadRole.LONG_RANGE
    if variance < 0.02:
        return HeadRole.DIFFUSE
    return HeadRole.MID_RANGE


def profile_heads(
    all_tensions: list[torch.Tensor],
) -> list[HeadStat]:
    """
    all_tensions: list of L tensors, each [B, T, H, W] (the `return_all=True`
                  output of `TensionLM.forward`).
    Returns one HeadStat per (layer, head).
    """
    stats: list[HeadStat] = []
    for l, tau in enumerate(all_tensions):
        # Average over batch; convert to numpy for cheap stats on CPU.
        t = tau.detach().float().mean(dim=0).cpu().numpy()   # T H W
        T, H, W = t.shape
        for h in range(H):
            vals   = t[:, h, :]                               # T W
            mean_t = float(vals.mean())
            peak   = float(vals.argmax(axis=-1).mean())
            var    = float(vals.var())
            stats.append(HeadStat(
                layer     = l,
                head      = h,
                mean_tau  = mean_t,
                peak_pos  = peak,
                variance  = var,
                role      = _classify(mean_t, peak, var, W),
            ))
    return stats


def select_signal_heads(
    stats: Iterable[HeadStat],
    include_mid_range: bool = True,
) -> list[tuple[int, int]]:
    """
    Return [(layer, head), ...] for heads that carry graph-level signal.
    `include_mid_range=False` is the stricter filter — use it when the graph
    is getting noisy.
    """
    allowed = {HeadRole.LONG_RANGE}
    if include_mid_range:
        allowed.add(HeadRole.MID_RANGE)
    return [(s.layer, s.head) for s in stats if s.role in allowed]


def format_profile(stats: Iterable[HeadStat]) -> str:
    """Human-readable table of head stats, same columns as visualise.py."""
    lines = [f"{'L':<4}{'H':<4}{'mean τ':<10}{'peak':<8}{'var':<10}role"]
    lines.append("─" * 48)
    prev_l = None
    for s in stats:
        if prev_l is not None and s.layer != prev_l:
            lines.append("")
        lines.append(
            f"{s.layer:<4}{s.head:<4}"
            f"{s.mean_tau:<10.3f}{s.peak_pos:<8.1f}{s.variance:<10.4f}"
            f"{s.role.value}"
        )
        prev_l = s.layer
    return "\n".join(lines)
