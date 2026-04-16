"""
ts_bridge.head_filter
======================

Classify TensionLM heads by specialisation and select the ones that produce
useful edges for the external graph.

Convention (matches `MultiHeadCausalTensionLayer._gather_window` in model.py):
    w = 0      → oldest token in the window (t - W)
    w = W - 1  → most recent prior token   (t - 1)

So `peak_pos` near 0 means the head attends to distant past (long-range
semantic), and peak_pos near W-1 means it tracks the most recent token
(short-range syntactic).  An earlier version of this file had the labels
inverted; the fix landed with the Phase 1 recalibration.

Activity is measured by **peak** τ, not mean τ.  Sigmoid tension produces
sparse/bimodal distributions (median ≈ 0, most mass ≈ 0, occasional spikes
near 1) — a head's capability shows up in its p95/max, not its mean.  Using
mean here marks every head "inactive" on real checkpoints.

Selection (what exports contribute to the graph):
    - LONG_RANGE heads always contribute.
    - MID_RANGE heads contribute when `include_mid_range=True` (default on
      current-scale models — essentially nothing classifies LONG_RANGE yet).
    - SYNTACTIC / INACTIVE / DIFFUSE are filtered out.

Known limitation — per-prompt instability:
    Head classification is content-dependent.  On the 117M-Curriculum
    checkpoint, pairwise Jaccard overlap of signal-head selections across
    different long prompts sits around 0.33–0.69; the intersection across
    5 prompts can be zero.  `profile_and_lock(one_prompt)` therefore locks
    to that prompt's activity pattern, not to intrinsic semantic heads.

    Phase 1.5 followup: corpus-level profiling.  Profile each head across
    a large mixed corpus, take the *average* concentration / max_tau /
    peak_pos, and lock the signal set from those aggregated stats.  The
    current API keeps `profile_and_lock` for generation-time use but the
    stable selection should come from a one-off pre-compute.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np
import torch


class HeadRole(str, Enum):
    INACTIVE   = "inactive"      # never fires sharply
    SYNTACTIC  = "syntactic"     # tracks recent (high w) — position, not semantics
    LONG_RANGE = "long_range"    # tracks distant past (low w) — semantic signal
    DIFFUSE    = "diffuse"       # flat over window — weak signal
    MID_RANGE  = "mid_range"     # peaky but not extreme either way


@dataclass
class HeadStat:
    layer:          int
    head:           int
    mean_tau:       float        # overall mean (informational only)
    max_tau:        float        # p99 — capability to fire sharply
    peak_pos:       float        # mean argmax over the window, in [0, W)
    concentration:  float        # mean max-over-w of per-query τ (higher = peakier)
    role:           HeadRole

    @property
    def carries_graph_signal(self) -> bool:
        """Whether this head should contribute to exported edges."""
        return self.role in (HeadRole.LONG_RANGE, HeadRole.MID_RANGE)


# Thresholds calibrated against the 117M-Curriculum checkpoint's τ distribution:
#   most heads max_tau ~ 0.4–0.95 (sparse spikes), mean_tau ~ 0.05–0.17.
#   Using mean < 0.15 as "inactive" miscategorises every head — fixed here.
_MAX_TAU_FLOOR   = 0.40   # a head must be capable of firing (p99 > 0.4) to count
_CONCENTRATION   = 0.35   # mean of max-over-w must exceed this to be non-diffuse


def _classify(mean_tau: float, max_tau: float, peak_pos: float,
              concentration: float, W: int) -> HeadRole:
    if max_tau < _MAX_TAU_FLOOR:
        return HeadRole.INACTIVE
    if concentration < _CONCENTRATION:
        return HeadRole.DIFFUSE
    if peak_pos > W * 0.7:
        # Attends to most-recent tokens → position/syntax tracking.
        return HeadRole.SYNTACTIC
    if peak_pos < W * 0.3:
        # Attends to distant past → long-range semantic.
        return HeadRole.LONG_RANGE
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
        t = tau.detach().float().mean(dim=0).cpu().numpy()    # T H W
        T, H, W = t.shape
        for h in range(H):
            vals = t[:, h, :]                                  # T W
            mean_t = float(vals.mean())
            # p99 captures the sharp-firing capability without being fooled
            # by a single outlier (which a plain max would be).
            max_t  = float(np.quantile(vals, 0.99))
            peak   = float(vals.argmax(axis=-1).mean())
            # Mean of per-query peaks: how sharply does τ[t, :] localise
            # mass on its best key?  Uniform τ → concentration ≈ 1/W; peaked
            # τ → concentration → 1.
            concentration = float(vals.max(axis=-1).mean())
            stats.append(HeadStat(
                layer         = l,
                head          = h,
                mean_tau      = mean_t,
                max_tau       = max_t,
                peak_pos      = peak,
                concentration = concentration,
                role          = _classify(mean_t, max_t, peak, concentration, W),
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
    """Human-readable table of head stats."""
    lines = [f"{'L':<4}{'H':<4}{'mean τ':<9}{'max τ':<9}"
             f"{'peak':<7}{'conc.':<8}role"]
    lines.append("─" * 52)
    prev_l = None
    for s in stats:
        if prev_l is not None and s.layer != prev_l:
            lines.append("")
        lines.append(
            f"{s.layer:<4}{s.head:<4}"
            f"{s.mean_tau:<9.3f}{s.max_tau:<9.3f}"
            f"{s.peak_pos:<7.1f}{s.concentration:<8.3f}"
            f"{s.role.value}"
        )
        prev_l = s.layer
    return "\n".join(lines)
