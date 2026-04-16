"""
ts_bridge.export
=================

TauExporter — turns TensionLM's internal τ field into weighted edges on a
`UniversalLivingGraph`.

Usage
-----
    from ts_bridge.export import TauExporter
    from ts_bridge.graph  import UniversalLivingGraph

    graph    = UniversalLivingGraph()
    exporter = TauExporter(graph, edge_threshold=0.3)

    logits, _, all_tensions = model(input_ids, return_all=True)
    exporter.ingest(input_ids[0].tolist(), all_tensions, tokenizer)

    graph.to_json("session_graph.json")

Semantics
---------
For each query position t and each key position s = t - w - 1 (w indexes the
window; s < t is causal), we aggregate τ across the selected heads:

    aggregated[t, s] = mean over (layer, head) in signal_heads of τ[l, t, h, w]

Where `signal_heads` is chosen by `head_filter.select_signal_heads`.

An edge src=s → dst=t with `weight = aggregated[t, s]` is emitted when
aggregated exceeds `edge_threshold` (default 0.3).  Nodes are upserted with:

    id       = f"{content}#{absolute_position}"
    content  = decoded token text
    activation initialised to 0.5, raised on reinforcement via upsert_node.

Position-qualified ids keep repeated tokens distinct at the graph level;
concept-level deduplication is a later-phase job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .graph       import UniversalLivingGraph
from .head_filter import HeadStat, profile_heads, select_signal_heads


@dataclass
class ExportStats:
    """Diagnostic summary returned from each `ingest` call."""
    tokens:        int
    signal_heads:  int
    total_heads:   int
    candidate_pairs: int   # (t, s) pairs evaluated
    edges_emitted: int
    nodes_touched: int
    mean_weight:   float   # over emitted edges


class TauExporter:
    """
    Stateless w.r.t. the graph it writes to — you can reuse one exporter
    across many `ingest` calls.  Head profile is computed per ingest unless
    you call `profile_and_lock(all_tensions)` first to fix it, which is the
    right pattern when you want stable head-selection across a session.
    """

    def __init__(
        self,
        graph:                UniversalLivingGraph,
        edge_threshold:       float = 0.3,
        include_mid_range:    bool  = True,
        head_override:        list[tuple[int, int]] | None = None,
        position_scoped_ids:  bool  = True,
    ):
        self.graph               = graph
        self.edge_threshold      = edge_threshold
        self.include_mid_range   = include_mid_range
        self.position_scoped_ids = position_scoped_ids
        self._locked_heads: list[tuple[int, int]] | None = head_override

    # ── Head selection ───────────────────────────────────────────────────────

    @staticmethod
    def _local_subset(all_tensions: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Filter `all_tensions` to layers that share the most common (H, W)
        shape.  Interleaved global layers carry a [T, H, T] τ rather than
        [T, H, W] — their aggregation is a Phase 2 concern.  Returning the
        local-only list lets head indices stay consistent between
        `profile_and_lock` and `ingest`.
        """
        shapes = [tuple(t.shape[-2:]) for t in all_tensions]
        canonical = max(set(shapes), key=shapes.count)
        return [t for t in all_tensions if tuple(t.shape[-2:]) == canonical]

    def profile_and_lock(self, all_tensions: list[torch.Tensor]) -> list[HeadStat]:
        """
        Profile heads once and reuse the selection for subsequent ingests.
        Use this when streaming tokens one-by-one during generation — you
        don't want the head selection jittering mid-sequence.
        """
        local = self._local_subset(all_tensions)
        stats = profile_heads(local)
        self._locked_heads = select_signal_heads(
            stats, include_mid_range=self.include_mid_range,
        )
        return stats

    def reset_head_lock(self) -> None:
        self._locked_heads = None

    def _current_signal_heads(
        self, local_tensions: list[torch.Tensor],
    ) -> list[tuple[int, int]]:
        """Takes the already-filtered local subset so layer indices align."""
        if self._locked_heads is not None:
            return self._locked_heads
        stats = profile_heads(local_tensions)
        return select_signal_heads(
            stats, include_mid_range=self.include_mid_range,
        )

    # ── Core ingest ───────────────────────────────────────────────────────────

    def ingest(
        self,
        token_ids:    Sequence[int],
        all_tensions: list[torch.Tensor],
        tokenizer,
        position_offset: int = 0,
    ) -> ExportStats:
        """
        token_ids    : list of ints, length T
        all_tensions : list of L tensors, each [B, T, H, W] — first batch item
                       used; additional items are averaged in
        tokenizer    : anything with .id_to_token(int) → str
        position_offset : absolute-position base, so successive generation
                          chunks produce unique node ids even if local T
                          starts at 0 each time
        Returns ExportStats.
        """
        if not all_tensions:
            return ExportStats(
                tokens=len(token_ids), signal_heads=0, total_heads=0,
                candidate_pairs=0, edges_emitted=0, nodes_touched=0,
                mean_weight=0.0,
            )

        T = len(token_ids)
        # Interleaved global layers carry a [T, H, T] τ instead of [T, H, W].
        # Phase 1 aggregates only over the canonical local-window shape —
        # global-layer edges need their own emission path (Phase 2, out of
        # scope here).  Pick the most common W across layers as canonical.
        shapes = [tuple(t.shape[-2:]) for t in all_tensions]       # (H, W) per layer
        canonical = max(set(shapes), key=shapes.count)
        H, W = canonical
        local_tensions: list[torch.Tensor] = []
        local_layer_idx: list[int] = []
        for l, t in enumerate(all_tensions):
            if tuple(t.shape[-2:]) == canonical:
                local_tensions.append(t)
                local_layer_idx.append(l)

        L = len(local_tensions)
        # Stack → [L, T, H, W], batch-mean, trim to T.
        tau = torch.stack([
            t.detach().float().mean(dim=0) for t in local_tensions
        ]).cpu().numpy()[:, :T, :, :]                             # L T H W

        # Head selection runs on the same local subset so indices align.
        signal_heads = self._current_signal_heads(local_tensions)

        # Build a mask over (layer, head) to vectorise the head aggregation.
        if not signal_heads:
            # Fall back to all heads rather than producing an empty graph —
            # likely model not yet trained enough to show specialisation.
            head_mask = np.ones((L, H), dtype=bool)
        else:
            head_mask = np.zeros((L, H), dtype=bool)
            for (l, h) in signal_heads:
                head_mask[l, h] = True

        # Aggregated[t, w] = mean over selected (l, h) of tau[l, t, h, w]
        # Vectorised: broadcast mask into L·H, mean over both.
        lh_tau = tau.transpose(1, 3, 0, 2).reshape(T, W, L * H)    # T W (L·H)
        mask_flat = head_mask.reshape(-1)                           # (L·H,)
        if mask_flat.any():
            aggregated = lh_tau[:, :, mask_flat].mean(axis=-1)      # T W
        else:
            aggregated = lh_tau.mean(axis=-1)                        # T W

        # Convert to (query_pos, key_pos) pairs, then emit edges.
        edges_emitted = 0
        weights_emitted: list[float] = []
        nodes_touched = set()
        candidate_pairs = 0

        for t in range(T):
            query_abs = t + position_offset
            query_tok = tokenizer.id_to_token(token_ids[t]) or f"<{token_ids[t]}>"
            query_id  = (f"{query_tok}#{query_abs}"
                         if self.position_scoped_ids else query_tok)

            # Upsert query node — activation scaled by the token's own
            # average aggregated τ (how much this token is "pulled on").
            own_pull = float(aggregated[t].mean())
            self.graph.upsert_node(
                id         = query_id,
                content    = query_tok,
                activation = min(1.0, 0.3 + own_pull),
                metadata   = {"absolute_pos": query_abs, "token_id": int(token_ids[t])},
            )
            nodes_touched.add(query_id)

            for w in range(W):
                # Key position: t - (W - w).  Forward-kernel convention.
                s = t - (W - w)
                if s < 0:
                    continue
                candidate_pairs += 1
                weight = float(aggregated[t, w])
                if weight < self.edge_threshold:
                    continue

                key_abs = s + position_offset
                key_tok = tokenizer.id_to_token(token_ids[s]) or f"<{token_ids[s]}>"
                key_id  = (f"{key_tok}#{key_abs}"
                           if self.position_scoped_ids else key_tok)

                self.graph.upsert_node(
                    id         = key_id,
                    content    = key_tok,
                    activation = min(1.0, 0.3 + weight),
                    metadata   = {"absolute_pos": key_abs, "token_id": int(token_ids[s])},
                )
                nodes_touched.add(key_id)

                self.graph.add_edge(
                    src     = key_id,
                    dst     = query_id,
                    weight  = weight,
                    relation="tension",
                    metadata={"distance": t - s},
                )
                edges_emitted += 1
                weights_emitted.append(weight)

        return ExportStats(
            tokens          = T,
            signal_heads    = int(head_mask.sum()),
            total_heads     = L * H,
            candidate_pairs = candidate_pairs,
            edges_emitted   = edges_emitted,
            nodes_touched   = len(nodes_touched),
            mean_weight     = float(np.mean(weights_emitted)) if weights_emitted else 0.0,
        )
