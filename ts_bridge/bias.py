"""
ts_bridge.bias
===============

Phase 2 — substrate → surface biasing.

The export pipeline (Phase 1) writes TensionLM's τ field into a graph.  This
module does the reverse: given a graph, produce an additive bias on the τ
precursor logit (pre-sigmoid) so the graph's accumulated structure directly
enters the attention computation.  No prompt injection, no retrieval — the
substrate's current state conditions the surface's generation at the
attention-mechanism level.

Interface
---------
    bias_engine = GraphBias.from_graph(graph, alpha=0.5)
    bias = bias_engine.local_bias(ctx_ids, tokenizer,
                                  window=cfg.window, batch=1)
    logits = model(ctx_tensor, tau_bias=bias)

Shape
-----
The returned bias is shaped [B, T, W] — the same [B, T, W] the local-window
kernel expects.  Key-position convention matches TauExporter.ingest:

    bias[b, t, w] = α · edge_weight(
        from = ctx_content[t - (W - w)],
        to   = ctx_content[t],
    )

i.e. w=0 is the oldest key in the window (distance W behind the query), and
w=W-1 is the newest (distance 1).  Out-of-bounds keys (t - (W - w) < 0) get
zero bias.

Content addressing
------------------
The Phase 1 exporter stores position-scoped node ids (`{content}#{abs_pos}`)
so repeat tokens stay distinct in the graph.  At bias time we deliberately
collapse across positions: the biasing question is concept-level ("has
`dolphins` tended to pull on `mammals`?"), not position-level.  We aggregate
edge weights by `(src.content, dst.content)` via max — same reinforcement
semantics the graph already uses for `add_edge`.

This is a pragmatic stand-in for the concept-extraction pass that Phase 1's
schema flagged as out-of-scope.  Once concepts exist, swap the key from
`content` to `concept_id` and the rest of this module is unchanged.

Global layers
-------------
Global-attention layers use a [B, T, T] bias shape and different semantics
(all past tokens, not a window).  `global_bias` builds the [B, T, T]
counterpart to `local_bias`: bias[b, t, s] = α · edge_weight(ctx[s], ctx[t])
for s < t, zero otherwise.  `TensionLM.forward` takes it via the
`tau_bias_global` kwarg; callers can pass local, global, or both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .graph import UniversalLivingGraph


def _content_of(node_id: str) -> str:
    """Strip `#pos` suffix if position-scoped; else return as-is."""
    pos = node_id.rfind("#")
    return node_id if pos < 0 else node_id[:pos]


@dataclass
class BiasStats:
    """Cheap diagnostics so callers can tell if the bias ever fires."""
    nonzero_pairs: int = 0     # number of (t, w) positions that received a boost
    max_weight:    float = 0.0 # largest edge weight applied on this forward
    mean_weight:   float = 0.0 # mean over nonzero positions


class GraphBias:
    """
    Content-addressed edge index + local-window bias builder.

    Cheap to construct (one pass over graph.edges), so rebuilding it every
    generation step is fine.  For very large graphs or very hot loops, keep
    one instance and call `local_bias` repeatedly.
    """

    def __init__(self, edge_map: dict[tuple[str, str], float], alpha: float):
        self.edge_map = edge_map
        self.alpha    = alpha

    # ── Build ────────────────────────────────────────────────────────────────

    @classmethod
    def from_graph(
        cls,
        graph:    UniversalLivingGraph,
        alpha:    float = 0.5,
        relation: str   = "tension",
    ) -> "GraphBias":
        """
        Collapse position-scoped edges to (src_content, dst_content) -> weight
        via max.  Only edges whose relation matches `relation` are included
        (keeps future typed edges from polluting the τ-bias channel).
        """
        edge_map: dict[tuple[str, str], float] = {}
        for e in graph.edges:
            if e.relation != relation:
                continue
            src_c = _content_of(e.src)
            dst_c = _content_of(e.dst)
            key   = (src_c, dst_c)
            prev  = edge_map.get(key, 0.0)
            if e.weight > prev:
                edge_map[key] = e.weight
        return cls(edge_map=edge_map, alpha=alpha)

    # ── Query / build ────────────────────────────────────────────────────────

    def lookup(self, src_content: str, dst_content: str) -> float:
        return self.edge_map.get((src_content, dst_content), 0.0)

    def local_bias(
        self,
        ctx_ids:    Sequence[int],
        tokenizer,
        window:     int,
        batch:      int = 1,
        device:     torch.device | str = "cpu",
        dtype:      torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, BiasStats]:
        """
        Build a [B, T, W] tensor of α · edge_weight values aligned to the
        causal-window kernel's (t, w) indexing.

        Returns (bias_tensor, stats).  The stats are cheap and handy for
        debugging ("did the graph ever actually bias anything on this run?").
        """
        T = len(ctx_ids)
        W = window

        # Decode ctx tokens once.
        contents = [
            tokenizer.id_to_token(int(tid)) or f"<{int(tid)}>"
            for tid in ctx_ids
        ]

        bias = torch.zeros((batch, T, W), dtype=dtype, device=device)
        hits = 0
        total_w = 0.0
        max_w   = 0.0

        for t in range(T):
            dst_c = contents[t]
            for w in range(W):
                s = t - (W - w)
                if s < 0:
                    continue
                weight = self.edge_map.get((contents[s], dst_c), 0.0)
                if weight <= 0.0:
                    continue
                boost = self.alpha * weight
                bias[:, t, w] = boost
                hits    += 1
                total_w += weight
                if weight > max_w:
                    max_w = weight

        stats = BiasStats(
            nonzero_pairs = hits,
            max_weight    = max_w,
            mean_weight   = total_w / hits if hits else 0.0,
        )
        return bias, stats

    def global_bias(
        self,
        ctx_ids:    Sequence[int],
        tokenizer,
        batch:      int = 1,
        device:     torch.device | str = "cpu",
        dtype:      torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, BiasStats]:
        """
        Build a [B, T, T] tensor of α · edge_weight values for the full-
        sequence (global) attention path.

            bias[b, t, s] = α · edge_weight(
                from = ctx_content[s],
                to   = ctx_content[t],
            )   for  s < t

        Zero for s ≥ t (no self-bias, and the causal mask inside the model
        zeros out s > t anyway).  Shape matches TensionLM's global-layer
        score tensor on its last dim, and is broadcast over H inside the
        layer.

        O(T²) Python loop; fine at T ≤ 2048 but a candidate for
        vectorisation if global-layer biasing becomes a hot path.
        """
        T = len(ctx_ids)

        contents = [
            tokenizer.id_to_token(int(tid)) or f"<{int(tid)}>"
            for tid in ctx_ids
        ]

        bias = torch.zeros((batch, T, T), dtype=dtype, device=device)
        hits = 0
        total_w = 0.0
        max_w   = 0.0

        for t in range(T):
            dst_c = contents[t]
            for s in range(t):
                weight = self.edge_map.get((contents[s], dst_c), 0.0)
                if weight <= 0.0:
                    continue
                boost = self.alpha * weight
                bias[:, t, s] = boost
                hits    += 1
                total_w += weight
                if weight > max_w:
                    max_w = weight

        stats = BiasStats(
            nonzero_pairs = hits,
            max_weight    = max_w,
            mean_weight   = total_w / hits if hits else 0.0,
        )
        return bias, stats
