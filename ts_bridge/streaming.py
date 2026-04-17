"""
ts_bridge.streaming
====================

Generation-time streaming variant of TauExporter.

The Phase 1 batch exporter ingests an entire forward pass at once — fine for
offline analysis, wasteful for autoregressive generation where one new query
token appears per step.  StreamingTauExporter primes on the prompt forward
pass, then per-step emits edges from only the new query (the last position in
the rolling context) back to its W keys.

This is the substrate Phase 2 (graph → surface biasing) needs: every
generation step writes its τ-edges into the live graph before the next step
runs, so the graph state available for biasing is always up to date with what
has just been said.

Pattern:
    streamer = StreamingTauExporter(graph,
                                    edge_threshold=0.27,
                                    head_override=corpus_signal_heads)
    _, _, all_tau = model(prompt_ctx, return_all=True)
    streamer.prime(prompt_ids, all_tau, tokenizer)

    ids = list(prompt_ids)
    abs_pos = len(ids) - 1
    next_id = sample_from(logits[0, -1])
    ids.append(next_id); abs_pos += 1

    for _ in range(max_new - 1):
        ctx_list = ids[-max_ctx:]
        _, _, all_tau = model(ctx_tensor, return_all=True)
        streamer.ingest_step(ctx_list, all_tau, tokenizer, query_abs_pos=abs_pos)
        next_id = sample_from(logits[0, -1])
        ids.append(next_id); abs_pos += 1
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .export import ExportStats, TauExporter
from .graph  import UniversalLivingGraph


@dataclass
class StreamStats:
    steps:         int = 0          # ingest_step calls
    edges_emitted: int = 0          # cumulative across steps
    nodes_touched: int = 0          # unique node ids the streamer has upserted


class StreamingTauExporter(TauExporter):
    """
    Subclass that adds `prime` (batch-ingest the prompt + lock heads) and
    `ingest_step` (emit edges for one new query token).  Reuses the parent's
    head selection, edge_threshold, and node-id conventions verbatim so the
    streamed graph is bit-identical to what a batch ingest of the same final
    sequence would produce — modulo the absence of self-edges from positions
    we never observed as "the last query" (i.e. trailing tokens that appear
    in context but never get a forward pass after them).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._touched_nodes: set[str] = set()
        self._stream_steps: int   = 0
        self._stream_edges: int   = 0

    # ── Prompt prime ────────────────────────────────────────────────────────

    def prime(
        self,
        prompt_ids,
        all_tensions: list[torch.Tensor],
        tokenizer,
        position_offset: int = 0,
    ) -> ExportStats:
        """
        Batch-ingest the prompt forward pass and lock heads so subsequent
        ingest_step calls reuse the same selection.  If the caller supplied
        head_override at construction time, that override stays — we only
        profile-and-lock when nothing is set.
        """
        if self._locked_heads is None:
            self.profile_and_lock(all_tensions)
        stats = self.ingest(prompt_ids, all_tensions, tokenizer,
                            position_offset=position_offset)
        # Track touched nodes by replaying the same id construction the
        # parent ingest used — graph itself doesn't expose a "what got
        # added in this call" API, so we just snapshot all node ids now.
        self._touched_nodes.update(self.graph.nodes.keys())
        return stats

    # ── Per-step streaming ingest ──────────────────────────────────────────

    def ingest_step(
        self,
        ctx_ids,
        all_tensions: list[torch.Tensor],
        tokenizer,
        query_abs_pos: int,
    ) -> int:
        """
        Emit edges from the LAST position in ctx_ids (the new query token,
        at absolute position `query_abs_pos`) back to its W key positions —
        the W tokens immediately preceding it in ctx_ids.  Returns the
        number of edges emitted this step.
        """
        if not all_tensions or not ctx_ids:
            return 0

        local_tensions = self._local_subset(all_tensions)
        if not local_tensions:
            return 0
        L = len(local_tensions)
        H = int(local_tensions[0].shape[-2])
        W = int(local_tensions[0].shape[-1])

        # Stack only the τ row for the last query position across local
        # layers: shape [L, H, W].  Average over batch (B usually 1).
        tau_last = torch.stack([
            t.detach().float()[:, -1, :, :].mean(dim=0) for t in local_tensions
        ]).cpu().numpy()                                       # L H W

        # Build the head mask the same way the batch ingest does.
        if self._locked_heads:
            head_mask = np.zeros((L, H), dtype=bool)
            for (l, h) in self._locked_heads:
                if 0 <= l < L and 0 <= h < H:
                    head_mask[l, h] = True
            if not head_mask.any():
                head_mask = np.ones((L, H), dtype=bool)
        else:
            head_mask = np.ones((L, H), dtype=bool)

        # aggregated[w] = mean over selected (l, h) of tau_last[l, h, w]
        lh = tau_last.transpose(2, 0, 1).reshape(W, L * H)     # W (L·H)
        aggregated = lh[:, head_mask.reshape(-1)].mean(axis=-1)  # W

        # Upsert query node.
        T_ctx = len(ctx_ids)
        query_int = int(ctx_ids[-1])
        query_tok = tokenizer.id_to_token(query_int) or f"<{query_int}>"
        query_id  = (f"{query_tok}#{query_abs_pos}"
                     if self.position_scoped_ids else query_tok)
        own_pull = float(aggregated.mean())
        self.graph.upsert_node(
            id         = query_id,
            content    = query_tok,
            activation = min(1.0, 0.3 + own_pull),
            metadata   = {"absolute_pos": query_abs_pos, "token_id": query_int},
        )
        self._touched_nodes.add(query_id)

        # Emit one edge per w that clears the threshold and has a valid key.
        edges_emitted = 0
        for w in range(W):
            key_offset_back = W - w                # 1..W (key is `back` tokens behind query)
            ctx_key_idx = T_ctx - 1 - key_offset_back
            if ctx_key_idx < 0:
                continue                           # key would be before ctx start
            weight = float(aggregated[w])
            if weight < self.edge_threshold:
                continue

            key_abs   = query_abs_pos - key_offset_back
            key_int   = int(ctx_ids[ctx_key_idx])
            key_tok   = tokenizer.id_to_token(key_int) or f"<{key_int}>"
            key_id    = (f"{key_tok}#{key_abs}"
                         if self.position_scoped_ids else key_tok)

            self.graph.upsert_node(
                id         = key_id,
                content    = key_tok,
                activation = min(1.0, 0.3 + weight),
                metadata   = {"absolute_pos": key_abs, "token_id": key_int},
            )
            self._touched_nodes.add(key_id)
            self.graph.add_edge(
                src      = key_id,
                dst      = query_id,
                weight   = weight,
                relation = "tension",
                metadata = {"distance": key_offset_back},
            )
            edges_emitted += 1

        self._stream_steps += 1
        self._stream_edges += edges_emitted
        return edges_emitted

    @property
    def stream_stats(self) -> StreamStats:
        return StreamStats(
            steps         = self._stream_steps,
            edges_emitted = self._stream_edges,
            nodes_touched = len(self._touched_nodes),
        )
