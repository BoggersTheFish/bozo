"""
ts_bridge.streaming_parity
===========================

Phase 1.5 acceptance: StreamingTauExporter's per-step emission must produce
the same edges as the batch TauExporter does on the same sequence, for every
query position the streamer observed as "last query."

Protocol
--------
1. Pick T tokens (coherent prompt + continuation, length ≤ cfg.max_seq_len).
2. Batch: one forward pass over all T, run `TauExporter.ingest`.
3. Stream: `prime` on the prompt prefix (positions 0..P-1), then for each
   p in P..T-1 run a fresh forward over tokens[0..p] and call
   `ingest_step(ctx_ids=tokens[0..p], query_abs_pos=p)`.
4. Compare the two resulting graphs on query positions 0..T-1 — every
   position has been "observed as last query" either via prime's batch
   ingest or via an ingest_step call, so no edges should be missing.

Parity relies on causal masking: τ[p, ·, ·] from forward(T) equals
τ[p, ·, ·] from forward(p+1), bit-identical, because token p's attention
distribution only depends on tokens 0..p.  The streaming path also uses the
exporter's `_locked_heads` (seeded by `prime`), so the head-aggregation
mask is identical between batch and stream.

Run:
    python -m ts_bridge.streaming_parity \
        --checkpoint checkpoints/117m-curriculum/pytorch_model.pt \
        --tokenizer  checkpoints/117m-curriculum/tokenizer.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ts_bridge import (                                            # noqa: E402
    StreamingTauExporter, TauExporter, UniversalLivingGraph,
)
from ts_bridge.smoke_test import load_model, token_ids_for         # noqa: E402


PROMPT = (
    "If all mammals are warm-blooded and all whales are mammals "
    "then whales are warm-blooded. Dolphins are mammals, so dolphins"
)


@torch.no_grad()
def batch_graph(model, tokenizer, ids, device, head_override, edge_threshold):
    g = UniversalLivingGraph()
    exp = TauExporter(g, edge_threshold=edge_threshold, head_override=head_override)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    _, _, all_tau = model(x, return_all=True)
    if head_override is None:
        exp.profile_and_lock(all_tau)
    stats = exp.ingest(ids, all_tau, tokenizer)
    return g, stats, exp._locked_heads


@torch.no_grad()
def stream_graph(model, tokenizer, ids, device, prime_len, head_override,
                 edge_threshold):
    g = UniversalLivingGraph()
    streamer = StreamingTauExporter(
        g, edge_threshold=edge_threshold, head_override=head_override,
    )
    # Prime on prompt prefix.
    prompt_ids = ids[:prime_len]
    x_p = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    _, _, all_tau_p = model(x_p, return_all=True)
    streamer.prime(prompt_ids, all_tau_p, tokenizer)

    # Per-step ingest for each subsequent position.
    for p in range(prime_len, len(ids)):
        ctx_ids = ids[: p + 1]
        x_s = torch.tensor(ctx_ids, dtype=torch.long, device=device).unsqueeze(0)
        _, _, all_tau_s = model(x_s, return_all=True)
        streamer.ingest_step(
            ctx_ids, all_tau_s, tokenizer, query_abs_pos=p,
        )
    return g, streamer.stream_stats, streamer._locked_heads


def _edge_key(e):
    return (e.src, e.dst, e.relation)


def compare(g_batch: UniversalLivingGraph,
            g_stream: UniversalLivingGraph) -> dict:
    batch_edges  = {_edge_key(e): e for e in g_batch.edges}
    stream_edges = {_edge_key(e): e for e in g_stream.edges}

    only_batch  = set(batch_edges)  - set(stream_edges)
    only_stream = set(stream_edges) - set(batch_edges)
    common      = set(batch_edges)  & set(stream_edges)

    weight_gap = 0.0
    for k in common:
        weight_gap = max(weight_gap,
                         abs(batch_edges[k].weight - stream_edges[k].weight))

    return {
        "batch_edges":   len(batch_edges),
        "stream_edges":  len(stream_edges),
        "common":        len(common),
        "only_batch":    len(only_batch),
        "only_stream":   len(only_stream),
        "max_weight_gap": weight_gap,
        "only_batch_sample":  sorted(only_batch)[:5],
        "only_stream_sample": sorted(only_stream)[:5],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer",  default=None)
    ap.add_argument("--device",     default="cpu")
    ap.add_argument("--prime_len",  type=int, default=12,
                    help="First N tokens are batch-ingested via prime; "
                         "positions prime_len..T-1 are streamed one by one")
    ap.add_argument("--edge_threshold", type=float, default=0.3)
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="Max allowed absolute weight gap on common edges")
    args = ap.parse_args()

    model, tokenizer, cfg = load_model(
        args.checkpoint, args.device, args.tokenizer,
    )
    ids = token_ids_for(PROMPT, tokenizer, cfg.max_seq_len)
    T = len(ids)
    assert args.prime_len < T, "prime_len must leave at least one streamed token"
    print(f"model: dim={cfg.dim} layers={cfg.num_layers} heads={cfg.num_heads} "
          f"W={cfg.window} tokens={T} prime={args.prime_len}")

    # Batch first — this also profiles heads on the full sequence.  We reuse
    # that head lock for streaming to remove head-selection drift from the
    # diff (corpus-lock is the real deployment pattern; here we just want the
    # two paths to make the same aggregation choice).
    g_batch, stats_batch, locked = batch_graph(
        model, tokenizer, ids, args.device,
        head_override=None, edge_threshold=args.edge_threshold,
    )
    g_stream, stats_stream, _ = stream_graph(
        model, tokenizer, ids, args.device,
        prime_len=args.prime_len,
        head_override=locked,
        edge_threshold=args.edge_threshold,
    )

    print(f"\nbatch:  edges={stats_batch.edges_emitted} "
          f"nodes={stats_batch.nodes_touched} "
          f"heads={stats_batch.signal_heads}/{stats_batch.total_heads}")
    print(f"stream: steps={stats_stream.steps} "
          f"edges={stats_stream.edges_emitted} "
          f"nodes_touched={stats_stream.nodes_touched}")

    report = compare(g_batch, g_stream)
    print("\n── parity ──")
    for k, v in report.items():
        print(f"  {k:>20s}  {v}")

    ok = (
        report["only_batch"]  == 0 and
        report["only_stream"] == 0 and
        report["max_weight_gap"] <= args.tol
    )
    print(f"\n{'✓ parity OK' if ok else '✗ parity FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
