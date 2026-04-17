"""
ts_bridge.bias_smoke
=====================

Phase 2 acceptance: the graph-bias path actually enters TensionLM's
attention computation and measurably moves next-token logits.

Three checks
------------
1. **No-op invariant.**  Empty graph ⇒ bias is all zeros ⇒ biased logits
   must equal unbiased logits.  Catches plumbing regressions immediately.

2. **Responsiveness.**  Seed the graph with a strong edge between tokens
   that both appear in the prompt.  The biased forward's τ field at the
   matched (query, key) pair will rise pre-sigmoid, changing downstream
   hidden states and thus the last-position logits.  We expect a
   non-trivial KL divergence between biased and unbiased next-token
   distributions.

3. **Directional.**  Two different seeded edges (A→B vs A→C) should
   produce two different biased distributions.  Confirms the bias
   pathway transmits which edge, not just "something is biased."

This is a mechanism test, not a truth test.  Phase 2 only claims the
graph can enter attention; whether a *truthful* graph produces *better*
completions depends on graph quality, which is the Phase 3 integration
concern.

Run:
    python -m ts_bridge.bias_smoke \
        --checkpoint checkpoints/diagnostic/latest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ts_bridge import GraphBias, UniversalLivingGraph              # noqa: E402
from ts_bridge.smoke_test import load_model, token_ids_for         # noqa: E402


PROMPT = "If all mammals are warm-blooded and all whales are mammals then"


def _kl(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(P || Q) over softmax(p) vs softmax(q) at one position."""
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    return float((p.exp() * (p - q)).sum().item())


def _seed_graph(pairs: list[tuple[str, str, float]]) -> UniversalLivingGraph:
    """Build a graph with position-scoped ids so GraphBias' content-collapse
    matches the real-pipeline code path."""
    g = UniversalLivingGraph()
    for i, (src, dst, w) in enumerate(pairs):
        g.upsert_node(id=f"{src}#__seed_s_{i}", content=src)
        g.upsert_node(id=f"{dst}#__seed_d_{i}", content=dst)
        g.add_edge(
            src=f"{src}#__seed_s_{i}",
            dst=f"{dst}#__seed_d_{i}",
            weight=w,
            relation="tension",
        )
    return g


@torch.no_grad()
def run_forward(model, ctx_tensor: torch.Tensor,
                bias: torch.Tensor | None) -> torch.Tensor:
    """Return last-position logits [vocab]."""
    out = model(ctx_tensor, tau_bias=bias)
    return out[0, -1].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer",  default=None)
    ap.add_argument("--device",     default="cpu")
    ap.add_argument("--alpha",      type=float, default=2.0,
                    help="Bias scale α applied to edge weights")
    ap.add_argument("--edge_weight", type=float, default=1.0,
                    help="Seeded edge weight (after α, bias magnitude ≈ α·this)")
    ap.add_argument("--top_k",      type=int, default=5)
    args = ap.parse_args()

    model, tokenizer, cfg = load_model(
        args.checkpoint, args.device, args.tokenizer,
    )
    ids = token_ids_for(PROMPT, tokenizer, cfg.max_seq_len)
    T   = len(ids)
    W   = cfg.window
    print(f"model: dim={cfg.dim} layers={cfg.num_layers} heads={cfg.num_heads} "
          f"W={W} T={T}")

    # Pick two prompt tokens as anchor contents for the seeded edges.
    # Use the first content-decodable tokens, skipping any that don't map back
    # to a clean string.  Default picks: mid-prompt token as src, last as dst
    # for edge A, and a different src for edge B.
    contents = [tokenizer.id_to_token(int(t)) or f"<{int(t)}>" for t in ids]
    # Dedup while preserving order.
    seen, uniq = set(), []
    for c in contents:
        if c not in seen:
            seen.add(c); uniq.append(c)
    if len(uniq) < 3:
        sys.exit("Need ≥3 distinct tokens in the prompt for the directional test.")
    # src_A / src_B are two different earlier tokens; dst is the final token
    # (the one whose query row drives the next-token logits).
    dst_tok   = contents[-1]
    src_A     = uniq[1]
    src_B     = uniq[2] if uniq[2] != src_A else uniq[3]
    print(f"probe edges:  '{src_A}' → '{dst_tok}'   vs   '{src_B}' → '{dst_tok}'")

    ctx = torch.tensor(ids, dtype=torch.long, device=args.device).unsqueeze(0)

    # ── 1. No-op invariant ──────────────────────────────────────────────────
    logits_unbiased = run_forward(model, ctx, None)
    bias_zero = torch.zeros(1, T, W, device=args.device)
    logits_zero = run_forward(model, ctx, bias_zero)
    noop_gap = (logits_unbiased - logits_zero).abs().max().item()
    print(f"\n[1] no-op invariant: max|Δlogit| = {noop_gap:.2e}   "
          f"{'✓' if noop_gap < 1e-5 else '✗'}")

    # ── 2. Responsiveness ───────────────────────────────────────────────────
    graph_A = _seed_graph([(src_A, dst_tok, args.edge_weight)])
    bias_A, stats_A = GraphBias.from_graph(graph_A, alpha=args.alpha).local_bias(
        ids, tokenizer, window=W, device=args.device,
    )
    logits_A = run_forward(model, ctx, bias_A)
    kl_A = _kl(logits_A, logits_unbiased)
    print(f"\n[2] responsiveness:  edge '{src_A}'→'{dst_tok}' "
          f"w={args.edge_weight} α={args.alpha}")
    print(f"    bias hits={stats_A.nonzero_pairs}  "
          f"max={stats_A.max_weight:.2f}  mean={stats_A.mean_weight:.2f}")
    print(f"    KL(biased||unbiased) at last pos = {kl_A:.4e}   "
          f"{'✓' if kl_A > 1e-6 else '✗'}")

    # ── 3. Directional ──────────────────────────────────────────────────────
    graph_B = _seed_graph([(src_B, dst_tok, args.edge_weight)])
    bias_B, stats_B = GraphBias.from_graph(graph_B, alpha=args.alpha).local_bias(
        ids, tokenizer, window=W, device=args.device,
    )
    logits_B = run_forward(model, ctx, bias_B)
    kl_AB = _kl(logits_A, logits_B)
    print(f"\n[3] directional:     edge '{src_B}'→'{dst_tok}' "
          f"vs edge A")
    print(f"    bias hits={stats_B.nonzero_pairs}  max={stats_B.max_weight:.2f}")
    print(f"    KL(A||B) at last pos = {kl_AB:.4e}   "
          f"{'✓' if kl_AB > 1e-6 else '✗'}")

    # ── Top-k diff under edge A ─────────────────────────────────────────────
    def _top(logits):
        probs = F.softmax(logits, dim=-1)
        p, i  = torch.topk(probs, args.top_k)
        return [
            (tokenizer.id_to_token(int(idx)) or f"<{int(idx)}>", float(pi))
            for pi, idx in zip(p.tolist(), i.tolist())
        ]
    print("\n── top-{} next-token probs ──".format(args.top_k))
    print("  unbiased : ", ", ".join(f"{t} ({p:.3f})" for t, p in _top(logits_unbiased)))
    print("  bias A   : ", ", ".join(f"{t} ({p:.3f})" for t, p in _top(logits_A)))
    print("  bias B   : ", ", ".join(f"{t} ({p:.3f})" for t, p in _top(logits_B)))

    ok = (noop_gap < 1e-5) and (kl_A > 1e-6) and (kl_AB > 1e-6)
    print(f"\n{'✓ Phase 2 bias mechanism OK' if ok else '✗ Phase 2 bias FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
