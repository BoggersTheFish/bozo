"""
ts_bridge.biased_generate
==========================

Phase 2.1 — closed-loop generation.

Every step:
  1. Build a [B, T, W] τ-bias from the current graph.
  2. Forward with `tau_bias`, return logits + all_tensions.
  3. `StreamingTauExporter.ingest_step` writes τ-edges for the last query
     position back into the same graph.
  4. Sample next token, append, loop.

The exporter's prompt prime happens once up front; after that the graph
and the surface are in a bidirectional loop — the graph biases the next
forward, the next forward updates the graph.

Sampling mirrors `model.generate` (nucleus + rep penalty) so biased vs
unbiased runs are directly comparable.

Trade-offs kept deliberate:
- Exported τ is the **post-bias** field, so graph-reinforcement is a
  positive feedback loop.  For clean Phase 2 acceptance you want
  `seed_weight` reasonable and `alpha` modest.  An `export=False` flag
  disables writeback entirely when you want a pure bias A/B.
- Global-attention layers are silently unbiased (their W-dim differs).
  TensionLM.forward already handles that; see bias.py for the split.

Run:
    python -m ts_bridge.biased_generate \
        --checkpoint checkpoints/diagnostic/latest.pt \
        --prompt "whales are" --alpha 1.0 --max_new 30 --ab
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ts_bridge import (                                             # noqa: E402
    GraphBias, StreamingTauExporter, UniversalLivingGraph,
)
from ts_bridge.smoke_test import load_model                         # noqa: E402


def _sample_next(
    logits: torch.Tensor,
    recent_ids: Sequence[int],
    temp: float,
    top_p: float,
    rep_penalty: float,
) -> int:
    """Mirror of model.generate's sampler so comparisons are apples-to-apples."""
    logits = logits.float().clone()
    for tok in set(recent_ids[-32:]):
        if logits[tok] > 0:
            logits[tok] /= rep_penalty
        else:
            logits[tok] *= rep_penalty
    logits = logits / max(temp, 1e-5)
    probs  = F.softmax(logits, dim=-1)
    sp, si = torch.sort(probs, descending=True)
    cum    = torch.cumsum(sp, dim=-1)
    mask   = (cum - sp) < top_p
    sp[~mask] = 0.0
    sp = sp / sp.sum()
    return int(si[torch.multinomial(sp, 1).item()].item())


@torch.no_grad()
def closed_loop_generate(
    model,
    tokenizer,
    prompt_ids:  list[int],
    *,
    max_new:     int   = 50,
    alpha:       float = 0.5,
    seed_graph:  UniversalLivingGraph | None = None,
    export:      bool  = True,
    temp:        float = 0.8,
    top_p:       float = 0.92,
    rep_penalty: float = 1.3,
    device:      str   = "cpu",
    head_override: list[tuple[int, int]] | None = None,
) -> tuple[list[int], UniversalLivingGraph]:
    """Closed-loop generation: graph biases forward, forward updates graph."""
    cfg     = model.cfg
    W       = cfg.window
    max_ctx = cfg.max_seq_len

    graph   = seed_graph or UniversalLivingGraph()
    streamer = StreamingTauExporter(
        graph, edge_threshold=0.3, head_override=head_override,
    )

    # Prime on the prompt so head-lock is settled and prompt edges exist.
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long,
                                 device=device).unsqueeze(0)
    _, _, all_tau_p = model(prompt_tensor, return_all=True)
    if export:
        streamer.prime(prompt_ids, all_tau_p, tokenizer)

    ids = list(prompt_ids)
    exported_upto = len(prompt_ids) - 1

    for _ in range(max_new):
        ctx = ids[-max_ctx:]
        ctx_tensor = torch.tensor(ctx, dtype=torch.long,
                                  device=device).unsqueeze(0)

        # Build bias from the current graph (cheap rebuild per step).
        bias_engine = GraphBias.from_graph(graph, alpha=alpha)
        bias, _ = bias_engine.local_bias(
            ctx, tokenizer, window=W, device=device,
        )

        logits, _, all_tau = model(
            ctx_tensor, return_all=True, tau_bias=bias,
        )

        # Export edges ending at current last position (if new).
        curr_last = len(ids) - 1
        if export and curr_last > exported_upto:
            streamer.ingest_step(
                ctx, all_tau, tokenizer, query_abs_pos=curr_last,
            )
            exported_upto = curr_last

        next_id = _sample_next(
            logits[0, -1], ids, temp, top_p, rep_penalty,
        )
        ids.append(next_id)

    return ids, graph


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer",  default=None)
    ap.add_argument("--device",     default="cpu")
    ap.add_argument("--prompt",     default="If all mammals are warm-blooded and all whales are mammals then")
    ap.add_argument("--max_new",    type=int, default=30)
    ap.add_argument("--alpha",      type=float, default=0.5)
    ap.add_argument("--temp",       type=float, default=0.8)
    ap.add_argument("--top_p",      type=float, default=0.92)
    ap.add_argument("--rep_penalty", type=float, default=1.3)
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--ab", action="store_true",
                    help="Also run an unbiased comparison and print both.")
    args = ap.parse_args()

    model, tokenizer, cfg = load_model(
        args.checkpoint, args.device, args.tokenizer,
    )
    prompt_ids = tokenizer.encode(args.prompt).ids[: cfg.max_seq_len]
    print(f"model: dim={cfg.dim} L={cfg.num_layers} H={cfg.num_heads} "
          f"W={cfg.window}  prompt_tokens={len(prompt_ids)}  α={args.alpha}")

    # Closed-loop biased run.
    _set_seed(args.seed)
    ids_bias, g_bias = closed_loop_generate(
        model, tokenizer, prompt_ids,
        max_new=args.max_new, alpha=args.alpha,
        temp=args.temp, top_p=args.top_p, rep_penalty=args.rep_penalty,
        device=args.device,
    )
    print(f"\n── biased (closed-loop, α={args.alpha}) ──")
    print(tokenizer.decode(ids_bias))
    print(f"final graph: {len(g_bias.nodes)} nodes / {len(g_bias.edges)} edges, "
          f"mean w={g_bias.mean_edge_weight():.3f}")

    if args.ab:
        _set_seed(args.seed)
        ids_un, _ = closed_loop_generate(
            model, tokenizer, prompt_ids,
            max_new=args.max_new, alpha=0.0, export=False,
            temp=args.temp, top_p=args.top_p, rep_penalty=args.rep_penalty,
            device=args.device,
        )
        print("\n── unbiased (α=0, export off) ──")
        print(tokenizer.decode(ids_un))

        # Simple overlap diagnostic: do the generated tails diverge?
        tail_bias = ids_bias[len(prompt_ids):]
        tail_un   = ids_un[len(prompt_ids):]
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(tail_bias, tail_un)) if a != b),
            min(len(tail_bias), len(tail_un)),
        )
        print(f"\ndiverge-at (first differing generated token index): {first_diff}"
              f" / {min(len(tail_bias), len(tail_un))}")


if __name__ == "__main__":
    main()
