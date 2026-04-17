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
- `export_mode` controls which τ field becomes graph edges.
    "biased"   — export the post-bias τ (positive feedback loop).
    "unbiased" — second forward per step with tau_bias=None for export
                 only; sampling still uses the biased forward.  Doubles
                 forward cost per step but decouples graph growth from
                 bias strength so α's effect on text can be measured
                 without feedback-loop confounds.
    "off"      — disable export entirely (pure bias A/B, no graph grows).
- Global-attention layers are silently unbiased (their W-dim differs).
  TensionLM.forward already handles that; see bias.py for the split.
- Prompt prime is always an unbiased forward — the graph is empty or
  seed-only at that point, and we want a clean starting graph state.

Run:
    python -m ts_bridge.biased_generate \
        --checkpoint checkpoints/diagnostic/latest.pt \
        --prompt "whales are" --alpha 1.0 --max_new 30 --ab

    # Break the positive feedback loop:
    python -m ts_bridge.biased_generate --checkpoint ... \
        --alpha 1.0 --export_mode unbiased
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


EXPORT_MODES = ("biased", "unbiased", "off")


@torch.no_grad()
def closed_loop_generate(
    model,
    tokenizer,
    prompt_ids:  list[int],
    *,
    max_new:     int   = 50,
    alpha:       float = 0.5,
    seed_graph:  UniversalLivingGraph | None = None,
    export_mode: str   = "biased",
    temp:        float = 0.8,
    top_p:       float = 0.92,
    rep_penalty: float = 1.3,
    device:      str   = "cpu",
    head_override: list[tuple[int, int]] | None = None,
) -> tuple[list[int], UniversalLivingGraph]:
    """
    Closed-loop generation: graph biases forward, forward updates graph.

    `export_mode`:
      - "biased":   graph ingests the post-bias τ (positive feedback).
      - "unbiased": second forward per step with tau_bias=None for export;
                    sampling still uses the biased forward.  Doubles the
                    per-step forward cost but decouples graph growth from
                    bias strength.
      - "off":      no export; pure bias experiment.
    """
    if export_mode not in EXPORT_MODES:
        raise ValueError(f"export_mode must be one of {EXPORT_MODES}, "
                         f"got {export_mode!r}")

    cfg     = model.cfg
    W       = cfg.window
    max_ctx = cfg.max_seq_len
    export  = export_mode != "off"

    graph   = seed_graph or UniversalLivingGraph()
    streamer = StreamingTauExporter(
        graph, edge_threshold=0.3, head_override=head_override,
    )

    # Prime on the prompt so head-lock is settled and prompt edges exist.
    # Always unbiased: graph is empty (or seed-only) at prime time, and we
    # want a consistent prompt-graph regardless of export_mode.
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long,
                                 device=device).unsqueeze(0)
    if export:
        _, _, all_tau_p = model(prompt_tensor, return_all=True)
        streamer.prime(prompt_ids, all_tau_p, tokenizer)

    ids = list(prompt_ids)
    exported_upto = len(prompt_ids) - 1

    for _ in range(max_new):
        ctx = ids[-max_ctx:]
        ctx_tensor = torch.tensor(ctx, dtype=torch.long,
                                  device=device).unsqueeze(0)

        # Build bias from the current graph (cheap rebuild per step).
        bias_engine = GraphBias.from_graph(graph, alpha=alpha)
        bias_local, _ = bias_engine.local_bias(
            ctx, tokenizer, window=W, device=device,
        )
        # Only build the [B, T, T] global bias when the model actually has
        # global layers — it's O(T²) work that would otherwise be wasted.
        bias_global = None
        if cfg.global_every > 0:
            bias_global, _ = bias_engine.global_bias(
                ctx, tokenizer, device=device,
            )

        logits, _, all_tau_biased = model(
            ctx_tensor, return_all=True,
            tau_bias=bias_local, tau_bias_global=bias_global,
        )

        # Export edges ending at current last position (if new).
        curr_last = len(ids) - 1
        if export and curr_last > exported_upto:
            if export_mode == "biased":
                all_tau_export = all_tau_biased
            else:  # "unbiased" — second forward without the graph bias
                _, _, all_tau_export = model(ctx_tensor, return_all=True)
            streamer.ingest_step(
                ctx, all_tau_export, tokenizer, query_abs_pos=curr_last,
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
    ap.add_argument("--export_mode", choices=EXPORT_MODES, default="biased",
                    help="What τ the graph ingests: biased post-bias field "
                         "(feedback loop), unbiased second forward (no loop), "
                         "or off (no export).")
    ap.add_argument("--ab", action="store_true",
                    help="Also run an α=0 no-bias baseline and print the "
                         "diverge-at-token diagnostic.")
    ap.add_argument("--loop_check", action="store_true",
                    help="Run biased and unbiased export modes at matched α "
                         "and print graph-growth difference. Diagnostic for "
                         "the positive feedback loop.")
    args = ap.parse_args()

    model, tokenizer, cfg = load_model(
        args.checkpoint, args.device, args.tokenizer,
    )
    prompt_ids = tokenizer.encode(args.prompt).ids[: cfg.max_seq_len]
    print(f"model: dim={cfg.dim} L={cfg.num_layers} H={cfg.num_heads} "
          f"W={cfg.window}  prompt_tokens={len(prompt_ids)}  "
          f"α={args.alpha}  export_mode={args.export_mode}")

    # Primary closed-loop run.
    _set_seed(args.seed)
    ids_bias, g_bias = closed_loop_generate(
        model, tokenizer, prompt_ids,
        max_new=args.max_new, alpha=args.alpha,
        export_mode=args.export_mode,
        temp=args.temp, top_p=args.top_p, rep_penalty=args.rep_penalty,
        device=args.device,
    )
    print(f"\n── closed-loop (α={args.alpha}, export={args.export_mode}) ──")
    print(tokenizer.decode(ids_bias))
    print(f"final graph: {len(g_bias.nodes)} nodes / {len(g_bias.edges)} edges, "
          f"mean w={g_bias.mean_edge_weight():.3f}")

    if args.ab:
        _set_seed(args.seed)
        ids_un, _ = closed_loop_generate(
            model, tokenizer, prompt_ids,
            max_new=args.max_new, alpha=0.0, export_mode="off",
            temp=args.temp, top_p=args.top_p, rep_penalty=args.rep_penalty,
            device=args.device,
        )
        print("\n── no-bias baseline (α=0, export off) ──")
        print(tokenizer.decode(ids_un))

        tail_bias = ids_bias[len(prompt_ids):]
        tail_un   = ids_un[len(prompt_ids):]
        first_diff = next(
            (i for i, (a, b) in enumerate(zip(tail_bias, tail_un)) if a != b),
            min(len(tail_bias), len(tail_un)),
        )
        print(f"\ndiverge-at (first differing generated token index): {first_diff}"
              f" / {min(len(tail_bias), len(tail_un))}")

    if args.loop_check:
        # Re-run at matched α with export_mode=unbiased so graph growth is
        # driven by the model's own τ rather than by the bias we just added.
        _set_seed(args.seed)
        ids_ub, g_ub = closed_loop_generate(
            model, tokenizer, prompt_ids,
            max_new=args.max_new, alpha=args.alpha,
            export_mode="unbiased",
            temp=args.temp, top_p=args.top_p, rep_penalty=args.rep_penalty,
            device=args.device,
        )
        print(f"\n── loop-check: unbiased-export rerun (α={args.alpha}) ──")
        print(tokenizer.decode(ids_ub))
        print(f"graph: {len(g_ub.nodes)} nodes / {len(g_ub.edges)} edges, "
              f"mean w={g_ub.mean_edge_weight():.3f}")
        # Compare to the primary run's graph (only meaningful if primary was
        # biased-export; otherwise the two runs have the same export mode).
        if args.export_mode == "biased":
            de = len(g_bias.edges) - len(g_ub.edges)
            dm = g_bias.mean_edge_weight() - g_ub.mean_edge_weight()
            print(f"feedback-loop delta: Δedges={de:+d}  "
                  f"Δmean_w={dm:+.3f}")


if __name__ == "__main__":
    main()
