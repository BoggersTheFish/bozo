"""
ts_bridge.alpha_sweep
======================

Phase 2.2 — α calibration.

Runs `closed_loop_generate` across a configurable set of α values and
reports the three signals that matter for picking a calibrated α:

  1. **diverge-at** — first generated-token index where the biased run
     differs from the α=0 baseline under a matched seed.  Monotone
     relationship with α is the "graph is steering the surface" signal.

  2. **graph growth** — nodes / edges / mean edge weight at end of the
     biased-export run.  Tracks how much structure the closed loop
     accumulated by the end of generation.

  3. **feedback-loop Δ** — same α, but export_mode="unbiased" (graph
     ingests a second forward with tau_bias=None).  The delta between
     biased-export and unbiased-export graphs is the positive-feedback
     footprint of α.  Small Δ means the bias is mostly "steering"; large
     Δ means α is mostly amplifying itself.

The harness does not pick α for you; it prints the numbers so you can.
A reasonable heuristic: the largest α where diverge-at is in [1, max_new/2]
**and** feedback-loop Δedges / Δmean-w are within an order of magnitude of
the α=0.25 reference.

Run:
    python -m ts_bridge.alpha_sweep \
        --checkpoint checkpoints/117m-curriculum/pytorch_model.pt \
        --tokenizer  checkpoints/117m-curriculum/tokenizer.json \
        --device cuda --force_triton \
        --alphas 0.25 0.5 1.0 2.0 \
        --max_new 30 --seed 42 \
        --json_out logs/alpha_sweep_117m.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ts_bridge.biased_generate import closed_loop_generate          # noqa: E402
from ts_bridge.smoke_test    import load_model                      # noqa: E402


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _diverge_at(tail_a: list[int], tail_b: list[int]) -> int:
    """First index where two generated tails differ, or the shorter length
    if one is a prefix of the other."""
    for i, (a, b) in enumerate(zip(tail_a, tail_b)):
        if a != b:
            return i
    return min(len(tail_a), len(tail_b))


def _graph_stats(g) -> dict:
    return {
        "nodes": len(g.nodes),
        "edges": len(g.edges),
        "mean_w": float(g.mean_edge_weight()),
    }


def _force_triton_on(model) -> None:
    """load_model forces use_triton=False for CPU safety; flip it back on
    at the layer level for a CUDA sweep."""
    for block in model.blocks:
        if not block.tension.global_layer:
            block.tension.use_triton = True
    model.cfg.use_triton = True


def sweep(
    model,
    tokenizer,
    cfg,
    prompt_ids: list[int],
    alphas:     list[float],
    *,
    max_new:    int   = 30,
    seed:       int   = 42,
    temp:       float = 0.8,
    top_p:      float = 0.92,
    rep_penalty: float = 1.3,
    device:     str   = "cpu",
) -> dict:
    """Return a dict of {alpha: {...}} plus the α=0 baseline entry."""
    results: dict = {"baseline": {}, "by_alpha": {}}

    # α=0 baseline.  export_mode="off" keeps the graph out of the loop.
    _set_seed(seed)
    t0 = time.time()
    ids_base, _ = closed_loop_generate(
        model, tokenizer, prompt_ids,
        max_new=max_new, alpha=0.0, export_mode="off",
        temp=temp, top_p=top_p, rep_penalty=rep_penalty, device=device,
    )
    tail_base = ids_base[len(prompt_ids):]
    results["baseline"] = {
        "alpha": 0.0,
        "export_mode": "off",
        "tail_ids": tail_base,
        "text": tokenizer.decode(ids_base),
        "elapsed_s": round(time.time() - t0, 2),
    }
    print(f"baseline (α=0, export=off): "
          f"{results['baseline']['elapsed_s']}s  "
          f"{len(tail_base)} new tokens")

    for alpha in alphas:
        entry: dict = {"alpha": alpha}

        # Biased run with biased-export (positive-feedback loop).
        _set_seed(seed)
        t0 = time.time()
        ids_b, g_b = closed_loop_generate(
            model, tokenizer, prompt_ids,
            max_new=max_new, alpha=alpha, export_mode="biased",
            temp=temp, top_p=top_p, rep_penalty=rep_penalty, device=device,
        )
        entry["biased"] = {
            "tail_ids":  ids_b[len(prompt_ids):],
            "text":      tokenizer.decode(ids_b),
            "graph":     _graph_stats(g_b),
            "diverge_at": _diverge_at(ids_b[len(prompt_ids):], tail_base),
            "elapsed_s": round(time.time() - t0, 2),
        }

        # Same α, export_mode="unbiased" — feedback-loop control.
        _set_seed(seed)
        t0 = time.time()
        ids_u, g_u = closed_loop_generate(
            model, tokenizer, prompt_ids,
            max_new=max_new, alpha=alpha, export_mode="unbiased",
            temp=temp, top_p=top_p, rep_penalty=rep_penalty, device=device,
        )
        entry["unbiased_export"] = {
            "tail_ids":  ids_u[len(prompt_ids):],
            "text":      tokenizer.decode(ids_u),
            "graph":     _graph_stats(g_u),
            "diverge_at": _diverge_at(ids_u[len(prompt_ids):], tail_base),
            "elapsed_s": round(time.time() - t0, 2),
        }

        # Feedback-loop delta: how much of the graph growth is the loop
        # amplifying itself, vs. how much comes from the model's own τ.
        entry["loop_delta"] = {
            "d_edges":  entry["biased"]["graph"]["edges"]
                        - entry["unbiased_export"]["graph"]["edges"],
            "d_mean_w": entry["biased"]["graph"]["mean_w"]
                        - entry["unbiased_export"]["graph"]["mean_w"],
        }

        results["by_alpha"][f"{alpha:g}"] = entry

        b, u, d = entry["biased"], entry["unbiased_export"], entry["loop_delta"]
        print(f"α={alpha:>5g}  "
              f"diverge@{b['diverge_at']:>2d}/{max_new}  "
              f"biased: {b['graph']['edges']:>4d}E  mw={b['graph']['mean_w']:.3f}  "
              f"│ unbiased: {u['graph']['edges']:>4d}E  mw={u['graph']['mean_w']:.3f}  "
              f"│ Δ={d['d_edges']:+d}E  Δmw={d['d_mean_w']:+.3f}  "
              f"({b['elapsed_s'] + u['elapsed_s']:.1f}s)")

    return results


def _print_summary(results: dict, prompt: str, max_new: int) -> None:
    print("\n" + "=" * 72)
    print("α-SWEEP SUMMARY")
    print("=" * 72)
    print(f"prompt   : {prompt!r}")
    print(f"max_new  : {max_new}")
    print(f"\nbaseline (α=0, no export):\n  {results['baseline']['text']}")
    for a, e in results["by_alpha"].items():
        b, u, d = e["biased"], e["unbiased_export"], e["loop_delta"]
        print(f"\n── α={a} ──")
        print(f"  biased   text : {b['text']}")
        print(f"  unbiased text : {u['text']}")
        print(f"  diverge@ biased={b['diverge_at']}/{max_new}  "
              f"unbiased={u['diverge_at']}/{max_new}")
        print(f"  graph    biased={b['graph']}  unbiased={u['graph']}")
        print(f"  loop Δ   edges={d['d_edges']:+d}  mean_w={d['d_mean_w']:+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer",  default=None)
    ap.add_argument("--device",     default="cpu")
    ap.add_argument("--prompt",
                    default="If all mammals are warm-blooded and all whales "
                            "are mammals then")
    ap.add_argument("--max_new",    type=int, default=30)
    ap.add_argument("--alphas",     type=float, nargs="+",
                    default=[0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--temp",       type=float, default=0.8)
    ap.add_argument("--top_p",      type=float, default=0.92)
    ap.add_argument("--rep_penalty", type=float, default=1.3)
    ap.add_argument("--force_triton", action="store_true",
                    help="Enable fused kernel on local layers (CUDA only).")
    ap.add_argument("--json_out",   default=None,
                    help="Optional path to write full results JSON.")
    args = ap.parse_args()

    model, tokenizer, cfg = load_model(
        args.checkpoint, args.device, args.tokenizer,
    )
    if args.force_triton and args.device != "cpu":
        _force_triton_on(model)
        print("force_triton: fused kernel enabled on local layers")

    prompt_ids = tokenizer.encode(args.prompt).ids[: cfg.max_seq_len]
    print(f"model: dim={cfg.dim} L={cfg.num_layers} H={cfg.num_heads} "
          f"W={cfg.window} global_every={cfg.global_every}")
    print(f"prompt_tokens={len(prompt_ids)}  max_new={args.max_new}  "
          f"alphas={args.alphas}  seed={args.seed}")

    results = sweep(
        model, tokenizer, cfg, prompt_ids, args.alphas,
        max_new=args.max_new, seed=args.seed,
        temp=args.temp, top_p=args.top_p, rep_penalty=args.rep_penalty,
        device=args.device,
    )

    meta = {
        "checkpoint": args.checkpoint,
        "prompt": args.prompt,
        "max_new": args.max_new,
        "alphas": args.alphas,
        "seed": args.seed,
        "model_cfg": {
            "dim": cfg.dim, "num_layers": cfg.num_layers,
            "num_heads": cfg.num_heads, "window": cfg.window,
            "global_every": cfg.global_every,
            "use_triton": bool(getattr(cfg, "use_triton", False)),
        },
    }
    results["meta"] = meta

    _print_summary(results, args.prompt, args.max_new)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
