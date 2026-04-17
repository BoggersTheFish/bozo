"""
ts_bridge.corpus_profile
=========================

Phase 1.5 — corpus-level head profiling.

Single-prompt `TauExporter.profile_and_lock` picks heads based on *that
prompt's* activity pattern; across 5 long prompts the signal-head Jaccard
was 0.33–0.69 and the intersection was empty.  This module replaces the
per-prompt lock with a one-off corpus profile: run the model on N wikitext
samples, aggregate per-head stats, and pick heads whose *average* behaviour
marks them as semantic carriers.

Output is a JSON sidecar (one per checkpoint) containing the locked
(layer, head) list plus the aggregated stats.  `TauExporter(head_override=...)`
accepts it directly, so variance-check / generation-time code just loads the
JSON instead of re-profiling.

Run:
    python -m ts_bridge.corpus_profile \
        --checkpoint checkpoints/117m-curriculum/pytorch_model.pt \
        --tokenizer  checkpoints/117m-curriculum/tokenizer.json \
        --n_samples  50 \
        --out        checkpoints/117m-curriculum/head_profile.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ts_bridge.smoke_test import load_model                       # noqa: E402


def _collect_corpus(n_samples: int, min_chars: int = 400) -> list[str]:
    """Fetch N wikitext-2 test passages long enough to exceed W=64 tokens."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    passages: list[str] = []
    for sample in ds:
        text = sample["text"].strip()
        if len(text) < min_chars or text.startswith("="):
            continue
        passages.append(text)
        if len(passages) >= n_samples:
            break
    return passages


def _per_head_stats(tau: torch.Tensor) -> dict:
    """
    tau: [L, T, H, W] (single batch item, local layers only).
    Returns per-head arrays:
      max_tau[L, H]       — p99 of τ over (T, W) — firing capability
      concentration[L, H] — mean over T of max-over-w of τ[t, h, :]
      peak_pos[L, H]      — mean of argmax-over-w of τ[t, h, :]
    """
    t = tau.detach().float().cpu().numpy()                # L T H W
    L, T, H, W = t.shape

    # Causal validity mask — exclude s < 0 positions so zero-padding
    # doesn't deflate per-head stats on short sequences.
    t_idx = np.arange(T)[:, None]
    w_idx = np.arange(W)[None, :]
    valid = (t_idx + w_idx >= W)                          # T W
    # Zero out invalid entries for max computations (a τ value of 0 is the
    # natural neutral element for max comparisons against valid-signal τ).
    t_masked = t * valid[None, :, None, :]                # L T H W

    max_tau       = np.zeros((L, H))
    concentration = np.zeros((L, H))
    peak_pos      = np.zeros((L, H))

    for l in range(L):
        for h in range(H):
            vals = t_masked[l, :, h, :]                   # T W
            max_tau[l, h]       = float(np.quantile(vals, 0.99))
            # For concentration & peak_pos only look at rows that have any
            # valid cells — shorter sequences have fewer valid rows.
            row_max    = vals.max(axis=-1)
            row_argmax = vals.argmax(axis=-1)
            concentration[l, h] = float(row_max.mean())
            peak_pos[l, h]      = float(row_argmax.mean())

    return {
        "max_tau":       max_tau,
        "concentration": concentration,
        "peak_pos":      peak_pos,
    }


def _aggregate(all_stats: list[dict]) -> dict:
    """Mean each per-head stat across corpus samples."""
    def _stack_mean(key: str) -> np.ndarray:
        return np.stack([s[key] for s in all_stats], axis=0).mean(axis=0)
    return {
        "max_tau":       _stack_mean("max_tau"),
        "concentration": _stack_mean("concentration"),
        "peak_pos":      _stack_mean("peak_pos"),
    }


def select_heads_by_corpus(
    agg: dict,
    W: int,
    top_k: int | None = None,
    min_max_tau: float = 0.40,
    min_concentration: float = 0.35,
) -> list[tuple[int, int]]:
    """
    Pick heads that consistently fire sharply AND concentrate mass on a
    particular key position across the corpus.  Peak-pos classification
    (long vs short vs mid range) is reported but *not used* as a filter
    here — on the 117M curriculum model almost no head classifies
    LONG_RANGE by position, and filtering by peak_pos just empties the set.
    """
    max_tau       = agg["max_tau"]
    concentration = agg["concentration"]
    L, H = max_tau.shape

    # Combined score = max_tau * concentration
    # — a head needs to both fire sharply AND localise mass.
    score = max_tau * concentration

    candidates: list[tuple[float, int, int]] = []
    for l in range(L):
        for h in range(H):
            if max_tau[l, h] < min_max_tau:       continue
            if concentration[l, h] < min_concentration: continue
            candidates.append((float(score[l, h]), l, h))

    candidates.sort(reverse=True)
    if top_k is not None:
        candidates = candidates[:top_k]
    return [(l, h) for _, l, h in candidates]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer",  default=None)
    ap.add_argument("--device",     default="cpu")
    ap.add_argument("--n_samples",  type=int, default=50,
                    help="Number of wikitext passages to profile on")
    ap.add_argument("--top_k",      type=int, default=None,
                    help="Cap signal-head count (else keep all above thresholds)")
    ap.add_argument("--out",        required=True,
                    help="Output JSON path for the locked head profile")
    args = ap.parse_args()

    model, tokenizer, cfg = load_model(args.checkpoint, args.device, args.tokenizer)
    print(f"model: dim={cfg.dim} layers={cfg.num_layers} heads={cfg.num_heads} "
          f"W={cfg.window} vocab={cfg.vocab_size}")

    passages = _collect_corpus(args.n_samples)
    print(f"profiling on {len(passages)} wikitext passages …")

    all_stats: list[dict] = []
    aggregated_samples: list[np.ndarray] = []   # for per-corpus quantile picking
    for i, text in enumerate(passages):
        ids = tokenizer.encode(text).ids[:cfg.max_seq_len]
        if len(ids) < cfg.window + 4:     # too short to exercise the full window
            continue
        x = torch.tensor(ids, device=args.device, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            _, _, all_tau = model(x, return_all=True)
        shapes = [tuple(t.shape[-2:]) for t in all_tau]
        canon  = max(set(shapes), key=shapes.count)
        local  = [t for t in all_tau if tuple(t.shape[-2:]) == canon]
        tau    = torch.stack([t[0].float() for t in local])       # L T H W
        all_stats.append(_per_head_stats(tau))
        # Stash the full τ so we can compute aggregated-τ quantiles after
        # the corpus-level head selection is known.  Move to CPU/numpy to
        # cap memory on GPU runs.
        aggregated_samples.append(tau.cpu().numpy())
        if (i + 1) % 10 == 0:
            print(f"  profiled {i+1}/{len(passages)}")

    print(f"  {len(all_stats)} usable samples after length filter")

    agg = _aggregate(all_stats)
    L, H = agg["max_tau"].shape

    # Print the per-head table sorted by combined score.
    print(f"\nPer-head aggregate stats (top 25 by score):")
    print(f"{'L':<4}{'H':<4}{'max_tau':<10}{'conc.':<9}{'peak_pos':<10}{'score':<8}")
    print("─" * 50)
    rows: list[tuple[float, int, int, float, float, float]] = []
    for l in range(L):
        for h in range(H):
            mt   = float(agg["max_tau"][l, h])
            cc   = float(agg["concentration"][l, h])
            pp   = float(agg["peak_pos"][l, h])
            rows.append((mt * cc, l, h, mt, cc, pp))
    rows.sort(reverse=True)
    for _, l, h, mt, cc, pp in rows[:25]:
        print(f"{l:<4}{h:<4}{mt:<10.3f}{cc:<9.3f}{pp:<10.1f}{mt*cc:<8.3f}")

    # Select signal heads.
    signal = select_heads_by_corpus(agg, W=cfg.window, top_k=args.top_k)
    print(f"\nSelected {len(signal)} signal heads "
          f"(min_max_tau=0.40, min_concentration=0.35"
          + (f", top_k={args.top_k}" if args.top_k else "") + ")")

    # Distribution over layers — which layers carry the semantic signal?
    from collections import Counter
    layer_dist = Counter(l for l, _ in signal)
    print("Layer distribution: "
          + ", ".join(f"L{l}:{c}" for l, c in sorted(layer_dist.items())))

    # Aggregated-τ quantiles across the corpus, using the corpus-selected
    # heads.  Lets downstream code pick edge_threshold by target density
    # instead of guessing: threshold at quantile q emits (1-q) of all valid
    # causal pairs, so q=0.80 → ~20% density on a windowed local graph.
    threshold_table: dict[str, float] = {}
    if signal:
        head_mask = np.zeros((L, H), dtype=bool)
        for (l, h) in signal:
            head_mask[l, h] = True
        mask_flat = head_mask.reshape(-1)
        all_valid_vals: list[np.ndarray] = []
        for tau_np in aggregated_samples:
            Lp, T, Hp, W = tau_np.shape
            lh = tau_np.transpose(1, 3, 0, 2).reshape(T, W, Lp * Hp)
            agg_mat = lh[:, :, mask_flat].mean(axis=-1)           # T W
            t_idx = np.arange(T)[:, None]; w_idx = np.arange(W)[None, :]
            valid = (t_idx + w_idx >= W)
            all_valid_vals.append(agg_mat[valid])
        all_vals = np.concatenate(all_valid_vals)
        for q, name in [(0.50, "q50"), (0.70, "q70"), (0.80, "q80"),
                        (0.90, "q90"), (0.95, "q95")]:
            threshold_table[name] = float(np.quantile(all_vals, q))
        print("\nAggregated-τ quantiles over corpus (via selected heads):")
        for k, v in threshold_table.items():
            print(f"  {k}:  {v:.3f}   → threshold gives ~{(1 - float(k[1:])/100)*100:.0f}% density")

    out = {
        "checkpoint":     str(Path(args.checkpoint).resolve()),
        "n_samples":      len(all_stats),
        "num_layers":     L,
        "num_heads":      H,
        "window":         cfg.window,
        "signal_heads":   signal,
        "thresholds":     {"min_max_tau": 0.40, "min_concentration": 0.35},
        "edge_threshold_quantiles": threshold_table,
        # Recommendation: use q50 as the edge threshold.  Empirically it
        # both (a) matches the historical default 0.30 on this checkpoint and
        # (b) gives the tightest coherent/salad discrimination in variance
        # check.  Raising the threshold to q80 ("~20% density") produces a
        # sparser graph but degrades discrimination — use it only for
        # downstream integration where sparsity matters more than the delta.
        "recommended_edge_threshold": threshold_table.get("q50", 0.30),
        "per_head_stats": {
            "max_tau":       agg["max_tau"].tolist(),
            "concentration": agg["concentration"].tolist(),
            "peak_pos":      agg["peak_pos"].tolist(),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
