"""
ts_bridge.exp2_replicate
=========================

Does the Exp 2 finding ("+25% mean τ / +60% more active edges for coherent
vs salad") hold on the 117M-Curriculum checkpoint?

The finding was published in the README against a 13.5M TS-native model on
open-web-math.  Phase 1's variance check showed the *exported graph* signal
is weak on the 117M curriculum model; B asks whether the underlying *internal*
τ-field signal replicates, or whether the claim is regime-specific.

Measured on the raw τ tensor (same input as the exporter, before any head
filtering, thresholding, or aggregation):
    - mean τ overall
    - fraction of τ values above 0.3 (the canonical "active edge" cutoff)
    - fraction above 0.5 (stricter)

Run:
    python -m ts_bridge.exp2_replicate \
        --checkpoint checkpoints/117m-curriculum/pytorch_model.pt \
        --tokenizer  checkpoints/117m-curriculum/tokenizer.json
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ts_bridge.smoke_test    import load_model, random_salad     # noqa: E402
from ts_bridge.variance_check import LONG_PROMPTS                # noqa: E402


def _tau_stats(model, ids: list[int], device: str) -> dict:
    """Return dict of raw-τ statistics over the local-window layers only."""
    x = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        _, _, all_tau = model(x, return_all=True)
    shapes = [tuple(t.shape[-2:]) for t in all_tau]
    canon  = max(set(shapes), key=shapes.count)
    local  = [t for t in all_tau if tuple(t.shape[-2:]) == canon]
    tau    = torch.stack([t[0].float() for t in local])         # L T H W

    # Apply causal validity mask — the kernel zero-fills positions s < 0,
    # so unfiltered τ means on short sequences include those zeros and
    # depress mean τ.  Mask keeps only (t, w) with s = t - (W - w) >= 0.
    L, T, H, W = tau.shape
    t_idx = torch.arange(T).unsqueeze(1)
    w_idx = torch.arange(W).unsqueeze(0)
    valid = (t_idx + w_idx >= W)                                # T W
    valid_full = valid[None, :, None, :].expand(L, T, H, W)      # L T H W
    vals = tau[valid_full]

    return {
        "mean_tau":   float(vals.mean()),
        "frac_gt_0.3": float((vals > 0.3).float().mean()),
        "frac_gt_0.5": float((vals > 0.5).float().mean()),
        "p95_tau":    float(torch.quantile(vals, 0.95)),
        "n_tokens":   T,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer",  default=None)
    ap.add_argument("--device",     default="cpu")
    ap.add_argument("--seed_base",  type=int, default=42)
    args = ap.parse_args()

    model, tokenizer, cfg = load_model(args.checkpoint, args.device, args.tokenizer)
    print(f"model: dim={cfg.dim} layers={cfg.num_layers} heads={cfg.num_heads} "
          f"W={cfg.window} vocab={cfg.vocab_size}")
    print(f"Replicating Exp 2 — raw τ stats on {len(LONG_PROMPTS)} "
          "coherent / length-matched salad pairs\n")

    keys = ["mean_tau", "frac_gt_0.3", "frac_gt_0.5", "p95_tau"]
    log_all: dict[str, list[float]] = {k: [] for k in keys}
    sal_all: dict[str, list[float]] = {k: [] for k in keys}

    print(f"  {'#':<3}{'tok':<5}{'mean τ L/S':<20}"
          f"{'>0.3 L/S':<20}{'>0.5 L/S':<20}")
    for i, prompt in enumerate(LONG_PROMPTS):
        log_ids = tokenizer.encode(prompt).ids[:cfg.max_seq_len]
        sal_ids = random_salad(cfg.vocab_size, len(log_ids),
                               seed=args.seed_base + i)

        sl = _tau_stats(model, log_ids, args.device)
        ss = _tau_stats(model, sal_ids, args.device)

        for k in keys:
            log_all[k].append(sl[k])
            sal_all[k].append(ss[k])

        print(f"  {i:<3}{sl['n_tokens']:<5}"
              f"{sl['mean_tau']:.4f} / {ss['mean_tau']:<11.4f}"
              f"{sl['frac_gt_0.3']:.3f} / {ss['frac_gt_0.3']:<12.3f}"
              f"{sl['frac_gt_0.5']:.3f} / {ss['frac_gt_0.5']:<12.3f}")

    print("\n── Aggregate (Exp 2 replication) ──")
    print(f"  {'metric':<16}{'logical':<20}{'salad':<20}{'delta (L/S)':<16}positive")
    print("  " + "─" * 80)
    for k in keys:
        lm = statistics.mean(log_all[k])
        ls = statistics.pstdev(log_all[k])
        sm = statistics.mean(sal_all[k])
        ss = statistics.pstdev(sal_all[k])
        delta = lm / sm if sm > 1e-9 else float("inf")
        positive = sum(1 for a, b in zip(log_all[k], sal_all[k]) if a > b)
        print(f"  {k:<16}{lm:.4f} ± {ls:.4f}    {sm:.4f} ± {ss:.4f}    "
              f"{delta:.3f}×           {positive}/{len(log_all[k])}")

    # Exp 2 published claims:
    #   +25% mean τ → delta ≥ 1.25 on mean_tau
    #   +60% active edges → delta ≥ 1.60 on frac_gt_0.3 (canonical cutoff)
    exp2_mean_delta = statistics.mean(log_all["mean_tau"]) / statistics.mean(sal_all["mean_tau"])
    exp2_active_delta = (statistics.mean(log_all["frac_gt_0.3"])
                         / max(statistics.mean(sal_all["frac_gt_0.3"]), 1e-9))
    print("\n── vs Exp 2 published claims ──")
    print(f"  mean τ     : observed {exp2_mean_delta:.3f}×     (Exp 2: 1.25×)")
    print(f"  active >0.3: observed {exp2_active_delta:.3f}×     (Exp 2: 1.60×)")


if __name__ == "__main__":
    main()
