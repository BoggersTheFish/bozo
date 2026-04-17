"""
ts_bridge.variance_check
=========================

Run the Phase 1 exporter on N coherent prompts (each paired with a length-
matched salad) and report mean ± std of the coherence-delta ratios.  The
single-prompt smoke test is directional evidence; this is the stability test.

Run:
    python -m ts_bridge.variance_check \
        --checkpoint checkpoints/117m-curriculum/pytorch_model.pt \
        --tokenizer  checkpoints/117m-curriculum/tokenizer.json
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ts_bridge import TauExporter, UniversalLivingGraph          # noqa: E402
from ts_bridge.smoke_test import (                               # noqa: E402
    export_for, load_model, random_salad,
)


_SHORT_PROMPTS = [
    "If all mammals are warm-blooded and all whales are mammals then whales are",
    "Socrates is a man and all men are mortal therefore Socrates is",
    "A dog is a mammal and every mammal is a vertebrate so a dog is a",
    "No fish are mammals and a whale is a mammal therefore a whale is not a",
    "When water reaches one hundred degrees it boils this pot is at one hundred so it",
    "Three plus four equals seven and seven plus two equals nine so three plus six",
    "Every bird has feathers and a sparrow is a bird so a sparrow has",
    "If it rains the grass gets wet it is raining therefore the grass is",
    "All squares are rectangles and all rectangles are quadrilaterals so every square is a",
    "If the light is red cars must stop the light is red therefore cars must",
]

# Long passages chosen so the tokenised length exceeds W=64 — heads can then
# distinguish genuinely-long-range vs genuinely-short-range attention, which
# the ~15-token smoke prompts cannot.  Each is a multi-step reasoning chain
# where later tokens depend on early premises.
LONG_PROMPTS = [
    (
        "All mammals are warm-blooded animals that nurse their young with milk. "
        "Whales live in the ocean and give birth to live offspring which they "
        "nurse with milk produced by specialised glands. Fish lay eggs in the "
        "water and do not nurse their young. Warm-blooded animals regulate "
        "their internal body temperature regardless of the surrounding water. "
        "Therefore whales are not fish but are"
    ),
    (
        "Every student who passes the final exam will receive a certificate. "
        "Students who attend at least eighty percent of the lectures and "
        "complete all of the assignments tend to pass the final exam. Maria "
        "attended every single lecture throughout the semester and completed "
        "every assignment on time with high marks. So it is reasonable to "
        "expect that Maria will receive a"
    ),
    (
        "Water freezes into ice at zero degrees Celsius and boils into steam "
        "at one hundred degrees Celsius under standard atmospheric pressure. "
        "The kettle in the kitchen contains ordinary tap water and has been "
        "heating on the stove for several minutes. The thermometer placed "
        "inside the kettle now reads one hundred degrees Celsius exactly. "
        "Therefore the water inside the kettle must be turning into"
    ),
    (
        "Every prime number greater than two is odd because an even number "
        "greater than two would be divisible by two and therefore not prime. "
        "The number seventeen cannot be divided evenly by any number other "
        "than one and itself so seventeen is a prime number. Since seventeen "
        "is a prime number greater than two it follows from the rule that "
        "seventeen must be"
    ),
    (
        "When a tree absorbs carbon dioxide through its leaves during "
        "photosynthesis the carbon is stored inside the wood as the tree "
        "grows. When forests are cut down and burned the stored carbon is "
        "released back into the atmosphere as carbon dioxide. Replanting "
        "forests therefore removes carbon from the atmosphere by"
    ),
    (
        "A triangle with all three sides equal in length is called "
        "equilateral. Every equilateral triangle is also equiangular meaning "
        "all three interior angles are equal. Since the interior angles of "
        "any triangle must add up to one hundred and eighty degrees this "
        "means every interior angle of an equilateral triangle measures"
    ),
    (
        "If a person runs faster than everyone else in the race they win the "
        "gold medal. Anna ran faster than every other competitor from the "
        "starting line to the finish line today. Every competitor crossed "
        "the finish line and Anna crossed it first by a considerable margin. "
        "The award ceremony will therefore give Anna the"
    ),
    (
        "Diamonds are made of pure carbon arranged in a tetrahedral crystal "
        "lattice under extreme heat and pressure deep inside the earth. "
        "Graphite is also made of pure carbon but its atoms are arranged in "
        "flat hexagonal sheets that slide over each other easily. Although "
        "diamond and graphite share the same chemical composition they have "
        "different"
    ),
    (
        "A number is divisible by six if and only if it is divisible by two "
        "and also divisible by three. The number twenty-four can be written "
        "as two times twelve so it is divisible by two and it can be written "
        "as three times eight so it is divisible by three. Therefore the "
        "number twenty-four must be divisible by"
    ),
    (
        "Electric cars draw their energy from batteries that must be "
        "recharged from a power source. When the power source is a coal-"
        "fired power plant the overall emissions from driving the electric "
        "car are only reduced if the plant itself emits less carbon per "
        "kilometre than a petrol engine. Cleaning up the electrical grid "
        "therefore directly reduces the emissions associated with"
    ),
]

# Default to the long set — classifier calibration only meaningful when
# prompt length exceeds W.
LOGICAL_PROMPTS = LONG_PROMPTS


def _summarise_ratio(name: str, values: list[float]) -> str:
    mean = statistics.mean(values)
    std  = statistics.pstdev(values) if len(values) > 1 else 0.0
    lo, hi = min(values), max(values)
    return f"  {name:<20s}  {mean:>6.2f}× ± {std:>4.2f}   range [{lo:.2f}, {hi:.2f}]"


def _safe_ratio(a: float, b: float, fallback: float) -> float:
    if b < 1e-9:
        return fallback
    return a / b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer",  default=None)
    ap.add_argument("--device",     default="cpu")
    ap.add_argument("--seed_base",  type=int, default=42)
    ap.add_argument("--head_profile", default=None,
                    help="Path to corpus-profiled head JSON (from ts_bridge.corpus_profile). "
                         "Pins the signal-head set instead of per-prompt profiling.")
    ap.add_argument("--edge_threshold", type=float, default=0.3)
    args = ap.parse_args()

    head_override: list[tuple[int, int]] | None = None
    edge_threshold = args.edge_threshold
    if args.head_profile:
        import json
        prof = json.loads(Path(args.head_profile).read_text())
        head_override = [tuple(lh) for lh in prof["signal_heads"]]
        print(f"using corpus-profiled head set: {len(head_override)} heads "
              f"(from {args.head_profile})")
        # q50 from the profile matches the historical default on 117M-Curriculum
        # and empirically gives the tightest coherent/salad discrimination.
        # Only auto-adopt when the caller is still on the hard-coded default.
        if abs(args.edge_threshold - 0.3) < 1e-9:
            q50 = prof.get("edge_threshold_quantiles", {}).get("q50")
            if q50 is not None:
                edge_threshold = float(q50)
                print(f"  using edge_threshold = {edge_threshold:.3f} "
                      "(q50 from corpus profile — tightest discrimination)")

    model, tokenizer, cfg = load_model(args.checkpoint, args.device, args.tokenizer)
    print(f"model: dim={cfg.dim} layers={cfg.num_layers} heads={cfg.num_heads} "
          f"W={cfg.window} vocab={cfg.vocab_size}")
    print(f"N prompts: {len(LOGICAL_PROMPTS)}\n")

    density_ratios: list[float]   = []
    weight_ratios:  list[float]   = []
    edge_ratios:    list[float]   = []
    gini_ratios:    list[float]   = []
    agg_mean_ratios: list[float]  = []
    # Track signal-head selection on the first prompt to see whether the new
    # classifier actually picks anything out.
    signal_heads_seen: list[int] = []
    total_heads_seen:  list[int] = []

    print(f"  {'#':<3}{'tokens':<7}{'heads':<10}"
          f"{'edges L/S':<13}{'density L/S':<19}"
          f"{'gini L/S':<17}{'agg μ L/S':<16}")
    for i, prompt in enumerate(LOGICAL_PROMPTS):
        log_ids = tokenizer.encode(prompt).ids[:cfg.max_seq_len]
        sal_ids = random_salad(cfg.vocab_size, len(log_ids),
                               seed=args.seed_base + i)

        g_log, s_log, _ = export_for(log_ids, model, tokenizer, args.device,
                                     head_override=head_override,
                                     edge_threshold=edge_threshold)
        g_sal, s_sal, _ = export_for(sal_ids, model, tokenizer, args.device,
                                     head_override=head_override,
                                     edge_threshold=edge_threshold)

        # Ratio fallbacks: when salad produces zero signal, use the logical
        # count as an "infinity proxy" capped at the number of candidate pairs.
        # This keeps the summary statistics finite without silently hiding
        # strong deltas.
        fallback = float(s_log.candidate_pairs or 1)
        dr = _safe_ratio(g_log.density(),          g_sal.density(),          fallback)
        wr = _safe_ratio(g_log.mean_edge_weight(), g_sal.mean_edge_weight(), 1.0)
        er = _safe_ratio(s_log.edges_emitted,      s_sal.edges_emitted,      fallback)

        gr = _safe_ratio(s_log.concentration, s_sal.concentration, 1.0)
        mr = _safe_ratio(s_log.aggregated_mean, s_sal.aggregated_mean, 1.0)

        density_ratios.append(dr)
        weight_ratios.append(wr)
        edge_ratios.append(er)
        gini_ratios.append(gr)
        agg_mean_ratios.append(mr)
        signal_heads_seen.append(s_log.signal_heads)
        total_heads_seen.append(s_log.total_heads)

        print(f"  {i:<3}{len(log_ids):<7}"
              f"{s_log.signal_heads:>3}/{s_log.total_heads:<6}"
              f"{s_log.edges_emitted:>3}/{s_sal.edges_emitted:<9}"
              f"{g_log.density():>5.3f}/{g_sal.density():<12.3f}"
              f"{s_log.concentration:>5.3f}/{s_sal.concentration:<10.3f}"
              f"{s_log.aggregated_mean:>5.3f}/{s_sal.aggregated_mean:<9.3f}")

    avg_signal = sum(signal_heads_seen) / len(signal_heads_seen)
    avg_total  = sum(total_heads_seen)  / len(total_heads_seen)
    print(f"\n  mean signal heads / total : {avg_signal:.1f} / {avg_total:.0f}")
    # Count how many runs ran through the classifier vs the all-heads fallback
    # — filtered and fallback runs should be analysed separately, but for now
    # we report both together and flag the breakdown.
    n_fallback = sum(1 for fh in signal_heads_seen if fh == total_heads_seen[signal_heads_seen.index(fh)])
    print(f"  head-fallback fired on    : {n_fallback}/{len(signal_heads_seen)} prompts")

    print("\n── Coherence delta over N prompts ──")
    print(_summarise_ratio("density ratio",      density_ratios))
    print(_summarise_ratio("mean-weight ratio",  weight_ratios))
    print(_summarise_ratio("edge-count ratio",   edge_ratios))
    print(_summarise_ratio("concentration (Gini) ratio", gini_ratios))
    print(_summarise_ratio("aggregated-mean ratio",      agg_mean_ratios))

    # Sanity: fraction of prompts where logical > salad on each metric.
    # <70% on any of these should make us suspicious of that metric's
    # calibration.  Density is the raw-throughput check; concentration and
    # aggregated-mean are the "is there structure at all" checks.
    def _positive_frac(rs: list[float]) -> str:
        pos = sum(1 for r in rs if r > 1.0)
        return f"{pos}/{len(rs)}"
    print(f"\n  density        > 1× on {_positive_frac(density_ratios)} prompts")
    print(f"  concentration  > 1× on {_positive_frac(gini_ratios)} prompts")
    print(f"  aggregated μ   > 1× on {_positive_frac(agg_mean_ratios)} prompts")


if __name__ == "__main__":
    main()
