"""
ts_bridge.smoke_test
=====================

Phase 1 acceptance test: run TauExporter against a trained checkpoint on
(a) a logical prompt and (b) length-matched random word salad, confirm the
exported graph is denser / higher-mean-weight for the coherent input.

This is the structural analogue of Exp 2 in the README (+25% τ / +60% edge
density on coherent vs salad text) — same phenomenon, now measured at the
graph-export layer instead of internal τ.

Also exercises JSON round-trip to prove the serialisation contract in
schema.md holds.

Run:
    python -m ts_bridge.smoke_test --checkpoint checkpoints/diagnostic/latest.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

# Allow running as a script from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import TensionConfig, TensionLM                       # noqa: E402
from ts_bridge import (                                           # noqa: E402
    TauExporter, UniversalLivingGraph,
    format_profile, profile_heads,
)


LOGICAL_PROMPT = (
    "If all mammals are warm-blooded and all whales are mammals then whales are"
)


def _migrate_fused_kv(state: dict, cfg: "TensionConfig") -> dict:
    """
    Older checkpoints fused K and V into a single `wkv` linear even when RoPE was
    enabled; current model.py splits them when use_rope=True.  Reshape the fused
    weight `[D*2, D]` back into `wk [D, D]` and `wv [D, D]` so the old weights
    load into the new module layout.

    Fused layout per `self.wkv(x).view(B, T, H, HD*2)` → split on last dim: rows
    of wkv.weight are grouped as [h0_k, h0_v, h1_k, h1_v, ...].
    """
    if not cfg.use_rope:
        return state
    out = {}
    H, HD, D = cfg.num_heads, cfg.head_dim, cfg.dim
    for k, v in state.items():
        if k.endswith(".wkv.weight"):
            prefix = k[: -len(".wkv.weight")]
            # [D*2, D] → [H, 2, HD, D]  (2 = k/v interleave within each head)
            w = v.view(H, 2, HD, D)
            out[f"{prefix}.wk.weight"] = w[:, 0].reshape(D, D)
            out[f"{prefix}.wv.weight"] = w[:, 1].reshape(D, D)
        else:
            out[k] = v
    return out


def load_model(ckpt_path: str, device: str = "cpu", tokenizer_path: str | None = None):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = dict(ckpt["cfg"])
    # Force reference path — smoke test should be runnable on CPU.
    cfg_dict["use_triton"] = False
    cfg = TensionConfig(**cfg_dict)
    model = TensionLM(cfg)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    state = _migrate_fused_kv(state, cfg)
    model.load_state_dict(state)
    model.eval().to(device)

    from tokenizers import Tokenizer
    tok_path = tokenizer_path or ckpt["tok_path"]
    tokenizer = Tokenizer.from_file(tok_path)
    return model, tokenizer, cfg


def token_ids_for(prompt: str, tokenizer, max_len: int) -> list[int]:
    ids = tokenizer.encode(prompt).ids
    return ids[:max_len]


def random_salad(vocab_size: int, length: int, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    # Avoid special/low ids — 256+ keeps us in content vocab for most BPE setups.
    return [rng.randint(256, vocab_size - 1) for _ in range(length)]


@torch.no_grad()
def export_for(
    ids: list[int],
    model,
    tokenizer,
    device: str,
    head_override: list[tuple[int, int]] | None = None,
    edge_threshold: float = 0.3,
) -> tuple[UniversalLivingGraph, "ExportStats", list]:
    """Run forward, profile+lock heads (or use override), ingest, return
    (graph, stats, head_stats).  `head_override` lets callers pin the
    signal-head set to a corpus-profiled one instead of per-prompt."""
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    _, _, all_tensions = model(x, return_all=True)

    graph = UniversalLivingGraph()
    exporter = TauExporter(
        graph,
        edge_threshold    = edge_threshold,
        include_mid_range = True,
        head_override     = head_override,
    )
    if head_override is None:
        head_stats = exporter.profile_and_lock(all_tensions)
    else:
        head_stats = []
    stats = exporter.ingest(ids, all_tensions, tokenizer)
    return graph, stats, head_stats


def summarise(label: str, graph: UniversalLivingGraph, stats) -> None:
    print(f"\n── {label} ──")
    print(f"  tokens                 {stats.tokens}")
    print(f"  signal / total heads   {stats.signal_heads} / {stats.total_heads}")
    print(f"  candidate pairs        {stats.candidate_pairs}")
    print(f"  edges emitted          {stats.edges_emitted}")
    print(f"  nodes touched          {stats.nodes_touched}")
    print(f"  mean edge weight       {stats.mean_weight:.4f}")
    print(f"  graph density          {graph.density():.4f}")
    print(f"  graph mean activation  {graph.mean_activation():.4f}")
    print(f"  graph mean edge w.     {graph.mean_edge_weight():.4f}")


def top_edges(graph: UniversalLivingGraph, k: int = 10) -> None:
    edges = sorted(graph.edges, key=lambda e: e.weight, reverse=True)[:k]
    for e in edges:
        d = e.metadata.get("distance", "?")
        print(f"    {e.src:>20s}  →  {e.dst:<20s}  w={e.weight:.3f}  dist={d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="Override tokenizer path (default: ckpt['tok_path'])")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json_out", default="logs/smoke_graph.json")
    ap.add_argument("--show_profile", action="store_true",
                    help="Print full per-head classification table")
    args = ap.parse_args()

    model, tokenizer, cfg = load_model(args.checkpoint, args.device, args.tokenizer)
    print(f"model: dim={cfg.dim} layers={cfg.num_layers} heads={cfg.num_heads} "
          f"W={cfg.window} vocab={cfg.vocab_size}")

    # Length-matched inputs.
    logical_ids = token_ids_for(LOGICAL_PROMPT, tokenizer, cfg.max_seq_len)
    salad_ids   = random_salad(cfg.vocab_size, len(logical_ids), seed=42)

    g_logical, stats_logical, head_stats = export_for(
        logical_ids, model, tokenizer, args.device,
    )
    g_salad, stats_salad, _ = export_for(
        salad_ids, model, tokenizer, args.device,
    )

    if args.show_profile:
        print("\n── Head profile (locked to logical prompt) ──")
        print(format_profile(head_stats))

    summarise("Coherent logical prompt", g_logical, stats_logical)
    print("\n  top-10 edges:")
    top_edges(g_logical, k=10)

    summarise("Random word salad (length-matched)", g_salad, stats_salad)

    # Structural-coherence delta — the Exp 2 analogue at the graph layer.
    def _safe_ratio(a: float, b: float) -> str:
        if b < 1e-9:
            return "∞ (salad baseline ≈ 0)"
        return f"{a / b:.2f}×"

    print("\n── Coherence delta (logical vs salad) ──")
    print(f"  density ratio       {_safe_ratio(g_logical.density(), g_salad.density())}")
    print(f"  mean-weight ratio   {_safe_ratio(g_logical.mean_edge_weight(), g_salad.mean_edge_weight())}")
    print(f"  edge-count ratio    {_safe_ratio(stats_logical.edges_emitted, stats_salad.edges_emitted)}")

    # JSON round-trip — proves schema.md contract.
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g_logical.to_json(out_path)
    g_round = UniversalLivingGraph.from_json(out_path)
    assert len(g_round.nodes) == len(g_logical.nodes), "node count mismatch after round-trip"
    assert len(g_round.edges) == len(g_logical.edges), "edge count mismatch after round-trip"
    print(f"\n✓ JSON round-trip OK — wrote {out_path} "
          f"({len(g_logical.nodes)} nodes, {len(g_logical.edges)} edges)")


if __name__ == "__main__":
    main()
