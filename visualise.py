"""
visualise.py — Tension field analysis and visualisation for TensionLM.

Modes:
  heatmap   Per-head tau heatmaps for a sentence (saved to PNG)
  token     Per-head pull map for a specific token (terminal output)
  stats     Head specialisation statistics over a sample of text
  layers    How tau evolves across layers for a specific token

Usage:
    python visualise.py --checkpoint checkpoints/tension_fw10b/latest.pt \\
        --mode heatmap \\
        --text "The history of artificial intelligence" \\
        --out tension_heatmap.png

    python visualise.py --checkpoint checkpoints/tension_fw10b/latest.pt \\
        --mode token \\
        --text "Manchester United won the Premier League title" \\
        --token_idx 2

    python visualise.py --checkpoint checkpoints/tension_fw10b/latest.pt \\
        --mode stats \\
        --sample_file data/fineweb-10B/val_0000.bin \\
        --sample_size 200

    python visualise.py --checkpoint checkpoints/tension_fw10b/latest.pt \\
        --mode layers \\
        --text "Scientists discovered that water conducts electricity" \\
        --token_idx -1
"""

from __future__ import annotations

import argparse
import sys
import math
from pathlib import Path

import numpy as np
import torch

from model import TensionConfig, TensionLM


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(ckpt_path: str):
    print(f"Loading {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg  = TensionConfig(**ckpt["cfg"])

    model = TensionLM(cfg)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(ckpt["tok_path"])

    ppl  = ckpt.get("val_ppl", "?")
    step = ckpt.get("step", "?")
    ppl_str = f"{ppl:.2f}" if isinstance(ppl, float) else str(ppl)
    print(f"  step={step}  val_ppl={ppl_str}")
    print(f"  layers={cfg.num_layers}  heads={cfg.num_heads}  window={cfg.window}  dim={cfg.dim}\n")
    return model, tokenizer, cfg


def encode(tokenizer, text: str):
    enc  = tokenizer.encode(text)
    ids  = enc.ids
    toks = [tokenizer.id_to_token(i) or f"<{i}>" for i in ids]
    return ids, toks


def get_all_tensions(model, ids: list[int]):
    """Return all_tensions: list of L tensors, each [1, T, H, W]."""
    with torch.no_grad():
        inp = torch.tensor([ids], dtype=torch.long)
        _, _, all_tensions = model(inp, return_all=True)
    return all_tensions  # list[L] of [1, T, H, W]


def tau_tensor(all_tensions) -> torch.Tensor:
    """Stack into [L, T, H, W], squeeze batch."""
    return torch.stack([t[0] for t in all_tensions])  # L T H W


# ── Mode: heatmap ─────────────────────────────────────────────────────────────

def mode_heatmap(model, tokenizer, cfg, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        sys.exit("matplotlib required: pip install matplotlib")

    ids, toks = encode(tokenizer, args.text)
    T = len(ids)
    all_tensions = get_all_tensions(model, ids)
    tau = tau_tensor(all_tensions)  # L T H W

    L, _, H, W = tau.shape
    fig, axes = plt.subplots(
        L, H,
        figsize=(H * 3, L * 2.5),
        squeeze=False,
    )
    fig.suptitle(f'Tension heatmap\n"{args.text}"', fontsize=11, y=1.01)

    for l in range(L):
        for h in range(H):
            ax = axes[l][h]
            # Flip W so col 0 = most recent past, col W-1 = oldest
            data = tau[l, :, h, :].flip(-1).numpy()  # T W
            im = ax.imshow(
                data, aspect="auto", vmin=0, vmax=1,
                cmap="viridis", interpolation="nearest",
            )
            if l == 0:
                ax.set_title(f"Head {h}", fontsize=8)
            if h == 0:
                ax.set_ylabel(f"L{l+1}", fontsize=8)
            ax.set_yticks(range(T))
            ax.set_yticklabels(toks, fontsize=6)
            ax.set_xlabel("recent → old", fontsize=6)
            ax.tick_params(labelsize=6)

    plt.colorbar(im, ax=axes, label="tau", fraction=0.015, pad=0.04)
    plt.tight_layout()
    out = args.out or "tension_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


# ── Mode: token ───────────────────────────────────────────────────────────────

def mode_token(model, tokenizer, cfg, args):
    ids, toks = encode(tokenizer, args.text)
    T = len(ids)
    t = args.token_idx % T  # support negative indices

    all_tensions = get_all_tensions(model, ids)
    tau = tau_tensor(all_tensions)  # L T H W
    W   = cfg.window

    layers_to_show = range(cfg.num_layers) if args.layer is None else [args.layer]

    print(f'Token "{toks[t]}" (pos {t}) — text: "{args.text}"')
    print("─" * 70)

    for l in layers_to_show:
        print(f"\nLayer {l + 1}:")
        for h in range(cfg.num_heads):
            scores = []
            for w_viz in range(min(W, t)):
                # w_viz=0 → most recent past token (t-1)
                # model stores w=W-1 as most recent, so model_w = W-1-w_viz
                model_w = W - 1 - w_viz
                src_tok = toks[t - w_viz - 1]
                val     = tau[l, t, h, model_w].item()
                scores.append((src_tok, val, t - w_viz - 1))

            # Sort by tau descending
            scores.sort(key=lambda x: x[1], reverse=True)
            top = scores[:8]  # show top 8

            parts = []
            for src_tok, val, pos in top:
                bar = "▓" * max(1, round(val * 10))
                parts.append(f"{src_tok}[{pos}]:{val:.2f}{bar}")

            print(f"  Head {h:2d}:  {'  '.join(parts)}")


# ── Mode: layers ──────────────────────────────────────────────────────────────

def mode_layers(model, tokenizer, cfg, args):
    ids, toks = encode(tokenizer, args.text)
    T = len(ids)
    t = args.token_idx % T

    all_tensions = get_all_tensions(model, ids)
    tau = tau_tensor(all_tensions)  # L T H W
    W   = cfg.window

    print(f'Layer evolution for token "{toks[t]}" (pos {t})')
    print(f'Text: "{args.text}"')
    print("─" * 70)

    for l in range(cfg.num_layers):
        # Average across heads
        tau_avg = tau[l, t, :, :].mean(0)  # W
        # Reverse window so index 0 = most recent past token
        tau_reversed = tau[l, t, :, :].mean(0).flip(0)  # W, reversed
        valid = min(W, t)
        top_k = min(6, valid)
        vals, idxs = tau_reversed[:valid].topk(top_k)

        parts = []
        for val, w_viz in zip(vals.tolist(), idxs.tolist()):
            src = toks[t - w_viz - 1]
            bar = "▓" * max(1, round(val * 10))
            parts.append(f"{src}:{val:.2f}{bar}")

        mean_tau = tau_avg[:min(W, t)].mean().item()
        print(f"  Layer {l+1:2d} (mean τ={mean_tau:.3f}):  {'  '.join(parts)}")


# ── Mode: stats ───────────────────────────────────────────────────────────────

def mode_stats(model, tokenizer, cfg, args):
    """Compute per-head statistics over a sample of text."""

    # Gather sample texts
    sample_texts = []
    if args.text:
        sample_texts = [args.text]
    elif args.sample_file:
        # Load from binary shard and decode
        arr = np.memmap(args.sample_file, dtype=np.uint16, mode="r")
        ids = arr[:args.sample_size * 64].tolist()
        # Split into chunks of 64
        chunks = [ids[i:i+64] for i in range(0, len(ids), 64)][:args.sample_size]
        sample_texts = [tokenizer.decode(chunk) for chunk in chunks]
    else:
        sys.exit("Provide --text or --sample_file for stats mode")

    L, H, W = cfg.num_layers, cfg.num_heads, cfg.window

    # Accumulators: per (layer, head)
    mean_tau_acc    = np.zeros((L, H))
    peak_pos_acc    = np.zeros((L, H))
    variance_acc    = np.zeros((L, H))
    count           = 0

    for i, text in enumerate(sample_texts):
        if not text.strip():
            continue
        try:
            ids, toks = encode(tokenizer, text)
            if len(ids) < 4:
                continue
            all_tensions = get_all_tensions(model, ids)
            tau = tau_tensor(all_tensions).numpy()  # L T H W

            T = tau.shape[1]
            for l in range(L):
                for h in range(H):
                    vals = tau[l, :, h, :]  # T W — includes padding zeros for early positions
                    mean_tau_acc[l, h]  += vals.mean()
                    peak_pos_acc[l, h]  += vals.argmax(axis=-1).mean()  # mean peak position
                    variance_acc[l, h]  += vals.var()
            count += 1

        except Exception as e:
            continue

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(sample_texts)} texts...")

    if count == 0:
        sys.exit("No valid texts processed")

    mean_tau_acc  /= count
    peak_pos_acc  /= count
    variance_acc  /= count

    print(f"\nHead specialisation stats over {count} texts")
    print(f"{'Layer':<6} {'Head':<5} {'Mean τ':<10} {'Peak pos':<12} {'τ variance':<12} {'Role guess'}")
    print("─" * 65)

    for l in range(L):
        for h in range(H):
            mt  = mean_tau_acc[l, h]
            pp  = peak_pos_acc[l, h]
            var = variance_acc[l, h]

            # Heuristic role assignment
            if mt < 0.15:
                role = "inactive"
            elif pp < W * 0.2 and var > 0.05:
                role = "local (recent)"
            elif pp > W * 0.7:
                role = "long-range"
            elif var < 0.02:
                role = "uniform/diffuse"
            else:
                role = "mid-range"

            print(f"  L{l+1:<4} H{h:<4} {mt:<10.3f} {pp:<12.1f} {var:<12.4f} {role}")
        if l < L - 1:
            print()

    # Summary: most active and least active heads
    flat_mean = mean_tau_acc.flatten()
    flat_idx  = np.unravel_index(flat_mean.argmax(), mean_tau_acc.shape)
    flat_min  = np.unravel_index(flat_mean.argmin(), mean_tau_acc.shape)
    print(f"\nMost active:  Layer {flat_idx[0]+1}, Head {flat_idx[1]}  (mean τ={mean_tau_acc[flat_idx]:.3f})")
    print(f"Least active: Layer {flat_min[0]+1}, Head {flat_min[1]}  (mean τ={mean_tau_acc[flat_min]:.3f})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="TensionLM tension field visualiser",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--mode",         required=True,
                   choices=["heatmap", "token", "stats", "layers"])
    p.add_argument("--text",         default=None,
                   help="Input text for heatmap / token / layers modes")
    p.add_argument("--token_idx",    default=-1,  type=int,
                   help="Token position to inspect (token/layers mode). -1 = last token")
    p.add_argument("--layer",        default=None, type=int,
                   help="Layer to show (token mode). None = all layers")
    p.add_argument("--out",          default=None,
                   help="Output PNG path (heatmap mode)")
    p.add_argument("--sample_file",  default=None,
                   help="Binary shard (.bin) to sample texts from (stats mode)")
    p.add_argument("--sample_size",  default=200, type=int,
                   help="Number of text samples for stats mode")
    return p.parse_args()


def main():
    args  = get_args()
    model, tokenizer, cfg = load_model_and_tokenizer(args.checkpoint)

    if args.mode == "heatmap":
        if not args.text:
            sys.exit("--text required for heatmap mode")
        mode_heatmap(model, tokenizer, cfg, args)

    elif args.mode == "token":
        if not args.text:
            sys.exit("--text required for token mode")
        mode_token(model, tokenizer, cfg, args)

    elif args.mode == "layers":
        if not args.text:
            sys.exit("--text required for layers mode")
        mode_layers(model, tokenizer, cfg, args)

    elif args.mode == "stats":
        mode_stats(model, tokenizer, cfg, args)


if __name__ == "__main__":
    main()
