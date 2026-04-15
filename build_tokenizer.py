"""
build_tokenizer.py — Train a domain-specific BPE tokenizer for TensionLM.

Trains on two sources:
  1. Generated logical inference examples (logic generator, no download needed)
  2. A streamed sample of mathematical text (open-web-math from HuggingFace)

Design rationale:
  - Vocab 16384: our domain is logic connectives, numbers, math symbols, and
    constrained mathematical English. This covers the domain with clean token
    boundaries while keeping the embedding table small (vs 32k or 50k for
    general LLMs). At 350M params, the freed embedding capacity goes into
    actual computation layers.
  - ByteLevel BPE: handles all Unicode including math symbols without <unk>.
  - Trained on domain data: tokens like "therefore", "implies", "sqrt", "frac"
    become single tokens naturally from frequency — not by hand-coding.
  - Minimal special tokens: <pad>, <bos>, <eos>, <unk>. Structural tokens for
    stage 3 (worked solutions format) will be added once that format is decided.

Usage:
    python build_tokenizer.py                        # default settings
    python build_tokenizer.py --vocab_size 16384 --math_samples 100000
    python build_tokenizer.py --logic_only           # skip HF download
"""

from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path


# ── Text generators ───────────────────────────────────────────────────────────

def generate_logic_corpus(n_chars: int, seed: int = 42) -> str:
    """Generate a block of logical inference text for tokenizer training."""
    random.seed(seed)
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_logic_data import sample_example

    buf = io.StringIO()
    written = 0
    while written < n_chars:
        ex = sample_example() + "\n\n"
        buf.write(ex)
        written += len(ex)
    return buf.getvalue()


def stream_math_corpus(n_samples: int) -> str:
    """Stream a sample of mathematical text from open-web-math (HuggingFace)."""
    print(f"  Streaming {n_samples:,} samples from open-web-math (HuggingFace)...")
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "open-web-math/open-web-math",
            split="train",
            streaming=True,
        )
        buf = io.StringIO()
        for i, row in enumerate(ds):
            if i >= n_samples:
                break
            text = row.get("text", "")
            if text.strip():
                buf.write(text[:2000])   # cap per-doc length to avoid huge outliers
                buf.write("\n\n")
        result = buf.getvalue()
        print(f"  Got {len(result)/1e6:.1f}M chars of math text")
        return result
    except Exception as e:
        print(f"  Warning: could not stream math corpus ({e})")
        print("  Proceeding with logic corpus only.")
        return ""


# ── Tokenizer training ────────────────────────────────────────────────────────

def train_tokenizer(corpus: str, vocab_size: int, out_path: str):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.processors import TemplateProcessing

    print(f"\nTraining BPE tokenizer — vocab {vocab_size:,} ...")
    print(f"  Corpus size: {len(corpus)/1e6:.1f}M chars")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # Post-processor: automatically wrap sequences with <bos> and <eos>
    # when encoding with add_special_tokens=True (optional — callers can skip it).
    # Disabled by default so the training pipeline controls BOS/EOS explicitly.

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        min_frequency=2,
        show_progress=True,
        # Merges tuned for a domain-specific corpus: more aggressive than
        # default so common math/logic phrases merge into single tokens.
        initial_alphabet=ByteLevel.alphabet(),
    )

    # Train from the corpus string split into lines (BpeTrainer expects an iterator)
    lines = [l for l in corpus.split("\n") if l.strip()]
    print(f"  Training on {len(lines):,} lines ...")
    tokenizer.train_from_iterator(lines, trainer=trainer)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_path)

    # ── Sanity checks ──
    print(f"\nTokenizer saved → {out_path}")
    print(f"Actual vocab size: {tokenizer.get_vocab_size():,}")

    checks = [
        "If P then Q. P is true. Therefore Q is true.",
        "Rule 1: if it is raining then the ground is wet. Therefore if it is raining then the ground is wet.",
        "To prove that the system is stable, assume the opposite: the system is unstable.",
        "The derivative of x squared is 2x.",
        "If A then B. If B then C. Therefore A implies C.",
        "The integral of 1 dx equals x plus a constant.",
        "therefore hence implies because since thus",
    ]

    print("\nSanity checks (text → n_tokens → decode):")
    total_chars = 0
    total_tokens = 0
    for text in checks:
        enc = tokenizer.encode(text)
        decoded = tokenizer.decode(enc.ids)
        match = "✓" if decoded.strip() == text.strip() else "✗"
        ratio = len(enc.ids) / max(len(text.split()), 1)
        print(f"  [{match}] {len(enc.ids):3d} tokens  ({ratio:.1f} tok/word)  {text[:60]}")
        total_chars += len(text)
        total_tokens += len(enc.ids)

    print(f"\nOverall: {total_tokens/total_chars:.2f} tokens/char on domain text")
    print(f"Special tokens: <pad>=0  <bos>=1  <eos>=2  <unk>=3")

    return tokenizer


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Train domain-specific tokenizer for TensionLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out",           default="data/tokenizer.json",
                   help="Output path for the tokenizer")
    p.add_argument("--vocab_size",    default=16384, type=int,
                   help="BPE vocabulary size")
    p.add_argument("--logic_chars",   default=20_000_000, type=int,
                   help="Characters of logic text to generate for training")
    p.add_argument("--math_samples",  default=80_000, type=int,
                   help="Number of open-web-math documents to stream")
    p.add_argument("--logic_only",    action="store_true",
                   help="Skip math corpus streaming (faster, logic domain only)")
    p.add_argument("--seed",          default=42, type=int)
    args = p.parse_args()

    print("=" * 60)
    print("TensionLM tokenizer builder")
    print(f"  vocab_size  : {args.vocab_size:,}")
    print(f"  logic_chars : {args.logic_chars/1e6:.0f}M")
    if not args.logic_only:
        print(f"  math_samples: {args.math_samples:,}")
    print(f"  output      : {args.out}")
    print("=" * 60)

    # Build training corpus
    print("\n[1/2] Generating logic corpus ...")
    logic_text = generate_logic_corpus(args.logic_chars, seed=args.seed)
    print(f"  Generated {len(logic_text)/1e6:.1f}M chars of logic text")

    math_text = ""
    if not args.logic_only:
        print("\n[2/2] Streaming math corpus ...")
        math_text = stream_math_corpus(args.math_samples)
    else:
        print("\n[2/2] Skipping math corpus (--logic_only)")

    corpus = logic_text + "\n\n" + math_text
    print(f"\nTotal corpus: {len(corpus)/1e6:.1f}M chars")

    # Train
    train_tokenizer(corpus, args.vocab_size, args.out)


if __name__ == "__main__":
    main()
