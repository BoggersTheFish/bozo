"""
TensionLM training script
==========================
Downloads WikiText data, trains a BPE tokenizer, then trains TensionLM
with mini-batching, gradient accumulation, cosine LR, and checkpointing.

Quick start (install deps first):
    pip install torch tokenizers datasets

Default run (WikiText-2, ~10-15 min/epoch on Pentium Silver):
    python train.py

Bigger data (~2h/epoch, much better quality after overnight run):
    python train.py --dataset wikitext-103-raw-v1 --max_tokens 10000000

Custom text file:
    python train.py --text_file my_book.txt

Resume interrupted training:
    python train.py --resume
"""

import argparse
import math
import multiprocessing
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import (
    TensionConfig, TensionLM,
    manifold_closure_loss, tension_diversity_loss,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Train TensionLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--text_file",   default=None,
                   help="Custom .txt file (skips auto-download)")
    p.add_argument("--dataset",     default="wikitext-2-raw-v1",
                   choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"],
                   help="HuggingFace dataset name")
    p.add_argument("--max_tokens",  default=None, type=int,
                   help="Cap training tokens (None = all)")
    # Tokeniser — 2048 keeps the LM-head output at 16 MB (bandwidth-limited CPU).
    # Use 4096 or 8192 for better text coverage on faster hardware.
    p.add_argument("--vocab_size",  default=2048, type=int)
    # Architecture
    # Defaults: ~3.5M params, trains overnight on Pentium-class CPU.
    # Quality preset: --dim 256 --num_layers 6 --window 16 (much slower).
    p.add_argument("--dim",         default=128,  type=int)
    p.add_argument("--num_layers",  default=4,    type=int)
    p.add_argument("--num_heads",   default=4,    type=int)
    p.add_argument("--window",      default=8,    type=int)
    p.add_argument("--ffn_mult",    default=3,    type=int)
    p.add_argument("--max_seq_len", default=256,  type=int)
    p.add_argument("--dropout",     default=0.10, type=float)
    p.add_argument("--grad_ckpt",   action="store_true",
                   help="Gradient checkpointing (saves ~40%% RAM, ~30%% slower)")
    # Training
    p.add_argument("--seq_len",     default=64,   type=int)
    p.add_argument("--batch_size",  default=32,   type=int)
    p.add_argument("--grad_accum",  default=2,    type=int,
                   help="Gradient accumulation (effective batch = batch_size × grad_accum)")
    p.add_argument("--lr",          default=3e-4, type=float)
    p.add_argument("--min_lr",      default=3e-5, type=float)
    p.add_argument("--warmup_steps",default=200,  type=int)
    p.add_argument("--epochs",      default=10,   type=int)
    p.add_argument("--weight_decay",default=0.10, type=float)
    p.add_argument("--clip_grad",   default=1.0,  type=float)
    # Aux loss weights (set to 0 to disable and speed up training)
    p.add_argument("--w_closure",   default=0.05, type=float,
                   help="Weight for ManifoldClosure loss (0 to disable)")
    p.add_argument("--w_diversity", default=0.02, type=float,
                   help="Weight for TensionDiversity loss (0 to disable)")
    # I/O
    p.add_argument("--out_dir",     default="checkpoints")
    p.add_argument("--resume",      action="store_true",
                   help="Resume from checkpoints/latest.pt")
    p.add_argument("--log_every",   default=50,   type=int)
    p.add_argument("--eval_every",  default=500,  type=int)
    p.add_argument("--save_every",  default=1000, type=int)
    return p.parse_args()


# ── LR Schedule ───────────────────────────────────────────────────────────────

def get_lr(step: int, warmup: int, total: int, lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup:
        return lr * max(step, 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ── Data ──────────────────────────────────────────────────────────────────────

def load_raw_text(args) -> tuple[str, str]:
    """Return (train_text, val_text)."""
    if args.text_file:
        text  = Path(args.text_file).read_text(encoding="utf-8", errors="replace")
        split = int(len(text) * 0.95)
        print(f"Loaded {args.text_file}: {len(text):,} chars")
        return text[:split], text[split:]

    print(f"Downloading {args.dataset} via HuggingFace datasets...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "\nMissing dependency.  Run:\n"
            "    pip install datasets\n"
            "Or pass --text_file path/to/corpus.txt"
        )
    ds    = load_dataset("wikitext", args.dataset)
    train = "\n".join(t for t in ds["train"]["text"]      if t.strip())
    val   = "\n".join(t for t in ds["validation"]["text"] if t.strip())
    print(f"  train: {len(train):,} chars | val: {len(val):,} chars")
    return train, val


def train_or_load_tokenizer(train_text: str, vocab_size: int, path: str):
    """Load tokenizer if saved, otherwise train BPE and save."""
    if os.path.exists(path):
        print(f"Loading tokenizer from {path}")
        from tokenizers import Tokenizer
        return Tokenizer.from_file(path)

    print(f"Training BPE tokenizer (vocab={vocab_size})...")
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    except ImportError:
        raise SystemExit(
            "\nMissing dependency.  Run:\n"
            "    pip install tokenizers"
        )

    tokenizer                = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer  = ByteLevel()
    tokenizer.decoder        = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        (line for line in train_text.split("\n") if line.strip()),
        trainer=trainer,
    )
    tokenizer.save(path)
    print(f"Tokenizer saved → {path}")
    return tokenizer


def tokenize(tokenizer, text: str, max_tokens: int | None = None) -> list[int]:
    """Encode text to token IDs, processing line-by-line for speed."""
    lines  = [l for l in text.split("\n") if l.strip()]
    ids: list[int] = []
    for enc in tokenizer.encode_batch(lines):
        ids.extend(enc.ids)
        if max_tokens and len(ids) >= max_tokens:
            break
    return ids[:max_tokens] if max_tokens else ids


class TokenDataset(Dataset):
    """Sliding window over a flat token stream.  stride=seq_len → non-overlapping."""
    def __init__(self, token_ids: list[int], seq_len: int, stride: int | None = None):
        self.data    = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride  = stride if stride is not None else seq_len  # non-overlapping by default

    def __len__(self) -> int:
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride)

    def __getitem__(self, idx: int):
        s = idx * self.stride
        c = self.data[s : s + self.seq_len + 1]
        return c[:-1], c[1:]


# ── Checkpointing ─────────────────────────────────────────────────────────────

def _unwrap(model) -> TensionLM:
    """Return the base module even if torch.compiled."""
    return getattr(model, "_orig_mod", model)


def save_checkpoint(out_dir, step, model, val_ppl, cfg, tok_path, args_dict):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "step":     step,
        "model":    _unwrap(model).state_dict(),
        "cfg":      cfg.__dict__,
        "tok_path": tok_path,
        "val_ppl":  val_ppl,
        "args":     args_dict,
    }
    # Numbered + latest
    numbered = os.path.join(out_dir, f"ckpt_{step:07d}.pt")
    latest   = os.path.join(out_dir, "latest.pt")
    torch.save(state, numbered)
    torch.save(state, latest)
    print(f"  Saved → {numbered}  (val ppl {val_ppl:.2f})")


def load_checkpoint(path: str, device: torch.device):
    return torch.load(path, map_location=device, weights_only=False)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches: int = 60) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y    = x.to(device), y.to(device)
        logits  = _unwrap(model)(x)
        total  += criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        n      += 1
    model.train()
    return math.exp(min(total / max(n, 1), 20))


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    # ── Setup ──
    device = torch.device("cpu")
    n_threads = multiprocessing.cpu_count()
    torch.set_num_threads(n_threads)
    print(f"CPU threads: {n_threads}")

    out_dir  = args.out_dir
    tok_path = os.path.join(out_dir, "tokenizer.json")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── Data ──
    train_text, val_text = load_raw_text(args)
    tokenizer            = train_or_load_tokenizer(train_text, args.vocab_size, tok_path)
    actual_vocab         = tokenizer.get_vocab_size()
    print(f"Vocab size: {actual_vocab}")

    print("Tokenising...")
    t0       = time.time()
    train_ids = tokenize(tokenizer, train_text, args.max_tokens)
    val_ids   = tokenize(tokenizer, val_text,   max_tokens=min(200_000, len(val_text) // 4))
    print(f"  train {len(train_ids):,} tokens | val {len(val_ids):,} tokens  ({time.time()-t0:.1f}s)")

    train_ds = TokenDataset(train_ids, args.seq_len)          # stride=seq_len (non-overlapping)
    val_ds   = TokenDataset(val_ids,   args.seq_len, stride=args.seq_len // 2)  # val: denser sampling
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, drop_last=False)
    print(f"  {len(train_ds):,} train sequences | {len(train_loader)} batches/epoch")

    # ── Model ──
    cfg = TensionConfig(
        vocab_size          = actual_vocab,
        dim                 = args.dim,
        num_layers          = args.num_layers,
        num_heads           = args.num_heads,
        window              = args.window,
        ffn_mult            = args.ffn_mult,
        max_seq_len         = args.max_seq_len,
        dropout             = args.dropout,
        use_grad_checkpoint = args.grad_ckpt,
    )
    model = TensionLM(cfg).to(device)

    try:
        model = torch.compile(model)
        print("torch.compile: enabled (first step will be slow — that's normal)")
    except Exception:
        print("torch.compile: not available (upgrade to PyTorch ≥2.0 for free speedup)")

    np = _unwrap(model).num_params
    print(f"\nModel: {np:,} parameters")
    print(f"  dim={cfg.dim}  layers={cfg.num_layers}  heads={cfg.num_heads}  "
          f"window={cfg.window}  ffn_mult={cfg.ffn_mult}  dropout={cfg.dropout}")
    print(f"  vocab={cfg.vocab_size}  max_seq_len={cfg.max_seq_len}  "
          f"grad_checkpoint={cfg.use_grad_checkpoint}")

    # ── Optimiser ──
    # Weight decay on weights only (not biases, norms, embeddings)
    decay   = [p for n, p in _unwrap(model).named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in _unwrap(model).named_parameters() if p.dim() <  2]
    optimizer = optim.AdamW(
        [{"params": decay,   "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps     = steps_per_epoch * args.epochs
    criterion       = nn.CrossEntropyLoss()

    # ── Resume ──
    start_step = 0
    if args.resume:
        ckpt_file = os.path.join(out_dir, "latest.pt")
        if os.path.exists(ckpt_file):
            ckpt = load_checkpoint(ckpt_file, device)
            _unwrap(model).load_state_dict(ckpt["model"])
            start_step = ckpt["step"]
            print(f"Resumed from step {start_step}  (prev val ppl: {ckpt.get('val_ppl', '?'):.2f})")
        else:
            print("No checkpoint found — starting fresh.")

    # ── Loop ──
    print(f"\nTraining: {args.epochs} epochs | eff. batch {args.batch_size * args.grad_accum} | "
          f"~{total_steps:,} optimiser steps")
    aux_enabled = args.w_closure > 0 or args.w_diversity > 0
    if not aux_enabled:
        print("Aux losses disabled (--w_closure 0 --w_diversity 0)")
    print("─" * 72)

    step      = start_step
    raw_step  = 0          # counts individual forward passes (before accum)
    best_ppl  = float("inf")
    t_start   = time.time()
    optimizer.zero_grad()

    for epoch in range(1, args.epochs + 1):
        for inputs, targets in train_loader:
            model.train()
            inputs, targets = inputs.to(device), targets.to(device)

            # Set LR for this step
            lr = get_lr(step, args.warmup_steps, total_steps, args.lr, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward
            if aux_enabled:
                logits, hidden, tensions = _unwrap(model)(inputs, return_all=True)
                loss_ce  = criterion(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
                loss_mcl = manifold_closure_loss(hidden)
                loss_div = tension_diversity_loss(tensions)
                loss     = loss_ce + args.w_closure * loss_mcl + args.w_diversity * loss_div
            else:
                logits   = model(inputs)
                loss_ce  = criterion(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
                loss     = loss_ce
                loss_mcl = loss_div = torch.tensor(0.0)

            (loss / args.grad_accum).backward()
            raw_step += 1

            if raw_step % args.grad_accum != 0:
                continue   # accumulate more gradients

            # Optimiser step
            nn.utils.clip_grad_norm_(_unwrap(model).parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            # ── Logging ──
            if step % args.log_every == 0:
                elapsed  = time.time() - t_start
                ppl      = math.exp(min(loss_ce.item(), 20))
                sps      = step / max(elapsed, 1)  # steps per second
                eta_h    = (total_steps - step) / max(sps * 3600, 1)
                print(
                    f"ep {epoch:2d} | step {step:6d}/{total_steps} | "
                    f"loss {loss.item():.4f} | ppl {ppl:7.1f} | "
                    f"cl {loss_mcl.item():.3f} | div {loss_div.item():.3f} | "
                    f"lr {lr:.1e} | ETA {eta_h:.1f}h"
                )

            # ── Validation ──
            if step % args.eval_every == 0:
                val_ppl = evaluate(model, val_loader, criterion, device)
                marker  = " ← best" if val_ppl < best_ppl else ""
                print(f"  ↳ val ppl {val_ppl:.2f}{marker}")
                if val_ppl < best_ppl:
                    best_ppl = val_ppl
                    save_checkpoint(out_dir, step, model, val_ppl, cfg, tok_path, vars(args))

            # ── Periodic save ──
            elif step % args.save_every == 0:
                val_ppl = evaluate(model, val_loader, criterion, device)
                save_checkpoint(out_dir, step, model, val_ppl, cfg, tok_path, vars(args))

        print(f"── Epoch {epoch} done ──────────────────────────────────────────────────────")

    # Final checkpoint
    val_ppl = evaluate(model, val_loader, criterion, device)
    print(f"\nFinal val ppl: {val_ppl:.2f}  |  Best: {best_ppl:.2f}")
    save_checkpoint(out_dir, step, model, val_ppl, cfg, tok_path, vars(args))
    print(f"\nTo generate text:\n    python generate.py --checkpoint {out_dir}/latest.pt")


if __name__ == "__main__":
    train(get_args())
