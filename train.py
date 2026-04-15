"""
TensionLM training script
==========================
Supports two data modes:
  - Small corpus (WikiText-2/103): downloads, tokenises in RAM, trains directly.
  - Large corpus (FineWeb etc.):   reads pre-tokenised uint16 shards from disk via
                                   memory-map. Prepare shards first with prepare_data.py.

Single GPU:
    python3 train.py --preset large --model tension

Multi-GPU (DDP via torchrun):
    torchrun --nproc_per_node=2 train.py --preset large --model tension

Resume interrupted run:
    python3 train.py --resume --out_dir checkpoints/tension_117m ...
    torchrun --nproc_per_node=2 train.py --resume --out_dir checkpoints/tension_117m ...

Large-scale run from shards with token budget:
    torchrun --nproc_per_node=2 train.py \\
        --data_dir data/fineweb-10B \\
        --train_tokens 10_000_000_000 \\
        --preset large
"""

import argparse
import bisect
import contextlib
import json
import math
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from model import (
    TensionConfig, TensionLM,
    manifold_closure_loss, tension_diversity_loss,
    constraint_consistency_loss, tension_entropy_loss,
)


# ── DDP ───────────────────────────────────────────────────────────────────────

def setup_ddp():
    """
    Auto-detect torchrun via LOCAL_RANK env var.
    Returns (rank, world_size, device, is_main).
    Single-process fallback when not launched with torchrun.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device, True
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size, torch.device(f"cuda:{local_rank}"), rank == 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Train TensionLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    p.add_argument("--model",   default="tension", choices=["tension", "transformer"])
    p.add_argument("--preset",  default=None, choices=["small", "medium", "large", "diagnostic", "350m"])
    # Data — small corpus (in-memory)
    p.add_argument("--text_file",  default=None)
    p.add_argument("--dataset",    default="wikitext-2-raw-v1",
                   choices=["wikitext-2-raw-v1", "wikitext-103-raw-v1"])
    p.add_argument("--max_tokens", default=None, type=int)
    # Data — large corpus (shards from prepare_data.py)
    p.add_argument("--data_dir",   default=None,
                   help="Pre-tokenised shard directory. Overrides --dataset when set.")
    # Token budget (overrides --epochs when set)
    p.add_argument("--train_tokens", default=None, type=int,
                   help="Total tokens to train on. Overrides --epochs.")
    # Logic mixing — catastrophic forgetting prevention
    p.add_argument("--logic_mix", default=0.0, type=float,
                   help="Fraction of each batch drawn from logic stage data (0=off). "
                        "e.g. --logic_mix 0.1 keeps 10%% logic data in stage 3 "
                        "to prevent catastrophic forgetting of constraint structure.")
    p.add_argument("--logic_dir", default=None,
                   help="Path to logic stage data dir (required when --logic_mix > 0)")
    # Tokeniser
    p.add_argument("--vocab_size",  default=2048,  type=int,
                   help="Vocabulary size. Common values: 2048 (small), 32768 (large preset), "
                        "50257 (GPT-2 tokenizer compatible).")
    # Architecture
    p.add_argument("--dim",         default=128,   type=int)
    p.add_argument("--num_layers",  default=4,     type=int)
    p.add_argument("--num_heads",   default=4,     type=int)
    p.add_argument("--window",      default=8,     type=int)
    p.add_argument("--ffn_mult",    default=3,     type=int)
    p.add_argument("--max_seq_len", default=256,   type=int)
    p.add_argument("--dropout",     default=0.10,  type=float)
    p.add_argument("--grad_ckpt",   action="store_true")
    p.add_argument("--no_osc",      action="store_true",
                   help="Disable OscillatoryModulation (ablation)")
    p.add_argument("--rope",        action="store_true",
                   help="Use Rotary Position Embeddings (replaces learned pos + OscillatoryModulation)")
    p.add_argument("--triton",      action="store_true",
                   help="Use fused Triton kernel for tension op (CUDA only)")
    p.add_argument("--global_every", default=0, type=int,
                   help="Interleaved global tension layer every N layers (0=off). "
                        "e.g. --global_every 4 with 12 layers → global at layers 3,7,11")
    # Training
    p.add_argument("--seq_len",       default=64,   type=int)
    p.add_argument("--batch_size",    default=32,   type=int)
    p.add_argument("--grad_accum",    default=2,    type=int)
    p.add_argument("--lr",            default=3e-4, type=float)
    p.add_argument("--min_lr",        default=3e-5, type=float)
    p.add_argument("--transfer_lr",   default=None, type=float,
                   help="Peak LR override for curriculum transfer (use with --resume). "
                        "Lower than --lr to avoid overwriting prior knowledge. Default: lr/3")
    p.add_argument("--warmup_steps",  default=200,  type=int)
    p.add_argument("--epochs",        default=10,   type=int)
    p.add_argument("--weight_decay",  default=0.10, type=float)
    p.add_argument("--clip_grad",     default=1.0,  type=float)
    # Aux losses
    p.add_argument("--w_closure",      default=0.01, type=float)
    p.add_argument("--w_diversity",    default=0.02, type=float)
    # TS-native losses
    p.add_argument("--w_consistency",  default=0.0,  type=float,
                   help="Constraint consistency loss weight (TS-native, 0=off)")
    p.add_argument("--w_entropy",      default=0.0,  type=float,
                   help="Tension entropy regularisation weight (TS-native, 0=off)")
    # I/O
    p.add_argument("--out_dir",    default="checkpoints")
    p.add_argument("--resume",     action="store_true")
    p.add_argument("--log_every",  default=50,   type=int)
    p.add_argument("--eval_every", default=500,  type=int)
    p.add_argument("--save_every", default=1000, type=int)
    p.add_argument("--log_csv",    default=None)
    # WandB
    p.add_argument("--wandb",         action="store_true",
                   help="Log metrics to Weights & Biases")
    p.add_argument("--wandb_project", default="tensionlm")

    # Apply preset as set_defaults so explicit CLI args always win
    pre, _ = p.parse_known_args()
    if pre.preset == "medium":
        p.set_defaults(
            dim=256, num_layers=6, num_heads=4, window=16,
            max_seq_len=512, seq_len=256, batch_size=16,
        )
    elif pre.preset == "large":
        # vocab_size=32768 is the default for the large preset.
        # Use --vocab_size 50257 for GPT-2 tokenizer compatibility (requires retraining tokenizer).
        p.set_defaults(
            dim=768, num_layers=12, num_heads=12, window=64, ffn_mult=3,
            max_seq_len=1024, seq_len=512, batch_size=8, grad_accum=8,
            vocab_size=32768, dataset="wikitext-103-raw-v1", warmup_steps=2000,
            rope=True, no_osc=True, triton=True, global_every=4,
        )
    elif pre.preset == "diagnostic":
        # ~50M params. For inspecting tension heads on logic data before the full run.
        # Triton off so return_tensions always uses the readable unfold path.
        # Full aux losses on — we want to see whether constraint structure forms.
        p.set_defaults(
            dim=512, num_layers=12, num_heads=8, window=64, ffn_mult=3,
            max_seq_len=512, seq_len=256, batch_size=16, grad_accum=4,
            vocab_size=16384, warmup_steps=500,
            rope=True, no_osc=True, triton=False, global_every=4,
            w_consistency=0.1, w_entropy=0.05, w_diversity=0.02, w_closure=0.01,
        )
    elif pre.preset == "350m":
        # ~338M params. The production model.
        # window=256: proofs regularly require attending back 200+ tokens.
        # global_every=3: long-range passes every 3 blocks without O(T²) everywhere.
        p.set_defaults(
            dim=1024, num_layers=24, num_heads=16, window=256, ffn_mult=3,
            max_seq_len=2048, seq_len=1024, batch_size=4, grad_accum=16,
            vocab_size=16384, warmup_steps=2000,
            rope=True, no_osc=True, triton=True, global_every=3,
            w_consistency=0.05, w_entropy=0.02, w_diversity=0.02, w_closure=0.01,
        )

    return p.parse_args()


# ── LR Schedule ───────────────────────────────────────────────────────────────

def get_lr(step: int, warmup: int, total: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * max(step, 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ── Datasets ──────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    """In-memory sliding window — for small corpora (WikiText-2/103)."""
    def __init__(self, token_ids: list, seq_len: int, stride: int | None = None):
        self.data    = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride  = stride if stride is not None else seq_len

    def __len__(self):
        return max(0, (len(self.data) - self.seq_len - 1) // self.stride)

    def __getitem__(self, idx):
        s = idx * self.stride
        c = self.data[s : s + self.seq_len + 1]
        return c[:-1], c[1:]


class ShardedTokenDataset(Dataset):
    """
    Memory-mapped dataset over pre-tokenised uint16 binary shards.
    Each shard is a flat array of token IDs written by prepare_data.py.
    Supports arbitrarily large datasets — only the active shard region is
    paged into RAM by the OS on demand.
    """
    def __init__(self, data_dir: str, seq_len: int, split: str = "train"):
        meta_path = os.path.join(data_dir, "metadata.json")
        meta      = json.load(open(meta_path))
        shards    = [s for s in meta["shards"] if s["split"] == split]
        if not shards:
            raise ValueError(f"No '{split}' shards found in {data_dir}")

        self.seq_len = seq_len
        self.mmaps: list[np.ndarray] = []
        self.shard_seqs: list[int]   = []

        for s in shards:
            arr    = np.memmap(s["path"], dtype=np.uint16, mode="r")
            n_seqs = max(0, (len(arr) - 1) // seq_len)
            self.mmaps.append(arr)
            self.shard_seqs.append(n_seqs)

        # Cumulative sequence offsets for O(log N) shard lookup
        self.cumul = [0]
        for n in self.shard_seqs:
            self.cumul.append(self.cumul[-1] + n)

    def __len__(self):
        return self.cumul[-1]

    def __getitem__(self, global_idx: int):
        si    = bisect.bisect_right(self.cumul, global_idx) - 1
        li    = global_idx - self.cumul[si]
        start = li * self.seq_len
        chunk = self.mmaps[si][start : start + self.seq_len + 1].astype(np.int64)
        x     = torch.from_numpy(chunk[:-1])
        y     = torch.from_numpy(chunk[1:])
        return x, y


# ── Mixed DataLoader (catastrophic forgetting prevention) ────────────────────

class MixedDataLoader:
    """
    Interleaves a primary dataloader with a secondary (logic-stage) loader at a
    specified ratio. When logic_frac=0 it is identical to iterating primary directly.

    On each batch there is a logic_frac probability of yielding a batch from the
    secondary loader instead of the primary. The secondary loader cycles
    indefinitely — it restarts automatically when exhausted.
    """
    def __init__(self, primary: DataLoader, secondary: DataLoader, logic_frac: float):
        self.primary        = primary
        self.secondary      = secondary
        self.logic_frac     = logic_frac
        self._secondary_iter = None

    def __len__(self):
        return len(self.primary)

    def _next_secondary(self):
        if self._secondary_iter is None:
            self._secondary_iter = iter(self.secondary)
        try:
            return next(self._secondary_iter)
        except StopIteration:
            self._secondary_iter = iter(self.secondary)
            return next(self._secondary_iter)

    def __iter__(self):
        import random
        for batch in self.primary:
            if self.logic_frac > 0 and random.random() < self.logic_frac:
                yield self._next_secondary()
            else:
                yield batch


# ── Checkpointing ─────────────────────────────────────────────────────────────

def _unwrap(model) -> TensionLM:
    """Peel torch.compile wrapper, then DDP wrapper, to reach the base module."""
    model = getattr(model, "_orig_mod", model)   # torch.compile
    if isinstance(model, DDP):
        model = model.module                      # DDP
    return model


def save_checkpoint(out_dir, step, model, optimizer, val_ppl, cfg, tok_path, args_dict,
                    max_ckpts: int = 3):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "step":      step,
        "model":     _unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg":       cfg.__dict__,
        "tok_path":  tok_path,
        "val_ppl":   val_ppl,
        "args":      args_dict,
        "arch":      args_dict.get("model", "tension"),
    }
    numbered = os.path.join(out_dir, f"ckpt_{step:07d}.pt")
    latest   = os.path.join(out_dir, "latest.pt")
    torch.save(state, numbered)
    torch.save(state, latest)
    print(f"  Saved → {numbered}  (val ppl {val_ppl:.2f})")

    # Prune old numbered checkpoints, keeping only the most recent max_ckpts
    existing = sorted(Path(out_dir).glob("ckpt_*.pt"))
    for old in existing[:-max_ckpts]:
        old.unlink()


def load_checkpoint(path: str, device: torch.device):
    return torch.load(path, map_location=device, weights_only=False)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches: int = 60) -> float:
    # Use the unwrapped base model to bypass DDP's buffer broadcast.
    # DDP normally broadcasts all buffers (including pos_buf) from rank 0 to
    # all ranks before each forward — but during eval only rank 0 runs forwards
    # while rank 1 waits at the barrier, causing a deadlock + 600s NCCL timeout.
    base = _unwrap(model)
    base.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y   = x.to(device), y.to(device)
        logits = base(x)
        total += criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        n     += 1
    base.train()
    return math.exp(min(total / max(n, 1), 20))


# ── Small-corpus data helpers (in-memory path) ────────────────────────────────

def load_raw_text(args) -> tuple[str, str]:
    if args.text_file:
        text  = Path(args.text_file).read_text(encoding="utf-8", errors="replace")
        split = int(len(text) * 0.95)
        print(f"Loaded {args.text_file}: {len(text):,} chars")
        return text[:split], text[split:]
    print(f"Downloading {args.dataset} via HuggingFace datasets...")
    from datasets import load_dataset
    ds    = load_dataset("wikitext", args.dataset)
    train = "\n".join(t for t in ds["train"]["text"]      if t.strip())
    val   = "\n".join(t for t in ds["validation"]["text"] if t.strip())
    print(f"  train: {len(train):,} chars | val: {len(val):,} chars")
    return train, val


def train_or_load_tokenizer(train_text: str, vocab_size: int, path: str):
    if os.path.exists(path):
        print(f"Loading tokenizer from {path}")
        from tokenizers import Tokenizer
        return Tokenizer.from_file(path)
    print(f"Training BPE tokenizer (vocab={vocab_size})...")
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    tokenizer               = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder       = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        min_frequency=2, show_progress=True,
    )
    tokenizer.train_from_iterator(
        (l for l in train_text.split("\n") if l.strip()), trainer=trainer,
    )
    tokenizer.save(path)
    print(f"Tokenizer saved → {path}")
    return tokenizer


def tokenize(tokenizer, text: str, max_tokens: int | None = None) -> list[int]:
    lines = [l for l in text.split("\n") if l.strip()]
    ids: list[int] = []
    for enc in tokenizer.encode_batch(lines):
        ids.extend(enc.ids)
        if max_tokens and len(ids) >= max_tokens:
            break
    return ids[:max_tokens] if max_tokens else ids


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    rank, world_size, device, is_main = setup_ddp()

    # TF32 matmuls on Ampere+ GPUs — free throughput, negligible accuracy loss
    torch.set_float32_matmul_precision("high")

    out_dir  = args.out_dir
    tok_path = os.path.join(out_dir, "tokenizer.json")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── CSV log (rank 0 only) ──
    csv_file = None
    if args.log_csv and is_main:
        existed  = os.path.exists(args.log_csv)
        csv_file = open(args.log_csv, "a")
        if not existed:
            csv_file.write("step,train_ppl,val_ppl,tokens\n")
            csv_file.flush()

    # ── WandB (rank 0 only) ──
    use_wandb = False
    if args.wandb and is_main:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
            use_wandb = True
            print("WandB: enabled")
        except ImportError:
            print("WandB: skipped (pip install wandb to enable)")

    # ── Data ──
    if args.data_dir:
        # Large corpus — memory-mapped shards
        if is_main:
            print(f"Loading shards from {args.data_dir}")
        meta      = json.load(open(os.path.join(args.data_dir, "metadata.json")))
        tok_path  = meta["tokenizer"]                 # use shard's tokenizer
        vocab_size = meta["vocab_size"]
        train_ds  = ShardedTokenDataset(args.data_dir, args.seq_len, split="train")
        val_ds    = ShardedTokenDataset(args.data_dir, args.seq_len, split="val")
        if is_main:
            print(f"  {len(train_ds):,} train seqs | {len(val_ds):,} val seqs")
    else:
        # Small corpus — in-memory.
        # Rank 0 downloads and trains the tokenizer first, then all ranks tokenize.
        # HF dataset download is cached on disk so subsequent ranks are fast.
        if is_main:
            train_text, val_text = load_raw_text(args)
            tokenizer  = train_or_load_tokenizer(train_text, args.vocab_size, tok_path)
            vocab_size = tokenizer.get_vocab_size()
            print(f"Vocab size: {vocab_size}")
        if world_size > 1:
            dist.barrier()  # wait for rank 0 to save tokenizer before others load it
        if not is_main:
            train_text, val_text = load_raw_text(args)
            from tokenizers import Tokenizer
            tokenizer  = Tokenizer.from_file(tok_path)
            vocab_size = tokenizer.get_vocab_size()
        t0        = time.time()
        train_ids = tokenize(tokenizer, train_text, args.max_tokens)
        val_ids   = tokenize(tokenizer, val_text,
                             max_tokens=min(200_000, len(val_text) // 4))
        if is_main:
            print(f"  train {len(train_ids):,} tokens | "
                  f"val {len(val_ids):,} tokens  ({time.time()-t0:.1f}s)")
        train_ds = TokenDataset(train_ids, args.seq_len)
        val_ds   = TokenDataset(val_ids, args.seq_len, stride=args.seq_len // 2)

    # ── DataLoaders ──
    n_workers = min(12, multiprocessing.cpu_count() - 2) if device.type == "cuda" else 0
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=n_workers, pin_memory=(device.type == "cuda"), drop_last=True,
        persistent_workers=(n_workers > 0),
        prefetch_factor=4 if n_workers > 0 else None,
    )
    # Val: no distributed sampler — rank 0 evaluates on full val set
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=n_workers,
        pin_memory=(device.type == "cuda"), drop_last=False,
        persistent_workers=(n_workers > 0),
        prefetch_factor=4 if n_workers > 0 else None,
    )
    if is_main:
        print(f"  {len(train_ds):,} train sequences | {len(train_loader)} batches/epoch")

    # ── Logic mixing (catastrophic forgetting prevention) ──
    if args.logic_mix > 0:
        if not args.logic_dir:
            raise ValueError("--logic_dir is required when --logic_mix > 0")
        logic_ds = ShardedTokenDataset(args.logic_dir, args.seq_len, split="train")
        logic_loader = DataLoader(
            logic_ds, batch_size=args.batch_size,
            shuffle=True, num_workers=min(4, n_workers), drop_last=True,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(n_workers > 0),
            prefetch_factor=2 if n_workers > 0 else None,
        )
        train_loader = MixedDataLoader(train_loader, logic_loader, args.logic_mix)
        if is_main:
            print(f"Logic mix: {args.logic_mix:.0%} of batches from {args.logic_dir}")

    # ── Model ──
    cfg = TensionConfig(
        vocab_size          = vocab_size,
        dim                 = args.dim,
        num_layers          = args.num_layers,
        num_heads           = args.num_heads,
        window              = args.window,
        ffn_mult            = args.ffn_mult,
        max_seq_len         = args.max_seq_len,
        dropout             = args.dropout,
        use_grad_checkpoint = args.grad_ckpt,
        use_oscillation     = not args.no_osc,
        use_rope            = args.rope,
        use_triton          = args.triton,
        global_every        = args.global_every,
    )
    if args.model == "transformer":
        from baseline import TransformerLM
        model = TransformerLM(cfg).to(device)
    else:
        model = TensionLM(cfg).to(device)

    # Compile BEFORE DDP wrap — torch.compile sees the base model, not the DDP wrapper.
    if device.type == "cuda":
        try:
            model = torch.compile(
                model,
                dynamic=False,
                mode="max-autotune",  # full autotune including CUDAGraphs
            )
            if is_main:
                print("torch.compile: max-autotune (CUDAGraphs enabled)")
        except Exception as e:
            if is_main:
                print(f"torch.compile: skipped ({e})")

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[device.index],
            find_unused_parameters=False,
            broadcast_buffers=False,
            static_graph=True,  # safe because architecture doesn't change during training
        )
        if is_main:
            print(f"DDP: {world_size} GPUs (static_graph=True)")
    else:
        if is_main and device.type == "cuda":
            n_gpus = torch.cuda.device_count()
            print(f"Single GPU mode ({n_gpus} GPU{'s' if n_gpus>1 else ''} available — "
                  f"use torchrun to use all)")

    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else None
    if is_main and use_amp:
        print("Mixed precision: bf16 autocast enabled")

    if is_main:
        np_count = _unwrap(model).num_params
        print(f"\nModel: {args.model}  |  {np_count:,} parameters")
        print(f"  dim={cfg.dim}  layers={cfg.num_layers}  heads={cfg.num_heads}  "
              f"window={cfg.window}  ffn_mult={cfg.ffn_mult}  dropout={cfg.dropout}")
        print(f"  vocab={cfg.vocab_size}  max_seq_len={cfg.max_seq_len}  "
              f"grad_checkpoint={cfg.use_grad_checkpoint}")

    # ── Optimiser ──
    decay   = [p for n, p in _unwrap(model).named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in _unwrap(model).named_parameters() if p.dim() <  2]
    optimizer = optim.AdamW(
        [{"params": decay,   "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    # ── Total steps ──
    # Token budget overrides epoch count when set.
    tokens_per_step = args.seq_len * args.batch_size * args.grad_accum * world_size
    if args.train_tokens:
        total_steps = args.train_tokens // tokens_per_step
        max_epochs  = 999999  # effectively infinite epochs; token budget is the limit
        if is_main:
            print(f"Token budget: {args.train_tokens/1e9:.1f}B tokens  "
                  f"→ {total_steps:,} steps")
    else:
        steps_per_epoch = len(train_loader) // args.grad_accum
        total_steps     = steps_per_epoch * args.epochs
        max_epochs      = args.epochs

    criterion = nn.CrossEntropyLoss()

    # ── Resume ──
    start_step    = 0
    tokens_seen   = 0
    if args.resume:
        ckpt_file = os.path.join(out_dir, "latest.pt")
        if os.path.exists(ckpt_file):
            ckpt = load_checkpoint(ckpt_file, device)
            _unwrap(model).load_state_dict(ckpt["model"])
            # Do NOT restore optimizer state on curriculum transfer — we want a
            # fresh optimiser so the new LR schedule takes effect cleanly.
            start_step  = 0   # reset step counter so LR schedule runs from scratch
            tokens_seen = 0
            # Apply transfer LR: default to lr/3 if not explicitly set
            if args.transfer_lr is not None:
                args.lr     = args.transfer_lr
                args.min_lr = args.transfer_lr / 10
            else:
                args.lr     = args.lr / 3
                args.min_lr = args.min_lr / 3
            if is_main:
                print(f"Curriculum transfer from step {ckpt['step']}  "
                      f"(prev val ppl: {ckpt.get('val_ppl', '?'):.2f})  "
                      f"peak LR → {args.lr:.2e}")
        elif is_main:
            print("No checkpoint found — starting fresh.")

    # ── Training loop ──
    aux_enabled = args.w_closure > 0 or args.w_diversity > 0
    if is_main:
        if args.train_tokens:
            print(f"\nTraining: {args.train_tokens/1e9:.1f}B token budget | "
                  f"eff. batch {args.batch_size * args.grad_accum * world_size} | "
                  f"~{total_steps:,} steps")
        else:
            print(f"\nTraining: {args.epochs} epochs | "
                  f"eff. batch {args.batch_size * args.grad_accum * world_size} | "
                  f"~{total_steps:,} steps")
        if not aux_enabled:
            print("Aux losses: disabled")
        print("─" * 72)

    step          = start_step
    raw_step      = 0
    best_ppl      = float("inf")
    t_start       = time.time()
    t_first_step  = None   # set after first optimizer step — excludes compile warmup
    done          = False
    optimizer.zero_grad()

    for epoch in range(1, max_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for inputs, targets in train_loader:
            model.train()
            inputs, targets = inputs.to(device), targets.to(device)

            lr = get_lr(step, args.warmup_steps, total_steps, args.lr, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with (amp_ctx if amp_ctx else contextlib.nullcontext()):
                if aux_enabled:
                    logits, hidden, tensions = model(inputs, return_all=True)
                    loss_ce   = criterion(
                        logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
                    loss_mcl  = manifold_closure_loss(hidden)
                    loss_div  = tension_diversity_loss(tensions, _unwrap(model).window_groups)
                    loss_cons = constraint_consistency_loss(tensions)
                    loss_ent  = tension_entropy_loss(tensions)
                    loss      = (loss_ce
                                 + args.w_closure      * loss_mcl
                                 + args.w_diversity    * loss_div
                                 + args.w_consistency  * loss_cons
                                 + args.w_entropy      * loss_ent)
                else:
                    logits   = model(inputs)
                    loss_ce  = criterion(
                        logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
                    loss     = loss_ce
                    loss_mcl = loss_div = loss_cons = loss_ent = torch.tensor(0.0)

            (loss / args.grad_accum).backward()
            raw_step    += 1
            tokens_seen += inputs.numel() * world_size

            if raw_step % args.grad_accum != 0:
                continue

            nn.utils.clip_grad_norm_(_unwrap(model).parameters(), args.clip_grad)
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            if t_first_step is None:
                t_first_step = time.time()

            # ── Logging (rank 0) ──
            if is_main and step % args.log_every == 0:
                elapsed = time.time() - t_first_step
                steps_so_far = step - start_step
                ppl     = math.exp(min(loss_ce.item(), 20))
                sps     = steps_so_far / max(elapsed, 1)
                eta_h   = (total_steps - step) / max(sps * 3600, 1)
                cons_str = f" | cons {loss_cons.item():.3f} | ent {loss_ent.item():.3f}" \
                           if (args.w_consistency > 0 or args.w_entropy > 0) else ""
                print(
                    f"ep {epoch:2d} | step {step:6d}/{total_steps} | "
                    f"loss {loss.item():.4f} | ppl {ppl:7.1f} | "
                    f"cl {loss_mcl.item():.3f} | div {loss_div.item():.3f}"
                    f"{cons_str} | "
                    f"lr {lr:.1e} | tok {tokens_seen/1e9:.2f}B | ETA {eta_h:.1f}h"
                )
                if csv_file:
                    csv_file.write(f"{step},{ppl:.4f},,{tokens_seen}\n")
                    csv_file.flush()
                if use_wandb:
                    import wandb
                    wandb.log({"train_ppl": ppl, "loss": loss.item(),
                               "lr": lr, "tokens": tokens_seen}, step=step)

            # ── Validation + checkpoint (rank 0, others barrier) ──
            if step % args.eval_every == 0:
                if is_main:
                    val_ppl = evaluate(model, val_loader, criterion, device)
                    marker  = " ← best" if val_ppl < best_ppl else ""
                    print(f"  ↳ val ppl {val_ppl:.2f}{marker}")
                    if csv_file:
                        csv_file.write(f"{step},,{val_ppl:.4f},{tokens_seen}\n")
                        csv_file.flush()
                    if use_wandb:
                        import wandb
                        wandb.log({"val_ppl": val_ppl}, step=step)
                    if val_ppl < best_ppl:
                        best_ppl = val_ppl
                        save_checkpoint(out_dir, step, model, optimizer,
                                        val_ppl, cfg, tok_path, vars(args))
                if world_size > 1:
                    dist.barrier()

            # ── Periodic save ──
            elif step % args.save_every == 0:
                if is_main:
                    val_ppl = evaluate(model, val_loader, criterion, device)
                    save_checkpoint(out_dir, step, model, optimizer,
                                    val_ppl, cfg, tok_path, vars(args))
                if world_size > 1:
                    dist.barrier()

            # ── Token budget check ──
            if args.train_tokens and tokens_seen >= args.train_tokens:
                done = True
                break

        if is_main:
            print(f"── Epoch {epoch} done "
                  f"({tokens_seen/1e9:.2f}B tokens seen) "
                  f"───────────────────────────────────────")
        if done or (not args.train_tokens and epoch >= args.epochs):
            break

    # ── Final checkpoint ──
    if is_main:
        val_ppl = evaluate(model, val_loader, criterion, device)
        print(f"\nFinal val ppl: {val_ppl:.2f}  |  Best: {best_ppl:.2f}")
        save_checkpoint(out_dir, step, model, optimizer,
                        val_ppl, cfg, tok_path, vars(args))
        if csv_file:
            csv_file.write(f"{step},,{val_ppl:.4f},{tokens_seen}\n")
            csv_file.close()
        if use_wandb:
            import wandb
            wandb.finish()
        print(f"\nTo generate text:\n    python generate.py --checkpoint {out_dir}/latest.pt")

    if world_size > 1:
        dist.destroy_process_group()


# ── Presets ───────────────────────────────────────────────────────────────────

def apply_preset(args):
    # Preset values are now injected via set_defaults in get_args() so CLI args
    # always take precedence. This function just prints the summary.
    if args.preset:
        print(f"Preset '{args.preset}' applied: dim={args.dim}  "
              f"layers={args.num_layers}  window={args.window}  vocab={args.vocab_size}")


if __name__ == "__main__":
    args = get_args()
    apply_preset(args)
    train(args)
