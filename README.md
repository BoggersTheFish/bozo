# TensionLM

A language model built on **CausalTensionGraphs** instead of softmax attention.

---

## The idea

Standard transformers use *softmax attention* — every position competes with every other for a fixed budget of weight that must sum to 1. If one token pulls harder, the others pull less. It's zero-sum.

TensionLM replaces this with *tension* — a sigmoid score computed independently for each token pair:

```
tau[t, w] = sigmoid( dot(q_t, k_{t-w-1}) / √head_dim )
output[t] = Σ_w  tau[t, w] * v_{t-w-1}
```

There is no competition. A token can be pulled hard by all its neighbors simultaneously, or barely pulled by any of them. The model learns which relationships matter without forcing a zero-sum tradeoff.

---

## Architecture

```
Embedding + LearnedPositional
  └─ × N  TensionBlock
          ├─ RMSNorm (pre-norm)
          ├─ MultiHeadCausalTensionLayer   ← the core
          ├─ OscillatoryModulation         ← learned wave-based positional signal
          └─ SwiGLU FFN
RMSNorm → LM head (weight-tied to embedding)
```

**`MultiHeadCausalTensionLayer`**
H heads, causal window of W tokens. Each head independently scores how strongly the current token is drawn toward each past neighbor using a scaled dot product, then aggregates neighbor values weighted by those tensions. Keys and values are projected jointly and gathered in a single vectorised `unfold` operation (no Python loops over T or W).

**`OscillatoryModulation`**
After each tension layer, hidden states are multiplicatively modulated by learned sinusoids. Each channel gets its own frequency and phase — free parameters the model tunes during training. Unlike fixed sinusoidal encodings, these waves are learned end-to-end and can specialise per layer.

**`SwiGLU FFN`**
```
out = proj( silu(gate(x)) * val(x) )
```
Gated activation as used in LLaMA/PaLM. More expressive than ReLU FFN at the same parameter count.

---

## Training signal

All three losses share a single computation graph:

| Loss | Weight | Purpose |
|------|--------|---------|
| `CrossEntropy` | 1.0 | Next-token prediction |
| `ManifoldClosureLoss` | 0.05 | First and last hidden states should be coherent |
| `TensionDiversityLoss` | 0.02 | Heads should spread tension, not collapse onto one position |

---

## Comparison: TensionLM vs Transformer

Both models are trained on **WikiText-2** with identical config (dim=128, 4 layers, 4 heads, BPE vocab=2048). The only difference is the mechanism: windowed sigmoid tension vs full softmax attention.

| | TensionLM | Transformer |
|---|---|---|
| Context mechanism | Sigmoid tension (window W) | Softmax attention (full O(T²)) |
| Position encoding | Learned embeddings + trainable per-channel waves | Learned embeddings |
| Weight competition | No — each pair is independent | Yes — sums to 1 |
| Parameters (~1M config) | ~1.15M | ~1.15M |

*Results below are from the 10-epoch WikiText-2 run.*

| Metric | TensionLM | Transformer |
|--------|-----------|-------------|
| Best val PPL | **57.7** | **57.8** |
| Final train PPL | 56.9 | 266.6 |

Plot: `results/comparison.png`
> **Result:** TensionLM wins by 0.1 PPL points (TensionLM 57.7 vs Transformer 57.8 on WikiText-2 val).

---

## 117M scale results (WikiText-103)

Trained for 3 epochs (~0.34B tokens) on WikiText-103 with the `--preset large` config. 2× RTX 4090 D, DDP, bf16, torch.compile.

| Metric | Value |
|--------|-------|
| Final val PPL | **32.01** |
| Best val PPL | **32.33** (step 5000) |
| Parameters | 118M |
| Training time | ~9h (2× RTX 4090 D) |

Model available on HuggingFace: [BoggersTheFish/TensionLM-117M](https://huggingface.co/BoggersTheFish/TensionLM-117M)

Sample output at val PPL 32:

> *"The war began in March, when an army of 42,000 men under his command became part of a new militia unit formed along the River Plate... This force was composed of two battalions under General Sir Thomas Fairfax and Major Robert Cornwallis, along with the British left wing."*


---

## Quick start

```bash
pip install torch tokenizers datasets
python3 train.py                          # default: TensionLM, WikiText-2, 10 epochs
python3 train.py --model transformer      # baseline for comparison
```

### Size presets

```bash
# Small (~1.1M params) — default, trains overnight on CPU
python3 train.py --preset small

# Medium (~5M params) — Milestone 1 scale test
python3 train.py --preset medium

# Large (~117M params) — GPU required
python3 train.py --preset large --model tension
```

### Generate text

```bash
python3 generate.py --checkpoint checkpoints/tension/latest.pt --prompt "The cat"
python3 generate.py --checkpoint checkpoints/tension/latest.pt  # interactive
```

### Evaluate perplexity

```bash
python3 eval.py --checkpoint checkpoints/tension/latest.pt
python3 eval.py --checkpoint checkpoints/tension/latest.pt --dataset wikitext-103-raw-v1
```

### Compare two runs

```bash
python3 compare.py \
  --tension     logs/tension.csv \
  --transformer logs/transformer.csv \
  --out         results/comparison.png
```

---

## Config

The `--preset` flag expands to these concrete hyperparameters:

| Preset | dim | layers | heads | window | vocab | ~params | Hardware |
|--------|-----|--------|-------|--------|-------|---------|----------|
| small  | 128 | 4      | 4     | 8      | 2048  | 1.1M    | CPU      |
| medium | 256 | 6      | 4     | 16     | 2048  | 5M      | CPU      |
| large  | 768 | 12     | 12    | 64     | 32768 | 117M    | GPU      |

All presets can be further tuned with individual flags (`--dim`, `--window`, etc.).

---

## GPU training (Milestone 2)

Single GPU:
```bash
python3 train.py --preset large --model tension \
  --out_dir checkpoints/tension_117m \
  --log_csv logs/tension_117m.csv \
  --epochs 3
```

Multi-GPU (DDP via torchrun):
```bash
torchrun --nproc_per_node=2 train.py --preset large --model tension \
  --out_dir checkpoints/tension_117m \
  --log_csv logs/tension_117m.csv \
  --epochs 3
```

The training loop auto-detects CUDA and enables:
- `bf16` mixed precision (`torch.autocast`)
- `torch.compile()` (Inductor backend)
- `pin_memory=True` DataLoader
- DDP (DistributedDataParallel) via `torchrun` when multiple GPUs are available

Estimated GPU training time at 117M params, WikiText-103, 3 epochs:

| Setup | Est. time |
|-------|-----------|
| 1× RTX 4090 | ~30h |
| 2× RTX 4090 (DDP) | ~17h |
| 1× A100 | ~10h |

### Large-scale training (FineWeb)

For billion-token runs, pre-tokenise into binary shards first:

```bash
# Tokenise FineWeb 10B into shards
python3 prepare_data.py \
  --dataset fineweb-10B \
  --out_dir data/fineweb-10B \
  --tokenizer checkpoints/tension_117m/tokenizer.json

# Train from shards with a token budget
torchrun --nproc_per_node=2 train.py \
  --data_dir data/fineweb-10B \
  --train_tokens 10_000_000_000 \
  --out_dir checkpoints/tension_117m_fw \
  --log_csv logs/tension_117m_fw.csv
```

---

## File map

| File | Purpose |
|------|---------|
| `model.py` | TensionLM architecture + aux losses + generation |
| `baseline.py` | Baseline transformer (same API, softmax attention) |
| `train.py` | Training pipeline — single GPU, DDP, token budget, WandB |
| `prepare_data.py` | Stream + tokenise large datasets into binary shards |
| `compare.py` | Plot loss curves from two CSV logs side by side |
| `eval.py` | Perplexity evaluation on any HuggingFace dataset |
| `fill_results.py` | Auto-fill README results table from CSV logs |
| `generate.py` | Inference CLI with sampling controls |
| `upload_hf.py` | Upload checkpoint and tokenizer to HuggingFace Hub |

---

## Tension field visualisation

What the model learns — which past tokens each new token is drawn toward:

```
Tension field — layer 2 — 'the cat sat on the mat'
  the           (no causal history)
  cat          ← the:0.43
  sat          ← cat:0.52 | the:0.45
  on           ← sat:0.53 | cat:0.19 | the:0.15
  the          ← on:0.54  | sat:0.38 | cat:0.31 | the:0.32
  mat          ← the:0.61 | on:0.41  | sat:0.52 | cat:0.60 | the:0.60
```

`mat` pulls nearly equally from `the`, `sat`, and `cat` — the whole predicate phrase, not just the adjacent word.

```bash
python3 generate.py --checkpoint checkpoints/tension/latest.pt \
  --prompt "the cat sat on the mat" --show_tension
```

---

## Open questions

1. **Does it scale?** The windowed context (window=64 at GPT-2 scale) means each layer only sees 64 tokens back. Long-range dependencies must emerge through depth. Does 12 layers of local tension produce comparable long-range coherence to full attention?

2. **Optimal window size?** Larger window = more compute but better context. Needs ablation at 5M scale.

3. **Is OscillatoryModulation carrying its weight?** It adds parameters and a sinusoidal positional signal. Worth ablating at 5M scale.
