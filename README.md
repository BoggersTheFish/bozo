# TensionLM

A language model built on **sigmoid tension** instead of softmax attention — an empirical implementation of the Thinking System (TS) theory of computation.

**[117M model on HuggingFace](https://huggingface.co/BoggersTheFish/TensionLM-117M)** | **[GitHub](https://github.com/BoggersTheFish/bozo)**

---

## The theory behind it

**Thinking System (TS)** is a framework for understanding computation — and cognition — as constraint relaxation over a graph of interdependent states.

The core claim: any system that processes information can be described as a graph where:
- **Nodes** are states (tokens, features, concepts)
- **Edges** are constraints (relationships, dependencies)
- **τ (tau)** is unresolved tension — the degree to which a constraint is active but not yet satisfied
- **Computation** is the system relaxing toward equilibrium under those constraints

Under TS, learning is the graph accumulating edges from experience. Inference is the graph finding a low-tension state given an input. Representations are not vectors — they are constraint patterns. A concept is defined not by what it *is* but by what it *constrains*.

**Why softmax attention contradicts this.**

Standard transformers use softmax attention — a zero-sum weight distribution over positions. If token A pulls hard, tokens B and C are forced to pull less. This is a competitive mechanism: information fights for a fixed budget. Under TS this is wrong. Real constraints don't cancel each other out. A concept can be simultaneously constrained by multiple past concepts — the constraints add, they don't compete.

**Why sigmoid tension is the right implementation.**

TensionLM replaces softmax with sigmoid tension:

```
τ[t, w] = sigmoid( dot(Q[t], K[t-w]) / √head_dim )
output[t] = Σ_w  τ[t, w] · V[t-w]  /  valid_count
```

Each token pair is scored independently. τ[t, w] measures how strongly token t is constrained by token t-w. There is no global normalisation — a high score at one position does not suppress any other. The tension field is the literal constraint graph the model has learned. It is inspectable, interpretable, and directly motivated by TS theory.

This is not an architectural trick. It is what TS predicts the mechanism should look like.

---

## Results

### Small scale — TensionLM vs Transformer (WikiText-2)

Both models: dim=128, 4 layers, 4 heads, BPE vocab=2048, ~1.15M parameters. Trained 10 epochs on WikiText-2. Identical config — only the mechanism differs.

| Metric | TensionLM | Transformer |
|--------|-----------|-------------|
| Best val PPL | **57.7** | 57.8 |
| Mechanism | Sigmoid tension (windowed) | Softmax attention (full O(T²)) |

TensionLM matches the transformer at identical parameter count with a fundamentally different computation.

### 117M scale — WikiText-103

Trained ~0.34B tokens on WikiText-103. Hardware: 2× RTX 4090, DDP, bf16, torch.compile.

| Metric | Value |
|--------|-------|
| Final val PPL | **32.01** |
| Parameters | 117M |
| Training time | ~9h |

### 117M scale — FineWeb-10B (in progress)

Currently training on 11.1B tokens from FineWeb. Early checkpoints tracking toward val PPL < 30.

---

## What the tension field actually shows

The tension field is inspectable — you can read what the model has learned directly from τ values.

**Composability:** In softmax attention, if a token attends strongly to position A, it attends less to position B — they compete. In TensionLM, a token can be simultaneously pulled by all its relevant predecessors at full strength. In our 117M model (layer 12, head 0), the token "title" in "Manchester United won the Premier League title" produces:

```
Head 0:  Manchester:0.95  United:0.95  League:0.86  Premier:0.71
```

τ=0.95 on two tokens simultaneously. In softmax this is impossible — weights sum to 1 so two positions at 0.95 would require negative weights elsewhere. In TensionLM it is the natural result: the token "title" is genuinely co-constrained by both "Manchester" and "United."

**Head specialisation:** Different heads specialise spontaneously into different constraint roles — some track local syntactic structure (recent tokens, high variance), some track long-range semantic content, some are uniformly diffuse. This is TS in action: each head discovers a different class of constraint present in the data.

**Empirical TS validation:** Coherent text produces a 25% higher mean τ and 60% more active edges than random word salad on the same vocabulary. TS predicts that structured input creates denser, more consistent constraint graphs. The tension field confirms this.

You can inspect the tension field yourself:

```bash
# Per-head heatmap for a sentence
python visualise.py --checkpoint checkpoints/tension_fw10b/latest.pt \
    --mode heatmap \
    --text "The history of artificial intelligence" \
    --out tension_heatmap.png

# Which past tokens pull hardest on a specific token
python visualise.py --checkpoint checkpoints/tension_fw10b/latest.pt \
    --mode token \
    --text "Manchester United won the Premier League title" \
    --token_idx -1

# How tension evolves across layers
python visualise.py --checkpoint checkpoints/tension_fw10b/latest.pt \
    --mode layers \
    --text "Scientists discovered that water conducts electricity" \
    --token_idx -1

# Head specialisation statistics over a sample
python visualise.py --checkpoint checkpoints/tension_fw10b/latest.pt \
    --mode stats \
    --sample_file data/fineweb-10B/val_0000.bin \
    --sample_size 200
```

---

## Architecture

```
Embedding
  └─ × N  TensionBlock
          ├─ RMSNorm (pre-norm)
          ├─ MultiHeadCausalTensionLayer   ← the constraint mechanism
          ├─ OscillatoryModulation         ← learned per-channel sinusoidal signal
          └─ SwiGLU FFN
RMSNorm → LM head (weight-tied to embedding)
```

**`MultiHeadCausalTensionLayer`**

H heads, causal window of W tokens. Each head independently computes τ[t, w] = sigmoid(dot(Q[t], K[t-w]) / scale) for all w in [0, W), then aggregates V[t-w] weighted by τ. No position competes with any other — all constraints are satisfied simultaneously. The output is mean-normalised by valid window length to keep magnitude stable.

At training scale, this is implemented as a fused Triton kernel that avoids materialising the B×T×H×W×HD intermediate tensor — a 64× memory reduction vs the naive unfold approach.

**`OscillatoryModulation`**

After each tension layer, hidden states are multiplicatively modulated by learned sinusoids. Each channel gets its own frequency and phase, tuned end-to-end. Encodes a form of temporal structure without fixed positional embeddings.

**`SwiGLU FFN`**

```
out = proj( silu(gate(x)) * val(x) )
```

Gated activation as in LLaMA/PaLM. More expressive than ReLU FFN at the same parameter count.

---

## Training signal

Three losses, one computation graph:

| Loss | Weight | Purpose |
|------|--------|---------|
| `CrossEntropy` | 1.0 | Next-token prediction |
| `ManifoldClosureLoss` | 0.05 | First and last hidden states stay coherent |
| `TensionDiversityLoss` | 0.02 | Heads spread tension rather than collapsing onto one position |

`TensionDiversityLoss` directly enforces the TS prediction that a good constraint graph uses diverse edge types — not just one mode of constraint.

---

## Quick start

```bash
pip install torch tokenizers datasets triton
python3 train.py                          # TensionLM, WikiText-2, 10 epochs
python3 train.py --model transformer      # baseline transformer for comparison
```

### Size presets

| Preset | dim | layers | heads | window | vocab | ~params | Hardware |
|--------|-----|--------|-------|--------|-------|---------|----------|
| small  | 128 | 4      | 4     | 8      | 2048  | 1.1M    | CPU      |
| medium | 256 | 6      | 4     | 16     | 2048  | 5M      | CPU/GPU  |
| large  | 768 | 12     | 12    | 64     | 32768 | 117M    | GPU      |

```bash
python3 train.py --preset small
python3 train.py --preset medium
python3 train.py --preset large --model tension
```

All presets can be overridden with individual flags (`--dim`, `--window`, etc.).

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

### Multi-GPU training (DDP)

```bash
torchrun --nproc_per_node=2 train.py --preset large --model tension \
  --out_dir checkpoints/tension_117m \
  --log_csv logs/tension_117m.csv
```

The training loop auto-detects CUDA and enables bf16 mixed precision, torch.compile (Inductor), gradient checkpointing, and DDP when launched with torchrun.

### Large-scale training (FineWeb)

```bash
# Pre-tokenize into binary shards first
python3 prepare_data.py \
  --dataset fineweb-10B \
  --out_dir data/fineweb-10B \
  --tokenizer checkpoints/tension_117m/tokenizer.json

# Train on shards
torchrun --nproc_per_node=2 train.py \
  --data_dir data/fineweb-10B \
  --train_tokens 10_000_000_000 \
  --out_dir checkpoints/tension_fw10b \
  --log_csv logs/tension_fw10b.csv
```

---

## File map

| File | Purpose |
|------|---------|
| `model.py` | TensionLM architecture, aux losses, generation |
| `baseline.py` | Baseline transformer (identical API, softmax attention) |
| `train.py` | Training pipeline — single GPU, DDP, token budget |
| `prepare_data.py` | Stream + tokenize large datasets into binary shards |
| `eval.py` | Perplexity evaluation on any HuggingFace dataset |
| `generate.py` | Inference CLI with sampling controls |
| `visualise.py` | Tension field inspection — heatmap, token, layers, stats modes |
| `compare.py` | Plot loss curves from two CSV logs side by side |
| `upload_hf.py` | Upload checkpoint and tokenizer to HuggingFace Hub |
| `triton_tension/` | Fused Triton kernels (fwd + bwd) for the tension mechanism |

---

## Open questions

These are the live research questions, not gaps to paper over:

1. **Does window depth substitute for window breadth?** W=64 means direct context is 64 tokens per layer. Long-range dependencies propagate through stacked layers. Does this break at document scale, or does depth compensate? Needs a W=512 vs W=64 ablation at 117M scale.

2. **Tau-mass normalisation.** Current normalisation divides by valid window count — every position counts equally whether τ=0.01 or τ=0.99. Tau-mass normalisation (divide by `Σ τ` instead) would weight the message by actual constraint strength. May improve stability at large windows.

3. **Does OscillatoryModulation earn its parameters?** It adds sinusoidal positional structure per channel. Needs ablation to confirm it actually helps vs learned absolute positional embeddings alone.

4. **Scaling law exponent.** Is TensionLM's loss-vs-compute curve competitive with transformers? This determines whether it's worth scaling to 1B+. Needs a proper Chinchilla-style sweep at 5–350M.
