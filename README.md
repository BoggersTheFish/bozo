# TensionLM

A language model built on **CausalTensionGraphs** instead of self-attention.

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
  └─ × 4  TensionBlock
          ├─ LayerNorm (pre-norm)
          ├─ MultiHeadCausalTensionLayer   ← the core
          ├─ OscillatoryModulation         ← learned wave-based positional signal
          └─ SwiGLU FFN
LayerNorm → LM head (weight-tied to embedding)
```

**`MultiHeadCausalTensionLayer`**
4 heads, causal window of 8 tokens. Each head independently scores how strongly the current token is drawn toward each past neighbor using a scaled dot product, then aggregates neighbor values weighted by those tensions. Keys and values are projected jointly and gathered in a single vectorised `unfold` operation (no Python loops over T or W).

**`OscillatoryModulation`**
After each tension layer, hidden states are multiplicatively modulated by learned sinusoids. Each of the 64 channels gets its own frequency and phase — free parameters the model tunes during training. Unlike fixed Transformer sinusoidal encodings, these waves are learned end-to-end and can specialise per layer.

**`SwiGLU FFN`**
```
out = proj( silu(gate(x)) * val(x) )
```
Gated activation as used in LLaMA/PaLM. More expressive than ReLU FFN at the same parameter count.

---

## Training signal

All three losses share a single computation graph — all backpropagate correctly:

| Loss | Weight | Purpose |
|------|--------|---------|
| `CrossEntropy` | 1.0 | Next-token prediction |
| `ManifoldClosureLoss` | 0.05 | First and last hidden states should be coherent |
| `TensionDiversityLoss` | 0.02 | Heads should spread tension, not collapse onto one position |

**Full-batch training** — all sequences stacked into a single tensor and processed in one forward pass per epoch. On CPU, per-call Python overhead dominates for small batches; batching everything together gives ~11x speedup over per-sample loops.

---

## Results (50 epochs, toy corpus)

```
Epoch  10  CE: 3.30   (random baseline: 4.87)
Epoch  20  CE: 1.77
Epoch  30  CE: 0.74
Epoch  40  CE: 0.46
Epoch  50  CE: 0.41
```

**Tension field** — what the model learned after training:

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

---

## Quick start

```bash
pip install torch
python tension_lm.py
```

Training runs on CPU. Default is 50 epochs (~7 min on a standard laptop). Increase `EPOCHS` in the config block for better generation quality.

To use in your own code:

```python
from tension_lm import TensionLM, generate, show_tensions, word_to_idx, idx_to_word

# model is trained when the script is imported/run
result = generate(model, "the cat", max_new=30, temp=0.85, top_p=0.92)
show_tensions(model, "the cat sat on the mat", layer=0)
```

---

## Config

```python
SEQ_LEN    = 16   # training sequence length
STRIDE     = 3    # sliding window stride over corpus
DIM        = 64   # model dimension
NUM_LAYERS = 4    # number of TensionBlocks
WINDOW     = 8    # causal look-back window per layer
NUM_HEADS  = 4    # attention heads per layer
EPOCHS     = 50   # ~8s/epoch on CPU; increase to 150+ for better quality
```

---

## TensionLM vs Transformer

| | Transformer | TensionLM |
|---|---|---|
| Context mechanism | Softmax (global, all-to-all) | Sigmoid tension (local window) |
| Position encoding | Fixed sinusoids or learned | Learned embeddings + trainable per-channel waves |
| Weight competition | Yes — sums to 1 | No — each pair is independent |
| Context range | Full sequence | Configurable window |

The local window is intentional — it forces the model to build long-range understanding through depth (4 layers of local context) rather than attending globally in one shot.
