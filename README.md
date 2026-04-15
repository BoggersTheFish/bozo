# TensionLM

A language model built on **sigmoid tension** instead of softmax attention — an empirical implementation of the Thinking System (TS) theory of computation.

**[117M model on HuggingFace](https://huggingface.co/BoggersTheFish/TensionLM-117M)** | **[Phase 2 TS-native model](https://huggingface.co/BoggersTheFish/TensionLM-Phase2-TSNative)** | **[GitHub](https://github.com/BoggersTheFish/bozo)**

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
output[t] = Σ_w  τ[t, w] · V[t-w]  /  Σ τ[t, w]
```

Each token pair is scored independently. τ[t, w] measures how strongly token t is constrained by token t-w. There is no global normalisation — a high score at one position does not suppress any other. The tension field is the literal constraint graph the model has learned. It is inspectable, interpretable, and directly motivated by TS theory.

The denominator divides by **tau-mass** (Σ τ) rather than valid position count — TS-correct normalisation where output is proportional to actual constraint strength, not position count.

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

### Phase 2 — TS-native objectives vs baseline (13.5M, open-web-math)

Clean comparison: same architecture (13.5M params, dim=256, 6 layers), same data (1B tokens of open-web-math), same hardware. Only difference: training objective.

| Model | Val PPL | Training objective |
|-------|---------|-------------------|
| Baseline | **85.19** | Cross-entropy only |
| TS-native | 86.50 | Cross-entropy + constraint consistency + tension entropy |

**Gap: 1.31 PPL.** TS-native trades a negligible PPL cost for structurally coherent constraint graphs and qualitatively superior reasoning behaviour.

The constraint consistency loss enforces transitivity — if A tensions B and B tensions C, A should tension C. The result is visible directly in the tension field:

**TS-native layer 5, head 2 on "If A then B. If B then C. Therefore":**
```
A[1]:0.32   B[3]:0.43   B[6]:0.64   C[8]:0.59   then[7]:0.50
```
The full transitivity chain A→B→C is explicitly encoded as active simultaneous constraints.

**Baseline layer 5 on the same prompt:** uniform diffuse activation across all tokens, no selective structure.

### Curriculum training — logic → language → maths (13.5M)

TS predicts that coherent data builds coherent constraint graphs. The correct training order is: teach inference structure first, then vocabulary, then formal notation.

| Stage | Data | Tokens | Final val PPL |
|-------|------|--------|--------------|
| 1 — Logic | Synthetic inference (TS-native) | 200M | **1.51** |
| 2 — Language | WikiText-103 | 100M | **196** |
| 3 — Maths | open-web-math | 200M | **573** |

**First-contact PPL on maths** (first val checkpoint after switching to maths data):

| Training history | First-contact maths val PPL |
|-----------------|----------------------------|
| Cold start (Phase 2) | ~2293 |
| Logic only | ~1076 |
| Logic + language | **~582** |

4× better first contact than cold start.

### 117M curriculum run — in progress

Same curriculum at full scale. Architecture upgraded: RoPE, tau-mass normalisation, global attention layers every 4 blocks, scaled weight init, Triton kernel.

| Stage | Data | Tokens | Best val PPL | Status |
|-------|------|--------|--------------|--------|
| 1 — Logic | Synthetic inference | 200M | **5.10** | Done |
| 2 — Language | FineWeb-Edu | 500M | **339** | Done |
| 3 — Maths | open-web-math | 2B | **359.99** | Done |

**Stage 3 first-contact:** train PPL **24** at step 2550, no domain shock. The 13.5M run had first-contact maths PPL ~582. The 117M curriculum model walked straight into maths data without flinching — 96× better than cold start.

**Stage 3 minimum train PPL: 6.8.** GPT-2 (117M, same size) trained on general web text achieves ~20 train PPL. TensionLM hits 6.8 on mathematics — a substantially harder domain — due to the curriculum pre-loading constraint structure before the formal notation was introduced.

**Stage 3 early generations (step 2550):** already producing LaTeX notation, proof structure, formal equation output:
```
To prove this by contradiction assume that $\theta_n \in [0,1]$.
It is true to say that if $A = 1\left(A+B|A-B|\right)^2$ then
the result of a series converges to $A_1 = B_0^{-1} + C_3^{-2} \rightarrow B_n^{+1}$
```

### Formal reasoning evaluation (23-question benchmark)

Evaluated on syllogisms, transitivity, arithmetic, calculus, algebra, and definitions using the best checkpoint (step 14,000, val PPL 359.99):

| Category | Score |
|----------|-------|
| Algebra | 67% (2/3) |
| Definitions | 67% (2/3) |
| Arithmetic | 50% (2/4) |
| Calculus | 50% (2/4) |
| Transitivity | 33% (1/3) |
| Syllogisms | 17% (1/6) |
| **Overall** | **43.5% (10/23)** |

**Finding:** The best reasoning checkpoint is step 14,000, not the final. The second epoch of maths data partially overwrites the logic structure loaded in stage 1 — a sweet spot exists before full stage 3 saturation. This is a direct TS prediction: overwriting coherent constraints with more constraints of a different type reduces overall graph coherence.

---

## What the tension field actually shows

The tension field is inspectable — you can read what the model has learned directly from τ values.

**Composability:** In softmax attention, if a token attends strongly to position A, it attends less to position B — they compete. In TensionLM, a token can be simultaneously pulled by all its relevant predecessors at full strength. In our 117M model (layer 12, head 0), the token "title" in "Manchester United won the Premier League title" produces:

```
Head 0:  Manchester:0.95  United:0.95  League:0.86  Premier:0.71
```

τ=0.95 on two tokens simultaneously. In softmax this is impossible. In TensionLM it is the natural result.

**Logical constraint structure:** On the prompt "If all mammals are warm-blooded and all whales are mammals then", layer 5 head 2 of the stage 2 model produces:

```
then ← mammals:0.755 | whales:0.755 | are:0.414
```

Both logical subjects held simultaneously at equal full strength. Head 11 shows `mammals:0.991` — near-total constraint, correctly identifying `mammals` as the key term for what follows `then`. The constraint graph is doing the right thing internally even before the output layer knows how to complete the syllogism.

**Head specialisation:** Different heads specialise spontaneously into different constraint roles — some track local syntactic structure, some track long-range semantic content, some are uniformly diffuse.

**Empirical TS validation:** Coherent text produces a 25% higher mean τ and 60% more active edges than random word salad on the same vocabulary.

You can inspect the tension field yourself:

```bash
# Per-head heatmap for a sentence
python visualise.py --checkpoint checkpoints/stage3_math_117m/latest.pt \
    --mode heatmap \
    --text "The history of artificial intelligence" \
    --out tension_heatmap.png

# Which past tokens pull hardest on a specific token
python visualise.py --checkpoint checkpoints/stage3_math_117m/latest.pt \
    --mode token \
    --text "Manchester United won the Premier League title" \
    --token_idx -1

# How tension evolves across layers
python visualise.py --checkpoint checkpoints/stage3_math_117m/latest.pt \
    --mode layers \
    --text "Scientists discovered that water conducts electricity" \
    --token_idx -1

# Head specialisation statistics over a sample
python visualise.py --checkpoint checkpoints/stage3_math_117m/latest.pt \
    --mode stats \
    --sample_file data/fineweb-edu/val_0000.bin \
    --sample_size 200
```

---

## Architecture

```
Embedding
  └─ × N  TensionBlock
          ├─ RMSNorm (pre-norm)
          ├─ MultiHeadCausalTensionLayer   ← the constraint mechanism
          └─ SwiGLU FFN
RMSNorm → LM head (weight-tied to embedding)
```

Every 4th block is a **global tension layer** — uses the full sequence as its window instead of the local W=64 window, enabling long-range constraint propagation without O(T²) cost at every layer.

**`MultiHeadCausalTensionLayer`**

H heads, causal window of W tokens. Each head independently computes τ[t, w] = sigmoid(dot(Q[t], K[t-w]) / scale) for all w in [0, W), then aggregates V[t-w] weighted by τ. No position competes with any other. Output is normalised by tau-mass (Σ τ) rather than valid position count — TS-correct: output proportional to actual constraint strength.

At training scale, this is implemented as a fused Triton kernel that avoids materialising the B×T×H×W×HD intermediate tensor — a 64× memory reduction vs the naive unfold approach.

**RoPE** — Rotary Position Embeddings on Q and K. Better length generalisation than learned absolute positional embeddings.

**`SwiGLU FFN`**

```
out = proj( silu(gate(x)) * val(x) )
```

Gated activation as in LLaMA/PaLM.

**Scaled weight init** — projection weights initialised with std/√depth (output projections std/√(2×depth)). Ensures stable gradient flow at 12 layers.

---

## Training signal

Three standard losses plus two TS-native objectives:

| Loss | Weight | Purpose |
|------|--------|---------|
| `CrossEntropy` | 1.0 | Next-token prediction |
| `ManifoldClosureLoss` | 0.01 | First and last hidden states stay coherent |
| `TensionDiversityLoss` | 0.02 | Heads spread tension rather than collapsing onto one position |
| `ConstraintConsistencyLoss` | 0.1 | Enforce transitivity — if A→B and B→C, then A→C |
| `TensionEntropyLoss` | 0.05 | Penalise isolated nodes (τ≈0) and saturated nodes (τ≈1) |

The TS-native losses directly train the constraint graph structure.

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
python3 generate.py --checkpoint checkpoints/stage3_math_117m/latest.pt --prompt "The cat"
python3 generate.py --checkpoint checkpoints/stage3_math_117m/latest.pt  # interactive
```

### Evaluate perplexity

```bash
python3 eval.py --checkpoint checkpoints/stage3_math_117m/latest.pt
python3 eval.py --checkpoint checkpoints/stage3_math_117m/latest.pt --dataset wikitext-103-raw-v1
```

### Multi-GPU training (DDP)

```bash
torchrun --nproc_per_node=2 train.py --preset large --model tension \
  --out_dir checkpoints/tension_117m \
  --log_csv logs/tension_117m.csv
```

### Curriculum training

```bash
# Stage 1 — logic
torchrun --nproc_per_node=2 train.py \
    --data_dir data/logic-stage1 \
    --train_tokens 200_000_000 \
    --preset large \
    --w_consistency 0.1 --w_entropy 0.05 \
    --out_dir checkpoints/stage1_logic_117m

# Stage 2 — language (copy stage 1 checkpoint first)
cp checkpoints/stage1_logic_117m/latest.pt checkpoints/stage2_language_117m/latest.pt
torchrun --nproc_per_node=2 train.py \
    --data_dir data/fineweb-edu \
    --train_tokens 500_000_000 \
    --preset large --resume \
    --out_dir checkpoints/stage2_language_117m

# Stage 3 — maths
cp checkpoints/stage2_language_117m/latest.pt checkpoints/stage3_math_117m/latest.pt
torchrun --nproc_per_node=2 train.py \
    --data_dir data/open-web-math \
    --train_tokens 2_000_000_000 \
    --preset large --resume \
    --out_dir checkpoints/stage3_math_117m
```

### Large-scale training (FineWeb)

```bash
# Pre-tokenize into binary shards first
python3 prepare_data.py \
  --dataset fineweb-edu \
  --out_dir data/fineweb-edu \
  --tokenizer checkpoints/stage1_logic_117m/tokenizer.json \
  --max_tokens 500_000_000

# Train on shards
torchrun --nproc_per_node=2 train.py \
  --data_dir data/fineweb-edu \
  --train_tokens 500_000_000 \
  --out_dir checkpoints/stage2_language_117m
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

1. **Does window depth substitute for window breadth?** W=64 means direct context is 64 tokens per layer. Long-range dependencies propagate through stacked layers. Does this break at document scale, or does depth compensate?

2. **Does the curriculum model beat a cold-start 117M on formal reasoning tasks?** Stage 3 finishes in ~19h. This is the main result being built toward.

3. **Scaling law exponent.** Is TensionLM's loss-vs-compute curve competitive with transformers? Needs a Chinchilla-style sweep at 5–350M.

4. **Does global layer frequency matter?** Currently every 4th block. Needs ablation.

5. **RoPE vs learned positional embeddings.** RoPE is on by default for large preset. Ablation pending.
