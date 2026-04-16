# TensionLM

A language model that *is* a constraint graph. Built as the **TS-native language surface** for the Thinking System — not a standalone LLM, but the linguistic surface of a larger graph-native reasoning architecture.

Instead of softmax attention — a zero-sum competition where strong constraints silence weak ones — TensionLM uses **sigmoid tension**: each query-key pair is scored independently, constraints add rather than compete, and the internal attention field is structurally identical to a weighted graph. This lets it be coupled directly to an external constraint graph ([`TS-Core`](https://github.com/BoggersTheFish/TS-Core)) without the lossy natural-language bottleneck of traditional LLM + retrieval systems.

**Checkpoints:** [117M Curriculum](https://huggingface.co/BoggersTheFish/TensionLM-117M-Curriculum) · [117M WikiText-103](https://huggingface.co/BoggersTheFish/TensionLM-117M) · [Phase 2 TS-Native](https://huggingface.co/BoggersTheFish/TensionLM-Phase2-TSNative) · [GitHub](https://github.com/BoggersTheFish/bozo)

---

## Where TensionLM fits

The Thinking System is three-layered:

| Layer | Role | Implementation |
|-------|------|----------------|
| **Substrate** | Constraint graph — where thinking happens | [`TS-Core`](https://github.com/BoggersTheFish/TS-Core) — wave-cycle propagation over a `UniversalLivingGraph` |
| **Surface** | Fluent language on top of settled graph state | **TensionLM** (this repo) |
| **Grounding** | Closed-loop learning from high-confidence traces | QLoRA pipeline inside [`BoggersTheAI`](https://github.com/BoggersTheFish/BoggersTheAI) |

The substrate does the thinking. The surface converts settled graph state into tokens. Because TensionLM's internal representation (sigmoid tension over heads/layers) is structurally the same as the graph (weighted edges over nodes), the two layers can exchange representations directly — no reconciliation through a natural-language bottleneck.

**This is the thesis.** Previous LLM + KG systems pay a representational tax at every hop: KG stores graphs, LLM reads text, translation between them loses structure and creates hallucination. A graph-native surface eliminates the tax.

---

## The mechanism

Standard transformers use softmax attention: scores across a row sum to one, so high activation on token A forces low activation on B and C. Constraints compete for a fixed budget.

Under TS, this is wrong. Real constraints don't cancel each other. A concept can be simultaneously constrained by multiple past concepts at full strength — the constraints add, they don't compete.

TensionLM replaces softmax with sigmoid tension:

```
τ[t, w] = sigmoid( dot(Q[t], K[t-w]) / √head_dim )
output[t] = Σ_w  τ[t, w] · V[t-w]  /  Σ τ[t, w]
```

Each token pair is scored independently. No global normalisation. The denominator divides by tau-mass (Σ τ) — output is proportional to actual constraint strength, not position count.

What this enables:
- **Simultaneous full-strength constraints** — two tokens can both pull at τ=0.95 without either suppressing the other.
- **Readable attention field** — τ[layer, head, query_tok, key_tok] is directly interpretable as edge weights, no post-hoc rationalisation needed.
- **Bidirectional coupling to an external graph** — see the [`ts_bridge/` work](./PLAN.md#phase-1--surface--substrate-export-current) for the surface ↔ substrate interface.

---

## Surface-viability evidence

The experiments below establish that TensionLM is a **viable TS-native surface** — the tension field carries real constraint structure that responds to coherence. They are not evidence that TensionLM beats transformers as a standalone LLM (that's the wrong comparison for this project — see [PLAN.md](./PLAN.md)).

### Experiment 1 — Mechanism parity (1.1M params, WikiText-2)

Identical config — dim=128, 4 layers, 4 heads, BPE vocab=2048. Only the mechanism differs.

| Metric | TensionLM | Transformer |
|--------|-----------|-------------|
| Best val PPL | **57.7** | 57.8 |

Sigmoid tension ties softmax at parity. The surface mechanism is not broken.

### Experiment 2 — TS-native objectives (13.5M params, open-web-math)

Constraint consistency and tension entropy losses shape the τ field into a coherent graph.

| Model | Val PPL | Objective |
|-------|---------|-----------|
| Baseline | **85.19** | Cross-entropy only |
| TS-native | 86.50 | + constraint consistency + tension entropy |

1.31 PPL cost for structurally coherent constraint graphs. TS-native layer 5, head 2 on *"If A then B. If B then C. Therefore"*:

```
A[1]:0.32   B[3]:0.43   B[6]:0.64   C[8]:0.59   then[7]:0.50
```

The full A→B→C transitivity chain encoded as simultaneous active constraints.

**Coherence response:** coherent text produces **+25% mean τ** and **+60% more active edges** than random word salad on the same vocabulary. The τ field measurably responds to coherence — which is exactly what you need from a surface that will be exporting edges to a graph.

### Experiment 3 — Curriculum matches the TS layering doctrine (13.5M)

Logic → language → maths. Substrate before surface. First-contact maths PPL on the first checkpoint after switching data:

| Training history | First-contact maths PPL |
|-----------------|------------------------|
| Cold start | ~2293 |
| Logic only | ~1076 |
| Logic + language | **~582** |

4× better first contact than cold start. The ordering (build graph structure first, layer language on top) mirrors the TS substrate-before-surface prescription.

### Experiment 4 — 117M Curriculum run

RoPE, tau-mass normalisation, global attention every 4 blocks, scaled weight init, fused Triton kernel.

| Stage | Data | Tokens | Best val PPL |
|-------|------|--------|--------------|
| 1 — Logic | Synthetic inference | 200M | **5.10** |
| 2 — Language | FineWeb-Edu | 500M | **339** |
| 3 — Maths | open-web-math | 2B | **359.99** (step 14,000) |

**Stage 3 first-contact train PPL: 24** — no domain shock. Cold start baseline: ~2293. **96× better.** Minimum train PPL: **6.8**.

### Experiment 5 — Constraint interference confirmed (step 14k formal eval)

23-question benchmark (syllogisms, transitivity, arithmetic, calculus, algebra, definitions). Evaluated at step 14,000:

| Category | Score |
|----------|-------|
| Algebra | 67% (2/3) |
| Definitions | 67% (2/3) |
| Arithmetic | 50% (2/4) |
| Calculus | 50% (2/4) |
| Transitivity | 33% (1/3) |
| Syllogisms | 17% (1/6) |
| **Overall** | **43.5% (10/23)** |

**The key empirical confirmation of TS theory:** step 14,000 outperforms the final checkpoint. The second epoch of maths data partially overwrites the logic structure loaded in stage 1 — dense new constraints displace existing coherent constraints, exactly as constraint-propagation dynamics predict. The fix (active in the new run) is `--logic_mix 0.10`: thread 10% logic data through stage 3 to keep the constraint structure alive.

This is not an LLM training bug. It's a predicted TS phenomenon confirmed at training scale. Surfaces built to carry constraint graphs exhibit constraint-graph dynamics even in their weight updates.

---

## What the tension field shows

**Simultaneous non-competitive constraints.** Layer 12, head 0 on *"Manchester United won the Premier League title"*:
```
Manchester:0.95  United:0.95  League:0.86  Premier:0.71
```
τ=0.95 on two tokens simultaneously. Impossible in softmax. Natural in TensionLM.

**Logical constraint structure.** Layer 5, head 2 on *"If all mammals are warm-blooded and all whales are mammals then"*:
```
then ← mammals:0.755 | whales:0.755 | are:0.414
```
Both logical subjects held simultaneously at equal full strength. Head 11: `mammals:0.991` — near-total constraint, correctly identifying the key inference term.

**Head specialisation** into syntactic tracking, long-range semantic, and diffuse background emerges without any head-role supervision. The long-range-semantic heads are the ones that will produce useful edges for export to the substrate graph (see [`ts_bridge` head filter](./PLAN.md#phase-1--surface--substrate-export-current)).

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

Every Nth block (configurable via `--global_every`) is a **global tension layer** — uses the full sequence as its window instead of local W, enabling long-range constraint propagation without O(T²) cost at every layer.

**Key components:**
- **Sigmoid tension:** Independent score per token pair, no global normalisation
- **Tau-mass normalisation:** Output ∝ total constraint strength
- **RoPE:** Rotary position embeddings on Q and K
- **SwiGLU FFN:** Gated activation as in LLaMA / PaLM
- **Fused Triton kernel:** 64× memory reduction vs naive unfold, covers forward and backward, **now optionally emits the τ field** for substrate export without materialising the O(B·T·H·W·HD) gather buffer
- **Scaled weight init:** std/√depth for stability at 12+ layers

**Biological training machinery** (viewable as TS-cycle operations applied to weight updates):
- `--sleep_every` — periodic inhibitory mini-steps on corrupted inputs, mirroring substrate consolidation
- `--ff_mode` — Hinton forward-forward goodness, mirroring contrastive constraint pressure
- `--sparse_grad` — only backprop through high-entropy τ positions, mirroring plastic-window updates
- `--decouple_optim` — separate QK and VO optimisers, letting structure-shaping and readout update at different rates

---

## Quick start

```bash
pip install torch tokenizers datasets triton
python3 train.py                      # TensionLM, WikiText-2, small preset
python3 train.py --model transformer  # baseline transformer for ablation cross-reference
```

### Generate

```bash
python3 generate.py --checkpoint checkpoints/math_stage3/ckpt_0014000.pt \
    --prompt "The derivative of x squared is"
```

### Inspect the tension field

```bash
# Per-head heatmap
python3 visualise.py --checkpoint checkpoints/math_stage3/ckpt_0014000.pt \
    --mode heatmap \
    --text "If all mammals are warm-blooded and all whales are mammals then" \
    --out tension_heatmap.png

# Which tokens pull hardest on a specific position
python3 visualise.py --checkpoint checkpoints/math_stage3/ckpt_0014000.pt \
    --mode token \
    --text "The integral of x squared is x cubed over 3" \
    --token_idx -1
```

### Evaluate

```bash
python3 formal_eval.py --checkpoint checkpoints/math_stage3/ckpt_0014000.pt
python3 eval.py        --checkpoint checkpoints/math_stage3/ckpt_0014000.pt
```

Note: standalone eval is retained as a surface-viability signal, but it is **not** the primary benchmark for this project. See [PLAN.md §Phase 3](./PLAN.md#phase-3--integration-ab-the-real-paper-experiment) for the integration A/B that is.

---

## Training losses

| Loss | Weight | Purpose |
|------|--------|---------|
| `CrossEntropy` | 1.0 | Next-token prediction |
| `ManifoldClosureLoss` | 0.01 | First and last hidden states stay coherent |
| `TensionDiversityLoss` | 0.02 | Heads spread tension rather than collapsing |
| `ConstraintConsistencyLoss` | 0.1 | Enforce transitivity A→B, B→C ⟹ A→C |
| `TensionEntropyLoss` | 0.05 | Penalise isolated (τ≈0) and saturated (τ≈1) nodes |

---

## Roadmap

See [PLAN.md](./PLAN.md) for the full phase plan. Short version:

- **Phase 0** — Kernel unblock for τ export ✅ (complete 2026-04-16)
- **Phase 1** — Surface → substrate export (`ts_bridge/`) — current
- **Phase 2** — Substrate → surface biasing (graph conditions generation)
- **Phase 3** — Integration A/B benchmark (TensionLM + TS-Core vs Llama + TS-Core)
- **Phase 4** — Multi-agent coherence (Wave 16 — two TS-native surfaces communicating via shared graph)
- **Parallel** — 117M Curriculum / 350M bio runs, keeping a competent surface trained

---

## File map

| File | Purpose |
|------|---------|
| `model.py` | TensionLM architecture, aux losses, generation |
| `baseline.py` | Baseline transformer (ablation cross-reference) |
| `train.py` | Training pipeline — DDP, token budget, logic mixing, biological training machinery |
| `prepare_data.py` | Stream + tokenise datasets into binary shards |
| `eval.py` | Perplexity evaluation (surface-viability signal) |
| `generate.py` | Inference CLI — standard, anchored, cached generation |
| `formal_eval.py` | Formal reasoning benchmark (surface-viability signal) |
| `visualise.py` | Tension field inspection — feeds Phase 1 head-specialisation filter |
| `compare.py` | Plot loss curves |
| `upload_hf.py` | Upload to HuggingFace Hub |
| `triton_tension/` | Fused Triton kernels — fwd + bwd, **now with optional τ emission** |
| `ts_bridge/` | **New (Phase 1)** — surface ↔ substrate interface |
