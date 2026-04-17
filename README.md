# TensionLM

A language model built on **sigmoid tension** instead of softmax attention — an implementation of Thinking System (TS) theory of computation as constraint relaxation.

**Current focus: mathematical and code reasoning.** Formal domains are the ideal fit for a constraint-relaxation LM — contradictions cannot exist by construction, so the constraint graph built during training is coherent by definition. Code has the same property (a program either parses and runs or it doesn't), with a larger, more varied training corpus.

**Models:** [117M Curriculum](https://huggingface.co/BoggersTheFish/TensionLM-117M-Curriculum) · [117M WikiText-103](https://huggingface.co/BoggersTheFish/TensionLM-117M) · [Phase 2 TS-Native](https://huggingface.co/BoggersTheFish/TensionLM-Phase2-TSNative) · [GitHub](https://github.com/BoggersTheFish/bozo)

---

## The mechanism

Standard transformers use softmax attention — a zero-sum weight distribution. If token A scores high, tokens B and C are forced lower. Constraints compete for a fixed budget.

Under TS, this is wrong. Real constraints don't cancel each other. A concept can be simultaneously constrained by multiple past concepts at full strength — the constraints add, they don't compete.

TensionLM replaces softmax with sigmoid tension:

```
τ[t, w] = sigmoid( dot(Q[t], K[t-w]) / √head_dim )
output[t] = Σ_w  τ[t, w] · V[t-w]  /  Σ τ[t, w]
```

Each token pair is scored independently. No global normalisation — high score at one position suppresses nothing else. The denominator divides by tau-mass (Σ τ) — output is proportional to actual constraint strength, not position count.

What this enables:
- **Simultaneous full-strength constraints** — two tokens can both pull at τ=0.95 without either suppressing the other.
- **Readable attention field** — τ[layer, head, query, key] is directly interpretable as edge weights.
- **Coherent with constraint-graph downstream consumers** — the internal τ field is structurally the same as a weighted graph, so the model's attention state can be exported as edges (see [Ecosystem integration](#ecosystem-integration-optional) below).

---

## Results

### Experiment 1 — Mechanism validation (1.1M params, WikiText-2)

Identical config — dim=128, 4 layers, 4 heads, BPE vocab=2048. Only the mechanism differs.

| Metric | TensionLM | Transformer |
|--------|-----------|-------------|
| Best val PPL | **57.7** | 57.8 |
| Mechanism | Sigmoid tension (windowed) | Softmax attention (O(T²)) |

The mechanism works at parity with softmax.

---

### Experiment 2 — TS-native objectives (13.5M params, open-web-math)

Same architecture, same data, same hardware. Only the training objective differs.

| Model | Val PPL | Objective |
|-------|---------|-----------|
| Baseline | **85.19** | Cross-entropy only |
| TS-native | 86.50 | + constraint consistency + tension entropy |

1.31 PPL cost for structurally coherent constraint graphs. The constraint consistency loss enforces transitivity — if A tensions B and B tensions C, A should tension C. The result is visible directly in the tension field.

**TS-native layer 5, head 2 on "If A then B. If B then C. Therefore":**
```
A[1]:0.32   B[3]:0.43   B[6]:0.64   C[8]:0.59   then[7]:0.50
```
The full A→B→C transitivity chain encoded as simultaneous active constraints.

**Empirical TS validation:** Coherent text produces **+25% mean τ** and **+60% more active edges** than random word salad on the same vocabulary. The constraint graph measurably responds to coherence.

> **117M replication note (Phase 1.2, 2026-04-17).** The coherence direction replicates on the 117M-Curriculum checkpoint — 10/10 coherent prompts beat length-matched salads on raw-τ mean — at a smaller magnitude (mean-τ ratio ≈ 1.056× vs 1.25× on 13.5M). Direction is robust; magnitude is regime-specific. Reproduce with `python -m ts_bridge.exp2_replicate`.

---

### Experiment 3 — Curriculum training (13.5M params)

Logic → language → maths. First-contact maths PPL (first checkpoint after switching to maths data):

| Training history | First-contact maths PPL |
|-----------------|------------------------|
| Cold start | ~2293 |
| Logic only | ~1076 |
| Logic + language | **~582** |

**4× better first contact than cold start.** Curriculum validated.

---

### Experiment 4 — 117M curriculum run

RoPE, tau-mass normalisation, global attention every 4 blocks, scaled weight init, fused Triton kernel.

| Stage | Data | Tokens | Best val PPL |
|-------|------|--------|--------------|
| 1 — Logic | Synthetic inference | 200M | **5.10** |
| 2 — Language | FineWeb-Edu | 500M | **339** |
| 3 — Maths | open-web-math | 2B | **359.99** (step 14,000) |

**Stage 3 first-contact train PPL: 24** — no domain shock. Cold start baseline: ~2293. **96× better.**

**Minimum train PPL: 6.8.** GPT-2 (117M) on general web text achieves ~20 train PPL. TensionLM hits 6.8 on mathematics — a harder domain — because the curriculum pre-loaded constraint structure.

---

### Experiment 5 — Formal reasoning evaluation

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

**Key finding:** Step 14,000 outperforms the final checkpoint. The second epoch of maths data partially overwrites the logic structure loaded in stage 1. This is a direct TS prediction — dense new constraints can displace existing coherent constraints. The next training run addresses this with `--logic_mix 0.10`.

---

## What the tension field shows

**Simultaneous non-competitive constraints.** In softmax, if a token attends strongly to position A it attends less to B — they compete. In TensionLM, layer 12 head 0 on "Manchester United won the Premier League title":
```
Head 0:  Manchester:0.95  United:0.95  League:0.86  Premier:0.71
```
τ=0.95 on two tokens simultaneously. Impossible in softmax. Natural in TensionLM.

**Logical constraint structure.** Layer 5 head 2 on "If all mammals are warm-blooded and all whales are mammals then":
```
then ← mammals:0.755 | whales:0.755 | are:0.414
```
Both logical subjects held simultaneously at equal full strength. Head 11: `mammals:0.991` — near-total constraint, correctly identifying the key inference term.

**Head specialisation** into syntactic tracking, long-range semantic, and diffuse background emerges without any head-role supervision.

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
- **Sigmoid tension** — independent score per token pair, no global normalisation
- **Tau-mass normalisation** — output ∝ total constraint strength, not position count
- **RoPE** — rotary position embeddings on Q and K
- **SwiGLU FFN** — gated activation as in LLaMA / PaLM
- **Fused Triton kernel** — 64× memory reduction vs naive unfold, forward and backward, with optional τ emission for aux losses / export
- **Scaled weight init** — std/√depth for stability at 12+ layers
- **Optional pre-sigmoid bias hook** — an external constraint graph can inject an additive bias on the τ precursor before the sigmoid (see `ts_bridge/` below)

---

## Roadmap

- **117M math run** — the next training run per [PLAN.md](./PLAN.md): logic → ProofPile → open-web-math with `--logic_mix 0.10`, W=256, global_every=3, vocab=50,257
- **Code reasoning run** — parallel track on code corpora (The Stack / starcoderdata subsets) to test the same mechanism in another formally-constrained domain
- **Scale to 1B** — once the math-reasoning model is validated
- **Paper** — *TensionLM: Constraint Relaxation as a Mechanism for Mathematical Reasoning*

### Ecosystem integration (optional)

TensionLM's τ field is structurally identical to a weighted graph, so the model can plug into the broader TS ecosystem ([`TS-Core`](https://github.com/BoggersTheFish/TS-Core), [`BoggersTheAI`](https://github.com/BoggersTheFish/BoggersTheAI), `BoggersTheMind`). The `ts_bridge/` package in this repo exposes that interface — it's not on the critical path for the LLM itself, but the plumbing is ready when it's wanted.

- **Phase 1 — Surface → substrate export** (landed): `TauExporter`, head-filter classifier, corpus-level head profile, corpus-derived edge-threshold quantiles. See `ts_bridge/schema.md` for the contract.
- **Phase 1.5 — Generation-time streaming** (landed): `StreamingTauExporter` — per-step edge writes during autoregressive generation. Bit-parity with batch ingest.
- **Phase 2.0 — Graph → surface biasing** (landed): optional `tau_bias` added to the τ precursor pre-sigmoid, head-agnostic, threaded through every local-attention layer. A/B smoke confirms no-op / responsive / directional mechanism.
- **Phase 2.1 — Closed loop** (landed): `biased_generate.py` — graph biases forward, forward writes τ-edges back each step.
- **Phase 2.2 — Hardening + α calibration** (landed): `--export_mode {biased,unbiased,off}` decouples graph growth from bias strength; global-layer `tau_bias_global: [B,T,T]` plumbed; Triton kernel accepts `Bias` with σ' recomputed from biased τ. α sweep on 117M-curriculum (`ts_bridge.alpha_sweep`): silent ≤ α=2, inflection α≈4, loop reliably opens at α=8 (+9–19 edges, +0.026–0.073 mean-w vs unbiased-export), text degrades past α=16.

All of the above is bidirectional plumbing between TensionLM and a `UniversalLivingGraph`-shaped substrate. Used or ignored as downstream needs dictate; the LLM stands on its own.

---

## Quick start

```bash
pip install torch tokenizers datasets triton
python3 train.py                      # TensionLM, WikiText-2, small preset
python3 train.py --model transformer  # baseline transformer for comparison
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
python3 eval.py --checkpoint checkpoints/math_stage3/ckpt_0014000.pt
```

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

## Open questions

1. Does `--logic_mix 0.10` prevent the step-14k degradation? (Active investigation on the next run.)
2. Does W=256 meaningfully improve multi-step proof following vs W=64?
3. What is the optimal logic_mix ratio for stage 3?
4. Does the mechanism's advantage transfer from math to code reasoning?
5. Is the Chinchilla-optimal token count different for TS-native training vs standard cross-entropy?

---

## File map

| File | Purpose |
|------|---------|
| `model.py` | TensionLM architecture, aux losses, generation |
| `baseline.py` | Baseline transformer (identical API) |
| `train.py` | Training pipeline — DDP, token budget, logic mixing, biological training machinery |
| `prepare_data.py` | Stream + tokenise datasets into binary shards |
| `eval.py` | Perplexity evaluation |
| `generate.py` | Inference CLI — standard, anchored, cached generation |
| `formal_eval.py` | Formal reasoning benchmark |
| `visualise.py` | Tension field inspection |
| `compare.py` | Plot loss curves |
| `upload_hf.py` | Upload to HuggingFace Hub |
| `triton_tension/` | Fused Triton kernels (fwd + bwd) with optional τ emission |
| `ts_bridge/` | Optional bidirectional coupling to a `UniversalLivingGraph` substrate |
