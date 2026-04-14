# TensionLM — Development Plan

## Goal

Build the best constraint-relaxation model possible under the Thinking System (TS) philosophy — with a specific focus on **formal reasoning, code, and mathematics** as the primary use case.

TS holds that computation is constraint relaxation: nodes, edges, τ as unresolved pressure, inference as equilibrium-finding. TensionLM implements this directly — sigmoid tension instead of softmax competition, inspectable constraint fields instead of black-box weights.

**The better use case insight:** general language modelling (FineWeb, web text) is the worst possible domain for a constraint-relaxation model. The internet is full of contradictory constraints — two documents making opposite claims create loops that can never relax to equilibrium. The model learns a fuzzy average that is neither true nor false.

Formal domains — mathematics, verified code, formal proofs — have *enforced* constraint consistency. Contradictions literally cannot exist. A model trained on formal data builds a constraint graph that is coherent by construction.

---

## Status

| Milestone | Status |
|-----------|--------|
| Proof of concept (1.1M, WikiText-2, CPU) | Done |
| Baseline comparison — tension vs transformer, same config | Done — PPL 57.7 vs 57.8 |
| GPU training setup (DDP, bf16, torch.compile) | Done |
| 117M run on WikiText-103 | Done — val PPL 32.01 |
| HuggingFace release | Done — BoggersTheFish/TensionLM-117M |
| Fused Triton kernel (fwd + bwd, gradcheck) | Done |
| Tension field visualiser (4 modes) | Done |
| Empirical TS validation (coherent vs salad density test) | Done — +25% mean τ, +60% edge density |
| Anchored generation (permanent prompt constraints) | Done |
| Phase 2 — TS-native vs baseline (13.5M, open-web-math) | Done — baseline 85.19 PPL, TS-native 86.50 PPL |
| Stage 1 logic dataset (200M synthetic tokens) | Done — `data/logic-stage1` |
| Curriculum training — logic → language → maths (13.5M) | Done — 4× better first-contact PPL vs cold start |
| Curriculum transfer LR fix + closure loss reduction | Done — `--transfer_lr`, `w_closure` default 0.01 |
| TS-native Phase 2 model on HuggingFace | Done — BoggersTheFish/TensionLM-Phase2-TSNative |
| Architecture upgrades (RoPE, tau-mass norm, global layers, scaled init) | Done |
| FineWeb-Edu pipeline (`--dataset fineweb-edu`, `--max_tokens`) | Done |
| 117M Stage 1 — logic (200M tokens) | Done — val PPL 5.10, train PPL 1.3 |
| 117M Stage 2 — language FineWeb-Edu (500M tokens) | Done — val PPL 339, train PPL ~65 |
| 117M Stage 3 — maths open-web-math (2B tokens) | **Running** — step 2550/30517, train PPL 24, ~19h left |

---

## Architecture — current 117M config

All optimisations applied before the 117M run:

| Feature | Status | Rationale |
|---------|--------|-----------|
| RoPE | On (default for large preset) | Better length generalisation |
| Tau-mass normalisation | On | TS-correct: output ∝ constraint strength |
| Global layers every 4 | On | Long-range constraint propagation |
| Scaled weight init | On | Stability at depth |
| Triton fused kernel | On | 64× memory reduction |
| Vectorised consistency loss | On | No Python loops |
| No OscillatoryModulation | On | RoPE handles positional structure |

---

## Phase 1 — Validate the mechanism ✓

**Done.** Sigmoid tension matches transformer at identical parameter count. FineWeb run abandoned — pivoted to formal reasoning as primary use case.

---

## Phase 2 — TS-native training objectives ✓

**Done.** TS-native objectives (constraint consistency + tension entropy) produce measurably more coherent constraint graphs. 1.31 PPL cost, explicit transitivity chains visible in tension field at layer 5.

---

## Phase 3 — Formal data curriculum ← current

**117M curriculum run in progress.**

Stage 1 complete: logic inference structure loaded, val PPL 5.10.
Stage 2 complete: language absorbed, val PPL 339, train PPL 65. FineWeb-Edu (educational quality filtered) used instead of WikiText-103 — better TS alignment, 5× more tokens.
Stage 3 running: first-contact maths train PPL 24 (no domain shock). LaTeX notation and proof structure appearing at step 2550. 19h remaining.

**Key result so far:** Stage 3 first-contact PPL 24 vs ~582 for 13.5M run, ~2293 for cold start. The curriculum is working dramatically better at scale.

**Next after stage 3 completes:**
- Run formal reasoning evaluation (syllogisms, simple proofs, equation solving)
- Compare against cold-start 117M baseline on same tasks
- Upload curriculum model to HuggingFace
- Write arXiv paper

---

## Phase 4 — Architecture refinements and ablations

With curriculum validated, close the open architectural questions.

### Ablation 1 — Window size
| Run | Window | Question |
|-----|--------|---------|
| Baseline | 64 | — |
| A | 128 | Does larger window help enough to justify 2× compute? |
| B | 512 | Is there return beyond 128? |

### Ablation 2 — Global layer frequency
Currently every 4th block. Test every 3rd and every 6th.

### Ablation 3 — Tau-mass vs valid-count normalisation
Already implemented tau-mass. Ablation to confirm improvement.

### Ablation 4 — RoPE vs learned positional embeddings
RoPE on by default. Ablation to confirm benefit.

---

## Phase 5 — Long-range context

With right objective + right data + right architecture, scale the window.

| Model scale | Window | Direct constraint reach |
|-------------|--------|--------------------------|
| 117M current | 64 | ~768 tokens (12 layers) |
| 117M Phase 5 | 512 | ~6,144 tokens |
| 1B | 1024 | ~24,576 tokens (24 layers) |

For formal reasoning specifically, long-range context is critical — a proof can depend on a definition stated 2,000 tokens earlier.

---

## Phase 6 — Scale to 1B then 7B

Scale only after Phases 2–5 are validated.

| Run | Params | Tokens | Hardware | Approx cost |
|-----|--------|--------|----------|-------------|
| Phase 3 formal run | 117M | ~2.7B total | 2× 4090 | ~$0 (own hardware) |
| Phase 4 ablations | 117M | ~30B total | 2× 4090 | ~$150 |
| Phase 5 W=512 run | 117M | 10B | 2× 4090 | ~$50 |
| Phase 6a | 1B | 100B | 8× A100 | ~$3,000 |
| Phase 6b | 7B | 1T | 64× H100 | ~$150,000 |

Phase 6a rentable on Vast.ai. Phase 6b requires a compute grant.

---

## Phase 7 — Persistent constraint graph (memory)

The full TS architecture:

```
Input → TensionLM (fast, local relaxation, per-token)
              ↕
     Persistent constraint graph (slow, global, cross-session)
```

TensionLM weights encode constraint *types* the model knows. The persistent graph encodes constraint *instances* it has encountered.

---

## Paper outline

**Title:** TensionLM: Sigmoid Tension as Constraint Relaxation for Language Modelling

**Sections:**
1. TS theory — computation as constraint relaxation, why softmax is wrong
2. Mechanism — sigmoid tension, tau-mass normalisation, global layers
3. Experiment 1 — mechanism comparison (tension vs transformer, same config)
4. Experiment 2 — TS-native objectives (constraint consistency + entropy, Phase 2 results)
5. Experiment 3 — curriculum training (logic → language → maths, first-contact PPL)
6. Experiment 4 — 117M curriculum run results (stage 3 pending)
7. Tension field analysis — simultaneous constraints, head specialisation, coherent vs salad
8. Discussion — implications for TS theory, scaling, formal reasoning

**Target:** arXiv cs.LG, after stage 3 completes.

---

## The invariant

Every decision in this plan preserves one thing: **token pairs are scored independently.** No position is suppressed because another scored higher. The model learns which constraints matter without being forced to forget other constraints in the same step.

Under TS this is not a design preference — it is the only correct implementation.

---

## File map

| File | Purpose |
|------|---------|
| `model.py` | TensionLM architecture, aux losses, generation |
| `baseline.py` | Baseline transformer (identical API, softmax attention) |
| `train.py` | Training pipeline — single GPU, DDP, token budget |
| `prepare_data.py` | Stream + tokenize large datasets into binary shards |
| `eval.py` | Perplexity evaluation on any HuggingFace dataset |
| `generate.py` | Inference CLI — standard and anchored generation |
| `visualise.py` | Tension field inspection — heatmap, token, layers, stats modes |
| `compare.py` | Plot loss curves from two CSV logs |
| `upload_hf.py` | Upload checkpoint and tokenizer to HuggingFace Hub |
| `triton_tension/` | Fused Triton kernels (fwd + bwd, gradcheck, ops wrapper) |
