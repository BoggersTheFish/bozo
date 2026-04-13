# TensionLM — Development Plan

## Goal

Build the best constraint-relaxation model possible under the Thinking System (TS) philosophy — with a specific focus on **formal reasoning, code, and mathematics** as the primary use case.

TS holds that computation is constraint relaxation: nodes, edges, τ as unresolved pressure, inference as equilibrium-finding. TensionLM implements this directly — sigmoid tension instead of softmax competition, inspectable constraint fields instead of black-box weights.

**The better use case insight:** general language modelling (FineWeb, web text) is the worst possible domain for a constraint-relaxation model. The internet is full of contradictory constraints — two documents making opposite claims create loops that can never relax to equilibrium. The model learns a fuzzy average that is neither true nor false.

Formal domains — mathematics, verified code, formal proofs — have *enforced* constraint consistency. Contradictions literally cannot exist. A model trained on formal data builds a constraint graph that is coherent by construction. And formal reasoning is also the domain where:
- Hallucination is most costly (wrong code, wrong proofs)
- Interpretability matters most (you need to understand *why*)
- The competition (GPT-4) is weakest relative to its general capability
- A small team can build a genuinely differentiated product

The current FineWeb run establishes the baseline and validates the mechanism. Everything after it pivots toward formal reasoning as the primary target.

This is not purely academic. Every training run produces real evidence about whether TS is the right theory of computation. The tension field visualisations are direct measurements of the constraint graph the model has learned.

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
| 11.1B token FineWeb run (117M) | **In progress** — val PPL ~35.6 at 33% |
| Tension field visualiser (4 modes) | Done |
| Empirical TS validation (coherent vs salad density test) | Done — +25% mean τ, +60% edge density |
| Anchored generation (permanent prompt constraints) | Done |

---

## Phase 1 — Validate the mechanism (current)

**Target:** Prove sigmoid tension works at GPT-2 scale. 117M parameters, 11.1B FineWeb tokens.

**Running now:**
```bash
torchrun --nproc_per_node=2 train.py \
  --data_dir data/fineweb-10B \
  --train_tokens 11_100_000_000 \
  --preset large \
  --out_dir checkpoints/tension_fw10b
```

**Success criterion:** val PPL ≤ 30.0 on WikiText-103 test. This validates the mechanism, not the use case. FineWeb is noisy web data — its role is to establish that sigmoid tension can learn at scale under the worst conditions. If it works here, it will work everywhere.

**After this run:**
- Eval on WikiText-103 test set
- Softmax baseline comparison at same scale (the paper's central table)
- Tension field heatmaps from final checkpoint for the paper
- Upload to HuggingFace
- arXiv preprint

---

## Phase 2 — TS-native training objectives (small scale)

**Do this before scaling.** This is the most important phase in the plan.

Next-token prediction is a transformer-era objective. It trains the model to recover statistical co-occurrence patterns — the constraint graph emerges as a side effect. The training signal never directly touches the graph structure. Current results are the **floor**, not the ceiling.

Under TS, the correct objective is: *given the surrounding constraints, what state would this position relax to?* Train the graph directly.

Test all objectives cheaply at 5M–30M scale before committing to an expensive run.

### Equilibrium prediction (primary candidate)

Mask a span of tokens. Given the constraint graph from surrounding context, predict the stable state the masked positions relax to. Unlike BERT's MLM (which predicts tokens independently), this is TS-native: masked positions are nodes with unresolved tension, objective is to find their equilibrium given their constraint neighbourhood.

The model is rewarded for finding a **globally consistent state**, not a locally probable token. This is a fundamentally different training signal — one that only makes sense if you have a readable constraint graph.

### Constraint consistency loss

If A tensions B strongly and B tensions C strongly, A should tension C — transitivity. Penalise graphs where transitivity is violated. Pushes toward internally coherent representations rather than statistically plausible sequences.

### Tension entropy regularisation

Each position should have meaningful, selective tension — not zero (isolated node) and not maximum everywhere (noise). The constraint graph should be sparse but non-trivial. Penalise both extremes directly.

### Constraint symmetry

If A constrains B, B constrains A to some degree. Pure one-directional tension is physically unusual under TS. Soft symmetry regulariser on τ values.

### Multi-scale consistency

Layer 1 constraint graph (local syntax) should be consistent with layer 12 (semantic meaning). A loss aligning shallow and deep constraint patterns forces coherence across scales.

**Decision gate:** if equilibrium prediction + constraint consistency loss beats cross-entropy at 30M scale by more than 1 PPL, switch all future runs to the new objective. If not, keep cross-entropy and add TS losses as auxiliary terms.

---

## Phase 3 — Formal data curriculum

Once the training objective is right, get the data right.

**The problem with web text:** the internet contains contradictory constraints — two documents making opposite claims. Under TS, contradictory data creates constraint loops that can never relax to equilibrium. The model learns a fuzzy average — statistically common but neither true nor false. This is the mechanism behind hallucination.

**The solution:** formal data has enforced constraint consistency. Contradictions cannot exist in verified mathematics or compiling code.

### Data tiers by constraint quality

| Tier | Data type | Constraint consistency | Examples |
|------|-----------|----------------------|---------|
| 1 (highest) | Formal proofs, verified code | Enforced — contradictions impossible | Lean4, Coq, Isabelle proofs; test-passing code |
| 2 | Mathematics | Very high — peer reviewed | ArXiv math, textbooks, Stack Exchange math |
| 3 | Code | High — compiles/runs | GitHub, Stack Overflow |
| 4 | Scientific text | Medium — peer reviewed | ArXiv CS/physics, PubMed |
| 5 (lowest) | Web text | Low — contradictions common | FineWeb, Common Crawl |

**Training curriculum:**
1. Warm up on Tier 1+2 exclusively — build a maximally coherent constraint graph
2. Introduce Tier 3 — let code structure reinforce formal reasoning
3. Add Tier 4 — scientific language grounded in formal results
4. Optional: small fraction of Tier 5 for language coverage

A 117M model trained bottom-up from formal data with TS-native objectives should produce dramatically better reasoning than the same model trained on FineWeb — because the constraint graph was coherent from the start.

This is also the differentiator from every other LLM. Nobody is training with this curriculum because nobody has a theory that says why it should work. TS provides the theory.

---

## Phase 4 — Architecture refinements and ablations

With the right objective and data established, close the open architectural questions.

### Ablation 1 — Window size

| Run | Window | Question |
|-----|--------|---------|
| Baseline | 64 | — |
| A | 128 | Does larger window help enough to justify 2× compute? |
| B | 512 | Is there return beyond 128? |

### Ablation 2 — Tau-mass normalisation

Current: `msg / valid_count` — treats all positions equally regardless of τ strength.
Proposed: `msg / τ.sum()` — weights message by actual constraint mass.

TS-correct: a node's output should be proportional to the total constraint acting on it, not the count of positions that could have constrained it.

### Ablation 3 — OscillatoryModulation

Remove at 5M scale. If PPL is unaffected, save the parameters for depth or width.

### Ablation 4 — Auxiliary losses

`ManifoldClosureLoss` and `TensionDiversityLoss` — useful at small scale, may constrain unnecessarily at large scale with good data. Test removing each.

### RoPE

At W=128+, learned absolute positional embeddings don't generalise beyond training length. Add Rotary Position Embeddings to Q and K. OscillatoryModulation becomes redundant once RoPE is in.

---

## Phase 5 — Long-range context

With right objective + right data + right architecture, scale the window.

The Triton kernel makes this O(T×W). At W=512, TensionLM is 4× cheaper than full O(T²) attention at seq_len=2048.

| Model scale | Window | Direct constraint reach |
|-------------|--------|--------------------------|
| 117M current | 64 | ~768 tokens (12 layers) |
| 117M Phase 5 | 512 | ~6,144 tokens |
| 1B | 1024 | ~24,576 tokens (24 layers) |
| 7B | 2048 | ~65,536 tokens (32 layers) |

For formal reasoning specifically, long-range context is critical — a proof can depend on a definition stated 2,000 tokens earlier. This is where TensionLM's O(T×W) advantage over transformers is most commercially relevant.

---

## Phase 6 — Scale to 1B then 7B

Scale only after Phases 2–5 are validated. Scaling with the wrong objective or wrong data is expensive and produces a ceiling you then have to retrain through.

### Architecture additions

**Grouped-Query Tension (GQT):** fewer K/V head groups than Q heads at 7B+ scale. Standard in LLaMA-2/3, Mistral. Reduces KV projection parameters by 8× with minimal quality loss.

### Infrastructure

**DDP → FSDP** at 7B (model state exceeds single GPU memory). PyTorch FSDP2.

### Compute

| Run | Params | Tokens | Hardware | Approx cost |
|-----|--------|--------|----------|-------------|
| Phase 1 (now) | 117M | 11.1B | 2× 4090 | ~$50 |
| Phase 2 experiments | 5–30M | ~5B total | 2× 4090 | ~$30 |
| Phase 3 formal run | 117M | 10B | 2× 4090 | ~$50 |
| Phase 4 ablations | 117M | ~30B total | 2× 4090 | ~$150 |
| Phase 5 W=512 run | 117M | 10B | 2× 4090 | ~$50 |
| Phase 6a | 1B | 100B | 8× A100 | ~$3,000 |
| Phase 6b | 7B | 1T | 64× H100 | ~$150,000 |

Phase 6a rentable on Vast.ai. Phase 6b requires a compute grant. The arXiv paper from Phase 1 + Phase 2 results (TS-native objectives beating cross-entropy) is the lever.

### Scaling law study

Before Phase 6b: 5 models at 30M, 60M, 125M, 350M, 1B — each to Chinchilla-optimal token count. Fit power law, compare exponent to transformer scaling law (Hoffmann et al. 2022).

---

## Phase 7 — Persistent constraint graph (memory)

The full TS architecture:

```
Input → TensionLM (fast, local relaxation, per-token)
              ↕
     Persistent constraint graph (slow, global, cross-session)
```

TensionLM weights encode constraint *types* the model knows. The persistent graph encodes constraint *instances* it has encountered. The graph accumulates edges over time — this is what TS means by learning.

At inference: TensionLM queries the graph to prime hidden states with relevant prior constraints.
After inference: the tension field from the output updates the graph with new edges.

This is distinct from RAG (retrieves text chunks). The persistent graph retrieves constraint patterns — the structural shape of what the model knows, not raw text.

Begins after Phase 6a. Architecturally separate, can be developed in parallel.

The combination of Phase 2 (TS-native training) + Phase 7 (persistent graph) produces a system that learns constraints correctly *and* accumulates them over time. That is the full TS account of intelligence.

---

## The invariant

Every decision in this plan preserves one thing: **token pairs are scored independently.** No position is suppressed because another scored higher. The model learns which constraints matter without being forced to forget other constraints in the same step.

Under TS this is not a design preference — it is the only correct implementation. Softmax competition is incoherent with a constraint-relaxation account of computation.

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
