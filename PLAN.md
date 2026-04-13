# TensionLM — Development Plan

## Goal

Build the best language model possible under the Thinking System (TS) philosophy.

TS holds that computation is constraint relaxation — a system of nodes and edges where τ (tension) represents unresolved constraint pressure, and inference is the process of the graph settling toward equilibrium. TensionLM is an LLM built to that spec: sigmoid tension instead of softmax attention, inspectable constraint fields instead of black-box weight matrices, non-competitive scoring instead of zero-sum normalisation.

The practical goal: a model that is not only competitive with transformers on standard benchmarks but is more interpretable, more theoretically grounded, and whose internal structure can be directly read as a learned constraint graph.

This is not purely academic. Every training run produces real evidence about whether TS is the right theory of computation. The tension field visualisations are direct measurements of the graph the model has learned.

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
| 11.1B token FineWeb run (117M) | **In progress** |
| Tension field visualiser (4 modes) | Done |
| Empirical TS validation (coherent vs salad density test) | Done — +25% mean τ, +60% edge density |

---

## Phase 1 — Validate at GPT-2 scale (current)

**Target:** 117M parameters, 11.1B FineWeb tokens. Val PPL competitive with GPT-2 117M (~29.4 on WikiText-103 test).

**Running now:**
```bash
torchrun --nproc_per_node=2 train.py \
  --data_dir data/fineweb-10B \
  --train_tokens 11_100_000_000 \
  --preset large \
  --out_dir checkpoints/tension_fw10b
```

**Success criterion:** val PPL ≤ 30.0 on WikiText-103 test after full training. This is the gate for everything that follows. If we land here, sigmoid tension is validated at GPT-2 scale and the scaling programme is justified. If we land at 33+, diagnose before scaling.

**After this run:**
- Eval on WikiText-103 test set (same as OpenAI's reported number — val set is not the same)
- Eval on HellaSwag zero-shot for cross-dataset comparison
- Generate tension field heatmaps from the final checkpoint for the paper
- Upload to HuggingFace

---

## Phase 2 — Ablations and architecture refinements

Before scaling, close the open architectural questions with cheap 117M runs (~10B tokens each, ~$50/run on 2× 4090).

### Ablation 1 — Window size

| Run | Window | Cost | Question |
|-----|--------|------|---------|
| Baseline (done) | 64 | — | — |
| Ablation A | 32 | ~$50 | Does smaller window hurt significantly? |
| Ablation B | 128 | ~$50 | Does larger window help enough to justify cost? |
| Ablation C | 512 | ~$50 | Is there a return beyond 128? |

This determines the window schedule for scaling (Phase 3). If W=128 beats W=64 by more than 0.5 PPL, the Triton kernel makes it worth the 2× compute.

### Ablation 2 — Softmax baseline at FineWeb scale

The WikiText-2 comparison (57.7 vs 57.8) is suggestive but small. Run an identical transformer at 117M/10B tokens. This is the paper's central table.

### Ablation 3 — Tau-mass normalisation

Current: `msg / valid_count` (uniform window normalisation)
Proposed: `msg / (τ.sum(-1).clamp(min=1e-6))` (tension-mass normalisation)

TS framing: dividing by valid count treats all positions as equal regardless of constraint strength. Dividing by τ-mass weights the message by actual constraint strength — positions with near-zero tension contribute negligibly. This may improve stability at large windows and better reflects TS semantics (a node's output should be proportional to the total constraint acting on it).

Test at 5M scale first — cheap and decisive.

### Ablation 4 — OscillatoryModulation

Ablate it out at 5M scale. If PPL is within noise, remove it and save the parameters for depth or width.

### Ablation 5 — Auxiliary losses at scale

`ManifoldClosureLoss` and `TensionDiversityLoss` are useful regularisers at small scale. At large scale with more data they may constrain unnecessarily. Test removing each at 117M/5B tokens.

---

## Phase 3 — Long-range context

**The problem.** W=64 means direct constraint reach is 64 tokens per layer. Long-range dependencies must propagate through stacked layers: 12 layers × 64-token hops = 768 effective tokens at best. This breaks at document scale.

**The solution.** Scale the window. The Triton kernel makes this O(T×W) — at W=512 it costs 8× more than W=64 but is still 4× cheaper than full O(T²) attention at seq_len=2048. Larger windows are the TS-correct answer because they literally extend the constraint graph's direct reach.

**RoPE.** At W=512+, learned absolute positional embeddings don't generalise. Replace with Rotary Position Embeddings applied to Q and K before the dot product. RoPE encodes relative position implicitly through rotation, generalises beyond training length, and is standard in all modern LLMs. OscillatoryModulation becomes redundant once RoPE is in — ablate and remove.

**Magnitude stability.** At large W, τ-mass normalisation (see Phase 2 Ablation 3) becomes essential rather than optional. Sum of W independent sigmoid values grows with W unless normalised by actual constraint mass.

**Validation target:** 117M, W=512, RoPE, 10B tokens. Compare val PPL vs W=64 baseline.

| Model scale | Window | Direct constraint reach |
|-------------|--------|--------------------------|
| 117M current | 64 | ~768 tokens (12 layers) |
| 117M Phase 3 | 512 | ~6,144 tokens |
| 1B | 1024 | ~24,576 tokens (24 layers) |
| 7B | 2048 | ~65,536 tokens (32 layers) |

---

## Phase 4 — Scale to 1B then 7B

### Architecture additions for scale

**Grouped-Query Tension (GQT).** At 7B+, K and V projections dominate memory. Use fewer K/V head groups than Q heads (e.g. 4 K/V groups for 32 Q heads). Each K/V group is shared across multiple Q heads. Reduces KV projection parameters by 8× with minimal quality loss — standard in LLaMA-2/3, Mistral, Gemma.

**Remove auxiliary losses.** At scale with enough data, `ManifoldClosureLoss` and `TensionDiversityLoss` may constrain unnecessarily. Phase 2 ablation decides whether they stay.

### Training infrastructure

**DDP → FSDP.** At 1B params, model state (weights + Adam + grads) is ~10 GB per GPU — fine on 4090. At 7B it's ~56 GB — exceeds any single card. Switch to PyTorch FSDP2 which shards all state across GPUs.

**Multi-node.** Beyond 8 GPUs, `torchrun` with c10d rendezvous backend. Training loop unchanged.

### Compute estimate

| Run | Params | Tokens | Hardware | Approx cost |
|-----|--------|--------|----------|-------------|
| Phase 1 (now) | 117M | 11.1B | 2× 4090 | ~$50 |
| Phase 2 ablations | 117M | ~50B total | 2× 4090 | ~$250 |
| Phase 3 validation | 117M | 10B | 2× 4090 | ~$50 |
| Phase 4a | 1B | 100B | 8× A100 | ~$3,000 |
| Phase 4b | 7B | 1T | 64× H100 | ~$150,000 |

Phase 4a is rentable (Lambda, CoreWeave, Vast.ai). Phase 4b requires a compute grant or commercial partnership. The arXiv paper from Phase 1 results is the lever — published results showing TensionLM matches GPT-2 at same compute, plus the O(T×W) efficiency argument, make the grant case.

### Scaling law study

Before Phase 4b, run a proper scaling law sweep:
- Train 5 models: 30M, 60M, 125M, 350M, 1B — each to Chinchilla-optimal token count
- Fit a power law to loss vs compute
- Compare exponent to transformer scaling law (Hoffmann et al. 2022)

If TensionLM's exponent is similar or better, the architecture scales. If it's worse, diagnose before spending 7B compute.

---

## Phase 5 — TS-native training objectives

### The problem with next-token prediction

Cross-entropy next-token prediction is a transformer-era objective. It was designed for a model that compresses everything into attention weights and produces a probability distribution over the vocabulary. It works — but it's a noisy proxy for what TensionLM is actually doing.

Next-token prediction asks: *given the past, what token comes next?* It trains the model to recover statistical co-occurrence patterns in text. The constraint graph emerges as a side effect of doing this well — the model learns constraints because constraints are the most efficient way to predict well. But the training signal never directly touches the constraint graph. It only rewards outputs.

Under TS, the correct objective is different: *given the surrounding constraints, what state would this position relax to?* Train the graph directly, not just the outputs.

Current results are the **floor**, not the ceiling. The model already works reasonably well despite the indirect training signal — which means the constraint structure is already latent in the text. Better objectives remove the noise and give the model a direct signal for what it's already trying to learn.

### TS-native objectives (to be explored after Phase 2)

**Equilibrium prediction (masked constraint resolution)**
Mask a span of tokens. Given the constraint graph from the surrounding context, predict the stable state the masked positions should relax to. Unlike BERT's masked language model (which predicts tokens independently), this is explicitly motivated by TS: the masked positions are nodes with unresolved tension, and the objective is to find their equilibrium given their constraint neighbourhood. The model is rewarded for finding a globally consistent state, not just a locally probable token.

**Constraint consistency loss**
If token A tensions token B strongly (high τ), and token B tensions token C strongly, then A should also tension C — transitivity of constraints. Penalise constraint graphs where this transitivity is violated. Pushes the model toward internally coherent representations rather than statistically plausible token sequences.

**Tension entropy regularisation**
Each position should have meaningful, selective tension — not zero (isolated node) and not maximum across all positions (noise). Penalise both extremes. The constraint graph should be sparse but non-trivial: some edges are strong, most are weak, none are absent entirely. This directly regularises the structure of the learned graph.

**Constraint symmetry**
Under TS, constraints are bidirectional in principle — if A constrains B, B constrains A to some degree. Pure one-directional tension is physically unusual. A soft symmetry regulariser on τ values encourages the model to build more physically coherent constraint graphs.

**Multi-scale consistency**
The constraint graph at layer 1 (local syntax) should be consistent with the constraint graph at layer 12 (semantic meaning). Currently layers are independent. A loss that aligns shallow and deep constraint patterns forces the model to build representations that are coherent across scales — syntactic structure and semantic structure point to the same underlying constraints.

### Data quality under TS

The internet contains contradictory constraints — two documents making opposite claims about the same thing. Under TS, contradictory data creates constraint loops that can never relax to equilibrium. The model learns a fuzzy average of the contradiction — which is neither true nor false, just statistically common. This is the mechanism behind hallucination.

The fix is not more data — it's **better constraint quality**:

- **Formal data first**: mathematics, verified code, formal proofs. Constraints are enforced by definition — contradictions literally cannot exist. A model trained heavily on formal data builds an extremely coherent internal constraint graph that generalises to messier domains.
- **Curated domain data**: textbooks, peer-reviewed papers, verified sources. High constraint consistency within domain.
- **Synthetic data from known ground truth**: generate training data from a system where you control the constraint graph, so coherence is guaranteed.

The current FineWeb training is necessary to establish the baseline. Future runs should increasingly weight formal and high-consistency data. This is not a data filtering problem — it's a constraint quality problem, and TS gives you a precise language to reason about it.

---

## Phase 6 — Memory and persistent constraint graphs

The current model has no persistence between runs. Every inference starts from zero — the constraint graph is rebuilt from scratch for each input. Under TS, this is incomplete. A full TS system accumulates edges in a persistent graph that updates with each observation.

The long-run architecture:

```
Input → TensionLM (fast, local relaxation, per-token)
              ↕
     Persistent constraint graph (slow, global, cross-session)
```

This is the two-timescale structure TS predicts: fast local constraint resolution (TensionLM) writing to and reading from a slow global constraint accumulator (the persistent graph). The TensionLM weights encode the constraint *types* the model knows about; the persistent graph encodes the constraint *instances* it has encountered.

Concretely this looks like:
- A graph database where nodes are concepts and edges are learned constraints with τ-weights
- At inference time, TensionLM queries the graph to prime its hidden states
- After inference, the tension field from the output updates the graph with new edges

This is distinct from RAG (retrieval-augmented generation), which retrieves text chunks. The persistent graph retrieves constraint patterns — the structural shape of what the model knows, not raw text.

This phase begins after Phase 4a produces a solid 1B model. The persistent graph is architecturally separate from TensionLM and can be developed in parallel.

The combination of Phase 5 (TS-native training) and Phase 6 (persistent graph) produces a system that learns constraints correctly *and* accumulates them over time — which is the full TS account of what intelligence is.

---

## The invariant

Every optimisation and scaling decision in this plan preserves one thing: **token pairs are scored independently.** No position is suppressed because another scored higher. The model learns which constraints matter without being forced to forget other constraints in the same step.

This is not a design preference. Under TS it is the only correct implementation. Softmax competition is incoherent with a constraint-relaxation account of computation. The whole project is structured around testing whether that theoretical commitment produces better models at scale.

---

## File map

| File | Purpose |
|------|---------|
| `model.py` | TensionLM architecture, aux losses, generation |
| `baseline.py` | Baseline transformer (identical API, softmax attention) |
| `train.py` | Training pipeline — single GPU, DDP, token budget |
| `prepare_data.py` | Stream + tokenize large datasets into binary shards |
| `eval.py` | Perplexity evaluation on any HuggingFace dataset |
| `generate.py` | Inference CLI |
| `visualise.py` | Tension field inspection — heatmap, token, layers, stats modes |
| `compare.py` | Plot loss curves from two CSV logs |
| `upload_hf.py` | Upload checkpoint and tokenizer to HuggingFace Hub |
| `triton_tension/` | Fused Triton kernels (fwd + bwd, gradcheck, ops wrapper) |
