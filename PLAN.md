# TensionLM — Development Plan

## Goal

Build a language model using sigmoid tension as the core attention mechanism, targeting **mathematical and code reasoning**. The architecture is TS-inspired: attention as constraint relaxation over a graph of interdependent states. Formal domains — mathematics and code — are the ideal training targets: contradictions cannot exist by construction, so the constraint graph built during training is coherent by definition.

**Why formal domains:** General web text creates irresolvable constraint loops (contradictory documents). Formal domains have enforced consistency. A model trained on formal data builds a constraint graph that is coherent by construction — exactly what TS predicts should produce structured, inspectable reasoning.

**The invariant:** Token pairs are always scored independently. No position is suppressed because another scored higher. The model learns which constraints matter without being forced to forget other constraints in the same step.

---

## What we know from completed experiments

### Mechanism validation (Exp 1 — 1.1M params)
Sigmoid tension matches transformer at identical parameter count (val PPL 57.7 vs 57.8). The mechanism works.

### TS-native objectives (Exp 2 — 13.5M params)
Constraint consistency + tension entropy losses cost 1.31 PPL but produce structurally coherent constraint graphs. Transitivity chains are directly visible in the tension field. Coherent text produces +25% mean τ and +60% more active edges than word salad — the graph measurably responds to coherence exactly as TS predicts.

### Curriculum training (Exp 3 — 13.5M params)
Logic → language → maths. First-contact maths PPL: cold start ~2293, logic only ~1076, logic+language ~582. 4× better than cold start. Curriculum validated.

### 117M curriculum run (Exp 4)
- Stage 1 logic (200M tokens): val PPL 5.10
- Stage 2 language FineWeb-Edu (500M tokens): val PPL 339
- Stage 3 maths open-web-math (2B tokens): best val PPL **359.99 at step 14,000**, min train PPL **6.8**
- First-contact train PPL: **24** vs ~2293 cold start — **96× better**

### Formal reasoning eval (Exp 5 — 23 questions, step 14k)
- Overall: **43.5%** — algebra 67%, calculus 50%, arithmetic 50%, transitivity 33%, syllogisms 17%
- Critical finding: **step 14,000 beats the final checkpoint** — epoch 2 of maths data partially overwrites stage 1 logic structure. Any new training must prevent this.

### logic_mix=0.10 preliminary (Exp 6 — 117M, abbreviated 2-stage run, 2026-04-18)
- **Run shape:** Stage 1 (200M synthetic logic, vocab=32768, from scratch, best val 9.56 @ step 500) → Stage 3 (1.1B open-web-math + `--logic_mix 0.10`, resumed from Stage 1 step-500). Stage 2 / ProofPile **skipped**. Vocab reused 117M-curriculum's 32768 tokenizer rather than the planned GPT-2 50257 — not a clean test of the full plan.
- **Final val ppl 317.96 / best 317.52** at step 16500. Monotone descent 1365 → 317 with no step-14k-style regression.
- **formal_eval:** 4/23 (17.4%) @ temp 0.1; 3/23 (13%) @ temp 0.3. Same overall rate as the forgotten 117M-curriculum baseline (3/23) despite better val ppl.
- **Signal:** transitivity 2/3 (67%) — logic templates measurably persist under logic_mix=0.10. Qualitative difference from forgotten baseline — coherent math English, no repetition loops, topical near-misses (e.g. Pythagorean → "hypoteniori").
- **What logic_mix=0.10 did NOT do:** induce arithmetic / calculus / algebra / definitional reasoning (all 0%). Stage 1 synthetic logic is too narrow a prior on its own; without Stage 2 (ProofPile) the math-language substrate is thin, and 1.1B tokens of open-web-math is a fraction of the planned 5B.
- **Takeaway:** logic_mix works as a forgetting-preventer on the data it was taught. Does not substitute for Stage 2 or for scale. Next run should include ProofPile and the planned 5B budget before judging the ratio.

### Tension field (117M)
- `Manchester:0.95 United:0.95` simultaneously — impossible in softmax
- Syllogism: both logical subjects at equal full strength simultaneously
- Head specialisation (syntactic / long-range semantic / diffuse) emerges without supervision

### Replication at 117M (Phase 1.2 sidebar)
The Exp 2 coherence response replicates qualitatively on 117M-Curriculum (10/10 coherent prompts > word salad on mean τ) but at ~half the magnitude (1.056× vs published 1.25× on 13.5M). Published numbers are regime-specific; the structural signal is present but weaker at 117M, which disciplines our claims about raw-τ coherence on the larger model.

---

## Completed phases

| Phase | Status | Key result |
|-------|--------|-----------|
| Proof of concept | ✓ | Mechanism validated at 1.1M |
| Baseline comparison | ✓ | PPL 57.7 vs 57.8 |
| TS-native objectives | ✓ | Coherent constraint graphs, +25% τ density |
| Curriculum training 13.5M | ✓ | 4× first-contact improvement |
| Architecture upgrades | ✓ | RoPE, tau-mass norm, global layers, Triton |
| 117M curriculum run | ✓ | Train PPL 6.8, formal eval 43.5% |
| Efficiency fixes | ✓ | ~2-3× training throughput on 2× 4090 |
| Kernel unblock (Phase 0) | ✓ | Fused Triton emits τ directly; 350M fits in 11 GB |

---

## Current phase — Math + Code Reasoning Model

**Target:** a TensionLM model that demonstrably outperforms an equivalent transformer on mathematical and code reasoning, with interpretable constraint structure as a first-class output.

### Architecture changes from 117M baseline

| Parameter | 117M baseline | New run |
|-----------|--------------|---------|
| vocab_size | 32,768 | 50,257 (GPT-2 tokenizer) |
| window W | 64 | 256 |
| global_every | 4 | 3 |
| logic_mix in stage 3 | 0 | 0.10 |
| max_seq_len | 1024 | 2048 |

**Why these changes:**
- vocab=50k: GPT-2 tokenizer is well-tested, compatible with existing maths and code corpora tokenisation
- W=256: Proofs and multi-function code regularly require attending back 500+ tokens. W=64 was the primary architectural bottleneck for real formal reasoning.
- global_every=3: More frequent long-range passes without O(T²) cost everywhere
- logic_mix=0.10: Directly addresses the catastrophic forgetting finding. 10% logic data throughout stage 3 keeps constraint structure active.

### Data plan

| Stage | Dataset | Tokens | Purpose |
|-------|---------|--------|---------|
| 1 — Logic | Synthetic inference (existing `data/logic-stage1`) | 200M | Load constraint structure |
| 2 — Formal language | Lean4 proofs + ProofPile + ArXiv maths abstracts | 500M | Mathematical language without notation overload |
| 3 — Math + code | open-web-math + MATH + The Stack (Python, Lean, Coq) + logic_mix=0.10 | 5B | Formal reasoning across maths and code; logic mix prevents forgetting |

**Why ProofPile for stage 2:** Formal proofs have explicit constraint chains — every step references previous steps. This is ideal TS-aligned data: maximally consistent, maximally structured. Better than FineWeb-Edu which mixes in general educational text.

**Why code in stage 3:** Code is formally verifiable constraint-chain data at scale. Python typing, Lean/Coq proofs, and well-typed codebases are dense in the kind of dependency structure sigmoid-tension attention is designed to represent. The mechanism should show particular advantages on code once enough is mixed in.

### Training plan

```bash
# Stage 1 — logic (reuse existing checkpoint if available)
torchrun --nproc_per_node=2 train.py \
  --data_dir data/logic-stage1 \
  --train_tokens 200_000_000 \
  --preset large \
  --window 256 \
  --max_seq_len 2048 \
  --vocab_size 50257 \
  --w_consistency 0.1 --w_entropy 0.05 \
  --global_every 3 \
  --out_dir checkpoints/math_stage1

# Stage 2 — formal language
torchrun --nproc_per_node=2 train.py \
  --data_dir data/proofpile \
  --train_tokens 500_000_000 \
  --preset large \
  --window 256 \
  --max_seq_len 2048 \
  --vocab_size 50257 \
  --global_every 3 \
  --resume --out_dir checkpoints/math_stage2

# Stage 3 — full maths + code with logic mixing
torchrun --nproc_per_node=2 train.py \
  --data_dir data/open-web-math \
  --train_tokens 5_000_000_000 \
  --preset large \
  --window 256 \
  --max_seq_len 2048 \
  --vocab_size 50257 \
  --global_every 3 \
  --logic_mix 0.10 \
  --logic_dir data/logic-stage1 \
  --w_consistency 0.05 \
  --resume --out_dir checkpoints/math_stage3
```

### Evaluation

Beyond perplexity, the model will be evaluated on:

1. **Extended formal reasoning benchmark** — expand the existing 23-question set to 100+ questions across: syllogisms, arithmetic, algebra, calculus, linear algebra, proof by contradiction, induction
2. **MATH dataset subset** — 500 problems across difficulty levels 1-5
3. **HumanEval / MBPP** — Python code completion as the first code-reasoning signal
4. **Tension field quality** — constraint transitivity score, head specialisation index, coherence ratio (coherent text τ density / random τ density)
5. **Forgetting metric** — formal reasoning score at step 2k, 5k, 10k, 20k, 50k — track whether logic_mix prevents the step-14k degradation seen in the baseline run

---

## Biological-training track (parallel)

350M runs use the "biological" training machinery already in `train.py`:
- `--decouple_optim` — separate consolidation / plastic-window optimisers
- `--sparse_grad` — activity-gated weight updates
- `--sleep_every` — periodic consolidation passes
- `--ff_mode` — forward-forward contrastive objective alongside cross-entropy

These are TS-inspired weight-update operations (consolidation / contrast / plastic-window); ablations so far suggest they stabilise long training runs. Kept as a parallel experimental track.

**350M stage 1 (bio)** — relaunch with the Phase 0 kernel fix. Stage 1 on synthetic logic (200M tokens).
**350M stage 2 (bio)** — `run_stage2_350m.sh` — open-web-math 1.1B tokens with `--logic_mix 0.10` plus biological machinery.

---

## Next phase — Scale to 1B

After the math+code reasoning model is validated:

| Run | Params | Window | Tokens | Hardware | Est. cost |
|-----|--------|--------|--------|----------|-----------|
| Math+code reasoning | 117M | 256 | 5.7B | 2× 4090 | ~$0 (own) |
| Ablation: W=64 vs W=256 | 117M | both | 1B each | 2× 4090 | ~$50 |
| Scale-up | 1B | 512 | 100B | 8× A100 (vast.ai) | ~$3,000 |

---

## Open questions to answer with this run

1. ~~Does logic_mix=0.10 prevent the step-14k degradation?~~ **Partially answered (Exp 6):** logic_mix=0.10 *does* preserve the transitivity templates taught in Stage 1 (67% transitivity at val ppl 317), and the val curve is monotone over 1.1B tokens with no step-14k-style regression. It does *not* substitute for Stage 2 or for the full token budget — math categories stay at 0% without them.
2. Does W=256 meaningfully improve multi-step proof following vs W=64?
3. Does ProofPile as stage 2 (vs FineWeb-Edu) produce better constraint structure?
4. What is the correct logic_mix ratio — 0.10 preserves what's taught but may need to be higher (0.15 – 0.20) to carry weaker priors through a longer stage 3.
5. Does sigmoid tension show an outsized advantage on code vs prose, as the constraint-dependency hypothesis predicts?

---

## Paper outline

**Title:** TensionLM: Constraint Relaxation as a Mechanism for Mathematical and Code Reasoning

**Contribution claim:** A language model whose attention mechanism directly implements constraint graph relaxation produces interpretable, structured reasoning on formal domains — and the constraint graph itself is a first-class output, not a black-box byproduct.

**Sections:**
1. TS-inspired theory — constraint relaxation, why softmax is wrong for formal reasoning
2. Mechanism — sigmoid tension, tau-mass normalisation, global layers, Triton kernel
3. Exp 1 — mechanism validation (tension vs transformer at 1.1M)
4. Exp 2 — TS-native objectives (constraint consistency, tension field coherence)
5. Exp 3 — curriculum training (first-contact PPL improvement)
6. Exp 4 — 117M baseline run results
7. Exp 5 — math + code reasoning model results (this run)
8. Tension field analysis — transitivity chains, head specialisation, coherence vs salad
9. Discussion — catastrophic forgetting finding, logic_mix solution, scaling outlook

**Target:** arXiv cs.LG / cs.AI

---

## Optional track — Ecosystem integration

TensionLM's internal representation is already a weighted graph, which makes it an unusually clean target for coupling to an external constraint-graph system. The `ts_bridge/` package explores this as an optional integration track — it does **not** gate the main LLM roadmap.

| Sub-phase | Status | Deliverable |
|-----------|--------|-------------|
| 1.0 | ✓ 91f0068 | `ts_bridge/` scaffold — graph shape, `TauExporter`, head filter, JSON round-trip |
| 1.1 | ✓ ac213ee | Head-classifier recalibration (quantile-based role assignment over corpus stats) |
| 1.2 | ✓ 48487b2 | Corpus-level head profiling + 117M Exp 2 replication (1.056× coherence vs published 1.25×) |
| 1.3 | ✓ 55543bf | Corpus-derived edge-threshold quantiles |
| 1.5 | ✓ | `StreamingTauExporter` — per-step τ export during generation, parity with batch ingest (527 edges, max-weight Δ = 1.2e-7) |
| 2.0 | ✓ | `GraphBias` + `tau_bias: [B,T,W]` plumbed through `MultiHeadCausalTensionLayer`, `TensionBlock`, `TensionLM`. No-op invariant exact; responsiveness KL ≈ 1.7e-2; directional KL ≈ 2.6e-1 |
| 2.1 | ✓ | Closed-loop `biased_generate` — graph biases forward, forward updates graph. Diverges at token 1/20 under matched seeds on diagnostic checkpoint |
| 2.2a | ✓ 034a61d | `--export_mode {biased,unbiased,off}` on `closed_loop_generate` to decouple graph growth from bias feedback |
| 2.2b | ✓ e4a10d8 | Global-layer graph bias — `tau_bias_global: [B,T,T]` plumbed through `TensionLM.forward` |
| 2.2c | ✓ 03f050b | Triton-fused graph bias — kernel `Bias` pointer + `HAS_BIAS` constexpr, σ' recomputed from biased τ, no dBias grad |
| 2.2d | ✓ | α calibration on 117M-curriculum. Sweep: α∈{0.25,0.5,1,2,4,8,16} × {seed=42, seed=7} × 60 new tokens. Silent ≤ α=2; inflection at α≈4 (seed-dependent); α=8 reliably opens the loop (+9–19 edges, +0.026–0.073 mean-w vs unbiased-export), text coherent; α=16 degrades |
| 3 | — | Integration A/B vs `BoggersTheAI` stock surface on held-out QA (contradiction rate, trace confidence) |

This track is cleanly separable from the main training roadmap. Landing it widens what the surface-level research can inform later; not landing it doesn't invalidate the LLM work.

---

## File map

| File | Purpose |
|------|---------|
| `model.py` | TensionLM architecture, aux losses, generation, KV cache |
| `baseline.py` | Baseline transformer (identical API) |
| `train.py` | Training pipeline — DDP, token budget, logic mixing, biological machinery |
| `prepare_data.py` | Stream + tokenise large datasets into binary shards |
| `eval.py` | Perplexity evaluation |
| `generate.py` | Inference CLI — standard, anchored, cached generation |
| `formal_eval.py` | Formal reasoning benchmark (expand to 100+ questions) |
| `visualise.py` | Tension field inspection — heatmap, token, layers, stats |
| `compare.py` | Plot loss curves |
| `upload_hf.py` | Upload to HuggingFace Hub |
| `triton_tension/` | Fused Triton kernels (fwd + bwd) |
| `ts_bridge/` | Optional ecosystem-integration package (exporter, streaming, graph bias, closed-loop gen) |
