# TensionLM — Development Plan

## Goal

Build a mathematical reasoning model using sigmoid tension as the core mechanism. The architecture implements Thinking System (TS) theory: computation as constraint relaxation over a graph of interdependent states. Formal mathematics is the ideal domain — contradictions cannot exist by construction, so the constraint graph built during training is coherent by definition.

**Why mathematics:** General web text creates irresolvable constraint loops (contradictory documents). Formal domains have enforced consistency. A model trained on formal data builds a constraint graph that is coherent by construction — exactly what TS predicts should produce structured, inspectable reasoning.

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

### Tension field (117M)
- `Manchester:0.95 United:0.95` simultaneously — impossible in softmax
- Syllogism: both logical subjects at equal full strength simultaneously
- Head specialisation (syntactic / long-range semantic / diffuse) emerges without supervision

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

---

## Current phase — Mathematical Reasoning Model

**Target:** A TensionLM model that demonstrably outperforms an equivalent transformer on mathematical reasoning, with interpretable constraint structure as a first-class output.

### Architecture changes from 117M baseline

| Parameter | 117M baseline | New run |
|-----------|--------------|---------|
| vocab_size | 32,768 | 50,257 (GPT-2 tokenizer) |
| window W | 64 | 256 |
| global_every | 4 | 3 |
| logic_mix in stage 3 | 0 | 0.10 |
| max_seq_len | 1024 | 2048 |

**Why these changes:**
- vocab=50k: GPT-2 tokenizer is well-tested, compatible with existing maths corpora tokenisation
- W=256: Proofs regularly require attending back 500+ tokens. W=64 was the primary architectural bottleneck for real mathematical reasoning.
- global_every=3: More frequent long-range passes without O(T²) cost everywhere
- logic_mix=0.10: Directly addresses the catastrophic forgetting finding. 10% logic data throughout stage 3 keeps constraint structure active.

### Data plan

| Stage | Dataset | Tokens | Purpose |
|-------|---------|--------|---------|
| 1 — Logic | Synthetic inference (existing data/logic-stage1) | 200M | Load constraint structure |
| 2 — Formal language | Lean4 proofs + ProofPile + ArXiv maths abstracts | 500M | Mathematical language without notation overload |
| 3 — Full maths | open-web-math + MATH dataset + logic_mix=0.10 | 5B | Mathematical reasoning; logic mix prevents forgetting |

**Why ProofPile for stage 2:** Formal proofs have explicit constraint chains — every step references previous steps. This is ideal TS data: maximally consistent, maximally structured. Better than FineWeb-Edu which mixes in general educational text.

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

# Stage 3 — full maths with logic mixing
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
3. **Tension field quality** — constraint transitivity score, head specialisation index, coherence ratio (coherent text τ density / random τ density)
4. **Forgetting metric** — formal reasoning score at step 2k, 5k, 10k, 20k, 50k — track whether logic_mix prevents the step-14k degradation seen in the baseline run

---

## Next phase — Scale to 1B

After the mathematical reasoning model is validated:

| Run | Params | Window | Tokens | Hardware | Est. cost |
|-----|--------|--------|--------|----------|-----------|
| Math reasoning | 117M | 256 | 5.7B | 2× 4090 | ~$0 (own) |
| Ablation: W=64 vs W=256 | 117M | both | 1B each | 2× 4090 | ~$50 |
| Scale-up | 1B | 512 | 100B | 8× A100 (vast.ai) | ~$3,000 |

---

## Open questions to answer with this run

1. Does logic_mix=0.10 prevent the step-14k degradation? (Track formal eval every 5k steps)
2. Does W=256 meaningfully improve multi-step proof following vs W=64?
3. Does ProofPile as stage 2 (vs FineWeb-Edu) produce better constraint structure?
4. What is the correct logic_mix ratio — does 0.10 over-constrain or under-constrain?

---

## Paper outline

**Title:** TensionLM: Constraint Relaxation as a Mechanism for Mathematical Reasoning

**Contribution claim:** A language model whose attention mechanism directly implements constraint graph relaxation produces interpretable, structured reasoning on formal mathematics — and the constraint graph itself is a first-class output, not a black-box byproduct.

**Sections:**
1. TS theory — constraint relaxation, why softmax is wrong for formal reasoning
2. Mechanism — sigmoid tension, tau-mass normalisation, global layers, Triton kernel
3. Exp 1 — mechanism validation (tension vs transformer at 1.1M)
4. Exp 2 — TS-native objectives (constraint consistency, tension field coherence)
5. Exp 3 — curriculum training (first-contact PPL improvement)
6. Exp 4 — 117M baseline run results
7. Exp 5 — mathematical reasoning model results (this run)
8. Tension field analysis — transitivity chains, head specialisation, coherence vs salad
9. Discussion — catastrophic forgetting finding, logic_mix solution, scaling outlook

**Target:** arXiv cs.LG / cs.AI

---

## File map

| File | Purpose |
|------|---------|
| `model.py` | TensionLM architecture, aux losses, generation, KV cache |
| `baseline.py` | Baseline transformer (identical API) |
| `train.py` | Training pipeline — DDP, token budget, logic mixing |
| `prepare_data.py` | Stream + tokenise large datasets into binary shards |
| `eval.py` | Perplexity evaluation |
| `generate.py` | Inference CLI — standard, anchored, cached generation |
| `formal_eval.py` | Formal reasoning benchmark (expand to 100+ questions) |
| `visualise.py` | Tension field inspection — heatmap, token, layers, stats |
| `compare.py` | Plot loss curves |
| `upload_hf.py` | Upload to HuggingFace Hub |
| `triton_tension/` | Fused Triton kernels (fwd + bwd) |
