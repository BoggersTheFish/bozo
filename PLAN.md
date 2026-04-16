# TensionLM — Development Plan

## What TensionLM is (and isn't)

TensionLM is the **TS-native language surface** for the Thinking System. Not a standalone LLM competing with transformers. Not a reasoning engine in its own right. A surface whose internal representation is already a constraint graph, so it can be coupled bidirectionally to the external graph that actually does the thinking.

Three-layer TS architecture:

| Layer | Role | Implementation |
|-------|------|----------------|
| **Substrate** | Constraint graph — where thinking happens | `TS-Core` / `UniversalLivingGraph`, wave-cycle propagation |
| **Surface** | Fluent language on top of settled graph state | **TensionLM** (this repo) — replaces generic LLM surface |
| **Grounding** | Closed-loop learning from high-confidence traces | QLoRA fine-tune in `BoggersTheAI/core/fine_tuner.py` |

## The thesis

> A language model whose attention mechanism is already a constraint graph, paired bidirectionally with an external constraint graph, eliminates the substrate-surface representational mismatch that forces traditional LLM+KG systems into hallucination.

The surface and the substrate speak the same representational language (sigmoid tension ≙ weighted edges). They are not a retriever bolted onto a decoder. They are one coherent graph running on two scales.

## What existing TensionLM experiments establish

Prior results (see README) validate that TensionLM is a **viable TS-native surface**:

- **Mechanism parity** (Exp 1, 1.1M) — sigmoid tension ties transformer PPL. The surface isn't broken.
- **Constraint graphs form** (Exp 2, 13.5M) — TS-native losses produce visible transitivity chains; coherent text produces +25% mean τ / +60% more active edges vs word salad. The τ field measurably responds to coherence.
- **Curriculum validates the surface ordering** (Exp 3, 4) — logic → language → maths produces 4× first-contact improvement at 13.5M, 96× at 117M. Building the graph *first*, language *second*, matches the TS doctrine (substrate precedes surface).
- **Constraint interference is real and predicted** (Exp 5) — step-14k outperforms the final checkpoint because epoch-2 dense maths constraints displace stage-1 logic structure. This is a direct TS prediction that got empirically confirmed.

What none of this establishes: whether coupling TensionLM to the TS-Core graph **actually improves the integrated system**. That's the next phase.

## Phase 0 — Kernel unblock (complete 2026-04-16)

Fused Triton kernel now optionally emits `tau[B,T,H,W]` directly, so `return_tensions=True` stays on the fused path instead of falling through to the unfold gather. 350M preset fits in 11 GB peak (previously OOM'd at 24 GB). All τ consumers downstream — aux losses, sparse-grad gate, FF goodness, and now graph export — are unblocked.

| Metric | Before | After |
|--------|--------|-------|
| 350M fwd+bwd peak | OOM (> 24 GB) | 10.9 GB |
| 350M + FF (double fwd) peak | OOM | 20.0 GB |
| Tau path memory (per layer, 350M dims) | 2.1 GB unfold gather | 126 MB fused |

## Phase 1 — Surface → Substrate export (current)

**Goal:** build a clean interface that turns TensionLM's internal tension field into weighted edges on a `UniversalLivingGraph`-shaped target, token by token during generation.

**Why first:** without this, everything about the integrated thesis is vapour. This is the shortest path from the current repo state to a verifiable integration artefact. Does *not* require TS-Core to be present — we target the graph shape, keep the implementation local, and make the shape trivially mergeable into TS-Core later.

**Deliverable:** `ts_bridge/` package in this repo:
- `ts_bridge/graph.py` — minimal graph shape compatible with `TS-Core/UniversalLivingGraph` (node: content, topics, activation, stability; edge: src, dst, weight, relation). Dict-backed for now.
- `ts_bridge/export.py` — `TauExporter` that hooks into TensionLM generation and emits edges per token.
- `ts_bridge/head_filter.py` — head-specialisation filter; only layers/heads classified long-range-semantic produce graph edges (syntactic-tracking heads are noise at the graph level).
- `ts_bridge/schema.md` — the interface contract. Once stable, this is what goes into TS-Core.

**Edge extraction:**
- For each generated token `t`, collect τ[layer, head, t, w] across all layers.
- Filter to heads tagged `long_range_semantic` by the head-specialisation classifier (from `visualise.py`).
- Aggregate: `edge_weight(t, t-w-1) = mean over selected heads of τ[·, ·, t, w]`.
- Threshold: emit edge only when aggregated weight > 0.3 (keeps the graph sparse).
- Node content = decoded token (pending a concept-extraction pass later).
- `stability` seeded at 0.5; `activation` seeded at the emitted-token log-probability.

**Validation:**
- Smoke test: generate a logical sentence ("If all mammals are warm-blooded and all whales are mammals then"); the exported subgraph should show whales ← mammals ← warm-blooded transitivity at layer ≥ 5.
- Coherence test: coherent text should produce a denser, more stable subgraph than random word salad under the same token budget. (Mirrors the Exp 2 coherence finding, now as an integrated-system diagnostic.)
- Round-trip: serialise exported graph to the `UniversalLivingGraph` JSON format, load into a TS-Core-shaped validator, confirm it round-trips.

**Out of scope for Phase 1:** the reverse direction (graph → LLM), concept extraction, deduplication across generations, writing directly into a live TS-Core instance.

## Phase 2 — Substrate → Surface biasing

**Goal:** let the graph bias TensionLM's generation — so the substrate's current state *directly enters the attention computation*, not via prompt injection.

**Mechanism:** at generation time, look up the local subgraph around recently emitted tokens. For each (query_tok, key_tok) pair, if a graph edge exists between the corresponding concepts with weight `e`, add `α · e` to the τ precursor logit before the sigmoid. Tunable α (start 0.5).

**Why this and not RAG:** RAG concatenates retrieved text into the prompt; the LLM has to re-derive the constraint structure from text. Graph biasing injects the constraint structure **directly into the attention mechanism**. The surface and substrate are structurally coupled, not stapled together.

**Deliverable:** `ts_bridge/bias.py` + a generation wrapper that takes a `UniversalLivingGraph` and biases `generate.py`.

**Validation:**
- Contradiction test: seed the graph with a known contradiction (e.g. "whales are fish" at high weight), generate completion. Biased generation should avoid or flag the contradiction; unbiased generation should not.
- A/B on short completions with/without graph bias, measured against human-labelled coherence.

## Phase 3 — Integration A/B (the real paper experiment)

**Setup:**
- **System A (control):** `BoggersTheAI` stock — Ollama/Llama-3.2 surface, `UniversalLivingGraph` substrate, unchanged pipeline.
- **System B (test):** same `BoggersTheAI` pipeline, TensionLM-117M-Curriculum as surface, bidirectional τ ↔ edge coupling via `ts_bridge`.

**Question:** does the TS-native surface produce a better-behaved integrated system?

**Metrics:**
| Axis | How measured |
|------|--------------|
| Answer correctness | Held-out QA set spanning graph domains (100 prompts, human-graded) |
| Contradiction rate | Fraction of answers containing statements contradicted by the graph at query time |
| Graph coherence | Mean tension trajectory over 1 h idle operation; lower = more settled |
| Trace confidence distribution | Histogram of per-trace confidence; System B should skew higher if surface/substrate agree |
| Closed-loop improvement | After N QLoRA cycles, does answer correctness improve monotonically? |

**Hypothesis:** System B produces fewer contradictions and a more-settled graph, not because TensionLM is a better LLM, but because its internal representation and the external graph do not need to be reconciled through a lossy natural-language bottleneck.

**Failure outcome is acceptable.** If System A ties or beats System B, that's the falsifiable result and we learn something important before further investment.

## Phase 4 — Multi-agent coherence (Wave 16)

TS is already on Wave 16 (multi-agent coordination). Two agents each with a TS-native surface should communicate **more coherently** than two agents with generic LLM surfaces — because their shared medium (the graph) and their private representations (τ fields) are structurally identical.

**Setup:** two `BoggersTheAI` instances, each with TensionLM surface + shared graph. Compare against two instances with Llama surfaces.

**Metric:** after N rounds of communication on an open-ended task, does the shared graph converge (low inter-agent tension) under TensionLM faster than under Llama?

This is the claim that's genuinely novel at the system level and publishable on its own merits. It depends on Phases 1–3 landing first.

## Parallel track — keep training a coherent surface

TensionLM needs to be a *competent* surface for any of Phases 1–4 to matter. Keep the 117M-Curriculum and 350M runs alive in parallel:

**350M stage 1 (bio)** — relaunch with the Phase 0 kernel fix. Stage 1 on synthetic logic (200M tokens), vocab=16384.

**350M stage 2 (bio)** — `run_stage2_350m.sh` — open-web-math 1.1B tokens with `--logic_mix 0.10` (catastrophic-forgetting fix from Exp 5) + the biological training machinery (`--decouple_optim`, `--sparse_grad`, `--sleep_every`, `--ff_mode`). These are not gratuitous LLM tricks; they are TS-cycle operations applied to weight updates (consolidation / contrastive constraints / plastic-window updates).

**117M-Curriculum** — the checkpoint that already hit formal_eval 43.5% at step 14k is the default surface for Phase 1 integration smoke tests. Don't block on 350M to start Phase 1.

## Paper restructure

Retiring: *"TensionLM: Constraint Relaxation as a Mechanism for Mathematical Reasoning"*

Target: *"Substrate-Surface Coherence in a Graph-Native Language Model: Eliminating Representational Mismatch in LLM+Graph Systems"*

**Sections:**
1. The substrate-surface mismatch in LLM+KG systems (motivation)
2. TS theory — computation as constraint relaxation; shared representation across scales
3. TensionLM as TS-native surface — sigmoid tension, tau-mass, global layers, Triton kernel
4. The τ ↔ edge bridge (Phase 1, 2) — interface, head filtering, bias injection
5. Prior results reframed as surface-viability evidence — Exps 1–5 from the current README, positioned as "the surface can carry graph structure" not "the surface beats softmax at math"
6. Integration A/B (Phase 3) — the real experiment
7. Multi-agent coherence (Phase 4) — Wave-16-aligned result
8. Discussion — scaling outlook, failure modes, open questions

**Target venue:** arXiv cs.LG / cs.AI (same as before); the narrative is cleaner and the claim more novel.

## Graph consolidation (prerequisite debt)

Four separate graph implementations currently exist across the ecosystem: TS-Core, BoggersTheAI, BoggersTheMind, BoggersThePulse. The interface spec produced in Phase 1 is the right time to consolidate — pick TS-Core's `UniversalLivingGraph` as source of truth, have all other repos import it. This is tracked but not in the critical path of the phases above; it's a quality-of-life fix that becomes more valuable as the integration work lands.

## Open questions

1. How much of TensionLM's τ field is signal at the graph scale? Head-specialisation analysis suggests 2-3 heads per layer are long-range-semantic; the rest are syntactic noise. Phase 1's head filter is the lever.
2. What's the right node granularity? Tokens are too fine; concepts from a topic-extraction pass may be the right target (matches `BoggersTheAI` query pipeline's topic extraction).
3. Does graph biasing need to be per-layer (bias every layer's τ) or only at the global-attention layers? The global layers are the long-range-coupling points — biasing only there may suffice and is cheaper.
4. Does logic_mix=0.10 actually prevent step-14k degradation in the new run? Needed for a coherent surface; answered by the 350M stage 3 run when it lands.
5. Is 117M enough for integrated Phase 3 to show a real delta, or do we need 350M as the minimum viable surface?

## Current repo layout

| File | Role |
|------|------|
| `model.py` | TensionLM architecture, aux losses, KV cache generation |
| `baseline.py` | Baseline transformer (kept for Phase 3 ablation cross-reference) |
| `train.py` | Training pipeline — DDP, token budget, logic mixing, biological training machinery |
| `prepare_data.py` | Streaming shard prep |
| `eval.py` / `formal_eval.py` | Standalone-surface eval (not the main target anymore, but retained) |
| `generate.py` | Inference CLI — standard, anchored, cached generation |
| `visualise.py` | Tension field inspection — inputs for Phase 1 head filtering |
| `triton_tension/` | Fused kernels (Phase 0 — tau output now supported) |
| `ts_bridge/` | **New (Phase 1)** — surface ↔ substrate interface |
