# TensionLM — Development Plan

## Goal

Release a GPT-2-scale language model (~117M parameters) built on the sigmoid tension mechanism instead of softmax attention. The release should be good enough that someone can clone the repo, run inference, and see that it works — with a clear explanation of why the architecture is different and why that matters.

---

## Current State

- Proof of concept complete: 1.1M param model trained on WikiText-2 (CPU, Pentium Silver N6000)
- Converges cleanly to val PPL ~60 over 10 epochs
- Architecture is clean and modular: `model.py`, `train.py`, `generate.py`
- BPE tokenizer (vocab=2048) trained from scratch on the dataset

The mechanism works. The question now is whether it scales.

---

## Milestone 1 — Prove It Scales (before GPU spend)

**Goal:** Validate that tension is competitive with a standard transformer at the same parameter count.

### Steps

1. **Train a baseline transformer** at identical config (dim=128, 4 layers, 4 heads, same data/tokenizer). This is the comparison that makes or breaks the scientific claim.
2. **Plot loss curves** side by side. If TensionLM matches or beats the transformer at the same param count, the core claim holds.
3. **Scale to 5M params** on the same CPU run — dim=256, 6 layers, window=16. Check that PPL continues to drop proportionally.

If scaling looks good, proceed to GPU training. If not, investigate why before spending money.

---

## Milestone 2 — GPU Training Setup

**Hardware:** Rent a GPU instance on Vast.ai or RunPod (RTX 3090/4090 or A100).

**Workflow:**
```bash
# On the GPU instance
npm install -g @anthropic-ai/claude-code   # Claude Code CLI
git clone https://github.com/BoggersTheFish/bozo.git
cd bozo
pip install -r requirements.txt
claude   # Claude works directly in the GPU terminal from here
```

**Before first GPU run, code needs:**
- `torch.compile()` enabled (broken on this CPU, will work on GPU with CUDA)
- `bf16` mixed precision training (`torch.autocast`)
- Gradient checkpointing enabled by default for large configs
- Larger vocab (32768 recommended at this scale)
- DataLoader with multiple workers instead of pre-tokenised tensor

**Estimated GPU training time (117M params, 10B tokens):**

| GPU | Est. time | Cost |
|-----|-----------|------|
| RTX 3090 (~$0.25/hr) | 35–50h | ~$10 |
| RTX 4090 (~$0.50/hr) | 15–20h | ~$10 |
| A100 (~$1.50/hr) | 6–8h | ~$10 |

TensionLM is ~1.5–2× cheaper to train than an equivalent transformer because windowed attention (O(T×W)) is much cheaper than full attention (O(T²)) at seq_len=1024.

---

## Milestone 3 — GPT-2 Scale Run

**Config:**
```python
vocab_size  = 32768
dim         = 768
num_layers  = 12
num_heads   = 12
window      = 64      # 64-token causal look-back
ffn_mult    = 3       # SwiGLU hidden = dim * 3
max_seq_len = 1024
```
This gives ~117M parameters.

**Dataset:** FineWeb (10B token sample) or OpenWebText — both available via HuggingFace datasets. Do not use WikiText-2 at this scale; it is too small to train a 117M param model meaningfully.

**Intermediate checkpoints:**
- 1B tokens — sanity check, generation quality
- 5B tokens — mid-run eval
- 10B tokens — full run, final release checkpoint

---

## Milestone 4 — Release

**What the release needs:**

- [ ] Updated README with architecture diagram and clear explanation of tension vs attention
- [ ] Model card: training data, parameter count, eval results, limitations
- [ ] Checkpoint hosted on HuggingFace Hub
- [ ] `generate.py` working out of the box from the checkpoint
- [ ] Perplexity reported on a standard benchmark (WikiText-103 test set) for comparison
- [ ] The baseline transformer result — so the claim "competitive with transformers" is backed by numbers

**What it does not need for v1:**
- Fine-tuning / instruction following
- A chat interface
- RLHF
- Anything beyond "this architecture trains and generates coherent text"

---

## Open Questions

1. **Does it scale?** The windowed context (window=64 at GPT-2 scale) means each layer only sees 64 tokens back. Long-range dependencies must emerge through depth. Does 12 layers of local tension produce comparable long-range coherence to full attention? This is the central empirical question.

2. **Optimal window size?** Larger window = more compute but better context. Needs ablation at the 5M scale before committing to the 117M run.

3. **Is the OscillatoryModulation carrying its weight?** It adds parameters and a sinusoidal positional signal. Worth ablating to see if it actually helps vs standard learned positional embeddings alone.

---

## File Map

| File | Purpose |
|------|---------|
| `tension_lm.py` | Original toy demo, 200-token corpus, word-level tokenizer |
| `model.py` | Clean architecture: TensionConfig, TensionLM, aux losses |
| `train.py` | Full pipeline: WikiText-2, BPE tokenizer, checkpointing |
| `generate.py` | Inference CLI with sampling controls |
| `test.py` | Early prototype / scratch file |
| `PLAN.md` | This file |
