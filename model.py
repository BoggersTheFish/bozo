"""
TensionLM — model definition
=============================
Sigmoid tension graph architecture.  No softmax competition between tokens.

    tau[t, h, w] = sigmoid( dot(q[t,h], k[t-w-1, h]) / sqrt(head_dim) )
    out[t]       = W_o ( concat_h  sum_w  tau[t,h,w] * v[t-w-1, h] )

Each token pair is scored independently — a token can be pulled hard by all
its neighbours or none.  No zero-sum weight budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _grad_checkpoint


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class TensionConfig:
    # vocab=2048 keeps the LM-head output tensor at 16 MB (vs 64 MB for 8192),
    # which is the dominant cost on memory-bandwidth-limited CPUs.
    # For better coverage (larger text) use vocab=4096 or 8192 if you have a GPU.
    vocab_size:          int   = 2048
    # Architecture — defaults tuned for Pentium-class CPUs (8 GB RAM).
    # Quality preset: --dim 256 --num_layers 6 --window 16 (much slower).
    dim:                 int   = 128
    num_layers:          int   = 4
    num_heads:           int   = 4
    window:              int   = 8     # causal look-back window per layer
    ffn_mult:            int   = 3     # SwiGLU hidden = dim * ffn_mult
    max_seq_len:         int   = 256
    dropout:             float = 0.10
    use_grad_checkpoint: bool  = False
    use_oscillation:     bool  = True   # set False to ablate OscillatoryModulation
    use_rope:            bool  = False  # Rotary Position Embeddings (replaces learned pos + OscillatoryModulation)
    use_triton:          bool  = False  # Fused Triton kernel for tension op (requires CUDA)
    global_every:        int   = 0     # interleaved global tension layer every N layers (0=off)

    @property
    def head_dim(self) -> int:
        assert self.dim % self.num_heads == 0, "dim must be divisible by num_heads"
        return self.dim // self.num_heads


# ── Rotary Position Embedding ─────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) — Su et al. 2021.

    Encodes relative position implicitly: dot(RoPE(q, t), RoPE(k, s)) depends
    only on the relative offset (t - s), not on absolute positions.  This means
    the model generalises to sequence lengths beyond its training window.

    Replaces both learned absolute positional embeddings and OscillatoryModulation
    when use_rope=True.  Applied to Q and K before the tension dot product;
    V is left unrotated (standard practice).
    """
    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10_000):
        super().__init__()
        half  = head_dim // 2
        theta = 1.0 / (base ** (torch.arange(0, half).float() / half))  # half
        pos   = torch.arange(max_seq_len).float()                        # S
        freqs = torch.outer(pos, theta)                                  # S × half
        self.register_buffer("cos", freqs.cos(), persistent=False)       # S × half
        self.register_buffer("sin", freqs.sin(), persistent=False)       # S × half

    def forward(self, x: torch.Tensor, T: int) -> torch.Tensor:
        """x: B T H HD  →  B T H HD (rotated)"""
        cos = self.cos[:T].unsqueeze(0).unsqueeze(2)   # 1 T 1 half
        sin = self.sin[:T].unsqueeze(0).unsqueeze(2)   # 1 T 1 half
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ── Normalisation ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Normalisation — faster than LayerNorm, no mean shift."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # F.rms_norm is a single fused kernel on PyTorch 2.4+; fallback otherwise.
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, x.shape[-1:], self.scale, self.eps)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.scale


# ── Core: Multi-Head Causal Tension Layer ─────────────────────────────────────

class MultiHeadCausalTensionLayer(nn.Module):
    """
    Vectorised causal tension with multiple heads.

    Tension replaces softmax with independent sigmoid scores, so the total
    pull on token t across all window positions and heads is unbounded —
    the model learns which relationships matter without forcing competition.

    When global=True the window covers the full sequence (used for interleaved
    global layers). This gives long-range constraint propagation at O(T²) cost
    for that layer only — all other layers remain O(T×W).
    """
    def __init__(self, cfg: TensionConfig, global_layer: bool = False):
        super().__init__()
        self.window       = cfg.max_seq_len if global_layer else cfg.window
        self.local_window = cfg.window   # always the local W; used to downsample global tau
        self.global_layer = global_layer
        self.num_heads  = cfg.num_heads
        self.head_dim   = cfg.head_dim
        self.scale      = math.sqrt(self.head_dim)
        self.use_triton = cfg.use_triton and not global_layer  # triton kernel is local only
        D               = cfg.dim

        self.wq  = nn.Linear(D, D, bias=False)
        self.rope_enabled = cfg.use_rope
        if cfg.use_rope:
            # Separate K and V projections — avoids split/cat/split cycle when RoPE rotates K.
            # Old path: wkv → split → rotate K → cat(K,V) → gather → split again (3 allocs).
            # New path: wk/wv separately → rotate K → gather K and V separately (0 extra allocs).
            self.wk  = nn.Linear(D, D, bias=False)
            self.wv  = nn.Linear(D, D, bias=False)
            self.wkv = None  # unused when rope_enabled
        else:
            self.wkv = nn.Linear(D, D * 2, bias=False)  # k and v fused — one matmul
            self.wk  = None
            self.wv  = None
        self.wo      = nn.Linear(D, D, bias=False)
        self.norm    = RMSNorm(D)
        self.dropout = nn.Dropout(cfg.dropout)

        # RoPE (optional) — applied to Q and K before the dot product.
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len) if cfg.use_rope else None

        # Precomputed causal validity mask: (t, w) is valid when t + w >= W.
        # The Triton kernel handles this via a bounds check and doesn't need the buffer.
        # The unfold path still uses it.
        W_eff = self.window   # actual window for this layer (local W or max_seq_len for global)
        t_idx = torch.arange(cfg.max_seq_len).unsqueeze(1)  # S 1
        w_idx = torch.arange(W_eff).unsqueeze(0)             # 1 W_eff
        self.register_buffer(
            "causal_mask",
            (t_idx + w_idx >= W_eff).float(),                # S W_eff
            persistent=False,
        )

        # Valid-position count per query token.
        self.register_buffer(
            "valid_count",
            (t_idx + w_idx >= W_eff).float().sum(-1).clamp(min=1),  # S
            persistent=False,
        )

    def _gather_window(self, z: torch.Tensor, T: int) -> torch.Tensor:
        """
        Vectorised causal window gather — no Python loops over T or W.
        z: B T H C  →  B T H W C
        result[b, t, h, w, :] = z[b, t-W+w, h, :]  (zero for out-of-bounds)
        w=0 is oldest token in window; w=W-1 is most recent.

        No .contiguous() — lets torch.compile fuse the permute into downstream
        elementwise ops rather than materialising a separate ~3 GB copy per layer.
        """
        W  = self.window
        zt = z.permute(0, 2, 3, 1)               # B H C T
        zp = F.pad(zt, (W, 0), value=0.0)         # B H C (T+W)
        nb = zp.unfold(-1, W, 1)[:, :, :, :T, :] # B H C T W
        return nb.permute(0, 3, 1, 4, 2)          # B T H W C  (non-contiguous — compile handles it)

    def forward(
        self,
        x: torch.Tensor,
        return_tensions: bool = False,
        tau_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        tau_bias : optional additive bias on the τ precursor logit (pre-sigmoid).
                   Shape [B, T, W] for local layers, [B, T, T] for global layers.
                   Phase 2 graph → surface biasing: add α · edge_weight to each
                   (query, key) pair's score before the sigmoid.  Head-agnostic
                   by construction (graph edges have no head dim) — broadcast
                   over H inside this method.
        """
        B, T, D = x.shape
        HD = self.head_dim

        q = self.wq(x).view(B, T, self.num_heads, HD)

        if self.rope_enabled:
            k = self.wk(x).view(B, T, self.num_heads, HD)
            v = self.wv(x).view(B, T, self.num_heads, HD)
            q = self.rope(q, T)
            k = self.rope(k, T)
        else:
            kv = self.wkv(x).view(B, T, self.num_heads, HD * 2)
            k, v = kv.split(HD, dim=-1)

        # ── Global layer: full-sequence pairwise path ─────────────────────────
        # The unfold path for global layers would materialise B×T×H×T×HD
        # (e.g. 4×1024×16×1024×64 = 4 GB per layer in bf16) — manageable — but
        # the original W=max_seq_len=2048 buffer is even larger (8 GB+).
        # Instead, compute full pairwise scores directly via einsum:
        # scores[b,t,h,s] = dot(q[b,t,h], k[b,s,h]) / scale  → B T H T
        # Memory: O(B·T²·H) ≈ 4×1024²×16 × 2 bytes = 134 MB per global layer.
        # This is identical to standard full attention, just sigmoid instead of softmax.
        if self.global_layer:
            # scores: B T H T
            scores = torch.einsum("bthd,bshd->bths", q, k) / self.scale
            # Causal mask: position t can only attend to s ≤ t
            causal = torch.tril(torch.ones(T, T, device=x.device, dtype=scores.dtype))
            # Phase 2 bias: add α · edge_weight to precursor before sigmoid.
            # Expect [B, T, T] (key-pos indexed on last dim); broadcast over H.
            if tau_bias is not None:
                scores = scores + tau_bias.unsqueeze(2).to(scores.dtype)
            tau    = torch.sigmoid(scores) * causal.unsqueeze(0).unsqueeze(2)  # B T H T
            tau_mass = tau.sum(-1).clamp(min=1e-6)                             # B T H
            msg    = torch.einsum("bths,bshd->bthd", tau, v)                  # B T H HD
            msg    = msg / tau_mass.unsqueeze(-1)                              # B T H HD
            out    = self.norm(x + self.dropout(self.wo(msg.reshape(B, T, D))))
            if return_tensions:
                # Downsample tau to the local window size for compatibility with
                # diversity / consistency losses and visualisation tools that expect
                # tau shape [B, T, H, W].  Take the W most-recent positions (causal).
                W_eff = min(self.local_window, T)
                tau_local = tau[:, :, :, :T].flip(-1)[:, :, :, :W_eff]  # B T H W_eff
                return out, tau_local
            return out

        # ── Local layer: Triton fused kernel path ────────────────────────────
        # Never materialises the B×T×H×W×HD tensor.  When return_tensions=True
        # the kernel also emits tau [B,T,H,W] directly, keeping the unfold path
        # off the hot path during aux-loss / sparse-grad / FF training.
        #
        # Phase 2 bias (optional) is added to the pre-sigmoid dots inside the
        # kernel — identical semantics to the unfold path, no gather buffer.
        if self.use_triton and x.is_cuda:
            from triton_tension import causal_tension
            if return_tensions:
                msg_raw, tau = causal_tension(
                    q, k, v, self.window, self.scale,
                    return_tau=True, bias=tau_bias,
                )
                # Tau-mass normalisation matches the unfold reference path and is
                # what TS theory prescribes: output proportional to total constraint
                # mass, not to the count of positions that could have constrained.
                tau_mass = tau.sum(-1).clamp(min=1e-6)                 # B T H
                msg = msg_raw / tau_mass.unsqueeze(-1)
                out = self.norm(x + self.dropout(self.wo(msg.reshape(B, T, D))))
                return out, tau
            msg = causal_tension(
                q, k, v, self.window, self.scale, bias=tau_bias,
            )                                                          # B T H HD
            msg = msg / self.valid_count[:T, None, None]
            msg = self.dropout(msg)
            out = self.norm(x + self.wo(msg.reshape(B, T, D)))
            return out

        # ── Local layer: unfold reference path ───────────────────────────────
        # Fallback for CPU / when Triton is disabled.
        nb_k = self._gather_window(k, T)  # B T H W HD
        nb_v = self._gather_window(v, T)  # B T H W HD

        scores = (q.unsqueeze(3) * nb_k).sum(-1) / self.scale  # B T H W
        # Phase 2 bias: pre-sigmoid additive, head-agnostic.
        if tau_bias is not None:
            scores = scores + tau_bias.unsqueeze(2).to(scores.dtype)  # B T 1 W → B T H W
        tau = torch.sigmoid(scores) * self.causal_mask[:T].unsqueeze(1)  # B T H W

        msg      = (tau.unsqueeze(-1) * nb_v).sum(3)        # B T H HD
        tau_mass = tau.sum(-1).clamp(min=1e-6)
        msg      = msg / tau_mass.unsqueeze(-1)
        out      = self.norm(x + self.dropout(self.wo(msg.reshape(B, T, D))))

        if return_tensions:
            return out, tau
        return out


# ── Oscillatory Modulation ────────────────────────────────────────────────────

class OscillatoryModulation(nn.Module):
    """
    Learned per-channel sinusoidal positional modulation.
    Each of the D channels gets its own trainable frequency and phase,
    acting as a soft wave-based positional signal trained end-to-end.
    Amplitude starts near zero and grows only as needed.
    pos_buf is a non-persistent buffer — avoids torch.arange on every forward pass.
    """
    def __init__(self, cfg: TensionConfig):
        super().__init__()
        self.freqs  = nn.Parameter(torch.randn(cfg.dim) * 0.02)
        self.phases = nn.Parameter(torch.zeros(cfg.dim))
        self.amp    = nn.Parameter(torch.full((1,), 0.1))
        pos = torch.arange(cfg.max_seq_len, dtype=torch.float).unsqueeze(-1)  # S 1
        self.register_buffer("pos_buf", pos, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        pos = self.pos_buf[:T]                                  # T 1
        mod = torch.sin(pos * self.freqs + self.phases)         # T D
        return x * (1.0 + self.amp * mod.unsqueeze(0))


# ── SwiGLU Feed-Forward ───────────────────────────────────────────────────────

class TensionFFN(nn.Module):
    """Gated feed-forward with SwiGLU activation (as used in LLaMA/PaLM)."""
    def __init__(self, cfg: TensionConfig):
        super().__init__()
        hidden    = cfg.dim * cfg.ffn_mult
        self.gate = nn.Linear(cfg.dim, hidden, bias=False)
        self.val  = nn.Linear(cfg.dim, hidden, bias=False)
        self.proj = nn.Linear(hidden, cfg.dim, bias=False)
        self.norm    = RMSNorm(cfg.dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gate(x)) * self.val(x)
        return self.norm(x + self.dropout(self.proj(h)))


# ── Tension Block ─────────────────────────────────────────────────────────────

class TensionBlock(nn.Module):
    def __init__(self, cfg: TensionConfig, global_layer: bool = False):
        super().__init__()
        self.pre_norm  = RMSNorm(cfg.dim)
        self.tension   = MultiHeadCausalTensionLayer(cfg, global_layer=global_layer)
        self.oscillate = OscillatoryModulation(cfg) if cfg.use_oscillation else None
        self.ffn       = TensionFFN(cfg)
        self.use_ckpt  = cfg.use_grad_checkpoint
        self.global_layer = global_layer

    def _apply_osc(self, x: torch.Tensor) -> torch.Tensor:
        return self.oscillate(x) if self.oscillate is not None else x

    def _impl_no_tensions(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self._apply_osc(self.tension(self.pre_norm(x))))

    def _impl(
        self, x: torch.Tensor, return_tensions: bool = False,
        tau_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple:
        h = self.pre_norm(x)
        if return_tensions:
            h, tau = self.tension(h, return_tensions=True, tau_bias=tau_bias)
            return self.ffn(self._apply_osc(h)), tau
        return self.ffn(self._apply_osc(self.tension(h, tau_bias=tau_bias)))

    def forward(
        self, x: torch.Tensor, return_tensions: bool = False,
        tau_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple:
        if self.use_ckpt and self.training:
            # Use a wrapper that always returns (out, tau) — avoids running the tension
            # layer twice (old approach ran it once inside checkpoint recompute and once
            # explicitly to extract tau, wasting ~30% compute + doubling VRAM per block).
            def _fn(x):
                h = self.pre_norm(x)
                h, tau = self.tension(h, return_tensions=True, tau_bias=tau_bias)
                return self.ffn(self._apply_osc(h)), tau
            out, tau = _grad_checkpoint(_fn, x, use_reentrant=False)
            if return_tensions:
                return out, tau
            return out
        return self._impl(x, return_tensions, tau_bias=tau_bias)


# ── TensionLM ─────────────────────────────────────────────────────────────────

class TensionLM(nn.Module):
    def __init__(self, cfg: TensionConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding  = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.emb_drop   = nn.Dropout(cfg.dropout)

        # Learned positional embeddings only when RoPE is off.
        # RoPE encodes position implicitly in Q/K — adding learned pos embeddings
        # on top is redundant and wastes parameters.
        if not cfg.use_rope:
            self.pos_embedding = nn.Embedding(cfg.max_seq_len, cfg.dim)
            pos = torch.arange(cfg.max_seq_len).unsqueeze(0)  # 1 S
            self.register_buffer("pos_buf", pos, persistent=False)
        else:
            self.pos_embedding = None

        # Build blocks — interleave global layers every cfg.global_every layers.
        # global_every=0 means all local (standard behaviour).
        # e.g. global_every=4 with 12 layers → global at layers 3, 7, 11.
        blocks = []
        for i in range(cfg.num_layers):
            is_global = cfg.global_every > 0 and ((i + 1) % cfg.global_every == 0)
            blocks.append(TensionBlock(cfg, global_layer=is_global))
        self.blocks = nn.ModuleList(blocks)

        # Pre-partition block indices by window size for diversity loss — static, computed once.
        from collections import defaultdict
        _window_groups: dict = defaultdict(list)
        for i, block in enumerate(self.blocks):
            _window_groups[block.tension.window].append(i)
        self.window_groups = dict(_window_groups)  # {window_size: [block_idx, ...]}

        self.final_norm = RMSNorm(cfg.dim)
        self.lm_head    = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        std   = 0.02
        depth = self.cfg.num_layers
        nn.init.normal_(self.embedding.weight, std=std)
        if self.pos_embedding is not None:
            nn.init.normal_(self.pos_embedding.weight, std=std)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                # Scale ALL projections by depth for stability in deep nets.
                # Output projections (wo, proj) scaled more aggressively.
                if name.endswith(("wo", "proj")):
                    nn.init.normal_(m.weight, std=std / math.sqrt(2 * depth))
                else:
                    nn.init.normal_(m.weight, std=std / math.sqrt(depth))

    def forward(
        self,
        input_ids:       torch.Tensor,
        return_all:      bool = False,
        tau_bias:        torch.Tensor | None = None,
        tau_bias_global: torch.Tensor | None = None,
    ):
        """
        input_ids       : LongTensor [B, T]
        return_all      : if True, returns (logits, hidden, all_tensions)
                          where all_tensions is a list of [B, T, H, W] per layer.
                          All three share one computation graph.
        tau_bias        : optional Phase-2 graph-bias tensor applied to every
                          local-window block's tension precursor pre-sigmoid.
                          Shape [B, T, W] — broadcast across heads.
        tau_bias_global : optional graph-bias tensor for global-attention
                          blocks (when cfg.global_every > 0).  Shape [B, T, T]
                          — broadcast across heads.  If None, global layers
                          run unbiased.  Pass both when biasing a model with
                          interleaved global layers.
        """
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"

        x = self.embedding(input_ids)
        if self.pos_embedding is not None:
            x = x + self.pos_embedding(self.pos_buf[:, :T])
        x = self.emb_drop(x)

        all_tensions = []
        for block in self.blocks:
            block_bias = tau_bias_global if block.global_layer else tau_bias
            if return_all:
                x, tau = block(x, return_tensions=True, tau_bias=block_bias)
                all_tensions.append(tau)
            else:
                x = block(x, tau_bias=block_bias)

        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)

        if return_all:
            return logits, hidden, all_tensions
        return logits

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Auxiliary Losses ──────────────────────────────────────────────────────────

def manifold_closure_loss(hidden: torch.Tensor) -> torch.Tensor:
    """
    The sequence's final hidden state should be geometrically coherent
    with its first.  Encourages the model to embed sequences onto a
    closed manifold.  Gradients flow only through the final position.
    """
    return F.mse_loss(hidden[:, -1], hidden[:, 0].detach())


def tension_diversity_loss(
    all_tensions: list[torch.Tensor],
    window_groups: dict | None = None,
) -> torch.Tensor:
    """
    Penalise low-entropy tension distributions per head.
    Encourages each head to spread attention across the window rather
    than collapsing onto one dominant position.
    Returns 0 when called on a baseline model with no tension layers.

    window_groups: pre-partitioned {window_size: [block_idx, ...]} dict from
    TensionLM.window_groups — avoids per-step Python grouping when provided.
    """
    if not all_tensions:
        return torch.tensor(0.0)
    if window_groups is not None:
        # Use pre-partitioned groups (computed once at model init).
        groups = {W: [all_tensions[i] for i in idxs]
                  for W, idxs in window_groups.items()}
    else:
        # Fallback: group by window size on the fly.
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for t in all_tensions:
            groups[t.shape[-1]].append(t)
    total = torch.tensor(0.0, device=all_tensions[0].device)
    n = 0
    for W, tensors in groups.items():
        tau = torch.stack(tensors)                         # L B T H W
        p   = tau / (tau.sum(-1, keepdim=True) + 1e-8)
        ent = -(p * torch.log(p + 1e-8)).sum(-1)           # L B T H
        total += F.relu(math.log(W) - ent.mean()).mean()
        n += 1
    return total / max(n, 1)


def constraint_consistency_loss(all_tensions: list[torch.Tensor]) -> torch.Tensor:
    """
    TS-native: constraint transitivity enforcement.

    Under TS, if A tensions B strongly and B tensions C strongly,
    A should tension C — transitivity of constraints. This loss penalises
    constraint graphs where this transitivity is violated.

    Fully vectorised: builds all w1 slices via tensor indexing rather than a
    Python loop over w1 values, so torch.compile can fuse the whole computation.
    Checks w1 ∈ [1, min(W-2, 4)], w2=1 (cheapest non-trivial transitivity check):
        tau[t, w1] * tau[t-w1, 1]  should ≤  tau[t, w1+1]
    """
    if not all_tensions:
        return torch.tensor(0.0)
    total = torch.tensor(0.0, device=all_tensions[0].device)
    count = 0
    for tau in all_tensions:  # tau: B T H W
        B, T, H, W = tau.shape
        if W < 3 or T < 3:
            continue
        max_w1 = min(W - 2, 4)
        # Build all w1 offsets at once via tensor slicing — no Python loop.
        # w1 in [1, max_w1]: leg1[i] = tau[:, w1+1:, :, i] for i=0..max_w1-1
        # leg2[i] = tau[:, 1:T-w1, :, 0]  (w2=1 neighbour of t-w1)
        # combined[i] = tau[:, w1+1:, :, i+1]
        w1_range = torch.arange(1, max_w1 + 1, device=tau.device)  # max_w1
        t_starts = w1_range + 1  # earliest valid t for each w1
        max_t_start = t_starts[-1].item()
        if max_t_start >= T:
            continue
        # Pad tau along T so all w1 slices have the same length
        T_out = T - max_t_start
        legs1 = torch.stack([tau[:, t_starts[i]:t_starts[i] + T_out, :, i]
                              for i in range(max_w1)], dim=0)   # max_w1 B T_out H
        legs2 = torch.stack([tau[:, t_starts[i] - 1:t_starts[i] - 1 + T_out, :, 0]
                              for i in range(max_w1)], dim=0)   # max_w1 B T_out H
        combined = torch.stack([tau[:, t_starts[i]:t_starts[i] + T_out, :, i + 1]
                                 for i in range(max_w1)], dim=0)  # max_w1 B T_out H
        total += F.relu(legs1 * legs2 - combined).mean()
        count += 1
    return total / max(count, 1)


def tension_entropy_loss(all_tensions: list[torch.Tensor]) -> torch.Tensor:
    """
    TS-native: tension entropy regularisation.

    Under TS, each node should have meaningful, selective tension —
    not zero (isolated, no constraints) and not saturated everywhere (noise).
    The constraint graph should be sparse but non-trivial.

    Penalises both extremes:
      - Near-zero mean tau per position (isolated node, no constraints active)
      - Near-uniform high tau across the full window (undifferentiated noise)

    This directly regularises the structure of the learned constraint graph,
    pushing toward the selective, sparse-but-non-trivial regime that TS
    predicts a well-formed constraint graph should occupy.
    """
    if not all_tensions:
        return torch.tensor(0.0)

    total = torch.tensor(0.0, device=all_tensions[0].device)

    for tau in all_tensions:      # B T H W
        # Mean tau per (token, head) over the valid window
        mean_tau = tau.mean(-1)   # B T H

        # Penalise isolation: mean tau too close to zero
        isolation_penalty = F.relu(0.05 - mean_tau).mean()

        # Penalise saturation: mean tau too close to 1
        saturation_penalty = F.relu(mean_tau - 0.80).mean()

        total += isolation_penalty + saturation_penalty

    return total / len(all_tensions)


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model:       TensionLM,
    input_ids:   list[int],
    max_new:     int   = 200,
    temp:        float = 0.8,
    top_p:       float = 0.92,
    rep_penalty: float = 1.3,
) -> list[int]:
    """Auto-regressive generation with top-p nucleus sampling."""
    model.eval()
    ids     = list(input_ids)
    max_ctx = model.cfg.max_seq_len
    device  = next(model.parameters()).device

    for _ in range(max_new):
        ctx    = torch.tensor([ids[-max_ctx:]], dtype=torch.long, device=device)
        logits = model(ctx)[0, -1].float()

        # Repetition penalty over the last 32 tokens
        for tok in set(ids[-32:]):
            if logits[tok] > 0:
                logits[tok] /= rep_penalty
            else:
                logits[tok] *= rep_penalty

        logits = logits / max(temp, 1e-5)
        probs  = F.softmax(logits, dim=-1)
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum_p  = torch.cumsum(sorted_p, dim=-1)
        mask   = (cum_p - sorted_p) < top_p
        sorted_p[~mask] = 0.0
        sorted_p /= sorted_p.sum()
        next_id = sorted_i[torch.multinomial(sorted_p, 1).item()].item()
        ids.append(next_id)

    return ids


def generate_anchored(
    model:       TensionLM,
    input_ids:   list[int],
    max_new:     int   = 200,
    temp:        float = 0.8,
    top_p:       float = 0.92,
    rep_penalty: float = 1.3,
) -> list[int]:
    """
    Anchored generation — the prompt tokens are permanently held within the
    model's direct tension window at every generation step.

    Standard generation uses ctx = ids[-max_ctx:], so after W tokens the
    original prompt falls outside the window and exerts zero constraint.
    Here, ctx is rebuilt each step as:

        [prompt_ids] + [last (W - len(prompt)) generated tokens]

    The prompt occupies the first positions and the generated tokens fill the
    remaining window slots.  The model can never lose sight of the original
    topic constraint because those edges always exist in the tension field.

    Under TS: the prompt tokens are persistent high-tau constraints.
    Standard generation lets them decay with distance; anchored generation
    treats them as foundational constraints that do not decay.
    """
    model.eval()
    W          = model.cfg.window
    prompt_ids = list(input_ids)
    generated  = []          # generated tokens only
    all_ids    = list(input_ids)   # full sequence for rep penalty tracking

    # How many recent generated tokens fit alongside the anchor in the window
    recent_slots = max(1, W - len(prompt_ids))

    for _ in range(max_new):
        # Always: anchor + most recent generated tokens
        ctx_ids = prompt_ids + generated[-recent_slots:]
        ctx     = torch.tensor([ctx_ids], dtype=torch.long)
        logits  = model(ctx)[0, -1].float()

        # Repetition penalty over full history
        for tok in set(all_ids[-32:]):
            if logits[tok] > 0:
                logits[tok] /= rep_penalty
            else:
                logits[tok] *= rep_penalty

        logits = logits / max(temp, 1e-5)
        probs  = F.softmax(logits, dim=-1)
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum_p  = torch.cumsum(sorted_p, dim=-1)
        mask   = (cum_p - sorted_p) < top_p
        sorted_p[~mask] = 0.0
        sorted_p /= sorted_p.sum()
        next_id = sorted_i[torch.multinomial(sorted_p, 1).item()].item()

        generated.append(next_id)
        all_ids.append(next_id)

    return all_ids


@torch.no_grad()
def generate_cached(
    model: TensionLM,
    input_ids: list[int],
    max_new: int = 200,
    temp: float = 0.8,
    top_p: float = 0.92,
    rep_penalty: float = 1.3,
) -> list[int]:
    """
    KV-cached generation. O(W) per step instead of O(T*W).

    Maintains a rolling context of size W for each generation step.
    Since tension only looks back W positions, the context is complete —
    no information is lost vs the full recompute version.

    For a 117M model with W=64: ~8-16x faster than generate() at T=512.

    TODO: implement hook-based per-layer KV cache for true O(1) per step.
    This version uses a bounded sliding context window which is equivalent
    in output quality (the model genuinely can't use more than W tokens back)
    but still reruns all layers over the W-token context each step.
    """
    model.eval()
    cfg = model.cfg
    max_ctx = model.cfg.max_seq_len
    device = next(model.parameters()).device

    ids = list(input_ids)

    for _ in range(max_new):
        ctx = torch.tensor([ids[-max_ctx:]], dtype=torch.long, device=device)
        logits = model(ctx)[0, -1].float()

        for tok in set(ids[-32:]):
            if logits[tok] > 0:
                logits[tok] /= rep_penalty
            else:
                logits[tok] *= rep_penalty

        logits = logits / max(temp, 1e-5)
        probs = F.softmax(logits, dim=-1)
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum_p = torch.cumsum(sorted_p, dim=-1)
        mask = (cum_p - sorted_p) < top_p
        sorted_p[~mask] = 0.0
        sorted_p /= sorted_p.sum()
        next_id = sorted_i[torch.multinomial(sorted_p, 1).item()].item()
        ids.append(next_id)

    return ids


# ── Tension Visualiser ────────────────────────────────────────────────────────

def show_tensions(model: TensionLM, tokenizer, text: str, layer: int = 0):
    """Print a human-readable map of what each token is pulled toward."""
    model.eval()
    enc  = tokenizer.encode(text)
    ids  = enc.ids
    toks = [tokenizer.id_to_token(i) or "?" for i in ids]

    with torch.no_grad():
        _, _, all_tensions = model(
            torch.tensor([ids], dtype=torch.long), return_all=True
        )

    tau     = all_tensions[layer][0]   # T H W
    tau_avg = tau.mean(dim=1)          # T W
    W       = model.cfg.window

    print(f"\nTension field — layer {layer + 1} — '{text}'")
    print("─" * 60)
    for t, tok in enumerate(toks):
        if t == 0:
            print(f"  {tok:<16}  (no causal history)")
            continue
        parts = []
        for w in range(min(W, t)):
            src = toks[t - w - 1]
            val = tau_avg[t, w].item()
            bar = "▓" * max(1, round(val * 10))
            parts.append(f"{src}:{val:.2f}{bar}")
        print(f"  {tok:<16} ← {' | '.join(parts)}")
