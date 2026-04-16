"""
Fused causal sigmoid tension kernels.

Forward: out[b,t,h] = Σ_w  sigmoid(dot(Q[t], K[t-(W-w)]) / scale) · V[t-(W-w)]
         for w in [0, W-1], skipping positions where t-(W-w) < 0

Optionally also emits tau[b,t,h,w] when HAS_TAU is true — lets callers use the
fused path with return_tensions=True instead of falling through to the unfold
path that materialises B×T×H×W×HD gather buffers (32 GB at 350M/W=256/seq=1024).

Backward: dQ, dK, dV via two kernels:
  - _bwd_dq_kernel: same loop structure as forward (Q-centric)
  - _bwd_dkv_kernel: reverse loop — for each K/V position, accumulate from
                     future query positions that looked at it
Both accept an optional dTau input — when HAS_DTAU is true, the gradient
flowing in through tau (from aux losses, sparse-grad gate, FF goodness) is
added to the d(dot) term before chain-ruling into Q and K.

Memory: O(B·T·H·HD) output + O(B·T·H·W) tau when requested.
Never materialises the O(B·T·H·W·HD) expanded window tensor.
"""

import triton
import triton.language as tl


# ── Forward ────────────────────────────────────────────────────────────────────

@triton.jit
def _fwd_kernel(
    Q, K, V, Out, Tau,
    T, H, W, scale,
    sq_b, sq_t, sq_h, sq_d,
    sk_b, sk_t, sk_h, sk_d,
    sv_b, sv_t, sv_h, sv_d,
    so_b, so_t, so_h, so_d,
    st_b, st_t, st_h, st_w,
    HD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_TAU: tl.constexpr,
):
    """One program per (batch, head, T-tile)."""
    pid    = tl.program_id(0)
    n_tile = tl.cdiv(T, BLOCK_T)
    bh     = pid // n_tile
    tile   = pid  % n_tile

    b  = bh // H
    h  = bh  % H
    t0 = tile * BLOCK_T

    t_off = t0 + tl.arange(0, BLOCK_T)   # [BLOCK_T]
    d_off = tl.arange(0, HD)              # [HD]
    t_ok  = t_off < T

    q_base = b * sq_b + h * sq_h
    q = tl.load(
        Q + q_base + t_off[:, None] * sq_t + d_off[None, :] * sq_d,
        mask=t_ok[:, None], other=0.0,
    ).to(tl.float32)

    acc = tl.zeros([BLOCK_T, HD], dtype=tl.float32)

    tau_base = b * st_b + h * st_h

    for w in range(W):
        # Source position: t - (W - w).  w=0 → oldest; w=W-1 → most recent past.
        ts      = t_off - (W - w)
        ts_safe = tl.maximum(ts, 0)
        src_ok  = (ts >= 0) & t_ok

        k_base = b * sk_b + h * sk_h
        v_base = b * sv_b + h * sv_h

        k = tl.load(
            K + k_base + ts_safe[:, None] * sk_t + d_off[None, :] * sk_d,
            mask=src_ok[:, None], other=0.0,
        ).to(tl.float32)

        v = tl.load(
            V + v_base + ts_safe[:, None] * sv_t + d_off[None, :] * sv_d,
            mask=src_ok[:, None], other=0.0,
        ).to(tl.float32)

        dots = (tl.sum(q * k, axis=1) / scale).to(tl.float32)
        tau  = tl.sigmoid(dots) * src_ok.to(tl.float32)

        acc += tau[:, None] * v

        if HAS_TAU:
            tl.store(
                Tau + tau_base + t_off * st_t + w * st_w,
                tau,
                mask=t_ok,
            )

    o_base = b * so_b + h * so_h
    tl.store(
        Out + o_base + t_off[:, None] * so_t + d_off[None, :] * so_d,
        acc,
        mask=t_ok[:, None],
    )


# ── Backward: dQ ──────────────────────────────────────────────────────────────

@triton.jit
def _bwd_dq_kernel(
    Q, K, V, dOut, dTau, dQ,
    T, H, W, scale,
    sq_b, sq_t, sq_h, sq_d,
    sk_b, sk_t, sk_h, sk_d,
    sv_b, sv_t, sv_h, sv_d,
    sd_b, sd_t, sd_h, sd_d,
    sdt_b, sdt_t, sdt_h, sdt_w,
    HD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_DTAU: tl.constexpr,
):
    """
    dQ: same window scan as forward.
    For each (t, w):
      tau  = σ(dot)
      dtau = ⟨dOut[t], V[ts]⟩ + (dTau[t, w] if HAS_DTAU)
      ddot = dtau · σ'(dot) = dtau · τ · (1 − τ)
      dq[t] += ddot · k[ts] / scale
    """
    pid    = tl.program_id(0)
    n_tile = tl.cdiv(T, BLOCK_T)
    bh     = pid // n_tile
    tile   = pid  % n_tile

    b  = bh // H
    h  = bh  % H
    t0 = tile * BLOCK_T

    t_off = t0 + tl.arange(0, BLOCK_T)
    d_off = tl.arange(0, HD)
    t_ok  = t_off < T

    q_base    = b * sq_b + h * sq_h
    dout_base = b * sd_b + h * sd_h
    dtau_base = b * sdt_b + h * sdt_h

    q = tl.load(
        Q + q_base + t_off[:, None] * sq_t + d_off[None, :] * sq_d,
        mask=t_ok[:, None], other=0.0,
    ).to(tl.float32)

    dout = tl.load(
        dOut + dout_base + t_off[:, None] * sd_t + d_off[None, :] * sd_d,
        mask=t_ok[:, None], other=0.0,
    ).to(tl.float32)

    dq = tl.zeros([BLOCK_T, HD], dtype=tl.float32)

    for w in range(W):
        ts      = t_off - (W - w)
        ts_safe = tl.maximum(ts, 0)
        src_ok  = (ts >= 0) & t_ok

        k_base = b * sk_b + h * sk_h
        v_base = b * sv_b + h * sv_h

        k = tl.load(
            K + k_base + ts_safe[:, None] * sk_t + d_off[None, :] * sk_d,
            mask=src_ok[:, None], other=0.0,
        ).to(tl.float32)

        v = tl.load(
            V + v_base + ts_safe[:, None] * sv_t + d_off[None, :] * sv_d,
            mask=src_ok[:, None], other=0.0,
        ).to(tl.float32)

        dots  = (tl.sum(q * k, axis=1) / scale).to(tl.float32)
        tau   = tl.sigmoid(dots) * src_ok.to(tl.float32)
        dtau  = tl.sum(dout * v, axis=1)

        if HAS_DTAU:
            dtau += tl.load(
                dTau + dtau_base + t_off * sdt_t + w * sdt_w,
                mask=t_ok, other=0.0,
            ).to(tl.float32) * src_ok.to(tl.float32)

        ddot = dtau * tau * (1.0 - tau) * src_ok.to(tl.float32)

        dq += ddot[:, None] * k / scale

    tl.store(
        dQ + q_base + t_off[:, None] * sq_t + d_off[None, :] * sq_d,
        dq,
        mask=t_ok[:, None],
    )


# ── Backward: dK and dV ────────────────────────────────────────────────────────

@triton.jit
def _bwd_dkv_kernel(
    Q, K, V, dOut, dTau, dK, dV,
    T, H, W, scale,
    sq_b, sq_t, sq_h, sq_d,
    sk_b, sk_t, sk_h, sk_d,
    sv_b, sv_t, sv_h, sv_d,
    sd_b, sd_t, sd_h, sd_d,
    sdt_b, sdt_t, sdt_h, sdt_w,
    HD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_DTAU: tl.constexpr,
):
    """
    For K/V position ts, accumulate gradients from all future query positions
    t in [ts+1, ts+W].  Inner loop delta = t - ts ∈ [1, W], which corresponds
    to forward-loop index w = W - delta (so dTau[t, w=W-delta]).
        tau  = σ(dot(Q[t], K[ts]) / scale)
        dV[ts] += τ · dOut[t]
        dtau = ⟨dOut[t], V[ts]⟩ + (dTau[t, W-delta] if HAS_DTAU)
        dK[ts] += dtau · σ'(dot) · Q[t] / scale
    """
    pid    = tl.program_id(0)
    n_tile = tl.cdiv(T, BLOCK_T)
    bh     = pid // n_tile
    tile   = pid  % n_tile

    b   = bh // H
    h   = bh  % H
    ts0 = tile * BLOCK_T

    ts_off = ts0 + tl.arange(0, BLOCK_T)
    d_off  = tl.arange(0, HD)
    ts_ok  = ts_off < T

    k_base = b * sk_b + h * sk_h
    v_base = b * sv_b + h * sv_h

    k = tl.load(
        K + k_base + ts_off[:, None] * sk_t + d_off[None, :] * sk_d,
        mask=ts_ok[:, None], other=0.0,
    ).to(tl.float32)

    v = tl.load(
        V + v_base + ts_off[:, None] * sv_t + d_off[None, :] * sv_d,
        mask=ts_ok[:, None], other=0.0,
    ).to(tl.float32)

    dk = tl.zeros([BLOCK_T, HD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_T, HD], dtype=tl.float32)

    q_base    = b * sq_b + h * sq_h
    dout_base = b * sd_b + h * sd_h
    dtau_base = b * sdt_b + h * sdt_h

    for delta in range(1, W + 1):
        tq      = ts_off + delta
        tq_safe = tl.minimum(tq, T - 1)
        q_ok    = (tq < T) & ts_ok

        q = tl.load(
            Q + q_base + tq_safe[:, None] * sq_t + d_off[None, :] * sq_d,
            mask=q_ok[:, None], other=0.0,
        ).to(tl.float32)

        dout = tl.load(
            dOut + dout_base + tq_safe[:, None] * sd_t + d_off[None, :] * sd_d,
            mask=q_ok[:, None], other=0.0,
        ).to(tl.float32)

        dots = (tl.sum(q * k, axis=1) / scale).to(tl.float32)
        tau  = tl.sigmoid(dots) * q_ok.to(tl.float32)

        dv += tau[:, None] * dout

        dtau = tl.sum(dout * v, axis=1)

        if HAS_DTAU:
            # Forward loop index for this (tq, ts) pair is w = W - delta.
            w_fwd = W - delta
            dtau += tl.load(
                dTau + dtau_base + tq_safe * sdt_t + w_fwd * sdt_w,
                mask=q_ok, other=0.0,
            ).to(tl.float32) * q_ok.to(tl.float32)

        ddot = dtau * tau * (1.0 - tau) * q_ok.to(tl.float32)
        dk  += ddot[:, None] * q / scale

    tl.store(
        dK + k_base + ts_off[:, None] * sk_t + d_off[None, :] * sk_d,
        dk,
        mask=ts_ok[:, None],
    )
    tl.store(
        dV + v_base + ts_off[:, None] * sv_t + d_off[None, :] * sv_d,
        dv,
        mask=ts_ok[:, None],
    )
