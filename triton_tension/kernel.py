"""
Fused causal sigmoid tension kernels.

Forward: out[b,t,h] = Σ_w  sigmoid(dot(Q[t], K[t-(W-w)]) / scale) · V[t-(W-w)]
         for w in [0, W-1], skipping positions where t-(W-w) < 0

Backward: dQ, dK, dV via two kernels:
  - _bwd_dq_kernel: same loop structure as forward (Q-centric)
  - _bwd_dkv_kernel: reverse loop — for each K/V position, accumulate from
                     future query positions that looked at it

Memory: O(B·T·H·HD) — never materialises the O(B·T·H·W·HD) expanded window tensor.
"""

import triton
import triton.language as tl


# ── Forward ────────────────────────────────────────────────────────────────────

@triton.jit
def _fwd_kernel(
    Q, K, V, Out,
    T, H, W, scale,
    sq_b, sq_t, sq_h, sq_d,
    sk_b, sk_t, sk_h, sk_d,
    sv_b, sv_t, sv_h, sv_d,
    so_b, so_t, so_h, so_d,
    HD: tl.constexpr,
    BLOCK_T: tl.constexpr,
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

    # Load Q: [BLOCK_T, HD]
    q_base = b * sq_b + h * sq_h
    q = tl.load(
        Q + q_base + t_off[:, None] * sq_t + d_off[None, :] * sq_d,
        mask=t_ok[:, None], other=0.0,
    ).to(tl.float32)

    acc = tl.zeros([BLOCK_T, HD], dtype=tl.float32)

    for w in range(W):
        # Source position: t - (W - w).  w=0 → oldest; w=W-1 → most recent past.
        ts      = t_off - (W - w)
        ts_safe = tl.maximum(ts, 0)          # clamp for safe pointer arithmetic
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

    o_base = b * so_b + h * so_h
    tl.store(
        Out + o_base + t_off[:, None] * so_t + d_off[None, :] * so_d,
        acc,
        mask=t_ok[:, None],
    )


# ── Backward: dQ ──────────────────────────────────────────────────────────────

@triton.jit
def _bwd_dq_kernel(
    Q, K, V, dOut, dQ,
    T, H, W, scale,
    sq_b, sq_t, sq_h, sq_d,
    sk_b, sk_t, sk_h, sk_d,
    sv_b, sv_t, sv_h, sv_d,
    sd_b, sd_t, sd_h, sd_d,
    HD: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
    dQ: same window scan as forward.
    dq[t] = Σ_w  sigma'(dot) · dot(dout[t], v[ts]) · k[ts] / scale
    where sigma'(x) = sigma(x) · (1 - sigma(x))
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
        dtau  = tl.sum(dout * v, axis=1)             # dot(dout, v): [BLOCK_T]
        ddot  = dtau * tau * (1.0 - tau) * src_ok.to(tl.float32)

        dq += ddot[:, None] * k / scale

    tl.store(
        dQ + q_base + t_off[:, None] * sq_t + d_off[None, :] * sq_d,
        dq,
        mask=t_ok[:, None],
    )


# ── Backward: dK and dV ────────────────────────────────────────────────────────

@triton.jit
def _bwd_dkv_kernel(
    Q, K, V, dOut, dK, dV,
    T, H, W, scale,
    sq_b, sq_t, sq_h, sq_d,
    sk_b, sk_t, sk_h, sk_d,
    sv_b, sv_t, sv_h, sv_d,
    sd_b, sd_t, sd_h, sd_d,
    HD: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
    For K/V position ts, accumulate gradients from all future query positions
    t in [ts+1, ts+W] that have ts in their window.

    delta = t - ts ∈ [1, W]:
        tau = sigmoid(dot(Q[t], K[ts]) / scale)
        dV[ts] += tau · dOut[t]
        dK[ts] += sigma'(dots) · dot(dOut[t], V[ts]) · Q[t] / scale
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

    # Load K and V for this tile: [BLOCK_T, HD]
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

    for delta in range(1, W + 1):
        tq      = ts_off + delta          # query position for this delta
        tq_safe = tl.minimum(tq, T - 1)  # clamp for safe pointer arithmetic
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
