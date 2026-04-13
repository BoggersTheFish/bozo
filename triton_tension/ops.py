"""
torch.autograd.Function wrapper around the Triton tension kernels.

Usage:
    from triton_tension.ops import causal_tension
    # Q, K, V: [B, T, H, HD]  (float32 or bfloat16, CUDA)
    out = causal_tension(Q, K, V, window=64, scale=8.0)
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from torch import Tensor
from .kernel import _fwd_kernel, _bwd_dq_kernel, _bwd_dkv_kernel

BLOCK_T = 16


def _strides(t: Tensor):
    assert t.ndim == 4
    return t.stride(0), t.stride(1), t.stride(2), t.stride(3)


# ── Reference (pure PyTorch) ──────────────────────────────────────────────────

def _ref_forward(Q: Tensor, K: Tensor, V: Tensor, W: int, scale: float) -> Tensor:
    """
    Reference implementation — identical semantics to the Triton kernel.
    Used for float64 inputs (gradcheck) and CPU.  Autograd differentiates
    through this naturally, so no manual backward needed.
    """
    B, T, H, HD = Q.shape
    kv = torch.cat([K, V], dim=-1)
    zt = kv.permute(0, 2, 3, 1)
    zp = F.pad(zt, (W, 0), value=0.0)
    nb = zp.unfold(-1, W, 1)[:, :, :, :T, :]
    nb = nb.permute(0, 3, 1, 4, 2)                      # B T H W 2*HD
    nb_k, nb_v = nb.split(HD, dim=-1)

    t_idx = torch.arange(T, device=Q.device).unsqueeze(1)
    w_idx = torch.arange(W, device=Q.device).unsqueeze(0)
    mask  = (t_idx + w_idx >= W).to(Q.dtype)             # T W

    tau = torch.sigmoid(
        (Q.unsqueeze(3) * nb_k).sum(-1) / scale
    ) * mask.unsqueeze(0).unsqueeze(2)                   # B T H W

    valid = mask.sum(-1).clamp(min=1)                    # T
    msg   = (tau.unsqueeze(-1) * nb_v).sum(3)            # B T H HD
    return msg / valid[None, :, None, None]


# ── Triton autograd.Function ──────────────────────────────────────────────────

class _CausalTensionFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, W: int, scale: float) -> Tensor:
        B, T, H, HD = Q.shape

        Out = torch.zeros(B, T, H, HD, dtype=torch.float32, device=Q.device)
        grid = (B * H * math.ceil(T / BLOCK_T),)

        _fwd_kernel[grid](
            Q, K, V, Out,
            T, H, W, scale,
            *_strides(Q), *_strides(K), *_strides(V), *_strides(Out),
            HD=HD, BLOCK_T=BLOCK_T,
        )

        ctx.save_for_backward(Q, K, V)
        ctx.W, ctx.scale = W, scale
        return Out.to(Q.dtype)

    @staticmethod
    def backward(ctx, dOut: Tensor):
        Q, K, V   = ctx.saved_tensors
        W, scale  = ctx.W, ctx.scale
        B, T, H, HD = Q.shape

        dOut  = dOut.contiguous().to(torch.float32)
        Qf    = Q.to(torch.float32).contiguous()
        Kf    = K.to(torch.float32).contiguous()
        Vf    = V.to(torch.float32).contiguous()

        dQ = torch.zeros_like(Qf)
        dK = torch.zeros_like(Kf)
        dV = torch.zeros_like(Vf)

        grid = (B * H * math.ceil(T / BLOCK_T),)

        _bwd_dq_kernel[grid](
            Qf, Kf, Vf, dOut, dQ,
            T, H, W, scale,
            *_strides(Qf), *_strides(Kf), *_strides(Vf), *_strides(dOut),
            HD=HD, BLOCK_T=BLOCK_T,
        )
        _bwd_dkv_kernel[grid](
            Qf, Kf, Vf, dOut, dK, dV,
            T, H, W, scale,
            *_strides(Qf), *_strides(Kf), *_strides(Vf), *_strides(dOut),
            HD=HD, BLOCK_T=BLOCK_T,
        )

        return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype), None, None


# ── Public API ─────────────────────────────────────────────────────────────────

@torch._dynamo.disable   # Triton kernel already fused — prevent AOT Autograd from retracing
def causal_tension(Q: Tensor, K: Tensor, V: Tensor, window: int, scale: float) -> Tensor:
    """
    Fused causal sigmoid tension.

    Routes to:
      - Triton kernel  for CUDA float32/bfloat16 (fast, no materialised window tensor)
      - PyTorch unfold for float64 or CPU (gradcheck, testing, CPU inference)

    Args:
        Q, K, V : [B, T, H, HD]
        window  : causal look-back W
        scale   : dot-product scale (typically sqrt(HD))

    Returns:
        [B, T, H, HD] — magnitude-normalised weighted sum of values
    """
    if not Q.is_cuda or Q.dtype == torch.float64:
        return _ref_forward(Q, K, V, window, scale)

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()
    return _CausalTensionFn.apply(Q, K, V, window, scale)
