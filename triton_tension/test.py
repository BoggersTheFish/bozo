"""
Numerical gradient checks for the Triton tension kernels.

Runs:
  - Forward: Triton output vs reference PyTorch (unfold) implementation
  - Backward: torch.autograd.gradcheck on the Triton function
  - Dtype: float32 and bfloat16 inputs

Usage:
    python3 triton_tension/test.py
"""

import math
import torch
import torch.nn.functional as F
from triton_tension.ops import causal_tension


# ── Reference implementation (matches model.py exactly) ──────────────────────

def ref_causal_tension(Q, K, V, window, scale, bias=None):
    """
    Reference: unfold-based implementation from model.py.
    Returns output in float32 for comparison.

    bias : optional [B, T, W], added to pre-sigmoid dots (broadcast over H).
    """
    B, T, H, HD = Q.shape
    W = window

    # Gather window for K and V jointly (same as _gather_window in model.py)
    kv = torch.cat([K, V], dim=-1)  # B T H 2*HD
    zt = kv.permute(0, 2, 3, 1).float()          # B H 2*HD T
    zp = F.pad(zt, (W, 0), value=0.0)             # B H 2*HD (T+W)
    nb = zp.unfold(-1, W, 1)[:, :, :, :T, :]      # B H 2*HD T W
    nb = nb.permute(0, 3, 1, 4, 2)                 # B T H W 2*HD

    nb_k, nb_v = nb.split(HD, dim=-1)
    q_f = Q.float()

    dots = (q_f.unsqueeze(3) * nb_k).sum(-1) / scale   # B T H W
    if bias is not None:
        dots = dots + bias.unsqueeze(2).float()        # broadcast over H

    tau = torch.sigmoid(dots)

    # Apply causal mask (same as model.py fix)
    t_idx = torch.arange(T, device=Q.device).unsqueeze(1)
    w_idx = torch.arange(W, device=Q.device).unsqueeze(0)
    mask = (t_idx + w_idx >= W).float()
    tau = tau * mask.unsqueeze(0).unsqueeze(2)

    msg = (tau.unsqueeze(-1) * nb_v).sum(3)  # B T H HD
    return msg


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_forward(B=2, T=32, H=4, HD=16, W=8, dtype=torch.float32):
    scale = math.sqrt(HD)
    torch.manual_seed(0)
    Q = torch.randn(B, T, H, HD, device="cuda", dtype=dtype)
    K = torch.randn(B, T, H, HD, device="cuda", dtype=dtype)
    V = torch.randn(B, T, H, HD, device="cuda", dtype=dtype)

    out_triton = causal_tension(Q, K, V, W, scale).float()
    out_ref    = ref_causal_tension(Q, K, V, W, scale)

    max_err = (out_triton - out_ref).abs().max().item()
    rtol    = 1e-3 if dtype == torch.float32 else 5e-2
    ok      = max_err < rtol
    print(f"  forward  {str(dtype):<20}  max_err={max_err:.2e}  {'OK' if ok else 'FAIL'}")
    assert ok, f"Forward mismatch: max error {max_err:.2e}"


def test_backward(B=1, T=16, H=2, HD=16, W=6, dtype=torch.float64):
    """gradcheck requires float64."""
    scale = math.sqrt(HD)
    torch.manual_seed(1)
    Q = torch.randn(B, T, H, HD, device="cuda", dtype=dtype, requires_grad=True)
    K = torch.randn(B, T, H, HD, device="cuda", dtype=dtype, requires_grad=True)
    V = torch.randn(B, T, H, HD, device="cuda", dtype=dtype, requires_grad=True)

    from torch.autograd import gradcheck
    ok = gradcheck(
        causal_tension,
        (Q, K, V, W, scale),
        eps=1e-4, atol=1e-3, rtol=1e-3,
        raise_exception=False,
    )
    print(f"  backward gradcheck                    {'OK' if ok else 'FAIL'}")
    assert ok, "Gradient check failed"


def test_manual_grad(B=2, T=24, H=4, HD=16, W=8, dtype=torch.float32):
    """Compare Triton backward vs autograd through reference."""
    scale = math.sqrt(HD)
    torch.manual_seed(2)

    def make(requires_grad=False):
        Q = torch.randn(B, T, H, HD, device="cuda", dtype=dtype, requires_grad=requires_grad)
        K = torch.randn(B, T, H, HD, device="cuda", dtype=dtype, requires_grad=requires_grad)
        V = torch.randn(B, T, H, HD, device="cuda", dtype=dtype, requires_grad=requires_grad)
        return Q, K, V

    Q0, K0, V0 = make()
    Qt, Kt, Vt = [x.detach().requires_grad_(True) for x in (Q0, K0, V0)]
    Qr, Kr, Vr = [x.detach().float().requires_grad_(True) for x in (Q0, K0, V0)]

    dLoss = torch.randn(B, T, H, HD, device="cuda", dtype=dtype)

    # Triton path
    out_t = causal_tension(Qt, Kt, Vt, W, scale)
    out_t.backward(dLoss)

    # Reference path (float32)
    out_r = ref_causal_tension(Qr, Kr, Vr, W, scale)
    out_r.backward(dLoss.float())

    for name, t_grad, r_grad in [
        ("dQ", Qt.grad, Qr.grad),
        ("dK", Kt.grad, Kr.grad),
        ("dV", Vt.grad, Vr.grad),
    ]:
        err = (t_grad.float() - r_grad).abs().max().item()
        ok  = err < 5e-3
        print(f"  manual   {name:<6} max_err={err:.2e}  {'OK' if ok else 'FAIL'}")
        assert ok, f"{name} gradient mismatch: {err:.2e}"


def test_forward_biased(B=2, T=32, H=4, HD=16, W=8, dtype=torch.float32):
    """Forward with a non-trivial [B, T, W] bias."""
    scale = math.sqrt(HD)
    torch.manual_seed(10)
    Q = torch.randn(B, T, H, HD, device="cuda", dtype=dtype)
    K = torch.randn(B, T, H, HD, device="cuda", dtype=dtype)
    V = torch.randn(B, T, H, HD, device="cuda", dtype=dtype)
    # Nonzero, non-uniform bias — exercises the load paths in every kernel.
    bias = torch.randn(B, T, W, device="cuda", dtype=torch.float32) * 0.5

    out_triton = causal_tension(Q, K, V, W, scale, bias=bias).float()
    out_ref    = ref_causal_tension(Q, K, V, W, scale, bias=bias)

    max_err = (out_triton - out_ref).abs().max().item()
    rtol    = 1e-3 if dtype == torch.float32 else 5e-2
    ok      = max_err < rtol
    print(f"  forward+bias  {str(dtype):<15}  max_err={max_err:.2e}  "
          f"{'OK' if ok else 'FAIL'}")
    assert ok, f"Biased forward mismatch: max error {max_err:.2e}"


def test_backward_biased(B=1, T=16, H=2, HD=16, W=6, dtype=torch.float64):
    """gradcheck with a constant bias tensor (bias itself has no grad)."""
    scale = math.sqrt(HD)
    torch.manual_seed(11)
    Q = torch.randn(B, T, H, HD, device="cuda", dtype=dtype, requires_grad=True)
    K = torch.randn(B, T, H, HD, device="cuda", dtype=dtype, requires_grad=True)
    V = torch.randn(B, T, H, HD, device="cuda", dtype=dtype, requires_grad=True)
    bias = torch.randn(B, T, W, device="cuda", dtype=dtype) * 0.3

    from torch.autograd import gradcheck

    def fn(q, k, v):
        return causal_tension(q, k, v, W, scale, bias=bias)

    ok = gradcheck(
        fn, (Q, K, V),
        eps=1e-4, atol=1e-3, rtol=1e-3,
        raise_exception=False,
    )
    print(f"  backward+bias gradcheck                {'OK' if ok else 'FAIL'}")
    assert ok, "Biased gradient check failed"


def test_manual_grad_biased(B=2, T=24, H=4, HD=16, W=8, dtype=torch.float32):
    """Compare Triton biased backward vs autograd through biased reference."""
    scale = math.sqrt(HD)
    torch.manual_seed(12)

    def make():
        return (
            torch.randn(B, T, H, HD, device="cuda", dtype=dtype),
            torch.randn(B, T, H, HD, device="cuda", dtype=dtype),
            torch.randn(B, T, H, HD, device="cuda", dtype=dtype),
        )

    Q0, K0, V0 = make()
    bias = torch.randn(B, T, W, device="cuda", dtype=torch.float32) * 0.5

    Qt, Kt, Vt = [x.detach().requires_grad_(True) for x in (Q0, K0, V0)]
    Qr, Kr, Vr = [x.detach().float().requires_grad_(True) for x in (Q0, K0, V0)]

    dLoss = torch.randn(B, T, H, HD, device="cuda", dtype=dtype)

    out_t = causal_tension(Qt, Kt, Vt, W, scale, bias=bias)
    out_t.backward(dLoss)

    out_r = ref_causal_tension(Qr, Kr, Vr, W, scale, bias=bias)
    out_r.backward(dLoss.float())

    for name, t_grad, r_grad in [
        ("dQ", Qt.grad, Qr.grad),
        ("dK", Kt.grad, Kr.grad),
        ("dV", Vt.grad, Vr.grad),
    ]:
        err = (t_grad.float() - r_grad).abs().max().item()
        ok  = err < 5e-3
        print(f"  manual+bias {name:<6} max_err={err:.2e}  {'OK' if ok else 'FAIL'}")
        assert ok, f"Biased {name} mismatch: {err:.2e}"


def test_large(B=4, T=512, H=12, HD=64, W=64):
    """Spot-check at training-scale dimensions."""
    scale = math.sqrt(HD)
    torch.manual_seed(3)
    Q = torch.randn(B, T, H, HD, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, T, H, HD, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, T, H, HD, device="cuda", dtype=torch.bfloat16)

    out_t = causal_tension(Q, K, V, W, scale).float()
    out_r = ref_causal_tension(Q, K, V, W, scale)

    max_err = (out_t - out_r).abs().max().item()
    rel_err = (out_t - out_r).abs() / (out_r.abs() + 1e-6)
    ok = rel_err.mean().item() < 0.02   # mean relative error < 2% for bf16
    print(f"  large    bf16 B={B} T={T} H={H} W={W}  max_err={max_err:.3f}  mean_rel={rel_err.mean():.4f}  {'OK' if ok else 'FAIL'}")
    assert ok


if __name__ == "__main__":
    print("Triton causal tension kernel tests\n")

    print("Forward correctness:")
    test_forward(dtype=torch.float32)
    test_forward(dtype=torch.bfloat16)

    print("\nManual gradient comparison:")
    test_manual_grad()

    print("\nGradcheck (float64):")
    test_backward()

    print("\nPhase-2 bias path:")
    test_forward_biased(dtype=torch.float32)
    test_forward_biased(dtype=torch.bfloat16)
    test_manual_grad_biased()
    test_backward_biased()

    print("\nLarge-scale spot check:")
    test_large()

    print("\nAll tests passed.")
