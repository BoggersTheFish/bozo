"""
Parity tests for the return_tau path.

Validates that:
  1. Forward tau matches the reference unfold tau.
  2. Output when return_tau=True matches output when return_tau=False (up to
     fp precision — same kernel, tau is a side effect).
  3. Backward dQ/dK/dV agree with autograd-through-reference, including when a
     gradient flows back through tau (sparse_grad gate / aux-loss path).
  4. Memory footprint of the fused-with-tau path is a small fraction of the
     unfold path at realistic training dimensions.
"""

import math
import torch
import torch.nn.functional as F

from triton_tension.ops import causal_tension, _ref_forward


def ref_out_and_tau(Q, K, V, W, scale):
    """Pure-PyTorch reference: returns raw Σ τ·V (unnormalised) and tau."""
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    msg, tau = _ref_forward(Qf, Kf, Vf, W, scale, return_tau=True)
    return msg, tau


def test_forward_tau(B=2, T=32, H=4, HD=16, W=8, dtype=torch.float32):
    scale = math.sqrt(HD)
    torch.manual_seed(0)
    Q = torch.randn(B, T, H, HD, device="cuda", dtype=dtype)
    K = torch.randn(B, T, H, HD, device="cuda", dtype=dtype)
    V = torch.randn(B, T, H, HD, device="cuda", dtype=dtype)

    out_t, tau_t = causal_tension(Q, K, V, W, scale, return_tau=True)
    out_r, tau_r = ref_out_and_tau(Q, K, V, W, scale)

    out_err = (out_t.float() - out_r).abs().max().item()
    tau_err = (tau_t.float() - tau_r).abs().max().item()
    tol = 1e-3 if dtype == torch.float32 else 5e-2

    print(f"  fwd+tau  {str(dtype):<20}  out_err={out_err:.2e}  tau_err={tau_err:.2e}")
    assert out_err < tol, f"out mismatch {out_err}"
    assert tau_err < tol, f"tau mismatch {tau_err}"


def test_output_identical_with_without_tau(B=2, T=16, H=2, HD=16, W=6,
                                            dtype=torch.float32):
    """Requesting tau must not change the Out values."""
    scale = math.sqrt(HD)
    torch.manual_seed(1)
    Q = torch.randn(B, T, H, HD, device="cuda", dtype=dtype)
    K = torch.randn(B, T, H, HD, device="cuda", dtype=dtype)
    V = torch.randn(B, T, H, HD, device="cuda", dtype=dtype)

    out_no_tau      = causal_tension(Q, K, V, W, scale)
    out_with_tau, _ = causal_tension(Q, K, V, W, scale, return_tau=True)
    err = (out_no_tau - out_with_tau).abs().max().item()
    print(f"  fwd      Out identical with/without tau  max_err={err:.2e}")
    assert err == 0.0, f"Out differs between return_tau paths: {err}"


def test_backward_through_tau(B=2, T=24, H=4, HD=16, W=8, dtype=torch.float32):
    """Gradient via both Out and tau must match autograd through reference."""
    scale = math.sqrt(HD)
    torch.manual_seed(2)

    def make():
        return [
            torch.randn(B, T, H, HD, device="cuda", dtype=dtype, requires_grad=True)
            for _ in range(3)
        ]

    # Triton path
    Qt, Kt, Vt = make()
    out_t, tau_t = causal_tension(Qt, Kt, Vt, W, scale, return_tau=True)
    # Fake downstream: a scalar that depends on both Out and tau.
    dOut_sample = torch.randn_like(out_t)
    dTau_sample = torch.randn_like(tau_t) * 0.1
    loss_t = (out_t * dOut_sample).sum() + (tau_t * dTau_sample).sum()
    loss_t.backward()

    # Reference path (autograd through _ref_forward)
    Qr = Qt.detach().clone().float().requires_grad_(True)
    Kr = Kt.detach().clone().float().requires_grad_(True)
    Vr = Vt.detach().clone().float().requires_grad_(True)
    out_r, tau_r = _ref_forward(Qr, Kr, Vr, W, scale, return_tau=True)
    loss_r = (out_r * dOut_sample.float()).sum() + (tau_r * dTau_sample.float()).sum()
    loss_r.backward()

    for name, t_grad, r_grad in [
        ("dQ", Qt.grad, Qr.grad),
        ("dK", Kt.grad, Kr.grad),
        ("dV", Vt.grad, Vr.grad),
    ]:
        err = (t_grad.float() - r_grad).abs().max().item()
        tol = 5e-3
        print(f"  bwd+tau  {name:<6} max_err={err:.2e}  {'OK' if err < tol else 'FAIL'}")
        assert err < tol, f"{name} bwd-through-tau mismatch: {err}"


def test_backward_only_out(B=2, T=24, H=4, HD=16, W=8, dtype=torch.float32):
    """When return_tau=True but tau is never used downstream, dQ/dK/dV must
    match the case where return_tau=False."""
    scale = math.sqrt(HD)
    torch.manual_seed(3)

    def make():
        return [
            torch.randn(B, T, H, HD, device="cuda", dtype=dtype, requires_grad=True)
            for _ in range(3)
        ]

    Qa, Ka, Va = make()
    out_a      = causal_tension(Qa, Ka, Va, W, scale)
    dOut       = torch.randn_like(out_a)
    (out_a * dOut).sum().backward()

    Qb, Kb, Vb = [x.detach().clone().requires_grad_(True) for x in (Qa, Ka, Va)]
    out_b, _   = causal_tension(Qb, Kb, Vb, W, scale, return_tau=True)
    (out_b * dOut).sum().backward()

    for name, ga, gb in [("dQ", Qa.grad, Qb.grad), ("dK", Ka.grad, Kb.grad),
                         ("dV", Va.grad, Vb.grad)]:
        err = (ga - gb).abs().max().item()
        print(f"  bwd      {name:<6} tau-unused parity  max_err={err:.2e}")
        assert err < 1e-5, f"{name} diverges between paths: {err}"


def test_memory_vs_unfold():
    """At 350M-ish dims, the fused-with-tau path should use far less memory
    than the unfold path (~B·T·H·W·HD·2 bytes = 2 GB per layer at bf16)."""
    B, T, H, HD, W = 4, 1024, 16, 64, 256
    scale = math.sqrt(HD)
    torch.manual_seed(4)

    Q = torch.randn(B, T, H, HD, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, T, H, HD, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, T, H, HD, device="cuda", dtype=torch.bfloat16)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()

    out, tau = causal_tension(Q, K, V, W, scale, return_tau=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    # Tau is the dominant new allocation: B*T*H*W*2 bytes (bf16 return).
    tau_bytes = B * T * H * W * 2
    expected_tau_mb = tau_bytes / 1e6
    # Unfold gather is B*T*H*W*HD*2 bytes — 64× larger.
    unfold_bytes = B * T * H * W * HD * 2
    print(f"  mem      B={B} T={T} H={H} W={W} HD={HD}  "
          f"peak={(peak-before)/1e6:.1f} MB  "
          f"tau_alone={expected_tau_mb:.1f} MB  "
          f"unfold_would_be={unfold_bytes/1e6:.1f} MB")
    # Sanity: peak should be well under the unfold footprint.
    assert (peak - before) < unfold_bytes / 4, \
        "Fused path using more memory than ~1/4 unfold — regression?"


if __name__ == "__main__":
    print("Triton tension kernel — return_tau parity tests\n")

    print("Forward tau parity:")
    test_forward_tau(dtype=torch.float32)
    test_forward_tau(dtype=torch.bfloat16)

    print("\nForward Out invariance:")
    test_output_identical_with_without_tau()

    print("\nBackward parity when tau is consumed:")
    test_backward_through_tau()

    print("\nBackward parity when tau is unused:")
    test_backward_only_out()

    print("\nMemory footprint:")
    test_memory_vs_unfold()

    print("\nAll return_tau tests passed.")
