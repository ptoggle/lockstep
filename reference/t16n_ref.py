"""Reference of the contract-2.6 float norm ("float norm", on top of contract 2.5): the RMSNorm as an fp32 computation whose
reduction order is pinned by element index, so that any lane width, thread mapping or vendor
reproduces it bit for bit.  This file IS the rule; the kernel (t16n_kernels.cu) is gated against it.

Rule, for a row x of width N (N a multiple of 8) in bf16 widened to binary32
(with a residual: x = fl32(x + res), and bf16(x) is the layer's residual output, as in v2.2):

  sq[j]   = fl32(x[j] * x[j])
  group g (elements 8g .. 8g+7):  a = sq[8g]; a = fl32(a + sq[8g+i]) for i = 1..7      (G = N/8 groups)
  S       = T(a[0..G)) where T(one value) = the value and, for n values,
            T(v) = fl32( T(v[0..P)) + T(v[P..n)) ),  P = the largest power of two BELOW n
            (so a power-of-two count is halved; 384 splits 256 | 128).  This is the adjacent-pairs
            balanced tree, which a warp computes by xor-shuffles 1, 2, 4, 8, 16 and a CTA by the same
            rule over its warp sums.
  S       = fl32( S + ne ),   ne = fl32( fl64(N) * fl64(eps) )   the model's rms_norm_eps as a published binary32 constant
            (contract 2.5 carries eps as an integer term; here it is the fp32 add the stock definition has)
  S       = max(S, 2^-126)                      (an all-zero row gives zeros)
  rs      = fl32( sN / fl32( sqrt(S) ) ),   sN = fl32( sqrt(N) )   (a published constant of the width; 64 for 4096)
            IEEE square root and IEEE division: basic operations, correctly rounded on every compliant CPU and GPU.
            (Contract 3.0 read 1/sqrt from the 2.2 table at 14 bits; 3.1 removes the table.)
  out[j]  = bf16( fl32( fl32( x[j] * rs ) * g[j] ) ),  g the norm's gain in bf16 widened
            (folded norms have g = 1 exactly and the second multiply is the identity)
  int8    = the v2.2 row quantiser on out (asc = fl32(max(amax, 1e-8) / 127), r = fl32(1 / asc),
            q = clamp(rn_even(fl32(out * r)), +-127)).

No fused multiply-add anywhere; every fl32 is one IEEE binary32 rounding to nearest even; subnormals
are not flushed.  The functions below are written with torch elementwise ops (each one rounding) and
a numpy twin for a second opinion.
"""
import math
import numpy as np
import torch

FLT_MIN = 2.0 ** -126



def sqrt_n(N):
    return float(np.float32(math.sqrt(N)))


def tree_sum(v):
    """v: [R, n] float32 tensor of partial sums -> [R]: the rule's tree (largest power of two below n)."""
    n = v.shape[1]
    if n == 1:
        return v[:, 0]
    P = 1 << (n.bit_length() - 1)
    if P == n:
        P = n // 2
    return tree_sum(v[:, :P]) + tree_sum(v[:, P:])


def row_sum_float(x2):
    """x2: [R, N] float32 -> S [R] by the rule (group chains of 8, then the tree)."""
    R, N = x2.shape
    assert N % 8 == 0
    sq = x2 * x2
    g = sq.reshape(R, N // 8, 8)
    a = g[:, :, 0]
    for i in range(1, 8):
        a = a + g[:, :, i]
    return tree_sum(a)


def eps_term(N, eps):
    """ne = fl32(fl64(N) * fl64(eps)): one binary64 multiply, one rounding to binary32."""
    return float(np.float32(float(N) * float(eps or 0.0)))


def rmsnorm_float(x, weight, sN=None, out_dtype=None, eps=0.0):
    """x: [..., N] (bf16 or fp32), weight: [N] gain (bf16 or fp32; folded norms pass ones).
    Returns the bf16 output (out_dtype=None -> bf16), plus the float32 pre-rounding value is not needed."""
    shp = x.shape
    N = shp[-1]
    x2 = x.reshape(-1, N).float()
    if sN is None:
        sN = sqrt_n(N)
    S = row_sum_float(x2)
    if eps:
        S = S + torch.tensor(eps_term(N, eps), dtype=torch.float32, device=x2.device)   # one binary32 add
    S = S.clamp(min=FLT_MIN)
    # tensor / tensor: torch's tensor / python-scalar is a reciprocal multiply, not a correctly rounded division
    rs = torch.full_like(S, sN) / torch.sqrt(S)
    y = x2 * rs[:, None]
    g = weight.detach().float().reshape(-1)
    if not bool((g == 1.0).all()):
        y = y * g[None, :]
    out = y.to(torch.bfloat16)
    if out_dtype is not None and out_dtype != torch.bfloat16:
        out = out.to(out_dtype)
    return out.reshape(shp)


# ------------------------------------------------------------------ numpy twin (float32 ops, no FMA)
def _bf16_rne_np(f32):
    u = f32.astype(np.float32).view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def rmsnorm_float_np(x_f32, g_f32, sN, eps=0.0):
    """x_f32: [R, N] np.float32; g_f32: [N].  Returns the bf16 output as float32 values."""
    R, N = x_f32.shape
    f32 = np.float32
    sq = (x_f32 * x_f32).astype(f32)
    grp = sq.reshape(R, N // 8, 8)
    a = grp[:, :, 0].copy()
    for i in range(1, 8):
        a = (a + grp[:, :, i]).astype(f32)

    def tree(v):
        n = v.shape[1]
        if n == 1:
            return v[:, 0]
        P = 1 << (n.bit_length() - 1)
        if P == n:
            P = n // 2
        return (tree(v[:, :P]) + tree(v[:, P:])).astype(f32)
    S = tree(a)
    if eps:
        S = (S + f32(eps_term(N, eps))).astype(f32)
    S = np.maximum(S, f32(FLT_MIN)).astype(f32)
    rs = (np.full_like(S, f32(sN)) / np.sqrt(S).astype(f32)).astype(f32)     # IEEE sqrt, IEEE division
    y = (x_f32 * rs[:, None]).astype(f32)
    if not np.all(g_f32 == 1.0):
        y = (y * g_f32[None, :].astype(f32)).astype(f32)
    return _bf16_rne_np(y)
