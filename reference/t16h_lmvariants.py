#!/usr/bin/env python3
"""T16h - lm_head VARIANTS, each with its arithmetic PINNED.

Every variant is a pure function of (x, W_bf16) plus a declared, data-independent
artifact (weight scales; optionally a calibrated per-in-channel smoothing vector).
All of them are:
  * row-local  (per-token scales only)      -> batch/composition invariant  (F3)
  * integer-accumulated (int32 planes, int64 combine) -> tiling/split-k invariant
  * free of any python-scalar divisor       -> device-invariant (the T5 F2 finding)

The reference for the int8 arm is contract_common.w8a8_linear /
lockstep_fullstack.w8a8_matmul; `a8` here reproduces it BIT-FOR-BIT (gate G0c)
and is cross-checked against the engine's own logits in t16h_ppl.py.

SPEC GRAMMAR       bf16 | [sq<AA>][a<A>][w<W>]      defaults a8 w8
  bf16       stock float lm_head (F.linear in the model dtype).  NOT a contract
             op - it is the quality ceiling and the F3 breaker.
  a8   (=a8w8)  TODAY'S CONTRACT: per-out-channel int8 W, per-token dynamic int8 A,
             one int32 GEMM, epilogue fl32(acc) * asc * ws.
  a<A>       a <A>-bit activation carried EXACTLY by two int8 planes:
             hi = a >> (A-8) in [-128,127], lo = a - (hi << (A-8)) in [0, 2^(A-8)-1].
  w<W>       the same two-plane carrier on the WEIGHT.
  sq<AA>     SmoothQuant fold at alpha = AA/100: s = amax_act^a / amax_w^(1-a),
             W' = W * s folded ONCE offline, runtime x' = x / s (a correctly
             rounded elementwise divide by a declared fp32 vector), then as above.

WHY PLANES AND NOT A SINGLE int16 GEMM: with K = 3584, |a| <= 32767 and |w| <= 127
the exact bound is 32767*127*3584 = 1.49e10 > 2^31, so a single int32
accumulation is NOT exact by construction.  Every plane product here obeys
|plane_acc| <= 128*127*3584 = 5.83e7, i.e. 36x of int32 headroom, and the
combine sum(plane << shift) is done in int64 where it is exact for every input
in range.  The plane loop order is PINNED (activation planes outer, ascending
shift; weight planes inner) - integer addition is associative, so the order is
not a correctness requirement, but it is declared so an independent
implementation has nothing to choose.
"""
from __future__ import annotations

import torch

INT8_MIN_ROWS = 17          # torch._int_mm requires M > 16


def _const(value: float, ref):
    """Device/dtype-resident scalar tensor divisor.  `tensor / python_scalar` is
    reciprocal-MULTIPLY on CUDA and true division on CPU (1 ulp apart on ~4% of
    channels) - the substitution SPEC 6.0.3 prohibits; see lockstep_fullstack._const."""
    return torch.tensor(float(value), device=ref.device, dtype=ref.dtype)


def _int_mm(a8, b8):
    """int8 x int8 -> int32.  CUDA: torch._int_mm (cuBLASLt, int32 accumulation,
    exact at our shapes).  CPU: int64 matmul narrowed to int32 - numerically
    identical because no accumulation overflows int32.  Same discipline as
    lockstep_fullstack._int_mm_exact: on CUDA a failure raises, never reroutes."""
    n = a8.shape[0]
    pad = (INT8_MIN_ROWS - n) if n < INT8_MIN_ROWS else 0
    if pad:
        a8 = torch.cat([a8, torch.zeros(pad, a8.shape[1], dtype=torch.int8,
                                        device=a8.device)])
    try:
        y = torch._int_mm(a8, b8)
    except Exception as e:
        if a8.is_cuda:
            raise RuntimeError("torch._int_mm failed on CUDA %s x %s: %r"
                               % (list(a8.shape), list(b8.shape), e))
        y = torch.matmul(a8.to(torch.int64), b8.to(torch.int64)).to(torch.int32)
    return y[:n]


def _planes(a_i32, sh):
    """a = (hi << sh) + lo, hi in [-128,127], lo in [0, 2^sh - 1].  sh = 0 -> one plane."""
    if sh == 0:
        return [(0, a_i32.to(torch.int8))]
    hi = a_i32 >> sh                                 # arithmetic shift = floor division
    lo = a_i32 - (hi << sh)
    return [(sh, hi.to(torch.int8)), (0, lo.to(torch.int8))]


def _quant_rows(x2, abits):
    amax_q = float((1 << (abits - 1)) - 1)
    asc = (x2.abs().amax(dim=1).clamp(min=1e-8) / _const(amax_q, x2))
    a = torch.clamp(torch.round(x2 / asc[:, None]), -amax_q, amax_q)
    return a.to(torch.int32), asc


def quantize_weight_int8(W):
    """Per-out-channel int8 weight, contract_common.w8a8_linear semantics."""
    Wf = W.detach().float()
    ws = Wf.abs().amax(dim=1).clamp(min=1e-8) / _const(127.0, Wf)
    W8 = torch.clamp(torch.round(Wf / ws[:, None]), -127, 127).to(torch.int8).contiguous()
    return W8, ws.contiguous()


def quantize_weight(W, wbits):
    """Per-out-channel <wbits>-bit weight carried by 1 or 2 int8 planes.
    wbits = 8 reproduces quantize_weight_int8 exactly."""
    Wf = W.detach().float()
    amax_q = float((1 << (wbits - 1)) - 1)
    ws = Wf.abs().amax(dim=1).clamp(min=1e-8) / _const(amax_q, Wf)
    Wq = torch.clamp(torch.round(Wf / ws[:, None]), -amax_q, amax_q).to(torch.int32)
    planes = [(sh, p.contiguous().t()) for sh, p in _planes(Wq, wbits - 8)]
    return planes, ws.contiguous()                   # each plane is [K, N] stride (1, K)


def smooth_scales(act_amax, W, alpha):
    """SmoothQuant per-IN-channel scale, folded once into the weight."""
    Wf = W.detach().float()
    w_amax = Wf.abs().amax(dim=0).clamp(min=1e-5)
    a_amax = act_amax.detach().float().to(Wf.device).clamp(min=1e-5)
    s = (a_amax ** float(alpha)) / (w_amax ** (1.0 - float(alpha)))
    return s.clamp(min=1e-5).contiguous()


def _mm_planes(x2, wplanes, ws, abits):
    a, asc = _quant_rows(x2, abits)
    aplanes = _planes(a, abits - 8)
    if len(aplanes) == 1 and len(wplanes) == 1:      # today's contract: ONE int32 GEMM
        return _int_mm(aplanes[0][1], wplanes[0][1]).float() * asc[:, None] * ws[None, :]
    acc = None
    for ash, ap in aplanes:                          # PINNED order
        for wsh, wp in wplanes:
            t = _int_mm(ap, wp).to(torch.int64) << (ash + wsh)
            acc = t if acc is None else acc + t
    return acc.to(torch.float32) * asc[:, None] * ws[None, :]


class Variant:
    """name -> logits.  `prepare` is called once with the bf16 lm_head weight and
    (for the sq family) the calibrated per-in-channel activation amax."""

    def __init__(self, name, kind, abits=8, wbits=8, alpha=None):
        self.name, self.kind = name, kind
        self.abits, self.wbits, self.alpha = abits, wbits, alpha
        self.wplanes = self.ws = self.sdiv = self.W_bf16 = None

    def prepare(self, W, act_amax=None):
        self.W_bf16 = W
        if self.kind == "bf16":
            return self
        Wq = W
        if self.alpha is not None:
            if act_amax is None:
                raise RuntimeError("variant %s needs a calibrated act_amax" % self.name)
            self.sdiv = smooth_scales(act_amax, W, self.alpha)
            Wq = W.detach().float() * self.sdiv[None, :]
        self.wplanes, self.ws = quantize_weight(Wq, self.wbits)
        return self

    def logits(self, x2):
        """x2 : [rows, K] float32 -> [rows, N] float32."""
        if self.kind == "bf16":
            return torch.nn.functional.linear(x2.to(self.W_bf16.dtype), self.W_bf16).float()
        xq = x2 if self.sdiv is None else (x2 / self.sdiv[None, :])
        return _mm_planes(xq, self.wplanes, self.ws, self.abits)

    def n_out(self):
        return int(self.W_bf16.shape[0])

    def gemms(self):
        return 0 if self.kind == "bf16" else (1 if self.abits == 8 else 2) * len(self.wplanes)

    def nbytes(self):
        n = sum(p.numel() for _, p in (self.wplanes or []))
        if self.sdiv is not None:
            n += 4 * self.sdiv.numel()
        if self.ws is not None:
            n += 4 * self.ws.numel()
        return n


def build(spec: str) -> Variant:
    """bf16 | [sq<AA>][a<A>][w<W>].  e.g. a8  a15  a15w15  sq50  sq50a15  sq80a15w15"""
    s = spec.strip()
    if s == "bf16":
        return Variant(s, "bf16")
    rest, alpha = s, None
    if rest.startswith("sq"):
        i = 2
        while i < len(rest) and rest[i].isdigit():
            i += 1
        if i == 2:
            raise ValueError("sq needs an alpha, e.g. sq50: %r" % spec)
        alpha = int(rest[2:i]) / 100.0
        rest = rest[i:]
    abits, wbits = 8, 8
    if "w" in rest:
        rest, wtxt = rest.split("w", 1)
        wbits = int(wtxt)
    if rest:
        if not rest.startswith("a"):
            raise ValueError("unknown lm_head variant spec: %r" % spec)
        abits = int(rest[1:])
    for b, nm in ((abits, "activation"), (wbits, "weight")):
        if b not in (8, 11, 15):
            raise ValueError("%s width %d not supported (8, 11, 15): %r" % (nm, b, spec))
    return Variant(s, "planes", abits=abits, wbits=wbits, alpha=alpha)


def needs_calibration(specs) -> bool:
    return any(sp.strip().startswith("sq") for sp in specs)
