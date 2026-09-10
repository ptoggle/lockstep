"""Lockstep FULL-STACK contract operators for vLLM 0.25.1 (WS3-T5).

WHAT THIS IS
------------
demo/vllm_lockstep_backend.py makes ATTENTION exact (a first-class 0.25.1
attention backend, T2/T7a).  Everything AROUND attention is still float in that
setup — RoPE, RMSNorm, the Q/K/V + MLP linears and the lm_head logits — so the
engine as a whole is NOT batch-invariant: cuBLAS picks different bf16 GEMM
kernels/split-k for different M (batch composition), and the resulting last-bit
jitter amplifies through 28 layers into different tokens.

This module arms the NON-ATTENTION contract operators so the pipeline is exact
END TO END.  It is a forward-port of the operators that already worked in the
old monkey-patch stack (demo/vllm_contract4.py, demo/vllm_contract2.py) onto
0.25.1's class layout, with two deliberate upgrades stated below.

PORT MAP (old symbol -> this module)
------------------------------------
| old                                             | here                        |
|-------------------------------------------------|-----------------------------|
| vllm_contract4.py `rope_contract` + deai::rope   | `_rope_forward` (torch)     |
| vllm_contract4.py `norm_contract` + deai::norm   | `_rmsnorm_forward` (torch)  |
| vllm_contract2.py:176-201 `logits_int`           | `_logits_forward`           |
| (checkpoint-provided W8A8 in the old stack)      | `_linear_apply` (NEW)       |
| SPECIFICATION 6.3.1 silu_det                     | `silu_contract` (opt-in)    |

The CUDA custom ops of demo/deai_kernels.cu (int15_rmsnorm, rope_q14,
head_logits) are OPTIONAL and NOT used by this first cut: a correct torch
implementation is what the first gate pass needs (exactness before speed).
Every operator below is written so that a kernel can replace it later without
changing a single bit (all reductions are exact-integer or row-local).

SEMANTICS AND THEIR SOURCES
---------------------------
1. W8A8 LINEAR  (`w8a8_matmul`) — bit-for-bit the golden recipe of
   paper_softmax/bigrun/contract_common.py::w8a8_linear (the implementation
   that produced the filed EV-5 quality numbers, SPECIFICATION 6.5(b)):
   per-out-channel weight scale ws = amax(|W|,dim=1)/127 from the FP32 weight,
   per-token dynamic activation scale asc = amax(|x|,dim=1)/127, half-even
   rounding, clamp [-127,127], `torch._int_mm` (int32 accumulation), dequant
   as `(int32.float() * ws[None,:]) * asc[:,None]` IN THAT ORDER (contract v2.1:
   the WEIGHT scale first - the order the Hopper CUTLASS fused epilogue evaluates,
   measured 0 mismatches over 310M outputs; before v2.1 it was asc first), rounded
   to x.dtype, then bias: `bf16(fl32(y8) + bias)` (a second rounding, so that the
   biased and unbiased linears share one GEMM epilogue).
   EXACT AND BATCH-INVARIANT BY CONSTRUCTION: int32 accumulation of int8 products
   cannot overflow at these shapes (K<=18944: |dot| <= 18944*127*127 = 3.1e8 <
   2^31), so any tiling / split-k / kernel choice cuBLASLt makes yields the same
   int32; amax is an exact associative reduction; everything else is row-local
   elementwise f32.  This is the operator that removes the M-dependent cuBLAS
   bf16 GEMM as a divergence source (T2_LOG.md "G2 divergence localization").
2. RMSNorm (`rmsnorm_contract`) — the §15 integer norm of vllm_contract2.py
   (int15_norm) / vllm_contract4.py (deai::norm), with ONE documented change:
   the per-layer STATIC input scale from the declared artifact
   (/workspace/eval/declared_artifact_v1.json, which only ever existed for the
   Llama-W8A8 checkpoint) is replaced by a per-ROW DYNAMIC power-of-two scale
   built by the quantizer rule of SPECIFICATION 6.1.5 / contract_common
   `quant_pow2` (frexp exponent, clip-safe decrement), widened from 7 bits to
   15 bits.  This is sound because RMSNorm is scale-invariant in its input: the
   input exponent cancels between the sum-of-squares and the Q14 reciprocal
   square root, so the output is identical for any choice of the row exponent
   that avoids clipping — and it removes the calibration artifact entirely.
   Everything else is the old operator: exact int64 sum of squares, the
   2^15-entry Q14 rsqrt table, half-away-from-zero requantization, integer gain
   G = round(w * 131072), one f32 multiply by sqrt(N)/131072, cast last.
   Row-local integer reduction => batch-invariant by construction.
3. RoPE (`rope_contract`) — vllm_contract2.py `rope_q14` / vllm_contract4.py
   deai::rope: the cos/sin table is FROZEN on the Q14 grid
   (round(x*16384)/16384) so it carries no libm/vendor dependence, and the
   rotation is evaluated in f32 with separate (non-contracted) multiplies and
   adds, cast back to the model dtype once.  Elementwise => batch-invariant.
   NOTE this differs from vLLM's own forward_native, which does the rotation in
   the INPUT dtype (bf16); f32 here is strictly higher quality and is what the
   old contract stack did.
4. LOGITS (`_logits_forward`) — vllm_contract2.py:176-201 verbatim in
   semantics: per-token dynamic int8 x per-channel int8 weight through
   `torch._int_mm`, f32 elementwise scales, then the engine's own gather and
   org_vocab_size slice.  In decode the logits GEMM has M = number of running
   sequences, i.e. it is THE most batch-composition-sensitive float GEMM in the
   stack; making it integer is what lets identical hidden states produce
   identical tokens.
5. SiLU (`silu_contract`, OPT-IN) — SPECIFICATION 6.3.1 step-for-step (10
   declared binary32 operations, FRAC table shared with the softmax engine).
   Stock `SiluAndMul` is already elementwise and therefore batch-invariant, so
   this is NOT needed for F3; it is needed for CROSS-VENDOR exactness (a
   vendor's expf is not correctly rounded).  Off by default so that the F3/F4
   numbers isolate the four mandated operators; enable with
   LOCKSTEP_FS_OPS=rmsnorm,rope,linear,logits,silu.

EXACTNESS-CRITICAL vs SPEED-CRITICAL
------------------------------------
Exactness-critical (never change without re-running F2): the dequant ORDER in
w8a8_matmul, half-even vs half-away rounding in each operator, the f32/bf16 cast
points, the Q14 table grid, the rsqrt table construction (python round(), i.e.
half-EVEN, exactly as the old stack built it).
Speed-critical only (bit-invisible "schedule"): row chunking of the linears
(LOCKSTEP_FS_ROWCHUNK), which kernel computes the int8 GEMM, whether the tables
live on GPU or CPU, dump instrumentation.

INTEGRATION RULES (0.25.1)
--------------------------
* Call `arm()` BEFORE constructing `LLM(...)`.  vLLM's CustomOp base class binds
  `self._forward_method = self.forward_cuda` at MODULE CONSTRUCTION time, so a
  class-level patch installed after the model is built would be ignored; arm()
  patches the classes first and additionally re-binds `_forward_method` on every
  armed instance (so post-hoc arming also works).
* Per-module state (int8 weights, integer gains, Q14 tables) is armed from a
  `Worker.load_model` hook so it exists BEFORE vLLM's memory profiling and
  kernel warmup — the same seam vllm_contract4.py used.  The int8 weight copies
  (~7 GB for a 7B bf16 model) are therefore counted by the profiler and the KV
  cache is sized around them.
* `enforce_eager=True` is assumed (T2/T7a protocol).  With torch.compile /
  CUDA graphs ON, these python patches must instead be registered as
  torch.library custom ops with fake impls (vllm_contract4.py:209-243 shows the
  pattern) or the traced graph will bake in the stale float path.
* vLLM 0.25.1's kernel warmup drives REAL forwards with dummy data through every
  patched op; STATS counters include that warmup work.

This module py_compiles and imports with NEITHER torch NOR vLLM installed
(authoring-Mac rule): every heavy import is guarded and lazy.

Usage
-----
    import lockstep_fullstack as fs
    fs.arm()                     # class patches + load-model hook
    llm = LLM(..., enforce_eager=True)
    print(json.dumps(fs.report(), indent=1))

    python lockstep_fullstack.py --report     # introspection only
    python lockstep_fullstack.py --selftest   # CPU op self-tests (needs torch)
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover - authoring environment
    torch = None  # type: ignore[assignment]


# ======================================================================================
# Configuration
# ======================================================================================

ALL_OPS = ("rmsnorm", "rope", "linear", "logits", "silu")
DEFAULT_OPS = "rmsnorm,rope,linear,logits"

OPS = tuple(o.strip() for o in os.environ.get("LOCKSTEP_FS_OPS", DEFAULT_OPS).split(",")
            if o.strip())
VERBOSE = os.environ.get("LOCKSTEP_FS_VERBOSE", "1") == "1"
COUNT_SAT = os.environ.get("LOCKSTEP_FS_COUNT", "0") == "1"      # norm saturation audit
ROWCHUNK = int(os.environ.get("LOCKSTEP_FS_ROWCHUNK", "1024"))   # schedule only
SKIP_NAMES = tuple(s for s in os.environ.get("LOCKSTEP_FS_SKIP", "").split(",") if s)

# T16h: the lm_head arithmetic variant.  "a8" is TODAY'S CONTRACT and the default,
# and with it every code path below is byte-identical to the pre-T16h file.  Any
# other spec is resolved by demo/t16h_lmvariants.py (a15 / a11 / sq<AA>[a15]).
LMHEAD = (os.environ.get("LOCKSTEP_FS_LMHEAD", "a8").strip() or "a8")
LMHEAD_CAL = os.environ.get("LOCKSTEP_FS_LMHEAD_CAL", "").strip()   # act_amax .pt (sq family)
# T16h: the same knob for the BODY linears.  "a8" is today's contract and the
# default; "a15" / "a15w15" widen the activation (and weight) with the same exact
# two-plane int8 carrier, at 2x / 4x the GEMMs.
FSLINEAR = (os.environ.get("LOCKSTEP_FS_LINEAR", "a8").strip() or "a8")
FSLINEAR_CAL = os.environ.get("LOCKSTEP_FS_LINEAR_CAL", "").strip()  # per-linear act_amax .pt

# F2 instrumentation
DUMP_DIR = os.environ.get("LOCKSTEP_FS_DUMP", "")
DUMP_N = int(os.environ.get("LOCKSTEP_FS_DUMP_N", "2"))
DUMP_LAYERS = tuple(x for x in os.environ.get("LOCKSTEP_FS_DUMP_LAYERS", "0,13").split(",") if x)
DUMP_WMAX = int(os.environ.get("LOCKSTEP_FS_DUMP_WMAX_ROWS", "8192"))  # weight rows per dump
DUMP_MAXROWS = int(os.environ.get("LOCKSTEP_FS_DUMP_MAXROWS", "64"))   # skip huge prefill calls

# ======================================================================================
# Contract constants (ported from vllm_contract2.py / vllm_contract4.py)
# ======================================================================================

RSQRT_MBITS = 14
RSQRT_NM = 1 << RSQRT_MBITS
G_BASE = 131072                  # integer gain scale for RMSNorm weights (2^17)
NORM_Y_CLAMP = (1 << 23) - 1     # int24 output container of the CUDA norm kernel
ROPE_Q = 16384                   # Q14 cos/sin grid
INT8_MIN_ROWS = 17               # torch._int_mm requires M > 16 (golden pads to 17)
SILU_C = -1.4426950216293335     # 0xBFB8AA3B, binary32 nearest -log2(e)   (SPEC 6.0.6)
F32_MIN_NORMAL = 1.1754943508222875e-38

STATS: Dict[str, Any] = {
    "armed": False,
    "ops": list(OPS),
    "lmhead": LMHEAD,
    "fslinear": FSLINEAR,
    "fslinear_cal": FSLINEAR_CAL,
    "n_rmsnorm": 0, "n_rope": 0, "n_linear": 0, "n_silu": 0, "n_logits": 0,
    "skipped_linear": [],
    "calls_rmsnorm": 0, "calls_rope": 0, "calls_linear": 0, "calls_logits": 0,
    "calls_silu": 0,
    "norm_saturations": 0,       # |y| clipped by the int24 container (expect 0)
    "norm_wide": 0,              # |y| >= 2^23 seen before clamp (kernel-port warning)
    "linear_rows": 0, "logits_rows": 0,
    "dumps": {},
}

_PATCHED: Dict[str, Any] = {}     # op -> dict(target=..., attr=..., orig=...)
_RESOLVED: Dict[str, Any] = {}    # symbol -> {"module":..., "ok":..., "error":...}
_ARMED_MODULES: List[Any] = []    # instances we mutated (for disarm)


def _log(msg: str) -> None:
    if VERBOSE:
        print("[lockstep-fs] " + msg, flush=True)


# ======================================================================================
# Lazily built tables (device-cached).  Built exactly as the old stack built them.
# ======================================================================================

_RSQRT_LUT_HOST = None
_rsqrt_cache: Dict[Any, Any] = {}
_frac_cache: Dict[Any, Any] = {}


def rsqrt_lut(device=None):
    """Q14 reciprocal-square-root table, identical to vllm_contract2.py:31-33.

    LUT[(k&1)*2^14 + m] = round(32768 / sqrt((1 + m/2^14) * 2^(k&1))).
    NOTE python's round() is half-to-EVEN; that is how the shipped table was
    built and is part of the operator's definition here.
    """
    global _RSQRT_LUT_HOST
    if torch is None:
        raise RuntimeError("torch required")
    if _RSQRT_LUT_HOST is None:
        _RSQRT_LUT_HOST = torch.tensor(
            [round(32768.0 / math.sqrt((1.0 + mm / RSQRT_NM) * (2.0 ** p)))
             for p in (0, 1) for mm in range(RSQRT_NM)], dtype=torch.int64)
    key = str(device)
    if key not in _rsqrt_cache:
        _rsqrt_cache[key] = _RSQRT_LUT_HOST.to(device)
    return _rsqrt_cache[key]


def frac_table(device=None):
    """The 8192-entry frozen FRAC table (SPEC 6.1.3), from contract_common when
    available (digest-checked there) and rebuilt from the seven Q30 coefficients
    otherwise.  Used only by the opt-in contract SiLU."""
    if torch is None:
        raise RuntimeError("torch required")
    key = str(device)
    if key in _frac_cache:
        return _frac_cache[key]
    tab = None
    try:
        import contract_common as cc  # golden; pod: /workspace/p2/contract_common.py
        tab = cc.table_on(device, 13, 15)
    except Exception as e:  # pragma: no cover - fallback path
        _log("contract_common unavailable for FRAC (%r); rebuilding from coefficients" % (e,))
        coeffs = [1073741823, -744260975, 257938938, -59583045, 10286509, -1369790, 117454]
        vals = []
        for k in range(1 << 13):
            t = k << (20 - 13)
            p = coeffs[6]
            for c in coeffs[5::-1]:
                p = c + ((p * t) // (1 << 20))
            vals.append((p + (1 << (29 - 15))) >> (30 - 15))
        tab = torch.tensor(vals, dtype=torch.int64, device=device)
    _frac_cache[key] = tab
    return tab


# ======================================================================================
# Core operators — device-agnostic torch, usable on CPU for the golden recompute
# ======================================================================================

def pow2_f32(e):
    """EXACT 2^e as float32 from the exponent field (e integer tensor in [-126, 127]).  Not
    torch.ldexp: on torch 2.11.0+cu130 ldexp faults with an illegal memory access when its
    tensors live on a CUDA device other than the process's current one (SPEC 7.23, P3)."""
    return ((e.to(torch.int32) + 127) << 23).view(torch.float32)


def pow2_f64(e):
    """EXACT 2^e as float64 (e integer tensor in [-1022, 1023])."""
    return ((e.to(torch.int64) + 1023) << 52).view(torch.float64)


def pow2_exponent(m, bits: int):
    """Per-row power-of-two exponent by the contract quantizer rule
    (contract_common.quant_pow2 / SPEC 6.1.5), generalized from 7 to `bits`.

    m: non-negative amax per row (f32).  Returns int64 exponent e such that
    round(x * 2^e) fits in [-(2^bits - 1), 2^bits - 1].
    """
    lim = float((1 << bits) - 1)
    mant, E = torch.frexp(m)
    e = (bits - 1) - (E.to(torch.int64) - 1)
    e = torch.where(m == 0, torch.zeros_like(e), e)
    ones = torch.ones_like(m)
    over = m * pow2_f32(e) > lim
    return (e - over.to(torch.int64)).clamp(-60, 60)


NORM_EPS_BOUND = float(1 << 50)   # the eps term never exceeds 2^50: ss + eps term < 2^51, exact in binary64
_ecap_cache: Dict[float, int] = {}


def norm_ecap(ne: float) -> int:
    """The per-model cap on the norm's pow2 exponent e: the largest e in [0, 60] with
    fl64(N * eps) * 4^e <= 2^50 (60 when N * eps == 0).  Pure binary64 arithmetic on
    powers of two, so every host computes the same integer (t16l_kernels.cu::norm_ecap)."""
    r = _ecap_cache.get(ne)
    if r is None:
        r = 60
        if ne > 0.0:
            r = 0
            while r < 60 and math.ldexp(ne, 2 * (r + 1)) <= NORM_EPS_BOUND:
                r += 1
        _ecap_cache[ne] = r
    return r


NORMF = os.environ.get("T16L_NORMF", "1") == "1"    # contract 2.6: the float norm (t16n_ref.rmsnorm_float is the rule)


def rmsnorm_rule(x, weight, eps=0.0, out_dtype=None):
    """The norm under the active contract: the 2.5 integer rule, or the 2.6 float rule (T16L_NORMF=1)."""
    if NORMF:
        import t16n_ref
        return t16n_ref.rmsnorm_float(x, weight, out_dtype=out_dtype, eps=eps)
    return rmsnorm_contract(x, weight, eps, out_dtype=out_dtype)


def rmsnorm_contract(x, weight, eps=0.0, out_dtype=None):
    """Integer RMSNorm (§15 of the contract runner; vllm_contract2.py::int15_norm)
    with a per-row dynamic pow2 input scale and the model's eps carried as an
    integer term (contract v2.2):  ss += floor(fl64(N * eps) * 4^e), e <= norm_ecap.

    eps     : the RMSNorm epsilon of the model (rms_norm_eps); 0 gives the v2.1 op

    x       : [..., N] float tensor (f32 recommended; bf16 accepted)
    weight  : [N] float tensor (the RMSNorm gain)
    returns : same shape as x, dtype = out_dtype or x.dtype

    All reductions are exact integer sums over one row => the result is
    independent of how many rows are in the batch and of any tiling.
    """
    shp = x.shape
    N = shp[-1]
    x2 = x.reshape(-1, N).float()

    eps = float(eps or 0.0)
    ne = float(N) * eps                                                  # one binary64 multiply
    e = pow2_exponent(x2.abs().amax(dim=-1), 15)                        # [R]
    if ne > 0.0:
        e = e.clamp(max=norm_ecap(ne))
    scale = pow2_f32(e)                                                  # 2^e, exact
    x16 = torch.clamp(torch.round(x2 * scale[:, None]), -32767, 32767).to(torch.int64)

    ss = (x16 * x16).sum(dim=-1, dtype=torch.int64)                     # exact
    if ne > 0.0:
        # eps in the quantised domain: N * eps * 2^(2e), floored; exact scaling by a power of two
        et = torch.full_like(ss, ne, dtype=torch.float64) * pow2_f64(2 * e)   # exact scaling by 4^e
        ss = ss + torch.floor(et).to(torch.int64)
    ss = ss.clamp(min=1)
    _, ex = torch.frexp(ss.to(torch.float64))
    k = (ex.to(torch.int64) - 1)
    m_hi = (ss >> (k - RSQRT_MBITS).clamp(min=0)) & (RSQRT_NM - 1)
    m_lo = (ss << (RSQRT_MBITS - k).clamp(min=0)) & (RSQRT_NM - 1)
    m = torch.where(k >= RSQRT_MBITS, m_hi, m_lo)
    b = rsqrt_lut(x.device)[(k & 1) * RSQRT_NM + m]                     # [R]
    ts = (k >> 1)

    prod = x16 * b[:, None]
    t = prod.sign() * ((prod.abs() + (1 << 14)) >> 15)        # half away from zero
    G32 = _gain_int(weight)
    v = t * G32[None, :]
    half2 = (torch.ones_like(ts) << (ts - 1).clamp(min=0))[:, None]
    shifted = v.sign() * ((v.abs() + half2) >> ts.clamp(min=1)[:, None])
    y = torch.where(ts[:, None] > 0, shifted, v)

    if COUNT_SAT:                    # opt-in: costs one device sync per call
        wide = int((y.abs() > NORM_Y_CLAMP).sum().item())
        if wide:
            STATS["norm_wide"] += wide
            STATS["norm_saturations"] += wide
    y = y.clamp(-NORM_Y_CLAMP, NORM_Y_CLAMP)

    out = y.float() * (math.sqrt(N) / G_BASE)
    return out.reshape(shp).to(out_dtype or x.dtype)


_gain_cache: Dict[int, Any] = {}


def _gain_int(weight):
    """G = round(w * 131072) as int64 (vllm_contract4.py:_arm).  Cached per weight
    tensor identity so a decode step does not re-quantize the gains."""
    key = id(weight)
    ent = _gain_cache.get(key)
    if ent is not None and ent[0] is weight:
        return ent[1]
    g = torch.round(weight.detach().float() * G_BASE).to(torch.int64)
    _gain_cache[key] = (weight, g)
    return g


def rope_contract(positions, query, key, cos_sin_q14, head_size, rotary_dim,
                  is_neox_style=True):
    """Frozen-Q14 RoPE evaluated in f32 (vllm_contract2.py::rope_q14 semantics,
    vllm_contract4.py's deai::rope arithmetic).

    positions   : [T] int64
    query       : [T, H*head_size] or [T, H, head_size]
    cos_sin_q14 : [max_pos, rotary_dim] f32 already on the Q14 grid
    """
    qshape = query.shape
    T = positions.numel()
    cs = cos_sin_q14.index_select(0, positions.reshape(-1))
    cos, sin = cs.chunk(2, dim=-1)                     # [T, rotary_dim/2]
    cos = cos.unsqueeze(-2)                            # [T, 1, rd/2]
    sin = sin.unsqueeze(-2)

    def _rot(t):
        if t is None:
            return None
        v = t.reshape(T, -1, head_size).float()
        rot, pas = v[..., :rotary_dim], v[..., rotary_dim:]
        if is_neox_style:
            x1, x2 = torch.chunk(rot, 2, dim=-1)
        else:
            x1, x2 = rot[..., ::2], rot[..., 1::2]
        o1 = x1 * cos - x2 * sin                       # three separate CR ops
        o2 = x2 * cos + x1 * sin
        if is_neox_style:
            o = torch.cat((o1, o2), dim=-1)
        else:
            o = torch.stack((o1, o2), dim=-1).flatten(-2)
        if pas.shape[-1]:
            o = torch.cat((o, pas), dim=-1)
        return o.to(t.dtype).reshape(t.shape)

    q = _rot(query).reshape(qshape)
    return q, (None if key is None else _rot(key))


_CONST_CACHE: Dict[Any, Any] = {}


def _const(value: float, ref):
    """A device/dtype-resident scalar TENSOR, so that `x / _const(v, x)` is a real
    correctly-rounded division on every device.

    MEASURED on this box (t5_divprobe.py, H100 / torch 2.11): PyTorch compiles
    `tensor / python_scalar` into RECIPROCAL-MULTIPLY on CUDA and into a true
    division on CPU, so `amax / 127.0` differs between the two devices by 1 ulp
    on ~4% of channels — exactly the substitution SPECIFICATION 6.0.3 prohibits.
    With a tensor divisor both devices produce the correctly rounded quotient
    (and agree with numpy). NOTE this makes the operator DEVICE-INVARIANT and
    therefore 1 ulp away, on those channels, from contract_common.w8a8_linear
    executed ON A GPU (which takes PyTorch's reciprocal path); it agrees with the
    same golden code executed on CPU. Device-invariance is the product, so the
    correctly rounded form wins; see demo/T5_LOG.md."""
    key = (float(value), str(ref.device), ref.dtype)
    t = _CONST_CACHE.get(key)
    if t is None:
        t = torch.tensor(float(value), device=ref.device, dtype=ref.dtype)
        _CONST_CACHE[key] = t
    return t


def quantize_rows_int8(x2):
    """Per-token dynamic int8 quantization, golden semantics
    (contract_common.w8a8_linear): asc = amax(|x|)/127, half-even round, clamp."""
    asc = (x2.abs().amax(dim=1).clamp(min=1e-8) / _const(127.0, x2))
    x8 = torch.clamp(torch.round(x2 / asc[:, None]), -127, 127).to(torch.int8)
    return x8, asc


def _int_mm_exact(a8, b8):
    """int8 x int8 -> int32 matrix product.  CUDA: torch._int_mm (cuBLASLt, int32
    accumulation, exact at our shapes).  CPU/fallback: int64 matmul, then narrow —
    numerically identical because no accumulation overflows int32."""
    n = a8.shape[0]
    pad = (INT8_MIN_ROWS - n) if n < INT8_MIN_ROWS else 0
    if pad:
        a8 = torch.cat([a8, torch.zeros(pad, a8.shape[1], dtype=torch.int8,
                                        device=a8.device)])
    try:
        y = torch._int_mm(a8, b8)
    except Exception as e:
        if a8.is_cuda:                      # loud: never silently change the kernel class
            raise RuntimeError("torch._int_mm failed on CUDA for shapes %s x %s: %r"
                               % (list(a8.shape), list(b8.shape), e))
        y = torch.matmul(a8.to(torch.int64), b8.to(torch.int64)).to(torch.int32)
    return y[:n]


def w8a8_matmul(x, W8, ws, bias=None, out_dtype=None, rowchunk: int = 0):
    """The golden W8A8 linear (contract_common.w8a8_linear), row-chunked.

    x    : [..., K] float;  W8 : [N, K] int8;  ws : [N] f32 weight scales
    bias : [N] float or None (added in f32, as the golden does)
    Chunking rows is bit-invisible (per-token scales; integer accumulation)."""
    shp = x.shape
    x2 = x.reshape(-1, shp[-1]).float()
    rows = x2.shape[0]
    if rows == 0:                      # legal (e.g. an MoE expert routed no tokens)
        return torch.zeros(*shp[:-1], W8.shape[0], dtype=out_dtype or x.dtype,
                           device=x.device)
    step = rowchunk or ROWCHUNK or rows
    W8t = W8.t()
    outs = []
    for r0 in range(0, max(rows, 1), step):
        r1 = min(r0 + step, rows)
        if r1 <= r0:
            break
        xc = x2[r0:r1]
        x8, asc = quantize_rows_int8(xc)
        y = (_int_mm_exact(x8, W8t).float() * ws[None, :]) * asc[:, None]   # v2.1 order
        y = y.to(out_dtype or x.dtype)
        if bias is not None:
            y = (y.float() + bias.float()).to(out_dtype or x.dtype)
        outs.append(y)
    y = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
    return y.reshape(*shp[:-1], W8.shape[0])


def fwht_blocks(x2, blk=256, scale=1.0 / 16.0):
    """The R4 / online-R1 butterfly on every `blk`-wide block of the last dim (the declared
    pinned-order fp32 butterfly: pair (i, i+h) -> (a+b, a-b), h = 1, 2, ..., blk/2, then one
    exact multiply by `scale`)."""
    sh = x2.shape
    D = sh[-1]
    nb = D // blk
    v = x2.float().reshape(-1, nb, blk)
    h = 1
    while h < blk:
        v = v.view(-1, nb, blk // (2 * h), 2, h)
        v = torch.stack((v[..., 0, :] + v[..., 1, :], v[..., 0, :] - v[..., 1, :]), dim=-2).reshape(-1, nb, blk)
        h *= 2
    return (v * scale).reshape(sh)


def outlier_side(ho, wo):
    """Contract v2.5 side path: side[m, n] = sum_{k in order} fl32(ho[m, k]) * fl32(wo[n, k]).
    ho [M, k] and wo [N, k] carry bf16 values, so each product is exact in binary32; the sum
    is sequential in k with binary32 round-to-nearest adds (the kernels' __fadd_rn chain)."""
    h = ho.float(); w = wo.float()
    side = torch.zeros(h.shape[0], w.shape[0], dtype=torch.float32, device=h.device)
    for k in range(h.shape[1]):
        side = side + h[:, k:k + 1] * w[None, :, k]
    return side


def w8a8_outlier_linear(x, oidx, wo, W8, ws, bias=None, out_dtype=None, rot=True):
    """The v2.5 norm-fed linear under R1 online (contract section 6.6): the declared outlier
    channels `oidx` of the bf16 row leave it as their exact values (ho), the rest takes the
    256-block butterfly / 16 and the per-token int8 quant, the GEMM runs on the published
    int8 weight (the rotated weight with those columns zeroed before the rounding), and
        y = bf16( fl32( bf16((fl32(acc) * ws[n]) * asc[m]) ) + side[m, n] ),  then the bias
    x [..., K] bf16 (or float holding bf16 values); oidx int [k]; wo [N, k] bf16; W8 [N, K]."""
    shp = x.shape
    x2 = x.reshape(-1, shp[-1]).float()
    idx = oidx.long()
    ho = x2[:, idx]
    xm = x2.clone(); xm[:, idx] = 0.0
    xr = fwht_blocks(xm) if rot else xm             # rot=False: the o_proj input (v2.5.1, no rotation)
    x8, asc = quantize_rows_int8(xr)
    y = ((_int_mm_exact(x8, W8.t()).float() * ws[None, :]) * asc[:, None]).to(torch.bfloat16)
    y = (y.float() + outlier_side(ho, wo)).to(torch.bfloat16)
    if bias is not None:
        y = (y.float() + bias.float()).to(torch.bfloat16)
    return y.to(out_dtype or x.dtype).reshape(*shp[:-1], W8.shape[0])


def quantize_weight_int8(W):
    """Per-out-channel int8 weight, golden semantics (contract_common).
    The /127 uses a tensor divisor — see `_const` for the measured reason."""
    Wf = W.detach().float()
    ws = Wf.abs().amax(dim=1).clamp(min=1e-8) / _const(127.0, Wf)
    W8 = torch.clamp(torch.round(Wf / ws[:, None]), -127, 127).to(torch.int8).contiguous()
    return W8, ws.contiguous()


def silu_contract(x):
    """SPECIFICATION 6.3.1: invariant sigmoid / SiLU, ten declared binary32 steps.
    x must be f32 (caller upcasts); returns f32."""
    frac = frac_table(x.device)
    xc = torch.where(x.abs() < F32_MIN_NORMAL, torch.zeros_like(x), x)   # 6.0.4
    t = xc * SILU_C                                     # (1) CR multiply
    t = torch.clamp(t, -30.0, 30.0)                     # (2)
    i = torch.ceil(t)                                   # (3) exact
    g = i - t                                           # (4) CR subtract
    k = torch.clamp(torch.trunc(g * 8192.0), max=8191.0).to(torch.int64)   # (5)
    f = frac[k].to(torch.float32) * (2.0 ** -15)        # (6) exact
    e2 = f * pow2_f32(i.to(torch.int64))                # (7) exact: f * 2^i, i in [-30, 30]
    den = 1.0 + e2                                      # (8) CR add
    sig = 1.0 / den                                     # (9) CR divide
    return xc * sig                                     # (10) CR multiply


# ======================================================================================
# Defensive resolution of the 0.25.1 symbols we patch
# ======================================================================================

def _try_import(paths: List[str], attr: str, key: str):
    errs = []
    for p in paths:
        try:
            mod = importlib.import_module(p)
        except Exception as e:
            errs.append("%s: %r" % (p, e))
            continue
        obj = getattr(mod, attr, None)
        if obj is None:
            errs.append("%s: no attr %s" % (p, attr))
            continue
        _RESOLVED[key] = {"module": p, "attr": attr, "ok": True,
                          "qualname": getattr(obj, "__qualname__", str(obj))}
        return obj
    _RESOLVED[key] = {"ok": False, "tried": paths, "attr": attr, "errors": errs}
    return None


def resolve_all() -> Dict[str, Any]:
    """Locate every vLLM class/function this module patches.  Never raises."""
    _RESOLVED.clear()
    out = {}
    out["RMSNorm"] = _try_import(
        ["vllm.model_executor.layers.layernorm"], "RMSNorm", "RMSNorm")
    out["RotaryEmbedding"] = _try_import(
        ["vllm.model_executor.layers.rotary_embedding.base",
         "vllm.model_executor.layers.rotary_embedding"],
        "RotaryEmbedding", "RotaryEmbedding")
    out["UnquantizedLinearMethod"] = _try_import(
        ["vllm.model_executor.layers.linear"], "UnquantizedLinearMethod",
        "UnquantizedLinearMethod")
    out["LinearBase"] = _try_import(
        ["vllm.model_executor.layers.linear"], "LinearBase", "LinearBase")
    out["LogitsProcessor"] = _try_import(
        ["vllm.model_executor.layers.logits_processor"], "LogitsProcessor",
        "LogitsProcessor")
    out["SiluAndMul"] = _try_import(
        ["vllm.model_executor.layers.activation"], "SiluAndMul", "SiluAndMul")
    out["Worker"] = _try_import(
        ["vllm.v1.worker.gpu_worker"], "Worker", "Worker")
    out["ParallelLMHead"] = _try_import(
        ["vllm.model_executor.layers.vocab_parallel_embedding"], "ParallelLMHead",
        "ParallelLMHead")
    # signatures help catch a renamed/reordered API early
    try:
        import inspect
        lp = out.get("LogitsProcessor")
        if lp is not None and hasattr(lp, "_get_logits"):
            _RESOLVED["LogitsProcessor"]["_get_logits_sig"] = str(
                inspect.signature(lp._get_logits))
        um = out.get("UnquantizedLinearMethod")
        if um is not None:
            _RESOLVED["UnquantizedLinearMethod"]["apply_sig"] = str(
                inspect.signature(um.apply))
        rn = out.get("RMSNorm")
        if rn is not None:
            _RESOLVED["RMSNorm"]["forward_native_sig"] = str(
                inspect.signature(rn.forward_native))
        ro = out.get("RotaryEmbedding")
        if ro is not None:
            _RESOLVED["RotaryEmbedding"]["forward_native_sig"] = str(
                inspect.signature(ro.forward_native))
    except Exception as e:  # pragma: no cover
        _RESOLVED["signature_error"] = repr(e)
    return out


# ======================================================================================
# F2 dump instrumentation
# ======================================================================================

def _dump_enabled(op: str, name: str = "") -> bool:
    if not DUMP_DIR:
        return False
    n = STATS["dumps"].get(op + "|" + name, 0)
    return n < DUMP_N


def _dump(op: str, name: str, payload: Dict[str, Any]) -> None:
    key = op + "|" + name
    n = STATS["dumps"].get(key, 0)
    STATS["dumps"][key] = n + 1
    try:
        os.makedirs(DUMP_DIR, exist_ok=True)
        safe = name.replace("/", "_").replace(".", "_") or "x"
        path = os.path.join(DUMP_DIR, "%s__%s__%d.pt" % (op, safe, n))
        cpu = {}
        for k, v in payload.items():
            cpu[k] = v.detach().to("cpu").clone() if hasattr(v, "detach") else v
        torch.save(cpu, path)
    except Exception as e:  # pragma: no cover
        _log("dump failed (%s): %r" % (key, e))


def _layer_of(name: str) -> Optional[str]:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _dump_layer_ok(name: str) -> bool:
    lay = _layer_of(name)
    return (lay is None) or (not DUMP_LAYERS) or (lay in DUMP_LAYERS)


# ======================================================================================
# Patched forwards
# ======================================================================================

def _install_rmsnorm(RMSNorm) -> bool:
    orig_native = RMSNorm.forward_native
    orig_cuda = getattr(RMSNorm, "forward_cuda", None)

    def forward_contract(self, x, residual=None):
        if not getattr(self, "_ls_on", False):
            return orig_native(self, x, residual)
        STATS["calls_rmsnorm"] += 1
        orig_dtype = x.dtype
        x32 = x.float()
        res_out = None
        if residual is not None:
            x32 = x32 + residual.float()          # vLLM's own residual semantics
            res_out = x32.to(orig_dtype)
        y = rmsnorm_rule(x32, self.weight, float(getattr(self, "variance_epsilon", 0.0)),
                             out_dtype=orig_dtype)
        nm = getattr(self, "_ls_name", "")
        if _dump_enabled("rmsnorm", nm) and _dump_layer_ok(nm) and \
                x32.reshape(-1, x32.shape[-1]).shape[0] <= DUMP_MAXROWS:
            _dump("rmsnorm", nm, {"x32": x32, "weight": self.weight, "y": y,
                                  "orig_dtype": str(orig_dtype)})
        if residual is not None:
            return y, res_out
        return y

    RMSNorm.forward_native = forward_contract
    if orig_cuda is not None:
        RMSNorm.forward_cuda = forward_contract
    _PATCHED["rmsnorm"] = {"cls": RMSNorm, "forward_native": orig_native,
                           "forward_cuda": orig_cuda, "fn": forward_contract}
    return True


def _install_rope(RotaryEmbedding) -> bool:
    orig_native = RotaryEmbedding.forward_native
    orig_cuda = getattr(RotaryEmbedding, "forward_cuda", None)

    def forward_contract(self, positions, query, key=None, offsets=None):
        cache = getattr(self, "_ls_q14", None)
        if cache is None:
            return orig_native(self, positions, query, key)
        STATS["calls_rope"] += 1
        pos = positions.reshape(-1)
        if offsets is not None:
            pos = pos + offsets.reshape(-1)
        q, k = rope_contract(pos, query, key, cache, self.head_size,
                             self.rotary_dim, getattr(self, "is_neox_style", True))
        nm = getattr(self, "_ls_name", "rope")
        if _dump_enabled("rope", nm) and pos.numel() <= DUMP_MAXROWS:
            _dump("rope", nm, {"positions": pos, "query": query, "key": key,
                               "cache": cache.index_select(0, pos), "q": q, "k": k,
                               "head_size": self.head_size,
                               "rotary_dim": self.rotary_dim,
                               "is_neox": bool(getattr(self, "is_neox_style", True))})
        return q, k          # vLLM's own forward_native returns a 2-tuple always

    RotaryEmbedding.forward_native = forward_contract
    if orig_cuda is not None:
        RotaryEmbedding.forward_cuda = forward_contract
    _PATCHED["rope"] = {"cls": RotaryEmbedding, "forward_native": orig_native,
                        "forward_cuda": orig_cuda, "fn": forward_contract}
    return True


def _install_linear(UnquantizedLinearMethod) -> bool:
    orig_apply = UnquantizedLinearMethod.apply

    def apply_contract(self, layer, x, bias=None):
        W8 = getattr(layer, "_ls_w8", None)
        if W8 is None:
            return orig_apply(self, layer, x, bias)
        STATS["calls_linear"] += 1
        rows = 1
        for d in x.shape[:-1]:
            rows *= d
        STATS["linear_rows"] += rows
        var = getattr(layer, "_ls_var", None)
        if var is None:
            y = w8a8_matmul(x, W8, layer._ls_ws, bias=bias, out_dtype=x.dtype)
        else:                                   # T16h linear variant (a15 / a15w15)
            shp = x.shape
            x2 = x.reshape(-1, shp[-1]).float()
            n = x2.shape[0]
            step = ROWCHUNK or n or 1
            parts = [var.logits(x2[r0:min(r0 + step, n)])
                     for r0 in range(0, max(n, 1), step) if r0 < n]
            y = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            if n == 0:
                y = torch.zeros(0, var.n_out(), dtype=torch.float32, device=x2.device)
            if bias is not None:
                y = y + bias.float()
            y = y.reshape(*shp[:-1], var.n_out()).to(x.dtype)
        nm = getattr(layer, "_ls_name", "")
        if _dump_enabled("linear", nm) and _dump_layer_ok(nm) and rows <= DUMP_MAXROWS:
            # dump the RAW float weight slice (not W8): the F2 checker then runs
            # contract_common.w8a8_linear itself — the golden code, unmodified.
            wr = min(W8.shape[0], DUMP_WMAX)
            _dump("linear", nm, {"x": x, "w": layer.weight.data[:wr],
                                 "ws": layer._ls_ws[:wr],
                                 "bias": (None if bias is None else bias[:wr]),
                                 "y": y[..., :wr], "w_rows": wr,
                                 "out_dtype": str(x.dtype)})
        return y

    UnquantizedLinearMethod.apply = apply_contract
    _PATCHED["linear"] = {"cls": UnquantizedLinearMethod, "apply": orig_apply,
                          "fn": apply_contract}
    return True


def _install_logits(LogitsProcessor) -> bool:
    orig = getattr(LogitsProcessor, "_get_logits", None)
    if orig is None:
        _RESOLVED.setdefault("LogitsProcessor", {})["_get_logits"] = "MISSING"
        return False

    def get_logits_contract(self, hidden_states, lm_head, embedding_bias=None, **kw):
        W8 = getattr(lm_head, "_ls_w8", None)
        if W8 is None:
            return orig(self, hidden_states, lm_head, embedding_bias, **kw)
        STATS["calls_logits"] += 1
        x2 = hidden_states.reshape(-1, hidden_states.shape[-1])
        STATS["logits_rows"] += x2.shape[0]
        var = getattr(lm_head, "_ls_var", None)
        if var is None:
            y = w8a8_matmul(x2, W8, lm_head._ls_ws, bias=embedding_bias,
                            out_dtype=hidden_states.dtype)
        else:                                   # T16h lm_head variant (a15 / sq...)
            rows = x2.shape[0]
            step = ROWCHUNK or rows or 1
            parts = [var.logits(x2[r0:min(r0 + step, rows)].float())
                     for r0 in range(0, max(rows, 1), step) if r0 < rows]
            y = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            if rows == 0:
                y = torch.zeros(0, var.n_out(), dtype=torch.float32, device=x2.device)
            if embedding_bias is not None:
                y = y + embedding_bias.float()
            y = y.to(hidden_states.dtype)
        if _dump_enabled("logits", "lm_head") and x2.shape[0] <= DUMP_MAXROWS:
            wr = min(W8.shape[0], DUMP_WMAX)
            _dump("logits", "lm_head", {"x": x2, "w": lm_head.weight.data[:wr],
                                        "ws": lm_head._ls_ws[:wr], "y": y[..., :wr],
                                        "w_rows": wr,
                                        "out_dtype": str(hidden_states.dtype)})
        scale = getattr(self, "scale", 1.0)
        if scale is not None and scale != 1.0:
            y = y * scale
        gather = getattr(self, "_gather_logits", None)
        if gather is not None:
            y = gather(y)
        else:                                        # TP=1 fallback
            try:
                from vllm.distributed import tensor_model_parallel_gather
                y = tensor_model_parallel_gather(y)
            except Exception:
                pass
        ovs = getattr(lm_head, "org_vocab_size", None)
        if y is not None and ovs is not None:
            y = y[..., :ovs]
        return y

    LogitsProcessor._get_logits = get_logits_contract
    _PATCHED["logits"] = {"cls": LogitsProcessor, "_get_logits": orig,
                          "fn": get_logits_contract}
    return True


def _install_silu(SiluAndMul) -> bool:
    orig_native = SiluAndMul.forward_native
    orig_cuda = getattr(SiluAndMul, "forward_cuda", None)

    def forward_contract(self, x):
        if not getattr(self, "_ls_on", False):
            return orig_native(self, x)
        STATS["calls_silu"] += 1
        d = x.shape[-1] // 2
        x32 = x.float()
        y = silu_contract(x32[..., :d]) * x32[..., d:]
        out = y.to(x.dtype)
        if _dump_enabled("silu", "act") and x.reshape(-1, x.shape[-1]).shape[0] <= DUMP_MAXROWS:
            _dump("silu", "act", {"x32": x32, "y": out, "out_dtype": str(x.dtype)})
        return out

    SiluAndMul.forward_native = forward_contract
    if orig_cuda is not None:
        SiluAndMul.forward_cuda = forward_contract
    _PATCHED["silu"] = {"cls": SiluAndMul, "forward_native": orig_native,
                        "forward_cuda": orig_cuda, "fn": forward_contract}
    return True


# ======================================================================================
# Arming
# ======================================================================================

def _rebind(mod, fn) -> None:
    """vLLM CustomOp binds self._forward_method at construction; re-point it."""
    try:
        import types
        if hasattr(mod, "_forward_method"):
            mod._forward_method = types.MethodType(fn, mod)
    except Exception as e:  # pragma: no cover
        _log("rebind failed: %r" % (e,))


def arm_model(model) -> Dict[str, int]:
    """Arm per-module state on an already-constructed vLLM model.

    Safe to call more than once (idempotent).  Returns the armed counts."""
    if torch is None:
        raise RuntimeError("torch required to arm a model")
    cls = resolve_all()
    RMSNorm = cls.get("RMSNorm")
    RotaryEmbedding = cls.get("RotaryEmbedding")
    LinearBase = cls.get("LinearBase")
    SiluAndMul = cls.get("SiluAndMul")
    ParallelLMHead = cls.get("ParallelLMHead")

    n_norm = n_rope = n_lin = n_silu = n_head = 0
    skipped: List[str] = []
    lm_heads = []

    for name, mod in model.named_modules():
        try:
            mod._ls_name = name
        except Exception:
            pass

        if "rmsnorm" in OPS and RMSNorm is not None and isinstance(mod, RMSNorm):
            mod._ls_on = True
            _gain_int(mod.weight)                       # pre-quantize the gain
            if "rmsnorm" in _PATCHED:
                _rebind(mod, _PATCHED["rmsnorm"]["fn"])
            _ARMED_MODULES.append(mod)
            n_norm += 1

        if "rope" in OPS and RotaryEmbedding is not None and isinstance(mod, RotaryEmbedding):
            cc = getattr(mod, "cos_sin_cache", None)
            if cc is not None:
                q14 = (torch.round(cc.detach().float() * ROPE_Q)
                       .clamp(-2 * ROPE_Q, 2 * ROPE_Q) / ROPE_Q)
                mod._ls_q14 = q14.contiguous()          # f32, exactly representable
                # keep vLLM's own cache on the same grid so any unpatched reader agrees
                mod.cos_sin_cache = q14.to(cc.dtype)
                if "rope" in _PATCHED:
                    _rebind(mod, _PATCHED["rope"]["fn"])
                _ARMED_MODULES.append(mod)
                n_rope += 1

        if "silu" in OPS and SiluAndMul is not None and isinstance(mod, SiluAndMul):
            mod._ls_on = True
            if "silu" in _PATCHED:
                _rebind(mod, _PATCHED["silu"]["fn"])
            _ARMED_MODULES.append(mod)
            n_silu += 1

        if ParallelLMHead is not None and isinstance(mod, ParallelLMHead):
            lm_heads.append((name, mod))

    # ---- linears -------------------------------------------------------------
    if "linear" in OPS and LinearBase is not None:
        for name, mod in model.named_modules():
            if not isinstance(mod, LinearBase):
                continue
            if getattr(mod, "_ls_w8", None) is not None:
                n_lin += 1
                continue
            leaf = name.split(".")[-1]
            if leaf == "gate":                 # MoE router stays float (contract_common)
                skipped.append(name + " (MoE router)")
                continue
            if any(s and s in name for s in SKIP_NAMES):
                skipped.append(name + " (env skip)")
                continue
            if any(mod is h for _, h in lm_heads) or "lm_head" in name:
                skipped.append(name + " (lm_head -> logits path)")
                continue
            w = getattr(mod, "weight", None)
            if w is None or not torch.is_floating_point(w):
                skipped.append(name + " (non-float weight: %s)" %
                               (None if w is None else w.dtype))
                continue
            qm = getattr(mod, "quant_method", None)
            qmn = type(qm).__name__ if qm is not None else "None"
            if "Unquantized" not in qmn:
                skipped.append(name + " (quant_method=%s)" % qmn)
                continue
            W8, ws = quantize_weight_int8(w.data)
            mod._ls_w8, mod._ls_ws = W8, ws
            if FSLINEAR != "a8":
                import t16h_lmvariants as _LV
                aa = None
                if _LV.needs_calibration([FSLINEAR]):
                    global _LINCAL
                    if _LINCAL is None:
                        if not FSLINEAR_CAL:
                            raise RuntimeError("LOCKSTEP_FS_LINEAR=%s needs "
                                               "LOCKSTEP_FS_LINEAR_CAL=<lin_amax .pt>" % FSLINEAR)
                        _LINCAL = torch.load(FSLINEAR_CAL, map_location="cpu")["amax"]
                        _log("linear SmoothQuant calibration loaded: %d linears from %s"
                             % (len(_LINCAL), FSLINEAR_CAL))
                    if name not in _LINCAL:
                        raise RuntimeError("no calibrated act_amax for linear %r "
                                           "(artifact has %d entries)" % (name, len(_LINCAL)))
                    aa = _LINCAL[name].to(w.device)
                mod._ls_var = _LV.build(FSLINEAR).prepare(w.data, aa)
            _ARMED_MODULES.append(mod)
            n_lin += 1

    # ---- lm_head / logits ----------------------------------------------------
    if "logits" in OPS:
        for name, mod in lm_heads:
            w = getattr(mod, "weight", None)
            if w is None or not torch.is_floating_point(w):
                skipped.append(name + " (lm_head non-float)")
                continue
            if getattr(mod, "_ls_w8", None) is None:
                W8, ws = quantize_weight_int8(w.data)
                mod._ls_w8, mod._ls_ws = W8, ws
                _ARMED_MODULES.append(mod)
            if LMHEAD != "a8" and getattr(mod, "_ls_var", None) is None:
                import t16h_lmvariants as _LV
                aa = None
                if _LV.needs_calibration([LMHEAD]):
                    if not LMHEAD_CAL:
                        raise RuntimeError("LOCKSTEP_FS_LMHEAD=%s needs "
                                           "LOCKSTEP_FS_LMHEAD_CAL=<act_amax .pt>" % LMHEAD)
                    aa = torch.load(LMHEAD_CAL, map_location=w.device)["act_amax"]
                mod._ls_var = _LV.build(LMHEAD).prepare(w.data, aa)
                _log("lm_head variant %s armed (%s)" % (LMHEAD, LMHEAD_CAL or "no calib"))
            n_head += 1

    STATS.update({"n_rmsnorm": n_norm, "n_rope": n_rope, "n_linear": n_lin,
                  "n_silu": n_silu, "n_logits": n_head, "skipped_linear": skipped})
    _log("armed: norms=%d rope=%d linears=%d lm_head=%d silu=%d (skipped %d)"
         % (n_norm, n_rope, n_lin, n_head, n_silu, len(skipped)))
    for s in skipped[:12]:
        _log("  skip: " + s)
    return {"rmsnorm": n_norm, "rope": n_rope, "linear": n_lin,
            "logits": n_head, "silu": n_silu}


_HOOKED = False
_LINCAL = None      # T16h: per-linear SmoothQuant act_amax artifact, loaded once


def _install_loader_hook(Worker) -> bool:
    """Arm module state at Worker.load_model — BEFORE vLLM's memory profiling and
    kernel warmup (the seam vllm_contract4.py:install_loader_hook used)."""
    global _HOOKED
    if _HOOKED:
        return True
    orig = Worker.load_model

    def load_model_hook(self, *a, **kw):
        out = orig(self, *a, **kw)
        try:
            model = self.model_runner.get_model()
        except Exception:
            model = getattr(getattr(self, "model_runner", None), "model", None)
        if model is None:
            raise RuntimeError("lockstep-fs: could not reach the model from Worker")
        counts = arm_model(model)
        print("LOCKSTEP-FS armed_at_load " + json.dumps(counts), flush=True)
        return out

    Worker.load_model = load_model_hook
    _PATCHED["loader_hook"] = {"cls": Worker, "load_model": orig}
    _HOOKED = True
    return True


def arm(model=None) -> Dict[str, Any]:
    """Install the class-level patches (and the load-model hook).  Call BEFORE
    constructing LLM().  If `model` is given, per-module state is armed now."""
    if torch is None:
        raise RuntimeError("torch required")
    torch.backends.cuda.matmul.allow_tf32 = False        # contract rule (6.0.3)
    try:
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    cls = resolve_all()
    missing = []
    if "rmsnorm" in OPS:
        if cls.get("RMSNorm") is None or not _install_rmsnorm(cls["RMSNorm"]):
            missing.append("rmsnorm")
    if "rope" in OPS:
        if cls.get("RotaryEmbedding") is None or not _install_rope(cls["RotaryEmbedding"]):
            missing.append("rope")
    if "linear" in OPS:
        if cls.get("UnquantizedLinearMethod") is None or \
                not _install_linear(cls["UnquantizedLinearMethod"]):
            missing.append("linear")
    if "logits" in OPS:
        if cls.get("LogitsProcessor") is None or not _install_logits(cls["LogitsProcessor"]):
            missing.append("logits")
    if "silu" in OPS:
        if cls.get("SiluAndMul") is None or not _install_silu(cls["SiluAndMul"]):
            missing.append("silu")
    if missing:
        raise RuntimeError("lockstep-fs: could not patch %s — resolution report: %s"
                           % (missing, json.dumps(_RESOLVED, default=str)))

    if cls.get("Worker") is not None:
        _install_loader_hook(cls["Worker"])
    else:
        _log("WARNING: vllm.v1.worker.gpu_worker.Worker not found; call "
             "arm_model(model) yourself after load")

    STATS["armed"] = True
    _log("class patches installed for: %s" % (",".join(sorted(_PATCHED)),))
    if model is not None:
        arm_model(model)
    return report()


def disarm() -> None:
    """Restore stock behaviour (class methods + per-module state)."""
    for mod in _ARMED_MODULES:
        for attr in ("_ls_on", "_ls_q14", "_ls_w8", "_ls_ws"):
            if hasattr(mod, attr):
                try:
                    delattr(mod, attr)
                except Exception:
                    pass
    _ARMED_MODULES.clear()
    for op, ent in list(_PATCHED.items()):
        cls = ent.get("cls")
        for attr in ("forward_native", "forward_cuda", "apply", "_get_logits",
                     "load_model"):
            if attr in ent and ent[attr] is not None:
                try:
                    setattr(cls, attr, ent[attr])
                except Exception:
                    pass
    _PATCHED.clear()
    STATS["armed"] = False
    _log("disarmed")


# ======================================================================================
# Introspection
# ======================================================================================

def report() -> Dict[str, Any]:
    if not _RESOLVED:
        try:
            resolve_all()
        except Exception as e:  # pragma: no cover
            _RESOLVED["resolve_error"] = repr(e)
    rep: Dict[str, Any] = {
        "have_torch": torch is not None,
        "ops_requested": list(OPS),
        "ops_patched": sorted(k for k in _PATCHED if k in ALL_OPS),
        "rowchunk": ROWCHUNK,
        "dump_dir": DUMP_DIR or None,
        "resolved": dict(_RESOLVED),
        "stats": {k: v for k, v in STATS.items() if k != "dumps"},
        "dumps": STATS["dumps"],
        "constants": {"G_BASE": G_BASE, "RSQRT_MBITS": RSQRT_MBITS, "NORM_EPS_BOUND": NORM_EPS_BOUND,
                      "NORM_Y_CLAMP": NORM_Y_CLAMP, "ROPE_Q": ROPE_Q,
                      "INT8_MIN_ROWS": INT8_MIN_ROWS},
    }
    try:
        import vllm
        rep["vllm_version"] = getattr(vllm, "__version__", "unknown")
    except Exception as e:
        rep["vllm_version"] = "unavailable (%r)" % (e,)
    if torch is not None:
        rep["torch_version"] = torch.__version__
        rep["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    return rep


# ======================================================================================
# CPU self-test (needs torch only; no vLLM)
# ======================================================================================

def selftest(verbose: bool = True) -> Dict[str, Any]:
    """Sanity + determinism checks for every operator, on CPU (and CUDA if
    present: the CPU/CUDA comparison is the F2 method in miniature)."""
    if torch is None:
        raise RuntimeError("torch required")
    res: Dict[str, Any] = {}
    torch.manual_seed(0)
    K, N, R = 256, 128, 33
    x = (torch.randn(R, K) * 3).to(torch.bfloat16).float()
    W = torch.randn(N, K)
    W8, ws = quantize_weight_int8(W)

    # 1. w8a8 vs the golden formula, and chunk invariance
    y_full = w8a8_matmul(x, W8, ws, rowchunk=10 ** 9)
    y_chunk = w8a8_matmul(x, W8, ws, rowchunk=7)
    res["w8a8_chunk_invariant"] = bool(torch.equal(y_full, y_chunk))
    x2 = x.float()
    asc = (x2.abs().amax(dim=1).clamp(min=1e-8) / 127.0)
    x8 = torch.clamp(torch.round(x2 / asc[:, None]), -127, 127).to(torch.int8)
    ref = ((torch.matmul(x8.to(torch.int64), W8.t().to(torch.int64)).to(torch.int32)
            .float() * ws[None, :]) * asc[:, None]).to(x.dtype)
    res["w8a8_matches_golden"] = bool(torch.equal(y_full, ref))
    # sub-batch invariance: rows 0..3 alone must equal the same rows in the batch
    y_sub = w8a8_matmul(x[:4], W8, ws)
    res["w8a8_row_invariant"] = bool(torch.equal(y_sub, y_full[:4]))

    # 2. rmsnorm: scale invariance of the row exponent + batch invariance
    w = torch.rand(K) * 2
    y1 = rmsnorm_rule(x, w)
    y2 = rmsnorm_rule(x[:5], w)
    res["rmsnorm_row_invariant"] = bool(torch.equal(y1[:5], y2))
    ref_f = (x.float() / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True))) * w
    res["rmsnorm_relerr_vs_float"] = float(
        ((y1.float() - ref_f).abs().max() / ref_f.abs().max()).item())

    # 3. rope: shape + determinism
    hs, rd, H = 64, 64, 4
    cache = (torch.round(torch.randn(128, rd) * 16384).clamp(-16384, 16384) / 16384)
    pos = torch.arange(R) % 128
    q = torch.randn(R, H * hs)
    k = torch.randn(R, 2 * hs)
    qo, ko = rope_contract(pos, q, k, cache, hs, rd, True)
    qo2, _ = rope_contract(pos[:5], q[:5], k[:5], cache, hs, rd, True)
    res["rope_row_invariant"] = bool(torch.equal(qo[:5], qo2))
    res["rope_shapes"] = [list(qo.shape), list(ko.shape)]

    # 4. silu spec vectors (SPEC 6.3.1 / 6.0.4 behaviour)
    try:
        xs = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.5, 3.0, -3.0, 10.0, -30.0, 30.0,
                           31.0, -100.0, 0.1, 7.25], dtype=torch.float32)
        ys = silu_contract(xs)
        ref_s = xs * torch.sigmoid(xs)
        res["silu_maxabs_vs_torch"] = float((ys - ref_s).abs().max().item())
        res["silu_negzero_is_poszero"] = (not bool(torch.signbit(ys[1]).item())
                                          and float(ys[1]) == 0.0)
    except Exception as e:
        res["silu_error"] = repr(e)

    # 5. CPU vs CUDA bit-equality (the F2 method)
    if torch.cuda.is_available():
        d = "cuda"
        # regression guard for the scalar-division hazard (see `_const`)
        big = torch.rand(200000, dtype=torch.float32) * 0.2 + 1e-3
        res["scale_div_cpu_eq_cuda"] = bool(torch.equal(
            big / _const(127.0, big), (big.to(d) / _const(127.0, big.to(d))).cpu()))
        res["scalar_div_hazard_present"] = not bool(torch.equal(
            (big.to(d) / 127.0).cpu(), big / 127.0))
        xr = torch.randn(4096, 512)
        a_cpu = quantize_rows_int8(xr)[1]
        a_gpu = quantize_rows_int8(xr.to(d))[1].cpu()
        res["act_scale_cpu_eq_cuda"] = bool(torch.equal(a_cpu, a_gpu))
        y_g = w8a8_matmul(x.to(d), W8.to(d), ws.to(d)).cpu()
        res["w8a8_cpu_eq_cuda"] = bool(torch.equal(y_g, y_full))
        n_g = rmsnorm_rule(x.to(d), w.to(d)).cpu()
        res["rmsnorm_cpu_eq_cuda"] = bool(torch.equal(n_g, y1))
        q_g, _ = rope_contract(pos.to(d), q.to(d), k.to(d), cache.to(d), hs, rd, True)
        res["rope_cpu_eq_cuda"] = bool(torch.equal(q_g.cpu(), qo))
        try:
            s_g = silu_contract(xs.to(d)).cpu()
            res["silu_cpu_eq_cuda"] = bool(torch.equal(s_g, ys))
        except Exception as e:
            res["silu_cuda_error"] = repr(e)
    if verbose:
        print(json.dumps(res, indent=1, sort_keys=True), flush=True)
    return res


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        resolve_all()
        print(json.dumps(report(), indent=1, sort_keys=True, default=str))
