#!/usr/bin/env python3
"""
LSSA-B8 GOLDEN — an independent numpy/python-int implementation of the "LSSA-B8" addendum
of BLOCK_EXPONENT_VARIANT.md (T16a, 2026-09-02).

WHAT LSSA-B8 IS (restated so this file is self-contained; the addendum is normative)
  * PREP is unchanged from LSSA-B: key offset mu, per-(block, kv-head) pow2 exponents
    e_k[b], e_v[b] over KB consecutive ABSOLUTE positions, per-(row, head) e_q, int8
    q8/k8/v8, G[i,b] = g_fold(dh, clamp(e_q[i] + e_k[b], -7, 23)) in [1, 2^30].
  * The weight is 8-bit and referenced to the (row, BLOCK) local grid point, not the row:
        m_S[b] = max over VISIBLE j in b of S[i,j]          (exact int32 tile max)
        NB_b   = least multiple of 2^B >= m_S[b]*G[b]
        d[j]   = NB_b - S[j]*G[b]     (>= 0)
        eh[j]  = T2[d[j] >> (B - n)]  with T2[k] = 0 for k >= W_LOC * 2^n
    T2[k] = RN_half_up(255 * 2^(-k/2^n)), generated from 40-digit Decimal, digest-pinned.
  * Per block the partials are EXACT integers:
        A_b = sum_{j in b} eh[j] * v8[j]   (|A_b| <= 255*127*128 = 4,145,280 < 2^22)
        l_b = sum_{j in b} eh[j]           (<= 255*128 = 32,640)
  * The cross-block scale is a per-(row, block) power of two applied when the block partial
    is FOLDED into a running fp32 output, in a COMMITTED ASCENDING block order, two-level
    (blocks inside a SEG-block segment, then segments):
        D = (NB_b - NB_run) >> B
        if D > 0:  O = rescale(O, D); l = rescale(l, D); NB_run = NB_b; s = 0
        else:      s = -D; if s > S_WIN: the block contributes exactly zero -> SKIP
        O = fl_RNE(O + f32(A_b) * 2^-(s + e_v[b]));  l = fl_RNE(l + f32(l_b) * 2^-s)
    rescale(x, D) is an EXPONENT-FIELD SUBTRACTION with flush when the biased exponent
    would fall below 1 (NOT a float multiply: FA3 is built with -ftz=true and the IEEE
    product (2^24-1)*2^-150 rounds UP to 2^-126 while a flush-before-rounding implementation
    gives 0).  This makes the rescale a pure integer operation, identical on every vendor.
  * Epilogue: out = f32_div_RN(O, l); l == 0 -> 0.  There is NO e_v_ref and no plane split:
    the V exponent rides in the term scale.

WHAT IS DIFFERENT FROM lssab_golden.py (LSSA-B): the weight width (8 vs 16 bits), the weight
reference (block grid point vs row grid point), the accumulation (a committed-order fp32 fold
vs an order-free integer sum with e_v_ref), and the loss of CROSS-BLOCK schedule invariance
(block order, SEG, S_WIN, W_LOC, n and the T2 digest become contract constants).  Batch
invariance, chunk invariance (block-aligned rule), K-tiling inside a block and M-tiling are
unchanged.

Nothing here is approximate.  The only floating point is (a) the pow2 quantiser (exact),
(b) the exact int8 score matmul carried in float32 (|partial| < 2^24), and (c) the fold and
epilogue, which are a fixed sequence of IEEE-754 binary32 RNE operations on operands that are
exactly representable.
"""
import os, sys, math, hashlib
from decimal import Decimal, getcontext
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lssab_golden as LG            # PREP, g_fold, i2f32, quantiser — reused, never modified

# ------------------------------------------------------------------ contract constants
BHI = B = 26
KB = 128              # contract key-block size (absolute positions)
SEG = 32              # fold segment, in blocks (SEG*KB = 4,096 keys)
# ---- LSSA-B8-LITE (T16m, 2026-09-04): a CANDIDATE prompt-row variant with two of the
# T16d-priced rules changed, selected by environment so that no existing number can move:
#   LSSAB8_SEG=0        -> SINGLE-LEVEL fold: every block folds into one accumulator in the
#                          committed ascending order (SEG = "all blocks"); no segment level.
#   LSSAB8_RESCALE=mul  -> rescale(x, D) = fl32(x * 2^-D) for D < 126, else +0.0: ONE IEEE
#                          RNE multiply (2^-D is exact), gradual underflow honoured; the
#                          kernel is built with -ftz=false so it computes the same function.
# Both are still exactly specified functions; they are a different contract version.
_SEG_ENV = os.environ.get("LSSAB8_SEG", "")
if _SEG_ENV not in ("", "32"):
    SEG = (1 << 30) if int(_SEG_ENV) <= 0 else int(_SEG_ENV)
RESCALE_MODE = os.environ.get("LSSAB8_RESCALE", "field")   # field (filed) | mul (lite)
S_WIN = 32            # exact-skip window, in doublings, inside a segment / across segments
W_LOC = 9             # block-local weight window, in doublings  (9 == the full u8 support)
NIDX = 7              # index bits per doubling
TOP = 255             # top weight
KTAB_DOUB = 17        # table length in doublings (17*2^n covers the clamped chain index)
EV_LO, EV_HI = -32, 40

getcontext().prec = 40
_LN2 = Decimal(2).ln()


def _rn_half_up(x):
    """Round a Decimal to the nearest integer, ties AWAY FROM ZERO (all T2 arguments are
    positive, so this is 'half-up').  Exactly one exact tie exists in the family — k = 2^n,
    value 127.5 — and it rounds to 128 under half-up AND under round-half-even; the rule is
    pinned anyway so a golden and a kernel cannot disagree by convention."""
    return int((x + Decimal("0.5")).to_integral_value(rounding="ROUND_FLOOR"))


_T2 = {}

def t2_table(n=NIDX, w_loc=W_LOC, top=TOP, ktab_doub=KTAB_DOUB):
    """T2[k] = RN_half_up(top * 2^(-k/2^n)) for k < w_loc*2^n, 0 beyond, length ktab_doub*2^n."""
    key = (n, w_loc, top, ktab_doub)
    if key in _T2:
        return _T2[key]
    N = 1 << n
    live = w_loc * N
    L = ktab_doub * N
    assert live <= L
    t = np.zeros(L, dtype=np.int64)
    for k in range(live):
        t[k] = _rn_half_up(Decimal(top) * (-Decimal(k) / Decimal(N) * _LN2).exp())
    _T2[key] = t
    return t


def t2_table_trunc(n=NIDX, w_loc=W_LOC, top=TOP, ktab_doub=KTAB_DOUB):
    """The L8-TR arm: truncation instead of round-to-nearest (a control; RN is mandatory)."""
    N = 1 << n
    live = w_loc * N
    t = np.zeros(ktab_doub * N, dtype=np.int64)
    for k in range(live):
        t[k] = int((Decimal(top) * (-Decimal(k) / Decimal(N) * _LN2).exp()).to_integral_value(
            rounding="ROUND_FLOOR"))
    return t


def digest(a):
    """sha256 of the ASCII csv of the table (stable across platforms and dtypes)."""
    return hashlib.sha256(",".join(str(int(x)) for x in np.asarray(a).ravel()).encode()).hexdigest()


def t2_digests():
    d = {}
    for n in (7, 8):
        for wl in (8, 9):
            d["T2_n%d_W%d" % (n, wl)] = digest(t2_table(n, wl))
    d["T2TR_n%d_W%d" % (NIDX, W_LOC)] = digest(t2_table_trunc(NIDX, W_LOC))
    return d


# ------------------------------------------------------------------- exact fp32 helpers
F32 = np.float32
_EXPMASK = np.uint32(0xFF << 23)
_NOTEXP = np.uint32(~(0xFF << 23) & 0xFFFFFFFF)


def rescale_mul(x, D):
    """The LITE rescale: fl32(x * 2^-D), one IEEE RNE multiply with an exact 2^-D, +0.0 for
    D >= 126 (the kernel's `(D >= 126) ? 0.f : x * 2^-D`).  Results below the normal range
    round to a subnormal (gradual underflow); the kernel is built with -ftz=false."""
    a = np.asarray(x, dtype=np.float32)
    assert D >= 0
    # D >= 126: the kernel multiplies by 0.0f, so the SIGN of zero follows x (-0.0 for a
    # negative input); a later `+ term` absorbs it, but O_row / l_row bit patterns carry it.
    sc = np.float32(0.0) if D >= 126 else np.float32(2.0 ** -D)
    res = (a * sc).astype(np.float32)
    return res if a.ndim else np.float32(res)


def rescale(x, D):
    """The CONTRACT rescale: subtract D from the biased exponent field; flush to +0 when the
    result would have a biased exponent < 1 (i.e. would be subnormal or underflow).  Exact for
    every normal input, and never rounds — so no vendor FTZ/denormal mode can change it.
    x: float32 array (or scalar), D: non-negative python int."""
    if RESCALE_MODE == "mul":
        return rescale_mul(x, D)
    a = np.asarray(x, dtype=np.float32)
    assert D >= 0
    u = a.view(np.uint32)
    e = ((u >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int64)
    assert not (e == 255).any(), "inf/nan in the fold"
    ne = e - D
    keep = (e >= 1) & (ne >= 1)
    out = np.where(keep, (u & _NOTEXP) | (ne.astype(np.uint64) << np.uint64(23)).astype(np.uint32),
                   u & np.uint32(0x80000000))          # signed zero preserved; value is +-0
    out = out.view(np.float32) * np.float32(1.0)       # normalise -0 handling below
    res = np.where(keep, out, np.float32(0.0)).astype(np.float32)
    return res if a.ndim else np.float32(res)


def f32_add(a, b):
    """IEEE-754 binary32 addition with round-to-nearest-even (numpy float32 semantics)."""
    return np.add(np.asarray(a, np.float32), np.asarray(b, np.float32), dtype=np.float32)


def f32_mul(a, b):
    return np.multiply(np.asarray(a, np.float32), np.asarray(b, np.float32), dtype=np.float32)


def f32_div(a, b):
    """One correctly-rounded binary32 division (the kernel MUST use __fdiv_rn, never
    div.approx — the T12a --use_fast_math trap)."""
    return np.divide(np.asarray(a, np.float32), np.asarray(b, np.float32), dtype=np.float32)


def pow2_f32(e):
    """2^e as an exact float32; requires -126 <= e <= 127."""
    assert -126 <= e <= 127, "term exponent %d outside the normal binary32 range" % e
    return np.float32(np.ldexp(1.0, e))


i2f32 = LG.i2f32                     # correctly-rounded int -> float32 (RNE)
f32_to_bf16_bits = LG.f32_to_bf16_bits
lssab_quantize = LG.lssab_quantize   # PREP is unchanged
g_fold_int = LG.g_fold_int
g_fold_vec = LG.g_fold_vec


# ================================================================== THE GOLDEN (core)
def lssab8_core(q8, k8, v8, eq, ek, ev, kb=KB, seg=SEG, s_win=S_WIN, w_loc=W_LOC,
                n=NIDX, bhi=BHI, causal=True, key_valid=None, table=None, weight="rn",
                skip=True, debug=False, rows=None, heads=None):
    """int8-valued q8 [H,T,dh], k8/v8 [KVH,T,dh], eq [H,T], ek/ev [nblk,KVH] -> out [H,T,dh] f32.

    weight: "rn" (contract), "tr" (truncation control), "l16" (16-bit block-local control).
    skip:   apply the exact s > S_WIN skip (must be bit-identical to skip=False).
    rows / heads: restrict which query rows / heads are evaluated (a pure work restriction --
            every row is an independent recurrence, so the rows that ARE computed are
            bit-identical to the full evaluation).  Uncomputed entries stay 0.
    """
    q8 = np.asarray(q8, np.float32); k8 = np.asarray(k8, np.float32); v8 = np.asarray(v8, np.float32)
    eq = np.asarray(eq, np.int64); ek = np.asarray(ek, np.int64); ev = np.asarray(ev, np.int64)
    H, T, dh = q8.shape
    KVH = k8.shape[0]
    assert H % KVH == 0 and k8.shape == (KVH, T, dh) and v8.shape == (KVH, T, dh)
    grp = H // KVH
    nblk = (T + kb - 1) // kb
    assert ek.shape == (nblk, KVH) and ev.shape == (nblk, KVH) and eq.shape == (H, T)
    assert ((ev >= EV_LO) & (ev <= EV_HI)).all(), "e_v outside the contract clamp [-32, 40]"
    if weight == "l16":
        tab = LG.table(13, 15); nn = 13; win_k = 16 << 13
    else:
        tab = table if table is not None else (
            t2_table(n, w_loc) if weight == "rn" else t2_table_trunc(n, w_loc))
        nn = n; win_k = w_loc << n
    idx_sh = bhi - nn

    out = np.zeros((H, T, dh), np.float32)
    dbg = {"skipped": 0, "blocks": 0, "seg_dropped": 0, "segments": 0,
           "l": np.zeros((H, T), np.float32), "O": np.zeros((H, T, dh), np.float32),
           "NB": np.zeros((H, T), np.int64), "maxterm": -10**9, "minterm": 10**9,
           "ftz": 0, "Abmax": 0}
    pos = np.arange(T)
    hlist = range(H) if heads is None else list(heads)
    rlist = range(T) if rows is None else list(rows)
    for h in hlist:
        g = h // grp
        S = np.matmul(q8[h], k8[g].T)
        assert np.abs(S).max(initial=0.0) < (1 << 24)
        S = S.astype(np.int64)
        for i in rlist:
            vis = np.ones(T, bool)
            if causal:
                vis &= (pos <= i)
            if key_valid is not None:
                vis &= np.asarray(key_valid, bool)
            if not vis.any():
                continue
            # ---- row-level state
            O_row = np.zeros(dh, np.float32); l_row = np.float32(0.0)
            NB_row = None
            nseg = (nblk + seg - 1) // seg
            for sgi in range(nseg):
                b0, b1 = sgi * seg, min(nblk, (sgi + 1) * seg)
                O_seg = np.zeros(dh, np.float32); l_seg = np.float32(0.0)
                NB_run = None
                for b in range(b0, b1):
                    lo, hi = b * kb, min(T, (b + 1) * kb)
                    vb = vis[lo:hi]
                    if not vb.any():
                        continue
                    dbg["blocks"] += 1
                    Gb = g_fold_int(dh, int(eq[h, i]) + int(ek[b, g]), bhi)
                    Sb = S[i, lo:hi]
                    mS = int(Sb[vb].max())
                    SGm = mS * Gb
                    NB_b = -((-SGm) >> bhi) << bhi
                    if NB_run is None:
                        NB_run = NB_b
                    D = (NB_b - NB_run) >> bhi
                    if D > 0:
                        O_seg = rescale(O_seg, int(D)); l_seg = rescale(l_seg, int(D))
                        NB_run = NB_b; s = 0
                    else:
                        s = int(-D)
                        if s > s_win:
                            dbg["skipped"] += 1
                            if skip:
                                continue
                    # ---- the chain (exact integers; |d| < 2^52 so int64 is exact)
                    d = NB_b - Sb.astype(np.int64) * np.int64(Gb)
                    win_d = win_k << idx_sh                       # the window in units of d
                    d = np.where(vb, d, np.int64(win_d))          # masked keys -> zero weight
                    assert (d >= 0).all() and d.max(initial=0) < (1 << 52)
                    if weight == "l16":
                        # the filed (13,15) table applied BLOCK-locally: index is the
                        # fractional part, the integer part is a right shift
                        idx = (d >> idx_sh) & ((1 << 13) - 1)
                        sh = np.minimum(d >> bhi, 63)
                        eh = np.where(d < win_d, tab[idx] >> sh, 0)
                    else:
                        kidx = np.minimum(d >> idx_sh, len(tab) - 1)
                        eh = tab[kidx]
                    eh = eh.astype(np.int64)
                    l_b = int(eh.sum())
                    A_b = np.matmul(eh.astype(np.float64), v8[g, lo:hi, :].astype(np.float64))
                    A_b = np.rint(A_b).astype(np.int64)
                    dbg["Abmax"] = max(dbg["Abmax"], int(np.abs(A_b).max(initial=0)))
                    if weight != "l16":
                        assert np.abs(A_b).max(initial=0) <= 4145280, "A_b out of the u8 bound"
                        assert l_b <= 32640
                    if l_b == 0 and not np.any(A_b):
                        continue
                    # ---- the fold (fp32, RNE, committed order)
                    evb = int(ev[b, g])
                    esc = -(s + evb)
                    dbg["maxterm"] = max(dbg["maxterm"], esc); dbg["minterm"] = min(dbg["minterm"], esc)
                    scO = pow2_f32(esc)
                    scl = pow2_f32(-s)
                    Af = np.asarray([i2f32(int(x)) for x in A_b], dtype=np.float32)
                    lf = i2f32(l_b)
                    O_seg = f32_add(O_seg, f32_mul(Af, scO))
                    l_seg = f32_add(l_seg, f32_mul(lf, scl))
                if NB_run is None:
                    continue
                dbg["segments"] += 1
                # ---- fold the finished segment into the row, same three rules
                if NB_row is None:
                    NB_row = NB_run
                D = (NB_run - NB_row) >> bhi
                if D > 0:
                    O_row = rescale(O_row, int(D)); l_row = rescale(l_row, int(D))
                    NB_row = NB_run; s = 0
                else:
                    s = int(-D)
                    if s > s_win:
                        dbg["seg_dropped"] += 1
                        if skip:
                            continue
                O_row = f32_add(O_row, rescale(O_seg, s))
                l_row = f32_add(l_row, rescale(l_seg, s))
            dbg["NB"][h, i] = 0 if NB_row is None else NB_row
            dbg["O"][h, i] = O_row; dbg["l"][h, i] = l_row
            if l_row != np.float32(0.0):
                out[h, i] = f32_div(O_row, l_row)
    if debug:
        return out, dbg
    return out


def lssab8_attention(q, k, v, mu=None, kb=KB, q_exp_mode="row", causal=True, key_valid=None,
                     **kw):
    """PREP + core.  q [H,T,dh], k/v [KVH,T,dh] float32."""
    p = lssab_quantize(q, k, v, mu, kb, q_exp_mode)
    return lssab8_core(p["q8"], p["k8"], p["v8"], p["eq"], p["ek"], p["ev"], kb=kb,
                       causal=causal, key_valid=key_valid, **kw)


# ------------------------------------------------------- the split-KV / decode structure
def lssab8_row_segments(q8, k8, v8, eq, ek, ev, h, i, kb=KB, seg=SEG, s_win=S_WIN,
                        w_loc=W_LOC, n=NIDX, bhi=BHI, causal=True, key_valid=None,
                        weight="rn"):
    """Compute ONE row's per-segment partials INDEPENDENTLY (as a split-KV decode kernel
    would: each segment sees only its own blocks) and then combine them in the committed
    segment order.  Must be bit-identical to lssab8_core's result for that row -- that
    identity is gate S7, and it is the reason the fold is two-level."""
    q8 = np.asarray(q8); k8 = np.asarray(k8); v8 = np.asarray(v8)
    H, T, dh = q8.shape
    nblk = (T + kb - 1) // kb
    nseg = (nblk + seg - 1) // seg
    base = np.ones(T, bool)
    if causal:
        base &= (np.arange(T) <= i)
    if key_valid is not None:
        base &= np.asarray(key_valid, bool)
    parts = []
    for sgi in range(nseg):
        kv = np.zeros(T, bool)
        kv[sgi * seg * kb:min(T, (sgi + 1) * seg * kb)] = True
        kv &= base
        if not kv.any():
            parts.append(None)
            continue
        _, d = lssab8_core(q8, k8, v8, eq, ek, ev, kb=kb, seg=seg, s_win=s_win, w_loc=w_loc,
                           n=n, bhi=bhi, causal=False, key_valid=kv, weight=weight,
                           debug=True, rows=[i], heads=[h])
        parts.append((d["O"][h, i].copy(), np.float32(d["l"][h, i]), int(d["NB"][h, i])))
    O = np.zeros(dh, np.float32); l = np.float32(0.0); NB = None
    for p in parts:
        if p is None:
            continue
        Os, ls, NBs = p
        if NB is None:
            NB = NBs
        D = (NBs - NB) >> bhi
        if D > 0:
            O = rescale(O, int(D)); l = rescale(l, int(D)); NB = NBs; s = 0
        else:
            s = int(-D)
            if s > s_win:
                continue
        O = f32_add(O, rescale(Os, s)); l = f32_add(l, rescale(ls, s))
    out = f32_div(O, l) if l != np.float32(0.0) else np.zeros(dh, np.float32)
    return out, O, l, NB


if __name__ == "__main__":
    print("LSSA-B8 golden — contract constants: B=%d KB=%d SEG=%d S_WIN=%d W_LOC=%d n=%d TOP=%d"
          % (B, KB, SEG, S_WIN, W_LOC, NIDX, TOP))
    t = t2_table()
    print("T2 (n=%d, W_LOC=%d): %d entries, %d live, T2[0]=%d T2[2^n]=%d T2[live-1]=%d T2[live]=%d"
          % (NIDX, W_LOC, len(t), W_LOC << NIDX, t[0], t[1 << NIDX], t[(W_LOC << NIDX) - 1],
             t[W_LOC << NIDX]))
    for k_, v_ in sorted(t2_digests().items()):
        print("  sha256(%s csv) = %s" % (k_, v_))
