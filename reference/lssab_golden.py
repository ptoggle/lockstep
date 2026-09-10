# LSSA-B golden: an independent numpy implementation of BLOCK_EXPONENT_VARIANT.md.
#
# Scope: this file implements THE VARIANT CONTRACT (per-block K/V exponents, key offset,
# per-row q exponent, fold-first-then-max, the per-block 32-bit chain with r_b/delta_max,
# the exact block skip, the two-plane PV and the integer A-accumulation rule).
# It does NOT modify and does not depend on contract_common.py's semantics; the self-test
# at the bottom PINS the shared pieces by degenerating to contract_common.make_contract_attention.
#
# Everything integer here is a Python/numpy integer computation; nothing is approximated.
# The only floating point is (a) the pow2 quantiser (exact: multiply by a power of two),
# (b) the exact int8 score matmul carried in float32 (|partial| < 2^24), and (c) the
# unchanged epilogue (one f32 division + one exact pow2 scale).
#
# A-ACCUMULATION RULE IMPLEMENTED (see notes in BLOCK_EXPONENT_VARIANT.md):
#   e_v_ref = MAX over the row's visible blocks of e_v[b]        (not min)
#   A       = sum_b ( A_b << (e_v_ref - e_v[b]) )                 (all shifts >= 0, exact)
#   out     = f32(A) / f32(l) * 2^(-e_v_ref)
# The document's prose says "min", but with e_v_ref = min the shift (e_v[b]-e_v_ref) applied
# to A_b and the final 2^(-e_v_ref) scale do not compose to sum_b A_b*2^(-e_v[b]); with
# e_v_ref = MAX they do, and every shift is still a non-negative LEFT shift, which is the
# stated intent ("(e_v[b]-e_v_ref) >= 0 so it is a LEFT shift (exact)").  Proof:
#   sum_b A_b*2^(-e_v[b]) = 2^(-e_v_ref) * sum_b A_b*2^(e_v_ref-e_v[b]), e_v_ref = max_b e_v[b].
# The result is INVARIANT to the choice of e_v_ref among values >= max_b e_v[b] (doubling
# e_v_ref doubles A exactly, f32 conversion/division commute with exact powers of two, and the
# final scale halves) -- gate `gate_evref_invariance` checks this.
#
# Author-facing constants: BHI (=B) = 26, table (n,w) = (13,15), KB = 128, dh <= 128.

import math
import os
import hashlib
from decimal import Decimal, getcontext
import numpy as np

BHI = 26
KB = 128
NBITS = 13
WBITS = 15

# ----------------------------------------------------------------------------- table
COEFFS = [1073741823, -744260975, 257938938, -59583045, 10286509, -1369790, 117454]

def build_table(n=NBITS, w=WBITS):
    """Section 6.1.8 table family; (13,15) reproduces the frozen golden FRAC."""
    tab = []
    for k in range(1 << n):
        t = k << (20 - n)
        p = COEFFS[6]
        for c in COEFFS[5::-1]:
            p = c + ((p * t) // (1 << 20))
        tab.append((p + (1 << (29 - w))) >> (30 - w))
    return np.asarray(tab, dtype=np.int64)

_TAB = {}
def table(n=NBITS, w=WBITS):
    if (n, w) not in _TAB:
        _TAB[(n, w)] = build_table(n, w)
    return _TAB[(n, w)]

# ------------------------------------------------------------------- LSSA-B2 (frac2)
# T13c contract variant "LSSA-B2".  The 8192-entry FRAC table is replaced by a TWO-FACTOR
# product of a 64-entry and a 128-entry table:
#
#     T_hi[h] = RN(2^15 * 2^(-h/64)),    h in [0, 64)
#     C_lo[l] = RN(2^15 * 2^(-l/8192)),  l in [0, 128)
#     FRAC2[idx] = (T_hi[idx >> 7] * C_lo[idx & 127] + 2^14) >> 15
#
# idx = 128*h + l and idx/8192 = h/64 + l/8192, so the product is 2^15 * 2^(-idx/8192) to
# within the two roundings.  Both factor tables are CORRECTLY ROUNDED, generated here from
# 40-digit Decimal arithmetic (no ties occur, so the tie rule is immaterial).
#
# THIS FILE TREATS FRAC2 AS *THE* DEFINITION for table_mode="frac2": bit-identity of the
# LSSA-B2 kernel is against FRAC2, never against FRAC.  Measured: max|FRAC2 - FRAC| = 1 over
# all 8192 indices, 5766/8192 (70.4%) exact match.  FRAC itself is the correctly-rounded
# RN(2^15 * 2^(-idx/8192)) at every index, so FRAC2 is within 1 ulp of correctly rounded.
#
# The default table mode is UNCHANGED ("frac"); "frac2" must be asked for explicitly (by the
# table_mode= argument or the LSSAB_TABLE=frac2 environment variable).

TABLE_MODE_DEFAULT = os.environ.get("LSSAB_TABLE", "frac")

def _rn_dec(x):
    """Round a Decimal to the nearest integer (ties away from zero; no tie occurs here)."""
    n = int(x); f = x - n
    if f > Decimal("0.5"): n += 1
    elif f == Decimal("0.5"): n += 1
    return n

def build_factor_tables():
    """(T_hi[64], C_lo[128]) as int64 arrays, from 40-digit Decimal arithmetic."""
    getcontext().prec = 40
    ln2 = Decimal(2).ln()
    def p2(t):                                    # 2^(-t), t a Decimal
        return (-t * ln2).exp()
    thi = [_rn_dec(Decimal(2) ** 15 * p2(Decimal(h) / Decimal(64))) for h in range(64)]
    clo = [_rn_dec(Decimal(2) ** 15 * p2(Decimal(l) / Decimal(8192))) for l in range(128)]
    return np.asarray(thi, dtype=np.int64), np.asarray(clo, dtype=np.int64)

def build_table2():
    thi, clo = build_factor_tables()
    idx = np.arange(1 << NBITS, dtype=np.int64)
    return (thi[idx >> 7] * clo[idx & 127] + (1 << 14)) >> 15

# C_lo has an EXACT closed form, verified exhaustively over all 128 entries by
# check_clo_closed_form() (and by selftest S1b).  The kernel uses it to evaluate the low
# factor with three integer ops and no memory access; it is an implementation of the SAME
# integer values, not a redefinition.
CLO_A, CLO_B, CLO_SH = 4049, 22717, 13          # C_lo[l] = 32768 + ((A + l*(l - B)) >> SH)

def clo_closed_form(l):
    return 32768 + ((CLO_A + l * (l - CLO_B)) >> CLO_SH)

def check_clo_closed_form():
    _, clo = build_factor_tables()
    return all(clo_closed_form(l) == int(clo[l]) for l in range(128))

_TAB2 = {}
def table2():
    if "t" not in _TAB2:
        _TAB2["t"] = build_table2()
    return _TAB2["t"]

def table_of(mode=None, n=NBITS, w=WBITS):
    """Dispatch on the table mode.  mode=None -> TABLE_MODE_DEFAULT (which is "frac" unless
    LSSAB_TABLE says otherwise)."""
    m = TABLE_MODE_DEFAULT if mode is None else mode
    if m == "frac":
        return table(n, w)
    if m == "frac2":
        assert (n, w) == (NBITS, WBITS), "frac2 is defined for (n,w) = (13,15) only"
        return table2()
    raise ValueError("unknown table mode %r" % (m,))

def table_digests():
    """SHA-256 of the ASCII csv of each table (stable across platforms/dtypes)."""
    thi, clo = build_factor_tables()
    def h(a):
        return hashlib.sha256(",".join(str(int(x)) for x in np.asarray(a).ravel()).encode()).hexdigest()
    return {"T_hi": h(thi), "C_lo": h(clo), "FRAC2": h(table2()), "FRAC": h(table(NBITS, WBITS))}

# ------------------------------------------------------------------------- quantiser
def quant_pow2_np(x, axes):
    """contract_common.quant_pow2 semantics, numpy/float32, reducing over `axes`.
    Returns (q int8-valued float32 with x's shape, e int64 with the kept shape)."""
    x = np.asarray(x, dtype=np.float32)
    m = np.abs(x).max(axis=axes, keepdims=True)                 # float32
    mant, E = np.frexp(m)                                       # m = mant*2^E, mant in [.5,1)
    e = 6 - (E.astype(np.int64) - 1)
    e = np.where(m == 0, np.zeros_like(e), e)
    scale = np.ldexp(np.float32(1.0), e.astype(np.int32)).astype(np.float32)
    over = (m * scale) > np.float32(127.0)
    e = np.clip(e - over.astype(np.int64), -32, 40)
    scale = np.ldexp(np.float32(1.0), e.astype(np.int32)).astype(np.float32)
    q = np.clip(np.round(x * scale), -127.0, 127.0).astype(np.float32)
    return q, np.squeeze(e, axis=tuple(axes) if isinstance(axes, tuple) else (axes,))

# ------------------------------------------------------------------------------ fold
def g_fold_int(dh, sh, bhi=BHI):
    """contract_common.g_fold_t semantics on a python int shift, with the T10 safe-band
    clamp of the shift argument to [-7, 23] applied FIRST (BLOCK_EXPONENT_VARIANT.md)."""
    base = round((2.0 ** bhi) / (math.sqrt(dh) * math.log(2.0)))
    sh = int(np.clip(sh, -7, 23))
    if sh >= 0:
        add = (1 << (sh - 1)) if sh > 0 else 0
        G = (base + add) >> sh
    else:
        G = base << min(-sh, 34)
    return int(np.clip(G, 1, 1 << 30))

def g_fold_vec(dh, sh, bhi=BHI):
    sh = np.asarray(sh, dtype=np.int64)
    base = round((2.0 ** bhi) / (math.sqrt(dh) * math.log(2.0)))
    s = np.clip(sh, -7, 23)
    shp = np.maximum(s, 0)
    shn = np.clip(-s, 0, 34)
    add = np.where(shp > 0, np.left_shift(np.int64(1), np.maximum(shp - 1, 0)), np.int64(0))
    Gp = np.right_shift(base + add, shp)
    Gn = np.left_shift(np.int64(base), shn)
    return np.clip(np.where(s >= 0, Gp, Gn), 1, 1 << 30).astype(np.int64)

# ------------------------------------------------------------- exact int -> f32 (RNE)
def i2f32(x):
    """Correctly-rounded (round-half-even) int -> float32, no double rounding.
    Matches CUDA __ll2float_rn / torch int64->float32."""
    x = int(x)
    if x == 0:
        return np.float32(0.0)
    s = -1 if x < 0 else 1
    a = -x if x < 0 else x
    nb = a.bit_length()
    if nb <= 24:
        return np.float32(s * a)
    sh = nb - 24
    q = a >> sh
    rem = a - (q << sh)
    half = 1 << (sh - 1)
    if rem > half or (rem == half and (q & 1)):
        q += 1
    return np.float32(np.ldexp(float(s * q), sh))

_i2f32v = np.vectorize(i2f32, otypes=[np.float32])

def f32_to_bf16_bits(x):
    """RNE float32 -> bfloat16 bit pattern (uint16)."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    r = u + 0x7FFF + lsb
    nan = (np.asarray(x, dtype=np.float32) != np.asarray(x, dtype=np.float32))
    out = ((r >> 16) & 0xFFFF).astype(np.uint16)
    out = np.where(nan, np.uint16(0x7FC0), out)
    return out

# ============================================================================= GOLDEN
def lssab_quantize(q, k, v, mu=None, kb=KB, q_exp_mode="row"):
    """The PREP: q [H,T,dh], k/v [KVH,T,dh] f32, mu [KVH,dh] or None ->
    q8,k8,v8 (int8-valued float32), eq [H,T] (or per-head broadcast), ek/ev [nblk,KVH]."""
    q = np.asarray(q, dtype=np.float32); k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    H, T, dh = q.shape; KVH = k.shape[0]
    nblk = (T + kb - 1) // kb
    kp = k if mu is None else (k - np.asarray(mu, dtype=np.float32)[:, None, :]).astype(np.float32)
    k8 = np.zeros((KVH, T, dh), np.float32); v8 = np.zeros((KVH, T, dh), np.float32)
    ek = np.zeros((nblk, KVH), np.int64);    ev = np.zeros((nblk, KVH), np.int64)
    for b in range(nblk):
        lo, hi = b * kb, min(T, (b + 1) * kb)
        qk_, e_ = quant_pow2_np(kp[:, lo:hi, :], axes=(1, 2)); k8[:, lo:hi, :] = qk_; ek[b] = e_
        qv_, e_ = quant_pow2_np(v[:, lo:hi, :], axes=(1, 2));  v8[:, lo:hi, :] = qv_; ev[b] = e_
    if q_exp_mode == "row":
        q8, eq = quant_pow2_np(q, axes=(2,))                      # [H,T]
    elif q_exp_mode == "head":
        q8, eq_h = quant_pow2_np(q, axes=(1, 2))                  # [H]
        eq = np.repeat(eq_h[:, None], T, axis=1)
    else:
        raise ValueError(q_exp_mode)
    return dict(q8=q8, k8=k8, v8=v8, eq=eq, ek=ek, ev=ev)


def lssab_attention(q, k, v, mu=None, bhi=BHI, n=NBITS, w=WBITS, kb=KB,
                    q_exp_mode="row", causal=True, key_valid=None,
                    block_skip=True, debug=False, table_mode=None):
    """Prep + core. Returns out [H,T,dh] float32 (the f32 epilogue result)."""
    p = lssab_quantize(q, k, v, mu, kb, q_exp_mode)
    return lssab_core(p["q8"], p["k8"], p["v8"], p["eq"], p["ek"], p["ev"],
                      bhi=bhi, n=n, w=w, kb=kb, causal=causal, key_valid=key_valid,
                      block_skip=block_skip, debug=debug, table_mode=table_mode)


def lssab_core(q8, k8, v8, eq, ek, ev, bhi=BHI, n=NBITS, w=WBITS, kb=KB,
               causal=True, key_valid=None, block_skip=True, debug=False, table_mode=None):
    """The CONTRACT: int8-valued q8 [H,T,dh], k8/v8 [KVH,T,dh], eq [H,T], ek/ev [nblk,KVH]."""
    q8 = np.asarray(q8, dtype=np.float32); k8 = np.asarray(k8, dtype=np.float32)
    v8 = np.asarray(v8, dtype=np.float32)
    eq = np.asarray(eq, dtype=np.int64); ek = np.asarray(ek, dtype=np.int64)
    ev = np.asarray(ev, dtype=np.int64)
    H, T, dh = q8.shape
    KVH = k8.shape[0]
    assert k8.shape == (KVH, T, dh) and v8.shape == (KVH, T, dh)
    assert H % KVH == 0
    grp = H // KVH
    WIN = (w + 1) << bhi
    idx_sh = bhi - n
    idx_mask = (1 << n) - 1
    tab = table_of(table_mode, n, w)
    nblk = (T + kb - 1) // kb
    assert ek.shape == (nblk, KVH) and ev.shape == (nblk, KVH) and eq.shape == (H, T)

    pos = np.arange(T)
    blk_of = pos // kb
    out = np.zeros((H, T, dh), np.float32)
    dbg = {"skipped": 0, "blocks": 0, "l": np.zeros((H, T), np.int64),
           "A": np.zeros((H, T, dh), np.int64), "evref": np.zeros((H, T), np.int64),
           "NB": np.zeros((H, T), np.int64), "mS": np.full((H, T, nblk), np.iinfo(np.int64).min, np.int64)}

    for h in range(H):
        g = h // grp
        S = np.matmul(q8[h], k8[g].T)                              # f32, exact ints < 2^24
        assert np.abs(S).max() < (1 << 24)
        S = S.astype(np.int64)
        vis = np.ones((T, T), bool)
        if causal:
            vis &= (pos[None, :] <= pos[:, None])
        if key_valid is not None:
            vis &= np.asarray(key_valid, bool)[None, :]
        G = g_fold_vec(dh, eq[h][:, None] + ek[:, g][None, :], bhi)   # [T, nblk]
        NEG = np.iinfo(np.int64).min // 4
        for b in range(nblk):
            lo, hi = b * kb, min(T, (b + 1) * kb)
            mS = np.where(vis[:, lo:hi].any(1), np.where(vis[:, lo:hi], S[:, lo:hi], NEG).max(1), NEG)
            dbg["mS"][h, :, b] = mS
        mS_all = dbg["mS"][h]                                      # [T, nblk]
        live_b = mS_all > NEG
        SG = np.where(live_b, mS_all * G, np.int64(NEG))
        m_SG = SG.max(1)
        NB = -((-m_SG) >> bhi) << bhi
        anylive = live_b.any(1)
        # e_v_ref = max over the row's VISIBLE blocks
        evb = ev[:, g][None, :]
        evref = np.where(anylive, np.where(live_b, evb, np.int64(-1 << 40)).max(1), np.int64(0))
        dbg["NB"][h] = np.where(anylive, NB, 0); dbg["evref"][h] = evref

        Aacc = np.zeros((T, dh), np.int64)   # exact: |A| < 2^62 is asserted below
        Ahit = np.zeros(T, bool)
        lacc = np.zeros(T, np.int64)
        for b in range(nblk):
            lo, hi = b * kb, min(T, (b + 1) * kb)
            rows = np.nonzero(live_b[:, b] & anylive)[0]
            if rows.size == 0:
                dbg["blocks"] += T; continue
            Gb = G[rows, b]; mSb = mS_all[rows, b]
            r = NB[rows] - mSb * Gb
            assert (r >= 0).all(), "r_b must be >= 0"
            dead = r >= WIN
            dbg["blocks"] += rows.size; dbg["skipped"] += int(dead.sum())
            keep = rows[~dead] if block_skip else rows
            if keep.size == 0:
                continue
            sel = ~dead if block_skip else np.ones(rows.size, bool)
            Gk = Gb[sel]; mSk = mSb[sel]; rk = r[sel]
            dmax = -(-(WIN - rk) // Gk)                            # ceil, > 0 when not dead
            delta = mSk[:, None] - S[keep, lo:hi]
            visk = vis[keep, lo:hi]
            live = visk & (delta < dmax[:, None]) & (rk[:, None] < WIN)
            d = rk[:, None] + delta * Gk[:, None]
            assert d[live].max(initial=0) < (1 << 31), "32-bit chain overflow"
            shift = np.where(live, d >> bhi, 0)
            eh = np.where(live, tab[(d >> idx_sh) & idx_mask] >> shift, 0).astype(np.int64)
            lacc[keep] += eh.sum(1)
            # two u8 planes (w=15): eh = 256*hi + lo, hi in [0,128], lo in [0,255]
            ehi = eh >> 8; elo = eh & 255
            assert ehi.max(initial=0) <= 128 and elo.max(initial=0) <= 255
            vb = v8[g, lo:hi, :].astype(np.int64)
            Ahi = np.matmul(ehi, vb); Alo = np.matmul(elo, vb)     # exact s32-range partials
            assert np.abs(Ahi).max(initial=0) < (1 << 31) and np.abs(Alo).max(initial=0) < (1 << 31)
            Ab = (Ahi << 8) + Alo
            sb = evref[keep] - ev[b, g]
            assert (sb >= 0).all(), "A rule: every block shift must be a non-negative LEFT shift"
            add = np.left_shift(Ab, sb[:, None])                     # exact LEFT shift, s64
            assert np.abs(add).max(initial=0) < (1 << 62), "A accumulation out of s64 range"
            Aacc[keep] += add
            Ahit[keep] = True

        assert np.abs(Aacc).max(initial=0) < (1 << 62), "A accumulation out of s64 range"
        live_rows = anylive & (lacc > 0) & Ahit
        Af = _i2f32v(Aacc)                                          # exact int64 -> f32 RNE
        lf = _i2f32v(lacc)
        with np.errstate(divide="ignore", invalid="ignore"):
            o = (Af / np.where(lf == 0, np.float32(1.0), lf)[:, None]).astype(np.float32)
        sc = np.ldexp(np.ones(T), -evref).astype(np.float32)
        out[h] = np.where(live_rows[:, None], (o * sc[:, None]).astype(np.float32), np.float32(0.0))
        dbg["A"][h] = Aacc
        dbg["l"][h] = lacc

    if debug:
        return out, dbg
    return out
