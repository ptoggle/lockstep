#!/usr/bin/env python3
"""LSSA-B9 numpy golden (Stage 2 item 2.0b): lssab8_golden's core with the two VALUE options of
the B9 addendum (BLOCK_EXPONENT_VARIANT.md).  lssab8_golden.py is imported, never modified;
with both options off lssab9_core == lssab8_core bit for bit (lssab9_selftest.py, S2).

  B9.1 sink_alone: key 0 (absolute position 0) is quantised on its own natural value grid
       e_v0; keys 1..KB-1 of block 0 on theirs; block 0 folds as TWO terms, key 0's first,
       each with its own exponent and the block's s.
  B9.2 v_planes = 2: the value on a 15-bit signed grid per block (magnitude limit 16383,
       e_v15 = the 7-bit exponent + 7), v_hi = v15 >> 8 (arithmetic, [-64, 63]),
       v_lo = v15 - (v_hi << 8) ([0, 255]); two partials A_hi, A_lo folded as two terms,
       high first, with exponents e_v15 - 8 and e_v15.
"""
import numpy as np
import lssab8_golden as G8
from lssab8_golden import (BHI, KB, SEG, S_WIN, W_LOC, NIDX, EV_LO, EV_HI, t2_table, t2_table_trunc,
                           rescale, f32_add, f32_mul, f32_div, pow2_f32, i2f32, g_fold_int)
import lssab_golden as LG
from lssab_golden import quant_pow2_np


def quant_pow2_bits_np(x, axes, bits):
    """quant_pow2_np generalised to a `bits`-bit magnitude (7 = int8)."""
    x = np.asarray(x, np.float32)
    lim = float((1 << bits) - 1)
    m = np.abs(x).max(axis=axes, keepdims=True)
    mant, E = np.frexp(m)
    e = (bits - 1) - (E.astype(np.int64) - 1)
    e = np.where(m == 0, 0, e)
    over = m * np.ldexp(np.float32(1.0), e) > lim
    e = np.clip(e - over.astype(np.int64), -32, 48)
    sc = np.ldexp(np.float32(1.0), e).astype(np.float32)
    q = np.clip(np.rint(x * sc), -lim, lim).astype(np.float32)
    return q, np.squeeze(e, axis=axes)


def lssab9_quantize(q, k, v, mu=None, kb=KB, q_exp_mode="row", sink_alone=False, v_planes=1, v_keyshift=0):
    """The B8 PREP (lssab_golden.lssab_quantize) with the B9 value options.

    v_keyshift = dmax > 0 (B9.3): key j of block b is quantised on the grid e_v[b] + d_j with
    d_j = clamp(e_j - e_v[b], 0, dmax), e_j the key's own 7-bit exponent; p["d8"] int64 [KVH,T]."""
    p = LG.lssab_quantize(q, k, v, mu, kb, q_exp_mode)
    v = np.asarray(v, np.float32)
    KVH, T, dh = v.shape
    nblk = (T + kb - 1) // kb
    vbits = 14 if v_planes == 2 else 7
    v8 = np.zeros((KVH, T, dh), np.float32); ev = np.zeros((nblk, KVH), np.int64)
    for b in range(nblk):
        lo, hi = b * kb, min(T, (b + 1) * kb)
        qv_, e_ = quant_pow2_bits_np(v[:, lo:hi, :], (1, 2), vbits); v8[:, lo:hi, :] = qv_; ev[b] = e_
    ev0 = None
    if sink_alone and T > 1:
        q0, e0 = quant_pow2_bits_np(v[:, :1, :], (1, 2), vbits)              # key 0 alone: [KVH]
        hi = min(kb, T)
        qr, er = quant_pow2_bits_np(v[:, 1:hi, :], (1, 2), vbits)            # keys 1..kb-1 as one block
        v8[:, :1, :] = q0; v8[:, 1:hi, :] = qr
        ev[0] = er
        ev0 = np.asarray(e0, np.int64)
    d8 = None
    if v_keyshift > 0:
        assert v_planes == 1, "B9.3 is the single-plane rule"
        _, ekey = quant_pow2_bits_np(v, (2,), vbits)                          # [KVH,T,1]: each key's own exponent
        ekey = np.asarray(ekey, np.int64).reshape(KVH, T)
        ev_tok = np.repeat(ev.T, kb, axis=1)[:, :T]                           # [KVH,T] the block exponent per key
        if ev0 is not None:
            ev_tok = ev_tok.copy(); ev_tok[:, 0] = ev0
        d8 = np.clip(ekey - ev_tok, 0, int(v_keyshift)).astype(np.int64)
        sc = np.ldexp(np.float32(1.0), (ev_tok + d8).astype(np.int32)).astype(np.float32)   # exact powers of two
        v8 = np.clip(np.rint(v * sc[..., None]), -127.0, 127.0).astype(np.float32)
    p["v8"] = v8; p["ev"] = ev; p["ev0"] = ev0; p["d8"] = d8
    if v_planes == 2:
        v_hi = np.floor(v8 / 256.0).astype(np.float32)
        v_lo = (v8 - v_hi * 256.0).astype(np.float32)
        assert np.abs(v_hi).max() <= 64 and v_lo.min() >= 0 and v_lo.max() <= 255
        p["v_hi"] = v_hi; p["v_lo"] = v_lo
    else:
        p["v_hi"] = p["v_lo"] = None
    return p


def lssab9_core(q8, k8, v8, eq, ek, ev, kb=KB, seg=SEG, s_win=S_WIN, w_loc=W_LOC,
                n=NIDX, bhi=BHI, causal=True, key_valid=None, table=None, weight="rn",
                skip=True, debug=False, rows=None, heads=None,
                ev0=None, v_planes=1, v_hi=None, v_lo=None, d8=None, ks_round=False):
    """int8-valued q8 [H,T,dh], k8/v8 [KVH,T,dh], eq [H,T], ek/ev [nblk,KVH] -> out [H,T,dh] f32.

    weight: "rn" (contract), "tr" (truncation control), "l16" (16-bit block-local control).
    skip:   apply the exact s > S_WIN skip (must be bit-identical to skip=False).
    rows / heads: restrict which query rows / heads are evaluated (a pure work restriction --
            every row is an independent recurrence, so the rows that ARE computed are
            bit-identical to the full evaluation).  Uncomputed entries stay 0.
    """
    q8 = np.asarray(q8, np.float32); k8 = np.asarray(k8, np.float32); v8 = np.asarray(v8, np.float32)
    if v_planes == 2:
        v_hi = np.asarray(v_hi, np.float32); v_lo = np.asarray(v_lo, np.float32)
    eq = np.asarray(eq, np.int64); ek = np.asarray(ek, np.int64); ev = np.asarray(ev, np.int64)
    H, T, dh = q8.shape
    KVH = k8.shape[0]
    assert H % KVH == 0 and k8.shape == (KVH, T, dh) and v8.shape == (KVH, T, dh)
    grp = H // KVH
    nblk = (T + kb - 1) // kb
    assert ek.shape == (nblk, KVH) and ev.shape == (nblk, KVH) and eq.shape == (H, T)
    assert ((ev >= EV_LO) & (ev <= (EV_HI + (7 if v_planes == 2 else 0)))).all(), "e_v outside the contract clamp"
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
                    l_b = int(eh.sum())                           # l keeps the UNshifted weight
                    if d8 is not None:                            # B9.3: the PV weight is eh >> d_j
                        dk = np.asarray(d8[g, lo:hi], np.int64)
                        if ks_round == "even":                    # round to nearest, ties to even
                            qq = eh >> dk; rem = eh & ((1 << dk) - 1); half = (1 << dk) >> 1
                            eh = qq + ((rem > half) | ((rem == half) & (dk > 0) & ((qq & 1) == 1))).astype(np.int64)
                        elif ks_round:
                            eh = (eh + ((1 << dk) >> 1)) >> dk
                        else:
                            eh = eh >> dk
                    # ---- the block's value partial(s) as (weights, value slice, exponent) groups:
                    # B8 has one group; B9.1 splits block 0 into key 0 (its own exponent) then the rest;
                    # B9.2 splits every group into a high and a low value plane (high first)
                    groups = []
                    if ev0 is not None and b == 0 and lo == 0:
                        eh0 = np.zeros_like(eh); eh0[0] = eh[0]
                        ehr = eh.copy(); ehr[0] = 0
                        groups.append((eh0, int(ev0[g]))); groups.append((ehr, int(ev[b, g])))
                    else:
                        groups.append((eh, int(ev[b, g])))
                    parts = []
                    for ehg, evg in groups:
                        if v_planes == 2:
                            Ah = np.rint(np.matmul(ehg.astype(np.float64), v_hi[g, lo:hi, :].astype(np.float64))).astype(np.int64)
                            Al = np.rint(np.matmul(ehg.astype(np.float64), v_lo[g, lo:hi, :].astype(np.float64))).astype(np.int64)
                            assert np.abs(Ah).max(initial=0) < (1 << 24) and np.abs(Al).max(initial=0) < (1 << 24)
                            parts += [(Ah, evg, 8), (Al, evg, 0)]
                        else:
                            A_b = np.rint(np.matmul(ehg.astype(np.float64), v8[g, lo:hi, :].astype(np.float64))).astype(np.int64)
                            dbg["Abmax"] = max(dbg["Abmax"], int(np.abs(A_b).max(initial=0)))
                            if weight != "l16":
                                assert np.abs(A_b).max(initial=0) < (1 << 24), "A_b out of the binary32-exact bound"
                                assert l_b <= 32640
                            parts.append((A_b, evg, 0))
                    if l_b == 0 and not any(np.any(P) for P, _, _ in parts):
                        continue
                    # ---- the fold (fp32, RNE, committed order): one term per part
                    scl = pow2_f32(-s)
                    for P, evg, off in parts:
                        esc = -(s + evg) + off
                        dbg["maxterm"] = max(dbg["maxterm"], esc); dbg["minterm"] = min(dbg["minterm"], esc)
                        scO = pow2_f32(esc)
                        Af = np.asarray([i2f32(int(x)) for x in P], dtype=np.float32)
                        O_seg = f32_add(O_seg, f32_mul(Af, scO))
                    lf = i2f32(l_b)
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




def lssab9_attention(q, k, v, mu=None, kb=KB, q_exp_mode="row", causal=True, key_valid=None,
                     sink_alone=False, v_planes=1, v_keyshift=0, ks_round=False, **kw):
    p = lssab9_quantize(q, k, v, mu, kb, q_exp_mode, sink_alone=sink_alone, v_planes=v_planes, v_keyshift=v_keyshift)
    return lssab9_core(p["q8"], p["k8"], p["v8"], p["eq"], p["ek"], p["ev"], kb=kb,
                       causal=causal, key_valid=key_valid, ev0=p["ev0"], v_planes=v_planes,
                       v_hi=p["v_hi"], v_lo=p["v_lo"], d8=p["d8"], ks_round=ks_round, **kw)
