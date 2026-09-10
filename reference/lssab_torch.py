#!/usr/bin/env python3
"""
T12b - LSSA-B reference implementation in torch, written FROM THE SPEC
(BLOCK_EXPONENT_VARIANT.md, 2026-09-01).  Independent of the kernel work.

WHAT LSSA-B IS (spec, restated so this file is self-contained)
  * Keys/values are partitioned into blocks of KB consecutive ABSOLUTE positions.
  * e_k[b,g], e_v[b,g] = clip-safe pow2 exponent (contract_common.quant_pow2's rule)
    of the amax of the block's rows over ALL channels.  Rows absent from a partial
    last block do not participate.
  * Key offset mu (T9) subtracted before quantization: k' = k - mu.
  * q8, e_q per (query row, head)   [spec: per-row; T10 proved per-row legal].
  * G[i,b] = g_fold(dh, sh) with sh = e_q[i] + e_k[b] CLAMPED to [-7, 23].
  * m_SG = max over live j of S[j]*G[b(j)]  (== max_b m_S[b]*G[b], G>0 constant in b).
    NB = least multiple of 2^B >= m_SG;  d[j] = NB - S[j]*G[b(j)].
  * table / window / integer normalizer l: unchanged.
  * PV with per-block V exponent: see "A RULE" below.
  * Epilogue: f32(A)/f32(l), then the 2^(-e_v_ref) pow2 scale in f32.

A RULE ACTUALLY IMPLEMENTED HERE (and the spec sign error it fixes)
  The spec's integer sketch reads
        e_v_ref = min_b e_v[b];  A = sum_b (A_b << (e_v[b] - e_v_ref))
  which is inconsistent with its own epilogue: the true value is
        sum_b A_b * 2^(-e_v[b]) = 2^(-e_v_ref) * sum_b A_b * 2^(e_v_ref - e_v[b]),
  so the shift is (e_v_ref - e_v[b]) and it is a LEFT (exact) shift only when
        e_v_ref = MAX_b e_v[b]   over the row's live blocks,
  not the min.  With e_v_ref = min the shift is a right shift (lossy) and the spec's
  own formula scales each block by 2^(+ (e_v[b]-e_v_min)) instead of 2^(-...), i.e. it
  inverts the relative weight of the blocks.  THIS FILE IMPLEMENTS THE MAX FORM:

        e_v_ref[i] = max over blocks b live for row i of e_v[b]        (a cummax)
        A[i]       = sum_b ( A_b[i] << (e_v_ref[i] - e_v[b]) )         (exact, integers)
        out[i]     = ( f32(A[i]) / f32(l[i]) ) * 2^(-e_v_ref[i])       (f32, exact pow2)

  Evaluated as a single exact f64 matmul: A = matmul(eh * 2^shift, v8), which equals the
  per-block integer accumulation term-by-term (a left shift distributes over the block
  sum).  Exactness bound, asserted at runtime: |eh| < 2^w, |v8| <= 127, T <= 2^17, so
  |A| <= 2^(w+7+17+shift_max) and we require that <= 2^53, i.e. shift_max <= 14 at w=15.
  Observed shift_max is recorded in `stats` for every run.

  NOTE the output is in fact INVARIANT to the choice of e_v_ref, because A carries the
  exact factor 2^(e_v_ref) and f32 rounding is exact under scaling by a power of two; the
  choice only fixes the bit-level definition that a kernel must match.

STRUCTURAL OBSERVATION USED THROUGHOUT: LSSA-B's fold structure is the SPECIAL CASE of
T10's arm R (per-row K exponents) in which the exponent is constant inside a block, so
T10's exactness argument ("fold first, then max") carries over verbatim.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/p2")
import torch
import contract_common as C
import t10_attn as T10                 # quant_rows / sdpa_attn / install_longctx (reuse)

quant_rows = T10.quant_rows            # per-row (amax over channels only), T9/T10 semantics

SH_LO, SH_HI = -7, 23                  # spec: clamp the g_fold shift to the safe band


# --------------------------------------------------------------------------- quantizers
def quant_pow2_blocks(x, KB):
    """x [g,T,dh] fp32 -> q (int8-valued fp32) [g,T,dh], e [g,nb] int64, scale_tok [g,T].

    Exponent rule byte-identical to contract_common.quant_pow2 except that the amax is
    taken over the rows of ONE BLOCK (and all channels) instead of the whole tensor.
    KB >= T  =>  one block  =>  identical to contract_common.quant_pow2.
    """
    g, T, dh = x.shape
    kb = KB if KB <= T else T
    nb = (T + kb - 1) // kb
    a = x.abs().amax(-1)                                            # [g,T]
    pad = nb * kb - T
    if pad:                                                          # zeros never raise an amax
        a = torch.cat([a, torch.zeros(g, pad, dtype=a.dtype, device=a.device)], 1)
    m = a.view(g, nb, kb).amax(-1)                                   # [g,nb]
    mant, E = torch.frexp(m)
    e = 6 - (E.to(torch.int64) - 1)
    e = torch.where(m == 0, torch.zeros_like(e), e)
    over = m * C.pow2_t(e) > 127.0
    e = (e - over.to(torch.int64)).clamp(-32, 40)
    sc = C.pow2_t(e)                                                 # [g,nb] exact pow2
    sctok = sc.repeat_interleave(kb, dim=1)[:, :T]                   # [g,T]
    q = torch.clamp(torch.round(x * sctok[..., None]), -127, 127)
    return q, e, kb


def ev_ref_rows(evb, kb, T):
    """e_v_ref[i] = max_{b live for row i} e_v[b] -- a cummax over blocks, gathered per row.
    evb [g,nb] int64 -> [g,T] int64."""
    cm = torch.cummax(evb, dim=1).values
    rb = torch.div(torch.arange(T, device=evb.device), kb, rounding_mode="floor")
    return cm[:, rb]


# --------------------------------------------------------------------- materialized form
def make_lssab_attn(KB, bhi=26, n=13, w=15, head_chunk=16, mu=None, qexp="row",
                    KB_k=None, KB_v=None, clamp_sh=True, stats=None):
    """eager_attention_forward replacement, materialized [T,T] (short context).

    Style/structure copied from verify_arm_g.make_attn so the degenerate case can be
    compared line-by-line.  KB_k / KB_v override KB for the K / V operand independently
    (a value >= T means 'whole tensor', i.e. arm-G granularity for that operand) --
    used for the K-only vs V-only granularity ablation.
    qexp: 'row' (spec) or 'head' (arm-G granularity, used by the anchor gate).
    """
    assert n <= bhi and qexp in ("row", "head")
    win = (w + 1) << bhi
    idx_sh = bhi - n
    idx_mask = (1 << n) - 1
    kbk = KB if KB_k is None else KB_k
    kbv = KB if KB_v is None else KB_v
    _mu_cache = {}

    def mu_on(li, dev):
        k = (li, str(dev))
        if k not in _mu_cache:
            _mu_cache[k] = mu[li].to(dev)
        return _mu_cache[k]

    def note(key, val, how="max"):
        if stats is None:
            return
        if how == "max":
            stats[key] = max(stats.get(key, -(1 << 62)), int(val))
        elif how == "min":
            stats[key] = min(stats.get(key, (1 << 62)), int(val))
        else:
            stats[key] = stats.get(key, 0) + int(val)

    def attn(module, query, key, value, attention_mask, *args, **kw):
        B, H, T, dh = query.shape
        assert B == 1, "B=1 windows only"
        dev = query.device
        li = int(getattr(module, "layer_idx", -1))
        KVH = key.shape[1]
        grp = H // KVH
        qf = query[0].float()
        kf = key[0].float()
        if mu is not None:
            kf = kf - mu_on(li, dev)[:, None, :]
        vf = value[0].float()

        k8, ekb, kb_k = quant_pow2_blocks(kf, kbk)          # ekb [KVH,nbk]
        v8, evb, kb_v = quant_pow2_blocks(vf, kbv)          # evb [KVH,nbv]
        ektok = ekb.repeat_interleave(kb_k, dim=1)[:, :T]   # [KVH,T] per-key K exponent
        evref = ev_ref_rows(evb, kb_v, T)                   # [KVH,T] per-row V reference
        # per-key V shift (e_v_ref[i] - e_v[b(j)]) is built per head-chunk below
        vblk = torch.div(torch.arange(T, device=dev), kb_v, rounding_mode="floor")
        evtok = evb[:, vblk]                                # [KVH,T] per-key V exponent
        del kf, vf

        mask = C.mask_on(T, dev)
        frac = C.table_on(dev, n, w)
        outs = []
        for h0 in range(0, H, head_chunk):
            h1 = min(h0 + head_chunk, H)
            kvi = torch.arange(h0, h1, device=dev) // grp
            if qexp == "row":
                q8, eq = quant_rows(qf[h0:h1])              # eq [hc,T]
                sh = eq[:, :, None] + ektok[kvi][:, None, :]        # [hc,T,T]
            else:
                q8, eq = C.quant_pow2(qf[h0:h1])            # eq [hc]
                sh = eq[:, None, None] + ektok[kvi][:, None, :]     # [hc,1,T]
            note("sh_min", sh.min(), "min"); note("sh_max", sh.max(), "max")
            if clamp_sh:
                note("sh_clamped", ((sh < SH_LO) | (sh > SH_HI)).sum(), "sum")
                sh = sh.clamp(SH_LO, SH_HI)
            G = C.g_fold_t(dh, sh, bhi)                     # [hc,T|1,T] int64
            S = torch.matmul(q8, k8[kvi].transpose(-1, -2)).to(torch.int64)
            SG = S * G
            del S, G, sh
            m = SG.masked_fill(~mask, -(1 << 62)).amax(-1, keepdim=True)
            NB = -((-m) >> bhi) << bhi
            d = NB - SG
            del SG, m, NB
            d = torch.where(mask, d, torch.full_like(d, win))
            note("d_max", torch.where(d < win, d, torch.zeros_like(d)).max(), "max")
            eh = torch.where(d < win,
                             frac[(d >> idx_sh) & idx_mask] >> torch.minimum(d >> bhi, torch.full_like(d, 63)),
                             torch.zeros_like(d))
            del d
            l = eh.sum(-1)                                   # int64 exact
            shift = evref[kvi][:, :, None] - evtok[kvi][:, None, :]     # [hc,T,T] >= 0
            smax = int(shift.max())
            note("vshift_max", smax, "max")
            assert w + 7 + int(math.ceil(math.log2(max(T, 2)))) + smax <= 53, \
                "f64 PV exactness bound violated: shift_max=%d T=%d" % (smax, T)
            ehs = eh.double() * C.pow2_t(shift, torch.float64)      # exact: x * 2^n
            A = torch.matmul(ehs, v8[kvi].double())          # exact integer f64 matmul
            del eh, ehs, shift
            of = A.to(torch.float32) / l.to(torch.float32)[..., None]
            outs.append(of * (2.0 ** (-evref[kvi].to(torch.float32)))[:, :, None])
        out = torch.cat(outs, 0)
        return out.to(query.dtype)[None].transpose(1, 2).contiguous(), None
    return attn


# ------------------------------------------------------------------------- blocked form
def make_lssab_long_attn(KB, bhi=26, n=13, w=15, head_chunk=7, qblk=1024, kblk=8192,
                         mu=None, qexp="row", KB_k=None, KB_v=None, clamp_sh=True,
                         stats=None):
    """Same arithmetic, flash-style two-pass over key blocks (long context).
    Structure copied from t10_attn.make_long_contract_attn; gated bit-identical to the
    materialized form above (t12b_gates.py, gate Q1c)."""
    assert n <= bhi and qexp in ("row", "head")
    win = (w + 1) << bhi
    idx_sh = bhi - n
    idx_mask = (1 << n) - 1
    kbk_req = KB if KB_k is None else KB_k
    kbv_req = KB if KB_v is None else KB_v
    _mu_cache = {}

    def mu_on(li, dev):
        k = (li, str(dev))
        if k not in _mu_cache:
            _mu_cache[k] = mu[li].to(dev)
        return _mu_cache[k]

    def note(key, val, how="max"):
        if stats is None:
            return
        if how == "max":
            stats[key] = max(stats.get(key, -(1 << 62)), int(val))
        elif how == "min":
            stats[key] = min(stats.get(key, (1 << 62)), int(val))
        else:
            stats[key] = stats.get(key, 0) + int(val)

    def attn(module, query, key, value, attention_mask, *args, **kw):
        B, H, T, dh = query.shape
        assert B == 1, "B=1 windows only"
        dev = query.device
        li = int(getattr(module, "layer_idx", -1))
        KVH = key.shape[1]
        grp = H // KVH
        qf = query[0].float()
        kf = key[0].float()
        if mu is not None:
            kf = kf - mu_on(li, dev)[:, None, :]
        vf = value[0].float()
        k8, ekb, kb_k = quant_pow2_blocks(kf, kbk_req)
        v8, evb, kb_v = quant_pow2_blocks(vf, kbv_req)
        del kf, vf
        assert T % kb_k == 0 and T % kb_v == 0, "T must be a multiple of the block size"
        assert qblk % kb_k == 0 and kblk % kb_k == 0
        assert qblk % kb_v == 0 and kblk % kb_v == 0
        evref = ev_ref_rows(evb, kb_v, T)                       # [KVH,T]
        frac = T10.frac32_on(dev, n, w)
        ar = torch.arange(T, device=dev)
        zero32 = torch.zeros((), dtype=torch.int32, device=dev)
        winT = torch.tensor(win, dtype=torch.int64, device=dev)
        outs = []
        for h0 in range(0, H, head_chunk):
            h1 = min(h0 + head_chunk, H)
            hc = h1 - h0
            kvi = torch.arange(h0, h1, device=dev) // grp
            if qexp == "row":
                q8, eq = quant_rows(qf[h0:h1])                  # [hc,T]
            else:
                q8, eq = C.quant_pow2(qf[h0:h1])                # [hc]
            K8 = k8[kvi]; V8 = v8[kvi]
            EKB = ekb[kvi]; EVB = evb[kvi]; EVR = evref[kvi]    # [hc,nb*]
            OUT = torch.empty((hc, T, dh), dtype=torch.float32, device=dev)
            for i0 in range(0, T, qblk):
                i1 = min(i0 + qblk, T)
                q8b = q8[:, i0:i1]
                rows = ar[i0:i1]
                eqb = eq[:, i0:i1, None] if qexp == "row" else eq[:, None, None]
                # ---- pass 1: exact integer row max of SG over valid j
                m = torch.full((hc, i1 - i0, 1), -(1 << 62), dtype=torch.int64, device=dev)
                for j0 in range(0, i1, kblk):
                    j1 = min(j0 + kblk, i1)
                    nbt = (j1 - j0) // kb_k
                    S = torch.matmul(q8b, K8[:, j0:j1].transpose(-1, -2))
                    sh = eqb + EKB[:, None, j0 // kb_k:j1 // kb_k]        # [hc,qb|1,nbt]
                    note("sh_min", sh.min(), "min"); note("sh_max", sh.max(), "max")
                    if clamp_sh:
                        note("sh_clamped", ((sh < SH_LO) | (sh > SH_HI)).sum(), "sum")
                        sh = sh.clamp(SH_LO, SH_HI)
                    Gb = C.g_fold_t(dh, sh, bhi)                          # [hc,qb|1,nbt]
                    SG = (S.to(torch.int64).view(hc, i1 - i0, nbt, kb_k)
                          * Gb[..., None]).view(hc, i1 - i0, j1 - j0)
                    if j1 > i0:
                        valid = ar[j0:j1][None, :] <= rows[:, None]
                        SG = SG.masked_fill(~valid[None], -(1 << 62))
                    m = torch.maximum(m, SG.amax(-1, keepdim=True))
                    del S, SG, Gb, sh
                NB = -((-m) >> bhi) << bhi
                # ---- pass 2: window, table, integer normalizer, exact f64 PV
                l = torch.zeros((hc, i1 - i0), dtype=torch.int64, device=dev)
                A = torch.zeros((hc, i1 - i0, dh), dtype=torch.float64, device=dev)
                for j0 in range(0, i1, kblk):
                    j1 = min(j0 + kblk, i1)
                    nbt = (j1 - j0) // kb_k
                    S = torch.matmul(q8b, K8[:, j0:j1].transpose(-1, -2))
                    sh = eqb + EKB[:, None, j0 // kb_k:j1 // kb_k]
                    if clamp_sh:
                        sh = sh.clamp(SH_LO, SH_HI)
                    Gb = C.g_fold_t(dh, sh, bhi)
                    d = NB - (S.to(torch.int64).view(hc, i1 - i0, nbt, kb_k)
                              * Gb[..., None]).view(hc, i1 - i0, j1 - j0)
                    if j1 > i0:
                        valid = ar[j0:j1][None, :] <= rows[:, None]
                        d = torch.where(valid[None], d, winT)
                    d = torch.clamp(d, max=win).to(torch.int32)
                    note("d_max", torch.where(d < win, d, zero32).max(), "max")
                    sel = frac.index_select(0, ((d >> idx_sh) & idx_mask).reshape(-1)).view_as(d)
                    eh = torch.where(d < win, sel >> (d >> bhi), zero32)
                    l += eh.sum(-1, dtype=torch.int64)
                    nbv = (j1 - j0) // kb_v
                    shift = (EVR[:, i0:i1, None]
                             - EVB[:, None, j0 // kb_v:j1 // kb_v])       # [hc,qb,nbv]
                    smax = int(shift.max())
                    note("vshift_max", smax, "max")
                    assert w + 7 + 17 + smax <= 53, "f64 PV bound: shift_max=%d" % smax
                    ehs = (eh.double().view(hc, i1 - i0, nbv, kb_v)
                           * C.pow2_t(shift, torch.float64)[..., None]).view(hc, i1 - i0, j1 - j0)   # exact: x * 2^n
                    A += torch.matmul(ehs, V8[:, j0:j1].double())
                    del S, d, sel, eh, ehs, Gb, sh, shift
                of = A.to(torch.float32) / l.to(torch.float32)[..., None]
                OUT[:, i0:i1] = of * (2.0 ** (-EVR[:, i0:i1].to(torch.float32)))[:, :, None]
            outs.append(OUT)
            del q8, K8, V8
        out = torch.cat(outs, 0)
        return out.to(query.dtype)[None].transpose(1, 2).contiguous(), None
    return attn
