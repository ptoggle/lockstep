#!/usr/bin/env python3
"""LSSA-B9 torch reference (Stage 2 item 2.0b, 2026-09-08): the LSSA-B8 rule with TWO OPTIONS on
the VALUE operand, measured in SPEC 7.23 to carry the 70B's attention cost (the value vectors
quantised to 7 bits with one exponent per 128-key block: +2.67 of the +2.83).  lssab8_torch.py
and lssab8_golden.py are imported for their helpers and are NOT modified; with both options
off this file is gated bit-identical to lssab8_torch (arm L9 == L8).

  sink_alone   the first key of the sequence (absolute position 0) forms its own VALUE block:
               e_v0 over key 0 alone, e_v[0'] over keys 1..KB-1, each its natural pow2
               exponent (no cap: the sink's values exceed its neighbours' by more than 2^8 on
               Llama, so a shift bounded by two weight planes buys nothing - measured).  Block
               0's partial becomes TWO integer partials folded as two consecutive fp32 terms,
               key 0's first, each with its own exponent and the block's s:
                   O = fl(O + f32(eh[0]*v8[0]) * 2^-(s + e_v0));  O = fl(O + f32(A_0') * 2^-(s + e_v[0']))
               K, the scores, the weights, NB and l are unchanged.  A kernel keeps key 0 out of
               block 0's P.V MMA and adds its term per row on the CUDA cores (128 multiply-adds).
  v_planes=2   the value is quantised on a 15-bit signed grid per block (magnitude limit 16383,
               e_v15 = the 7-bit block exponent + 7) and split into a signed high plane
               v_hi = v15 >> 8 (arithmetic shift, in [-64, 63]) and an unsigned low plane
               v_lo = v15 - (v_hi << 8) in [0, 255]; the block partials A_hi = sum eh*v_hi and A_lo = sum eh*v_lo (both
               < 2^24, exact in binary32) fold as two consecutive terms, high first:
                   O = fl(O + f32(A_hi) * 2^(8 - s - e_v15));  O = fl(O + f32(A_lo) * 2^(-s - e_v15))
               l is unchanged.  A kernel evaluates two P.V products per block.
Everything else is B8's, including the committed block order, S_WIN, the rescale rule and the
epilogue.  Measured (SPEC 7.23) on Llama-3.1-70B: see attn_bisect.py arms L9-sink, L9-v16.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/p2")
import torch
import contract_common as C
import t10_attn as T10
import lssab_torch as LB
import lssab8_golden as G8
import lssab8_torch as B8

quant_rows = T10.quant_rows
quant_pow2_blocks = LB.quant_pow2_blocks
SH_LO, SH_HI = LB.SH_LO, LB.SH_HI
BHI, KB, SEG_KEYS, S_WIN, W_LOC, NIDX = G8.BHI, G8.KB, G8.SEG * G8.KB, G8.S_WIN, G8.W_LOC, G8.NIDX
NB_SENT = B8.NB_SENT
rescale_t = B8.rescale_t
_table_on = B8._table_on


def quant_pow2_blocks_bits(x, KBv, bits):
    """LB.quant_pow2_blocks generalised to a `bits`-bit magnitude (7 = the contract's int8)."""
    g, T, dh = x.shape
    kb = KBv if KBv <= T else T
    nb = (T + kb - 1) // kb
    a = x.abs().amax(-1)
    pad = nb * kb - T
    if pad:
        a = torch.cat([a, torch.zeros(g, pad, dtype=a.dtype, device=a.device)], 1)
    m = a.view(g, nb, kb).amax(-1)
    lim = float((1 << bits) - 1)
    mant, E = torch.frexp(m)
    e = (bits - 1) - (E.to(torch.int64) - 1)
    e = torch.where(m == 0, torch.zeros_like(e), e)
    over = m * C.pow2_t(e) > lim
    e = (e - over.to(torch.int64)).clamp(-32, 48)
    sc = C.pow2_t(e)
    sctok = sc.repeat_interleave(kb, dim=1)[:, :T]
    q = torch.clamp(torch.round(x * sctok[..., None]), -lim, lim)
    return q, e, kb


def make_lssab9_long_attn(kb=KB, bhi=BHI, n=NIDX, w_loc=W_LOC, weight="rn", s_win=S_WIN,
                          seg_keys=SEG_KEYS, head_chunk=7, qblk=1024, kblk=8192, mu=None,
                          qexp="row", clamp_sh=True, stats=None, skip=True,
                          sink_alone=False, v_planes=1, v_keyshift=0, ks_round=False, v_sub=1):
    """eager_attention_forward replacement implementing LSSA-B9 (B8 + the value options).

    v_sub = 2 | 4 (B9.4, sub-block value exponents): each 128-key block's value is quantised per
    sub-block of 128 / v_sub keys on its own exponent, and the block folds v_sub terms in key order
    (the PV split into v_sub K-slices, one fold term each; the sink term, if on, stays first).
    v_keyshift = dmax > 0 (B9.3, "per-key value exponent, weight-compensated"): every key j of
    block b is quantised on ITS OWN grid e_v[b] + d_j with d_j = clamp(e_j - e_v[b], 0, dmax)
    (e_j the key's natural 7-bit exponent), and the PV product uses the weight shifted right by
    d_j (floor, or round-half-up with ks_round); l keeps the unshifted weight.  The fold is B8's
    (one term per block on e_v[b]).  The sink token has d = 0, so this subsumes sink_alone.

    qblk / kblk / head_chunk are SCHEDULE (gated bit-identical across values); kb, seg_keys,
    s_win, w_loc, n, weight and the ASCENDING block order are CONTRACT.
    """
    assert n <= bhi and qexp in ("row", "head")
    assert seg_keys % kb == 0, "a segment must be a whole number of blocks"
    seg_blocks = seg_keys // kb
    idx_sh = bhi - (13 if weight == "l16" else n)
    win_d = (16 << bhi) if weight == "l16" else (w_loc << bhi)     # window in units of d
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
        Bsz, H, T, dh = query.shape
        assert Bsz == 1, "B=1 windows only"
        dev = query.device
        li = int(getattr(module, "layer_idx", -1))
        KVH = key.shape[1]
        grp = H // KVH
        qf = query[0].float()
        kf = key[0].float()
        if mu is not None:
            kf = kf - mu_on(li, dev)[:, None, :]
        vf = value[0].float()
        k8, ekb, kb_k = quant_pow2_blocks(kf, kb)          # [KVH,T,dh], [KVH,nblk]
        vbits = 14 if v_planes == 2 else 7                 # magnitude bits: int8 = 7, the 15-bit signed grid = 14
        v8, evb, kb_v = quant_pow2_blocks_bits(vf, kb, vbits)  # v on the (7 | 15)-bit block grid
        # sink_alone: key 0 on its OWN natural grid e_v0; keys 1..kb-1 of block 0 on theirs
        if sink_alone and T > 1:
            v0, ev0, _ = quant_pow2_blocks_bits(vf[:, :1], 1, vbits)          # [KVH,1,dh], [KVH,1]
            vr, evr, _ = quant_pow2_blocks_bits(vf[:, 1:min(kb, T)], kb, vbits)  # keys 1..kb-1 as one block
            v8 = v8.clone(); v8[:, :1] = v0; v8[:, 1:min(kb, T)] = vr
            ev_sink = ev0[:, 0]                                                 # [KVH]
            evb = evb.clone(); evb[:, 0] = evr[:, 0]                            # block 0's exponent = keys 1..kb-1
        else:
            ev_sink = None
        SUB = None
        if v_sub > 1:
            assert v_planes == 1 and v_keyshift == 0 and kb % v_sub == 0
            skb = kb // v_sub
            v8s, evs, _ = quant_pow2_blocks_bits(vf, skb, vbits)          # [KVH,T,dh], [KVH,nsub]
            if sink_alone and T > 1:                                        # key 0 alone, keys 1..skb-1 as one sub-block
                v0, ev0s, _ = quant_pow2_blocks_bits(vf[:, :1], 1, vbits)
                vr, evr, _ = quant_pow2_blocks_bits(vf[:, 1:min(skb, T)], skb, vbits)
                v8s = v8s.clone(); v8s[:, :1] = v0; v8s[:, 1:min(skb, T)] = vr
                evs = evs.clone(); evs[:, 0] = evr[:, 0]
                ev_sink = ev0s[:, 0]
            v8 = v8s; SUB = evs                                              # per-sub-block exponents
        D8 = None
        if v_keyshift > 0:
            _, ekey, _ = quant_pow2_blocks_bits(vf, 1, 7)               # [KVH,T] each key's own exponent
            ev_tok = evb.repeat_interleave(kb, dim=1)[:, :T]            # the block exponent per key
            if sink_alone and T > 1:
                ev_tok = ev_tok.clone(); ev_tok[:, 0] = ev_sink
            D8 = (ekey - ev_tok).clamp(0, int(v_keyshift))              # [KVH,T] the per-key shift
            sc = C.pow2_t(ev_tok + D8)
            v8 = torch.clamp(torch.round(vf * sc[..., None]), -127.0, 127.0)
        del kf, vf
        # quant_pow2_blocks caps the block at T when T < KB; nblk is 1 either way and the
        # exponent is the same amax, so the two agree with the golden's ceil(T/KB) blocking.
        assert kb_k == min(kb, T) and kb_v == min(kb, T), "unexpected prep block size"
        assert int(evb.min()) >= -32 and int(evb.max()) <= 40, "e_v outside the contract clamp"
        nblk = (T + kb - 1) // kb
        Tp = nblk * kb
        if Tp != T:                                         # pad the LAST partial block with
            pad = Tp - T                                    # zeros; those positions are never
            z = torch.zeros(k8.shape[0], pad, dh, dtype=k8.dtype, device=dev)
            k8 = torch.cat([k8, z], 1); v8 = torch.cat([v8, z], 1)
            if D8 is not None:
                D8 = torch.cat([D8, torch.zeros(D8.shape[0], pad, dtype=D8.dtype, device=dev)], 1)
        if v_planes == 2:                                   # the planes of the PADDED value
            v_hi = torch.floor(v8 / 256.0)                  # arithmetic shift right by 8
            v_lo = v8 - v_hi * 256.0                        # in [0, 255]
            assert float(v_hi.abs().max()) <= 64 and float(v_lo.min()) >= 0 and float(v_lo.max()) <= 255
        ar = torch.arange(Tp, device=dev)
        tab = _table_on(dev, weight, n, w_loc)
        tabL = tab.numel()
        NEG = -(1 << 62)
        outs = []
        for h0 in range(0, H, head_chunk):
            h1 = min(h0 + head_chunk, H)
            hc = h1 - h0
            kvi = torch.arange(h0, h1, device=dev) // grp
            if qexp == "row":
                q8, eq = quant_rows(qf[h0:h1])                    # [hc,T,dh], [hc,T]
            else:
                q8, eq = C.quant_pow2(qf[h0:h1])                  # [hc]
            K8 = k8[kvi]; V8 = v8[kvi]; EKB = ekb[kvi]; EVB = evb[kvi]
            EVS = ev_sink[kvi] if ev_sink is not None else None
            DK = D8[kvi] if D8 is not None else None                    # [hc,Tp] per-key shifts
            EVSUB = SUB[kvi] if SUB is not None else None               # [hc,nsub] sub-block exponents
            if v_planes == 2:
                VHI = v_hi[kvi]; VLO = v_lo[kvi]
            OUT = torch.empty((hc, T, dh), dtype=torch.float32, device=dev)
            for i0 in range(0, T, qblk):
                i1 = min(i0 + qblk, T)
                nq = i1 - i0
                q8b = q8[:, i0:i1]
                rows = ar[i0:i1]
                eqb = eq[:, i0:i1, None] if qexp == "row" else eq[:, None, None]
                O_row = torch.zeros((hc, nq, dh), dtype=torch.float32, device=dev)
                l_row = torch.zeros((hc, nq), dtype=torch.float32, device=dev)
                NB_row = torch.full((hc, nq), NB_SENT, dtype=torch.int64, device=dev)
                nblk_vis = (i1 - 1) // kb + 1                      # causal: no later block
                nseg = (nblk_vis + seg_blocks - 1) // seg_blocks
                for sg in range(nseg):
                    O_seg = torch.zeros((hc, nq, dh), dtype=torch.float32, device=dev)
                    l_seg = torch.zeros((hc, nq), dtype=torch.float32, device=dev)
                    NB_seg = torch.full((hc, nq), NB_SENT, dtype=torch.int64, device=dev)
                    b_lo = sg * seg_blocks
                    b_hi = min(nblk_vis, (sg + 1) * seg_blocks)
                    bstep = max(1, kblk // kb)
                    for bt0 in range(b_lo, b_hi, bstep):
                        bt1 = min(b_hi, bt0 + bstep)
                        j0, j1 = bt0 * kb, bt1 * kb
                        S = torch.matmul(q8b, K8[:, j0:j1].transpose(-1, -2)).to(torch.int64)
                        sh = eqb + EKB[:, None, bt0:bt1]           # [hc,nq|1,nbt]
                        note("sh_min", sh.min(), "min"); note("sh_max", sh.max(), "max")
                        if clamp_sh:
                            note("sh_clamped", ((sh < SH_LO) | (sh > SH_HI)).sum(), "sum")
                            sh = sh.clamp(SH_LO, SH_HI)
                        Gb = C.g_fold_t(dh, sh, bhi)               # [hc,nq|1,nbt] int64
                        # -------- THE FOLD: strictly sequential over blocks, ASCENDING
                        for t in range(bt1 - bt0):
                            b = bt0 + t
                            keys = ar[j0 + t * kb:j0 + (t + 1) * kb]
                            valid = (keys[None, :] <= rows[:, None]) & (keys[None, :] < T)
                            SGt = S[..., t * kb:(t + 1) * kb] * Gb[..., t, None]
                            SGm = torch.where(valid[None], SGt,
                                              torch.full_like(SGt, NEG)).amax(-1)   # [hc,nq]
                            live = SGm > NEG
                            NBb = torch.where(live, -((-SGm) >> bhi) << bhi,
                                              torch.full_like(SGm, NB_SENT))
                            d = NBb[..., None] - SGt
                            d = torch.where(valid[None], d, torch.full_like(d, win_d))
                            note("d_max", torch.where(d < win_d, d, torch.zeros_like(d)).max(), "max")
                            if weight == "l16":
                                shr = (d >> bhi).clamp(min=0, max=63)
                                sel = tab.index_select(
                                    0, ((d >> idx_sh) & 8191).clamp(min=0).reshape(-1)
                                ).view_as(d).to(torch.int64)
                                eh = torch.where(d < win_d, sel >> shr, torch.zeros_like(d))
                            else:
                                ki = (d >> idx_sh).clamp(min=0, max=tabL - 1)
                                eh = tab.index_select(0, ki.reshape(-1)).view_as(d).to(torch.int64)
                            lb = eh.sum(-1)                                    # [hc,nq] int64
                            sl = slice(j0 + t * kb, j0 + (t + 1) * kb)
                            if DK is not None:                                 # B9.3: the weight-compensated key grid
                                dk = DK[:, None, sl]                           # [hc,1,kb]
                                if ks_round == "even":                         # round to nearest, ties to even
                                    q = eh >> dk; rem = eh & ((1 << dk) - 1); half = (1 << dk) >> 1
                                    eh = q + ((rem > half) | ((rem == half) & (dk > 0) & ((q & 1) == 1))).to(eh.dtype)
                                elif ks_round:                                 # round half up
                                    eh = (eh + ((1 << dk) >> 1)) >> dk
                                else:                                          # floor
                                    eh = eh >> dk
                            # the block's value partials: a list of (A, exponent offset) terms in
                            # the declared order; B8 has exactly one, (eh . v8, e_v[b])
                            terms = []
                            def _partials(ehx, Vsl, e_base):
                                if v_planes == 2:
                                    Ah = torch.matmul(ehx.double(), VHI[:, Vsl].double())
                                    Al = torch.matmul(ehx.double(), VLO[:, Vsl].double())
                                    assert int(Ah.abs().max().item()) < (1 << 24) and int(Al.abs().max().item()) < (1 << 24), "two-plane A bound"
                                    return [(Ah, e_base - 8), (Al, e_base)]
                                A = torch.matmul(ehx.double(), V8[:, Vsl].double())
                                if weight != "l16":
                                    assert int(A.abs().max().item()) < (1 << 24), "A_b bound (exact in binary32)"
                                return [(A, e_base)]
                            if EVSUB is not None:                              # B9.4: one term per sub-block, in key order
                                skb = kb // v_sub
                                for u in range(v_sub):
                                    ehu = torch.zeros_like(eh); ehu[..., u * skb:(u + 1) * skb] = eh[..., u * skb:(u + 1) * skb]
                                    ssl = slice(sl.start + u * skb, sl.start + (u + 1) * skb)
                                    if sink_alone and b == 0 and u == 0 and EVS is not None:
                                        eh0 = torch.zeros_like(eh); eh0[..., :1] = eh[..., :1]
                                        ehu = ehu.clone(); ehu[..., :1] = 0
                                        terms += [(torch.matmul(eh0.double(), V8[:, sl.start:sl.start + 1].double()), EVS[:, None].expand(hc, nq))]
                                    terms += [(torch.matmul(ehu[..., u * skb:(u + 1) * skb].double(), V8[:, ssl].double()), EVSUB[:, None, b * v_sub + u].expand(hc, nq))]
                            elif sink_alone and b == 0 and EVS is not None:
                                # key 0's term first, on its own grid; then keys 1..kb-1 on the block's
                                eh0 = eh.clone(); eh0[..., 1:] = 0
                                ehr = eh.clone(); ehr[..., :1] = 0
                                terms += _partials(eh0, sl, EVS[:, None].expand(hc, nq))
                                terms += _partials(ehr, sl, EVB[:, None, b].expand(hc, nq))
                            else:
                                terms += _partials(eh, sl, EVB[:, None, b].expand(hc, nq))
                            if weight != "l16":
                                assert int(lb.max().item()) <= 32640, "l_b bound"
                            Dd = ((NBb - NB_seg).clamp(min=0) >> bhi).clamp(max=255)
                            sgap = ((NB_seg - NBb).clamp(min=0) >> bhi).clamp(max=s_win + 1)
                            dead = (sgap > s_win) if skip else (~live)
                            note("skipped", int((dead & live).sum()), "sum")
                            O_seg = rescale_t(O_seg, Dd[..., None])
                            l_seg = rescale_t(l_seg, Dd)
                            NB_seg = torch.maximum(NB_seg, NBb)
                            se = torch.where(dead, torch.zeros_like(sgap), sgap)
                            m = (~dead).to(torch.float32)
                            for (A_t, ev_t) in terms:                          # the declared term order
                                eO = -(se + ev_t)
                                assert int(eO.min()) >= -126 and int(eO.max()) <= 127, \
                                    "term exponent outside the normal binary32 range"
                                note("term_exp_min", eO.min(), "min")
                                note("term_exp_max", eO.max(), "max")
                                scO = m * C.pow2_t(eO)
                                O_seg = O_seg + A_t.to(torch.float32) * scO[..., None]
                            scl = m * C.pow2_t(-se)
                            l_seg = l_seg + lb.double().to(torch.float32) * scl
                            del SGt, SGm, d, eh, terms, NBb
                        del S, Gb, sh
                    # -------- fold the finished segment into the row, the SAME three rules
                    Dd = ((NB_seg - NB_row).clamp(min=0) >> bhi).clamp(max=255)
                    sgap = ((NB_row - NB_seg).clamp(min=0) >> bhi).clamp(max=s_win + 1)
                    dead = (sgap > s_win) | (NB_seg == NB_SENT)
                    O_row = rescale_t(O_row, Dd[..., None])
                    l_row = rescale_t(l_row, Dd)
                    NB_row = torch.maximum(NB_row, NB_seg)
                    se = torch.where(dead, torch.zeros_like(sgap), sgap)
                    m = (~dead).to(torch.float32)
                    O_row = O_row + rescale_t(O_seg, se[..., None]) * m[..., None]
                    l_row = l_row + rescale_t(l_seg, se) * m
                nz = l_row != 0
                den = torch.where(nz, l_row, torch.ones_like(l_row))
                OUT[:, i0:i1] = torch.where(nz[..., None], O_row / den[..., None],
                                            torch.zeros_like(O_row))
                del O_row, l_row, NB_row
            outs.append(OUT)
            del q8, K8, V8
        out = torch.cat(outs, 0)
        return out.to(query.dtype)[None].transpose(1, 2).contiguous(), None
    return attn


def make_lssab8_attn(kb=KB, bhi=BHI, n=NIDX, w_loc=W_LOC, weight="rn", head_chunk=16,
                     mu=None, qexp="row", stats=None, **kw):
    """Short-context entry point.  Same algorithm, one query block and one key tile
    (qblk/kblk are schedule; gate Q1c checks the two agree bit-for-bit)."""
    return make_lssab8_long_attn(kb=kb, bhi=bhi, n=n, w_loc=w_loc, weight=weight,
                                 head_chunk=head_chunk, qblk=1 << 20, kblk=1 << 20,
                                 mu=mu, qexp=qexp, stats=stats, **kw)




def make_lssab9_attn(head_chunk=16, **kw):
    return make_lssab9_long_attn(head_chunk=head_chunk, qblk=1 << 20, kblk=1 << 20, **kw)


ARM_SPECS = {
    "L9":          dict(sink_alone=False, v_planes=1),      # == L8-RN-W9, gated bit-identical
    "L9-sink":     dict(sink_alone=True, v_planes=1),
    "L9-v16":      dict(sink_alone=False, v_planes=2),
    "L9-sink-v16": dict(sink_alone=True, v_planes=2),
    # B9.3: per-key value exponent compensated on the weight (floor / round), shift capped at dmax
    "L9-ks3":      dict(sink_alone=False, v_planes=1, v_keyshift=3),
    "L9-ks7":      dict(sink_alone=False, v_planes=1, v_keyshift=7),
    "L9-ks3r":     dict(sink_alone=False, v_planes=1, v_keyshift=3, ks_round=True),
    "L9-ks7r":     dict(sink_alone=False, v_planes=1, v_keyshift=7, ks_round=True),
    "L9-sink-ks3": dict(sink_alone=True, v_planes=1, v_keyshift=3),
    "L9-sink-ks7r": dict(sink_alone=True, v_planes=1, v_keyshift=7, ks_round=True),
    "L9-ks3e":     dict(sink_alone=False, v_planes=1, v_keyshift=3, ks_round="even"),
    "L9-ks2e":     dict(sink_alone=False, v_planes=1, v_keyshift=2, ks_round="even"),
    "L9-ks1e":     dict(sink_alone=False, v_planes=1, v_keyshift=1, ks_round="even"),
    "L9-ks2r":     dict(sink_alone=False, v_planes=1, v_keyshift=2, ks_round=True),
    "L9-ks1r":     dict(sink_alone=False, v_planes=1, v_keyshift=1, ks_round=True),
    "L9-ks2":      dict(sink_alone=False, v_planes=1, v_keyshift=2),
    # B9.4: sub-block value exponents (2 = 64-key halves, 4 = 32-key quarters), with the sink term
    "L9-sub2":      dict(sink_alone=False, v_planes=1, v_sub=2),
    "L9-sink-sub2": dict(sink_alone=True, v_planes=1, v_sub=2),
    "L9-sink-sub4": dict(sink_alone=True, v_planes=1, v_sub=4),
}


def build_arm9(name, bhi=BHI, head_chunk=7, mu=None, qblk=1024, kblk=8192, stats=None, long=True, qexp="row"):
    spec = ARM_SPECS.get(name)
    if spec is None:
        return None
    if long:
        return make_lssab9_long_attn(head_chunk=head_chunk, qblk=qblk, kblk=kblk, mu=mu, bhi=bhi, stats=stats, qexp=qexp, **spec)
    return make_lssab9_attn(head_chunk=head_chunk, mu=mu, bhi=bhi, stats=stats, qexp=qexp, **spec)
