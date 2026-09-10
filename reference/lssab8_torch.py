#!/usr/bin/env python3
"""
LSSA-B8 reference implementation in torch, written from the "LSSA-B8" addendum of
BLOCK_EXPONENT_VARIANT.md (T16a, 2026-09-02).  Bit-for-bit equal to lssab8_golden.py.

WHAT IT IS (the addendum is normative; this restatement makes the file self-contained)
  PREP is LSSA-B's, unchanged and shared with lssab_torch.py: key offset mu, per-block
  (KB absolute positions) K/V pow2 exponents, per-row q exponents, G = g_fold(dh,
  clamp(e_q + e_k, -7, 23)).
  Per (row, BLOCK b), ascending in absolute block order:
      m_SG[b] = max over visible j in b of S[j]*G[b]     (exact integers)
      NB_b    = least multiple of 2^B >= m_SG[b]
      d[j]    = NB_b - S[j]*G[b] >= 0
      eh[j]   = T2[d[j] >> (B - n)],  T2[k] = 0 for k >= W_LOC * 2^n
      A_b     = sum_j eh[j]*v8[j]   (|A_b| <= 4,145,280 < 2^22, exact in f32)
      l_b     = sum_j eh[j]         (<= 32,640)
  and the COMMITTED-ORDER fp32 fold, two-level (blocks inside a segment of SEG blocks,
  then segments into the row), with
      D = (NB_b - NB_run) >> B
      D > 0  -> O = rescale(O, D), l = rescale(l, D), NB_run = NB_b, s = 0
      D <= 0 -> s = -D ; s > S_WIN -> the block contributes exactly zero (exact skip)
      O = fl_RNE(O + f32(A_b) * 2^-(s + e_v[b])) ;  l = fl_RNE(l + f32(l_b) * 2^-s)
  rescale(x, D) subtracts D from the BIASED EXPONENT FIELD and flushes to +0 when the
  result would have a biased exponent < 1 (an integer operation; never a float multiply,
  so no vendor FTZ mode and no round-up-to-2^-126 boundary can appear).
  Epilogue: out = O / l with ONE correctly-rounded fp32 division; l == 0 -> 0.

BRANCHLESS FORM USED HERE, and why it is the same function
  D_eff = clamp(NB_b - NB_run, min=0) >> B   (clamped to 255: rescale by >= 255 is total flush)
  s     = clamp(NB_run - NB_b, min=0) >> B   (clamped to S_WIN+1, where the term is masked to 0)
  When NB_b > NB_run this gives (D_eff = D, s = 0); when NB_b <= NB_run it gives
  (D_eff = 0 -> rescale is the identity, s = -D).  Dead blocks (no visible key in the row)
  carry NB_b = a sentinel far below every real NB, so they never move NB_run, are always
  masked as skipped, and have A_b = l_b = 0 anyway.

THE FOLD IS A SEQUENTIAL LOOP OVER BLOCKS.  torch.cumsum / any parallel scan would round
differently and is FORBIDDEN in this reference.  Vectorisation is over rows, heads and dh
only -- all of which are independent recurrences.

ARMS (name -> parameters), consumed by build_arm8() from t12b_ppl.py / t12b_long.py:
  L8-RN-W9  (== L8)  n=7, W_LOC=9, KB=128, RN      the contract candidate / default
  L8-RN-W8           n=7, W_LOC=8, KB=128, RN      the speed fallback (no index clamp)
  L8-TR              n=7, W_LOC=9, KB=128, trunc   the control: RN must be mandatory
  L8-RN-n8           n=8, W_LOC=9, KB=128, RN      the index-width control (expect identical)
  L8-RN-KB64         n=7, W_LOC=9, KB= 64, RN      the quality fallback (2x fold cost)
  L16                16-bit block-local (13,15) table, SAME fold -- isolates the fold/order
                     change from the 8-bit change
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/p2")
import torch
import contract_common as C
import t10_attn as T10
import lssab_torch as LB                 # PREP helpers, shared with LSSA-B (never modified)
import lssab8_golden as G8               # the T2 table family + the contract constants

quant_rows = T10.quant_rows
quant_pow2_blocks = LB.quant_pow2_blocks
SH_LO, SH_HI = LB.SH_LO, LB.SH_HI

BHI = G8.BHI
KB = G8.KB
SEG_KEYS = G8.SEG * G8.KB               # 4,096 keys per fold segment (SEG scales with KB)
S_WIN = G8.S_WIN
W_LOC = G8.W_LOC
NIDX = G8.NIDX

NB_SENT = -(1 << 58)                    # a multiple of 2^26 far below every real NB


# ------------------------------------------------------------------- the contract rescale
def rescale_t(x, D):
    """x float32 tensor, D int64 tensor (>= 0) broadcastable to x.  Subtract D from the
    biased exponent field; flush to +0.0 when the result would be < 1 (subnormal/underflow).
    Exact for every normal input and never rounds.
    LSSAB8_RESCALE=mul (the LITE variant): fl32(x * 2^-D), +0.0 for D >= 126, mirroring
    lssab8_golden.rescale_mul."""
    if G8.RESCALE_MODE == "mul":
        Dc = torch.as_tensor(D, device=x.device).to(torch.int64)
        Db = torch.broadcast_to(Dc, x.shape)
        sc = C.pow2_t((-Db).clamp(min=-126))                    # (the mul variant; the exponent-field rule is the default)
        sc = torch.where(Db >= 126, torch.zeros_like(sc), sc)
        return x * sc
    u = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    e = (u >> 23) & 0xFF
    ne = e - D
    keep = (e >= 1) & (ne >= 1)
    y = (u & 0x807FFFFF) | (ne.clamp(min=0) << 23)
    y = torch.where(keep, y, torch.zeros_like(y))
    y = torch.where(y >= (1 << 31), y - (1 << 32), y)
    return y.to(torch.int32).view(torch.float32)


def _table_on(dev, kind, n, w_loc):
    if kind == "l16":
        return T10.frac32_on(dev, 13, 15)
    t = G8.t2_table(n, w_loc) if kind == "rn" else G8.t2_table_trunc(n, w_loc)
    return torch.tensor(t.astype("int64"), dtype=torch.int32, device=dev)


# ------------------------------------------------------------------------------- the core
def make_lssab8_long_attn(kb=KB, bhi=BHI, n=NIDX, w_loc=W_LOC, weight="rn", s_win=S_WIN,
                          seg_keys=SEG_KEYS, head_chunk=7, qblk=1024, kblk=8192, mu=None,
                          qexp="row", clamp_sh=True, stats=None, skip=True):
    """eager_attention_forward replacement implementing LSSA-B8.

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
        v8, evb, kb_v = quant_pow2_blocks(vf, kb)
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
                            Ab = torch.matmul(eh.double(), V8[:, j0 + t * kb:j0 + (t + 1) * kb]
                                              .double())                       # [hc,nq,dh] exact
                            if weight != "l16":
                                assert int(Ab.abs().max().item()) <= 4145280, "A_b bound"
                                assert int(lb.max().item()) <= 32640, "l_b bound"
                            Dd = ((NBb - NB_seg).clamp(min=0) >> bhi).clamp(max=255)
                            sgap = ((NB_seg - NBb).clamp(min=0) >> bhi).clamp(max=s_win + 1)
                            dead = (sgap > s_win) if skip else (~live)
                            note("skipped", int((dead & live).sum()), "sum")
                            O_seg = rescale_t(O_seg, Dd[..., None])
                            l_seg = rescale_t(l_seg, Dd)
                            NB_seg = torch.maximum(NB_seg, NBb)
                            se = torch.where(dead, torch.zeros_like(sgap), sgap)
                            ev_t = EVB[:, None, b].expand(hc, nq)
                            eO = -(se + ev_t)
                            assert int(eO.min()) >= -126 and int(eO.max()) <= 127, \
                                "term exponent outside the normal binary32 range"
                            note("term_exp_min", eO.min(), "min")
                            note("term_exp_max", eO.max(), "max")
                            m = (~dead).to(torch.float32)
                            scO = m * C.pow2_t(eO)                       # m in {0, 1}: exact
                            scl = m * C.pow2_t(-se)
                            O_seg = O_seg + Ab.to(torch.float32) * scO[..., None]
                            l_seg = l_seg + lb.double().to(torch.float32) * scl
                            del SGt, SGm, d, eh, Ab, NBb
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


# ----------------------------------------------------------------- harness arm dispatch
ARM_SPECS = {
    "L8":          dict(kb=128, n=7, w_loc=9, weight="rn"),
    "L8-RN-W9":    dict(kb=128, n=7, w_loc=9, weight="rn"),
    "L8-RN-W8":    dict(kb=128, n=7, w_loc=8, weight="rn"),
    "L8-TR":       dict(kb=128, n=7, w_loc=9, weight="tr"),
    "L8-RN-n8":    dict(kb=128, n=8, w_loc=9, weight="rn"),
    "L8-RN-KB64":  dict(kb=64,  n=7, w_loc=9, weight="rn"),
    "L16":         dict(kb=128, n=13, w_loc=16, weight="l16"),
}


def build_arm8(name, bhi=BHI, head_chunk=7, mu=None, qblk=1024, kblk=8192, stats=None,
               long=True, qexp="row"):
    """Return an eager_attention_forward for an LSSA-B8 arm, or None if `name` is not one.

    ADDITIVE HOOK: t12b_ppl.py / t12b_long.py call this FIRST; every pre-existing arm name
    (E, G, C, R, H, B*, LB*) returns None here and falls through to the untouched dispatch,
    so no existing number can move."""
    spec = ARM_SPECS.get(name)
    if spec is None:
        return None
    sk = dict(spec)
    if long:
        return make_lssab8_long_attn(head_chunk=head_chunk, qblk=qblk, kblk=kblk, mu=mu,
                                     bhi=bhi, stats=stats, qexp=qexp, **sk)
    return make_lssab8_attn(head_chunk=head_chunk, mu=mu, bhi=bhi, stats=stats, qexp=qexp, **sk)
