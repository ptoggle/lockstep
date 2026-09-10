# Self-test + adversarial generator for lssab_golden.py (the LSSA-B contract golden).
#   S1  table identity vs contract_common.build_table(13,15)
#   S2  DEGENERATE: kb >= T, mu = 0, q_exp_mode='head'  ==  contract_common
#       make_contract_attention, BIT-FOR-BIT  (pins table, g_fold, S, NB/d chain, PV, epilogue)
#   S3  e_v_ref invariance (result must not depend on the reference exponent choice)
#   S4  block-skip ON == block-skip OFF, bit-for-bit
#   S5  chunk invariance (prep of a prefix == prefix of the prep) and append-only
#   S6  adversarial suite: r_b == 0, r_b == 2^B-1, delta == delta_max +- 1, dead blocks,
#       partial last block, e_v spread, ties, GQA, T in {1..8192}
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lssab_golden as LG

BHI, N, W, KB = LG.BHI, LG.NBITS, LG.WBITS, LG.KB
WIN = (W + 1) << BHI
BASE_G = LG.g_fold_int(128, 0, BHI)

def bits(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)

# --------------------------------------------------------------- score construction
# q row = [64, 1, 1, ..., 1]  (amax 64 in [64,127] -> e_q = 0, q8 == q)
# k row = [a, b, 0, ..., 0]   -> S = 64a + b ; one key per block pinned to |.|=127 so e_k = 0
def build_qk_from_scores(Starget, dh=128, T=None, pin_block=KB):
    """Starget [T_q, T_k] ints. Returns q [1,Tq,dh], k [1,Tk,dh] float32 with
    quantisation exactly the identity (e_q = e_k = 0) and q8.k8 == Starget."""
    Tq, Tk = Starget.shape
    assert Tq == 1, "single-query-row constructor"
    # q: channel 0 = 64 (sets amax -> e_q = 0), channels 1..dh-2 = 1, channel dh-1 = 0.
    q = np.zeros((1, Tq, dh), np.float32); q[0, :, 0] = 64.0; q[0, :, 1:dh - 1] = 1.0
    k = np.zeros((1, Tk, dh), np.float32)
    a = np.clip(np.round(Starget[0] / 64.0), -127, 127)
    rem = Starget[0] - 64 * a
    assert np.abs(rem).max() <= 127, "score out of constructible range"
    k[0, :, 0] = a; k[0, :, 1] = rem
    # pin one entry per block to 127 in the channel q zeroes -> block amax 127 -> e_k = 0,
    # and the pin contributes exactly 0 to every score.
    for b in range(0, Tk, pin_block):
        k[0, b, dh - 1] = 127.0
    return q, k

def check_identity_quant(q, k):
    p = LG.lssab_quantize(q, k, k, None, KB, "row")
    assert (p["eq"] == 0).all(), p["eq"]
    assert (p["ek"] == 0).all(), p["ek"]
    assert np.array_equal(p["q8"], q) and np.array_equal(p["k8"], k)
    return p

# ------------------------------------------------------------------------ S1: table
MODE = LG.TABLE_MODE_DEFAULT          # "frac" (default) or "frac2" (LSSAB_TABLE=frac2)

def s1_table():
    t = LG.build_table(13, 15)
    try:
        import contract_common as CC
        ref = CC.table_np(13, 15)
        assert np.array_equal(t, ref), "table mismatch vs contract_common"
        src = "contract_common (frozen FRAC)"
    except Exception as ex:
        src = "contract_common unavailable (%r)" % (ex,)
        return False, src
    return True, src

# ------------------------------------------------------- S1b: the LSSA-B2 (frac2) table
def s1b_frac2():
    """T13c.  Regenerate T_hi/C_lo from 40-digit Decimal, rebuild FRAC2 from them, and
    check (a) the two-factor product IS the frac2 table, (b) it differs from FRAC by at
    most 1 ulp and only where the two roundings disagree, (c) the kernel's closed form for
    C_lo reproduces the Decimal-generated table exactly, (d) the digests."""
    thi, clo = LG.build_factor_tables()
    ok_shape = (thi.shape == (64,) and clo.shape == (128,))
    idx = np.arange(8192, dtype=np.int64)
    f2 = (thi[idx >> 7] * clo[idx & 127] + (1 << 14)) >> 15
    ok_def = np.array_equal(f2, LG.table2())
    f = LG.table(13, 15)
    d = f2 - f
    ulp = int(np.abs(d).max())
    nexact = int((d == 0).sum())
    ok_ulp = (ulp <= 1)
    ok_clo = LG.check_clo_closed_form()
    ok_rng = (int(f2.min()) >= 0 and int(f2.max()) <= (1 << 15))
    dig = LG.table_digests()
    print("  S1b T_hi[0..3]=%s  C_lo[0..3]=%s" % (list(map(int, thi[:4])), list(map(int, clo[:4]))))
    print("  S1b max|FRAC2-FRAC| = %d ; exact match %d/8192 (%.2f%%) ; diff histogram %s"
          % (ulp, nexact, 100.0 * nexact / 8192,
             {int(k): int(v) for k, v in zip(*np.unique(d, return_counts=True))}))
    print("  S1b C_lo closed form 32768+((%d + l*(l-%d))>>%d) exact over all 128 l: %s"
          % (LG.CLO_A, LG.CLO_B, LG.CLO_SH, ok_clo))
    for k in ("T_hi", "C_lo", "FRAC2", "FRAC"):
        print("  S1b sha256(%-5s csv) = %s" % (k, dig[k]))
    return ok_shape and ok_def and ok_ulp and ok_clo and ok_rng

# -------------------------------------------------------------------- S2: degenerate
def s2_degenerate(seed=0, H=4, KVH=2, T=64, dh=128, verbose=True):
    import torch
    import contract_common as CC
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = np.random.RandomState(seed)
    q = g.standard_normal((H, T, dh)).astype(np.float32)
    k = g.standard_normal((KVH, T, dh)).astype(np.float32)
    v = g.standard_normal((KVH, T, dh)).astype(np.float32)
    # golden: one block covering everything, mu = 0, per-HEAD q exponent
    out_g = LG.lssab_attention(q, k, v, mu=None, kb=max(T, KB), q_exp_mode="head", causal=True)
    # G0 (frac2 mode): compare against a contract_common-EQUIVALENT path carrying FRAC2.
    # contract_common.py is NOT modified: its table accessor is substituted for the duration
    # of the call and then restored, so make_contract_attention runs its own arithmetic with
    # the frac2 table.  In the default ("frac") mode nothing is substituted.
    _saved = CC.table_on
    if MODE == "frac2":
        _f2 = torch.tensor(np.asarray(LG.table2()), dtype=torch.int64, device=dev)
        CC.table_on = lambda d_, n_=13, w_=15: _f2
    qt = torch.tensor(q, device=dev)[None]
    kt = torch.tensor(k, device=dev)[None]
    vt = torch.tensor(v, device=dev)[None]
    try:
        fn = CC.make_contract_attention(BHI, head_chunk=16, n=13, w=15)
        out_c, _ = fn(None, qt, kt, vt, None)
    finally:
        CC.table_on = _saved
    out_c = out_c[0].transpose(0, 1).contiguous().cpu().numpy()      # [H,T,dh] f32
    same = np.array_equal(bits(out_g), bits(out_c))
    if verbose and not same:
        d = np.nonzero(bits(out_g) != bits(out_c))
        print("  degenerate mismatch at", len(d[0]), "of", out_g.size,
              "first:", [x[0] for x in d], out_g[tuple(x[0] for x in d)],
              out_c[tuple(x[0] for x in d)])
    return same

# ------------------------------------------------- S3/S4: invariances on random data
def s3_s4_invariance(seed=1, H=4, KVH=2, T=400, dh=128):
    g = np.random.RandomState(seed)
    q = g.standard_normal((H, T, dh)).astype(np.float32)
    k = g.standard_normal((KVH, T, dh)).astype(np.float32)
    v = (g.standard_normal((KVH, T, dh)) * (1 + 3 * g.rand(KVH, T, 1))).astype(np.float32)
    mu = (0.1 * g.standard_normal((KVH, dh))).astype(np.float32)
    a = LG.lssab_attention(q, k, v, mu=mu, causal=True, block_skip=True)
    b = LG.lssab_attention(q, k, v, mu=mu, causal=True, block_skip=False)
    skipeq = np.array_equal(bits(a), bits(b))
    return skipeq, a

def s3_evref(seed=2, H=2, KVH=1, T=300, dh=128, bump=3):
    """The result must be invariant to using a LARGER e_v_ref than max_b e_v[b]."""
    g = np.random.RandomState(seed)
    q = g.standard_normal((H, T, dh)).astype(np.float32)
    k = g.standard_normal((KVH, T, dh)).astype(np.float32)
    v = g.standard_normal((KVH, T, dh)).astype(np.float32)
    p = LG.lssab_quantize(q, k, v, None, KB, "row")
    a = LG.lssab_core(p["q8"], p["k8"], p["v8"], p["eq"], p["ek"], p["ev"])
    b = _core_with_evref_bump(p, bump)
    return np.array_equal(bits(a), bits(b))

def _core_with_evref_bump(p, bump):
    import types
    out, dbg = LG.lssab_core(p["q8"], p["k8"], p["v8"], p["eq"], p["ek"], p["ev"], debug=True)
    H, T, dh = out.shape
    out2 = np.zeros_like(out)
    for h in range(H):
        for r in range(T):
            A = dbg["A"][h, r]
            l = int(dbg["l"][h, r])
            if l == 0:
                continue
            ev = int(dbg["evref"][h, r]) + bump
            Af = LG._i2f32v(np.asarray([int(x) << bump for x in A], dtype=object))
            out2[h, r, :] = ((Af / LG.i2f32(l)).astype(np.float32)
                             * np.float32(np.ldexp(1.0, -ev))).astype(np.float32)
    return out2

# -------------------------------------------------------------- S5: chunk invariance
def s5_chunk(seed=3, KVH=2, T=520, dh=128):
    g = np.random.RandomState(seed)
    k = g.standard_normal((KVH, T, dh)).astype(np.float32)
    v = g.standard_normal((KVH, T, dh)).astype(np.float32)
    q = g.standard_normal((2, T, dh)).astype(np.float32)
    full = LG.lssab_quantize(q, k, v, None, KB, "row")
    ok = True
    for cut in (128, 256, 384, 512, 519):
        part = LG.lssab_quantize(q[:, :cut], k[:, :cut], v[:, :cut], None, KB, "row")
        nb = (cut + KB - 1) // KB
        # complete blocks must be untouched by later appends
        nfull = cut // KB
        if nfull:
            ok &= np.array_equal(part["ek"][:nfull], full["ek"][:nfull])
            ok &= np.array_equal(part["ev"][:nfull], full["ev"][:nfull])
            ok &= np.array_equal(part["k8"][:, :nfull * KB], full["k8"][:, :nfull * KB])
            ok &= np.array_equal(part["v8"][:, :nfull * KB], full["v8"][:, :nfull * KB])
        ok &= np.array_equal(part["eq"], full["eq"][:, :cut])
    return bool(ok)

# ------------------------------------------------------------- S6: adversarial suite
def adversarial_cases(limit=None):
    """Yields dicts with q,k,v,mu,causal,tag. Cases are constructed so the golden's
    own asserts (r_b >= 0, d < 2^31, plane ranges) are exercised at their boundaries."""
    cases = []
    rs = np.random.RandomState(20260901)

    # --- A. targeted r_b and delta boundaries with an exactly-constructed score row
    #    single query row at position T-1, keys 0..T-1, causal
    def tgt(Tk, scores, tag, evspread=0):
        S = np.zeros((1, Tk), np.int64)
        S[0, :] = np.clip(np.asarray(scores, np.int64), -8255, 8255)
        q1, k1 = build_qk_from_scores(S)
        # replicate the single query row into position Tk-1 of a full causal problem
        q = np.zeros((1, Tk, 128), np.float32); q[0] = q1[0, 0][None, :]
        v = (rs.standard_normal((1, Tk, 128)) * 8).astype(np.float32)
        if evspread:
            for b in range(0, Tk, KB):
                v[0, b:b + KB] *= np.float32(2.0 ** (-(b // KB) * evspread))
        return dict(q=q, k=k1, v=v, mu=None, causal=True, tag=tag, focus_row=Tk - 1)

    G = BASE_G
    # r_b == 0 on the max block: all scores <= 0 with the max exactly 0
    Tk = 384
    sc = -rs.randint(1, 300, size=Tk).astype(np.int64); sc[Tk - 1] = 0
    cases.append(tgt(Tk, sc, "r0_maxzero"))
    # r_b == 2^B - 1 : search m so that (-m*G) mod 2^B == 2^B-1  <=>  m*G == 1 mod 2^B
    m = None
    for cand in range(1, 1 << 14):
        if (cand * G) % (1 << BHI) == 1:
            m = cand; break
        if ((-cand) * G) % (1 << BHI) == 1:
            m = -cand; break
    if m is not None:
        sc = -rs.randint(1, 300, size=Tk).astype(np.int64) + m; sc[Tk - 1] = m
        cases.append(tgt(Tk, sc, "r_max_2B_minus_1"))
    # r_b == 1 and r_b == 2 (near the bottom edge)
    for want in (1, 2, 3):
        mm = None
        for cand in range(1, 1 << 15):
            if ((-cand * G) % (1 << BHI)) == want:
                mm = cand; break
        if mm is not None:
            sc = -rs.randint(1, 200, size=Tk).astype(np.int64) + mm; sc[Tk - 1] = mm
            cases.append(tgt(Tk, sc, "r_eq_%d" % want))
    # delta == delta_max, delta_max-1, delta_max+1 on the max block
    for off in (-1, 0, 1):
        sc = np.full(Tk, -8000, np.int64)
        mx = 500
        sc[Tk - 1] = mx
        dmax = -(-(WIN - ((-mx * G) % (1 << BHI))) // G)
        for j in range(Tk - 40, Tk - 1):
            sc[j] = mx - (dmax + off)
        sc[:Tk - 40] = mx - dmax - 5
        cases.append(tgt(Tk, sc, "delta_boundary_%+d" % off))
    # blocks entirely dead (a whole 128-block far below the row max)
    sc = rs.randint(-20, 20, size=Tk).astype(np.int64)
    sc[128:256] = -8000
    sc[Tk - 1] = 300
    cases.append(tgt(Tk, sc, "dead_block"))
    # ties everywhere (all scores equal -> every eh identical, l = T*eh)
    for val in (0, 137, -55):
        cases.append(tgt(256, np.full(256, val, np.int64), "ties_%d" % val))
    # e_v spread across blocks
    for sp in (1, 2, 4, 8, 16):
        sc = rs.randint(-40, 40, size=512).astype(np.int64)
        cases.append(tgt(512, sc, "ev_spread_%d" % sp, evspread=sp))

    # --- B. random full problems, incl. GQA and partial last blocks
    for T in (1, 2, 3, 17, 63, 64, 127, 128, 129, 130, 255, 256, 257, 383, 384, 512, 513, 1000):
        for H, KVH in ((1, 1), (4, 2), (7, 1)):
            g = np.random.RandomState(1000 + T * 13 + H)
            q = g.standard_normal((H, T, 128)).astype(np.float32)
            k = g.standard_normal((KVH, T, 128)).astype(np.float32)
            v = g.standard_normal((KVH, T, 128)).astype(np.float32)
            mu = (0.3 * g.standard_normal((KVH, 128))).astype(np.float32) if (T % 3) else None
            cases.append(dict(q=q, k=k, v=v, mu=mu, causal=True, tag="rand_T%d_H%d" % (T, H)))
    # GQA at the model shape
    for T in (128, 320, 1024):
        g = np.random.RandomState(7 * T)
        cases.append(dict(q=g.standard_normal((28, T, 128)).astype(np.float32),
                          k=g.standard_normal((4, T, 128)).astype(np.float32),
                          v=g.standard_normal((4, T, 128)).astype(np.float32),
                          mu=(0.2 * g.standard_normal((4, 128))).astype(np.float32),
                          causal=True, tag="gqa28x4_T%d" % T))
    # heavy-tail value blocks: a per-block outlier that moves e_v block by block
    for T in (256, 640):
        g = np.random.RandomState(31 * T)
        v = g.standard_normal((2, T, 128)).astype(np.float32)
        for b in range(0, T, KB):
            v[:, b] *= np.float32(2.0 ** ((b // KB) % 5))       # e_v spread 0..4
        cases.append(dict(q=g.standard_normal((4, T, 128)).astype(np.float32),
                          k=g.standard_normal((2, T, 128)).astype(np.float32),
                          v=v, mu=None, causal=True, tag="v_outlier_T%d" % T))
    # explicit e_v spread ladder on full random problems
    for sp in (1, 2, 4, 8):
        T = 512
        g = np.random.RandomState(555 + sp)
        v = g.standard_normal((2, T, 128)).astype(np.float32)
        for b in range(0, T, KB):
            v[:, b:b + KB] *= np.float32(2.0 ** (-(b // KB) * sp))
        cases.append(dict(q=g.standard_normal((4, T, 128)).astype(np.float32),
                          k=g.standard_normal((2, T, 128)).astype(np.float32),
                          v=v, mu=None, causal=True, tag="ev_ladder_%d" % sp))
    # --- C. bulk random sweep (>= 200 cases total), multi-block rows, GQA, mixed shapes
    for seed in range(40):
        for (H, KVH) in ((1, 1), (2, 1), (4, 2), (8, 4)):
            T = int(rs.choice([1, 5, 33, 65, 96, 128, 160, 200, 256, 300, 384, 400, 511, 512,
                               600, 640, 700, 768, 900, 1024, 1200, 1400, 1536, 1800, 2048]))
            g = np.random.RandomState(90000 + seed * 17 + H)
            mu = (0.25 * g.standard_normal((KVH, 128))).astype(np.float32) if (seed % 2) else None
            cases.append(dict(q=g.standard_normal((H, T, 128)).astype(np.float32),
                              k=g.standard_normal((KVH, T, 128)).astype(np.float32),
                              v=g.standard_normal((KVH, T, 128)).astype(np.float32),
                              mu=mu, causal=True, tag="sweep%02d_T%d_H%d" % (seed, T, H)))
    # --- D. long context: multi-block rows at the campaign lengths
    for T in (2048, 4096, 8192):
        g = np.random.RandomState(4242 + T)
        cases.append(dict(q=g.standard_normal((2, T, 128)).astype(np.float32),
                          k=g.standard_normal((1, T, 128)).astype(np.float32),
                          v=g.standard_normal((1, T, 128)).astype(np.float32),
                          mu=None, causal=True, tag="long_T%d" % T))
    if limit:
        cases = cases[:limit]
    return cases

def main():
    print("=" * 78)
    print("TABLE MODE: %s" % MODE)
    ok1, src = s1_table(); print("S1 table vs %s: %s" % (src, "PASS" if ok1 else "SKIP/FAIL"))
    ok1b = s1b_frac2(); print("S1b LSSA-B2 two-factor table (frac2): %s" % ("PASS" if ok1b else "FAIL"))
    try:
        ok2 = s2_degenerate()
        ok2b = s2_degenerate(seed=5, H=8, KVH=4, T=100)
        print("S2 degenerate (kb>=T, mu=0, per-head e_q) vs contract_common: %s"
              % ("PASS" if (ok2 and ok2b) else "FAIL"))
    except Exception as ex:
        ok2 = False; print("S2 degenerate: ERROR %r" % (ex,))
    ok4, _ = s3_s4_invariance(); print("S4 block-skip ON == OFF: %s" % ("PASS" if ok4 else "FAIL"))
    ok3 = s3_evref(); print("S3 e_v_ref invariance: %s" % ("PASS" if ok3 else "FAIL"))
    ok5 = s5_chunk(); print("S5 chunk invariance / append-only: %s" % ("PASS" if ok5 else "FAIL"))
    cs = adversarial_cases()
    print("S6 adversarial suite: %d cases" % len(cs))
    nfail = 0
    stats = dict(skipped=0, blocks=0)
    for c in cs:
        try:
            o, d = LG.lssab_attention(c["q"], c["k"], c["v"], mu=c["mu"], causal=c["causal"],
                                      debug=True)
            o2 = LG.lssab_attention(c["q"], c["k"], c["v"], mu=c["mu"], causal=c["causal"],
                                    block_skip=False)
            assert np.array_equal(bits(o), bits(o2)), "skip on/off differ"
            assert np.isfinite(o).all(), "non-finite"
            stats["skipped"] += d["skipped"]; stats["blocks"] += d["blocks"]
        except Exception as ex:
            nfail += 1; print("  FAIL %-24s %r" % (c["tag"], ex))
    print("S6: %d/%d ok; dead (row,block) pairs %d / %d (%.2f%%)"
          % (len(cs) - nfail, len(cs), stats["skipped"], stats["blocks"],
             100.0 * stats["skipped"] / max(1, stats["blocks"])))
    allok = ok1 and ok1b and ok2 and ok3 and ok4 and ok5 and nfail == 0
    print("=" * 78); print("GOLDEN SELFTEST: %s" % ("PASS" if allok else "FAIL"))
    return 0 if allok else 1

if __name__ == "__main__":
    sys.exit(main())
