#!/usr/bin/env python3
"""
LSSA-B8 self-test: lssab8_torch == lssab8_golden BIT-FOR-BIT, plus the invariances the
addendum claims and the boundary rules it pins.  CPU only, no GPU, no model.

  S0  T2 table family: values, the single exact tie at k = 2^n, the zero floor at 9
      doublings (so W_LOC = 10 is bit-identical to W_LOC = 9), digests.
  S1  rescale(): the exponent-field rule vs the naive float multiply, incl. the documented
      boundary where IEEE rounds (2^24-1)*2^-150 UP to 2^-126 and the rule gives 0;
      golden rescale == torch rescale over a grid.
  S2  torch == golden bit-for-bit, random q/k/v, T in {256, 320, 512, 1000, 1024, 2048}, GQA.
  S3  200-doubling jumps between blocks (exercises rescale, flush and the exact skip).
  S4  FTZ boundary vectors: V scaled so terms land at the bottom of the normal range.
  S5  degenerate rows: T=1, single visible key, all-equal scores, dead blocks, e_v ladders.
  S6  skip ON == skip OFF, bit-for-bit (both implementations).
  S7  segment-split fold == sequential fold (the split-KV decode structure).
  S8  128-aligned chunk invariance: prep append-only, and every row of a prefix identical.
  S9  decode row == prefill row.
  S10 schedule invariance: qblk / kblk / head_chunk do not change a bit.
  S11 every arm (L8-RN-W9 / W8 / TR / n8 / KB64 / L16) torch == golden.
"""
import os, sys, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import lssab8_golden as G8
import lssab8_torch as L8
import lssab_selftest as LS8            # build_qk_from_scores (score constructor), reused

BHI, KB, SEG, S_WIN, W_LOC, NIDX = G8.BHI, G8.KB, G8.SEG, G8.S_WIN, G8.W_LOC, G8.NIDX
BASE_G = G8.g_fold_int(128, 0, BHI)

N_PASS = 0
N_FAIL = 0
FAILS = []


def check(name, ok, extra=""):
    global N_PASS, N_FAIL
    if ok:
        N_PASS += 1
        print("  PASS  %-46s %s" % (name, extra))
    else:
        N_FAIL += 1
        FAILS.append(name)
        print("  FAIL  %-46s %s" % (name, extra))
    return ok


def bits(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)


class _Mod:
    layer_idx = 0


def torch_out(q, k, v, arm="L8-RN-W9", **kw):
    """Run the torch reference and return [H,T,dh] float32 numpy."""
    H, T, dh = q.shape
    KVH = k.shape[0]
    spec = dict(L8.ARM_SPECS[arm])
    fn = L8.make_lssab8_long_attn(head_chunk=kw.pop("head_chunk", 4),
                                  qblk=kw.pop("qblk", 512), kblk=kw.pop("kblk", 1024),
                                  **spec, **kw)
    qt = torch.tensor(q)[None]                      # [1,H,T,dh]
    kt = torch.tensor(k)[None]
    vt = torch.tensor(v)[None]
    o, _ = fn(_Mod(), qt, kt, vt, None)             # [1,T,H,dh]
    return o[0].transpose(0, 1).contiguous().numpy()


def golden_out(q, k, v, arm="L8-RN-W9", **kw):
    spec = L8.ARM_SPECS[arm]
    kb = spec["kb"]
    seg = (SEG * KB) // kb
    w = spec["weight"]
    return G8.lssab8_attention(q, k, v, mu=None, kb=kb, seg=seg,
                               n=(NIDX if w == "l16" else spec["n"]),
                               w_loc=(W_LOC if w == "l16" else spec["w_loc"]),
                               weight=w, **kw)


def cmp_case(q, k, v, tag, arm="L8-RN-W9", **kw):
    a = torch_out(q, k, v, arm=arm, **kw)
    b = golden_out(q, k, v, arm=arm)
    same = np.array_equal(bits(a), bits(b))
    if not same:
        d = np.nonzero(bits(a) != bits(b))
        extra = "%d/%d elements differ, first %s: torch=%r golden=%r" % (
            len(d[0]), a.size, [int(x[0]) for x in d],
            float(a[tuple(x[0] for x in d)]), float(b[tuple(x[0] for x in d)]))
    else:
        extra = ""
    return same, extra


# ------------------------------------------------------------------------------ S0
def s0_table():
    ok = True
    t7 = G8.t2_table(7, 9)
    t7w8 = G8.t2_table(7, 8)
    t7w10 = G8.t2_table(7, 10)
    t8 = G8.t2_table(8, 9)
    ok &= check("S0 T2 length = 17*2^n (zero-padded to the chain bound)",
                len(t7) == 17 * 128 and len(t8) == 17 * 256, "n=7: %d B, n=8: %d B" % (len(t7), len(t8)))
    ok &= check("S0 T2[0] = 255", int(t7[0]) == 255)
    ok &= check("S0 the single exact tie at k = 2^n is 127.5 -> 128 (half-up == RNE)",
                int(t7[128]) == 128 and int(t8[256]) == 128)
    # exhaustive tie scan
    from decimal import Decimal
    ties = [k for k in range(9 * 128)
            if (Decimal(255) * (-Decimal(k) / Decimal(128) * G8._LN2).exp()) % 1 == Decimal("0.5")]
    ok &= check("S0 exactly ONE exact tie in the n=7 family", ties == [128], "ties at k=%s" % ties)
    ok &= check("S0 last live entry T2[9*2^n - 1] = 1, T2[9*2^n] = 0",
                int(t7[9 * 128 - 1]) == 1 and int(t7[9 * 128]) == 0)
    ok &= check("S0 W_LOC=10 is BIT-IDENTICAL to W_LOC=9 (the u8 support ends at 9 doublings)",
                np.array_equal(t7, t7w10), "-> 'widen to W10' is NOT an available fallback")
    ok &= check("S0 W_LOC=8 is a strict truncation of W_LOC=9",
                np.array_equal(t7w8[:1024], t7[:1024]) and (t7w8[1024:] == 0).all()
                and (t7[1024:9 * 128] > 0).any())
    ok &= check("S0 n=8 T2[2047] = 1 (the design text said 2)", int(t8[2047]) == 1)
    for kk, vv in sorted(G8.t2_digests().items()):
        print("        sha256(%-12s csv) = %s" % (kk, vv))
    return ok


# ------------------------------------------------------------------------------ S1
def s1_rescale():
    ok = True
    xs = [1.0, -1.0, 0.5, 3.0, 1.9999999, float(np.float32(np.ldexp(2.0 ** 24 - 1, -150 + 23))),
          float(np.float32(np.ldexp(1.0, -120))), float(np.float32(np.ldexp(1.0, -126))),
          1e30, -1e-30, 0.0]
    x = np.asarray(xs, dtype=np.float32)
    for D in (0, 1, 3, 23, 24, 100, 126, 127, 200, 255):
        g = G8.rescale(x, D)
        t = L8.rescale_t(torch.tensor(x), torch.tensor(D, dtype=torch.int64)).numpy()
        if not np.array_equal(bits(g), bits(t)):
            ok &= check("S1 golden rescale == torch rescale (D=%d)" % D, False)
    ok &= check("S1 golden rescale == torch rescale over the grid", ok)
    # the documented FTZ boundary: IEEE rounds this product UP to 2^-126, the rule gives 0
    xb = np.float32(np.ldexp(float(2 ** 24 - 1), -23))          # 1.9999999 (all-ones mantissa)
    naive = np.float32(np.float64(xb) * np.float64(2.0) ** -127)
    ruled = G8.rescale(np.asarray([xb], np.float32), 127)[0]
    ok &= check("S1 FTZ boundary: exponent-field rule = 0 where a float multiply rounds UP",
                float(ruled) == 0.0 and float(naive) != 0.0,
                "naive=%r rule=%r" % (float(naive), float(ruled)))
    ok &= check("S1 rescale is exact (never rounds) for normal results",
                float(G8.rescale(np.asarray([1.9999999], np.float32), 3)[0])
                == float(np.float32(1.9999999) / 8))
    ok &= check("S1 flush returns +0.0 (sign cleared), including for negative inputs",
                bits(G8.rescale(np.asarray([-1.0], np.float32), 200))[0] == 0)
    return ok


# ---------------------------------------------------------------------------- cases
def random_cases():
    cs = []
    for T in (256, 320, 512, 1000, 1024, 2048):
        for (H, KVH) in ((1, 1), (4, 2), (7, 1)):
            g = np.random.RandomState(1000 + T + H)
            cs.append(dict(tag="rand_T%d_H%dx%d" % (T, H, KVH),
                           q=g.standard_normal((H, T, 128)).astype(np.float32),
                           k=g.standard_normal((KVH, T, 128)).astype(np.float32),
                           v=g.standard_normal((KVH, T, 128)).astype(np.float32)))
    return cs


def jump_cases():
    """200-doubling jumps: consecutive blocks whose NB differ by ~200 doublings, so the
    fold rescales hard and the exact skip fires."""
    cs = []
    G = BASE_G
    for jump in (50, 126, 127, 200, 400):
        dS = int(round(jump * (1 << BHI) / G))
        for T in (512, 1024):
            nb = T // KB
            S = np.zeros((1, T), np.int64)
            for b in range(nb):
                lvl = (b % 2) * dS                       # alternate high / low blocks
                S[0, b * KB:(b + 1) * KB] = lvl - np.arange(KB) % 37
            S = np.clip(S, -8255, 8255)
            q1, k1 = LS8.build_qk_from_scores(S)
            q = np.zeros((1, T, 128), np.float32); q[0] = q1[0, 0][None, :]
            g = np.random.RandomState(7 * T + jump)
            v = (g.standard_normal((1, T, 128)) * 8).astype(np.float32)
            cs.append(dict(tag="jump%d_T%d" % (jump, T), q=q, k=k1, v=v))
    return cs


def ftz_cases():
    """V blocks scaled by huge negative powers of two so the fold terms and the running O
    land at the bottom of the normal binary32 range and the flush rule is exercised."""
    cs = []
    for sp in (0, 8, 16, 24, 32, 40):
        T = 512
        g = np.random.RandomState(555 + sp)
        v = g.standard_normal((1, T, 128)).astype(np.float32)
        for b in range(0, T, KB):
            v[:, b:b + KB] *= np.float32(2.0 ** (-(b // KB) * sp))
        cs.append(dict(tag="ftz_evladder_%d" % sp,
                       q=g.standard_normal((2, T, 128)).astype(np.float32),
                       k=g.standard_normal((1, T, 128)).astype(np.float32), v=v))
    # a block whose V is tiny but nonzero, next to one whose V is huge
    T = 384
    g = np.random.RandomState(99)
    v = g.standard_normal((1, T, 128)).astype(np.float32)
    v[:, :KB] *= np.float32(2.0 ** 30); v[:, KB:2 * KB] *= np.float32(2.0 ** -30)
    cs.append(dict(tag="ftz_extreme_v",
                   q=g.standard_normal((2, T, 128)).astype(np.float32),
                   k=g.standard_normal((1, T, 128)).astype(np.float32), v=v))
    return cs


def degenerate_cases():
    cs = []
    for T in (1, 2, 128, 129, 130, 255, 257, 384):
        g = np.random.RandomState(31 * T)
        cs.append(dict(tag="deg_T%d" % T,
                       q=g.standard_normal((2, T, 128)).astype(np.float32),
                       k=g.standard_normal((1, T, 128)).astype(np.float32),
                       v=g.standard_normal((1, T, 128)).astype(np.float32)))
    # all scores equal (every eh identical, l = n*eh)
    for val in (0, 137, -55):
        T = 256
        S = np.full((1, T), val, np.int64)
        q1, k1 = LS8.build_qk_from_scores(S)
        q = np.zeros((1, T, 128), np.float32); q[0] = q1[0, 0][None, :]
        g = np.random.RandomState(abs(val) + 5)
        cs.append(dict(tag="ties_%d" % val, q=q, k=k1,
                       v=g.standard_normal((1, T, 128)).astype(np.float32)))
    # a whole block far below the row max (dead block)
    T = 512
    S = np.random.RandomState(3).randint(-20, 20, size=(1, T)).astype(np.int64)
    S[0, KB:2 * KB] = -8000
    S[0, T - 1] = 300
    q1, k1 = LS8.build_qk_from_scores(S)
    q = np.zeros((1, T, 128), np.float32); q[0] = q1[0, 0][None, :]
    cs.append(dict(tag="dead_block", q=q, k=k1,
                   v=np.random.RandomState(4).standard_normal((1, T, 128)).astype(np.float32)))
    return cs


# ------------------------------------------------------------------------- the runner
def run_cmp(cases, label, arm="L8-RN-W9", **kw):
    nfail = 0
    t0 = time.time()
    for c in cases:
        same, extra = cmp_case(c["q"], c["k"], c["v"], c["tag"], arm=arm, **kw)
        if not same:
            nfail += 1
            print("        MISMATCH %-22s %s" % (c["tag"], extra))
    return check("%s (%d cases, arm %s)" % (label, len(cases), arm), nfail == 0,
                 "%.1fs" % (time.time() - t0))


def s6_skip(cases):
    nfail = 0
    for c in cases[:8]:
        a = golden_out(c["q"], c["k"], c["v"])
        b = G8.lssab8_attention(c["q"], c["k"], c["v"], skip=False)
        if not np.array_equal(bits(a), bits(b)):
            nfail += 1
            print("        golden skip ON != OFF for %s" % c["tag"])
    for c in cases[:4]:
        a = torch_out(c["q"], c["k"], c["v"])
        b = torch_out(c["q"], c["k"], c["v"], skip=False)
        if not np.array_equal(bits(a), bits(b)):
            nfail += 1
            print("        torch skip ON != OFF for %s" % c["tag"])
    return check("S6 exact skip ON == OFF, bit-for-bit", nfail == 0)


def s7_segsplit():
    nfail = ntot = 0
    for seed, T in ((0, 320), (1, 640), (2, 1024)):
        g = np.random.RandomState(seed)
        q = g.standard_normal((2, T, 128)).astype(np.float32)
        k = g.standard_normal((1, T, 128)).astype(np.float32)
        v = g.standard_normal((1, T, 128)).astype(np.float32)
        p = G8.lssab_quantize(q, k, v, None, KB, "row")
        o = G8.lssab8_core(p["q8"], p["k8"], p["v8"], p["eq"], p["ek"], p["ev"])
        for (h, i) in ((0, T - 1), (1, T - 1), (0, T // 2), (1, KB + 1), (0, KB - 1)):
            r, _, _, _ = G8.lssab8_row_segments(p["q8"], p["k8"], p["v8"], p["eq"], p["ek"],
                                                p["ev"], h, i)
            ntot += 1
            if not np.array_equal(bits(r), bits(o[h, i])):
                nfail += 1
    return check("S7 segment-split fold == sequential fold", nfail == 0, "%d rows" % ntot)


def s8_chunk():
    """128-aligned chunk invariance: the prep of a prefix equals the prefix of the prep
    (append-only), and every row of the prefix has a bit-identical output."""
    T = 640
    g = np.random.RandomState(11)
    q = g.standard_normal((2, T, 128)).astype(np.float32)
    k = g.standard_normal((1, T, 128)).astype(np.float32)
    v = g.standard_normal((1, T, 128)).astype(np.float32)
    full = G8.lssab8_attention(q, k, v)
    ok = True
    for cut in (128, 256, 384, 512):
        part = G8.lssab8_attention(q[:, :cut], k[:, :cut], v[:, :cut])
        if not np.array_equal(bits(part), bits(full[:, :cut])):
            ok = False
            print("        chunk cut=%d differs" % cut)
    check("S8 128-aligned chunk invariance (aligned cuts bit-identical)", ok)
    # the NON-aligned cut MUST be allowed to differ: the last block is re-quantised over
    # fewer rows.  Constructed so the trailing block's amax really changes, which is the
    # reason the engine needs the T13B block-aligned split hook.
    k2 = k.copy(); v2 = v.copy()
    k2[:, 505] *= np.float32(64.0); v2[:, 505] *= np.float32(64.0)
    full2 = G8.lssab8_attention(q, k2, v2)
    part2 = G8.lssab8_attention(q[:, :500], k2[:, :500], v2[:, :500])
    return check("S8 non-aligned cut differs (block-aligned rule is REQUIRED)",
                 not np.array_equal(bits(part2), bits(full2[:, :500])),
                 "documents why the T13B chunk-alignment hook is mandatory")


def s9_decode():
    """A decode row (the last row of a T-length prefill) equals the same row computed by
    the split-KV segment structure a decode kernel uses."""
    ok = True
    for T in (256, 512, 1024, 4224):
        g = np.random.RandomState(T)
        q = g.standard_normal((2, T, 128)).astype(np.float32)
        k = g.standard_normal((1, T, 128)).astype(np.float32)
        v = g.standard_normal((1, T, 128)).astype(np.float32)
        p = G8.lssab_quantize(q, k, v, None, KB, "row")
        o = G8.lssab8_core(p["q8"], p["k8"], p["v8"], p["eq"], p["ek"], p["ev"], rows=[T - 1])
        r, _, _, _ = G8.lssab8_row_segments(p["q8"], p["k8"], p["v8"], p["eq"], p["ek"],
                                            p["ev"], 0, T - 1)
        if not np.array_equal(bits(r), bits(o[0, T - 1])):
            ok = False
            print("        decode row differs at T=%d" % T)
    return check("S9 decode row == prefill row (split-KV structure)", ok)


def s10_schedule():
    g = np.random.RandomState(77)
    T = 1024
    q = g.standard_normal((4, T, 128)).astype(np.float32)
    k = g.standard_normal((2, T, 128)).astype(np.float32)
    v = g.standard_normal((2, T, 128)).astype(np.float32)
    ref = torch_out(q, k, v, qblk=512, kblk=1024, head_chunk=4)
    ok = True
    for (qb, kbk, hcs) in ((256, 512, 1), (1024, 4096, 2), (128, 128, 4), (1 << 20, 1 << 20, 3)):
        o = torch_out(q, k, v, qblk=qb, kblk=kbk, head_chunk=hcs)
        if not np.array_equal(bits(o), bits(ref)):
            ok = False
            print("        schedule qblk=%d kblk=%d hc=%d differs" % (qb, kbk, hcs))
    return check("S10 schedule invariance (qblk / kblk / head_chunk)", ok)


def main():
    print("=" * 96)
    print("LSSA-B8 SELF-TEST  (torch %s, numpy %s)  B=%d KB=%d SEG=%d S_WIN=%d W_LOC=%d n=%d"
          % (torch.__version__, np.__version__, BHI, KB, SEG, S_WIN, W_LOC, NIDX))
    print("=" * 96)
    print("S0 T2 table family")
    s0_table()
    print("S1 the contract rescale (exponent-field subtraction with flush)")
    s1_rescale()
    rc = random_cases()
    jc = jump_cases()
    fc = ftz_cases()
    dc = degenerate_cases()
    print("S2-S5 torch == golden, bit-for-bit")
    run_cmp(rc, "S2 random q/k/v, T=256..2048, GQA")
    run_cmp(jc, "S3 200-doubling jumps between blocks")
    run_cmp(fc, "S4 FTZ boundary / e_v ladders")
    run_cmp(dc, "S5 degenerate rows")
    print("S6-S10 invariances")
    s6_skip(rc + jc)
    s7_segsplit()
    s8_chunk()
    s9_decode()
    s10_schedule()
    print("S11 every arm")
    small = [c for c in rc if c["q"].shape[1] in (256, 320, 512)][:6] + dc[:6]
    for arm in ("L8-RN-W9", "L8-RN-W8", "L8-TR", "L8-RN-n8", "L8-RN-KB64", "L16"):
        run_cmp(small, "S11 arm", arm=arm)
    print("=" * 96)
    print("LSSA-B8 SELFTEST: %d/%d checks PASS%s"
          % (N_PASS, N_PASS + N_FAIL, "" if N_FAIL == 0 else ("  FAILED: %s" % FAILS)))
    print("=" * 96)
    return 0 if N_FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
