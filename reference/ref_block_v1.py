# M-T2 GOLDEN REFERENCE - executable contract spec for one transformer block, fwd+bwd+Adam.
# The CUDA and Metal implementations must match this BIT-FOR-BIT.
# Contract rules: fp32 scalar-IEEE ops in DECLARED order (no fma anywhere); all reductions =
# in-place binary tree (pow2 lengths); all quant scales = powers of two (exact muls); integer
# GEMMs exact; backward GEMMs fp32 with sequential-k accumulation; STE through quantizers;
# runtime-opaque seed (no constant folding possible on GPU ports).
import numpy as np
import sys

F = np.float32
import os
if os.environ.get("DEAI_MT4") == "1":
    T, D, H, DH, DF = 256, 1024, 16, 64, 4096
    if os.environ.get("DEAI_T1024") == "1":
        T = 1024
else:
    T, D, H, DH, DF = 64, 128, 4, 32, 512   # seq, d_model, heads, d_head, d_ff (all pow2)
LOG2D = D.bit_length() - 1
SWIGLU = os.environ.get("DEAI_SWIGLU") == "1"
V3Q = os.environ.get("DEAI_V3Q") == "1"
V3QC = os.environ.get("DEAI_V3QC") == "1"
V4M = os.environ.get("DEAI_V4M") == "1"    # mixed tier: dgrad int8, wgrad fp32
V4 = (os.environ.get("DEAI_V4") == "1") or V4M
V3QB = (os.environ.get("DEAI_V3QB") == "1") or V3QC
LOG2DFG = (DF // 2).bit_length() - 1                     # 11 for DFG=2048

def bgemm(a, b):
    """Backward GEMM: fp32 declared seq-k (<=v3) or int8 dyn-pow2 + exact int GEMM (V4)."""
    if not V4:
        return gemm_seqk(a, b)
    qa, ea = quant8_dyn(a)
    qb, eb = quant8_dyn(b)
    return (gemm_int(qa, qb).astype(F) * F(2.0**(-ea-eb))).astype(F)

def wgemm(a, b):
    """Weight-grad GEMM: stays declared fp32 under V4M (quality tier); int8 under V4-full."""
    return gemm_seqk(a, b) if V4M else bgemm(a, b)

def g_eff_int(base_sh, ea, eb):
    sh = base_sh - ea - eb
    sh = max(-40, min(20, sh))
    G = (586 << sh) if sh >= 0 else ((586 + (1 << (-sh - 1))) >> (-sh))
    return max(1, min(G, 1 << 30))

def fwht_rows(x):      # unnormalized Walsh-Hadamard over last axis (pow2), RNE adds/subs, declared order
    a = x.astype(F).copy()
    n = a.shape[-1]; ln = 1
    while ln < n:
        for i in range(0, n, 2 * ln):
            u = a[..., i:i+ln].copy(); v = a[..., i+ln:i+2*ln].copy()
            a[..., i:i+ln] = (u + v).astype(F)
            a[..., i+ln:i+2*ln] = (u - v).astype(F)
        ln *= 2
    return a

def _e_from_max(m):    # vectorized clip-safe pow2 exponent (same formula as quant8_dyn)
    m = m.astype(F)
    e = 6 - ((m.view(np.uint32).astype(np.int64) >> 23) - 127)
    e = np.where(m == 0, 0, e)
    over = (m * np.ldexp(np.float32(1.0), e).astype(F)).astype(F) > F(127.0)
    e = np.where(over, e - 1, e)
    return np.clip(e, -32, 40)

def quant8_rows(x):    # per-row (token) pow2 scales
    e = _e_from_max(np.max(np.abs(x), axis=1))
    q = np.clip(np.rint((x * np.ldexp(np.float32(1.0), e)[:, None].astype(F)).astype(F)), -127, 127).astype(np.int64)
    return q, e

def quant8_cols(w):    # per-column (channel) pow2 scales
    e = _e_from_max(np.max(np.abs(w), axis=0))
    q = np.clip(np.rint((w * np.ldexp(np.float32(1.0), e)[None, :].astype(F)).astype(F)), -127, 127).astype(np.int64)
    return q, e
DFG = DF // 2                                            # gate width (pow2)

def silu_det(x):
    """Deterministic SiLU: sigma via FRAC-LUT exp2 (no libm/SFU). Declared IEEE op order.
       t=-x*log2e (RNE); clip; i=ceil; g=i-t in [0,1); 2^-g via FRAC (trunc index); 2^t=ldexp."""
    c = F(-1.4426950408889634)
    t = np.clip((x * c).astype(F), F(-30.0), F(30.0))
    i = np.ceil(t).astype(np.int64)
    g = (i.astype(F) - t).astype(F)
    k = np.minimum((g * F(8192.0)).astype(F).astype(np.int64), 8191)
    frac = (FRAC[k].astype(F) * F(2.0**-15)).astype(F)
    twot = np.ldexp(frac, i).astype(F)                   # exact pow2 scaling
    sig = (F(1.0) / (F(1.0) + twot).astype(F)).astype(F)
    return (x * sig).astype(F), sig
EPS = F(1e-6)
QE = 5                                   # quant exponent: scale 2^5=32 (pow2 -> exact)
G_, BHI, BLUT = 586, 20, 13              # block-float softmax constants (proven contract)

def sm32(x):
    x = (x + 0x9E3779B9) & 0xFFFFFFFF; x ^= x >> 16
    x = (x * 0x21F0AAAD) & 0xFFFFFFFF; x ^= x >> 15
    x = (x * 0x735A2D97) & 0xFFFFFFFF; x ^= x >> 15
    return x
def rr(k, n):
    return np.array([sm32((k*0x9E3779B9 + i*0x85EBCA6B + 0xABCD1234) & 0xFFFFFFFF)
                     for i in range(n)], dtype=np.uint32)
def u2f(u):   # (float)(int)u * 2^-31  (I2F RNE, pow2 mul exact)
    return (u.astype(np.int32).astype(F) * F(2.0**-31)).astype(F)

# FRAC table via the frozen deg-6 Q30 int64 poly (platform-identical integer computation)
_C = [1073741823, -744260975, 257938938, -59583045, 10286509, -1369790, 117454]
def build_frac():
    f = np.arange(8192, dtype=np.int64); t = f << 7
    p = np.full_like(f, _C[6])
    for k in range(5, -1, -1): p = _C[k] + ((p * t) >> 20)
    return (p + (1 << 14)) >> 15
FRAC = build_frac()

def tree_sum(a):    # in-place binary tree over last axis (pow2 length), fp32 RNE per add
    a = a.astype(F).copy(); n = a.shape[-1]; s = 1
    while s < n:
        idx = np.arange(0, n, 2*s)
        a[..., idx] = (a[..., idx] + a[..., idx+s]).astype(F)
        s *= 2
    return a[..., 0]

def quant8(x, e=QE):   # rint(x*2^e) clamp - pow2 scale, RNE
    return np.clip(np.rint((x * F(2.0**e)).astype(F)), -127, 127).astype(np.int64)

def quant8_dyn(x):     # dynamic per-tensor pow2: largest e with m*2^e <= 127 (clip-safe, MXFP8 lesson)
    m = F(np.max(np.abs(x)))                      # associative max -> deterministic any order
    if m == 0:
        return np.zeros_like(x, dtype=np.int64), 0
    e = 6 - ((np.float32(m).view(np.uint32) >> 23).astype(np.int64) - 127)
    if F(m * F(2.0**e)) > F(127.0):               # exact fp compare; avoid silent clip of max
        e -= 1
    e = int(max(-32, min(40, e)))
    return np.clip(np.rint((x * F(2.0**e)).astype(F)), -127, 127).astype(np.int64), e

def gemm_int(a, b):    # exact integer GEMM (int64; GPU uses int32, bounds verified)
    return a @ b

def gemm_seqk(a, b):   # fp32 GEMM, sequential-k declared order: acc = RNE(acc + RNE(a_k*b_k))
    acc = np.zeros((a.shape[0], b.shape[1]), dtype=F)
    for k in range(a.shape[1]):
        acc = (acc + (a[:, k:k+1] * b[k:k+1, :]).astype(F)).astype(F)
    return acc

def rmsnorm_fwd(x, g):
    ss = tree_sum((x * x).astype(F))                     # [T]
    ms = (ss * F(2.0**-LOG2D)).astype(F)                     # /D exact (D=128)
    r = np.sqrt((ms + EPS).astype(F)).astype(F)          # CR sqrt
    xh = (x / r[:, None]).astype(F)                      # CR div
    return (xh * g[None, :]).astype(F), r, xh

def rmsnorm_bwd(dy, g, r, xh):
    dxh = (dy * g[None, :]).astype(F)
    dg = tree_sum((dy * xh).astype(F).T)                 # [D] tree over T
    dot = tree_sum((dxh * xh).astype(F))                 # [T]
    t1 = (xh * ((dot * F(2.0**-LOG2D)).astype(F))[:, None]).astype(F)
    dx = ((dxh - t1).astype(F) / r[:, None]).astype(F)
    return dx, dg

def softmax_bf(S, G=G_):   # block-float softmax on int scores (G adjustable for dyn logit scales)
    m = S.max(axis=-1)                                   # int max (associative)
    NB = -((-(m.astype(np.int64) * G)) >> BHI) << BHI    # ceil-shift, per row (int64)
    d = NB[:, None].astype(np.int64) - S.astype(np.int64) * G               # >= 0 (int64: MT4 logits range)
    eh = np.where(d < (16 << BHI), FRAC[(d >> (BHI-BLUT)) & 8191] >> (d >> BHI), 0)
    l = eh.sum(axis=-1)                                  # int sum exact
    p = (eh.astype(F) / l.astype(F)[:, None]).astype(F)  # I2F + CR div (l<2^22, eh<2^16: exact I2F)
    return p, eh, l

def attn_fwd(x, Wqkv, ewq):
    q8, ex = quant8_dyn(x)
    qkv = (gemm_int(q8, Wqkv).astype(F) * F(2.0**(-ex-ewq))).astype(F)   # dyn pow2 deq
    Q = qkv[:, 0*D:1*D].reshape(T, H, DH); K = qkv[:, 1*D:2*D].reshape(T, H, DH)
    V = qkv[:, 2*D:3*D].reshape(T, H, DH)
    O = np.zeros((T, H, DH), dtype=F); Ps = []
    for h in range(H):
        if V3QC:
            Qq, eq = quant8_dyn(Q[:, h]); Kq, ek = quant8_dyn(K[:, h]); Vq, ev = quant8_dyn(V[:, h])
        else:
            Qq = quant8(Q[:, h]); Kq = quant8(K[:, h]); Vq = quant8(V[:, h]); eq = ek = ev = QE
        S = gemm_int(Qq, Kq.T)                                     # int scores
        mask = np.tril(np.ones((T, T), dtype=bool))
        S = np.where(mask, S, -(1 << 20))                          # causal: huge-negative -> ehat 0
        p, eh, l = softmax_bf(S, g_eff_int(10, eq, ek) if V3QC else G_)
        A = gemm_int(eh, Vq)                                       # int PV (<2^31)
        of = (A.astype(F) / l.astype(F)[:, None]).astype(F)        # RNE I2F(A) (A<2^29: rounds, ok) + CR div
        O[:, h] = (of * np.ldexp(np.float32(1.0), -ev).astype(F)).astype(F)
        Ps.append((p, Qq, Kq, Vq, Q[:, h], K[:, h], V[:, h], eq, ek, ev))
    return O.reshape(T, D), Ps, (q8, ex)

def block_fwd(x, params):
    g1, Wqkv, Wo, g2, Wup, Wdn = params
    h1, r1, xh1 = rmsnorm_fwd(x, g1)
    Wqkv8, ewq = quant8_dyn(Wqkv)
    ao_pre, Ps, q8x = attn_fwd(h1, Wqkv8, ewq)
    ao8, eao = quant8_dyn(ao_pre)
    Wo8, ewo = quant8_dyn(Wo)
    ao = (gemm_int(ao8, Wo8).astype(F) * F(2.0**(-eao-ewo))).astype(F)
    x2 = (x + ao).astype(F)
    h2, r2, xh2 = rmsnorm_fwd(x2, g2)
    u8, eu = quant8_dyn(h2)
    Wup8, ewu = quant8_dyn(Wup)
    up = (gemm_int(u8, Wup8).astype(F) * F(2.0**(-eu-ewu))).astype(F)
    if SWIGLU:
        gate = up[:, :DFG]; val = up[:, DFG:]
        sact, sig = silu_det(gate)
        rel = (sact * val).astype(F)
    else:
        gate = val = sig = None
        rel = np.maximum(up, F(0)).astype(F)
    if V3QB:
        rl8, er = quant8_dyn(fwht_rows(rel))             # rotate rows, per-tensor dyn scale
        Wdn8, ewd = quant8_dyn(fwht_rows(Wdn.T).T)       # rotate Wdn along DFG dim (cols of Wdn.T)
        dn = (gemm_int(rl8, Wdn8).astype(F) * F(2.0**(-er-ewd-LOG2DFG))).astype(F)
    elif V3Q:
        rl8, er = quant8_rows(rel)                       # er: [T]
        Wdn8, ewd = quant8_cols(Wdn)                     # ewd: [D]
        sc = np.ldexp(np.float32(1.0), -(er[:, None] + ewd[None, :])).astype(F)
        dn = (gemm_int(rl8, Wdn8).astype(F) * sc).astype(F)
    else:
        rl8, er = quant8_dyn(rel)
        Wdn8, ewd = quant8_dyn(Wdn)
        dn = (gemm_int(rl8, Wdn8).astype(F) * F(2.0**(-er-ewd))).astype(F)
    y = (x2 + dn).astype(F)
    E = dict(ewq=ewq, eao=eao, ewo=ewo, eu=eu, ewu=ewu, er=er, ewd=ewd,
             W8=(Wqkv8, Wo8, Wup8, Wdn8), ao8=ao8, rl8=rl8, u8=u8, glu=(gate, val, sig))
    cache = (h1, r1, xh1, Ps, ao_pre, x2, r2, xh2, up, rel, E, q8x)
    return y, cache

def block_bwd(dy, x, params, cache):
    g1, Wqkv, Wo, g2, Wup, Wdn = params
    h1, r1, xh1, Ps, ao_pre, x2, r2, xh2, up, rel, E, q8xT = cache
    Wqkv8, Wo8, Wup8, Wdn8 = E["W8"]
    Wqkv_f = (Wqkv8.astype(F) * F(2.0**-E["ewq"])).astype(F)       # dequantized dyn-scale weights
    Wo_f = (Wo8.astype(F) * F(2.0**-E["ewo"])).astype(F)
    Wup_f = (Wup8.astype(F) * F(2.0**-E["ewu"])).astype(F)
    if V3QB:
        Wdn_f = Wdn.astype(F)                            # STE vs master weights
    elif V3Q:
        Wdn_f = (Wdn8.astype(F) * np.ldexp(np.float32(1.0), -E["ewd"])[None, :].astype(F)).astype(F)
    else:
        Wdn_f = (Wdn8.astype(F) * F(2.0**-E["ewd"])).astype(F)
    h2f = (xh2 * g2[None, :]).astype(F)
    # FFN backward (STE through quants)
    ddn = dy
    dWdn = wgemm(rel.T, ddn)
    drel = bgemm(ddn, Wdn_f.T)
    if SWIGLU:
        gate, val, sig = E["glu"]
        sact = (gate * sig).astype(F)
        dval = (drel * sact).astype(F)
        t1 = (F(1.0) - sig).astype(F)
        t2 = (gate * t1).astype(F)
        t3 = (F(1.0) + t2).astype(F)
        dsilu = (sig * t3).astype(F)
        dgate = ((drel * val).astype(F) * dsilu).astype(F)
        dup = np.concatenate([dgate, dval], axis=1).astype(F)
    else:
        dup = (drel * (up > 0).astype(F)).astype(F)
    dWup = wgemm(h2f.T, dup)
    dh2 = bgemm(dup, Wup_f.T)
    dx2_a, dg2 = rmsnorm_bwd(dh2, g2, r2, xh2)
    dx2 = (dy + dx2_a).astype(F)
    # attention backward
    dao = dx2
    ao8_f = (E["ao8"].astype(F) * F(2.0**-E["eao"])).astype(F)
    dWo = wgemm(ao8_f.T, dao)
    dao_pre = bgemm(dao, Wo_f.T)
    dO = dao_pre.reshape(T, H, DH)
    dqkv = np.zeros((T, 3*D), dtype=F)
    for h in range(H):
        p, Qq, Kq, Vq, Qf, Kf, Vf = Ps[h][:7]
        dOf = dO[:, h]
        dV = bgemm(p.T, dOf)
        if V3QC:
            eq, ek, ev = Ps[h][7:]
            dp = bgemm(dOf, Vf.T)                              # STE vs master V
            rowdot = tree_sum((p * dp).astype(F))
            ds = (p * (dp - rowdot[:, None]).astype(F)).astype(F)
            dsn = (ds * np.ldexp(np.float32(1.0), eq + ek - 10).astype(F)).astype(F)
            dQ = bgemm(dsn, Kf)                                # STE vs masters
            dK = bgemm(dsn.T, Qf)
        else:
            Vq_f = (Vq.astype(F) * F(2.0**-QE)).astype(F)
            dp = bgemm(dOf, Vq_f.T)
            rowdot = tree_sum((p * dp).astype(F))                  # [T] tree
            ds = (p * (dp - rowdot[:, None]).astype(F)).astype(F)
            Kq_f = (Kq.astype(F) * F(2.0**-QE)).astype(F)
            Qq_f = (Qq.astype(F) * F(2.0**-QE)).astype(F)
            dQ = bgemm(ds, Kq_f)
            dK = bgemm(ds.T, Qq_f)
        dqkv[:, 0*D+h*DH:0*D+(h+1)*DH] = dQ
        dqkv[:, 1*D+h*DH:1*D+(h+1)*DH] = dK
        dqkv[:, 2*D+h*DH:2*D+(h+1)*DH] = dV
    q8v, exv = q8xT
    q8x_f = (q8v.astype(F) * F(2.0**-exv)).astype(F)
    dWqkv = wgemm(q8x_f.T, dqkv)
    dh1 = bgemm(dqkv, Wqkv_f.T)
    dx_a, dg1 = rmsnorm_bwd(dh1, g1, r1, xh1)
    dx = (dx2 + dx_a).astype(F)
    dbg = dict(dh2=dh2, dao_pre=dao_pre, dqkv=dqkv, dh1=dh1, dx2=dx2)
    return dx, [dg1, dWqkv, dWo, dg2, dWup, dWdn], dbg

def adam_step(params, grads, m, v):
    # declared: separate mul/add RNE ops, NO fma. lr=1e-3, b1=.9, b2=.999, eps=1e-8, no bias corr.
    out_p, out_m, out_v = [], [], []
    for p_, g_, m_, v_ in zip(params, grads, m, v):
        m2 = ((F(0.9)*m_).astype(F) + (F(0.1)*g_).astype(F)).astype(F)
        gg = (g_*g_).astype(F)
        v2 = ((F(0.999)*v_).astype(F) + (F(0.001)*gg).astype(F)).astype(F)
        den = (np.sqrt(v2).astype(F) + F(1e-8)).astype(F)
        upd = (F(0.001) * (m2/den).astype(F)).astype(F)
        out_p.append((p_ - upd).astype(F)); out_m.append(m2); out_v.append(v2)
    return out_p, out_m, out_v

def Hsh(x):
    b = np.ascontiguousarray(x, dtype=F).view(np.uint32).ravel().astype(np.uint64)
    i = np.arange(len(b), dtype=np.uint64)
    return int(((b * ((i*np.uint64(0x9E3779B97F4A7C15)) | np.uint64(1))).sum()) & np.uint64(0xFFFFFFFFFFFFFFFF))

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    x = u2f(rr(10+seed, T*D)).reshape(T, D)
    params = [ (u2f(rr(11+seed, D))*F(0.5)+F(1)).astype(F),          # g1
               (u2f(rr(12+seed, D*3*D))*F(0.05)).astype(F).reshape(D,3*D),
               (u2f(rr(13+seed, D*D))*F(0.05)).astype(F).reshape(D,D),
               (u2f(rr(14+seed, D))*F(0.5)+F(1)).astype(F),          # g2
               (u2f(rr(15+seed, D*DF))*F(0.05)).astype(F).reshape(D,DF),
               (u2f(rr(16+seed, DF*D))*F(0.05)).astype(F).reshape(DF,D) ]
    m0 = [np.zeros_like(p) for p in params]; v0 = [np.zeros_like(p) for p in params]
    y, cache = block_fwd(x, params)
    dy = u2f(rr(17+seed, T*D)).reshape(T, D)
    dx, grads, dbg = block_bwd(dy, x, params, cache)
    p2, m2, v2 = adam_step(params, grads, m0, v0)
    h1c, _, _, Psc, aoprec, x2c, _, _, upc, _ = cache
    print("MT2DBG h1=%016x p0=%016x aop=%016x x2=%016x up=%016x dh2=%016x daop=%016x dqkv=%016x dh1=%016x" % (
        Hsh(h1c), Hsh(Psc[0][0]), Hsh(aoprec), Hsh(x2c), Hsh(upc),
        Hsh(dbg["dh2"]), Hsh(dbg["dao_pre"]), Hsh(dbg["dqkv"]), Hsh(dbg["dh1"])))
    print("MT2REF y=%016x dx=%016x " % (Hsh(y), Hsh(dx)) +
          " ".join("g%d=%016x" % (i, Hsh(g)) for i, g in enumerate(grads)) + " " +
          " ".join("p%d=%016x" % (i, Hsh(p)) for i, p in enumerate(p2)))
