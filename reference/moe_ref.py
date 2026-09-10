"""Contract v3 (DRAFT, SPEC 7.27): the torch REFERENCE of a mixture-of-experts layer under the contract.

This file is the definition. Every operation is an integer operation or a named binary32 / bf16
rounding, so the same function evaluated on a CPU or any GPU produces the same bits. It reuses the
dense contract's pieces: the per-row int8 quantisation (rowquant), the v2.1 linear epilogue, the SiLU
table, and the attention rule's exponential table (contract_common.table_on).

    y = moe_forward(x, gate, experts, cfg)

x: bf16 [T, H]. gate: (w8 int8 [E, H] column-major as [H, E], ws fp32 [E]). experts: per expert e,
(w13_8 int8 [2I, H] as [H, 2I], ws13 fp32 [2I], w2_8 int8 [H, I] as [I, H], ws2 fp32 [H]).
cfg: E, k, renormalize, and the table (n, w).
"""
from __future__ import annotations
import torch
from lockstep_fullstack import fwht_blocks

BF = torch.bfloat16
F32 = torch.float32


def bfr(x: torch.Tensor) -> torch.Tensor:
    """fl32 -> bf16 -> fl32: one bf16 rounding, kept in binary32 for the next operation."""
    return x.to(BF).to(F32)


def rowquant(x: torch.Tensor):
    """Contract v2.1 per-row int8 quantisation: asc = amax / 127 (binary32), q = rint(x / asc) clamped."""
    xf = x.to(F32)
    am = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    # the divisor is a TENSOR: PyTorch's CUDA division by a python scalar multiplies by the reciprocal
    # (the dense stack's `_const`, lockstep_fullstack): a scalar 127.0 here gave the gate logits one bf16
    # ulp of GPU-against-CPU difference on 915 real tokens (M0, 2026-09-10)
    d127 = torch.tensor(127.0, dtype=F32, device=xf.device)
    asc = am / d127
    q = torch.clamp(torch.round(xf / asc), -127, 127).to(torch.int8)
    return q, asc.squeeze(-1)


def lin_epilogue(acc: torch.Tensor, ws: torch.Tensor, asc: torch.Tensor) -> torch.Tensor:
    """y8 = bf16( (fl32(acc) * ws[n]) * asc[m] ) - the declared order (weight scale first)."""
    return bfr((acc.to(F32) * ws[None, :].to(F32)) * asc[:, None].to(F32))


INT8_MIN_ROWS = 17   # torch._int_mm needs more than 16 rows; zero rows do not change a single product


def int_mm(a8: torch.Tensor, b8: torch.Tensor) -> torch.Tensor:
    """int8 x int8 -> int32, exact (int64 on CPU narrowed; torch._int_mm on CUDA, rows zero-padded to 17)."""
    if a8.is_cuda:
        n = a8.shape[0]
        if n < INT8_MIN_ROWS:
            a8 = torch.cat([a8, torch.zeros(INT8_MIN_ROWS - n, a8.shape[1], dtype=torch.int8, device=a8.device)])
        return torch._int_mm(a8, b8)[:n]
    return (a8.to(torch.int64) @ b8.to(torch.int64)).to(torch.int32)


# The attention rule's exponential (contract_common.make_contract_attention): d is a log2 distance on a
# 2^BHI grid; the weight is frac[(d >> (BHI - N)) & (2^N - 1)] >> (d >> BHI), zero beyond (W + 1) doublings.
BHI, TAB_N, TAB_W = 26, 13, 15
ROUTER_C = float(torch.tensor(2.0 ** BHI, dtype=torch.float64).item() / 0.6931471805599453)   # 2^26 / ln 2, as a binary32 constant below


def table_scores(dgap: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
    """dgap fp32 >= 0 (m - g, exact in binary32 for bf16 inputs) -> integer score = the table exponential of
    exp(-dgap): d = rint(fl32(dgap) * fl32(2^26 / ln 2)) (one binary32 multiply, rounded once), then the
    attention rule's lookup. Integers out; no libm."""
    c = torch.tensor(ROUTER_C, dtype=F32, device=dgap.device)
    d = torch.round(dgap.to(F32) * c).to(torch.int64).clamp(min=0)
    win = (TAB_W + 1) << BHI
    idx = (d >> (BHI - TAB_N)) & ((1 << TAB_N) - 1)
    sh = torch.minimum(d >> BHI, torch.full_like(d, 63))
    s = tab[idx] >> sh
    return torch.where(d < win, s, torch.zeros_like(s))


def router(x: torch.Tensor, gate_w8: torch.Tensor, gate_ws: torch.Tensor, k: int, renormalize: bool,
           tab: torch.Tensor, win: int = 0):
    """SPEC 7.27 steps 1-4. Returns (idx int64 [T, k] by score rank, w bf16 [T, k], g bf16 [T, E])."""
    q, asc = rowquant(x)
    g = lin_epilogue(int_mm(q, gate_w8), gate_ws, asc)                      # [T, E] bf16 values in fp32
    return router_from_logits(g, k, renormalize, tab, win)


def router_from_logits(g: torch.Tensor, k: int, renormalize: bool, tab: torch.Tensor, win: int = 0):
    """SPEC 7.27 steps 2-4 from the bf16 gate logits g [T, E] (bf16 values, any float dtype)."""
    g = g.to(F32)
    m = g.amax(dim=-1, keepdim=True)
    s = table_scores(m - g, tab)                                             # integer scores, the max logit -> the largest
    T, E = s.shape
    key = s * E + (E - 1 - torch.arange(E, device=s.device, dtype=torch.int64))[None, :]   # ties: the lower index wins
    order = torch.argsort(key, dim=-1, descending=True)[:, :k]
    s_sel = torch.gather(s, 1, order)
    S = s_sel.sum(dim=-1, keepdim=True) if renormalize else s.sum(dim=-1, keepdim=True)
    S = S.clamp(min=1)
    w = (s_sel.to(F32) / S.to(F32)).to(BF)                                   # one binary32 division, one rounding
    return order, w, g.to(BF)


def expert_swiglu(x_rows: torch.Tensor, w13_8: torch.Tensor, ws13: torch.Tensor, w2_8: torch.Tensor, ws2: torch.Tensor,
                  silu_fn, rot: bool = False) -> torch.Tensor:
    """Contract v2.5 SwiGLU block on the rows routed to one expert. rot=True: the dense stack's online
    rotations - R1 (the 256-block Sylvester Hadamard / 16, exact binary32 adds: lockstep_fullstack.fwht_blocks)
    on the block input before its row quant, R4 (the same transform) on the SiLU output before the down
    quant - with w13 / w2 folded by the same block rotation on their input columns at arm time."""
    xin = fwht_blocks(x_rows.to(F32)) if rot else x_rows
    q, asc = rowquant(xin)
    y = lin_epilogue(int_mm(q, w13_8), ws13, asc)                              # [t, 2I] bf16 values
    I = y.shape[-1] // 2
    gate, up = y[:, :I], y[:, I:]
    h = silu_fn(gate, up)                                                      # the declared SiLU table * up, bf16
    hin = fwht_blocks(h.to(F32)) if rot else h
    q2, asc2 = rowquant(hin)
    return lin_epilogue(int_mm(q2, w2_8), ws2, asc2)                           # [t, H] bf16 values in fp32


def moe_forward(x: torch.Tensor, gate, experts, k: int, renormalize: bool, tab: torch.Tensor, win: int, silu_fn,
                shared=None, routing=None, rot: bool = False):
    """SPEC 7.27 steps 1-6. x bf16 [T, H] -> y bf16 [T, H]; also returns the routing (idx, w).
    routing=(idx, w) injects a routing computed elsewhere (the attribution runs: stock's router with the
    contract's experts); the declared router is the default."""
    if routing is None:
        idx, w, _ = router(x, gate[0], gate[1], k, renormalize, tab, win)
    else:
        idx, w = routing[0].to(torch.int64), routing[1].to(BF)
    T, H = x.shape
    acc = torch.zeros(T, H, dtype=F32, device=x.device)
    E = len(experts)
    # combine in ASCENDING expert index: for each expert, the tokens that selected it, in token order
    for e in range(E):
        hit = (idx == e)
        if not hit.any():
            continue
        tok = hit.any(dim=-1).nonzero(as_tuple=True)[0]
        we = w[tok][hit[tok]].to(F32)                                          # the token's weight for expert e
        out = expert_swiglu(x[tok], *experts[e], silu_fn, rot=rot)
        acc[tok] += bfr(we[:, None] * out)                                    # bf16( w * out ), then fp32 accumulate
    if shared is not None:
        acc += bfr(shared(x).to(F32))
    return acc.to(BF), idx, w


# ============================================================================ contract v3 addendum: the GPT-OSS family
# (SPEC 7.27 addendum, draft 2026-09-10): biased gate and expert linears, the clamped SwiGLU with the alpha-scaled
# sigmoid table, gate / up de-interleaved at fold time, the R4 block a parameter (256 / 64 / none).
SWIGLU_ALPHA = torch.tensor(1.702, dtype=F32)      # binary32 literal
SWIGLU_LIMIT = 7.0


def sigmoid_contract(x: torch.Tensor) -> torch.Tensor:
    """SPEC 6.3.1 steps (1)-(9): the invariant sigmoid (silu_contract without its final multiply). x f32 -> f32."""
    from lockstep_fullstack import frac_table as _ft, SILU_C, F32_MIN_NORMAL, pow2_f32
    frac = _ft(x.device)
    xc = torch.where(x.abs() < F32_MIN_NORMAL, torch.zeros_like(x), x)
    t = torch.clamp(xc * SILU_C, -30.0, 30.0)
    i = torch.ceil(t)
    g = i - t
    k = torch.clamp(torch.trunc(g * 8192.0), max=8191.0).to(torch.int64)
    f = frac[k].to(F32) * (2.0 ** -15)
    e2 = f * pow2_f32(i.to(torch.int64))
    return 1.0 / (1.0 + e2)


def lin_epilogue_b(acc: torch.Tensor, ws: torch.Tensor, asc: torch.Tensor, bias=None) -> torch.Tensor:
    """The declared epilogue with an optional bias: bfr( bfr(fl32(acc) * ws * asc) + b ) - the dense deq1s order."""
    y = lin_epilogue(acc, ws, asc)
    if bias is not None:
        y = bfr(y + bias.to(F32))
    return y


def act_swigluoai(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """GPT-OSS: gc = min(g, 7); uc = clamp(u, -7, 7); glu = bfr(gc * sig(bfr(1.702 * gc))); h = bfr((uc + 1) * glu).
    gate / up already de-interleaved (contiguous halves); bf16 values in f32 in, bf16 values in f32 out."""
    g = gate.to(F32); u = up.to(F32)
    gc = torch.clamp(g, max=SWIGLU_LIMIT)
    uc = torch.clamp(u, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    arg = bfr(gc * SWIGLU_ALPHA.to(g.device))
    glu = bfr(gc * sigmoid_contract(arg))
    return bfr((uc + 1.0) * glu)


def act_silu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Contract v2.5 SwiGLU: bfr( silu_table(g) * u )."""
    from lockstep_fullstack import silu_contract
    return bfr(silu_contract(gate.to(F32)) * up.to(F32))


def expert_block(x_rows: torch.Tensor, w13_8, ws13, w2_8, ws2, act, b13=None, b2=None, rot_blk: int = 256) -> torch.Tensor:
    """The expert block with biases and a parametrised R1 / R4 block (rot_blk 256: the v2.5 rotation, 64: the
    64-block Sylvester / 8, 0: no rotation). Weights folded with the same block rotation on their input columns."""
    def rot(v):
        if not rot_blk: return v
        return fwht_blocks(v.to(F32), blk=rot_blk, scale=1.0 / (rot_blk ** 0.5))
    q, asc = rowquant(rot(x_rows))
    y = lin_epilogue_b(int_mm(q, w13_8), ws13, asc, b13)
    I = y.shape[-1] // 2
    h = act(y[:, :I], y[:, I:])
    q2, asc2 = rowquant(rot(h))
    return lin_epilogue_b(int_mm(q2, w2_8), ws2, asc2, b2)


def router_b(x, gate_w8, gate_ws, k, renormalize, tab, win, gate_bias=None):
    """The SPEC 7.27 router with an optional bias on the gate logits (GPT-OSS): the biased epilogue, then the table
    scores and top-k exactly as router()."""
    if gate_bias is None:
        return router(x, gate_w8, gate_ws, k, renormalize, tab, win)
    q, asc = rowquant(x)
    g = lin_epilogue_b(int_mm(q, gate_w8), gate_ws, asc, gate_bias).to(BF)
    return router_from_logits(g, k, renormalize, tab, win)


def moe_forward_b(x, gate, experts, k, renormalize, tab, win, act, rot_blk=256, routing=None):
    """moe_forward for the addendum: gate = (w8col, ws, bias | None); experts[e] = (w13col, ws13, w2col, ws2, b13 | None,
    b2 | None); act = act_silu | act_swigluoai; rot_blk the R1 / R4 block."""
    if routing is None:
        idx, w, _ = router_b(x, gate[0], gate[1], k, renormalize, tab, win, gate[2] if len(gate) > 2 else None)
    else:
        idx, w = routing[0].to(torch.int64), routing[1].to(BF)
    T, H = x.shape
    acc = torch.zeros(T, H, dtype=F32, device=x.device)
    for e in range(len(experts)):
        hit = (idx == e)
        if not hit.any(): continue
        tok = hit.any(dim=-1).nonzero(as_tuple=True)[0]
        we = w[tok][hit[tok]].to(F32)
        ex = experts[e]
        out = expert_block(x[tok], ex[0], ex[1], ex[2], ex[3], act, ex[4] if len(ex) > 4 else None, ex[5] if len(ex) > 5 else None, rot_blk)
        acc[tok] += bfr(we[:, None] * out)
    return acc.to(BF), idx, w


# ============================================================================ addendum item 5: the MXFP4-consuming operand
# The expert weight stays the checkpoint's MXFP4: per output row n and 32-wide K-block b, doubled mantissas m2 in
# {0, +-1, +-2, +-3, +-4, +-6, +-8, +-12} (int8) and an exponent k (E8M0). Declared evaluation of acc[m, n] for an int8 row a:
#   s_b = sum_{j in b} a[m, j] * m2[n, j]                     exact int32 (|s| <= 32 * 127 * 12)
#   acc = fl32( acc + fl32(s_b) * 2^(k[n, b] - 128) )           in ASCENDING b: the product is exact (a power of two), so the
#                                                              add is one correctly rounded binary32 operation per block
# then the epilogue y = bf16( acc * asc[m] ) (+ bias). No rotation (the weights cannot be folded).
MX_BLOCK = 32


def int_mm_mx(a8: torch.Tensor, m2: torch.Tensor, kexp: torch.Tensor) -> torch.Tensor:
    """a8 int8 [T, K]; m2 int8 [N, K] doubled mantissas; kexp uint8/int [N, K/32] E8M0 -> fp32 [T, N] (the declared acc)."""
    T, K = a8.shape; N = m2.shape[0]; nb = K // MX_BLOCK
    scale = torch.ldexp(torch.ones(N, nb, dtype=F32, device=a8.device), kexp.to(torch.int32) - 128)   # exact powers of two
    acc = torch.zeros(T, N, dtype=F32, device=a8.device)
    for b in range(nb):
        ab = a8[:, b * MX_BLOCK:(b + 1) * MX_BLOCK]; mb = m2[:, b * MX_BLOCK:(b + 1) * MX_BLOCK]
        if a8.is_cuda:
            s = (ab.to(torch.float64) @ mb.to(torch.float64).t()).to(F32)              # exact: |s| < 2^16
        else:
            s = (ab.to(torch.int64) @ mb.to(torch.int64).t()).to(F32)
        acc = acc + s * scale[:, b][None, :]                                            # one binary32 rounding per block
    return acc


def lin_epilogue_mx(acc: torch.Tensor, asc: torch.Tensor, bias=None) -> torch.Tensor:
    y = bfr(acc * asc[:, None].to(F32))
    if bias is not None: y = bfr(y + bias.to(F32))
    return y


def expert_block_mx(x_rows, w13, w2, act, b13=None, b2=None):
    """w13 = (m2 [2I, H], kexp [2I, H/32]); w2 = (m2 [H, I], kexp [H, I/32]). No rotation."""
    q, asc = rowquant(x_rows)
    y = lin_epilogue_mx(int_mm_mx(q, w13[0], w13[1]), asc, b13)
    I = y.shape[-1] // 2
    h = act(y[:, :I], y[:, I:])
    q2, asc2 = rowquant(h)
    return lin_epilogue_mx(int_mm_mx(q2, w2[0], w2[1]), asc2, b2)


def moe_forward_mx(x, gate, experts, k, renormalize, tab, win, act, routing=None):
    """experts[e] = ((m2_13, kexp_13), (m2_2, kexp_2), b13 | None, b2 | None); gate as moe_forward_b."""
    if routing is None:
        idx, w, _ = router_b(x, gate[0], gate[1], k, renormalize, tab, win, gate[2] if len(gate) > 2 else None)
    else:
        idx, w = routing[0].to(torch.int64), routing[1].to(BF)
    T, H = x.shape
    acc = torch.zeros(T, H, dtype=F32, device=x.device)
    for e in range(len(experts)):
        hit = (idx == e)
        if not hit.any(): continue
        tok = hit.any(dim=-1).nonzero(as_tuple=True)[0]
        we = w[tok][hit[tok]].to(F32)
        ex = experts[e]
        out = expert_block_mx(x[tok], ex[0], ex[1], act, ex[2] if len(ex) > 2 else None, ex[3] if len(ex) > 3 else None)
        acc[tok] += bfr(we[:, None] * out)
    return acc.to(BF), idx, w


# ============================================================================ item 5b: block-scaled ACTIVATIONS (MX2)
# The activation row is quantised per 32-wide block to int8 with a power-of-two exponent: ka_b = ceil(log2(max|x_b| / 127))
# (floored at -126), q = rint(x / 2^ka_b) clamped; the block sum s_b = sum_j q[m, j] m2[n, j] is exact in int32 and the
# declared accumulate becomes acc = fl32( acc + fl32(s_b) * 2^(ka[m, b] + kw[n, b] - 128) ) in ascending b - the product is
# still a power-of-two scaling (exact), one rounding per block. No per-row scale; the epilogue is y = bf16(acc) (+ bias).
# Zero blocks get ka = -126 and q = 0.
def blockquant(x: torch.Tensor):
    """x bf16 [T, K] -> (q int8 [T, K], ka int32 [T, K/32]) - the block exponents as signed ints."""
    xf = x.to(F32); T, K = xf.shape; nb = K // MX_BLOCK
    xb = xf.reshape(T, nb, MX_BLOCK)
    am = xb.abs().amax(dim=-1)                                                    # [T, nb]
    ka = torch.where(am > 0, torch.ceil(torch.log2(am / 127.0)), torch.full_like(am, -126.0)).clamp(min=-126.0, max=127.0)
    # guard: 127 * 2^ka >= am must hold (ceil of the exact log2 can round down in fp32): bump where it does not
    bad = (127.0 * torch.ldexp(torch.ones_like(ka), ka.to(torch.int32))) < am
    ka = torch.where(bad, ka + 1.0, ka)
    sc = torch.ldexp(torch.ones_like(ka), ka.to(torch.int32))                      # 2^ka, exact
    q = torch.clamp(torch.round(xb / sc[..., None]), -127, 127).to(torch.int8).reshape(T, K)
    return q, ka.to(torch.int32)


def int_mm_mx2(q: torch.Tensor, ka: torch.Tensor, m2: torch.Tensor, kexp: torch.Tensor) -> torch.Tensor:
    """q int8 [T, K], ka int32 [T, nb]; m2 int8 [N, K], kexp uint8 [N, nb] -> fp32 [T, N]."""
    T, K = q.shape; N = m2.shape[0]; nb = K // MX_BLOCK
    acc = torch.zeros(T, N, dtype=F32, device=q.device)
    for b in range(nb):
        qb = q[:, b * MX_BLOCK:(b + 1) * MX_BLOCK]; mb = m2[:, b * MX_BLOCK:(b + 1) * MX_BLOCK]
        s = (qb.to(torch.float64) @ mb.to(torch.float64).t()).to(F32) if q.is_cuda else (qb.to(torch.int64) @ mb.to(torch.int64).t()).to(F32)
        e = ka[:, b][:, None] + (kexp[:, b].to(torch.int32) - 128)[None, :]         # [T, N] exponents
        acc = acc + s * torch.ldexp(torch.ones_like(s), e)                             # exact product, one rounding
    return acc


def expert_block_mx2(x_rows, w13, w2, act, b13=None, b2=None):
    q, ka = blockquant(x_rows)
    y = int_mm_mx2(q, ka, w13[0], w13[1]); y = bfr(y)
    if b13 is not None: y = bfr(y + b13.to(F32))
    I = y.shape[-1] // 2
    h = act(y[:, :I], y[:, I:])
    q2, ka2 = blockquant(h.to(BF))
    o = bfr(int_mm_mx2(q2, ka2, w2[0], w2[1]))
    if b2 is not None: o = bfr(o + b2.to(F32))
    return o


def moe_forward_mx2(x, gate, experts, k, renormalize, tab, win, act, routing=None):
    if routing is None:
        idx, w, _ = router_b(x, gate[0], gate[1], k, renormalize, tab, win, gate[2] if len(gate) > 2 else None)
    else:
        idx, w = routing[0].to(torch.int64), routing[1].to(BF)
    T, H = x.shape
    acc = torch.zeros(T, H, dtype=F32, device=x.device)
    for e in range(len(experts)):
        hit = (idx == e)
        if not hit.any(): continue
        tok = hit.any(dim=-1).nonzero(as_tuple=True)[0]
        we = w[tok][hit[tok]].to(F32)
        ex = experts[e]
        out = expert_block_mx2(x[tok], ex[0], ex[1], act, ex[2] if len(ex) > 2 else None, ex[3] if len(ex) > 3 else None)
        acc[tok] += bfr(we[:, None] * out)
    return acc.to(BF), idx, w
