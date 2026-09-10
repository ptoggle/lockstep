#!/usr/bin/env python3
"""Published test vectors of contract v2.5's non-attention operators (docs/contract/NON_ATTENTION_STACK_v2.5.md).

Generates deterministic inputs (seeded, small shapes) and the reference outputs from the
pure-torch reference `lockstep_fullstack.py` on the CPU, and publishes a digest (sha256 of
the raw output bytes, dtype and shape in the preimage as in lockstep_manifest) per vector -
plus the first values in the clear, so an implementer can locate a divergence.

    python contract_vectors.py                 -> writes results/contract_vectors.json (repo layout) or ./contract_vectors.json
    python contract_vectors.py --check FILE    -> recompute here and compare every digest (exit 3 on any difference)

Anything that reproduces these digests on any hardware implements the same function; the
engine kernels are gated against the same reference (t16l_egates.py, t16m_gate.py).
"""
import hashlib
import json
import os
import sys

import torch

# Inputs come from numpy's PCG64 stream, whose output is stable across numpy versions and platforms by policy
# (NEP 19); torch.randn is not (torch 2.11 on x86 and 2.14 on arm64 return different values for one seed,
# found 2026-09-10 when the vectors were regenerated on a second machine).
import numpy as _np
class _Gen:
    def __init__(self, seed): self.g = _np.random.Generator(_np.random.PCG64(int(seed)))
    def randn(self, *shape): return torch.from_numpy(self.g.standard_normal(size=shape, dtype=_np.float64).astype(_np.float32))
    def rand(self, *shape): return torch.from_numpy(self.g.random(size=shape, dtype=_np.float64).astype(_np.float32))
def _randn(*shape, generator=None): return generator.randn(*shape)
def _rand(*shape, generator=None): return generator.rand(*shape)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lockstep_fullstack as F  # noqa: E402

VERSION = "contract-vectors-v2.6 (float norm)" if F.NORMF else "contract-vectors-v1 (v2.1)"


def digest(t: torch.Tensor) -> str:
    c = t.detach().to("cpu").contiguous().reshape(-1)
    h = hashlib.sha256()
    h.update(b"lockstep-tensor-v1|")
    h.update(str(t.dtype).replace("torch.", "").encode())
    h.update(b"|")
    h.update("x".join(str(int(d)) for d in t.shape).encode())
    h.update(b"|")
    h.update(memoryview(c.view(torch.uint8).numpy()) if c.numel() else b"")
    return h.hexdigest()


def head(t: torch.Tensor, n=6):
    f = t.detach().to("cpu").reshape(-1)[:n]
    if f.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return [float(v) for v in f.float()]
    return [int(v) for v in f]


def vectors():
    g = _Gen(20260906)
    out = {"version": VERSION, "torch": torch.__version__, "vectors": {}}

    def rec(name, t, note=""):
        out["vectors"][name] = {"dtype": str(t.dtype).replace("torch.", ""), "shape": [int(d) for d in t.shape],
                                "sha256": digest(t), "head": head(t), "note": note}

    # 1. activation quantisation (per token int8) and 2. weight quantisation
    x = (_randn(5, 512, generator=g) * 3).to(torch.bfloat16)
    q, asc = F.quantize_rows_int8(x.float())
    rec("act_quant.q", q, "clamp(rn_even(x / asc), +-127), asc = max(|x|, 1e-8) / 127 (one divide each)")
    rec("act_quant.asc", asc)
    W = (_randn(96, 512, generator=g) * 0.05).to(torch.bfloat16)
    W8, ws = F.quantize_weight_int8(W)
    rec("weight_quant.w8", W8); rec("weight_quant.ws", ws)
    # 3. the linear with the v2.1 epilogue, with and without a bias
    y = F.w8a8_matmul(x, W8, ws, None, out_dtype=torch.bfloat16)
    rec("linear.v2_1.no_bias", y, "bf16((fl32(acc) * ws[n]) * asc[m])")
    bias = (_randn(96, generator=g) * 0.1).float()
    yb = F.w8a8_matmul(x, W8, ws, bias, out_dtype=torch.bfloat16)
    rec("linear.v2_1.bias", yb, "bf16(fl32(y8) + bias[n]): a second rounding")
    # 3b. contract v2.5: the norm-fed linear under R1 online with declared outlier channels
    g5 = _Gen(25)
    x5 = (_randn(5, 512, generator=g5) * 3).to(torch.bfloat16)
    x5[:, 17] = x5[:, 17] * 400; x5[:, 300] = x5[:, 300] * 150       # two massive channels
    W5 = (_randn(96, 512, generator=g5) * 0.05).to(torch.bfloat16)
    oidx = torch.tensor([17, 300, 511], dtype=torch.int32)
    W64 = W5.double(); wo = W64[:, oidx.long()].to(torch.bfloat16).contiguous(); W64[:, oidx.long()] = 0.0
    Wr = F.fwht_blocks(W64.float()).to(torch.bfloat16)            # the block rotation on the columns (fp32 here; the fold uses fp64)
    W8o, wso = F.quantize_weight_int8(Wr)
    rec("outlier.oidx", oidx, "the declared channels, ascending int32 (a published constant)")
    rec("outlier.wo", wo, "bf16 of the folded weight's outlier columns [N, k]")
    rec("outlier.w8", W8o, "the int8 weight of the block-rotated weight with the outlier columns zeroed")
    rec("outlier.ho", x5[:, oidx.long()].contiguous(), "the row's outlier values, exact bf16")
    rec("outlier.side", F.outlier_side(x5[:, oidx.long()], wo), "sum_k fl32(ho[m,k]) * fl32(wo[n,k]), sequential binary32 adds")
    rec("linear.v2_5.outliers.no_bias", F.w8a8_outlier_linear(x5, oidx, wo, W8o, wso, out_dtype=torch.bfloat16),
        "bf16(fl32(bf16((fl32(acc)*ws)*asc)) + side)")
    rec("linear.v2_5.outliers.bias", F.w8a8_outlier_linear(x5, oidx, wo, W8o, wso, bias, out_dtype=torch.bfloat16),
        "then bf16(fl32(y) + bias[n])")
    # 3c. contract v2.5.1: the same rule on an unrotated input (o_proj): no block butterfly
    W64n = W5.double(); won = W64n[:, oidx.long()].to(torch.bfloat16).contiguous(); W64n[:, oidx.long()] = 0.0
    W8n, wsn = F.quantize_weight_int8(W64n.to(torch.bfloat16))
    rec("outlier.norot.w8", W8n, "the int8 weight with the outlier columns zeroed, no rotation")
    rec("linear.v2_5_1.outliers_norot.no_bias", F.w8a8_outlier_linear(x5, oidx, won, W8n, wsn, out_dtype=torch.bfloat16, rot=False),
        "o_proj: rowquant of the masked row (no butterfly), the same side path")
    # 4. integer RMSNorm, gain 1 and a general gain, widths 512 and 128
    for N in (512, 128):
        xn = (_randn(7, N, generator=g) * 2).to(torch.bfloat16)
        rec("rmsnorm.gain1.N%d" % N, F.rmsnorm_contract(xn, torch.ones(N, dtype=torch.bfloat16), out_dtype=torch.bfloat16),
            "SPEC 15 integer norm, per-row pow2 scale, rsqrt table")
        gain = (1.0 + 0.1 * _randn(N, generator=g)).to(torch.bfloat16)
        rec("rmsnorm.gain.N%d" % N, F.rmsnorm_contract(xn, gain, out_dtype=torch.bfloat16))
    # 5. RoPE on the Q14 grid, neox pairing, full and partial rotary dim
    hs = 128
    pos = torch.arange(6)
    inv = 1.0 / (10000 ** (torch.arange(0, hs, 2).float() / hs))
    ang = pos[:, None].float() * inv[None, :]
    cache = torch.cat([ang.cos(), ang.sin()], -1)
    q14 = (torch.round(cache * 16384).clamp(-16384, 16384) / 16384).float()
    qh = (_randn(6, 2 * hs, generator=g)).to(torch.bfloat16)
    kh = (_randn(6, 1 * hs, generator=g)).to(torch.bfloat16)
    for rd in (128, 96):
        qq, kk = F.rope_contract(pos, qh.clone(), kh.clone(), q14 if rd == 128 else torch.cat([q14[:, :rd // 2], q14[:, hs // 2:hs // 2 + rd // 2]], -1), hs, rd, True)
        rec("rope.q.rd%d" % rd, qq, "3 correctly rounded fp32 ops, one bf16 rounding")
        rec("rope.k.rd%d" % rd, kk)
    # 6. SiLU-and-mul (SPEC 6.3.1), incl. tie-dense and extreme inputs
    xs = torch.cat([_randn(4, 256, generator=g) * 4, torch.round(_randn(2, 256, generator=g) * 2) / 2,
                    torch.tensor([[-40.0, 40.0, 0.0, 1e-30, -1e-30, 0.5, -0.5, 30.0] + [0.25] * 248])], 0).to(torch.bfloat16)
    # the reference takes binary32 (the epilogue's widened bf16); SiLU-and-mul is silu(gate) * up, rounded to bf16
    gate = xs.float(); up = xs.flip(-1).float()
    rec("silu", F.silu_contract(gate), "ten declared fp32 steps, 8192-entry frac table; binary32 in and out")
    rec("silu_mul", (F.silu_contract(gate) * up).to(torch.bfloat16), "bf16( fl32( silu(gate) ) * up )")
    # 7. the R4 butterfly on one 256-lane block (8 pinned stages, exact 2^-4)
    v = (_randn(3, 256, generator=g)).to(torch.bfloat16).float()
    def had_block(row):
        r = row.clone()
        n = 256; h = 1
        while h < n:
            for i in range(0, n, 2 * h):
                a = r[i:i + h].clone(); b = r[i + h:i + 2 * h].clone()
                r[i:i + h] = (a + b).float(); r[i + h:i + 2 * h] = (a - b).float()
            h *= 2
        return r * 0.0625
    r4 = torch.stack([had_block(row) for row in v])
    rec("r4.block256", r4, "stage order: pairs at distance 1, 2, 4, ..., 128; each stage one fp32 add/sub; then exact 2^-4")
    # 10. v2.2: the model eps as an integer term (its own generator so the 17 v2.1 digests stand)
    g2 = _Gen(2026_09_06)
    for N in (512, 128):
        gain = (1.0 + 0.1 * _randn(N, generator=g2)).to(torch.bfloat16)
        xt = (_randn(7, N, generator=g2) * 3e-3).to(torch.bfloat16)      # mean square ~ eps (Mistral-7B embeddings)
        rec("rmsnorm.eps1e-5.N%d" % N, F.rmsnorm_contract(xt, gain, 1e-5, out_dtype=torch.bfloat16),
            "ss += floor(fl64(N*eps) * 4^e), e capped at norm_ecap(N*eps); the rows have mean square ~ eps")
        xs = (_randn(7, N, generator=g2) * 1e-6).to(torch.bfloat16)      # far below eps: e hits the cap
        rec("rmsnorm.eps1e-6.subeps.N%d" % N, F.rmsnorm_contract(xs, gain, 1e-6, out_dtype=torch.bfloat16),
            "rows far below eps: e hits the cap, the eps term dominates ss")
    # 11. contract 2.6: the float norm (t16n_ref.rmsnorm_float), drawn last so that every digest above stands
    if F.NORMF:
        import t16n_ref
        g3 = _Gen(2026_09_10)
        for N, eps in ((4096, 1e-6), (3072, 1e-5), (128, 1e-6)):
            xn = (_randn(6, N, generator=g3) * 2).to(torch.bfloat16)
            xn[1] = 0; xn[2, 7] = 3.0; xn[3] = (_randn(N, generator=g3) * 1e-19).to(torch.bfloat16)
            wn = torch.ones(N) if N != 128 else (1.0 + 0.1 * _randn(N, generator=g3)).to(torch.bfloat16)
            rec("norm_float.N%d.out" % N, t16n_ref.rmsnorm_float(xn, wn, eps=eps),
                "contract 2.6: fp32 sum of fl32(x^2) in the index-pinned tree (8-element chains, largest-power-of-two split), "
                "S = fl32(S + fl32(N*eps)), rs = fl32(fl32(sqrt N) / fl32(sqrt S)) (IEEE sqrt and division, no table), "
                "out = bf16(fl32(fl32(x * rs) * g))")
            rec("norm_float.N%d.S" % N, t16n_ref.row_sum_float(xn.float()), "the pinned-tree fp32 sum of squares itself")
    return out


def main(argv):
    if len(argv) >= 2 and argv[0] == "--check":
        ref = json.load(open(argv[1]))
        got = vectors()
        bad = [k for k, v in ref["vectors"].items() if got["vectors"].get(k, {}).get("sha256") != v["sha256"]]
        print("CONTRACT-VECTORS %s: %d vectors, %d differ%s" % ("PASS" if not bad else "FAIL", len(ref["vectors"]), len(bad), (": " + ", ".join(bad)) if bad else ""))
        return 0 if not bad else 3
    out = vectors()
    dst = os.path.join(HERE, "..", "results", "contract_vectors.json")
    if not os.path.isdir(os.path.dirname(dst)):
        dst = "contract_vectors.json"
    json.dump(out, open(dst, "w"), indent=1)
    print("CONTRACT-VECTORS wrote %s (%d vectors)" % (dst, len(out["vectors"])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
