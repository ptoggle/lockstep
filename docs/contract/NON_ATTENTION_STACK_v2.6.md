# Lockstep contract v2.6: the non-attention stack (normative)

Contract 2.6 is contract 2.5 with ONE section replaced: the RMSNorm (section 4) is no longer the
integer norm but the **float norm**, an fp32 computation whose reduction order is pinned by element
index and whose reciprocal square root is one IEEE square root and one IEEE division. Everything
else, the quantisers (1, 2), the linear and its epilogue (3), RoPE (5), the folds, the online R1
rotation and the declared outlier channels (6), SiLU (7), the head (8) and tensor parallelism (9),
is unchanged and is read from `NON_ATTENTION_STACK_v2.5.md`: the norm's bf16 output goes on to the
256-block rotation, the outlier split and the per-token quantiser exactly as under 2.5. This
document states the replaced section, the additions to the manifest, and the references and gates
that pin the new rule. Switch: `T16L_NORMF=1` on the reference, the kernels, the manifest tooling and
the served engine; a manifest prepared under one setting is refused under the other
(`export_meta.normf`, 2 for this rule, absent or 0 for the integer norm).

## 0. Arithmetic conventions (addendum)

The conventions of 2.2 section 0 hold: every written `fl32` operation is one correctly rounded
binary32 operation in the written order, no fused multiply-add, no reassociation, no fast-math, no
TF32, and subnormals are not flushed. Section 4 below introduces a binary32 SUM whose order is part
of the rule: an implementation MUST perform exactly the written additions, pairing exactly the
written operands. Nothing in this contract permits a reduction whose order is left to the
implementation.

## 4. The float norm

For a row `x` of width `N` (a multiple of 8) in bf16, widened to binary32. With a residual input
(`norm(x + res)`), `x + res` is one binary32 add per element and its bf16 rounding is ALSO the
layer's residual output, as in 2.2.

```
sq[j]   = fl32( x[j] * x[j] )                                          j = 0 .. N-1
group g (elements 8g .. 8g+7), g = 0 .. G-1, G = N / 8:
        a[g] = sq[8g];  a[g] = fl32( a[g] + sq[8g+i] )  for i = 1, 2, ..., 7   (in that order)
S       = T( a[0 .. G) )
        T(v) of one value is that value; of n > 1 values, T(v) = fl32( T(v[0..P)) + T(v[P..n)) )
        with P the largest power of two strictly below n  (a power of two splits in halves;
        384 splits 256 | 128; 768 splits 512 | 256).  This is the adjacent-pairs balanced tree.
S       = fl32( S + ne ),   ne = fl32( fl64(N) * fl64(eps) )   eps = the model's rms_norm_eps, ne a published
                                 binary32 constant of the (width, model); 2.5 carries eps as an integer term
                                 (its `norm_ecap`), here it is the fp32 add of the stock definition
S       = max( S, 2^-126 )                                             (an all-zero row -> zeros)
rs      = fl32( sN / fl32( sqrt(S) ) ),   sN = fl32( sqrt(N) )   a published constant of the width (64 for 4096)
        one IEEE binary32 square root (correctly rounded, as IEEE 754 requires of sqrt) and one IEEE
        binary32 division: basic operations, no table, no approximation.  (an earlier draft read
        1/sqrt of the 14-bit-truncated mantissa from the 2.2 table; the norm no longer uses any table.)
out[j]  = bf16( fl32( fl32( x[j] * rs ) * g[j] ) )     g = the norm's gain, bf16 widened to binary32
```

`g` is the norm's gain as served: 1 under the 2.2 fold, the model's gain under R1 online (2.4), the
bf16 reciprocal of the published scale under the smoothing fold (2.3); in every case the bf16 value
that 2.5 publishes as `G = rn_even(g * 2^17)`, and published as itself (`<norm>.g`) here. When `g = 1`
exactly the second multiply is the identity and an implementation may omit it. Qwen3's per-head
`q_norm` / `k_norm` (width 128) use the same rule with their gains. What follows `out` is 2.5's:
under R1 online the 256-block rotation, the outlier split and the per-token int8 quantiser
(sections 1, 6.5, 6.6); a kernel may fuse the quantiser onto the norm where 2.5 quantises `out`
directly, taking the row maximum from the input maximum through the monotone transform when `g = 1`.

Why this order and these constants. The tree is defined by element index alone so that a 32-lane
warp (xor-shuffles by 1, 2, 4, 8, 16), a 64-lane wavefront, a CPU, or any thread mapping reproduces
the same additions; a CTA combines its warp sums and chunk sums by the same rule. The reciprocal
square root is NOT a transcendental function: it is one square root and one division, both of which
IEEE 754 requires to be correctly rounded, so `rs` is the same bits on every compliant machine
(CUDA: `sqrt.rn.f32` and `div.rn.f32`, i.e. `-prec-sqrt=true -prec-div=true`; no `--use_fast_math`).
`sN` carries `sqrt(N)` with one rounding for widths that are not perfect squares and none for those
that are. The guard at `2^-126` keeps `rs` finite (sqrt(2^-126) = 2^-63); an all-zero row gives zeros;
rows whose fp32 sum of squares overflows (elements near `2^64`) are outside the model's domain and
are defined by the arithmetic above (S = +inf, rs = 0).

## 10. Published constants, the manifest (addendum)

In addition to 2.5's constants, every norm publishes `<norm>.g` (bf16 [N], the gain as served) and the
manifest carries `export_meta.normf = 2` and `contract_version = "2.6"`. `<norm>.G` and `<norm>.s`
(the integer gain and scale) are still written for tooling that reads them; under 2.6 they are not
used by the engine. A manifest without `normf` is a 2.5 manifest and is refused by a 2.6 engine.

## 11. Reference implementations and gates (addendum)

| what | reference (torch / numpy, CPU-capable) | engine implementation | gate |
|---|---|---|---|
| section 4 (2.6) | `lssa_runtime/t16n_ref.py` (`rmsnorm_float`, and its numpy twin `rmsnorm_float_np`; `row_sum_float` is the sum alone) | `t16n_kernels.cu` (`normf_k`, `normf_small_k`), routed by `t16n_stack.py` from `t16l_stack` (all norms) and `t16m_stack` (the fused forward's two norm sites) | `t16n_test.py` (120 cases: widths 128 .. 8192, folded and per-channel gains, with and without residual, 1 .. 1024 rows, edge rows; torch and numpy twins must agree), `lockstep_fullstack.py --selftest` (`rmsnorm_row_invariant`, `rmsnorm_cpu_eq_cuda`), `t16l_egates.py` (per-op identity on real activations through `lockstep_fullstack.rmsnorm_rule`) |
| cross-architecture determinism | `results/contract3_20260910/xarch/xarch_test.py` (fixed `inputs.npz`, the kernel, the torch and numpy references, sha256 of every output) | the same kernel source built for each GPU | identical digests on H100 (sm_90, CUDA 13.0), RTX 4090 (sm_89, CUDA 12.8) and RTX 5090 (sm_120, CUDA 12.8), 72 cases, 492 digests, 10 September 2026 (`xarch/RESULTS.md`; re-run with the eps term pending) |
| test vectors | `contract_vectors.py` under `T16L_NORMF=1` -> `contract-vectors-v2.6 (float norm)`: `norm_float.N{4096,3072,128}.out`, `.S` (with the eps term) | | `python contract_vectors.py --check` |

Contract versions: v1, v2, v2.1 to v2.5 as in `NON_ATTENTION_STACK_v2.5.md`; v2.6 (this document; its drafts were numbered 3.0 and 3.1 on a 2.2 base before the 2.5 merge). A served engine declares its
version in `LOCKSTEP-CONFIG`; a manifest prepared under one version is not valid under another.
