# Lockstep contract v2.5: the non-attention stack (normative)

This document is the contract for every operator of a decoder layer other than attention,
plus the head, the rotations folded into the published weights, and the tensor-parallel
reduction. It is written for an implementer or a verifier who has only this text, the
attention contract (`BLOCK_EXPONENT_VARIANT.md`, LSSA-B and LSSA-B8) and the reference code
named at the end. Where the text and the reference disagree, the reference code and its
test vectors (`results/contract_vectors.json`) are the contract, and the text is a bug to
fix here.

Version: **v2.5** (2026-09-07). v2.5 adds DECLARED OUTLIER CHANNELS to the R1-online norm-fed
linears (section 6.6): a published list of `k` input channels per `qkv_proj` / `gate_up_proj`
(default 8, chosen by calibration amax) whose exact bf16 values bypass the block rotation and
the int8 quant and re-enter through an exact side path in the epilogue. It is what the
256-block rotation of v2.4 lacked on Llama: the per-token int8 quant of a row holding one
2,000-scale channel, diluted only over 256 channels, left every ordinary channel one level
(Llama-3.1-70B in emulation: +66 percent; with 8 declared channels +0.94, against +1.11 for
the offline residual rotation). The norm was NOT the cost: the 15-bit row quantisation alone
reads within 0.1 percent of bf16 on Llama, Qwen and Phi (SPEC 7.22), and the v2.4 paragraph
below is corrected by that measurement. With the outlier channels, R1 ONLINE is the default
of `lockstep prepare` (it wins or ties the offline fold on every measured model and keeps the
checkpoint's basis for the embedding, the head and the residual; the ladder ends at the v2.2
fold), and R2 (6.2) is off by default. A v2.4 manifest (no `.oidx` constants) is a v2.5
manifest with `k = 0` and stays valid; v2.2 / v2.3 manifests stay valid as before. v2.4 adds R1 ONLINE as a per-model option (section 6.5): the
residual stream, the embedding, the head and the weights stay in the original basis, the norms
keep their gains, and the norm output is rotated online by the R4 block rotation before its
int8 quant. It removes the norm-gain fold that costs Phi-4-mini its band (+1.72 -> +0.52) and
improves Qwen and Mistral (Qwen2.5-7B +0.29 -> +0.06), but it removes the residual rotation
too, and on models with massive activations (Llama) the integer norm's 15-bit row quantisation
of the UNROTATED residual then starves the ordinary channels (Llama-3.1-70B: exact on the
synthetic probe, 2.4x worse than stock on real text at every length). The offline R1 of v2.2 /
v2.3 therefore stays the default; `lockstep prepare` escalates to online on a quality-gate
failure and records the choice in `family.rotation.R1`; v2.2 / v2.3 manifests remain valid. v2.3 adds the smoothing fold (section 6.4): a published
per-channel scale folded into neighbouring constants before the rotations, so a norm may carry
a non-unit integer gain and the MLP weights carry the inverse; the served operators are
unchanged and the 21 vectors stand. v2.2 carries the model's RMSNorm epsilon into the integer
norm (section 4): v2.1 dropped it, which over-normalises rows whose mean square is at or
below eps - Mistral-7B's embedding rows (mean square 7e-6 against eps 1e-5) lost 35 percent
of their scale at layer 0 and the whole stack read perplexity 26 against 5.2; Llama-3.1-8B's
lost 4 percent, Qwen's under 1 (SPEC 7.20). Every v2.1 manifest must be re-prepared: the
constants are unchanged, the arithmetic and therefore the fingerprint are not. v2.1 declared
the linear epilogue in the other order (section 5); everything else is unchanged from v2. The attention rules are versioned
separately (LSSA-B8 addendum, table `T2` digest `7a42a785...`).

## 0. Arithmetic conventions

- `fl32(x)`: the IEEE-754 binary32 value of `x`; `bf16(x)`: round-to-nearest-even of a
  binary32 value to bfloat16; every product, sum, quotient written below is ONE correctly
  rounded binary32 operation unless it is an integer operation. No fused multiply-add, no
  reassociation, no fast-math, no TF32: an implementation MUST evaluate each written
  operation as one IEEE operation in the written order.
- Integers are exact; int32 accumulation of int8 x int8 products is exact for every shape
  in this contract (K <= 2^15 and |a|,|w| <= 127 give |acc| < 2^29).
- `rn_even(x)`: round a binary32 to the nearest integer, ties to even (`rintf`).
- All per-row scales are binary32; all tables are exact integers.

## 1. Activation quantisation (per token, int8)

For a row `x` (binary32 values, the row's width `K`):

```
amax = max_k |x[k]|                       (exact, order-free)
asc  = fl32( max(amax, 1e-8) / 127 )      one binary32 divide (T16L_QDIV=1, the default form)
q[k] = clamp( rn_even( fl32( x[k] / asc ) ), -127, 127 )    one binary32 divide per element
```

`q` is int8, `asc` binary32. The divide form is the contract; `x * (1/asc)` (QDIV=0) is a
different function and is not v2.1.

## 2. Weight quantisation (per output channel, int8), published constants

For a weight matrix `W` [N, K] in bf16, widened to binary32:

```
ws[n]   = fl32( max_k |W[n,k]| clamp_min 1e-8 / 127 )
W8[n,k] = clamp( rn_even( fl32( W[n,k] / ws[n] ) ), -127, 127 )
```

`W8` and `ws` are published constants of the model (they live in the manifest, section 8)
AFTER the folds of section 6 have been applied to `W`. A verifier recomputes them from the
published bf16 checkpoint and the fold rules, or trusts the manifest's digests.

## 3. The linear: int8 GEMM and the v2.1 epilogue

```
acc[m,n] = sum_k q[m,k] * W8[n,k]                   int32, exact
y8[m,n]  = bf16( fl32( fl32( fl32(acc[m,n]) * ws[n] ) * asc[m] ) )     WEIGHT scale first
out      = y8                                       no bias
out[m,n] = bf16( fl32( fl32(y8[m,n]) + bias[n] ) )  with a bias: a SECOND rounding
```

`bias[n]` is the published binary32 bias (after the norm-gain fold of section 6.1).
`fl32(acc)` is exact for |acc| < 2^24 and correctly rounded otherwise. v2 declared
`(fl32(acc) * asc[m]) * ws[n]` with the bias added before a single rounding; v2.1 swapped
the scale order because the Hopper CUTLASS int8 GEMM's fused epilogue evaluates exactly
`(acc * ws) * asc` (SPEC 2.6), and declared the bias as a separate rounded add because no
fused bias epilogue reproduces a simple form. The two orders differ by one bf16 ULP on
about 5e-6 of outputs.

The `down_proj` input is rotated online by R4 (section 6.3) before quantisation.

## 4. Integer RMSNorm (SPEC 15) and the norm-gain fold

Gains are folded into the consumer linears (section 6.1) so a norm in the served model has
gain 1, or under the v2.3 smoothing fold (6.4) the bf16 reciprocal of the published scale;
either way it is published as the integer `G[k] = rn_even(g[k] * 2^17)` and the op is defined
for any gain. For a row `x` of width `N` in bf16, widened to binary32:

```
ne     = fl64(N) * fl64(eps)          one binary64 multiply; eps is the model's rms_norm_eps
ecap   = the largest e in [0, 60] with ne * 4^e <= 2^50   (60 when ne = 0; binary64, exact:
                                 every factor is a power of two)  -- `norm_ecap`
m      = max_k |x[k]|
e      = pow2_exponent(m, 15):  e = 15 - E where m = f * 2^E, 1/2 <= f < 1 (frexp);
                                 if fl32(m * 2^e) > 32767 then e -= 1; clamp e to [-60, 60];
                                 e = 0 when m = 0;  then e = min(e, ecap)
xq[k]  = clamp( rn_even( fl32( x[k] * 2^e ) ), -32767, 32767 )     int32
ss     = sum_k xq[k]^2  +  floor( ne * 4^e )                        int64, exact (the eps term is
                                 N * eps in the quantised domain, scaled by the exact power
                                 of two 4^e and floored; ecap keeps ss + term below 2^51);
                                 ss = max(ss, 1)
r      = rsqrt_lut(ss / N):  the table `rsqrt_lut` (reference: 2 x 16384 entries of
                             rn(32768 / sqrt((1 + m/16384) * 2^p)), p in {0,1}) indexed by
                             the leading bits of the binary32 mean of squares, as in the
                             reference `rmsnorm_contract` - the table and the indexing are
                             the contract, not this sentence
y[k]   = clamp( (xq[k] * r * G[k]) >> (shifts of the reference), -8388607, 8388607 )   int32
out[k] = bf16( fl32( y[k] ) * s_out ),   s_out = fl32( sqrt(N) / 2^17 )
```

The eps term makes `r` the contract's value of `1 / sqrt(mean(x^2) + eps)`, the stock
definition, to the table's resolution; with eps = 0 the op is v2.1's. The cap `ecap` is a
function of `N * eps` only (26 for N = 4,096 and eps = 1e-5), so a row whose largest
element is below 32767 * 2^-ecap (about 5e-4 there) is quantised with step 2^-ecap instead
of its own power of two - still 15 bits of resolution for the elements that matter, and the
row's `ss` is then dominated by the eps term, as it is in binary32 for the stock op.

With a residual input (`norm(x + res)`), `x + res` is one binary32 add per element and its
bf16 rounding is ALSO the layer's residual output (the reference `forward_contract`).
The exact integer chain (shift positions, the LUT index) is `lockstep_fullstack.rmsnorm_contract`
and its kernel twin `t16l_kernels.cu::dynnorm_k`, gated bit-identical; the test vectors
pin it. Rows of width 128 (Qwen3's per-head q_norm / k_norm) use the same op.

## 5. RoPE, frozen Q14

The frozen table is a PUBLISHED constant of the manifest, verified by its hash. Its derivation is
vLLM's cache - cos and sin in binary32, held in bf16 - placed on the Q14 grid; a CPU recomputes it
to within the bf16 rounding-boundary entries where the device's cos / sin differ in the last bit
(Qwen2.5-7B: 1,263 of 4,194,304 entries, 0.03 percent; the other 482 tensors of a manifest
recompute byte for byte). An implementer reproduces the served arithmetic from the published
table, not from the formula.

The rotary cos/sin cache is frozen once per model: `q14 = clamp(rn_even(cache * 2^14), -2^15, 2^15) / 2^14`
(a binary32 value on the Q14 grid with |value| <= 2, computed from vLLM's cache for the
model's rope parameters, including Llama-3.1 scaling and Phi-4's partial rotary dim; the
range reaches 2 because longrope multiplies cos and sin by a scale above 1 - Phi-4-mini's
1.19 - and a clamp at 1 silently destroyed that scale, +3 points of perplexity). For a
long/short rope (Phi-4) the frozen table is the concatenation [short rows | long rows] and
the row index carries the offset `original_max_position` when the deployment's
`max_model_len` exceeds it; both the table and the offset are published constants of the
manifest, so a manifest is bound to the `max_model_len` it was prepared at. For a head
vector `t` (bf16, widened) and its position's `(cos, sin)` rows:

```
o1 = fl32( t1 * cos ) ;  o2 = fl32( t2 * sin ) ;  out1 = bf16( fl32( o1 - o2 ) )
o3 = fl32( t2 * cos ) ;  o4 = fl32( t1 * sin ) ;  out2 = bf16( fl32( o3 + o4 ) )
```

three correctly rounded binary32 operations per output, one bf16 rounding; the neox
pairing (`t1 = t[:rd/2]`, `t2 = t[rd/2:]`) as in the reference; lanes beyond `rotary_dim`
pass through. Applied to q and k after the qkv linear (and after Qwen3's q/k norm).

## 6. Folds and rotations (offline, into the published constants)

### 6.1 Norm gains
Every `input_layernorm` gain folds into `qkv_proj`'s input columns, every
`post_attention_layernorm` gain into `gate_up_proj`'s, the final norm's into `lm_head`'s,
all in binary64 then rounded to bf16 (`W[n,k] * g[k]`); the norms keep gain 1. A TIED head
(Phi-4-mini) is untied first: the embedding rows never carry the gain.

### 6.2 R1 and R2 (offline)
R1: an orthogonal `Q` [H, H] = `(Hadamard * random signs) / sqrt(row norm)`, signs from
`torch.Generator(cpu).manual_seed(seed)` (seed 0), Hadamard by `hadamard_for(H)`:
Sylvester when `H` is a power of two, Paley type I when `H - 1` is a prime = 3 (mod 4),
otherwise block-diagonal Sylvester on the largest power-of-two block dividing `H`.
(Paley I at 4,096 or 3,072 is NOT a Hadamard matrix; the builder refuses any `Q` with
max |Q Q^T - I| > 1e-9.) The embedding rows and the head rows are rotated `e -> e Q`,
`W_lm -> W_lm Q`; every input-side weight (qkv, gate_up) `W -> W Q^T`; every output-side
weight (o_proj, down_proj) `W -> Q W`, all in binary64 then bf16. The residual stream then
lives in the rotated basis; RMSNorm is invariant to it because its gain is 1.
R2: a Sylvester Hadamard on the head dimension (`sylvester(128) / sqrt(128)`), block-diagonal
over kv heads on the v rows of qkv (and its bias) and over q heads on o_proj's input.
`Q`'s kind, `H_sha256` and `signs_sha256` are published in the manifest.

R2 is OPTIONAL and off by default since Stage 2 (`lockstep prepare --r2 on` enables it; the
manifest records `family.rotation.R2`, and the served arithmetic is the same either way, since
R2 lives entirely in the published weights). Measured on Llama-3.1-70B at TP=4 (SPEC 7.22): with
R2 every arm of the stack costs 5 to 8 points of perplexity more than without, whichever
linear is quantised, and o_proj's int8 without R2 costs nothing - the rotated value vectors sit
in the bf16 KV cache, where the small components that carry a token's information round at the
scale of the large ones, and the sink token's value enters every row. Llama-3.1-8B and
Llama-3.2-3B read 0.2 and 1.3 points better without it, Qwen2.5-7B and Mistral-7B are
unchanged, Phi-4-mini reads 0.25 worse, inside its band.

### 6.3 R4 (online) on the down_proj input
The SiLU output row (width `I`, the rank's slice zero-padded to a multiple of 256) is
transformed per 256-lane block by the 8-stage Sylvester butterfly, each stage `a + b`,
`a - b` as ONE binary32 add / subtract, in the pinned stage order of the reference, then
multiplied by the exact `2^-4`; `down_proj`'s published `W8` carries `W (blockdiag(H256)/16)`
so the composition is the identity in exact arithmetic. SiLU(0) x 0 = 0, so the padding
lanes are exactly zero. The rotated row is then quantised by section 1.

### 6.4 The smoothing fold (v2.3)

The per-token int8 activations of the MLP linears are the quality cost on models whose MLP
inputs keep a per-token structure the rotations do not flatten (Phi-4-mini, SPEC 7.21).
A per-input-channel scale `s[k]` is chosen offline from calibration statistics and folded
so that the served arithmetic sees `x / s` where the original model saw `x`:

```
down_proj    (input silu(gate) * up, its own basis, per tensor-parallel rank):
             W_up'[k, :]   = W_up[k, :] / s_dn[k]        (rows of the up half of gate_up)
             W_down'[:, k] = W_down[:, k] * s_dn[k]      then R4 as before
gate_up_proj (input the post-attention norm output, in the R1-rotated basis):
             g_inv[k]      = bf16( 1 / s_gu[k] )         the norm's published gain (G = rn_even(g_inv * 2^17))
             W_gu'[:, k]   = (W_gu Q1)[:, k] / g_inv[k]  (fp64, the exact inverse of the bf16 gain)
qkv_proj     the same rule on the input norm, when enabled
```

`s` is a constant of the manifest like any other: the fold is exact by construction for ANY
`s`, so the choice is a quality decision, not a correctness one. The published choice
(`lockstep_calibrate.py`, `manifest.smoothing`): `s[k] = clamp(sqrt(amax_x[k]) /
sqrt(amax_w[k]), 1/100, 100)` with `amax_x` the channel's absolute maximum over 16 windows of
1,024 corpus tokens on the stock model (the gate_up statistics taken in the rotated basis
`x Q1`) and `amax_w` the weight column's, and `s = 1` for channels the calibration never
lit. The fold is applied by level: 0 = none (the default: the v2.2 constants, so a v2.2
manifest is a v2.3 manifest at level 0), 1 = gate_up through the norm gain (the level that
moves Phi-4-mini), 2 = also down_proj, 3 = also qkv; `lockstep prepare` escalates a level
only when the quality gate fails, in a fresh process, and records the level in the
manifest. Exactness does not depend on the level; the fingerprint does.

### 6.6 Declared outlier channels (v2.5)

Under R1 online the norm output `h` (bf16, gain applied, original basis) is rotated in
256-blocks and quantised per token. A row carrying a massive activation - Llama's residual
stream holds single channels in the hundreds to thousands on the first and delimiter tokens
(channels 788, 1384, 4062 on the 8B; 1532, 6857, 7306 on the 70B, the same channels in every
layer) - is diluted only over its own 256-block, so the row's int8 scale is set by that block
and the ordinary channels round to a level or two. The offline residual rotation spreads the
channel over the whole width (sqrt(8192) = 90 against 16) and hides the problem at the cost
of the gain fold. v2.5 removes the channel from the quantised row instead:

```
published, per norm-fed linear (qkv_proj, gate_up_proj), from the calibration amax of its
input on a stock engine over the calibration windows (lockstep_calibrate.py, stats_version 3):
  oidx      int32 [k], distinct, ascending, 1 <= k <= 32 (default k = 8): the k channels of largest amax
  wo        bf16  [N, k] = bf16( W64[:, oidx] )   W64 = the fp64 weight AFTER every fold of
            sections 6.1 to 6.5 (norm gain, smoothing, R2 on the v rows) and BEFORE the block rotation
  W8, ws    the int8 weight of ( W64 with the oidx columns set to zero ) H_256 / 16   (6.5)
            (gate_up: rows padded to 256 as in 6.3; wo takes the same row padding)

per token, row h (bf16 [K]):
  ho[j]     = h[oidx[j]]                                  exact, bf16              (r4quant_outl)
  hm        = h with hm[oidx[j]] = 0
  x8, asc   = quant_pow2_rows( H_256(hm) / 16 )           the R4 op of 6.3 on the masked row
  acc       = x8 . W8^T                                   int32, exact
  y8[n]     = bf16( (fl32(acc[n]) * ws[n]) * asc )        the v2.1 epilogue (section 3)
  side[n]   = sum_{j = 0}^{k-1} fl32(ho[j]) * fl32(wo[n, j])     each product exact in binary32
                                                          (two 8-bit significands); the sum is
                                                          SEQUENTIAL in j with binary32 rn adds,
                                                          starting from 0
  y[n]      = bf16( fl32(y8[n]) + side[n] )
  with a bias:  bf16( fl32(y[n]) + bias[n] )              as in section 3
```

`y` then feeds the rope (qkv) or the SiLU (gate_up) exactly as before. The value is exact by
construction for any choice of `oidx`: `W(hm) + W[:, oidx] h[oidx] = W h` in real
arithmetic, and the rotation acts on `hm` alone. The choice is a published constant a
verifier checks for consistency (`verify_constants.py` recomputes `wo` and `W8` from the
checkpoint and the manifest's `oidx`); recomputing the amax needs a stock run of the
calibration windows and is not part of the fingerprint. The side path costs `k` multiply-adds
per output element against `K` for the GEMM (one part in a thousand at `k = 8`). Only the
norm-fed linears carried it in v2.5. **v2.5.1** extends the same rule to the other two linears,
per tensor-parallel rank (their inputs are the rank's slice): `o_proj`'s input (the rank's
attention output, in the head basis the served model uses - R2's when R2 is on) with
`oidx` from the calibration amax of that input and NO rotation (the row is quantised by the
plain per-token rule with the outlier channels at zero, `rowquant_outl`), and `down_proj`'s
input (silu(gate) * up on the rank's slice) with the outliers removed BEFORE the R4 pad and
butterfly (`r4quant_outl` with `had = 1`; `wo` is taken from the down weight before its R4
fold). The manifest records `outliers.k` and the list of linears; `k = 0` (or a v2.4
manifest) is the v2.4 arithmetic; a v2.5 manifest (norm-fed linears only) stays valid.

### 6.5 R1 online (v2.4)

Under the offline R1 (6.1 to 6.3) the residual stream is rotated, so a norm cannot apply its
per-channel gain and the gain is folded into the consumer's weight columns before the rotation
mixes them. On Phi-4-mini the gains vary widely across channels and the per-row int8 weight
scale then loses the small columns: the fold alone costs 0.4 points and the weights carry
0.73 of the linears' 1.15 under it (SPEC 7.21). R1 online removes the fold:

```
residual, embedding, head, o_proj, down_proj outputs:  the ORIGINAL basis (no R1)
norm:            gain g kept, published as G = rn_even(g * 2^17) (section 4)
qkv_proj, gate_up_proj input:   x8 = quant_pow2_rows( H_256(norm output) / 16 )   the R4 op of 6.3
                                W' = W H_256 / 16 per 256-block of the input columns (offline, fp64)
smoothing (6.4):  in the original basis: G' = rn_even(bf16(g / s) * 2^17), W' columns * (g / bf16(g / s))
```

The op the linear input takes is exactly the down_proj input's (the 256-block Sylvester
with the exact 2^-4 normalisation, then the per-token int8 quant), so the kernels, the
reference and the vectors are unchanged; only the arming differs (the linear's `mode` is 1
for qkv and gate_up, exported with the constants). `hidden` must be a multiple of 256 (the
envelope requires it already). The manifest records `family.rotation.R1` (`true` offline,
`"online"`) and `smoothing.r1`; the ladder `lockstep prepare` climbs on a quality-gate failure
is offline/0, online/0, online/1, online/2, online/3, and `--r1 online` starts at online/0.

## 7. SiLU and mul (SPEC 6.3.1)

For gate value `x1` and up value `x2` (binary32 from the v2.1 epilogue), the ten declared
steps of the reference `silu_contract` (the `frac` table, 8,192 entries of 15-bit integers,
`contract_common.table_on(13, 15)`):

```
xc = 0 if |x1| < 2^-126 else x1
t  = clamp( fl32( xc * -1.4426950216293335 ), -30, 30 )
ii = ceil(t) ;  g = fl32( ii - t ) ;  k = min( trunc( fl32( g * 8192 ) ), 8191 )
f  = fl32( frac[k] * 2^-15 ) ;  e2 = ldexp(f, ii) ;  den = fl32( 1 + e2 ) ;  sig = fl32( 1 / den )
out = bf16( fl32( fl32( xc * sig ) * x2 ) )
```

## 8. The head (`a15w15`) and the logits

The head's activation row is quantised to 15 bits (`amax_q = 16383`, section 1 with 16383
in place of 127), split into two int8 planes `hi = a >> 7`, `lo = a - (hi << 7)`; the
published head weight is likewise two int8 planes (`wbits = 15`). The four int32 GEMMs are
combined exactly in int64: `s = acc_hh << 14 + acc_hl << 7 + acc_lh << 7 + acc_ll`, then
`logits = bf16( fl32( fl32( fl32(s) * asc[m] ) * ws[n] ) )` (`fl32(s)` round-to-nearest-even
from int64), sliced to `org_vocab_size`. One stacked GEMM `[2M, K] x [K, 2N]` evaluates the
same four products (exact) and is what the engine runs.

## 9. Tensor parallelism

The R4 padding is per rank (section 6.3). The row-parallel linears' partial outputs are
reduced `bf16( fl32(y_0) + fl32(y_1) + ... + fl32(y_{W-1}) )` in RANK ORDER, one rounding;
vLLM's custom all-reduce (rank-ordered fp32, one rounding) and the all-gather + fold
(`t16mc.tp_rank_sum`) both evaluate it and are gated equal. At TP > 1 the o_proj epilogue,
the reduction and the norm run as three ops.

## 10. Published constants, the manifest

`docs/PLUGIN.md`: `lockstep.manifest.json` names the contract version, the model, the
rotation kind and hashes, the key offset (attention), `tp.world_size`, and the sha256 of
every constant tensor (`W8`, `ws`, biases, `G`, `q14`, the rotated embedding, the head
planes) - the preimage is `lockstep-tensor-v1|<dtype>|<shape>|<raw little-endian bytes>`.
`probe.json` is the fingerprint: 256 fixed token ids and the bf16 bit pattern of every
prompt log-prob plus the 8 greedy ids, which a served engine must reproduce exactly.

## 11. Reference implementations and gates

| what | reference (torch, CPU-capable) | engine implementation | gate |
|---|---|---|---|
| sections 1, 3, 4, 5, 7, 8 | `lssa_runtime/lockstep_fullstack.py` (`quantize_rows_int8`, `w8a8_matmul`, `rmsnorm_contract`, `rope_contract`, `silu_contract`), `t16h_lmvariants.py` | `t16l_kernels.cu`, `t16m_kernels.cu` (fused), `t16l_stack.py`, `t16m_stack.py` | `t16l_egates.py` (per-op identity on real activations), `t16m_gate.py kernels` (fused == composed, 514 cases), CI |
| section 6 | `t16l_stack._fold_and_rotate`, `hadamard_for`, `build_Q1`; `t16k_rot.py` offline | the manifest constants | bisect probes `results/lp_llama_bis_*.json`, SPEC 7.16 |
| section 6.6 | `lockstep_fullstack.w8a8_outlier_linear`, `outlier_side`, `fwht_blocks`; `t16l_stack._pick_outliers`, `_split_outliers` | `r4quant_outl` (t16l / t16m / t16mc), the side path in `deq_k`, `deqrope_k`, `deqsilu_k`, `deqsilur4_k`, `r4_cl_k` | `t16l_kcheck.py` (6 cases), `t16m_gate.py kernels` (v2.5 section), 7 vectors, `t16l_egates.py` |
| section 9 | `t16mc.tp_rank_sum` | custom all-reduce | `t16s_car_gate.py` (32 / 32), `t16s_tp_gate.py` |
| test vectors | `lssa_runtime/contract_vectors.py` -> `results/contract_vectors.json` | | `python contract_vectors.py --check` |
| end to end | `lockstep_selftest.py` (fingerprint), `verify_rows.py` (attention rows), `t16h_ppl.py` (quality) | | `lockstep_plugin_gate.py`, E6, E1 / E4v2 |

Contract versions: v1 (portable prompt and decode kernels, `lsb_pass_a/b`, `lsb_dec`),
v2 (LSSA-B8 for every row + the non-attention stack with the activation-scale-first
epilogue), v2.1, v2.2 (eps), v2.3 (smoothing), v2.4 (R1 online), v2.5 (this document). A served engine declares its version in
`LOCKSTEP-CONFIG`; a manifest prepared under one version is not valid under another.
