# Lockstep specification and measured benchmarks

> **Paths in this document.** It was written against the campaign monorepo and cites files
> by those paths. In this repository they map as follows: `demo/<file>`,
> `lssa/portable/<file>` and `paper_softmax/bigrun/<file>` are all `lssa_runtime/<file>`
> (the runtime bundle is flat, exactly the `/workspace/p2` layout of the pods);
> `handover/results/` and `results/` are `results/`; `handover/pod/` is `pod/`;
> `handover/README.md` is superseded by the top-level `README.md`.


Verifiable exact attention inside a production inference engine, on a real model, with
every number in this document measured in one session on one box and reproducible from
the raw evidence in `results/`. Numbers marked **measured** come from that evidence.
Numbers marked *filed* come from the campaign logs named beside them and were not
re-measured here. Nothing modelled appears in this document.

| | |
|---|---|
| Model | Qwen2.5-7B-Instruct (28 layers, 28 heads, 4 KV heads, head dim 128, bf16 weights) |
| Engine | vLLM 0.25.1, torch 2.11+cu130, CUDA graphs FULL_AND_PIECEWISE with torch compilation mode 0 (the configuration that produces correct output, section 7), prefix caching off on both stacks |
| Session | 3 to 4 September 2026, RunPod pod `lc-handover` (H100) and `lc-verify-4090` (RTX 4090), FA3 `ce088ab`, key offset `.pt` sha `af5aea20...` (`results/PROVENANCE.txt`) |
| Engine mode for every ratio | both stacks in full CUDA graphs (`FULL_AND_PIECEWISE`) with torch compilation mode 3 and the `eager` backend, i.e. graphs on, inductor off: the only mode in which token identity can be gated (section 7.1); vLLM's default (inductor on) is reported beside it for reference |
| Box | 1x NVIDIA H100 SXM 80GB, RunPod secure cloud |
| Baseline | the same engine with its stock FlashAttention-3 backend, bf16 KV cache, same flags, same session, alternated |
| Lockstep stack | contract attention for every row (LSSA-B8 for prompt rows, LSSA-B for generated rows), key offset armed; linear layers, norms and output head stock bf16 |
| Evidence | `results/*.json`, checksums in `results/SHA256SUMS.pod`, toolchain in `results/PROVENANCE.txt` |

## 1. What is being claimed

1. **Exactness.** For every attention row the engine produces, the bf16 output is bit-identical to a reference implementation of the contract that runs on any hardware, including a CPU. The reference is a python-integer and scalar-fp32 program with no vendor dependency.
2. **Verifiability.** Because 1 holds, a verifier who is given a row's inputs recomputes it and compares bits. A cheaper attention, a skipped key, a perturbed prompt or an altered KV entry changes bits and is detected with certainty, not with a tolerance. Section 3 is that check, run.
3. **Cost.** With both stacks in full CUDA graphs and inductor off, the exact attention runs at 0.93 to 0.98x of stock time to first token from 2k to 32k tokens and, after this session's decode work, 0.92x / 0.91x / 0.87x of stock decode throughput at batch 1 / 8 / 32 (section 4, final rows). No parity wording is used anywhere: the campaign's pre-registered rule forbids it unless a cell reads at least 0.95x.
4. **Quality.** The perplexity of the contract attention against stock bf16 attention is in section 4.5. The pre-registered band is +0.30%.

Not claimed here: full-network verification. The linear layers in the timed
configuration are stock bf16. The fully verifiable non-attention stack is built and
gated in eager mode; its state and open defects are in section 7.

## 2. The contract

The forward pass's attention is redefined as an exact function of integers plus a short
list of pinned IEEE binary32 operations. Anything integer may be re-associated freely,
which is what makes any tiling, any split, any vendor produce the same bits. Nothing
floating-point may be re-associated. The normative text is
`paper_softmax/bigrun/BLOCK_EXPONENT_VARIANT.md`; its LSSA-B8 addendum is the rule for
prompt rows and the sections above it are the rule for generated rows.

### 2.1 Quantisation (shared by both rules)

- Keys are partitioned into blocks of `KB = 128` consecutive absolute positions. One block is one engine page, so prefill chunk boundaries must fall on multiples of 128; the engine enforces this with a scheduler hook. This is a contract property: a block's exponent is the amax over all 128 rows, so a mid-block cut changes the function.
- Per (block, kv-head) exponents `e_k`, `e_v`, and per (row, head) `e_q`, are powers of two chosen by a clip-safe rule and clamped to [-32, 40]; `q8`, `k8`, `v8` are int8. Scores `S = q8 . k8` are exact int32 with `|S| < 2^21`.
- A published per-(layer, kv-head, channel) key offset `mu` is subtracted from every key before quantisation. Softmax is invariant to any constant added to a score row, so the transformation is exact by construction and needs no zero-point term. Artifact: `keq_qwen25_7b_instruct.pt` (`keq-1`, bin sha `756923b1...`, manifest sha `22cb03ad...`), applied on read so that the stored key is the raw one.

### 2.2 LSSA-B8, the prompt-row rule (constants are contract version `v`)

| Constant | Value | Why |
|---|---|---|
| `B` | 26 | one doubling of the score scale is `2^26` |
| `W_LOC` | 9 doublings | the block-local weight window; 10 is provably the same table, so 9 is the ceiling, not a choice |
| `n` | 7 index bits | bit-indistinguishable from 8 on every offline metric, table halves |
| `TOP` | 255 | the top weight |
| `T2` table | `RN_half_up(255 * 2^(-k/2^7))`, zero from `9*2^7`, 2,176 bytes | generated from 40-digit decimal, sha256 `7a42a785940f557d5d5a4188fa5683ea00cf8b29e8d5213c75c581a5a0d92fbb` |
| `SEG`, `S_WIN` | 32 blocks, 32 doublings | the two-level fold and its exact-skip window |
| block order | ascending absolute block index | pinned; FA3's native descending walk computes a different function |
| rescale | exponent-field subtraction, flush to +0.0 below the normal range | closes the flush-to-zero gap; a float multiply would round |
| epilogue | one correctly rounded fp32 division, then round-to-nearest-even to bf16; `l == 0` gives 0 | `__fdiv_rn`, never `div.approx` |

Per (row, block): `m_S = max S` over visible keys (exact); `NB_b` = the least multiple of
`2^B` at or above `m_S * G[b]`; `d[j] = NB_b - S[j] * G[b]`; `eh[j] = T2[d >> (B - n)]`.
Block partials `A_b = sum eh * v8` and `l_b = sum eh` are exact integers below `2^24`, so
their conversion to fp32 is exact and any MMA order inside a block is legal. The fold over
blocks is a sequential fp32 recurrence in the committed order: rescale on a rising max,
skip exactly when the block is more than `S_WIN` doublings below, accumulate `O += f32(A_b)
* 2^-(s + e_v)` and `l += f32(l_b) * 2^-s`, both with round-to-nearest-even. Segments of
`SEG` blocks are folded the same way, which is what lets split-KV decode equal prefill.

What is surrendered: cross-block schedule invariance. LSSA-B8 is a schedule-committed
function, not an approximation of a schedule-invariant one; the impossibility of an exact
schedule-invariant single pass is proven in the filed specification, section 6.2.5.

### 2.3 LSSA-B, the generated-row rule

The filed per-block key-exponent variant: a 32-bit chain per block, the 8,192-entry
`FRAC` table (`(n, w) = (13, 15)`, sha `f472476c...`) generated from seven frozen integer
coefficients, and an order-free integer accumulation using `e_v_ref = max e_v[b]` with
exact left shifts. Append-only KV means a block's exponent is fixed once written.

### 2.4 Two rules, fixed by position

A row's rule is decided by whether its token is a prompt token or a generated token,
read from the engine's own scheduling array, never inferred from a query length. Both
rules read the same KV lattice. Gate E8-B8 proves they are different functions: prompt ids
differ between the two kernels, generated ids do not.

### 2.5 Toolchain laws (contract, not tuning)

No fast-math; `-ftz=false` appended after FA3's `--use_fast_math` (the IEEE product
`(2^24 - 1) * 2^-150` rounds up to `2^-126` while a flush gives 0, and the two differ in
the final division); `div.rn` asserted; TF32 off; a device-resident divisor tensor rather
than a python scalar (CUDA compiles `x / 127.0` to a reciprocal multiply, CPU does not);
no FMA contraction in RoPE. Each of these was found by a gate failing, not by inspection.

### 2.6 The non-attention stack (contract v2.1: the linear epilogue re-declared)

The exact non-attention ops of `demo/lockstep_fullstack.py` are unchanged (integer
RMSNorm with the per-row dynamic power-of-two scale, frozen-Q14 RoPE in fp32 with one
bf16 rounding, the SPEC 6.3.1 table SiLU, per-token int8 activations, per-channel int8
weights, `torch._int_mm` accumulation) except for the order of the two scale
multiplications in the linear epilogue. Contract v2.1 declares

    y8  = bf16( (fl32(acc) * ws[n]) * asc[m] )          weight scale first
    out = y8                                             (no bias)
    out = bf16( fl32(y8) + bias[n] )                     (bias: a second rounding)

where v2 declared `(fl32(acc) * asc[m]) * ws[n]` with the bias added before the single
rounding. The reason is measured, not aesthetic: the Hopper CUTLASS int8 GEMM in vLLM
(`cutlass_scaled_mm`) evaluates exactly the v2.1 order in its fused epilogue - 0
mismatches over 310,378,496 outputs on the gate_up shape and 0 on every other shape
(`results/t16m_gemm.json`) - and it runs the prefill-sized GEMMs at 1.4 to 1.7x the speed
of bf16 cuBLAS, where the v2 order forced a separate epilogue pass and lost that. Its
bias epilogue does NOT reproduce any simple declared form (86 to 1.4M mismatches of 37.7M
for every candidate), so the bias is declared as a separate rounded add and applied by the
consumer kernel on both routes. Two GEMM routes evaluate the declared function
bit-identically: `torch._int_mm` (cuBLASLt) with the epilogue inside the fused consumer
below `T16M_MCUT` rows (default 128, the measured crossover), and CUTLASS above it. The
v2.1 order is one bf16 ULP away from v2 on about 5e-6 of outputs; the quality battery is
re-run on it.

Under tensor parallelism two more rules are declared. The R4 rotation acts on each rank's
slice of the intermediate width, zero-padded to a multiple of 256 (the 7B at TP=4 holds
4,736 of the 18,944 lanes per rank, padded to 4,864; SiLU(0) x 0 = 0, so the padding is
exact), which makes R4 a per-rank rotation rather than the global one. And the partial
outputs of the row-parallel linears (o_proj, down_proj) are reduced as
`bf16( fl32(y_0) + fl32(y_1) + ... )` in rank order with one rounding, a pinned sum the
verifier reproduces from the ranks' bf16 partials, where stock lets NCCL choose the order
and rounds at every hop. The fused epilogue of o_proj cannot straddle the reduction, so at
TP > 1 that layer runs the epilogue, the reduction and the norm+quant as three ops.

## 3. Verification: what the verifier does and what this package runs

The prover's engine records, for every (sequence, step) at layers 0, 13 and 27, the
quantised `q8/k8/v8` with their exponents and the bf16 output row. The verifier
(`verify_rows.py`) rebuilds each sequence's KV prefix from the dumped pages, recomputes
every output row with the reference implementation (`lssab8_torch` for prompt rows,
`lssab_torch` for generated rows, with `mu`), and compares the int16 bit pattern of every
bf16 output. There is no tolerance; a single differing bit fails the row. The GPU run
checks every dumped row; the CPU run checks a sample of sequence-steps on the host with
no CUDA, which is the property that lets a challenger use any machine.

The same verifier was run on a second machine of a different architecture, an RTX 4090
(Ada) rented for the purpose: it reproduced every dumped row of the H100 engine bit for
bit on its GPU in 19.9 s and on its CPU in 906 s, and its CPU self-test reproduced the
python golden 28 of 28 (section 4, "Verifier"). That is the cross-device half of the
thesis: the prover ran on Hopper, the verifier on Ada and on two CPUs, no tolerance.

In the deployed protocol the same recomputation is what settles a dispute: the provider
commits per-layer leaves under a keyed BLAKE3 tree (section 4.7 prices it), a
post-commitment beacon selects rows, and a mismatch on any selected row is a fraud proof.
The dispute machinery (reveal-or-slash, the zkVM proof of a leaf transition) was drilled in
Phase 0 and is not re-run here.

The oracle hierarchy and every gate family this package runs are listed in
`IMPLEMENTATION.md` sections 2 and 3. The rule that governs all of them: the golden is
never edited; a mismatch is fixed in the kernel; no tolerance is ever widened.

## 4. Measured benchmarks

All cells below are **measured** in this session. Rows are labelled by engine mode and by
the Lockstep build they ran on; "m1" to "m11" are this session's decode iterations (section
7.2), each gated bit-identical before it was timed (the m9 files carry an "m6" tag from the
queue template that ran them). The stock arm is the same vLLM build
with `--attention-backend FLASH_ATTN` and the plugin inert, in the same mode.

<!-- BEGIN TABLES -->
### Single-stream TTFT and decode, CUDA graphs SHIPPED TP READING (SPEC 7.13): 72B, TP=4, whole stack, routed native reduction with the cut at 4,096 rows, both arms under NCCL_GRAPH_MIXING_SUPPORT=0, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 21.72 ms | 30.33 ms | 0.716x (stock time / ours) |
| ttft_p128_b8 | 77.83 ms | 101.60 ms | 0.766x (stock time / ours) |
| ttft_p2048_b1 | 149.53 ms | 182.85 ms | 0.818x (stock time / ours) |
| ttft_p2048_b8 | 1143.43 ms | 1139.93 ms | 1.003x (stock time / ours) |
| ttft_p8192_b1 | 605.96 ms | 605.45 ms | 1.001x (stock time / ours) |
| ttft_p8192_b8 | 4813.07 ms | 4811.61 ms | 1.000x (stock time / ours) |
| ttft_p8192_b4 | 2407.53 ms | 2405.15 ms | 1.001x (stock time / ours) |
| ttft_p32640_b1 | 2854.10 ms | 2992.34 ms | 0.954x (stock time / ours) |
| prefill_p128_b32 | 288.84 ms | 292.61 ms | 0.987x (stock time / ours) |
| decode_b1 | 63.3 tok/s | 62.2 tok/s | 0.982x (ours / stock) |
| decode_b8 | 490.1 tok/s | 483.6 tok/s | 0.987x (ours / stock) |
| decode_b32 | 1786.9 tok/s | 1716.5 tok/s | 0.961x (ours / stock) |

Seam: `None`, errflags[0]=?, varlen launches: None; vsh_violations=None

### Single-stream TTFT and decode, CUDA graphs DISCRIMINATOR (SPEC 7.13): 72B, TP=4, the environment alone (NCCL_GRAPH_MIXING_SUPPORT=0) with the earlier all-gather reduction, both arms under it, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 21.72 ms | 30.13 ms | 0.721x (stock time / ours) |
| ttft_p128_b8 | 77.83 ms | 99.85 ms | 0.779x (stock time / ours) |
| ttft_p2048_b1 | 149.53 ms | 182.23 ms | 0.821x (stock time / ours) |
| ttft_p2048_b8 | 1143.43 ms | 1333.86 ms | 0.857x (stock time / ours) |
| ttft_p8192_b1 | 605.96 ms | 705.37 ms | 0.859x (stock time / ours) |
| ttft_p8192_b8 | 4813.07 ms | 5589.71 ms | 0.861x (stock time / ours) |
| ttft_p8192_b4 | 2407.53 ms | 2797.77 ms | 0.861x (stock time / ours) |
| ttft_p32640_b1 | 2854.10 ms | 3382.61 ms | 0.844x (stock time / ours) |
| prefill_p128_b32 | 288.84 ms | 339.34 ms | 0.851x (stock time / ours) |
| decode_b1 | 63.3 tok/s | 62.3 tok/s | 0.984x (ours / stock) |
| decode_b8 | 490.1 tok/s | 483.6 tok/s | 0.987x (ours / stock) |
| decode_b32 | 1786.9 tok/s | 1722.6 tok/s | 0.964x (ours / stock) |

Seam: `None`, errflags[0]=?, varlen launches: None; vsh_violations=None

### Single-stream TTFT and decode, CUDA graphs SPEC 7.13: 72B, TP=4, whole stack, routed native reduction at the 512-row cut, both arms under NCCL_GRAPH_MIXING_SUPPORT=0 (W1 loss on mixed steps), medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 21.72 ms | 30.30 ms | 0.717x (stock time / ours) |
| ttft_p128_b8 | 77.83 ms | 95.32 ms | 0.816x (stock time / ours) |
| ttft_p2048_b1 | 149.53 ms | 160.45 ms | 0.932x (stock time / ours) |
| ttft_p2048_b8 | 1143.43 ms | 1152.22 ms | 0.992x (stock time / ours) |
| ttft_p8192_b1 | 605.96 ms | 606.87 ms | 0.998x (stock time / ours) |
| ttft_p8192_b8 | 4813.07 ms | 4811.89 ms | 1.000x (stock time / ours) |
| ttft_p8192_b4 | 2407.53 ms | 2415.65 ms | 0.997x (stock time / ours) |
| ttft_p32640_b1 | 2854.10 ms | 2993.07 ms | 0.954x (stock time / ours) |
| prefill_p128_b32 | 288.84 ms | 292.74 ms | 0.987x (stock time / ours) |
| decode_b1 | 63.3 tok/s | 62.2 tok/s | 0.982x (ours / stock) |
| decode_b8 | 490.1 tok/s | 482.8 tok/s | 0.985x (ours / stock) |
| decode_b32 | 1786.9 tok/s | 1721.2 tok/s | 0.963x (ours / stock) |

Seam: `None`, errflags[0]=?, varlen launches: None; vsh_violations=None

### Single-stream TTFT and decode, CUDA graphs REJECTED (SPEC 7.13): 72B, TP=4, the all-to-all reduction on vLLM's own pynccl communicator from 512 rows (prefill parity, decode 0.84x), 16-warp kernel, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 24.55 ms | 32.52 ms | 0.755x (stock time / ours) |
| ttft_p128_b8 | 80.71 ms | 89.60 ms | 0.901x (stock time / ours) |
| ttft_p2048_b1 | 152.00 ms | 158.44 ms | 0.959x (stock time / ours) |
| ttft_p2048_b8 | 1147.19 ms | 1140.31 ms | 1.006x (stock time / ours) |
| ttft_p8192_b1 | 607.01 ms | 606.96 ms | 1.000x (stock time / ours) |
| ttft_p8192_b8 | 4814.06 ms | 4822.17 ms | 0.998x (stock time / ours) |
| ttft_p8192_b4 | 2417.68 ms | 2409.25 ms | 1.003x (stock time / ours) |
| ttft_p32640_b1 | 2860.42 ms | 2995.99 ms | 0.955x (stock time / ours) |
| prefill_p128_b32 | 291.18 ms | 291.45 ms | 0.999x (stock time / ours) |
| decode_b1 | 63.3 tok/s | 53.3 tok/s | 0.842x (ours / stock) |
| decode_b8 | 489.7 tok/s | 395.5 tok/s | 0.808x (ours / stock) |
| decode_b32 | 1783.7 tok/s | 1482.0 tok/s | 0.831x (ours / stock) |

Seam: `None`, errflags[0]=?, varlen launches: None; vsh_violations=None

### Single-stream TTFT and decode, CUDA graphs MECHANISM TEST (SPEC 7.13): 72B, TP=4, routed c10d native reduction with NCCL_GRAPH_MIXING_SUPPORT=0, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 24.55 ms | 30.30 ms | 0.810x (stock time / ours) |
| ttft_p128_b8 | 80.71 ms | 95.32 ms | 0.847x (stock time / ours) |
| ttft_p2048_b1 | 152.00 ms | 160.45 ms | 0.947x (stock time / ours) |
| ttft_p2048_b8 | 1147.19 ms | 1152.22 ms | 0.996x (stock time / ours) |
| ttft_p8192_b1 | 607.01 ms | 606.87 ms | 1.000x (stock time / ours) |
| ttft_p8192_b8 | 4814.06 ms | 4811.89 ms | 1.000x (stock time / ours) |
| ttft_p8192_b4 | 2417.68 ms | 2415.65 ms | 1.001x (stock time / ours) |
| ttft_p32640_b1 | 2860.42 ms | 2993.07 ms | 0.956x (stock time / ours) |
| prefill_p128_b32 | 291.18 ms | 292.74 ms | 0.995x (stock time / ours) |
| decode_b1 | 63.3 tok/s | 62.2 tok/s | 0.983x (ours / stock) |
| decode_b8 | 489.7 tok/s | 482.8 tok/s | 0.986x (ours / stock) |
| decode_b32 | 1783.7 tok/s | 1721.2 tok/s | 0.965x (ours / stock) |

Seam: `None`, errflags[0]=?, varlen launches: None; vsh_violations=None

### Single-stream TTFT and decode, CUDA graphs 72B TP=4 DISCRIMINATOR (SPEC 7.13): native reduction armed but its cut set to 1e6 rows (never taken), old kernel, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 24.55 ms | 31.01 ms | 0.792x (stock time / ours) |
| ttft_p128_b8 | 80.71 ms | 99.17 ms | 0.814x (stock time / ours) |
| ttft_p2048_b1 | 152.00 ms | 179.35 ms | 0.847x (stock time / ours) |
| ttft_p2048_b8 | 1147.19 ms | 1337.71 ms | 0.858x (stock time / ours) |
| ttft_p8192_b1 | 607.01 ms | 702.50 ms | 0.864x (stock time / ours) |
| ttft_p8192_b8 | 4814.06 ms | 5590.55 ms | 0.861x (stock time / ours) |
| ttft_p8192_b4 | 2417.68 ms | 2799.98 ms | 0.863x (stock time / ours) |
| ttft_p32640_b1 | 2860.42 ms | 3387.32 ms | 0.844x (stock time / ours) |
| prefill_p128_b32 | 291.18 ms | 338.02 ms | 0.861x (stock time / ours) |
| decode_b1 | 63.3 tok/s | 60.8 tok/s | 0.961x (ours / stock) |
| decode_b8 | 489.7 tok/s | 472.0 tok/s | 0.964x (ours / stock) |
| decode_b32 | 1783.7 tok/s | 1685.6 tok/s | 0.945x (ours / stock) |

Seam: `None`, errflags[0]=?, varlen launches: None; vsh_violations=None

### Single-stream TTFT and decode, CUDA graphs 72B TP=4 with the PURE pipelined decode kernel (SPEC 7.14; short units, no merged fallback), old reduction, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 24.55 ms | 31.11 ms | 0.789x (stock time / ours) |
| ttft_p128_b8 | 80.71 ms | 99.41 ms | 0.812x (stock time / ours) |
| ttft_p2048_b1 | 152.00 ms | 180.75 ms | 0.841x (stock time / ours) |
| ttft_p2048_b8 | 1147.19 ms | 1335.89 ms | 0.859x (stock time / ours) |
| ttft_p8192_b1 | 607.01 ms | 705.41 ms | 0.860x (stock time / ours) |
| ttft_p8192_b8 | 4814.06 ms | 5587.80 ms | 0.862x (stock time / ours) |
| ttft_p8192_b4 | 2417.68 ms | 2794.24 ms | 0.865x (stock time / ours) |
| ttft_p32640_b1 | 2860.42 ms | 3378.78 ms | 0.847x (stock time / ours) |
| prefill_p128_b32 | 291.18 ms | 337.76 ms | 0.862x (stock time / ours) |
| decode_b1 | 63.3 tok/s | 60.3 tok/s | 0.952x (ours / stock) |
| decode_b8 | 489.7 tok/s | 466.3 tok/s | 0.952x (ours / stock) |
| decode_b32 | 1783.7 tok/s | 1663.2 tok/s | 0.932x (ours / stock) |

Seam: `None`, errflags[0]=?, varlen launches: None; vsh_violations=None

### Single-stream TTFT and decode, CUDA graphs 72B TP=4 CONTROL (SPEC 7.13): the earlier all-gather reduction on the current build, 16-warp kernel, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 24.55 ms | 31.37 ms | 0.783x (stock time / ours) |
| ttft_p128_b8 | 80.71 ms | 99.32 ms | 0.813x (stock time / ours) |
| ttft_p2048_b1 | 152.00 ms | 179.56 ms | 0.847x (stock time / ours) |
| ttft_p2048_b8 | 1147.19 ms | 1336.66 ms | 0.858x (stock time / ours) |
| ttft_p8192_b1 | 607.01 ms | 703.21 ms | 0.863x (stock time / ours) |
| ttft_p8192_b8 | 4814.06 ms | 5588.17 ms | 0.861x (stock time / ours) |
| ttft_p8192_b4 | 2417.68 ms | 2794.77 ms | 0.865x (stock time / ours) |
| ttft_p32640_b1 | 2860.42 ms | 3380.28 ms | 0.846x (stock time / ours) |
| prefill_p128_b32 | 291.18 ms | 338.26 ms | 0.861x (stock time / ours) |
| decode_b1 | 63.3 tok/s | 60.8 tok/s | 0.961x (ours / stock) |
| decode_b8 | 489.7 tok/s | 470.8 tok/s | 0.961x (ours / stock) |
| decode_b32 | 1783.7 tok/s | 1683.7 tok/s | 0.944x (ours / stock) |

Seam: `None`, errflags[0]=?, varlen launches: None; vsh_violations=None

### Single-stream TTFT and decode, CUDA graphs T16v + split target 2 (SPEC 7.15): whole verified stack, pipelined decode kernel, LOCKSTEP_LSSAB_SPLIT_CTA=2, no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.46 ms | 9.70 ms | 1.182x (stock time / ours) |
| ttft_p128_b8 | 27.39 ms | 23.93 ms | 1.145x (stock time / ours) |
| ttft_p2048_b1 | 53.29 ms | 42.39 ms | 1.257x (stock time / ours) |
| ttft_p2048_b8 | 375.21 ms | 310.30 ms | 1.209x (stock time / ours) |
| ttft_p8192_b1 | 213.40 ms | 174.48 ms | 1.223x (stock time / ours) |
| ttft_p8192_b8 | 1601.50 ms | 1400.39 ms | 1.144x (stock time / ours) |
| ttft_p8192_b4 | 805.75 ms | 700.13 ms | 1.151x (stock time / ours) |
| ttft_p32640_b1 | 1108.19 ms | 1046.89 ms | 1.059x (stock time / ours) |
| prefill_p128_b32 | 96.40 ms | 79.51 ms | 1.212x (stock time / ours) |
| decode_b1 | 165.1 tok/s | 165.1 tok/s | 1.000x (ours / stock) |
| decode_b8 | 1290.8 tok/s | 1285.0 tok/s | 0.995x (ours / stock) |
| decode_b32 | 5200.5 tok/s | 4713.7 tok/s | 0.906x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs split target 2 with the 16-warp kernel (SPEC 7.15 policy A/B), medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.46 ms | 9.61 ms | 1.193x (stock time / ours) |
| ttft_p128_b8 | 27.39 ms | 24.51 ms | 1.118x (stock time / ours) |
| ttft_p2048_b1 | 53.29 ms | 42.97 ms | 1.240x (stock time / ours) |
| ttft_p2048_b8 | 375.21 ms | 310.36 ms | 1.209x (stock time / ours) |
| ttft_p8192_b1 | 213.40 ms | 174.93 ms | 1.220x (stock time / ours) |
| ttft_p8192_b8 | 1601.50 ms | 1395.17 ms | 1.148x (stock time / ours) |
| ttft_p8192_b4 | 805.75 ms | 699.26 ms | 1.152x (stock time / ours) |
| ttft_p32640_b1 | 1108.19 ms | 1046.37 ms | 1.059x (stock time / ours) |
| prefill_p128_b32 | 96.40 ms | 79.96 ms | 1.206x (stock time / ours) |
| decode_b1 | 165.1 tok/s | 168.3 tok/s | 1.019x (ours / stock) |
| decode_b8 | 1290.8 tok/s | 1299.8 tok/s | 1.007x (ours / stock) |
| decode_b32 | 5200.5 tok/s | 4693.9 tok/s | 0.903x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs split target 4 with the 16-warp kernel (SPEC 7.15 policy A/B), medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.46 ms | 9.51 ms | 1.205x (stock time / ours) |
| ttft_p128_b8 | 27.39 ms | 24.39 ms | 1.123x (stock time / ours) |
| ttft_p2048_b1 | 53.29 ms | 42.53 ms | 1.253x (stock time / ours) |
| ttft_p2048_b8 | 375.21 ms | 310.57 ms | 1.208x (stock time / ours) |
| ttft_p8192_b1 | 213.40 ms | 174.81 ms | 1.221x (stock time / ours) |
| ttft_p8192_b8 | 1601.50 ms | 1403.44 ms | 1.141x (stock time / ours) |
| ttft_p8192_b4 | 805.75 ms | 704.53 ms | 1.144x (stock time / ours) |
| ttft_p32640_b1 | 1108.19 ms | 1047.73 ms | 1.058x (stock time / ours) |
| prefill_p128_b32 | 96.40 ms | 79.72 ms | 1.209x (stock time / ours) |
| decode_b1 | 165.1 tok/s | 171.8 tok/s | 1.040x (ours / stock) |
| decode_b8 | 1290.8 tok/s | 1333.5 tok/s | 1.033x (ours / stock) |
| decode_b32 | 5200.5 tok/s | 4773.0 tok/s | 0.918x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs T16v (SPEC 7.14): END-TO-END VERIFIED STACK with the two-group PIPELINED decode kernel, compile cache off, one stack per process at GMU 0.85, no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.46 ms | 9.63 ms | 1.190x (stock time / ours) |
| ttft_p128_b8 | 27.39 ms | 24.68 ms | 1.110x (stock time / ours) |
| ttft_p2048_b1 | 53.29 ms | 42.42 ms | 1.256x (stock time / ours) |
| ttft_p2048_b8 | 375.21 ms | 312.62 ms | 1.200x (stock time / ours) |
| ttft_p8192_b1 | 213.40 ms | 175.65 ms | 1.215x (stock time / ours) |
| ttft_p8192_b8 | 1601.50 ms | 1404.01 ms | 1.141x (stock time / ours) |
| ttft_p8192_b4 | 805.75 ms | 700.86 ms | 1.150x (stock time / ours) |
| ttft_p32640_b1 | 1108.19 ms | 1047.03 ms | 1.058x (stock time / ours) |
| prefill_p128_b32 | 96.40 ms | 79.80 ms | 1.208x (stock time / ours) |
| decode_b1 | 165.1 tok/s | 166.3 tok/s | 1.007x (ours / stock) |
| decode_b8 | 1290.8 tok/s | 1293.7 tok/s | 1.002x (ours / stock) |
| decode_b32 | 5200.5 tok/s | 4759.9 tok/s | 0.915x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs T16s (SPEC 7.13): 72B, TP=4, whole verifiable stack, 16-warp decode kernel + the NATIVE all-to-all reduction from 512 rows (vLLM all-gather below), no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 24.55 ms | 32.55 ms | 0.754x (stock time / ours) |
| ttft_p128_b8 | 80.71 ms | 89.46 ms | 0.902x (stock time / ours) |
| ttft_p2048_b1 | 152.00 ms | 157.48 ms | 0.965x (stock time / ours) |
| ttft_p2048_b8 | 1147.19 ms | 1138.01 ms | 1.008x (stock time / ours) |
| ttft_p8192_b1 | 607.01 ms | 605.02 ms | 1.003x (stock time / ours) |
| ttft_p8192_b8 | 4814.06 ms | 4806.95 ms | 1.001x (stock time / ours) |
| ttft_p8192_b4 | 2417.68 ms | 2404.73 ms | 1.005x (stock time / ours) |
| ttft_p32640_b1 | 2860.42 ms | 2991.38 ms | 0.956x (stock time / ours) |
| prefill_p128_b32 | 291.18 ms | 292.82 ms | 0.994x (stock time / ours) |
| decode_b1 | 63.3 tok/s | 58.2 tok/s | 0.920x (ours / stock) |
| decode_b8 | 489.7 tok/s | 451.4 tok/s | 0.922x (ours / stock) |
| decode_b32 | 1783.7 tok/s | 1616.2 tok/s | 0.906x (ours / stock) |

Seam: `None`, errflags[0]=?, varlen launches: None; vsh_violations=None

### Single-stream TTFT and decode, CUDA graphs SUPERSEDED (SPEC 7.13): 72B, TP=4, native reduction on EVERY row count (the c10d all-gather in the decode graph cost 4 decode points), no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 24.55 ms | 33.03 ms | 0.743x (stock time / ours) |
| ttft_p128_b8 | 80.71 ms | 90.64 ms | 0.890x (stock time / ours) |
| ttft_p2048_b1 | 152.00 ms | 158.66 ms | 0.958x (stock time / ours) |
| ttft_p2048_b8 | 1147.19 ms | 1137.84 ms | 1.008x (stock time / ours) |
| ttft_p8192_b1 | 607.01 ms | 606.54 ms | 1.001x (stock time / ours) |
| ttft_p8192_b8 | 4814.06 ms | 4802.86 ms | 1.002x (stock time / ours) |
| ttft_p8192_b4 | 2417.68 ms | 2399.98 ms | 1.007x (stock time / ours) |
| ttft_p32640_b1 | 2860.42 ms | 2989.11 ms | 0.957x (stock time / ours) |
| prefill_p128_b32 | 291.18 ms | 291.00 ms | 1.001x (stock time / ours) |
| decode_b1 | 63.3 tok/s | 58.2 tok/s | 0.919x (ours / stock) |
| decode_b8 | 489.7 tok/s | 453.3 tok/s | 0.926x (ours / stock) |
| decode_b32 | 1783.7 tok/s | 1621.1 tok/s | 0.909x (ours / stock) |

Seam: `None`, errflags[0]=?, varlen launches: None; vsh_violations=None

### Single-stream TTFT and decode, CUDA graphs SHIPPED KERNEL (T16r, SPEC 7.12): Qwen2.5-72B-Instruct, TP=4 on 4x H100, the WHOLE VERIFIABLE STACK with the 16-warp decode kernel (per-rank R4 padding, pinned rank-order all-gather reduction), no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 24.55 ms | 31.22 ms | 0.786x (stock time / ours) |
| ttft_p128_b8 | 80.71 ms | 99.67 ms | 0.810x (stock time / ours) |
| ttft_p2048_b1 | 152.00 ms | 179.69 ms | 0.846x (stock time / ours) |
| ttft_p2048_b8 | 1147.19 ms | 1334.01 ms | 0.860x (stock time / ours) |
| ttft_p8192_b1 | 607.01 ms | 705.68 ms | 0.860x (stock time / ours) |
| ttft_p8192_b8 | 4814.06 ms | 5589.48 ms | 0.861x (stock time / ours) |
| ttft_p8192_b4 | 2417.68 ms | 2795.67 ms | 0.865x (stock time / ours) |
| ttft_p32640_b1 | 2860.42 ms | 3382.20 ms | 0.846x (stock time / ours) |
| prefill_p128_b32 | 291.18 ms | 339.54 ms | 0.858x (stock time / ours) |
| decode_b1 | 63.3 tok/s | 60.7 tok/s | 0.959x (ours / stock) |
| decode_b8 | 489.7 tok/s | 470.8 tok/s | 0.961x (ours / stock) |
| decode_b32 | 1783.7 tok/s | 1681.4 tok/s | 0.943x (ours / stock) |

Seam: `None`, errflags[0]=?, varlen launches: None; vsh_violations=None

### Single-stream TTFT and decode, CUDA graphs SHIPPED KERNEL (T16r, SPEC 7.12): END-TO-END VERIFIED STACK with the 16-warp decode kernel, compile cache off, one stack per process at GMU 0.85, no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.46 ms | 9.56 ms | 1.198x (stock time / ours) |
| ttft_p128_b8 | 27.39 ms | 24.27 ms | 1.129x (stock time / ours) |
| ttft_p2048_b1 | 53.29 ms | 42.65 ms | 1.250x (stock time / ours) |
| ttft_p2048_b8 | 375.21 ms | 308.31 ms | 1.217x (stock time / ours) |
| ttft_p8192_b1 | 213.40 ms | 174.19 ms | 1.225x (stock time / ours) |
| ttft_p8192_b8 | 1601.50 ms | 1393.43 ms | 1.149x (stock time / ours) |
| ttft_p8192_b4 | 805.75 ms | 699.41 ms | 1.152x (stock time / ours) |
| ttft_p32640_b1 | 1108.19 ms | 1042.27 ms | 1.063x (stock time / ours) |
| prefill_p128_b32 | 96.40 ms | 79.86 ms | 1.207x (stock time / ours) |
| decode_b1 | 165.1 tok/s | 166.2 tok/s | 1.006x (ours / stock) |
| decode_b8 | 1290.8 tok/s | 1288.8 tok/s | 0.998x (ours / stock) |
| decode_b32 | 5200.5 tok/s | 4671.0 tok/s | 0.898x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs NOT SHIPPED: 72B, TP=4, whole stack with the python-wrapped routed reduction (regressed by its dispatch cost; SPEC 7.11), one stack per process, no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 24.55 ms | 36.23 ms | 0.678x (stock time / ours) |
| ttft_p128_b8 | 80.71 ms | 108.30 ms | 0.745x (stock time / ours) |
| ttft_p2048_b1 | 152.00 ms | 189.00 ms | 0.804x (stock time / ours) |
| ttft_p2048_b8 | 1147.19 ms | 1369.58 ms | 0.838x (stock time / ours) |
| ttft_p8192_b1 | 607.01 ms | 729.21 ms | 0.832x (stock time / ours) |
| ttft_p8192_b8 | 4814.06 ms | 5788.78 ms | 0.832x (stock time / ours) |
| ttft_p8192_b4 | 2417.68 ms | 2896.00 ms | 0.835x (stock time / ours) |
| ttft_p32640_b1 | 2860.42 ms | 3598.53 ms | 0.795x (stock time / ours) |
| prefill_p128_b32 | 291.18 ms | 345.27 ms | 0.843x (stock time / ours) |
| decode_b1 | 63.3 tok/s | 51.6 tok/s | 0.815x (ours / stock) |
| decode_b8 | 489.7 tok/s | 401.9 tok/s | 0.821x (ours / stock) |
| decode_b32 | 1783.7 tok/s | 1445.3 tok/s | 0.810x (ours / stock) |

Seam: `None`, errflags[0]=?, varlen launches: None; vsh_violations=None

### Single-stream TTFT and decode, CUDA graphs Qwen2.5-72B-Instruct, TP=4 on 4x H100: the WHOLE VERIFIABLE STACK (per-rank R4, pinned TP reduction, C++ ops), one stack per process, no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 24.55 ms | 30.79 ms | 0.797x (stock time / ours) |
| ttft_p128_b8 | 80.71 ms | 99.28 ms | 0.813x (stock time / ours) |
| ttft_p2048_b1 | 152.00 ms | 179.16 ms | 0.848x (stock time / ours) |
| ttft_p2048_b8 | 1147.19 ms | 1331.20 ms | 0.862x (stock time / ours) |
| ttft_p8192_b1 | 607.01 ms | 702.09 ms | 0.865x (stock time / ours) |
| ttft_p8192_b8 | 4814.06 ms | 5582.94 ms | 0.862x (stock time / ours) |
| ttft_p8192_b4 | 2417.68 ms | 2790.66 ms | 0.866x (stock time / ours) |
| ttft_p32640_b1 | 2860.42 ms | 3376.92 ms | 0.847x (stock time / ours) |
| prefill_p128_b32 | 291.18 ms | 338.40 ms | 0.860x (stock time / ours) |
| decode_b1 | 63.3 tok/s | 60.3 tok/s | 0.952x (ours / stock) |
| decode_b8 | 489.7 tok/s | 464.5 tok/s | 0.948x (ours / stock) |
| decode_b32 | 1783.7 tok/s | 1645.2 tok/s | 0.922x (ours / stock) |

Seam: `None`, errflags[0]=?, varlen launches: None; vsh_violations=None

### Single-stream TTFT and decode, CUDA graphs END-TO-END VERIFIED STACK, FUSED, C++ ops, COMPILE CACHE OFF (T16m v2.3, the valid measurement), one stack per process at GMU 0.85, no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.46 ms | 9.60 ms | 1.193x (stock time / ours) |
| ttft_p128_b8 | 27.39 ms | 24.39 ms | 1.123x (stock time / ours) |
| ttft_p2048_b1 | 53.29 ms | 42.59 ms | 1.251x (stock time / ours) |
| ttft_p2048_b8 | 375.21 ms | 309.62 ms | 1.212x (stock time / ours) |
| ttft_p8192_b1 | 213.40 ms | 174.75 ms | 1.221x (stock time / ours) |
| ttft_p8192_b8 | 1601.50 ms | 1400.73 ms | 1.143x (stock time / ours) |
| ttft_p8192_b4 | 805.75 ms | 699.77 ms | 1.151x (stock time / ours) |
| ttft_p32640_b1 | 1108.19 ms | 1044.57 ms | 1.061x (stock time / ours) |
| prefill_p128_b32 | 96.40 ms | 79.86 ms | 1.207x (stock time / ours) |
| decode_b1 | 165.1 tok/s | 165.4 tok/s | 1.002x (ours / stock) |
| decode_b8 | 1290.8 tok/s | 1286.4 tok/s | 0.997x (ours / stock) |
| decode_b32 | 5200.5 tok/s | 4706.8 tok/s | 0.905x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs fused C++ ops, 7B model (this model had no stale cache issue: same numbers as t16m24), one stack per process at GMU 0.85, no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.46 ms | 11.99 ms | 0.956x (stock time / ours) |
| ttft_p128_b8 | 27.39 ms | 26.92 ms | 1.017x (stock time / ours) |
| ttft_p2048_b1 | 53.29 ms | 44.74 ms | 1.191x (stock time / ours) |
| ttft_p2048_b8 | 375.21 ms | 312.78 ms | 1.200x (stock time / ours) |
| ttft_p8192_b1 | 213.40 ms | 177.63 ms | 1.201x (stock time / ours) |
| ttft_p8192_b8 | 1601.50 ms | 1413.68 ms | 1.133x (stock time / ours) |
| ttft_p8192_b4 | 805.75 ms | 704.80 ms | 1.143x (stock time / ours) |
| ttft_p32640_b1 | 1108.19 ms | 1053.03 ms | 1.052x (stock time / ours) |
| prefill_p128_b32 | 96.40 ms | 81.86 ms | 1.178x (stock time / ours) |
| decode_b1 | 165.1 tok/s | 166.1 tok/s | 1.006x (ours / stock) |
| decode_b8 | 1290.8 tok/s | 1291.3 tok/s | 1.000x (ours / stock) |
| decode_b32 | 5200.5 tok/s | 4727.3 tok/s | 0.909x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs END-TO-END VERIFIED STACK, FUSED (T16m v2.2: fused glue, routed int8 GEMM, contract v2.1 epilogue), one stack per process at GMU 0.85, no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.46 ms | 12.02 ms | 0.953x (stock time / ours) |
| ttft_p128_b8 | 27.39 ms | 26.93 ms | 1.017x (stock time / ours) |
| ttft_p2048_b1 | 53.29 ms | 44.87 ms | 1.188x (stock time / ours) |
| ttft_p2048_b8 | 375.21 ms | 315.07 ms | 1.191x (stock time / ours) |
| ttft_p8192_b1 | 213.40 ms | 177.05 ms | 1.205x (stock time / ours) |
| ttft_p8192_b8 | 1601.50 ms | 1415.92 ms | 1.131x (stock time / ours) |
| ttft_p8192_b4 | 805.75 ms | 707.67 ms | 1.139x (stock time / ours) |
| ttft_p32640_b1 | 1108.19 ms | 1057.75 ms | 1.048x (stock time / ours) |
| prefill_p128_b32 | 96.40 ms | 82.34 ms | 1.171x (stock time / ours) |
| decode_b1 | 165.1 tok/s | 166.0 tok/s | 1.005x (ours / stock) |
| decode_b8 | 1290.8 tok/s | 1289.3 tok/s | 0.999x (ours / stock) |
| decode_b32 | 5200.5 tok/s | 4714.4 tok/s | 0.907x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs attention-only control for the row above: same session, same stock arm, one stack per process at GMU 0.85, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.46 ms | 13.50 ms | 0.849x (stock time / ours) |
| ttft_p128_b8 | 27.39 ms | 29.88 ms | 0.917x (stock time / ours) |
| ttft_p2048_b1 | 53.29 ms | 55.51 ms | 0.960x (stock time / ours) |
| ttft_p2048_b8 | 375.21 ms | 387.68 ms | 0.968x (stock time / ours) |
| ttft_p8192_b1 | 213.40 ms | 217.42 ms | 0.982x (stock time / ours) |
| ttft_p8192_b8 | 1601.50 ms | 1681.11 ms | 0.953x (stock time / ours) |
| ttft_p8192_b4 | 805.75 ms | 846.85 ms | 0.951x (stock time / ours) |
| ttft_p32640_b1 | 1108.19 ms | 1197.22 ms | 0.926x (stock time / ours) |
| prefill_p128_b32 | 96.40 ms | 99.87 ms | 0.965x (stock time / ours) |
| decode_b1 | 165.1 tok/s | 158.5 tok/s | 0.959x (ours / stock) |
| decode_b8 | 1290.8 tok/s | 1243.2 tok/s | 0.963x (ours / stock) |
| decode_b32 | 5200.5 tok/s | 4728.1 tok/s | 0.909x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs SUPERSEDED (the two-engine harness never armed the exact stack: attention-only numbers mislabelled as full stack; kept as evidence), medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.38 ms | 13.33 ms | 0.854x (stock time / ours) |
| ttft_p128_b8 | 27.31 ms | 30.66 ms | 0.891x (stock time / ours) |
| ttft_p2048_b1 | 51.93 ms | 55.36 ms | 0.938x (stock time / ours) |
| ttft_p2048_b8 | 377.37 ms | 386.97 ms | 0.975x (stock time / ours) |
| ttft_p8192_b1 | 213.33 ms | 217.19 ms | 0.982x (stock time / ours) |
| ttft_p8192_b8 | 1600.07 ms | 1682.97 ms | 0.951x (stock time / ours) |
| ttft_p8192_b4 | 809.83 ms | 846.78 ms | 0.956x (stock time / ours) |
| ttft_p32640_b1 | 1112.44 ms | 1195.91 ms | 0.930x (stock time / ours) |
| prefill_p128_b32 | 98.43 ms | 108.50 ms | 0.907x (stock time / ours) |
| decode_b1 | 170.1 tok/s | 161.5 tok/s | 0.950x (ours / stock) |
| decode_b8 | 1315.9 tok/s | 1266.5 tok/s | 0.963x (ours / stock) |
| decode_b32 | 5194.1 tok/s | 4833.0 tok/s | 0.930x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs SUPERSEDED (same harness defect; attention-only numbers), medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.29 ms | 13.46 ms | 0.839x (stock time / ours) |
| ttft_p128_b8 | 27.64 ms | 30.63 ms | 0.902x (stock time / ours) |
| ttft_p2048_b1 | 52.16 ms | 56.10 ms | 0.930x (stock time / ours) |
| ttft_p2048_b8 | 372.80 ms | 386.28 ms | 0.965x (stock time / ours) |
| ttft_p8192_b1 | 211.72 ms | 217.07 ms | 0.975x (stock time / ours) |
| ttft_p8192_b8 | 1603.85 ms | 1680.98 ms | 0.954x (stock time / ours) |
| ttft_p8192_b4 | 810.57 ms | 847.43 ms | 0.957x (stock time / ours) |
| ttft_p32640_b1 | 1105.65 ms | 1195.02 ms | 0.925x (stock time / ours) |
| prefill_p128_b32 | 97.60 ms | 108.79 ms | 0.897x (stock time / ours) |
| decode_b1 | 164.7 tok/s | 158.2 tok/s | 0.961x (ours / stock) |
| decode_b8 | 1287.5 tok/s | 1240.5 tok/s | 0.964x (ours / stock) |
| decode_b32 | 5069.2 tok/s | 4727.4 tok/s | 0.933x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs Qwen2.5-72B-Instruct (64 heads / 8 kv-heads), TP=4 on 4x H100: contract v2 v8 kernel, no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 24.55 ms | 30.06 ms | 0.817x (stock time / ours) |
| ttft_p128_b8 | 80.71 ms | 84.95 ms | 0.950x (stock time / ours) |
| ttft_p2048_b1 | 152.00 ms | 157.23 ms | 0.967x (stock time / ours) |
| ttft_p2048_b8 | 1147.19 ms | 1176.39 ms | 0.975x (stock time / ours) |
| ttft_p8192_b1 | 607.01 ms | 616.48 ms | 0.985x (stock time / ours) |
| ttft_p8192_b8 | 4814.06 ms | 4894.48 ms | 0.984x (stock time / ours) |
| ttft_p8192_b4 | 2417.68 ms | 2453.78 ms | 0.985x (stock time / ours) |
| ttft_p32640_b1 | 2860.42 ms | 2993.09 ms | 0.956x (stock time / ours) |
| prefill_p128_b32 | 291.18 ms | 300.52 ms | 0.969x (stock time / ours) |
| decode_b1 | 63.3 tok/s | 60.7 tok/s | 0.959x (ours / stock) |
| decode_b8 | 489.7 tok/s | 466.9 tok/s | 0.953x (ours / stock) |
| decode_b32 | 1783.7 tok/s | 1686.0 tok/s | 0.945x (ours / stock) |

Seam: `None`, errflags[0]=?, varlen launches: None; vsh_violations=None

### Single-stream TTFT and decode, CUDA graphs Qwen2.5-14B-Instruct (40 heads / 8 kv-heads): contract v2 v8 kernel (head groups), no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 17.31 ms | 20.33 ms | 0.852x (stock time / ours) |
| ttft_p128_b8 | 51.23 ms | 55.74 ms | 0.919x (stock time / ours) |
| ttft_p2048_b1 | 99.95 ms | 106.35 ms | 0.940x (stock time / ours) |
| ttft_p2048_b8 | 728.84 ms | 760.74 ms | 0.958x (stock time / ours) |
| ttft_p8192_b1 | 416.82 ms | 438.54 ms | 0.950x (stock time / ours) |
| ttft_p8192_b8 | 3239.80 ms | 3409.78 ms | 0.950x (stock time / ours) |
| ttft_p8192_b4 | 1618.65 ms | 1717.36 ms | 0.943x (stock time / ours) |
| ttft_p32640_b1 | 2301.13 ms | 2509.68 ms | 0.917x (stock time / ours) |
| prefill_p128_b32 | 186.96 ms | 197.73 ms | 0.946x (stock time / ours) |
| decode_b1 | 87.3 tok/s | 84.7 tok/s | 0.971x (ours / stock) |
| decode_b8 | 682.1 tok/s | 663.5 tok/s | 0.973x (ours / stock) |
| decode_b32 | 2600.2 tok/s | 2469.1 tok/s | 0.950x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 7008; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs SUPERSEDED: filed as the end-to-end verified stack, but the two-engine harness never armed the exact stack (no armed_at_load banner in its log); these are attention-only numbers. The armed measurement is the t16m22 table (T16L_ARM=1), no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.00 ms | 12.89 ms | 0.853x (stock time / ours) |
| ttft_p128_b8 | 27.51 ms | 30.79 ms | 0.893x (stock time / ours) |
| ttft_p2048_b1 | 50.66 ms | 55.02 ms | 0.921x (stock time / ours) |
| ttft_p2048_b8 | 374.35 ms | 388.02 ms | 0.965x (stock time / ours) |
| ttft_p8192_b1 | 208.91 ms | 214.49 ms | 0.974x (stock time / ours) |
| ttft_p8192_b8 | 1600.21 ms | 1675.91 ms | 0.955x (stock time / ours) |
| ttft_p8192_b4 | 809.23 ms | 841.38 ms | 0.962x (stock time / ours) |
| ttft_p32640_b1 | 1106.40 ms | 1189.72 ms | 0.930x (stock time / ours) |
| prefill_p128_b32 | 96.09 ms | 106.50 ms | 0.902x (stock time / ours) |
| decode_b1 | 170.0 tok/s | 161.8 tok/s | 0.952x (ours / stock) |
| decode_b8 | 1317.9 tok/s | 1270.8 tok/s | 0.964x (ours / stock) |
| decode_b32 | 5199.3 tok/s | 4847.1 tok/s | 0.932x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs SHIPPED: CONTRACT v2 v6 (v4 kernel + prep fused into lsb_dec8 on decode-only steps, split cap 64): no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.12 ms | 13.02 ms | 0.854x (stock time / ours) |
| ttft_p128_b8 | 27.30 ms | 29.87 ms | 0.914x (stock time / ours) |
| ttft_p2048_b1 | 51.99 ms | 56.09 ms | 0.927x (stock time / ours) |
| ttft_p2048_b8 | 373.92 ms | 387.54 ms | 0.965x (stock time / ours) |
| ttft_p8192_b1 | 214.51 ms | 217.55 ms | 0.986x (stock time / ours) |
| ttft_p8192_b8 | 1601.08 ms | 1676.84 ms | 0.955x (stock time / ours) |
| ttft_p8192_b4 | 807.00 ms | 843.34 ms | 0.957x (stock time / ours) |
| ttft_p32640_b1 | 1106.66 ms | 1190.91 ms | 0.929x (stock time / ours) |
| prefill_p128_b32 | 96.79 ms | 108.03 ms | 0.896x (stock time / ours) |
| decode_b1 | 164.8 tok/s | 158.6 tok/s | 0.962x (ours / stock) |
| decode_b8 | 1285.5 tok/s | 1246.8 tok/s | 0.970x (ours / stock) |
| decode_b32 | 5067.2 tok/s | 4730.2 tok/s | 0.934x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs CONTRACT v2 v7 (v6 + half-block staging, ~48 KB shared memory; evidence only, slower): no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.44 ms | 13.52 ms | 0.846x (stock time / ours) |
| ttft_p128_b8 | 27.91 ms | 30.68 ms | 0.910x (stock time / ours) |
| ttft_p2048_b1 | 51.64 ms | 55.59 ms | 0.929x (stock time / ours) |
| ttft_p2048_b8 | 375.57 ms | 384.50 ms | 0.977x (stock time / ours) |
| ttft_p8192_b1 | 209.38 ms | 217.82 ms | 0.961x (stock time / ours) |
| ttft_p8192_b8 | 1601.51 ms | 1676.29 ms | 0.955x (stock time / ours) |
| ttft_p8192_b4 | 810.14 ms | 849.65 ms | 0.954x (stock time / ours) |
| ttft_p32640_b1 | 1114.96 ms | 1191.72 ms | 0.936x (stock time / ours) |
| prefill_p128_b32 | 96.80 ms | 106.68 ms | 0.907x (stock time / ours) |
| decode_b1 | 164.7 tok/s | 159.1 tok/s | 0.966x (ours / stock) |
| decode_b8 | 1285.3 tok/s | 1251.3 tok/s | 0.974x (ours / stock) |
| decode_b32 | 5059.7 tok/s | 4630.7 tok/s | 0.915x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs CONTRACT v2 FINAL (v4 kernel, split cap 64, sub-segment units up to 128k): no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.38 ms | 13.48 ms | 0.845x (stock time / ours) |
| ttft_p128_b8 | 28.62 ms | 30.88 ms | 0.927x (stock time / ours) |
| ttft_p2048_b1 | 52.72 ms | 56.98 ms | 0.925x (stock time / ours) |
| ttft_p2048_b8 | 380.02 ms | 384.56 ms | 0.988x (stock time / ours) |
| ttft_p8192_b1 | 210.44 ms | 217.53 ms | 0.967x (stock time / ours) |
| ttft_p8192_b8 | 1604.60 ms | 1676.23 ms | 0.957x (stock time / ours) |
| ttft_p8192_b4 | 809.25 ms | 842.52 ms | 0.961x (stock time / ours) |
| ttft_p32640_b1 | 1115.27 ms | 1194.43 ms | 0.934x (stock time / ours) |
| prefill_p128_b32 | 97.19 ms | 106.88 ms | 0.909x (stock time / ours) |
| decode_b1 | 169.4 tok/s | 163.6 tok/s | 0.966x (ours / stock) |
| decode_b8 | 1313.3 tok/s | 1262.8 tok/s | 0.962x (ours / stock) |
| decode_b32 | 5170.5 tok/s | 4682.5 tok/s | 0.906x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs CONTRACT v2 v5 (v4 + K read once per block, lanes = keys, heads inner), no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 12.32 ms | 14.38 ms | 0.857x (stock time / ours) |
| ttft_p128_b8 | 28.12 ms | 31.01 ms | 0.907x (stock time / ours) |
| ttft_p2048_b1 | 51.51 ms | 55.67 ms | 0.925x (stock time / ours) |
| ttft_p2048_b8 | 374.64 ms | 386.85 ms | 0.968x (stock time / ours) |
| ttft_p8192_b1 | 213.86 ms | 220.41 ms | 0.970x (stock time / ours) |
| ttft_p8192_b8 | 1602.32 ms | 1680.30 ms | 0.954x (stock time / ours) |
| ttft_p8192_b4 | 806.99 ms | 844.89 ms | 0.955x (stock time / ours) |
| ttft_p32640_b1 | 1109.46 ms | 1196.11 ms | 0.928x (stock time / ours) |
| prefill_p128_b32 | 98.04 ms | 107.59 ms | 0.911x (stock time / ours) |
| decode_b1 | 164.6 tok/s | 159.1 tok/s | 0.966x (ours / stock) |
| decode_b8 | 1286.9 tok/s | 1234.8 tok/s | 0.960x (ours / stock) |
| decode_b32 | 5066.9 tok/s | 4599.1 tok/s | 0.908x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs CONTRACT v2 v4 (v3b + vector q loads + one cooperative V transpose per block), no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.35 ms | 13.38 ms | 0.848x (stock time / ours) |
| ttft_p128_b8 | 27.87 ms | 30.77 ms | 0.906x (stock time / ours) |
| ttft_p2048_b1 | 51.78 ms | 54.68 ms | 0.947x (stock time / ours) |
| ttft_p2048_b8 | 374.94 ms | 385.05 ms | 0.974x (stock time / ours) |
| ttft_p8192_b1 | 210.23 ms | 217.82 ms | 0.965x (stock time / ours) |
| ttft_p8192_b8 | 1604.36 ms | 1678.81 ms | 0.956x (stock time / ours) |
| ttft_p8192_b4 | 809.20 ms | 845.64 ms | 0.957x (stock time / ours) |
| ttft_p32640_b1 | 1104.93 ms | 1188.02 ms | 0.930x (stock time / ours) |
| prefill_p128_b32 | 98.26 ms | 104.79 ms | 0.938x (stock time / ours) |
| decode_b1 | 165.4 tok/s | 160.2 tok/s | 0.968x (ours / stock) |
| decode_b8 | 1289.4 tok/s | 1238.9 tok/s | 0.961x (ours / stock) |
| decode_b32 | 5191.7 tok/s | 4702.3 tok/s | 0.906x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs CONTRACT v2 FINAL SHAPE (v3b: SEG=32 units, merged single-segment fold, multi-head staging), no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.29 ms | 13.22 ms | 0.854x (stock time / ours) |
| ttft_p128_b8 | 28.00 ms | 30.77 ms | 0.910x (stock time / ours) |
| ttft_p2048_b1 | 51.68 ms | 55.55 ms | 0.930x (stock time / ours) |
| ttft_p2048_b8 | 373.15 ms | 384.98 ms | 0.969x (stock time / ours) |
| ttft_p8192_b1 | 211.98 ms | 220.73 ms | 0.960x (stock time / ours) |
| ttft_p8192_b8 | 1602.44 ms | 1685.03 ms | 0.951x (stock time / ours) |
| ttft_p8192_b4 | 811.10 ms | 847.52 ms | 0.957x (stock time / ours) |
| ttft_p32640_b1 | 1108.27 ms | 1196.84 ms | 0.926x (stock time / ours) |
| prefill_p128_b32 | 96.79 ms | 109.70 ms | 0.882x (stock time / ours) |
| decode_b1 | 165.0 tok/s | 159.5 tok/s | 0.967x (ours / stock) |
| decode_b8 | 1290.2 tok/s | 1238.1 tok/s | 0.960x (ours / stock) |
| decode_b32 | 5090.1 tok/s | 4598.6 tok/s | 0.903x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs CONTRACT v2 FINAL SHAPE (v3: filed SEG=32, sub-segment units, staged folds), no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 12.05 ms | 14.00 ms | 0.860x (stock time / ours) |
| ttft_p128_b8 | 28.75 ms | 31.00 ms | 0.928x (stock time / ours) |
| ttft_p2048_b1 | 51.77 ms | 56.54 ms | 0.916x (stock time / ours) |
| ttft_p2048_b8 | 374.47 ms | 385.50 ms | 0.971x (stock time / ours) |
| ttft_p8192_b1 | 212.08 ms | 217.09 ms | 0.977x (stock time / ours) |
| ttft_p8192_b8 | 1603.39 ms | 1681.06 ms | 0.954x (stock time / ours) |
| ttft_p8192_b4 | 808.84 ms | 846.78 ms | 0.955x (stock time / ours) |
| ttft_p32640_b1 | 1112.58 ms | 1196.39 ms | 0.930x (stock time / ours) |
| prefill_p128_b32 | 97.62 ms | 107.55 ms | 0.908x (stock time / ours) |
| decode_b1 | 164.6 tok/s | 153.8 tok/s | 0.934x (ours / stock) |
| decode_b8 | 1287.9 tok/s | 1194.1 tok/s | 0.927x (ours / stock) |
| decode_b32 | 5080.9 tok/s | 4462.2 tok/s | 0.878x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs CONTRACT v2c (lsb_dec8 with block mode for single-segment rows), no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.32 ms | 13.40 ms | 0.845x (stock time / ours) |
| ttft_p128_b8 | 27.38 ms | 30.45 ms | 0.899x (stock time / ours) |
| ttft_p2048_b1 | 51.53 ms | 56.73 ms | 0.908x (stock time / ours) |
| ttft_p2048_b8 | 373.14 ms | 388.66 ms | 0.960x (stock time / ours) |
| ttft_p8192_b1 | 212.42 ms | 224.12 ms | 0.948x (stock time / ours) |
| ttft_p8192_b8 | 1605.21 ms | 1724.41 ms | 0.931x (stock time / ours) |
| ttft_p8192_b4 | 807.21 ms | 868.53 ms | 0.929x (stock time / ours) |
| ttft_p32640_b1 | 1113.09 ms | 1286.48 ms | 0.865x (stock time / ours) |
| prefill_p128_b32 | 98.49 ms | 107.28 ms | 0.918x (stock time / ours) |
| decode_b1 | 165.2 tok/s | 159.9 tok/s | 0.968x (ours / stock) |
| decode_b8 | 1288.2 tok/s | 1235.9 tok/s | 0.959x (ours / stock) |
| decode_b32 | 5085.5 tok/s | 4656.2 tok/s | 0.916x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs CONTRACT v2 (generated rows on LSSA-B8 via lsb_dec8, split cap 128), no inductor, full graphs, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.47 ms | 13.45 ms | 0.853x (stock time / ours) |
| ttft_p128_b8 | 28.49 ms | 32.17 ms | 0.886x (stock time / ours) |
| ttft_p2048_b1 | 52.42 ms | 55.93 ms | 0.937x (stock time / ours) |
| ttft_p2048_b8 | 378.88 ms | 385.52 ms | 0.983x (stock time / ours) |
| ttft_p8192_b1 | 210.01 ms | 217.95 ms | 0.964x (stock time / ours) |
| ttft_p8192_b8 | 1597.54 ms | 1668.82 ms | 0.957x (stock time / ours) |
| ttft_p8192_b4 | 805.37 ms | 842.78 ms | 0.956x (stock time / ours) |
| ttft_p32640_b1 | 1108.03 ms | 1189.24 ms | 0.932x (stock time / ours) |
| prefill_p128_b32 | 96.93 ms | 103.14 ms | 0.940x (stock time / ours) |
| decode_b1 | 164.8 tok/s | 157.0 tok/s | 0.953x (ours / stock) |
| decode_b8 | 1287.0 tok/s | 1217.6 tok/s | 0.946x (ours / stock) |
| decode_b32 | 5065.7 tok/s | 4602.5 tok/s | 0.909x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs no inductor, full graphs, m11 (hybrid FRAC staging, split cap 128) + filed prompt kernel, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.14 ms | 14.66 ms | 0.760x (stock time / ours) |
| ttft_p128_b8 | 28.67 ms | 30.19 ms | 0.950x (stock time / ours) |
| ttft_p2048_b1 | 51.63 ms | 56.61 ms | 0.912x (stock time / ours) |
| ttft_p2048_b8 | 374.74 ms | 384.70 ms | 0.974x (stock time / ours) |
| ttft_p8192_b1 | 210.20 ms | 219.09 ms | 0.959x (stock time / ours) |
| ttft_p8192_b8 | 1603.18 ms | 1677.45 ms | 0.956x (stock time / ours) |
| ttft_p8192_b4 | 808.94 ms | 847.06 ms | 0.955x (stock time / ours) |
| ttft_p32640_b1 | 1112.47 ms | 1195.20 ms | 0.931x (stock time / ours) |
| prefill_p128_b32 | 98.73 ms | 105.38 ms | 0.937x (stock time / ours) |
| decode_b1 | 165.4 tok/s | 157.4 tok/s | 0.952x (ours / stock) |
| decode_b8 | 1288.6 tok/s | 1218.7 tok/s | 0.946x (ours / stock) |
| decode_b32 | 5082.9 tok/s | 4405.4 tok/s | 0.867x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs no inductor, full graphs, m10 (m9 + cached FRAC table, split cap 128) + filed prompt kernel, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.55 ms | 13.70 ms | 0.844x (stock time / ours) |
| ttft_p128_b8 | 27.50 ms | 30.84 ms | 0.892x (stock time / ours) |
| ttft_p2048_b1 | 51.29 ms | 55.30 ms | 0.927x (stock time / ours) |
| ttft_p2048_b8 | 377.19 ms | 388.72 ms | 0.970x (stock time / ours) |
| ttft_p8192_b1 | 214.06 ms | 218.14 ms | 0.981x (stock time / ours) |
| ttft_p8192_b8 | 1609.60 ms | 1683.37 ms | 0.956x (stock time / ours) |
| ttft_p8192_b4 | 807.89 ms | 845.44 ms | 0.956x (stock time / ours) |
| ttft_p32640_b1 | 1109.13 ms | 1190.95 ms | 0.931x (stock time / ours) |
| prefill_p128_b32 | 96.72 ms | 106.06 ms | 0.912x (stock time / ours) |
| decode_b1 | 165.2 tok/s | 157.3 tok/s | 0.952x (ours / stock) |
| decode_b8 | 1289.5 tok/s | 1219.2 tok/s | 0.946x (ours / stock) |
| decode_b32 | 5077.8 tok/s | 4411.5 tok/s | 0.869x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs no inductor, full graphs, m9 (double-buffered decode kernel; file tagged m6 by the queue template) + filed prompt kernel, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 12.02 ms | 13.80 ms | 0.871x (stock time / ours) |
| ttft_p128_b8 | 27.76 ms | 32.54 ms | 0.853x (stock time / ours) |
| ttft_p2048_b1 | 50.75 ms | 55.31 ms | 0.918x (stock time / ours) |
| ttft_p2048_b8 | 374.53 ms | 383.28 ms | 0.977x (stock time / ours) |
| ttft_p8192_b1 | 210.99 ms | 214.60 ms | 0.983x (stock time / ours) |
| ttft_p8192_b8 | 1596.18 ms | 1671.73 ms | 0.955x (stock time / ours) |
| ttft_p8192_b4 | 805.43 ms | 842.48 ms | 0.956x (stock time / ours) |
| ttft_p32640_b1 | 1107.89 ms | 1189.20 ms | 0.932x (stock time / ours) |
| prefill_p128_b32 | 96.10 ms | 107.16 ms | 0.897x (stock time / ours) |
| decode_b1 | 165.3 tok/s | 157.3 tok/s | 0.952x (ours / stock) |
| decode_b8 | 1289.3 tok/s | 1218.2 tok/s | 0.945x (ours / stock) |
| decode_b32 | 5078.8 tok/s | 4378.1 tok/s | 0.862x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs no inductor, full graphs, LITE prompt-row kernel + m5 decode, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.71 ms | 13.28 ms | 0.882x (stock time / ours) |
| ttft_p128_b8 | 27.27 ms | 30.04 ms | 0.908x (stock time / ours) |
| ttft_p2048_b1 | 51.59 ms | 54.34 ms | 0.949x (stock time / ours) |
| ttft_p2048_b8 | 371.52 ms | 380.66 ms | 0.976x (stock time / ours) |
| ttft_p8192_b1 | 211.99 ms | 214.93 ms | 0.986x (stock time / ours) |
| ttft_p8192_b8 | 1599.60 ms | 1658.06 ms | 0.965x (stock time / ours) |
| ttft_p8192_b4 | 803.05 ms | 835.97 ms | 0.961x (stock time / ours) |
| ttft_p32640_b1 | 1111.67 ms | 1155.34 ms | 0.962x (stock time / ours) |
| prefill_p128_b32 | 96.86 ms | 104.89 ms | 0.923x (stock time / ours) |
| decode_b1 | 170.0 tok/s | 155.1 tok/s | 0.912x (ours / stock) |
| decode_b8 | 1316.8 tok/s | 1199.0 tok/s | 0.911x (ours / stock) |
| decode_b32 | 5207.7 tok/s | 4486.5 tok/s | 0.862x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs no inductor, full graphs, m5 (decode-shaped pass B, one head per warp), medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.37 ms | 13.66 ms | 0.833x (stock time / ours) |
| ttft_p128_b8 | 28.32 ms | 31.11 ms | 0.910x (stock time / ours) |
| ttft_p2048_b1 | 52.97 ms | 56.60 ms | 0.936x (stock time / ours) |
| ttft_p2048_b8 | 376.73 ms | 382.89 ms | 0.984x (stock time / ours) |
| ttft_p8192_b1 | 209.60 ms | 219.25 ms | 0.956x (stock time / ours) |
| ttft_p8192_b8 | 1601.96 ms | 1683.10 ms | 0.952x (stock time / ours) |
| ttft_p8192_b4 | 808.16 ms | 847.61 ms | 0.953x (stock time / ours) |
| ttft_p32640_b1 | 1110.34 ms | 1186.59 ms | 0.936x (stock time / ours) |
| prefill_p128_b32 | 97.50 ms | 105.11 ms | 0.928x (stock time / ours) |
| decode_b1 | 165.3 tok/s | 152.3 tok/s | 0.921x (ours / stock) |
| decode_b8 | 1288.4 tok/s | 1176.2 tok/s | 0.913x (ours / stock) |
| decode_b32 | 5087.7 tok/s | 4399.3 tok/s | 0.865x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs vLLM default (inductor ON, full graphs), both stacks, m8, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 8.72 ms | 10.60 ms | 0.823x (stock time / ours) |
| ttft_p128_b8 | 24.81 ms | 27.67 ms | 0.897x (stock time / ours) |
| ttft_p2048_b1 | 48.91 ms | 51.63 ms | 0.947x (stock time / ours) |
| ttft_p2048_b8 | 370.93 ms | 382.64 ms | 0.969x (stock time / ours) |
| ttft_p8192_b1 | 207.85 ms | 214.79 ms | 0.968x (stock time / ours) |
| ttft_p8192_b8 | 1591.38 ms | 1670.22 ms | 0.953x (stock time / ours) |
| ttft_p8192_b4 | 799.16 ms | 839.37 ms | 0.952x (stock time / ours) |
| ttft_p32640_b1 | 1102.25 ms | 1181.50 ms | 0.933x (stock time / ours) |
| prefill_p128_b32 | 94.91 ms | 103.21 ms | 0.920x (stock time / ours) |
| decode_b1 | 169.1 tok/s | 160.2 tok/s | 0.947x (ours / stock) |
| decode_b8 | 1319.0 tok/s | 1239.5 tok/s | 0.940x (ours / stock) |
| decode_b32 | 5201.9 tok/s | 4466.1 tok/s | 0.859x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs same, decode-shaped pass B v1 (seven heads per warp), medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.24 ms | 13.26 ms | 0.848x (stock time / ours) |
| ttft_p128_b8 | 28.16 ms | 30.97 ms | 0.909x (stock time / ours) |
| ttft_p2048_b1 | 52.58 ms | 56.16 ms | 0.936x (stock time / ours) |
| ttft_p2048_b8 | 377.24 ms | 384.94 ms | 0.980x (stock time / ours) |
| ttft_p8192_b1 | 215.02 ms | 220.60 ms | 0.975x (stock time / ours) |
| ttft_p8192_b8 | 1607.57 ms | 1684.57 ms | 0.954x (stock time / ours) |
| ttft_p8192_b4 | 805.73 ms | 843.22 ms | 0.956x (stock time / ours) |
| ttft_p32640_b1 | 1111.05 ms | 1191.84 ms | 0.932x (stock time / ours) |
| prefill_p128_b32 | 97.52 ms | 107.82 ms | 0.904x (stock time / ours) |
| decode_b1 | 164.8 tok/s | 147.7 tok/s | 0.896x (ours / stock) |
| decode_b8 | 1287.7 tok/s | 1147.6 tok/s | 0.891x (ours / stock) |
| decode_b32 | 5056.3 tok/s | 3922.8 tok/s | 0.776x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs same, before the decode-shaped kernel (memsets folded, fused epilogue), medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.33 ms | 13.54 ms | 0.837x (stock time / ours) |
| ttft_p128_b8 | 27.73 ms | 31.32 ms | 0.885x (stock time / ours) |
| ttft_p2048_b1 | 51.68 ms | 55.67 ms | 0.928x (stock time / ours) |
| ttft_p2048_b8 | 376.72 ms | 385.92 ms | 0.976x (stock time / ours) |
| ttft_p8192_b1 | 210.57 ms | 218.70 ms | 0.963x (stock time / ours) |
| ttft_p8192_b8 | 1602.73 ms | 1674.74 ms | 0.957x (stock time / ours) |
| ttft_p8192_b4 | 805.40 ms | 841.72 ms | 0.957x (stock time / ours) |
| ttft_p32640_b1 | 1105.29 ms | 1191.24 ms | 0.928x (stock time / ours) |
| prefill_p128_b32 | 95.63 ms | 106.29 ms | 0.900x (stock time / ours) |
| decode_b1 | 169.5 tok/s | 144.3 tok/s | 0.851x (ours / stock) |
| decode_b8 | 1314.2 tok/s | 1108.1 tok/s | 0.843x (ours / stock) |
| decode_b32 | 5174.9 tok/s | 3971.4 tok/s | 0.767x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs same, incremental prep only, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 12.18 ms | 13.81 ms | 0.882x (stock time / ours) |
| ttft_p128_b8 | 27.65 ms | 30.97 ms | 0.893x (stock time / ours) |
| ttft_p2048_b1 | 51.82 ms | 54.91 ms | 0.944x (stock time / ours) |
| ttft_p2048_b8 | 377.03 ms | 384.77 ms | 0.980x (stock time / ours) |
| ttft_p8192_b1 | 210.90 ms | 219.06 ms | 0.963x (stock time / ours) |
| ttft_p8192_b8 | 1606.38 ms | 1681.19 ms | 0.956x (stock time / ours) |
| ttft_p8192_b4 | 806.63 ms | 848.25 ms | 0.951x (stock time / ours) |
| ttft_p32640_b1 | 1108.10 ms | 1193.71 ms | 0.928x (stock time / ours) |
| prefill_p128_b32 | 98.49 ms | 106.95 ms | 0.921x (stock time / ours) |
| decode_b1 | 169.4 tok/s | 141.9 tok/s | 0.838x (ours / stock) |
| decode_b8 | 1313.6 tok/s | 1080.0 tok/s | 0.822x (ours / stock) |
| decode_b32 | 5176.9 tok/s | 3860.6 tok/s | 0.746x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs FULL_AND_PIECEWISE CUDA graphs, no inductor, LSSA-B8-LITE prompt rows (candidate contract variant), medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.71 ms | 13.28 ms | 0.882x (stock time / ours) |
| ttft_p128_b8 | 27.27 ms | 30.04 ms | 0.908x (stock time / ours) |
| ttft_p2048_b1 | 51.59 ms | 54.34 ms | 0.949x (stock time / ours) |
| ttft_p2048_b8 | 371.52 ms | 380.66 ms | 0.976x (stock time / ours) |
| ttft_p8192_b1 | 211.99 ms | 214.93 ms | 0.986x (stock time / ours) |
| ttft_p8192_b8 | 1599.60 ms | 1658.06 ms | 0.965x (stock time / ours) |
| ttft_p8192_b4 | 803.05 ms | 835.97 ms | 0.961x (stock time / ours) |
| ttft_p32640_b1 | 1111.67 ms | 1155.34 ms | 0.962x (stock time / ours) |
| prefill_p128_b32 | 96.86 ms | 104.89 ms | 0.923x (stock time / ours) |
| decode_b1 | 170.0 tok/s | 155.1 tok/s | 0.912x (ours / stock) |
| decode_b8 | 1316.8 tok/s | 1199.0 tok/s | 0.911x (ours / stock) |
| decode_b32 | 5207.7 tok/s | 4486.5 tok/s | 0.862x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs FULL_AND_PIECEWISE CUDA graphs, no inductor, after the capture fix + host-cost change, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.45 ms | 13.41 ms | 0.854x (stock time / ours) |
| ttft_p128_b8 | 30.04 ms | 31.21 ms | 0.962x (stock time / ours) |
| ttft_p2048_b1 | 51.55 ms | 55.96 ms | 0.921x (stock time / ours) |
| ttft_p2048_b8 | 376.10 ms | 388.55 ms | 0.968x (stock time / ours) |
| ttft_p8192_b1 | 214.75 ms | 220.18 ms | 0.975x (stock time / ours) |
| ttft_p8192_b8 | 1608.16 ms | 1685.92 ms | 0.954x (stock time / ours) |
| ttft_p8192_b4 | 809.27 ms | 847.30 ms | 0.955x (stock time / ours) |
| ttft_p32640_b1 | 1105.49 ms | 1188.95 ms | 0.930x (stock time / ours) |
| prefill_p128_b32 | 95.88 ms | 105.36 ms | 0.910x (stock time / ours) |
| decode_b1 | 164.8 tok/s | 125.1 tok/s | 0.759x (ours / stock) |
| decode_b8 | 1313.0 tok/s | 973.0 tok/s | 0.741x (ours / stock) |
| decode_b32 | 5170.9 tok/s | 3562.5 tok/s | 0.689x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs FULL_AND_PIECEWISE CUDA graphs, no inductor, after the capture fix, before the host-cost change, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 11.30 ms | 13.36 ms | 0.845x (stock time / ours) |
| ttft_p128_b8 | 27.78 ms | 29.66 ms | 0.937x (stock time / ours) |
| ttft_p2048_b1 | 50.69 ms | 55.63 ms | 0.911x (stock time / ours) |
| ttft_p2048_b8 | 373.51 ms | 379.34 ms | 0.985x (stock time / ours) |
| ttft_p8192_b1 | 210.50 ms | 215.53 ms | 0.977x (stock time / ours) |
| ttft_p8192_b8 | 1590.59 ms | 1668.97 ms | 0.953x (stock time / ours) |
| ttft_p8192_b4 | 804.26 ms | 838.00 ms | 0.960x (stock time / ours) |
| ttft_p32640_b1 | 1106.43 ms | 1175.88 ms | 0.941x (stock time / ours) |
| prefill_p128_b32 | 96.33 ms | 105.31 ms | 0.915x (stock time / ours) |
| decode_b1 | 169.5 tok/s | 127.2 tok/s | 0.750x (ours / stock) |
| decode_b8 | 1315.2 tok/s | 976.4 tok/s | 0.742x (ours / stock) |
| decode_b32 | 5175.1 tok/s | 3560.9 tok/s | 0.688x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs FULL_AND_PIECEWISE with inductor (vLLM default), after the capture fix, medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 8.62 ms | 10.72 ms | 0.805x (stock time / ours) |
| ttft_p128_b8 | 24.91 ms | 27.66 ms | 0.901x (stock time / ours) |
| ttft_p2048_b1 | 48.37 ms | 51.14 ms | 0.946x (stock time / ours) |
| ttft_p2048_b8 | 368.77 ms | 377.69 ms | 0.976x (stock time / ours) |
| ttft_p8192_b1 | 205.35 ms | 216.26 ms | 0.950x (stock time / ours) |
| ttft_p8192_b8 | 1593.59 ms | 1670.78 ms | 0.954x (stock time / ours) |
| ttft_p8192_b4 | 800.68 ms | 832.92 ms | 0.961x (stock time / ours) |
| ttft_p32640_b1 | 1102.14 ms | 1168.61 ms | 0.943x (stock time / ours) |
| prefill_p128_b32 | 94.88 ms | 103.17 ms | 0.920x (stock time / ours) |
| decode_b1 | 174.4 tok/s | 130.0 tok/s | 0.746x (ours / stock) |
| decode_b8 | 1353.2 tok/s | 998.8 tok/s | 0.738x (ours / stock) |
| decode_b32 | 5344.8 tok/s | 3639.8 tok/s | 0.681x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs FULL_AND_PIECEWISE with inductor, BEFORE the capture fix (Lockstep output was WRONG in this mode), medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 8.76 ms | 11.25 ms | 0.779x (stock time / ours) |
| ttft_p128_b8 | 25.41 ms | 27.78 ms | 0.915x (stock time / ours) |
| ttft_p2048_b1 | 48.93 ms | 52.87 ms | 0.925x (stock time / ours) |
| ttft_p2048_b8 | 369.71 ms | 378.73 ms | 0.976x (stock time / ours) |
| ttft_p8192_b1 | 208.23 ms | 216.01 ms | 0.964x (stock time / ours) |
| ttft_p8192_b8 | 1593.80 ms | 1660.89 ms | 0.960x (stock time / ours) |
| ttft_p8192_b4 | 801.96 ms | 836.36 ms | 0.959x (stock time / ours) |
| ttft_p32640_b1 | 1105.28 ms | 1162.60 ms | 0.951x (stock time / ours) |
| prefill_p128_b32 | 94.25 ms | 102.16 ms | 0.923x (stock time / ours) |
| decode_b1 | 168.9 tok/s | 182.7 tok/s | 1.082x (ours / stock) |
| decode_b8 | 1318.1 tok/s | 1436.1 tok/s | 1.090x (ours / stock) |
| decode_b32 | 5204.6 tok/s | 5705.2 tok/s | 1.096x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs compilation mode 0 = no CUDA graphs at all (eager engine), medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 8.77 ms | 11.84 ms | 0.741x (stock time / ours) |
| ttft_p128_b8 | 24.45 ms | 27.73 ms | 0.882x (stock time / ours) |
| ttft_p2048_b1 | 48.04 ms | 52.39 ms | 0.917x (stock time / ours) |
| ttft_p2048_b8 | 368.58 ms | 379.41 ms | 0.971x (stock time / ours) |
| ttft_p8192_b1 | 206.61 ms | 213.07 ms | 0.970x (stock time / ours) |
| ttft_p8192_b8 | 1595.46 ms | 1672.81 ms | 0.954x (stock time / ours) |
| ttft_p8192_b4 | 804.52 ms | 841.96 ms | 0.956x (stock time / ours) |
| ttft_p32640_b1 | 1106.43 ms | 1169.83 ms | 0.946x (stock time / ours) |
| prefill_p128_b32 | 94.69 ms | 102.46 ms | 0.924x (stock time / ours) |
| decode_b1 | 125.7 tok/s | 121.4 tok/s | 0.967x (ours / stock) |
| decode_b8 | 1023.2 tok/s | 945.6 tok/s | 0.924x (ours / stock) |
| decode_b32 | 3798.7 tok/s | 3442.1 tok/s | 0.906x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Single-stream TTFT and decode, CUDA graphs PIECEWISE with inductor (attention outside the graph), medians of 3, both stacks in one process

| Cell | Stock FA3 | Lockstep | Lockstep / stock |
|---|---|---|---|
| ttft_p128_b1 | 8.81 ms | 10.65 ms | 0.827x (stock time / ours) |
| ttft_p128_b8 | 24.93 ms | 29.42 ms | 0.847x (stock time / ours) |
| ttft_p2048_b1 | 48.45 ms | 52.77 ms | 0.918x (stock time / ours) |
| ttft_p2048_b8 | 368.50 ms | 377.60 ms | 0.976x (stock time / ours) |
| ttft_p8192_b1 | 204.89 ms | 212.22 ms | 0.965x (stock time / ours) |
| ttft_p8192_b8 | 1587.55 ms | 1661.72 ms | 0.955x (stock time / ours) |
| ttft_p8192_b4 | 794.73 ms | 827.97 ms | 0.960x (stock time / ours) |
| ttft_p32640_b1 | 1097.47 ms | 1164.14 ms | 0.943x (stock time / ours) |
| prefill_p128_b32 | 96.22 ms | 100.26 ms | 0.960x (stock time / ours) |
| decode_b1 | 164.3 tok/s | 124.6 tok/s | 0.759x (ours / stock) |
| decode_b8 | 1288.1 tok/s | 970.8 tok/s | 0.754x (ours / stock) |
| decode_b32 | 5112.2 tok/s | 3601.9 tok/s | 0.705x (ours / stock) |

Seam: `lssab-b8-1launch-varlen`, errflags[0]=0, varlen launches: 4088; vsh_violations=0

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs SHIPPED TP READING (SPEC 7.13): 72B, TP=4, whole stack, routed native reduction, cut 4,096 rows, both arms under NCCL_GRAPH_MIXING_SUPPORT=0

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 2052.3 | 4228.4 | 15184.0 / 32668.4 | 22.71 / 55.85 |  |
|  | Lockstep | 500 / 0 | 1602.0 | 3300.8 | 20975.9 / 45621.3 | 32.54 / 77.04 | 0.781x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 3178.3 | 3972.9 | 15890.1 / 31554.7 | 19.41 / 19.85 |  |
|  | Lockstep | 256 / 0 | 2493.0 | 3116.2 | 20225.7 / 40263.0 | 24.78 / 25.30 | 0.784x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 197.6 | 12845.0 | 20382.2 / 39920.6 | 154.51 / 154.84 |  |
|  | Lockstep | 128 / 0 | 151.7 | 9861.3 | 26645.1 / 52141.3 | 192.34 / 192.55 | 0.768x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs DISCRIMINATOR (SPEC 7.13): 72B, TP=4, environment alone with the earlier all-gather reduction, both arms under it

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 2066.1 | 4257.0 | 15082.2 / 32510.1 | 22.71 / 58.91 |  |
|  | Lockstep | 500 / 0 | 1580.8 | 3257.0 | 21031.0 / 45972.9 | 32.47 / 81.08 | 0.765x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 3164.1 | 3955.1 | 15901.4 / 31583.9 | 19.50 / 20.09 |  |
|  | Lockstep | 256 / 0 | 2498.8 | 3123.5 | 20200.1 / 40188.4 | 24.75 / 25.30 | 0.790x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 197.9 | 12865.2 | 20334.2 / 39883.3 | 154.56 / 155.22 |  |
|  | Lockstep | 128 / 0 | 151.7 | 9861.0 | 26668.6 / 52154.3 | 192.38 / 192.69 | 0.766x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs SPEC 7.13: 72B, TP=4, whole stack, routed native reduction at the 512-row cut, both arms under NCCL_GRAPH_MIXING_SUPPORT=0 (W1 loss on mixed steps)

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 2075.6 | 4276.6 | 15164.8 / 32307.7 | 22.70 / 56.39 |  |
|  | Lockstep | 500 / 0 | 1454.7 | 2997.3 | 20264.0 / 45145.8 | 32.06 / 72.34 | 0.701x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 3178.9 | 3973.7 | 15889.4 / 31558.0 | 19.42 / 19.85 |  |
|  | Lockstep | 256 / 0 | 2524.9 | 3156.1 | 19944.8 / 39719.1 | 24.55 / 25.00 | 0.794x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 197.6 | 12841.6 | 20384.9 / 39926.4 | 154.46 / 154.71 |  |
|  | Lockstep | 128 / 0 | 168.6 | 10959.3 | 23931.1 / 46781.7 | 172.45 / 172.68 | 0.853x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs T16s2 (SPEC 7.13): 72B, TP=4, whole stack, all-to-all reduction on vLLM's pynccl communicator from 512 rows

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 2078.9 | 4283.4 | 15002.1 / 32461.7 | 22.75 / 55.97 |  |
|  | Lockstep | 500 / 0 | 1488.6 | 3067.1 | 21610.4 / 46705.3 | 32.93 / 70.91 | 0.716x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 3242.5 | 4053.2 | 15594.1 / 30945.5 | 19.00 / 19.42 |  |
|  | Lockstep | 256 / 0 | 2480.9 | 3101.2 | 20327.4 / 40411.4 | 25.01 / 25.42 | 0.765x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 197.8 | 12860.2 | 20359.3 / 39897.8 | 154.48 / 154.72 |  |
|  | Lockstep | 128 / 0 | 168.8 | 10971.4 | 23803.7 / 46532.9 | 171.49 / 171.73 | 0.853x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs T16v + split target 2 (SPEC 7.15): whole stack, pipelined decode kernel, LOCKSTEP_LSSAB_SPLIT_CTA=2

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5397.2 | 11120.2 | 6027.1 / 12498.3 | 8.43 / 17.92 |  |
|  | Lockstep | 500 / 0 | 4453.0 | 9175.0 | 7388.9 / 15850.4 | 10.98 / 20.11 | 0.825x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8182.0 | 10227.5 | 6254.0 / 12280.6 | 7.45 / 7.55 |  |
|  | Lockstep | 256 / 0 | 7422.3 | 9277.9 | 6859.5 / 13507.8 | 8.25 / 8.39 | 0.907x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 585.2 | 38040.8 | 6852.7 / 13376.7 | 51.59 / 51.77 |  |
|  | Lockstep | 128 / 0 | 577.1 | 37514.1 | 7043.2 / 13542.4 | 49.15 / 49.30 | 0.986x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs split target 2 with the 16-warp kernel (7.15 policy A/B)

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5434.9 | 11198.0 | 5967.7 / 12499.4 | 8.45 / 18.55 |  |
|  | Lockstep | 500 / 0 | 4475.2 | 9220.6 | 7378.5 / 15829.8 | 11.02 / 21.29 | 0.823x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8359.8 | 10449.8 | 6105.7 / 12006.0 | 7.31 / 7.41 |  |
|  | Lockstep | 256 / 0 | 7482.3 | 9352.9 | 6786.9 / 13408.4 | 8.24 / 8.34 | 0.895x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 585.5 | 38058.7 | 6855.4 / 13385.0 | 51.57 / 51.76 |  |
|  | Lockstep | 128 / 0 | 574.0 | 37309.9 | 7079.6 / 13621.7 | 49.41 / 49.53 | 0.980x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs split target 4 with the 16-warp kernel (7.15 policy A/B)

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5472.9 | 11276.4 | 6015.3 / 12398.1 | 8.34 / 17.75 |  |
|  | Lockstep | 500 / 0 | 4529.3 | 9332.0 | 7282.6 / 15605.1 | 10.80 / 20.45 | 0.828x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8347.7 | 10434.6 | 6112.9 / 12029.7 | 7.31 / 7.41 |  |
|  | Lockstep | 256 / 0 | 7362.2 | 9202.7 | 6958.1 / 13640.3 | 8.29 / 8.53 | 0.882x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 588.4 | 38248.8 | 6797.9 / 13314.0 | 51.56 / 51.72 |  |
|  | Lockstep | 128 / 0 | 574.5 | 37344.5 | 7045.7 / 13603.4 | 49.53 / 49.66 | 0.976x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs T16v (SPEC 7.14): whole verifiable stack with the two-group pipelined decode kernel, no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5362.6 | 11049.0 | 6057.6 / 12533.0 | 8.42 / 17.11 |  |
|  | Lockstep | 500 / 0 | 4548.1 | 9370.9 | 7342.2 / 15542.1 | 10.75 / 19.08 | 0.848x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8151.5 | 10189.4 | 6275.7 / 12328.6 | 7.49 / 7.58 |  |
|  | Lockstep | 256 / 0 | 7262.9 | 9078.7 | 7043.3 / 13844.9 | 8.40 / 8.65 | 0.891x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 583.7 | 37939.8 | 6884.8 / 13413.9 | 51.64 / 51.82 |  |
|  | Lockstep | 128 / 0 | 574.2 | 37321.7 | 7065.0 / 13624.2 | 49.43 / 49.61 | 0.984x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs T16s (SPEC 7.13): 72B, TP=4, whole stack, 16-warp kernel + NATIVE all-to-all reduction from 512 rows, no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 2071.8 | 4268.6 | 15127.2 / 32340.6 | 22.77 / 57.94 |  |
|  | Lockstep | 500 / 0 | 1532.8 | 3158.2 | 21287.2 / 47275.5 | 33.14 / 72.15 | 0.740x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 3183.8 | 3979.8 | 15836.2 / 31480.3 | 19.40 / 19.84 |  |
|  | Lockstep | 256 / 0 | 2395.8 | 2994.8 | 20983.9 / 41794.3 | 25.91 / 26.38 | 0.752x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 197.3 | 12821.7 | 20426.2 / 39981.4 | 154.66 / 154.97 |  |
|  | Lockstep | 128 / 0 | 169.6 | 11024.9 | 23748.8 / 46420.5 | 171.16 / 171.46 | 0.860x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs SUPERSEDED (SPEC 7.13): 72B, TP=4, native reduction on every row count, no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 2050.8 | 4225.4 | 15357.6 / 32543.3 | 22.70 / 58.82 |  |
|  | Lockstep | 500 / 0 | 1531.6 | 3155.7 | 21093.0 / 46960.9 | 33.37 / 72.72 | 0.747x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 3229.0 | 4036.3 | 15792.4 / 31122.3 | 19.07 / 19.76 |  |
|  | Lockstep | 256 / 0 | 2392.6 | 2990.8 | 21047.1 / 41839.9 | 25.98 / 26.44 | 0.741x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 197.9 | 12863.8 | 20353.9 / 39875.0 | 154.33 / 154.61 |  |
|  | Lockstep | 128 / 0 | 169.6 | 11027.0 | 23743.4 / 46411.2 | 171.05 / 171.31 | 0.857x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs SHIPPED KERNEL (T16r, SPEC 7.12): 72B, TP=4, whole verifiable stack with the 16-warp decode kernel, no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 2058.1 | 4240.4 | 14936.4 / 32459.3 | 22.74 / 61.13 |  |
|  | Lockstep | 500 / 0 | 1575.8 | 3246.7 | 21039.9 / 46616.1 | 32.69 / 78.64 | 0.766x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 3177.2 | 3971.6 | 15903.9 / 31561.4 | 19.41 / 19.85 |  |
|  | Lockstep | 256 / 0 | 2460.0 | 3074.9 | 20450.9 / 40747.8 | 25.14 / 25.68 | 0.774x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 197.3 | 12822.6 | 20382.6 / 39985.0 | 154.85 / 155.35 |  |
|  | Lockstep | 128 / 0 | 151.8 | 9869.4 | 26614.6 / 52067.1 | 192.12 / 192.33 | 0.770x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs SHIPPED KERNEL (T16r, SPEC 7.12): whole verifiable stack with the 16-warp decode kernel, no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5380.3 | 11085.5 | 6016.0 / 12631.2 | 8.44 / 18.87 |  |
|  | Lockstep | 500 / 0 | 4570.7 | 9417.3 | 7297.0 / 15388.1 | 10.64 / 20.54 | 0.850x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8170.6 | 10213.3 | 6247.1 / 12294.4 | 7.50 / 7.58 |  |
|  | Lockstep | 256 / 0 | 7283.3 | 9104.2 | 6992.2 / 13761.3 | 8.40 / 8.53 | 0.891x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 584.8 | 38009.1 | 6858.7 / 13386.4 | 51.59 / 51.81 |  |
|  | Lockstep | 128 / 0 | 574.7 | 37358.6 | 7064.7 / 13611.5 | 49.43 / 49.57 | 0.983x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs NOT SHIPPED: 72B, TP=4, whole stack with the python-wrapped routed reduction (SPEC 7.11), no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 1700.2 | 3503.0 | 17727.6 / 38293.1 | 27.25 / 63.38 |  |
|  | Lockstep | 500 / 0 | 1535.4 | 3163.5 | 21301.4 / 46805.0 | 33.12 / 80.91 | 0.903x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 2656.5 | 3320.6 | 18975.0 / 37724.2 | 23.25 / 23.77 |  |
|  | Lockstep | 256 / 0 | 2378.6 | 2973.2 | 21274.1 / 42274.9 | 26.10 / 26.63 | 0.895x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 171.4 | 11143.1 | 23431.2 / 45960.2 | 178.00 / 178.41 |  |
|  | Lockstep | 128 / 0 | 150.0 | 9747.8 | 26922.1 / 52656.8 | 194.24 / 194.63 | 0.875x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs Qwen2.5-72B-Instruct, TP=4, the WHOLE VERIFIABLE STACK, no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 2046.4 | 4216.4 | 15030.6 / 32366.5 | 22.76 / 56.28 |  |
|  | Lockstep | 500 / 0 | 1549.6 | 3192.7 | 21254.3 / 46964.1 | 33.24 / 80.08 | 0.757x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 3184.0 | 3980.0 | 15833.4 / 31489.7 | 19.41 / 19.82 |  |
|  | Lockstep | 256 / 0 | 2391.7 | 2989.6 | 21040.2 / 41994.4 | 25.89 / 26.42 | 0.751x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 197.5 | 12835.6 | 20383.9 / 39955.4 | 154.67 / 155.04 |  |
|  | Lockstep | 128 / 0 | 150.3 | 9772.0 | 26805.9 / 52526.4 | 194.04 / 194.25 | 0.761x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs END-TO-END VERIFIED STACK, FUSED, C++ ops, COMPILE CACHE OFF (T16m v2.3, the valid measurement), no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5411.5 | 11149.8 | 6031.6 / 12496.1 | 8.45 / 17.89 |  |
|  | Lockstep | 500 / 0 | 4487.1 | 9245.2 | 7446.4 / 15956.4 | 10.93 / 20.09 | 0.829x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8172.9 | 10216.2 | 6268.5 / 12289.3 | 7.46 / 7.56 |  |
|  | Lockstep | 256 / 0 | 7175.4 | 8969.2 | 7114.2 / 13987.1 | 8.53 / 8.79 | 0.878x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 582.6 | 37870.5 | 6889.6 / 13446.8 | 51.75 / 52.02 |  |
|  | Lockstep | 128 / 0 | 569.4 | 37014.1 | 7103.9 / 13698.6 | 49.83 / 49.95 | 0.977x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs END-TO-END VERIFIED STACK, FUSED, C++-registered ops (T16m v2.3), no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5239.7 | 10795.8 | 6346.8 / 12996.8 | 8.69 / 19.18 |  |
|  | Lockstep | 500 / 0 | 4293.7 | 8846.6 | 7761.9 / 16571.4 | 11.56 / 22.71 | 0.819x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8144.8 | 10181.0 | 6307.5 / 12361.1 | 7.48 / 7.57 |  |
|  | Lockstep | 256 / 0 | 7133.2 | 8916.5 | 7116.6 / 14028.7 | 8.60 / 8.72 | 0.876x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 582.7 | 37875.2 | 6896.6 / 13454.1 | 51.83 / 52.03 |  |
|  | Lockstep | 128 / 0 | 544.1 | 35363.9 | 7457.6 / 14393.5 | 52.25 / 52.50 | 0.934x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs END-TO-END VERIFIED STACK, FUSED (T16m v2.2), no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5251.5 | 10820.2 | 6350.4 / 13024.0 | 8.69 / 19.00 |  |
|  | Lockstep | 500 / 0 | 4259.6 | 8776.4 | 7866.6 / 16848.9 | 11.54 / 22.64 | 0.811x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8134.6 | 10168.2 | 6296.6 / 12353.9 | 7.51 / 7.60 |  |
|  | Lockstep | 256 / 0 | 7131.6 | 8914.6 | 7135.4 / 14051.6 | 8.57 / 8.72 | 0.877x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 582.3 | 37849.0 | 6884.3 / 13451.1 | 51.90 / 52.08 |  |
|  | Lockstep | 128 / 0 | 543.7 | 35337.4 | 7457.5 / 14392.6 | 52.37 / 52.62 | 0.934x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs END-TO-END VERIFIED STACK, fused glue only (T16m interim), no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5119.9 | 10548.9 | 6240.4 / 12744.3 | 8.71 / 18.19 |  |
|  | Lockstep | 500 / 0 | 4031.7 | 8306.9 | 8371.4 / 17984.5 | 12.28 / 26.61 | 0.787x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8152.7 | 10190.9 | 6272.2 / 12318.8 | 7.48 / 7.58 |  |
|  | Lockstep | 256 / 0 | 6863.4 | 8579.2 | 7455.1 / 14664.8 | 8.88 / 9.03 | 0.842x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 581.7 | 37810.8 | 6907.4 / 13464.6 | 51.79 / 52.10 |  |
|  | Lockstep | 128 / 0 | 437.9 | 28462.3 | 9289.1 / 17970.8 | 65.57 / 65.71 | 0.753x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs Qwen2.5-72B-Instruct, TP=4, contract v2 v8, no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 2033.1 | 4189.1 | 14897.1 / 32382.1 | 22.71 / 61.18 |  |
|  | Lockstep | 500 / 0 | 1850.1 | 3812.0 | 16700.9 / 37116.2 | 26.10 / 64.63 | 0.910x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 3181.5 | 3976.9 | 15843.3 / 31514.7 | 19.43 / 19.85 |  |
|  | Lockstep | 256 / 0 | 2933.8 | 3667.2 | 17188.4 / 34146.3 | 21.12 / 21.47 | 0.922x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 197.6 | 12845.0 | 20382.2 / 39918.4 | 154.45 / 154.87 |  |
|  | Lockstep | 128 / 0 | 185.6 | 12062.7 | 21523.5 / 42309.2 | 156.52 / 157.32 | 0.939x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs Qwen2.5-14B-Instruct, contract v2 v8, no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 2850.7 | 5873.6 | 11059.9 / 23859.4 | 16.48 / 29.97 |  |
|  | Lockstep | 500 / 0 | 2483.8 | 5117.6 | 12859.8 / 27840.6 | 19.31 / 35.93 | 0.871x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 4270.7 | 5338.4 | 11890.5 / 23475.7 | 14.46 / 14.65 |  |
|  | Lockstep | 256 / 0 | 3922.0 | 4902.4 | 12914.3 / 25545.6 | 15.77 / 15.94 | 0.918x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 284.2 | 18469.8 | 14000.7 / 27582.4 | 107.32 / 107.59 |  |
|  | Lockstep | 128 / 0 | 269.8 | 17537.3 | 14700.2 / 28973.6 | 107.48 / 107.78 | 0.950x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs END-TO-END VERIFIED STACK (T16L_ARM=1 + contract v2), no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5396.9 | 11119.7 | 6132.5 / 12438.7 | 8.52 / 17.92 |  |
|  | Lockstep | 500 / 0 | 3831.0 | 7893.3 | 8895.7 / 18779.2 | 12.98 / 27.37 | 0.710x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8313.0 | 10391.3 | 6145.5 / 12096.3 | 7.34 / 7.45 |  |
|  | Lockstep | 256 / 0 | 6498.4 | 8123.0 | 7852.9 / 15504.7 | 9.40 / 9.59 | 0.782x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 585.6 | 38065.5 | 6849.5 / 13385.7 | 51.62 / 51.84 |  |
|  | Lockstep | 128 / 0 | 390.6 | 25390.8 | 10443.5 / 20215.9 | 73.78 / 74.08 | 0.667x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs SHIPPED: CONTRACT v2 v6 (fused prep), no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5349.9 | 11022.9 | 6078.7 / 12687.3 | 8.66 / 18.61 |  |
|  | Lockstep | 500 / 0 | 4754.8 | 9796.7 | 7041.1 / 14556.4 | 9.90 / 19.88 | 0.889x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8136.4 | 10170.5 | 6296.3 / 12363.5 | 7.49 / 7.58 |  |
|  | Lockstep | 256 / 0 | 7638.4 | 9548.0 | 6697.5 / 13170.3 | 8.01 / 8.10 | 0.939x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 582.5 | 37862.7 | 6888.3 / 13444.5 | 51.82 / 52.01 |  |
|  | Lockstep | 128 / 0 | 543.2 | 35306.2 | 7358.6 / 14365.3 | 52.65 / 53.05 | 0.932x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs CONTRACT v2 v7 (half-block staging; evidence only), no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5075.7 | 10457.9 | 6252.6 / 12736.0 | 8.63 / 17.63 |  |
|  | Lockstep | 500 / 0 | 4639.8 | 9559.8 | 7213.1 / 14864.3 | 10.23 / 22.36 | 0.914x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8151.1 | 10188.8 | 6274.2 / 12328.0 | 7.48 / 7.57 |  |
|  | Lockstep | 256 / 0 | 7345.7 | 9182.1 | 6925.6 / 13674.5 | 8.36 / 8.45 | 0.901x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 584.1 | 37964.2 | 6865.8 / 13413.4 | 51.69 / 51.94 |  |
|  | Lockstep | 128 / 0 | 537.2 | 34914.8 | 7420.2 / 14512.9 | 53.35 / 53.66 | 0.920x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs CONTRACT v2 FINAL (v4 kernel, cap 64), no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5094.4 | 10496.5 | 6248.1 / 12907.7 | 8.74 / 18.18 |  |
|  | Lockstep | 500 / 0 | 4506.8 | 9285.7 | 7172.8 / 15144.7 | 10.43 / 19.24 | 0.885x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8124.3 | 10155.3 | 6317.2 / 12381.7 | 7.51 / 7.60 |  |
|  | Lockstep | 256 / 0 | 7422.2 | 9277.8 | 6881.2 / 13532.9 | 8.26 / 8.36 | 0.914x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 581.5 | 37796.2 | 6905.9 / 13460.7 | 51.80 / 52.01 |  |
|  | Lockstep | 128 / 0 | 544.5 | 35392.3 | 7346.0 / 14323.1 | 52.48 / 52.73 | 0.936x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs CONTRACT v2 v5, no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5148.9 | 10608.7 | 6214.3 / 12906.8 | 8.68 / 17.90 |  |
|  | Lockstep | 500 / 0 | 4549.0 | 9372.7 | 7317.0 / 15309.1 | 10.44 / 20.26 | 0.883x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8174.9 | 10218.6 | 6248.9 / 12287.0 | 7.49 / 7.58 |  |
|  | Lockstep | 256 / 0 | 7380.1 | 9225.1 | 6915.8 / 13615.2 | 8.33 / 8.40 | 0.903x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 581.5 | 37796.7 | 6929.4 / 13477.5 | 51.78 / 51.98 |  |
|  | Lockstep | 128 / 0 | 543.2 | 35305.7 | 7370.9 / 14345.2 | 52.55 / 52.72 | 0.934x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs CONTRACT v2 v4, no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5331.1 | 10984.2 | 6326.1 / 12787.7 | 8.75 / 17.63 |  |
|  | Lockstep | 500 / 0 | 4588.4 | 9453.8 | 7286.4 / 15343.8 | 10.46 / 21.68 | 0.861x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8136.5 | 10170.6 | 6300.4 / 12354.4 | 7.49 / 7.58 |  |
|  | Lockstep | 256 / 0 | 7417.3 | 9271.6 | 6868.8 / 13545.9 | 8.27 / 8.36 | 0.912x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 584.2 | 37973.9 | 6874.0 / 13408.1 | 51.54 / 51.78 |  |
|  | Lockstep | 128 / 0 | 548.3 | 35636.5 | 7272.1 / 14215.9 | 52.29 / 52.58 | 0.938x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs CONTRACT v2 v3b (SEG=32 units, staged folds), no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5285.9 | 10891.0 | 6285.1 / 12910.9 | 8.74 / 18.68 |  |
|  | Lockstep | 500 / 0 | 4617.4 | 9513.7 | 7252.7 / 15191.9 | 10.29 / 19.19 | 0.874x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8136.8 | 10171.0 | 6285.2 / 12342.9 | 7.50 / 7.59 |  |
|  | Lockstep | 256 / 0 | 7318.9 | 9148.6 | 6975.6 / 13715.7 | 8.37 / 8.45 | 0.899x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 582.6 | 37871.3 | 6889.0 / 13441.9 | 51.80 / 52.00 |  |
|  | Lockstep | 128 / 0 | 546.3 | 35511.8 | 7308.8 / 14276.2 | 52.49 / 52.69 | 0.938x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs CONTRACT v2 v3 (SEG=32 units, staged folds), no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5270.5 | 10859.3 | 6193.1 / 12916.2 | 8.79 / 18.25 |  |
|  | Lockstep | 500 / 0 | 4457.0 | 9183.2 | 7382.1 / 15394.4 | 10.43 / 20.52 | 0.846x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8146.6 | 10183.3 | 6305.2 / 12355.5 | 7.50 / 7.58 |  |
|  | Lockstep | 256 / 0 | 7219.2 | 9024.1 | 7113.4 / 13931.7 | 8.55 / 8.63 | 0.886x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 587.0 | 38153.9 | 6818.8 / 13348.2 | 51.58 / 51.82 |  |
|  | Lockstep | 128 / 0 | 550.7 | 35793.2 | 7258.5 / 14171.7 | 52.04 / 52.31 | 0.938x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs CONTRACT v2c (block mode), no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5218.8 | 10752.9 | 6196.4 / 12763.3 | 8.68 / 17.90 |  |
|  | Lockstep | 500 / 0 | 4472.5 | 9215.0 | 7206.6 / 15495.0 | 10.45 / 19.87 | 0.857x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8166.2 | 10207.7 | 6255.2 / 12297.9 | 7.48 / 7.58 |  |
|  | Lockstep | 256 / 0 | 7319.9 | 9149.9 | 6965.4 / 13705.5 | 8.38 / 8.45 | 0.896x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 582.3 | 37851.4 | 6888.3 / 13454.5 | 51.81 / 52.04 |  |
|  | Lockstep | 128 / 0 | 550.2 | 35762.1 | 7290.5 / 14201.2 | 52.00 / 52.34 | 0.945x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs CONTRACT v2 (lsb_dec8 + SEG=8 prompt kernel), no inductor, full graphs

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5316.9 | 10954.9 | 6230.8 / 12886.1 | 8.68 / 17.40 |  |
|  | Lockstep | 500 / 0 | 4549.2 | 9373.2 | 7209.6 / 15146.0 | 10.34 / 19.91 | 0.856x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8133.5 | 10166.9 | 6292.3 / 12347.1 | 7.51 / 7.60 |  |
|  | Lockstep | 256 / 0 | 7555.3 | 9444.1 | 6764.4 / 13291.9 | 8.11 / 8.20 | 0.929x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 583.2 | 37908.1 | 6902.0 / 13430.9 | 51.57 / 51.80 |  |
|  | Lockstep | 128 / 0 | 554.1 | 36016.2 | 7240.4 / 14092.2 | 51.60 / 51.80 | 0.950x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs no inductor, full graphs, m11 (hybrid FRAC staging, split cap 128)

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5326.0 | 10973.6 | 6224.6 / 13019.1 | 8.68 / 18.15 |  |
|  | Lockstep | 500 / 0 | 4510.5 | 9293.3 | 7410.7 / 15690.8 | 10.64 / 19.78 | 0.847x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8135.7 | 10169.6 | 6306.6 / 12351.4 | 7.51 / 7.59 |  |
|  | Lockstep | 256 / 0 | 6907.4 | 8634.2 | 7362.1 / 14522.9 | 8.89 / 8.97 | 0.849x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 585.2 | 38037.7 | 6848.3 / 13379.9 | 51.60 / 51.80 |  |
|  | Lockstep | 128 / 0 | 538.3 | 34987.5 | 7421.1 / 14501.0 | 53.22 / 53.57 | 0.920x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs no inductor, full graphs, m10 (cached FRAC table, split cap 128)

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 4661.6 | 9604.8 | 6055.5 / 12767.9 | 8.70 / 19.67 |  |
|  | Lockstep | 500 / 0 | 4430.6 | 9128.7 | 7482.8 / 15943.4 | 10.81 / 20.81 | 0.950x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8147.8 | 10184.7 | 6280.9 / 12330.1 | 7.50 / 7.59 |  |
|  | Lockstep | 256 / 0 | 6815.5 | 8519.4 | 7459.3 / 14724.1 | 9.01 / 9.08 | 0.836x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 585.7 | 38068.9 | 6860.2 / 13385.0 | 51.55 / 51.75 |  |
|  | Lockstep | 128 / 0 | 541.2 | 35178.2 | 7366.3 / 14410.2 | 53.04 / 53.29 | 0.924x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs no inductor, full graphs, m9 (double-buffered chunks)

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5296.4 | 10912.6 | 6233.9 / 12852.1 | 8.74 / 18.53 |  |
|  | Lockstep | 500 / 0 | 4446.7 | 9161.9 | 7462.0 / 15678.4 | 10.83 / 23.55 | 0.840x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8131.2 | 10163.9 | 6304.4 / 12363.6 | 7.51 / 7.60 |  |
|  | Lockstep | 256 / 0 | 6857.1 | 8571.3 | 7437.1 / 14625.0 | 8.94 / 9.03 | 0.843x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 584.8 | 38014.0 | 6860.7 / 13391.7 | 51.63 / 51.83 |  |
|  | Lockstep | 128 / 0 | 540.7 | 35145.8 | 7392.9 / 14423.5 | 53.00 / 53.22 | 0.925x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs no inductor, full graphs, m8

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5332.6 | 10987.2 | 6237.7 / 12809.8 | 8.67 / 18.74 |  |
|  | Lockstep | 500 / 0 | 4253.8 | 8764.4 | 7605.6 / 16185.7 | 10.93 / 22.22 | 0.798x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8171.2 | 10214.0 | 6230.3 / 12292.3 | 7.51 / 7.60 |  |
|  | Lockstep | 256 / 0 | 6762.4 | 8453.0 | 7515.1 / 14818.0 | 9.10 / 9.18 | 0.828x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 582.8 | 37884.7 | 6878.4 / 13449.9 | 51.86 / 52.11 |  |
|  | Lockstep | 128 / 0 | 534.5 | 34739.9 | 7465.8 / 14589.0 | 53.58 / 53.89 | 0.917x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs no inductor, full graphs, after the capture fix + host-cost change

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5278.8 | 10876.3 | 6303.7 / 12911.2 | 8.62 / 18.41 |  |
|  | Lockstep | 500 / 0 | 3408.9 | 7023.7 | 9382.8 / 20385.3 | 14.30 / 23.50 | 0.646x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8294.9 | 10368.6 | 6181.9 / 12116.7 | 7.36 / 7.45 |  |
|  | Lockstep | 256 / 0 | 5219.5 | 6524.4 | 9616.4 / 19118.9 | 11.90 / 11.97 | 0.629x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 581.9 | 37824.3 | 6905.4 / 13471.5 | 51.87 / 52.04 |  |
|  | Lockstep | 128 / 0 | 491.1 | 31923.2 | 7931.8 / 15676.3 | 58.16 / 58.59 | 0.844x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs FULL_AND_PIECEWISE CUDA graphs, no inductor, after the capture fix, before the host-cost change

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5275.6 | 10869.9 | 6282.1 / 12863.8 | 8.75 / 20.58 |  |
|  | Lockstep | 500 / 0 | 3358.4 | 6919.5 | 9695.2 / 20737.8 | 14.65 / 24.82 | 0.637x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8311.6 | 10389.4 | 6127.0 / 12086.0 | 7.35 / 7.45 |  |
|  | Lockstep | 256 / 0 | 5222.8 | 6528.6 | 9636.2 / 19129.6 | 11.92 / 11.98 | 0.628x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 582.3 | 37847.8 | 6906.5 / 13466.9 | 51.79 / 52.00 |  |
|  | Lockstep | 128 / 0 | 490.0 | 31848.7 | 7959.5 / 15721.0 | 58.38 / 58.70 | 0.841x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs FULL_AND_PIECEWISE with inductor (vLLM default), after the capture fix

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5542.2 | 11419.0 | 6063.2 / 12426.7 | 8.24 / 18.34 |  |
|  | Lockstep | 500 / 0 | 3413.9 | 7033.9 | 9681.0 / 20445.2 | 14.20 / 24.06 | 0.616x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8547.7 | 10684.6 | 5965.6 / 11760.1 | 7.15 / 7.24 |  |
|  | Lockstep | 256 / 0 | 5249.9 | 6562.4 | 9556.3 / 19025.8 | 11.84 / 11.90 | 0.614x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 583.0 | 37895.9 | 6899.6 / 13451.5 | 51.85 / 51.99 |  |
|  | Lockstep | 128 / 0 | 489.7 | 31833.4 | 7985.1 / 15734.1 | 58.48 / 58.71 | 0.840x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs FULL_AND_PIECEWISE with inductor, BEFORE the capture fix (Lockstep output was WRONG)

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5572.7 | 11481.9 | 5885.5 / 12212.3 | 8.29 / 18.45 |  |
|  | Lockstep | 500 / 0 | 5319.9 | 10961.0 | 6513.1 / 13649.4 | 9.23 / 22.58 | 0.955x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8398.3 | 10497.9 | 6072.3 / 11959.6 | 7.27 / 7.38 |  |
|  | Lockstep | 256 / 0 | 9440.7 | 11800.9 | 5480.2 / 10702.9 | 6.40 / 6.50 | 1.124x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 584.5 | 37992.4 | 6861.1 / 13407.7 | 51.72 / 51.93 |  |
|  | Lockstep | 128 / 0 | 506.0 | 32892.9 | 7894.9 / 15661.0 | 58.40 / 58.76 | 0.866x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs compilation mode 0 = no CUDA graphs (eager engine)

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 4422.5 | 9112.0 | 6506.6 / 13763.0 | 9.41 / 21.64 |  |
|  | Lockstep | 500 / 0 | 3354.7 | 6912.1 | 9322.8 / 20379.8 | 14.41 / 23.35 | 0.759x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 6885.1 | 8606.4 | 7506.8 / 14622.4 | 9.05 / 9.28 |  |
|  | Lockstep | 256 / 0 | 5088.7 | 6360.8 | 9835.2 / 19568.8 | 12.24 / 12.35 | 0.739x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 581.7 | 37808.5 | 6816.6 / 13370.1 | 51.89 / 52.10 |  |
|  | Lockstep | 128 / 0 | 490.1 | 31855.6 | 7944.3 / 15684.5 | 58.23 / 58.58 | 0.843x |

### Serving, `vllm bench serve`, 64 concurrent sequences, prefix caching off, CUDA graphs PIECEWISE with inductor (attention outside the graph)

| Workload | Stack | completed / failed | output tok/s | total tok/s | TTFT p50 / p99 ms | TPOT p50 / p99 ms | Lockstep / stock (output tok/s) |
|---|---|---|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests, saturated | stock | 500 / 0 | 5574.4 | 11485.4 | 5883.6 / 12099.3 | 8.18 / 18.13 |  |
|  | Lockstep | 500 / 0 | 3492.3 | 7195.4 | 9274.5 / 19928.2 | 14.06 / 25.32 | 0.626x |
| W2 decode-heavy, 128 in / 512 out, 256 requests | stock | 256 / 0 | 8344.9 | 10431.1 | 6079.7 / 12027.7 | 7.37 / 7.42 |  |
|  | Lockstep | 256 / 0 | 5229.6 | 6536.9 | 9559.5 / 19058.1 | 11.88 / 11.97 | 0.627x |
| W3 prefill-heavy, 4096 in / 64 out, 128 requests | stock | 128 / 0 | 583.2 | 37911.0 | 6879.0 / 13425.1 | 51.79 / 51.99 |  |
|  | Lockstep | 128 / 0 | 490.5 | 31882.4 | 7935.2 / 15694.4 | 58.49 / 58.69 | 0.841x |

### Quality: wikitext-2 perplexity from the engine, 300 windows, ctx 1024, stride 512, eager

| Arm | ppl (stride 512, 300 w) | ppl (all positions, 300 w) | 60-window disjoint cell | delta vs stock |
|---|---|---|---|---|
| stock bf16 FlashAttention-3 | 6.8334 | 8.0670 | 8.3883 | +0.000% |
| Lockstep, portable LSSA-B for all rows | 301436.3589 | 303448.8162 | 292791.3373 | +4411141.731% |
| Lockstep, LSSA-B8 seam (the adopted two-rule stack) | 6.8393 | 8.0781 | 8.4162 | +0.087% |
| Lockstep, LSSA-B8-LITE seam (candidate variant) | 6.8393 | 8.0781 | 8.4162 | +0.087% |
| the FUSED verified stack, contract v2.1 (exact int8 linears, integer norms, Q14 RoPE, table SiLU, contract-v2 attention; T16m kernels) | 6.8511 | 8.0965 | 8.4363 | +0.260% |
| the FUSED verified stack at TP=4 (per-rank R4 padding, pinned rank-order reduction) | 6.8548 | 8.0970 | 8.4369 | +0.313% |

### Kernel standalone, ms per layer, B=1, 28 heads, 4 KV heads, head dim 128, causal, medians of 50

| Build | T=2048 | T=8192 | T=32768 | T=65536 | T=131072 |
|---|---|---|---|---|---|
| stock_first | 0.051 | 0.709 | 11.245 | 45.074 | 179.123 |
| b8vareng | 0.103 | 1.107 | 16.141 | 63.847 | 253.155 |
| stock_last | 0.051 | 0.709 | 11.238 | 45.050 | 179.705 |
| **exact / stock** | 2.020x | 1.561x | 1.435x | 1.416x | 1.413x |

### Prefill stage split, one 31,000-token prompt, max_tokens=1, medians of 3 (eager, CUDA events)

| Run | wall ms | prep | pass A | pass B (portable) | gather | attend (exact kernel) | host ms |
|---|---|---|---|---|---|---|---|
| lssab mnbt=8192 | 1081.1 | 15.3 | 0.3 | 0.3 | 2.6 | 462.5 | 384.6 |
| lssab mnbt=2048 | 1181.2 | 24.1 | 1.3 | 1.3 | 10.6 | 522.4 | 280.2 |
| stock mnbt=8192 | 1027.0 | - | - | - | - | - | - |

### Commitment cost: keyed GPU BLAKE3 over 8,256-byte per-(layer, token) leaves plus a binary parent tree, 28 layers

| Prompt tokens | leaves hashed | GB/s | us per prompt token | hash total ms (28 layers) | inline, % of stock TTFT | overlapped on a side stream, % of stock TTFT |
|---|---|---|---|---|---|---|
| 8192 | 1.764 GB | 759 | 0.492 | 4.03 | 1.94% | 1.95% |
| 32256 | 6.944 GB | 946 | 0.317 | 10.24 | 0.93% | 0.40% |

Gate: 8 of 8 leaves match the python reference and 8 of 8 match PyPI `blake3` 1.0.9. TTFT anchors are this session's stock p8192 / p32640 cells.

### Decode profile: GPU time per generated token per layer (300-token context, 64 tokens, CUDA graphs no inductor)

| Run | ms per token | attention-path kernels (us per token per layer) |
|---|---|---|
| lssab_b1 | 7.910 | nvjet_sm90_tst_192x8_64x8_4x1_v_bz 92.5; nvjet_sm90_tst_64x8_64x16_2x1_v_bz 45.6; lsb_prep 42.8; lsb_pass_b<9, 64, 1, 0> 27.2; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 13.8; nvjet_sm90_tst_384x8_64x4_2x1_v_bz 12.9; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 10.9; lsb_pass_a<9, 128> 4.9 |
| lssab_b1 (m1 incremental prep) | 6.936 | nvjet_sm90_tst_192x8_64x8_4x1_v_bz 92.4; nvjet_sm90_tst_64x8_64x16_2x1_v_bz 45.6; lsb_pass_b<9, 64, 1, 0> 27.1; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 13.8; nvjet_sm90_tst_384x8_64x4_2x1_v_bz 12.9; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 10.9; lsb_prep 8.2; lsb_pass_a<9, 128> 5.0 |
| lssab_b1 (m3 folded memsets + fused epilogue) | 6.910 | nvjet_sm90_tst_192x8_64x8_4x1_v_bz 92.4; nvjet_sm90_tst_64x8_64x16_2x1_v_bz 45.6; lsb_pass_b<9, 64, 1, 0> 32.5; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 13.8; nvjet_sm90_tst_384x8_64x4_2x1_v_bz 12.9; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 10.9; lsb_prep 8.0; lsb_pass_a<9, 128> 5.3 |
| lssab_b1 (m4 decode-shaped pass B v1) | 6.676 | nvjet_sm90_tst_192x8_64x8_4x1_v_bz 92.6; nvjet_sm90_tst_64x8_64x16_2x1_v_bz 45.6; lsb_dec 24.9; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 13.8; nvjet_sm90_tst_384x8_64x4_2x1_v_bz 12.9; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 10.9; lsb_prep 8.1; lsb_pass_a<9, 128> 5.3 |
| lssab_b1 (m5 decode-shaped pass B, one head per warp) | 6.498 | nvjet_sm90_tst_192x8_64x8_4x1_v_bz 91.9; nvjet_sm90_tst_64x8_64x16_2x1_v_bz 45.6; lsb_dec 19.0; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 13.8; nvjet_sm90_tst_384x8_64x4_2x1_v_bz 12.9; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 11.0; lsb_prep 8.0; lsb_pass_a<9, 128> 5.3 |
| lssab_b1 (m9 double-buffered chunks; file tagged m6) | 6.260 | nvjet_sm90_tst_192x8_64x8_4x1_v_bz 91.9; nvjet_sm90_tst_64x8_64x16_2x1_v_bz 45.6; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 13.8; lsb_dec 13.0; nvjet_sm90_tst_384x8_64x4_2x1_v_bz 12.9; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 11.0; lsb_prep 6.2; lsb_pass_a<9, 128> 5.3 |
| lssab_b1 (m10 cached FRAC table) | 6.238 | nvjet_sm90_tst_192x8_64x8_4x1_v_bz 91.9; nvjet_sm90_tst_64x8_64x16_2x1_v_bz 45.6; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 13.8; nvjet_sm90_tst_384x8_64x4_2x1_v_bz 12.9; lsb_dec 12.2; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 11.0; lsb_prep 6.2; lsb_pass_a<9, 128> 5.3 |
| lssab_b1 (m11 hybrid FRAC staging) | 6.254 | nvjet_sm90_tst_192x8_64x8_4x1_v_bz 91.9; nvjet_sm90_tst_64x8_64x16_2x1_v_bz 45.6; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 13.8; nvjet_sm90_tst_384x8_64x4_2x1_v_bz 12.9; lsb_dec 12.2; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 11.0; lsb_prep 6.2; lsb_pass_a<9, 128> 5.3 |
| lssab_b1 (CONTRACT v2: lsb_dec8, one plane, no pass A) | 6.293 | nvjet_sm90_tst_192x8_64x8_4x1_v_bz 92.2; nvjet_sm90_tst_64x8_64x16_2x1_v_bz 45.9; lsb_dec8 19.2; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 13.7; nvjet_sm90_tst_384x8_64x4_2x1_v_bz 12.9; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 11.0; lsb_prep 6.3; nvjet_sm90_tst_192x160_64x4_1x2_h_ 1.9 |
| lssab_b1 (CONTRACT v2c: lsb_dec8 + block mode) | 6.203 | nvjet_sm90_tst_192x8_64x8_4x1_v_bz 92.0; nvjet_sm90_tst_64x8_64x16_2x1_v_bz 45.9; lsb_dec8 16.0; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 13.8; nvjet_sm90_tst_384x8_64x4_2x1_v_bz 12.9; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 11.0; lsb_prep 6.3; nvjet_sm90_tst_192x160_64x4_1x2_h_ 1.9 |
| lssab_b1 (CONTRACT v2 v3b: SEG=32 units, staged folds) | 6.186 | nvjet_sm90_tst_192x8_64x8_4x1_v_bz 91.9; nvjet_sm90_tst_64x8_64x16_2x1_v_bz 45.4; lsb_dec8 15.8; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 13.8; nvjet_sm90_tst_384x8_64x4_2x1_v_bz 12.9; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 11.0; lsb_prep 6.3; nvjet_sm90_tst_192x160_64x4_1x2_h_ 1.9 |
| lssab_b1 (CONTRACT v2 FINAL: v4 kernel, cap 64) | 6.167 | nvjet_sm90_tst_192x8_64x8_4x1_v_bz 91.8; nvjet_sm90_tst_64x8_64x16_2x1_v_bz 45.5; lsb_dec8 14.9; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 13.8; nvjet_sm90_tst_384x8_64x4_2x1_v_bz 12.9; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 11.0; lsb_prep 6.3; nvjet_sm90_tst_192x160_64x4_1x2_h_ 1.9 |
| lssab_b1 (CONTRACT v2 v6: fused prep) | 6.240 | nvjet_sm90_tst_192x8_64x8_4x1_v_bz 92.1; nvjet_sm90_tst_64x8_64x16_2x1_v_bz 45.8; lsb_dec8 22.3; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 13.8; nvjet_sm90_tst_384x8_64x4_2x1_v_bz 12.9; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 10.9; nvjet_sm90_tst_192x160_64x4_1x2_h_ 1.9; nvjet_sm90_tst_256x104_64x4_2x1_v_ 1.1 |
| lssab_b1 (END-TO-END VERIFIED STACK) | 6.337 | cutlass::Kernel2<cutlass_80_tensor 80.4; cutlass::Kernel2<cutlass_80_tensor 61.4; lsb_dec8 22.7; dynnorm_k<256, 14> 13.3; r4quant_k 8.2; deq_k 8.0; rowquant_k<256, 14> 7.2; silumul_k 2.7 |
| lssab_b1 (CONTRACT v2 v7: half-block staging) | 6.226 | nvjet_sm90_tst_192x8_64x8_4x1_v_bz 92.1; nvjet_sm90_tst_64x8_64x16_2x1_v_bz 45.8; lsb_dec8 21.8; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 13.8; nvjet_sm90_tst_384x8_64x4_2x1_v_bz 12.9; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 10.9; nvjet_sm90_tst_192x160_64x4_1x2_h_ 1.9; nvjet_sm90_tst_256x104_64x4_2x1_v_ 1.1 |
| stock_b1 | 5.930 | nvjet_sm90_tst_192x8_64x8_4x1_v_bz 91.7; nvjet_sm90_tst_64x8_64x16_2x1_v_bz 45.8; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 13.6; nvjet_sm90_tst_384x8_64x4_2x1_v_bz 12.8; nvjet_sm90_tst_64x8_64x16_4x1_v_bz 10.9; cutlass::device_kernel<flash::enab 8.8; cutlass::device_kernel<flash::Flas 3.5; vllm::reshape_and_cache_flash_kern 2.2 |
| lssab_b32 | 12.490 | nvjet_sm90_tst_192x32_64x7_4x1_v_b 97.2; lsb_pass_b<9, 64, 1, 0> 57.9; nvjet_sm90_tst_128x32_64x10_4x1_v_ 47.6; lsb_prep 46.9; nvjet_sm90_tst_192x208_64x4_2x1_v_ 45.4; nvjet_sm90_tst_128x248_64x4_2x1_v_ 26.6; nvjet_sm90_tst_64x32_64x16_4x1_v_b 14.5; nvjet_sm90_tst_384x32_64x4_2x1_v_b 13.3 |
| lssab_b32 (m3 folded memsets + fused epilogue) | 11.663 | nvjet_sm90_tst_192x32_64x7_4x1_v_b 97.0; lsb_pass_b<9, 64, 1, 0> 63.2; nvjet_sm90_tst_128x32_64x10_4x1_v_ 47.7; nvjet_sm90_tst_192x208_64x4_2x1_v_ 45.0; nvjet_sm90_tst_128x248_64x4_2x1_v_ 26.4; lsb_prep 19.3; nvjet_sm90_tst_64x32_64x16_4x1_v_b 14.5; nvjet_sm90_tst_384x32_64x4_2x1_v_b 13.3 |
| lssab_b32 (m4 decode-shaped pass B v1) | 11.598 | nvjet_sm90_tst_192x32_64x7_4x1_v_b 94.6; lsb_dec 66.5; nvjet_sm90_tst_128x32_64x10_4x1_v_ 47.7; nvjet_sm90_tst_192x208_64x4_2x1_v_ 45.2; nvjet_sm90_tst_128x248_64x4_2x1_v_ 26.4; lsb_prep 19.4; nvjet_sm90_tst_64x32_64x16_4x1_v_b 14.6; nvjet_sm90_tst_384x32_64x4_2x1_v_b 13.3 |
| lssab_b32 (m5 decode-shaped pass B, one head per warp) | 10.449 | nvjet_sm90_tst_192x32_64x7_4x1_v_b 94.0; nvjet_sm90_tst_128x32_64x10_4x1_v_ 47.7; nvjet_sm90_tst_192x208_64x4_2x1_v_ 44.2; lsb_dec 27.6; nvjet_sm90_tst_128x248_64x4_2x1_v_ 25.9; lsb_prep 19.4; nvjet_sm90_tst_64x32_64x16_4x1_v_b 14.6; nvjet_sm90_tst_384x32_64x4_2x1_v_b 13.3 |
| lssab_b32 (m9 double-buffered chunks; file tagged m6) | 10.661 | nvjet_sm90_tst_192x32_64x7_4x1_v_b 94.2; nvjet_sm90_tst_128x32_64x10_4x1_v_ 47.7; nvjet_sm90_tst_192x208_64x4_2x1_v_ 45.8; lsb_dec 30.2; nvjet_sm90_tst_128x248_64x4_2x1_v_ 26.8; lsb_prep 21.4; nvjet_sm90_tst_64x32_64x16_4x1_v_b 14.6; nvjet_sm90_tst_384x32_64x4_2x1_v_b 13.3 |
| lssab_b32 (m10 cached FRAC table) | 10.603 | nvjet_sm90_tst_192x32_64x7_4x1_v_b 94.3; nvjet_sm90_tst_128x32_64x10_4x1_v_ 47.7; nvjet_sm90_tst_192x208_64x4_2x1_v_ 45.3; lsb_dec 28.6; nvjet_sm90_tst_128x248_64x4_2x1_v_ 26.5; lsb_prep 21.5; nvjet_sm90_tst_64x32_64x16_4x1_v_b 14.6; nvjet_sm90_tst_384x32_64x4_2x1_v_b 13.3 |
| lssab_b32 (m11 hybrid FRAC staging) | 10.519 | nvjet_sm90_tst_192x32_64x7_4x1_v_b 94.1; nvjet_sm90_tst_128x32_64x10_4x1_v_ 47.7; nvjet_sm90_tst_192x208_64x4_2x1_v_ 44.7; lsb_dec 28.5; nvjet_sm90_tst_128x248_64x4_2x1_v_ 26.2; lsb_prep 21.4; nvjet_sm90_tst_64x32_64x16_4x1_v_b 14.6; nvjet_sm90_tst_384x32_64x4_2x1_v_b 13.3 |
| lssab_b32 (CONTRACT v2: lsb_dec8, one plane, no pass A) | 10.166 | nvjet_sm90_tst_192x32_64x7_4x1_v_b 94.2; nvjet_sm90_tst_128x32_64x10_4x1_v_ 47.7; nvjet_sm90_tst_192x208_64x4_2x1_v_ 45.7; nvjet_sm90_tst_128x248_64x4_2x1_v_ 26.8; lsb_dec8 24.5; lsb_prep 21.0; nvjet_sm90_tst_64x32_64x16_4x1_v_b 14.6; nvjet_sm90_tst_384x32_64x4_2x1_v_b 13.3 |
| lssab_b32 (CONTRACT v2c: lsb_dec8 + block mode) | 10.109 | nvjet_sm90_tst_192x32_64x7_4x1_v_b 93.9; nvjet_sm90_tst_128x32_64x10_4x1_v_ 47.7; nvjet_sm90_tst_192x208_64x4_2x1_v_ 45.0; nvjet_sm90_tst_128x248_64x4_2x1_v_ 26.4; lsb_dec8 22.9; lsb_prep 21.0; nvjet_sm90_tst_64x32_64x16_4x1_v_b 14.6; nvjet_sm90_tst_384x32_64x4_2x1_v_b 13.3 |
| lssab_b32 (CONTRACT v2 v3b: SEG=32 units, staged folds) | 10.158 | nvjet_sm90_tst_192x32_64x7_4x1_v_b 94.0; nvjet_sm90_tst_128x32_64x10_4x1_v_ 47.7; nvjet_sm90_tst_192x208_64x4_2x1_v_ 45.3; nvjet_sm90_tst_128x248_64x4_2x1_v_ 26.5; lsb_dec8 25.7; lsb_prep 21.0; nvjet_sm90_tst_64x32_64x16_4x1_v_b 14.6; nvjet_sm90_tst_384x32_64x4_2x1_v_b 13.3 |
| lssab_b32 (CONTRACT v2 FINAL: v4 kernel, cap 64) | 10.168 | nvjet_sm90_tst_192x32_64x7_4x1_v_b 93.7; nvjet_sm90_tst_128x32_64x10_4x1_v_ 47.7; nvjet_sm90_tst_192x208_64x4_2x1_v_ 45.8; nvjet_sm90_tst_128x248_64x4_2x1_v_ 26.7; lsb_dec8 24.2; lsb_prep 21.1; nvjet_sm90_tst_64x32_64x16_4x1_v_b 14.6; nvjet_sm90_tst_384x32_64x4_2x1_v_b 13.3 |
| lssab_b32 (CONTRACT v2 v6: fused prep) | 9.794 | nvjet_sm90_tst_192x32_64x7_4x1_v_b 94.6; nvjet_sm90_tst_128x32_64x10_4x1_v_ 47.7; nvjet_sm90_tst_192x208_64x4_2x1_v_ 45.2; lsb_dec8 30.1; nvjet_sm90_tst_128x248_64x4_2x1_v_ 26.4; nvjet_sm90_tst_64x32_64x16_4x1_v_b 14.6; nvjet_sm90_tst_384x32_64x4_2x1_v_b 13.3; nvjet_sm90_tst_64x16_64x16_4x2_h_b 11.6 |
| lssab_b32 (END-TO-END VERIFIED STACK) | 11.465 | cutlass::Kernel2<cutlass_80_tensor 149.8; cutlass::Kernel2<cutlass_80_tensor 63.2; deq_k 37.3; lsb_dec8 29.4; silumul_k 20.4; cutlass::Kernel2<cutlass_80_tensor 17.3; r4quant_k 17.1; dynnorm_k<256, 14> 16.7 |
| lssab_b32 (CONTRACT v2 v7: half-block staging) | 10.001 | nvjet_sm90_tst_192x32_64x7_4x1_v_b 96.3; nvjet_sm90_tst_128x32_64x10_4x1_v_ 47.7; nvjet_sm90_tst_192x208_64x4_2x1_v_ 45.0; lsb_dec8 35.2; nvjet_sm90_tst_128x248_64x4_2x1_v_ 26.4; nvjet_sm90_tst_64x32_64x16_4x1_v_b 14.6; nvjet_sm90_tst_384x32_64x4_2x1_v_b 13.3; nvjet_sm90_tst_64x16_64x16_4x2_h_b 12.2 |
| stock_b32 | 9.241 | nvjet_sm90_tst_192x32_64x7_4x1_v_b 91.4; nvjet_sm90_tst_128x32_64x10_4x1_v_ 47.6; nvjet_sm90_tst_192x192_64x4_2x1_v_ 47.2; nvjet_sm90_tst_256x128_64x4_1x2_h_ 27.8; cutlass::device_kernel<flash::enab 18.9; nvjet_sm90_tst_64x32_64x16_4x1_v_b 14.5; nvjet_sm90_tst_384x32_64x4_2x1_v_b 13.3; nvjet_sm90_tst_64x16_64x16_4x2_h_b 11.4 |

### Kernel standalone, LSSA-B8-LITE vs filed vs stock, ms per layer

| Build | T=2048 | T=8192 | T=32768 | T=65536 | T=131072 | T=262144 |
|---|---|---|---|---|---|---|
| stock_first | 0.051 | 0.701 | 11.300 | 45.005 | 179.139 | 713.561 |
| b8liteeng | 0.089 | 1.003 | 14.764 | 58.122 | 230.790 | 919.899 |
| b8vareng | 0.102 | 1.105 | 16.128 | 63.448 | 251.548 | 1001.315 |
| stock_last | 0.052 | 0.708 | 11.145 | 45.206 | 180.452 | 716.745 |
| **b8liteeng / stock** | 1.745x | 1.431x | 1.307x | 1.291x | 1.288x | 1.289x |
| **b8vareng / stock** | 2.000x | 1.576x | 1.427x | 1.410x | 1.404x | 1.403x |

### Long context, Qwen2.5-7B-Instruct-1M (dual-chunk stripped, dense FA3 baseline), batch 1, no inductor, full graphs

| Cell | Stock | Lockstep, LITE prompt-row kernel | Lockstep, filed prompt-row kernel |
|---|---|---|---|
| ttft_p32768_b1 | 1100.8 | 1147.1 (0.960x) | 1180.1 (0.933x) |
| ttft_p65536_b1 | 2938.0 | 3111.0 (0.944x) | 3254.4 (0.903x) |
| ttft_p131072_b1 | 8819.5 | 9550.4 (0.923x) | 10132.7 (0.870x) |
| ttft_p262144_b1 | 29438.5 | 32624.7 (0.902x) | 34922.4 (0.843x) |
| decode_p131072_b1 | 96.0 | 53.8 (0.561x) | 52.5 (0.547x) |

Long-context TTFT with the CONTRACT v2 prompt kernel (b8v2eng, SEG=8) + lsb_dec8, vs stock (filed run), 1M model dense, B1:

| cell | stock | Lockstep v2 | ratio |
|---|---|---|---|
| TTFT 32k | 1107.2 ms | 1267.7 ms | 0.873x |
| TTFT 64k | 2938.6 ms | 3663.4 ms | 0.802x |
| TTFT 128k | 8820.7 ms | 11874.0 ms | 0.743x |
| TTFT 256k | 29298.8 ms | 41820.8 ms | 0.701x |

Long-context decode, 128-token slope (N=16 -> 144, median of 3), 1M model dense, LITE prompt kernel + m11 decode kernel, split cap 128:

| cell | stock | Lockstep | ratio |
|---|---|---|---|
| decode at 512k context, B1, 128-token slope, T16v (SPEC 7.14): whole verified stack with the two-group pipelined decode kernel | 66.40 tok/s | 59.29 tok/s | 0.893x |
| TTFT at 512k context, B1, T16v (SPEC 7.14): whole verified stack with the two-group pipelined decode kernel | 106090.2 ms | 117130.0 ms | 0.906x |
| decode at 512k context, B1, 128-token slope, SHIPPED KERNEL (T16r, SPEC 7.12): whole verified stack with the 16-warp decode kernel | 66.40 tok/s | 56.04 tok/s | 0.844x |
| TTFT at 512k context, B1, SHIPPED KERNEL (T16r, SPEC 7.12): whole verified stack with the 16-warp decode kernel | 106090.2 ms | 117125.4 ms | 0.906x |
| decode at 512k context, B1, 128-token slope, attention-only | 66.40 tok/s | 49.90 tok/s | 0.751x |
| TTFT at 512k context, B1, attention-only | 106090.2 ms | 119235.2 ms | 0.890x |
| decode at 512k context, B1, 128-token slope, END-TO-END VERIFIED STACK, FUSED, C++ ops, bf16 weights freed after quantisation (T16m v2.3) | 66.40 tok/s | 51.70 tok/s | 0.779x |
| TTFT at 512k context, B1, END-TO-END VERIFIED STACK, FUSED, C++ ops, bf16 weights freed after quantisation (T16m v2.3) | 106090.2 ms | 117139.9 ms | 0.906x |
| TTFT at 976k context, B1, attention-only, ONE H100 of the 4x H100 node (the harness runs TP=1; file tag historical) | 359802.0 ms | 447203.9 ms | 0.805x |
| decode at 976k context, B1, 128-token slope, WHOLE VERIFIABLE STACK, ONE H100 of the 4x H100 node (TP=1), pipelined decode kernel, both arms in one session (SPEC 7.14) | 37.55 tok/s | 38.16 tok/s | 1.016x |
| TTFT at 976k context, B1, WHOLE VERIFIABLE STACK, ONE H100 of the 4x H100 node (TP=1), pipelined decode kernel, both arms in one session (SPEC 7.14) | 359844.5 ms | 445224.2 ms | 0.808x |
| **decode at 32k context, B1, 128-token slope, SHIPPED long-context numbers (contract v2, v4 kernel unfused, cap 64)** | 152.29 tok/s | 130.31 tok/s | **0.856x** |
| decode at 32k context, B1, 128-token slope, v6 with the fusion FORCED ON (evidence: the shipped policy keeps fusion off above 64k, so the FINAL rows are the shipped long-context numbers) | 152.29 tok/s | 126.96 tok/s | 0.834x |
| decode at 32k context, B1, 128-token slope, v7 (half-block staging, fused prep) | 152.29 tok/s | 123.48 tok/s | 0.811x |
| decode at 32k context, B1, 128-token slope, T16v + split target 2 (SPEC 7.15): pipelined decode kernel, LOCKSTEP_LSSAB_SPLIT_CTA=2 (both arms at GMU 0.80) | 151.58 tok/s | 135.11 tok/s | 0.891x |
| TTFT at 32k context, B1, T16v + split target 2 (SPEC 7.15): pipelined decode kernel, LOCKSTEP_LSSAB_SPLIT_CTA=2 | 1101.8 ms | 1010.2 ms | 1.091x |
| decode at 32k context, B1, 128-token slope, T16v (SPEC 7.14): whole verified stack with the two-group pipelined decode kernel (both arms at GMU 0.80) | 151.58 tok/s | 135.18 tok/s | 0.892x |
| TTFT at 32k context, B1, T16v (SPEC 7.14): whole verified stack with the two-group pipelined decode kernel | 1101.8 ms | 1010.3 ms | 1.091x |
| decode at 32k context, B1, 128-token slope, SHIPPED KERNEL (T16r, SPEC 7.12): whole verified stack with the 16-warp decode kernel (both arms at GMU 0.80) | 151.58 tok/s | 131.90 tok/s | 0.870x |
| TTFT at 32k context, B1, SHIPPED KERNEL (T16r, SPEC 7.12): whole verified stack with the 16-warp decode kernel | 1101.8 ms | 1005.6 ms | 1.096x |
| decode at 32k context, B1, 128-token slope, END-TO-END VERIFIED STACK, FUSED, C++ ops, COMPILE CACHE OFF (the valid measurement) (both arms at GMU 0.80) | 151.58 tok/s | 130.72 tok/s | 0.862x |
| TTFT at 32k context, B1, END-TO-END VERIFIED STACK, FUSED, C++ ops, COMPILE CACHE OFF (the valid measurement) | 1101.8 ms | 1007.9 ms | 1.093x |
| decode at 32k context, B1, 128-token slope, SUPERSEDED: fused C++ ops but a STALE cached graph from an earlier build (compile cache keyed on the model config) (both arms at GMU 0.80) | 151.58 tok/s | 115.36 tok/s | 0.761x |
| TTFT at 32k context, B1, SUPERSEDED: fused C++ ops but a STALE cached graph from an earlier build (compile cache keyed on the model config) | 1101.8 ms | 1102.0 ms | 1.000x |
| decode at 32k context, B1, 128-token slope, SUPERSEDED: stale cached graph (see t16m24) (both arms at GMU 0.80) | 151.58 tok/s | 115.03 tok/s | 0.759x |
| TTFT at 32k context, B1, SUPERSEDED: stale cached graph (see t16m24) | 1101.8 ms | 1100.6 ms | 1.001x |
| decode at 32k context, B1, 128-token slope, attention-only control, same session, same GMU 0.80 stock arm (both arms at GMU 0.80) | 151.58 tok/s | 129.01 tok/s | 0.851x |
| TTFT at 32k context, B1, attention-only control, same session, same GMU 0.80 stock arm | 1101.8 ms | 1144.2 ms | 0.963x |
| decode at 32k context, B1, 128-token slope, END-TO-END VERIFIED STACK, fused glue only (T16m interim) (both arms at GMU 0.80) | 151.58 tok/s | 116.72 tok/s | 0.770x |
| TTFT at 32k context, B1, END-TO-END VERIFIED STACK, fused glue only (T16m interim) | 1101.8 ms | 1098.4 ms | 1.003x |
| decode at 32k context, B1, 128-token slope, END-TO-END VERIFIED STACK (T16L_ARM=1 + v2; both arms at GMU 0.80, the int8 weight copies need the headroom) | 151.58 tok/s | 114.84 tok/s | 0.758x |
| TTFT at 32k context, B1, END-TO-END VERIFIED STACK (exact int8 linears are faster than bf16 at this M and pay for the attention) | 1101.8 ms | 1100.0 ms | 1.002x |
| **decode at 64k context, B1, 128-token slope, SHIPPED long-context numbers (contract v2, v4 kernel unfused, cap 64)** | 139.87 tok/s | 109.36 tok/s | **0.782x** |
| decode at 64k context, B1, 128-token slope, v6 with the fusion FORCED ON (evidence: the shipped policy keeps fusion off above 64k, so the FINAL rows are the shipped long-context numbers) | 139.87 tok/s | 113.34 tok/s | 0.810x |
| **decode at 64k context, B1, 128-token slope, FINAL, units up to 128k (shipped policy)** | 139.87 tok/s | 117.30 tok/s | **0.839x** |
| **decode at 128k context, B1, 128-token slope, SHIPPED long-context numbers (contract v2, v4 kernel unfused, cap 64)** | 118.33 tok/s | 102.68 tok/s | **0.868x** |
| decode at 128k context, B1, 128-token slope, v6 with the fusion FORCED ON (evidence: the shipped policy keeps fusion off above 64k, so the FINAL rows are the shipped long-context numbers) | 118.33 tok/s | 97.65 tok/s | 0.825x |
| decode at 128k context, B1, 128-token slope, v7 (half-block staging, fused prep) | 118.33 tok/s | 89.53 tok/s | 0.757x |
| **decode at 128k context, B1, 128-token slope, FINAL, units up to 128k (shipped policy)** | 118.33 tok/s | 102.33 tok/s | **0.865x** |
| decode at 128k context, B1, 128-token slope, T16v + split target 2 (SPEC 7.15): pipelined decode kernel, LOCKSTEP_LSSAB_SPLIT_CTA=2 (both arms at GMU 0.80) | 118.73 tok/s | 113.63 tok/s | 0.957x |
| TTFT at 128k context, B1, T16v + split target 2 (SPEC 7.15): pipelined decode kernel, LOCKSTEP_LSSAB_SPLIT_CTA=2 | 8792.0 ms | 9148.0 ms | 0.961x |
| decode at 128k context, B1, 128-token slope, T16v (SPEC 7.14): whole verified stack with the two-group pipelined decode kernel (both arms at GMU 0.80) | 118.73 tok/s | 111.82 tok/s | 0.942x |
| TTFT at 128k context, B1, T16v (SPEC 7.14): whole verified stack with the two-group pipelined decode kernel | 8792.0 ms | 9148.4 ms | 0.961x |
| decode at 128k context, B1, 128-token slope, SHIPPED KERNEL (T16r, SPEC 7.12): whole verified stack with the 16-warp decode kernel (both arms at GMU 0.80) | 118.73 tok/s | 96.29 tok/s | 0.811x |
| TTFT at 128k context, B1, SHIPPED KERNEL (T16r, SPEC 7.12): whole verified stack with the 16-warp decode kernel | 8792.0 ms | 9146.6 ms | 0.961x |
| decode at 128k context, B1, 128-token slope, END-TO-END VERIFIED STACK, FUSED, C++ ops, COMPILE CACHE OFF (the valid measurement) (both arms at GMU 0.80) | 118.73 tok/s | 94.53 tok/s | 0.796x |
| TTFT at 128k context, B1, END-TO-END VERIFIED STACK, FUSED, C++ ops, COMPILE CACHE OFF (the valid measurement) | 8792.0 ms | 9159.7 ms | 0.960x |
| decode at 128k context, B1, 128-token slope, SUPERSEDED: fused C++ ops but a STALE cached graph from an earlier build (compile cache keyed on the model config) (both arms at GMU 0.80) | 118.73 tok/s | 86.74 tok/s | 0.731x |
| TTFT at 128k context, B1, SUPERSEDED: fused C++ ops but a STALE cached graph from an earlier build (compile cache keyed on the model config) | 8792.0 ms | 9086.5 ms | 0.968x |
| decode at 128k context, B1, 128-token slope, SUPERSEDED: stale cached graph (see t16m24) (both arms at GMU 0.80) | 118.73 tok/s | 87.42 tok/s | 0.736x |
| TTFT at 128k context, B1, SUPERSEDED: stale cached graph (see t16m24) | 8792.0 ms | 9101.2 ms | 0.966x |
| decode at 128k context, B1, 128-token slope, attention-only control, same session, same GMU 0.80 stock arm (both arms at GMU 0.80) | 118.73 tok/s | 94.75 tok/s | 0.798x |
| TTFT at 128k context, B1, attention-only control, same session, same GMU 0.80 stock arm | 8792.0 ms | 9554.5 ms | 0.920x |
| decode at 128k context, B1, 128-token slope, END-TO-END VERIFIED STACK, fused glue only (T16m interim) (both arms at GMU 0.80) | 118.73 tok/s | 86.20 tok/s | 0.726x |
| TTFT at 128k context, B1, END-TO-END VERIFIED STACK, fused glue only (T16m interim) | 8792.0 ms | 9060.5 ms | 0.970x |
| decode at 128k context, B1, 128-token slope, END-TO-END VERIFIED STACK (T16L_ARM=1 + v2; both arms at GMU 0.80, the int8 weight copies need the headroom) | 118.73 tok/s | 87.54 tok/s | 0.737x |
| TTFT at 128k context, B1, END-TO-END VERIFIED STACK (exact int8 linears are faster than bf16 at this M and pay for the attention) | 8792.0 ms | 9063.5 ms | 0.970x |
| **decode at 256k context, B1, 128-token slope, SHIPPED long-context numbers (contract v2, v4 kernel unfused, cap 64)** | 94.03 tok/s | 82.64 tok/s | **0.879x** |
| decode at 256k context, B1, 128-token slope, v6 with the fusion FORCED ON (evidence: the shipped policy keeps fusion off above 64k, so the FINAL rows are the shipped long-context numbers) | 94.03 tok/s | 79.78 tok/s | 0.849x |
| decode at 256k context, B1, 128-token slope, v7 (half-block staging, fused prep) | 94.03 tok/s | 70.44 tok/s | 0.749x |
| decode at 32k context, B1, 128-token slope, CONTRACT v2 v3 | 152.29 tok/s | 129.65 tok/s | 0.851x |
| decode at 32k context, B1, 128-token slope, CONTRACT v2 v4 | 152.29 tok/s | 127.67 tok/s | 0.838x |
| decode at 32k context, B1, 128-token slope, CONTRACT v2 v4cap64 | 152.29 tok/s | 130.80 tok/s | 0.859x |
| decode at 32k context, B1, 128-token slope, CONTRACT v2 v4cap32 | 152.29 tok/s | 128.96 tok/s | 0.847x |
| decode at 32k context, B1, 128-token slope, CONTRACT v2 v5cap128 | 152.29 tok/s | 125.15 tok/s | 0.822x |
| decode at 32k context, B1, 128-token slope, CONTRACT v2 v5cap64 | 152.29 tok/s | 128.69 tok/s | 0.845x |
| decode at 128k context, B1, 128-token slope, m11 (contract v1 decode) | 118.33 tok/s | 89.35 tok/s | 0.755x |
| decode at 128k context, B1, 128-token slope, CONTRACT v2 v5, split cap 128 | 118.33 tok/s | 87.35 tok/s | 0.738x |
| decode at 128k context, B1, 128-token slope, CONTRACT v2 v5, split cap 64 | 118.33 tok/s | 95.80 tok/s | 0.810x |
| decode at 128k context, B1, 128-token slope, CONTRACT v2 v4, split cap 64 | 118.33 tok/s | 102.61 tok/s | 0.867x |
| decode at 128k context, B1, 128-token slope, CONTRACT v2 v4, split cap 32 | 118.33 tok/s | 103.16 tok/s | 0.872x |
| decode at 128k context, B1, 128-token slope, CONTRACT v2 v4 (vector q, cooperative V transpose) | 118.33 tok/s | 91.53 tok/s | 0.774x |
| decode at 128k context, B1, 128-token slope, CONTRACT v2 v3 (SEG=32 units, staged folds) | 118.33 tok/s | 91.12 tok/s | 0.770x |
| decode at 128k context, B1, 128-token slope, CONTRACT v2 (lsb_dec8) | 118.33 tok/s | 56.64 tok/s | 0.479x |
| decode at 256k context, B1, 128-token slope, m11 (contract v1 decode) | 94.03 tok/s | 63.38 tok/s | 0.674x |
| decode at 256k context, B1, 128-token slope, CONTRACT v2 v5, split cap 128 | 94.03 tok/s | 79.62 tok/s | 0.847x |
| decode at 256k context, B1, 128-token slope, CONTRACT v2 v5, split cap 64 | 94.03 tok/s | 78.96 tok/s | 0.840x |
| decode at 256k context, B1, 128-token slope, CONTRACT v2 v4, split cap 64 | 94.03 tok/s | 83.12 tok/s | 0.884x |
| decode at 256k context, B1, 128-token slope, CONTRACT v2 v4, split cap 32 | 94.03 tok/s | 73.00 tok/s | 0.776x |
| decode at 256k context, B1, 128-token slope, CONTRACT v2 v4 (vector q, cooperative V transpose) | 94.03 tok/s | 82.43 tok/s | 0.877x |
| decode at 256k context, B1, 128-token slope, CONTRACT v2 v3 (SEG=32 units, staged folds) | 94.03 tok/s | 76.35 tok/s | 0.812x |
| decode at 256k context, B1, 128-token slope, CONTRACT v2 (lsb_dec8) | 94.03 tok/s | 32.13 tok/s | 0.342x |

8-token-slope cells (noisy: they subtract two ~10 s runs; kept as evidence):

- 131k decode, Lockstep m9, split cap 32: 77.97 tok/s (stock: 96.0 / 104.8 / 355.2 tok/s over its runs)
- 131k decode, Lockstep m9, split cap 128: 97.23 tok/s (stock: 96.0 / 104.8 / 355.2 tok/s over its runs)
- 131k decode, Lockstep m10, split cap 32: 88.18 tok/s (stock: 96.0 / 104.8 / 355.2 tok/s over its runs)
- 131k decode, Lockstep m10, split cap 128: 86.84 tok/s (stock: 96.0 / 104.8 / 355.2 tok/s over its runs)
- 131k decode, Lockstep m11, split cap 128: 88.18 tok/s (stock: 96.0 / 104.8 / 355.2 tok/s over its runs)
- 131k decode, Lockstep m11, split cap 128: 95.28 tok/s (stock: 96.0 / 104.8 / 355.2 tok/s over its runs)

### The end-to-end example: four real chat prompts (46 / 51 / 1,439 / 2,869 tokens), greedy, 64 tokens

| Run | mode | TTFT alone ms (short / code / summary / needle) | batched TTFT ms | decode tok/s (4 streams) | run == re-run | alone == batched | needle answered | sensible text |
|---|---|---|---|---|---|---|---|---|
| lssab | FULL_AND_PIECEWISE, no inductor [72B TP=4, whole stack, routed native reduction (cut 4,096), both arms under NCCL_GRAPH_MIXING_SUPPORT=0 (7.13, shipped)] | 24.2 / 27.2 / 146.3 / 264.8 | 340.1 | 183.9 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor [72B TP=4, whole stack, routed native reduction, both arms under NCCL_GRAPH_MIXING_SUPPORT=0 (7.13)] | 24.2 / 27.2 / 134.4 / 233.2 | 340.8 | 183.1 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor [72B TP=4, whole stack, pynccl all-to-all reduction from 512 rows (7.13)] | 27.6 / 28.1 / 130.4 / 230.5 | 341.2 | 150.8 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor [72B TP=4, whole stack, 16-warp kernel + NATIVE all-to-all reduction from 512 rows (7.13)] | 26.2 / 29.2 / 131.4 / 231.1 | 340.7 | 173.2 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor [SUPERSEDED: native reduction on every row count (7.13)] | 26.1 / 29.3 / 130.3 / 231.2 | 339.3 | 172.3 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor [72B TP=4, whole stack, SHIPPED 16-warp kernel (7.12)] | 24.8 / 27.7 / 144.9 / 262.0 | 394.4 | 181.6 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor [7B, whole stack, SHIPPED 16-warp kernel (7.12)] | 9.0 / 8.8 / 36.5 / 66.1 | 92.9 | 484.8 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor [72B TP=4, whole stack, chunked-fold kernel] | 25.4 / 27.7 / 144.8 / 262.0 | 393.0 | 178.3 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor [7B, whole stack, chunked-fold kernel (t16m24)] | 8.8 / 8.7 / 36.0 / 66.0 | 92.7 | 490.3 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor [7B, whole stack (t16m23)] | 11.3 / 11.2 / 38.2 / 68.0 | 95.0 | 488.3 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor | 11.4 / 11.5 / 38.5 / 68.6 | 95.4 | 487.0 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor | 11.7 / 11.8 / 47.4 / 85.7 | 123.8 | 442.5 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (CONTRACT v2 decode) | 26.6 / 26.4 / 114.4 / 211.0 | 320.2 | 176.7 | True | False | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (CONTRACT v2 decode) | 18.6 / 18.9 / 73.0 / 145.6 | 219.0 | 255.5 | True | False | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (CONTRACT v2 decode) | 11.9 / 11.9 / 53.6 / 97.5 | 141.4 | 426.6 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (CONTRACT v2 decode) | 11.9 / 12.1 / 38.9 / 72.5 | 109.8 | 469.8 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (CONTRACT v2 decode) | 12.0 / 12.0 / 38.8 / 71.2 | 109.5 | 462.4 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (CONTRACT v2 decode) | 12.0 / 12.2 / 37.8 / 73.0 | 110.1 | 467.7 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (CONTRACT v2 decode) | 12.1 / 12.1 / 38.0 / 71.9 | 110.1 | 462.5 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (CONTRACT v2 decode) | 12.1 / 12.1 / 38.0 / 71.6 | 112.0 | 468.5 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (CONTRACT v2 decode) | 11.9 / 11.9 / 38.3 / 71.8 | 110.4 | 460.5 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (CONTRACT v2 decode) | 12.1 / 12.1 / 38.2 / 72.3 | 110.2 | 452.1 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (CONTRACT v2 decode, block mode) | 12.0 / 12.1 / 38.2 / 71.9 | 111.3 | 433.4 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (CONTRACT v2 decode) | 12.0 / 12.1 / 37.6 / 72.0 | 108.8 | 433.8 | True | True | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (m11 decode kernel) | 12.2 / 12.2 / 37.8 / 72.2 | 111.7 | 481.4 | True | False | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (m10 decode kernel) | 12.0 / 12.3 / 37.3 / 72.7 | 111.1 | 477.5 | True | False | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (m9 decode kernel) | 12.1 / 12.6 / 38.4 / 74.6 | 109.6 | 474.5 | True | False | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (m8 decode kernel) | 12.2 / 12.2 / 38.6 / 73.2 | 109.9 | 468.2 | True | False | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor (after T16m host-cost change) | 12.0 / 12.1 / 37.8 / 71.6 | 110.0 | 358.4 | True | False | True | yes |
| lssab | FULL_AND_PIECEWISE, no inductor | 12.2 / 12.3 / 38.2 / 71.1 | 109.1 | 357.9 | True | False | True | yes |
| lssab | FULL_AND_PIECEWISE, inductor on (after the capture fix) | 9.7 / 9.9 / 35.8 / 68.7 | 112.4 | 369.5 | True | False | True | yes |
| lssab | eager | 13.0 / 13.1 / 37.1 / 68.7 | 102.4 | 261.3 | True | False | True | yes |
| lssab | PIECEWISE, inductor on | 9.7 / 9.8 / 35.8 / 68.9 | 109.1 | 365.2 | True | False | True | yes |
| lssab | FULL_AND_PIECEWISE, inductor on (BEFORE the capture fix) | 9.7 / 9.7 / 36.4 / 69.5 | 107.3 | 726.4 | False | False | False | NO |
| stock | FULL_AND_PIECEWISE, no inductor [72B TP=4 stock] (CONTRACT v2 decode) | 23.2 / 22.6 / 107.3 / 209.1 | 310.6 | 191.0 | True | True | True | yes |
| stock | FULL_AND_PIECEWISE, no inductor [14B stock] (CONTRACT v2 decode) | 17.0 / 16.8 / 69.4 / 143.2 | 207.7 | 266.4 | True | False | True | yes |
| stock | FULL_AND_PIECEWISE, no inductor [7B stock] | 10.7 / 10.8 / 36.0 / 69.0 | 106.9 | 502.4 | True | False | True | yes |
| stock | FULL_AND_PIECEWISE -> vLLM disables graphs (eager engine) | 8.5 / 8.4 / 32.6 / 66.4 | 102.9 | 375.0 | True | True | True | yes |
| stock | PIECEWISE, inductor on | 8.4 / 8.4 / 33.5 / 67.7 | 103.4 | 510.1 | True | True | True | yes |
| stock | FULL_AND_PIECEWISE, inductor on | 8.7 / 8.4 / 34.5 / 66.8 | 104.3 | 520.5 | True | False | True | yes |

Lockstep arm: prompt rows by `LSSA-B8 exact kernel in FlashAttention-3 (lssab-b8-1launch-varlen)`, generated rows by `LSSA-B portable split-KV decode (lssa_lssab.so)`, errflags=[0, 16595920, 1441857536, 0], key offset `keq_qwen25_7b_instruct.pt`

Answers (Lockstep, eager):

- **short** (46 prompt tokens): Integer addition is associative because integers are exact and do not suffer from rounding errors, meaning that the way in which numbers are grouped does not affect the result. In contrast, floating-point addition is not associative due to rounding errors and loss of precision, which can lead to dif
- **code** (51 prompt tokens): Certainly! Below is a Python function `fib(n)` that returns the n-th Fibonacci number iteratively. The function includes a docstring that explains its purpose, parameters, and return value.  ```python def fib(n):     """     Calculate the n-th Fibonacci number iteratively.      Parameters:     n (in
- **summary** (1439 prompt tokens): Robert Boulter is a versatile British actor who has appeared in various television series, plays, and films since the early 2000s, including roles in "The Bill," "Judge John Deed," and "The Long Firm," as well as starring in productions like "Mercury Fur" and "
- **needle** (2869 prompt tokens): lockstep-7a42

Verifier (H100 GPU, every dumped row): **PASS**, prompt rows 370020 bad 0, generated rows 2352 bad 0, maxdiff 0.0e+00, 12+84 sequence-steps, 3 s

Verifier (H100 host CPU, sampled sequence-steps, no CUDA): **PASS**, prompt rows 370020 bad 0, generated rows 168 bad 0, maxdiff 0.0e+00, 12+6 sequence-steps, 105 s

### Gates

- E6n: CUDA-graph capture is bit-inert (no inductor): full == piecewise == eager token ids: **PASS**
  - `{"shape": "p1024x3", "full_eq_piecewise": true, "full_eq_eager": true, "piecewise_eq_eager": true, "full_self_consistent": true, "full_alone_eq_batched": true, "eager_alone_eq_batched": true}`
  - `{"shape": "p46x3", "full_eq_piecewise": true, "full_eq_eager": true, "piecewise_eq_eager": true, "full_self_consistent": true, "full_alone_eq_batched": false, "eager_alone_eq_batched": false}`
  - `{"shape": "p1000x3", "full_eq_piecewise": true, "full_eq_eager": true, "piecewise_eq_eager": true, "full_self_consistent": true, "full_alone_eq_batched": true, "eager_alone_eq_batched": true}`
  - `{"shape": "p2048x2_mid", "full_eq_piecewise": true, "full_eq_eager": true, "piecewise_eq_eager": true, "full_self_consistent": true, "full_alone_eq_batched": false, "eager_alone_eq_batched": false}`
  - `{"shape": "p128x8", "full_eq_piecewise": true, "full_eq_eager": true, "piecewise_eq_eager": true, "full_self_consistent": true, "full_alone_eq_batched": false, "eager_alone_eq_batched": false}`
  - `{"shape": "stock_p1024x3", "stock_full_eq_eager": false, "stock_full_self_consistent": true}`
- E8m: token identity before vs after the T16m host-cost change: **PASS**
  - `{"shape": "p1024x3", "mode": "eager", "after_eq_before": true, "self_consistent": true, "cached_layers": null}`
  - `{"shape": "p1024x3", "mode": "full", "after_eq_before": true, "self_consistent": true, "cached_layers": null}`
  - `{"shape": "p46x3", "mode": "eager", "after_eq_before": true, "self_consistent": true, "cached_layers": null}`
  - `{"shape": "p46x3", "mode": "full", "after_eq_before": true, "self_consistent": true, "cached_layers": null}`
  - `{"shape": "p1000x3", "mode": "eager", "after_eq_before": true, "self_consistent": true, "cached_layers": null}`
  - `{"shape": "p1000x3", "mode": "full", "after_eq_before": true, "self_consistent": true, "cached_layers": null}`
  - `{"shape": "p2048x2_mid", "mode": "eager", "after_eq_before": true, "self_consistent": true, "cached_layers": null}`
  - `{"shape": "p2048x2_mid", "mode": "full", "after_eq_before": true, "self_consistent": true, "cached_layers": null}`
  - `{"shape": "p128x8", "mode": "eager", "after_eq_before": true, "self_consistent": true, "cached_layers": null}`
  - `{"shape": "p128x8", "mode": "full", "after_eq_before": true, "self_consistent": true, "cached_layers": null}`
- E8p: token identity before vs after the incremental prep (m1): **PASS**
  - `{"shape": "p1024x3", "mode": "eager", "after_eq_before": true, "self_consistent": true, "counters": {"tiles": 3049872, "tiles_dead": 48, "ctas": 388752, "vsh_max": 4, "vsh_violations": 0, "shadow_exhausted": 0, "shadow_evicted": 0, "prep_grid_short": 0}}`
  - `{"shape": "p1024x3", "mode": "full", "after_eq_before": true, "self_consistent": true, "counters": {"tiles": 3059616, "tiles_dead": 48, "ctas": 398496, "vsh_max": 4, "vsh_violations": 0, "shadow_exhausted": 0, "shadow_evicted": 0, "prep_grid_short": 0}}`
  - `{"shape": "p46x3", "mode": "eager", "after_eq_before": true, "self_consistent": true, "counters": {"tiles": 313152, "tiles_dead": 0, "ctas": 138768, "vsh_max": 0, "vsh_violations": 0, "shadow_exhausted": 0, "shadow_evicted": 0, "prep_grid_short": 0}}`
  - `{"shape": "p46x3", "mode": "full", "after_eq_before": true, "self_consistent": true, "counters": {"tiles": 322896, "tiles_dead": 0, "ctas": 148512, "vsh_max": 0, "vsh_violations": 0, "shadow_exhausted": 0, "shadow_evicted": 0, "prep_grid_short": 0}}`
  - `{"shape": "p1000x3", "mode": "eager", "after_eq_before": true, "self_consistent": true, "counters": {"tiles": 2638608, "tiles_dead": 0, "ctas": 364560, "vsh_max": 4, "vsh_violations": 0, "shadow_exhausted": 0, "shadow_evicted": 0, "prep_grid_short": 0}}`
  - `{"shape": "p1000x3", "mode": "full", "after_eq_before": true, "self_consistent": true, "counters": {"tiles": 2648352, "tiles_dead": 0, "ctas": 374304, "vsh_max": 4, "vsh_violations": 0, "shadow_exhausted": 0, "shadow_evicted": 0, "prep_grid_short": 0}}`
  - `{"shape": "p2048x2_mid", "mode": "eager", "after_eq_before": true, "self_consistent": true, "counters": {"tiles": 6364512, "tiles_dead": 56, "ctas": 364896, "vsh_max": 5, "vsh_violations": 0, "shadow_exhausted": 0, "shadow_evicted": 0, "prep_grid_short": 0}}`
  - `{"shape": "p2048x2_mid", "mode": "full", "after_eq_before": true, "self_consistent": true, "counters": {"tiles": 6374256, "tiles_dead": 56, "ctas": 374640, "vsh_max": 5, "vsh_violations": 0, "shadow_exhausted": 0, "shadow_evicted": 0, "prep_grid_short": 0}}`
  - `{"shape": "p128x8", "mode": "eager", "after_eq_before": true, "self_consistent": true, "counters": {"tiles": 685440, "tiles_dead": 0, "ctas": 274176, "vsh_max": 5, "vsh_violations": 0, "shadow_exhausted": 0, "shadow_evicted": 3, "prep_grid_short": 0}}`
  - `{"shape": "p128x8", "mode": "full", "after_eq_before": true, "self_consistent": true, "counters": {"tiles": 695184, "tiles_dead": 0, "ctas": 283920, "vsh_max": 5, "vsh_violations": 0, "shadow_exhausted": 0, "shadow_evicted": 3, "prep_grid_short": 0}}`
- E8t: token identity before vs after the decode-shaped pass B (m5): **PASS**
- E8w: token identity, m8: **PASS**
- E8x: token identity, m10 (m9 passed the same cells; see hb_q25.log): **PASS**
- E6final: CUDA-graph capture bit-inert under the FINAL contract-v2 build (full == eager token ids, 4 shapes): **PASS**
  - `{"tag": "p1024x3", "ok": true}`
  - `{"tag": "p128x8", "ok": true}`
  - `{"tag": "p2048x2_mid", "ok": true}`
  - `{"tag": "p46x3", "ok": true}`
- E6 on Qwen2.5-14B-Instruct (v8 kernel): full == eager token ids: **PASS**
- E6 on the FUSED verified stack, C++ ops, compile cache off: full == eager token ids: **PASS**
- E6 on the FUSED verified stack with C++-registered ops (T16m v2.3): full == eager token ids: **PASS**
- E6 on the FUSED verified stack (T16m v2.2, routed GEMM): full == eager token ids: **PASS**
- E6 on the fused verified stack (T16m interim): full == eager token ids: **PASS**
- E6 on Qwen2.5-72B-Instruct, TP=4: full == eager token ids: **PASS**
- E6 on the end-to-end verified stack (T16L_ARM=1 + v2): full == eager token ids: **PASS**
- E6v2: CUDA-graph capture bit-inert under CONTRACT v2 (full == eager token ids): **PASS**
- E6i (informational): the same probe with inductor ON: **FAIL**
  - `{"shape": "p1024x3", "full_eq_piecewise": false, "full_eq_eager": false, "piecewise_eq_eager": false, "full_self_consistent": true, "full_alone_eq_batched": true, "eager_alone_eq_batched": true}`
  - `{"shape": "p46x3", "full_eq_piecewise": false, "full_eq_eager": false, "piecewise_eq_eager": false, "full_self_consistent": true, "full_alone_eq_batched": false, "eager_alone_eq_batched": false}`
  - `{"shape": "p1000x3", "full_eq_piecewise": false, "full_eq_eager": false, "piecewise_eq_eager": true, "full_self_consistent": true, "full_alone_eq_batched": true, "eager_alone_eq_batched": true}`
  - `{"shape": "p2048x2_mid", "full_eq_piecewise": false, "full_eq_eager": false, "piecewise_eq_eager": true, "full_self_consistent": true, "full_alone_eq_batched": true, "eager_alone_eq_batched": false}`
  - `{"shape": "p128x8", "full_eq_piecewise": false, "full_eq_eager": true, "piecewise_eq_eager": false, "full_self_consistent": true, "full_alone_eq_batched": true, "eager_alone_eq_batched": false}`
  - `{"shape": "stock_p1024x3", "stock_full_eq_eager": false, "stock_full_self_consistent": true}`
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) (after T16m): **PASS** rows=336000 bad=0 maxdiff=0.000e+00 (30 sequence-steps)
- E4(mu ON) GENERATED rows vs lssab_torch(mu) (after T16m): **PASS** rows=2352 bad=0 maxdiff=0.000e+00 (84 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) (after T16m): **PASS** rows=516096 bad=0 maxdiff=0.000e+00 (9 sequence-steps)
- E4(mu ON) GENERATED rows vs lssab_torch(mu) (after T16m): **PASS** rows=1764 bad=0 maxdiff=0.000e+00 (63 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) (after T16m): **PASS** rows=336000 bad=0 maxdiff=0.000e+00 (30 sequence-steps)
- E4(mu ON) GENERATED rows vs lssab_torch(mu) (after T16m): **PASS** rows=2352 bad=0 maxdiff=0.000e+00 (84 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) (after T16m): **PASS** rows=50400 bad=0 maxdiff=0.000e+00 (6 sequence-steps)
- E4(mu ON) GENERATED rows vs lssab_torch(mu) (after T16m): **PASS** rows=33432 bad=0 maxdiff=0.000e+00 (1194 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) (after T16m): **PASS** rows=336000 bad=0 maxdiff=0.000e+00 (30 sequence-steps)
- E4(mu ON) GENERATED rows vs lssab_torch(mu) (after T16m): **PASS** rows=2352 bad=0 maxdiff=0.000e+00 (84 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) (after T16m): **PASS** rows=516096 bad=0 maxdiff=0.000e+00 (9 sequence-steps)
- E4(mu ON) GENERATED rows vs lssab_torch(mu) (after T16m): **PASS** rows=1764 bad=0 maxdiff=0.000e+00 (63 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) (after T16m): **PASS** rows=516096 bad=0 maxdiff=0.000e+00 (9 sequence-steps)
- E4(mu ON) GENERATED rows vs lssab_torch(mu) (after T16m): **PASS** rows=1764 bad=0 maxdiff=0.000e+00 (63 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_mid_v2]: **PASS** rows=336000 bad=0 maxdiff=0.000e+00 (30 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_mid_v2]: **PASS** rows=2352 bad=0 maxdiff=0.000e+00 (84 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_long_v2]: **PASS** rows=112000 bad=0 maxdiff=0.000e+00 (4 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_long_v2]: **PASS** rows=13328 bad=0 maxdiff=0.000e+00 (476 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_2k_v2full]: **PASS** rows=516096 bad=0 maxdiff=0.000e+00 (9 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_2k_v2full]: **PASS** rows=1764 bad=0 maxdiff=0.000e+00 (63 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_5k_v2full]: **PASS** rows=280000 bad=0 maxdiff=0.000e+00 (6 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_5k_v2full]: **PASS** rows=392 bad=0 maxdiff=0.000e+00 (14 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_mid_v2b]: **PASS** rows=504000 bad=0 maxdiff=0.000e+00 (48 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_mid_v2b]: **PASS** rows=19656 bad=0 maxdiff=0.000e+00 (702 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_short_v2c]: **PASS** rows=23184 bad=0 maxdiff=0.000e+00 (18 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_short_v2c]: **PASS** rows=19656 bad=0 maxdiff=0.000e+00 (702 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_mid_v2c]: **PASS** rows=112000 bad=0 maxdiff=0.000e+00 (10 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_mid_v2c]: **PASS** rows=6608 bad=0 maxdiff=0.000e+00 (236 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_mid_v3]: **PASS** rows=504000 bad=0 maxdiff=0.000e+00 (48 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_mid_v3]: **PASS** rows=19656 bad=0 maxdiff=0.000e+00 (702 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_5k_v3]: **PASS** rows=280000 bad=0 maxdiff=0.000e+00 (6 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_5k_v3]: **PASS** rows=2184 bad=0 maxdiff=0.000e+00 (78 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_mid_v3b]: **PASS** rows=504000 bad=0 maxdiff=0.000e+00 (48 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_mid_v3b]: **PASS** rows=19656 bad=0 maxdiff=0.000e+00 (702 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_mid_v4]: **PASS** rows=504000 bad=0 maxdiff=0.000e+00 (48 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_mid_v4]: **PASS** rows=19656 bad=0 maxdiff=0.000e+00 (702 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_mid_v5]: **PASS** rows=504000 bad=0 maxdiff=0.000e+00 (48 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_mid_v5]: **PASS** rows=19656 bad=0 maxdiff=0.000e+00 (702 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_mid_final]: **PASS** rows=504000 bad=0 maxdiff=0.000e+00 (48 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_mid_final]: **PASS** rows=19656 bad=0 maxdiff=0.000e+00 (702 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_long_final]: **PASS** rows=50400 bad=0 maxdiff=0.000e+00 (6 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_long_final]: **PASS** rows=33432 bad=0 maxdiff=0.000e+00 (1194 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_mid_v6]: **PASS** rows=504000 bad=0 maxdiff=0.000e+00 (48 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_mid_v6]: **PASS** rows=19656 bad=0 maxdiff=0.000e+00 (702 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_long_v6]: **PASS** rows=50400 bad=0 maxdiff=0.000e+00 (6 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_long_v6]: **PASS** rows=33432 bad=0 maxdiff=0.000e+00 (1194 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_mid_v7]: **PASS** rows=504000 bad=0 maxdiff=0.000e+00 (48 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_mid_v7]: **PASS** rows=19656 bad=0 maxdiff=0.000e+00 (702 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_long_v7]: **PASS** rows=50400 bad=0 maxdiff=0.000e+00 (6 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_long_v7]: **PASS** rows=33432 bad=0 maxdiff=0.000e+00 (1194 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_mid_v8_7b]: **PASS** rows=504000 bad=0 maxdiff=0.000e+00 (48 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_mid_v8_7b]: **PASS** rows=19656 bad=0 maxdiff=0.000e+00 (702 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_long_v8b_7b]: **PASS** rows=50400 bad=0 maxdiff=0.000e+00 (6 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_long_v8b_7b]: **PASS** rows=9912 bad=0 maxdiff=0.000e+00 (354 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_mid_14b]: **PASS** rows=720000 bad=0 maxdiff=0.000e+00 (48 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_mid_14b]: **PASS** rows=28080 bad=0 maxdiff=0.000e+00 (702 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_long_14b]: **PASS** rows=72000 bad=0 maxdiff=0.000e+00 (6 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_long_14b]: **PASS** rows=28560 bad=0 maxdiff=0.000e+00 (714 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu) [e1_72b]: **PASS** rows=86400 bad=0 maxdiff=0.000e+00 (18 sequence-steps)
- E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target] [e1_72b]: **PASS** rows=16992 bad=0 maxdiff=0.000e+00 (1062 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu): **PASS** rows=336000 bad=0 maxdiff=0.000e+00 (30 sequence-steps)
- E4(mu ON) GENERATED rows vs lssab_torch(mu): **PASS** rows=2352 bad=0 maxdiff=0.000e+00 (84 sequence-steps)
- E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu): **PASS** rows=516096 bad=0 maxdiff=0.000e+00 (9 sequence-steps)
- E4(mu ON) GENERATED rows vs lssab_torch(mu): **PASS** rows=1764 bad=0 maxdiff=0.000e+00 (63 sequence-steps)
- E7a(mu ON) prep q8/e_q byte-equal to host reference: **PASS** elements=132572160 bad=0
- E7b(mu ON) prep KV lattice + e_k/e_v byte-equal to host ref: **PASS** sequences=48 mismatched-tensors=0 []
- `gate_selftest.log`: PASS  S11 arm (12 cases, arm L8-RN-W8)               2.9s
- `gate_selftest.log`: PASS  S11 arm (12 cases, arm L8-TR)                  3.7s
- `gate_selftest.log`: PASS  S11 arm (12 cases, arm L8-RN-n8)               2.9s
- `gate_selftest.log`: PASS  S11 arm (12 cases, arm L8-RN-KB64)             3.7s
- `gate_selftest.log`: PASS  S11 arm (12 cases, arm L16)                    2.9s
- `gate_selftest.log`: LSSA-B8 SELFTEST: 28/28 checks PASS
- `selftest_seg8.log`: PASS  S7 segment-split fold == sequential fold       15 rows
- `selftest_seg8.log`: PASS  S9 decode row == prefill row (split-KV structure)
- `selftest_seg8.log`: LSSA-B8 SELFTEST: 28/28 checks PASS
- `gate_t16n_v2.log`: K10d 5 segments            Tp=5000   gen=4   split=128  bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=20 segs=100
- `gate_t16n_v2.log`: K10e 9 seg / 3 CTAs        Tp=9000   gen=3   split=3    bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=12 segs=108
- `gate_t16n_v2.log`: K10f 9 seg / 1 CTA         Tp=9000   gen=2   split=1    bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=4 segs=36
- `gate_t16n_v2.log`: K10g block edge            Tp=1151   gen=3   split=4    bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=8 segs=16
- `gate_t16n_v2.log`: K10h scale 4               Tp=2500   gen=3   split=8    bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=12 segs=36
- `gate_t16n_v2.log`: T16N GATE: 8 PASS / 0 FAIL
- `gate_t16n_v2b.log`: T16N GATE: 8 PASS / 0 FAIL
- `gate_t16n_v2c.log`: K10i block mode 1 blk      Tp=46     gen=8   split=8    bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=4 segs=4
- `gate_t16n_v2c.log`: K10j block mode 3 blk      Tp=300    gen=8   split=8    bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=12 segs=12
- `gate_t16n_v2c.log`: K10k block mode 8 blk      Tp=1000   gen=6   split=8    bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=32 segs=32
- `gate_t16n_v2c.log`: K10l block mode 8 blk s2   Tp=900    gen=4   split=2    bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=8 segs=8
- `gate_t16n_v2c.log`: T16N GATE: 12 PASS / 0 FAIL
- `gate_t16n_v3.log`: K10d 5 segments            Tp=5000   gen=4   split=128  bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=160 segs=320
- `gate_t16n_v3.log`: K10e 9 seg / 3 CTAs        Tp=9000   gen=3   split=3    bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=12 segs=36
- `gate_t16n_v3.log`: K10f 9 seg / 1 CTA         Tp=9000   gen=2   split=1    bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=4 segs=12
- `gate_t16n_v3.log`: K10g block edge            Tp=1151   gen=3   split=4    bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=8 segs=8
- `gate_t16n_v3.log`: K10h scale 4               Tp=2500   gen=3   split=8    bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=20 segs=20
- `gate_t16n_v3.log`: T16N GATE: 16 PASS / 0 FAIL
- `gate_t16n_v3b.log`: T16N GATE: 16 PASS / 0 FAIL
- `gate_t16n_v4.log`: T16N GATE: 16 PASS / 0 FAIL
- `gate_t16n_v5.log`: T16N GATE: 16 PASS / 0 FAIL
- `gate_t16n_final.log`: T16N GATE: 16 PASS / 0 FAIL
- `gate_t16n_v6.log`: T16N GATE: 16 PASS / 0 FAIL
- `gate_t16n_v7.log`: T16N GATE: 16 PASS / 0 FAIL
- `gate_t16n_v8.log`: K10q G=5 (40/8) 3 blk      Tp=300    gen=6   split=8    bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=8 segs=8
- `gate_t16n_v8.log`: K10r G=5 (40/8) 2 seg      Tp=5000   gen=3   split=16   bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=80 segs=160
- `gate_t16n_v8.log`: K10s G=8 (64/8) 3 blk      Tp=300    gen=6   split=8    bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=8 segs=8
- `gate_t16n_v8.log`: K10t G=8 (64/8) 2 seg      Tp=5000   gen=3   split=16   bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=80 segs=160
- `gate_t16n_v8.log`: K10u G=4 (32/8) 1 seg      Tp=1000   gen=4   split=4    bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=8 segs=8
- `gate_t16n_v8.log`: T16N GATE: 21 PASS / 0 FAIL
- `gate_t16n_72box.log`: T16N GATE: 21 PASS / 0 FAIL
- `gate_t16n_4090_v7_fused.log`: T16N GATE: 16 PASS / 0 FAIL
- `gate_t16n_4090.log`: T16N GATE: 16 PASS / 0 FAIL
- `gate_t16n_4090_v6_fused.log`: T16N GATE: 16 PASS / 0 FAIL
- `gate_k1k8_final_v1.log`: K-GATES: 43 PASS / 0 FAIL
- `v2_gate.log`: LSSA-B8 EXACT KERNEL GATE  [build b8v2eng]  FA3_DIR=/root/fa3_b8v2eng/hopper  W_LOC=9  expect=match
- `v2_gate.log`: PASS  G2 kernel == lssab8_torch (== golden, S2/S11) on the 239-case suite 239/239 cases, 4.1s
- `v2_gate.log`: T16d GATE [b8v2eng]: PASS   (8 PASS / 0 FAIL, 16.5s)   signatures -> b8x_sig_b8v2eng.json
- `v2_gate_neg1.log`: LSSA-B8 EXACT KERNEL GATE  [build b8vareng_vs_seg8]  FA3_DIR=/root/fa3_b8vareng/hopper  W_LOC=9  expect=differ
- `v2_gate_neg1.log`: T16d GATE [b8vareng_vs_seg8]: FAIL   (negative control; 1/3 checks as expected)
- `v2_gate_neg2.log`: LSSA-B8 EXACT KERNEL GATE  [build b8v2eng_vs_filed]  FA3_DIR=/root/fa3_b8v2eng/hopper  W_LOC=9  expect=differ
- `v2_gate_neg2.log`: T16d GATE [b8v2eng_vs_filed]: FAIL   (negative control; 1/3 checks as expected)
- `gate_kernel.log`: PASS  G2 kernel == lssab8_torch (== golden, S2/S11) on the 239-case suite 239/239 cases, 4.0s
- `gate_kernel.log`: PASS  G3 exact block skip ON == OFF                        0 dead of 337920 (row,block) pairs
- `gate_kernel.log`: PASS  G5 determinism over 20 launches
- `gate_kernel.log`: PASS  G7 128-aligned chunk invariance                      T1024->512 ok T2048->1280 ok T4096->3968 ok
- `gate_kernel.log`: PASS  G8 split-KV segment fold == the sequential fold (golden, CPU) 3 rows at T=2048
- `gate_kernel.log`: T16d GATE [fa3_b8vareng]: PASS   (8 PASS / 0 FAIL, 16.2s)   signatures -> b8x_sig_fa3_b8vareng.json
- `gate_k1k8.log`: K-GATES: 43 PASS / 0 FAIL
- `gate_k9.log`: == K9: decode with FRESHLY PACKED per-step k/v (L crosses a 128 block) ==
- `gate_k9.log`: K9: 10 PASS / 0 FAIL
- `gate_e2.log`: [PASS] E2 ids across mnbt {512,640,700,2048,8192} (attention-only) all token ids identical: True
- `gate_e2.log`: [FAIL] E2 cache LATTICE across mnbt, layer 0 (attention-only) layer0 mismatches=48
- `gate_e2.log`: [FAIL] E2 cache LATTICE across mnbt, layers 13+27 (attention-only) L13=48 L27=48 mismatches
- `gate_e2.log`: [PASS] E2 ids across mnbt {512,640,700,2048,8192} (FS-armed) all token ids identical: True
- `gate_e2.log`: [PASS] E2 cache LATTICE across mnbt, layer 0 (FS-armed)     layer0 mismatches=0
- `gate_e2.log`: [PASS] E2 cache LATTICE across mnbt, layers 13+27 (FS-armed) L13=0 L27=0 mismatches
- `gate_e6.log`: [FAIL] E6 ours graph(FULL_DECODE_ONLY, no-inductor) == eager token-identical: False
- `gate_e6.log`: [PASS] E6 ours graph(FULL_AND_PIECEWISE, no-inductor) == eager token-identical: True
- `gate_e6.log`: [PASS] E6 ours graph(PIECEWISE, no-inductor) == eager       token-identical: True
- `gate_e6.log`: [FAIL] E6 CONTROL stock graph(FULL_DECODE_ONLY, no-inductor) == eager token-identical: False
- `gate_e6.log`: [PASS] E6 CONTROL stock graph(PIECEWISE, no-inductor) == eager token-identical: True

### T16m: the fused verified stack (fused glue, routed int8 GEMM, contract v2.1 epilogue)

- Kernel gate (every fused op against the composition of the T16l ops it replaces, both GEMM routes, random + tie-dense data, M = 1..2048): **PASS** cases=410 bad=0
- Engine gate, fused vs unfused per-layer outputs (layers 0/13/27, every real forward) and token ids, GEMM route cut = 128 rows: **PASS** layer_calls=75 bad=0 ids_identical=True
- Engine gate, fused vs unfused per-layer outputs (layers 0/13/27, every real forward) and token ids, GEMM route cut = 1 rows: **PASS** layer_calls=75 bad=0 ids_identical=True
- Engine gate, fused vs unfused per-layer outputs (layers 0/13/27, every real forward) and token ids, GEMM route cut = 1000000000 rows: **PASS** layer_calls=75 bad=0 ids_identical=True
- GEMM routes, standalone on H100 SXM (medians, us): qkv_M1: bf16 21, int_mm 18, int_mm+deq 21, cutlass 28, cutlass_b 32; qkv_M32: bf16 20, int_mm 18, int_mm+deq 21, cutlass 25, cutlass_b 33; qkv_M256: bf16 26, int_mm 32, int_mm+deq 37, cutlass 30, cutlass_b 38; qkv_M8192: bf16 354, int_mm 375, int_mm+deq 503, cutlass 213, cutlass_b 221; o_M1: bf16 19, int_mm 18, int_mm+deq 21, cutlass 25; o_M32: bf16 19, int_mm 18, int_mm+deq 20, cutlass 24; o_M256: bf16 25, int_mm 21, int_mm+deq 25, cutlass 29; o_M8192: bf16 279, int_mm 294, int_mm+deq 397, cutlass 170; gate_up_M1: bf16 103, int_mm 62, int_mm+deq 66, cutlass 68; gate_up_M32: bf16 110, int_mm 63, int_mm+deq 71, cutlass 70; gate_up_M256: bf16 121, int_mm 111, int_mm+deq 147, cutlass 79; gate_up_M8192: bf16 2849, int_mm 3845, int_mm+deq 4107, cutlass 1708; down_M1: bf16 61, int_mm 54, int_mm+deq 58, int_mm_splitk4 74, cutlass 45; down_M32: bf16 62, int_mm 51, int_mm+deq 55, int_mm_splitk4 70, cutlass 46; down_M256: bf16 75, int_mm 94, int_mm+deq 99, int_mm_splitk4 112, cutlass 65; down_M8192: bf16 1427, int_mm 1252, int_mm+deq 1441, int_mm_splitk4 1732, cutlass 871
- Epilogue order gate (CUTLASS fused epilogue vs the candidate declared orders, mismatches over every element): qkv: {'n': 37748736, 'O1_mismatch': 189, 'O2_mismatch': 0, 'bias_bf16_O2+b': 1370814, 'bias_bf16_O1+b': 1370817, 'bias_bf16_O2+bbf': 86, 'bias_bf16_bf16(O2)+b': 10114097}; o: {'n': 29360128, 'O1_mismatch': 138, 'O2_mismatch': 0}; gate_up: {'n': 310378496, 'O1_mismatch': 1473, 'O2_mismatch': 0}; down: {'n': 29360128, 'O1_mismatch': 145, 'O2_mismatch': 0}
- GPU time, same node (H100 SXM, torch profiler, CUDA kernels only), the fused stack against the attention-only build: decode B1 (300-token context, 64 tokens): fused 5.661 ms/token vs attention-only 6.248 (1.104x); decode B32 (300-token context, 64 tokens): fused 9.105 ms/token vs attention-only 9.741 (1.070x); decode B64 (300-token context, 64 tokens): fused 12.961 ms/token vs attention-only 13.540 (1.045x); prefill of 512 tokens (eager): fused 12.18 ms vs attention-only 12.63 (1.037x); prefill of 8192 tokens (eager): fused 167.20 ms vs attention-only 195.20 (1.167x); decode-only at 30k context by slope (GEN 144 minus GEN 16, B1): fused 6.890 ms/token GPU; decode-only at 30k context by slope (GEN 144 minus GEN 16, B1): attention-only 7.536 ms/token GPU; same-harness wall-clock decode slope at 32k on the 4x H100 node (1M model, fresh compile cache): fused FULL_AND_PIECEWISE 129.44 tok/s, fused PIECEWISE 123.73 tok/s, attention-only FULL_AND_PIECEWISE 127.63 tok/s, attention-only PIECEWISE 125.77 tok/s; CPU-side attribution at 30k (1M model, same node): fused wall 7.670 ms/token, GPU 7.946, CPU self 7.534 (mostly waiting on the GPU); CPU-side attribution at 30k (1M model, same node): attention-only wall 7.958 ms/token, GPU 8.077, CPU self 5.291 (mostly waiting on the GPU). Wall-clock ratios below are lower on steps CUDA graphs do not capture (prefill chunks, mixed steps): the python-registered custom ops cost ~10 us of CPU per call more than an aten op (measured 13.8 vs 4.9 us for one quantiser), 6 calls per layer.

### Needle retrieval at 512k (1M model, depths 5 / 50 / 95%)

- `needle_stock_p524288` (stock): 4 / 6 needles found; per cell: L=524288 keys=1 d=0.05 1/1, L=524288 keys=1 d=0.5 1/1, L=524288 keys=1 d=0.95 1/1, L=524288 keys=4 d=0.05 0/1, L=524288 keys=4 d=0.5 0/1, L=524288 keys=4 d=0.95 1/1
- `needle_lssab_p524288_attn` (lssab): 4 / 6 needles found; per cell: L=524288 keys=1 d=0.05 1/1, L=524288 keys=1 d=0.5 1/1, L=524288 keys=1 d=0.95 1/1, L=524288 keys=4 d=0.05 0/1, L=524288 keys=4 d=0.5 0/1, L=524288 keys=4 d=0.95 1/1
- `needle_lssab_p524288_t16m23f` (lssab): 4 / 6 needles found; per cell: L=524288 keys=1 d=0.05 1/1, L=524288 keys=1 d=0.5 1/1, L=524288 keys=1 d=0.95 1/1, L=524288 keys=4 d=0.05 0/1, L=524288 keys=4 d=0.5 0/1, L=524288 keys=4 d=0.95 1/1

### T16o: lsb_dec8 with the scores and PV on the int8 tensor cores (mma.sync m16n8k32, exact int32)

- rig gate, all K10 cases: T16N GATE: 21 PASS / 0 FAIL
- rig gate, head-group variants: T16N GATE: 5 PASS / 0 FAIL
- engine, MMA kernel: E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu): **PASS** rows=151200 bad=0 maxdiff=0.000e+00 (18 sequence-steps)
- engine, MMA kernel: E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target]: **PASS** rows=29736 bad=0 maxdiff=0.000e+00 (1062 sequence-steps)
- E6 on the whole verifiable stack at TP=4 (7B, per-rank R4, pinned reduction): full == eager ids **PASS**, self-consistent True / True
- engine, MMA kernel, E6 full == eager ids: **PASS**
- decode at 32k (1M model, 4x H100 node, one GPU, attention-only, same session): dp4a kernel 129.01 tok/s, MMA kernel 126.03 tok/s (0.977x)
- decode at 128k (1M model, 4x H100 node, one GPU, attention-only, same session): dp4a kernel 93.38 tok/s, MMA kernel 90.59 tok/s (0.970x)
- decode at 512k (1M model, 4x H100 node, one GPU, attention-only, same session): dp4a kernel 50.14 tok/s, MMA kernel 47.64 tok/s (0.950x)

### T16p: the row-fold staging fix (1M-token decode)

- rig gate, all K10 cases, T16q pipeline kernel (not shipped: correct, not faster): T16N GATE: 21 PASS / 0 FAIL
- rig, head-group variants, T16q: T16N GATE: 5 PASS / 0 FAIL
- rig, split-64 vs split-1 at 262k / 512k / 1M, T16q: K10z T=262144 split64      Tp=262144 gen=2   split=64   bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=4 segs=260
- rig, split-64 vs split-1 at 262k / 512k / 1M, T16q: K10z T=524288 split64      Tp=524288 gen=2   split=64   bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=4 segs=516
- rig, split-64 vs split-1 at 262k / 512k / 1M, T16q: K10z T=1000000 split64     Tp=1000000 gen=2   split=64   bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=4 segs=980
- rig, split-64 vs split-1 at 262k / 512k / 1M, T16q: T16N GATE: 3 PASS / 0 FAIL
- rig gate, all K10 cases, chunked-fold kernel (shipped): T16N GATE: 21 PASS / 0 FAIL
- rig, split-64 vs split-1 at 262k / 512k / 1M, chunked-fold kernel (shipped): K10z T=262144 split64      Tp=262144 gen=2   split=64   bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=4 segs=260
- rig, split-64 vs split-1 at 262k / 512k / 1M, chunked-fold kernel (shipped): K10z T=524288 split64      Tp=524288 gen=2   split=64   bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=4 segs=516
- rig, split-64 vs split-1 at 262k / 512k / 1M, chunked-fold kernel (shipped): K10z T=1000000 split64     Tp=1000000 gen=2   split=64   bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=4 segs=980
- rig, split-64 vs split-1 at 262k / 512k / 1M, chunked-fold kernel (shipped): T16N GATE: 3 PASS / 0 FAIL
- rig gate, all K10 cases, first fix (unstaged fold): T16N GATE: 21 PASS / 0 FAIL
- rig, split-64 vs split-1 at 262k / 512k / 1M, first fix: K10z T=262144 split64      Tp=262144 gen=2   split=64   bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=4 segs=260
- rig, split-64 vs split-1 at 262k / 512k / 1M, first fix: K10z T=524288 split64      Tp=524288 gen=2   split=64   bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=4 segs=516
- rig, split-64 vs split-1 at 262k / 512k / 1M, first fix: K10z T=1000000 split64     Tp=1000000 gen=2   split=64   bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=4 segs=980
- rig, split-64 vs split-1 at 262k / 512k / 1M, first fix: T16N GATE: 3 PASS / 0 FAIL
- rig, split-64 vs split-1 at 262k / 512k (1M crashed), before the fix: K10z T=262144 split64      Tp=262144 gen=2   split=64   bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=4 segs=260
- rig, split-64 vs split-1 at 262k / 512k (1M crashed), before the fix: K10z T=524288 split64      Tp=524288 gen=2   split=64   bits=EQ  maxdiff=0.000e+00 bad_steps=0 ctas=4 segs=516
- rig, split-64 vs split-1 at 262k / 512k (1M crashed), before the fix: res.append(run_decode8("K10z T=%d split64" % Tp, Tp, 2, 64, seed=31))
- engine, fixed kernel: E1(B8, mu ON) PROMPT rows vs lssab8_torch(mu): **PASS** rows=151200 bad=0 maxdiff=0.000e+00 (18 sequence-steps)
- engine, fixed kernel: E4v2(mu ON) GENERATED rows vs lssab8_torch(mu) [contract v2 target]: **PASS** rows=29736 bad=0 maxdiff=0.000e+00 (1062 sequence-steps)
- engine, fixed kernel, E6 full == eager ids: **PASS**
- 1M tokens on ONE H100 of the 4x H100 node (TP=1; file tags say tp4 for historical reasons), attention-only, chunked-fold kernel, split cap scaled (the shipped configuration): decode 33.45 tok/s (0.859x of stock's 38.94)
- 1M tokens on ONE H100 of the 4x H100 node (TP=1), attention-only, chunked-fold kernel, split cap scaled (the shipped configuration): TTFT 447.7 s (0.804x of stock's 359.8 s)
- 1M tokens on ONE H100 of the 4x H100 node (TP=1; file tags say tp4 for historical reasons), WHOLE VERIFIABLE STACK, chunked-fold kernel, split cap scaled, bf16 weights freed (the shipped configuration): decode 33.50 tok/s (0.860x of stock's 38.94)
- 1M tokens on ONE H100 of the 4x H100 node (TP=1), WHOLE VERIFIABLE STACK, chunked-fold kernel, split cap scaled, bf16 weights freed (the shipped configuration): TTFT 445.3 s (0.808x of stock's 359.8 s)
- 1M tokens on ONE H100 of the 4x H100 node (TP=1; file tags say tp4 for historical reasons), attention-only, first fix (unstaged fold), split cap scaled: decode 17.20 tok/s (0.442x of stock's 38.94)
- 1M tokens on ONE H100 of the 4x H100 node (TP=1), attention-only, first fix (unstaged fold), split cap scaled: TTFT 447.3 s (0.804x of stock's 359.8 s)
- 1M tokens on ONE H100 of the 4x H100 node (TP=1; file tags say tp4 for historical reasons), WHOLE VERIFIABLE STACK, first fix (unstaged fold), split cap scaled, bf16 weights freed: decode 17.44 tok/s (0.448x of stock's 38.94)
- 1M tokens on ONE H100 of the 4x H100 node (TP=1), WHOLE VERIFIABLE STACK, first fix (unstaged fold), split cap scaled, bf16 weights freed: TTFT 445.2 s (0.808x of stock's 359.8 s)
- 1M tokens on ONE H100 of the 4x H100 node (TP=1; file tags say tp4 for historical reasons), attention-only, fixed kernel, split cap 64 (before the cap scaling; at KVH=4 on one GPU the scaled cap is also 64, so the difference to the next row is run-to-run variance): decode 16.46 tok/s (0.423x of stock's 38.94)
- 1M tokens on ONE H100 of the 4x H100 node (TP=1), attention-only, fixed kernel, split cap 64 (before the cap scaling; at KVH=4 on one GPU the scaled cap is also 64, so the difference to the next row is run-to-run variance): TTFT 447.7 s (0.804x of stock's 359.8 s)
- 1M tokens on ONE H100 of the 4x H100 node (TP=1; file tags say tp4 for historical reasons), WHOLE VERIFIABLE STACK, fixed kernel, bf16 weights freed, split cap 64: decode 17.31 tok/s (0.444x of stock's 38.94)
- 1M tokens on ONE H100 of the 4x H100 node (TP=1), WHOLE VERIFIABLE STACK, fixed kernel, bf16 weights freed, split cap 64: TTFT 445.3 s (0.808x of stock's 359.8 s)

### Full verifiable stack, eager, contract v2.1 declared ops (unfused reference of T16m)

- A per-op bit identity on real activations: **True**; linear_down_proj 60 calls / 0 bad, linear_gate_up_proj 60 calls / 0 bad, linear_o_proj 60 calls / 0 bad, linear_qkv_proj 60 calls / 0 bad, rmsnorm 60 calls / 0 bad, rmsnorm_residual 50 calls / 0 bad, rope 60 calls / 0 bad, silu 60 calls / 0 bad
- B F3 token identity: {'self_consistent': True, 'alone_eq_batched': True, 'perm_eq_batched': True, 'ids_sha': '478014e0131c0ca316b3cf06f473022eb116e9d9650bcc610fc3c4d64367b746'}
- C commit replay (CPU torch leaf + PyPI blake3): {'rows': 48, 'leaf_bad': 0, 'hash_bad': 0, 'detail': []}

### Full verifiable stack, eager (built, not the timed configuration)

- A per-op bit identity on real activations: **True**; linear_down_proj 60 calls / 0 bad, linear_gate_up_proj 60 calls / 0 bad, linear_o_proj 60 calls / 0 bad, linear_qkv_proj 60 calls / 0 bad, rmsnorm 60 calls / 0 bad, rmsnorm_residual 50 calls / 0 bad, rope 60 calls / 0 bad, silu 60 calls / 0 bad
- B F3 token identity: {'self_consistent': True, 'alone_eq_batched': True, 'perm_eq_batched': True, 'ids_sha': 'da3ea34e4f4b0806cb1edfc611142c74ea5e7009bd83c5c72ff1f92de4a5d942'}
- C commit replay (CPU torch leaf + PyPI blake3): {'rows': 48, 'leaf_bad': 0, 'hash_bad': 0, 'detail': []}
<!-- END TABLES -->

## 5. Reading the numbers

<!-- BEGIN READING -->
The tables above are the evidence; this is the reading. Every ratio is Lockstep against the
same vLLM build with the plugin inert, in the same engine mode, on the same GPU in the same
session, and every Lockstep build that was timed had first passed the gates of section 3
bit-identically. The shipped configuration is contract v2 (section 7.5): prompt rows and
generated rows on the LSSA-B8 rule with the filed constants, the filed prompt kernel, the
`lsb_dec8` decode kernel with the KV-quantise step fused in (v6; v5 first-phase layout), split
cap 64, full CUDA graphs, inductor off.

**Where it stands against a 15% bar and a 10% bar (the FINAL rows of section 4).**
Single-stream decode 0.962 / 0.970 / 0.934x at batch 1 / 8 / 32. Prefill 0.925 / 0.967 /
0.934x at 2k / 8k / 32k tokens, 0.988x at 2k with 8 streams. Long-context prefill on the 1M
model 0.94 / 0.90 / 0.87 / 0.84x at 32k / 65k / 131k / 262k with the filed prompt kernel
(0.96 / 0.94 / 0.92 / 0.90x with the LITE candidate). Long-context decode on the 128-token
slope: 0.856x at 32k, 0.839x at 65k, 0.865x at 131k, 0.879x at 262k.
The four-stream chat example decodes at 0.957x with the needle found; serving 0.889 /
0.939 / 0.932x. Within 15%: every cell except the 128-token single-stream prompt (a 2 ms fixed host
cost, 0.85x) and long-context prefill at 262k with the filed prompt kernel (0.84x; 0.90x with
the LITE candidate). Within 10%: decode at every
batch size, prefill from 2k up, all three serving workloads, and the chat example; not yet
long-context decode (0.84 to 0.88x).

**What moved it in this session.** At the start the fair decode numbers were 0.76 / 0.74 /
0.69x and long-context decode 0.75 / 0.67x, with the filed figures above 1.0x measured on
graphs that ran no Lockstep kernel (7.1). Contract v1's decode kernel went through eleven
gated iterations (7.2); contract v2 then replaced its rule for generated rows with the
prompt-row rule and rebuilt the decode kernel around exact block partials and staged folds
(7.5). None of it changed a bit of any output relative to its own reference: the E4v2 gate
holds on 33,432 generated rows over 1,194 steps and the E6 gates on every graph shape.

**What the remaining gap is made of.** At batch 32 and in the ShareGPT serving workloads
the decode kernel is 24 us per layer per token and the KV-quantise step 21 us, against
stock's whole attention path of 24 us; the quantise step does one row per sequence and its
cost is latency, which the bare-metal profiler run is meant to explain. At long context the
kernel main loop is within reach of stock per block, and the difference is the fixed cost
per CTA and the folds; at 32k every launch configuration lands at the same number, which
locates the limiter outside the instrumented loop. Neither is a contract question.

**Quality and verification, unchanged.** The B8 rule's quality is the measured +0.052%
(base, 300 windows) and +0.087% on the engine path, now for every row; the generated rows
moved from a rule measured at +0.102% to this one. The RTX 4090 verifier reproduces the H100
engine's rows bit for bit on its GPU and CPU, and the contract-v2 decode kernel itself passes
its 16-case gate compiled for Ada, so it proves on both architectures. The linear layers in the timed configuration
are stock bf16; the exact non-attention stack of T16l is built and gated, and its timing as
one build with the attention path is the END-TO-END VERIFIED STACK rows read below.

**Model sizes and the whole verifiable stack (the 14B, 72B and END-TO-END rows of section 4).**
The same v8 kernel, with head groups of 5 and 8 instead of 7, runs Qwen2.5-14B-Instruct
(40 heads / 8 kv-heads) and Qwen2.5-72B-Instruct (64 / 8, tensor-parallel 4 on four H100
SXM) with per-model key offsets computed by the same tool, and the gates hold on both: E1 on
792,000 prompt rows and E4v2 on 56,640 generated rows bit-identical on the 14B, E6 full ==
eager on both models, the 72B at TP=4. Fair 14B numbers: decode 0.971 / 0.973 / 0.950x at
batch 1 / 8 / 32, prefill 0.940 / 0.950 / 0.917x at 2k / 8k / 32k, the chat example 0.948x
TTFT and 0.959x decode, serving 0.871 / 0.918 / 0.950x. 72B at TP=4: decode 0.959 / 0.953 / 0.945x at
batch 1 / 8 / 32, prefill 0.967 / 0.985 / 0.956x at 2k / 8k / 32k (0.817x on the 128-token
prompt, the same fixed host cost), the chat example 0.970x TTFT and 0.925x decode, serving
0.910 / 0.922 / 0.939x; the 72B single-stream arms ran one stack per process (two 72B
engines do not fit on the node) and were merged. The 128-token single-stream prompt is the
one cell outside 15% on every model, and it is a fixed cost, not a kernel cost.
The whole verifiable stack (exact int8 linears, integer norms, frozen RoPE, table SiLU and
the contract-v2 attention), in its FUSED form with natively registered ops (T16m, section
7.9; `T16L_ARM=1 T16M_FUSE=1 T16M_CPP=1 VLLM_DISABLE_COMPILE_CACHE=1`, the shipped
configuration of the full stack), measured one engine per process against stock (the
`t16m24` rows): decode 1.002 / 0.997 / 0.905x at batch 1 / 8 / 32, prefill 1.25 / 1.22 /
1.06x at 2k / 8k / 32k tokens and 1.19x on the 128-token prompt, the chat example 1.17x
TTFT and 0.94x decode, serving 0.85 / 0.89 / 0.98x (0.83 / 0.88 / 0.98 before 7.12),
long-context decode 0.89x at 32k (TTFT 1.09x) and 0.94x at 131k (TTFT 0.96x) with the
two-group pipelined decode kernel of 7.14 (0.87 / 0.81x with the sixteen-warp kernel of
7.12, 0.86 / 0.80x with the chunked-fold one) and 0.89x at 512k (TTFT 0.91x; 0.84x with
the sixteen-warp kernel, 0.78x with the chunked-fold one) and 1.02x at 1M tokens on one
H100 (TTFT 0.81x; 0.86x decode before), with needle
retrieval at 512k identical to stock. Set against the attention-only build in the same
sessions (0.851 / 0.798 / 0.751x decode at 32k / 131k / 512k; serving 0.885 / 0.914 /
0.936x) the exact non-attention stack costs nothing anywhere and gains where prefill
dominates: where the full stack trails stock, the attention-only build trails by the same
amount, so the remaining gap is the attention decode kernel and the seam's per-step cost
at batch 32 and on the decode-heavy serving workloads (7.8). Where the exact stack is
faster than stock the reason is arithmetic: int8 weights are half the bytes bf16 weights
are, and the Hopper CUTLASS int8 GEMM runs the prefill-sized products at 1.4 to 1.7x the
speed of bf16 cuBLAS under the v2.1 epilogue. Two earlier readings of these rows in this
document were wrong and are kept as evidence: the two-engine single-stream harness never
armed the exact stack (section 7.8), and the handover pod served the 1M model's fused runs
a stale compiled graph from an earlier build (7.9), which is why the pre-`t16m24` fused
long-context rows read 0.76 / 0.74x. Quality of the whole verifiable stack: +0.26%
perplexity over stock on the 300-window battery (attention alone +0.087%). The whole
verifiable stack also runs under tensor parallelism now (section 2.6, the TP rules: per-rank
R4 padding and the pinned rank-order reduction): the 7B at TP=4 passes E6 at +0.31%
perplexity, and the 72B at TP=4 on four H100s decodes at 0.982 / 0.987 / 0.961x of stock with the
native reduction and the NCCL environment of 7.13 (0.959 / 0.961 / 0.943 with the
sixteen-warp kernel of 7.12 alone, 0.952 / 0.948 / 0.922 before it),
the attention-only level, with TTFT 0.85 to 0.87x against attention-only's 0.96 to 0.99x
and serving 0.78 / 0.78 / 0.77x (0.76 / 0.75 / 0.76 before 7.12; 0.70 / 0.79 / 0.85 with
the 512-row cut of 7.13) against attention-only's 0.91 / 0.92 / 0.94x, with TTFT at parity
from 8,192 prompt tokens (0.85x before 7.13); the prefill
and serving gaps are the reduction's doubled traffic (7.8), not the arithmetic. What this
document cannot claim: sub-10% at batch 32 and on the decode-heavy serving workloads
(0.83 / 0.88x), or at long context (0.78 to 0.86x), all of which are the attention decode
path; parity on 72B prefill under tensor parallelism until the reduce-scatter form of the
reduction exists; and parity at 1M, where the decode crash is fixed (7.8): with the shipped
kernel the attention-only stack decodes one million tokens on four H100s at 33.5 tok/s
against stock's 38.9 (0.86x) with TTFT 448 s against 360 s (0.80x), and the whole
verifiable stack reads the same there: 33.5 tok/s (0.86x) and TTFT 445 s (0.81x). So the
long-context reading is now one number at every length from 32k to 1M: the whole verifiable
stack costs what the attention decode kernel costs, 14 to 22 percent, and nothing more.

**How to compare.** Mode to mode only. Inductor on is vLLM's default and is timed for both
stacks; it is not a gate for either (7.1). Compilation mode 0 is not a graph mode. The
8-token-slope long-context decode cells are noise and are labelled as such; the 128-token
slope is the measurement.
<!-- END READING -->

## 6. Filed context this session did not re-measure

| Quantity | Value | Source |
|---|---|---|
| Kernel family floor on Hopper for the filed prompt-row rule | 1.57x of stock FA3 (register file, not arithmetic) | `demo/T16D_LOG.md` section 11.5 |
| Contract quality vs exact attention, offline torch reference | +0.052% base model 300 windows [+0.018, +0.093]; worst cell +0.144% Instruct 120 windows; flat to 131k | `paper_softmax/bigrun/T16B_LOG.md` |
| Full committed stack quality (two-plane int8 linears) | +0.2335% [+0.173, +0.294] | `demo/T16H_LOG.md` section 5.2 |
| Rotated one-GEMM stack quality | +0.209% at 300 windows, +0.250% at 580 | `demo/T16K_LOG.md` section 3 |
| Cross-device bit-exactness of the base contract | 90 of 90 cases on A100, H100, x86, ARM | `RESEARCH_LOG.md` steps 6 and 7 |

## 7. What this session found, fixed, and left open

### 7.1 The full-graph defect, its cause, and the mode that can be gated

Before this session, every filed graph-mode number (T14, T16e, T16f, T16j, including the
decode figure of 1.08 to 1.10x and the serving figures near 1.0x) was taken on captured
graphs that ran none of the Lockstep decode kernels. Two causes, both in the engine seam:
every kernel launch went to the legacy stream 0, which a CUDA-graph capture on vLLM's
side stream never records; and the seam's `seqskip` mask, set on the eager prefill step
for the prompt sequence, was never cleared by a replay, so the captured row map emitted an
empty tile list (`wcnt = [1, 0]`) and pass B computed nothing. The model read a stale
attention buffer and produced garbage from the second token. Nobody caught it because
the graph-equals-eager gate (E6) always ran with compilation mode 0, and vLLM 0.25.1
silently disables graphs in that mode, so E6 compared eager with eager, and the serving
harness counts completions without inspecting tokens.

Fix (`demo/vllm_lockstep_backend.py`, `demo/lssab_fa3.py`): launches on the current
stream, the mask reset recorded inside the graph, capture-frozen fields sized at the
engine ceiling, and the per-step decisions cached on layer 0. Gate E6n now proves
capture is bit-inert: with inductor off, Lockstep's tokens are identical across eager,
piecewise and full graphs at five batch shapes, self-consistent, and identical before and
after every later change (E8 series). Stock's own full-graph tokens differ from its eager
tokens in the same mode.

Inductor is not a gate for anyone: with it on, stock's tokens differ between full and
piecewise graphs and across processes (E6i control), because compiled non-attention
kernels are selected per compilation. It remains a legitimate speed setting for both
stacks and is reported as such.

### 7.2 Decode: what it really cost, and what was done

With the kernels actually in the graph, decode read 0.76 / 0.74 / 0.69x of stock at batch
1 / 8 / 32. A per-kernel profile of a captured decode step (300-token context) put the
whole gap in two kernels: `lsb_prep` at 42.8 us per layer per token, which re-read the
open block twice from the shadow for every token, and the prefill-shaped pass B at 27 to
32 us, a 64-row wgmma tile with 190 KB of shared memory running seven real rows, flat in
context length from 100 to 4,000 tokens. Stock's entire attention was 12.3 us.

Iterations, each gated bit-identical (K1 to K9 in both launch shapes, E1/E4 on 33,432
generated rows over 1,194 decode steps, E8 token identity against the pre-change ids):

| Change | What it is | Effect |
|---|---|---|
| m1 incremental prep | a running block amax per shadow slot (a max of maxes is the same max), and quantise only the new row when the block exponents are unchanged (the old rows are the same function of the same inputs); counters show 96% of steps take the short path | prep 42.8 to 8.3 us; batch-1 decode 0.76 to 0.84x |
| m2 pass B early exit | empty split CTAs leave before loading the table | none: pass B's cost is per-CTA latency, not the idle CTAs |
| m3 folded memsets, fused epilogue | zero-fills done inside prep, the split epilogue run by the last-arriving CTA (order-free integer partials) | neutral: eight graph nodes became four, the time moved |
| m4, m5 decode-shaped pass B | `lsb_dec`: one CTA per (sequence, kv-head, split), one warp per head, K and V staged per chunk, the same chain helpers, the same byte planes, the same integer partials into the same buffers; selected only for single-row tiles | pass B 32 to 19 us at batch 1 and 63 to 27 us at batch 32; decode 0.92 / 0.91 / 0.87x |
| m6 prefetched block params | | none, reverted |
| m7, m8 | a one-warp prep path for the single-new-row case; 64-key split units so a short-context CTA does one round | decode 0.938 / 0.936 / 0.858x; `lsb_dec` 14.6 us at batch 1 but 32.5 us at batch 32 (twice the working CTAs, each staging the 16 KB FRAC table) |
| m9 double-buffered chunks | K and V chunks issued with `cp.async` one chunk ahead; split cap 32 to 128 | `lsb_dec` 13.0 / 30.3 us; decode 0.952 / 0.945 / 0.862x; the 131k decode cell at cap 128 read 97 tok/s on the 8-token slope (see 7.4 on that cell's noise) |
| m10 table through L1 | the FRAC table read through the cache instead of staged | `lsb_dec` 12.2 / 28.6 us at 300 tokens, but 12% slower at 131k where a CTA walks 16 chunks |
| m11 hybrid staging | staged only when a CTA owns four or more chunks (same table, same indices) | keeps m10 at short context and m9 at long; the shipped kernel: decode 0.952 / 0.946 / 0.867x, TTFT 0.912 / 0.959 / 0.931x at 2k / 8k / 32k |

Where the batch-32 gap is now (profile of a captured decode step, per layer per token):
stock's attention path is 24.3 us (FA3 decode 18.9, KV write 3.2, split combine 2.3); the
Lockstep path is 60.6 us, of which `lsb_dec` is 28.6, `lsb_prep` 21.5 and `lsb_pass_a` 10.6.
The kernel itself is 7.5 us of the 36 us gap; the two helper kernels that stock does not
have are 28.6 us of it. Prep's work at batch 32 is one new row per sequence, so its cost is
its chain of dependent memory hops and the step-level zero-fill of the split accumulators,
not arithmetic; performance counters are locked in the RunPod container
(`ERR_NVGPUCTRPERM`), so the split of that 21.5 us is not measured. The two changes that
close it, in order: have the last-arriving split CTA re-zero the accumulators it consumed so
prep does no zero-fill (the invariant must hold across mixed steps and the prompt-row seam
too), and fold pass A's row maxima into `lsb_dec` as a first phase over the same chunks.

### 7.3 LSSA-B8-LITE, the candidate prompt-row variant for long context

Two of the three T16d-priced rules are changed behind environment switches in the golden
and torch references (`LSSAB8_SEG=0`, `LSSAB8_RESCALE=mul`): a single-level fold in the
committed ascending order, and a one-multiply rescale `fl32(x * 2^-D)` with gradual
underflow (the kernel is built with `-ftz=false`). The kernel build `b8liteeng` passes
G2 (239 of 239 cases against the LITE torch reference), G3, G5, G7, G8, and the in-engine
E1 gate on 336,000 prompt rows against the LITE reference; the negative controls confirm
the gate can see the change (33 of 33 cases differ from the filed golden). Below one
4,096-key segment the two functions coincide, so its 1k-window perplexity equals the
filed variant's (+0.087%). Standalone it runs 1.74 / 1.43 / 1.31 / 1.29 / 1.29 / 1.29x of
stock FA3 at 2k to 262k (the filed kernel 1.98 / 1.58 / 1.43 / 1.41 / 1.40 / 1.40x). In the
engine on the 1M-context model it lifts TTFT from 0.938 / 0.903 / 0.871 / 0.839x (filed) to
0.960 / 0.944 / 0.924 / 0.902x at 32k / 65k / 131k / 262k. It is a candidate: adopting it
means re-running the 300-window quality battery and re-pinning the contract constants.

### 7.5 Contract v2: one rule for every row (T16n, built and gated in this session)

The cycle accounting of 7.2 and 7.4 says the generated-row rule (LSSA-B) is the cost:
32-bit probabilities need four byte planes per V multiply, a global row-max pass before
the main pass, and 32-bit accumulators that are zero-filled and atomically summed. The
prompt-row rule (LSSA-B8) has none of that: 8-bit weights with a per-block exponent,
exact s32 block partials, and a two-level fp32 fold in a committed order whose segment
level exists precisely so that split-KV decode equals prefill (golden gate S7/S9). Two
timing-only variants of the v1 kernel priced the change before it was built: one byte
plane took the 131k decode cell from 0.755x to 0.836x and removing the per-chunk block
parameter loads took it to 0.898x.

Contract v2 puts generated rows on LSSA-B8 and changes one constant, the fold segment,
from 32 blocks to 8 (1,024 keys), so that a 32k context already yields 128 decode CTAs.
The reference is `lssab8_row_segments` in the golden with `LSSAB8_SEG=8`; the self-test
passes 28 of 28 there, including S7 and S9. The kernel is `lsb_dec8`
(`lssa/portable/lssa_lssab_t13b.cu`): one CTA per (sequence, kv-head, segment), one warp
per head, whole 128-key blocks staged by `cp.async` with double buffering, the exact
s32 block max, the B8 chain with the T2 table (digest 7a42a785) in shared memory, one
u8 x s8 dp4a plane into an exact block partial, the one-pass block fold, one segment
partial per CTA written without atomics, and the row-level fold run by the last-arriving
CTA. Host switch: `LOCKSTEP_V2=1` (backend flag bit 32; `LSSAB8_SEG=8` for every
reference). The prompt kernel is rebuilt with `B8X_SEG=8` (`b8v2eng`), which
`patch_b8x.py` now takes from the environment.

**What the long-context cells then showed (measured, 128-token slope).** With the segment at
8 blocks the v2 decode reads 56.9 tok/s at 131k and 32.1 at 262k against v1's 89.4 and 63.4,
while the cycle accounting of the v2 main loop is *faster* than v1's over the same keys (100k
cycles per 8 blocks against 124k). The time is in the part the accounting does not cover: the
last-arriving CTA folds the row's segments one after another from L2, and a 131k row has 128
of them at SEG = 8. The SEG = 8 prompt kernel also runs 8 to 30% slower at 32k to 262k (its
segment close runs four times as often). Both point the same way and the design converges on:
keep the filed segment of 32 blocks (no prompt-kernel rebuild, no constant re-pin; the
contract change is the single sentence "generated rows use LSSA-B8"), obtain the decode
parallelism inside a segment from the exact integer block partials (sub-segment units below
32k, a wide CTA per segment above), and stage every fold's inputs in shared memory so the
tail is microseconds. The B8 decode arithmetic, its gates, the engine seam and the batch-32
and serving gains carry over unchanged.

**The restructure (v3 to v5) and what each measured, 128-token slope, 1M model, fair mode.**
v3 (filed segment, units, staged folds): 32k 0.851x, 131k 0.770x, 262k 0.812x; short context
fell to 0.934 / 0.927 / 0.878x because a single-segment row paid two global round trips. v3b
(blocks fold straight into the row for single-segment rows, several heads per staging round):
0.967 / 0.960 / 0.903x, serving 0.874 / 0.899 / 0.938x, e2e 0.912x. v4 (q as 16-byte loads;
V transposed once per block, cooperatively, into the dead K buffer with a bank-free swizzle, so
phase 2 is one load and one dp4a per four keys): phase 2 fell from 4.7k to 1.4k cycles per
block, 262k 0.877x. A sizing slip had capped the block-partial scratch below a 131k context,
so every 131k cell before the fix ran whole-segment CTAs; with units engaged at 131k they
measured *slower* (87.4 against 102.6 tok/s), and the split cap matters there because a
launched CTA that finds no unit still reserves its shared memory before exiting (cap 128:
91.5; cap 64: 102.6; cap 32: 103.2 tok/s). v5 (K read once per block, lanes owning keys)
was a wash against v4. The shipped configuration is therefore the v4 kernel, split cap 64,
sub-segment units only up to 64k keys, whole segments above; its battery is the FINAL rows of
section 4. At 32k every configuration lands at 0.85x, so the 32k limiter is a fixed
per-layer cost outside the instrumented main loop (the folds are the suspect), which the
bare-metal profiler run is meant to settle.

**Cross-architecture.** The same `lsb_dec8` source compiled for Ada (`sm_89`, RTX 4090 bare
metal, CUDA 12.4) passes the same 16-case gate bit for bit (`results/gate_t16n_4090.log`), so
under contract v2 the decode kernel proves on Hopper and on Ada; the prompt kernel remains the
FA3 (Hopper) build, with the 4090 as verifier for prompt rows.

**The profiler-driven pair (v6, v7).** v6 folds the KV-quantise step into the decode CTA
(its base is the v5 first-phase layout, lanes owning keys with heads as the inner loop, which
had measured level with v4; the numbers below are v6 as measured)
that owns a sequence's last block and has every unit CTA quantise its own seven q rows (a pure
function of the row, so no CTA waits for another); on a decode-only step the prep launch
disappears. It is bit-identical on both machines (16 of 16; in the engine 504,000 prompt and
33,432 generated rows; graph identity on four shapes, tokens equal to the unfused build). The
first attempt set the fusion flag from the rows the portable path owns, which on a chunked
prefill excludes the seam's prompt chunk, so the seam's lattice was never quantised (E1
caught it: 336,000 bad rows); the flag now requires a step with no prompt rows at all. Fair
mode: decode 0.962 / 0.970 / 0.934x at batch 1 / 8 / 32 (from 0.906 at 32), serving 0.889 /
0.939 / 0.932x, the chat example 0.957x. Its own long-context cells read 3 to 5% below the
unfused kernel (131k 97.7 against 102.7 tok/s, 262k 79.8 against 82.6): the fused quantise is a
serial prologue on one unit CTA, and a whole-segment CTA at that length pays it as a straggler.
The fusion is therefore a per-engine policy, on for engines whose context ceiling is at most
64k keys (`LOCKSTEP_V2_FUSE_MAXBLK`, 512 blocks) and off above, so a long-context engine keeps
the unfused kernel's numbers. **v6 with that policy is the shipped build.** v7 (half-block staging
so the footprint fits four CTAs per SM) is also bit-identical everywhere but slower at every
length on the H100 (batch 32 0.915x, 32k 123.5 and 131k 89.5 tok/s against 130.3 and 102.7):
the barriers per half-block and the second transpose cost more than the occupancy returns, so
on Hopper the long-context limiter is not warp count. It is kept as evidence, not shipped.

Gates passed so far: K10a to K10h, the kernel against `lssab8_torch` across one to nine
segments and split factors 1 to 128 (8 of 8, max difference 0); the v1 K gates on the same
library (43 of 43, the v1 path is untouched); in the engine, E1 on 336,000 prompt rows and
E4v2 on 2,352 generated rows over 84 sequence-steps, bit-identical. The remaining v2 cells
(graph identity, single-stream, long-context decode, the SEG=8 prompt kernel's G gates and
its long-context TTFT, serving) are in section 4 where present. Quality is the B8 quality
already measured (+0.052% base, +0.087% engine path); adopting v2 re-pins the contract
constants and calls for the 300-window battery on the final build.

### 7.6 Retrieval battery and the end-to-end verified stack (T16g, T16h in engine form)

**Retrieval (`t16g_needle.py`, `results/t16g_*.json`).** Qwen2.5-7B-Instruct-1M, greedy, the
same engine settings for both stacks: a random 8-hex passkey placed at depths 5 / 25 / 50 / 75 /
95% of a 16k, 32k, 65k and 131k filler context, single-key and four-key variants, three
draws per cell (120 cells). Stock: 120 of 120. Lockstep under contract v2: 120 of 120, and
the generated answer string equals stock's in every cell. The pre-registered kill line
(a loss at any depth) is not triggered at any length.

**The fully committed stack (`T16L_ARM=1` with contract v2; `results/*_fullstack_v2*`).**
Exact int8 linears (`torch._int_mm` with the declared-order epilogue), integer RMSNorm,
frozen-Q14 RoPE, table SiLU, and the contract-v2 attention, under full CUDA graphs. The decode
profile shows the int8 tensor-op GEMMs and the quantise / dequantise kernels in place of the
bf16 kernels, so the stack is armed inside the graphs. Gates under v2: F3 (self-consistent,
alone equals batched, permuted equals batched, which bf16 linears never give) and the
commitment replay (48 rows, 0 bad). Fair mode: single-stream decode 0.952 / 0.964 / 0.932x at
batch 1 / 8 / 32, prefill 0.921 / 0.974 / 0.930x at 2k / 8k / 32k; the four-stream chat example
0.85x with TTFT 0.76x; serving 0.710 / 0.782 / 0.667x. The single-stream cells are the attention-
only numbers; the batched and serving cells are not, and the profile says why: around the int8
GEMMs sit per-token quantise, R4 quantise, dequantise, concatenation and fill kernels of about
40 us per layer per token at batch 1. Fusing the quantisers into the norms and the dequantise
into the GEMM epilogue (the declared-order question of 7.7) is the next engineering item for the
full stack; it does not touch the contract.

### 7.7 Profiler findings (RTX 4090 bare metal, Nsight Compute, `results/ncu_*.csv`, `results/raw_*.csv`)

The same decode kernel source compiled for Ada passes the 16-case gate bit for bit, so the
standalone decode step (`lsb_rowmap`, `lsb_prep`, `lsb_dec8` on synthetic data,
`portable/t16n_prof.py`) was profiled where performance counters are available.

| kernel, regime | duration | what the counters say |
|---|---|---|
| `lsb_prep`, 32 sequences x 300 tokens | 27 us | compute 1.8%, memory 14 GB/s, 77% of cycles no eligible warp; stalls per issue: instruction-cache miss 7.8, long scoreboard 4.5. A dependent-load chain run by one warp per CTA, not work. |
| `lsb_rowmap`, same | 38 us (once per step) | instruction-cache miss 11 to 22 per issue; a serial single-thread loop. |
| `lsb_dec8`, 1 x 32k | 91 us | DRAM 37%, one CTA per SM (79 KB shared memory is Ada's whole allotment), stalls: long scoreboard, barrier. |
| `lsb_dec8`, 1 x 131k | 232 us | DRAM 62%, same occupancy limit, same stalls. |

Reading: the KV-quantise step's cost is latency and code footprint, so it should be folded
into the decode CTA that owns the last block (with every unit CTA quantising its own seven q
rows, which is deterministic and therefore exact); and the decode kernel's shared-memory
footprint, not its arithmetic, sets how many warps cover the chunk-load latency, so half-block
staging with the table read through L1 (about 40 KB, 64 registers) is the lever for long
context on both architectures.

### 7.9 T16m: the fused verifiable stack (built, gated and timed in this session)

The brief was to bring the whole verifiable stack to the attention-only numbers. The
starting profile (batch 1, 300-token context) put the non-attention glue at 56 us per layer
against stock's 17 us for the same roles: four dequantisation passes, three activation
quantisers, the R4 butterfly, two norms, a padding concatenation with its zero fill and a
slice copy for `torch._int_mm`'s 17-row minimum, and two `.contiguous()` copies feeding
RoPE. At batch 32 the same glue was 120 us per layer. `demo/t16m_kernels.cu` replaces it
with five fused kernels and `demo/t16m_stack.py` with one decoder-layer forward:
`norm_quant` (RMSNorm, residual and the int8 row quant in one CTA per row),
`lin_rope` (the qkv GEMM, its epilogue, the bias and the Q14 rotation, v dequantised in
place, no copies), `rowquant_pad` (into a buffer already 17 rows tall), `lin_norm_quant`
(the o_proj GEMM, its epilogue, the post norm, the residual and the next quant),
`lin_silu_r4quant` (the gate_up GEMM, a wide epilogue-and-SiLU pass, then the R4 butterfly
and quant per row) and `lin_out`. Every intermediate bf16 rounding the separate kernels
performed through memory is performed in registers, so each fused op is the exact
composition of the declared ops, and it is gated as such: 316 kernel cases against the
composition of the T16l kernels (random and tie-dense data, rows 1 to 2048, both GEMM
routes, `results/gate_t16m_kernels_v21.json`), the engine gate (per-layer output hashes of
layers 0, 13 and 27 on every real forward plus the ids, fused against unfused, identical at
three route cuts, `results/gate_t16m_engine_m*.json`), E6 full == eager on three graph
shapes, the per-op gate on real activations (`fullstack_egates_v21.json`: every armed op on
layers 0, 13 and 27, 0 bad; it had recorded 0 calls under the plugin until this session
found that the plugin's own arming re-installed the plain patches over the checkers) and
F3.

Two findings on the way. The first fused SiLU kernel folded the ten-step SiLU into the
per-row R4 CTA and took 35 us per row against 3 us for the wide kernel it replaced: forty
dependent IEEE divisions per thread; the shipped op runs the epilogue-and-SiLU wide and the
R4 quant per row, and is faster at every row count. And the int8 GEMM route mattered more
than the glue at prefill sizes: `torch._int_mm` reaches cuBLASLt's sm80 CUTLASS kernels on
an H100, while vLLM's Hopper CUTLASS int8 GEMM is 1.4 to 1.7x faster than bf16 at 2k to 8k
rows but evaluates its fused epilogue with the weight scale first. Contract v2.1 (section
2.6) declares that order; the route is chosen by row count inside each op (`T16M_MCUT`,
default 128, the measured crossover, `results/t16m_gemm.json`), and both routes are gated
bit-identical.

What it measured (section 4, the `t16m22`, `t16m21`, `attn85` rows). GPU time on one node
against the attention-only build: batch-1 decode 5.66 vs 6.25 ms per token, batch 32 9.11
vs 9.74, batch 64 12.96 vs 13.54, a 512-token prefill 12.2 vs 12.6 ms, an 8,192-token
prefill 167 vs 195 ms. Wall-clock against stock, one engine per process: decode 1.005 /
0.999 / 0.909x at batch 1 / 8 / 32, TTFT 1.19 / 1.21 / 1.05x at 2k / 8k / 32k, the chat
example 1.13x TTFT and 0.966x decode, serving 0.811 / 0.877 / 0.934x. The gap between the
GPU-time and the wall-clock pictures is CPU: a python-registered custom op costs 13.8 us
per call where the direct kernel call costs 4.9 us (an aten matmul 10.3 us), six ops per
layer, and that shows on every step CUDA graphs do not capture: prefill chunks and the mixed
steps of the ShareGPT workloads. It does not show on captured decode steps at short context, which is why
batch-1 decode is at parity with stock. At long context the wall-clock deficit turned out to be an artifact of the measuring
pod, not of the stack: vLLM keys its compiled-graph cache on the model config and not on
the Lockstep arming, and the handover pod had been serving the 1M model's "fused" runs a
graph compiled by an earlier build, which is why every fused long-context number there was
identical across builds (0.759 / 0.761x at 32k, 0.736 / 0.731x at 131k) and why freeing
the bf16 weights crashed inside a cached graph. On the 4x H100 node with a fresh cache,
the same harness reads fused 129.4 against attention-only 127.6 tok/s at 32k under full
graphs (and 123.7 against 125.8 piecewise), and the CPU-side attribution at 30k reads
fused 7.67 ms per token wall against attention-only 7.96, with the fused arm's own CPU
work about 0.4 ms per token higher (graph launch of more nodes, allocations) and hidden
behind the GPU. The rule that follows, `VLLM_DISABLE_COMPILE_CACHE=1` for every armed run,
is applied to the `t16m24` rows of section 4, which are the handover pod's re-measurement
of the fused stack under it. Quality on the v2.1 order: +0.26% perplexity
over stock at 300 windows (attention alone +0.087%).

### 7.10 T16o: the decode kernel's inner products on the int8 tensor cores (built, gated, not faster)

The long-context reading of 7.9 leaves the attention decode kernel as the whole remaining
gap, and the obvious lever was its arithmetic: `lsb_dec8` computes the scores and the PV
partials with `dp4a` on CUDA cores, one query per head. `prepz` bit 128 (`LOCKSTEP_DEC8_MMA=1`,
`LSB_GATE_MMA=1` in the rig) replaces both with `mma.sync.m16n8k32` int8 tensor-core tiles:
the head group is the M dimension padded to 16 rows, keys or dims are N, and each warp owns
two tiles for all heads. Because int32 accumulation of int8 products is exact in any order,
the block partials are the same integers `dp4a` produced and every statement after them is
unchanged, which the gates confirm: 21 of 21 rig cases and all 5 head-group variants
bit-identical to the torch reference on the first build, E1 on 151,200 prompt rows and E4v2
on 29,736 generated rows bit-identical in the engine, E6 full == eager. It is not faster:
126.0 against 129.0 tok/s at 32k, 90.6 against 93.4 at 131k and 47.6 against 50.1 at 512k
on the same GPU. That is a
measurement worth having, because it closes the arithmetic hypothesis: the kernel's time is
in its memory pipeline and synchronisation per block (a double-buffered `cp.async` stage of
one 16 KB K block and one 16 KB V block, five block-wide barriers, a shared-memory V
transpose), not in the multiplies. The instrumented build's per-phase cycle counters
(`LSB_DEC_TIMING`, `results/t16o_phase_timers.log`) say where it is: per CTA at 131k
(16 blocks per CTA, 256 CTAs, two per SM) the `dp4a` kernel spends 36k cycles waiting on
`cp.async`, 52k in the score phase, 47k in the table chain and the V transpose, 32k in PV
and 4k in the prologue of 218k in all, about 8 us per block; the tensor-core variant takes
the score phase to 38k and the PV to nothing, and gives it all back in the two extra
block-wide barriers and the shared-memory staging its fragment layout needs (its chain
bucket reads 95k), 222k in all. No phase dominates; the cost is the five barriers and the
short dependent chains between them at sixteen warps per SM. The lever is therefore the
block pipeline, not the arithmetic: more keys per barrier (two to four blocks per phase),
a deeper `cp.async` stage of smaller blocks, and a V layout in the lattice that needs no
transpose in the decode kernel (which touches the prompt kernel's reads, so it is a layout
decision, not a contract one). That is the remaining path to parity at long context; the
half-block pipeline tried in 7.5 (v7) regressed, so it is not a free change. The
tensor-core variant stays behind its flag.

### 7.11 T16q: the decode kernel's memory pipeline (built, gated, not faster) and the routed
tensor-parallel reduction (built, gated, NOT shipped: it regressed the 72B)

The pipeline design from 7.10: K and V single-buffered in shared memory with separate
`cp.async` groups, K(blk+1) issued the moment the score phase has consumed K(blk) and
V(blk+1) the moment the transpose has consumed V(blk), the transpose into its own 16 KB
buffer instead of the dead K buffer, one barrier fewer per block, and a footprint of about
70 KB that fits three CTAs per SM instead of two. Every declared statement is unchanged and
the gates say so: rig 21 of 21, the 5 head-group variants, split paths EQ at 262k / 512k /
1M, and E1 (151,200 rows), E4v2 (29,736 rows) and E6 in the engine, all on the first build
(`results/gate_t16n_t16q*.log`). The phase timers (`results/t16q_phase_timers.log`) show
the design did what it was meant to and that it was not the limiter: per CTA at 131k the
`cp.async` wait fell from 34k cycles to 2.9k, and the total rose from 216k to 236k, with
the score, chain and PV phases each 20 to 50 percent longer. The reason is the grid: the
decode step launches 256 CTAs on 132 SMs, which already fit in one wave at two per SM, so
the third slot per SM is never used and the kernel is instruction-latency-bound at about
fifteen warps per SM. Memory is no longer the constraint at all; warps are. The next
structure is therefore more warps per block of keys (sixteen-warp CTAs working two key
halves of one block, or finer key splits with the fold overhead they bring), which is the
kernel work that 7.12 did.

The tensor-parallel reduction of 2.6 now has its equal-traffic form: an all-to-all of
column blocks, each rank's fold of its own block with the same kernel, then an all-gather
of the reduced blocks, wrapped as one opaque op (`t16m_tp::tp_reduce`). Under a real
4-rank group it is bit-identical to the all-gather form on every case (`t16m_tp_gate.py`)
and, measured on the same node, 2.4 to 2.7x faster at 2,048 rows (325 to 186 us at
N = 3,584; 873 to 361 us at N = 8,192) and slower at decode sizes, where the all-to-all's
latency dominates (78 to 100 us at one row). The op routes by row count, the all-gather
form below 256 rows and the reduce-scatter form above. Measured in the 72B engine
(`72bfs2` rows of section 4) it REGRESSED: decode 0.82x against 0.95x with the plain
form, TTFT 0.80x against 0.85x. The reason is not the arithmetic or the traffic but the
wrapper: the routed op is a python custom op, and its dispatch cost on every uncaptured
step is the cost the native registration of 7.9 removed; at decode sizes it takes the
all-gather path anyway and pays that cost for nothing. So the shipped reduction stays the
all-gather form with the fused fold, and the reduce-scatter form waits for a native
wrapper (the all-to-all through vLLM's communicator, registered in C++), which is the
remaining item for 72B prefill and serving parity under tensor parallelism.

### 7.12 T16r: the sixteen-warp decode kernel (built, gated, faster; shipped until 7.14, now behind `LOCKSTEP_DEC8_P=0`)

The structure 7.11 asked for: `lsb_dec8w`, 512 threads, two warps per head, each warp
owning one 64-key half of the 128-key block. The score phase keeps the lane-per-key layout
with each warp covering a 32-key group for a pair of heads; the per-head chain runs on both
warps of the head over their own halves, and the three quantities the declared statements
need from the whole block are combined exactly through shared memory: the block score max
(the max of the two half maxima), the weight sum (the integer sum of the two half sums),
and the PV partial (the hf=1 warp's integer accumulators added to the hf=0 warp's before the
fold). Integer max and integer addition are associative, so the combined values are the
values the eight-warp kernel computes, and the fold, the segment and row epilogues and every
statement after them run unchanged on the head's first warp. The kernel is selected by
`prepz` bit 512 (`LOCKSTEP_DEC8_W=1`, the shipped default); the eight-warp kernel remains
in the build behind the flag.

The one defect on the way was not in the kernel but in the fused prep it calls on
decode-only steps (7.6): `b8_prep_kv_block` reduced its per-warp amax through a fixed
eight-slot shared array indexed by warp number, so with sixteen warps the upper eight wrote
past the K slots into the V slots, the V exponent absorbed K magnitudes, and the new K/V
rows were quantised on the wrong scale. The signature was exact: the rig passed 21 of 21
with the separate prep and failed 21 of 21 with the fused one (`results/gate_t16r_fuse*.log`),
a bisect of the kernel's own splits (PV split off, chain split off, both off, combine-only)
changed nothing, and the helper is now thread-count agnostic (`blockDim.x` strides, a
sixteen-slot array, the reduction over `blockDim.x / 32` warps).

Gates on the fixed build, all first-run: rig 21 of 21 and the head-group cases with the
fused prep, the split paths at 262k / 512k / 1M (`results/gate_t16n_t16r3*.log`); in the
engine E1 151,200 prompt rows and E4v2 29,736 generated rows bit-identical to the declared
reference with mu on, E6 full graph equal to eager, and the generated token ids equal to the
shipped chunked-fold kernel's and to the T16q kernel's on the same prompts
(`results/gate_e1_t16r3.json`, `results/e6_t16r3_*.json`).

Measured (`results/t16r3_clean_timers.log`, `results/longdec_lssab_p*_t16r3.log`,
`results/single_noinductor_t16r3.json`): the decode step alone, one layer, 64-way split,
against the shipped chunked-fold kernel, two measurements (one on an idle node, one with a
rig running on a neighbouring GPU): 0.237 / 0.246 against 0.257 / 0.249 ms at 32k (8 and 1
percent less) and 0.335 / 0.336 against 0.349 / 0.354 ms at 131k (4 and 5 percent less);
the T16q and T16q2 builds measured the same way on the idle node were 0.253 / 0.369 and
0.260 / 0.368 ms, which is why neither shipped. The eight-warp path of the same build still
passes the rig 21 of 21 (`LOCKSTEP_DEC8_W=0`). In the engine, whole verified stack, 1M model at GMU 0.80 both arms: decode at 32k
131.90 tok/s against stock 151.58 (0.870x, was 0.862x with the shipped kernel) and at 131k
96.29 against 118.73 (0.811x, was 0.796x), and at 512k (GMU 0.90, one measurement, as the
earlier cell) 56.04 against 66.40 (0.844x, was 0.779x); TTFT unchanged (the kernel is
decode-only). The gain grows with context because the attention step's share does.
Short-context single-stream decode is unchanged within noise (B1 1.006x, B8 0.998x, B32
0.898x against 1.002 / 0.997 / 0.905 before). Serving, whole verified stack, both arms in
the same session (`HNI_t16r3` rows of section 4): W1 ShareGPT saturated 0.849x (was 0.829),
W2 0.891x (was 0.878), W3 prefill-heavy 0.983x (was 0.977), every request completed on
both stacks. The 72B at TP=4 on four H100s, whole verifiable stack (`72bfs3` rows of
section 4): decode 0.959 / 0.961 / 0.943x at batch 1 / 8 / 32 against 0.952 / 0.948 /
0.922 with the chunked-fold kernel, TTFT 0.79 to 0.87x unchanged, serving 0.766 / 0.774 /
0.770x against 0.757 / 0.751 / 0.761 (the 72B serving gap is the reduction's traffic, 7.11). The remaining long-context gap is still the
attention step, now latency-bound at 16 warps per CTA and two CTAs per SM; the next
structure is a finer key split across CTAs with the fold cost it brings, and it is not built.

Two follow-ups measured after it. A split-policy sweep (`results/split_sweep.log`, 32 to
256 splits, three kernels, 32k and 131k): for the shipped kernel 64 splits is the optimum
at both lengths and every larger count is slower, so the grid cannot supply more overlap
without paying for it in partials, and the split policy stays. And T16u, the first block's
K/V loads issued ahead of the fused per-token prep (the prep is 31 percent of a CTA's
cycles at batch 32 with 300-token contexts): bit-identical under every gate, 21 of 21, the
head-group cases, 262k / 512k / 1M and the eight-warp path (`results/gate_t16n_t16u.log`),
and not faster - 0.241 against 0.229 ms per step at batch 32, unchanged at 32k and 131k
(`results/t16u_timers.log`). The prep's latency was not on the critical path the loads
could hide; the change is not shipped.

### 7.13 T16s: the native tensor-parallel reduction (built, gated; 72B measurement pending)

The wrapper 7.11 asked for. `t16mc::tp_reduce(y, ws, mcut, group)` is one C++ op
(`demo/t16m_tp.cpp`): it resolves vLLM's tensor-parallel device group by its registered
c10d name, and runs either form of the declared rank-order reduction with no python between
the collectives: below `mcut` rows the all-gather of the full partials and the fold kernel,
from `mcut` up the all-to-all of column blocks, each rank's fold of its own block, and the
all-gather of the reduced blocks. The collectives go through the same process group as
vLLM's own all-gather, so the op captures into CUDA graphs the same way. It is the default
reduction when the fused stack is armed under tensor parallelism (`T16M_TP_NATIVE=1`); the
previous forms stay behind their flags.

Gate (`t16s_tp_gate.py`, four ranks, real NCCL group): bit-identical to the all-gather
reference on 81 cases, every case through both routes (N = 896 / 3,584 / 8,192, M = 1 to
8,192, including rows on either side of the cut). Timing on the same node, one call:

| N | M | python all-gather form (shipped before) | native, routed | native all-gather form | native all-to-all form |
|---|---|---|---|---|---|
| 3,584 | 1 | 73.4 us | 27.2 | 25.0 | 49.7 |
| 3,584 | 64 | 70.0 | 26.9 | 26.8 | 65.1 |
| 3,584 | 256 | 333.3 | 70.8 | 49.9 | 63.7 |
| 3,584 | 2,048 | 359.1 | 181.8 | 233.8 | 181.5 |
| 3,584 | 8,192 | 1,148.5 | 557.7 | 818.6 | 557.5 |
| 8,192 | 1 | 68.2 | 25.4 | 25.1 | 47.0 |
| 8,192 | 2,048 | 682.7 | 347.1 | 484.8 | 347.2 |
| 8,192 | 8,192 | 2,535.1 | 1,180.2 | 1,801.0 | 1,181.2 |

At decode sizes the native op is 2.7x faster than the python form it replaces, which is
the dispatch cost 7.11 attributed; at prefill sizes 2 to 2.1x, which is the traffic. The
all-gather form still wins at 256 rows, so the cut is 512 (`T16M_TP_RS_MCUT`).

In the 72B engine the first form, the native op on every row count (`72bfs4` rows of
section 4), moved TTFT to parity - 0.96 to 1.01x at 2k to 32k prompt tokens against 0.85 to
0.87 before, the attention-only control's level - and cost four decode points: 0.919 /
0.926 / 0.909x against 0.959 / 0.961 / 0.943 with the graph-captured vLLM all-gather. The
reason is where the microbench cannot see: inside the captured decode graph the c10d
all-gather runs on the process group's own stream with event hand-offs on both sides, and
the 72B makes 160 such calls per token; vLLM's all-gather is stream-ordered. So the shipped
routing keeps the captured vLLM all-gather and the fused fold below 512 rows and takes the
native all-to-all form from 512 rows up, where the traffic is what matters; the branch is a
shape guard, so the compiled model carries one graph per side of the cut. Serving under
the first form said the same thing from the other side: the prefill-heavy W3 went 0.770 to
0.857x (mean TTFT 26.6 to 23.7 s against stock's 20.4) while the decode-heavy W1 and W2
went 0.766 / 0.774 to 0.747 / 0.741. The routed form (`72bfs5` rows) kept the prefill
side - TTFT 0.96 to 1.01x at 2k to 32k, W3 0.860x - and did NOT recover the decode side:
single-stream decode 0.920 / 0.922 / 0.906x and W1 / W2 0.740 / 0.752, the same as the
all-native run, although at decode sizes it runs exactly the earlier all-gather path. So the
decode loss is not the route. A control run of the earlier path on the current build
(`72bfs6`, `T16M_TP_NATIVE=0`) separates the reduction change from whatever else moved on
disk between the runs. The control (`72bfs6`) read 0.961 / 0.961 / 0.944x: the loss belongs
to the change. A discriminator (`72bfs8`, the native path armed with its cut at a million
rows so it never executes) read 0.961 / 0.964 / 0.945x: the branch, the group lookup and
the op's registration cost nothing; the loss appears only once torch's own NCCL
communicator has actually run a collective in the process. That is the signature of
NCCL's graph-mixing support: when one communicator is used outside CUDA graphs on a device
whose other communicator (vLLM's) runs inside captured graphs, every graph-launched
collective pays a synchronisation, and the 72B launches 160 of them per token. Two fixes
measured next: the mixing support disabled by environment (`NCCL_GRAPH_MIXING_SUPPORT=0`,
`72bfs9`), and the all-to-all carried by vLLM's own pynccl communicator as send/receive
pairs in one NCCL group on the current stream, so no second communicator exists
(`T16M_TP_NATIVE=2`, `72bfs10`).

The environment switch confirmed the mechanism and found more than the loss: with the
mixing support off, the routed native reduction reads decode 0.983 / 0.986 / 0.965x
(`72bfs9`) - two points ABOVE the earlier path's 0.961 / 0.961 / 0.944 - with TTFT 0.95 to
1.00x at 2k to 32k. vLLM's own collectives inside the captured decode graphs had been
paying the same synchronisation all along, on both stacks. Fairness therefore requires
the stock arm re-measured under the same environment (`72bnm_stock`, the `72bfs11` rows);
the reading that ships is that pair. The other fix, the all-to-all as send / receive
pairs on vLLM's own pynccl communicator (`72bfs10`), kept the prefill parity (TTFT 0.96 to
1.01x) and made decode WORSE, 0.842 / 0.808x: that communicator is the one the decode
graphs launch, and using it outside a graph engages the same synchronisation on every
graph launch. It is bit-identical (7B at TP=4, E6 full equal to eager, tokens equal to the
all-gather run) and not shipped; `T16M_TP_NATIVE=2` keeps it selectable.

The fair pair (`72bfs11` rows, both arms under the environment): stock's own numbers move
within 0.2 percent on decode and gain 12 percent on the 128-token TTFT; against them the
whole verifiable stack reads decode 0.982 / 0.985 / 0.963x at batch 1 / 8 / 32, TTFT 0.93
to 1.00x from 2,048 prompt tokens (0.72x at 128 tokens, where stock's gain landed), and the
chat e2e 0.96x decode at 0.90x TTFT. Serving under the pair splits: W2 0.794x and W3 0.853x
against 0.774 / 0.770 before the change, and the saturated chat workload W1 0.701x against
0.766 - a loss the single-stream and long-prompt cells do not show, so it belongs to the
MIXED prefill-plus-decode steps that only saturated chat produces, where the all-to-all
runs on the c10d group outside any graph. A discriminator run (`72bfs12`: the environment
alone with the earlier reduction) separates the two: it reads decode 0.984 / 0.987 /
0.964x - the environment alone carries the whole decode gain - with W1 0.765, W2 0.790 and
W3 0.766, TTFT 0.85x. So the all-to-all is worth its prefill parity and the W3 gain, and it
costs W1 on the mixed prefill-plus-decode steps of saturated chat, where a step of a few
hundred to a few thousand rows carries decode rows that then wait on it. The resolution is
the cut: raised to 4,096 rows the all-to-all serves the large prefill steps (W3's prompts,
the 8k and 32k TTFT cells) and saturated chat's mixed steps stay on the captured
all-gather (`72bfs13` rows, `T16M_TP_RS_MCUT=4096`); that pair is the shipped reading:
decode 0.982 / 0.987 / 0.961x at batch 1 / 8 / 32, TTFT 1.00x at 8,192 prompt tokens (batch
1, 4 and 8), 0.954x at 32k, 0.82x at 2,048 tokens batch 1 (that step has 2,048 rows and
stays on the all-gather; 1.00x at batch 8), the chat e2e 0.965x decode at 0.91x TTFT, and
serving W1 0.781x (its best reading; 0.766 before the change), W2 0.784x, W3 0.768x. The
512-row cut is the knob for prefill-heavy deployments: W3 0.853x and TTFT parity from
2,048 tokens, at W1 0.701x. The mechanism behind the trade is the serving engine's
2,048-token prefill chunks: every chunked step carries decode rows, and on saturated chat
those rows wait on the all-to-all.

### 7.14 T16v: the two-group pipelined decode kernel (built, gated, faster, SHIPPED)

The intra-CTA pipeline the split sweep of 7.12 asked for. `lsb_dec8p` is 512 threads as two
eight-warp groups. Group A computes the even blocks of a unit and folds EVERY block in
block order, the declared fold order; group B computes the odd blocks and hands its
results - the exact integer PV partial per head, the weight sum, the block exponent and
e_v - to A through two shared slots guarded by named barriers (producer arrives, consumer
syncs, one pair per slot for full and empty), so B runs up to two blocks ahead and the two
groups overlap each other's score, chain and table phases. Each group is the eight-warp
kernel of 7.8 verbatim on its own K, V, score and weight buffers with its own group barrier
(`bar.sync 1+group, 256`), and V is transposed in place through registers, so a CTA stays
at 96 KB and two fit per SM. Integer results are exact and the fold runs in one place in
block order, so every declared statement is unchanged; `prepz` bit 1024
(`LOCKSTEP_DEC8_P=1`) selects it.

Gates, first build, fused prep: rig 21 of 21 and the head-group cases, the split paths at
262k / 512k / 1M (`results/gate_t16n_t16v.log`), 64 registers, 8 bytes of spill; the
sixteen-warp and eight-warp paths of the same build 21 of 21; in the engine E1 151,200 rows
and E4v2 29,736 rows bit-identical, E6 full graph equal to eager, and the generated tokens
equal to the chunked-fold and sixteen-warp kernels' (`results/gate_e1_t16v.json`,
`results/e6_t16v_*.json`). The shipping build (`results/gate_t16n_t16v4.log`: the same
source with the merged fallback of 7.15 compiled out, 64 registers, 8 bytes of spill) passes
the rig on all three kernel paths, the head-group cases and 262k / 512k / 1M, and, promoted
to the shipped name with the backend defaults, reproduces the gated tokens
(`results/gate_shipped_t16v4_smoke.log`).

Measured in the engine, whole verified stack (`t16v` rows of section 4), against the
sixteen-warp kernel of 7.12: long-context decode at 131k 111.82 tok/s against stock 118.73,
0.942x (was 0.811x; 0.796x before 7.12); at 512k (GMU 0.90, one measurement, as the earlier
cells) 59.29 against 66.40, 0.893x (was 0.844x; 0.779x before 7.12); at 1M tokens (the 1M
model on ONE H100 of the four-GPU node, both arms in one session, GMU 0.88 / 0.90 as the
earlier cells) 38.16 against 37.55, 1.016x - parity, was 0.860x with the chunked-fold
kernel - at a TTFT of 0.81x that is the prompt kernel's and unchanged; at 32k 135.18
against 151.58, 0.892x (was 0.870x); single-stream decode B32 0.915x (was 0.898x), B1 and
B8 at parity, TTFT unchanged. The
decode step alone on an idle GPU, interleaved with the sixteen-warp kernel, 64 splits:
0.312 against 0.346 ms at 131k, and 1.5 to 3 percent slower at 32k, 65k and batch 32,
where a unit of four blocks leaves the two groups little to overlap and the hand-off shows;
the engine's own split policy gives longer units than the rig's 64 there, which is why the
engine gains at 32k where the rig loses. Serving under the shipped split policy is unchanged
with it (W1 / W2 / W3 0.848 / 0.891 / 0.984x against 0.850 / 0.891 / 0.983), and the
serving-shaped rig says why (`results/b64_attribution.log`, `results/b64_split_attribution.log`,
batch 64, 7B head shape): at 300-token contexts the two kernels are equal at every split
count (0.31 to 0.33 ms per layer step); at 2k the pipelined kernel at ONE split is 0.388 ms
against 0.429 for the shipped kernel at the policy's four (units of four blocks, one CTA
prologue per unit); at 8k 0.622 at two splits against 0.751 at four. The lever for serving
is therefore the split target (`LOCKSTEP_LSSAB_SPLIT_CTA`, 8 SMs' worth of CTAs today)
together with the pipelined kernel, measured in 7.15. On the 72B at TP=4 (32k ceiling,
short contexts, every unit short) the pure pipelined kernel reads 0.952 / 0.952 / 0.932x
against the sixteen-warp kernel's 0.961 / 0.961 / 0.944 on the same day (`72bfs7` against
`72bfs6`): the hand-off with nothing to overlap. The merged form (`DEC8P_MINUB`, the pipelined
body for units of eight blocks or more and the sixteen-warp body below, one launch, no
host decision because the launch bits are captured into the graphs) is 7.15.

### 7.15 T16v3: the merged decode kernel (built, gated, not shipped) and the split target

The merged kernel: `lsb_dec8p` running the pipelined body of 7.14 when a CTA's unit holds
`DEC8P_MINUB` (eight) blocks or more and the sixteen-warp body of 7.12 below that, decided
inside the kernel from the unit length, because the launch bits are captured into the CUDA
graphs and a host-side choice per step is not available; the sixteen-warp body is the same
code (`lsb_dec8w_body`, called by both kernels). Bit-identical on every path on the first
build: rig 21 of 21 on the pipelined, sixteen-warp and eight-warp paths, the head-group
cases, 262k / 512k / 1M (`results/gate_t16n_t16v3.log`). Not faster: compiling both bodies
into one kernel raises the register pressure (34 bytes of spill against 8 for the pure
pipelined kernel, 0 for the sixteen-warp one), and the interleaved timers
(`results/t16v3_timers.log`) put its sixteen-warp path 3 percent behind the standalone
sixteen-warp kernel at 32k (0.240 / 0.246 against 0.232 / 0.233 ms) and its pipelined path
behind the pure pipelined kernel at 131k. Since the pure pipelined kernel already beats
the sixteen-warp kernel in the engine on every 7B cell of 7.14 - batch 32 and short serving
included, because the engine's split policy gives it longer units than the rig - and loses
only one point on the 72B's short-context engine (7.14), the pure pipelined kernel ships as
the default (`LOCKSTEP_DEC8_P=1`; `DEC8P_MINUB` defaults to 0, which makes the fallback
dead code and restores the pure kernel's register allocation) and the sixteen-warp kernel
stays behind `LOCKSTEP_DEC8_P=0` for short-context tensor-parallel engines.

The split target: the engine's split policy asked for eight SMs' worth of CTAs per launch,
which at serving shapes made units of four blocks and paid a CTA prologue per unit
(7.14). `LOCKSTEP_LSSAB_SPLIT_CTA` is measured at 2 and 4 against the shipped 8 in the
engine (single-stream and serving, `sc2` / `sc4` rows) and at 2 together with the
pipelined kernel (`t16vsc2` rows). With the sixteen-warp kernel the A/B is mixed: the
target of 4 lifts single-stream decode to 1.04 / 1.03 / 0.92x at batch 1 / 8 / 32 (against
1.01 / 1.00 / 0.90 at 8), but saturated ShareGPT serving (W1) falls from 0.850 to 0.828x
at 4 and 0.823 at 2, W2 and W3 move within the 2 percent that the stock arm itself moves
between runs. The serving-shaped rig of 7.14 predicted this: at 300-token contexts, which
dominate W1, fewer CTAs buy nothing and cost parallelism. The pipelined kernel at a
target of 2 (`t16vsc2` rows) reads the same way: single-stream 1.000 / 0.995 / 0.906x
(against 1.007 / 1.002 / 0.915 at 8), W1 0.825, W2 0.907, W3 0.986, long-context decode
135.11 tok/s at 32k (135.18 at 8) and 113.63 at 131k (111.82 at 8, 0.957x against 0.942).
A 1.6 percent gain at 131k against a 2.5 point loss on saturated chat serving: the shipped
target stays at 8, and `LOCKSTEP_LSSAB_SPLIT_CTA=2` is documented for long-context
deployments that do not serve short chat traffic.

### 7.16 Group A: the second model family (Llama-3.1-8B-Instruct; attention only measured, whole stack pending)

The question of `docs/PLAN_GROUP_A.md`: is the contract a property of Qwen2.5? Llama-3.1-8B-Instruct
(`NousResearch/Meta-Llama-3.1-8B-Instruct`, the ungated mirror; 32 query heads, 8 kv-heads, head
dim 128, G = 4; llama3 RoPE scaling) was run with no kernel change: the key offset came from
`k_equalize.py`, whose statistics hook now follows the model family's `eager_attention_forward`
(`keq_llama31_8b_instruct.pt`, sha `c22e1d59...`), and the shipped pipelined kernel took G = 4 as
gated by rig family K10u.

Attention only, first run: E1 172,800 prompt rows and E4v2 33,984 generated rows bit-identical
to `lssab8_torch(mu)` at layers 0, 15 and 31 (`results/gate_e1_llama.json`); E6 full graphs equal
to eager; single-stream against stock in the same session (`llama` rows of section 4): decode
0.958 / 0.963 / 0.943x at batch 1 / 8 / 32, TTFT 0.78x at 128 tokens, 0.94 / 0.95 at 2k, 0.98 /
0.95 at 8k, 0.93x at 32k. Those are the attention-only ratios the Qwen2.5-7B read before the
fused int8 stack lifted its prefill above stock.

The whole verifiable stack, second run, again with no code change (the fused stack quantises
vLLM's own `cos_sin_cache`, so the llama3 RoPE scaling came for free; the layer module names
are the same; the untied head is generic): E6 full graphs equal to eager, the per-op identity
gate of `t16l_egates.py`, and `t16m_gate.py compare` with 72 of 72 layer calls identical and
the ids identical between the fused and unfused forwards (`results/spin_llama_*.log`,
`results/egates_llama.log`). The whole-stack single-stream and serving cells (`llama_fs`
rows) and the perplexity of stock, attention-only and whole stack follow; the first attempt at
the single-stream cell hit an out-of-memory at GMU 0.85 because the 128k-token vocabulary's
head keeps its bf16 copy beside the int8 fold, so the cell runs with `T16L_FREE_BF16=1`, as
the 72B does. With the weights freed, the whole verifiable stack on Llama reads decode 1.017 /
1.021 / 0.977x at batch 1 / 8 / 32 against stock in the same session - above parity at batch 1
and 8, and at batch 32 two points better than the Qwen2.5-7B's 0.915 (the 128k-token head is a
larger share of Llama's step, and the int8 head wins it). No constant of the contract was
changed, so the first answer is: the contract, attention and non-attention, carries to a
second family as written, at the same or better cost.

Perplexity of the second family (`results/ppl_llama_*.log`, the harness's `T16H_STACK`
switch, 300 windows): stock bf16 6.5268, attention only 6.5453 (+0.28 percent, inside the
+0.30 band as on Qwen2.5). The whole-stack figure is pending: the first run freed the bf16
weights and the harness's prompt-logprob path reads the bf16 head, which returned a
meaningless number; it is re-run with the weights kept.

A third family, Qwen3-8B (36 layers, 32 / 8 heads, head dim 128, q_norm and k_norm before
RoPE; `Qwen/Qwen3-8B`, ungated), attention only, again with no code change and its own key
offset (`keq_qwen3_8b.pt`, sha `8bebf847...`): E1 172,800 prompt rows and E4v2 33,984
generated rows bit-identical at layers 0, 17 and 35, E6 full graphs equal to eager
(`results/gate_e1_qwen3.json`, `results/e6_qwen3_*.json`); single-stream against stock in
the same session (`qwen3` rows of section 4): decode 0.965 / 0.971 / 0.960x at batch 1 / 8 /
32, TTFT 0.80x at 128 tokens, 0.95 to 0.97x from 2k to 8k, 0.91x at 32k - the attention-only
level of the other two families; perplexity attention only 8.6507 against stock's
8.7210 (-0.81 percent). The qk-norm path of the whole stack (the norms are the integer RMSNorm already)
is the next A3 run.

A fourth family, Phi-4-mini-instruct (24 / 8 heads, G = 3, head dim 128, longrope, a
declared sliding window of 262,144 tokens), attention only: the backend first refused the
window outright; with the rule that a window at or beyond the engine's context ceiling is
no window (a Group B change on the backend, captured at construction), E1 129,600 prompt
rows and E4v2 25,488 generated rows bit-identical, E6 full graphs equal to eager
(`results/gate_e1_phi4.json`). Perplexity attention only 8.5601 against stock's 8.5276,
+0.38 percent - the first family outside the +0.30 band, by 0.08 points, with three query
heads per kv-head (the key offset's statistics have a third of the heads to average over);
single-stream (`phi4` rows of section 4): decode 0.884 / 0.936 / 0.955x at batch 1 / 8 / 32
(G = 3 leaves five of a CTA's eight head warps idle at batch 1), TTFT 0.92 to 0.95x from
2k to 8k, 0.89x at 32k. A window smaller than the context stays outside the contract as
written.

**Open defect, the whole stack on Llama (Group A, found 2026-09-06).** With the fused
non-attention stack armed, Llama-3.1-8B's prompt log-probabilities are wrong (a per-token
log-probability of -600 where stock reads -15 to -1, and a different next token after a
512-token wikitext prompt), while attention only is right (+0.28 percent perplexity) and
the fused stack's own gates pass: E6 (fused full graphs equal to fused eager), the per-op
gate against the declared ops, and the fused-equals-unfused compare. Those gates compare
the verifiable stack against ITSELF and against its declared statements; none of them
compares Llama's whole-stack output against stock bf16, which the Qwen2.5 perplexity did.
So the stack computes its declared function consistently and that function is wrong for
Llama somewhere the 7B-shaped gate never looked: the fused ops were gated at Qwen2.5-7B's
widths (3,584 / 4,608 / 18,944) and Llama's are 4,096 / 6,144 / 14,336 with a 128k-token
head. The route-discriminating probe (`lssa_runtime/llama_lp_probe.py`, each GEMM route
forced) and the fused-op gate at Llama widths (`T16M_GATE_SHAPE=llama8b`) are running; the
Llama whole-stack timing cells of this section stand as timings of a wrong function until
the defect is fixed, and its perplexity is not claimed.


**The whole-stack defect, found and fixed (2026-09-06).** Bisected with one fused op armed at
a time (`T16L_OPS`) and then one rotation at a time (`T16L_ROT`), on a 512-token wikitext
prompt's per-position log-probs against stock (`lp_llama_bis_*.json`): every single op is
wrong with the default rotations (max |d log-prob| 690 to 930), all ops together with the
rotations OFF are right (max 12.4, within the attention-only noise), R2 or R4 alone are
right (max 3.0 / 3.2), R1 alone reproduces the defect (929.8). The cause is
`build_Q1`: the R1 rotation was a Paley type-I Hadamard matrix of order n, which is only a
Hadamard matrix when n - 1 is a prime congruent to 3 mod 4. That holds for 3,584 (Qwen2.5-7B)
and 8,192 (72B) and fails for 4,096 (Llama-3.1-8B and Qwen3-8B: 4,095 = 3^2 . 5 . 7 . 13)
and 3,072 (Phi-4-mini: 3,071 = 37 . 83), where the construction returned a non-orthogonal
matrix and the embedding, the norms' consumers and the lm_head were folded through a
transform that is not undone. The fix (`hadamard_for` in `t16l_stack.py` and `t16k_rot.py`):
Sylvester for powers of two, Paley I where it is valid, block-diagonal Sylvester otherwise,
and a hard gate - `build_Q1` raises when max |Q Q^T - I| exceeds 1e-9, so a non-orthogonal
R1 can never fold silently again. The 7B and 72B constants are unchanged (their kind stays
`paley1`, same hashes). With the fix, Llama-3.1-8B's WHOLE stack: perplexity 6.5540 against
stock's 6.5268 (+0.42 percent; attention only +0.28); E6 full graphs equal to eager and
self-consistent (`results/e6_llama_fix_*.json`); per-op identity (`egates_llama_fix.log`) and
fused == unfused 72 of 72 layer calls with identical ids; the probe's max |d log-prob| against
stock falls from 929.8 to the attention-only band. Single-stream, whole stack against stock
(`single_noinductor_llama_fix.json`): decode 1.006 / 0.998 / 0.947x at batch 1 / 8 / 32,
TTFT 1.13x at 2k, 1.20x at 8k, 1.04x at 32k, 0.885x at 128 tokens - the exact int8 linears
put Llama's prefill ABOVE stock, as on the 7B. The pre-fix whole-stack rows (`llama_fs`,
serving W3 1.015x) were timings of a wrong model and are marked superseded in the tables.
Two further family-specific findings from the same battery: Qwen3-8B's per-head q_norm /
k_norm sit between the qkv GEMM and RoPE, where the fused `lin_rope` op has no slot - the
fused-layer guard tested a `qk_norm` attribute vLLM's Qwen3 does not have, so all 36 layers
were fused WITHOUT the norms (perplexity 32,418); the guard now keys on `q_norm` / `k_norm`
and those layers take the unfused exact path (Group B file, `t16m_stack.py`). Phi-4-mini ties
lm_head to the embedding, which the R1 fold refused; embedding rows and head take the same
transform (e -> e Q, W -> W Q), so the tied tensor is rotated once and shared
(`R1_tied_lm_head`). Both families' whole-stack batteries are the `*_fix` rows, and they
read differently:

- **Qwen3-8B, whole stack** (`single_noinductor_qwen3_fix.json`, `ppl_qwen3_fullk_fix.log`):
  E6 full == eager and self-consistent, per-op identity, fused == unfused 72 / 72 with
  identical ids; perplexity 8.7507 against stock's 8.7210 (+0.34 percent; attention only
  -0.81). Speed is BELOW the attention-only build - decode 0.888 / 0.867 / 0.873x at batch
  1 / 8 / 32 against 0.965 / 0.971 / 0.960; TTFT 0.73x at 2k, 0.81x at 8k, 0.80x at 32k -
  because every layer has q_norm / k_norm and therefore runs the UNFUSED exact path (the
  python-registered ops, one 128-wide norm launch per q and per k). The fix is a fused
  `lin_rope` with the per-head norm between the GEMM and RoPE (Group B, B2'''); the
  arithmetic is already declared and gated, only the launch count changes.
- **Phi-4-mini, whole stack** (`single_noinductor_phi4_fix.json`, `ppl_phi4_fullk_fix.log`):
  E6, per-op identity and fused == unfused 72 / 72 all pass - the stack is self-consistent -
  but its perplexity is 9.8123 against stock's 8.5276 (+15 percent; attention only +0.38),
  so one of the declared non-attention ops loses far more on this model than on the other
  three. The bisect by op and by rotation on 30 windows (`pod/a9_phi4_bisect.sh`,
  `results/ppl_phi4_bis_*.log`) is the open item; its timing rows (decode 0.70 / 0.70 / 0.73x)
  are recorded as timing only. The bisect (300 windows each, `results/ppl_phi4_bis_*.log`)
  reads: attention only 8.5601; whole stack 9.8123; no rotations 10.5251; R2 + R4 only
  10.2934; R1 only 9.9172; linears alone without rotations 10.5331; and the norm alone,
  the logits alone and SiLU + RoPE alone all 9.44 - the SAME number whatever single op is
  armed, which points at what arming does before any op runs: the norm-gain fold. Phi-4
  ties `lm_head.weight` to the embedding (one Parameter), so folding the final norm's gain
  into the head multiplied the EMBEDDING table by that gain - a +10 percent floor that
  every variant shares. The fix unties the head before the fold (one bf16 copy of the
  200k x 3,072 table, `untied_lm_head` in the fold metadata); the R1 rotation then applies
  to the embedding and the head separately as on every other family. The remaining gap
  between 9.44 and 9.81 is the int8 linears on Phi-4's residual stream, which the rotations
  reduce (10.53 without them) but do not remove. With the head untied
  (`single_noinductor_phi4_fix2.json`, `ppl_phi4_fullk_fix2.log`): perplexity 8.9179, +4.6
  percent against stock (from +15); E6 full == eager, per-op identity and fused == unfused
  72 / 72 pass; decode 0.703 / 0.705 / 0.743x at batch 1 / 8 / 32 (attention only 0.884 /
  0.936 / 0.955), TTFT 0.63x at 2k, 0.72x at 8k, 0.73x at 32k. Two open items on this
  family, both contract-level rather than defects: the residual stream's massive
  activations make the per-token int8 linears cost 4 points of perplexity where the other
  three families lose under 0.5 (the remedies are the two-plane `a15` activations the
  lm_head already uses, at twice the GEMM cost, or a declared per-channel smoothing fold),
  and the single-stream speed, which tracks the per-launch glue of a 32-layer, 3,072-wide
  model at batch 1 (SPEC 7.17's cluster quantisers address exactly that regime).
### 7.17 Group B: performance where deployments live (measurements; the reduction through the custom all-reduce)

The question of `docs/PLAN_GROUP_B.md`: what the whole verifiable stack spends on saturated
short-context serving, and what moves it. Three measurements answered it, and the first
shipping change follows from them.

**B1, the decode kernel at short units** (`results/b1_attrib72.log`, the sixteen-warp
timing build, batch 32 and 64 at 300 and 2k tokens, fused prep on and off, splits 1 and 8):
with three blocks per CTA the fused prep is 10k of 34k to 46k cycles per CTA, and the
separate prep launch is slower overall (0.260 against 0.234 ms per step at batch 32), so
that latency is inherent; the kernel is about 35 us per layer step at batch 64 with
300-token contexts, one quarter of what the profile below charges to attention.

**B1b, the saturated-chat profile** (`results/w1_prof_*.json`, `lssa_runtime/w1_prof.py`:
128 ShareGPT requests, 64 concurrent, 2,048-token chunks, torch profiler, kernels bucketed
by name, Qwen2.5-7B on one H100). Stock: 3,570 ms of GPU time, 81 percent in bf16 cuBLAS
linears, 11.5 percent attention. The whole verifiable stack: 3,991 ms (+421), decomposed
against stock as: int8 linears -605 ms (2,094 against 2,895: the cuBLASLt sm80 CUTLASS
kernels `torch._int_mm` reaches at 64 rows, still faster than bf16); the fused glue kernels
+382 ms (normquant 6.4 us, r4quant 10.7, deqsilu 9.0, rowquant 4.0 per launch at 64 rows,
against stock's 2.7 / 5.9 / 3.6 for rms_norm, act_and_mul and rotary); per-step int64 and
copy kernels outside the graphs +298 ms, which the attention-only profile does not have
(20 ms), so they belong to the fused stack's path; the decode kernel +178 ms over stock's
attention (594 against 410). So at saturated chat the attention kernel is a fifth of the
gap; the fused glue at 64 rows and the copies are the larger two, and both are cheaper to
fix than a kernel.

**B4a, the reduction on mixed steps** (`results/tp_trace_*.jsonl`, the opaque trace op of
`t16m_stack.py`, 72B at TP=4, serving W1 and W3 with the 512-row cut): on the uncaptured
steps, 160 reductions per step at 0.18 to 0.50 ms each, 30 to 65 ms per step against a
NCCL all-reduce of 0.15 to 0.2 ms at those sizes. The absolute cost of BOTH forms was the
serving gap, not the choice between them.

**B4c, the declared reduction through vLLM's custom all-reduce.** vLLM's IPC all-reduce
(`csrc/custom_all_reduce.cuh`, one-shot below 512 KB and two-shot above) accumulates in
fp32 in rank order 0..N-1 and rounds once at the end - exactly the declared
`bf16(fl32(y_0) + fl32(y_1) + ...)`. Gated on four ranks against the all-gather-and-fold
form (`lssa_runtime/t16s_car_gate.py`): 32 of 32 cases bit-identical (N = 896 / 3,584 /
8,192, M = 1 to 4,096; the 64 MB case is the buffer's own size and is skipped). Timing:

| N | M | all-gather + fold (shipped before) | custom all-reduce (rank order) | NCCL all-reduce (stock, unpinned order) |
|---|---|---|---|---|
| 3,584 | 1 | 74.5 us | 11.3 | 27.8 |
| 3,584 | 64 | 71.5 | 12.6 | 28.2 |
| 3,584 | 256 | 173.6 | 22.0 | 27.3 |
| 3,584 | 2,048 | 324.0 | 97.8 | 101.1 |
| 8,192 | 64 | 70.4 | 16.6 | 34.2 |
| 8,192 | 256 | 108.9 | 35.0 | 43.4 |
| 8,192 | 2,048 | 678.2 | 211.0 | 196.2 |

The pinned reduction now costs what stock's unpinned one costs, or less. The routing mode
`T16M_TP_NATIVE=3` uses the engine's own registered instance inside captured graphs and a
64 MB instance of its own on uncaptured steps, with the routed forms as the fallback for
sizes or topologies it does not cover. The 7B TP=4 identity gate passes: E6 full graphs
equal to eager and the ids equal to the all-gather run's (`results/e6_tp4_car_*.json`).
On the 72B at TP=4, both arms under `NCCL_GRAPH_MIXING_SUPPORT=0` (`72bfs14` rows): decode
1.037 / 1.039 / 1.038x at batch 1 / 8 / 32 against stock - the whole verifiable stack ABOVE
stock on the 72B - TTFT 1.10x at 2k tokens batch 1, 1.00x at 2k batch 8 and at 8k, 0.954x at
32k, 0.90x at 128 tokens; the chat e2e 193.6 tok/s against stock's 191.1. Serving, 64
concurrent, same session, both stacks in full graphs (`results/*_repHNI_72bfs14_*.json`):

| Workload | stock out tok/s | custom all-reduce (72bfs14) | ratio | before (72bfs13, all-gather + fold) |
|---|---|---|---|---|
| W1 ShareGPT chat, 500 requests | 2,077.8 | 1,897.5 | 0.913x | 0.781x |
| W2 mixed, 256 | 3,243.6 | 3,184.8 | 0.982x | 0.784x |
| W3 4k prefill-heavy, 128 | 198.0 | 194.6 | 0.983x | 0.768x |

Every 72B serving workload moves inside 10 percent, two of them inside 2; the routed
all-gather form cost 20 points on each. `T16M_TP_NATIVE=3` is therefore the default in
`t16m_stack.py` from this commit (the routed forms stay as modes 1 and 2 and as the
fallback where the custom all-reduce declines a size or a topology). What remains on the
72B is W1 at 0.913x, the same short-unit attention deficit as the 7B's W1, which the
remaining Group B items (B2'', B2', B2) address.

**B2'', the lm_head epilogue in one kernel** (`t16mc.lm_combine`, `lssa_runtime/t16m_kernels.cu`;
`lm_fused_logits` in `t16m_stack.py`; gate `results/t16m_kernels_b2pp.json`). The copy-shape
profile of saturated chat (`w1_prof_shapes.py`, 455 shape groups) charged every one of its
copy / add / to kernels to tensors of shape [rows, 152,064]: the lm_head's two-plane
(`a15w15`) combine, `sum_i int64(acc_i) << sh_i`, then `.float() * asc * ws`, `.to(bf16)`
and the `org_vocab_size` slice - eleven passes over a 39 MB tensor at 64 rows, all outside
the graphs. The kernel does the four steps in one pass: the int64 sum is exact, the
int64-to-fp32 conversion rounds to nearest even as torch's does, the two fp32 products are
correctly rounded, one bf16 rounding, and only the first `org_vocab_size` columns are
written. Its second form stacks the two activation planes into one GEMM
`[2M, K] x [K, 2N]` against the two weight planes laid side by side (`T16M_LMSTACK=1`,
default), so each weight plane is read once; the four quadrants of the int32 result are the
four plane products, and int32 accumulation is exact, so the tiling cannot change a bit.
Gated against the torch variant at 1 to 300 rows, both int variants (`a8`, `a15w15`), random
and tie-dense activations, full and sliced widths, both forms, and the torch path through the
re-pointed weight views: 517 of 517 cases bit-identical. Timing on an H100, N = 152,064:

| rows | torch path | one-kernel epilogue | + stacked GEMM | stock bf16 head |
|---|---|---|---|---|
| 1 | 0.943 ms | 0.889 | 0.496 | 0.364 |
| 64 | 1.697 | 0.980 | 0.640 | 0.381 |
| 256 | 4.614 | 1.934 | 1.956 | 0.437 |

At the rows a decode step samples (one per sequence, 64 at saturated chat) the head goes
from 4.5x stock's to 1.7x; at 256 rows the int8 GEMM itself is the floor (cuBLASLt's sm80
int8 kernels, SPEC 7.10). The hook wraps t16l's `_get_logits` and falls through to it for a
bf16 head, a SmoothQuant variant, an embedding bias or a non-unit logit scale. In the engine
(7B, full graphs, `results/e6_b2pp_lmfuse{1,0}.json`) the generated ids with the fused head
equal the ids with it off, and E6 full == eager holds. Qwen3-8B's per-head q_norm / k_norm
have no slot in the fused `lin_rope`; the fused-layer guard now keys on those attributes and
leaves such layers on the unfused exact path (found by Group A, SPEC 7.16).

In the engine, same session, both arms (`results/single_noinductor_b2pp.json`, stock arm
verified un-armed: the harness arms the exact stack whenever `T16L_ARM=1` is in the
environment even for `ONLY=stock`, so the first stock run of this cell was polluted and
re-run clean): decode 1.016 / 1.005 / 0.928x at batch 1 / 8 / 32, TTFT 0.907x at 128
tokens, 1.136x at 2k, 1.186x at 8k, 1.061x at 32k. Saturated serving
(`*_repHNI_b2pp_*.json`): W1 0.821x, W2 0.895x, W3 0.996x against 0.848 / 0.891 / 0.984
before the fused head - NO measurable change at W1, whose run-to-run spread across this
document's repeats is +-0.03 (0.82 to 0.86 for the same build). The lm_head was a real
cost (0.6 ms of a ~14 ms saturated step) but not the W1 deficit; that stays with the
attention at short units (B2', B2).

**B2', the glue kernels' latency at small row counts** (`lssa_runtime/t16m_glue_prof.py`,
kernel durations from the torch profiler, `results/t16m_glue_prof_{pre,b2p}.json`;
CUDA-event timing carries a ~10 us launch floor and is useless here). At 64 rows:
`normquant` 4.74 us against stock's `rms_norm` 1.96; `rowquant` 2.09; `r4quant` 9.53
against `act_and_mul` 4.64; and the fused `deqsilur4` 25.9 us - the largest single glue
cost, 28 launches per step. Two bit-identical changes were gated (514 / 514 at both
widths, `results/t16m_kernels_b2p_*.json`): the block reductions fold their per-warp
partials with shuffles (two barriers instead of three) and the R4 kernels run 32 warps
per row (3 blocks each instead of 5). Effect: `deqsilur4` 25.9 -> 23.3 us at 64 rows
(59.0 -> 51.4 at 256), `normquant` unchanged (4.74 -> 4.75), `r4quant` 9.5 -> 10.0. The
reductions were not the latency. Staging the SiLU fraction table in shared memory
(`results/t16m_kernels_b2p2_*.json`, 514 / 514) moved `deqsilur4` 23.3 -> 22.2 us: not the
gather either. What remains at small M is the per-thread instruction chain itself (24
elements per thread through deq, SiLU with its correctly rounded divide, the Hadamard
stages and the quantiser), so the third change spreads one row over FOUR CTAs as a Hopper
thread-block cluster (`r4_cl_k<DEQ>`, `results/t16m_kernels_b2p3_*.json` and `_b2p4_`,
514 / 514 at both widths): each CTA owns a quarter of the row's 256-blocks and the row max
crosses the cluster through distributed shared memory; the per-element arithmetic is the
same code and the max is order-free. Durations (`t16m_glue_prof_b2p4.json`):

| rows | `deqsilur4` one CTA per row | cluster of 4 | `r4quant` one CTA | cluster of 4 |
|---|---|---|---|---|
| 1 | 19.8 us | 9.2 | 8.2 | 5.8 |
| 16 | 20.6 | 9.7 | 8.6 | 6.2 |
| 32 | ~21 | 18.2 | ~9.9 | 11.3 |
| 64 | 22.3 | 27.3 | 10.3 | 16.8 |
| 256 | 49.0 | 81.9 | 22.7 | 49.4 |

The cluster form wins by 2.2x up to 16 rows and loses from 32, where M x 4 CTAs of 1,024
threads no longer fit one wave, so the launcher routes by row count: cluster up to
`T16M_R4CL_MAXM` (16) rows on sm_90 and above, the one-CTA kernel otherwise and on older
parts (`T16M_R4CL=0` disables it). That is the single-stream regime (batch 1 to 16), where
the fused MLP's glue drops from 20 to 9 us per layer; saturated serving (64 rows) is
unchanged by it, and its remaining glue cost is the one-CTA chain at 22 us. In the engine
(`results/single_noinductor_b2p.json`, E6 full == eager with ids equal to the b2pp run's,
`e6_b2p_*.json`) the 7B single-stream cells do not move: decode 1.020 / 1.007 / 0.920x
against 1.016 / 1.005 / 0.928 before, TTFT 1.14 / 1.18 / 1.05x at 2k / 8k / 32k. The 11 us
per layer the kernel gives back at batch 1 is inside the graph's launch pipelining, not on
the step's critical path; the change is kept for what it is (a shorter kernel, gated) and
the single-stream cells stay where the decode kernel and the GEMM route put them.
### 7.18 Same fingerprint across hardware

`verify_rows.py` replays the dumped rows of one H100 engine run (`e2e_lssab_eager.json.pt`,
268 MB: q8 / k8 / v8, exponents, bf16 outputs at layers 0 / 13 / 27) with the reference
implementation of the contract - `lssab8_torch` for prompt rows, `lssab_torch` for generated
rows, the role from the engine's own `is_prefilling` - and compares bf16 bit patterns with
no tolerance. Results by device (`results/verify_*.json`):

| Device | prompt rows | generated rows | bad | time |
|---|---|---|---|---|
| RTX 4090, GPU (`verify_gpu.json`) | 370,020 | 2,352 | 0 | 3.2 s |
| RTX 4090 host CPU, x86-64 (`verify_cpu.json`, MAX_STEPS 6) | 370,020 | 168 | 0 | 105.3 s |
| Apple M4 Max CPU, ARM64, torch 2.14 (`verify_apple_cpu.json`, all steps) | 370,020 | 2,352 | 0 | 194.5 s |

The Apple run used the verifier and reference modules exactly as shipped (no CUDA, no
vLLM). The Mac's GPU (Metal, `DEVICE=mps`) cannot run it yet: torch 2.14 has no MPS
implementation of `frexp`, which the pow2-exponent step uses (`verify_apple_mps.log`).
Blackwell: `pod/blackwell_verify.sh` builds the portable kernel with `-arch=native`, runs
the rig (21 families, head groups, both warp counts) against the torch reference and
replays the same dump on the GPU and the host CPU; it needs the dump, `keq_*.pt`, the
reference modules and `t16n_gate.py` / `t13b_gate.py` in `/workspace/p2`. On 2026-09-06
RunPod had no B200 or RTX 5090 to assign (stock "Low"; two requests were withdrawn after
90 minutes unassigned), so that row is open, not failed. A B200 (Blackwell) pod was rented on 2026-09-06 for the same replay plus the
portable decode kernel's rig; its rows follow when the pod boots.

### 7.19 The plugin, phases I and II: manifest, auto-configuration, fingerprint self-test

`docs/PLAN_PLUGIN.md` is the design, `docs/PLUGIN.md` the user guide. A prepared MANIFEST
directory is the unit of deployment: `lockstep.manifest.json` (model and family facts, the
rotation kind and hashes, `tp.world_size`, the sha256 of every file and every tensor),
`constants.safetensors` (the armed constants exactly as `t16l_stack.arm_model` produces
them, exported from the load hook), `keq.pt`, `probe.json` (a fixed 256-token probe with
the engine's per-position prompt log-probs as bf16 bit patterns and its 8 greedy tokens:
the fingerprint) and `prepare_report.json`. `lockstep_prepare.py` writes it in one command.
Serving reads it from ONE variable, `LOCKSTEP_MANIFEST=<dir>`: `lockstep_config.resolve`
derives the kernel set from the GPU (sm 9.x with the FA3 artifacts -> the exact route;
anything else -> the portable kernels, which are contract v1 and therefore NOT the same
bits: recorded as `manifest_compatible=False`, and the self-test decides), the reduction
mode from the TP world and a CUDA-context-free P2P probe (custom all-reduce when every pair
has P2P), the NCCL variable at TP > 1, and the fused-layer set from the family; it fills
the legacy environment variables the modules read at import without overriding any the
user set, and prints one `LOCKSTEP-CONFIG` line with the reasons. `t16l_stack.arm_model`
then imports the constants (no fold, no quantisation at load) after verifying every
tensor's sha256, and refuses a manifest prepared at another world size.
`lockstep_selftest.check_engine` reruns the probe in-process or over HTTP and compares
bits; `pod/lockstep_serve.sh` wraps `vllm serve` with it and kills a server that fails.

Measured on the 7B, one H100 (`results/lockstep_plugin_gate.json`): `lockstep_prepare.py`
with an existing key offset took 77 s and wrote 483 constants (8.73 GB). The gate, second run
(the first found two defects in the gate itself: the probe had been written without the
CUSTOM attention backend, and the E6 harness imported the backend before the manifest's
environment - both fixed):

| item | result |
|---|---|
| self-test, manifest-served engine, full graphs | PASS, 256 of 256 positions bit-equal, 8 of 8 greedy ids |
| E6 ids against the legacy fold-at-load run (`e6_b2p_full.json`) | PASS, 3 prompts x 32 tokens identical |
| one byte of `constants.safetensors` flipped | refused at load: constants sha256 mismatch |
| one log-prob of `probe.json` edited | refused: probe sha256 mismatch |
| served at another TP world size (env and manifest forms) | refused: "manifest prepared at tp=1, serving at tp=2" |
| legacy engine reproduces the probe | PASS (the fingerprint is the model's, not the load path's) |
| engine build | manifest 33.3 s against fold-at-load 19.2 s; of the 33 s, 6.6 s were the file-level sha256 and 9.6 s the 483 per-tensor digests, single-threaded; the per-tensor digests now run in a thread pool and the file digest only under `LOCKSTEP_VERIFY=full` |

`vllm serve` from the manifest (`pod/lockstep_serve.sh`, prefix caching ON): healthy after
61 s, the HTTP self-test PASS twice in a row (the second request hits the prefix cache;
`results/plugin_selftest_http_{1,2}.json`), 1.5 s per check. The pip-installed package
(`pyproject.toml`: the `vllm.general_plugins` entry point and the `lockstep` console script)
serves the same way with no `PYTHONPATH`: `lockstep serve` resolved the configuration,
started `vllm serve`, passed the self-test and answered a completion
(`results/plugin_cli_serve.log`). Two defects found by the wider runs and fixed: the
load-hook export's temporaries collided with vLLM's memory profiling on Llama-3.1-8B and
Qwen3-8B (CUDA OOM at the KV-cache allocation; the export now runs by worker RPC after the
engine is built), and the import expected the live module's unpadded shape where the R4 fold
pads per rank (72B at TP=4: gate_up 14,848 against 14,784; the export now records the
shapes and the import accepts the padding rule). The final gate on the installed package
(`results/lockstep_plugin_gate_final.json`, the runtime with every fix of this section):
7 of 7 PASS, and the manifest engine builds in 18.7 s against 19.0 s for fold-at-load once
the tensor digests are zero-copy and hashed in parallel (an earlier run: 22.4 against 22.1). The 72B at TP=4 (`results/plugin_selftest_72b_tp4.json`,
`results/manifest_qwen25_72b_tp4/`): four rank shards prepared in one command (the driver
hung at teardown after writing them - the known worker-outlives-parent hazard - and was
killed by PID), served from the manifest in full graphs with the custom all-reduce chosen
by the P2P probe, self-test 256 of 256 positions bit-equal, engine build 81 s. The other
families (`results/plugin_selftest_{llama31_8b,qwen3_8b,phi4_mini_4096}.json`): Llama-3.1-8B
and Qwen3-8B prepare and serve with 0 differing bits (engine builds 20.8 s and 24.4 s) once
the prepare engine's sequence budget is capped at 32 (vLLM's default 256 sized the decode
workspaces past one H100). Phi-4-mini exposed a binding the contract had not stated: its
longrope builds a different cos / sin cache from `max_model_len` (short or long factors), so
a manifest prepared at 4,096 and served at 8,192 reproduced nothing (238 of 256 positions)
while the same manifest served at 4,096 is bit-exact. The frozen Q14 table is therefore a
per-deployment constant: `lockstep_prepare.py` records `max_model_len` in the manifest, and
the import compares the served engine's rotary cache with the manifest's table entry by
entry and refuses a mismatch with the cause named. The stacked lm_head GEMM now runs in row
chunks sized by the vocabulary (about 256 MB of int32 per chunk), after Phi-4's 200k
vocabulary at 2,048 profile rows asked for a 3.3 GB temporary. The diagnosis then went one
step further: the served engine's rotary table at 8,192 was IDENTICAL to the manifest's, yet
the outputs differed, because vLLM's `Phi3LongRoPEScaledRotaryEmbedding` is a plain
`nn.Module`, not the `RotaryEmbedding` CustomOp the runtime patches - Phi-4's rope had
never been frozen or routed through the declared op in any earlier run (its `cos_sin_cache`
attribute is a leftover; the module reads `long_short_cos_sin_cache` and adds the
original-length row offset when `max_model_len` exceeds 4,096). The runtime now freezes the
concatenated short/long table on the Q14 grid, replaces the module's forward with the
declared op carrying the row offset, exports and imports that table, applies the offset in
the fused layer, and refuses a served table that differs. Phi-4-mini re-prepared at 8,192
(519 constants, 53 s) and served at 8,192: 256 of 256 positions bit-equal, engine build
17.7 s (`results/plugin_selftest_phi4_mini_lr2_8192.json`); served at 4,096 it is refused
with the cause named ("longrope row offset differs: manifest prepared with offset 4,096,
this engine has 0"). One more contract correction came out of it: the Q14 freeze clamped
the table to [-1, 1], and longrope multiplies cos and sin by 1.19, so the first frozen run
lost 3 points of perplexity (9.1906); the grid now keeps |value| <= 2 (contract v2.1
section 5, the other families' tables are unchanged since theirs never exceed 1) and the
whole-stack perplexity with the frozen longrope is 8.9204 against stock's 8.5276 (+4.6
percent, `results/ppl_phi4_fullk_lr2.log`), the same as with the unfrozen rope: the
frozen table costs nothing, and Phi-4's remaining loss is the int8 linears (SPEC 7.16).

### 7.20 Stage 1: every supported model and GPU, or a clean refusal

`docs/PLAN_STAGE1.md`. The matrix (`results/s1/matrix.json`, `results/s1/`): each model
prepared by `lockstep prepare` (key offset calibrated by the tool), served from its manifest
in full CUDA graphs with the fingerprint compared bit for bit, and its whole-stack
perplexity against stock on the same node (30 windows); the manifest and the checkpoint
deleted afterwards to respect the H100 volume's quota. The GPU matrix: the RTX 4090 (Ada)
resolves the portable route, contract v1, `manifest_compatible=False`, and the plugin
refuses to serve the Hopper manifest (`results/s1/config_rtx4090.txt`); the published
contract vectors recompute with 0 differing digests on the 4090's, the H100's and the
4xH100's host CPUs (`results/s1/contract_vectors_hosts.txt`). The rows follow.

**What the matrix found, in the order it found it.** (1) Qwen3-4B: the fold derived the head
dimension as hidden over heads (80) where the config declares 128 (32 heads x 128 on a
2,560-wide residual), so the v-row slice of the fused qkv weight was off by 2,944 rows; the
fold now reads the config's `head_dim`. (2) Mistral-7B: vLLM's decoder layer passes an extra
keyword (`t_cond`) that the fused layer's forward did not accept; it now takes the extras and
defers to the original forward when one is set. (3) Qwen2.5-1.5B and 3B: the fingerprint
reproduced itself bit for bit while the whole-stack perplexity read 69,030 and 171 against
8.9 and 7.9 - vLLM's Qwen2 makes the tied head THE EMBEDDING MODULE (`self.lm_head =
self.model.embed_tokens`), so the untie that fixed Phi-4 (a shared Parameter on two modules)
did nothing and the final norm's gain and the R1 rotation folded into the embedding table.
The fold now gives such models their own head module. This is the case the fingerprint
cannot see: a wrong fold is reproduced faithfully. Two guards were added because of it: the
per-op identity gate inside `lockstep prepare` (each armed op against the declared reference
on real activations, `report.per_op`) and a stock sanity check (the probe's mean prompt
log-prob against a stock engine on the same ids, within 0.15 nats, `report.stock_check`;
prepare refuses with exit 6 otherwise). (4) The first load-time armed-module scan refused
every model: vLLM's `RotaryEmbedding` holds a helper child (`ApplyRotaryEmb`) that the scan
counted as an unarmed rotary; helper children of an armed module now count as covered.
(5) The negatives refuse at the config, before any download or GPU work, with the reason:
Qwen2.5-0.5B (head dimension 64, hidden 896), Llama-3.2-1B (head dimension 64), the GPTQ
checkpoint (`quantization_config`), Mistral-7B-v0.1 at 8,192 (window 4,096). (6) The two
14B-class models pass at +0.31 (Qwen2.5-14B) and +0.14 percent (Phi-4): the 4.6 percent of
Phi-4-mini is that model's, not the family's. (7) Two models pass EVERY bit-level gate and
are still wrong: Qwen2.5-1.5B (attention only 8.241 against stock 8.247, E1 64,800 prompt
rows and E4v2 12,744 generated rows all exact, whole stack 44,515 with fused or unfused
kernels and 52,966 without rotations; every per-op identity 0 bad; the fused kernels 514 /
514 at its widths) and Mistral-7B-v0.3 (fingerprint exact, whole stack 26.5 against 5.3).
The probe-based stock check caught the first (-1.30 nats) and missed the second (+0.15,
inside the 0.03 to 0.15 spread of correct stacks): a mean log-prob over 256 tokens of a
near-random prompt is not a quality gate. The 30-window whole-stack perplexity against stock
on the same node is, and `lockstep prepare` now runs it (`report.quality`, refuses above 1
percent unless `--no-quality-gate`). The bisect by op on the 1.5B (one op armed at a time, no rotations, 30 windows; stock
8.247): norm 8.242, rope 8.249, SiLU 8.239, logits 8.246, the int8 linears 8.480 (+2.8
percent; the same without the CUTLASS route, 8.472 with the multiply form of the quantiser)
- no single op explains 44,515, and the attention rows replay exact (E1 64,800 and E4v2
12,744 rows, 0 bad; attention only 8.241). The cause was the untie itself: `copy.copy` of
an `nn.Module` shares its `_parameters` dictionary, so assigning the new head weight replaced
the embedding's Parameter as well and the untie was a no-op in the fold AND at import (the
import then freed the head's bf16 weight and found the embedding at shape [0]). The copy now
takes its own parameter, buffer and submodule dictionaries. Mistral-7B (untied; attention
rows exact, attention only 5.177 against 5.173; whole stack 24.06 fused, 24.06 unfused,
23.33 without rotations) is a different defect, bisected next. (8) The Mistral bisect (one op
armed at a time, gains folded, 30 windows; stock 5.17): norm 26.34, rope 5.18, SiLU 5.18,
logits 5.18, the int8 linears 5.21 - the integer norm alone, whose per-op identity gate
passes (the kernel IS its reference). The reference dropped the model's epsilon
(`rmsnorm_contract(x, weight, eps_unused=None)`): the stock op is `x / sqrt(mean(x^2) + eps)`
and Mistral-7B-v0.3's embedding rows have mean square 5.5e-6 to 9.9e-6 (percentiles 10 to 99)
against eps 1e-5, so layer 0's input norm over-scaled every token by 30 to 40 percent.
The same omission costs Llama-3.1-8B 3 to 5 percent at layer 0 (mean square 1.2e-4, eps
1e-5) and Qwen2.5-7B 0.2 to 0.7 (2e-4, eps 1e-6), Phi-4-mini under 1 - which is why only
Mistral collapsed and why every earlier 'whole stack' figure carried a small silent error
of the same kind. CONTRACT v2.2 (`docs/contract/NON_ATTENTION_STACK_v2.2.md` section 4): the
eps enters the integer chain as `ss += floor(fl64(N * eps) * 4^e)` with `e` capped at
`norm_ecap(N * eps)` (the largest e with the term at or below 2^50, so the sum stays exact in
binary64 for the torch reference's frexp); pure power-of-two binary64 arithmetic, the same
integers on every host and device. The reference, `t16l_kernels.cu::dynnorm_k`, the fused
`t16m_kernels.cu::normquant_k` and every armed module (`_t16l_eps` from vLLM's
`variance_epsilon`) take the eps; the kernel check adds rows at and below eps; two published
vectors pin the term (`rmsnorm.eps1e-5.N*`, `rmsnorm.eps1e-6.subeps.N*`; the 17 v2.1
digests are unchanged, eps = 0 is v2.1's op). The manifest's `contract_version` is 2.2 and
the reader refuses a 2.1 manifest with the instruction to re-prepare: the constants are the
same, the fingerprint is not. Every positive row of the matrix is re-run under v2.2 below.
(9) Under v2.2 the per-op gate failed Qwen3-8B (o_proj, one call of 24, 252 of 256 rows) and
Phi-4-mini (the norm, about 1,000 rows) while every other model passed; the dumped failing
call had 8,192 rows - vLLM's MEMORY-PROFILING forward (`max_num_batched_tokens` dummy
tokens over placeholder block tables and uninitialised KV memory), whose attention output
was non-finite in those runs and finite in others (the same command on the H100 NVL, three
times, clean; twice at the harness defaults, NaN at layer 3 both times, with and without
the v2.2 eps). NaN rows split the kernel and the reference: `fmaxf` skips NaN where torch's
`amax` propagates it, so the row scale differs and every element of the row reads as bad.
vLLM discards the profiling output; the checker now scores real forwards only
(`CHK_ACTIVE` after the engine is built), and traces the first non-finite output of every
op and of every attention module (`T16L-NONFINITE`, `DUMP_BAD`). (10) The commitment
replay (section C of `t16l_egates.py`, the CPU recompute of the int8 leaf against the GPU
pack) failed 48 of 48 leaves on Qwen3-4B and Qwen3-8B with the BLAKE3 digests correct:
`leaf_bytes` was the constant 8,256 = 64 heads x 129 (the 7B's 28 + 4 heads, doubled for q
/ k / v / o) while the pack kernel writes 2 (HQ + KVH) (dh + 1) bytes per row - 10,320 for
a 32 / 8-head model (Llama-3.1-8B, Qwen3, Mistral) - so rows overran one another and the
last row wrote 2,064 bytes past the buffer. The leaf size now follows the shapes at the
first commit, the pack guards it, and the kernel check adds the 10,320-byte hash and a
32 / 8-head pack case. Neither (9) nor (10) touches the served arithmetic: the fingerprints
and perplexities above stand; the commitment path (`T16L_HASH`) is exercised by the gate
harness, not by the plugin's serving path.

**The matrix under contract v2.2** (every row re-prepared after the eps change; `lockstep
prepare` with the envelope, the per-op identity gate on real forwards, the stock check, the
fused-kernel gate at the model's widths and the 30-window quality gate on one-GPU rows;
then served from the manifest in full CUDA graphs with the fingerprint compared bit for bit;
then the 30-window whole-stack perplexity against stock on the same node; `results/s1/v22/`,
`results/s1/v22/matrix.json`):

| model | pod, TP | prepare | fingerprint | whole-stack ppl vs stock (v2.1 in brackets) |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct (tied head) | H100 NVL | PASS 224 s | 0 bits | +0.29 (+0.26) |
| Qwen2.5-3B-Instruct (tied) | H100 NVL | PASS 119 s | 0 bits | +0.25 (+0.22) |
| Llama-3.2-3B-Instruct (tied, llama3 rope) | H100 NVL | PASS 131 s | 0 bits | -0.14 (-0.33) |
| Qwen3-4B (q / k norm, tied, head_dim 128 on 2,560) | H100 NVL | PASS 134 s | 0 bits | -0.26 |
| Mistral-7B-Instruct-v0.3 | H100 | PASS 287 s | 0 bits | +0.26 (26.3 -> 5.35: the eps) |
| Qwen2.5-7B-Instruct | H100 | PASS 251 s | 0 bits | +0.29 (+0.26) |
| the same 7B manifest, served on an H200 | H200 | - | 0 bits | +0.29, the perplexities bit-identical to the H100's |
| the same 7B manifest, served on an H100 NVL | H100 NVL | - | 0 bits | +0.29, bit-identical |
| Llama-3.1-8B-Instruct | H100 | PASS 347 s | 0 bits | +0.40 (+0.42) |
| Qwen3-8B | H100 | PASS 428 s | 0 bits | +0.35 (+0.34) |
| Phi-4-mini-instruct (longrope, bound to 8,192) | H100 | gates PASS, quality gate FAIL (exit 8) | 0 bits | **+1.72 (+4.6): outside the band, item (13)** |
| Qwen2.5-14B-Instruct (Paley R1) | H100 | PASS 411 s | 0 bits | +0.29 (+0.31) |
| Phi-4 (14B) | H100 | PASS 388 s | 0 bits | +0.12 (+0.14) |
| Qwen2.5-32B-Instruct | 4xH100, TP=2 | PASS 491 s | 0 bits | +0.10 (+0.12) |
| Qwen3-32B | 4xH100, TP=2 | PASS 469 s | 0 bits | -0.04 (+0.02) |
| Llama-3.1-70B-Instruct | 4xH100, TP=4 | PASS 881 s | 0 bits | **+4.94 (+5.12): outside the band, item (11)** |
| Qwen2.5-72B-Instruct | 4xH100, TP=4 | PASS 873 s | 0 bits | +0.98 (+0.31 at 300 windows under v2.1; inside the band, at its edge) |
| Qwen2.5-0.5B, Llama-3.2-1B, Qwen2.5-7B-GPTQ-Int4, Mistral-7B-v0.1 at 8,192 | any | REFUSED at the config (exit 4), 4 of 4 | - | - |

The GPU matrix: one manifest prepared on the H100 SXM reproduces its fingerprint on the H200
and on the H100 NVL (0 differing bits, and the 30-window perplexities of both arms agree to
the last bit across the three SKUs); the RTX 4090 resolves the portable kernel set and
refuses to serve it; the published vectors recompute on every host CPU. Blackwell: no
capacity was granted in the session. The H200 pod's driver (570, CUDA 12.8) predates the
venv's CUDA 13 torch; `cuda-compat-13-0` with `LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat`
runs it unchanged (kernel check 39 / 39, fused kernels 514 / 514 there). The kernel check
(`t16l_kcheck.py`) stands at 39 / 39 on the H100 and the H100 NVL after this session: the
six earlier failures were the harness's own dequant references still written in the v2
order (the served path, `linear_vs_golden`, always passed) and the fused CUTLASS bias record
is informational by design.

(11) Llama-3.1-70B at TP=4 is the one positive outside the band: +4.94 percent (+5.12 under
v2.1), with every bit-level gate passing and the fingerprint reproduced from four shards.
The v2.1 attribution on the same node (`results/s1/diag/`): stock 2.692, attention only
2.769 (+2.9 percent), the whole stack 2.883 (+7.1), the stack without rotations 3.422 - so
about 3 points are the attention rule itself on this model and 2 the int8 linears, where
the 8B of the same family costs 0.4 in total. The tensor-parallel prepare SKIPS the stock
check and the quality gate (`"reason": "tp=4"`: both arms would need a second engine on the
same four GPUs), which is why the row passed `lockstep prepare`; the gate must run at TP,
and the attention cost on the 70B (its key offset calibration, the block exponent on 64 query
heads over 8 kv heads) is the open item. (12) Phi-4-mini's manifest is bound to the 8,192
context it was prepared at (the longrope offset, section 5 of the contract); the perplexity
harness opened its engine at 2,048, the import refused - as designed - and the prepare's
internal quality gate recorded ERROR and still returned PASS. `lockstep_config --env` now
exports `T16H_MAXLEN`, the harness opens at it, the gate runs both arms at the prepared
length, and an arm that produces no result fails the prepare. With both arms at 8,192 the
row reads +1.72 percent (stock 8.370, Lockstep 8.514; +4.6 under v2.1, so the eps carried 3
of its points), and `lockstep prepare` refuses it at the 1 percent tolerance (exit 8, the
manifest removed) - the fingerprint row above is the prepare of the same constants before the
harness fix. (13) Phi-4-mini therefore joins the 70B as a model the contract serves exactly
but degrades beyond the band: its outlier activations under the per-token int8 linears
(README section 9); `--quality-tolerance 2` prepares it, deliberately.

### 7.21 Stage 1b: the two rows outside the band, attributed

`docs/ROADMAP.md` Stage 1b; results in `results/s1b/`. Every arm is a 30-window whole-stack
perplexity against stock on the same node, at the model's bound context length.

**Phi-4-mini (+1.72 under v2.2), one op armed at a time** (no rotations except the `_rot`
arm; every armed arm carries the exact attention, which alone is +0.38, so the per-op
numbers below are attention plus that op):

| arm | ppl | vs stock 8.370 | the op alone (attention subtracted) |
|---|---|---|---|
| attention only, with the key offset | 8.402 | +0.38 | +0.38 |
| attention only, no key offset | 8.423 | +0.63 | the key offset is worth 0.25 |
| norm | 8.398 | +0.33 | ~0 |
| rope (frozen Q14 longrope table) | 8.409 | +0.47 | +0.09 |
| SiLU | 8.396 | +0.31 | ~0 |
| logits (a15w15 head) | 8.399 | +0.34 | ~0 |
| the int8 linears, no rotations | 8.819 | +5.37 | +5.0 |
| the int8 linears with R1 / R4 | 8.524 | +1.84 | +1.46 |
| the whole stack, fused | 8.514 | +1.72 | |

The int8 linears are the row: 1.5 of the 1.7 points, the rotations already taking 3.5 off.
The activation census (`lssa_runtime/activation_census.py`, `results/s1b/census_*`) says
Phi-4-mini's linear inputs quantise no worse per token than Qwen2.5-7B's (down_proj relative
RMS error 0.053 against 0.087; dynamic range 43 against 75), and Qwen2.5-7B serves at +0.29 -
so the loss is not the activation outliers of the earlier hypothesis. The fake-quant
attribution (`lssa_runtime/linear_fakequant.py`: the HF model with one kind of linear
quantised one way at a time, plus the remedies) decides between the activations and the
weights and tests two-plane 15-bit activations, the smoothing fold and the Hadamard; its
rows follow (`results/s1b/fq_*.json`; HF eager, 30 windows, the model's own bf16 as the base):

| linears quantised as | Phi-4-mini (bf16 8.077) | Qwen2.5-7B (bf16 5.744) |
|---|---|---|
| int8 weights only, per channel | +0.03 | +0.18 |
| int8 activations only, per token | +3.18 | +1.95 |
| both | +3.18 | +2.22 |
| both, qkv only / o only | +0.51 / -0.05 | -0.45 / -0.12 |
| both, gate_up only / down only | +1.94 / +2.22 | +0.96 / +1.84 |
| both, with a 256-block Hadamard on the input (the R4 rule on every linear) | +0.74 | -0.08 |
| both, with the smoothing fold alone (alpha 0.5) | +2.17 | +1.16 |
| both, Hadamard and the smoothing fold | **+0.11** | +0.12 |
| two-plane 15-bit activations, int8 weights | +0.15 | +0.11 |
| two-plane 15-bit activations and the Hadamard | +0.26 | -0.08 |

The weights cost nothing; the per-token int8 ACTIVATIONS of the two MLP linears are the
row, on both models - and the Hadamard alone, which is what the contract does today (R4 on
down_proj, R1 on the residual stream), takes Qwen to zero and leaves Phi-4-mini at +0.74:
its MLP inputs keep a per-token structure the rotation does not flatten. Two remedies land
inside the band and are exact by construction: the smoothing fold (a published per-channel
scale, folded into the preceding norm's integer gain or the producing linear's rows and into
the consumer's columns, before the rotation; no runtime cost, no kernel change) and two-plane
15-bit activations (a second int8 GEMM on the low plane, the head's `a15` rule). Contract v2.3
takes the smoothing fold (1b.3); `a15` stays the fallback for a model the fold does not
reach.

Where the scale lives (`results/s1b/fq2_*.json`): the residual stream is R1-rotated, so a
scale on the norm-fed linears can only be applied in the rotated basis (through the norm's
integer gain), while down_proj's input has its own basis where the scale goes before R4:

| variant (all with the Hadamard) | Phi-4-mini | Qwen2.5-7B |
|---|---|---|
| no scale (the v2.2 contract) | +0.74 | -0.08 |
| scale in the rotated basis, every linear | +0.48 | +0.36 |
| down_proj only, scale in its own basis then R4 | -0.08 | -0.05 |
| down_proj only, scale in the rotated basis | +0.43 | -0.01 |
| the v2.3 candidate: down in its basis, gate_up and qkv in the rotated basis | +0.17 | +0.22 |

The down_proj scale is neutral or better on both; the rotated-basis scale on the norm-fed
linears is what Phi-4-mini needs and what costs Qwen 0.3. Contract v2.3 therefore applies
the fold by LEVEL: 1 = down_proj (the default), 2 = also gate_up, 3 = also qkv, and
`lockstep prepare` escalates a level only when the quality gate fails, recording the level
in the manifest (`docs/contract/NON_ATTENTION_STACK_v2.3.md` section 6.4; the statistics
from `lockstep_calibrate.py` on a stock engine, 16 windows, reused across levels). Every
level is exact by construction: the scale is a published constant, the served operators and
the 21 vectors are unchanged, only the constants and therefore the fingerprint move.

In the real stack the down_proj scale alone did NOT move Phi-4-mini (+1.76 in the gate
against +1.72 without; the fake-quant row had no Hadamard-only-on-down baseline, and R4
already does that job), cost Qwen2.5-7B 0.4 in the gate (+0.59 against +0.13) and left
Llama-3.1-8B where it was (+0.42 against +0.40). The levels were therefore reordered with
the norm-fed scale first and NONE as the default: level 0 is the v2.2 constants exactly
(a 2.2 manifest is accepted as a 2.3 manifest at level 0, and the probe of a level-0
prepare is compared byte for byte with the v2.2 manifest's below), so no model that passes
at level 0 changes; the first escalation in-process tripped the attention key offset's
per-layer counter ("layer 32 asked for" of 32), and the escalation now re-executes the
prepare in a fresh process with the calibration reused.
Level 0 is v2.2, proven: the Qwen2.5-7B prepared under v2.3 at level 0 carries the same
constants (sha `dc308744cf74825a`) and the same fingerprint file (`probe.json` sha
`2f08f97bb2121ebd`) as the v2.2 matrix row, byte for byte, and serves it with 0 differing
bits at +0.29 (`results/s1b/identity_qwen25_7b_level0.txt`). The first escalation with the
gain-inclusive statistics read +1.72 / +1.65 / +1.53 / +1.59 at levels 0 to 3 on Phi-4-mini
(the mechanics work: a fresh process per level, the calibration reused, the manifest removed
on the final FAIL). With the unit-gain statistics the levels read +1.72 / +1.86 / +1.62 /
+1.64: the smoothing fold does not move Phi-4-mini in the real stack, whatever the basis or
the level, where the fake-quant model promised 0.74 -> 0.17. The fake-quant lacked one thing
the real fold does: the norm gain cannot live in the norm once the residual is rotated, so it
is folded into the weight COLUMNS before R1, and Phi's gains vary widely across channels -
which inflates the per-row int8 weight quantisation after the rotation mixes those columns.
The scan `results/s1b/fq3_*.json` simulates the real fold in fake-quant (x / g, W g, then the
Hadamard) with an alpha scan of the smoothing and the two-plane 15-bit activations as the
fallback (`results/s1b/fq3b_*.json`; the first pass divided by a POSITIVE floor of the gain
and Phi-4-mini has negative gains, so it read 1e15 - the calibration tool had the same bug,
fixed):

| under the real fold (x / g, W g, then the Hadamard) | Phi-4-mini | Qwen2.5-7B |
|---|---|---|
| no gain fold (the earlier fake-quant baseline) | +0.74 | -0.08 |
| the real fold, int8 both | **+1.15** | +0.18 |
| the real fold, int8 weights, 15-bit activations | +0.73 | +0.08 |
| the real fold, smoothing alpha 0.5 / 0.25 / 0.75 / 0 | +0.65 / +0.77 / +5.39 / +1.33 | -0.07 / +0.64 / +0.21 / +0.01 |
| the real fold, 15-bit activations and smoothing 0.5 | +0.43 | -0.16 |

The fold costs 0.4 on Phi-4-mini and 0.26 on Qwen, and under it the WEIGHTS carry 0.73 of
the 1.15: the norm gains, which vary widely across Phi's channels, scale the weight columns
before R1 mixes them, and the per-row int8 weight scale then loses the small columns. No
scale on the activation side reaches the band from there. The remedy is structural: keep
the gain in the norm (its integer `G`, as the contract already allows) and rotate the norm
OUTPUT online with the 256-block Hadamard the down_proj input already uses (R4), so the
residual stream, the embedding, the head and the weights stay in the original basis, the
weights quantise as the unfolded model's (+0.74 -> the "no gain fold" row), and the v2.3
smoothing scale applies in the original basis, where the earlier scan measured +0.11. This
is contract v2.4 ("R1 online"), a per-model choice so models inside the band keep their v2.2
constants (`docs/contract/NON_ATTENTION_STACK_v2.5.md` section 6.5).

**Contract v2.4 on Phi-4-mini** (`results/s1b/manifest_phi4_mini_v24.json`,
`report_phi4_mini_v24.json`): `lockstep prepare --r1 online` climbs the ladder itself -
online / level 0 fails the gate, online / level 1 (the gate_up scale in the original basis,
folded into the norm's integer gain) PASSES at +0.46 percent (stock 8.370, Lockstep 8.409,
30 windows); the manifest records `rotation.kind = online_sylvester_blockdiag256`, `R1 =
"online"`, the level; the served engine reproduces the fingerprint with 0 differing bits
(build 16.9 s) and the whole-stack perplexity against stock on the same node reads +0.55
(8.852 against 8.901), from +1.72 under v2.2 and +4.6 under v2.1. Every op is the same op;
only the arming changed: 32 norms keep their gains, the norm output takes the R4 rotation
before its quant, and one published scale per MLP input channel.

Two defects of the first v2.4 build, both found by the gates: (a) the R1-online flag was read
at import, and `lockstep prepare` sets the arming environment after importing the runtime,
so level 0 of `--r1 online` ran the offline fold (the same perplexity to the last digit as
the offline level 0) while the re-executed level 1 was online; the flag is read at fold and
arm time now. (b) The online branch was inserted inside the R2 block of the v rows and
swallowed the v BIAS's R2 rotation: Qwen2.5-7B (qkv bias) read 47,233 offline while
Phi-4-mini (no bias) was untouched; the tensor diff of the offline export against the v2.2
manifest named exactly the 28 `qkv_proj.bias` tensors and nothing else. The diff of a fresh
export against a known manifest is the fastest instrument this repository has for a fold
regression, and the prepare's stock check (-1.71 nats) had already refused the row.

With both fixed, at R1 online and NO scale (level 0): Qwen2.5-7B prepare PASS (stock check
-0.001 nats, quality gate +0.28), served 0 bits, whole-stack +0.06 (from +0.29 offline);
Phi-4-mini prepare PASS (quality gate +0.59), served 0 bits, whole-stack +0.52 (from +1.72);
and the offline export is byte-identical to the v2.2 manifest again (483 tensors, 0
differing). The structural change alone does the work; the smoothing scale adds nothing
measurable on top (Phi-4-mini +0.55 with it). R1 online is therefore the default of v2.4 and
`lockstep prepare`, the smoothing ladder stays as the escalation, and the matrix is re-run
under it (below). The cost is two small kernels per layer where one fused kernel ran
(the norm output is rotated by the R4 op before its quant, and o_proj's epilogue no longer
fuses the norm); the serving numbers of section 7.13 are re-measured in Stage 2.

**The constants recomputed from the public checkpoint (VERIFIER step 3.1, executed for the
first time).** `lssa_runtime/verify_constants.py` rebuilds a manifest's constants from the
Hugging Face checkpoint on a CPU with no serving engine - the folds and rotations in binary64
in the contract's order, the bf16 roundings, the int8 quantisers, the head planes - and
compares every tensor digest. Qwen2.5-1.5B-Instruct under v2.4 (`results/s1b/
verify_constants_qwen25_1p5b.json`): 483 tensors in the manifest, 482 match byte for byte -
all 112 int8 weights, 113 scales, 57 integer gains, 57 norm scalars, 112 modes, 28 biases,
both head planes and the embedding. The one that does not is the rope Q14 table: it is vLLM's cache held in bf16 and then
frozen on the Q14 grid, and a CPU recomputes it to within the bf16 rounding-boundary entries
where the device's binary32 cos / sin differ in the last bit - 1,263 of 4,194,304 entries on
the 7B (0.03 percent; `results/s1b/rope_diag`). The table is therefore a PUBLISHED constant
verified by its hash, and the verifier reports it apart from the constants proper; the
contract text says so (section 5). The remaining half of "independent"
is a person who is not us running the same tool from `docs/VERIFIER.md`.

**The matrix under R1 ONLINE** (contract v2.4's per-model option, measured on every row to
decide the default; every row re-prepared with every gate inside `lockstep prepare`, served in
full graphs from the manifest, 30-window whole-stack perplexity against stock on the same node;
`results/s1/v24/`; the manifests of record stay the offline rows of 7.20 except where the
ladder escalates):

| model | pod, TP | quality gate | fingerprint | whole-stack ppl vs stock (offline, 7.20) |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | H100 | +0.14 | 0 bits | +0.21 (+0.29) |
| Qwen2.5-3B-Instruct | H100 | +0.12 | 0 bits | +0.11 (+0.25) |
| Llama-3.2-3B-Instruct | H100 | +0.59 | 0 bits | +0.69 (-0.14) |
| Qwen3-4B | H100 | -0.56 | 0 bits | -0.60 (-0.26) |
| Mistral-7B-Instruct-v0.3 | H100 | +0.09 | 0 bits | +0.21 (+0.26) |
| Qwen2.5-7B-Instruct | H100 | +0.28 | 0 bits | +0.06 (+0.29) |
| Llama-3.1-8B-Instruct | H100 | +0.55 | 0 bits | +0.49 (+0.40) |
| Qwen3-8B | H100 | -0.31 | 0 bits | -0.49 (+0.35) |
| Phi-4-mini-instruct | H100 | +0.59 | 0 bits | +0.52 (+1.72, refused) |
| Qwen2.5-14B-Instruct | H100 (prepared at 0.60) | +0.72 | 0 bits | +0.17 (+0.29) |
| Phi-4 (14B) | H100 (prepared at 0.60) | +0.08 | 0 bits | +0.22 (+0.12) |
| Qwen2.5-32B-Instruct | 4xH100, TP=2 | -0.07 | 0 bits | +0.05 (+0.10) |
| Qwen3-32B | 4xH100, TP=2 | +0.22 | 0 bits | +0.09 (-0.04) |
| Qwen2.5-72B-Instruct | 4xH100, TP=4 | +0.92 | 0 bits | +1.15 (+0.98) |
| Llama-3.1-70B-Instruct | 4xH100, TP=4 | not run (gate skipped) | 0 bits at 256 tokens | worse than stock at every real-text length, see below (+4.94, refused) |

Of the fifteen positives, ten improve, four lose 0.1 to 0.8 and stay inside the band
(Llama-3.2-3B, Llama-3.1-8B, Phi-4, the 72B), and the 70B fails; the 72B sits at the edge of the band
(+0.92 in the gate, +1.15 on the 30-window harness) where v2.2 read +0.98 - the online
rotation of an 8,192-wide norm output at TP=4 costs it a little, and it is the one model on
which v2.4 is not a gain. Phi-4-mini, refused under v2.2, passes.
The two 14B rows first failed inside the prepare engine with an out-of-memory: the online
rotation of a weight's input columns was applied to the whole fp64 matrix at once (a 14B's
gate_up is 1.1 GB in fp64, twice), where the offline rotation had always run in 8,192-row
chunks; the online rotation now runs in place by row chunks - and the rows still failed at 0.85
and 0.80, at vLLM's KV-cache allocation after arming, not in the fold: the online path's
profiling forward keeps more activation memory (the norm output and its rotated int8 copy per
layer), and vLLM's budget at those fractions leaves the allocation short. The same 14B
prepares at 0.60 online and at 0.85 offline; the online rows are measured at 0.60 (a
prepare-time setting, not a property of the manifest).

**Why the online 70B fails, and the decision.** Served from its online manifest at TP=4 the
70B reproduces the fingerprint with 0 differing bits and reads 33.6 on the 30-window harness.
The bisect by prompt length on REAL text (`results/s1b/lenbisect_*.log`, mean prompt log-prob
of one wikitext window; stock / Lockstep online):

| tokens | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|
| stock | -2.625 | -2.626 | -2.318 | -1.478 | -1.009 | -0.824 |
| online | -3.484 | -3.225 | -2.684 | -2.006 | -1.611 | -2.112 |

Worse at every length, from 64 tokens, by 0.4 to 1.3 nats - while the synthetic probe (256
tokens of a formula) matched stock within 0.003 nats. The variable is content, not length:
real text carries massive activations (single channels in the thousands on the first and
delimiter tokens), and under R1 online the integer norm quantises the UNROTATED residual row
in 15 bits from the row's amax, so a 2,000-scale channel leaves the ordinary channels a few
levels each. The offline residual rotation of v2.2 spreads that channel over all 8,192
before the norm sees it - that, not the gain fold, is what R1 was buying on Llama; the 70B's
gains are all at most 1.0 (`model.norm.G`), so norm saturation is ruled out. The same effect
is the 0.1 to 0.8 the two Llama rows lose under online, while Qwen, Mistral and Phi, whose
norm inputs are tamer, gain. DECISION: offline R1 stays the default and the manifests of
record are the v2.2 rows; the ladder escalates to online on a quality-gate failure (Phi-4-mini:
offline/0 refused, online/0 +0.52, accepted); online is a per-model option recorded in the
manifest. The clean resolution - an integer norm whose row quantisation survives massive
activations without the residual rotation (a wider input grid, or the rotation applied
inside the norm) - is a Stage 2 contract item. The ladder accepted, end to end, with the
default settings on Phi-4-mini (`results/s1/v24/h100/manifest_phi4_mini.json`): offline / 0
refused at +1.72, online / 0 passed at +0.59 in a fresh process, the manifest records
`rotation.R1 = "online"`, the served engine reproduces its fingerprint with 0 differing bits
and reads +0.52 on the 30-window harness; 504 s in all.

**Llama-3.1-70B at TP=4 (1b.1), the per-op arms** (stock 2.6915; every armed arm carries the
exact attention, +2.89): norm +2.95, rope +3.03, SiLU +2.97, logits +2.98 - all inside 0.15
of attention alone, so the norm, the rope table, the SiLU table and the head cost nothing on
the 70B; the whole fused stack +6.20 at the 8,192 bound (the matrix row read +4.94 at the
harness's 2,048). The int8 linears are therefore the other 3 points, as on Phi-4-mini, and
the v2.3 fold is the candidate there too. The linear arms with the bf16 weights freed (the
first pass ran out of memory at TP=4 with them retained): the int8 linears without
rotations 3.026 (+12.4), with R1 / R4 2.862 (+6.3, of which attention 2.9, so the linears
3.3), the whole fused stack without rotations 3.016 (+12.1). On the 70B the rotations take
6 points off the linears and 3.3 remain, twice Phi-4-mini's 1.5 - the same per-token int8
activation failure at 8,192 wide and 80 deep. The attention rule's 2.9 stays the open item
of Stage 2; the 70B's v2.3 escalation below can address the linears only.

**The gates at tensor parallelism (1b.4).** `lockstep prepare` ran the stock check and the
quality gate only at TP=1 (`"reason": "tp=4"`), which is how the 70B passed at +4.9. Both now
run at every world size: the stock engine from a guarded script file (vLLM's workers
re-import the driver's main module) and the perplexity harness behind `pymain.py`, and the
fused-kernel gate at the RANK's widths (the global widths failed at TP=2 on Qwen3-32B: 64
query heads on one GPU). Acceptance on the 4xH100: Qwen3-32B at TP=2, stock check PASS
(-0.010 nats), quality gate PASS (+0.16 percent, 30 windows, stock 6.263 against 6.273),
fingerprint 0 bits, whole-stack -0.04, and with the kernel gate at the rank's widths (5120, 5120,
12800, 32 heads, 4 kv heads) 574 of 574 cases. Llama-3.1-70B at TP=4 (the manifest written to
the pod's container disk: the volume's quota counts MooseFS trash, and the 263 GB checkpoint
beside four rank shards tripped it twice): stock check PASS (-0.001 nats), kernel gate 514 of
514 at the rank's widths (8192, 2560, 7168, 16 heads, 2 kv heads), quality gate FAIL +6.20 at
the 8,192 bound (stock 2.6915, Lockstep 2.8585; +4.94 at the harness's 2,048), fingerprint 0
bits - the verdict the gate exists for. Its level-1 escalation then failed inside
`lockstep_calibrate.py` at TP=4: the tool's RPC functions travel to the workers by value
(cloudpickle, for functions of the driver's main module), so the module global the install
wrote was not the copy the collect read ("KeyError: calib" reproduced on Qwen3-32B at TP=2);
the state now lives on the worker object and the calibration runs at TP=2 in 49 s. The
smoothing levels cannot reach the attention rule's 2.9 in any case.

**Blackwell (1b.5, `results/s1b/b200/`).** A B200 (sm 10.0, driver 595, torch 2.11 cu130)
was granted for the first time. The portable decode kernel built for it passes the rig, the
H100 engine dump replays with 0 differing bits on its GPU (370,020 prompt rows, 2,352
generated rows, 1.8 s) and on its host CPU (181 s, sampled steps); the config resolves the
portable kernel set and contract v1 on it; and the plugin now REFUSES the Hopper manifest at
load ("this GPU (NVIDIA B200, sm (10, 0)) cannot reproduce a Hopper manifest:
kernel_set=portable"). Before this session it did not: `manifest_compatible=False` was a
reason, not a refusal, and the plugin went on to arm and died building the Hopper kernels -
the same outcome by accident on the RTX 4090, which had no portable `.so`. A GPU that
cannot reproduce a manifest refuses in strict mode now, by rule.

**Llama-3.1-70B at TP=4 (1b.1), the first arms** (stock 2.6915 at the 8,192 bound): attention
only +2.89, attention only without the key offset +3.10 - the key offset is worth 0.2 and
the attention rule itself carries the 3 points on this model, against 0.38 on Phi-4-mini and
about 0.3 on the 8B of the same family. There is no per-layer attention switch (the paged
cache holds the contract's int8 layout, so a stock layer would need its own cache), so the
depth hypothesis - 80 layers of the same per-layer error against 32, on a model whose stock
perplexity is 2.7 - stays a hypothesis; the remedies (a finer block exponent, a wider table)
are kernel changes for Stage 2. The per-op arms follow.

### 7.22 Stage 2, item 1: the norm-fed linears under R1 online, and the declared outlier channels (contract v2.5)

The Stage 1b decision (7.21) left R1 online as a per-model option because the two Llama rows
lost under it and Llama-3.1-70B failed outright (34 against 2.69). The attribution there -
the integer norm's 15-bit row quantisation of the unrotated residual - was a hypothesis, and
Stage 2 opened by measuring it. It was WRONG.

**The norm alone costs nothing** (`lssa_runtime/norm_probe.py`, `results/s2/norm/norm_*.json`:
the HF model in bf16, EVERY RMSNorm replaced by the contract op with the model's own gain on
the unrotated residual - the R1-online situation - 30 windows of 1,024, stride 512, scored
on the second half; a census of the norm inputs on one real-text window):

| model | bf16 | 15-bit grid, int epilogue (v2.4) | 24-bit grid | 24-bit + float epilogue | rows with amax / rms > 64 | max amax / rms |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 6.2754 | -0.04% | -0.05% | -0.06% | 0 | 48 |
| Qwen2.5-7B | 5.7445 | -0.04% | -0.02% | -0.05% | 0 | 53 |
| Phi-4-mini | 8.0766 | -0.09% | -0.05% | -0.05% | 0 | 51 |

A row's amax / rms never exceeds 53 on real text, so the 15-bit grid leaves the ordinary
channel 600 levels, and no wider grid or epilogue moves perplexity by more than the noise.
The per-row relative error of the op against a binary64 norm is 1.7e-3 mean (the bf16
output rounding) at every setting. The norm is not the item.

**The int8 linears are** (`lssa_runtime/linear_fakequant2.py`, `results/s2/norm/fq2_*.json`:
the HF model with every linear fake-quantised per the contract - per-token int8 activations,
per-output-channel int8 weights - under the rotation each mode gives the linear's input:
ONLINE = 256-block on the norm output for qkv / gate_up, 128 per head for o_proj, 256 for
down (the served v2.4 arithmetic); OFFLINE = the unit-gain norm output rotated full-width
with the gain in the weight for qkv / gate_up (the served v2.2 fold); OUTL k = the k input
channels of largest calibration amax removed from the row before rotation and quantisation,
their exact bf16 product added back):

| mode | Llama-3.1-8B (6.2754) | Llama-3.1-70B (2.6506) |
|---|---|---|
| int8, no rotation | +1.64% | |
| int8, online (v2.4) | +0.10% | **+66.41%** |
| int8, offline (v2.2) | +0.23% | +1.11% |
| online, qkv only / o only / gate_up only / down only | -0.03 / +0.00 / +0.01 / +0.05 | |
| online + outliers 8 on qkv, gate_up | +0.13% | **+0.94%** |
| online + outliers 8 on down only | -0.01% | +68.65% |
| online + outliers 8 on all four | -0.00% | +0.77% |
| online + outliers 32 on all four | +0.11% | +0.82% |
| online + outliers 2 on all four | +0.12% | |
| offline + outliers 8 on down | +0.15% | +0.74% |
| 15-bit two-plane activations, int8 weights, online | +0.05% | +31.68% |

The emulation reproduces the served failure (+66 percent on the 70B under the v2.4
arithmetic, against +0.10 on the 8B), and it names the remedy: eight declared channels per
norm-fed linear take the 70B from +66 to +0.94, BELOW the offline fold's +1.11, and the same
eight on the 8B read +0.13 (offline +0.23). Thirty-two channels do no better than eight;
outliers on the down_proj input do nothing online and 0.2 to 0.4 of a point on top
(+0.77 with all four kinds; +0.74 for offline with down outliers), a later item. Two-plane
15-bit activations do NOT rescue the online 70B (+31.7): the loss is not activation
precision alone - the same few input columns also dominate the int8 rows of the norm-fed
WEIGHTS once only a 256-block mixes them, and removing the columns from the quantised weight
is the other half of the fix (the weight / activation split is measured in `fq2b_llama70b.json`).
The channels are the same in every layer (8B: 788, 1384, 4062; 70B: 1532, 6857, 7306), the
classic massive-activation channels of Llama-3.

**Contract v2.5** (`docs/contract/NON_ATTENTION_STACK_v2.5.md` section 6.6): per norm-fed
linear, `oidx` (int32 [k], the k channels of largest calibration amax, default k = 8) and
`wo` (bf16 [N, k], the folded weight's columns) are published constants; `r4quant_outl`
returns the row's outlier values exactly and quantises the rest through the 256-block
butterfly; the int8 weight is the block-rotated weight with those columns zeroed; every
consumer epilogue adds `side[n] = sum_j fl32(ho[j]) * fl32(wo[n, j])` (exact products,
sequential binary32 adds) to the bf16 v2.1 value before the bias: `y = bf16(fl32(y8) + side)`.
Exact by construction for any `oidx`; k multiply-adds per output against K for the GEMM.
The constants come from the calibration statistics `lockstep_calibrate.py` already records
(`qkv_raw`, `gate_up_raw`), so `lockstep prepare --r1 online` now runs the calibration at
level 0 too; `--outliers 0` gives the v2.4 arithmetic, and a v2.4 manifest (no `.oidx`) loads
as v2.5 with k = 0. Gates: `t16l_kcheck.py` 45 / 45 (six new: `r4quant_outliers_{rand,wide}`
at widths 3,584 and 8,192, `deq_outlier_side_*` with and without a bias, against
`lockstep_fullstack.w8a8_outlier_linear`), `t16m_gate.py kernels` (the fused `deq_rope`,
`deq_silu`, `deq_silu_r4quant` with the side path on both GEMM routes, `r4quant_outl` in all
three libraries, against the T16l composition and the torch reference), 7 new contract vectors
(28; the 21 of v2.1 to v2.4 unchanged), the per-op identity gate's linear checker carries the
side path, `verify_constants.py` recomputes `wo` and the zeroed-column `w8` from the checkpoint
and the manifest's `oidx`.

**Served, first pass** (`results/s2/matrix/matrix.md`, R1 online, k = 8, R2 on): the six H100
rows prepare, reproduce their fingerprints with 0 differing bits in full CUDA graphs and read
Llama-3.1-8B +0.46, Llama-3.2-3B +0.74, Phi-4-mini +0.60, Qwen2.5-7B +0.08, Mistral-7B +0.23,
Qwen3-8B -0.35 against stock (v2.4 online: +0.49, +0.69, +0.52, +0.06, +0.21, -0.49). The
70B at TP=4 FAILS its quality gate at +10.24 percent (2.9671 against 2.6915) - from +1169
under v2.4, still ten times the emulation's +0.94.

**The served gap, bisected on the 70B** (`results/s2/matrix/h100x4/attrib_llama70b.md`; 30
windows, every arm with the exact attention, whose own cost is +2.89 - re-measured this
session, 2.7692, with stock 2.6915):

| arm (what is int8 / armed) | R2 on | R2 off |
|---|---|---|
| v2.5 online, all ops, fused | 2.9671 (+10.24) | pending |
| v2.5 online, all ops, unfused torch path | 2.9671 (identical) | |
| v2.5 online, linears only | 2.9696 | 2.7944 (+3.82) |
| online, qkv only / gate_up only / o_proj only / down only | 2.9700 / 3.0008 / 2.9963 / 2.9775 | o_proj only 2.7675 (+2.82) |
| v2.4 online (no outliers), linears only | 30.954 | pending |
| offline, all ops, fused | 2.8585 (+6.20) | pending |
| offline, linears only / qkv+gate_up / o+down | 2.8620 / 2.8548 / 2.8996 | |

The per-kind arms are the tell: whichever single linear kind is quantised, every online arm
sits at +10 to +11.5 and every offline arm at +6 to +8, and quantising all four is no worse
than one. The cost is not the quantisation of any linear; it is a constant that the FOLD
introduces whenever anything is armed, and the one fold both modes share that the emulation
never applied is R2 - the per-head Sylvester rotation of the value rows of qkv and the columns
of o_proj (section 6.2). Without it the online linears cost 0.9 over the attention, the
emulation's number, and o_proj int8 on its own is free. R2 rotates a value vector whose energy
sits in a few components into 128 components of equal magnitude; the KV cache stores it in
bf16, so the small components that carried the token's information are rounded at the scale
of the large ones, and the sink token's value - attended to by every row - carries that noise
into every output. The rotation is exact in fp64 and useless for the thing it was built for:
o_proj's per-token int8 without it costs nothing on the 70B. Stage 1b's "int8 linears +3.3
on the 70B" (7.21) was R2. Tensor parallelism is not a factor: the emulation with the served
row-parallel arithmetic (per-rank slices, bf16 partials folded in rank order, `TPEMU=4`) reads
+0.78 online and +0.79 offline; nor is the BOS token (the emulation with BOS-prefixed windows:
v2.4 online +186, v2.5 +0.63, offline +0.97). `lockstep prepare --r2 off` (the manifest records
`family.rotation.R2`) is the remedy, measured:

| arm on the 70B | R2 on | R2 off |
|---|---|---|
| v2.5 online, all ops, fused | 2.9671 (+10.24) | 2.8101 (+4.41) |
| offline, all ops, fused | 2.8585 (+6.20) | 2.8012 (+4.08) |
| v2.4 online (no outliers), linears only | 30.954 | 4.4163 (+64.1, the emulation's +66) |
| online, qkv only, 32 channels | | 2.7648 (+2.72, below the attention alone) |

Without R2 the two R1 modes are within 0.3 of each other on the 70B and the emulation is
faithful to the last point (+64 served against +66 emulated for v2.4), which settles the
attribution: the outlier channels are necessary (v2.4 without them is +64), sufficient for the
linears (about 0.9 over the attention), and the remaining 2.9 is the attention rule alone.

**Served, second pass, both R2 settings** (`results/s2/matrix/matrix.md`; R1 online, k = 8;
every row prepare PASS, per-op identity 0 bad, kernel gate 553 / 553, fingerprint 0 differing
bits in full CUDA graphs):

| model | R2 on | R2 off | v2.2 offline (R2 on) |
|---|---|---|---|
| Llama-3.1-8B | +0.46 | +0.30 | +0.40 |
| Llama-3.2-3B | +0.74 | -0.57 | -0.14 |
| Phi-4-mini | +0.60 | +0.85 | refused at +1.72 |
| Qwen2.5-7B | +0.08 | +0.06 | +0.29 |
| Mistral-7B | +0.23 | +0.21 | +0.21 |
| Qwen3-8B | -0.35 | -0.51 | +0.07 |
| Qwen2.5-72B, TP=4 | +1.15 (v2.4) | +0.89 (gate +0.70) | +0.98 |
| Llama-3.1-70B, TP=4 | +10.24 (gate FAIL) | +3.71 (served, gate skipped) | +4.94 / +6.20 |

R2 hurts every Llama row and is neutral on Qwen and Mistral; Phi-4-mini alone reads 0.25
better with it, inside its band. DECISIONS: `--r2 off` is the default (`--r2 on` stays a
per-model option, recorded in the manifest; choosing it inside the ladder is item 2.0c);
`--r1 online` with `--outliers 8` is the default, since it wins or ties the offline fold on
every measured row and keeps the embedding, the head and the residual in the checkpoint's
basis; the ladder is online/0 -> online/1 -> online/2 -> online/3 -> offline/0. The 70B stays
outside the band by the attention rule's +2.89 (item 2.0b) plus about 1 point of int8 linears
that the side path on o_proj / down_proj would halve (item 2.0c). Qwen2.5-72B at TP=4 under the
new default: quality gate +0.70, served +0.89 with 0 differing bits (its v2.4-online row with
R2 read +1.15, the v2.2 row +0.98), which confirms the default on the largest Qwen.

Two tool notes from this pass: `norm_probe.py` returns invalid output under `device_map=auto`
(the 70B run; the single-GPU runs are the ones cited) and the served norm on the unrotated
70B residual was already measured in 7.21 at +0.06 over the attention; and arms that set
`T16L_ROT=` (no rotations) are invalid under R1 online, because the fold rotates the
norm-fed weight columns for the online op whether or not that linear is armed
(`T16L_LIN_KINDS`, a diagnostic knob the export refuses, arms a subset of kinds correctly).

### 7.23 Stage 2, item 2.0b: the attention rule on deep models, bisected in the torch reference

The 70B's remaining +2.89 (7.22) is the exact attention alone. `lssa_runtime/attn_bisect.py`
(the T12b driver generalised to Llama and to `device_map=auto`; the golden references are
imported, not modified; diagnostic arms that are not contract candidates are built in the
driver with exact softmax and named `DIAG`) on Llama-3.1-70B (4xH100, HF eager, 30 windows
of 1,024, stride 512) and Llama-3.1-8B (the control, one H100), `results/s2/attn/`:

| arm | what is quantised | 70B (E = 2.6911) | 8B (E = 6.2678) |
|---|---|---|---|
| L8-RN-W9 | the served contract (7-bit q per row, k / v per 128-key block, 8-bit block-local weights, key offset) | +2.83 (served attention-only arm: +2.89) | +0.26 (served: about +0.3) |
| L16 | 16-bit block-local weights, same fold | +2.80 | +0.21 |
| L8-RN-KB64 | 64-key blocks | +2.80 | +0.25 |
| B128 / Bk128 / Bv128 | the pre-B8 rule (n=13, w=15); K-only / V-only block exponents | +2.72 / +3.06 / +2.86 | +0.22 / +0.21 / +0.25 |
| L8-RN-W9-nomu | no key offset | +3.03 | +0.22 |
| DIAG q7row | q, k, v at 7 bits PER ROW, exact softmax | +0.24 | +0.15 |
| DIAG kv7blk-k | k alone at 7 bits per 128-key block, exact softmax | +0.17 | +0.01 |
| DIAG kv7blk-v | **v alone at 7 bits per 128-key block**, exact softmax | **+2.67** | +0.19 |
| DIAG q15 | q, k, v at 15 bits per row | +0.05 | (the first run's arm had an off-by-one grid; corrected) |

The cause is one operand and one granularity: the VALUE vectors quantised to 7 bits with an
exponent shared by a 128-key block. On the 70B a block holds a few value vectors far larger
than the rest (the sink and delimiter tokens), the shared exponent is theirs, and the other
keys' values keep one or two bits; per row the same 7 bits cost 0.24 for all three operands.
The keys do not have the problem (+0.17 at the same granularity), the weight table does not
(16-bit: +2.80), the fold's block size does not (64: +2.80), the key offset is worth 0.2 in
the other direction (+3.03 without it). The 8B pays 0.19 for the same reason, at its own scale.

**The multi-GPU "race", found and fixed (P3 of `docs/PLAN_STAGE2.md`).** The torch reference on a
`device_map=auto` model produced a NaN in a block partial at a different window in each of two
runs (28, then 13), passed all 30 windows with per-layer synchronisation, and reproduced to the
last digit under `CUDA_LAUNCH_BLOCKING=1`. `attn_race.py` (per-layer checksums accumulated on
the stream, read only after the forward) showed two non-blocking runs IDENTICAL to each other
and both different from the blocking run at window 1, layer 19 - the first layer on cuda:1 -
with equal inputs and an all-zero output; `attn_race2.py` recomputed the arm on the captured
inputs and got a third value; `attn_race3.py` ran the arm alone on random inputs: identical
three times on cuda:0, an ILLEGAL MEMORY ACCESS on cuda:1, and identical to cuda:0 when GPU 1
is the only visible device. The statement is `torch.ldexp(ones, e)` in the block quantiser:
on this pod's torch 2.11.0+cu130, `torch.ldexp` faults whenever its tensors live on a CUDA
device that is not the process's current device, while `pow(2, e)`, the exponent-field
construction and `ldexp` under a device guard all give the right value (each in a fresh
process: the fault poisons the context). Every other primitive the references use (frexp,
round, index_select, cummax, where, fp64 matmul, int views, shifts) is clean on the second
device. Asynchronous, the fault surfaced as a NaN, a zero block or a wrong value depending on
timing, which is what blocking launches masked. FIX: `contract_common.pow2_t` /
`lockstep_fullstack.pow2_f32 / pow2_f64` build 2^e from the exponent field (the value the
contract text specifies: an exact power of two) and replace every `torch.ldexp` on the live
reference paths (`contract_common`, `lssab_torch`, `lssab8_torch`, `lssab9_torch`,
`verify_arm_g`, `t10_attn`, `lockstep_fullstack`, `t16l_egates`, `vllm_lockstep_backend`); the
numpy goldens are untouched. Gates: `lssab_selftest`, `lssab8_selftest`, `lssab9_selftest`
(torch == numpy bit for bit), the 28 published vectors unchanged, `t16l_kcheck` and
`t16m_gate` (kernels == the torch reference), and a new `ref_device_gate.py` that runs every
reference op on a non-current device and demands bit identity with cuda:0. The 70B arms are
re-run without blocking launches and must reproduce 2.7671. The earlier `norm_probe.py`
failure on the 70B (7.22) was the same fault.

**The value operand's granularity and width** (the same driver, exact softmax and exact q / k,
70B, `results/s2/attn/bisect4_llama70b.jsonl`):

| value quantisation | 70B |
|---|---|
| 7-bit, one exponent per 128 / 64 / 32 / 16 / 8 keys | +2.67 / +2.42 / +2.22 / +2.01 / +1.82 |
| 7-bit, one exponent per key | +0.09 |
| 7-bit per 128 keys, key 0 (the sink) taken out of block 0's exponent and quantised alone | +0.46 |
| 9 / 11 / 15-bit, one exponent per 128 keys | +0.10 / +0.08 / +0.02 |
| 9-bit per 32 keys | +0.05 |

Two more bits of value mantissa at the contract's own block size remove the cost; the sink
key alone is most of it (2.67 -> 0.46 by giving key 0 its own exponent); sub-blocks alone
help slowly (8-key blocks still +1.82), because the large values are not confined to one
block. The 8B pays 0.19 for the same reason at its own scale.

**LSSA-B9, the two options in the torch reference with the real fold** (`lssab9_torch.py`,
the B8 golden untouched; `lssab9_gate.py`: 0 differing bits against B8 with both options off on
four shapes; `results/s2/attn/bisect9*_*.jsonl`):

| arm | rule | 70B | 8B | kernel cost |
|---|---|---|---|---|
| L9 | B8 (both off) | +2.83 (= L8 to the last digit) | +0.26 (= L8) | |
| L9-sink | key 0 in its own value block (B9.1) | **+0.73** | **+0.08** | one rank-1 update per row for block 0 |
| L9-v16 | 15-bit value in two int8 planes (B9.2) | **+0.21** | -0.08 | 2x the P.V product, the V half of the KV cache at 16 bits |
| L9-sink-v16 | both | +0.29 | -0.01 | |

Against the exact-softmax diagnostics the real fold adds about 0.25 on the 70B (the 8-bit
weights and the block chain, which the sink rule does not touch). The whole-stack projection
for the 70B with the int8 linears at about 1 point: near +1.8 with the sink rule, near +1.2
with the two-plane value; under 1 percent needs the two-plane value AND the linears at 32
declared channels (the served qkv-only arm at k = 32 sat below the attention alone, 7.22).
DECISION for the kernel step: B9.1 first (it is nearly free and takes every Llama row
inside its band by a margin: the 8B's attention from 0.26 to 0.08), B9.2 as the per-model
option for deployments that want the 70B under 1 percent and accept the P.V cost, measured
when the kernel exists; the kill line of `docs/PLAN_STAGE2.md` applies to both.

**Performance of the v2.5 default stack** (Qwen2.5-7B, one H100, the Group B recipe:
`t13b_bench.py` single-stream both arms one process each, CUDA graphs; `t14_bench.py`
saturated serving; `results/s2/perf/`):

| single-stream, ratio Lockstep / stock (time; > 1 is faster) | v2.5 default | Group B record (v2.2) |
|---|---|---|
| decode B1 / B8 / B32 | 1.048 / 1.052 / 1.057 | 1.09 / 1.08 / 1.10 |
| TTFT p128 B1 / B8 | 0.749 / 0.894 | (0.82 chat) |
| TTFT p2048 B1 / B8 | 0.923 / 0.964 | 0.936 |
| TTFT p8192 B1 / B4 / B8 | 0.953 / 0.958 / 0.957 | 0.947 |
| TTFT p32640 B1 | 0.918 | 0.892 |
| prefill p128 B32 | 0.958 | |

Single-stream is the record within noise. Saturated serving, whole stack, W1 (ShareGPT,
500 prompts) output throughput against stock in the same session (`results/s2/perf/`):

| Lockstep arm (whole stack, W1) | tok/s | ratio | reference |
|---|---|---|---|
| v2.2 fold (offline R1, R2 on, no outliers), today | 4,250 | 0.788 | the Group B whole-stack record on this pod: 4,485 / 5,465 = 0.821 (b2pp, 09-06), earlier sessions 0.82 to 0.85 |
| R1 online, no outliers (v2.4), today | 4,114 | 0.762 | |
| R1 online, 8 outlier channels (v2.5 default), today | 4,090 | 0.759 | |
| the same, first run of the session (W1 / W2 / W3) | 4,095 / 6,547 / 460 | 0.759 / 0.802 / 0.785 | |

The 0.996x of T16j (7.17) is the ATTENTION-ONLY arm and is not the reference for the whole
stack; the whole stack has read 0.82 to 0.85 on this pod since Group B, and that gap is item
2.1's (the per-launch glue at short context). Against it the v2.5 default costs about 3
percent in saturated serving: the online path's separate rotation-and-quant kernel on the norm
output and the unfused o_proj at TP = 1 (about 3 percent, v2.4 online against the v2.2 fold),
the outlier side path under 1 percent (k = 8 against k = 0); single-stream and decode are
unchanged. Both are kernel fusions without arithmetic change (the block butterfly and quant
inside `normquant_k`; the side path in the fused `lin_norm_quant`), filed under item 2.1.

**2.0c, measured, is neutral:** the side path on o_proj and down_proj (v2.5.1, built and
gated: kcheck 51 / 51, fused gate 586 / 586, 30 vectors, per-op identity 0 bad on all four
kinds, the verifier recomputes the constants) reads Llama-8B +0.38 (v2.5 +0.30), Qwen2.5-7B
+0.07 (+0.06) and Llama-70B at TP=4 +3.70 (+3.71), all with 0 differing bits
(`results/s2/matrix/*/…_v251.json`). The emulation's 0.2 to 0.4 on the 70B did not survive
serving. The default outlier set stays the norm-fed linears; `LOCKSTEP_OUTLIER_KINDS` adds the
other two as a per-model option, recorded in the manifest.
**P1, LSSA-B9.1 in every kernel (2026-09-08; addendum B9.4).** One flag selects the rule
everywhere (`LOCKSTEP_SINK=1`, `lockstep prepare --attn-rule b9.1`, the manifest's `attention`
block; 0 is the B8 kernel bit for bit). Portable prep and decode (`lssa_lssab_t13b.cu`): block
0's row 0 is scanned into its own amax and exponent (`ev0`, per page, written once); the decode
kernels take key 0's u8 weight out of the plane after `l_b` has summed it, form its s32 product
from the transposed V tile and fold it first with `2^-(s + e_v0)`; under split-KV the sink is
slot 0 of the block-partial buffer (a zero-weight pseudo-block folded before block 0, which is
the golden's two-term block without a new rule). The FA3 seam (`patch_b9.py` on the shipped
`b8x + b8eng + b8varlen` stack, build `b9vareng`): the thread holding tile column 0 keeps
`eh[0]`, zeroes it in the u8 A-operand, adds it back for `l`; the fold applies
`f32(eh[0] * v0[c]) * 2^-(s + e_v0)` before the block term, `eh[0]` shared across the quad by
one width-4 shuffle, `v0` and `e_v0` gathered per (batch element, kv head) by the engine glue.
Gates on the H100: `lssab8_gate` on `b9vareng` with the sink off 8 / 8 (the B8 signatures);
`lssab9_seam_gate` 48 / 48 (bf16 output bits, the fp32 bit patterns of O_row and l_row and
NB_row against `lssab9_golden` on 14 shape-by-sink cases, the negative control differing on
every case where `e_v0 != e_v[0]`, determinism, the exact skip, 1k to 4k against
`lssab9_torch`); `t16n_gate` on the base, pipelined and 16-warp decode kernels 21 / 21 with the
sink off (`lssab8_torch`) and 21 / 21 with it on (`lssab9_torch(sink_alone)`); the v1 K-gates
(`t13b_gates`, `t15_kgate` K9 10 / 10) unchanged on the new `.so`. The `b9vareng` build is now
the default FA3 directory when present (`lockstep_config.default_fa3_dir`), one build for both
rules.

**P2, the online path in one launch (2026-09-08).** `normrotquant_k` (`t16m_kernels.cu`)
computes the deq epilogue with its side path, the residual, the integer RMSNorm, the online R1
(the 256-block butterfly in `r4quant_k`'s pinned order, the exact 2^-4), the declared-outlier
split and the int8 quant in ONE CTA per row, in `r4quant_k`'s thread layout; `norm_rot_quant`
and `lin_norm_rot_quant` (the o_proj GEMM + everything after it, at TP = 1) replace the
`norm_quant(want_h) -> r4quant_outl` pair and the `lin_out -> norm_quant -> r4quant` triple
that item 2.1 attributed the v2.5 serving cost to. No arithmetic changes: `t16m_gate` proves
the fusion bit-identical to the gated two-launch path on 600 new cases (N = 3,072 / 3,584 /
4,096 / 8,192, M = 1 to 300, 0 / 8 / 32 outliers, with and without residual, bias and the side
path, both GEMM routes), 1,186 / 1,186 in all. `T16M_NRQ=0` keeps the two-launch form for the
attribution arm.

**B9.1 served (2026-09-08; `results/s2/attn/b9/`).** Llama-3.1-8B on ONE build (`b9vareng` +
the promoted portable `.so`, v2.5 default: R1 online, 8 outliers, R2 off), 30 windows against
stock on the same node: LSSA-B8 +0.30 (the prepare's own gate +0.25), LSSA-B9.1 +0.25 (gate
+0.22), both prepares PASS with the per-op gate and the fingerprint, and the served rows
bit-identical to the rule's torch reference on a dump spin: E1 172,800 prompt rows and E4
33,984 generated rows, 0 bad, against `lssab8_torch(mu)` under B8 and against
`lssab9_torch(sink_alone, mu)` under B9.1 (`gate_e1_llama31_8b_b8.json`, `..._b91.json`).
The manifests record the rule (`attention.rule`, `sink`). Llama-3.1-70B at TP = 4 under B9.1
(k = 8): whole-stack +1.77 percent (2.7391 against 2.6915), from +3.71 under B8; the 1 percent
prepare gate refuses it, as it should, so the row is filed as measured and the manifest is
re-prepared under a 3 percent tolerance for the fingerprint (marked). The torch reference had
predicted +0.73 for the attention alone (B8 +2.83) and the int8 linears about +0.9, and the
served number is their sum. With k = 32 declared outlier channels on the norm-fed linears
(the same build, `--outliers 32`) the 70B reads +1.17 percent (2.7230), fingerprint
reproduced with 0 differing bits, and its served rows at TP = 4 are bit-identical to the
rule's reference on a dump spin (E1 86,400 prompt rows and E4 16,992 generated rows of
rank 0's kv-head slice, 0 bad, against `lssab9_torch(sink_alone, mu)`;
`gate_e1_llama31_70b_b91_k32e.json`): the outlier set buys 0.6 of the remaining points, and the
last 0.2 to 0.7 on the 70B is the value grid (B9.2, +0.21 in the reference), which is the
open P4 decision against its 10 percent prompt-throughput kill line.

**A correction to the single-stream record, and the cost of P1 / P2 as built (2026-09-08).**
The single-stream harness (`t13b_bench.py`) ran its STOCK arm in a shell that still exported
`LOCKSTEP_PREFILL` for the lockstep arm, and the site-packages vLLM plugin's legacy branch then
armed the int8 non-attention stack inside the "stock" process (its log carries `T14-PLUGIN ...
T16l full stack registered`). Every filed single-stream ratio since the plugin was installed
into site-packages was therefore against an ARMED stock: a clean stock (every `LOCKSTEP_*` /
`T16*` variable unset, `results/s2/perf/clean/single_stockclean.json`) reads Qwen2.5-7B
decode 163.9 / 1,280 / 5,047 tok/s at batch 1 / 8 / 32 and TTFT 208 ms at 8k, against the
filed "stock" 162 / 1,264 / 4,566 and 219 ms, and the v2.5 stack of 2026-09-07 against it is
TTFT 0.86 to 0.91 and decode 0.94 / 0.94 / 0.86, not 0.92 to 0.95 and 1.05. The saturated
serving numbers are unaffected (that harness disarms the stock server). The harness now strips
the arming variables in its stock branch.

On that clean footing, the two P1 / P2 builds cost what they cost, one change at a time on
the same day (`results/s2/perf/clean/`): the LSSA-B9.1 rule itself is free (sink on equals
sink off within noise on every shape); the portable `.so` is neutral (the decode kernels
have the same 64 registers and the same 34 / 52 spill bytes before and after); the
`b9vareng` seam costs about 4 percent of TTFT at 8k and 10 percent at 32k against
`b8vareng` even with the sink OFF (0.783 against 0.870 at 32k). The attribution builds
(`patch_b9.py` knob `B9_STATIC`: the sink code compiled away in whole, `off`; the chain's
key-0 predicate only, `chain0`; the fold's sink term only, `fold0`; every build still carrying
the members and parameters) read, at 32k against the clean stock, b8 0.869 (three builds of
it, 0.868 to 0.869), b9 0.781, off 0.818, chain0 0.795, fold0 0.822: the fold's per-block
work (two quad shuffles and the block-0 branch on every block of every row) is 4 of the 10
points and the extra state in a 168-register kernel the other 5; the chain's predicate is
free. The V2 form moves the shuffles under the block-0 branch and, under the shipped SERIAL
step, keeps one `eh0` array instead of two (the pending block's fold runs before the current
block's chain); gated (B8 signatures 8 / 8 with the sink off, the B9 seam gate 48 / 48) it
first read 0.790 / 0.787 at 32k, one point back of ten - and the `B8X_LEAN` register-relief
transform read 0.783, and, decisively, a rebuild of the UNCHANGED B8 seam read 0.821 against
the shipped build's 0.871. The cause was the build, not the rule: the Sep-3 seam was compiled
through ninja with the toolchain FA3's `setup.py` downloads (the nvcc 12.6 front end with ptxas
12.8, which it selects for its measured performance), while every build since ran from a bare
ssh shell without ninja on PATH, and torch's non-ninja path ignores FA3's `PYTORCH_NVCC` and
compiles with the system CUDA 12.4 ptxas. Rebuilt through ninja with the downloaded toolchain
(`mk_var.sh` now puts the venv's ninja on PATH), the same sources read, at 32k against the
clean stock, B8 0.871 (the shipped build 0.871), the B9 seam 0.867 with the sink on and 0.866
with it off (both the V1 and V2 forms), and at 8k 0.896 to 0.900 against 0.903 to 0.905
(`single_seam_S17` to `S24`): LSSA-B9.1 costs the seam under half a point. The `b9vareng`
build on both pods is now the toolchain-correct V2 form (the old one kept as
`fa3_b9vareng_sysnvcc`), and the earlier `b9var*` attribution rows in this section are all
system-ptxas builds compared against a downloaded-toolchain one. The P2 fused kernel as first built launched
1,024 threads per row, which
caps it at 64 registers and spills - TTFT 0.60 to 0.65 and decode 0.68 to 0.75 against
0.78 to 0.87 with the fusion off. The warp-count-templated form (8 / 16 warps for 7B / 70B
rows; gated 1,186 / 1,186 on the 4xH100) removes the spill but still measures below the
two-launch path on the same clean stock: TTFT 0.80 / 0.82 / 0.76 at 2k / 8k / 32k against
0.84 / 0.86 / 0.78, decode 0.78 / 0.77 / 0.72 against 0.86 / 0.84 / 0.78 (`single_b9s1nw.json`
against `single_b9s1nrq0.json`). One CTA per row carrying the integer norm's 64-bit chain on
24 elements per thread plus three block reductions loses to the 512-thread norm kernel
followed by the 4-CTA cluster rotation kernel that the two-launch path uses at small M. The
fused op stays in the library as `T16M_NRQ=1`; the two-launch path is the default, and item
2.1's remedy for those two launches is the cluster form of the fusion, not this one.

**Saturated serving under the two rules (2026-09-08, Qwen2.5-7B, one H100, both stacks in the
same session, the two-launch online path, `results/s2/perf/clean/serve/`).** W1 ShareGPT /
W2 decode-heavy / W3 prefill-heavy, lockstep over stock: the LSSA-B8 configuration
(`b8vareng`, sink off) 0.728 / 0.728 / 0.766; the B9 seam with the sink off 0.703 / 0.734 /
0.750; the LSSA-B9.1 configuration (the B9 seam, sink on) 0.693 / 0.745 / 0.758. The rule
itself is inside the run-to-run band (W2 reads higher with it on); the seam build costs 2 to 3
points on the prefill-heavy workloads, consistent with its 3 to 4 percent of TTFT at 2k to 8k.
Against the 2026-09-07 row (0.759 / 0.802 / 0.785) the same B8 configuration reads 3 to 7
points lower today with the same stock throughput. The same B8 configuration repeated an hour
later reads 0.745 / 0.802 / 0.775, and with the previous portable `.so` (the register-handoff
form, the same 64 registers and spill bytes as the pre-B9 kernel) 0.751 / 0.740 / 0.760: the
run-to-run band of the serving harness on this host today is 4 to 7 points (the host carried
a load average above 30 from other tenants all day), the portable kernels are not the cause,
and every serving difference between the rules and the seams above sits inside that band.
The reproducible signal for the seam is the single-stream 32k row (three `b8vareng` builds
within 0.2 percent of each other).

**The served matrix under LSSA-B9.1, and the default (2026-09-08; `results/s2/attn/b9/matrix/`).**
The prepare's own quality gate, 30 windows against stock on the same node, gate to gate against
the B8 rows of 7.22 (same v2.5 defaults: R1 online, 8 outliers, R2 off):

| model | LSSA-B8 | LSSA-B9.1 |
|---|---|---|
| Llama-3.1-8B | +0.25 | +0.22 |
| Llama-3.2-3B | -0.81 | -0.11 |
| Phi-4-mini | +0.71 | +0.55 |
| Qwen2.5-7B | +0.26 | +0.27 |
| Mistral-7B | +0.20 | +0.09 |
| Qwen3-8B | -0.44 | -0.31 |
| Qwen2.5-72B (TP = 4) | +0.70 | +0.14 |
| Llama-3.1-70B (TP = 4) | +3.71 | +1.77 (k = 8), +1.17 (k = 32) |

Every B9.1 row carries a 0-bit fingerprint. B9.1 wins or ties on every row, by 0.6 on the 72B and by
2 to 2.5 points on the 70B, at under half a point of TTFT once the seam is built with the right
toolchain (above) and nothing measurable in decode or saturated serving; `lockstep prepare`
therefore defaults to `--attn-rule b9.1` from
this date, the manifest names the rule, and the config picks the seam build from it.

**The 70B's last point: LSSA-B9.3 (2026-09-08).** The two-plane value (B9.2) is out on cost
(a second PV wgmma per block); sub-block exponents are out on quality (64-key halves read
+2.73 attention-alone on the 70B, no better than B8's +2.83: the cost is inside the
half-blocks too). The rule that works is per-key: every key on its own grid `e_v + d_j`
(`d_j = min(3, e_j - e_v)`) with the u8 weight shifted right by `d_j` in the chain and `l`
on the unshifted weight (addendum B9.5). Attention alone against bf16, 30 windows:
Llama-3.1-70B B8 +2.83, B9.1 +0.73, B9.3 floor +0.79, B9.3 round-half-up +0.38 (cap 3 or 7
alike; with the sink term too +0.38); Llama-3.1-8B B8 +0.26, B9.1 +0.08, B9.3 floor +0.01,
round-half-up +0.23. The rounding direction is a bias the two models take in opposite
directions, and the full set (attention alone, 30 windows, cap 3 unless stated) is:

| arm (cap, rounding) | Llama-3.1-70B | Llama-3.1-8B |
|---|---|---|
| B8 | +2.83 | +0.26 |
| B9.1 (sink alone) | +0.73 | +0.08 |
| B9.3 cap 3, floor | +0.79 | +0.01 |
| B9.3 cap 3, half up | +0.38 | +0.23 |
| B9.3 cap 3, ties to even | +0.63 | -0.08 |
| B9.3 cap 2, floor / half up / even | +0.81 / +0.64 / +0.69 | +0.02 / +0.14 / -0.01 |
| B9.3 cap 1, half up / even | +1.17 / +1.51 | +0.36 / +0.15 |
| B9.3 cap 7, floor / half up | +2.23 / +0.38 | +0.10 / +0.20 |

Round half up at cap 3 is the 70B's best and ties to even the 8B's, so the rounding is a
per-model manifest parameter (`attention.ksr`, `lockstep prepare --ks-round`), round half up
the default; every form is one integer expression and every form is gated.

The B9.3 seam as built (`b93vareng`: the B9 stack, `B9_KS=1`, the IADD `l` form, the
rounding fixed at build time - a runtime rounding switch had demoted the score fragment to
local memory, 1,184 bytes of stack) is bit-identical to the golden and to `lssab9_torch` on
every gate case (48 / 48 at cap 3 round half up, on both pods), and its cost at 32k against
the clean stock, one change at a time: the IADD `l` form alone 0.858 against the dp4a form's
0.869 (one point); the B9.3 build with the shift off 0.807 and on 0.776 - the predicated
per-score work and the per-block shift loads, 6 and 3 points, against the timing probe's
1 to 2 (the probe carried a table lookup where the exact form carries the extract and the
shift). The branch-free build (the rule unconditional in a `B9_KS=1` build, chosen by the
manifest as the seam builds already are; gated 20 / 20 on the rule-only suite) reads 0.808
at 32k, 0.877 at 8k and 0.841 at 2k against the B9 seam's 0.869 / 0.901 / 0.856, decode
0.938 against 0.933: LSSA-B9.3 costs the prompt side 6 points at 32k, 2.5 at 8k and 1.5 at
2k, and nothing in decode - inside the plan's kill line for the 70B's last point, and the
reason it is a per-model rule rather than the default.

**LSSA-B9.3 served (2026-09-08; `results/s2/attn/b93/`).** Llama-3.1-70B at TP = 4 with 32
declared outlier channels under B9.3 (cap 3, round half up): **+0.81 percent** (2.7134
against 2.6915, 30 windows, the prepare's own gate under the default 1 percent tolerance),
fingerprint reproduced, and its served rows bit-identical to `lssab9_torch(v_keyshift=3,
round half up, mu)` on a dump spin at TP = 4: E1 86,400 prompt rows and E4 16,992 generated
rows, 0 bad. Llama-3.1-8B under the same rule: +0.33 (the reference arm +0.23; B9.1 +0.22,
B8 +0.30), E1 172,800 and E4 33,984 rows, 0 bad. The first engine runs failed E1 on both
models (16,800 and 29,793 prompt rows) while the standalone seam gate passed: the glue's
packer wrote a sequence's last tile as a full eight packed bytes, which overran the next
sequence's first keys in the 16-aligned gather scratch and raced its own gather, so a
sequence's shifts depended on CTA order; the packer now stays inside the sequence, and the
E1 gate is what catches multi-sequence faults the single-sequence seam gate cannot. With
this the 70B is inside 1 percent with a 0-bit fingerprint and bit-identical rows, which is
the phase's exit condition in its first form. The chain's cost, measured as a timing-only probe carrying the
exact instruction mix on the correct-toolchain seam: 0.854 against 0.868 at 32k and 0.893
against 0.900 at 8k (1 to 2 percent). Built: the golden (`lssab9_golden` `v_keyshift`,
`ks_round`), the torch reference, the self-test (228 / 228 with the B9.3 arms), the portable
prep and decode kernels (the per-key shift in the V page's spare byte; the decode chains'
`b93_shift`), the seam (`patch_b9.py` `B9_KS=1`, packed two-bit shifts per key in two
registers per block, `lssab_set_ks`), the glue, and the rule's plumbing (`--attn-rule b9.3`,
manifest `attention.ks / ksr`, `LOCKSTEP_KS / LOCKSTEP_KSR`); gates and served rows follow.

### 7.8 Open

- Long-context decode under contract v2 is 0.856x at 32k, 0.868x at 131k and 0.879x at 262k
  (final rows, 128-token slope), from 0.755x and 0.674x at the start of the session. The
  next 10 points are engineering: the folds and per-CTA fixed cost at 32k, the launched-but-
  idle CTAs under CUDA graphs (the split cap is the current fix), and a deeper `cp.async`
  pipeline. The 4090 bare-metal profiler run is the instrument for the first two.
- Batch-32 decode is 0.934x and the ShareGPT serving workloads 0.889 / 0.939x after the
  KV-quantise fusion (v6); the decode kernel itself (30 us per layer per token at batch 32
  against stock's 24 us attention path) is what remains there.
- The 128-token single-stream TTFT (0.85x) is a fixed per-step host cost of the seam, about
  2 ms over 28 layers, invisible from 2k tokens up.
- Contract v2 replaces the generated-row rule; adopting it means re-running the 300-window
  quality battery on the final build (the rule itself is the measured B8 one) and updating
  the addendum text ("generated rows use LSSA-B8; the segment stays 32 blocks").
- Dequantisation order of the int8 linears (contract decision): the exact route 1.29 to 1.43x of bf16, the fused CUTLASS route 0.68 to 0.85x, inadmissible under the declared order; a power-of-two weight scale makes the two orders coincide (`demo/t16l_pow2.py`, unmeasured).
- Batch invariance of the whole engine needs the exact linears armed (stock bf16 cuBLAS picks kernels by batch shape); the attention rows are invariant (E5b) and the full verifiable stack passes F3 in eager.
- The fused verified stack (7.9) is faster than the attention-only build in GPU time at every
  row count profiled, but its wall-clock ratios on uncaptured steps carry the CPU cost of
  python-registered custom ops (~10 us per call over an aten op, six per layer; serving
  0.81 / 0.88x on the decode-heavy workloads against 0.885 / 0.914x attention-only). The fix
  is registering the fused ops in C++ (`TORCH_LIBRARY`) with the GEMM route inside; no
  kernel or contract change.
- Pod hazard, TP=4: after a tensor-parallel bench the four engine workers can outlive the
  parent (62 GB held per GPU, 0 percent utilisation); the next run on that node then fails
  with an out-of-memory in whatever allocates first. Kill them by PID
  (`nvidia-smi --query-compute-apps=pid`) before launching anything; `gz3_relaunch.sh` in
  the session scripts does exactly that. The single-GPU pod has not shown it.
- Long-context full-stack decode (0.759x at 32k, 0.736x at 131k) did not move with the glue
  fusion or the GEMM route; the sixteen-warp decode kernel of 7.12 later moved it to 0.870x
  at 32k, 0.811x at 131k and 0.844x at 512k, and the rest of this item stands. The same-session attention-only control at the same GMU 0.80
  (`attn80` rows, section 4) reads 0.851x at 32k and 0.798x at 131k, so the exact
  non-attention stack costs 9 to 11 points there - about 1 ms per token at 32k, 1.7 ms at
  131k, growing with context - while its TTFT at 131k is 1.03x against the control's
  0.92x. The decode-only profiler cannot run behind this backend with prefix caching (the
  replay of a cached 131k prompt hangs), so the attribution is by slope-differenced
  profiles at 30k (`decode_prof_lssab_b1_30k_g{16,144}_{fused,attn}`); see 7.9.
- HARNESS DEFECT, found and fixed this session: `t13b_bench.py` (single-stream, two engines
  in one process) never armed the exact stack, because the vLLM plugin arms on the first
  engine in a process and the harness set the Lockstep switch only for the second. Every
  `single_noinductor_fullstack*` and `_t16m`, `_t16m21` single-stream row is therefore an
  attention-only measurement mislabelled as the full stack (their logs have no
  `armed_at_load` banner); the armed rows are `single_noinductor_t16m22` (one stack per
  process, `ONLY=`). Serving, e2e, long-context and the profiles were armed (their logs carry
  the banner).
- Tensor-parallel runs (72B): the flat harnesses must run through `pod/pymain.py` (vLLM spawns
  its workers and re-imports the driver's main module) and the E1 dumps are saved from the
  workers by RPC before shutdown (the executor SIGKILLs workers 4 s after SIGTERM, which
  truncated a 400 MB exit-time save) and cleared by RPC after engine build: vLLM's profiling
  and graph-capture forwards (32 dummy 256-token sequences, 2-token sequences on placeholder
  block tables) are dumped like real rows, and in the first 72B run they were counted as
  generated rows (12,288 "bad" rows, all at context 256, every head, every layer) and
  polluted the checker's per-sequence key accumulation so that the real rows of two
  sequences were skipped. In-process engines (TP=1) never saw this because the driver clears
  the list itself. A gate that reports bad rows at a context length no real sequence had is
  the signature.
- DEFECT, 1M-token decode (CORRECTION: the long-context harness runs one engine and never read the tensor-parallel setting, so every 1M cell in this document ran on ONE H100 of the four-GPU node; the `_tp4` file tags and the earlier 'TP=4' wording were in error, and the numbers stand as single-GPU measurements): with the 1M model, the attention-only stack prefills
  1,000,000 tokens correctly (TTFT 447 s against stock's 360 s) and then hits
  `cudaErrorIllegalAddress` in the first decode steps (`results/longdec_lssab_p1000000_attn_tp4.log`);
  512k decodes correctly and retrieves needles identically to stock. The rig now has the
  long cases (`LSB_GATE_ONLY=gz`, `LSB_GATE_NOREF=1`: the split-64 kernel against its own
  split-1 fold, which the contract makes bit-identical, since the torch oracle is quadratic
  per step at these lengths): the decode kernel alone is EQ at 262,144 and 524,288 keys
  (`results/gate_t16n_gz.log`), and the 1,000,000-key case is the run named `gz1m`. The
  rig then reproduced the illegal address at 1,000,000 keys with the kernel alone, and the
  cause is the row fold: it stages one head's segment partials (132 floats each) into the
  64 KB K/V double buffer before folding, which holds 124 segments, about 508k keys; 512k
  is 128 segments and survived by overrunning into buffers that are dead at that point, 1M
  is 245 and ran off the end of shared memory. The first fix (T16p) folded straight from global
  memory when the partials exceed the stage: correct (rig 21 of 21, split paths EQ at
  262,144, 524,288 and 1,000,000 keys, 980 segments, `results/gate_t16n_p1m_gz.log`; E1,
  E4v2 and E6 in the engine) but latency-bound, since the one CTA that completes a row then
  walks 245 partials per element from global memory: the 1M engine decoded at 0.45x of stock
  (`_p1m` rows), and scaling the split cap with the rank's kv-head count (also needed: a
  tensor-parallel rank holds one kv-head, and the cap of 64 CTAs per kv-head had left half
  the GPU idle) did not move it. The shipped fix stages the partials in chunks of 124
  segments and carries the fold accumulators across chunks in registers, the same
  statements in the same order (`_p1m2` rows: rig, engine gates, and the 1M engine rows).
- The whole verifiable stack at 512k on one 80 GB H100 does not fit while the bf16 weights
  are retained beside the int8 copies (out of memory at GMU 0.90, 0.88 and 0.83); the
  weight-freeing switch (`T16L_FREE_BF16=1`) fails under graph capture ("data is not
  allocated yet") and needs to be made graph-safe before the fused stack can be timed
  there.
- HAZARD, torch compile cache: vLLM keys its AOT-compiled graph cache on the model config,
  not on the Lockstep arming, so an armed run can load a graph compiled by an unarmed run
  of the same model (the log line "Directly load the compiled graph(s) ... from the cache").
  With the bf16 weights freed (`T16L_FREE_BF16=1`) that graph fails with "data is not
  allocated yet"; without freeing it runs, and the gates (E6, the engine gate) show it
  computes the fused ops correctly, but the rule for armed runs is
  `VLLM_DISABLE_COMPILE_CACHE=1`.
- Tensor parallelism, the reduction's cost: the declared rank-order reduction is
  implemented as an all-gather of the ranks' bf16 partials plus one fused fp32 fold kernel
  (`t16mc::tp_rank_sum`). Its traffic is twice an all-reduce's (each rank receives every
  partial, not a reduced slice), which shows on prefill at 72B. The equal-traffic form of the
  same declared arithmetic is a pinned reduce-scatter: an all-to-all of column blocks, the
  fold on each rank's own block, then the all-gather of the reduced blocks; it needs a
  graph-capturable all-to-all behind a custom op and is not built yet.
- Never run: the memory sanitiser on the engine.

### 7.24 Stage 2, item 2.1 opener: the re-baseline against a clean stock (2026-09-09)

**Why.** 7.23 found that every filed single-stream ratio since the plugin was installed into
site-packages had been measured against a stock that the plugin had ARMED (the harness leaked
`LOCKSTEP_PREFILL` into the stock arm). The README withdrew those rows. This section is their
replacement, and the starting line of item 2.1: every one-GPU family, the long-context model and a
single-family size sweep, each lockstep arm armed FROM ITS MANIFEST (`lockstep_config.py --env`,
the served configuration: contract v2.5, LSSA-B9.1, R1 online, 8 declared outlier channels), each
stock arm in a shell with every `LOCKSTEP_` / `T16` / `LSSAB` variable unset, both in full CUDA
graphs without inductor, REP=3 medians, the same session on the same node. Every manifest was
written by `lockstep prepare` with all of its gates on the same host that day (quality, 30 windows:
Qwen2.5-7B +0.27, Qwen3-8B -0.31, Phi-4-mini +0.55, Mistral-7B +0.09, the 1M model +0.23,
Qwen2.5-1.5B +0.23, 3B +0.21, 14B +0.52; Llama-3.1-8B from its 7.23 manifest, +0.22), and every lockstep arm's
log carries `T16L armed_from_manifest`. Raw rows: `results/s2/perf/rebase/` (`make_table.py`
regenerates its README). Hosts: `lc-handover` (one H100 SXM, load average 11 to 29 on 208 cores
during the runs) and a second H100 SXM rented for the size sweep (`lc-sweep`, load 11 to 17,
terminated after). The 7B's stock arm was run before and after its lockstep arm: the two agree
within 0.5 percent on every row but one (TTFT 2k batch 1, 4 percent), so the host is quiet enough
for a baseline; the 7B rows below agree with the 2026-09-07 clean session (7.23) and the lower
band recorded on the loaded day (0.78x) was that day's host.

**The families, one H100 SXM (lockstep over stock; TTFT at 128 / 2k / 8k / 32k tokens, batch 1;
decode at batch 1 / 8 / 32):**

| model | TTFT 128 | 2k | 8k | 32k | decode 1 | 8 | 32 | serving W1 / W2 / W3 |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 0.81 | 0.86 | 0.90 | 0.87 | 0.94 | 0.94 | 0.86 | 0.75 / 0.81 / 0.77 |
| Llama-3.1-8B | 0.79 | 0.85 | 0.92 | 0.86 | 0.92 | 0.90 | 0.86 | 0.73 / 0.78 / 0.79 |
| Mistral-7B | 0.77 | 0.85 | 0.91 | 0.85 | 0.84 | 0.87 | 0.81 | |
| Phi-4-mini | 0.78 | 0.78 | 0.81 | 0.80 | 0.92 | 0.95 | 0.90 | |
| Qwen3-8B | 0.56 | 0.45 | 0.50 | 0.55 | 0.73 | 0.72 | 0.72 | |

Batched prompts sit with the batch-1 rows (7B: 2k x8 0.88, 8k x8 0.87, 8k x4 0.88, 128 x32 0.89;
Llama 0.88 / 0.87 / 0.88 / 0.87; the full grid is in the results README).

**The size sweep, Qwen2.5 on one H100 SXM, the same manifests and harness:**

| model | hidden / layers / kv heads | TTFT 128 | 2k | 8k | 32k | decode 1 | 8 | 32 |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 1,536 / 28 / 2 | 0.60 | 0.49 | 0.53 | 0.66 | 0.51 | 0.50 | 0.49 |
| Qwen2.5-3B | 2,048 / 36 / 2 | 0.59 | 0.59 | 0.67 | 0.69 | 0.61 | 0.61 | 0.58 |
| Qwen2.5-7B | 3,584 / 28 / 4 | 0.81 | 0.86 | 0.90 | 0.87 | 0.94 | 0.94 | 0.86 |
| Qwen2.5-14B | 5,120 / 48 / 8 | 0.82 | 0.93 | 0.96 | 0.89 | 1.08 | 1.07 | 0.99 |

The 32B does not prepare on one H100: `lockstep prepare` keeps the folded weights in torch beside
the vLLM engine and twice 65 GB exceeds 80 GB (out of memory at `--gmu 0.95`); it belongs to the
tensor-parallel round with the 72B and 70B. The 14B's calibration needs the engine's memory budget
at 0.55 (28 GB of weights plus 6 GB of KV cache for a 32k request; 0.45 left 5.9 GB).

**Long context, Qwen2.5-7B-Instruct-1M (dense), one H100 SXM, its own manifest (`rb_qwen1m`, key
offset calibrated by prepare), `t16f_long.py`:** TTFT 32k 0.86x (1.10 against 1.27 s), 65k 0.85x
(2.95 / 3.48 s), 131k 0.82x (8.79 / 10.70 s), 262k 0.81x (29.4 / 36.2 s); decode at 131k **1.15x**
(110.8 against 96.7 tok/s: the int8 KV cache halves the bytes per generated token). Stock's own
attention share of TTFT grows from 37 percent at 32k to 82 percent at 262k, so above 100k the
ratio is the prompt kernel's (item 2.3); below it, it is the glue's.

**What the re-baseline settles.**

1. The gap is a fixed per-layer cost, and it scales with model size exactly as such a cost must:
   the same stack is 0.49 to 0.66x on the 1.5B, 0.58 to 0.69x on the 3B, 0.86 to 0.94x on the 7B
   and 0.93 to 0.96x TTFT with decode at 1.07 to 1.08x on the 14B, whose quality gate reads +0.52. A kernel deficit would be flat across sizes; a fixed cost per launch is
   amortised by the width and depth of the layer. The 72B's serving at 0.91 to 0.98x (7.13, TP=4) is
   the same curve's far end. Item 2.1 is therefore a launch-count problem in the glue, largest on
   small models and short prompts (the 128-token rows are 5 to 8 points below the 2k rows on every
   family), and the remedy is fewer launches per layer, not a faster kernel.
2. Phi-4-mini's decode is at 0.92 to 0.95x, against the 0.70x filed under the armed stock and the
   older stack; its prompts sit at a flat 0.80x, consistent with its narrow hidden width (3,072)
   and the same fixed cost.
3. Qwen3-8B is at half of stock on prompts and 0.72x on decode. Its configuration is identical to
   the 7B's (the same seam, the same manifest path, `armed_from_manifest`); the difference is the
   per-head q / k norm, which the fused layer hands to the unfused exact path. Item 2.2 is not the
   "0.87 to 1.0x" polish the roadmap recorded, it is a 2x prompt gap on a whole family, and moves
   ahead of 2.3 and 2.4.
4. Found on the way and fixed: a fresh pod compiled the portable kernel from the top-level copy of
   its source, which had fallen behind `lssa/portable/` since the B9 work (ABI mismatch cu=472
   py=496 on the engine's struct); `pod_setup_h100.sh` now compiles from `lssa/portable/` and the
   copies are level. Also: the pod volume of `lc-handover` was at its 200 GB quota, which made
   `prepare` fail while writing constants and then let the harness run an unarmed lockstep arm
   (the portable prefill, 0.26x at 32k) - the harness now refuses a manifest without constants and
   checks the `armed_from_manifest` line before it files a row.

The single-stream rows filed in 7.12 to 7.20 for the 72B at TP=4 stay withdrawn until the
tensor-parallel round re-measures them.

### 7.25 Stage 2, item 2.1: the fixed per-layer cost, attributed and half removed (2026-09-09)

**The question.** 7.24 showed the same stack at 0.5x on a 1.5B, 0.6x on a 3B, 0.9x on a 7B and 0.95x
on a 14B against a clean stock: a fixed cost per layer amortised by width and depth. Item 2.1 is that
cost. Raw rows: `results/s2/perf/s21/` (the per-kernel profiles, every probe, every job log including
the failed tries); the plan and its log: `docs/PLAN_21.md`.

**S0, where the time is** (`pod/decode_prof.py`, per-kernel CUDA time inside the real graphs, per
token per layer). Qwen2.5-1.5B decode at batch 1: stock 78 us of kernels per layer (bf16 GEMMs 45,
attention 12, glue 12), lockstep 121 - int8 GEMMs 48 (cuBLASLt's sm80 CUTLASS kernel at one row:
0.44 to 0.69 TB/s where the bf16 read runs at 2.6), the decode kernel 23 against FA3's 12, glue 37.5
(r4quant 10.2 for two launches, normquant 8.2 for two, deqrope 5.2, r4_cl 5.1, deqsilu 4.4, deq 2.8,
rowquant 1.6). The wall clock is 4.7 ms per token against 2.4, and 1.3 ms of the 2.3 ms gap is not
in any kernel: the gaps between 18 to 20 dependent launches per layer against stock's 10. Qwen2.5-7B
decode at batch 1: lockstep's kernel time is BELOW stock's (5.68 against 5.95 ms per token, int8
GEMMs 125 against 165 us per layer) and the wall clock 0.94x - on the 7B the whole gap was launch
count. 7B decode at batch 32: glue 120 us per layer against 26 (one CTA per row on 32 SMs). 7B
prefill at 2k: int8 GEMMs 749 against 1,246 us per layer (1.66x), the seam 129 against 73, the glue
998 against 138 - deqsilu 475, r4quant 287 for three, normquant 97 for two, deqrope 78, the KV prep
51. One cause read straight from the source: the contract v2.5 outlier side path re-read the
linear's 16-byte weight row from L2 for every element (32 bytes of gathers per 6 bytes of payload).

**What was built, what each measured, what stayed** (every kernel gated bit-identical: t16m_gate
1,186 / 1,186 on every build that ran, and on the final build E1 151,200 prompt rows + E4v2 29,736
generated rows bad=0 against `lssab9_torch[sink](mu)`, E6 full graphs == eager - `handover/s1l`).

| change | probe / microbench | engine (7B; 1.5B) | verdict |
|---|---|---|---|
| S1f: row-tiled epilogues `deqsilu_t_k`, `deqrope_t_k` (one thread owns a column across 8 rows, the side-path weights in registers, all loads issued before the chain) | deqsilu at 2k rows 300 us from 475; at 32 rows 31.6 from 45.8 | 7B TTFT 0.86 / 0.90 / 0.87 -> 0.99 / 1.09 / 0.98 at 2k / 8k / 32k; 1.5B 0.49 / 0.53 / 0.66 -> 0.70 / 0.82 / 0.80 | KEPT |
| S1g: staged rotation `r4quant_s_k` (8 warps stream the row block by block, the rotated row parked in smem; from 128 rows) | 41 us at 2,048 rows from 95 (the SiLU width), 8.2 from 14.9 at 256; equal at 64, slower at 32 | in the S1f row above | KEPT |
| S1g: the norm's 256-thread form from 256 rows | 20.4 from 27.5 us at 2,048 rows (1.5B), 43 from 49 (7B) | in the S1f row above | KEPT |
| S1r: the epilogue tiles at 4 rows per thread (from 8) | deqsilu 7.8 from 9.5 us at 64 rows, 12.1 from 13.6 at 128, 158 from 167 at 2,048 (stock 79) | inside the noise on the 7B's prompts | KEPT |
| S2: int8 GEMV `gemv_s8_k` (a warp streams weight rows, K contiguous, dp4a against the staged activation quads, exact int32; rows per warp by N, the k loop unrolled 4) at rows <= 2 | one row: 1.5B 17.8 against cuBLASLt's 42.9 us for the four linears, 7B 99.7 against 111.5; at four rows the 7B's wide and deep linears lose | 7B decode 0.93 -> 0.97 at batch 1; 1.5B 0.52 -> 0.59; prompts unchanged | KEPT |
| S1a: fused norm + R1 + quant as a 4-CTA cluster per row (<= 16 rows) | 8.8 us at one row against 8.3 to 10.2 for the two kernels: the kernels are latency chains, not launch overhead (stock's fused norm: 2.2 us) | first run: no effect (the Python row-count branch is baked by graph capture; the route moved into the C++ launcher); measured alone: 7B decode 0.93 -> 0.88, 1.5B 0.52 -> 0.47 - a cluster launch inside the graphs costs more than the launch it saves | KILLED |
| S1a-v2: the warp form (1 to 4 warps per row, shuffle reductions) | 31.8 us at one row against 8.7 | - | KILLED |
| S3: programmatic dependent launch on every glue kernel | trigger at entry: the next grid resident and spinning through the GEMMs, 7B decode 0.94 -> 0.59; trigger after the reductions: the fused SiLU-rotation kernel's bf16 path diverged (54 gate cases; the same source with the macro empty passes - bisected), so compiled out; implicit exit trigger: neutral to -2 points | - | KILLED |

**The kept set, measured as the defaults** (S1m, `results/s2/perf/s21/s1m`; the engine gates on
these defaults: E1 151,200 rows bad=0, E4v2 29,736 rows bad=0, E6 PASS; every lockstep arm
`armed_from_manifest`, one clean stock arm per model, the same session):

| model | TTFT 128 | 2k | 8k | 32k | 2k x8 | 128 x32 | decode 1 | 8 | 32 | serving W1 / W2 / W3 |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B, 7.24 | 0.81 | 0.86 | 0.90 | 0.87 | 0.88 | 0.89 | 0.94 | 0.94 | 0.86 | 0.75 / 0.81 / 0.77 |
| Qwen2.5-7B, 2.1 kept set | 0.84 | **1.01** | **1.08** | 0.98 | **1.06** | **1.06** | 0.96 | 0.91 | 0.83 | 0.78 / 0.77 / 0.88 |
| Qwen2.5-3B, 7.24 | 0.59 | 0.59 | 0.67 | 0.69 | 0.62 | 0.62 | 0.61 | 0.61 | 0.58 | |
| Qwen2.5-3B, 2.1 kept set | 0.63 | 0.79 | 0.88 | 0.84 | 0.85 | 0.85 | 0.70 | 0.60 | 0.59 | |
| Qwen2.5-1.5B, 7.24 | 0.60 | 0.49 | 0.53 | 0.66 | 0.49 | 0.53 | 0.51 | 0.50 | 0.49 | |
| Qwen2.5-1.5B, 2.1 kept set | 0.53 | 0.67 | 0.83 | 0.71 | 0.75 | 0.82 | 0.59 | 0.51 | 0.50 | |

The 7B's prompts are at parity or above from 2k, its batch-1 decode at 0.96x; the 3B's prompts moved
20 points and the 1.5B's 18 to 30. Serving on the 7B: W3 (prefill-heavy) 0.77 -> 0.88 (0.90 with the
prep rewrite, S1q), W1 0.75 -> 0.78 (0.76 in S1q), W2 inside the harness band. What did not move: decode at batch 8 to 32 on every size (the
glue at 8 to 64 rows and the decode kernel), the 128-token prompt, and the small models' decode,
which the attention kernel's fixed cost now dominates.

**What the campaign cost in wrong turns, filed so they are not repeated.** (1) A Python branch on
`x.shape[0]` in the fused layer is baked by dynamo / graph capture at trace time - route by row count
inside the launcher. (2) A mechanical rewrite of every launch site also rewrote the launch inside the
launch helper into a call to itself: an infinite host loop at the first kernel, six hours at 100
percent CPU with the GPU idle, found by bisecting the source by version (the standalone test had
cleared `griddepcontrol.wait`, which is a no-op in a plain launch). (3) The pod's overlay filled with
manifests (nvcc segfaults with 61 MB free) and the volume hit its quota again; manifests live on the
volume behind a symlink now, and a job refuses a manifest without constants and checks the
`armed_from_manifest` line before it files a row. (4) `pkill -f` inside an ssh command whose text also
names the script kills the ssh's own shell; killed torch-extension builds leave a lock file on which
the next build sleeps forever.

**Three attributions that followed, and what they set** (`handover/s1n`, `s4c`, `p128`):

- The glue at 8 to 64 rows (the batch-8-to-32 decode glue): at 8 to 16 rows the cluster rotation
  (6.0 to 6.7 us) and the 512-thread norm (5.2); at 32 the staged rotation (10.4) beats the cluster
  (11.9); at 64 the one-CTA form (9.7) edges the staged (10.4) and the cluster is 17.4; the tiled SiLU
  epilogue 5 to 6 us at 8 to 32 rows (stock 4.5), 9.6 at 64 (stock 4.7). Routes set from these. Each
  kernel is 2 to 3 us above stock's, six per layer, about 15 us of the layer's 230 at batch 8 - and
  the batch-8 profile has lockstep's kernel time AT stock's (6.54 against 6.71 ms per step) with the
  wall clock 0.91x: the launch gaps again, about 1 us per kernel, which without PDL only fewer kernels
  can remove (the o_proj epilogue folded into the norm is one; the rest is structural).
- The decode kernel's fixed cost (the `LSB_DEC_TIMING` build): at batch 1 and 300 tokens 37,600 cycles
  per CTA (21 us): the prologue 11,160 (four to five dependent loads - the tile entry, the length, the
  page-table row, the query rows and per-row constants, the last block's value exponent), the segment
  partials 8,300, the phase-1 chain 7,870, the PV 4,470; three 128-key chunks at about 6,900 each.
  The engine already splits the chunks across CTAs, so a layer step is prologue + one chunk + the
  combine. The remedy - a prep-written per-tile header so the prologue is one load - is metadata only
  and worth 3 to 4 of the 23 us; filed behind the items above.
- The 128-token prompt (7B 0.84x): per layer the int8 GEMM 112 us against bf16's 182, the KV prep
  `lsb_prep` 47 us - a FIXED cost (51 at 2k): one CTA per (block, kv-head) walking 64 elements per
  thread with a page-table address per element - the seam 31 against FA3's 13, the norm 13.9 per call
  at 128 rows against 2.7. The prep's block path was rewritten warp-per-row (T16y: one page address
  per row, coalesced rows, the same per-element arithmetic) and is under the rig and engine gates as
  this is written - and passed them: rig 21 / 21 on both warp counts, head groups and the long splits, E1
  151,200 rows bad=0, E4v2 29,736 rows bad=0, chunked == single prefill; the 128-token prompt 0.877x
  against 0.825 in the same session (128 x8 0.994 against 0.949, 2k 1.016 against 0.992), the longer
  prompts unchanged. Kept; the shipped kernel.

**Where parity stands after 2.1, stated for the roadmap.** Measured at parity or above: the 7B's
prompts from 2k tokens (1.03 / 1.08 / 0.99x), its batched prompts (1.07x), the 14B's decode (1.08x,
7.24). Within the harness band of parity: the 7B's batch-1 decode (0.96x). Within reach of the same
kind of work: the 7B's batch-8 decode (0.90x: kernel time at stock's, the gap is launch gaps; two or
three merges that do not add latency). STRUCTURAL, not reachable by tuning: batch-32 decode (0.82x)
and saturated serving (0.76 / 0.79 / 0.90x), which is the launch-count floor (about 18 launches per
layer against stock's 10) where each gap matters more, plus the mixed prefill+decode step - closing
it needs either programmatic dependent launch done right (two attempts failed here, one on speed and
one on a correctness effect not explained) or a persistent layer kernel; a multi-week item, filed as
2.1's remainder. BY DESIGN: the 1.5B and 3B (0.67 to 0.89x prompts, 0.59 to 0.67x decode), whose gap
is the decode kernel's fixed cost and the contract's one-row norm latency; a realistic ceiling of about
0.8x on prompts and 0.65x on decode. None of this week's numbers say anything about mixture-of-experts
models (Stage 3), whose stock baseline is FP8 / MXFP4, not bf16, and whose target is written as
15 percent overhead, not parity.

**A correction to the "structural" reading, from the gaps measured (later the same day;
`results/s2/perf/s21/gaps`, `host`, `gapby`).** The inter-kernel gaps were read from the CUDA event
timestamps rather than inferred. Per token on the 7B, both stacks replaying full CUDA graphs (vLLM's
log shows FULL and PIECEWISE captures for both): stock 393 kernels, lockstep 443 - 13 percent more,
not twice; lockstep's busy time BELOW stock's at every batch (5.45 against 5.96 ms at batch 1, 6.42
against 6.72 at 8, 9.11 against 9.20 at 32); the gaps between consecutive kernels 1.19 / 1.16 / 1.06 ms
against stock's 0.43 / 0.37 / 0.24 - a mean of 2.4 to 2.7 us per gap against 0.66 to 1.13 (inside the
graph stock launches every kernel 0.45 to 0.55 us after its predecessor) - and host-side stalls of
200 us or more of 0.77 / 0.88 / 1.24 ms per token against stock's 0.01 / 0.01 / 0.07. The batch-32 wall
gap is therefore not the kernels and not mainly their number: it is the per-launch gap and about a
millisecond of host stall per step. The stall's shape is in the per-step kernel list: stock's lm_head
is one GEMM per step, the fused int8 lm_head (B2'') ran, per step and outside the graphs (vLLM does
not capture compute_logits), about fifteen eager torch ops - the row abs / amax / clamp / div / round /
clamp / cast, the plane shift / sub / casts, two cats - before its stacked GEMM and combine; a stock
decode step issues 14 eager launches per token in all. Two remedies are under gate as this is written:
`lm_prep`, the row quantisation and plane split in one launch straight into the stacked operand
(three launches per step in that section), and the placed-trigger form of programmatic dependent
launch (the trigger after each glue kernel's loads or reductions, the two kernels whose bf16 path
diverged under a trigger excluded), which passes the kernel gate with and without the attribute and
reads +4 points at batch 1 and +1 at 8 to 32 on the 7B (S3b). Both passed and both stayed: `lm_prep` removes a
cudaStreamSynchronize the torch lm_head path issued on every decode step and 14 of its eager launches
(the lockstep step's host side now matches stock's: one graph launch, 20 eager launches against 14,
in-graph gaps 0.63 us); the placed-trigger PDL passes E1 / E4v2 / E6 under the launch attribute and
reads, one stock arm per model, 7B decode 0.98 / 0.94 / 0.85 at batch 1 / 8 / 32 (from 0.94 / 0.93 /
0.85), 3B 0.74 / 0.62 / 0.60, 1.5B 0.62 / 0.52 / 0.50 - three to six points at batch 1, neutral to minus
three at batch 32. The remainder at batch 32 is now a measurement question before it is an engineering
one: by the profiler's accounting the batch-8 step's GPU time is at or below stock's, while the bench
reads 0.94x; a test of the engine configuration the two harnesses differ in (max_model_len 8,192 / 0.4
against 32,768 / 0.85) is queued.

**S5a, the discrepancy resolved: the fused lm_head route was never live in the bench (2026-09-09,
`results/s2/perf/s21/s5a`; the shape profile in `w1shape`, the engine test in `mlen`).** The
engine-configuration test said no: at max_model_len 8,192 / 0.4 the bench reads decode 0.987 / 0.952 /
0.870, at 32,768 / 0.85 0.971 / 0.928 / 0.851 - two points. The W1 profile re-run with input shapes
named the "misc torch" bucket (334 ms against stock's 25): every large int64 op is on
[rows, 152064], the vocabulary - `to(int64)`, `<<`, `add`, `copy_` and two fp32 scalings per step -
i.e. `Variant.logits`, the torch plane combine that `lm_prep` / `lm_combine` had replaced; and the
profile with `T16M_LMPREP=1` and `=0` gave the same GPU time. Reading the arming code: `t16l_stack._install`
is not idempotent, and every harness that sets T16L_ARM=1 arms twice (itself, before the engine is
built, and again through the vLLM plugin's load hook). The second `arm()` installs t16l's
`_get_logits` - the torch path, with its cudaStreamSynchronize per step - on top of t16m's fused
wrapper, and that hook never calls what it wrapped. So the fused lm_head was dead in every
t13b_bench and t14_bench run filed in 7.24 and 7.25, and in the W1 profiles; the decode profile
harness arms once, which is why S3c saw `lm_prep` act there and the bench did not (0.921 against
0.919). The fix (`t16m_stack._install_lm_fuse`): the test is "is our wrapper the live hook" (a
marker on the function), not "did we install once"; if anything sits above it, it wraps again,
and the fall-through still reaches whatever is there. Confirmed live in the bench's configuration:
`lm_fuse_installs` 2, `lm_fused_calls` 1,540 of 1,540 decode steps. Gates under the double-armed
configuration: t16m kernel gate 1,186 / 1,186, E6 full graphs == eager. Re-measured, one process per
stack, REP=3, the 7B on the H100 (`s5a/single_7b.json`, `t14_res/*S5A*`):

| row | stock | lockstep | ratio | before S5a |
|---|---|---|---|---|
| TTFT 128 / 2k / 8k / 32k, batch 1 | 8.72 / 48.76 / 207.5 / 1107 ms | 9.79 / 48.80 / 199.1 / 1138 ms | 0.89 / 1.00 / 1.04 / 0.97 | 0.87 / 1.03 / 1.09 / 0.99 |
| TTFT 128 / 2k / 8k, batch 8; 8k batch 4 | 24.4 / 366.5 / 1594 / 800 ms | 25.6 / 350.1 / 1528 / 768 ms | 0.95 / 1.05 / 1.04 / 1.04 | - / 1.07 / - / - |
| prefill 128 x 32 | 93.6 ms | 93.5 ms | 1.00 | - |
| decode batch 1 / 8 / 32 | 165.0 / 1287.5 / 5078.9 tok/s | 186.4 / 1401.5 / 5238.6 tok/s | **1.13 / 1.09 / 1.03** | 0.98 / 0.94 / 0.85 |
| serving W1 chat / W2 mixed / W3 prefill-heavy, output tok/s | 5551 / 8379 / 585 | 4781 / 7675 / 558 | **0.86 / 0.92 / 0.95** | 0.76 / 0.79 / 0.90 |
| serving W1 / W2 / W3, mean TTFT | 5955 / 6216 / 6883 ms | 7257 / 6782 / 7207 ms | 0.82 / 0.92 / 0.96 | 0.70 / 0.79 / 0.90 |

The "structural" reading above is withdrawn: batch-32 decode was the torch lm_head's four int64
conversions, four shifts, three adds, two scalings and one host sync per step on a
[32, 152064] tensor, not a launch-count floor. Decode is now above parity at every batch, prompts at
or above parity from 2k tokens, the 128-token prompt at 0.89 (its remainder is the prompt's fixed
cost, 7.24), and saturated serving at 0.86 / 0.92 / 0.95. What every number filed before S5a for
OTHER models (Llama-8B, Mistral-7B, Phi-4-mini, Qwen3-8B, the size sweep, long context, the 72B)
has in common: measured with the torch lm_head path live; their decode and serving rows are
lower bounds and are to be re-measured (their prepared weights were deleted with the pods; the
re-measurement is a prepare-and-bench pass per family). Serving's remainder is now the glue at
many rows, measured in the same session (`w1shape/glue_mixed`): the norm 2.5x to 4.9x and the
rotation about 10x off their byte floors at 2,112 rows, deq_silu at its int32 byte floor - a
kernel item (Nsight Compute on the three kernels is queued), and the decode kernel against FA3 in
mixed steps (+361 ms of 4,373 in W1).

**S6, the glue at many rows without hardware counters (2026-09-09, `results/s2/perf/s21/s6c`, `s6e`).**
Nsight Compute is refused on the pod (no GPU performance-counter permission), so the forms were designed
from `cuobjdump --dump-resource-usage` and the microbench, and each one measured. Two latency remedies
gate clean and are slower - the norm with its row in shared memory (sixteen rows resident per SM
instead of five: 40.4 against 38.8 us at 2,112 rows) and the rotation rotated twice with no staging
(126.8 against 101.0 at the SiLU width) - so neither kernel is latency-bound; the one-CTA register
rotation at many rows is slower still (148 against 100: residency does matter, one row per SM loses).
KEPT: the norm's integer chain in 32-bit arithmetic where the ranges prove it exact (|q| <= 32767 and
the LUT value <= 32768, so q * bb fits int32; t <= 32768; t * |G| one 32x32 -> 64 multiply; the rounding
shift, clamp and sign on the magnitude) - 4.42 / 7.33 / 16.6 / 29.3 us against 4.99 / 9.17 / 22.2 /
38.8 at 64 / 320 / 1,088 / 2,112 rows, 11 to 25 percent at every row count, gate 1,186 / 1,186; and the
rotation with 32 elements per lane (three cross-lane butterfly stages instead of five, the stage order
and each binary32 (a + b, a - b) unchanged) at the hidden width only (16.6 against 19.1 at 2,112 rows;
at the SiLU width its one-CTA form loses residency, 107.7 against 100.4, and its staged form is slower
everywhere, 111.3 - killed). With both as defaults, the 7B against S5A's stock arm: TTFT 0.89 / 1.02 /
1.04 / 0.99 at 128 / 2k / 8k / 32k (batch 8: 0.96 / 1.07 / 1.05), decode 1.15 / 1.13 / 1.03 at batch 1 /
8 / 32, serving W1 / W2 / W3 0.88 / 0.93 / 0.97 (mean TTFT 0.85 / 0.93 / 0.96). The rotation with its
butterfly compiled out is still 79 us at 2,112 rows of the SiLU width (byte floor about 40), so the
butterfly is a fifth of that kernel and the remainder is its body: the one per-element operation that
is neither a load, a max nor a store is the contract's quantisation division, `__fdiv_rn`, which every
quant epilogue runs per element; its cost is being measured (the same kernels with a multiply, not a
candidate: the contract pins the division) and the exact remedy, if it is the remainder, is a fast
path that takes the division only where its rint could differ.
Measured next (`s6f`): the division's cost is two instructions against eight, not latency - the exact
fast path (the candidate from the multiply, the reference division only within 2^-13 of a half-integer)
is neutral at the SiLU width (101.9 against 100.8 us) and -4 percent at the hidden width, so the rotation
is instruction-throughput-bound, about 27 instructions per element against 40 us of bytes; kept as exact
and not slower. The SiLU's `1 / den` as the correctly rounded reciprocal (`__frcp_rn`, the same bits by
definition) takes 11 to 12 percent off deq_silu (159 against 180 us at 2,112 rows). The 7B with all of S6
as defaults, against S5A's stock arm: TTFT 0.90 / 1.02 / 1.05 / 0.99 at 128 / 2k / 8k / 32k (batch 8:
0.95 / 1.06 / 1.06), decode 1.15 / 1.13 / 1.04 at batch 1 / 8 / 32, serving W1 / W2 / W3 0.88 / 0.93 /
0.97 (mean TTFT 0.85 / 0.93 / 0.97). The glue work has reached its plateau for serving: what separates
W1 from parity is now, in the W1 profile's own terms, the attention in mixed prefill + decode steps (the
decode kernel's per-key cost against FA3: +361 ms of 4,373) and the rotation, which has no stock
counterpart (276 ms); the first is a kernel project measured in weeks (S4c: 7,400 cycles per 128-key
chunk), the second is the contract's own cost.

### 7.26 Stage 2, item 2.4: saturated serving - the decode kernel at serving shapes (2026-09-09)

**The question.** After 7.25 the 7B is above parity on every single-stream row and serving reads chat
0.88x, mixed 0.93x, prefill-heavy 0.97x; chat is the workload a customer runs. The W1 profile put the
remainder in the fused glue at 64-row steps (+720 ms of 4,373, mostly at or near its byte floor after
S6) and in attention (+361 ms: the decode kernel 43 us per launch against FA3's 20.6). The plan
(`docs/PLAN_24.md`) and its log carry the measurements; this section files what they settled.

**What the decode kernel costs, measured at 64 sequences** (`results/s2/perf/s24/p0`, in-engine
per-kernel time; decode-only by differencing GEN=96 against GEN=32 so the prefill share cancels): the
kernel 33.5 / 40.8 / 56.4 / 85.0 us per step-layer at 128 / 256 / 512 / 1,024 context against FA3's
decode 22.0 / 29.1 / 37.7 / 60.0 - about 26 + 0.053 x context against 15 + 0.04 x context: a fixed cost
11 us above FA3's and a slope 1.3x FA3's; the decode-only step 0.92x / 0.91x at 512 / 1,024, with the
attention difference 80 to 95 percent of the step's gap (the int8 linears cover the glue at 64 rows but
not the attention). The timing build (`s7t`, the rig, NSPLIT=1) per CTA: the prologue 13.3 k cycles of
35.2 k at 128 context (38 percent), 12.6 k of 45.1 k at 256, 11.1 k of 67.8 k at 512; 9 k to 11 k cycles
per 128-key chunk split evenly between the partials, the chain and the PV; the cp.async wait under 4
percent.

**Measured dead, again.** The scores and the PV on the int8 tensor cores (`-DLSB_IMMA=1/2`, mma.sync
m16n8k32, rows = the GQA heads): gate clean, not faster at any context (53.2 / 81.6 / 138.4 and 144.2 /
253.3 against the dp4a body's 52.7 / 83.1 / 139.2 / 240.5). This is 7.10's (T16o) verdict re-measured
in the pipelined kernel before the spec was re-read - 1.5 pod-hours, filed as the wrong turn it was:
the block is bound by its barriers and the short dependent chains between them, not its arithmetic.
Routing decode rows through the exact FA3-B8 prompt kernel was read for cost and is the FA3-M1-class
project (per-sequence launches, no packed GQA, no exact split combine): weeks.

**What moved: the split policy.** `auto_split_nt` targeted 8 CTAs per SM (T16j, tuned on W4's few
31k-token sequences, where the splits are what makes the kernel fill the machine). At 64 sequences
the tiles alone are 256 CTAs, one full wave at two per SM, and every split added a 13 k-cycle
prologue per (sequence, head) for no parallelism: 512 context, 4 blocks, was cut five ways. The rule
now: no split once the tiles put one CTA on every SM (`LOCKSTEP_LSSAB_SPLIT_FULL`, default 1); below
that the T16j target stands. Metadata only: the fold is exact in any split (K7, E8). On the gated
build (`s7s2`): the kernel 44.1 / 63.2 us at 512 / 1,024 against the old policy's 54.1 / 83.6 (-18 / -24
percent), 1.17x / 1.05x FA3's decode-only. A first sweep had read the same numbers off an UNGATED
rowmap build whose garbage shadow slots changed the prologue's path; it was withdrawn and redone -
the rule is the one 7.25 already had: no performance number off a build that has not passed its gate.

**The 7B with it** (`s7p4`, against 7.25 S5a's stock arm, REP=3): TTFT 0.90 / 1.03 / 1.06 / 0.99 at 128 /
2k / 8k / 32k (batch 8: 0.97 / 1.07 / 1.06; 8k x 4 1.06; 128 x 32 prefill 1.02); decode 1.14 / 1.10 /
1.01 at batch 1 / 8 / 32; serving W1 / W2 / W3 0.87 / 0.97 / 0.97 (mean TTFT 0.87 / 0.97 / 0.96). The
mixed workload moved 3.4 points (its decode rows sit at 500 to 2,000 context, where the old policy
split); chat is flat because its contexts are under 256 tokens, where no split happens under either
policy. Its remainder, stated for the roadmap: the kernel's prologue (the dependent metadata loads,
the in-kernel prep of the newest block, the table copy: 13 k cycles, 38 percent of a CTA at 128
context) and the 64-row glue (eight launches per layer at 2 to 14 us each, each at its measured floor
per form). Items 2.2 and 2.3 and the family re-measurement come before any further work on this
kernel; the FA3-class rewrite of the decode path is the one large lever and a separate decision.
Filed after the section above, the same evening: the rowmap's shadow tables and per-sequence metadata
in shared memory (`s7r3`, K-rig 43 / 43, t16n 21 / 21) take the per-step rowmap from 139 to 124 us,
and with the no-split default the decode kernel at 64 sequences reads 27.7 / 34.1 us at 128 / 256
context (the old policy split there too once the context grew past a block), which puts the
decode-only step at parity at chat contexts: 9.14 against 9.13 ms per token at 128, 11.93 against
11.79 at 256. The chat serving remainder therefore sits in the mixed steps and the longer contexts,
and its next attribution is a W1 profile under the new policy.

### 7.27 Contract v3, DRAFT: the router of a mixture-of-experts layer (2026-09-10, Stage III M0; the reference replays GPU == CPU bit for bit on 915 real tokens of Qwen3-30B-A3B, `results/s3/m0`)

The expert path of every open model in use today (PLAN_3, section 1) begins with a discrete decision:
which k of E experts a token visits, and with what weights. Under the contract the decision is part
of the declared arithmetic and part of the fingerprint - a verifier that checks the expert GEMMs but
not the routing can be shown a correct tensor for the wrong computation. The declaration, in the
order the reference evaluates it (`lssa_runtime/moe_ref.py`, the definition; every step integer or a
named binary32 operation, so a CPU replays it):

1. **Gate logits.** The gate is a declared linear (contract v2.1): the token's per-row int8
   quantisation with the row scale `asc`, the gate's per-channel int8 weights with `ws`, the int32
   product, the epilogue `g[e] = bf16( (fl32(acc) * ws[e]) * asc )`. E values per token, bf16.
2. **Scores on the grid.** `m = max_e g[e]`; `d[e] = rint( fl32(m - g[e]) * fl32(2^26 / ln 2) )` - the
   difference of two bf16 values is exact in binary32, one binary32 multiply by a declared constant,
   rounded once to a non-negative integer: a log2 distance on the attention rule's 2^26 grid. The score
   is that rule's exponential, `s[e] = TAB[(d >> 13) & 8191] >> (d >> 26)` for `d < 16 << 26` (the 13-bit
   index over 15 doublings, values in [1, 2^16]) and 0 beyond. Integers; no libm anywhere.
3. **Top-k.** The k largest `s[e]`; ties broken by the LOWER expert index. A total order, so the
   selection is a function of the integers alone.
4. **Routing weights.** With renormalisation (Qwen3's `norm_topk_prob`): `S = sum over the k selected
   s[e]` (an integer, at most k * 2^16), `w[e] = bf16( fl32(s[e]) / fl32(S) )` - one binary32 division
   per selected expert, rounded once to bf16. Without renormalisation: `S = sum over all E` instead.
   Sigmoid scoring, the DeepSeek score bias and grouped top-k, and V4's sqrtsoftplus are further
   declared forms of steps 2-4, added per model family (PLAN_3, 3.1).
5. **The experts.** Each selected expert is the declared SwiGLU block of contract v2.5 on the token's
   row, rotations included: R1 (the 256-block Sylvester Hadamard over 16, exact binary32 adds) on the
   block input before its row quant, the gate_up linear with the folded weight and the v2.1 epilogue,
   the SiLU table, R4 (the same transform) on the activation before its row quant, the down linear
   with the folded weight - per expert, with per-expert weight scales in the manifest. Measured on
   Qwen3-30B-A3B (30 windows of 2,048 tokens): +0.12 percent perplexity against stock with the
   rotations, +0.71 without; the router's own cost is nil (`results/s3/m0`).
6. **Combine.** `y = bf16( sum_j fl32(bf16( fl32(w[e_j]) * fl32(out_j) )) )` over the selected experts in
   ASCENDING expert index, binary32 accumulation in that order; the shared expert, where the model has
   one, is added last with its own declared gate. The order is written so that expert parallelism
   returns partials in any arrival order and folds them in the declared one (the TP rule).

What the fingerprint covers: the routing (`e_j`, `w[e_j]` per token) and the output. What the
verifier replays on a CPU for a sampled row: steps 1 to 6 from the token's input and the manifest,
with the expert weights of the selected experts only. Gate: bit-identical to the reference on every
routed token of a 300-window run (M0), and the routing decisions of the reference against vLLM's
`ops.topk_softmax` on the same gate logits reported as an agreement rate, not a gate: the contract
model IS the declared one, and perplexity against stock judges whether the declaration lost anything.


**M1 MEASURED (2026-09-10, H100, Qwen3-30B-A3B, `results/s3/m1`, PLAN_3).** The router, the combine and the
whole expert path of this draft exist as kernels and are bit-identical to the torch reference:
- `t16moe::route` (one warp per token: int8 gate accumulator -> declared epilogue -> the 2^26 log2-grid table
  scores, top-k with lower-index ties, bf16 weights): 60 / 60 cases, random and tie-dense rows.
- `t16moe::combine` (the k rows per token in ascending expert index, bf16(w * row) summed in binary32): 20 / 20.
- `t16moe_gmm::gmm`, the exact grouped int8 GEMM (CUTLASS 4.3 classic grouped GEMM, int32 accumulation, the
  per-problem table filled on device from the expert offsets, no host sync): 42 / 42 cases exact against
  per-expert `torch._int_mm` at E 8 / 32 / 128, both expert shapes, both tile configurations, empty and
  one-row experts.
- The fused route (`moe_fused.MoEFused`: rowquant + int8 gate + `route`, R1 + row quant, expert GEMMs +
  `deq_silu_r4quant` + `deq_rows`, `combine`) on 915 real tokens of the layer input at layers 0 and 24:
  routing sets, routing order, weights and output all EQUAL to `moe_ref.moe_forward(rot=True)`, 0 rows differ.
The grouped epilogues are the dense kernels (`deq_k`, `deqsilur4_k`) with a per-row expert index selecting the
ws block - the same binary32 operations in the same order per element. Engine perplexity with the fused path in all 48 layers: 8.81400582424298, identical to the reference
arming's value (M1e). Under vLLM's CUDA graphs (armed at model load) the fused path measures 0.87 / 1.04 / 1.26x of stock at decode
batch 1 / 8 / 32 and 0.86 / 0.88x on TTFT 2k / 8k (M1F11, after the M1f kernel round), and SERVED (ShareGPT, 64
concurrent, 500 prompts) 1.32-1.37x of stock on request and output-token throughput with median TPOT 17.3 vs 24.0 ms,
reproduced to 0.2% in a second rep (M1S4 / M1S5; M1E9's first numbers
were 0.62 / 0.88 / 1.06x and 0.74 / 0.73x). Still draft: shared experts, and the GPT-OSS family (intermediate 2,880 is not a
multiple of 256: the R4 block or its absence is an M0-style decision before any kernel).
**Contract v3 addendum, DRAFT (2026-09-10): the GPT-OSS family (M2).** GPT-OSS-20B / 120B differ from Qwen3-MoE in
five declared places, everything else (router table scores, top-k, dispatch order, combine) is SPEC 7.27 as is:
1. The gate linear has a BIAS: `g = bf16( bf16( fl32(acc) * ws[e] * asc ) + b_g[e] )` - the dense epilogue with bias
   (deq1s: the bias add after the dequant, one more bf16 rounding), b_g the checkpoint's bf16 router bias.
2. The experts' up projection has a bias b13[e] (2I) and the down projection a bias b2[e] (H): both added in the
   declared epilogue exactly as the dense stack's biased linears (bfr(y + b)).
3. The activation is the clamped SwiGLU of the checkpoint: with g = the gate column and u = the up column (the
   checkpoint interleaves them, g = col 2j, u = col 2j + 1; the fold de-interleaves the rows of w13 so the two
   halves are contiguous - a permutation, exact), `gc = min(g, 7)`, `uc = min(max(u, -7), 7)`,
   `glu = bfr( gc * sig( fl32(1.702) * gc ) )` with sig the contract's sigmoid table on the bf16-rounded argument
   `bfr(fl32(1.702) * gc)`, `h = bfr( (uc + 1) * glu )`. The constants 7, -7, 1, 1.702 are binary32 literals.
4. The intermediate width 2,880 is not a multiple of 256: the R4 rotation before the down projection is either
   the 64-block Sylvester / 8 (2,880 = 45 x 64) or absent; the choice is made by the M2-M0 perplexity
   measurement (the rotation exists to tame int8 outliers; it is not part of the model).
5. The published weights are MXFP4 (4-bit mantissa, power-of-two block scales). The contract's int8 expert
   weights are quantized from the EXACT dequantization of the MXFP4 values (a finite set of bf16-representable
   numbers); an alternative that consumes the MXFP4 values directly (int4 mantissa x 2^k into the int8
   operand, exact by construction) keeps the model identity and the footprint and is assessed in M2-M0.
Attention: sinks and a sliding window on alternate layers - the LSSA-B8 rule with a sink term (gated in T16) and
the window as a mask; outside this addendum's scope until the MoE block is measured.
**M2-0 MEASURED (2026-09-10, H100, GPT-OSS-20B, 1,024 real tokens, layers 0 and 12).** The addendum reference
(items 1-3 above, int8 experts from the exact MXFP4 dequant, rotation 64-block or none) replays GPU == CPU bit for
bit in all four cells. Routing agreement with stock's bf16 router 95.5% / 97.8%. Relative L2 of the layer's MoE
output to stock's MXFP4 kernels: the dequantised model evaluated in bf16 0.35% / 0.33% (the stock kernels' own
noise); the contract without rotation 4.0% / 4.6%; with the 64-block rotation 2.2% / 2.9%. The hidden size 2,880
is not a multiple of 256 either, so the 64-block rotation applies on both sides of the expert block. Raw-wikitext perplexity
(stock 367.41; contract 345.90 without and 365.69 with the rotation, both BELOW stock) is not a measure for this
harmony-format chat model. The in-distribution measure, perplexity on ShareGPT conversations rendered with the harmony
chat template (30 x 2,048 tokens): stock 15.101, the contract with the 64-block rotation 15.531 (+2.85%), without
rotation 16.539 (+9.5%) - the int8-from-dequant weights cost real quality on this family (per-row int8 scaling
crushes the MXFP4 block scales), so item 5 is the GPT-OSS contract; the int8 form is the gated interim. The FUSED kernel
path equals the reference bit for bit on real tokens at layers 0 and 12, with the 64-block rotation (M2K3) and without
rotation (M2K5).
Engine perplexity of the fused path 365.68665819674897 = the reference arming's (M2E1). Graph-mode speed vs stock
MXFP4 (int8 form, 64-block): 0.85 / 0.72 / 0.71x at decode 1 / 8 / 32, 0.80 / 0.83x TTFT (M2E2). Parity from batch 8 up needs item 5 (the MXFP4-consuming operand): stock reads
4-bit weights and nearly every expert is active on this family.

**Stage III kernel inventory (2026-09-10, `lssa_runtime/t16moe_kernels.cu`, `t16moe_gmm.cu`, the grouped forms in
`t16m_kernels.cu`; every kernel gated bit-identical against the torch reference of this section, `moe_m1_gate.py`,
`moe_m1b_gate.py`, `moe_gmm90_gate.py`).**
| op | what | gate |
|---|---|---|
| `t16moe::gate_route` | one block per token: the contract row quant of the unrotated row, the int8 gate dot (dp4a, 8 lanes per expert, interleaved reads), the declared epilogue (+ optional gate bias), the table scores, top-k with lower-index ties, the bf16 weights; optionally the R1-rotated row quant of the same row (256- or 64-block) | 54 + 8 cases vs rowquant_pad + _int_mm + deq_rows + route, and vs r4quant_pad / the torch 64-block quant |
| `t16moe::route` | the router alone from bf16 logits (one warp per token) | 60 cases, random and tie-dense |
| `t16moe::dispatch` | one launch: offs, the canonical (expert, token) order by a block-wide scan, rowexp, inv, and the a8 / asc row gather | 18 cases vs the torch argsort block |
| `t16moe::r1quant` | the rotated row quant for prefill rows, 256- or 64-block | vs r4quant_pad (256) and torch (64) |
| `t16moe_gmm::gmm` | CUTLASS 4 classic grouped int8 GEMM, int32 accumulation, device-filled group table (no host sync); 128x128 / 64x128 tiles, and a GEMV form for few rows | 64 cases vs fp64, empty / one-row / dense groups |
| `t16moe_gmm::gmm_deq` | the GEMV with the contract dequant (+ optional bias) fused | 8 + 2 cases vs gmm + deq_rows_g |
| `deq_silu_r4quant_g` | the grouped gate-up epilogue: dequant (+ bias), SiLU table or the clamped SwiGLU, R4 (256- or 64-block), row quant; multi-row (warp per 256 columns) or per-row CTA forms | 60 + 6 cases vs the per-row form and the torch reference (all pass, M2K4) |
| `deq_rows_g` | the grouped down epilogue (+ bias), 8 columns per thread | 15 + 2 cases |
| `t16moe::combine` | the k expert rows per token in ascending expert index, bf16(w * row) summed in binary32, block per token, 8 columns per thread | 20 cases |
| `t16moe_gmm90::gmm90` | the Hopper TMA ptr-array int8 grouped GEMM (exact; 4x slower than the classic kernel in this CUTLASS: not used) | 12 cases |
| `t16moe_mx::mx_gemv` | item 5: the MXFP4-consuming block-scaled integer GEMV (4-bit block-major weights, exact int32 block sums, the pinned-order binary32 block-scaled accumulate, fused epilogue with bias) for few rows | 10 cases vs the torch item-5 reference; the fused path on it equals the reference on real tokens at layers 0 / 12, T 48 / 1,024 (M2X2) |
| `t16moe_mx::mx_gemm` | item 5 on int8 tensor cores: 64 x 128 tiles, mma m16n8k32 per 32-wide K block, the per-block step on every element in block order (the same declared order), a flat per-expert tile list | 5 cases identical to the GEMV / reference; the fused path on it equals the reference on real tokens (M2X3 / M2X4); 108 TOPS unpipelined |
| `t16moe_mx::mx_gemv2` | the GEMV with 8 lanes per column: the exact int32 block sums in parallel, one lane's ordered accumulate | identical to the GEMV and the reference in every case (M2X5 / M2X8) |
| `t16moe_mx::mx_gemm2` / `mx_gemm16n<NBLK>` | the pipelined tensor-core forms (cp.async double buffering, the B tile unpacked once per stage): 128-row tiles and 16-row tiles with 32- / 64- / 128-wide K stages, chosen by rows per expert; the fast 4-bit unpack (`__byte_perm` magnitude table, sign byte masks, the E2M1 negative zero masked) | identical to the GEMV and the reference in every case (M2X8; the 64- / 128-wide stages in M2X9 / M2X10) |
The engine arming (`moe_stack.py`) replaces `MoERunner._forward_impl` at model load, before torch.compile and CUDA-graph
capture; the vLLM plugin does it under `LOCKSTEP_MOE=1`.
**Item 5, the MXFP4-consuming operand, DRAFT design (2026-09-10).** For every expert weight row n and 32-wide K-block b the
checkpoint gives 4-bit values m (E2M1, values in {0, 0.5, 1, 1.5, 2, 3, 4, 6} with sign) and one exponent k_{n,b} (E8M0).
Declared evaluation of acc[m, n] for an int8 activation row a: (i) per block, s_{m,n,b} = sum_{j in b} a[m, j] * (2 m[n, j])
in int32 - the doubled mantissas are integers in [-12, 12], so the block sum is exact and |s| <= 32 * 127 * 12;
(ii) across blocks in ASCENDING b, acc = fma( fl32(s_{m,n,b}), 2^(k_{n,b} - 128), acc ) in binary32 - a pinned-order
sequence of 90 (K = 2,880) correctly rounded fused multiply-adds per output, the 2^-1 of the doubling folded into the
exponent; (iii) the epilogue as today: y = bf16( fl32(acc) * asc[m] ) (+ bias). Step (ii) is deterministic on every
device that follows the order; it forbids split-K and any reassociation of the block sum. On tensor cores the block
sums (i) are int8 x int8 -> int32 mma tiles over a 32-wide K slice (the mantissas stored as int8), the per-block
fma (ii) on the accumulator tile with the block's scales broadcast along m - the structure of CUTLASS's blockwise-
scaled FP8 GEMM (example 68) with an integer mainloop and a pinned block order. Footprint: the 4-bit mantissas
stay 4-bit in memory (unpacked to int8 in shared memory), so the weight read equals stock's. Implementation note: the
block sum s is an integer below 2^16 in magnitude and 2^(k - 128) a power of two, so fl32(s) * 2^(k - 128) is exact and
the step's single rounding may be taken by one fused multiply-add, fmaf(fl32(s), 2^(k - 128), acc) - identical bits to the
multiply-then-add form; the power of two itself is the binary32 with exponent field k - 1 (k >= 1), not a library call. The reference is
`moe_ref` with this evaluation replacing int_mm + lin_epilogue for the experts; the M2C0-style GPU == CPU replay and
the routing / output-distance rows judge it as for the int8 form.
**Item 5b, block-scaled activations, DRAFT (2026-09-10).** Measured on GPT-OSS-20B (M2C1): with the weights exact (item 5)
and the activation row quantised to int8 per row, the layer's distance to stock stays 3.6% / 4.1% - the activation quant
dominates on this family and the rotation cannot be folded into fixed MXFP4 weights. Declared alternative: the
activation row is quantised per 32-wide block, `ka[m, b] = ceil(log2(max_j |x[m, 32b + j]| / 127))` (floored at -126,
raised by one where 127 * 2^ka < max - a binary32 rounding guard), `q = rint(x / 2^ka)` clamped to [-127, 127]; the
block sum `s = sum_j q[m, j] * m2[n, j]` is exact in int32 and the accumulate becomes
`acc = fl32( acc + fl32(s) * 2^(ka[m, b] + kw[n, b] - 128) )` in ascending b - still a power-of-two scaling of an
exactly representable integer, one correctly rounded add per block; the epilogue y = bf16(acc) (+ bias), no per-row
scale. Reference `moe_ref.blockquant / int_mm_mx2 / moe_forward_mx2`. MEASURED (M2C2): layer distance 2.2% / 2.9% (item 5: 3.6% /
4.1%) but chat perplexity 15.254 against item 5's 14.790 and stock's 15.101 - not adopted; item 5 stands.
**Item 5 MEASURED (2026-09-10, M2P4):** chat-templated ShareGPT perplexity of the MXFP4-consuming operand 14.790 vs stock
15.101 (the int8 forms 15.531 / 16.539) - the declared exact-weight evaluation beats the stock kernels' own rounding.
Item 5 is the GPT-OSS contract; its GEMV kernel (`t16moe_mx::mx_gemv`) serves decode rows, the tensor-core form (int8 mma
over 32-wide K slices with the per-block scaling in pinned order) the prefill rows.
The fused item-5 kernel path inside the engine reproduces the reference arming's perplexity to the last digit (M2E3,
raw windows 329.49283843993334 both).
