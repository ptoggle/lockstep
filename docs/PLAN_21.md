# Stage 2, items 2.1 to 2.4: the plan to parity

Written 2026-09-09 from the clean re-baseline (SPEC 7.24). Updated at the end of every stage with
what was measured, what was kept and what was killed.

## What the re-baseline says the gap is

The same stack, against a clean stock on one H100, is 0.5x on a 1.5B, 0.6x on a 3B, 0.9x on a 7B
and 0.95x (1.08x decode) on a 14B. A kernel-throughput deficit is flat across sizes; a fixed cost
per layer is amortised by the layer's width and depth and produces exactly this curve. The
per-layer cost is the number of launches the verifiable layer needs where stock needs about nine
(fused add+norm, qkv GEMM, rotary with the KV write, attention, o GEMM, fused add+norm, gate_up
GEMM, silu_and_mul, down GEMM). The served lockstep layer (contract v2.5, R1 online, the two-launch
online path, `T16M_NRQ=0`, TP=1) issues, per layer and step:

| step | op (`t16mc::`) | kernels | stock's equivalent |
|---|---|---|---|
| 1 | `norm_quant` | integer RMSNorm + residual (`normquant_k`) | fused_add_rms_norm |
| 2 | `r4quant_outl` | the R4 rotation (cluster) + int8 quant + outlier side path (`r4quant_k`) | - |
| 3 | `lin_rope` | int8 GEMM (`torch._int_mm`, cuBLASLt, 1 to 2 kernels below 128 rows) + `deqrope_k` (epilogue, bias, side path, rotary) | qkv GEMM + rotary |
| 4 | attention | the KV prep (int8 block quant; fused into the decode kernel, `DEC8_P=1`) + the decode kernel; on prompts the FA3 seam + the glue gathers | attention |
| 5 | `rowquant_pad` | int8 quant of the attention output (`rowquant_k`) | - |
| 6 | `lin_out` (o_proj) | int8 GEMM + `deq_k` | o GEMM |
| 7 | `norm_quant` | as 1 | fused_add_rms_norm |
| 8 | `r4quant_outl` | as 2 | - |
| 9 | `lin_silu_r4quant` | int8 GEMM + `deqsilu_k` + `r4quant_k` | gate_up GEMM + silu_and_mul |
| 10 | `lin_out` (down) | int8 GEMM + `deq_k` | down GEMM |

About 17 to 20 launches against 9 to 10. Under full CUDA graphs a launch costs no host time, but
each kernel still pays its ramp and tail (3 to 5 us at one row), and a chain of 18 dependent small
kernels is 18 ramps. On a 1.5B a layer step at batch 1 is about 85 us on stock; the extra 80 us
the lockstep layer shows is the ramps and the one-CTA-per-row kernels among them. Every fusion
below composes integer and pinned-fp32 operations that are already declared, so each is gated
bit-for-bit against the composed kernels (t16m_gate) and against the reference on real
activations (E1 / E4), and nothing in the contract changes.

## The stages

Each stage: build, gate (t16m_gate kernels, t16n rig where the decode kernel is touched, E1 / E4 /
E6 on the 7B), measure on the SAME harness as SPEC 7.24 (`s2_rebase*.sh`: single-stream 1.5B / 3B /
7B, serving W1 / W2 / W3 on the 7B, the 14B as the control that nothing regresses at size), file
in SPEC, keep or kill. A stage is kept only if the sweep moves and no gate changes.

**S0, attribution (running).** Per-kernel CUDA time of one decode step and one 2k prefill, stock
against lockstep, on the 1.5B and the 7B at batch 1 and the 7B at batch 32 (`pod/decode_prof.py`,
torch profiler under the real graphs). Output: the launch count per layer, the time per launch,
and the split between the GEMMs, the glue, the attention path and anything outside the graphs
(SPEC 7.17 found +298 ms per saturated step of int64 and copy kernels outside the graphs on the
fused stack). This decides the order of S1 to S3.

**S1, the glue at one launch per stock launch (decode first).** Targets, in the order the profile
will most likely rank them:

- S1a `norm_rot_quant` as a CLUSTER kernel: the residual add, the integer norm, the R4 rotation and
  the int8 quant with the outlier side path in one launch of a 4-CTA cluster per row (the form
  `r4_cl_k` already uses for the rotation), replacing steps 1+2 and 7+8. The one-CTA-per-row form
  (`normrotquant_k`, P2) is gated but slower because one CTA carries the 64-bit chain over the
  whole row; the cluster spreads it. Saves 2 launches per layer.
- S1b the down epilogue into the next layer's norm: `lin_out` (down) returns the int32
  accumulator and the next layer's step 1 consumes it (`deq_norm_quant` exists for the python path;
  the layer boundary carries acc + scales instead of bf16). Saves 1 launch; also removes one bf16
  round trip of the hidden state.
- S1c `deqsilu_k` + `r4quant_k` as one cluster kernel (the single-kernel form `deq_silu_r4quant`
  exists and is slower for the same one-CTA reason). Saves 1.
- S1d the KV prep into `deqrope_k`: the rotary epilogue already produces k and v; writing the int8
  block-quantised pages there removes the prep from the attention path. Saves 1 to 2, and the
  decode kernel's fused prep becomes unnecessary work to drop.
- S1e the o_proj epilogue + residual + norm + rot + quant as one launch after the GEMM
  (`lin_norm_rot_quant` exists under `T16M_NRQ=1`; it becomes the cluster form of S1a fed by the
  accumulator). Folds steps 6+7+8 into GEMM + 1.

Expected: 17 to 20 launches down to 10 to 12. Kill line: a fusion that does not move the 1.5B's
decode by at least its launch's share is dropped.

**S2, the int8 GEMM below 128 rows.** Below `T16M_MCUT` the route is `torch._int_mm` (cuBLASLt),
whose small-M kernels are not tuned for one row, followed by a separate epilogue. If S0 charges the
GEMMs more than stock's bf16 GEMMs at batch 1 (the 14B's 1.08x decode says the int8 weight read
wins at size; the 1.5B's 0.5x says something else dominates there), build an int8 GEMV / small-M
GEMM with the declared epilogue fused (int32 exact accumulation is order-free, so any correct
integer kernel is bit-identical to the contract; the epilogue is the pinned fp32 pair). Kill line:
not faster than `_int_mm` + epilogue at M = 1 and 8 on the 1.5B and 7B.

**S3, outside the graphs.** Whatever S0 shows between the graphs (index and copy kernels, the
per-step buffer pins of the attention backend, host syncs). These are per step, not per layer,
and hit serving hardest (7.17). Remove or move into the graph.

**S4, serving.** The multi-sequence decode CTA (7.17: one CTA serves several short sequences with
one prologue) for chat-length contexts, and the mixed prefill+decode step (7.13: the portable pass
sized from the prompt chunk, fixed in T16j; re-profile under the current stack). Measured on
W1 / W2 / W3.

**S5, prefill.** After decode: the glue at large row counts (the cluster kernels are already the
right shape there), the seam at 0.87x at 32k (2.3's wgmma-native lookup).

**S6, Qwen3 (item 2.2).** The per-head q / k RMSNorm sits between the qkv GEMM and the rotary,
which is exactly where `deqrope_k` runs; an integer RMSNorm over 128 elements per head inside that
epilogue (the contract's norm, SPEC 15, on a 128-wide row) puts the family on the fused layer.
Measured against the 7.24 rows (0.45 to 0.56x TTFT, 0.72x decode).

## Parity, defined

Close to parity means, on the same clean harness: the 7B at or above 0.95x on single-stream TTFT
from 2k, decode at batch 1 and 8, and serving W1 / W2 / W3; the 3B at or above 0.90x; the 1.5B at
or above 0.85x; the 14B not below its 7.24 row; Qwen3-8B within 5 points of the 7B; every gate
unchanged and every fingerprint 0 bits.

## Log

- 2026-09-09: plan written; S0 launched on lc-handover (`s21_prof_h100.sh`).
- 2026-09-09, S0 measured (`results/s2/perf/prof/`, per token per layer, torch profiler inside the graphs;
  KERNEL time only - the wall clock adds the gaps between dependent kernels, which is where the
  launch count is paid). Qwen2.5-1.5B decode at batch 1: stock 78 us of kernels per layer
  (bf16 GEMMs 45, attention 12, glue 12), lockstep 121 (int8 GEMMs 48 through cuBLASLt's sm80
  CUTLASS kernel at one row, the decode kernel 23, glue 37.5: r4quant 10.2 for two launches,
  normquant 8.2 for two, deqrope 5.2, r4_cl 5.1, deqsilu 4.4, deq 2.8 for two, rowquant 1.6);
  wall 2.38 ms against 4.7 ms per token, so about 1.1 ms of the 2.3 ms gap is inter-kernel gaps
  (18 to 20 launches against 10). Qwen2.5-7B decode at batch 1: lockstep kernel time is BELOW
  stock's (5.68 against 5.95 ms per token: int8 GEMMs 125 against 165 us per layer) and the wall
  clock is 0.94x, so on the 7B the whole gap is the launch count. 7B decode at batch 32: the glue
  is 120 us per layer against stock's 26 (deqsilu 45.8, r4quant 39.2, normquant 17.7, deqrope
  10.8: one CTA per row, 32 CTAs on 132 SMs, each a long chain) - item 2.4 is the glue at mid M.
  7B prefill at 2k: int8 GEMMs 749 against stock's 1,246 us per layer (1.66x), the seam 129
  against 73, the glue 998 against 138 (deqsilu 475, r4quant 287 for three, normquant 97 for two,
  deqrope 78, the KV prep 51): the glue at large M is latency-bound (one CTA of 512 to 1,024
  threads per row with three block reductions; 2,048 CTAs at about two resident per SM), and with
  it at stock's level the 7B's TTFT would be ABOVE parity on the GEMM gain alone. 1.5B prefill at
  2k: glue 555 against 71 us per layer, GEMMs 164 against 259.
  ORDER, from this: (1) the glue at every M - a tiled, bandwidth-bound form for M > 16 (row-tile
  x 256-block tiles, the per-row amax as an order-free first pass) for deqsilu / r4quant /
  normquant / deqrope, which is TTFT, batch-32 decode and serving at once; the cluster fusions
  (S1a) for M <= 16, which is batch-1 decode; (2) an int8 GEMV for M <= 16 (S2: 48 us against a
  bf16 read of 45 on the 1.5B, half the bytes for the same time); (3) the decode kernel's fixed
  cost (23 us at a 300-token context on both models against FA3's 12, S4); (4) the seam (S5).
- 2026-09-09, S1a measured. Gate: t16m kernels 1,186 / 1,186, P2 600 / 600, the cluster form
  bit-identical at every row count (probe). Engine: NO change on the 1.5B or the 7B - the fused op
  never ran: the Python branch `0 < M <= 16` on `x.shape[0]` is baked by dynamo / graph capture at
  trace time, so the two-launch path stayed in every graph. The route now lives in the C++ launcher
  (warp form <= 64 rows, cluster form, the two gated kernels above that). The probe also says WHY
  a launch merge alone cannot pay: at one row the cluster form takes 8.8 us where the two kernels
  take 8.3 us (1.5B width) to 10.2 us (7B width); the kernels are latency chains (three block
  reductions, the 64-bit norm, the butterfly), not launch overhead; stock's fused_add_rms_norm is
  2.2 us. So S1a-v2: a WARP form (one to four warps per row, shuffle reductions, no block barrier),
  and S3 brought forward: programmatic dependent launch (`T16M_PDL=1`) on every glue kernel, since
  the 1.5B's wall clock (4.7 ms per token) exceeds its kernel time (3.4 ms) by 1.3 ms - about
  2.3 us of gap per kernel, where stock shows 0.6 us: stock's cuBLAS / FA3 kernels overlap their
  launch under PDL and ours did not. Both queued (S1h).
- 2026-09-09, S1f built: the outlier side path re-read wo[n, 0..ko) per element (32 bytes of gathers
  per 6 bytes of payload); deqsilu_t_k / deqrope_t_k hold the column's weights in registers across
  8 rows. Two runs lost to the pod: the overlay disk filled with manifests (nvcc segfaults with 61 MB
  free) and the volume hit its quota again; manifests now live on the volume behind a symlink.
- 2026-09-09, the hang: six hours of a "hung" gate were a host infinite loop - the regex that turned
  every `X<<<g,b,s,st>>>(` into `launch_k(X, ...)` also rewrote the launch INSIDE launch_k into a call to
  itself (bisected by source version: original / pre-PDL / plain-launch sites; the standalone test
  showed griddepcontrol.wait is a no-op in a plain launch, so the PDL theory was wrong). Fixed; gate
  1,186 / 1,186 with every new kernel present.
- 2026-09-09, S1a-v2 (warp form) KILLED: 31.8 us at one row against 8.7 us for the two kernels at the
  1.5B width - one warp serialises the row's 64-bit chain (48 elements per lane, 256 bytes of stack);
  the block form's 512 threads beat barrier latency. Default off. The fused op keeps the cluster form
  (<= 16 rows, neutral in kernel time, one launch fewer) and the two kernels above that.
- 2026-09-09, S1h / S2 measured (`handover/s1h`, `handover/s2`; PDL on, cluster norm <= 16, S1f epilogues,
  then + GEMV <= 16). 7B TTFT: 0.92 / 0.995 / 0.93 (S1h) and 1.01 / 1.08 / 0.98 (S2) at 2k / 8k / 32k,
  2k x8 0.98 / 1.07, 128 x32 0.98 / 1.08 - the 7B's PROMPTS ARE AT OR ABOVE PARITY with the tiled
  epilogues in (from 0.86 / 0.90 / 0.87). 7B DECODE collapsed: 0.59 (S1h) and 0.77 / 0.51 / 0.80 (S2)
  from 0.94 / 0.94 / 0.86; 1.5B decode 0.30 / 0.45 from 0.51. Cause, from the profiles: with PDL every
  glue kernel triggers its dependents at entry, so the next kernel's whole grid is resident and
  spinning at griddepcontrol.wait while the GEMM runs (the GEMV read 146 us per layer of "duration",
  the cluster norm 79, deqrope 66: waiting time, not work) - the classic PDL pitfall; the trigger has
  to sit after the main loads, not at the top, or PDL stays off. S1j separates PDL / cluster / GEMV.
  GEMV probe (bit-equal on every shape): at M = 1 it beats cuBLASLt on the 1.5B (29.7 against 42.8 us
  for the four linears) and on the 7B's qkv / o (7.3 against 9.4) but loses on the 7B's wide
  (gate_up 55.7 against 51.6) and deep (down 75 against 40.6) shapes, and from M = 4 it loses
  everywhere (MT dp4a per 4-byte load): too few CTAs (8 for the 1.5B's qkv, 112 for the 7B's down)
  and no loads in flight. GEMV v2: rows per warp by N (1 / 2 / 4, about 1,000 warps), the k loop
  unrolled 4, routed at M <= 4 only.
- 2026-09-09, S1i measured (`handover/s1i`; all on, PDL still the entry-trigger form). The staged
  rotation: 41 us at 2,048 rows against 95 (one-CTA), 8.2 against 14.9 at 256, equal at 64, SLOWER at
  32 (50 against 39 in the batch-32 engine profile) - routed from 128 rows, the cluster form up to 32.
  Prompts, all three sizes, against the 7.24 rows: 7B 1.01 / 1.06 / 0.98 (from 0.86 / 0.90 / 0.87),
  3B 0.78 / 0.87 / 0.83 (from 0.59 / 0.67 / 0.69), 1.5B 0.73 / 0.82 / 0.78 (from 0.49 / 0.53 / 0.66) at
  2k / 8k / 32k. 7B 2k-prefill kernel time 44.7 ms per token (was 53.3; stock 41.4); what remains there:
  deqsilu_t 300 us per layer (stock's silu_and_mul 42), r4quant_s 158 for three, normquant 97 for two,
  the KV prep 74, the seam 128 against 73. Decode still broken by the entry-trigger PDL (7B 0.76 /
  0.51 / 0.80); S1j / S1k separate it. Next in the glue: the epilogues' loads hoisted (RT x 2 in flight),
  normquant at 256 threads from 64 rows.
- 2026-09-09, S4a: the standalone rig's step time includes a host sync (0.19 ms floor at batch 1, 300
  tokens), so it cannot resolve the 23 us the engine profile charges to `lsb_dec8p` per layer at that
  context; the split count does not matter below 2k (0.19 to 0.25 ms) and matters from 32k (1.25 ms at
  1 split, 0.29 at 16). The fixed cost has to be read inside the kernel (the `LSB_DEC_TIMING` build's
  per-CTA cycle counters) - deferred behind the glue work; on the 7B it is 5 percent of a layer step.
- 2026-09-09, a gate regression: from the S1j build on, `deq_silu_r4quant(y8)` (the fused one-CTA
  SiLU + rotation op fed bf16 instead of the accumulator) differs from its accumulator path, at every
  row count, and `lin_gate_up` with SILU_FUSED on follows; `deq_silu(wide)` (the tiled epilogue, both
  inputs) and `r4quant_pad` pass. The kernel's two branches are unchanged since the passing build
  (817bcdf) apart from an inert trigger statement; bisect T1 (trigger compiled out) and T0 (817bcdf
  rebuilt) running.
- 2026-09-09, S1j measured (`handover/s1j`, gate 1,186 / 1,186; one stock arm per model, each change alone):

  | 7B, ratio | TTFT 2k | 8k | 32k | 2k x8 | decode 1 | 8 | 32 |
  |---|---|---|---|---|---|---|---|
  | base (tiled epilogues, staged r4quant, norm 256, loads hoisted) | 0.99 | 1.09 | 0.98 | 1.07 | 0.93 | 0.91 | 0.84 |
  | + PDL (implicit trigger) | 1.02 | 1.09 | 0.98 | 1.07 | 0.93 | 0.89 | 0.82 |
  | + cluster fused norm | 1.00 | 1.09 | 0.98 | 1.07 | 0.88 | 0.86 | 0.82 |
  | + GEMV v2 (rows <= 4) | 1.03 | 1.10 | 0.98 | 1.07 | 0.97 | 0.92 | 0.84 |
  | all three | 1.01 | 1.10 | 0.99 | 1.06 | 0.92 | 0.85 | 0.82 |

  1.5B: base 0.70 / 0.82 / 0.80 TTFT and 0.52 / 0.50 / 0.50 decode; + GEMV 0.76 / 0.85 / 0.81 and
  0.59 / 0.51 / 0.49; PDL and the cluster norm 2 to 4 points worse on decode.
  KEPT: the tiled epilogues, the staged rotation, the norm's 256-thread form, the GEMV at rows <= 2
  (the v2 probe: 1.5B 17.8 against 42.9 us for the four linears at one row, 7B 99.7 against 111.5;
  at four rows the 7B's gate_up / down lose). KILLED: PDL in every form; the cluster fused norm (a
  cluster launch inside the graphs costs more than the launch it saves); the warp form. The 7B's
  prompts are at parity or above and its batch-1 decode at 0.97x; what remains on the 7B is decode at
  batch 8 to 32 (the glue at 8 to 32 rows, the decode kernel's 28 us) and the 128-token prompt (0.80x:
  a 128-row prefill is one wave of every kernel). The 1.5B is at 0.76 to 0.85x on prompts and 0.59x
  on decode: its decode is now the attention kernel (23 of about 95 us per layer) and the glue's
  latency floor (about 27 us of 4 to 5 us kernels) - S4 and a lower-latency one-row norm are next.
- 2026-09-09, S1l: ENGINE GATES PASS on the 2.1 build (cluster norm on, GEMV off, every other change in):
  E1 prompt rows 151,200 bad=0, E4v2 generated rows 29,736 bad=0 against lssab9_torch[sink](mu), E6 full
  graphs == eager. The norm's 256-thread form: 20.4 against 27.5 us at 2,048 rows (1.5B width), 43.0
  against 48.7 (7B); a little slower at 64 rows - routed from 256. S1m repeats the engine gates with the
  KEPT defaults (GEMV at rows <= 2, cluster norm off) and measures the kept set on the sweep + serving.
- 2026-09-09, S1m, the kept set as defaults (`handover/s1m`): engine gates PASS on the defaults (E1
  151,200 / E4v2 29,736 rows bad=0, E6); 7B TTFT 0.84 / 1.01 / 1.08 / 0.98 (128 / 2k / 8k / 32k),
  decode 0.96 / 0.91 / 0.83, serving 0.78 / 0.77 / 0.88; 3B TTFT 0.79 / 0.88 / 0.84, decode 0.70 /
  0.60 / 0.59; 1.5B TTFT 0.67 / 0.83 / 0.71, decode 0.59 / 0.51 / 0.50. Filed as SPEC 7.25.
- 2026-09-09, S4c (the LSB_DEC_TIMING build, `handover`/`s21_s4c`): the decode kernel at batch 1 and a
  300-token context spends 37,600 cycles per CTA (21 us at 1.75 GHz, the 23 us the engine profile
  charges): prologue 11,160 (30 percent), the segment partials 8,300 (22), the phase-1 chain 7,870 (21),
  the phase-2 PV 4,470 (12), cp.async waits 610; three 128-key chunks per CTA, about 6,900 cycles per
  chunk. At 2k: 147,000 cycles, 17 chunks, 7,400 per chunk. The engine already splits the chunks
  across CTAs (auto_split_nt: 3 at 300 tokens), so the fixed cost is the prologue plus one chunk plus
  the combine - the prologue is the target (6.4 us: what it does is read next).
- 2026-09-09, the decode kernel's prologue, read: the tile list entry -> the sequence's length ->
  its page-table row -> the 7 query rows, the per-row eq / nb, and the last block's page -> its value
  exponent: four to five dependent global loads (about 1 us each at one row) before the first chunk
  can start, then one __syncthreads; the engine splits the chunks across CTAs (nsplit = 3 at 300
  tokens), so a layer step is prologue + one chunk + the last-CTA combine. Remedy (S4d, not built): a
  per-tile header written by the prep kernel (length, row0, eq / nb / evr, page of the last block) so
  the decode kernel's prologue is one load; metadata only, the arithmetic untouched. Worth 3 to 4 us
  of the 23 per layer at batch 1 - 4 percent on the 1.5B, 2 on the 7B - so it sits behind the
  mid-row glue (S1n2) and the short prompt in the order.
- 2026-09-09, P128 (`handover/p128`): the 128-token prompt on the 7B (0.84x): lockstep 8.48 ms of
  kernels against stock's 6.62. Per layer: the int8 GEMM 112 us against bf16's 182 (a 70 us gain);
  the KV prep `lsb_prep` 47 us - a FIXED cost (51 at 2k, 74 at the seam's 2k profile): 1.3 ms per
  prompt whatever its length, 14 percent of a 128-token prompt; the seam 31 against FA3's 13 (an
  18 us fixed cost of its own); the norm 13.9 per call at 128 rows against 2.7 (the 512-thread form;
  the 256-thread form from 256 rows - to try from 128); deqsilu 24 against 6.3; the rotation 7 x 3;
  deqrope 9.1; the gathers 9.4. Batch-8 decode: kernel time at parity (6.54 against 6.71 ms per step)
  and the wall clock 0.91x - the launch gaps again, about 1 us per kernel; without PDL the only
  remedy is fewer kernels (the o_proj epilogue into the norm saves one; the rest is structural).
  ORDER: the KV prep's fixed cost first (it is also 2.6 percent of every 2k prompt and part of the
  seam's fixed cost at chat-length serving), then the norm's form at 128 rows.
- 2026-09-09, S1n2 (`handover/s1n`, the 7B width, the SiLU width for the rotation): at 8 to 16 rows the
  cluster rotation (6.0 to 6.7 us) and the 512-thread norm (5.2) stay; at 32 rows the staged rotation
  (10.4) beats the cluster (11.9); at 64 the one-CTA form (9.7) edges the staged (10.4) and the cluster
  is 17.4. Defaults: staged rotation from 32 rows, cluster to 16, the 512-thread norm below 256 rows.
  The tiled SiLU epilogue is 5 to 6 us at 8 to 32 rows (stock 4.5) and 9.6 at 64 (stock 4.7). These are
  the batch-8-to-32 decode glue: 2 to 3 us above stock per kernel, six kernels - about 15 us of the
  layer's 230; the batch-8 wall gap (0.91x at kernel parity) is the launch gaps, not these.
- 2026-09-09, S5a, T16y KEPT (`handover/s5a`): the warp-per-row KV prep passes the rig (21 / 21 on both
  warp counts, head groups 2 / 2, the 262k / 524k splits 2 / 2), E1 151,200 rows bad=0, E4v2 29,736 rows
  bad=0, E2 chunked == single prefill. 7B in the same session, T16y against the old kernel: the 128-token
  prompt 0.877 against 0.825, 128 x8 0.994 against 0.949, 2k 1.016 against 0.992; everything from 8k
  unchanged (the prep is 2 to 3 percent there). Promoted to `lssa_lssab.so` on lc-handover (the old kept
  as `lssa_lssab_pre_t16y.so`); the repo's `lssa/portable/lssa_lssab_t13b.cu` and its top-level copy
  carry it (16cffde).
- 2026-09-09, S1p (`handover/s1p`): lin_norm_h (the o_proj epilogue folded into the norm) gates clean
  (kernel 1,186 / 1,186, E1 151,200 / E4v2 29,736 rows bad=0, E6) and is SLOWER: 7B decode 0.93 / 0.89 /
  0.80 with it against 0.96 / 0.90 / 0.82 without, prompts equal, 1.5B equal - the dequant of a whole
  row now runs inside the norm's one CTA per row (the separate deq_k spread it over many CTAs) and the
  norm reads int32 instead of bf16. KILLED (kept as `T16M_LNH=1`). A launch saved is not a win when the
  merged kernel is slower than the pair: the same lesson as the cluster fused norm.
  The 7B's rows in this session with every kept default and the T16y prep: TTFT 0.88 / 1.03 / 1.08 / 0.99
  at 128 / 2k / 8k / 32k, 2k x8 1.07, 128 x32 1.08; decode 0.96 / 0.90 / 0.82.
- 2026-09-09, S1q (`handover/s1q`, the kept set + T16y): 3B TTFT 0.68 / 0.80 / 0.89 / 0.84 at 128 / 2k /
  8k / 32k (from 0.63 / 0.79 / 0.88 / 0.84), decode 0.67 / 0.59; 7B serving W1 / W2 / W3 0.76 / 0.79 / 0.90
  (S1m: 0.78 / 0.77 / 0.88 - inside the harness band; W3 up 13 points from the 7.24 row of 0.77). The
  prep rewrite shows where prompts are short: the 128-token rows.
- 2026-09-09, S1r (`handover/s1r`): the epilogue tiles at 4 rows per thread, gate 1,186 / 1,186: the tiled
  SiLU epilogue at the 7B's SiLU width 4.2 us at 16 rows (stock 4.5), 7.8 at 64 (was 9.5), 12.1 at 128
  (13.6), 24.8 at 256 (27.8), 158 at 2,048 (167 with the 8-row tile; stock 79). KEPT as the default at every
  row count. In the engine the 7B's 128-token and 2k prompts sit inside the noise of the two settings.
  The SiLU epilogue at many rows is now the largest glue item at prefill (2x stock's): its per-element
  chain (the table SiLU, the side path) rather than its tiling.
- 2026-09-09, the gaps MEASURED from the CUDA event timestamps (`handover/gaps`, 7B, 64 decode tokens
  after a 300-token prompt, both stacks in full graphs - vLLM's log shows FULL and PIECEWISE captures for
  both): per token, stock 393 kernels / lockstep 443 (13 percent more, not twice); busy time lockstep
  BELOW stock at every batch (5.45 against 5.96 ms at b1, 6.42 against 6.72 at b8, 9.11 against 9.20 at
  b32); the gaps between consecutive kernels 1.19 / 1.16 / 1.06 ms per token against stock's 0.43 /
  0.37 / 0.24 - a MEAN gap of 2.4 to 2.7 us against 0.66 to 1.13; and stalls of 200 us or more (host
  side) 0.77 / 0.88 / 1.24 ms per token against stock's 0.01 / 0.01 / 0.07. So the batch-32 wall gap is
  NOT the kernels (at parity) and not mainly their number: it is (a) three to four times the per-launch
  gap and (b) about 1 ms of host-side stall per step that stock does not have. Both are attributable
  and neither needs a kernel rewrite: the host profile (cuda runtime calls per token) and the gap-by-
  kernel profile are queued. The 2.1 remainder is therefore a backend / launch-path item, not a
  kernel-latency floor as stated earlier today.
- 2026-09-09, the per-step kernels (from the filed S0 profiles, calls that are multiples of the step
  count): stock runs its lm_head as ONE GEMM per step; the fused int8 lm_head (B2'') runs, per step and
  outside the CUDA graphs (vLLM does not capture compute_logits), about fifteen eager torch ops - the
  row abs / amax / clamp / div / round / clamp / cast, the plane shift / sub / two casts, two cats, the
  stacked GEMM and lm_combine - each a 10-to-30 us host issue for a few microseconds of GPU work. That
  is the shape of the ~1 ms host-side stall per step the gap measurement found. S3c: `lm_prep`, the row
  quant + plane split in one launch straight into the stacked operand (three launches per step);
  queued with its gate (the lm_combine cases compare the fused logits against the torch reference),
  the host / gap profile on and off, the decode rows and E6.
- 2026-09-09, S3b (`handover/s3b`): PDL v3 (the trigger after each glue kernel's loads / reductions,
  the two suspect kernels excluded) passes the kernel gate both without and WITH the launch attribute
  (1,186 / 1,186 twice). Decode, one session, PDL on against off: 7B 0.977 / 0.931 / 0.850 against
  0.934 / 0.922 / 0.840 at batch 1 / 8 / 32; 1.5B 0.616 / 0.521 / 0.506 against 0.576 / 0.513 / 0.529;
  prompts within noise (128 tokens -2, 2k +1). A real but modest gain at batch 1 (+4), one point at 8
  to 32. Held as a candidate default until the engine gates run under it (S3d, with lm_prep).
- 2026-09-09, the host side of a stock decode step (`handover/host`, 7B b8): per token one cudaGraphLaunch,
  14 eager cudaLaunchKernel (the lm_head GEMM, the sampler, the input-prep kernels), 5 cudaMemcpyAsync,
  and the inter-kernel gaps 0.25 ms per token at a mean of 0.84 us. The lockstep arm of this and of the
  gap-by-kernel profile was lost to a job that copied the python without the kernel source (the cached
  extension lacked the new op); both are re-queued behind S3d with the lm_head prep on and off, which is
  the comparison that matters: the stock count of 14 eager launches per token is the target for the
  lockstep step's eager section.
- 2026-09-09, the gap before each kernel, stock (`handover/gapby`, 7B b8 / b32): inside the graph every
  kernel launches 0.45 to 0.55 us after its predecessor (the FA3 kernels, the rotary, the norms, the
  bf16 GEMMs alike); the exceptions are the split-K GEMM's launch (4.1 us at b8, once per layer) and the
  device memsets (1.5 us, 29 per token). So the graph engine's floor is about 0.5 us per dependent
  node; the lockstep mean of 2.4 to 2.7 us per gap (S0 / the gap measurement) is three to five times
  that, and the lockstep arms of this profile (queued again with the kernel source in place) will say
  which of its kernels carry it - cluster launches, the dynamic-shared-memory kernels, or the eager
  section around the lm_head.
- 2026-09-09, S3c (`handover/s3c`): `lm_prep` gates clean (1,186 / 1,186; E6 full == eager). The host side of
  a lockstep decode step, with the torch lm_head path: ONE cudaStreamSynchronize per step (5.6 ms of
  host wait per 64 steps' worth) and 34 eager launches per step against stock's 14 - the stall the gap
  measurement saw; with `lm_prep`: no stream sync, 20 eager launches, the mean in-graph gap 0.63 us
  (stock 0.84 to 0.97), host stalls 0.9 ms per 64 steps (stock 0.8 to 1.2), busy 6.42 ms per step
  against stock's 6.72. By the profiler's accounting the batch-8 step is now at or below stock's; the
  bench still reads 0.92x (0.921 against 0.919 without) - so the bench's decode rows are measuring
  something the 300-token profile does not: the decode kernel at the bench's longer context is the
  candidate (S4c: 7,400 cycles per 128-key chunk, 147,000 per CTA at 2k against 37,600 at 300).
- 2026-09-09, S3d (`handover/s3d`): with lm_prep on, PDL v3 under the ENGINE gates: E1 151,200 rows
  bad=0, E4v2 29,736 rows bad=0, E6 full == eager - all under T16M_PDL=1. Decode, PDL on against off,
  one stock arm per model: 7B 0.983 / 0.943 / 0.847 against 0.941 / 0.929 / 0.854 (batch 1 / 8 / 32);
  3B 0.738 / 0.618 / 0.597 against 0.706 / 0.623 / 0.627; 1.5B 0.618 / 0.520 / 0.503 against 0.561 /
  0.521 / 0.525; prompts: 128 tokens -2 points, 2k and 8k +1 to +2. KEPT as the default: +3 to +6 at
  batch 1 on every size, +1 at batch 8 on the 7B, -1 to -3 at batch 32 (inside the band on the 7B).
  The 7B, all kept defaults: TTFT 0.87 / 1.03 / 1.09 (128 / 2k / 8k), decode 0.98 / 0.94 / 0.85.
- 2026-09-09, the replacement profiles (`handover/prof2`, lockstep arms with the kernel source in place).
  Batch 32 decode, lockstep with `lm_prep`: 426 kernels per step, busy 9.197 ms against stock's 9.234,
  in-graph gaps 0.256 ms (stock 0.233) at a mean of 0.60 us, host stalls 2.3 ms per 64 steps (stock 1.4).
  The gap before each kernel is 0.46 to 0.53 us for every lockstep kernel (the int8 GEMMs, the rotation,
  the norm, the epilogues, the decode kernel) - the same floor stock sits on. So at batch 32 the
  profiler also accounts the lockstep step at or below stock's GPU time, while the bench reads 0.85x;
  the MLEN test (engine max_model_len / memory fraction, both stacks, both settings) is the pending
  discriminator between an engine-configuration effect and a real cost the 300-token profile misses.
  Saturated chat (W1, 128 requests, 64 sequences): lockstep 4,373 ms of GPU for 3,768 out tok/s against
  stock's 3,584 ms and 4,658 tok/s (0.81x); `lm_prep` on against off is +2.7% tok/s. The decomposition
  is the first clean account of the serving gap, 789 ms: int8 linears 2,284 ms against bf16 2,905
  (-621); attention 618 decode kernel + 102 seam glue + 51 B8 seam = 771 against FA3's 410 (+361);
  the fused glue 935 against stock's norm/rope/act 215 (+720: r4quant 276 ms at 7.3 us per launch, deq_silu
  233 at 16 us against act_and_mul's 5.9, normquant 203 at 7.2 against 4.6, deqrope 89); and
  "sampling / misc torch" 334 against 25 (+309) - int64 elementwise kernels of 40 to 60 us each
  (CUDAFunctor_add<long> x1,542, lshift<long> x2,056, direct_copy x2,570), three to five per step, i.e.
  index arithmetic on tensors of millions of elements in the seam's Python planning, not a kernel at all.
  Queued (`s21_w1shape`): the same profile with input shapes and call sites to name the op, and the glue
  microbench at serving row counts (64 / 320 / 1,088 / 2,112). The deq_silu figure is close to its byte
  floor: at 2k rows it reads 2 x 18,944 int32 per row (310 MB) where act_and_mul reads bf16 (155 MB).
- 2026-09-09, MLEN (`handover/mlen`, 7B, both stacks, one process per stack, REP=3): the engine
  configuration is NOT the discrepancy. At max_model_len 8,192 / memory fraction 0.4 (the profile
  harness's setting) the bench reads TTFT 0.873, decode 0.987 / 0.952 / 0.870 (batch 1 / 8 / 32);
  at 32,768 / 0.85 (the bench's) 0.878, 0.971 / 0.928 / 0.851 - two points, inside the band. So the
  bench's batch-8 and batch-32 rows measure a cost the profiler does not see. Reading the two
  accounts together: the bench's stock step at batch 32 is 6.3 ms of wall (5,074 tok/s), the
  profiler's 9.2 ms of busy - the profiler's own per-kernel overhead (about 2 ms on a 426-kernel step)
  inflates BOTH arms' GPU time, and a step whose GPU time is inflated by 2 ms can no longer be
  host-bound. The hypothesis that fits every number: at short context the lockstep arm's step is
  HOST-bound (the backend's Python planning, ctypes argument packing and NumPy metadata per step),
  which the profiler masks. Test queued (`s21_hoststep`, `host_step_prof.py`): the bench's own engine
  configuration, no profiler, GPUModelRunner.execute_model wrapped with perf_counter and a CUDA event
  pair - host time inside the step, host time between steps, GPU time entry to exit, both stacks,
  batch 32 and 8.
- 2026-09-09, the shape profile (`handover/w1shape`, W1 lockstep with input shapes and call sites): the
  "misc torch" bucket is the LM_HEAD. Every large int64 op is on [rows, 152064] - the vocabulary -
  `aten::to(int64)`, `__lshift__`, `add` and `copy_` on [64, 152064] (and 63, 50, 44, 31, 30 rows: the
  number of sequences sampled that step), i.e. `t16h_lmvariants.Variant.logits`, the torch plane
  combine `_int_mm(...).to(int64) << shift` summed over the four plane products, plus its two fp32
  scalings `* ws` and `* asc`. That is the path `lm_prep` / `lm_combine` replaced - and the W1 profile
  with `T16M_LMPREP=1` and `=0` gave the same GPU time (4,373 against 4,362 ms), so in SERVING the fused
  lm_head route is NOT taken: t16m's `_get_logits` wrapper falls through to t16l's torch path (its
  guard: an int variant, no SmoothQuant scale, no embedding bias, unit scale), or t16l's hook is
  installed after t16m's. In the decode profile harness (S3c) the same route DID engage (34 to 20 eager
  launches). Queued: the live hook's identity and `STATS["lm_fused_calls"]` printed at the end of
  the W1 run and of the host-step probe. If the fused route engages in serving, the bucket shrinks
  from 334 ms toward stock's 25 - about 7% of the lockstep serving GPU time, for free.
- 2026-09-09, the glue at serving row counts (`handover/w1shape/glue_mixed`, 7B widths, us per launch,
  lockstep against the stock kernel it replaces): norm_quant 5.0 / 9.2 / 22.1 / 38.9 at 64 / 320 /
  1,088 / 2,112 rows against rms_norm 2.0 / 2.9 / 5.3 / 8.0 (2.5x to 4.9x); r4quant 13.8 / 26.0 / 60.3 /
  100.8 (no stock counterpart; its byte floor at 2,112 rows is about 9 us: 15 MB of bf16 in, 7.5 MB of
  int8 out); deq_silu(tiled) 10.8 / 34.0 / 96.7 / 180.0 against silu_and_mul 4.7 / 11.5 / 44.3 / 81.0
  (2.2x, and at or near its byte floor: int32 in where stock reads bf16). So the norm and the rotation
  are 4x and 10x off their byte floors at many rows - the compute is small (the rotation is 80 fp32
  ops and 40 shuffles per 8 elements; ~5 us of an H100 for 2,112 rows), so the loss is latency and
  residency: one CTA per row, three block reductions in series, a dependent chain per row of ~10 us,
  and only a few CTAs resident. Queued (`s21_ncu`): Nsight Compute on r4quant_s_k, normquant_k,
  deqsilu_t_k and the two stock kernels at 2,112 rows - registers per thread, achieved occupancy,
  DRAM throughput, warp stall reasons - before any kernel is rewritten.
- 2026-09-09, S5a ROOT CAUSE of the profiler-against-bench discrepancy, found by reading the arming
  code once the shape profile had named the op: `t16l_stack._install` is not idempotent, and every
  harness that sets T16L_ARM=1 arms TWICE - once itself (`t16m_stack.arm()` before the engine is
  built) and once through the vLLM plugin's load hook. The second `arm()` runs t16l's installer again,
  which puts t16l's `_get_logits` (the torch int64 plane combine, `Variant.logits`, with its per-step
  stream sync) ON TOP of t16m's fused wrapper; t16l's hook never calls what it wrapped, so the fused
  lm_head route is dead in every t13b_bench and t14_bench run to date, and in W1. The decode profile
  harness arms once (it does not set T16L_ARM), which is why S3c saw `lm_prep` take effect there and
  the bench did not (0.921 against 0.919). FIX (t16m_stack._install_lm_fuse): the test is "is our
  wrapper the live hook" (a marker attribute on the function), not "did we install once"; if anything
  sits above it, wrap again - the fall-through still reaches whatever is there. What the bench has
  been paying per decode step at batch 32 for this: 4 int64 conversions, 4 shifts, 3 adds and 2 fp32
  scalings on [32, 152064] plus one host sync, i.e. most of the 0.94 ms per step between the profiler's
  account and the bench's. Re-measurement queued behind the probes: kernel gate, E6 full == eager in
  the bench's own double-armed configuration, the 7B bench (all rows, both stacks) and serving W1-W3.
- 2026-09-09, S5a fix confirmed live (`handover/hoststep2`): in the bench's configuration the live
  hook is now `t16m_stack._install_lm_fuse.<locals>.get_logits`, `lm_fuse_installs` = 2 (the wrapper
  re-asserted itself after the plugin's second arm) and `lm_fused_calls` = 1,540 - every decode step of
  the run; in a W1 run 518 of 518 steps. Stock's live hook is vLLM's own. The host-against-GPU probe
  itself recorded no calls (the wrapped `GPUModelRunner.execute_model` is not the entry vLLM uses on
  this build) and is dropped: the root cause was found by reading the arming code, and the bench
  re-measurement is the test that matters.
- 2026-09-09, S5a MEASURED (`handover/s5a`, `t14_res/*S5A*`; gate 1,186 / 1,186, E6 full == eager,
  both under the double-armed configuration). 7B, one process per stack, REP=3: TTFT 0.89 / 1.00 /
  1.04 / 0.97 at 128 / 2k / 8k / 32k (batch 8: 0.95 / 1.05 / 1.04; 8k batch 4 1.04; 128 x 32 prefill 1.00);
  decode 1.13 / 1.09 / 1.03 at batch 1 / 8 / 32 (186.4 / 1,401.5 / 5,238.6 against 165.0 / 1,287.5 /
  5,078.9 tok/s); serving W1 / W2 / W3 output throughput 0.86 / 0.92 / 0.95 (from 0.76 / 0.79 / 0.90),
  mean TTFT 0.82 / 0.92 / 0.96. The batch-32 "structural floor" is withdrawn in SPEC 7.25; every
  other family's decode and serving rows were measured with the dead route and are lower bounds.
- 2026-09-09, S6a BUILT (serving's glue at many rows). Nsight Compute is refused on the pod
  (ERR_NVGPUCTRPERM: the container has no GPU performance-counter permission), so the design is from
  the resource usage (`cuobjdump --dump-resource-usage`) and the microbench: normquant_k<256,14> holds
  the row in registers (48 per thread) and runs three block reductions in series, so five CTAs - five
  rows - are resident per SM and each row's ~12 us latency chain overlaps with four others (39 us at
  2,112 rows = 3.2 waves x 12 us; byte floor about 9); r4quant_s_k stages the rotated fp32 row in
  shared memory (74 KB at the SiLU width), two to three CTAs resident, 101 us at 2,112 rows against a
  floor of about 40. Two forms, both bit-identical by construction (the same per-element operations
  in the same order; the reductions are order-free or exact): `normquant_s_k<128>` keeps the row in
  shared memory as int16 q, then in place as the bf16 norm output (2N bytes), reads the input twice
  (the second read is an L2 hit), 128 threads, so 16 rows are resident per SM; `r4quant_2p_k` rotates
  every block twice - once for the row max, once to quantise - with no staging (44 registers, no dynamic
  shared memory, five to eight CTAs resident), 16-byte loads and 8-byte stores. Routed from 256 rows
  (T16M_NQS, T16M_R42P; 0 = off); the kernel gate's 300 / 1,024-row norm and rotation cases run them.
  Queued (`s21_s6a`): build, gate, the glue microbench with the forms on and off (the rotation at the
  hidden width added as a row), then lockstep prompt rows and serving W1-W3 against S5A's stock arm.
- 2026-09-09, S6a MEASURED and KILLED (`handover/s6a`): both forms gate clean (1,186 / 1,186) and both
  are slower. The shared-memory norm: 11.4 / 24.7 / 40.4 us against 9.2 / 22.2 / 38.8 at 320 / 1,088 /
  2,112 rows (with the residual 59.7 against 44.0); the rotate-twice rotation: 41.6 / 77.9 / 126.8
  against 26.0 / 60.3 / 101.0 at the SiLU width, 6.9 / 16.8 / 27.6 against 5.6 / 12.3 / 19.2 at the
  hidden width. So the latency model was wrong for both: sixteen rows resident instead of five did
  not move the norm (it is bound by its integer chain: the 64-bit multiply, negations and shifts per
  element), and the rotation is bound by the shuffle pipe (five cross-lane butterfly stages = 40
  shuffles per 8 elements; 2,112 rows of the SiLU width are 200 M shuffles, about 28 us of an H100),
  which rotating twice doubles. Both are off by default (T16M_NQS=0, T16M_R42P=0), kept for the record.
  The one-CTA register form (T16M_R4S=0, `r4k`) at many rows: 148 / 76 us at 2,112 rows (SiLU / hidden)
  against the staged form's 100 / 19 - residency matters for the rotation, one row per SM loses.
- 2026-09-09, S6b / S6c / S6d BUILT: (b) the norm's integer chain in 32-bit arithmetic where the ranges
  prove it exact - |q| <= 32767 and the LUT value <= 32768 (round(32768 / sqrt(r)), r >= 1) so q * bb
  fits int32, t = (|prod| + 2^14) >> 15 <= 32768, t * |G| is one 32x32 -> 64 multiply, the rounding
  shift and the clamp on the magnitude with the sign of prod * G (y = 0 at t = 0 as before); the 64x64
  multiply, the 64-bit negations and compares are gone; (c) the rotation with 32 elements per lane -
  stages 1 .. 16 in registers, only 32 / 64 / 128 cross lanes: 3 shuffles per element instead of 5,
  the stage order and each stage's binary32 (a + b, a - b) unchanged, only WHERE a pair is computed
  moves - as a one-CTA form (`r4quant_w_k`, ceil(nb / 4) warps, off by default after `r4k`) and (d) as
  the staged form (`r4quant_sw_k`, T16M_R4SW, default 256 rows). Queued (`s21_s6c`): gate under each
  rotation route, then the microbench per form; an engine run only for a form that wins.
- 2026-09-09, S6b / S6c / S6d MEASURED (`handover/s6c`; gate 1,186 / 1,186 under each rotation route).
  S6b, the 32-bit norm chain, KEPT: 4.42 / 7.33 / 16.6 / 29.3 us against 4.99 / 9.17 / 22.2 / 38.8 at 64 /
  320 / 1,088 / 2,112 rows (with the residual 4.63 / 8.0 / 18.7 / 35.0 against 5.22 / 9.8 / 24.0 / 44.0):
  11 to 25 percent off the norm at every row count, the same integers. The rotation forms: the staged
  32-per-lane form (S6d) is slower everywhere (111.3 / 24.5 against 100.4 / 19.1 at 2,112 rows, SiLU /
  hidden width) - killed; the one-CTA 32-per-lane form (S6c) is slower at the SiLU width (107.7
  against 100.4) and faster at the hidden width (4.98 / 11.6 / 16.6 against 5.54 / 11.4 / 19.1) - kept
  for nb <= 16 from 256 rows (T16M_R4W). So the shuffle count was not the rotation's binding resource
  either; the rotation at the SiLU width stays at 100 us for 2,112 rows (byte floor about 40), and
  what binds it is not yet measured - without hardware counters the next probe is a build with the
  butterfly compiled out (bytes only) and one with the shuffles compiled out (register stages only).
  Queued (`s21_s6e`): the engine with S6b + S6c as defaults - 7B all rows and serving W1-W3, lockstep
  arm against S5A's stock arm.
- 2026-09-09, S6e bench rows (`handover/s6e`, lockstep arm with S6b + S6c as defaults, gate 1,186 / 1,186;
  against S5A's stock arm, S5A lockstep in brackets): TTFT 128 0.89 (0.89), 128 x 8 0.96 (0.95), 2k 1.02
  (1.00), 2k x 8 1.07 (1.05), 8k 1.04 (1.04), 8k x 8 1.05 (1.04), 8k x 4 1.05 (1.04), 32k 0.99 (0.97),
  128 x 32 prefill 1.00 (1.00); decode 1.15 / 1.13 / 1.03 (1.13 / 1.09 / 1.03) at batch 1 / 8 / 32.
  Serving W1-W3 running.
- 2026-09-09, S6e serving (`t14_res/*S6E*`, against S5A's stock arm): W1 / W2 / W3 output throughput
  0.88 / 0.93 / 0.97 (S5A 0.86 / 0.92 / 0.95), mean TTFT 0.85 / 0.93 / 0.96. The rotation probe
  (`handover/s6c/glue_probe`, the butterfly compiled out, had=0): the staged kernel at the SiLU width
  is 79.2 us at 2,112 rows WITHOUT the rotation against 100.9 with it (byte floor about 40), and the
  hidden-width form 14.0 against 16.8 (floor about 4). So the butterfly is a fifth of the kernel; the
  rest is the kernel body - and the one per-element operation left in it that is not a load, a max or
  a store is the contract's quantisation DIVISION (`__fdiv_rn`, a multi-instruction sequence), which
  every quant epilogue (the rotation, the norm's int8 output, rowquant) runs once per element. Probe
  queued (`s21_qdiv`): the same kernels with qdiv=0 (a multiply) - not a candidate (the contract pins
  the division), a measurement of what the division costs; if it is the remainder, the exact remedy
  is a fast path that takes the division only where its rint could differ (a candidate from the
  multiply, the reference division whenever the candidate is within 2^-13 of a half-integer; the error
  bound 3u * 128 < 2^-13 makes every other case provably identical).
- 2026-09-09, the division probe (`handover/s6c/glue_qdiv`, the same kernels with qdiv=0, a multiply -
  not a candidate, the contract pins the division): the rotation at the SiLU width 17.1 / 48.3 / 85.6 us
  against 25.9 / 60.3 / 100.8 at 320 / 1,088 / 2,112 rows (15 to 34 percent), at the hidden width 4.0 /
  7.2 / 11.6 against 5.0 / 11.7 / 16.8 (20 to 31), the norm 6.8 / 15.8 / 28.0 against 7.4 / 17.0 / 29.5
  (5 to 8). S6f BUILT: `quant_div_rint` - the candidate from the multiply, the reference division only
  when the candidate is within 2^-13 of a half-integer (with r = RN(1 / asc), c = RN(v r) and d =
  RN(v / asc) are both within 3u |v / asc| of the exact quotient, so for |v / asc| <= 256 within 3 * 2^-16
  < 2^-13 of each other, and rint is constant between half-integers; beyond 256 both clamp) - applied
  at all 15 quant-epilogue sites; and the SiLU's `1 / den` as `__frcp_rn(den)`, the correctly rounded
  reciprocal, the same bits by definition. Queued (`s21_s6f`): gate, microbench, the 7B bench and
  serving against S5A's stock arm.
- 2026-09-09, S6f MEASURED (`handover/s6f`; gate 1,186 / 1,186). The exact fast-path division is
  NEUTRAL on the rotation at the SiLU width (101.9 against 100.8 us at 2,112 rows; the multiply form
  reads 85) and -4 percent at the hidden width (16.2 against 16.8): the division's instructions are
  replaced by about as many (multiply, floor, two subtractions, abs, compare, rint), which says the
  rotation kernel is INSTRUCTION-throughput-bound, not division-bound - about 27 instructions per
  element, 40 M elements, against 40 us of bytes; what the qdiv=0 probe measured was two instructions
  against eight, not the division's latency. Kept (exact by proof, not slower). The SiLU's `__frcp_rn`
  IS a gain: deq_silu(tiled) 159 against 180 us at 2,112 rows, 86 against 97 at 1,088 (-11 to -12
  percent). Bench rows against S5A's stock arm: decode 1.15 / 1.13 / 1.04 at batch 1 / 8 / 32 (190.2 /
  1,453.6 / 5,271.0 tok/s), TTFT 0.90 / 1.02 / 1.05 / 0.99 at 128 / 2k / 8k / 32k, batch 8 0.95 / 1.06 /
  1.06, 8k x 4 1.06, 128 x 32 prefill 1.00. Serving running.
- 2026-09-09, S6f serving (`t14_res/*S6F*`): W1 / W2 / W3 0.875 / 0.933 / 0.968 (S6E 0.880 / 0.931 /
  0.968; S5A 0.861 / 0.916 / 0.954), mean TTFT 0.85 / 0.93 / 0.97. The glue plateau for serving is
  reached at about 0.88 / 0.93 / 0.97; the remainder is the decode kernel in mixed steps and the
  rotation itself.
