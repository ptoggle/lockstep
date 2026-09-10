# Plan: Stage 2 item 2.4 - saturated serving inside 5 percent (the decode kernel on tensor cores)

## The ultimate goal, and why this item

Stage 2's exit is a dense model at parity that a customer can deploy: single-stream at parity, and
its serving workload inside 5 percent of stock (ROADMAP, Stage 2). After item 2.1 (SPEC 7.24-7.25, S5a
and S6) the 7B is above parity on every single-stream row - decode 1.15 / 1.13 / 1.04x at batch 1 / 8 /
32, prompts 0.90 / 1.02 / 1.05 / 0.99x at 128 / 2k / 8k / 32k - and serving reads chat 0.88x, mixed
0.93x, prefill-heavy 0.97x. Chat is the workload that matters to a customer and it is 7 points short.

The W1 profile (SPEC 7.25 S5a, `results/s2/perf/s21/prof2`) says where those points are. Of 4,373 ms of
lockstep GPU time against stock's 3,584: the int8 linears save 621 ms; the lm_head is now at stock's
(S5a); the fused glue costs 720 ms more than stock's norm / rope / activation, of which the rotation
(276 ms, no stock counterpart) is the contract's own operation and the rest is at or near its byte
floor after S6; and ATTENTION costs 361 ms more - the decode kernel 618 ms plus 102 ms of seam glue and
51 ms of prompt-row seam, against FlashAttention-3's 410 ms for both prefill and decode rows. Per launch
in a mixed step of 64 sequences: the decode kernel 43 us against FA3's 20.6 for a whole step. The
attention item is the one large piece left, and it is a kernel item, not a routing one.

## Why the decode kernel is slow, from what is measured (no hardware counters on the pod)

S4c (the LSB_DEC_TIMING build) put a 128-key block at about 7,400 cycles per group of 256 threads:
scores 22 percent, the weight chain 21, the PV 12, the prologue 30 at short context. The block body
(`lsb_dec8p`, compute_block) does its two integer dot products on CUDA cores with dp4a - per block,
per group: 128 keys x 7 heads x 128 dims = 115 k MACs for the scores and the same for the PV, i.e. about
230 dp4a per thread plus the loads, the V transpose (16 byte_perm + 16 shared stores per thread), four
LUT lookups per lane per head warp and six group barriers. With two groups per CTA and two CTAs per SM,
32 warps contend for four schedulers: 32 x ~700 instructions per block / 4 = 5,600 cycles for the four
blocks in flight - the measured 7,400. The kernel is instruction-bound on dp4a. FA3 does its dot
products on the tensor cores (wgmma, bf16) and is not.

## The lever: the same integers on the tensor cores

Both dot products are INTEGER and EXACT: S = q8 . k8 accumulates in int32 (at most 128 x 127 x 127 <
2^21), and the PV partial accumulates u8 weights x s8 values (128 x 255 x 127 < 2^23). Integer sums are
order-free, so the SAME int32 values come out of any evaluation order - the contract pins the integers,
not the instruction. Hopper's integer MMA (`mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`, and the
`.u8.s8` form for the PV) computes 16 x 8 x 32 = 4,096 MACs per warp instruction against dp4a's 128:
32x denser. In decode the M dimension is the GQA group (7 heads on the 7B, zero-padded to 16), so the
useful density is 14x; the scores per block become 64 MMAs (16 key tiles x 4 k-steps) and the PV 64
(16 dim tiles x 4 key steps) - 16 per warp against ~230 dp4a per thread - and the block's cost moves to
the parts that are not arithmetic: the transpose, the LUT chain, the barriers. Those are the next
items once the arithmetic is off the critical path.

The fragments fit the shared-memory layouts the kernel already has:
- scores: A = Q (row-major s8 [16][128] from `sm.q[h]`, rows 7..15 zero), B = K^T in "col" layout, which
  is K row-major [key][dim] with dims contiguous - exactly `kbg[key * 8 + (chunk ^ (key & 7))]` (the
  16-byte chunk of 16 dims, 4 words of 4 dims); the b0 / b1 registers of lane l are word (l & 3) of
  chunks 2 kstep and 2 kstep + 1 of key (tile * 8 + (l >> 2)) - conflict-free under the XOR swizzle.
  The C fragment holds (head = l >> 2, keys tile * 8 + 2 (l & 3) + {0, 1}) in c0 / c1 (rows 8-15 in
  c2 / c3 are the padding and are dropped); they go to `sm.sc[grp][h][key]` with the visibility mask,
  and everything after (max, NB, LUT, weights) is unchanged.
- PV: A = P, the u8 weights `sm.eh[grp][h][key]` (row = head, k = key: a0 is one 32-bit load of keys
  kstep * 32 + 4 (l & 3)); B = V^T in col layout = the transposed `vt[dim * 32 + (word ^ ((dim >> 2) &
  31))]` the kernel already builds (b0 is word kstep * 8 + (l & 3) of dim tile * 8 + (l >> 2)). The C
  fragment (head, dims tile * 8 + 2 (l & 3) + {0, 1}) is written to a shared scratch and re-read in the
  layout the fold expects (warp h, lane holds dims 4 l .. 4 l + 3), which the score buffer provides
  (dead at that point; the sink stash writes it afterwards from the same warp).
- Every other byte of the kernel - the exponent handling, the segment folds, the split-KV combine,
  the B9.1 sink and B9.3 value-shift paths - is untouched.

## Phases

- P0 (measure, running): the attention kernels per decode step at serving shapes - 64 sequences at
  512 / 1,024 / 2,048 / 4,096 context, stock FA3 against the lockstep decode kernel, in-engine
  per-kernel time (`decode_prof.py`). This is the number the item must move.
- P1 (build): the scores on IMMA in `lsb_dec8p` (`compute_block`), behind a compile-time switch
  (`-DLSB_IMMA=1`) so the dp4a body remains the reference build. Gate: the kernel rig (t13b_gate K1-K9,
  t16n_gate under ks = 0 and 3, the dec8p variant) bit-identical; then the engine gates E1 / E4v2 / E6 /
  E7 with the new .so. Measure: P0 again.
- P2 (build): the PV on IMMA, the same discipline.
- P3 (rebalance): with the arithmetic off the critical path, the block's remaining cost - the barrier
  count per key (256-key blocks), the V transpose (or the value cache written transposed by the prep
  kernel, a layout change that keeps every value), the LUT chain - each measured by the timing build.
- P4 (measure, file): the 7B bench (all rows) and serving W1-W3 against a clean stock arm; SPEC 7.26;
  then the family re-measurement (Llama-8B, Mistral-7B, Phi-4-mini, Qwen3-8B, the size sweep) whose rows
  are lower bounds since S5a.

## Targets and kill lines

- Target: the decode kernel's per-step time at 64 sequences and 1-2k context within 1.3x of FA3's
  (from about 2x); W1 at or above 0.95x, W2 / W3 at or above 0.97x.
- Kill line for a form: not bit-identical on the rig, or slower than the dp4a body at any of the four
  P0 shapes. A form that gates clean and does not move P0 is filed and turned off, as in item 2.1.
- Budget: the H100 (lc-handover, USD 3.49 / h); each build-gate-measure cycle is about 40 minutes of
  pod time; P1 + P2 within a day of pod time if the fragments go in cleanly.

## Log
- 2026-09-09, P1 + P2 BUILT (`lssa/portable/lssa_lssab_t13b.cu`, `-DLSB_IMMA=1` scores, `=2` scores + PV;
  `=0` is the dp4a body, untouched): warp wg takes key tiles 2 wg and 2 wg + 1 for the scores (rows =
  heads, K = 128 dims in four steps, B words straight from the swizzled K block, conflict-free) and dim
  tiles 2 wg and 2 wg + 1 for the PV (A = the u8 weight rows, B = the transposed V words), the C
  fragments scattered through the dead score rows into the fold's per-warp layout (one extra group
  barrier per block). Every other line of the kernel is the same. Queued (`s21_s7p1`): three builds,
  the kernel rig (K-gates, t16n under ks = 0 / 3, the dec8p variant) per build, then the P0 shapes per
  build against P0's stock arm.
- 2026-09-09, P0 (`handover/s7p0`, 64 sequences, GEN=32, in-engine per-kernel time): at UNIFORM
  contexts the decode step is at parity - total ms per token 27.98 against stock's 27.85 at 512, 50.1
  against 51.2 at 1,024 (lockstep faster); the decode kernel 52.7 / 83.1 us per step-layer against
  stock's FA3 54.4 / 108.2 (stock's figure includes its prefill step; ours goes to the seam). So the
  kernel's per-key throughput is NOT what W1 pays: W1's decode-kernel launches average 43 us against
  FA3's 20.6 in steps whose contexts are short (chat) and mixed with prompt chunks. The item is the
  fixed cost per launch at short context (the prologue, the split-path glue: prep / rowmap / gather
  102 ms, the B8 seam 51 ms) - S4c's 30 percent prologue at 300 tokens, and the T16j-style sizing of
  the launch by the whole step. P0b queued: contexts 128 and 256 (GEN=64), both stacks; S7 P1/P2's
  gate-and-measure follows it (the tensor-core forms are still worth having if they gate clean).
- 2026-09-09, P0 read again with the prefill share removed: stock's FA3 figure at GEN=32 includes its
  prefill chunks (4 x 8,192 tokens at 512 context, about 11 ms of the 48.7; 8 chunks at 1,024, about
  45 ms of 96.9), so FA3's DECODE cost is about 42 us per step-layer at 512 and 58 at 1,024 against the
  decode kernel's 52.7 and 83.1: the kernel is 1.25x to 1.45x FA3 at uniform contexts, and about 2x in
  W1 (43 against 20.6 us per launch at chat contexts). Both parts of the item stand: the per-key
  throughput (S7 P1 / P2 on the tensor cores) and the fixed cost per launch at short context (P0b).
  P0c queued behind S7 P1: GEN=96 at 512 / 1,024 for both stacks, so the decode-only cost per
  kernel is the difference against GEN=32, prefill excluded, for every build.
- 2026-09-09, P0b (`handover/s7p0b`, 64 sequences, GEN=64, so the one prefill chunk is a negligible
  share): the decode kernel 33.5 / 40.8 us per step-layer at 128 / 256 context against FA3's 22.0 /
  29.1; with P0's 52.7 / 83.1 / 139 / 240 at 512 / 1,024 / 2,048 / 4,096 the kernel fits about 26 + 0.053
  x context us and FA3 about 15 + 0.04 x context: a FIXED cost per launch about 11 us above FA3's and a
  per-key slope about 1.3x FA3's. At chat contexts (about 300) the fixed part is 70 percent of the gap.
  The step totals at 64 sequences: 9.34 against 9.13 ms per token at 128 (0.977), 12.15 against 11.79
  at 256 (0.970), 27.98 against 27.85 at 512, 50.1 against 51.2 at 1,024, 95.2 against 97.7 at 2,048,
  190.4 against 195.1 at 4,096 - the int8 linears cover the attention gap from 512 up. What W1 pays
  beyond attention is the fused glue at 64-row decode steps: from the W1 profile, 935 ms against
  stock's 215 over 514 steps = +1.4 ms per step, i.e. eight glue launches per layer at 2 to 14 us each
  for a few megabytes apiece (the rotation at the SiLU width 14 us at 64 rows for 3.6 MB, deq_silu
  10.8, the norm 4.4 x 2, rope 5, rowquant 2.2) - latency-bound small launches, at their measured
  floor per form (S1n2, S6). So the item splits: (i) the kernel's fixed cost per launch (the prologue's
  dependent loads, S4d; the launch sizing) - 0.3 ms per step; (ii) the slope (S7 P1 / P2) - 0.1 ms per
  step at chat contexts, more at long ones; (iii) the 64-row glue - the largest, and structural
  unless launches merge.
- 2026-09-09, S7 P1 measured (`handover/s7p1`, LSB_IMMA=1, the scores on the tensor cores; the gate rig
  hit an ABI mismatch - the pod's lssa/portable gate scripts were older than the kernel's LsbDev, 488
  against 496 bytes - and is re-queued with the matching scripts): the decode kernel 53.2 / 81.6 /
  138.4 us per step-layer at 512 / 1,024 / 2,048 against the dp4a body's 52.7 / 83.1 / 139.2. NO CHANGE:
  the scores' 115 k MACs per block were not the block's cost. So S4c's "scores 22 percent" was time
  spent in that phase, not time bound by its arithmetic - the phase is bound by what surrounds the
  dp4a (the K block's arrival, the group barrier, the shared-memory traffic). P2 (the PV) is measured
  next; the timing build at 64-sequence shapes (S7T) will say where the 7,400 cycles per block go.
- 2026-09-09, a wrong turn, filed so it is not repeated: SPEC 7.10 (T16o) already put the decode
  kernel's inner products on the int8 tensor cores - built, gated (rig 21 / 21, E1 / E4v2 / E6), NOT
  faster (126 against 129 tok/s at 32k) - and 7.11 (T16q) reworked its memory pipeline with the same
  verdict; the instrumented build then said no phase dominates: the cost is five block-wide barriers
  and the short dependent chains between them, at sixteen warps per SM. Today's P1 (the scores on
  mma.sync in lsb_dec8p) re-measured exactly that (no change at any context) before the spec was
  re-read. About 1.5 pod-hours. The rule that would have caught it: grep SPEC.md for the lever's
  name before building. What 7.10 left as the path - more keys per barrier, a V layout that needs
  no transpose - stands; what P0b adds is that at chat contexts the FIXED cost per launch (26 against
  FA3's 15 us at 64 sequences) outweighs the slope, and that the 64-row glue is the larger W1 item
  of the two. The S7 forms stay behind their flag (LSB_IMMA=0 is the build) once P2's number is in.
- 2026-09-09, S7 P2 measured (LSB_IMMA=2, scores + PV on the tensor cores): the decode kernel 144.2 /
  253.3 us per step-layer at 2,048 / 4,096 against 139.2 / 240.5, and the step totals 28.08 / 50.25 /
  95.37 / 190.8 ms per token against 27.98 / 50.11 / 95.17 / 190.4 - slightly SLOWER at every context,
  T16o's verdict again (the extra barrier and the scatter cost more than the dp4a they replace). Both
  forms stay behind the flag; LSB_IMMA=0 is the build. S7r BUILT: the rowmap's shadow tables in shared
  memory (one thread's O(nseq x nsh) dependent global loads become shared-memory reads; every write
  goes to both; the slot decisions are unchanged) - the 139 us per step at 64 sequences is the target;
  queued behind the gates.
- 2026-09-09, P0c (`handover/s7p0c`, GEN=96 against P0's GEN=32: the 64 extra tokens are pure decode,
  the prefill share cancels). Decode-only at 64 sequences, per step-layer: FA3 37.7 / 60.0 us at 512 /
  1,024 context, the decode kernel 56.4 / 85.0 - 1.50x / 1.42x; per token-step the stock step 7.07 /
  7.71 ms, lockstep 7.70 / 8.44 - 0.92x / 0.91x, and the attention difference (0.52 / 0.70 ms) is 80 to
  95 percent of the step's gap: at 64 sequences the int8 linears cover the glue but not the attention.
  With P0b's fit (26 + 0.053 x context against 15 + 0.04 x context): both the fixed cost and the slope.
- 2026-09-09, the one attention path not yet tried, read for cost: decode rows through the exact FA3-B8
  prompt kernel. The seam (`lssab_fa3.py`) is a prompt-path design - one FA3 launch PER SEQUENCE over
  a dense gathered K / K-major V (`fg_gather`), `num_splits = 1`, `pack_gqa = False`, the B8 side
  buffers (e_q, NB, m_S) offset per launch. Decode rows would need one varlen launch over every
  decode sequence on the paged lattice with packed GQA heads and a split-KV combine that performs
  the contract's exact fold - a redesign of the seam and of the fork's combine, the FA3-M1-class
  project, weeks. Not an iteration.
- 2026-09-09, S7T (`handover/s7t`, the LSB_DEC_TIMING build in the rig, NSPLIT=1, 64 sequences = 256
  CTAs, one wave at two per SM), per CTA: 128 context 35.2 k cycles (prologue 13.3 k = 38 percent, 2
  chunks), 256: 45.1 k (prologue 12.6 k, 3 chunks), 512: 67.8 k (prologue 11.1 k, 5 chunks) - about 9 k
  to 11 k cycles per 128-key chunk, split evenly between the partials, the chain and the PV, the
  cp.async wait under 4 percent; the tensor-core build is the same to within noise. The step at 512
  in the rig is 39 us of kernel against the ENGINE's 56 for the same shape - and the difference is the
  engine's split: `auto_split_nt` targets 8 CTAs per SM (T16j, tuned on 31k-token prompts with few
  sequences), so 256 tiles become about 1,280 CTAs of one chunk each and the 13 k-cycle prologue is
  paid five times per (sequence, head). The cheap lever at 64 sequences is the split policy (fewer
  splits when the tiles already fill the machine); the deeper one is the prologue itself (S4d).
- 2026-09-09, S7r measured WRONG and caught by the number (`handover/s7r`): the rowmap went 139 -> 469
  us per step. The shared-memory fill loop ran after the kernel's per-sequence early return, i.e.
  with one thread and a stride of 256, so only every 256th tag was initialised and the walk read
  garbage (the slot decisions changed; the gate rerun on that build must FAIL and is left queued as a
  check on the rig). Fixed: block 0 keeps every thread through the fill (the tile emission is
  thread 0's alone), then one thread walks the shared copy. Re-queued as s7r2 behind the gates.
- 2026-09-09, S7s, the split sweep (`handover/s7s`, 64 sequences, GEN=64; the rowmap in these cells
  is the broken s7r build, so only the decode kernel's own time is read): the kernel per step-layer,
  CTA target 1 per SM (no split at 256 tiles): 31.1 / 35.3 / 43.6 / 67.9 us at 128 / 256 / 512 / 1,024;
  target 2: 33.0 / 39.8 / 48.9; the shipped target 8: 33.5 / 40.8 / 52.7 / 83.1 (P0 / P0b). Unsplit
  is -7 / -13 / -17 / -18 percent, and against FA3's decode-only 37.7 / 60.0 at 512 / 1,024 the kernel
  is now 1.16x / 1.13x (from 1.50x / 1.42x). POLICY (lssab_engine.auto_split_nt): no split once the
  tiles alone put one CTA on every SM (LOCKSTEP_LSSAB_SPLIT_FULL, default 1); below that the T16j
  target of 8 per SM stands (W4's few long sequences need the splits). Metadata only (the split is
  the engine's choice; the fold is exact in any split, K7 / E8). Queued: s7r2 (the fixed rowmap +
  this policy) then S7 P4: the 7B bench and serving W1-W3 against S5A's stock arm.
- 2026-09-09, S7s read again with the whole sweep: at 128 context there is no split under any target
  (one block per sequence: nsplit = 1), and at 256 one split - so the shipped target reads 31.2 / 35.8
  against the unsplit 31.1 / 35.3 there: the same. The policy's effect starts where the shipped
  target splits 4 to 8 blocks four or five ways: 512 and 1,024 context (-17 / -18 percent). So the
  fixed cost per launch at chat contexts (P0b: 26 against FA3's 15 us) is the PROLOGUE proper (13 k
  cycles: the dependent metadata loads, the in-kernel prep of the newest block, the table copy) plus
  the per-step rowmap, not the split. Targets 2 and 4 are between or worse (77.2 at 1,024 with two
  splits against 67.9 unsplit and 68.2 with three): no split is best or tied at every context.
- 2026-09-09, S7s WITHDRAWN pending a clean rerun: the sweep's target-8 cells read 43.5 / 68.5 us at
  512 / 1,024 - the same as the unsplit 43.6 / 67.9 - so the split never mattered; the 52.7 / 83.1 of
  P0 were measured with the SHIPPED .so and the default rowmap, and every sweep cell ran the BROKEN
  s7r build, whose garbage shadow slots change the in-kernel prep's path (a slot that looks fresh
  skips or re-does work) and so the decode kernel's prologue. The apparent 17 percent is
  contamination, not a policy effect. The policy knob stays (default back to T16j's: never skip the
  split) and the sweep is re-queued on the gated s7r2 build, target 1 against 8 at 512 / 1,024,
  before the final measurement. Lesson, filed: never read a performance number off a build that has
  not passed its gate - the gate was queued behind the measurement to save a pod-hour and cost two.
- 2026-09-09, the rig back in service (`handover/s7g2`, the lssa/portable gate scripts brought level
  with the kernel's struct, 496 bytes both sides): t16n dec8p on the shipped-source build 21 / 21 at
  ks = 3 and at ks = 0. The K-rig read 0 / 43 with maxdiff about 3.5 - run with LSB_GATE_KS=3 by
  mistake (the K-rig's oracle is the ks = 0 reference; the campaign's K-rig line was KS=0 SINK=0):
  a harness-configuration error, to be rerun at KS=0 on the s7r2 build, not a kernel finding.
  Also found: the final measurement script had lost its wait line in the sed that derived it and
  exited at once ("no s7r2 .so") both times it was launched - no engine run overlapped the sweep.
- 2026-09-09, the rig on the broken s7r build (`handover/s7g2`): t16n dec8p 2 / 21 at ks = 3 and at ks = 0
  (the shipped source 21 / 21 on the same rig) - the garbage shadow slots are caught as bit
  differences, as they should be. The corrected s7r2 build is gating on the same rig.
- 2026-09-09, s7r2 (the corrected rowmap) gates clean on the decode rig: t16n dec8p 21 / 21 at ks = 3
  (the K-rig line in that job was the KS=3 mis-run again, 0 / 43 like the shipped source; the KS=0 run
  is queued). Its 64-sequence measurement runs next, then the clean split sweep on this build.
- 2026-09-09, s7r2 measured (`handover/s7r2`, 64 sequences): the decode kernel 33.5 / 40.8 us at 128 /
  256 - exactly P0b's numbers, so the S7 source at LSB_IMMA=0 IS the shipped kernel and the
  contaminated sweep's 43.6 at 512 is confirmed as contamination; the rowmap 133 us per step against
  139 - the tag walk was not its cost. What remains in its one serial thread: QS / SL / bt[.,0] reads
  per sequence (dependent global loads, two loops of 64). s7r3 BUILT: those cached in shared memory
  by every thread alongside the tags (metadata only). Queued behind the K-rig, before the families.
- 2026-09-09, S7s2, the split sweep on the GATED s7r2 build (`handover/s7s2`, 64 sequences, GEN=64):
  no split (target 1 per SM) 44.1 / 63.2 us per step-layer at 512 / 1,024 against the T16j policy's
  54.1 at 512 (1,024 pending) - the policy effect is real: -18 percent at 512, and against FA3's
  decode-only 37.7 / 60.0 the kernel is 1.17x / 1.05x. The default is set to it (LOCKSTEP_LSSAB_SPLIT_FULL=1)
  before the final measurement starts. The contaminated sweep's reading is withdrawn in both
  directions: only the gated build counts.
- 2026-09-09, S7s2 complete: the T16j policy at 1,024 context reads 83.6 us against the unsplit 63.2
  (-24 percent); at 512, 54.1 against 44.1 (-18). The rowmap on the s7r2 build 127 to 133 us per step.
  S7 P4 (the 7B with the s7r2 kernel and the no-split default) first rows: prefill 128 x 32 1.02x,
  decode 1.14 / 1.10 / 1.01x at batch 1 / 8 / 32 (S6f read 1.15 / 1.13 / 1.04 - batch 8 and 32 are 2 to 3
  points lower than the previous run; at batch 32 the tiles are 128 CTAs, below one per SM, so the
  policy does not act there and the difference is the s7r2 rowmap or run-to-run noise - to be read
  against the prompt rows and serving when they land).
- 2026-09-09, S7 P4 MEASURED (`handover/s7p4`, `t14_res/*S7P4*`; the s7r2 kernel + the no-split default;
  t16m kernel gate 1,186 / 1,186; against S5A's stock arm): TTFT 0.90 / 1.03 / 1.06 / 0.99 at 128 / 2k /
  8k / 32k (batch 8: 0.97 / 1.07 / 1.06; 8k x 4 1.06; 128 x 32 prefill 1.02) - every prompt row its best
  yet; decode 1.14 / 1.10 / 1.01 at batch 1 / 8 / 32 (S6f 1.15 / 1.13 / 1.04); serving W1 / W2 / W3
  output throughput 0.865 / 0.967 / 0.968 (S6f 0.875 / 0.933 / 0.968), mean TTFT 0.867 / 0.966 / 0.963
  (S6f 0.849 / 0.932 / 0.965). The split policy moves what it touches: the MIXED workload +3.4 points
  (its decode rows sit at 500 to 2,000 context, where the old policy split), the prompt rows +1 to +3;
  chat is flat at 0.87 because its contexts are under 256 tokens, where no split happens under either
  policy - its remainder is the kernel's prologue and the 64-row glue, as P0b said. The decode rows at
  batch 8 / 32 are 2 to 3 points under S6f with the policy inactive there (128 tiles); the s7r2 rowmap
  is the only other difference and it is 6 us per step faster, so this is read as noise until the
  K-rig at KS=0 (queued) and s7r3 say otherwise.
- 2026-09-09, the K-rig at its oracle's configuration (KS=0, SINK=0): 43 / 43 on the s7r2 build AND on the
  shipped .so - the rig is healthy, the 0 / 43 readings were the KS=3 mis-run, and the rowmap cache is
  clean on every K case. s7r3 (the per-sequence metadata cached too): t16n dec8p 21 / 21; its K-rig
  and 64-sequence measurement run next, then the families.
- 2026-09-09, s7r3 MEASURED and KEPT (`handover/s7r3`; K-rig 43 / 43 at KS=0, t16n 21 / 21): the rowmap
  124 us per step (from 139: the tags, then the per-sequence metadata, in shared memory); and with the
  no-split default the decode kernel at 64 sequences reads 27.7 / 34.1 us per step-layer at 128 / 256
  (from 33.5 / 40.8: the old policy split there too, once the context grew past one block during
  generation), so the decode-only STEP is at parity at chat contexts: 9.14 against stock's 9.13 ms per
  token at 128, 11.93 against 11.79 at 256 (from 0.977 / 0.970). Promoted as the shipped kernel
  (`lssa_lssab.so`, the previous one kept as `lssa_lssab_pre_s7r3.so`). What W1 still pays at 0.87 is
  therefore in its MIXED steps (the prompt chunks with their seam and glue at 2k rows) and its longer
  contexts, not in the 64-row decode step; a W1 profile with the new policy is queued behind the
  families to re-attribute it.
- 2026-09-09, the family re-measurement begins (`handover/fam2`, the s7r3 kernel, the no-split default,
  the fixed lm_head route; one process per stack, REP=3). Qwen2.5-1.5B: TTFT 0.65 / 0.75 / 0.78 / 0.79
  at 128 / 2k / 8k / 32k (batch 8: 0.74 / 0.78 / 0.80), decode 0.86 / 0.70 / 0.68 at batch 1 / 8 / 32
  (filed before: 0.60 / 0.75 / 0.76 and 0.62 / 0.52 / 0.50). Qwen2.5-3B: TTFT 0.72 / 0.83 / 0.89 / 0.86
  (batch 8: 0.79 / 0.93 / 0.91), decode 0.94 / 0.78 / 0.74 (before 0.70 / 0.80 / 0.88 and 0.74 / 0.62 /
  0.60). The small models gain most on batch-1 decode (the lm_head route) and stay "by design" on
  prompts and batched decode: their per-layer fixed cost against a thin layer.
- 2026-09-09, Llama-3.1-8B re-measured (`handover/fam2`, the s7r3 kernel, the no-split default, the
  fixed route): TTFT 0.91 / 1.04 / 1.08 / 0.99 at 128 / 2k / 8k / 32k, batched prompts 1.01 / 1.09 / 1.07
  (8k x 4 1.08), 128 x 32 prefill 1.04, decode 1.10 / 1.05 / 1.01 at batch 1 / 8 / 32 (filed before: TTFT
  0.85 / 0.92 / 0.86, decode 0.92 / 0.90 / 0.86). A second family at or above parity on every
  single-stream row but the 128-token prompt - Stage II's single-stream exit criterion holds on two
  families with the same stack and no per-family tuning.
- 2026-09-09, Mistral-7B-v0.3 re-measured: TTFT 0.91 / 1.05 / 1.07 / 0.97 at 128 / 2k / 8k / 32k, batched
  prompts 1.00 / 1.10 / 1.07 (8k x 4 1.07), 128 x 32 prefill 1.04, decode 1.13 / 1.09 / 1.04 at batch 1 /
  8 / 32 (filed before: TTFT 0.85 / 0.91 / 0.85, decode 0.84 / 0.87 / 0.81). A third family at or above
  parity on every single-stream row but the 128-token prompt and the 32k prompt (0.97).
- 2026-09-09, Phi-4-mini re-measured: TTFT 0.97 / 1.00 / 1.04 / 0.94 at 128 / 2k / 8k / 32k, batched prompts
  0.96 / 1.09 / 1.08 (8k x 4 1.05), 128 x 32 prefill 1.03, decode 1.18 / 1.16 / 1.11 at batch 1 / 8 / 32
  (filed before: TTFT 0.78 / 0.81 / 0.80, decode 0.92 / 0.95 / 0.90). The fourth family at or above parity
  on the single-stream rows but the 32k prompt (0.94); its "flat 20 points down on prompts" is gone.
- 2026-09-09, Qwen3-8B re-measured: TTFT 0.62 / 0.45 / 0.51 / 0.56 at 128 / 2k / 8k / 32k, batched prompts
  0.44 / 0.48 / 0.50, 128 x 32 prefill 0.46, decode 0.82 / 0.82 / 0.86 at batch 1 / 8 / 32 (filed before:
  TTFT 0.45 / 0.50 / 0.55, decode 0.73 / 0.72 / 0.72). Decode gained the lm_head fix; the prompts did
  not move because every layer still runs the unfused exact path (the per-head q / k norm) - item 2.2
  (PLAN_22) is confirmed as the largest gap left in Stage II, a 2x on prompts for the family.
- 2026-09-09, the family job ends: Qwen2.5-14B's prepare failed in `lockstep_calibrate.py` (rc=1; the log
  is read next) and its rows are not re-taken; the other six families are. Summary of the
  re-measurement on the fixed route, the S6 glue and the S7 split policy (TTFT 2k / 8k / 32k; decode
  1 / 8 / 32): Llama-3.1-8B 1.04 / 1.08 / 0.99, 1.10 / 1.05 / 1.01; Mistral-7B 1.05 / 1.07 / 0.97, 1.13 /
  1.09 / 1.04; Phi-4-mini 1.00 / 1.04 / 0.94, 1.18 / 1.16 / 1.11; Qwen2.5-7B 1.03 / 1.06 / 0.99, 1.14 /
  1.10 / 1.01; Qwen3-8B 0.45 / 0.51 / 0.56, 0.82 / 0.82 / 0.86 (item 2.2); Qwen2.5-3B 0.83 / 0.89 / 0.86,
  0.94 / 0.78 / 0.74 and 1.5B 0.75 / 0.78 / 0.79, 0.86 / 0.70 / 0.68 (by design). Four families at or
  above parity on the single-stream rows with one stack and no per-family tuning.
- 2026-09-09, the 14B's prepare failed on the weight download: "Disk quota exceeded" on /workspace (the
  HF cache had grown by four families' bf16 weights). Re-queued behind the W1 profile with the
  regenerable weights and manifests of the families already benched dropped first (Qwen3-8B kept for
  item 2.2).
- 2026-09-09, W1 re-attributed under the new policy (`handover/w1p2`, lockstep, 128 requests): GPU
  3,931 ms (from 4,373 in the S5a-era profile) against stock's 3,584 - 0.91x in GPU time where the
  throughput reads 0.87x (the rest is the host side of a 64-sequence step). By bucket: int8 linears
  2,084 + 194 ms against bf16 2,905 (-627); the decode kernel 605 ms at 42 us per launch (43 before:
  the policy did NOT move it here - the 64-sequence chat step is ragged, and a launch's time follows
  its LONGEST sequence while FA3's varlen schedule follows the mean; at W1's mean context FA3 reads
  20.6 us per launch) + seam glue 96 + B8 seam 51 against FA3's 410 (+342); the fused glue 990
  against stock's norm / rope / act 215 (+775: the rotation 381 ms of it, ours only; deq_silu 237;
  the norm 184; rope 90; rowquant 41); lm_head at stock's (26 against 25). Run-to-run the step
  composition varies (the GEMM launch counts differ by 10 percent between profiles), so per-kernel
  means across runs carry that noise. NEXT CANDIDATE for 2.4, filed not built: a PER-TILE split -
  today nsplit is one number per launch, so a ragged step either splits every sequence (the old
  policy: the prologue paid many times) or none (the new one: the longest sequence sets the wave);
  a split count per tile, ceil(blocks / target), from the rowmap, with the grid mapping and the
  combine counters per tile, would balance the wave the way FA3's scheduler does. Metadata only;
  about a day; worth perhaps 200 of the 342 ms.
- 2026-09-10, S7u (the per-tile split) gates clean: t16n dec8p 21 / 21 at ks = 3 and ks = 0, K-rig 43 / 43
  at KS=0 (K7 / K8 are the split paths). Its 64-sequence cells and serving W1-W3 run next.
- 2026-09-10, S7u (the per-tile split) MEASURED and KILLED (`handover/s7u`): with the cap from the longest
  sequence and four blocks per CTA, the decode kernel at 64 uniform sequences 62.5 / 86.0 / 141.7 us at
  512 / 1,024 / 2,048 against the no-split 44.1 / 63.2; serving W1 0.860 against 0.865, W2 0.921 against
  0.967, W3 0.968 against 0.968. Splitting long sequences costs more in prologues than it recovers in
  wave balance on this kernel: the prologue is the price of every CTA, and the ragged wave was the
  cheaper evil. Off by default (LOCKSTEP_LSSAB_SPLIT_TB=0, the S7s rule); the kernel's per-tile clamp
  is a no-op at cap 1 and stays. This closes the cheap levers on chat serving at 0.87x: what is left is
  the decode path itself (PLAN_S2R, D).
- 2026-09-10, Qwen2.5-14B re-measured (prepare at gmu 0.55 - the outlier calibration's 25 GB beside the
  engine): TTFT 0.89 / 1.10 / 1.09 / 0.99 at 128 / 2k / 8k / 32k, batched prompts 1.04 / 1.13 / 1.09
  (8k x 4 1.09), 128 x 32 prefill 1.15, decode 1.01 / 1.12 / 0.80 at batch 1 / 8 / 32 (filed before: TTFT
  0.95 at 8k, decode 1.08 at batch 1). The batch-32 decode row is the one number under 0.9 on the
  family: at 32 sequences x 8 KV heads the 14B's 256 tiles cross the S7s no-split threshold (one CTA
  per SM) where the 7B's 128 do not - the first case where that rule acts on a single-stream row, and
  it reads as a loss. A one-cell check (the same row with LOCKSTEP_LSSAB_SPLIT_FULL=0) is queued
  behind M0 to settle whether the rule needs the blocks-per-CTA condition it lacks.
