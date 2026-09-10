# LSSA-B: per-block key exponents — contract variant specification (2026-09-01)

Purpose: a variant of the filed contract that (i) keeps the whole per-score chain 32-bit,
(ii) makes the KV cache APPEND-ONLY (no rescale of old entries, no epoch groups, one
launch per layer, CUDA-graph capturable), (iii) yields exact tile-skip for free, and
(iv) stays chunk-invariant and batch-invariant by construction. Runtime math otherwise
identical to SPECIFICATION 6.1/6.2. Quality must be re-measured (T12b) before adoption.

## Definitions (all integers unless stated; BHI = B = 26, dh = 128, table (n, w))
- Keys are partitioned into BLOCKS of KB = 128 consecutive ABSOLUTE positions
  [0,128), [128,256), ... (fixed by position, never by chunk or batch composition).
- Key offset: per (layer, kv-head, channel) published constant mu (T9). k' = k - mu.
- Per-block key exponent: for block b and kv-head g, e_k[b,g] = pow2 exponent of the
  amax of k' over the block's rows AND all channels (clip-safe frexp rule of
  contract_common.quant_pow2 applied to that amax; clamp [-32,40]). Missing rows in the
  last (partial) block do not participate. k8 = clamp(round_half_even(k' * 2^e_k), -127, 127).
- Values: v8, e_v per (block, kv-head) with the SAME rule (block granularity for V too).
- Queries: q8, e_q per (query row, head) — per-row (T10 proved per-row exactness-compatible).
- Fold: G[b] = g_fold(dh, e_q + e_k[b], BHI) per (row, block), clamp [1, 2^30]. The
  g_fold shift argument sh = e_q + e_k[b] MUST be clamped to the safe band [-7, 23] before
  1 << (sh-1) (T10 finding: overflow at sh >= 64).
- Scores: S[j] = sum_c q8[c] * k8[j][c] exact (< 2^24 in fp32 or exact in s32 IMMA).
- Row bias: m_SG = max over ALL live keys j of S[j] * G[b(j)]  (64-bit per (row, block):
  max_j in b of S[j] is a 32-bit tile max m_S[b]; m_SG = max_b (m_S[b] * G[b]); note
  max_j S[j]*G = G * max_j S[j] because G > 0 is constant within the block).
  NB = least multiple of 2^B >= m_SG.
- Per score: d[j] = NB - S[j] * G[b(j)]. eh[j] = table[(d >> (B - n)) & (2^n - 1)] >> (d >> B)
  if d < (w+1) * 2^B else 0. l = sum eh (s64). A = sum eh[j] * v8[j] * 2^(-e_v[b(j)]) —
  NOTE the per-block V exponent: accumulate A per block scale group exactly. Simplest exact
  rule: accumulate A_b = sum_{j in b} eh[j]*v8[j] in s32 planes per block (bounded by the
  WPV 66,311-key rule, trivially satisfied per 128-key block), then A = sum_b A_b * 2^(-e_v[b])
  performed in f64 (exact: A_b < 2^38, e_v in [-32,40]) — OR, preferred and exact in
  integers (CORRECTED 2026-09-01 after T12a and T12b independently found a sign error in
  the first draft, which used min): define e_v_ref = MAX_b e_v[b] over the row's live
  blocks and accumulate A = sum_b (A_b << (e_v_ref - e_v[b])); every shift is >= 0 (a LEFT
  shift, exact), and sum_b A_b*2^(-e_v[b]) = 2^(-e_v_ref) * sum_b A_b*2^(e_v_ref-e_v[b]).
  A fits s64 for T <= 2^17 given an e_v spread <= 16 (measured spread <= 3 on Qwen2.5-7B,
  T12b); if the spread exceeds 16, fall back to the f64 accumulation. The golden implements
  this rule (paper_softmax/bigrun/lssab_golden.py, lssab_torch.py) and the kernel matches it
  bit-for-bit (T12a gate K1).
- Epilogue: unchanged — f32(A) / f32(l) single correctly-rounded division, then 2^(-e_v_ref)
  pow2 scale in f32 (exact), masked rows -> 0.

## 32-bit chain per block (derived, T11): with G = G[b], m_S[b] the block's raw max:
  r_b = NB - m_S[b] * G[b]  (>= 0, 64-bit computed ONCE per (row, block))
  if r_b >= (w+1) * 2^B: the ENTIRE block contributes exactly zero -> skip it (exact skip).
  else: for each key j in b: delta = m_S[b] - S[j] (s32, >= 0); if delta >= delta_max[b]
        := ceil(((w+1)*2^B - r_b) / G[b]) then eh = 0 else d = r_b + delta * G[b] (< 2^31),
        index/shift/gather as usual. All 32-bit. (Verified bound: max d < 1.55e9 for every
        (B=26, w in {15,16,21}, sh in [2,14]).)

## Invariance properties (must be gated, not assumed)
- CHUNK invariance: e_k[b], k8 depend only on the block's own rows -> identical for any
  chunking of the prompt. Append-only: a new block never changes an old block.
- BATCH invariance: all quantities are per-sequence functions of that sequence's tokens.
- TILE invariance: kernel tile sizes are SCHEDULE; block size KB=128 is CONTRACT.
- Exactness: bit-identical to an independent numpy golden of THIS document.

## What changes vs the filed contract (for the record / second provisional)
Per-block K/V exponents (was: per-tensor/per-KV-head); key offset (T9); fold-first-then-max
(T10); block-granular exact skip. Everything else identical.

---

# ADDENDUM — LSSA-B8: block-local 8-bit weight, committed-order single-pass fold

**STATUS: DRAFT (T16a, 2026-09-02). Contract-version bump ACCEPTED by the owner on 2026-09-02
(decision delegated: "do whatever is best to reach parity"); the committed block order is adopted for LSSA-B8** — LSSA-B8 deliberately surrenders cross-block schedule invariance
(SPECIFICATION 6.2.5 proves no exact schedule-*invariant* single pass exists; a schedule-
*committed* one is exactly specified and is a different function, not an approximation).
Nothing in the sections above this line is changed. LSSA-B (the filed variant) remains the
adopted contract until a quality battery (T16b) and a kernel gate (T16d) say otherwise.

Motivation: one u8 x s8 PV per k-step, no pass A, no second PV plane, no `e_v_ref`
machinery, no 8,192-entry table, and no `T <= 65,536` bound (`l` is folded in fp32).

## B8.0 Constants (all part of the contract version `v`)

| symbol | value | note |
|---|---|---|
| `B` = `BHI` | 26 | unchanged: one doubling = `2^B` |
| `KB` | 128 | key block, absolute positions; unchanged |
| `SEG` | 32 blocks (= 4,096 keys) | fold segment; scales with `KB` so `SEG*KB` = 4,096 |
| `S_WIN` | 32 doublings | exact-skip window inside a segment and across segments |
| `W_LOC` | **9** doublings (primary); **8** = speed fallback | block-local weight window |
| `n` | **7** index bits per doubling | `n = 8` is bit-indistinguishable (OFFLINE), so the table is halved |
| `TOP` | 255 | the top weight |
| table length | `17 * 2^n` (2,176 B at `n=7`) | zero-padded to the chain's proven index bound |
| block order | **ASCENDING** absolute block index | pinned; see B8.6 |
| `T2` digest | `sha256(T2_n7_W9 csv)` = `7a42a785940f557d5d5a4188fa5683ea00cf8b29e8d5213c75c581a5a0d92fbb` | |

Other filed digests: `T2_n7_W8` `77910b38cdc718a57e6c187956360747399a3c9a77d7dbf98ce0ac262e1da03f`,
`T2_n8_W9` `87b4e18c1fa1efc5638052184ddad7d6a7e61c95874fc5ad2d805e3093a5b1cf`,
`T2_n8_W8` `93b7f0d49eeea7b710fb07e0b6b81bd74aa080138a292292e87048b681c518b5`,
`T2TR_n7_W9` (truncation control, NOT a contract table)
`c95669494db6a64b11f51741b1d83615e49a8df4e885d90148f0765665d68155`.

## B8.1 PREP — unchanged

Identical to the sections above: key offset `mu`; per-`(block, kv-head)` `e_k[b]`, `e_v[b]`
by the clip-safe `frex`-rule of `contract_common.quant_pow2` over the block's rows and all
channels, clamped to `[-32, 40]`; per-`(row, head)` `e_q`; `int8` `q8/k8/v8`;
`G[i,b] = g_fold(dh, clamp(e_q[i] + e_k[b], -7, 23))` in `[1, 2^30]`; exact
`S[i,j] = q8 . k8` with `|S| < 2^21`. `lssab8_golden.py` imports the LSSA-B prep verbatim.

## B8.2 The weight table `T2`

    T2[k] = RN(TOP * 2^(-k / 2^n))   for 0 <= k < W_LOC * 2^n
    T2[k] = 0                        for W_LOC * 2^n <= k < 17 * 2^n

generated from **40-digit `Decimal`** arithmetic. **Tie rule: HALF-UP (ties away from zero),
stated normatively.** Exactly one exact tie exists in the family — `k = 2^n`, value `127.5`
— and it rounds to `128` under half-up *and* under round-half-even, so the rule is
immaterial in value; it is pinned so a golden and a kernel cannot disagree by convention.
Verified: `T2[0] = 255`, `T2[2^n] = 128`, `T2[9*2^n - 1] = 1`, `T2[9*2^n] = 0`.
(Correction to the design text: at `n = 8` the entry `T2[2047]` is **1**, not 2.)

**DECISION D1 — `W_LOC = 9` is the MAXIMUM, not a tunable.** `RN(255 * 2^-x) = 0` for
`x > log2(510) = 8.994`, so a u8 weight referenced to `TOP = 255` has no support past 9
doublings: `W_LOC = 10` produces a **bit-identical table** to `W_LOC = 9` (same sha256;
gate S0). "Widen to `W10`" is therefore *not* an available fallback. The fallback ladder if
`W_LOC = 9` ever fails a quality bar is: `KB = 64` (2x fold cost), then `L16`.
`W_LOC = 9` costs **nothing** over `W_LOC = 8` in table bytes or in the chain (both are the
same 2,176-byte zero-padded table); it costs one index clamp — see B8.3.

## B8.3 The chain, per `(row i, block b)`

    m_S[b] = max over VISIBLE keys j in b of S[i,j]        (exact s32 tile max; FA3's rowmax slot)
    NB_b   = least multiple of 2^B >= m_S[b] * G[b]
    d[j]   = NB_b - S[i,j] * G[b]                          (>= 0, a mathematical integer)
    eh[j]  = T2[ d[j] >> (B - n) ]                         (T2[k] = 0 for k >= W_LOC * 2^n)

Masked / invisible keys are defined to have `eh = 0`.

**Implementation device (schedule, NOT contract), with its correctness condition.** A kernel
evaluates the chain in 32-bit with `r_loc = NB_b - m_S[b]*G[b]` in `[0, 2^B)`,
`S_c = max(S, m_S[b] - dm)`, `d = r_loc + (m_S[b] - S_c) * G[b]`, `k = d >> (B-n)`.
This is bit-identical to the definition **iff `dm * G[b] >= W_LOC * 2^B`**, because every
clamped key then lands in the table's zero region and every unclamped key is unchanged:

* `W_LOC = 8`: `dm = 1 << max(0, 29 - floor_log2(G))` gives `dm*G` in `[2^29, 2^30)` = 8
  doublings; `d < 2^26 + 2^30 < 2^31` (signed-safe) and `k < 2,176` — **no index clamp**.
* `W_LOC = 9`: `dm = 1 << max(0, 30 - floor_log2(G))` gives `dm*G` in `[2^30, 2^31)` >= 9
  doublings; `d < 2^26 + 2^31 < 2^32` so `d` must be carried as **unsigned** 32-bit, and
  `k < 4,224`, so either **one `IMNMX` clamping `k` to 2,175** (2,176-byte table) or a
  4,224-byte zero-padded table. Both give identical values.

## B8.4 Block partials — exact integers

    A_b[i,:] = sum_{j in b} eh[j] * v8[j]     |A_b| <= 255 * 127 * 128 = 4,145,280 < 2^22
    l_b[i]   = sum_{j in b} eh[j]             l_b  <= 255 * 128       =    32,640

`s32` accumulation of `int8` products is exact by specification on every vendor (Phase 0:
90/90 bit-identical on A100 / H100 / x86 / ARM) and an exact sum is order-invariant, so **any
K-tiling inside a block and any MMA ordering give the same `A_b`**. Both bounds are below
`2^24`, so `f32(A_b)` and `f32(l_b)` are **exact** (and the magic-number I2F path
`IADD + FADD` is valid, with 49,024 counts = 1.17% of margin).

## B8.5 The fold — committed order, fp32, two-level

Per row, over **segments** of `SEG` consecutive blocks by absolute index, ascending; inside a
segment over the segment's visible blocks, ascending:

    D = (NB_b - NB_run) >> B                      (a signed count of doublings)
    D >  0 : O = rescale(O, D); l = rescale(l, D); NB_run = NB_b; s = 0
    D <= 0 : s = -D ; if s > S_WIN the block contributes EXACTLY zero -> skip it
    O = fl_RNE( O + f32(A_b) * 2^-(s + e_v[b]) )
    l = fl_RNE( l + f32(l_b) * 2^-s )

with `NB_run`, `O`, `l` initialised at the segment's first visible block. The finished
segment `(O_seg, l_seg, NB_seg)` is then folded into the row by **the same three rules**.

**`rescale(x, D)` is an EXPONENT-FIELD SUBTRACTION, not a float multiply**: subtract `D`
from the biased exponent field of `x`; if `x` is zero or the result's biased exponent would
be `< 1`, the result is `+0.0` (sign cleared). It is exact for every normal input and never
rounds. This closes the FTZ gap: FA3 is built with `--use_fast_math`, which implies
`-ftz=true`, and the IEEE product `(2^24 - 1) * 2^-150` **rounds UP to `2^-126`** while a
flush-before-rounding implementation gives 0 — the two differ, and neither is a safe
contract. Gate S1 pins the boundary case.

**DECISION D2 — the design's `if D >= 126: reset to 0` rule is DROPPED.** It is *not*
implied by the rescale rule (a value with biased exponent > `D` survives a `D = 126`
rescale as a normal number), so keeping it would define a third, less accurate function.
The exponent-field rule is total for every `D` and needs no special case.

**Ranges.** `s <= S_WIN = 32` and `e_v` in `[-32, 40]`, so the term scale exponent
`-(s + e_v)` is in `[-72, 32]` and, with `1 <= |A_b| < 2^22`, every nonzero term is a
**normal** binary32 (`> 2^-72`, `< 2^54`). The reference asserts this at runtime.

**Epilogue.** `out = O / l` with ONE correctly-rounded binary32 division — the kernel MUST
use `__fdiv_rn`, never `div.approx` (the T12a `--use_fast_math` trap) — then RNE to bf16.
**If `l == 0` the row output is 0** (the degenerate rule, unchanged). There is no
`e_v_ref`, no cummax and no plane split: the V exponent rides in the term scale.

**The reference must never use `cumsum` or any parallel scan.** The fold is a sequential
RNE recurrence; a scan re-associates the additions and rounds differently. Vectorisation
over rows, heads and `dh` is legal (independent recurrences); vectorisation over blocks is
not.

## B8.6 Invariance — what holds and what is surrendered

HOLDS (gated, not assumed): batch invariance; **chunk invariance for 128-ALIGNED chunk
boundaries** (append-only blocks, unchanged rule — the engine's block-aligned split hook is
therefore mandatory, gate S8 constructs a non-aligned cut that really differs); K-tiling
inside a block; M-tiling; tile size; `qblk`/`kblk`/head-chunk (gate S10); split-KV decode ==
prefill (gate S7/S9), which is *why* the fold is two-level.

SURRENDERED: cross-block schedule invariance. **Block order (ASCENDING), `SEG`, `S_WIN`,
`W_LOC`, `n`, the `T2` digest, the tie rule and the rescale rule are contract constants.**
FA3's causal mainloop walks `n_block` DESCENDING and would compute a different function;
the walk order must be pinned in the kernel and gated with a descending-walk negative
control.

Inherited, not new: a row's function depends on whether its position was a prompt or a
generated token (the open block is re-quantised per step, T13B §2.2).

## B8.7 The exact skip, and why NO SKIP CREDIT IS CLAIMED

A block with `s > S_WIN` contributes exactly zero *by definition of the recurrence*, so
skipping it is not an approximation. But the predicate is **segment-local**: `s` is measured
against the segment's own running max, which is all a single pass knows. OFFLINE on the
captured real score rows (T16A_LOG.md §2), the segment-local rate at `S_WIN = 32` is
**0.29% / 0.76% / 0.00%** of visible `(row, block)` pairs at 262k / 1M / 32k, against a
row-global rate of 2.5% / 15.9% / 0.0% — and both collapse further under the AND-over-rows
effect a CTA tile imposes (T11 MEASURED 4.17% dead pairs -> 0.0% dead CTA tiles).
**No performance model built on this contract may take a tile-skip credit.**

## B8.8 O_row placement — an implementation constraint, not contract

The prefill CTA must hold BOTH the segment accumulator `O_seg` and the row accumulator
`O_row` at every segment boundary. The tile register budget (`S` 64 + PV 68 + `O` 68 + `P`
16 = 216) omits `O_row` and FA3's ~56-66 registers of addressing/loop/pipeline state that
T13A §4.2 measured (stock FA3 reaches R226 with 160 tile registers), so ~272-282 against the
240 consumer cap: **a ~30-40 register spill is the base case, not a risk.** T16c must
measure `STACK` bytes and test an `O_row`-in-SMEM variant before any exact build.

## B8.9 Reference implementations and gates

| file | role |
|---|---|
| `lssab8_golden.py` | python-int chain, explicit scalar fp32 RNE fold, exponent-field rescale, split-KV segment helper. PREP imported from `lssab_golden.py` (unmodified). |
| `lssab8_torch.py` | torch reference: per-block maxima, block-local `d`, `T2` gather, per-block `A_b` as an exact integer matmul, a **sequential** fold loop over blocks vectorised over rows/heads/`dh`. Arms `L8-RN-W9` (default), `L8-RN-W8`, `L8-TR`, `L8-RN-n8`, `L8-RN-KB64`, `L16`. |
| `lssab8_selftest.py` | the CPU gate: 28/28 checks, 119 torch-vs-golden cases bit-for-bit. |
| `l8_offline.py` | the OFFLINE scan behind D1, B8.7 and T16A_LOG.md. |

`t12b_ppl.py` / `t12b_long.py` gained one additive `build_arm8()` hook that returns `None`
for every pre-existing arm name, so no T12b number can move; `lssab_selftest.py` still
reports `GOLDEN SELFTEST: PASS` (239/239).

# ADDENDUM — LSSA-B9: the value operand's grid (Stage 2 item 2.0b, 2026-09-08, PROVISIONAL)

**Why.** Bisected on Llama-3.1-70B in the torch reference (SPEC 7.23): of the exact attention's
+2.83 percent of perplexity, +2.67 is the VALUE operand quantised to 7 bits with one exponent per
128-key block. The keys at the same granularity cost +0.17, the weight table nothing (16-bit
+2.80), the block size nothing (64-key +2.80), the key offset is worth 0.2. The first key of the
sequence (the attention sink, absolute position 0) sets block 0's value exponent for its 127
neighbours and is most of the cost (+0.46 with key 0 quantised alone); the rest is the spread of
value magnitudes inside every block (one more bit of value mantissa: +0.12; 9 bits per 128 keys:
+0.10; 15 bits: +0.02).

**Two options on B8, both OFF = B8 bit for bit** (`lssab9_torch.py`, gated identical to
`lssab8_torch.py` with both off). Everything not named here - PREP for q and k, the key offset,
the scores, `G`, `NB_b`, `d`, the table `T2`, `l_b`, the fold's rescale and skip rules, the
segment structure, the epilogue - is B8's unchanged.

## B9.1 `sink_alone` — key 0 is its own value block

Value exponents: `e_v0` over key 0 alone; `e_v[0']` over keys `1 .. KB-1`; `e_v[b]` for `b >= 1`
as in B8; each the natural pow2 exponent of the 7-bit rule (NO cap between key 0 and its
block: on Llama the sink's values exceed its neighbours' by more than 2^8, and a shift bounded
by two u8 weight planes was measured to buy nothing, 8B +0.26 against +0.08 for this rule).
Block 0's partial is TWO integer partials in the declared order

    A_0  = eh[0] * v8[0]                      (one key; |A_0| < 2^15)
    A_0' = sum_{j = 1}^{KB-1} eh[j] * v8[j]

folded as two consecutive terms with the SAME `s` (the block's doublings gap; NB_0 is still the
max over all KB keys) and their own exponents:

    O = fl(O + f32(A_0)  * 2^-(s + e_v[0]));   O = fl(O + f32(A_0') * 2^-(s + e_v[0']))

`l_b` is unchanged (one sum over the block). Keys are not split: `e_k[0]` is over the whole block.
Kernel cost: key 0's column zeroed in block 0's P.V MMA and one rank-1 fp32 update per row
(128 multiply-adds) in block 0's fold, on the CUDA cores.

## B9.2 `v_planes = 2` — a 15-bit value in two int8 planes

Per `(block, kv-head)`: `e_v15 = pow2 exponent for a 14-bit magnitude` (the 7-bit rule's exponent
plus 7, the same clip-safe frexp rule with limit `2^14 - 1`, clamp `[-32, 48]`);
`v15 = clamp(rn_even(v * 2^e_v15), -16383, 16383)` (a 15-bit signed value);
`v_hi = v15 >> 8` (arithmetic shift, in `[-64, 63]`, a signed int8 plane);
`v_lo = v15 - (v_hi << 8)` (in `[0, 255]`, an unsigned int8 plane).
Block partials, both exact in binary32 (`|A_hi| < 255 * 64 * 128 < 2^21`, `|A_lo| < 255 * 255 * 128 < 2^24`):

    A_hi = sum_j eh[j] * v_hi[j];    A_lo = sum_j eh[j] * v_lo[j]

folded as two consecutive terms, HIGH first:

    O = fl(O + f32(A_hi) * 2^(8 - s - e_v15));   O = fl(O + f32(A_lo) * 2^-(s + e_v15))

`l_b` unchanged. Term exponents stay in the normal binary32 range (asserted by the reference).
Kernel cost: two P.V products per block (an s8 x s8 and an s8 x u8 IMMA), the V half of the KV
cache at 16 bits per element (the K half stays 8).

With both options on, block 0 folds four terms: key 0 high, key 0 low, keys 1..KB-1 high, low.

## B9.5 `v_keyshift` (LSSA-B9.3) - every key on its own grid, compensated on the weight

The value operand's remaining cost after B9.1 is inside the block: 127 keys share one exponent
and the small-norm keys lose their low bits. B9.3 gives every key j of block b its own grid
`e_v[b] + d_j`, with `d_j = min(dmax, e_j - e_v[b])` and `e_j` the key's own 7-bit exponent
(the same amax rule as the block's, applied to one row; `d_j >= 0` because the block's amax
bounds the row's), so `v8[j] = round(v[j] * 2^(e_v[b] + d_j))` clipped to +-127, and stores
`d_j` in two bits (dmax = 3). The PV product must then carry `2^-d_j`, and it is put on the
weight: the plane's operand is `eh'[j] = shift(eh[j], d_j)` where `shift` is one of the
declared roundings (floor; round half up; round to nearest ties to even), an exact integer,
while `l_b = sum_j eh[j]` keeps the UNSHIFTED weight - the normalisation counts the key's
true weight, the product carries its finer value. The fold, the exponents, the sink handling
(unneeded: the dominant key of a block has `d = 0`) are B8's. Cost: two bits per key in the
lattice (the V row's spare byte), and per score one 2-bit extract and one integer shift in
the chain (measured 1 to 2 percent of TTFT at 8k to 32k on Hopper, SPEC 7.23). The rounding
is part of the rule and a per-model manifest parameter: the two models measured prefer
different roundings (SPEC 7.23; round half up is the 70B's best and the default, ties to
even the 8B's). Served: Llama-3.1-70B at TP = 4 with 32 outlier channels +0.81 percent, the
prompt and generated rows bit-identical to the reference; Llama-3.1-8B +0.33 (SPEC 7.23).

## B9.3 Gates and the decision

`lssab9_gate.py`: identity to B8 with both off (0 differing bits on four shapes, PASS) and
finiteness of each option. Measured (SPEC 7.23, `attn_bisect.py` arms), attention alone
against bf16 on 30 windows: Llama-3.1-70B: B8 +2.83, B9.1 +0.73, B9.2 +0.21, both +0.29;
Llama-3.1-8B: +0.26, +0.08, -0.08, -0.01. Decision: B9.1 is the candidate default (nearly free
in the kernel); B9.2 is the per-model option for a 70B under 1 percent, at twice the P.V
product, subject to the plan's kill line (no more than 10 percent of prompt throughput at 32k)
once the kernel exists. Both are recorded in the manifest as the attention rule's version.

## B9.4 B9.1 in the kernels (2026-09-08, built and gated)

`lssab9_golden.py` is the oracle (numpy; `lssab9_selftest.py` ties `lssab9_torch.py` to it,
176 / 176). Three implementations carry the rule, and one flag selects it everywhere
(`LOCKSTEP_SINK=1`, manifest `attention.rule = LSSA-B9.1`; 0 is the B8 kernel bit for bit):

- **PREP** (`lsb_prep` and the fused decode-step prep): block 0's row 0 is scanned into its own
  amax; `ev0[page][kvh]` holds `e_v0`, written on the visit that scans row 0 and re-read on
  every later visit (row 0 is never re-quantised with any other exponent); the block's amax,
  and the running shadow amax, cover rows 1..KB-1.
- **DECODE** (`lsb_dec8`, `lsb_dec8w`, `lsb_dec8p`): key 0's u8 weight is taken out of the
  plane (`eh[0] = 0` after `l_b` summed it), its s32 product `eh[0] * v8[0][c]` is formed from
  the transposed V tile, and the fold applies it first with `2^-(s + e_v0)`, then the block
  term. Under split-KV the sink is one more exact partial: slot 0 of the block-partial buffer
  (`l = 0`, block 0's NB, `e_v0`), block b at slot b + 1, so the segment fold needs no new rule
  (a zero-weight pseudo-block folded before block 0 is the golden's two-term block).
- **SEAM** (`patch_b9.py` on `b8x + b8eng + b8varlen` = build `b9vareng`): the thread holding
  tile column 0 of block 0 keeps `eh[0]`, writes 0 into the u8 A-operand, adds it back for
  `l` before the quad reduce; the fold folds `f32(eh[0] * v0[c]) * 2^-(s + e_v0)` before the
  block term, `eh[0]` reaching the quad by one width-4 shuffle and `v0` (key 0's int8 row) and
  `e_v0` arriving per (batch element, kv head) from the engine glue (`fg_aux`: `ev0c`, `v0c`).

Gates: `lssab8_gate.py` on `b9vareng` with the sink off 8 / 8 (the B8 kernel is unchanged);
`lssab9_seam_gate.py` 48 / 48 (out bits and the fp32 bit patterns of O_row, l_row and NB_row
against the numpy golden on 14 shape-by-sink cases, the negative control differing whenever
`e_v0 != e_v[0]`, determinism, the exact skip, and 1k to 4k against `lssab9_torch`);
`t16n_gate.py` 21 / 21 with the sink off (against `lssab8_torch`) and 21 / 21 with it on
(against `lssab9_torch(sink_alone)`), on the base, pipelined and 16-warp decode kernels.
Served (SPEC 7.23): Llama-3.1-8B B8 +0.30 / B9.1 +0.25 on one build, E1 / E4 0 bad against each
rule's reference; Llama-3.1-70B at TP = 4 B9.1 +1.77 from +3.71.
