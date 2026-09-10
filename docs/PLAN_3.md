# Plan: Stage III - the open mixture-of-experts models, exact and verifiable

Written 2026-09-10. Stage II's dense stack is at parity on four families (SPEC 7.25-7.26, PR #9); its
remaining items (2.2 Qwen3 q/k norm, 2.3 long context, 2.5 multi-node TP, the decode-path rewrite) go
to the backlog with their plans written (PLAN_22, PLAN_S2R). Stage III is the capability that does
not exist yet, and it is built the way the dense stack was: the contract first, a torch reference that
is the definition, one model exact end to end, then speed against the stock FP8 / FP4 baseline.

## 1. What is actually used (OpenRouter, week of 2026-09-08)

Token volume, not benchmarks, decides which models matter to the customers we name. Of the top
twelve models by daily tokens on OpenRouter on 2026-09-08, eight are open-weight, and every one of
them is a mixture of experts:

| rank | model | total / active | experts (routed, top-k, shared) | attention | shipped precision | licence |
|---|---|---|---|---|---|---|
| 1 | Tencent Hy4 Preview | 770B / 49B | 256 + 1 shared, top-8; 77 MoE layers of 78 | Gated DeepSeek Sparse Attention (indexer + sparse), iHC residuals | bf16 + FP8 release | open weights |
| 3, 10, 11 | GLM 5.3 Flash / 5.3 / 5.2 | 320B / 18B (Flash) | 288, top-8 | hybrid: KDA linear attention + NoPE sparse MLA, 45 layers | FP8 | MIT |
| 4, 6 | DeepSeek V4 Flash (two listings) | 284B / 13B | 256, top-k | MLA head dim 512 + compressed sparse attention (CSA / HCA), mHC residuals | FP4 experts, FP8 rest | open |
| 7 | Tencent Hy3 | (previous generation of 1) | | | | open |
| 9 | Nvidia Nemotron 3 Ultra | 550B / 55B | Latent MoE (8192 -> 2048 -> 8192), MTP | hybrid Mamba + attention | NVFP4 | open |
| 12 | Meta Muse Spark 1.3 | | | | | open-ish |
| (monthly leaders earlier in 2026) | Xiaomi MiMo V2.5 / Pro | 310B / 15B; 1.02T / 42B | MoE, 3 MTP heads | SWA + global attention 6:1, window 128, attention sink bias | | MIT |
| (monthly leader, Feb 2026) | Kimi K2.5 / K3 | 1T / 32B; 2.8T / 104B | 896, top-16 (K3), Stable LatentMoE | KDA linear + gated MLA 3:1 | MXFP4 (K3) | Kimi licence |
| (free tier, 18 providers) | OpenAI GPT-OSS-120B | 117B / 5.1B | 128, top-4 | GQA, attention sinks, sliding windows | MXFP4 experts | Apache-2.0 |
| (our current family) | Qwen3.5 397B-A17B; Qwen3.8-Flash-Next 125B-A6B; Qwen3-235B-A22B; Qwen3-30B-A3B | | 512, top-10 + 1 shared (3.8); 128, top-8 (Qwen3) | Gated DeltaNet + sparse attention (3.5 / 3.8); plain GQA (Qwen3) | bf16 / FP8 | Apache-2.0 |

Sources: the OpenRouter rankings as reported on 2026-09-08 (tokenmaxxing.com/openrouter-rankings;
openrouter.ai/rankings, data through Sep 8), presenc.ai's 2026 usage study (Chinese-origin models at
about 46 percent of identified tokens), and the model cards / vLLM recipes cited in the log below.

Three facts follow, and they set the plan:
1. **The expert path is universal.** Router, token-to-expert dispatch, grouped expert GEMMs, combine,
   a shared expert. Build it once, exactly, and it serves every model above.
2. **Attention is no longer one kernel.** Every frontier open model interleaves a linear-attention
   family (Gated DeltaNet, KDA, Mamba) or a sparse-attention family (DSA, CSA / HCA, sliding windows
   with sinks) with full attention, and three of them use MLA. Each family is a new declared rule
   and a new exact kernel. This is the long pole, longer than the expert path.
3. **The weights ship quantised.** FP8, FP4, MXFP4, NVFP4. The stock baseline is not bf16 any more and
   the contract cannot re-quantise a quantised checkpoint and call the result the model: the declared
   arithmetic has to consume the shipped low-precision weights exactly (their scales and block
   formats become contract inputs), and "exact" is measured against the bf16 upcast of that
   checkpoint run by the reference, not against a bf16 model that no longer exists.

## 2. The contract, v3 (what is declared, in order of build)

- **3.1 The router.** Gate logits from the declared int8 GEMM (or from the shipped FP8 gate weights
  upcast exactly), the softmax or sigmoid on the declared table, top-k with a written tie-break
  (lowest expert index wins), optional renormalisation on a fixed grid, the routing weights as
  declared bf16, the shared-expert gate. Batch-invariant by construction; reference in torch;
  vectors published. This is where "full semantics" starts: the router is a discrete decision, and
  one flipped bit there changes which experts run - the fingerprint must cover the routing, not only
  the arithmetic after it.
- **3.2 Dispatch and combine.** The permutation of tokens by (expert, token) order, the per-expert
  row counts as declared integers, the combine as a rank-order sum over the k experts in declared
  order (the same fold rule TP uses), the shared expert added last. Expert parallelism with a pinned
  all-to-all when the model spans GPUs.
- **3.3 The expert GEMMs.** Grouped int8 GEMM with the v2.1 epilogue per expert where the checkpoint
  is bf16 (Qwen3); for FP8 / FP4 / MXFP4 checkpoints, the declared dequant of the shipped block
  format into the int8 or bf16 operand the contract multiplies, per-expert scales in the manifest.
  CUTLASS grouped GEMM as the route, the exact epilogue outside it, as with the dense linears.
- **3.4 Attention rules for the hybrids.** In order of model usage: sliding windows with a sink
  (GPT-OSS, MiMo); MLA (DeepSeek, GLM, Kimi: the absorbed low-rank KV inside the LSSA-B8 rule at head
  dims 512 / 576); Gated DeltaNet / KDA (Qwen3.5 / 3.8, GLM, Kimi: a recurrent state update - a new
  exactness problem, integer state on a declared grid); DeepSeek's sparse attention (an indexer that
  selects keys: a discrete decision, like the router, that the fingerprint must cover).
- **3.5 Multi-token prediction heads** (Nemotron, MiMo, DeepSeek): the draft path must be exact or
  declared off; off first.

## 3. Milestones, each with its gate

- **M0 - the reference and the first vehicle (this week, one H100).** Qwen3-30B-A3B: 48 layers, 128
  experts top-8, plain GQA, SwiGLU, bf16, 61 GB - runs on one H100, and every block but the expert
  path is already exact in our stack. Torch reference of the MoE block under 3.1 - 3.3; vLLM's
  `Qwen3MoeSparseMoeBlock` / `FusedMoE` replaced by the contract path (the same arming pattern as
  t16m); the per-op identity gate on captured activations; the fingerprint on 300 windows.
  Gate: bit-identical to the reference on every routed token; perplexity against stock bf16 within
  the dense stack's band. Kill line: none - this is the definition.
- **M1 - the expert path fast enough to measure (week 2).** Grouped int8 GEMM via CUTLASS with the
  exact epilogue; the dispatch / combine as one kernel each; the router on the fused dense path. Gate:
  M0's identity; measure against stock bf16 on the 30B-A3B (single-stream and serving).
- **M2 - the first model people use: GPT-OSS-120B (weeks 3-4, one H100).** MXFP4 expert weights
  consumed exactly (the block format declared), 128 experts top-4, attention sinks and sliding windows
  as contract rules (3.4 first family). Gate: fingerprint + 300 windows against the stock MXFP4
  engine; serving against stock on one H100. This is the first number a customer can use.
- **M3 - the 8 x H100 class (weeks 5-8).** Qwen3-235B-A22B (bf16, classic GQA: the expert path at
  scale with expert parallelism and the pinned all-to-all, TP across 8 GPUs - item 2.5's reduction is
  built here) then GLM-5.3-Flash or DeepSeek V4 Flash (FP8 / FP4 checkpoints, MLA + the sparse /
  linear attention families). Gate: exact at TP / EP against the TP=1 reference fingerprint of the
  same manifest; serving within the 15 percent line against stock FP8.
- **M4 - verification at scale.** Per-expert commitments, the routing decisions inside the commitment
  root, sampled replay of routed tokens on CPU, the dispute protocol on a routed token.

## 4. Hardware and budget

M0 - M2 on the existing H100 (lc-handover). M3 needs an 8 x H100 node for about two weeks (about
USD 25 / h, rented per campaign and killed). The roadmap's Stage III figure (3 - 5 months, USD 25k)
stands; the first customer-usable number (M2) is about a month away.

## 5. Kill lines and the honest unknowns

- If the exact grouped GEMM cannot reach 70 percent of the stock fused-MoE kernel on the 30B-A3B by
  M1's end, the speed target for Stage III moves from 15 to 25 percent overhead and the roadmap says so.
- The linear-attention families (Gated DeltaNet, KDA, Mamba) have no exact integer formulation yet;
  if one cannot be found that the reference can replay on a CPU, those models are served with that
  layer family declared "stock bf16, not verified" and the scope statement says so.
- Low-precision checkpoints: the contract's exactness claim becomes "this model, as shipped, evaluated
  exactly as declared" - the perplexity comparison is against the shipped precision's stock engine.

## Log
- 2026-09-10, M0 read: vLLM 0.25.1's `Qwen3MoeSparseMoeBlock` runs the gate INSIDE `FusedMoE`
  (`is_internal_router`: the block hands `hidden_states` in as `router_logits` and the layer applies
  the gate, routes and runs the experts in one call); routing is `ops.topk_softmax` - an fp32 softmax
  over the gate logits, top-k, optional renormalisation (`norm_topk_prob`) - with sigmoid, grouped
  top-k with a score bias (DeepSeek V3), `sqrtsoftplus` with renormalisation (DeepSeek V4) and MiniMax's
  bias form as the other routing methods in `fused_moe/config.py`. So the contract router has four
  declared scoring functions to cover across the usage table, and the replacement point is the
  `FusedMoE` layer, not the block: the same arming pattern as t16m's linears. M0's first build: the
  torch reference of `softmax -> top-k (ties: lowest index) -> renormalise` on declared fp32 with a
  table softmax, compared bit for bit against `ops.topk_softmax` on captured gate logits, then the
  expert path. The 30B-A3B download and the stock engine's structure dump are queued (`s21_m0prep`).
- 2026-09-10, the replacement point, read in vLLM 0.25.1: `FusedMoE(...)` is a factory; it returns a
  `MoERunner` (`fused_moe/runner/moe_runner.py`) that owns the gate (`is_internal_router`), a
  `RoutedExperts` module with the expert weights (`w13_weight` [E, 2I, H], `w2_weight` [E, H, I] in
  bf16 for Qwen3), the optional shared experts, and `forward(hidden_states, router_logits)` ->
  `_forward_entry` (a registered custom op, so the graph captures it) -> `_forward_impl` (routing,
  dispatch, the quant method's `apply`, combine, the shared-expert add). The contract path replaces
  `MoERunner.forward` on armed layers the way t16m replaces the decoder layer's forward: at arm time
  the per-expert int8 weights and scales come from the manifest (or are folded from `w13_weight` /
  `w2_weight` on the bf16 checkpoint), the gate's from `self.gate`; at run time the fused contract op
  takes the block's input and returns its output, registered as a custom op so CUDA graphs capture it.
  M0's harness hooks the block (`Qwen3MoeSparseMoeBlock.mlp`) above this point, so it measures the
  whole replacement surface.
- 2026-09-10, `MoERunner._forward_impl` read: the gate is `self.gate(hidden_states)` (a bf16
  ReplicatedLinear; its logits are bf16, then `ops.topk_softmax` works in fp32 on them), followed by
  the optional dispatch (expert parallelism), the quant method's `apply` (routing + the fused expert
  kernel) and the combine with the shared experts. So stock's routing decisions already depend on a
  bf16 GEMM that is not reproducible across hardware - the contract's declared int8 gate with its
  bf16 epilogue (7.27 step 1) is a different but DEFINED function; M0 reports the agreement rate
  between the two as information, and the 300-window perplexity decides whether the declared router
  costs quality. The whole of `_forward_impl` is what the contract op replaces for an armed layer;
  `forward`'s input / output transforms and padding around it stay.
- 2026-09-10, M0's first run on the real model (`handover/m0`): the stock 30B-A3B boots in 31 to 41 s on
  one H100 and generates; the capture works (w13 [128, 1536, 2048], w2 [128, 2048, 768], k = 8,
  renormalise on); the reference failed in `torch._int_mm`, which needs more than 16 rows, on
  per-expert groups of 1 to 3 tokens - zero-row padding to 17 (the dense stack's rule) fixes it
  without changing a product; the harness's prompts were also too short (26 tokens) and are now
  about 1,000. Re-run queued (`s21_m0c`).
- 2026-09-10, M1's route, read: vLLM 0.25.1 ships `ops.cutlass_moe_mm` - one grouped GEMM over all
  experts (per-expert problem sizes and token offsets, per-token and per-channel scales) documented
  as the FP8 path; whether the CUTLASS grouped kernel behind it takes int8 operands with an int32 or
  exactly-scaled output is the first thing M1 checks on the pod (the dense stack's `cutlass_scaled_mm`
  int8 route does, and its fused epilogue proved admissible). If it does, the expert path is: the
  contract's dispatch permutation -> one grouped int8 GEMM for gate_up (the declared epilogue, bf16
  out) -> the SiLU table -> the per-expert row quant -> one grouped int8 GEMM for down -> the combine
  in declared order. If it does not, `torch._int_mm` per expert is the exact reference route and the
  grouped kernel is written as in Stage 1. The manifest side: the dense fold (`t16l_stack._fold_and_rotate`)
  walks the layers' linears - for the experts it walks `w13_weight[e]` / `w2_weight[e]` with the same
  `quantize_weight_int8` and per-expert `ws`; under R1 online the norm gains stay in the norms and the
  router's input is the same rotated, quantised row the experts see.
- 2026-09-10, M1's route settled by a probe: vLLM's `cutlass_moe_mm` refuses int8 ("A tensors must be
  of type float8_e4m3fn", grouped_mm_c3x_sm90.cu) - the shipped grouped kernel is FP8-only. So the
  exact expert GEMMs are ours: M1 runs correctness on `torch._int_mm` per expert (256 launches per
  layer on the 30B-A3B, exact, slow) and the speed form is a CUTLASS 3.x grouped int8 GEMM with the
  int32 output and the declared epilogue outside it, written the way Stage 1 wrote the dense route
  (M1b, days). For the FP8 / FP4 checkpoints (M2 onward) the question inverts: the shipped grouped
  FP8 kernel's epilogue must be shown admissible against the declared dequant, as 7.25's probe did for
  the dense int8 epilogue, or be bypassed the same way.
- 2026-09-10, M0 on 915 real tokens (`handover/m0`, layers 0 and 24): the reference ran end to end on
  GPU (0.13 s) and CPU (26 s) and disagreed with itself and with vLLM - routing SET agreement 0.0 with
  top-1 agreement 0.96 / 0.97. The cause is in my router draft, not the model: the scores indexed
  the exponential table directly with a 2^-16-grid difference, where the attention rule's table is
  indexed by bits 13..25 of a log2 distance on a 2^26 grid and shifted by the doublings (d >> 26) -
  so every expert but the top one read the table's tail. Fixed: `d = rint(fl32(m - g) * fl32(2^26 /
  ln 2))`, one binary32 multiply rounded once (the SiLU's own pattern), then the rule's lookup; the
  spec text (7.27 step 2) is corrected with it. The harness now reports the determinism stage by
  stage (gate logits, indices, weights, output) so the next disagreement names its operation. Re-run
  queued.
- 2026-09-10, M0 with the corrected scores (`handover/m0`, 915 tokens): the declared router agrees with
  vLLM's on the expert SET for 91 percent of tokens at layer 0 and 74 percent at layer 24 (top-1 0.97 /
  0.96; 99 / 97 percent of the selected experts shared), and the contract block's output is 3.3 / 12.5
  percent of stock's in relative L2 - the declared int8 gate is a close but different router, as
  expected, and the 300-window perplexity will judge it. Determinism, by stage: indices equal on GPU
  and CPU; the GATE LOGITS differ by one bf16 ulp on some elements, and everything after follows. The
  operation is the row quantisation's `am / 127.0`: PyTorch's CUDA division by a python scalar
  multiplies by the reciprocal, the CPU divides - the dense stack learned this (`_const`) and the
  reference now divides by a tensor. Re-run queued.
- 2026-09-10, M0 DETERMINISM PASSES (`results/s3/m0`): on 915 real tokens of Qwen3-30B-A3B at layers 0 and
  24, the contract's MoE reference evaluated on the H100 and on the CPU agrees bit for bit at every
  stage - gate logits, expert indices, routing weights, output (0 rows differ). The definition replays
  anywhere, which is what a verifier needs. Against stock on the same activations: the expert SET
  agrees on 91 / 74 percent of tokens (top-1 97 / 96, 99 / 97 percent of the selected experts shared),
  output 3.3 / 12.5 percent relative L2 - the declared int8 gate is a defined router close to stock's;
  the perplexity gate (running: 30 windows of 2,048 tokens, stock against the armed engine) says what
  the difference costs. Two lessons from the day filed for the next reference: index the exponential
  table on the rule's own grid, and divide by tensors on CUDA.

## M1 design (written ahead of the perplexity verdict; the expert path does not depend on it)

The engine op `t16moe::forward(x bf16 [T, H]) -> y bf16 [T, H]`, registered as a custom op so the
full-graph capture takes it, replacing `MoERunner.forward` on armed layers. Five launches per layer,
each exact by the same rules as the dense stack, against stock's three (gate GEMM, fused MoE kernel,
combine):
1. **router_k**: the declared int8 gate (the dense `lin_*` route: `rowquant` + `_int_mm` + epilogue) then
   one kernel for the scores on the 2^26 grid, the table lookup, top-k by integer key with the
   lower-index tie rule, the k weights as one binary32 division each - and the dispatch metadata: the
   per-expert counts (an integer histogram), their exclusive prefix (expert offsets), and the
   permutation of (token, slot) pairs sorted by (expert, token). Deterministic by construction: the
   sort key is unique.
2. **gather_quant_k**: the permuted rows quantised per row (the same `rowquant`) into the dispatched int8
   operand [T * k, H] with their row scales - the row quant is per token, so it is computed once per
   token and copied k times.
3. **grouped GEMM 1** (gate_up): per expert, rows [off[e], off[e+1]) x `w13_8[e]` [H, 2I] -> int32, then the
   declared epilogue to bf16 and the SiLU table and the per-row quant into the int8 operand for the
   down GEMM. Correctness route: `torch._int_mm` per expert with the dense epilogue kernels on the
   expert's row range (exact, 2 x 128 launches per layer on the 30B-A3B). Speed route (M1b): a CUTLASS
   3.x grouped int8 GEMM on sm90 (`GemmUniversal` with `GroupProblemShape`, int8 x int8 -> int32), the
   epilogue outside it as today - vLLM's `cutlass_moe_mm` is FP8-only, so this kernel is ours.
4. **grouped GEMM 2** (down): per expert -> int32 -> the declared epilogue -> bf16 rows [T * k, H] in
   dispatched order.
5. **combine_k**: for each token, its k rows gathered back in ASCENDING expert index, `bf16( fl32(w) *
   fl32(row) )` each, summed in binary32 in that order, the shared expert added last; bf16 out. One
   thread block per token row; the order is fixed by the sort, so any arrival order of the GEMM tiles
   gives the same sum.

Gates: the kernel gate extends to the five launches against `moe_ref` on random and tie-dense data
(ties in the scores are the router's hard case); the engine gates E1 / E4v2 (captured rows bit-identical
to the reference), E6 (full graphs == eager), the fingerprint, and the 300-window perplexity against
stock. Expert parallelism (M3) inserts a pinned all-to-all between 2 and 3 and before 5 - the
permutation is already by expert, so the exchange is a contiguous-range send.
- 2026-09-10, the perplexity gate's first attempt OOMed on both arms: stock because `prompt_logprobs`
  materialises a [2,048, 152k] logits block per window beside an engine at 0.95 of the card; the armed
  engine because the int8 expert copies (29 GB on the 30B-A3B) sat beside the bf16 expert weights (58
  GB) they replace. Fixed: 0.80 of the card, one sequence at a time, and the bf16 experts freed once
  the contract path owns the layer. Re-run queued. Note for the manifest design: the armed engine
  must never hold both - `free_bf16` is already the dense stack's rule.
- 2026-09-10, M0's QUALITY GATE, first reading (`results/s3/m0/ppl_*.json`, 30 windows of 2,048 tokens of
  wikitext-2, 61,410 tokens): stock Qwen3-30B-A3B perplexity 8.8036; the engine with all 48 expert
  blocks on the contract's reference (the declared int8 gate, int8 experts, no rotations, the dense
  blocks stock) 8.8665 - +0.71 percent. The path works end to end on a real model (1,440 MoE calls, 97 s
  for the eager per-expert reference against stock's 2.3 s). The cost is above the dense stack's band
  (+0.05 to +0.34 percent) and has two candidate sources: the declared router (the 9 / 26 percent of
  tokens whose expert set differs from stock's) and the experts' int8 path without the v2.5 rotations
  and outlier channels. Attribution run queued: stock's routing (bf16 gate, `ops.topk_softmax`) with the
  contract's experts, so the router's share is the difference.
- 2026-09-10, the quality cost attributed (`results/s3/m0/ppl_arm2.json`): stock's own routing with the
  contract's experts reads 8.8719 (+0.78 percent) against the fully declared path's 8.8665 (+0.71) and
  stock's 8.8036 - so the declared int8 router costs NOTHING (it reads slightly better than stock's bf16
  routing on these windows) and the whole 0.7 percent is the experts' int8 path without the v2.5
  tools. Next (queued, ARM=4): the dense stack's R1 on the block input and R4 on the SiLU output, exact
  online (`fwht_blocks`, the 256-block Sylvester Hadamard over 16), with w13 / w2 folded by the same
  block rotation on their input columns at arm time (`_block_rot_cols_`, fp64, one bf16 rounding).
  The outlier channels follow if needed (they need the MoE input's calibration statistics, which
  `lockstep_calibrate` does not yet collect).
- 2026-09-10, M0's QUALITY GATE PASSES with the rotations (`results/s3/m0/ppl_arm4.json`): the contract's
  expert path with R1 on the block input and R4 before the down linear reads perplexity 8.8140 against
  stock's 8.8036 - +0.12 percent on 61,410 tokens, inside the dense stack's band (+0.05 to +0.34), from
  +0.71 without the rotations. The declared int8 router is free (the attribution above). So contract
  v3's expert rule is the dense v2.5 rule per expert - the row quant under the 256-block rotation,
  the weights folded - and 7.27's draft drops its "no rotation" caveat. The rotated reference's
  determinism across devices is being re-checked (queued); the outlier channels are held in reserve.
- 2026-09-10, M1 begins: `t16moe::route` written - the contract router as one warp per token (the
  2^26-grid scores through the table, k rounds of warp argmax with the lower index winning ties, one
  binary32 division per weight, the dispatch histogram by atomics, which is order-free for a count).
  Its constant fl32(2^26 / ln 2) = 96,817,624 matches the reference's to the bit. Its gate against
  `moe_ref` on random and tie-dense logits (60 cases) is queued behind the rotated reference's
  determinism check, whose first layer already reads GPU == CPU at every stage.
- 2026-09-10, the rotated reference is deterministic across devices too (`results/s3/m0/moe_m0_rot_*`):
  GPU == CPU at every stage on both layers. M0 is complete: the definition (router + rotated experts)
  replays anywhere and costs +0.12 percent perplexity. M1's router gate did not run on its first try -
  the extension registered its torch library without the module entry point the JIT loader needs -
  fixed; the gate and the fused-route check (`moe_fused`: the block out of the dense stack's gated
  kernels plus the router, per-expert `_int_mm`, the combine in declared order) are queued.
- 2026-09-10, M1b's kernel route: the pod's FA3 checkouts carry CUTLASS 4 with the classic grouped GEMM
  (`gemm/kernel/gemm_grouped.h`, `device::GemmGrouped`): int8 x int8 -> int32 with per-problem shapes and
  pointers, which is exactly the exact expert GEMM the contract needs (the epilogue stays ours, outside
  the kernel, as for the dense linears). The sm90 wgmma form (`GroupProblemShape`) is the speed
  follow-up if the classic kernel's int8 tensor-core path (sm80 mma.sync) is not enough. `t16moe::combine`
  is written and wired into the fused route (the torch loop kept as the reference option); its gate
  runs with the router's.
- 2026-09-10, M1G3 MEASURED (H100): `t16moe::route` gate 60 / 60 (random + tie-dense rows against the torch
  reference of SPEC 7.27's router), `t16moe::combine` gate 20 / 20 (ascending-expert bf16 products summed
  in binary32). The fused route's first check died on a harness bug (a wrong loader name); re-queued as M1G2b.
- 2026-09-10, M1G2b MEASURED (H100, Qwen3-30B-A3B, 915 real tokens of the layer's MoE input): the fused route
  (`moe_fused.MoEFused`: rowquant + int8 gate GEMM + declared epilogue, `t16moe::route`, R1 + row quant,
  per-expert `torch._int_mm` + `deq_silu_r4quant` + `_int_mm` + `deq_rows`, `t16moe::combine`) is
  BIT-IDENTICAL to `moe_ref.moe_forward(rot=True)` at layers 0 and 24: routing sets and order equal, weights
  equal, output equal (0 rows differ). 0.244 s reference vs 0.0149 s fused (the fused one still loops over
  128 experts: that loop is what M1b removes).
- 2026-09-10, M1b BUILT: `t16moe_gmm.cu` - the exact grouped int8 GEMM on CUTLASS 4.3's classic grouped GEMM
  (sm80 int8 mma.sync, kDeviceOnly scheduling, the per-problem table filled on device from the expert
  offsets: no host sync, graph-capturable), two tile configurations (128x128 for prompts, 64x128 for decode);
  grouped forms of the two expert epilogues (`deq_rows_g`, `deq_silu_r4quant_g`: the SAME kernels with a
  per-row expert index selecting the ws block, nullptr in the dense launches); `moe_fused` grouped path
  (stacked [E, N, K] weights, dispatched rows, 2 GEMM launches + 2 epilogue launches per layer instead of
  4 per active expert). Gates queued: `moe_m1b_gate.py` (vs per-expert `_int_mm`, empty / one-row / dense
  expert counts, both tiles), the dense kernel gate (the signature change must stay 1,186 / 1,186), and the
  fused check with `GROUPED=1` at layers 0 and 24.
- NOTE for M2 (GPT-OSS-120B): its expert intermediate is 2,880 = 11.25 x 256, so the v2.5 R4 rotation's
  256-blocks do not tile it; contract v3 needs either a 64-block (2,880 = 45 x 64) R4 for that family or
  no rotation before the down projection (a perplexity decision, M0-style, before the kernel work). GPT-OSS
  also has a clamped SwiGLU (alpha 1.702, limit 7) with biases: a declared-function addition to SPEC 7.27.
- 2026-09-10, M1b gate, first run (H100): `t16moe_gmm` 42 / 42 cases exact against per-expert `torch._int_mm`
  (E = 8 / 32 / 128, the Qwen3-30B-A3B shapes K 2,048 -> N 1,536 and K 768 -> N 2,048, both tile
  configurations, decode-like 0-3 rows per expert, prompt-like up to 300 with an empty expert, and the
  almost-all-empty distribution); the harness then crashed in the REFERENCE on the last, odd shape
  (N 256, K 64: cuBLASLt refuses it). Reference switched to an exact int64 matmul; full rerun queued (M1B2).
- 2026-09-10, M1b MEASURED (H100): the dense kernel gate after the epilogue signature change (`rowexp` on
  `deq_k` / `deqsilur4_k`, nullptr in the dense launches): T16M kernels PASS 1,186 / 1,186, 0 bad. The
  GROUPED fused route (2 grouped GEMM launches + 2 grouped epilogue launches instead of the 128-expert loop)
  on 915 real tokens at layer 0: routing sets / order / weights / output EQUAL to the reference, 0 rows
  differ; 2.35 ms against the loop's 14.9 ms and the torch reference's 250 ms. Layer 24 and the engine
  perplexity / speed (M1e) follow.
- 2026-09-10, M2 sizing note: GPT-OSS-120B ships MXFP4 experts (~65 GB) and stock vLLM serves it on ONE
  H100 80 GB with on-the-fly dequant; the contract's int8 experts for the same model are ~117 GB, so an
  exact int8 GPT-OSS-120B needs TP=2 (a rented 2xH100) or an int4 expert weight rule (a contract decision:
  the MXFP4 weights ARE the published model, so a contract that consumes the MXFP4 values exactly - int4
  mantissa times a power-of-two block scale into the int8 GEMM's operand, exact by construction - keeps both
  the footprint and the model identity; to be assessed M0-style). GPT-OSS-20B (int8 ~21 GB) fits the
  single H100 and is the M2 model on lc-handover; 120B follows on the rented pair.
- 2026-09-10, M1b layer 24 (H100): GROUPED fused route EQUAL to the reference on 915 real tokens (sets,
  order, weights, output; 0 rows differ), 1.56 ms vs the reference's 258 ms. M1 kernel work is complete
  and gated; M1e (engine perplexity + speed) running.
- 2026-09-10, M1e first run (H100, Qwen3-30B-A3B): the armed runs died before the first token - the engine
  harness never loaded the `t16moe` extension (the check harness loaded it itself); `moe_fused` now loads it
  for every consumer. The STOCK rows did run: CUDA graphs decode 210.7 / 889.5 / 2,231 tok/s at batch
  1 / 8 / 32 (eager stock: 33.4 / 255 / 1,005 - eager is not a comparison point for decode); the TTFT rows of
  that run are INVALID (prefix caching served the timed repeat: 13 ms at 8k) and are re-measured with prefix
  caching off. Re-queued as M1E2 with the grouped-GEMM gate rerun (fp64 reference) and the per-stage profile.
- 2026-09-10, M1E2 MEASURED (H100, Qwen3-30B-A3B): grouped GEMM gate rerun 48 / 48 exact (fp64 reference).
  ENGINE PERPLEXITY with the fused grouped path in all 48 MoE layers (1,440 calls, 30 x 2,048 wikitext-2
  tokens): 8.81400582424298 - IDENTICAL to the last digit to the torch reference arming (ARM=4, M0:
  8.81400582424298). The kernel stack IS the declared contract end to end. Stock graphs reference rows,
  prefix caching off: decode 203.7 / 889.9 / 2,222.8 tok/s at batch 1 / 8 / 32; TTFT 44.9 ms at 2k,
  168.8 ms at 8k. The fused path under CUDA graphs crashed on its first bench step (illegal memory access,
  during a step, not capture); the eager bench and the profile follow in the chain, and a
  CUDA_LAUNCH_BLOCKING reproduction is queued (M1E3) to name the op.
- 2026-09-10, M1E2 eager rows (H100, Qwen3-30B-A3B, prefix caching off, max_num_seqs 32): the fused grouped
  path runs CLEAN in eager mode inside the engine over the whole bench (221,760 MoE calls): decode 30.0 /
  238.7 / 928.5 tok/s at batch 1 / 8 / 32 against stock EAGER 33.4 / 255.3 / 1,005 (0.90 / 0.93 / 0.92x -
  both launch-bound in eager, not the comparison); TTFT 106.5 ms at 2k, 377.6 ms at 8k against stock
  GRAPHS 44.9 / 168.8 ms. The crash is therefore CUDA-graph specific: a standalone capture + replay check
  of the fused block at decode batch sizes with padded-row variants (NaN / inf / zero rows) is queued (M1G4)
  beside the launch-blocking engine run (M1E3).
- 2026-09-10, M1G4 (H100): standalone CUDA-graph capture + replay of the fused grouped block at T = 1 / 2 / 8 /
  32 / 64 / 256 / 512 with normal, zero, NaN-row, inf-row and 1e4-scale inputs: 35 / 35 replays equal the
  eager forward bit for bit, no fault. The launch-blocking engine run places the fault in
  `torch.cuda.CUDAGraph.replay` of vLLM's graph runner (capture succeeded). So the fault is specific to the
  engine's graph runner (shared pool across capture sizes, the padded decode batch, or the compiled graph's
  buffer planning around the opaque op), not to the kernels. Discriminators queued (M1E4): PIECEWISE graphs
  and FULL graphs with a single capture size.
- 2026-09-10, M1P2 MEASURED (H100, eager, CUDA events per stage, medians of 20; E 128, H 2,048, I 768, k 8):

  | rows | total ms | gate | route | r4quant | dispatch (torch) | gmm up | deq_silu_r4quant_g | gmm down | deq_rows_g | combine (+inv) |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 1 | 0.379 | 0.083 | 0.011 | 0.009 | 0.134 | 0.029 | 0.014 | 0.015 | 0.006 | 0.077 |
  | 8 | 0.411 | 0.085 | 0.012 | 0.010 | 0.120 | 0.065 | 0.019 | 0.037 | 0.007 | 0.056 |
  | 32 | 0.472 | 0.086 | 0.012 | 0.007 | 0.120 | 0.098 | 0.027 | 0.054 | 0.008 | 0.060 |
  | 256 | 0.642 | 0.103 | 0.012 | 0.009 | 0.101 | 0.126 | 0.130 | 0.075 | 0.022 | 0.064 |
  | 2,048 | 1.842 | 0.105 | 0.017 | 0.013 | 0.154 | 0.221 | 0.930 | 0.144 | 0.143 | 0.116 |

  Reading. Decode (1-32 rows) is LAUNCH-BOUND: ~20 launches per MoE layer, of which the torch dispatch
  (argsort, two gathers, cumsum, casts: ~14 tiny ops) is 0.12-0.13 ms and the gate's three launches
  0.08 ms; the kernels themselves are a few microseconds each. Stock's graph decode step is 4.9 ms for
  48 layers (~0.10 ms per layer, attention included), so the MoE block must come to ~0.05 ms per layer
  under graphs: the launch COUNT is the lever (graphs remove the CPU gap, not the per-kernel floor of
  ~3-5 us). Prefill (2,048 rows = 16,384 dispatched rows) is one kernel: `deq_silu_r4quant_g` 0.93 ms of
  1.84 - the grouped wrapper forces the per-row CTA form where the dense path has the tiled / cluster
  forms from 32 rows. PLAN (M1f): (1) `t16moe::dispatch` - one deterministic counting-sort launch
  producing offs / order / rowexp / inv AND gathering the a8 / asc rows (replaces ~14 ops); (2) the gate
  fused into the route for decode rows (int8 GEMV + epilogue + table scores + top-k in one launch);
  (3) the combine's `inv` produced by dispatch (removes 3 ops); (4) the grouped cluster / tiled form of
  deq_silu_r4quant for prefill. Expected launches per layer: ~20 -> 7. Measured, not assumed, once the
  graph fault is resolved.
- 2026-09-10, M1E4 (H100): the graph fault reproduces with PIECEWISE graphs (attention outside) and with FULL
  graphs restricted to ONE capture size (32): not the shared pool across sizes, not the full-graph attention.
  The standalone block in the same T range replays clean (M1G4). Next discriminators (M1E5): the MoE op as a
  splitting op (runs eagerly between graph pieces) and the pipeline cut after each stage under graphs.
- 2026-09-10, M1G5 (H100): the standalone block in vLLM's graph pattern - 19 capture sizes (1 to 512) captured
  into ONE memory pool, 76 interleaved replays with fresh inputs: 76 / 76 equal to eager, no fault. The
  block's own graph behaviour is clean; the fault needs the engine's context (bisect M1E5 running).
- 2026-09-10, M1F MEASURED (H100): with `t16moe::dispatch` and the multi-row grouped SiLU-R4 kernel the fused
  grouped route is EQUAL to the reference at layers 0 and 24 (915 real tokens, 0 rows differ), the dense
  kernel gate stays 1,186 / 1,186, route 60 / 60, combine 20 / 20 (the dispatch and multi-row gates were
  skipped by an early exit in the gate script; re-queued as M1F2). Per-stage profile (eager, ms per layer):

  | rows | total | gate | route | r4quant | dispatch | gmm up | SiLU-R4 | gmm down | deq_rows_g | combine |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 1 | 0.206 (was 0.379) | 0.076 | 0.011 | 0.009 | 0.008 | 0.023 | 0.012 | 0.016 | 0.006 | 0.046 |
  | 32 | 0.331 (0.472) | 0.079 | 0.012 | 0.007 | 0.010 | 0.094 | 0.014 | 0.054 | 0.008 | 0.052 |
  | 256 | 0.444 (0.642) | 0.097 | 0.012 | 0.009 | 0.032 | 0.122 | 0.024 | 0.075 | 0.022 | 0.053 |
  | 2,048 | 1.045 (1.842) | 0.098 | 0.016 | 0.013 | 0.194 | 0.224 | 0.109 | 0.143 | 0.143 | 0.105 |

  Dispatch 0.134 -> 0.008 ms at decode; SiLU-R4 at 2,048 rows 0.930 -> 0.109 ms. Next by size: the combine
  at 46 us for ONE token is a latency-bound loop (one warp per token, 64 dependent iterations of 8 loads) -
  rewritten block-per-token with the k loads issued together (M1F2); the gate's three launches (76 us eager,
  cuBLASLt host cost) - to be fused into the route for decode rows; the grouped GEMM at 1-8 rows (23 + 16 us)
  - a GEMV form for tiny row counts; the dispatch's serial row copy at 2,048 rows (0.194 ms) - parallel rows.
- 2026-09-10, M1E5 (H100): under the engine's graphs EVERY cut faults, including the cut after stage 1 (gate
  linear + route, returning x): the fault is in the gate path - rowquant_pad, torch._int_mm (cuBLASLt),
  deq_rows or route - not in the expert path. Prime suspect: the cuBLASLt int8 GEMM inside vLLM's graph
  capture (workspace semantics); control runs queued (M1E7): the gate stage bisected op by op, the cuBLAS call
  replaced by an exact fp64 matmul, and by the grouped int8 GEMM as a single expert (graph-safe by
  construction; also the natural replacement if cuBLASLt is the cause).
- 2026-09-10, M1F2 MEASURED (H100): all five kernel gates pass - route 60 / 60, combine (block-per-token form)
  20 / 20, dispatch 18 / 18, multi-row SiLU-R4 60 / 60 (against the per-row form at nb 3 / 4 / 8 / 16, M 1 to
  4,096, three magnitudes), gate_route 42 / 42 (against rowquant_pad + _int_mm + deq_rows + route at H 2,048 /
  4,096, T 1 to 64). Fused route at layer 0 EQUAL to the reference (915 tokens). Profile (eager, ms per
  layer): T=1 0.142 (gate+route 0.055, combine 0.011 - was 0.046), T=8 0.207, T=32 0.259, T=256 0.405,
  T=2,048 1.004. The fused gate kernel itself is slow at 55 us (2 threads per expert over H = 2,048: a
  512-step dp4a chain) - widened to 8 threads per expert (1,024-thread block). The dispatch row copy at
  2,048 rows (0.193 ms) parallelised (8 rows per pass, warp per row).
- 2026-09-10, M1E6 (H100): the MoE op as a SPLITTING op (run eagerly between graph pieces) also faults - as
  it must: a piece's captured input address is our op's output at capture time, and an eagerly run op that
  allocates its output each call cannot provide a static address (vLLM's own splitting ops write into
  pre-allocated buffers). Not a discriminator for the in-graph fault; the in-stage bisect (M1E7) is.
- 2026-09-10, M1F3 MEASURED (H100): the fused gate + route path (T <= 64) is EQUAL to the reference on 48 real
  tokens at layer 0 (routing sets / order / weights / output), and the 915-token check stays equal; all five
  gates pass again (200 cases). Dispatch with the parallel row copy: 0.193 -> 0.090 ms at 2,048 rows; the
  block at 2,048 rows 0.900 ms (was 1.842 this morning), at 1 row 0.143 ms (was 0.379).
- 2026-09-10, M1E7 (H100) - ROOT CAUSE of the graph fault, by elimination: the cut after the row quant ALONE
  (a dense-stack kernel that has run under graphs for months) faults too, so no MoE kernel is at fault. With
  CUDA graphs on, vLLM captures during engine construction; the harness armed the layers AFTER `LLM()`
  returned - so the captured graphs hold the STOCK fused-MoE kernels reading the bf16 expert weights that the
  arming then freed: every replay reads freed memory. Eager mode has no capture, hence clean; the
  standalone captures were armed before capture, hence clean. Fix: arm inside the worker's model load
  (`GPUModelRunner.load_model` wrapped; the dense stack's plugin arms at the same point), before
  torch.compile and capture (M1E9 queued: graph-mode bench + perplexity).
- 2026-09-10, M1E8 (H100): the cuBLAS-free fused path (gate GEMM on the grouped int8 kernel as one expert at
  prefill rows; gate_route below 65 rows) keeps the engine perplexity at 8.81400582424298 - identical again.
  The graph bench faulted as predicted (this run still armed after construction). Stock graph reference
  reproduced to 0.1%: decode 203.9 / 889.5 / 2,222.8 tok/s, TTFT 44.8 / 168.9 ms at 2k / 8k.
- 2026-09-10, M1F4 (H100): the 1,024-thread gate_route passes its gate (42 / 42) but is NOT faster in the eager
  profile (0.061 vs 0.055 ms): at this size the eager per-stage numbers are HOST time between launches, not
  kernel time. A graph-replay profile (the block and its stage prefixes captured and replayed) is queued
  (M1P3) to read the true GPU floor per layer; the engine graph bench (M1E9) is the number that matters.
- 2026-09-10, M1E9 MEASURED (H100, Qwen3-30B-A3B, CUDA graphs, prefix caching off, medians of 3) - THE FUSED
  CONTRACT-v3 MoE PATH RUNS CLEAN UNDER vLLM's CUDA GRAPHS once armed at model load. First graph-mode speed,
  fused / stock (stock 203.9 / 889.5 / 2,222.8 tok/s; TTFT 44.8 / 168.9 ms):

  | row | fused | stock | ratio |
  |---|---|---|---|
  | decode batch 1 | 126.0 tok/s | 203.9 | 0.62x |
  | decode batch 8 | 780.6 | 889.5 | 0.88x |
  | decode batch 32 | 2,360.7 | 2,222.8 | 1.06x |
  | TTFT 2k | 60.6 ms | 44.8 | 0.74x |
  | TTFT 8k | 231.3 ms | 168.9 | 0.73x |

  Reading: at batch 32 the exact int8 grouped path is already FASTER than stock's fused bf16 MoE kernels;
  at batch 1 the deficit is 3.0 ms per step over 48 layers = ~63 us per layer of fixed cost (9 launches,
  the grouped GEMM's scheduler at 8 one-row problems, the 2 KB-per-row dispatch) - the graph-replay
  profile (M1P3) apportions it. Prefill at 0.73x: the gate GEMM on the grouped kernel (0.10 ms), the
  dispatch copy (0.09), deq_rows_g (0.14) and the two grouped GEMMs (0.37) at 2,048 rows - the GEMMs are
  the sm80 mma.sync path; the sm90 wgmma grouped kernel is the lever there.
  Perplexity with the arming at load: 8.81400582424298 (identical, 61,410 tokens) - the fourth identical
  reading of the fused path against the reference arming.
- 2026-09-10, M1P3 MEASURED (H100, CUDA-graph replay, GPU time per layer): the fused block's floor is 0.111 ms
  at 1 row, 0.175 at 8, 0.225 at 32, 0.289 at 256, 0.805 at 2,048. Stage prefixes (the 3-launch gate path):
  gate + route 0.073 ms at 1 row (cuBLASLt's int8 GEMM at M = 17 is most of it), + r4quant + dispatch 0.011,
  + grouped GEMM up + SiLU-R4 0.026, + grouped GEMM down + dequant 0.015, combine ~0.010. The fused
  gate_route kernel, by difference, ~0.049 ms at 1 row: its weight reads were STRIDED (each of the 8 threads of
  an expert walking its own 256-byte segment: a 256 KB read as ~64k L2 sectors from one block) - changed to
  interleaved lanes (32 contiguous bytes per expert per step). Budget: stock's whole layer is ~0.102 ms at 1
  row and our block adds 0.063, so stock's MoE block is ~0.048 ms; parity needs ours at ~0.05: the gate fix
  (~-0.04) and GEMV forms of the grouped GEMM at <= 64 dispatched rows (~-0.02; the CUTLASS grouped kernel
  schedules 132+ CTAs over 8 one-row problems). At 2,048 rows: gate on the grouped kernel 0.115 (N = 128:
  one wave of 128x128 tiles), dispatch 0.10, GEMM up + SiLU 0.32, GEMM down + dequant 0.28.
- 2026-09-10, M1F5 MEASURED (H100): vectorised grouped dequant (8 columns per thread) 15 / 15 against the scalar
  form, all other gates pass (220 cases), fused route EQUAL to the reference at T = 48 and 915. At 2,048 rows:
  deq_rows_g 0.143 -> 0.071 ms, combine 0.066 -> 0.039 (eager); the block under graph replay 0.805 -> 0.705 ms
  (this run still had the strided gate_route; its coalesced form follows in M1F6, the GEMV grouped GEMM in
  M1F7, the engine re-measurement in M1E10). Two more prefill changes shipped for M1F7: the dispatch scan at
  1,024 threads (16k pairs in 16 tiles instead of 64) and the gate GEMM on the 64x128 tile (N = 128 gave one
  wave of 128 CTAs on the 128x128 tile).
- 2026-09-10, M1F6 MEASURED (H100, graph replay, GPU ms per layer): the coalesced gate_route is 0.019 ms at 1 row
  (the strided form ~0.049) and the block's floor at 1 row falls 0.107 -> 0.066 ms; 8 rows 0.151, 32 rows
  0.184, 256 rows 0.274, 2,048 rows 0.702. All gates pass (220 cases), the fused route EQUAL to the reference
  at T = 48. Stage costs at 1 row now: gate_route 0.019, r4quant + dispatch 0.011, GEMM up + SiLU-R4 0.022,
  GEMM down + dequant 0.014, combine ~0.010 (the GEMV form of the two GEMMs follows in M1F7). The fused gate
  beats the GEMM gate path at every measured T below 2,048 (0.028 vs 0.097 ms at 256 rows; 0.141 vs 0.115
  at 2,048): its use extended from T <= 64 to T <= 512 (gate cases at 300 and 512 rows added).
- 2026-09-10, M1F7 MEASURED (H100): the GEMV grouped-GEMM form is exact (gmm gate 64 / 64 over the three
  forms), all other gates pass, fused route EQUAL at T = 48 - but at 8 dispatched rows it is NO faster than
  the CUTLASS kernel (GEMM up + SiLU-R4 prefix 0.022 ms both ways; the block at 1 row 0.068 ms): 96 blocks of
  1,024 threads on 132 SMs, one block per SM, latency-bound. Re-tiled to 64 columns x 8 lanes (512 threads,
  twice the blocks). The 1,024-thread dispatch scan took the 2,048-row block from 0.702 to 0.654 ms
  (r4quant + dispatch prefix 0.218 -> 0.169). Floor arithmetic at 1 row: the layer's expert weights are
  59 MB int8 = 18 us at 3.3 TB/s (stock reads 118 MB of bf16); our two GEMM stages take ~37 us.
- 2026-09-10, M1E10 MEASURED (H100, Qwen3-30B-A3B, CUDA graphs, prefix caching off, medians of 3) after the M1f
  round (coalesced gate_route, GEMV grouped GEMM at few rows, vectorised dequant / combine, 1,024-thread
  dispatch, gate GEMM on the 64x128 tile), fused / stock (stock 203.9 / 889.5 / 2,222.8 tok/s; 44.8 / 168.9 ms):

  | row | M1E9 (morning) | M1E10 | ratio now |
  |---|---|---|---|
  | decode batch 1 | 126.0 tok/s | 165.0 | 0.81x (was 0.62) |
  | decode batch 8 | 780.6 | 833.1 | 0.94x (was 0.88) |
  | decode batch 32 | 2,360.7 | 2,784.6 | 1.25x (was 1.06) |
  | TTFT 2k | 60.6 ms | 51.2 | 0.88x (was 0.74) |
  | TTFT 8k | 231.3 ms | 189.7 | 0.89x (was 0.73) |

  Perplexity 8.81400582424298 - identical, fifth reading. Remaining at batch 1: ~1.2 ms per step over 48
  layers = ~25 us per layer over stock's MoE block; the GEMM stages are at ~2x the weight-read floor (59 MB
  int8 per layer = 18 us). Next (M1F8 / M1E11): gate_route up to 512 rows, the re-tiled GEMV.
- 2026-09-10, M1F8 MEASURED (H100): gate_route 54 / 54 with the 300- and 512-row cases, gmm 64 / 64, all other
  gates pass; the fused route EQUAL to the reference on 300 real tokens (the fused gate path). Graph floor per
  layer: 1 row 0.069 ms, 8 rows 0.143 (was 0.151), 32 rows 0.185, 256 rows 0.260 (was 0.267), 512 rows 0.330,
  2,048 rows 0.657. The 64-column GEMV gave nothing at 1-8 rows (GEMM up + SiLU prefix 21.5 us both ways);
  re-tiled again to 32 columns x 8 lanes (256 threads, 384-512 blocks at 8 rows) for M1F10.
- 2026-09-10, M1E11 MEASURED (H100, graphs, medians of 3) with gate_route up to 512 rows and the 64-column
  GEMV: decode 170.0 / 873.7 / 2,786.7 tok/s = 0.83 / 0.98 / 1.25x of stock (M1E10: 0.81 / 0.94 / 1.25);
  TTFT 51.1 / 189.9 ms = 0.88 / 0.89x. Perplexity 8.81400582424298, sixth identical reading.
- 2026-09-10, M1F9 MEASURED (H100): `gmm_deq` (the GEMV with the contract dequant fused, down projection at few
  rows) 8 / 8 against gmm + deq_rows_g; all gates pass (235 cases); fused route EQUAL at T = 48. Graph floor
  per layer: 1 row 0.066 ms, 8 rows 0.138, 32 rows 0.188, 256 rows 0.260, 2,048 rows 0.655. ENGINE (graphs,
  medians of 3): decode 170.2 / 895.4 / 2,785.6 tok/s = 0.83 / 1.01 / 1.25x of stock - batch 8 at PARITY;
  TTFT 51.6 / 192.4 ms = 0.87 / 0.88x. Perplexity 8.81400582424298, seventh identical reading.
  Remaining at batch 1 (~18 us per layer over stock's MoE block): the up GEMV + SiLU-R4 stage (21.5 us for a
  24 MB weight read whose floor is 7 us), gate_route 19 us (its 256 KB weight read is L2-resident; the
  kernel is one block per token, latency-bound), r4quant + dispatch 11 us (two launches; fold the R1 quant into
  gate_route and let the GEMV gather rows by token index instead of copying them). Prefill (0.87x): the two
  grouped GEMMs are 0.52 of the 0.655 ms at 2,048 rows on the sm80 mma.sync path - the sm90 wgmma grouped
  kernel (CUTLASS 3 GroupProblemShape, int8) is the lever, a build item of its own.
- 2026-09-10, M1F10 MEASURED (H100): the 32-column GEMV (384-512 blocks at 8 rows) is exact (64 / 64) and, like
  the 64- and 128-column forms, leaves the 1-row block at 0.066 ms: the GEMM-up + SiLU-R4 stage stays ~21 us
  whatever the tiling, so it is not block-count-limited. Per-stage floor at 1 row now: gate_route 19, r4quant +
  dispatch 11, GEMV up ~12 + SiLU-R4 ~9, GEMV down with fused dequant ~10, combine ~4 - five launches each a
  few microseconds over its memory floor. The batch-1 ratio (0.83x) is the sum of those; batch 8 is at parity
  and batch 32 at 1.25x. DECISION: after M1F11 (the folded row quant, the last cheap launch saved) the
  batch-1 tuning stops here and the work moves to (a) the sm90 wgmma grouped GEMM for prefill (the 0.52 ms
  of the 0.655 ms block at 2,048 rows) and (b) M2, GPT-OSS-20B. Remaining batch-1 levers, filed: the SiLU-R4
  stage at few rows (its 32 KB table load per CTA, one CTA for 8 rows), fusing the SiLU into the up GEMV.

## M2: GPT-OSS-20B (started 2026-09-10)
Facts from vLLM 0.25.1's `gpt_oss.py`: router = Linear with bias; `FusedMoE(renormalize=True, has_bias=True,
activation="swigluoai")` - the same softmax-then-top-k-then-renormalise routing as Qwen3 (contract v3 router
applies; the bias joins the gate epilogue); experts with biases; the activation
`(clamp(up, -7, 7) + 1) * g * sigmoid(1.702 g)` with `g = clamp(gate, max 7)`, gate / up columns INTERLEAVED;
attention with sinks and a sliding window on even layers; weights MXFP4. Contract addendum drafted (SPEC 7.27).
Steps:
- M2-0 reference: `moe_ref` generalised - gate bias, expert biases, the clamped SwiGLU with the sigmoid table on
  `bfr(1.702 g)`, de-interleaved fold, R4 as 64-block / 256-block / none; GPU == CPU bit-identity on captured
  real tokens (the M0 harness on GPT-OSS-20B).
- M2-M0 quality: perplexity on GPT-OSS-20B - stock MXFP4 vs the contract (int8 from dequantised MXFP4) with
  R4-64 / no R4; the MXFP4-consuming int8 operand as the alternative. Decide the R4 rule and the weight rule.
- M2-1 kernels: `gate_route` with bias; the epilogue kernel variant `deq_swigluoai_r4quant_g` (clamps, the
  alpha-scaled sigmoid table, the +1, biases; R4-64 if chosen); the down epilogue with bias (`deq_rows_g` /
  `gmm_deq` with bias). Gates against the reference as for Qwen3.
- M2-2 engine: arming at load for the GPT-OSS runner (the `MLPBlock` slices `[:, :hidden_size]`); perplexity
  identical to the reference arming; graph-mode speed vs stock MXFP4 (a different baseline class: stock's MXFP4
  kernels read 4-bit weights, ours 8-bit - at batch 1 the weight read is the floor, so expect below parity
  there unless the MXFP4-consuming operand is adopted).
- M2-3 attention: sinks + sliding window under the LSSA-B8 rule (the dense stack's T16 sink gate), E1 / E4 /
  E6 on GPT-OSS-20B. Then 120B on a rented pair (TP=2).
- 2026-09-10, M1F11 MEASURED (H100): gate_route with the R1 row quant folded in - 54 / 54 against
  rowquant_pad + _int_mm + deq_rows + route AND r4quant_pad (the rotated quant bit-identical), all gates pass,
  fused route EQUAL at T = 48 and 300. Graph floor per layer: 1 row 0.061 ms (from 0.066), 8 rows 0.132,
  32 rows 0.183, 256 rows 0.258, 2,048 rows 0.655. ENGINE (graphs, medians of 3): decode 177.9 / 921.1 /
  2,811.6 tok/s = 0.87 / 1.04 / 1.26x of stock; TTFT 52.1 / 192.2 ms = 0.86 / 0.88x. Perplexity
  8.81400582424298, eighth identical reading. Batch-1 tuning stops here per the M1F10 decision; the day's
  decode trajectory: 0.62 -> 0.81 -> 0.83 -> 0.83 -> 0.87x at batch 1, 0.88 -> 1.04x at batch 8,
  1.06 -> 1.26x at batch 32.
- 2026-09-10, M2 download done (GPT-OSS-20B, 3 safetensors, 13.1 GB, after freeing ~46 GB of regenerable caches
  and dumps on the pod's quota). Config: 24 layers, hidden 2,880, expert intermediate 2,880, 32 experts, top-4,
  64 heads x 64 / 8 KV heads, sliding window 128 on every other layer, yarn rope x32, MXFP4 experts only (router,
  attention, embeddings, lm_head in bf16). CONSEQUENCE: the HIDDEN size 2,880 is not a multiple of 256 either -
  the v2.5 rotations (R1 on the block input, R4 before the down projection, and the dense stack's norm / rotate
  kernels) do not tile it; the M2 contract uses the 64-block Sylvester / 8 on both sides or no rotation, decided
  by perplexity (M2-M0); the first M2 milestone arms the MoE blocks only and leaves attention and the dense
  linears stock bf16.
- 2026-09-10, M2BOOT (H100): stock vLLM 0.25.1 serves GPT-OSS-20B on the single H100 (boot 21 s, eager; MoE quant
  method `GptOssMxfp4MoEMethod`, expert biases held in fp32 [32, 5,760] / [32, 2,880] in the interleaved layout,
  router bf16 [32, 2,880] + bias); a greedy generation answers "Paris". Checkpoint layout: `gate_up_proj_blocks`
  uint8 [32, 5,760, 90, 16] (90 blocks of 32 FP4 per row, K = 2,880) + `gate_up_proj_scales` uint8 [32, 5,760, 90],
  `down_proj_blocks` [32, 2,880, 90, 16] + scales, biases bf16, router bf16 - gate / up interleaved along the
  output rows. M2C0 queued: the addendum reference at layers 0 and 12 with ROT 0 and 64 on 1,024 real tokens.
- 2026-09-10, M1G90 (H100): the Hopper TMA warp-specialized ptr-array int8 grouped GEMM (CUTLASS 3 CollectiveBuilder,
  cooperative 128x128x128, int32 accumulation, device-filled group table) compiles at the first attempt and is
  EXACT (12 / 12 against fp64, incl. empty and one-row groups). Eager timing at the 2,048-token prompt shape:
  0.97 ms vs the classic kernel's 0.25 ms (106 vs 420 TOPS) - but eager event timing includes the ptr-array
  adapter's heavier host-side setup; re-measured under CUDA-graph replay with three schedules (cooperative
  128x128, pingpong 64x128 cluster 2x1, cooperative 256x128 cluster 1x2) in M1G90B.
- 2026-09-10, M2C0 first run: all four cells died on a KeyError for the expert tensors - the harness read MODEL
  from the environment and the pod's env.sh exports MODEL=Qwen/Qwen2.5-7B-Instruct by default, so the loader
  indexed the Qwen checkpoint (the pod convention "pin MODEL on the command line" applies to every harness
  with a MODEL default; the checkpoint loader now prints its snapshot path). Relaunched with MODEL pinned,
  the perplexity job chained behind it.
- 2026-09-10, M2C0 MEASURED (H100, GPT-OSS-20B, 1,024 real wikitext tokens, layers 0 and 12): the addendum reference
  (exact MXFP4 dequant -> int8 experts, biased gate and experts, clamped SwiGLU with the alpha-scaled sigmoid
  table) replays GPU == CPU BIT FOR BIT in all four cells (ROT 0 / 64 at both layers; ~65 s on the CPU). Routing
  agreement with stock's router 95.5% (layer 0) / 97.8% (layer 12). Output distance (relative L2 of the layer's
  MoE output): the bf16 evaluation of the dequantised MXFP4 model vs stock's MXFP4 kernels 0.35% / 0.33% (the
  stock kernels' own noise floor); ours, no rotation, 4.0% / 4.6%; ours with the 64-block rotation 2.2% / 2.9%.
  The rotation halves the int8 error; both are far above the noise floor - the perplexity run (M2P0: stock,
  ROT 0, ROT 64) decides whether int8-from-MXFP4 is admissible or the MXFP4-consuming operand is required.
- 2026-09-10, M2P0 (H100): stock GPT-OSS-20B perplexity on the raw wikitext-2 windows (30 x 2,048) is 367.41 -
  a harmony-format chat model on unformatted text; the number itself is not comparable to Qwen's 8.80, the
  stock-vs-contract DELTA on identical windows is the admissibility measure (as it was for Qwen: 8.8036 ->
  8.8140). The armed cells died freeing the stock experts: the MXFP4 quant method wraps them in triton-kernels
  tensors whose dtype is not a torch dtype - they stay resident (13 GB beside 19 GB of int8 experts). M2P1
  (ROT 0 / 64) re-queued. SERVING: the MoE arming now lives in `moe_stack.py` and the lockstep vLLM plugin
  arms it under LOCKSTEP_MOE=1 at model load; `t14_bench.py` gained the `moe` stack - the first served
  comparison (ShareGPT W1, 64 concurrent, 500 prompts, stock vs moe on Qwen3-30B-A3B) is queued (M1S1).
- 2026-09-10, M1G90B MEASURED (H100, CUDA-graph replay, 2,048-token prompt shape, 16,384 dispatched rows): the
  Hopper TMA warp-specialized ptr-array int8 grouped GEMM is exact (12 / 12) in all three schedules but 4x
  SLOWER than the classic sm80-style kernel - cooperative 128x128 0.968 ms (107 TOPS), pingpong 64x128 0.966,
  cooperative 256x128 1.163, against the classic 128x128 at 0.238 ms (433 TOPS) for N 1,536 / K 2,048; the same
  ratio at N 2,048 / K 768 (0.52-0.66 vs 0.151 ms). VERDICT: the CUTLASS 3 int8 ptr-array mainloop as the
  collective builder instantiates it here is not the wgmma fast path; the classic kernel stays for prefill
  (`t16moe_gmm90.cu` kept as an exact, slower alternative). The prefill lever moves to the rest of the block
  (dispatch copy, the gate GEMM tile, the combine) and to the sm90 kernel only if a hand-built int8 wgmma
  mainloop is written - a separate, larger item.
- 2026-09-10, M2-1 BUILT (gates queued, M2K1 / M2K3): the addendum kernels - `sigmoid1_t` / `swigluoai1_t` device
  functions (the table sigmoid alone; the clamped SwiGLU with the alpha-scaled argument, bf16 roundings as the
  reference); `deqsilur4_mr_k<ACT, BLK>` with the expert bias and the 64-block rotation (cross-lane butterfly
  stages limited to 8-lane groups, scale 1/8, lane liveness per group for D = 2,880 = 11.25 x 256); bias in
  `deq_gv_k` / `gmm_deq`; `gate_route<BLK>` with the gate bias and the 64-block folded row quant; `r1quant<BLK>`
  for prefill rows (bit-identical to r4quant_pad at 256). `moe_fused` takes the family parameters (act, blk,
  biases); `moe_stack.arm` dispatches on model_type (gpt_oss -> the checkpoint loader, blk from
  LOCKSTEP_MOE_BLK). Queued: M2K3 fused-vs-reference on real tokens (layers 0 / 12, ROT 64 / 0, T 48 / 1,024),
  M2E1 engine perplexity (must equal M2P2's ROT-64 reading) and the graph bench vs stock MXFP4.
- 2026-09-10, M1S1 (H100, Qwen3-30B-A3B, `vllm serve` + `vllm bench serve`, ShareGPT W1, 64 concurrent, 500 prompts):
  the "moe" arm DID NOT ARM (no plugin line in the engine log, the bf16 expert footprint intact), so the run is a
  stock-vs-stock reproducibility datum: 10.66 vs 10.75 req/s, 2,166 vs 2,185 output tok/s, median TPOT 24.00 vs
  23.96 ms, P99 TPOT 87.8 vs 117.7 ms (the tail varies 30% run to run at saturation). In a bare process with the
  job's environment the plugin installs the load_model hook; the served engine core did not - under
  investigation (where vLLM loads general plugins for the engine-core process; how the dense plugin reaches the
  worker).
- 2026-09-10, M2K1 (H100): with the addendum kernels in the dense file the dense kernel gate stays 1,186 / 1,186 and the
  seven existing MoE gates pass (235 cases); the two new addendum gates did not run - a harness bug (a loop
  variable shadowing the reference module), fixed and re-queued (M2K4).
- 2026-09-10, M2P2 MEASURED (H100, GPT-OSS-20B, 30 x 2,048 raw wikitext tokens): stock MXFP4 367.41; the contract
  with no rotation 345.90; with the 64-block rotation 365.69 - BOTH contract variants score LOWER than stock, by 5.9%
  and 0.5%. Reading: on raw text this harmony-format chat model is far off its distribution (ppl ~370), and the
  int8 perturbation moves the number in either direction by amounts unrelated to quality - raw-wikitext perplexity
  is not an admissibility measure for GPT-OSS (for Qwen3-30B-A3B at 8.80 it was). Added an in-distribution
  measure: perplexity on ShareGPT conversations rendered with the model's harmony chat template (`moe_arm2`
  FMT=chat), stock / ROT 0 / ROT 64 queued (M2P3). The M2C0 layer-level distances (2.2-2.9% with the rotation vs
  4.0-4.6% without, against a 0.35% kernel floor) remain the direct evidence; the chat perplexity decides.
- 2026-09-10, M2K3 MEASURED (H100, GPT-OSS-20B, real tokens): the FUSED kernel path with the 64-block rotation (gate_route
  with gate bias and the 64-block folded quant at T = 48; r1quant + the grouped GEMMs + the SwiGLU-OAI epilogue with
  biases + gmm_deq / deq_rows_g with bias at T = 1,024) is EQUAL to the addendum reference at layers 0 and 12 -
  routing sets, order, weights and output, 0 rows differ, 4 / 4 cells. The no-rotation cells failed on the harness
  (blk 256 does not tile H = 2,880): a `had` switch added to gate_route / r1quant (plain row quant with the 64-block
  kernels) and `MoEFused(blk=0)`; gated and re-queued (M2K5).
- 2026-09-10, M2E1 MEASURED (H100, GPT-OSS-20B, the fused kernel path armed at load, 64-block rotation): ENGINE
  PERPLEXITY 365.68665819674897 - IDENTICAL to the torch reference arming (M2P2 ROT 64) to the last digit: the
  kernel path is the declared addendum contract end to end on this family too. Graph-mode speed, fused / stock
  MXFP4 (stock: decode 309.1 / 1,575.2 / 4,827.0 tok/s at batch 1 / 8 / 32, TTFT 35.0 / 124.8 ms):

  | row | fused | ratio |
  |---|---|---|
  | decode batch 1 | 261.8 tok/s | 0.85x |
  | decode batch 8 | 1,124.5 | 0.71x |
  | decode batch 32 | 2,043.5 | 0.42x |
  | TTFT 2k | 43.4 ms | 0.81x |
  | TTFT 8k | 149.8 ms | 0.83x |

  READING. Two effects. (1) Structural: on this family (32 experts, top-4, hidden = intermediate = 2,880) nearly
  every expert is active from batch 8 up, so a decode step reads the whole expert set - 796 MB per layer as int8,
  half that as stock's MXFP4; the int8 contract's weight-read floor is ~2x stock's from batch 8 (Qwen3-30B-A3B,
  128 experts top-8, reads a fraction of its experts and stock reads bf16, twice OUR bytes: there we are at
  1.26x). (2) A routing error of ours at batch 32: 128 dispatched rows took the GEMV path (chosen at R <= 128),
  which reads each expert's weights once PER ROW - four times over on 32 experts; the threshold is now
  min(128, 2E) rows (Qwen unchanged), re-benched (M2E2). CONSEQUENCE for the roadmap: parity on GPT-OSS at
  batch >= 8 requires the addendum's item 5 - the MXFP4-consuming operand (the 4-bit mantissas and the
  power-of-two block scales into an exact block-scaled integer GEMM), a new kernel class; the int8-from-dequant
  contract stands as the correct, deterministic, gated form and as the batch-1 / prefill path.
- M2-4 (next, after the chat perplexity and the served-arm results): the MXFP4-consuming operand as drafted in the
  addendum (item 5): the torch reference first (block sums in int32, the pinned-order fp32 block-scaled accumulate,
  the declared epilogue), M2C0-style replay + distances on real tokens (this form's error against stock should be
  the routing / activation quant alone, the weights being exact), then the kernel (integer mainloop over 32-wide
  K slices with per-block fma scaling; 4-bit weights in memory).
- 2026-09-10, M1S2 (H100, served, ShareGPT W1 64 concurrent 500 prompts): the served "moe" arm did NOT arm - no marker
  file, no logger line (11.06 req/s, 2,249 tok/s, median TPOT 23.9 ms: a third stock-class datum; the tail P99 TPOT
  30 ms this time vs 88 / 118 ms in M1S1 - the saturation tail is not reproducible run to run). In a bare process
  with the same environment the plugin installs the load hook; in the served engine core nothing of ours prints.
  Trace lines at the plugin entry, the install and the load hook (stderr) and a 20-prompt served diagnostic
  queued (M1S3). Until it arms, the serving row for the MoE stack is unmeasured; the engine harness's graph
  numbers (0.87 / 1.04 / 1.26x decode) stand as the in-engine measurement.
- 2026-09-10, M2K4 MEASURED (H100): all nine MoE kernel gates pass, 263 cases - route 60, combine 20, dispatch 18,
  multi-row SiLU-R4 60, gate_route 54, vectorised dequant 15, gmm_deq 8, the addendum epilogue forms (SwiGLU-OAI,
  biases, 64-block; deq_rows_g with bias) 8 / 8, and the addendum gate_route (bias, 64-block folded quant at
  H = 2,880), r1quant (vs r4quant_pad at 256 and vs torch at 64) and gmm_deq-with-bias 20 / 20.
- 2026-09-10, M2P3 MEASURED (H100, GPT-OSS-20B, 30 x 2,048 tokens of ShareGPT conversations rendered with the harmony
  chat template - in distribution): stock MXFP4 15.101; the contract with the 64-block rotation 15.531 (+2.85%);
  without rotation 16.539 (+9.5%). Now the measure is sensitive and monotone with the layer distances (M2C0):
  the rotation halves the cost, and the int8-from-dequant expert weights cost +2.85% perplexity on this family
  (Qwen3-30B-A3B: +0.12%). Mechanism: an MXFP4 row carries a power-of-two scale per 32 columns; per-row int8
  re-quantisation of the dequantised values crushes the low-exponent blocks under the row's maximum. DECISION:
  the addendum's item 5 - the MXFP4-consuming operand (weights exact, only the activation quant remains) - is the
  GPT-OSS contract for quality as well as for the batch-8+ speed floor; the int8 form stays as the gated interim.
  Queued: the item-5 reference's layer distances (M2C1) and its chat perplexity (M2P4, expected near stock).
- 2026-09-10, M2-4 BUILT (gate queued, M2X1): `t16moe_mx::mx_gemv` - the MXFP4-consuming block-scaled integer GEMV for
  few rows: the checkpoint's 4-bit codes re-laid BLOCK-MAJOR ([E, nb, N, 16 bytes], exponents [E, nb, N]) so a
  warp's 32 columns read 512 contiguous bytes per block; per output one thread walks the 90 blocks in order - 8
  dp4a int32 on the doubled mantissas (a 16-entry table), then `acc = RN(acc + fl32(s) * 2^(k - 128))`, the
  declared single rounding per block - and the fused epilogue bf16(bfr(acc * asc) + bias). The weights stay 4-bit
  in memory: the same bytes stock reads. Gate vs the torch reference (`int_mm_mx` + `lin_epilogue_mx`) at the
  GPT-OSS shapes with random codes / exponents. Next after the gate: the fused path on it (decode rows on the
  GEMV; prefill rows on a tensor-core form - int8 mma over 32-wide K slices with the per-block scaling in
  pinned order, the item's larger build), then the engine perplexity (must equal M2P4's reference value).
- 2026-09-10, M2K5 MEASURED (H100): the NO-ROTATION fused GPT-OSS path (had = 0 in gate_route / r1quant, the 64-block
  kernels without the butterflies) is EQUAL to the addendum reference at layers 0 and 12, T = 48 and 1,024 (4 / 4);
  all nine gates pass with the had = 0 cases (265 cases). Both rotation variants of the int8 form are therefore
  gated end to end; the contract choice between them (and item 5) is the quality measurement's (M2P3 / M2P4).
- 2026-09-10, M2E2 MEASURED (H100, GPT-OSS-20B int8 form, 64-block, graphs): with the GEMV threshold at min(128, 2E) rows
  the batch-32 row goes 2,043.5 -> 3,439.9 tok/s (0.42x -> 0.71x of stock's 4,827); batch 1 261.9 (0.85x), batch 8
  1,128.6 (0.72x); TTFT 44.0 / 150.1 ms (0.80 / 0.83x). The remaining gap from batch 8 up is the int8-vs-4-bit weight
  read (stock reads half the bytes with nearly every expert active); the MXFP4-consuming kernels (M2-4) address it.
- 2026-09-10, M2C1 MEASURED (H100, GPT-OSS-20B, 1,024 real tokens, layers 0 / 12): the MXFP4-consuming reference (exact
  weights, per-row int8 activations) replays GPU == CPU bit for bit, but its distance to stock is 3.6% / 4.1% - NO
  better than the int8 form without rotation (4.0 / 4.6%) and worse than int8 with the 64-block rotation (2.2 / 2.9%).
  READING: on this family the ACTIVATION quant dominates the error, not the weights (hidden 2,880 rows with
  outliers; a per-row int8 scale), and the rotation that tames it cannot be folded into fixed MXFP4 weights.
  The fix that keeps the weights exact is block-scaled ACTIVATIONS (item 5b): int8 per 32-wide block with a
  power-of-two exponent (ka = ceil(log2(max|x_b| / 127))), the block sum still exact in int32 and the declared
  accumulate `acc = RN(acc + fl32(s_b) * 2^(ka[m, b] + kw[n, b] - 128))` still one rounding per block; no per-row
  scale. Reference written (`moe_ref.blockquant / int_mm_mx2 / moe_forward_mx2`); distances and chat perplexity
  queued (M2C2). The int8 + 64-block rotation form (chat ppl +2.85%) stays the best gated form until then.
- 2026-09-10, M1S3 (H100, served diagnostic): the plugin DOES run in the API server and the engine core, from our copy
  (`/workspace/p2/lockstep_vllm_plugin.py`), but with LOCKSTEP_MOE unset - the bench script that sets it for the
  server had been overwritten at every job start by the stage directory's older copy (the jobs copy `stage_s2/*.py`
  first; the patched `t14_bench.py` had only been edited in place). Shipped into the stage directory; the served MoE
  arm re-queued (M1S4). The three earlier "moe" serving rows (M1S1 / M1S2) are stock-vs-stock.
- 2026-09-10, M2P4 MEASURED (H100, GPT-OSS-20B, chat-templated ShareGPT, 30 x 2,048 tokens): the MXFP4-consuming
  operand (exact weights, per-row int8 activations, no rotation) reaches perplexity 14.790 - BELOW stock's 15.101
  (-2.1%); the int8 forms were +2.85% (64-block rotation) and +9.5% (none). Reading: stock's Triton MXFP4 kernels
  carry their own bf16/fp16 rounding (the 0.35% "noise floor" of M2C0 is that), while the declared form sums each
  block exactly in int32 and accumulates in binary32 once per block; the activation quant, though 3.6-4.1% in
  layer-level L2, costs less perplexity than stock's kernel rounding. DECISION (final for M2): item 5 is the
  GPT-OSS contract - quality better than stock, the same weight bytes as stock, no rotation, bit-identical GPU ==
  CPU. Item 5b (block-scaled activations) is measured next (M2C2) as a refinement, not a requirement.
- 2026-09-10, M2X1 MEASURED (H100): `t16moe_mx::mx_gemv` is bit-identical to the torch item-5 reference in 10 / 10 cases
  (GPT-OSS shapes, small odd shapes, with and without bias). BUILT next: `mx_gemm`, the tensor-core form for prefill
  rows - 64 x 128 tiles, int8 mma m16n8k32 per 32-wide K block on the unpacked nibbles, the per-block binary32 step
  on every accumulator element in block order (identical declared order to the GEMV), a flat per-expert tile list;
  gate (identity with the GEMV and the reference) and timing vs the GEMV and the int8 classic kernel queued (M2X3).
- 2026-09-10, M2X2 MEASURED (H100, GPT-OSS-20B, real tokens): the FUSED item-5 path (gate_route with bias, plain row quant,
  `mx_gemv` with the fused epilogue for both projections, the bf16-input SwiGLU-OAI epilogue, combine) is EQUAL to the
  item-5 reference at layers 0 and 12, T = 48 and 1,024 (4 / 4: sets, order, weights, output). The dense kernel gate
  stays 1,186 / 1,186 after the epilogue's bf16-input path; all nine MoE gates pass (265 cases). The GEMV form reads
  the weights once per row (1.7-2.4 s at 1,024 tokens): the tensor-core form (M2X3 / M2X4) is the prefill path.
- 2026-09-10, queued at the end of the chain (after M2E3 engine identity + bench, M2C2 block-scaled activations, M1S4 the
  served Qwen arm, M2X3 / M2X4 the tensor-core MX GEMM gate and fused check): M2S1 - GPT-OSS-20B SERVED, stock MXFP4 vs
  the item-5 contract through the plugin (LOCKSTEP_MOE=1 with the MX operand), ShareGPT W1 at 64 concurrent, 500
  prompts - the first served row for the GPT-OSS contract.
- 2026-09-10, M2E3 MEASURED (H100, GPT-OSS-20B, the fused item-5 path armed at load): ENGINE PERPLEXITY 329.49283843993334 on
  the raw windows = the item-5 reference arming's 329.49283843993334 to the last digit - the fused MXFP4-consuming
  kernel path IS the declared contract end to end in the engine (and raw 329 < stock 367). SPEED was a tenth of
  stock (decode 32.6 / 170.9 / 476.3 tok/s; TTFT 333 / 1,189 ms): both MX kernels unpacked the 4-bit codes through a
  `__constant__` 16-entry table indexed per lane - divergent constant-cache reads serialise (the GEMV ~100x off its
  weight-read floor). Replaced by a register-resident 64-bit packed table (no memory access per nibble); the gate /
  timing job (M2X3) and the fused checks pick up the new kernels, the engine bench re-queued (M2E4).
- 2026-09-10, M2C2 MEASURED (H100, GPT-OSS-20B): item 5b (exact MX weights + block-scaled int8 activations) replays GPU == CPU
  and halves the layer-level distance (2.2% / 2.9% vs item 5's 3.6% / 4.1%), but its chat perplexity is 15.254 - worse
  than item 5's 14.790 and above stock's 15.101. The layer L2 distance and the perplexity disagree for the second time
  today (the int8 + rotation form had the smallest distance of the int8 forms and +2.85%; item 5 has the largest
  distance of the MX forms and the best perplexity): the perplexity is the measure, the distance a diagnostic. NOT
  ADOPTED; item 5 (per-row int8 activations, exact MXFP4 weights, no rotation) stands as the GPT-OSS contract. All
  chat-perplexity readings on the same 61,410 tokens: stock 15.101, item 5 14.790, item 5b 15.254, int8 + 64-block
  rotation 15.531, int8 no rotation 16.539.
- 2026-09-10, M2X3 MEASURED (H100, register-table kernels): `mx_gemm` is EXACT - identical to the GEMV and the reference in
  5 / 5 cases (up to 5,405 dispatched rows) - and the GEMV gate stays 10 / 10. Timing under graph replay at the
  2,048-token prompt shape (8,192 dispatched rows over 32 experts): mx_gemm 2.51 ms (108 TOPS) vs the int8 classic
  CUTLASS kernel 0.47 ms (583 TOPS) at N 5,760 / K 2,880, and 1.26 vs 0.24 ms at N 2,880 - a first, unpipelined
  mma kernel (single-buffered 64 x 128 tiles, the nibble unpack in the load path) at 5x the classic kernel; the GEMV
  at 8 rows 0.104 / 0.054 ms - ~5x its weight-read floor: one thread per output walking 90 blocks, 184 blocks
  in flight, latency-bound. BUILT: `mx_gemv2` - 8 lanes per column compute the exact int32 block sums in parallel
  (order-free), stage them in shared memory, one lane performs the single ordered binary32 accumulate (the pinned
  order unchanged); wired for few rows; gate + timing + fused check queued (M2X5). The mma kernel's pipelining
  (cp.async double buffering, 128-row tiles, a wider unpack) is the prefill item that follows.
- 2026-09-10, M1S4 MEASURED (H100, Qwen3-30B-A3B SERVED: `vllm serve` + `vllm bench serve`, ShareGPT W1, 64 concurrent,
  500 prompts, prefix caching off, graphs) - the MoE stack ARMED in the served engine (48 layers, the marker file, the
  plugin trace in the engine core):

  | row | MoE stack (rep 4) | stock (reps 1 / 2) | ratio |
  |---|---|---|---|
  | request throughput | 14.57 req/s | 10.66 / 11.06 | 1.32-1.37x |
  | output token throughput | 2,961.9 tok/s | 2,166 / 2,249 | 1.32-1.37x |
  | median TPOT | 17.29 ms | 24.00 / 23.89 | 0.72x (faster) |
  | P99 TPOT | 26.8 ms | 87.8 / 30.2 | - |
  | median TTFT (queueing at saturation) | 11.1 s | 16.5 / 15.0 | 0.70x |

  Consistent with the engine harness (batch 32 decode 1.26x): at 64 concurrent sequences the exact int8 grouped
  path with the fused gate / dispatch / combine kernels beats stock's fused bf16 MoE by a third. Second reps of
  both arms queued (M1S5) to bound the run-to-run variance (stock's two reps differ by 4%).
- 2026-09-10, BUILT (gate + timing queued, M2X6): `mx_gemm2`, the pipelined tensor-core form of item 5 - 128 x 128 tiles
  (8 warps, 32 x 64 per warp = 2 x 8 mma per K block), two-stage cp.async double buffering of the A tile, the PACKED
  4-bit B tile and the block's exponents, the B fragments unpacked in registers from the packed bytes (4 nibbles per
  fragment word), the per-block binary32 step in block order unchanged. Wired for the many-row path; the 8-lane GEMV
  (`mx_gemv2`) for few rows. Chain now: M2X4 (fused check on the first mma form) -> M2S1 (GPT-OSS served, stock vs
  item 5) -> M2E4 (item-5 engine re-bench) -> M2X5 (gemv2 gate / timing / fused check) -> M1S5 (second serving reps,
  Qwen) -> M2X6 (gemm2 gate / timing / fused check at 1,024 tokens).
- 2026-09-10, M2X4 MEASURED (H100, GPT-OSS-20B, real tokens): the fused item-5 path with the tensor-core GEMM (`mx_gemm`) on
  the 1,024-token rows and the GEMV on the 48-token rows is EQUAL to the item-5 reference at layers 0 and 12 (4 / 4
  cells). The mma form's declared order (per-block binary32 step on every element, blocks ascending) reproduces the
  GEMV's and the reference's bits exactly.
- 2026-09-10, M2S1 MEASURED (H100, GPT-OSS-20B SERVED, ShareGPT W1, 64 concurrent, 500 prompts): stock MXFP4 19.15 req/s,
  3,821 tok/s, median TPOT 8.9 ms; the item-5 arm (ARMED, 24 layers, marker) 3.08 req/s, 616 tok/s, TPOT 83.9 ms -
  0.16x. Cause: at 64 concurrent the step dispatches 256 rows over 32 experts (~8 per expert); the 128-row tensor-core
  tile does 16x its useful work per weight byte and the 8-lane GEMV would re-read the weights once per row (8x the
  bytes). BUILT: `mx_gemm16` - a 16-row x 128-column tile (8 warps along N, one m16 row tile, the same pipeline and
  per-block step) so each expert's weights are read once per 16 rows; `mx_gemm2` picks 16 or 128 rows by rows per
  expert, the GEMV only near one row per expert. Queued (M2X7): gates of all forms, timing in the served decode regime
  (256 rows over 32 experts) against the int8 kernel, fused checks, the engine bench and the served comparison again.
  Floor arithmetic: the expert set is 398 MB per layer as 4-bit -> 9.5 GB per step -> ~2.9 ms at 3.3 TB/s against
  stock's 8.9 ms step.
- 2026-09-10, M2E4 MEASURED (H100, GPT-OSS-20B, item 5 in the engine with the register-table kernels, the 8-lane GEMV up to
  one row per expert and the 128-row pipelined tile above): decode 45.3 / 160.8 / 450.4 tok/s (0.15 / 0.10 / 0.09x of
  stock), TTFT 221 / 731 ms (0.16 / 0.17x) - far below what the standalone kernel timings predict (the GEMV at 8 rows
  ~0.1 ms, the expert set of a layer ~15 us of weight read at batch 1). Something in the fused MX block, not the GEMM
  alone, costs ~0.9 ms per layer at batch 1. A per-stage graph-replay profile of the MX block at GPT-OSS shapes
  (`moe_mx_prof.py`: gate_route, dispatch, up GEMM in each form, the bf16-input activation epilogue, down GEMM,
  combine) is queued (M2P5) - profile, don't guess.
- 2026-09-10, M2X5 timings (H100, graph replay, GPT-OSS shapes; the log read early): gemv2 at 8 rows 0.111 ms - no faster
  than the GEMV (0.103); the 16-row tile at the served decode regime (256 rows over 32 experts) 0.39 ms vs the int8
  kernel's 0.20 (the 128-row tile 2.16, the GEMV 2.15); at 8,192 rows the pipelined 128-row tile 5.3 ms vs the
  unpipelined 2.5 and the int8 kernel 0.46. DIAGNOSIS: the 4-bit unpack - a per-nibble table walk of ~10
  instructions, 8 nibbles per word, repeated per row in the GEMV and per FRAGMENT (four warps along M) in the pipelined
  tiles - is instruction-bound at ~4 G instructions per GEMM (~80 us at 8 rows, scaling with rows), not the weight
  read. FIX: `mx_unpack8_fast` - the 3-bit magnitudes through `__byte_perm` (an 8-byte table lookup for four nibbles in
  one instruction), the sign bits as byte masks, the signed int8x4 words by a carry-free per-byte negate (magnitudes
  <= 12); the tensor-core kernels now unpack each B tile ONCE into an int8 shared tile after the cp.async wait and
  read plain fragments. Exactness unchanged (integers). Queued (M2X8): gates of every form, timing, fused checks,
  the per-stage MX profile, the engine bench and the served GPT-OSS rerun.
- 2026-09-10, M2X5 MEASURED (H100): the 8-lane GEMV, the 16-row and 128-row pipelined tiles are identical to the GEMV and
  the reference in every gate case (GEMM-identity gate 5 / 5, GEMV 10 / 10), and the fused item-5 path on the 8-lane
  GEMV equals the reference on 48 real tokens. Timings as read above (the unpack diagnosis); the fast-unpack round
  (M2X8) re-measures them.
- 2026-09-10, M1S5 MEASURED (H100, Qwen3-30B-A3B SERVED, second reps): stock 10.97 req/s, 2,230 tok/s, median TPOT 23.83 ms,
  P99 30.9; the MoE stack (armed, marker) 14.54 req/s, 2,956 tok/s, median TPOT 17.33 ms, P99 22.9 - 1.33x on request
  and output throughput, TPOT 0.73x, reproducing rep 4 (14.57 / 2,962 / 17.29) within 0.2%. Stock's three reps:
  10.66 / 11.06 / 10.97 req/s. The served MoE row is reproducible: 1.32-1.37x of stock at 64 concurrent.
- 2026-09-10, M2X6 / M2X7 / the first M2X8 (H100): the FAST UNPACK BROKE EXACTNESS - the GEMV gate 0 / 10, the fused checks
  differ on every row (maxabs 13.5), while the GEMM-identity gate still passes (every form wrong the same way). Cause: the
  carry-free per-byte negate `(m ^ 0xFF) + 1` is wrong for m = 0 - E2M1's NEGATIVE ZERO (code 8) - where it carries 0x100
  into the neighbouring byte; random codes and the real weights both contain it. Fixed by masking the sign where the
  magnitude is zero. The speed numbers of those runs stand as UPPER BOUNDS of the fast kernels (arithmetic errors do not
  change timing): GEMV at 8 rows 0.059 ms (was 0.103), the 16-row tile at 256 rows over 32 experts 0.31 ms (was 0.39;
  int8 0.20), the 128-row pipelined tile at 8,192 rows 4.2 ms (int8 0.46: the pipelined tile stays slow - its per-K-block
  barriers and small mma work per load; the unpipelined 64-row form 2.2 ms), and in the engine decode 105 / 687 / 2,156
  tok/s (0.34 / 0.44 / 0.45x), the served row 10.66 req/s (0.56x of stock's 19.15). The quality of those runs is void;
  the fixed kernels re-gate in M2X8 (relaunched).
- 2026-09-10, M2P5 MEASURED (H100, the item-5 block per stage under graph replay, GPT-OSS shapes; timing valid though the
  kernels of that run carried the negative-zero bug): at 1 token the FULL forward is 0.327 ms while the stages sum to
  0.104 (gate_route 0.011, dispatch 0.005, up GEMV 0.045, activation 0.007, down GEMV 0.031, combine 0.005): ~0.22 ms
  of glue inside the fused forward that the stage profile does not see - an exact-sequence prefix profile of the MX
  forward is added (MXPREFIX) and runs in the relaunched M2X8. At 8 tokens the GEMV (0.30 ms at 32 rows) already loses
  to the 16-row tile (0.18): the GEMV threshold is now 16 rows. At 64 tokens (256 rows) the expert GEMMs take
  0.29 + 0.19 ms per layer on the 16-row tile against the int8 kernel's 0.20 + 0.10 reading twice the bytes: the tile
  is ~3x off its weight-read floor (90 K-blocks x 3 barriers per block, little work per barrier); a 64-wide K stage is
  the next change after the exactness re-gate.
- 2026-09-10, BUILT (gate + timing queued, M2X9, after the corrected fast-unpack round M2X8): `mx_gemm16b` - the 16-row tile
  with a 64-wide K stage (two 32-blocks per cp.async stage and per barrier set; the per-block binary32 step still per
  32-block in ascending order), the default for the 16-row tile. Expected: ~half the barrier cost per weight byte in the
  served decode regime (256 rows over 32 experts: 0.31 ms per projection on the 32-wide stage vs the int8 kernel's 0.20).
  M2X9 also re-runs the engine bench and the served GPT-OSS comparison on the corrected kernels.
- 2026-09-10, M2X8 gates (H100, the fast unpack with the negative-zero mask): EXACT again - the GEMV gate 10 / 10, every
  form identical in the GEMM-identity gate 5 / 5 (the 8-lane GEMV, the 64-row mma, the 16-row and 128-row pipelined
  tiles). Timing with the correct fast unpack: the 16-row tile at 256 rows over 32 experts 0.356 / 0.211 ms (up /
  down) against the int8 kernel's 0.200 / 0.105; the GEMV at 8 rows 0.093 / 0.072 ms (the mask costs ~0.03 over the
  broken form's 0.059); the 128-row pipelined tile at 8,192 rows 4.28 ms (the unpipelined 64-row form 2.35, int8 0.46).
  The fused checks, the profiles, the engine bench and the served rerun on these kernels follow in the same job; the
  64-wide K stage (M2X9) is the next timing.
- 2026-09-10, BUILT (gate + timing queued, M2X10): the 16-row tile generalised to NBLK 32-blocks per pipeline stage (64- and
  128-wide K: 2 or 4 blocks per cp.async stage and barrier set), the per-block step unchanged. Barrier arithmetic for
  the served decode regime: 90 K-blocks x 3 barriers at 32-wide ~ the measured 0.36 ms; 128-wide -> ~23 stages -> a
  ~0.1 ms floor-class time against the int8 kernel's 0.20 reading twice the bytes.
- 2026-09-10, M2X8 MEASURED (H100, the corrected fast unpack): the fused item-5 path is EQUAL to the reference on real
  tokens again (T = 48 and 1,024). The exact-sequence prefix profile matches the stage sums (forward at 1 token 0.140 ms =
  gate_route 0.010 + dispatch 0.004 + up GEMV 0.064 + activation 0.005 + down GEMV 0.051 + combine 0.003): there is no
  hidden glue - the earlier 0.33 ms reading was the broken kernels' own cost. ENGINE (graphs): decode 187.6 / 620.9 /
  1,934.8 tok/s = 0.61 / 0.39 / 0.40x of stock MXFP4; TTFT 181 / 604 ms = 0.19 / 0.21x. Where the time is: at 1 token
  the two GEMVs (0.115 of 0.140 ms per layer; weight floor ~15 us: latency-bound at 4 rows); at 32-64 tokens the 16-row
  tile's two projections 0.29 + 0.21 ms per layer (barrier-bound; the 64- and 128-wide K stages are queued); at
  prefill the pipelined 128-row tile is slower than the unpipelined 64-row form (4.3 vs 2.35 ms at 8k rows; int8
  CUTLASS 0.46) - prefill rows now take the 64-row form until a proper multistage mma pipeline exists. The served rerun
  on these kernels follows in the same job.
- 2026-09-10, M2X8 served (H100, GPT-OSS-20B, item 5 with the corrected fast unpack, the 16-row tile at the 32-wide K stage,
  ShareGPT W1 64 concurrent 500 prompts): 10.18 req/s, 2,031 tok/s, median TPOT 27.6 ms vs stock's 19.15 / 3,821 / 8.9 -
  0.53x (the broken-math run read 0.56x: the arithmetic fix cost nothing). The served regime is the 16-row tile's
  barrier-bound case; the 64- and 128-wide K stages (M2X9 / M2X10) are its next readings. Nsight Compute is on the pod:
  a kernel profile of the GEMV at 4 rows and the 16-row tile at 256 rows is queued (M2N1) to read stall reasons and
  achieved bandwidth directly.
- 2026-09-10, M2X9 timings (H100, graph replay, 256 rows over 32 experts, exact in every case): the 16-row tile at the
  32-wide K stage 0.354 / 0.211 ms (up / down), at 64-wide 0.395 / 0.243, at 128-wide 0.470 / 0.273 - WIDER STAGES ARE
  SLOWER. The barrier hypothesis is wrong. Achieved bandwidth: ~265 MB of 4-bit weights in 0.354 ms = 0.75 TB/s, 23% of
  HBM, while the int8 CUTLASS kernel moves twice the bytes in 0.20 ms (2.65 TB/s). The two-stage cp.async ring hides one
  iteration of tiny work behind each HBM round trip: latency-bound, which the Nsight Compute profile (M2N1, queued) should
  confirm as long-scoreboard stalls; the remedy is a deeper pipeline (3-4 stages) and more work per block (more columns
  or more rows per tile when the regime allows), not wider K. The 32-wide stage stays the default.
- 2026-09-10, BUILT (gate + timing queued, M2X11, after the Nsight profile M2N1): `mx_gemm16s<STAGES>` - the 16-row tile with a
  4- or 8-deep cp.async ring of 32-wide K blocks (`wait_group<STAGES - 1>`), so each HBM round trip is hidden behind
  STAGES - 1 iterations of work instead of one; the same unpack-once stage and the same declared per-block step. The
  32-wide two-stage form stays the default until the timing reads. Chain: M2X9 (64-wide, incl. the engine bench and the
  served rerun at that default) -> M2X10 (128-wide) -> M2N1 (Nsight Compute: stall reasons, DRAM %, occupancy of the GEMV
  at 4 rows and the 16-row tile at 256 rows) -> M2X11 (the 4- / 8-stage rings).
- 2026-09-10, BUILT (gate + timing in M2X11): the MX GEMV templated on columns x lanes, with a 32-lane x 8-column form for
  1-token decode (2,880 blocks at 4 rows instead of 720, each lane's chain ~3 blocks instead of 11 - the latency-bound
  regime of the batch-1 row), used below 17 rows; the 16-row tile keeps the 32-wide two-stage default until the deeper
  rings read. A GPT-OSS engine bench and served rerun on the current defaults is queued behind M2X11 (M2E6).
- 2026-09-10, M2X9 MEASURED (H100, GPT-OSS-20B, the 64-wide K stage as the 16-row tile's default, prefill on the 64-row form):
  the fused path EQUAL to the reference at T = 64 and 1,024; engine decode 187.7 / 513.5 / 1,492.5 tok/s (0.61 / 0.33 /
  0.31x - the batch-8 / 32 rows worse with the 64-wide stage, as its timing predicted), TTFT 109 / 382 ms (0.31 / 0.33x -
  prefill on the 64-row form up from 0.19 / 0.21x); served 10.21 req/s, 2,037 tok/s, TPOT 25.9 ms (0.53x). The 32-wide
  stage is the default again; the deeper rings (M2X11) and the profile (M2N1) decide the next form.
- 2026-09-10, M2X10 MEASURED (H100, graph replay; every form exact, GEMV 10 / 10, GEMM-identity 5 / 5 including the 4- and
  8-stage rings and the 32-lane GEMV): the deeper cp.async rings do not help the 16-row tile at 256 rows (2-stage 0.354,
  4-stage 0.355, 8-stage 0.378 ms) and the 32-lane GEMV is SLOWER than the 8-lane (0.098 vs 0.055 ms at 4 rows; 0.228 vs
  0.103 at 8). Neither barriers, nor pipeline depth, nor block count limits these kernels; the remaining candidates are
  instruction issue (the per-element ldexpf / fmul / fadd step and the unpack per K-block) or shared-memory traffic. No
  further blind variants: the Nsight Compute profile (M2N1, next in the chain) reads the stall reasons and the achieved
  DRAM / SM throughput directly. The GEMV default is back to 8 lanes; the 16-row tile stays at the 32-wide two-stage form.
- 2026-09-10, BUILT (gate + timing queued, M2X12, after M2E6): the block scale 2^(k - 128) as an exact bit-constructed binary32
  (exponent field k - 1 for the normal range, ldexpf only for k = 0) in every MX kernel - the same value as ldexpf, ~2
  instructions instead of ldexpf's range handling, on the hottest per-element step of the declared accumulate. The gate
  re-checks identity of every form; the timing reads whether the per-element step was the issue-bound part.
- 2026-09-10, M2N1 (H100): Nsight Compute cannot read the GPU performance counters in this container (ERR_NVGPUCTRPERM) -
  no hardware stall reasons. Replaced by PROFILE-BY-ABLATION (M2X13, queued): timing-only variants of the 16-row tile with
  the nibble unpack removed, the per-block binary32 step removed, the mma removed, and loads only - each isolates one
  cost component at the served decode regime (256 rows over 32 experts); the outputs of the ablation kernels are not
  used for anything.
- 2026-09-10, M2X11 MEASURED (H100): every form exact (GEMV 10 / 10, GEMM-identity 5 / 5 with the 4- / 8-stage rings and
  the 32-lane GEMV). Timing at 256 rows over 32 experts, this run: 2-stage vs 4-stage vs 8-stage (see the lines above
  in the results file) - the deeper rings within a few percent of the two-stage form, the 32-lane GEMV slower: the
  profile-by-ablation (M2X13) decides the next change.
- 2026-09-10, M2X11 re-read (the run included the bit-constructed power of two): the 16-row tile at 256 rows 0.354 -> 0.299 ms
  (up) and 0.211 -> 0.177 (down), the 128-row tile 1.73 -> 1.21, the GEMV at 4 rows 0.055 -> 0.052 - a 16% gain from
  removing ldexpf from the per-element step: instruction issue is a real part of the cost. One more exact reduction
  shipped for M2X12 / M2X13: the step `acc = RN(acc + RN(d * s))` as `fmaf(d, s, acc)` - identical bits because the
  product d * s is an exact power-of-two scaling of an integer below 2^24 (one rounding either way), one instruction
  instead of two, in every MX kernel. The ablation (M2X13) then apportions what remains among unpack, mma and step.
- 2026-09-10, M2E6 MEASURED (H100, GPT-OSS-20B, item 5 on the current defaults: 8-lane GEMV <= 16 rows, the 16-row tile at the
  32-wide stage, the 64-row form for prefill; the fused path EQUAL at T = 48): engine decode 190.3 / 524.2 / 1,521.2 tok/s
  (0.62 / 0.33 / 0.32x of stock), TTFT 107 / 376 ms (0.31 / 0.32x); served 10.32 req/s, 2,060 tok/s, TPOT 25.4 ms (0.54x).
  DISCREPANCY: the standalone block at 32 tokens measures 0.54 ms per layer (graph replay), which predicts ~2,000 tok/s
  at batch 32 against the 1,521 measured; the int8 form's engine rows match its standalone block. Something costs ~0.25 ms
  per layer inside the engine on the MX path only. Added an in-engine per-call CUDA-event timer to the arming
  (LOCKSTEP_MOE_TIMER, eager engines) and queued (M2E7) the MX and int8 forms side by side at decode batch 32 and 1.
- 2026-09-10, M2X12 MEASURED (H100, the fmaf step): every form exact (GEMV 10 / 10, GEMM-identity 5 / 5). The 16-row tile at
  256 rows 0.306 / 0.177 ms - unchanged from the bit-constructed scale alone (0.299 / 0.177): the per-element step is no
  longer the limiter; the GEMV at 4 rows 0.051 / 0.032; the pipelined 128-row tile improved to 2.94 ms at 8k rows while
  the unpipelined 64-row form read 2.88 (from 2.33, run-to-run or the fmaf's scheduling - both stay far from the int8
  kernel's 0.46). Next readings: the ablation (M2X13) for the decode tile's remaining cost, the in-engine timer (M2E7)
  for the engine-vs-standalone gap.
- 2026-09-10, M2X13 MEASURED - PROFILE BY ABLATION (H100, the 16-row tile at 256 rows over 32 experts, graph replay, timing
  only): full 0.301 ms; without the nibble unpack 0.216 (-0.085); without the per-block step 0.216 (-0.085); without the
  mma 0.265 (-0.036); loads and barriers only 0.119 (for N = 5,760, K = 2,880; the N = 2,880 projection: 0.181 / 0.124 /
  0.169 / 0.176 / 0.094). READING: the load skeleton alone moves the 265 MB of 4-bit weights at ~2.2 TB/s - bandwidth-class
  - and the compute (unpack ~85 us, the step ~85, mma ~36) ADDS to it instead of overlapping: with every warp doing load,
  unpack, mma and step in lock step behind two barriers per K block, the SM has nothing to overlap the memory wait with.
  The remedy the data supports is WARP SPECIALISATION: producer warps that issue the cp.async loads and unpack the B tile
  into an int8 ring, consumer warps that run the mma and the per-block step, synchronised per stage by named barriers -
  the compute (~0.2 ms) then hides under the loads (~0.12) toward the int8 kernel's 0.20 ms. Built as `mx_gemm16ws`
  (gate + timing queued, M2X14). Not the remedy: wider K stages, deeper uniform rings, more blocks (all measured).
- 2026-09-10, M2E7 MEASURED (H100, GPT-OSS-20B, eager engine, CUDA events around each fused MoE call, trimmed means): the
  MX form 0.185 ms per layer at 1 token (min 0.181), 0.881 at 32 tokens (min 0.748), 8.8 at 4,096 prompt tokens; the
  int8 form 0.125 / 0.310 (min 0.255) / 2.29. The int8 block in the engine matches its standalone profile; the MX block at
  32 tokens costs 0.75-0.88 ms against 0.54 standalone (random, balanced routing). READING: with the real routing the
  rows per expert are skewed, heavy experts span two or three 16-row tiles and each tile re-reads that expert's column
  slice (184 KB per block): the weight traffic grows with the tile count, not with the rows. The minimum is one pass
  over each active expert's weights - a tile that holds all of an expert's rows at this batch (up to ~64) - which the
  128-row pipelined tile did read once but paid 4x the compute per byte (1.2 ms). The warp-specialised tile (M2X14)
  tests whether compute can hide under the loads at all; if it does, its 32- or 64-row version is the decode kernel.
- 2026-09-10, BUILT (gate + timing queued, M2X15, after M2X14): the warp-specialised tile generalised to 16 / 32 / 64 rows
  (consumer warps hold ROWS / 16 m16 sub-tiles per 16 columns; the producers load ROWS rows of A; the B tile, its unpack
  and the exponents unchanged) - a tile covering all of an expert's rows at decode batch reads its weights ONCE. The gate
  adds a SKEWED-routing timing (256 rows with counts ~ 1 / rank over 32 experts, the engine's reality) for the 16-row
  tile, the three warp-specialised heights and the int8 kernel.
- 2026-09-10, M2X14 MEASURED (H100): the warp-specialised 16-row tile is EXACT (every form identical, 5 / 5) but only 8% faster
  (0.277 vs 0.300 ms at 256 rows; 0.166 vs 0.176): with four producer warps doing the loads AND the unpack (the largest
  compute term), the producers are the critical path. BUILT (M2X16, after M2X15): `mx_gemmws2<ROWS>` - producers load
  only (A rows, the packed B tile, the exponents into the ring); each consumer warp unpacks its own 16 columns from the
  packed ring into a per-warp int8 tile before its mma and step, so the unpack, the mma and the step share the consumer
  side while the loads run ahead. Gates and timing (balanced and skewed routing) at 16 / 32 / 64 rows.
- 2026-09-10, M2X15 MEASURED (H100, N=5760 K=2880, 256 rows over 32 experts): the taller warp-specialised tiles LOSE - balanced
  ws16 0.269 / ws32 0.333 / ws64 0.454 ms against the pipelined 16-row tile's 0.300 and int8 grouped GEMM's 0.201; balanced
  routing leaves 8 rows per expert, so a tall tile runs mostly empty and there are fewer blocks. SKEWED routing (counts
  proportional to 1/rank, max 61 rows per expert): gemm16 0.359, ws16 0.326, ws32 0.362, ws64 0.456, int8 0.203 - the 16-row
  tiles pay for re-reading a 61-row expert's weights four times, and int8 is FLAT (its grouped GEMM does not). All forms
  identical (5 / 5). Decision: the decode tile is a 16-row warp-specialised form; M2X16 decides v1 (producers unpack) vs v2
  (consumers unpack). `LOCKSTEP_MOE_MXBK` selects the form in the engine (32 pipelined, 200 v1, 300 v2); `LOCKSTEP_MOE_MXPF`
  optionally puts prefill rows on a selected tile form (default: the 64-row `mx_gemm`).
- 2026-09-10, M2X16 MEASURED (H100, 256 rows over 32 experts): the loads-only-producer variant (`mx_gemmws2`, consumers unpack
  their own 16 columns) is EXACT (every form identical, 5 / 5) and NOT faster: ws2-16 0.275 balanced / 0.331 skewed against
  ws16 v1 0.269 / 0.326; ws2-32 0.325 / 0.367, ws2-64 0.460 / 0.484 (N=5760); at N=2880 0.166 / 0.189 against 0.166 / 0.187.
  Moving the unpack to the consumers did not shorten the critical path: the per-K-block consumer work (unpack + step + mma) is
  now the longer side, and the 16-row tile is bound by the same ~0.27 ms the v1 form already reached. DECISION: the decode
  tile is the v1 warp-specialised 16-row form (`LOCKSTEP_MOE_MXBK=200`); the GPT-OSS engine bench and served run (M2E8) go
  on it. The remaining gap to int8's 0.201 ms is the unpack itself (the nibble -> int8 expansion that int8 weights never pay).
- 2026-09-10, PR #10 RECORD CHECK (the v31 float-norm cross-arch kit, `results/contract3_20260910/xarch`): the committed
  digests do NOT reproduce from the committed inputs and the committed reference for 22 of 72 cases - exactly the cases
  where the eps term changes a bit (widths 128 / 3072 / 4096 with eps, M >= 7 plus N3072 channel M1). With eps = 0 all 72
  reproduce, on ARM (this Mac, numpy 2.5.3), x86 (numpy 1.26.3 and 2.3.5, torch 2.4.1 CPU and 2.11 CPU) and the H100 (torch
  2.4.1 CUDA): the v31 record was made before the eps term was added to the rule and the kit was updated without re-running
  it. Not an arithmetic defect: every CPU and GPU evaluation of the committed reference agrees bit for bit across five
  environments. The v2.6 record (`xarch26`) is the one that binds; S26G regenerates its inputs and re-runs it on the merged
  sources, and the ARM leg of the norm twin is re-run from those digests.
  CONFIRMED on the H100 (the committed kit's `xarch_test.py run`, torch 2.11 + cu130 build): local gate PASS, kernel == torch
  reference == numpy twin on 72 / 72, and the kernel's own digests differ from the committed record on the same 22 cases -
  so the record, not the kernel or the references, is the stale element.
- 2026-09-10, M2E8 MEASURED (H100, GPT-OSS-20B, the warp-specialised 16-row decode tile `LOCKSTEP_MOE_MXBK=200`, on the tree
  AFTER the PR #10 merge): engine decode 189.9 / 437.8 / 1,261.5 tok/s, TTFT 128 / 453 ms; served ShareGPT W1 8.55 req/s,
  1,706 tok/s, TPOT 31.4 ms (0.45x of the stock 19.15 / 3,821 / 8.9). SLOWER than M2E6 on EVERY row (190.3 / 524.2 / 1,521;
  107 / 376; 10.32 req/s) including prefill, which the tile change does not touch - so the tile is not the cause. Candidates:
  the merged tree (t16m_kernels.cu, moe_fused, the 2.6 default), or the node. Not interpreted further; M2E9 runs stock and
  both tiles back to back on the current tree (stock first and last, clocks logged), which separates node, tree and tile.
- 2026-09-10, S26G MEASURED (H100, the MERGED tree: main = PR #10 + our Stage III work, T16L_NORMF default 1): every gate
  passes - t16n_test 160 / 160, t16m_gate kernels 1,186 / 0, t16l_r4_test PASS, t16m_qknorm_test 48 / 0, contract vectors
  written, the v2.6 cross-arch kit local gate PASS on 158 cases / 538 digests (kernel == torch reference == numpy twin on
  the merged sources), the MoE gates (deq_rows_g 15, gmm_deq 8, addendum epilogue 8, R1 gate 20, MX GEMV 10, MX GEMM 5)
  all PASS. The PR's recorded xarch26 digests are NOT comparable: the committed kit regenerates 430 arrays (sha
  af8935225283df14) where the record says 442 (90c0b36a5f098aad) - the kit's case list changed after the record was
  made, the same pattern as the v31 record. CPU leg: the numpy twin on this ARM Mac reproduces the H100 torch reference
  digests of the merged run on 72 / 72 norm cases (`xarch26/cpu_norm_check.py`, `digests_h100merge.json`). So the v2.6
  arithmetic holds on Hopper, Ada, Blackwell (their record), x86 and ARM CPUs (ours); what needs fixing in the PR's
  records is the records.
- 2026-09-10, M2X17 MEASURED (H100): `mx_i2f` (the exact magic-number int -> fp32) is bit-identical in every form (GEMV 10 / 10,
  GEMM-identity 5 / 5) and changes NOTHING in time: gemm16 0.310 (0.300 before), ws16 0.275 (0.269), ablation no-fp32step
  0.223 against full 0.309 - the "step" term is still 0.085 ms, so it is not the I2F instruction. What the ablation removes
  with the step is also the per-fragment scale fetch: 16 byte loads from shared memory and 16 shifts per lane per K block
  (the lane's two columns in each of eight n-fragments), against eight mma. Next (M2X18): an ablation that keeps the scale
  fetch and drops only the fmaf (ABL 105), and a form that stores the block exponents transposed so a lane fetches its 16
  scale bytes in one 16-byte load (bk 201 on the warp-specialised tile). mx_i2f stays (exact, free).
- 2026-09-10, M2E9 MEASURED (H100, GPT-OSS-20B, PAIRED on the current tree, stock first and last): stock 309.1 / 1,574.6 /
  4,833.4 tok/s, TTFT 36.6 / 126.6 ms, and again 307.7 / 1,576.8 / 4,842.4, 35.5 / 128.2 - the node reproduces stock to
  0.5%. Armed, pipelined tile (bk 32): 189.6 / 427.6 / 1,220.3, TTFT 130.9 / 462.4; armed, warp-specialised tile (bk 200):
  189.7 / 427.2 / 1,219.8, 131.0 / 462.7 - IDENTICAL to 0.1%, so either the tile knob does not reach the engine or the
  tile is not on the critical path; and BOTH are 18-22% slower than M2E6 (524.2 / 1,521.2; 107 / 376) on the same node
  with stock unchanged. The slowdown is in the tree between M2E6 and now (the PR #10 merge, mx_i2f, the tile knob) and
  it touches prefill too. Not interpreted further: M2E10 runs the in-engine timer on the current tree and the engine
  bench on the pre-merge runtime (commit e6cb1bf, shipped to /workspace/p2/stage_pre) back to back, and logs the tile
  form at arming. (The first M2E9 attempt hung 30 minutes on a stale torch-extension build lock left by a gate job I
  killed; the lock was removed and the armed runs relaunched.)
- 2026-09-10, M2X18 MEASURED (H100): the persistent tile (bk 210) FAILS - GEMM-identity 0 / 5 (every element differs) and a
  0.0066 ms "time": the kernel never ran (a launch that returns at once, most likely a launch-configuration error the
  wrapper did not check: the tile loop raises the consumers' live state and `__launch_bounds__(384)` may no longer fit).
  bk 210 is NOT selectable as a default and stays a variant. Added `C10_CUDA_KERNEL_LAUNCH_CHECK()` to the mx_gemm2
  wrapper so a failed launch raises, and `__launch_bounds__(384, 1)` on the persistent kernel; M2X19 re-runs the gate
  after M2E10. The ws16 default re-read 0.276 / 0.320 (balanced / skewed) and int8 0.201 / 0.203 in the same run.
- 2026-09-10, M2E10 / M2E12 / M2E13 MEASURED (H100, GPT-OSS-20B, the engine bench on OLDER TREES with the same harness; stock
  reproduces to 0.5% throughout): the pre-merge runtime e6cb1bf reads the SAME as the current tree (189.3 / 437.9 / 1,250;
  127.6 / 452 ms) and its in-engine per-call MoE cost is unchanged from M2E7 (0.185 / 0.88 ms) - so the PR #10 merge is NOT
  the cause. M2E6's own tree dd45a5d reproduces M2E6 (513 / 1,494; 109 / 383). Commit bisect: ca5db3f 513 / 1,493 (109 /
  384), b1da7c3 (bit-constructed pow2 in every kernel) 524 / 1,521 (107 / 378) - M2E6 exactly - and 17c35bf (the
  per-block step as a single fmaf) 437 / 1,251 (128 / 452). THE FMAF FORM IS THE REGRESSION: identical rounding (the
  product is exact), no change in the standalone tile timings (M2X12), and 17-20% slower on every engine row including
  prefill. Not interpreted further (a plausible mechanism is a changed instruction schedule in the GEMV and the 64-row
  prefill form, which the standalone gate does not time at engine shapes). FIX: every kernel back to the separate
  `__fmul_rn` + `__fadd_rn` (same rounding), keeping `mx_i2f`; M2E14 re-gates identity and re-runs the engine bench and
  the served W1. Two lessons for the perf handover: an engine-shape timing belongs in the standalone gate, and a "no
  change" reading on one shape is not a "no change" result.
- 2026-09-10, M2E14 MEASURED (H100, GPT-OSS-20B, the per-block step back to separate multiply + add): every MX form
  identical (GEMV 10 / 10, GEMM-identity 5 / 5); engine decode 190.2 / 508.4 / 1,479.5 tok/s, TTFT 110.4 / 390.6 ms -
  within 3% of M2E6 (190.3 / 524.2 / 1,521.2; 107 / 376) against the regressed 189.7 / 427 / 1,220 and 131 / 462; served
  ShareGPT W1 9.98 req/s, 1,991 tok/s, TPOT 26.3 ms = 0.52x of stock (M2E6: 10.32 / 2,060 / 25.4 = 0.54x). The regression
  is CLOSED; the standalone tile timings are unchanged (gemm16 0.317, ws16 0.282, int8 0.200 ms), which is the point: the
  fmaf form cost nothing in the standalone gate and 17-20% in the engine.
