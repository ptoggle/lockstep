# Evidence index

Every claim the project makes, the artifact that backs it, and how to re-run it. Written
2026-09-10 for the paper (`paper/main.tex`, version 0.2.0) and for anyone auditing the
repository. A claim without a row here is not a claim the project makes. Where a record is
marked *pending* the claim in the paper is worded to what exists.

Legend: **gate** = a test that must pass with zero tolerance; **file** = a committed result;
**theorem** = a checked Lean declaration (`formal/`); **log** = a dated entry in a plan log.

## 1. The contract is a total, executable definition

| claim | evidence | re-run |
|---|---|---|
| The attention rule LSSA-B / B8 / B9.x is fully declared (constants, order, roundings, masks) | `docs/contract/BLOCK_EXPONENT_VARIANT.md` (normative); T2 table digest `7a42a785...` | `python lssa_runtime/lssab8_selftest.py` (S0-S11, 28 checks / 119 cases) |
| The dense decoder is fully declared (linears, outlier side path, norm, RoPE, SiLU, head, TP) | `docs/contract/NON_ATTENTION_STACK_v2.6.md` over `v2.5.md`; SPEC sections 2, 7.20-7.26; 36 contract vectors identical on x86 (torch 2.11, numpy 2.3) and Apple silicon (torch 2.14, numpy 2.5) after the generator moved to numpy's PCG64 stream (torch.randn is not stable across torch versions: 2026-09-10) | `make vectors` (`python tools/contract_vectors.py --check conformance/contract_vectors.json`) |
| The MoE block is fully declared (gate, table router, tie rule, dispatch, experts, combine; GPT-OSS addendum; exact-MXFP4 rule) | SPEC 7.27 (a `docs/contract/MOE_v3.md` file is a Stage IV S0 deliverable) | `python lssa_runtime/moe_m1_gate.py`, `moe_mx_gate.py` |
| The scalar golden is independent of torch (python ints + numpy) | `lssa_runtime/lssab8_golden.py`, `lssab9_golden.py`, `t16n_ref.py` (numpy twin) | `lssab8_selftest.py`, `t16n_test.py` |
| Integer safety bounds and fold determinism are theorems | **theorems** `score_abs_le`, `block_value_abs_le`, `block_weight_le`, `lssa_deterministic`, `scheduled_refines_canonical`, `splitKV_refines_canonical` in `formal/Lockstep/Attention/` | `lake build && lake exe lockstep-formal check` |
| The binary32 / bfloat16 model is executable at bit level | **theorems/defs** `addRNE`, `divideRNE`, `rescaleNat`, `toBF16RNE` (`formal/Lockstep/Numeric/Binary32.lean`); bridge artifact `results/lean_contract_artifacts.json` | `python3 lssa_runtime/verify_lean_artifacts.py` (+ `--negative`) |
| What the formalisation does NOT prove | `docs/LEAN_PROOF_SCOPE.md` (trust boundaries: CUDA, compiler, hashes, runtime) | - |

## 2. The engine computes the contract (implementation agreement)

| claim | evidence | re-run |
|---|---|---|
| Decode kernels == torch reference | **gate** K1-K8 43/43, K9 10/10 (SPEC 7.26; `lssa/portable/t13b_gate.py`, `t15_kgate.py`); t16n 21/21 sink on/off | `LSB_GATE_KS=0 LSB_GATE_SINK=0 python lssa/portable/t13b_gate.py` |
| FA3 prompt seam == golden, incl. the descending-walk NEGATIVE control | **gate** G0-G8 8/8, seam 48/48 (`lssab8_gate.py`, `lssab9_seam_gate.py`; SPEC 7.23) | `python lssa_runtime/lssab8_gate.py` |
| Fused dense kernels == declared ops on captured real activations | **gate** `t16m_gate.py kernels` 1,186 / 0 (SPEC 7.25-7.26; PLAN_3 S26G on the merged tree) | `python lssa_runtime/t16m_gate.py kernels` |
| Float norm 2.6 kernel == torch == numpy | **gate** `t16n_test.py` 160 / 160; R4 quantiser `t16l_r4_test.py` 48; fused head norm `t16m_qknorm_test.py` 48 (PLAN_3 S26G) | those scripts |
| Engine prompt / generated rows == reference (E1 / E4) | **gate** E1 151,200 rows (7B), 172,800 (Llama-3.1-8B), 86,400 (70B TP=4); E4 29,736 (SPEC 7.20, 7.23, 7.25) | the engine's H100 gate script (not in this repository) |
| Token identity across batch composition, graph modes, TP | **gate** F3, E6 (SPEC 7.25), TP custom all-reduce 32 cases (`t16s_car_gate.py`) | `python lssa_runtime/t16l_egates.py` |
| MoE kernels == reference | **gate** route 60, combine 20, dispatch 18, grouped GEMM 64, epilogues 8-20 (263 cases); MX GEMV 10/10, MX tile identity 5 forms (SPEC 7.27; `results/s3/logs/`) | `python lssa_runtime/moe_m1_gate.py`, `moe_mx_gate.py` |
| MoE engine perplexity identical across runs / kernel forms | **file** `results/s3/engine/engine_ppl_arm1_g1_e1.json`; PLAN_3 M1E.. (8.81400582424298, eight readings), M2E3 | `MODE=ppl ARM=1 python lssa_runtime/moe_engine.py` |
| Chunk boundaries: 128-aligned invariance holds, a non-aligned cut DIFFERS (negative control S8) | **gate** `lssab8_selftest.py` S8; scheduler hook `install_lssab_chunk_alignment` | `lssab8_selftest.py` |

## 3. The same bits across devices

| claim | evidence | re-run |
|---|---|---|
| The 2.6 kit's 538 digests are identical on H100 sm_90, RTX 4090 sm_89 and RTX 5090 sm_120 (regenerated 2026-09-10 from the committed kit with shipped inputs `af8935225283df14`; torch 2.11 / 2.4 / 2.8, CUDA 13.0 / 12.4 / 12.8) | **file** `conformance/xarch26/RESULTS.md` (regenerated record), `digests_h100merge.json`, `digests_ada_rerun.json`, `digests_blackwell_rerun.json`; the earlier author record (`digests_{h100,ada,blackwell_pro6000}.json`, RTX PRO 6000) is kept as history | `CONFORMANCE.md` |
| The merged tree reproduces the kit locally on the H100 (158 cases / 538 digests, kernel == torch == numpy) | **file** `results/contract3_20260910/xarch26/digests_h100merge.json` (PLAN_3 S26G) | same |
| The norm reference is bit-identical on x86 and ARM CPUs (72 / 72 cases vs the H100 reference) | **file** `results/contract3_20260910/xarch26/cpu_norm_check.py` + `digests_h100merge.json` (PLAN_3 S26G) | `python cpu_norm_check.py <dir with inputs.npz> digests_h100merge.json` |
| An H100 attention dump replays with zero differing bits on RTX 4090, x86, Apple M4 Max, B200 | **file** `results/verify_*.json` (SPEC 7.18, 7.21) | `python lssa_runtime/verify_rows.py <dump.pt> <keq.pt> <report.json>` |
| Hopper SKUs (H100 SXM / NVL / H200) reproduce one manifest's fingerprint | SPEC 7.20 | `lockstep selftest <manifest> --url ...` |
| NOT claimed: A100 (the Phase 0 "90/90" record is not in this repository) | - | - |

## 4. Speed against stock vLLM (same node, same session, one process per stack)

| claim | evidence | re-run |
|---|---|---|
| Qwen2.5-7B TTFT 0.90 / 1.03 / 1.06 / 0.99, decode 1.14 / 1.10 / 1.01, served 0.87 / 0.97 / 0.97 (2026-09-09) | SPEC 7.24-7.26; PLAN_24 "S7 P4 MEASURED"; **files** `results/s2/perf/`, `t14_res/*S7P4*` | `python lssa_runtime/t13b_bench.py` (ONLY=stock \| lssab), `t14_bench.py` |
| Llama-3.1-8B, Mistral-7B, Phi-4-mini, size sweep, 7B-1M rows (2026-09-09) | PLAN_24 log; README section 3 | same |
| Qwen3-8B and Phi-4-mini under 2.6 (2026-09-10, second node) | **files** `results/contract3_20260910/logs26/final/TABLE.txt`, `perf26_*.json` | `results/contract3_20260910/logs26/final/v26_final.sh` |
| Qwen2.5-72B TP=4 served 0.91 / 0.98 / 0.98 (2026-09-06) | SPEC 7.17 | `t14_bench.py` at TP=4 |
| Qwen3-30B-A3B decode 0.87 / 1.04 / 1.26, TTFT 0.86 / 0.88, served 1.32-1.37x (2026-09-10) | PLAN_3 M1F11 (stock arm M1E9), M1S4 / M1S5; **files** `results/s3/served/stock_rep{1,5}_w1_rinf.json`, `moe_rep{4,5}_w1_rinf.json`, `results/s3/engine/` | `MODE=bench python moe_engine.py`; `t14_bench.py moe` |
| GPT-OSS-20B exact-MXFP4 decode 0.62 / 0.33 / 0.32, TTFT 0.31 / 0.32, served 0.54x (M2E6); after the fmaf regression (M2E8, 0.45x) and its fix, 0.52x and 190 / 508 / 1,480 tok/s (M2E14) | PLAN_3 M2E6, M2E8, M2E9, M2E10, M2E12, M2E13, M2E14; **files** `results/s3/served/moe_repG{5,8}_w1_rinf.json`, `stock_repG1_w1_rinf.json`, `results/s3/logs/` | same with `MODEL=openai/gpt-oss-20b LOCKSTEP_MOE_MX=1` |
| GPT-OSS-20B int8-expert route 0.85 / 0.72 / 0.71, TTFT 0.80 / 0.83 (early, M2E2) | SPEC 7.27; PLAN_3 M2E2 | same with `LOCKSTEP_MOE_MX=0` |
| WITHDRAWN: Qwen2.5-72B TP=4 single-stream rows (stock arm had been armed) | SPEC 7.24 closing paragraph | - |

## 5. Quality against stock

| claim | evidence | re-run |
|---|---|---|
| Dense families within the 1% wikitext-2 band under 2.5 (7B +0.27, Llama-8B +0.22, Llama-3B -0.11, Mistral +0.09, Qwen3-8B -0.31, Phi-4-mini +0.55, 72B +0.14, 70B +0.81 under B9.3) | SPEC 7.23 matrix, 7.24 prepare gates; **files** `results/s1/`, `results/s2/` | `lockstep prepare` quality gate |
| 2.6 vs 2.5: Qwen3-8B -0.035% [-0.105, +0.036], Phi-4-mini +0.011% [-0.071, +0.090] (300 windows, paired bootstrap) | **files** `results/contract3_20260910/logs26/ppl_*.log` | `results/contract3_20260910/logs26/perf/v26_perf.sh` |
| Qwen3-30B-A3B v3 +0.12% (rotations on) / +0.71% (off), 30 x 2,048 windows | SPEC 7.27; `results/s3/m0` | `python lssa_runtime/moe_arm2.py` |
| GPT-OSS chat perplexity: exact-MXFP4 14.790 vs stock 15.101 (-2.06%); int8 experts 15.531 (+2.85%); item 5b 15.254 | SPEC 7.27; PLAN_3 M2P3 / M2P4 / M2C2 | `FMT=chat MX=1 python lssa_runtime/moe_arm2.py` |

## 6. Verification protocol and commitment

| claim | evidence | re-run |
|---|---|---|
| Manifest binds constants by SHA-256 per tensor with a domain-tagged preimage; the engine refuses digest, TP, context and contract-version mismatches | `lssa_runtime/lockstep_manifest.py`, `lockstep_config.py`; SPEC 7.19-7.20 (4 of 4 refusals) | `lockstep prepare`, `lockstep selftest` |
| The fingerprint (256-token probe, bf16 log-prob bits + 8 greedy ids) reproduces on a live server | `lssa_runtime/lockstep_selftest.py`; `results/lockstep_plugin_gate_final.json` | `lockstep selftest <manifest> --url` |
| BLAKE3 leaf/tree commitment over q/k/v/o rows is exact and measured | `lssa_runtime/t16k_blake3.cu`; SPEC (T16l): 48 rows byte-identical to a CPU recompute, BLAKE3 == PyPI library 8/8 | `t16l_egates.py` (C) |
| Sampling detection probability | **theorem** `detection_certain_when_sample_exceeds_valid_rows` (`formal/Lockstep/Evidence/Audit.lean`); paper Proposition 9.2 | `lake build` |
| NOT claimed: live commitment on the serving path, signatures, a semantic root, an interpreter-as-spec | `docs/PLAN_4.md` (Stage IV) | - |

## 7. Known open items that bound the claims

- GPT-OSS attention is stock FlashAttention (head dim 64, sliding window, sinks are outside the rule); its MoE layers are on the contract. PLAN_4 S7.
- The GPT-OSS exact-MXFP4 engine rows regressed 18-22% between M2E6 and the merged tree with stock and the per-call MoE cost unchanged; bisected to commit `17c35bf` (the per-block step as one fmaf; M2E13); reverted to the separate multiply and add (same rounding); M2E14 re-gated identity (10/10, 5/5) and reads 190 / 508 / 1,480 tok/s, 110 / 391 ms, served 0.52x - within 3% of M2E6. Closed.
- The xarch26 record was regenerated on H100, RTX 4090 and RTX 5090 from the committed kit (2026-09-10, 0 of 538 differ); the author's earlier record and the v31 record are history.
- Single-stream cells are medians of three without confidence intervals; quality uses perplexity only.
- The Lean development covers the attention core, the 2.5 decoder bounds and the evidence checkers; the 2.6 norm and the MoE contract are not yet formalised.
- Conformance inputs must be shipped or generated by a version-stable stream: `torch.randn` with one seed returns different values on torch 2.11 (x86) and 2.14 (arm64). The contract vectors now use numpy's PCG64; the cross-architecture kit ships its `inputs.npz` by sha256 and must not be regenerated with torch on another host (its own README says so; the two stale records in the private history were partly this).
