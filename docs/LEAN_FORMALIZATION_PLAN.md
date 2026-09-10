# Lockstep Lean Formalization Plan

Status: implementation plan for contract v2.4 and LSSA-B8.

The implemented theorem statements, premise strength, and unproved boundaries are documented in
[`LEAN_PROOF_SCOPE.md`](LEAN_PROOF_SCOPE.md).

## Objective

Build a public, machine-checked Lean 4 model of the Lockstep contract that:

1. defines the admitted input domain and bit-level arithmetic semantics;
2. proves the LSSA-B8 safety, refinement, chunking, and split-KV claims;
3. certifies witnesses for prohibited arithmetic and scheduling substitutions;
4. proves safety properties for the non-attention decoder operators and ordered tensor-parallel reduction;
5. proves canonical transcript and manifest validation properties;
6. proves the post-commitment sampling probability theorem;
7. provides a useful local approximation-error bound;
8. generates machine-readable constants and contract vectors consumed by runtime gates;
9. records GPU layout, partition, and schedule obligations separately from unproved hardware semantics; and
10. checks family-level invariants for sliding-window, sparse-expert, and multihead-latent decoders.

The formal model is the root oracle for generated formal artifacts. Python, Torch, and CUDA remain independently implemented consumers checked against those artifacts. A Lean proof does not by itself prove the production binaries correct.

## Toolchain and trust boundary

- Lean 4 is pinned by `lean-toolchain`.
- The project uses Lake and only Lean's standard library. This keeps the dependency and trust surface small.
- Fixed-width fields use `BitVec`, `UInt8`, `UInt16`, and `UInt32` semantics.
- The formal arithmetic layer models the contract operations needed by the proofs. It does not delegate a theorem to host floating-point execution.
- Concrete finite obligations are discharged by kernel reduction or proof-producing decision procedures; the tree must contain no `sorry` or `admit`.
- Cryptographic results assume collision resistance and binding. They do not prove SHA-256 or BLAKE3 cryptographic security.
- GPU results prove logical layout and work-partition obligations. PTX/SASS instruction semantics, compiler correctness, and hardware conformance remain explicit assumptions backed by existing gates.

## Formal model boundary

### Admitted domain

`Lockstep.Contract.Domain` defines contract parameters and validates:

- `headDim = 128`, `keyBlock = 128`, `segmentBlocks = 32`;
- `scoreBits = 26`, `indexBits = 7`, `weightTop = 255`;
- exponent ranges `[-32, 40]` and the clamped score-scale range `[-7, 23]`;
- nonzero head counts, grouped-query divisibility, tensor-parallel world size, and positive dimensions;
- model features admitted by the dense Llama-like v2.4 envelope;
- explicit rejection reasons for invalid configurations.

### Numeric semantics

`Lockstep.Numeric` defines:

- signed integer ranges and exact accumulation bounds;
- raw binary32 and bfloat16 bit encodings;
- normal, zero, subnormal, infinity, and NaN classification;
- exponent-field rescaling with the contract's positive-zero flush rule;
- round-to-nearest-even integer-to-bfloat16 conversion for the finite range used by artifacts;
- exact power-of-two scale metadata used by the LSSA recurrence.

The model keeps rounded additions and divisions abstract only where no algebraic property is required beyond determinism and an explicit operation order. Concrete bridge vectors use exact representable integer and power-of-two cases. The paper must retain the platform assumption for general correctly rounded binary32 add, multiply, and divide.

## Work packages and theorem inventory

### WP1 — Contract domain

Deliverables:

- `ContractParams`, `ModelConfig`, `Admitted`, and `RejectReason`;
- executable validation corresponding to manifest refusal rules;
- proofs that validated parameters imply every static LSSA dimension and exponent invariant.

Principal theorems:

- `validateParams_sound`;
- `validateModel_sound`;
- `admitted_head_group_divides`;
- `admitted_lssa_constants`.

### WP2 — Bit-level numeric semantics

Deliverables:

- binary32 field extraction and construction;
- contract `rescaleBits`;
- signed fixed-width range predicates;
- exact integer conversion lemmas used by block partials.

Principal theorems:

- `rescale_zero`;
- `rescale_flushes_small_exponent`;
- `rescale_preserves_significand`;
- `rescale_subtracts_exponent`;
- `rescale_id`;
- `int_block_partial_exact_binary32`.

### WP3 — Quantization and T2

Deliverables:

- exponent selection over a finite integer proxy domain;
- int8 quantization and clamp range proof;
- T2 structural model, including live and zero regions;
- deterministic serialization of the structural table artifact.

The transcendental table values remain the published digest-pinned constants. Lean proves all structural properties used by LSSA and validates the published anchor entries and zero tail.

Principal theorems:

- `quantized_in_int8_range`;
- `score_scale_in_range`;
- `table_index_in_bounds`;
- `t2_zero_tail`;
- `t2_anchor_values`.

### WP4 — Canonical LSSA evaluator

Deliverables:

- score, weight, block partial, block state, segment state, and row state definitions;
- canonical ascending two-level fold;
- a scheduled evaluator parameterized by legal within-block term order and legal segment work partition.

Principal theorems:

- `score_abs_le`;
- `block_value_abs_le`;
- `block_weight_le`;
- `all_partial_sums_safe`;
- `block_reorder_invariant`;
- `scheduled_refines_canonical`;
- `lssa_deterministic`.

### WP5 — Chunking and split-KV

Deliverables:

- absolute key-block partitioning;
- aligned prompt chunk partitions;
- arbitrary legal segment-to-CTA assignment;
- independent segment construction followed by the committed row fold.

Principal theorems:

- `aligned_chunk_blocks_equal`;
- `aligned_chunk_refines_canonical`;
- `segment_assignment_complete_disjoint`;
- `splitKV_refines_canonical`.

### WP6 — Certified negative controls

Deliverables are kernel-checked witnesses that:

- ascending and descending rounded folds differ;
- aligned and non-aligned block construction can differ;
- half-up and truncating table rules differ;
- correctly rounded and deliberately approximate division can differ;
- exponent-field and multiplication/FTZ rescaling can differ;
- manifest tampering is rejected.

Principal theorems:

- `descending_fold_counterexample`;
- `nonaligned_chunk_counterexample`;
- `table_truncation_counterexample`;
- `approx_division_counterexample`;
- `multiply_rescale_counterexample`;
- `tampered_manifest_counterexample`.

### WP7 — Decoder and tensor parallelism

Deliverables:

- activation and weight quantization range proofs;
- int8 linear accumulator safety for `K ≤ 2^15`;
- RMSNorm sum-of-squares safety under its declared cap;
- two-plane head recomposition safety;
- R4 butterfly index and coverage proof;
- ordered rank-fold model.

Principal theorems:

- `linear_acc_abs_lt`;
- `rmsnorm_sum_lt_int64`;
- `head_recomposition_lt_int64`;
- `r4_pairs_complete_disjoint`;
- `rank_partition_refines_ordered_fold`;
- `decoder_static_safety`.

### WP8 — Transcript and manifest

Deliverables:

- canonical little-endian encoders for signed words and row transcripts;
- a decoder for the formal row header and payload;
- an authenticated trace chain with an executable sound checker;
- typed tensor digest preimage encoding;
- manifest field validation and fail-closed tamper model.

Principal theorems:

- `word_encode_decode`;
- `row_encode_decode`;
- `row_encoding_injective`;
- `check_chain_sound`;
- `check_trace_sound`;
- `tensor_header_separates_dtype`;
- `tensor_header_separates_shape`;
- `manifest_validation_sound`;
- `manifest_tamper_rejected`.

### WP9 — Audit protocol

Deliverables:

- finite index sets, samples without replacement, valid/invalid row predicates;
- count of clean samples and detection numerator;
- canonical Merkle openings and an executable batch checker;
- request/manifest/trace commitment before beacon-derived sampling;
- exact reference-row comparison for every opened index.

Principal theorems:

- `clean_sample_count`;
- `detection_count`;
- `detection_certain_when_sample_exceeds_valid_rows`;
- `opened_difference_is_nonconformance`.
- `check_opening_sound`;
- `check_openings_sound`.

### WP10 — Approximation bounds

Deliverables:

- scalar quantization reconstruction bound in an unclamped cell;
- weighted-average perturbation bounds;
- a compositional local bound separating value quantization and normalized-weight perturbation;
- an executable certificate reconstructing the observed error and enforcing a policy limit;
- a runtime row certificate binding source tensors, filed LSSA output, and softmax comparison.

Principal theorems:

- `quantization_error_le_half_step`;
- `weighted_average_value_error`;
- `normalized_weight_perturbation_bound`;
- `local_attention_error_bound`.
- `fidelity_certificate_sound`.

This is a local mathematical fidelity result, not a model-level perplexity or downstream-task guarantee.

### WP11 — Generated artifacts and runtime bridge

Deliverables:

- `lake exe lockstep-formal export` emits deterministic JSON containing contract constants, T2 structural anchors, certified witnesses, and formal bridge vectors;
- `results/lean_contract_artifacts.json` is checked in;
- `lssa_runtime/verify_lean_artifacts.py` recomputes corresponding runtime values and rejects divergence;
- the CPU gate runs the bridge verifier.

Acceptance:

- exported JSON is byte-stable across two runs;
- Python verification passes;
- mutating a checked field makes verification fail.

### WP12 — GPU obligations

Deliverables:

- a formal logical grid for rows, heads, blocks, segments, channels, and CTA assignments;
- proofs of complete/disjoint score-term and segment coverage;
- explicit assumptions for WGMMA integer semantics and PTX/SASS binary32 behavior;
- a machine-readable obligation report consumed by the artifact bridge.

This work does not claim full CUDA, compiler, or hardware verification.

### WP13 — Extended model families

Deliverables:

- absolute-coordinate suffix semantics for finite sliding windows;
- a decidable family certificate for GQA/MLA attention and dense/MoE feed-forward blocks;
- executable deterministic MoE routing, dispatch, and combine semantics;
- executable exact-integer MLA latent expansion with split-RoPE concatenation;
- manifest-carried family certificates, validated fail-closed.

Principal theorems:

- `visible_range_full`;
- `visible_range_finite`;
- `check_family_contract_sound`;
- `mixtral_contract_accepts`;
- `deepseek_contract_accepts`;
- `zero_window_rejected`.

This work extends the semantic and manifest contract. Production serving remains narrower:
finite-window GQA is wired through the portable B8 path; MoE grouped GEMM and MLA attention
remain refused until their CUDA/vLLM adapters pass model-level gates.

## Repository layout

```text
lean-toolchain
lakefile.toml
formal/
  Lockstep.lean
  Lockstep/
    Contract/Domain.lean
    Numeric/Bits.lean
    Numeric/Bounds.lean
    Attention/Table.lean
    Attention/LSSA.lean
    Attention/Schedule.lean
    Attention/Negative.lean
    Decoder/Operators.lean
    Decoder/TensorParallel.lean
    Decoder/Families.lean
    Evidence/Transcript.lean
    Evidence/Trace.lean
    Evidence/Manifest.lean
    Evidence/Audit.lean
    Runtime/Plan.lean
    Analysis/Approximation.lean
    GPU/Obligations.lean
    Artifacts.lean
  Main.lean
results/lean_contract_artifacts.json
lssa_runtime/verify_lean_artifacts.py
```

## Build and verification commands

```sh
lake build
lake exe lockstep-formal check
lake exe lockstep-formal export results/lean_contract_artifacts.json
python lssa_runtime/verify_lean_artifacts.py results/lean_contract_artifacts.json
python lssa_runtime/verify_lean_artifacts.py --negative results/lean_contract_artifacts.json
```

The complete verification also runs the existing CPU contract-vector and LSSA self-tests and recompiles the paper.

## Paper integration

The manuscript will:

- add a mechanized-semantics subsection and theorem-to-artifact table;
- distinguish proved contract properties from tested implementation agreement;
- state all formal assumptions, including binary32 platform behavior and hash binding;
- cite the exact Lean toolchain and checked artifact;
- add the formal files and commands to the reproduction map;
- avoid claiming formal CUDA correctness or model-level quality guarantees.

## Completion criteria

The plan is complete only when:

- every module above exists and builds;
- every named theorem has a checked Lean declaration with no `sorry` or `admit`;
- generated artifacts are deterministic and pass positive and mutation-negative runtime checks;
- existing relevant CPU gates still pass;
- the manuscript compiles without new diagnostics and its formalization pages are visually inspected;
- the current branch contains only intended source and generated artifacts; and
- the branch is committed, pushed, and represented by an updated or new pull request.
