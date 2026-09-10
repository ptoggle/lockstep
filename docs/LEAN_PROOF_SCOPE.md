# Lean proof scope

## Short answer

The Lean development proves internal properties of the formal Lockstep model: validator soundness, bit-vector rescaling laws, integer bounds, ordered-list refinements, serialization round trips, combinatorial audit counts, and logical index reconstruction.

It does **not** prove that the Python, Torch, CUDA, compiler, GPU, hash implementations, or deployed server implement that model. Those links remain executable conformance checks and explicit assumptions.

The accurate assurance statement is:

> For values satisfying the formal premises, the Lean definitions have the stated bounds, round-trip properties, and refinement equalities. The runtime gates separately test that generated artifacts and selected implementation executions agree with those definitions.

## Trust and interpretation

The project is pinned to Lean 4.33.1 by `lean-toolchain` and uses Lean's standard library. All declarations in `formal/**/*.lean` build without `sorry`, `admit`, or project-defined axioms.

That absence of admissions does not enlarge a theorem's statement. Several results are conditional on predicates that already encode important obligations, such as canonical ordering or nearest-grid behavior. The sections below state those premises explicitly.

The formal development has four kinds of result:

1. **Bit-vector and arithmetic proofs.** These derive properties from exact `BitVec`, `Int`, `Nat`, and list definitions.
2. **Conditional refinement proofs.** These prove equality after a schedule or partition has been shown to reconstruct the canonical order.
3. **Executable finite witnesses.** These certify concrete examples by kernel reduction or a proof-producing decision procedure.
4. **Bridge artifacts.** Lean exports deterministic data; Python independently recomputes corresponding values and compares them. This comparison is a test, not a Lean theorem about Python.

## Proven by module

### Contract admission

File: `formal/Lockstep/Contract/Domain.lean`

Lean defines `ContractParams`, `ModelConfig`, `ParamsValid`, `ModelValid`, `Admitted`, `RejectReason`, `validateParams`, and `validateModel`.

Proved:

- `canonicalParams_valid`: the checked-in canonical constants satisfy `ParamsValid`.
- `validateParams_sound`: `validateParams p = .ok ()` implies `ParamsValid p`.
- `validateModel_sound`: `validateModel p m = .ok ()` implies `Admitted p m`.
- `admitted_head_group_divides`: admitted models satisfy KV-head/query-head and tensor-parallel/query-head divisibility.
- `admitted_lssa_constants`: admission fixes head dimension 128, key block 128, 32 blocks per segment, score bits 26, index bits 7, and maximum table weight 255.

Boundary:

- This proves soundness of the Lean validator. It does not prove that the production manifest parser invokes an equivalent validator on every path.
- `Admitted` is the static envelope represented in this module, not a formalization of every dynamic tensor or request invariant.

### Extended model-family admission

File: `formal/Lockstep/Decoder/Families.lean`

Lean defines absolute-coordinate sliding-window ranges and a family certificate over
grouped-query or multihead-latent attention and dense or sparse-expert feed-forward blocks.

Proved:

- `visible_range_full` and `visible_range_finite`: finite windows select an absolute suffix
  and do not renumber cache positions.
- `check_family_contract_sound`: acceptance by the executable Boolean checker implies every
  declared family invariant, including positive window and MLA dimensions and
  `1 <= topK <= experts`.
- `mixtral_contract_accepts` and `deepseek_contract_accepts`: representative MoE/GQA and
  MoE/MLA profiles satisfy the family-level shape contract.
- `zero_window_rejected`: a zero-width sliding window fails admission.

Boundary:

- `lssa_runtime/lockstep_families.py` implements deterministic expert routing and dispatch,
  route-order integer combination, and exact-integer MLA expansion. Lean proves admission
  of the surrounding shapes; it does not prove those Python algorithms equivalent.
- Family admission is not a production serving claim. The current vLLM adapter executes
  finite sliding-window GQA, while MoE grouped GEMM and MLA attention remain fail-closed at
  the serving envelope until their kernel adapters and end-to-end gates exist.

### Fixed-width numeric semantics

Files:

- `formal/Lockstep/Numeric/Bits.lean`
- `formal/Lockstep/Numeric/Binary32.lean`
- `formal/Lockstep/Attention/Concrete.lean`

Lean models binary32 values as `BitVec 32` and bfloat16 values as `BitVec 16`. It defines
field extraction and packing, finite exact-dyadic decoding, rational round-to-nearest-even,
integer/power-of-two conversion, binary32 addition and division, exponent-field rescaling,
bfloat16 conversion, and a concrete LSSA evaluator from quantized q/k/v rows to 128
bfloat16 output words.

Proved:

- `pack_fields` and `pack_roundtrip`: field extraction and reconstruction agree for all 32-bit words.
- `rescale_zero`: either IEEE zero encoding is normalized to positive zero.
- `rescale_flushes_small_exponent`: a finite value whose biased exponent does not survive subtraction is flushed to positive zero.
- `rescale_preserves_significand`: a surviving normal value keeps its sign and fraction bits.
- `rescale_subtracts_exponent`: a surviving normal value's biased exponent is reduced by the requested distance.
- `rescale_id`: distance zero is the identity on normal finite values.
- `int8_contract_range_fits` and `int32_score_range_fits`: the declared ranges fit their signed word widths.
- `int_block_partial_exact_binary32` and `int_block_weight_exact_binary32`: the LSSA block bounds are below the integer-exact binary32 threshold.
- `scaled_one`, `add_one_one`, `divide_one_two`, and the bfloat16 tie vectors are
  kernel-reduced checks of the concrete arithmetic.
- `single_token_row_vector` checks the complete quantized-input-to-bfloat16 evaluator.

Boundary:

- `rescaleBits`, `addRNE`, `divideRNE`, and `toBF16RNE` are executable bit-level definitions;
  they do not delegate evaluation to host floating-point arithmetic.
- Non-finite inputs are outside the admitted LSSA fold domain. The concrete add and divide
  functions return a canonical NaN for unsupported non-finite combinations.
- The concrete evaluator starts from quantized integer q/k/v rows. Source binary32 tensor
  exponent selection and quantization remain a separate implementation boundary.
- Arbitrary binary32 multiplication is not modeled; LSSA term scaling is represented
  directly as exact integer-times-power-of-two conversion.

### Integer accumulation bounds

Files:

- `formal/Lockstep/Numeric/Bounds.lean`
- `formal/Lockstep/Attention/LSSA.lean`

Proved:

- `sum_natAbs_le_length_mul`: a bounded signed list has sum magnitude at most length times the term bound.
- `sublist_sum_safe`: the same reasoning bounds every sublist reduction.
- `score_abs_le`: at most 128 score terms of magnitude at most `127 * 127` produce magnitude at most 2,064,512.
- `block_value_abs_le`: at most 128 products of an unsigned weight at most 255 and a signed value of magnitude at most 127 produce magnitude at most 4,145,280.
- `block_weight_le`: the block weight sum is at most 32,640.
- `all_partial_sums_safe`: every sublist partial of a valid weighted block remains below 4,145,280.

Consequences inside the formal model:

- The score and block sums fit signed int32.
- Block value and weight integers are exactly representable in binary32.
- Reordering or reassociating these exact integer additions cannot introduce overflow within the proved envelope.

Boundary:

- The score theorem receives a list of already formed product terms. It does not formalize the CUDA WGMMA instruction that produces those terms.

### Quantization ranges and T2

File: `formal/Lockstep/Attention/Table.lean`

Proved:

- `quantized_in_int8_range`: `clampInt8` returns a value in `[-127, 127]`.
- `quantized_exponent_in_range` and `selected_exponent_proxy_in_range`: the formal exponent clamp returns a value in `[-32, 40]`.
- `score_scale_in_range`: the score-scale clamp returns a value in `[-7, 23]`.
- `t2LiveValues_size`, `t2_live_length`, and `t2_length`: the exact checked-in nonzero prefix has 1,152 entries and the structural table has 2,176 entries.
- `table_index_in_bounds`: the clamped table index is always below 2,176.
- `t2_zero_tail`: every entry from index 1,152 onward is zero.
- `t2_anchor_values`: `T2[0] = 255`, `T2[128] = 128`, `T2[1151] = 1`, and `T2[1152] = 0`.
- `half_up_tie_value` and `trunc_tie_value`: the exact tie example distinguishes half-up rounding from truncation.

Lean also defines `t2Csv`, a deterministic comma-separated serialization of all 2,176 entries.

Boundary:

- The 1,152 live entries are checked-in constants. Lean does not derive the transcendental values from a formal exponential function.
- `t2Digest` is a declared string. Lean does not implement SHA-256 or prove internally that hashing `t2Csv` yields that string.
- `lssa_runtime/verify_lean_artifacts.py` independently regenerates the Decimal table, compares the complete CSV, and computes the digest. That is the runtime bridge test for this boundary.

### Canonical LSSA evaluator

File: `formal/Lockstep/Attention/LSSA.lean`

Lean defines score terms, weighted block terms, block partials, fold state, segment and row
states, and a committed block transition. Each segment is evaluated from a fresh state and
its completed state is merged into the row by the separate ordered outer recurrence. The
canonical evaluator constructs consecutive 32-block segments.

The rounded boundary is represented by `RoundOps`, which supplies deterministic operations for:

- zero classification;
- conversion from scaled signed and unsigned integers;
- addition;
- exponent-field rescaling;
- division.

Proved:

- `two_level_fold_is_not_flat`: an executable rounded witness distinguishes independent
  segment evaluation and outer merging from a flattened block fold.
- `lssa_deterministic`: the canonical evaluation relation has at most one output.

Boundary:

- The structural refinement theorems remain parametric over `RoundOps`.
  `Lockstep.Attention.Concrete.binary32RoundOps` instantiates the interface with executable
  bit-level addition, division, conversion, rescaling, and zero classification.
- `lssa_deterministic` proves uniqueness relative to the formal evaluator; equivalence of
  Python, Torch, and CUDA implementations to that evaluator remains an executable bridge.
- The structural evaluator accepts materialized block metadata. `evalQuantizedRow` supplies
  the executable score, table-weight, block-partial, two-level-fold, and bfloat16 path from
  quantized q/k/v inputs.

### Scheduling, chunking, and split-KV

File: `formal/Lockstep/Attention/Schedule.lean`

Proved:

- `block_reorder_invariant`: if scheduled and canonical blocks have equal metadata and permutation-equivalent term lists, their exact integer partials are equal.
- `scheduled_refines_canonical`: a block-by-block `ScheduleRefines` derivation preserves the canonical formal row state.
- `aligned_chunk_blocks_equal`: an `AlignedChunks` witness exposes that its flattened block list equals the canonical list.
- `aligned_chunk_refines_canonical`: aligned chunks evaluate to the canonical row when their flattening is the canonical block list.
- `segment_assignment_complete_disjoint`: a valid indexed segment assignment preserves the
  canonical nested segment values and has no duplicate global indices.
- `splitKV_refines_canonical`: independently evaluated ordered segment groups produce the
  canonical two-level formal row state.

Important premise strength:

- `AlignedChunks chunks canonical` is defined as `chunks.flatten = canonical`.
- `SegmentAssignment.Valid` requires the flattened global indices to equal `List.range total`
  and the nested block values to equal the canonical semantic segments.
- Therefore these theorems prove permitted intra-block permutation and work assignment are
  inert after canonical block and segment reconstruction has been established. They do not
  yet prove that an arbitrary chunker, work-list builder, or CUDA kernel satisfies those
  validity predicates.

### Certified prohibited-substitution witnesses

File: `formal/Lockstep/Attention/Negative.lean`

Kernel-checked witnesses establish that:

- `descending_fold_counterexample`: reversing a small rounded fold can change its result.
- `nonaligned_chunk_counterexample`: independently normalizing a split weighted mean can change the result.
- `table_truncation_counterexample`: truncating the live table before index 1,151 changes a published nonzero entry.
- `approx_division_counterexample`: exact natural-number division by 3 differs from a deliberate power-of-two denominator approximation.
- `multiply_rescale_counterexample`: rescaling only the old accumulator differs from scaling the old and fresh terms together.

File: `formal/Lockstep/Evidence/Manifest.lean`

- `tampered_manifest_counterexample`: manifests differing in the T2 digest do not validate as equal.

Boundary:

- Except for the actual T2 entry and manifest record, these are small natural-number models of the prohibited transformations. They prove existence of observable differences in those models.
- They are not formal IEEE-754 or CUDA executions. The repository's CPU and GPU negative gates exercise the production binary32 and kernel paths separately.

### Decoder arithmetic and R4 indexing

File: `formal/Lockstep/Decoder/Operators.lean`

Proved:

- `activation_quantization_range` and `weight_quantization_range`: the formal int8 clamp stays in `[-127, 127]`.
- `linear_acc_abs_lt`: at most `2^15` int8-product terms produce magnitude below `2^29`.
- `rmsnorm_sum_lt_int64`: a square sum below `2^45` plus an epsilon term at most `2^50` remains below `2^63`.
- `head_recomposition_lt_int64`: four magnitude-bounded head partials at shifts 14, 7, 7, and 0 remain below `2^63`.
- `r4_pairs_complete_disjoint`: a lane in `[0, 256)` is reconstructed uniquely from its 128-wide pair index and upper-lane bit.
- `decoder_static_safety`: the closed numerical inequalities underlying those bounds hold.

Boundary:

- The RMSNorm and head results are conditional on their stated input caps.
- These theorems do not model the complete RMSNorm lookup algorithm, RoPE arithmetic, SiLU table program, or fused CUDA epilogues.

### Tensor-parallel ordered reduction

File: `formal/Lockstep/Decoder/TensorParallel.lean`

Lean defines an abstract `RankRoundOps`, an ordered rank accumulator, and a final rounding operation.

Proved:

- `rank_partition_complete_disjoint`: a valid indexed rank schedule has canonical values and no duplicate rank indices.
- `rank_partition_refines_ordered_fold`: a valid scheduled rank list produces the same result as the canonical rank-ordered list.

Important premise strength:

- `RankPartition.Valid` requires scheduled indices to equal `List.range worldSize` and scheduled values to equal the canonical list.
- The theorem proves preservation after ordered reconstruction. It does not prove an NCCL or custom collective implements that reconstruction.

### Transcript serialization

File: `formal/Lockstep/Evidence/Transcript.lean`

Proved:

- `word_encode_decode`: four-byte little-endian encoding of any `BitVec 32` decodes to the same word.
- `words_encode_decode`: the property lifts to lists of words.
- `row_encode_decode`: the formal row header and payload round-trip.
- `row_encoding_injective`: equal formal row encodings imply equal row records.

Boundary:

- A signed 32-bit value is represented by its 32-bit word, so the byte theorem preserves the bit pattern rather than proving a separate signed arithmetic conversion.
- The formal row format contains a magic word, layer, row, head, and word payload. It is not a proof about every field and byte in the production q/k/v/o leaf packer.

### Authenticated trace checking

Files:

- `formal/Lockstep/Evidence/Trace.lean`
- `lssa_runtime/lockstep_trace.py`

Proved:

- `check_chain_sound`: every row accepted by the executable formal checker has the canonical sequence number, expected previous digest, and digest of the canonical `rowEncode` bytes.
- `check_trace_sound`: acceptance binds the checked chain to the expected deployment context and externally committed root.

Executable bridge:

- The runtime format is a strict, length-delimited little-endian stream. HMAC-SHA256 binds the raw manifest digest, nonce, sequence number, previous authenticator, and exact row bytes.
- `TraceWriter` supports streaming production; `verify_trace` rejects wrong keys, contexts, committed roots, sequence numbers, authenticators, truncation, and trailing bytes.
- `verify_rows.py` emits this trace when `LOCKSTEP_TRACE_OUT` is set, preserving each actual 16-bit output word in the formal 32-bit payload container.

Boundary:

- Lean treats the hash primitive as an abstract function. It proves checker soundness relative to that function, not HMAC-SHA256 collision resistance, key secrecy, or correctness of Python's cryptographic library.
- The deployment must commit the reported root before revealing the trace or authentication key; that protocol path is modeled separately.

### Typed tensor preimages and manifests

File: `formal/Lockstep/Evidence/Manifest.lean`

Proved:

- `tensor_header_separates_dtype`: distinct dtype strings produce distinct formal headers.
- `tensor_header_separates_shape`: distinct shape lists produce distinct formal headers.
- `manifest_validation_sound`: if the exact-record validator accepts, the candidate manifest equals the expected manifest.
- `manifest_tamper_rejected`: replacing the T2 digest with a different string causes rejection.

Boundary:

- `tensorPreimage` is a typed sequence of domain, dtype, shape, separator, and byte atoms. It establishes logical domain separation.
- It is not a byte-level formalization of the production ASCII header or `lockstep_manifest.py`.
- `validateManifest` is exact structure equality. The theorem does not establish equivalence with every production manifest validation branch.
- Hash collision resistance and binding are assumptions, not Lean theorems.

### Sampled auditing

File: `formal/Lockstep/Evidence/Audit.lean`

Lean defines finite row-index samples, a no-duplicates validity predicate, clean-sample counts, canonical Merkle path folding, exact row openings, and a batch opening checker.

Proved:

- `choose_zero_of_lt`: `choose n k = 0` when `n < k`.
- `clean_sample_count`: the modeled count of all-clean samples is `choose (totalRows - badRows) sampleSize`.
- `detection_count`: the modeled count of detecting samples is total samples minus clean samples.
- `detection_certain_when_sample_exceeds_valid_rows`: if the sample size exceeds the number of valid rows, the clean count is zero and the detecting count equals the total count.
- `opened_difference_is_nonconformance`: a formal opened row unequal to the expected row causes the exact comparison predicate to return false.
- `check_opening_sound`: an accepted opening has the challenged index, exact replayed row, canonical path orientation, and committed root.
- `check_openings_sound`: the result lifts to the complete ordered challenge batch.

Executable bridge:

- `lockstep_audit.py` creates a canonical commitment binding the raw manifest, request, trace chain root, Merkle root, row count, trace nonce, and commitment nonce.
- A later beacon value and round derive an unbiased deterministic sample without replacement. The challenge digest binds the prior commitment and beacon statement.
- The opener returns only challenged rows and Merkle paths. The verifier independently re-derives the sample, checks request and manifest binding, enforces exact opening order and count, compares each row with reference replay, and verifies every path.

Boundary:

- The combinatorial count model does not define a probability measure; the runtime uses an executable rejection sampler and sparse partial Fisher-Yates selection.
- Lean treats Merkle leaf and node hashes as abstract functions. SHA-256 collision resistance and the external beacon's unpredictability, authenticity, and publication after the commitment remain deployment assumptions.

### Local approximation bounds

File: `formal/Lockstep/Analysis/Approximation.lean`

Proved:

- `quantization_error_le_half_step`: the formal half-step inequality follows from the `RoundsToNearestGrid` premise.
- `weighted_average_value_error`: bounded per-value errors produce a weighted-numerator error at most total weight times the error bound.
- `normalized_weight_perturbation_bound`: bounded weight errors multiplied by bounded values produce the stated length-times-product numerator bound.
- `local_attention_error_bound`: value and weight contributions compose through the triangle inequality.
- `local_attention_error_zero`: zero bounds on both contributions imply zero combined magnitude.
- `fidelity_certificate_sound`: if the executable certificate reconstructs the observed signed error, checks each component bound, and checks their sum against policy, then observed error is within policy.

Executable bridge:

- `lssa_fidelity.py` accepts a concrete float32 q/k/v NPZ row, reruns the filed LSSA-B8 golden including bfloat16 output rounding, and compares it with a stable float64 softmax reference.
- The certificate binds all typed input arrays by SHA-256, pins the T2 digest and LSSA constants, records exact bfloat16 output words, decomposes error into value-grid and weight/fold components, and applies an explicit maximum-error policy.
- Verification replays the entire calculation and requires exact equality with the canonical JSON claim; altered metrics or input tensors are rejected.

Important premise strength:

- `RoundsToNearestGrid x q step` is itself defined as the half-step inequality. The first theorem does not derive nearest-grid behavior from the production quantizer.
- The weighted theorems are conditional list inequalities over error terms. `lssa_fidelity.py` supplies concrete measured components for an individual row; Lean does not prove that Python's float64 softmax or NumPy implements real-number softmax.
- Certificates are row-local measurements, not perplexity, downstream accuracy, or a whole-model fidelity theorem.

### Logical GPU work coordinates

File: `formal/Lockstep/GPU/Obligations.lean`

Proved:

- `coordinate_complete`: quotient/remainder coordinates reconstruct a linear index.
- `coordinate_disjoint`: equal coordinates imply equal original indices.
- `score_term_coverage`: the construction is instantiated for 128-term score blocks.
- `segment_block_coverage`: it is instantiated for 32-block segments.
- `row_head_channel_coverage` and `row_head_channel_disjoint`: nested row/head/channel coordinates reconstruct indices and are injective.
- `cta_segment_coverage` and `cta_segment_disjoint`: CTA/segment quotient-remainder coordinates reconstruct indices and are injective.
- The logical-obligation and hardware-assumption inventories are nonempty.

Boundary:

- `GridShape` records rows, heads, blocks, segments, channels, and CTAs, but the coordinate theorems are generic arithmetic identities rather than a proof about a concrete kernel launch.
- Lean does not prove that a CUDA grid, work queue, memory address calculation, synchronization primitive, or compiled kernel uses these coordinates.
- WGMMA integer behavior, PTX/SASS binary32 behavior, compiler preservation, and the CUDA memory model remain explicit assumptions in the generated artifact.

## Generated artifact and bridge

Files:

- `formal/Lockstep/Artifacts.lean`
- `formal/Main.lean`
- `results/lean_contract_artifacts.json`
- `lssa_runtime/verify_lean_artifacts.py`

Lean deterministically exports:

- canonical contract constants;
- integer safety bounds;
- the complete T2 CSV, structural lengths, digest string, and anchor values;
- bit-level rescale vectors;
- negative-control witness outputs;
- a little-endian transcript vector;
- a sampled-audit count example;
- logical GPU obligations and hardware assumptions.

Verified behavior:

- Two final exports compare byte-for-byte equal.
- The Python bridge independently reconstructs the expected JSON object and accepts the checked-in artifact.
- Its `--negative` path changes the T2 digest and confirms that the same comparison rejects the mutation.

Boundary:

- This bridge establishes agreement for the exported finite values. It is not a proof of semantic equivalence between Lean and the full Python/Torch/CUDA programs.
- Python's Decimal and SHA-256 implementations are outside Lean's trusted kernel.

## End-to-end claim that is not proved

The repository does not contain a Lean theorem of the form

```text
production_server(request, manifest) = formal_lockstep(request, manifest)
```

for all admitted requests.

Establishing that statement would additionally require formal semantics and refinement proofs for at least:

- the complete tensor-to-block quantization pipeline;
- correctly rounded binary32 add, multiply, divide, and conversions;
- all non-attention operators, including RMSNorm, RoPE, SiLU, and fused epilogues;
- Python and Torch reference implementations;
- CUDA C++, PTX, SASS, synchronization, and memory behavior;
- compiler lowering and linked libraries;
- manifest parsing, tensor hashing, and server refusal paths;
- the connection between deployed request state and authenticated transcript bytes.

The current repository addresses those links with exact runtime comparisons, CPU self-tests, contract vectors, row replay, negative controls, and GPU gates. Those are valuable implementation evidence, but they remain tests rather than universal Lean proofs.

## Reproduction

From the repository root:

```sh
lake build
lake exe lockstep-formal check
lake exe lockstep-formal export results/lean_contract_artifacts.json
python3 lssa_runtime/verify_lean_artifacts.py
python3 lssa_runtime/verify_lean_artifacts.py --negative
```

Relevant implementation checks, outside the Lean proof kernel:

```sh
uv run --with numpy --with torch python lssa_runtime/lssab8_selftest.py
uv run python lssa_runtime/contract_vectors.py --check results/contract_vectors.json
```

The plan and theorem inventory are in `docs/LEAN_FORMALIZATION_PLAN.md`. The manuscript's higher-level proof-to-artifact table is in Section 8.2 of `paper/main.tex`.
