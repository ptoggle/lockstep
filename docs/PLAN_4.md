# Stage IV: full semantics - the interpreter as the specification

Written 2026-09-10, revised the same day after an adversarial review (section 8 lists what the review changed).
Owner: the verification side (mixture-of-experts performance is with the performance owner). Status: PLAN, not started.
Every claim about the current tree was checked in the checkout on the day of writing (four survey reports, condensed
in section 1).

## 0. What "full semantics" means and why it is the next stage

The cryptographer's requirement: the output of inference must be a TOTAL FUNCTION of (committed model, input, declared
parameters) with every operation's rounding, order and tie rule pinned, in a form a third party can execute without our
engine and without trusting a vendor library. Verification of any kind - re-execution on other hardware, audit
sampling, refereed dispute, a zero-knowledge circuit, a TEE re-executor - checks a claim against that definition.
Without it the strongest statement we can make is "our GPU kernels agree with our torch references": reproducibility,
not semantics.

Where the field is (survey, 2024-2026): batch-invariant engines (Thinking Machines, SGLang, vLLM, EigenAI) define the
reference as "what the kernel computes" and hold within one GPU family; hardware-emulation verifiers (Hawkeye,
MMA-Sim) reverse-engineer vendor rounding; refereed and statistical schemes (Verde, TAO, DiFR) either need bitwise
reproducibility float stacks cannot give or accept tolerances that one-ULP faults evade (Chen 2026); zkML proves a
re-quantised integer model, not the served one. The reference-model pattern that works as a conformance specification
is TOSA's: one normative document with per-op pseudocode, an executable dependency-free reference, exact integer
semantics, hashed per-operator conformance vectors, declared limits. Nobody has that for a whole LLM with a declared
decode loop. We are close: the operators are integer-first and gated bit-identical, the references are CPU-capable,
the manifest exists.

## 1. Where the tree is today (the survey, condensed)

**Have.** CPU-capable references for every operator: quantisers, the v2.1 linear epilogue, the outlier side path,
R1 / R4 rotations, the 2.5 integer norm and the 2.6 float norm (with a numpy twin), Q14 RoPE, the table SiLU, the
lm_head plane variants, attention LSSA-B / B8 / B9.x as torch references AND independent numpy / python-int goldens,
MoE v3 + addendum + item 5, exact MXFP4 dequant. Engine identity gates per op and per token (E1 / E4 / E6 / F3 /
K1-K10), 128-aligned chunk enforcement, split exactness, cross-arch digests. A manifest with per-tensor SHA-256, the
constants export, `verify_constants.py` (recomputes every constant from the public checkpoint), a probe fingerprint.
A §16 integer sampler (`contract_runner.py`: xorshift64, integer top-k) from an older contract tier.

**Missing.** (1) No composition: nothing on CPU takes a manifest and a prompt and produces token ids. (2) The decode
loop is vLLM's: no argmax tie rule, no sampling declaration, no log-softmax definition (today vLLM's fp32 `log_softmax`,
not portable). (3) Token ROLE (prompt row -> B8, generated row -> B) comes from vLLM's `is_prefilling`: after a
preemption a generated token is re-prefilled and changes rule, so the function depends on memory pressure; the block
exponent of a partially filled boundary block is fixed at first write and the append rule for generated keys is not
written. (4) The manifest has no root hash, no signature, no tokenizer or chat-template hash, no contract-document
hash; `contract_version` comes from an environment variable at import; `outliers` / `smoothing` are written but not
validated; `lockstep_git` may be "unknown". (5) A running server declares nothing over HTTP. (6) The runtime BLAKE3 is
keyed with one hardcoded key: a MAC, not a signature, and the same for every deployment. (7) Conformance vectors cover
the non-attention operators only. (8) GPT-OSS attention has nothing: sinks and the 128 window are outside the rule,
head dim 64 is refused, the score bound was derived for dh 128. (9) No standalone §9 TP reduction reference. (10) The
head epilogue order in `t16h_lmvariants` (`(acc * asc) * ws`) and the §2.6 declaration (`(acc * ws) * asc`) disagree.
(11) The Lean formalisation  pins the 2.5 norm. (12) Prefix caching: banned by plan text, "fine" in the runbook.
(13) Behaviour on NaN / Inf is an assertion in the goldens and a clamp in the quantiser, not a rule.

## 2. The target

`lockstep_interp` computes

    F(root, world_size, prompt_ids, params) -> (token_ids, finish_reason, logit_digests)

where `root` commits to everything in section 5, `world_size` is an explicit argument (rank slicing and the §9
reduction are part of the function), `params` = {max_tokens, stop_ids, eos_ids, mode: greedy | sampled(seed, top_k,
temperature_q)} under the declared decode loop, the model graph comes from a machine-readable IR derived from the
manifest, and every arithmetic step is an integer operation of declared width or a named IEEE binary32 operation at a
stated site. Log-probabilities are NOT part of F (their log-softmax is undeclared); F returns per-step digests of the
bf16 logits row so a claim can still be checked at the logit level. Tokenizer and chat template are identifiers in the
commitment, not part of F: F is on ids.

Two implementations of one IR: v0 (torch on CPU, composed from today's references: the bootstrap oracle) and v1 (numpy
+ python integers, no torch: the reference in the trusted base). Precedence clause: on any conflict between prose and
the v1 source, the v1 source is normative and the prose is corrected. The IR declares limits (max context, max batch,
vocab, head dim, window) TOSA-style.

Non-goals: interpreter throughput beyond audit use (an 8B 512-token replay in under an hour on a workstation CPU);
kernel proofs (S6 states the semantics so a proof or a circuit can target it); sampling beyond the declared integer
sampler; per-token dispute trees (roadmap 2.11) - out of the semantic root, into a separate deployment attestation.

## 3. Stages

Each stage names deliverables, the gate that closes it, the kill line, and how it is parallelised (agents for
surveys, independent modules and adversarial review; every merge gated by a human-read result). One owner plus agents.

### S0 - Semantics inventory, the IR, the missing clauses (weeks 1-2)

`docs/contract/SEMANTICS_v1.md`: the index of every section a total function needs. Existing sections by reference;
the missing ones written as rules, each with a first conformance vector:
- decode loop: token role = ORIGIN (a token generated by the model is a generated row forever; a preemption /
  recompute must reproduce it under the B rule or preemption is banned in the manifest); the prompt / generation
  boundary; the 128-aligned chunk clause; block exponent fixed at first write and the saturating append rule for
  generated keys; stop ids, eos ids, max_tokens, finish_reason; argmax tie = lowest index;
- the §16 integer sampler re-declared: integer logit scaling stated, `top_k` tie rule, the EXP_LUT generated by a
  declared integer recurrence (not `math.exp`) with its digest, temperature as an integer parameter or removed, seed
  derivation per request, the modulo bias stated;
- the head: the epilogue order pinned (decide t16h's or §2.6's, gate the other), padded vocab slice, bf16 cast site;
- embedding lookup and the residual stream dtype at every site; max context and position binding (YaRN / longrope);
- totality: a NaN / Inf argument per site (bounded integers from bf16 weights make NaN unreachable, or explicit
  propagation rules), the quantiser's clamp as a rule, overflow bounds as declared widths;
- invariance clauses: batch composition, split, graph mode, TP rank order; prefix caching decided with an argument
  (full-block hits under the aligned rule are plausibly exact: prove or ban; the runbook and the plan must agree);
- `world_size` as an argument; per-rank slicing and R4 padding as IR fields.
The IR: `lockstep_ir.json` from `lockstep_ir.py` (manifest + config -> layers of op nodes with contract op names,
constants by manifest tensor name + hash, declared parameters, limits), canonical JSON (sorted keys, fixed separators),
hashed. Gate S0: the IR round-trips for Qwen3-8B and Qwen3-30B-A3B (every constant the engine loads referenced once;
`verify_constants` recomputes every one) AND every new clause has a vector. Kill line: a section that cannot be written
as a total rule.

### S1 - Interpreter v0, the torch composition (weeks 2-4)

`lssa_runtime/lockstep_interp/`: loader (manifest + constants via `lockstep_manifest.load_constants`), IR executor,
layer wiring (residual, qkv / o with the fused-layer variants as declared, R1 / R4 sites, outlier side path), the
attention driver (B8 on prompt rows, B on generated rows by ORIGIN, B9.x by manifest rule, `mu` from keq, batch 1 per
sequence, chunk size a parameter that must not change the output), the MoE block through `moe_ref`, the head through
`t16h_lmvariants` under the pinned order, the decode loop with the tie rule and the §16 sampler. Models: Qwen3-8B
(dense, 2.6, q/k norm) and Qwen3-30B-A3B (MoE). Four agent-built modules against the S0 IR (loader + wiring;
attention driver; MoE + head; decode loop + CLI), one integrator.
Gate S1 ("I1"): for 64 prompts (32 chat, 32 synthetic incl. tie-dense and 8k), 64 new tokens each, greedy and 16
sampled seeds, the served engine's token ids equal the interpreter's and the per-step bf16 logit digests match - zero
differences. "I2": the existing E1 / E4 row dumps replay through the interpreter's operators bit-exact. Kill line: an
I1 difference not attributable to a declared clause; written up before either side changes (the engine is never the
reference). Note: I1 needs S2(a) in the engine for the tie rule and the sampler; greedy-without-ties runs first.

### S2 - Close the engine-side holes (weeks 3-6)

(a) A vLLM sampler patch implementing the declared argmax tie rule and the §16 sampler with the declared seed
derivation; gate: 16 seeds identical engine vs interpreter. (b) A standalone §9 TP reduction reference and the
`world_size` emulation in the interpreter; gate: TP 2 / 4 engine == interpreter on the I1 set. (c) Preemption: either
the engine reproduces the ORIGIN rule under recompute (gate: forced preemption mid-generation, ids unchanged) or the
manifest bans it and the engine refuses. (d) The head epilogue order fixed on whichever side S0 decided.
GPT-OSS attention (head dim 64, the 128 window, the sink logit, YaRN) is NOT in this stage: it is a contract addendum
the size of B9 (moved to S7). Its feasibility spike is week 1's task: re-derive the score bound for dh 64, write the
window as a block-aligned key range in the B8 fold (KB = 128 makes it at most two blocks), the sink as one extra
denominator term folded first at a declared exponent, YaRN as a Q14 table derivation - one page, no code. If any
piece cannot be made exact and portable, GPT-OSS attention is declared out of scope for SEMANTICS v1 and stays stock,
stated in the manifest.

### S3 - Commitment, declaration, the claim (weeks 4-7)

Manifest v2 with a SEMANTIC ROOT: unkeyed, domain-separated BLAKE3 Merkle root over canonical-JSON leaves for
{per-tensor digests, config.json, tokenizer files and chat template (as identifiers), `lockstep_ir.json`, the contract
documents' hashes, `contract_version` (from the manifest, never the environment), the interpreter v1 release tag and
source hash (named separately in the claim so the hash is not circular), the attention rule, `world_size`, the banned
features (prefix caching per S0's decision, inductor off, preemption per S2(c))}. Deployment attestation SEPARATE from
the semantic root: graph modes, GPU, driver, the runtime BLAKE3 commitment key. `outliers` / `smoothing` validated;
`lockstep_git="unknown"` refused. Signing: an Ed25519 deployment key, its public key published out of band (a registry,
roadmap 2.12), signs (root, request hash, output ids, optional step commitments); the keyed BLAKE3 stays a runtime
integrity MAC only. The engine: `/lockstep/manifest`, `/lockstep/fingerprint` (the probe result, signed),
`system_fingerprint` = root in every completion. `docs/CLAIM_FORMAT.md`: (root, world_size, prompt_ids, params,
output_ids, signature); `lockstep_verify.py` re-executes a claim with the interpreter and checks the signature. Gate S3:
an independent script (`b3sum` / `sha256sum` and a 40-line canonicaliser, no repo imports) recomputes the root from the
manifest directory; a claim from a served engine verifies on a machine without a GPU from the public checkpoint + the
manifest; a claim with a forged signature is rejected.

### S4 - Interpreter v1, dependency-free, and the conformance packs (weeks 6-10)

numpy + python integers: int8 matmuls as float64 BLAS (exact: every int8 x int8 partial sum stays below 2^53, so any
summation order gives the same integer; the argument written next to the code), the 2.6 norm's numpy twin, the
attention goldens (already numpy), MoE and the head planes ported, the table SiLU from `contract_common`, the sampler.
No float32 numpy op except the declared IEEE sites (sqrt, divide, the norm's chains). The trusted base after S4,
stated in the document: CPython, numpy's float64 BLAS and IEEE float32 paths, the platform sqrt / divide, `blake3`,
the safetensors parser, the canonical-JSON encoder, the OS. Conformance packs are GENERATED FROM v1 (Golden-Ruler
style: hashed JSON per section incl. attention B8 / B / B9.x, MoE, head planes, TP reduction, decode loop; tie-dense
and edge-case inputs) and v0 is checked against them, not the other way round. Gate S4: v1 == v0 on every pack and on
the I1 set; v1 identical on x86 and ARM and the H100 host's CPU; the 8B 512-token replay under one hour. Kill line: a
site that needs a library transcendental (it becomes a table).

### S5 - Invariance as specification and adversarial evidence (weeks 8-12)

Write the invariance clauses as rules with their gates as conformance (F3, K3 / G7 / S8 negative control, K7 / K10z,
E6, TP rank order, the prefix-caching decision). Fault injection on the I1 set: a one-ULP epilogue fault, a swapped
reduction order, a wrong tie-break, a re-quantised expert, a preemption mid-generation, a swapped model behind a
correct fingerprint - each must be caught by re-execution (the argument Chen 2026 makes against tolerance schemes,
shown on our stack). Cross-arch packs on H100, Ada, x86, ARM with the records generated from the committed kits (the
the earlier contribution lesson); Blackwell added only when hardware is confirmed. The product statement: a signed transcript plus a
declared re-execution sampling rate is what catches an operator swap. Gate S5: every fault detected; every pack
identical on every host.

### S6 - The formal hook (from week 6, parallel, with the Lean author)

Extend the Lean development to 2.6; state the IR ops as Lean definitions over integers and an executable IEEE binary32 model
(TorchLean-style), attention B8 and MoE included. Gate S6: the Lean extraction reproduces the S4 pack digests for the
sections it covers. Proof-friendliness: 2.6's IEEE sqrt / divide are portable but costly in a circuit; keep the 2.5
integer norm selectable and document a "2.7 integer-only profile" as the zk-facing variant if a proof system is
pursued. No proofs promised in Stage IV.

### S7 - GPT-OSS attention as a rule (after S4; the kernel is the performance owner's)

Addendum B10 from the week-1 spike: dh 64 bound, block-aligned window, the sink term, the YaRN table; `lssab10_torch`
+ numpy golden + selftest (200 cases incl. window edges and sink dominance); the envelope accepts dh 64 / windows for
IRs that declare B10. Only then does the GPU seam work start, on the performance side.

## 4. Order, dependencies, calendar

S0 -> S1 -> I1 is the spine and the first artifact the cryptographer can hold (week 4). S2(a-d) needs only S0 and
runs beside S1. S3 needs S0. S4 needs v0 as its oracle. S5 needs S3 and S4. S6 needs S0's IR. S7 needs S4.

| week | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 inventory, clauses, IR (+ GPT-OSS spike) | x | x | | | | | | | | | | | | |
| S1 v0 + I1 | | x | x | x | | | | | | | | | | |
| S2 engine holes (sampler, TP ref, preemption, head) | | | x | x | x | x | | | | | | | | |
| S3 root, signing, endpoints, claim | | | | x | x | x | x | | | | | | | |
| S4 v1 + packs | | | | | | x | x | x | x | x | | | | |
| S5 invariance, faults, cross-arch | | | | | | | | x | x | x | x | x | | |
| S6 formal hook | | | | | | x | x | x | x | x | x | x | x | x |
| S7 GPT-OSS attention rule | | | | | | | | | | x | x | x | x | x |

Credibility: the spine S0-S1-S3-S4 in ten weeks for one owner with agents matches this repository's rate; S5, S6 and
S7 together are not one owner's twelve weeks and are scheduled to fourteen with S6 shared with the Lean author and
S7's kernel half on the performance side.

## 5. What the root commits to and what stays outside it

Inside the semantic root: weights (per-tensor digests), config, tokenizer and template identifiers, IR, contract
documents, contract version, interpreter release, attention rule, world size, banned features. Outside, in the
deployment attestation: GPU, driver, graph modes, the runtime MAC key, the signing public key. A claim names the root
and the interpreter release separately and carries the signature; verifying it needs the public checkpoint, the
manifest and the interpreter - not the operator.

## 6. How this compares to the state of the art

| property | TOSA ref model | StableHLO interp | ONNX ref | batch-invariant engines | Chen 2026 | this plan |
|---|---|---|---|---|---|---|
| exact integer semantics | int profile | tolerances | no | no (fixed-order fp) | GEMM only | every op |
| attention + decode loop in the spec | no | no | no | engine-defined | no | S0 / S2 |
| dependency-free executable, precedence clause | yes | APFloat | numpy | no | no | S4 |
| hashed conformance packs per section, generated independently of the implementation under test | yes | partial | no | no | no | S4 |
| declared limits | yes | no | no | n/a | no | S0 IR |
| cross-hardware identity | by construction | n/a | n/a | one GPU family | shown | H100 / Ada / x86 / ARM |
| commitment to weights + spec + tokenizer + interpreter, signed | no | no | no | no | no | S3 |
| formal linkage | no | no | no | no | no | S6 |

## 7. Decisions the owner must take

1. Sampling: greedy plus the §16 integer sampler as the declared sampled mode (recommended), or greedy only.
2. Prefix caching: prove full-block exactness under the aligned rule, or ban it in the manifest (S0 decides which to
   attempt; my recommendation: ban for v1, prove later).
3. Preemption: reproduce the ORIGIN rule under recompute (engine work) or ban and refuse (recommended for v1).
4. The head epilogue order: t16h's `(acc * asc) * ws` (what is served and gated) or §2.6's - pick one, gate the other.
5. GPT-OSS attention: S7 after the spike (recommended) or out of scope for SEMANTICS v1.
6. The zk-facing "2.7 integer-only profile": keep the 2.5 norm selectable (recommended; costs nothing).
7. Log-probabilities: out of F (recommended; logit digests instead) or a declared integer log-softmax (more work).

## 8. What the adversarial review changed

Log-probs removed from F (undeclared log-softmax); token role by origin and the preemption rule added; block-exponent-
at-first-write and generated-key append rule added; head epilogue order conflict surfaced; prefix caching made a
decision; NaN / Inf totality argument required per site; `world_size` made an argument; tokenizer and template made
identifiers outside F; sampler details (tie rule, LUT derivation, temperature, seed, modulo) pinned; GPT-OSS attention
moved out of S2 to S7 with a week-1 spike; int64 numpy matmul replaced by exact float64 BLAS; the trusted base after S4
stated; packs generated from v1 and v0 checked against them; the S3 gate rewritten as an independent recomputation;
the keyed BLAKE3 demoted to a runtime MAC and Ed25519 signing with an out-of-band public key added; interpreter release
named separately from its hash; Blackwell removed from the S5 gate; preemption added to the fault list; the S6 gate made
a digest reproduction; precedence clause and IR limits added; the semantic root separated from the deployment
attestation; calendar extended to fourteen weeks with S6 / S7 shared.

## 9. Risks

- I1 finds engine behaviour that is neither a kernel bug nor a declared clause (scheduling, roles, chunk boundaries).
  That is the gate's purpose; each becomes a clause or a fix; the engine is never the reference.
- v1 speed on a 30B MoE: float64 BLAS keeps it in audit range; v0 remains the fast oracle.
- YaRN: a smooth interpolation is float arithmetic; the declared form must be a table derivation or it is not portable.
- Scope creep into performance: nothing in Stage IV changes served speed; every engine change is a declaration
  (sampler, endpoints, manifest, preemption rule) gated bit-identical against the interpreter.
