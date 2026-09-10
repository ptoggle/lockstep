# Changelog (contract history)

- **0.2.0 (2026-09-10, DOI 10.5281/zenodo.22695303)** - cross-architecture record regenerated from the committed kit on H100, RTX 4090 and RTX 5090 (0 / 538 differ). First export of the public specification set. Contract vectors regenerated with a version-stable input stream (numpy PCG64) and verified identical on x86 / torch 2.11 and arm64 / torch 2.14; 36 vectors.

- **v3 (2026-09-10)** - mixture of experts: int8 gate, table router with lowest-index ties, bf16 routing weights, int8 experts under the dense contract, ascending-index combine; GPT-OSS addendum (gate and expert biases, clamped SwiGLU, 64-lane rotation); the exact-MXFP4 rule (per 32-block exact int32 partial, one rounded binary32 update in ascending block order). `docs/SPEC.md` section 7.27.
- **2.6 (2026-09-10)** - the float RMSNorm: index-pinned fp32 reduction (8-chains, balanced tree), one IEEE sqrt and one IEEE division; 2.5's integer norm stays selectable. `docs/contract/NON_ATTENTION_STACK_v2.6.md`.
- **2.5 (2026-09-07)** - declared outlier channels (default eight) on R1-online norm-fed linears with an exact side path.
- **2.4** - R1 online as a per-model option. **2.3** - the smoothing fold. **2.2** - the norm carries the model's epsilon. **2.1** - the linear epilogue order declared (weight scale first).
- **Attention** - LSSA-B (v1), LSSA-B8 (v2: block-local 8-bit weights, ascending order, two-level fold), addenda B9.1 (position zero alone) and B9.3 (per-key value grid). `docs/contract/BLOCK_EXPONENT_VARIANT.md`.
