# Artifact description

Prepared in the format expected by artifact evaluation (Available / Functional / Reproduced).

**Available.** This repository, tagged; the manuscript in `paper/`; the evidence index `docs/EVIDENCE.md` mapping every claim to its artifact.

**Functional (CPU only, no GPU, no vLLM).** `make verify` runs: the attention goldens' selftests (LSSA-B 239 cases, LSSA-B8 28 checks / 119 cases, B9 228 cases), the MXFP4 dequant selftest, the contract vectors check, the Lean build and its bridge verifier with the mutation test, and the CPU leg of the float-norm cross-architecture check (72 cases, from the committed `norm_inputs.npz`). Expect PASS on every line; a differing bit anywhere is a defect.

**Reproduced (GPU).** The kernel gates, engine gates and the paired measurements against stock vLLM need the optimized implementation, which is not in this repository; `docs/EVIDENCE.md` names each gate, its case count, the result file and the command, and `results/` holds the raw result files of every row in the paper.

Environment: see `ENVIRONMENT.md`.
