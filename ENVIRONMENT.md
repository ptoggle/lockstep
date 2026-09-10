# Environment

- Python 3.11 or later; `numpy`, `torch` (CPU build is enough), `safetensors` for `tools/verify_constants.py`.
- Lean 4.33.1 via `elan` (pinned in `lean-toolchain`); `lake build` fetches nothing beyond the standard library.
- The measured results were produced with vLLM 0.25.1, torch 2.11 + CUDA 13.0 on NVIDIA H100 80GB (one node per measurement), and the cross-architecture kits on H100 (sm_90), RTX 4090 (sm_89) and RTX PRO 6000 Blackwell (sm_120) with torch 2.9.1 + CUDA 12.8; CPU legs on x86 (numpy 1.26 / 2.3, torch 2.4 / 2.11) and Apple silicon (numpy 2.5). Each result file carries its own versions where the harness recorded them.
