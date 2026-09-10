# Conformance: run the kit on your device and submit the digests

Current record (0 of 538 digests differ): H100 sm_90 (torch 2.11 / CUDA 13.0), RTX 4090 sm_89 (torch 2.4 / CUDA 12.4), RTX 5090 sm_120 (torch 2.8 / CUDA 12.8); CPU legs on x86 and Apple silicon. See `conformance/xarch26/RESULTS.md`.

The claim "the same bits on every device" is only as strong as the set of devices anyone has run the kit on. Adding one takes about ten minutes on a GPU host with torch, numpy, ninja and nvcc.

1. `cd conformance/xarch26`
2. `python xarch26_test.py gen` regenerates `inputs.npz` from the fixed seed (the file is not committed; the script prints its sha256, which must match the value in `RESULTS.md`).
3. `python xarch26_test.py run <tag>` builds the kernel sources for your GPU, checks every kernel against its torch reference and numpy twin on your machine, and writes `digests_<tag>.json`.
4. On a CPU-only host, `make cpu-norm` (or `python cpu_norm_check.py . digests_h100merge.json`) runs the numpy twin of the float norm against the committed H100 reference digests (72 cases); it reads the committed `norm_inputs.npz` (the 108 norm arrays of the shipped inputs) so no GPU or torch is needed.
5. Open a pull request adding `digests_<tag>.json`, a line in `RESULTS.md` with the device, driver, torch and CUDA versions, and the output of `python compare.py`.

A differing digest is a report, not a failure of the submission: file it with the case name and both digests.
