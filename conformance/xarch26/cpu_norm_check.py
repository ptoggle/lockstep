# CPU leg of the contract-2.6 float-norm cross-arch check: the numpy twin alone (no torch), every norm case of the kit,
# against the ref_out16 digests of a GPU run.  python cpu_norm_check.py <dir with inputs.npz> <digests_<tag>.json>
# 2026-09-10: Apple M-series (arm64, numpy 2.5.3) vs digests_h100merge.json: 72 / 72 identical.
import sys, json, hashlib, math, platform, numpy as np
HERE = sys.argv[1]; DIG = sys.argv[2]
FLT_MIN = 2.0 ** -126; f32 = np.float32
def sqrt_n(N): return float(np.float32(math.sqrt(N)))
def eps_term(N, eps): return float(np.float32(float(N) * float(eps or 0.0)))
def bf16_rne(v):
    u = v.astype(np.float32).view(np.uint32).astype(np.uint64); lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) & 0xFFFF0000).astype(np.uint32).view(np.float32)
def normf_np(x, g, sN, eps):
    R, N = x.shape; sq = (x * x).astype(f32); grp = sq.reshape(R, N // 8, 8); a = grp[:, :, 0].copy()
    for i in range(1, 8): a = (a + grp[:, :, i]).astype(f32)
    def tree(v):
        n = v.shape[1]
        if n == 1: return v[:, 0]
        P = 1 << (n.bit_length() - 1)
        if P == n: P = n // 2
        return (tree(v[:, :P]) + tree(v[:, P:])).astype(f32)
    S = tree(a)
    if eps: S = (S + f32(eps_term(N, eps))).astype(f32)
    S = np.maximum(S, f32(FLT_MIN)).astype(f32)
    rs = (np.full_like(S, f32(sN)) / np.sqrt(S).astype(f32)).astype(f32)
    y = (x * rs[:, None]).astype(f32)
    if not np.all(g == 1.0): y = (y * g[None, :].astype(f32)).astype(f32)
    return bf16_rne(y)
def bf16_to_f32(i16): return (i16.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)
import os
# norm_inputs.npz holds the 108 norm arrays of the shipped inputs (committed); inputs.npz is the full kit file
inp = np.load(HERE + "/inputs.npz") if os.path.exists(HERE + "/inputs.npz") else np.load(HERE + "/norm_inputs.npz")
dig = json.load(open(DIG))["cases"]
ok = bad = 0; dump = {}
for name, c in dig.items():
    if not name.startswith("norm_"): continue
    key, res = (name[:-4], True) if name.endswith("_res") else (name, False)
    N = int(key.split("_")[1][1:]); gain = key.split("_")[2]
    x = bf16_to_f32(inp[key + "_x"]); r = bf16_to_f32(inp[key + "_r"]); w = bf16_to_f32(inp[key + "_w"])
    xin = (x + r).astype(f32) if res else x
    eps = {4096: 1e-6, 3072: 1e-5, 128: 1e-6}.get(N, 0.0)
    y = normf_np(xin, w, sqrt_n(N), eps)
    dump[name] = y.view(np.uint32)
    h = hashlib.sha256(np.ascontiguousarray((y.view(np.uint32) >> 16).astype(np.uint16)).tobytes()).hexdigest()
    if h == c["outputs"]["ref_out16"]: ok += 1
    else: bad += 1; print("DIFF", name, h[:16], c["outputs"]["ref_out16"][:16])
if len(sys.argv) > 3: np.savez(sys.argv[3], **dump)
print("ARM_NORMF26_CHECK", platform.machine(), platform.system(), "numpy", np.__version__, "cases", ok + bad, "identical", ok, "differ", bad)
