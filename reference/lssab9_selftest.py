#!/usr/bin/env python3
"""LSSA-B9 self-test: lssab9_torch == lssab9_golden BIT-FOR-BIT for every arm (L9 == the B8 pair,
L9-sink, L9-v16, L9-sink-v16) on random, jump and degenerate cases, plus L9 == lssab8 golden.
CPU only."""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import lssab8_golden as G8, lssab9_golden as G9, lssab8_torch as L8, lssab9_torch as L9
import lssab8_selftest as S8
N_PASS = N_FAIL = 0
def check(name, ok, extra=""):
    global N_PASS, N_FAIL
    N_PASS += ok; N_FAIL += (not ok)
    print("  %-56s %s %s" % (name, "PASS" if ok else "FAIL", extra), flush=True)
class _Mod: layer_idx = 0
def bits(x): return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
def torch_out(q, k, v, arm, **kw):
    spec = dict(L9.ARM_SPECS[arm])
    fn = L9.make_lssab9_long_attn(head_chunk=4, qblk=512, kblk=1024, **spec, **kw)
    o, _ = fn(_Mod(), torch.tensor(q)[None], torch.tensor(k)[None], torch.tensor(v)[None], None)
    return o[0].transpose(0, 1).contiguous().numpy()
def golden_out(q, k, v, arm):
    spec = L9.ARM_SPECS[arm]
    return G9.lssab9_attention(q, k, v, mu=None, kb=128, seg=32, n=7, w_loc=9, weight="rn", **spec)
def cmp(q, k, v, tag, arm):
    a = torch_out(q, k, v, arm); b = golden_out(q, k, v, arm)
    same = np.array_equal(bits(a), bits(b))
    extra = "" if same else "%d/%d differ" % (int((bits(a) != bits(b)).sum()), a.size)
    check("%s %s torch == golden" % (arm, tag), same, extra)
    return same
cases = [(c["tag"], c["q"], c["k"], c["v"]) for c in (S8.random_cases() + S8.jump_cases() + S8.degenerate_cases())]
# S1: L9 == B8 golden bit for bit (both off)
for tag, q, k, v in cases[:6]:
    b8 = G8.lssab8_attention(q, k, v, mu=None, kb=128, seg=32, n=7, w_loc=9, weight="rn")
    b9 = golden_out(q, k, v, "L9")
    check("S1 L9 golden == B8 golden %s" % tag, np.array_equal(bits(b8), bits(b9)))
# S2: every arm, torch == golden
for arm in ("L9", "L9-sink", "L9-v16", "L9-sink-v16", "L9-ks3", "L9-ks7r", "L9-sink-ks3"):
    for tag, q, k, v in (cases if arm in ("L9", "L9-sink", "L9-v16", "L9-sink-v16") or os.environ.get("LSSAB9_FULL") else cases[::3]):
        cmp(q, k, v, tag, arm)
# S3: a sink-like case: key 0's value 300x the rest
rng = np.random.default_rng(9)
for T in (1, 2, 128, 129, 300):
    q = rng.standard_normal((4, T, 128)).astype(np.float32); k = rng.standard_normal((2, T, 128)).astype(np.float32)
    v = rng.standard_normal((2, T, 128)).astype(np.float32); v[:, 0] *= 300.0
    for arm in ("L9-sink", "L9-sink-v16", "L9-ks7r", "L9-sink-ks3"):
        cmp(q, k, v, "sink300_T%d" % T, arm)
print("LSSAB9-SELFTEST pass=%d fail=%d" % (N_PASS, N_FAIL))
sys.exit(0 if N_FAIL == 0 else 1)
