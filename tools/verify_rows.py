#!/usr/bin/env python3
"""The verifier's job: replay committed attention rows against the reference.

Takes the dump written by e2e_demo.py (DUMP=1) -- the engine's own int-quantised
q/k/v inputs and bf16 outputs for every (sequence, step) at the dumped layers --
and recomputes every row with the REFERENCE implementation of the contract:

    prompt rows     -> lssab8_torch  (LSSA-B8 rule, the prompt-row contract)
    generated rows  -> lssab_torch   (LSSA-B rule,  the generated-row contract)

The role comes from the engine's own is_prefilling array in the dump, never from
a query length.  The check is BIT EQUALITY (bf16 bit patterns), no tolerance.
DEVICE=cpu runs the reference on the CPU -- the point of the contract is that a
verifier needs no GPU -- optionally on a sample of sequence-steps (MAX_STEPS).

usage: verify_rows.py <dump.pt> <keq_*.pt> <out.json>
env:   DEVICE=cuda|cpu  MAX_STEPS (per layer, 0 = all)  LOCKSTEP_LSSAB_MU_KIND
Logic is t16e_b8_check.py's (the E1/E4 gate), with device control and sampling.
"""
import os, sys, json, time, torch
sys.path.insert(0, "/workspace/p2")
import lssab_torch as LT
import lssab8_torch as LT8
import lssab8_golden as G8

KB, DH, BHI = 128, 128, 26
DUMP, MUP, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DEV = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAXS = int(os.environ.get("MAX_STEPS", "0"))
KIND = os.environ.get("LOCKSTEP_LSSAB_MU_KIND", "MU_MID")
torch.set_num_threads(max(1, os.cpu_count() // 2))
_o = torch.load(MUP, map_location="cpu", weights_only=False)
MU = _o[KIND].to(torch.float32).to(DEV)
print("verifier device=%s  mu %s %s sha=%s  |  B8: n=%d W_LOC=%d SEG=%d S_WIN=%d TOP=%d"
      % (DEV, KIND, tuple(MU.shape), _o.get("bin_sha256") or _o.get("sha256"),
         G8.NIDX, G8.W_LOC, G8.SEG, G8.S_WIN, G8.TOP), flush=True)


class Mod:
    def __init__(self, li=0): self.layer_idx = li


_fn = {}


def oracle(q, k, v, li, prompt):
    H = q.shape[1]
    key = (H, bool(prompt))
    if key not in _fn:
        if prompt:
            _fn[key] = LT8.make_lssab8_attn(kb=KB, bhi=BHI, n=G8.NIDX, w_loc=G8.W_LOC,
                                            weight="rn", head_chunk=H, mu=MU,
                                            qexp="row", stats={})
        else:
            _fn[key] = LT.make_lssab_attn(KB, bhi=BHI, n=13, w=15, head_chunk=H,
                                          mu=MU, qexp="row", stats={})
    o, _ = _fn[key](Mod(li), q.transpose(0, 1)[None].contiguous(),
                    k.transpose(0, 1)[None].contiguous(),
                    v.transpose(0, 1)[None].contiguous(), None)
    return o[0]


t0 = time.time()
D = torch.load(DUMP, map_location=DEV, weights_only=False)
acc, per_layer = {}, {}
n = dict(p_rows=0, p_bad=0, g_rows=0, g_bad=0, p_steps=0, g_steps=0, skipped=0)
worst = 0.0
for d in D:
    L, qsl, sl = d["layer"], d["qsl"].tolist(), d["sl"].tolist()
    pref = d.get("pref")
    if pref is None:
        n["skipped"] += 1
        continue
    pl = per_layer.setdefault(L, dict(p_rows=0, p_bad=0, g_rows=0, g_bad=0, steps=0))
    for i in range(d["nseq"]):
        a0, a1 = qsl[i], qsl[i + 1]
        if a1 <= a0:
            continue
        key = (L, int(d["bt0"][i]))
        kk, vv = acc.get(key, (None, None))
        k_i, v_i = d["k"][a0:a1].to(DEV), d["v"][a0:a1].to(DEV)
        kk = k_i if kk is None else torch.cat([kk, k_i], 0)
        vv = v_i if vv is None else torch.cat([vv, v_i], 0)
        acc[key] = (kk, vv)
        T = kk.shape[0]
        if T != sl[i]:
            continue                       # this step's KV is not complete yet
        if MAXS and pl["steps"] >= MAXS:
            continue
        pl["steps"] += 1
        qq = torch.zeros(T, d["q"].shape[1], DH, dtype=d["q"].dtype, device=DEV)
        qq[T - (a1 - a0):] = d["q"][a0:a1].to(DEV)
        prompt = bool(pref[i])
        ref = oracle(qq, kk, vv, L, prompt)[T - (a1 - a0):]
        got = d["out"][a0:a1].to(DEV)
        bad = int((got.view(torch.int16) != ref.view(torch.int16)).any(-1).sum().item())
        mx = (got.float() - ref.float()).abs().max().item()
        worst = max(worst, mx)
        rows = got.shape[0] * got.shape[1]
        if prompt:
            n["p_rows"] += rows; n["p_bad"] += bad; n["p_steps"] += 1
            pl["p_rows"] += rows; pl["p_bad"] += bad
        else:
            n["g_rows"] += rows; n["g_bad"] += bad; n["g_steps"] += 1
            pl["g_rows"] += rows; pl["g_bad"] += bad
secs = time.time() - t0
ok = (n["p_bad"] == 0 and n["g_bad"] == 0 and (n["p_rows"] + n["g_rows"]) > 0)
res = dict(device=DEV, dump=DUMP, max_steps=MAXS, secs=round(secs, 1), pass_=ok,
           prompt_rows=n["p_rows"], prompt_bad=n["p_bad"], prompt_steps=n["p_steps"],
           generated_rows=n["g_rows"], generated_bad=n["g_bad"],
           generated_steps=n["g_steps"], maxdiff=worst, skipped_no_role=n["skipped"],
           per_layer={str(k): v for k, v in sorted(per_layer.items())})
json.dump(res, open(OUT, "w"), indent=1)
print("[%s] prompt rows %d bad %d | generated rows %d bad %d | maxdiff %.3e | "
      "%d+%d sequence-steps on %s in %.1fs -> %s"
      % ("PASS" if ok else "FAIL", n["p_rows"], n["p_bad"], n["g_rows"], n["g_bad"],
         worst, n["p_steps"], n["g_steps"], DEV, secs, OUT), flush=True)
sys.exit(0 if ok else 1)
