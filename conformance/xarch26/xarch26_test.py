# Cross-architecture determinism and correctness kit, contract 2.6 on main (10 Sep 2026):
#   the float norm (t16n), the R4 quantiser with the slot count compiled per row width (t16l r4quant<MAXB>),
#   the fused path's rotate-and-quantise forms (t16m r4quant_pad / r4quant_outl / deq_silu_r4quant: cluster, one-CTA,
#   staged and 32-per-lane forms by row count), and the fused q/k head-norm epilogue (t16m deq_norm_rope).
#
#   python xarch26_test.py gen        -> inputs.npz (fixed CPU seed; the SiLU table from contract_common on the host that
#                                        generates; ship the file, never regenerate elsewhere)
#   python xarch26_test.py run [tag]  -> builds the three extensions for THIS GPU, checks every kernel against its
#                                        reference on this machine, writes digests_<tag>.json (sha256 of every output).
#                                        Identical digests on two machines = bit-identical results.
# Needs torch (CUDA), numpy, ninja, nvcc; alongside: t16n_ref.py, t16n_kernels.cu, t16l_kernels.cu, t16k_blake3.cu, t16m_kernels.cu, t16m_tp.cpp.
import os, sys, json, hashlib, platform, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t16n_ref as R

NORM_CASES = [(N, gain, M) for N in (4096, 3072, 2048, 8192, 1024, 128) for gain in ("folded", "channel") for M in (1, 7, 64)]
R4L_CASES = [(nb, M, had, outl) for nb in (16, 48) for M in (1, 7, 40) for had in (1, 0) for outl in (0, 1) if not (outl and not had)]   # the outlier form needs the rotation
R4M_CASES = [(nb, M, outl) for nb in (12, 16, 32, 48) for M in (1, 3, 17, 40, 300) for outl in (0, 1)]
DSR_CASES = [(nb, M, outl) for nb in (32, 48) for M in (1, 17, 40) for outl in (0, 1)]
HN_CASES = [(M, outl, bias) for M in (1, 7, 40, 300) for outl in (0, 1) for bias in (0, 1)]
HQ, KVH, HS, RD = 32, 8, 128, 128          # Qwen3-8B heads
KO = 8


def sha(t):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().contiguous()
        b = t.view(torch.uint8).numpy().tobytes() if t.dtype in (torch.bfloat16, torch.float32, torch.int8) else t.numpy().tobytes()
    else:
        b = np.ascontiguousarray(t).tobytes()
    return hashlib.sha256(b).hexdigest()


def edge_rows(x, N, g):
    M = x.shape[0]
    if M >= 7:
        x[1] = 0; x[2] = 0; x[2, 5] = 3.0
        x[3] = (torch.arange(N) % 64 - 32).float().mul(0.125).to(torch.bfloat16)
        x[4] = (torch.randn(N, generator=g) * 1e-30).to(torch.bfloat16)
        x[5] = (torch.randn(N, generator=g) * 3e4).to(torch.bfloat16)
        x[6] = (torch.randn(N, generator=g) * 1e-19).to(torch.bfloat16)
    return x


def i16(t): return t.view(torch.int16).numpy()


def gen():
    g = torch.Generator(device="cpu").manual_seed(20260910)
    out = {}
    for (N, gain, M) in NORM_CASES:
        x = (torch.randn(M, N, generator=g) * torch.rand(M, 1, generator=g) * 4).to(torch.bfloat16)
        x[:, ::997] *= 40
        x = edge_rows(x, N, g)
        r = (torch.randn(M, N, generator=g) * 0.5).to(torch.bfloat16)
        w = torch.ones(N, dtype=torch.bfloat16) if gain == "folded" else (1.0 + 0.1 * torch.randn(N, generator=g)).to(torch.bfloat16)
        key = "norm_N%d_%s_M%d" % (N, gain, M)
        out[key + "_x"] = i16(x); out[key + "_r"] = i16(r); out[key + "_w"] = i16(w)
    for (nb, M, had, outl) in R4L_CASES:
        N = nb * 256
        x = (torch.randn(M, N, generator=g) * torch.rand(M, 1, generator=g) * 4).to(torch.bfloat16)
        x[:, ::509] *= 30
        if M >= 7: x[1] = 0; x[2] = 0; x[2, 7] = 5.0; x[3] = (torch.arange(N) % 32 - 16).float().mul(0.25).to(torch.bfloat16)
        key = "r4l_nb%d_M%d_had%d_o%d" % (nb, M, had, outl)
        out[key + "_x"] = i16(x)
        out[key + "_oidx"] = torch.sort(torch.randperm(N, generator=g)[:KO]).values.to(torch.int32).numpy()
    for (nb, M, outl) in R4M_CASES:
        N = nb * 256
        x = (torch.randn(M, N, generator=g) * torch.rand(M, 1, generator=g) * 4).to(torch.bfloat16)
        x[:, ::509] *= 30
        if M >= 3: x[1] = 0; x[2] = (torch.arange(N) % 32 - 16).float().mul(0.25).to(torch.bfloat16)
        key = "r4m_nb%d_M%d_o%d" % (nb, M, outl)
        out[key + "_x"] = i16(x)
        out[key + "_oidx"] = torch.sort(torch.randperm(N, generator=g)[:KO]).values.to(torch.int32).numpy()
    for (nb, M, outl) in DSR_CASES:
        D = nb * 256
        key = "dsr_nb%d_M%d_o%d" % (nb, M, outl)
        out[key + "_acc"] = torch.randint(-(1 << 20), 1 << 20, (M, 2 * D), generator=g, dtype=torch.int32).numpy()
        out[key + "_asc"] = (torch.rand(M, generator=g) * 0.02 + 0.005).float().numpy()
        out[key + "_ws"] = (torch.rand(2 * D, generator=g) * 1e-3 + 1e-4).float().numpy()
        out[key + "_ho"] = i16((torch.randn(M, KO, generator=g) * 2).to(torch.bfloat16))
        out[key + "_wo"] = i16((torch.randn(2 * D, KO, generator=g) * 0.05).to(torch.bfloat16))
    N = (HQ + 2 * KVH) * HS
    for (M, outl, bias) in HN_CASES:
        key = "hn_M%d_o%d_b%d" % (M, outl, bias)
        out[key + "_acc"] = torch.randint(-(1 << 20), 1 << 20, (M, N), generator=g, dtype=torch.int32).numpy()
        out[key + "_asc"] = (torch.rand(M, generator=g) * 0.02 + 0.005).float().numpy()
        out[key + "_ws"] = (torch.rand(N, generator=g) * 1e-3 + 1e-4).float().numpy()
        out[key + "_bias"] = (torch.randn(N, generator=g) * 0.1).float().numpy()
        out[key + "_ho"] = i16((torch.randn(M, KO, generator=g) * 2).to(torch.bfloat16))
        out[key + "_wo"] = i16((torch.randn(N, KO, generator=g) * 0.05).to(torch.bfloat16))
        out[key + "_pos"] = torch.randint(0, 4096, (M,), generator=g, dtype=torch.int64).numpy()
        out[key + "_gq"] = i16((1.0 + 0.2 * torch.randn(HS, generator=g)).to(torch.bfloat16))
        out[key + "_gk"] = i16((1.0 + 0.2 * torch.randn(HS, generator=g)).to(torch.bfloat16))
    out["hn_cache"] = (torch.rand(4096, RD, generator=g) * 2 - 1).float().numpy()
    # the SiLU fraction table (a published constant): from the generating host's contract_common
    sys.path.insert(0, os.environ.get("LOCKSTEP_HOME", "/workspace/p2"))
    import contract_common as cc
    out["frac"] = cc.table_on("cpu", 13, 15).to(torch.int64).numpy()
    np.savez(os.path.join(HERE, "inputs.npz"), **out)
    print("inputs.npz written:", len(out), "arrays, sha256 of the file", hashlib.sha256(open(os.path.join(HERE, "inputs.npz"), "rb").read()).hexdigest()[:16])


def quantise_ref(out16):
    o = out16.float()
    amax = o.abs().amax(dim=-1).clamp(min=1e-8)
    asc = amax / torch.full_like(amax, 127.0)
    r = torch.ones_like(asc) / asc
    q = torch.clamp(torch.round(o * r[:, None]), -127, 127).to(torch.int8)
    return q, asc


def fwht_ref(x):
    M, N = x.shape; v = x.reshape(M, N // 256, 256).clone(); h = 1
    while h < 256:
        v = v.reshape(M, N // 256, 256 // (2 * h), 2, h); a = v[:, :, :, 0, :].clone(); b = v[:, :, :, 1, :].clone()
        v[:, :, :, 0, :] = a + b; v[:, :, :, 1, :] = a - b; v = v.reshape(M, N // 256, 256); h *= 2
    return (v * 0.0625).reshape(M, N)


def quant_div_ref(v):      # the contract's exact-division quantiser: asc = fl32(max(amax, 1e-8) / 127), q = clamp(rint(v / asc))
    amax = v.abs().amax(dim=1).clamp(min=1e-8)
    asc = amax / torch.full_like(amax, 127.0)
    q = torch.clamp(torch.round(v / asc[:, None]), -127, 127).to(torch.int8)
    return q, asc


def side_ref(y8, ho, wo):
    s = torch.zeros(y8.shape, dtype=torch.float32, device=y8.device)
    for k in range(ho.shape[1]):
        s = s + ho[:, k].float().unsqueeze(1) * wo[:, k].float().unsqueeze(0)
    return (y8.float() + s).to(torch.bfloat16)


def run(tag):
    from torch.utils import cpp_extension
    dev = "cuda"
    props = torch.cuda.get_device_properties(0)
    info = {"gpu": props.name, "capability": "%d.%d" % (props.major, props.minor), "torch": torch.__version__,
            "cuda": torch.version.cuda, "python": platform.python_version(), "host": platform.node(),
            "inputs_sha": hashlib.sha256(open(os.path.join(HERE, "inputs.npz"), "rb").read()).hexdigest()[:16]}
    print(json.dumps(info), flush=True)
    flags = ["-O3", "-prec-div=true", "-prec-sqrt=true", "-fmad=false"]
    t0 = time.time()
    KN = cpp_extension.load(name="t16n_kernels_x26", sources=[os.path.join(HERE, "t16n_kernels.cu")], extra_cuda_cflags=flags, extra_include_paths=[HERE], verbose=False)
    KL = cpp_extension.load(name="t16l_kernels_x26", sources=[os.path.join(HERE, "t16l_kernels.cu")], extra_cuda_cflags=flags, extra_include_paths=[HERE], verbose=False)
    KM = cpp_extension.load(name="t16m_kernels_x26", sources=[os.path.join(HERE, "t16m_kernels.cu"), os.path.join(HERE, "t16m_tp.cpp")], extra_cuda_cflags=flags, extra_include_paths=[HERE], verbose=False)
    print("built in %.0f s" % (time.time() - t0), flush=True)
    inp = np.load(os.path.join(HERE, "inputs.npz"))
    B = lambda k: torch.from_numpy(inp[k]).view(torch.bfloat16).to(dev)
    F = lambda k: torch.from_numpy(inp[k]).to(dev)
    digests = {"info": info, "cases": {}}
    bad = 0
    def rec(name, outs, ok, note=""):
        nonlocal bad
        bad += (not ok)
        digests["cases"][name] = {"outputs": {k: sha(v) for k, v in outs.items()}, "local_check": "PASS" if ok else "FAIL " + note}
        if not ok: print("  [FAIL]", name, note, flush=True)
    E_I32 = torch.empty(0, dtype=torch.int32, device=dev); E_BF = torch.empty(0, dtype=torch.bfloat16, device=dev)
    # ---- A. the float norm
    for (N, gain, M) in NORM_CASES:
        key = "norm_N%d_%s_M%d" % (N, gain, M)
        x = B(key + "_x"); r = B(key + "_r"); w = B(key + "_w"); gf = w.float().contiguous()
        sN = R.sqrt_n(N); eps = {4096: 1e-6, 3072: 1e-5, 128: 1e-6}.get(N, 0.0); ne = R.eps_term(N, eps)
        for res in (None, r):
            name = key + ("_res" if res is not None else "")
            xin = x if res is None else (x.float() + res.float())
            y_ref = R.rmsnorm_float(xin, w, sN, eps=eps)
            y_np = R.rmsnorm_float_np(xin.float().cpu().numpy().astype(np.float32), gf.cpu().numpy(), sN, eps)
            q_ref, asc_ref = quantise_ref(y_ref)
            quant = N != 128
            a8 = torch.empty(64, N, dtype=torch.int8, device=dev) if quant else None
            asc = torch.empty(64, dtype=torch.float32, device=dev) if quant else None
            got = KN.normf(x, res, None if gain == "folded" else gf, ne, sN, a8, asc)
            d16 = int((got[0].view(torch.int16) != y_ref.view(torch.int16)).sum())
            twins = int((y_ref.float().cpu().numpy().view(np.uint32) != y_np.view(np.uint32)).sum())
            d8 = int((a8[:M] != q_ref).sum()) if quant else 0; da = int((asc[:M].view(torch.int32) != asc_ref.view(torch.int32)).sum()) if quant else 0
            dr = int((got[1].view(torch.int16) != (res.float() + x.float()).to(torch.bfloat16).view(torch.int16)).sum()) if res is not None else 0
            outs = {"out16": got[0], "ref_out16": y_ref}
            if quant: outs.update({"a8": a8[:M], "asc": asc[:M]})
            if res is not None: outs["res_out"] = got[1]
            rec(name, outs, (d16 | d8 | da | dr | twins) == 0, "y=%d q=%d asc=%d res=%d twins=%d" % (d16, d8, da, dr, twins))
    # ---- B. t16l r4quant<MAXB> (the unfused path's rotate-and-quantise), exact division
    for (nb, M, had, outl) in R4L_CASES:
        key = "r4l_nb%d_M%d_had%d_o%d" % (nb, M, had, outl)
        x = B(key + "_x"); oidx = F(key + "_oidx") if outl else None
        xf = x.float()
        if outl:
            hor = x[:, oidx.long()].contiguous(); xf = xf.clone(); xf[:, oidx.long()] = 0.0
        vref = fwht_ref(xf) if had else xf
        a8r, ascr = quant_div_ref(vref)
        got = KL.r4quant(x, 1, had, oidx)
        d8 = int((got[0] != a8r).sum()); da = int((got[1].view(torch.int32) != ascr.view(torch.int32)).sum())
        dh = int((got[2].view(torch.int16) != hor.view(torch.int16)).sum()) if outl else 0
        outs = {"a8": got[0], "asc": got[1]}
        if outl: outs["ho"] = got[2]
        rec(key, outs, (d8 | da | dh) == 0, "a8=%d asc=%d ho=%d" % (d8, da, dh))
    # ---- C. t16m r4quant_pad / r4quant_outl (cluster / one-CTA / staged / wide forms by M), vs t16l r4quant and torch
    for (nb, M, outl) in R4M_CASES:
        key = "r4m_nb%d_M%d_o%d" % (nb, M, outl)
        x = B(key + "_x"); oidx = F(key + "_oidx") if outl else None
        ref = KL.r4quant(x, 1, 1, oidx)
        xf = x.float()
        if outl: xf = xf.clone(); xf[:, oidx.long()] = 0.0
        a8t, asct = quant_div_ref(fwht_ref(xf))
        got = KM.r4quant_outl(x, oidx, 1, 17) if outl else KM.r4quant_pad(x, 1, 1, 17)
        d8 = int((got[0][:M] != ref[0]).sum()); da = int((got[1].view(torch.int32) != ref[1].view(torch.int32)).sum())
        dt = int((got[0][:M] != a8t).sum()) + int((got[1].view(torch.int32) != asct.view(torch.int32)).sum())
        dh = int((got[2].view(torch.int16) != ref[2].view(torch.int16)).sum()) if outl else 0
        outs = {"a8": got[0][:M], "asc": got[1]}
        if outl: outs["ho"] = got[2]
        rec(key, outs, (d8 | da | dt | dh) == 0, "vs t16l a8=%d asc=%d ho=%d; vs torch %d" % (d8, da, dh, dt))
    # ---- D. t16m deq_silu_r4quant (the gate-up epilogue: dequant, SiLU x mul, rotate, quantise), vs t16l deq -> silumul -> r4quant
    frac = F("frac")
    for (nb, M, outl) in DSR_CASES:
        key = "dsr_nb%d_M%d_o%d" % (nb, M, outl)
        acc = F(key + "_acc"); asc = F(key + "_asc"); ws = F(key + "_ws")
        ho = B(key + "_ho") if outl else None; wo = B(key + "_wo") if outl else None
        y = KL.deq(acc, asc, ws, None, ho, wo) if outl else KL.deq(acc, asc, ws, None)
        sref = KL.silumul(y, frac); a8r, ascr = KL.r4quant(sref.contiguous(), 1, 1)
        g1 = KM.deq_silu_r4quant(acc, E_BF, asc, ws, frac, 1, 1, 17, M, ho, wo)
        y8 = KL.deq(acc, asc, ws, None)
        g2 = KM.deq_silu_r4quant(E_I32, y8.contiguous(), asc, ws, frac, 1, 1, 17, M, ho, wo)
        d1 = int((g1[0][:M] != a8r).sum()) + int((g1[1].view(torch.int32) != ascr.view(torch.int32)).sum())
        d2 = int((g2[0][:M] != a8r).sum()) + int((g2[1].view(torch.int32) != ascr.view(torch.int32)).sum())
        rec(key, {"a8": g1[0][:M], "asc": g1[1], "a8_y8route": g2[0][:M], "ref_a8": a8r}, (d1 | d2) == 0, "acc route %d, y8 route %d" % (d1, d2))
    # ---- E. t16m deq_norm_rope (the fused q/k head norm), vs deq_rows -> normf -> ropeq14
    cache = F("hn_cache"); sN = R.sqrt_n(HS); ne = R.eps_term(HS, 1e-6); N = (HQ + 2 * KVH) * HS
    for (M, outl, bias) in HN_CASES:
        key = "hn_M%d_o%d_b%d" % (M, outl, bias)
        acc = F(key + "_acc"); asc = F(key + "_asc"); ws = F(key + "_ws"); pos = F(key + "_pos")
        b = F(key + "_bias") if bias else None
        ho = B(key + "_ho") if outl else None; wo = B(key + "_wo") if outl else None
        gq = B(key + "_gq").float().contiguous(); gk = B(key + "_gk").float().contiguous()
        y = KM.deq_rows(acc, asc, ws, b, M, ho, wo)
        q = y[:, :HQ * HS]; k = y[:, HQ * HS:(HQ + KVH) * HS]; v = y[:, (HQ + KVH) * HS:]
        qn = KN.normf(q.reshape(-1, HS).contiguous(), None, gq, ne, sN, None, None)[0].reshape(M, HQ * HS)
        kn = KN.normf(k.reshape(-1, HS).contiguous(), None, gk, ne, sN, None, None)[0].reshape(M, KVH * HS)
        qr, kr = KL.ropeq14(pos, qn.contiguous(), kn.contiguous(), cache, HS, RD)
        ref = torch.cat([qr.reshape(M, -1), kr.reshape(M, -1), v], 1)
        got = KM.deq_norm_rope(acc, E_BF, asc, ws, b, pos, cache, gq, gk, ne, sN, HQ, KVH, HS, RD, M, ho, wo)
        y8 = KM.deq_rows(acc, asc, ws, None, M)
        yy = side_ref(y8, ho, wo) if outl else y8
        if bias: yy = (yy.float() + b).to(torch.bfloat16)
        q2 = yy[:, :HQ * HS]; k2 = yy[:, HQ * HS:(HQ + KVH) * HS]; v2 = yy[:, (HQ + KVH) * HS:]
        qn2 = KN.normf(q2.reshape(-1, HS).contiguous(), None, gq, ne, sN, None, None)[0].reshape(M, HQ * HS)
        kn2 = KN.normf(k2.reshape(-1, HS).contiguous(), None, gk, ne, sN, None, None)[0].reshape(M, KVH * HS)
        qr2, kr2 = KL.ropeq14(pos, qn2.contiguous(), kn2.contiguous(), cache, HS, RD)
        ref2 = torch.cat([qr2.reshape(M, -1), kr2.reshape(M, -1), v2], 1)
        got2 = KM.deq_norm_rope(E_I32, y8, asc, ws, b, pos, cache, gq, gk, ne, sN, HQ, KVH, HS, RD, M, ho, wo)
        d1 = int((got.view(torch.int16) != ref.view(torch.int16)).sum()); d2 = int((got2.view(torch.int16) != ref2.view(torch.int16)).sum())
        rec(key, {"out": got, "out_y8route": got2, "ref": ref}, (d1 | d2) == 0, "acc route %d, y8 route %d" % (d1, d2))
    digests["local_gate"] = "PASS" if bad == 0 else "FAIL (%d cases)" % bad
    h = hashlib.sha256()
    for name in sorted(digests["cases"]):
        for k, v in sorted(digests["cases"][name]["outputs"].items()):
            if not k.startswith("ref"): h.update((name + k + v).encode())
    digests["all_kernel_outputs"] = h.hexdigest()
    h = hashlib.sha256()
    for name in sorted(digests["cases"]):
        for k, v in sorted(digests["cases"][name]["outputs"].items()):
            if k.startswith("ref"): h.update((name + k + v).encode())
    digests["all_reference_outputs"] = h.hexdigest()
    n_out = sum(len(c["outputs"]) for c in digests["cases"].values())
    path = os.path.join(HERE, "digests_%s.json" % tag)
    json.dump(digests, open(path, "w"), indent=1)
    print("XARCH26", tag, info["gpu"], "sm_%s" % info["capability"].replace(".", ""), "local gate", digests["local_gate"],
          "| %d cases, %d output digests" % (len(digests["cases"]), n_out),
          "| kernel digest", digests["all_kernel_outputs"][:24], "| reference digest", digests["all_reference_outputs"][:24], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen()
    else:
        run(sys.argv[2] if len(sys.argv) > 2 else "run")
