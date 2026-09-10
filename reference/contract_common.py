# Shared contract implementation for the Phase-1 big run.
# Semantics IDENTICAL to box/g2_ppl.py (the filed EV-5 methodology). Extensions only:
#   - device-agnostic (per-device FRAC/mask caches) for device_map=auto sharding
#   - head-chunked attention (bounded transients at 70B head counts; per-head math unchanged)
#   - Qwen3-MoE/Llama eager patch targets; W8A8 skips lm_head AND MoE routers ('.gate')
# Exactness guards: TF32 pinned OFF; fp32 score matmul exact (|S| <= 127*127*128 < 2^24);
# f64 PV exact (eh<=2^15, |v8|<=127, T<=2^17 -> sums < 2^40 << 2^53).
import math, hashlib, json, os, platform, subprocess, time
import numpy as np, torch
import ref_block_v1 as R

FRAC_DIGEST = "f472476ccea65299feb63f1ff863714a35910d2099470bafb5d3be0e92a598cd"

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try: torch.set_float32_matmul_precision("highest")
except Exception: pass

def frac_check():
    d = hashlib.sha256(np.asarray(R.FRAC, dtype="<u2").tobytes()).hexdigest()
    assert d == FRAC_DIGEST, "FRAC digest mismatch: " + d
    assert not torch.backends.cuda.matmul.allow_tf32, "TF32 must be off"
    return d

# The seven frozen Q30 coefficients (Section 6.1.3) and the parameterized table family
# (Section 6.1.8): FRAC_{n,w}[k] = (Horner(t = k<<(20-n)) + 2^(29-w)) >> (30-w).
# Verified points (Appendix E, re-checked locally 2026-08-31): (13,15),(13,16),(14,16) all
# 0 misrounded; (13,17)/(14,15)/(13,13) are FAIL points and must not be used.
COEFFS = [1073741823, -744260975, 257938938, -59583045, 10286509, -1369790, 117454]
VERIFIED_NW = {(13, 15), (12, 14), (12, 15), (13, 14), (13, 16), (14, 16), (11, 15),
               (15, 16)}   # (15,16) verified 2026-08-31 (grid search, 0/32768 misrounded)

def build_table(n, w):
    assert (n, w) in VERIFIED_NW, "(n,w)=(%d,%d) is not a verified point" % (n, w)
    tab = []
    for k in range(1 << n):
        t = k << (20 - n); p = COEFFS[6]
        for c in COEFFS[5::-1]:
            p = c + ((p * t) // (1 << 20))
        tab.append((p + (1 << (29 - w))) >> (30 - w))
    return np.asarray(tab, dtype=np.int64)

_tables, _frac_t, _masks = {}, {}, {}
def table_np(n, w):
    if (n, w) not in _tables:
        t = build_table(n, w)
        if (n, w) == (13, 15):
            assert np.array_equal(t, np.asarray(R.FRAC)), "build_table(13,15) != golden FRAC"
        _tables[(n, w)] = t
    return _tables[(n, w)]

def table_on(dev, n=13, w=15):
    k = (str(dev), n, w)
    if k not in _frac_t:
        _frac_t[k] = torch.tensor(table_np(n, w), dtype=torch.int64, device=dev)
    return _frac_t[k]

def frac_on(dev):          # backward-compat alias for the reference embodiment
    return table_on(dev, 13, 15)

def mask_on(T, dev):
    k = (T, str(dev))
    if k not in _masks:
        _masks[k] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))
    return _masks[k]

def pow2_t(e, dtype=None):
    """EXACT 2^e as a float tensor, built from the exponent field: no torch.ldexp.  torch.ldexp
    (2.11.0+cu130) faults with an illegal memory access when its tensors live on a CUDA device
    that is not the process's CURRENT device (SPEC 7.23, P3: the 70B under device_map=auto),
    and the fault is asynchronous, so it surfaced as a NaN or a zero block at a random window.
    float32 for e in [-126, 127], float64 for e in [-1022, 1023] (the normal ranges; the
    references assert their exponents inside them).  e: an integer tensor."""
    e = e.to(torch.int64)
    if dtype == torch.float64:
        return ((e + 1023) << 52).view(torch.float64)
    return ((e.to(torch.int32) + 127) << 23).view(torch.float32)


def quant_pow2(x):
    """x [h,T,dh] fp32 -> int8-valued fp32 q, e [h] int64. Contract semantics (g2_ppl.py):
    per-head pow2 dyn scale, frexp exponent, clip-safe decrement, clamp [-32,40], half-even rint."""
    m = x.abs().amax(dim=(-2, -1))
    mant, E = torch.frexp(m)
    e = 6 - (E.to(torch.int64) - 1)
    e = torch.where(m == 0, torch.zeros_like(e), e)
    over = m * pow2_t(e) > 127.0
    e = (e - over.to(torch.int64)).clamp(-32, 40)
    q = torch.clamp(torch.round(x * pow2_t(e)[:, None, None]), -127, 127)
    return q, e

def g_fold_t(dh, sh, bhi):
    """Tensor half-up shift of base G, clamp [1, 2^30] (g2_ppl.py semantics)."""
    base = round((2.0 ** bhi) / (math.sqrt(dh) * math.log(2.0)))
    shp = sh.clamp(min=0); shn = (-sh).clamp(min=0, max=34)
    add = torch.where(shp > 0, torch.ones_like(shp) << (shp - 1).clamp(min=0), torch.zeros_like(shp))
    Gp = (base + add) >> shp
    Gn = base << shn
    return torch.where(sh >= 0, Gp, Gn).clamp(1, 1 << 30)

def make_contract_attention(bhi, head_chunk=16, n=13, w=15):
    """Contract eager-attention replacement, head-chunked; (n,w) table family per 6.1.8.
    (13,15) reproduces g2_ppl.py bit-for-bit. Window = (w+1) doublings; index = n bits at
    bhi-n..bhi-1; shift = d >> bhi."""
    assert n <= bhi, "index width n must not exceed bhi"
    win = (w + 1) << bhi; idx_sh = bhi - n; idx_mask = (1 << n) - 1
    def contract_attention(module, query, key, value, attention_mask, *a, **kw):
        B, H, T, dh = query.shape
        assert B == 1, "B=1 windows only"
        dev = query.device
        KVH = key.shape[1]; grp = H // KVH
        qf = query[0].float(); kf = key[0].float(); vf = value[0].float()
        k8, ek = quant_pow2(kf); v8, ev = quant_pow2(vf)      # per-KV-head, once
        mask = mask_on(T, dev); frac = table_on(dev, n, w)
        outs = []
        for h0 in range(0, H, head_chunk):
            h1 = min(h0 + head_chunk, H)
            q8, eq = quant_pow2(qf[h0:h1])
            kvi = torch.arange(h0, h1, device=dev) // grp
            S = torch.matmul(q8, k8[kvi].transpose(-1, -2)).to(torch.int64)   # exact in fp32
            G = g_fold_t(dh, eq + ek[kvi], bhi)[:, None, None]
            SG = S * G
            m = SG.masked_fill(~mask, -(1 << 62)).amax(-1, keepdim=True)
            NB = -((-m) >> bhi) << bhi
            d = NB - SG
            d = torch.where(mask, d, torch.full_like(d, win))
            eh = torch.where(d < win,
                             frac[(d >> idx_sh) & idx_mask] >> torch.minimum(d >> bhi, torch.full_like(d, 63)),
                             torch.zeros_like(d))
            l = eh.sum(-1)                                                     # int64 exact
            A = torch.matmul(eh.double(), v8[kvi].double())                    # exact f64 integer PV
            of = A.to(torch.float32) / l.to(torch.float32)[..., None]          # contract f32 epilogue
            outs.append(of * (2.0 ** (-ev[kvi].to(torch.float32)))[:, None, None])
        out = torch.cat(outs, 0)
        return out.to(query.dtype)[None].transpose(1, 2).contiguous(), None
    return contract_attention

def patch_eager(fn):
    """Install fn as eager_attention_forward for all model families we evaluate."""
    import transformers.models.qwen2.modeling_qwen2 as mq
    import transformers.models.qwen3.modeling_qwen3 as mq3
    mq.eager_attention_forward = fn
    mq3.eager_attention_forward = fn
    for name in ("qwen3_moe", "llama"):
        try:
            mod = __import__("transformers.models.%s.modeling_%s" % (name, name), fromlist=["x"])
            mod.eager_attention_forward = fn
        except Exception as ex:
            print("patch skipped for %s: %r" % (name, ex), flush=True)

def w8a8_linear(lin):
    """Identical to g2_ppl.py: per-out-channel int8 W, per-token dyn int8 A, torch._int_mm.

    DEVICE-DEPENDENCE NOTE (found 2026-09-01, T5/F2): `tensor / 127.0` with a python
    scalar compiles to reciprocal-MULTIPLY on CUDA but true division on CPU (1 ulp apart
    on ~4% of channels) - the substitution SPEC 6.0.3 prohibits. Fixed below with a
    device-resident tensor divisor (true division on every backend). All measurements
    through Step 250 used the scalar form on CUDA; W8A8 is the industry-baseline QUALITY
    arm (never hashed), so no filed number or bit contract is affected - but cross-vendor
    bit-identity claims for W8A8 linears require this fixed form."""
    W = lin.weight.data.float()
    _d127 = torch.tensor(127.0, dtype=torch.float32, device=W.device)
    ws = W.abs().amax(dim=1).clamp(min=1e-8) / _d127
    W8 = torch.clamp(torch.round(W / ws[:, None]), -127, 127).to(torch.int8).contiguous()
    bias = lin.bias.data.clone() if lin.bias is not None else None
    def fwd(x):
        sh = x.shape; x2 = x.reshape(-1, sh[-1]).float()
        asc = x2.abs().amax(dim=1).clamp(min=1e-8) / torch.tensor(127.0, dtype=torch.float32, device=x2.device)
        x8 = torch.clamp(torch.round(x2 / asc[:, None]), -127, 127).to(torch.int8)
        n = x8.shape[0]; pad = (17 - n) if n < 17 else 0   # incl. n==0 (MoE expert routed no tokens)
        if pad: x8 = torch.cat([x8, torch.zeros(pad, x8.shape[1], dtype=torch.int8, device=x8.device)])
        y = torch._int_mm(x8, W8.t())[:n].float() * asc[:, None] * ws[None, :]
        if bias is not None: y = y + bias.float()
        return y.reshape(*sh[:-1], W8.shape[0]).to(x.dtype)   # explicit dim: n==0 rows legal
    lin.forward = fwd

def apply_w8a8(model):
    nrep, nskip = 0, 0
    for name, mod in model.model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if name.split(".")[-1] == "gate":       # MoE router stays fp — pre-registered (EVAL_PLAN)
                nskip += 1; continue
            w8a8_linear(mod); nrep += 1
    print("W8A8 linears replaced: %d (lm_head excluded, %d MoE routers skipped)" % (nrep, nskip), flush=True)
    return nrep, nskip

def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation="eager",
        device_map="auto", low_cpu_mem_usage=True).eval()
    return tok, model

def input_device(model):
    return model.get_input_embeddings().weight.device

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def provenance(extra=None):
    import transformers
    gpus = []
    try:
        for i in range(torch.cuda.device_count()):
            gpus.append(torch.cuda.get_device_name(i))
    except Exception: pass
    p = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "torch": torch.__version__, "transformers": transformers.__version__,
         "cuda": getattr(torch.version, "cuda", None), "gpus": gpus,
         "host": platform.node(), "frac_sha256": frac_check(),
         "tf32_matmul": torch.backends.cuda.matmul.allow_tf32}
    if extra: p.update(extra)
    return p
