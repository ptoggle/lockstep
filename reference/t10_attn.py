#!/usr/bin/env python3
"""
T10 - long-context contract attention (blocked), plus the long-context engineering fixes.

Blocked, flash-style, TWO-PASS evaluation of exactly the arithmetic that
contract_common.make_contract_attention (arm C) and verify_arm_g.make_attn (arms G/H/J)
evaluate with a materialized [T,T] score matrix.  Necessary because at T=131072 one head's
int64 score matrix is 137 GiB.

WHY THE BLOCKING IS BIT-IDENTICAL (gated in t10_gates.py, never assumed):
  * m = max_j SG[i,j] is an INTEGER max -> associative -> any j order, any tiling;
  * NB, d, index, shift, table lookup are per-element;
  * l = sum_j eh is an INTEGER sum in int64: eh < 2^w <= 2^15, T <= 2^17 -> l < 2^32;
  * A = sum_j eh*v8[j] accumulates in float64 over INTEGERS with |eh*v8| <= 2^15*2^7 = 2^22
    and T <= 2^17, so |A| <= 2^39 << 2^53: every partial sum is exactly representable, the
    f64 accumulation is exact and order-independent (the contract's own PV argument,
    contract_common header).  Hence block-accumulating A and l is exact.
  * d is clamped to [0, win] (win = (w+1)<<bhi = 2^30 at w=15,bhi=26) before the lookup, so
    the lookup/shift stage runs in int32 with IDENTICAL values -- pure traffic reduction.

KINDS
  C  the contract as filed: no offset, ONE pow2 K exponent per (kv head, window)
  G  + published per-(layer,kv-head,channel) key offset mu           (the T9 fix)
  R  + per-ROW (per key token) K exponents.  SPEC 6.1.5-legal.  The per-key pow2 scale is
     folded into the temperature G BEFORE the row max, so every SG[i,j] in a row is on ONE
     common fixed-point log2 scale and the integer comparison across keys is preserved
     (see T10_LOG.md section on exactness).
  H  = G + declared Walsh-Hadamard pre-transform on q and k  (fold factor dh = 2^7)
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import contract_common as C

try:
    import verify_arm_g as VG          # reuse T9's quant_rows / fwht verbatim
    quant_rows = VG.quant_rows
    fwht = VG.fwht
except Exception:                       # pragma: no cover - identical bodies, gated in t10_gates
    def quant_rows(x):
        m = x.abs().amax(dim=-1)
        mant, E = torch.frexp(m)
        e = 6 - (E.to(torch.int64) - 1)
        e = torch.where(m == 0, torch.zeros_like(e), e)
        over = m * C.pow2_t(e) > 127.0
        e = (e - over.to(torch.int64)).clamp(-32, 40)
        q = torch.clamp(torch.round(x * C.pow2_t(e)[..., None]), -127, 127)
        return q, e

    def fwht(x):
        orig = x.shape; n = orig[-1]; y = x.reshape(-1, n); h = 1
        while h < n:
            y = y.view(-1, n // (2 * h), 2, h)
            u = y[:, :, 0, :]; v = y[:, :, 1, :]
            y = torch.stack((u + v, u - v), dim=2).reshape(-1, n)
            h *= 2
        return y.view(orig)

_frac32 = {}
def frac32_on(dev, n, w):
    k = (str(dev), n, w)
    if k not in _frac32:
        _frac32[k] = C.table_on(dev, n, w).to(torch.int32).contiguous()
    return _frac32[k]


def make_long_contract_attn(kind, bhi=26, n=13, w=15, head_chunk=7, qblk=1024, kblk=8192,
                            mu=None, stats=None):
    """Return an eager_attention_forward replacement.  mu: [L,KVH,dh] fp32 (CPU ok)."""
    assert kind in ("C", "G", "R", "H")
    assert n <= bhi
    assert (kind == "C") == (mu is None), "kinds G/R/H need mu; kind C must not have one"
    win = (w + 1) << bhi
    idx_sh = bhi - n
    idx_mask = (1 << n) - 1
    _mu_cache = {}

    def mu_on(li, dev):
        k = (li, str(dev))
        if k not in _mu_cache:
            _mu_cache[k] = mu[li].to(dev)
        return _mu_cache[k]

    def attn(module, query, key, value, attention_mask, *args, **kw):
        B, H, T, dh = query.shape
        assert B == 1, "B=1 windows only"
        dev = query.device
        li = int(getattr(module, "layer_idx", -1))
        KVH = key.shape[1]
        grp = H // KVH
        qf = query[0].float()
        kf = key[0].float()
        if mu is not None:
            kf = kf - mu_on(li, dev)[:, None, :]
        vf = value[0].float()
        extra = 0
        if kind == "H":
            qf = fwht(qf); kf = fwht(kf)
            extra = int(round(math.log2(dh)))         # (Hq).(Hk) = dh*(q.k), dh a power of 2
        v8, ev = C.quant_pow2(vf)
        if kind == "R":
            k8, ekr = quant_rows(kf)                  # ekr [KVH,T] - per KEY TOKEN
            ek = None
        else:
            k8, ek = C.quant_pow2(kf)                 # ek  [KVH]   - per KV HEAD
            ekr = None
        del kf, vf
        frac = frac32_on(dev, n, w)
        ar = torch.arange(T, device=dev)
        zero32 = torch.zeros((), dtype=torch.int32, device=dev)
        winT = torch.tensor(win, dtype=torch.int64, device=dev)
        outs = []
        for h0 in range(0, H, head_chunk):
            h1 = min(h0 + head_chunk, H)
            hc = h1 - h0
            q8, eq = C.quant_pow2(qf[h0:h1])          # eq [hc], per QUERY HEAD
            kvi = torch.arange(h0, h1, device=dev) // grp
            K8 = k8[kvi]                              # [hc,T,dh]
            V8 = v8[kvi]
            EV = ev[kvi]
            if kind == "R":
                Gj = C.g_fold_t(dh, eq[:, None] + ekr[kvi] + extra, bhi)      # [hc,T]
                if stats is not None:
                    sh_all = eq[:, None] + ekr[kvi] + extra
                    stats["sh_min"] = min(stats.get("sh_min", 10**9), int(sh_all.min()))
                    stats["sh_max"] = max(stats.get("sh_max", -10**9), int(sh_all.max()))
                    stats["G_clamp_lo"] = stats.get("G_clamp_lo", 0) + int((Gj == 1).sum())
                    stats["G_clamp_hi"] = stats.get("G_clamp_hi", 0) + int((Gj == (1 << 30)).sum())
                    stats["G_n"] = stats.get("G_n", 0) + int(Gj.numel())
            else:
                Gh = C.g_fold_t(dh, eq + ek[kvi] + extra, bhi)[:, None, None]  # [hc,1,1]
            OUT = torch.empty((hc, T, dh), dtype=torch.float32, device=dev)
            for i0 in range(0, T, qblk):
                i1 = min(i0 + qblk, T)
                q8b = q8[:, i0:i1]
                rows = ar[i0:i1]
                # ---------------- pass 1: exact integer row max of SG over valid j
                m = torch.full((hc, i1 - i0, 1), -(1 << 62), dtype=torch.int64, device=dev)
                for j0 in range(0, i1, kblk):
                    j1 = min(j0 + kblk, i1)
                    S = torch.matmul(q8b, K8[:, j0:j1].transpose(-1, -2))     # fp32, |S|<2^24 exact
                    SG = S.to(torch.int64) * (Gj[:, None, j0:j1] if kind == "R" else Gh)
                    if j1 > i0:                                                # diagonal tile
                        valid = ar[j0:j1][None, :] <= rows[:, None]
                        SG = SG.masked_fill(~valid[None], -(1 << 62))
                    m = torch.maximum(m, SG.amax(-1, keepdim=True))
                    del S, SG
                NB = -((-m) >> bhi) << bhi
                # ---------------- pass 2: window, table, integer normalizer, exact f64 PV
                l = torch.zeros((hc, i1 - i0), dtype=torch.int64, device=dev)
                A = torch.zeros((hc, i1 - i0, dh), dtype=torch.float64, device=dev)
                for j0 in range(0, i1, kblk):
                    j1 = min(j0 + kblk, i1)
                    S = torch.matmul(q8b, K8[:, j0:j1].transpose(-1, -2))
                    d = NB - S.to(torch.int64) * (Gj[:, None, j0:j1] if kind == "R" else Gh)
                    if j1 > i0:
                        valid = ar[j0:j1][None, :] <= rows[:, None]
                        d = torch.where(valid[None], d, winT)
                    d = torch.clamp(d, max=win).to(torch.int32)                # in [0, win]
                    sel = frac.index_select(0, ((d >> idx_sh) & idx_mask).reshape(-1)).view_as(d)
                    eh = torch.where(d < win, sel >> (d >> bhi), zero32)
                    l += eh.sum(-1, dtype=torch.int64)
                    A += torch.matmul(eh.double(), V8[:, j0:j1].double())
                    del S, d, sel, eh
                of = A.to(torch.float32) / l.to(torch.float32)[..., None]
                OUT[:, i0:i1] = of * (2.0 ** (-EV.to(torch.float32)))[:, None, None]
            outs.append(OUT)
            del q8, K8, V8
        out = torch.cat(outs, 0)
        return out.to(query.dtype)[None].transpose(1, 2).contiguous(), None
    return attn


# ------------------------------------------------------------------ arm E (deployed)
def sdpa_attn(module, query, key, value, attention_mask, *args, **kw):
    """Arm E, byte-for-byte the function verify_arm_e.py uses, with one addition: above
    2^30 elements per repeated head block the call is split PER KV HEAD.  Attention is
    per-head, so the split is bit-identical to the unsplit call (phase1_tau.py, same fix)."""
    B, H, T, dh = query.shape
    KVH = key.shape[1]; grp = H // KVH
    if B * H * T * dh > (1 << 30):
        outs = []
        for j in range(KVH):
            o = torch.nn.functional.scaled_dot_product_attention(
                query[:, j * grp:(j + 1) * grp],
                key[:, j:j + 1].expand(B, grp, T, dh),
                value[:, j:j + 1].expand(B, grp, T, dh), is_causal=True)
            outs.append(o.transpose(1, 2).contiguous())
        return torch.cat(outs, dim=2), None
    k = key.repeat_interleave(grp, 1); v = value.repeat_interleave(grp, 1)
    o = torch.nn.functional.scaled_dot_product_attention(query, k, v, is_causal=True)
    return o.transpose(1, 2).contiguous(), None


# ------------------------------------------------------------------ long-context fixes
def install_longctx(mlp_chunk=16384):
    """The engineering fixes from phase1_tau.py, applied identically to EVERY arm:
      * amputate the model-level causal-mask builder (a [T,T] fp mask is 17 GiB at 131072;
        full windows + is_causal / our own tiled predicate need no mask tensor at all);
      * chunk the MLP over tokens (pointwise per token, so no cross-token reduction);
      * flash-only sdpa backend (the mem-efficient backend has int32-offset illegal
        accesses at long T; the math backend would materialize T^2)."""
    import transformers.models.qwen2.modeling_qwen2 as mq
    mq.Qwen2Model._update_causal_mask = lambda self, *a, **kw: None
    if not getattr(mq.Qwen2MLP, "_t10_chunked", False):
        _orig_mlp = mq.Qwen2MLP.forward
        def _chunked(self, x, _CH=mlp_chunk):
            if x.shape[1] <= _CH:
                return _orig_mlp(self, x)
            return torch.cat([_orig_mlp(self, x[:, i:i + _CH]) for i in range(0, x.shape[1], _CH)], dim=1)
        mq.Qwen2MLP.forward = _chunked
        mq.Qwen2MLP._t10_chunked = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)


def load_model_longctx(model_id, attn_impl="eager"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
        device_map="auto", low_cpu_mem_usage=True, use_sliding_window=False).eval()
    return tok, model
