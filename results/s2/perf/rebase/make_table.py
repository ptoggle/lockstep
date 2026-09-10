"""results/s2/perf/rebase: the Stage 2 item 2.1 re-baseline against a CLEAN stock (2026-09-09).
Every lockstep arm armed FROM ITS MANIFEST (lockstep_config.py --env; the served configuration: contract v2.5, LSSA-B9.1,
R1 online, 8 outliers), every stock arm in a shell with every LOCKSTEP_/T16/LSSAB variable unset. Regenerates README.md."""
import json, glob, os
D = os.path.dirname(os.path.abspath(__file__))
def ratio(r):
    return r["stock"] / r["lssab"] if r["unit"] == "ms" else r["lssab"] / r["stock"]
COLS = ["ttft_p128_b1", "ttft_p128_b8", "ttft_p2048_b1", "ttft_p2048_b8", "ttft_p8192_b1", "ttft_p8192_b8", "ttft_p8192_b4",
        "ttft_p32640_b1", "prefill_p128_b32", "decode_b1", "decode_b8", "decode_b32"]
out = ["# Re-baseline against a clean stock (2026-09-09)", "",
       "Single-stream, one H100 SXM, both stacks in full CUDA graphs without inductor, REP=3 medians; lockstep over stock",
       "(above 1.0 is faster than stock). `single_<model>.json` holds both arms; `single_<model>_stock2.json` is the stock arm",
       "repeated after the lockstep arm (Qwen2.5-7B). Hosts: lc-handover (7B, Llama, Qwen3, Phi, Mistral, 1M) at load 11-29 on",
       "208 cores; lc-sweep (the Qwen2.5 size sweep) at load 11-17.", "",
       "| model | " + " | ".join(c.replace("ttft_", "TTFT ").replace("_b", " x") for c in COLS) + " |",
       "|---|" + "---|" * len(COLS)]
for f in sorted(glob.glob(D + "/single_*.json")):
    b = os.path.basename(f)[7:-5]
    if b.endswith(("_stock", "_lssab", "_stock2")): continue
    rows = {r["metric"]: r for r in json.load(open(f))["rows"] if "stock" in r and "lssab" in r}
    out.append("| %s | " % b + " | ".join("%.3f" % ratio(rows[c]) if c in rows else "" for c in COLS) + " |")
out += ["", "Saturated serving (64 concurrent, `t14_bench.py`, output tokens per second, lockstep over stock):", "",
        "| model | W1 ShareGPT | W2 decode-heavy | W3 prefill-heavy |", "|---|---|---|---|"]
for n in ("qwen25_7b", "llama31_8b"):
    cells = []
    for wl in ("w1_rinf", "w2", "w3"):
        try:
            s = json.load(open("%s/stock_repRB_%s_%s.json" % (D, n, wl))); l = json.load(open("%s/lssab_repRB_%s_%s.json" % (D, n, wl)))
            cells.append("%.3f (%.0f / %.0f)" % (l["output_throughput"] / s["output_throughput"], l["output_throughput"], s["output_throughput"]))
        except Exception: cells.append("")
    out.append("| %s | %s |" % (n, " | ".join(cells)))
try:
    s = json.load(open(D + "/long_stock.json")); l = json.load(open(D + "/long_lssab.json"))
    out += ["", "Long context, Qwen2.5-7B-Instruct-1M (dense), `t16f_long.py`, its own manifest (`rb_qwen1m`, key offset calibrated by prepare):", "",
            "| row | stock | lockstep | ratio |", "|---|---|---|---|"]
    sr = {r["metric"]: r for r in s["rows"]}; lr = {r["metric"]: r for r in l["rows"]}
    def val(r):
        for k in ("ttft_ms", "tok_s", "toks_per_s", "decode_tok_s", "value"):
            if k in r: return k, r[k]
        for k, v in r.items():
            if isinstance(v, (int, float)) and k not in ("plen", "attn_calls"): return k, v
        return None, None
    for k in sr:
        if k not in lr: continue
        ka, av = val(sr[k]); kb, bv = val(lr[k])
        if av is None or bv is None: continue
        rt = av / bv if "ttft" in k else bv / av
        out.append("| %s (%s) | %.2f | %.2f | %.3f |" % (k, ka, av, bv, rt))
except Exception as e:
    out.append("\n(long-context table: %s)" % e)
open(D + "/README.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
