#!/usr/bin/env python3
"""Lockstep phase II: the end-to-end bit fingerprint of a served model.

A manifest directory (docs/PLAN_PLUGIN.md) carries `probe.json`: a fixed prompt of token
ids, the per-position log-prob of each prompt token as recorded by the engine that
PREPARED the manifest, and the first 8 greedy tokens. `check_engine` replays that prompt
on a running engine and compares (a) every prompt log-prob as a bf16 bit pattern and
(b) the greedy ids. Any difference is a FAIL: the engine is not computing the function
the manifest was prepared for, and a verifier replaying the same manifest would reject it.

Where it runs
  1. `lockstep_prepare.py` calls `write_probe(llm, manifest_dir)` (records the fingerprint);
  2. CLI: `python lockstep_selftest.py <manifest_dir> --url http://127.0.0.1:8000 [--model NAME]`
     against a running `vllm serve` (exit 0 PASS, 3 FAIL; `LOCKSTEP_SELFTEST=warn` -> exit 0);
  3. `pod/lockstep_serve.sh` (serve + wait healthy + this CLI; kills the server on FAIL);
  4. in-process: `check_engine(llm, manifest_dir)`.

probe.json (probe_version 1)
  {"probe_version": 1,
   "ids": [<probe_len token ids>],
   "prompt_logprobs_bits": [null, <int: bf16 bit pattern of log p(ids[i] | ids[:i])>, ...],
   "prompt_logprobs": [null, <the same as floats, for humans>, ...],
   "generated_ids": [<up to 8 greedy ids>],
   "sampling": {"temperature": 0, "max_tokens": 8, "prompt_logprobs": 0},
   "model": "<hf id>"}
  Position 0 has no log-prob (vLLM returns None there). The floats are the engine's fp32
  log-probs; the comparison is on their bf16 rounding (round-to-nearest-even), which is
  exactly `torch.tensor(x).to(torch.bfloat16).view(torch.int16) & 0xffff`.

The HTTP request (OpenAI-compatible completions; token ids are accepted as the prompt):
  POST <url>/v1/completions
  {"model": <served name>, "prompt": [<ids>], "max_tokens": 8, "temperature": 0.0,
   "echo": true, "logprobs": 0, "return_token_ids": true}
  -> choices[0].logprobs.token_logprobs = [null, lp_1, ..., lp_{n-1}, <8 generated lps>]
     (with echo, vLLM prepends the prompt log-probs; `logprobs: 0` = the actual token only),
     choices[0].token_ids = the generated ids (vLLM's `return_token_ids` extension; if the
     server lacks it, the completion text is re-tokenised with --tokenizer and the report
     says so in "generated_ids_source").

In-engine per-op identity stays `t16l_egates.py`; this is the fingerprint a verifier sees.
vLLM is imported lazily (inside the functions that need it), so the CLI runs anywhere.
"""
import os
import sys
import json
import time
import struct
import hashlib
import argparse
import urllib.request
import urllib.error

PROBE_VERSION = 1
PROBE_FILE = "probe.json"
MANIFEST_FILE = "lockstep.manifest.json"
PROBE_MAX_TOKENS = 8
PROBE_SAMPLING = {"temperature": 0, "max_tokens": PROBE_MAX_TOKENS, "prompt_logprobs": 0}
FIXED_ID_BASE = 1000          # probe_ids without a corpus: 1000 .. 1000+n-1

EXIT_PASS = 0
EXIT_FAIL = 3

_REGISTERED = None            # on_plugin_registered's record (one per process)


# --------------------------------------------------------------------------- bits
def bf16_bits(x):
    """bf16 bit pattern (int in [0, 0xffff]) of the float x, round-to-nearest-even.

    Identical to `int(torch.tensor(x).to(torch.bfloat16).view(torch.int16)) & 0xffff`:
    x (a Python double) is first rounded to fp32 (what torch.tensor does), then the low 16
    bits are rounded away with ties to even. None stays None (position 0). Every NaN maps
    to torch's canonical 0x7fc0, so two NaNs compare equal."""
    if x is None:
        return None
    b = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    if (b & 0x7f800000) == 0x7f800000 and (b & 0x007fffff):
        return 0x7fc0                                       # NaN: c10's canonical quiet NaN
    lo, hi = b & 0xffff, b >> 16
    if lo > 0x8000 or (lo == 0x8000 and (hi & 1)):
        hi += 1
    return hi & 0xffff


def bf16_bits_torch(x):
    """The torch form of bf16_bits (for the maintenance check; needs torch)."""
    import torch                                            # noqa: PLC0415
    if x is None:
        return None
    return int(torch.tensor(float(x), dtype=torch.float32).to(torch.bfloat16)
               .view(torch.int16).item()) & 0xffff


# --------------------------------------------------------------------------- probe ids
def probe_ids(hf_id_or_tokenizer, n, corpus=None):
    """Deterministic probe prompt of n token ids.

    corpus given (a path to a text file, or the text itself when it is not a path):
    the corpus's first n tokens under the model's tokenizer (no special tokens).
    corpus None: the fixed range FIXED_ID_BASE .. FIXED_ID_BASE+n-1 (no tokenizer needed;
    every model with a vocabulary above 1000+n accepts it)."""
    n = int(n)
    if n < 2:
        raise ValueError("probe_len must be >= 2 (position 0 carries no log-prob)")
    if corpus is None:
        return list(range(FIXED_ID_BASE, FIXED_ID_BASE + n))
    text = corpus
    if isinstance(corpus, str) and os.path.exists(corpus):
        with open(corpus, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    tok = hf_id_or_tokenizer
    if isinstance(tok, str):
        from transformers import AutoTokenizer                 # noqa: PLC0415
        tok = AutoTokenizer.from_pretrained(tok)
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) < n:
        raise ValueError("corpus has %d tokens, probe needs %d" % (len(ids), n))
    return [int(t) for t in ids[:n]]


# --------------------------------------------------------------------------- run: in-process
def run_probe_inprocess(llm, ids):
    """(prompt log-probs as floats with None at position 0, generated ids) from a vLLM LLM.

    SamplingParams(temperature=0.0, max_tokens=8, prompt_logprobs=0): the log-prob of the
    actual token at position i is out.prompt_logprobs[i][ids[i]].logprob; position 0 is None."""
    from vllm import SamplingParams                            # noqa: PLC0415
    from vllm.inputs import TokensPrompt                       # noqa: PLC0415
    ids = [int(t) for t in ids]
    sp = SamplingParams(temperature=0.0, max_tokens=PROBE_MAX_TOKENS, prompt_logprobs=0)
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)[0]
    plp = out.prompt_logprobs
    if plp is None or len(plp) < len(ids):
        raise RuntimeError("engine returned %s prompt log-probs for %d ids"
                           % (None if plp is None else len(plp), len(ids)))
    lps = [None]
    for i in range(1, len(ids)):
        d = plp[i]
        e = d.get(ids[i]) if d is not None else None
        if e is None:
            raise RuntimeError("no prompt log-prob for position %d (token %d)" % (i, ids[i]))
        lps.append(float(getattr(e, "logprob", e)))
    gen = [int(t) for t in out.outputs[0].token_ids]
    return lps, gen


# --------------------------------------------------------------------------- run: HTTP
def _http_json(url, payload=None, timeout=600.0):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:800]
        raise RuntimeError("HTTP %d from %s: %s" % (e.code, url, body)) from None


def served_models(url, timeout=30.0):
    """The ids the server lists at /v1/models."""
    d = _http_json(url.rstrip("/") + "/v1/models", timeout=timeout)
    return [m["id"] for m in d.get("data", [])]


def run_probe_http(url, model_name, ids, timeout=600.0, tokenizer=None):
    """Same as run_probe_inprocess, through the OpenAI-compatible /v1/completions of a
    running vLLM server. Returns (lps, gen) and sets run_probe_http.last_source to where
    the generated ids came from ("return_token_ids" or "retokenized")."""
    ids = [int(t) for t in ids]
    req = {"model": model_name, "prompt": ids, "max_tokens": PROBE_MAX_TOKENS,
           "temperature": 0.0, "echo": True, "logprobs": 0, "return_token_ids": True}
    d = _http_json(url.rstrip("/") + "/v1/completions", req, timeout=timeout)
    ch = d["choices"][0]
    lp = ch.get("logprobs") or {}
    tl = lp.get("token_logprobs")
    if tl is None or len(tl) < len(ids):
        raise RuntimeError("server returned %s token_logprobs for a %d-token echo"
                           % (None if tl is None else len(tl), len(ids)))
    got_prompt = ch.get("prompt_token_ids")
    if got_prompt is not None and [int(t) for t in got_prompt] != ids:
        raise RuntimeError("server echoed a different prompt (%d ids, first %s)"
                           % (len(got_prompt), got_prompt[:4]))
    lps = [None] + [float(v) for v in tl[1:len(ids)]]
    if tl[0] is not None:
        # vLLM's echo puts null at the first prompt position; another server might not
        lps[0] = None
    gen = ch.get("token_ids")
    if gen is not None:
        gen = [int(t) for t in gen]
        run_probe_http.last_source = "return_token_ids"
    else:
        if tokenizer is None:
            raise RuntimeError("server did not return token_ids (no return_token_ids "
                               "support); pass tokenizer= / --tokenizer to re-tokenise")
        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer             # noqa: PLC0415
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        gen = [int(t) for t in tokenizer.encode(ch.get("text", ""), add_special_tokens=False)]
        run_probe_http.last_source = "retokenized"
    return lps, gen


run_probe_http.last_source = None


# --------------------------------------------------------------------------- probe file
def _model_id_of(llm):
    for path in (("llm_engine", "model_config", "model"), ("llm_engine", "vllm_config",
                                                          "model_config", "model")):
        o = llm
        try:
            for a in path:
                o = getattr(o, a)
            if isinstance(o, str):
                return o
        except AttributeError:
            pass
    return None


def make_probe(ids, lps, gen, model):
    return {"probe_version": PROBE_VERSION,
            "ids": [int(t) for t in ids],
            "prompt_logprobs_bits": [bf16_bits(x) for x in lps],
            "prompt_logprobs": [None if x is None else float(x) for x in lps],
            "generated_ids": [int(t) for t in gen],
            "sampling": dict(PROBE_SAMPLING),
            "model": model}


def write_probe(llm, manifest_dir, probe_len=256, ids=None):
    """Run the probe on the (armed) in-process engine twice, require the two runs to agree
    bit for bit, and write <manifest_dir>/probe.json. Returns the probe dict. The manifest's
    hash of probe.json is written by lockstep_prepare (step 6), not here."""
    model = _model_id_of(llm)
    if ids is None:
        ids = probe_ids(model, probe_len)
    ids = [int(t) for t in ids]
    lps, gen = run_probe_inprocess(llm, ids)
    lps2, gen2 = run_probe_inprocess(llm, ids)
    rep = compare(make_probe(ids, lps, gen, model), lps2, gen2)
    if not rep["pass"]:
        raise RuntimeError("probe is not self-consistent across two runs: %s" % json.dumps(rep))
    probe = make_probe(ids, lps, gen, model)
    os.makedirs(manifest_dir, exist_ok=True)
    path = os.path.join(manifest_dir, PROBE_FILE)
    with open(path, "w") as f:
        json.dump(probe, f, indent=1)
    print("LOCKSTEP-SELFTEST wrote %s: %d ids, %d generated, model=%s"
          % (path, len(ids), len(gen), model), flush=True)
    return probe


def _sha256_file(path):
    try:
        import lockstep_manifest                               # noqa: PLC0415
        return lockstep_manifest.sha256_file(path)
    except (ImportError, AttributeError):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()


def _read_manifest(manifest_dir):
    try:
        import lockstep_manifest                               # noqa: PLC0415
        return lockstep_manifest.read_manifest(manifest_dir)
    except (ImportError, AttributeError):
        with open(os.path.join(manifest_dir, MANIFEST_FILE)) as f:
            return json.load(f)


def verify_probe_file(manifest_dir):
    """probe.json's sha256 against lockstep.manifest.json. Returns
    {"ok", "file", "expected", "got"}; a missing entry or file is ok=False. When the
    manifest does not exist YET (lockstep_prepare runs the self-test at step 5, before it
    hashes everything at step 6) ok is None: unverified, not tampered."""
    if not os.path.exists(os.path.join(manifest_dir, MANIFEST_FILE)):
        path = os.path.join(manifest_dir, PROBE_FILE)
        return {"ok": None, "file": path, "expected": None,
                "got": _sha256_file(path) if os.path.exists(path) else None,
                "note": "no %s yet (prepare time): probe unverified" % MANIFEST_FILE}
    m = _read_manifest(manifest_dir)
    ent = m.get("probe") or {}
    fname = ent.get("file", PROBE_FILE)
    path = os.path.join(manifest_dir, fname)
    exp = ent.get("sha256")
    got = _sha256_file(path) if os.path.exists(path) else None
    return {"ok": bool(exp) and got is not None and got == exp, "file": path,
            "expected": exp, "got": got}


def _normalise_probe(p):
    """Accept this module's probe_version-1 layout, and lockstep_prepare's fallback layout
    ("format": "lockstep-probe-v1": bits as 4-hex-digit strings) by converting it."""
    if p.get("probe_version") == PROBE_VERSION:
        return p
    if p.get("format") == "lockstep-probe-v1":
        q = dict(p)
        q["probe_version"] = PROBE_VERSION
        q["prompt_logprobs_bits"] = [None if b is None else (int(b, 16) if isinstance(b, str) else int(b))
                                     for b in p["prompt_logprobs_bits"]]
        q.setdefault("sampling", {"temperature": p.get("temperature", 0),
                                  "max_tokens": p.get("max_tokens", PROBE_MAX_TOKENS),
                                  "prompt_logprobs": 0})
        return q
    raise ValueError("unknown probe layout: probe_version=%r format=%r (this module reads "
                     "probe_version %d)" % (p.get("probe_version"), p.get("format"), PROBE_VERSION))


def load_probe(manifest_dir, verify=True):
    """The probe dict; with verify=True its sha256 must match the manifest first (a
    manifest that does not exist yet leaves the probe unverified, see verify_probe_file)."""
    if verify:
        v = verify_probe_file(manifest_dir)
        if v["ok"] is False:
            raise ValueError("probe.json sha256 mismatch in %s: manifest %s, file %s"
                             % (manifest_dir, v["expected"], v["got"]))
        path = v["file"]
    else:
        path = os.path.join(manifest_dir, PROBE_FILE)
    with open(path) as f:
        p = json.load(f)
    return _normalise_probe(p)


# --------------------------------------------------------------------------- compare
def compare(expected_probe, got_logprobs, got_ids):
    """{"pass", "n_positions", "n_bits_differ", "first_diff_position", "generated_equal",
    "max_abs_diff"} for the engine's (log-probs, generated ids) against the probe."""
    exp_bits = expected_probe["prompt_logprobs_bits"]
    exp_f = expected_probe.get("prompt_logprobs") or [None] * len(exp_bits)
    got_bits = [bf16_bits(x) for x in got_logprobs]
    n = len(exp_bits)
    rep = {"pass": False, "n_positions": n, "n_bits_differ": 0, "first_diff_position": None,
           "generated_equal": False, "max_abs_diff": 0.0}
    if len(got_bits) != n:
        rep["error"] = "got %d log-probs, probe has %d" % (len(got_bits), n)
        rep["n_bits_differ"] = n
        rep["first_diff_position"] = 0
        return rep
    mx = 0.0
    for i in range(n):
        if exp_bits[i] != got_bits[i]:
            rep["n_bits_differ"] += 1
            if rep["first_diff_position"] is None:
                rep["first_diff_position"] = i
                rep["first_diff"] = {"expected_bits": exp_bits[i], "got_bits": got_bits[i],
                                     "expected": exp_f[i], "got": got_logprobs[i]}
        a, b = exp_f[i], got_logprobs[i]
        if a is not None and b is not None:
            d = abs(float(a) - float(b))
            if d != d or d > mx:                                # NaN counts as infinite
                mx = float("inf") if d != d else d
    rep["max_abs_diff"] = mx
    exp_gen = [int(t) for t in expected_probe["generated_ids"]]
    got_gen = [int(t) for t in got_ids]
    rep["generated_equal"] = (exp_gen == got_gen)
    if not rep["generated_equal"]:
        rep["generated_expected"] = exp_gen
        rep["generated_got"] = got_gen
    rep["pass"] = (rep["n_bits_differ"] == 0 and rep["generated_equal"])
    return rep


# --------------------------------------------------------------------------- check
def check_engine(llm_or_url, manifest_dir, model_name=None, tokenizer=None, timeout=600.0):
    """The self-test. In-process when given an object with .generate (a vLLM LLM), over
    HTTP when given a URL string. probe.json's sha256 is checked against the manifest
    BEFORE the probe is trusted; on a mismatch the report fails without touching the engine
    (no manifest file yet, as at prepare step 5: unverified, probe_sha256_ok None).
    Never raises for an engine-side failure: the report carries "error"."""
    t0 = time.time()
    # probe_sha256_ok: True verified, False MISMATCH (tampered), None not verifiable (no
    # manifest yet, or the manifest itself failed to read; the error says which)
    rep = {"pass": False, "manifest_dir": manifest_dir, "probe_sha256_ok": None}
    try:
        v = verify_probe_file(manifest_dir)
        rep["probe_sha256_ok"] = v["ok"]
        if v.get("note"):
            rep["probe_sha256_note"] = v["note"]
        if v["ok"] is False:
            rep["error"] = ("probe.json sha256 mismatch: manifest %s, file %s"
                            % (v["expected"], v["got"]))
            return rep
        probe = load_probe(manifest_dir, verify=False)
        ids = probe["ids"]
        rep["probe_model"] = probe.get("model")
        if hasattr(llm_or_url, "generate"):
            rep["mode"] = "inprocess"
            lps, gen = run_probe_inprocess(llm_or_url, ids)
        elif isinstance(llm_or_url, str):
            rep["mode"] = "http"
            url = llm_or_url
            if model_name is None:
                names = served_models(url)
                model_name = (probe.get("model") if probe.get("model") in names
                              else (names[0] if names else probe.get("model")))
            rep["model_name"] = model_name
            lps, gen = run_probe_http(url, model_name, ids, timeout=timeout,
                                      tokenizer=tokenizer or probe.get("model"))
            rep["generated_ids_source"] = run_probe_http.last_source
        else:
            raise TypeError("check_engine wants a vLLM LLM or a URL string, got %r"
                            % (type(llm_or_url),))
        rep.update(compare(probe, lps, gen))
    except Exception as e:                                       # noqa: BLE001
        rep["pass"] = False
        rep["error"] = "%s: %s" % (type(e).__name__, e)
    rep["elapsed_s"] = round(time.time() - t0, 3)
    return rep


def on_plugin_registered(config, manifest):
    """Hook for lockstep_vllm_plugin.register(): records that a manifest-armed engine is
    being built and which self-test mode applies. It runs NO check: at register time there
    is no engine yet. The check runs at engine level (check_engine in-process, or the CLI
    from pod/lockstep_serve.sh once /health is up)."""
    global _REGISTERED
    mode = getattr(config, "selftest", None) or os.environ.get("LOCKSTEP_SELFTEST", "strict")
    mdir = None
    if isinstance(manifest, dict):
        mdir = manifest.get("dir") or manifest.get("manifest_dir")
    mdir = mdir or os.environ.get("LOCKSTEP_MANIFEST")
    _REGISTERED = {"time": time.time(), "pid": os.getpid(), "manifest_dir": mdir,
                   "selftest": mode,
                   "model": (manifest.get("model") or {}).get("hf_id") if isinstance(manifest, dict) else None}
    print("LOCKSTEP-SELFTEST armed mode=%s manifest=%s (engine-level check pending)"
          % (mode, mdir), flush=True)
    return _REGISTERED


# --------------------------------------------------------------------------- CLI
def _main(argv=None):
    ap = argparse.ArgumentParser(description="Lockstep self-test: replay the manifest's probe "
                                 "on a running vllm serve and compare bit for bit.")
    ap.add_argument("manifest_dir")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None, help="served model name (default: the probe's "
                    "hf id if served, else the first id at /v1/models)")
    ap.add_argument("--tokenizer", default=None, help="hf id/path used only if the server "
                    "lacks return_token_ids")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default=None, help="write the report json here too")
    ap.add_argument("--check-bf16", action="store_true", help="maintenance: compare bf16_bits "
                    "with the torch form on 200k values and exit")
    a = ap.parse_args(argv)
    if a.check_bf16:
        import random                                          # noqa: PLC0415
        rnd = random.Random(0)
        vals = [rnd.uniform(-40, 1) for _ in range(100000)]
        vals += [-float(struct.unpack("<f", struct.pack("<I", (rnd.getrandbits(32) & 0x7fffffff) | 0x8000))[0])
                 for _ in range(100000)]
        vals += [0.0, -0.0, 1e-45, -1e-45, 3.4e38, float("inf"), -float("inf"), float("nan")]
        bad = sum(1 for v in vals if bf16_bits(v) != bf16_bits_torch(v))
        print("LOCKSTEP-SELFTEST bf16_bits vs torch: %d/%d differ" % (bad, len(vals)))
        return 0 if bad == 0 else 1
    rep = check_engine(a.url, a.manifest_dir, model_name=a.model, tokenizer=a.tokenizer,
                       timeout=a.timeout)
    rep["url"] = a.url
    mode = os.environ.get("LOCKSTEP_SELFTEST", "strict").lower()
    rep["selftest_mode"] = mode
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    verdict = "PASS" if rep["pass"] else "FAIL"
    print("LOCKSTEP-SELFTEST %s %s" % (verdict, json.dumps(rep, sort_keys=True)), flush=True)
    if rep["pass"]:
        return EXIT_PASS
    return EXIT_PASS if mode == "warn" else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(_main())
