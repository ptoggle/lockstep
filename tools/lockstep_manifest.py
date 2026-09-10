"""Lockstep manifest (plugin phase I): the on-disk unit of deployment.

A manifest DIRECTORY holds

    lockstep.manifest.json     schema below; sha256 of every file and every tensor
    constants.safetensors      the armed constants exactly as t16l_stack.arm_model
                               produces them (rank 0; rank i > 0: constants.rank<i>.*)
    keq.pt                     the k_equalize.py key-offset artifact
    probe.json                 the fingerprint prompt + per-position log-probs + 8 ids
    prepare_report.json        timings, hardware, versions, gate results (optional)

and this module is the ONLY reader/writer of `lockstep.manifest.json`: schema
validation (`validate_manifest` / `read_manifest` / `write_manifest`), the hashing
conventions (`sha256_file`, `sha256_tensor`), the constants container
(`save_constants` / `load_constants`, safetensors when importable, else a
torch.save container with the same tensor names) and the `family_block_from_config`
derivation of the model block from an HF config.

MANIFEST SCHEMA (manifest_version 1; docs/PLAN_PLUGIN.md)
----------------------------------------------------------
  manifest_version   1
  contract_version   "2.5"
  lockstep_git       git sha of the tree that prepared it ("unknown" if not a checkout)
  model              hf_id, config_sha256, dtype, hidden, layers, heads, kv_heads,
                     head_dim, intermediate, vocab, org_vocab, tied_embeddings, qk_norm,
                     rotary_dim, sliding_window, max_position, architecture
  family             rotation{kind,R1,R2,R4,seed,H_sha256,signs_sha256}, untied_lm_head,
                     norm_folds, fused_layers, unfused_reason, lmhead_variant,
                     activation_bits, lut, frac
  keq                file, sha256, mu_kind
  tp                 world_size, R4_pad
  constants          file, sha256, count, tensors{name: {shape, dtype, sha256}},
                     export_meta (the t16l_stack.export_constants meta dict, rank 0)
  constants_ranks    [constants block per rank] - REQUIRED when tp.world_size > 1
                     (index = rank; entry 0 equals `constants`)
  probe              file, sha256
  prepared_on        gpu, sm, torch, vllm, date
  report             file, sha256                                   (optional)

TENSOR DIGEST: sha256 over the ASCII header
    "lockstep-tensor-v1|<dtype>|<d0>x<d1>x...|"
followed by the raw little-endian bytes of the contiguous CPU tensor (bf16 is hashed
as its 16-bit words).  A 0-d tensor has the shape field "" (empty).

CONSTANT NAMES (set by t16l_stack.export_constants): `<module>.w8` int8 [N,K],
`<module>.ws` fp32 [N], `<module>.mode` 0-d int32, `<module>.bias` fp32 [N],
`<norm>.G` int32 [N], `<norm>.s` 0-d fp32, `<rope>.q14` fp32, `<embed>.weight` bf16,
`<lm_head>.plane0` / `.plane1` int8 [N,K] row-major, `<lm_head>.ws` fp32 [N]
(`<lm_head>.sdiv` fp32 [K] for SmoothQuant variants, `<lm_head>.weight` bf16 for the
bf16 variant).  Module names are vLLM's `named_modules()` names.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

MANIFEST_VERSION = 1
CONTRACT_VERSION = ("2.6" if os.environ.get("T16L_NORMF", "1") == "1" else "2.5")   # 2.2: the RMSNorm eps; 2.3: the smoothing fold; 2.4: R1 online (SPEC 7.21); 2.5: declared outliers (SPEC 7.22); 2.6: the float norm (T16L_NORMF=1)
MANIFEST_FILE = "lockstep.manifest.json"
PROBE_FILE = "probe.json"
KEQ_FILE = "keq.pt"
REPORT_FILE = "prepare_report.json"
TENSOR_DIGEST_TAG = b"lockstep-tensor-v1|"

REQUIRED_KEYS = ("manifest_version", "contract_version", "lockstep_git", "model", "family",
                 "keq", "tp", "constants", "probe", "prepared_on")
MODEL_KEYS = ("hf_id", "config_sha256", "dtype", "hidden", "layers", "heads", "kv_heads",
              "head_dim", "intermediate", "vocab", "org_vocab", "tied_embeddings", "qk_norm",
              "rotary_dim", "sliding_window", "max_position")
FAMILY_KEYS = ("rotation", "untied_lm_head", "norm_folds", "fused_layers", "unfused_reason",
               "lmhead_variant", "activation_bits", "lut", "frac")
ROTATION_KEYS = ("kind", "R1", "R2", "R4", "seed", "H_sha256", "signs_sha256")
CONSTANTS_KEYS = ("file", "sha256", "count", "tensors", "export_meta")
PREPARED_KEYS = ("gpu", "sm", "torch", "vllm", "date")

_DTYPES = {"int8": torch.int8, "uint8": torch.uint8, "int16": torch.int16, "int32": torch.int32,
           "int64": torch.int64, "float16": torch.float16, "bfloat16": torch.bfloat16,
           "float32": torch.float32, "float64": torch.float64, "bool": torch.bool}


class ManifestError(RuntimeError):
    """A manifest, a file or a tensor failed validation; nothing must be served."""


# ------------------------------------------------------------------ hashing
def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Hex sha256 of a file's bytes, streamed in `chunk`-byte reads."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dtype_name(dtype: torch.dtype) -> str:
    """'torch.int8' -> 'int8' (the manifest's dtype spelling)."""
    return str(dtype).replace("torch.", "")


def dtype_from_name(name: str) -> torch.dtype:
    """Inverse of dtype_name; raises ManifestError on an unknown dtype."""
    try:
        return _DTYPES[name]
    except KeyError:
        raise ManifestError("unknown tensor dtype %r in manifest" % (name,))


def _raw_bytes(t: torch.Tensor) -> bytes:
    """Raw little-endian bytes of a CPU tensor (bf16 via its 16-bit words)."""
    if sys.byteorder != "little":
        raise ManifestError("tensor digests are defined on little-endian hosts only")
    c = t.detach().to("cpu")
    if not c.is_contiguous():
        c = c.contiguous()
    c = c.reshape(-1)
    if c.numel() == 0:
        return b""
    # zero copy: a memoryview over the tensor's own storage.  hashlib.update reads it without
    # copying and releases the GIL, so the per-tensor digests of a manifest run in parallel;
    # .tobytes() copied 8.7 GB under the GIL and made 208 threads no faster than one.
    return memoryview(c.view(torch.uint8).numpy())


def sha256_tensor(t: torch.Tensor) -> str:
    """Hex sha256 of a tensor: header 'lockstep-tensor-v1|<dtype>|<shape>|' + raw bytes.

    dtype and shape are part of the preimage, so an int8 [N,K] and the same bytes
    seen as [K,N] hash differently.  The tensor is copied to CPU if needed."""
    h = hashlib.sha256()
    h.update(TENSOR_DIGEST_TAG)
    h.update(dtype_name(t.dtype).encode("ascii"))
    h.update(b"|")
    h.update("x".join(str(int(d)) for d in t.shape).encode("ascii"))
    h.update(b"|")
    h.update(_raw_bytes(t))
    return h.hexdigest()


def tensor_entries(tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    """{name: {shape, dtype, sha256}} for every tensor, names sorted."""
    out: Dict[str, Dict[str, Any]] = {}
    for name in sorted(tensors):
        t = tensors[name]
        out[name] = {"shape": [int(d) for d in t.shape], "dtype": dtype_name(t.dtype),
                     "sha256": sha256_tensor(t)}
    return out


# --------------------------------------------------------- constants container
def have_safetensors() -> bool:
    """True when `safetensors.torch` imports."""
    try:
        import safetensors.torch  # noqa: F401
        return True
    except Exception:
        return False


def constants_ext() -> str:
    """'.safetensors' when safetensors is importable, else '.pt' (torch.save container)."""
    return ".safetensors" if have_safetensors() else ".pt"


def constants_filename(rank: int = 0, ext: Optional[str] = None) -> str:
    """'constants<ext>' for rank 0, 'constants.rank<i><ext>' otherwise."""
    ext = constants_ext() if ext is None else ext
    return ("constants%s" % ext) if rank == 0 else ("constants.rank%d%s" % (rank, ext))


def export_meta_filename(rank: int = 0) -> str:
    """'export_meta.json' for rank 0, 'export_meta.rank<i>.json' otherwise."""
    return "export_meta.json" if rank == 0 else "export_meta.rank%d.json" % rank


def save_constants(path: str, tensors: Dict[str, torch.Tensor],
                   metadata: Optional[Dict[str, str]] = None) -> str:
    """Write {name: CPU tensor} to `path` (safetensors by extension '.safetensors', else a
    torch.save container {'format': ..., 'tensors': {...}}) and return the file sha256.
    Every tensor is stored contiguous on CPU; the container never aliases storage."""
    cpu = {k: v.detach().to("cpu").contiguous() for k, v in tensors.items()}
    if path.endswith(".safetensors"):
        from safetensors.torch import save_file
        save_file(cpu, path, metadata=metadata or {})
    else:
        torch.save({"format": "lockstep-constants-v1", "metadata": metadata or {},
                    "tensors": cpu}, path)
    return sha256_file(path)


def _read_constants_file(path: str) -> Dict[str, torch.Tensor]:
    """Read a constants container into {name: CPU tensor}."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, dict) or "tensors" not in obj:
        raise ManifestError("%s is not a lockstep constants container" % path)
    return dict(obj["tensors"])


def constants_block(manifest: Dict[str, Any], rank: int = 0) -> Dict[str, Any]:
    """The constants block for `rank`: manifest['constants_ranks'][rank] when present,
    else manifest['constants'] (rank must then be 0)."""
    ranks = manifest.get("constants_ranks")
    if ranks:
        if rank < 0 or rank >= len(ranks):
            raise ManifestError("no constants block for tp rank %d (manifest has %d)"
                                % (rank, len(ranks)))
        return ranks[rank]
    if rank != 0:
        raise ManifestError("manifest carries a single constants block but tp rank %d asked"
                            % rank)
    return manifest["constants"]


def load_constants(mdir: str, manifest: Dict[str, Any], device, rank: int = 0
                   ) -> Dict[str, torch.Tensor]:
    """Load the rank's constants file, verify the FILE sha256 and EVERY tensor's shape,
    dtype and sha256 against the manifest, and return {name: tensor on `device`}.

    Raises ManifestError naming the first (up to 8) mismatching / missing / extra tensors.
    Nothing is returned on any mismatch."""
    blk = constants_block(manifest, rank)
    path = os.path.join(mdir, blk["file"])
    if not os.path.isfile(path):
        raise ManifestError("constants file missing: %s" % path)
    # The file-level sha256 (6.6 s single-threaded over 8.7 GB) adds nothing to the per-tensor
    # digests, which cover every byte the engine will use; it is checked when LOCKSTEP_VERIFY=full
    # (and always for the small files by verify_files).
    if os.environ.get("LOCKSTEP_VERIFY", "tensors") == "full":
        got = sha256_file(path)
        if got != blk["sha256"]:
            raise ManifestError("constants file sha256 mismatch for %s: manifest %s, file %s"
                                % (blk["file"], blk["sha256"], got))
    cpu = _read_constants_file(path)
    want = blk["tensors"]
    bad: List[str] = []
    missing = sorted(set(want) - set(cpu))
    extra = sorted(set(cpu) - set(want))
    for n in missing:
        bad.append("%s: missing from file" % n)
    for n in extra:
        bad.append("%s: not in manifest" % n)
    if len(cpu) != int(blk["count"]) or len(want) != int(blk["count"]):
        bad.append("count: manifest %s, tensors listed %d, in file %d"
                   % (blk["count"], len(want), len(cpu)))
    to_hash: List[str] = []
    for name in sorted(want):
        if name not in cpu:
            continue
        t = cpu[name]
        e = want[name]
        if [int(d) for d in t.shape] != [int(d) for d in e["shape"]]:
            bad.append("%s: shape %s != manifest %s" % (name, list(t.shape), e["shape"]))
            continue
        if dtype_name(t.dtype) != e["dtype"]:
            bad.append("%s: dtype %s != manifest %s" % (name, dtype_name(t.dtype), e["dtype"]))
            continue
        to_hash.append(name)
    # per-tensor sha256 in a thread pool: hashlib releases the GIL on large buffers, so the
    # 483 digests of the 7B (9.6 s serial) take a few seconds on a many-core host.
    if not bad:
        from concurrent.futures import ThreadPoolExecutor
        nthreads = max(1, min(16, int(os.environ.get("LOCKSTEP_VERIFY_THREADS", str(os.cpu_count() or 8)))))
        with ThreadPoolExecutor(max_workers=nthreads) as ex:
            digests = list(ex.map(lambda n: sha256_tensor(cpu[n]), to_hash))
        for name, h in zip(to_hash, digests):
            if h != want[name]["sha256"]:
                bad.append("%s: sha256 %s != manifest %s" % (name, h[:16], want[name]["sha256"][:16]))
            if len(bad) >= 8:
                break
    if bad:
        raise ManifestError("constants verification FAILED (%s, rank %d): %s"
                            % (blk["file"], rank, "; ".join(bad[:8])))
    return {k: v.to(device) for k, v in cpu.items()}


# ------------------------------------------------------------- manifest i/o
def _need(d: Dict[str, Any], keys: Iterable[str], where: str) -> None:
    """Raise ManifestError listing the keys of `keys` absent from dict `d`."""
    if not isinstance(d, dict):
        raise ManifestError("%s: expected an object, got %s" % (where, type(d).__name__))
    miss = [k for k in keys if k not in d]
    if miss:
        raise ManifestError("%s: missing keys %s" % (where, miss))


def _check_file_block(b: Dict[str, Any], where: str) -> None:
    """A {file, sha256} block: both present, file is a bare name, sha256 is 64 hex chars."""
    _need(b, ("file", "sha256"), where)
    f = b["file"]
    if not isinstance(f, str) or not f or os.path.basename(f) != f:
        raise ManifestError("%s.file must be a bare file name, got %r" % (where, f))
    s = b["sha256"]
    if not (isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)):
        raise ManifestError("%s.sha256 is not a hex sha256: %r" % (where, s))


def _check_constants_block(b: Dict[str, Any], where: str) -> None:
    """Validate one constants block (file/sha256/count/tensors/export_meta)."""
    _need(b, CONSTANTS_KEYS, where)
    _check_file_block(b, where)
    if not isinstance(b["count"], int) or b["count"] < 0:
        raise ManifestError("%s.count must be a non-negative int" % where)
    tens = b["tensors"]
    if not isinstance(tens, dict):
        raise ManifestError("%s.tensors must be an object" % where)
    if len(tens) != b["count"]:
        raise ManifestError("%s.count=%d but %d tensors listed" % (where, b["count"], len(tens)))
    for name, e in tens.items():
        _need(e, ("shape", "dtype", "sha256"), "%s.tensors[%s]" % (where, name))
        if not isinstance(e["shape"], list) or not all(isinstance(d, int) for d in e["shape"]):
            raise ManifestError("%s.tensors[%s].shape must be a list of ints" % (where, name))
        dtype_from_name(e["dtype"])
        _check_file_block({"file": b["file"], "sha256": e["sha256"]},
                          "%s.tensors[%s]" % (where, name))
    if not isinstance(b["export_meta"], dict):
        raise ManifestError("%s.export_meta must be an object" % where)


def validate_manifest(m: Dict[str, Any]) -> None:
    """Schema validation of a manifest dict (manifest_version 1).  Raises ManifestError."""
    _need(m, REQUIRED_KEYS, "manifest")
    if m["manifest_version"] != MANIFEST_VERSION:
        raise ManifestError("manifest_version %r, this reader supports %d"
                            % (m["manifest_version"], MANIFEST_VERSION))
    if not isinstance(m["contract_version"], str):
        raise ManifestError("contract_version must be a string")
    if m["contract_version"] in ("2.2", "2.3") and not (m.get("smoothing") or {}).get("level") \
            and (m.get("smoothing") or {}).get("r1", "offline") == "offline":
        pass                                   # v2.4 at r1=offline, smoothing level 0 IS the v2.2 arithmetic and constants
    elif m["contract_version"] == "2.4":
        pass                                   # v2.5 with no declared outliers IS the v2.4 arithmetic (no .oidx constants)
    elif m["contract_version"] in ("2.5", "2.6"):
        pass                                   # both live in this runtime: 2.6 = 2.5 with the float norm (T16L_NORMF=1).  The env
                                               # plan exports the knob FROM THE MANIFEST (lockstep_config.resolve) and
                                               # t16l_stack.import_constants refuses a mismatch (export_meta.normf)
    elif m["contract_version"] != CONTRACT_VERSION:
        raise ManifestError("manifest contract_version %r, this runtime computes contract %s: re-run "
                            "lockstep prepare (the arithmetic changed, so the fingerprint would not reproduce)"
                            % (m["contract_version"], CONTRACT_VERSION))
    if m.get("attention") is not None:
        a = m["attention"]
        if not isinstance(a, dict) or a.get("rule") not in ("LSSA-B8", "LSSA-B9.1", "LSSA-B9.3"):
            raise ManifestError("attention.rule must be LSSA-B8, LSSA-B9.1 or LSSA-B9.3 (got %r)" % (a,))
        if bool(a.get("sink")) != (a["rule"] == "LSSA-B9.1"):
            raise ManifestError("attention.sink must match attention.rule")
        if bool(a.get("ks", 0)) != (a["rule"] == "LSSA-B9.3"):
            raise ManifestError("attention.ks must match attention.rule")
    _need(m["model"], MODEL_KEYS, "model")
    _need(m["family"], FAMILY_KEYS, "family")
    _need(m["family"]["rotation"], ROTATION_KEYS, "family.rotation")
    _check_file_block(m["keq"], "keq")
    _need(m["keq"], ("mu_kind",), "keq")
    _need(m["tp"], ("world_size", "R4_pad"), "tp")
    ws = m["tp"]["world_size"]
    if not isinstance(ws, int) or ws < 1:
        raise ManifestError("tp.world_size must be an int >= 1, got %r" % (ws,))
    _check_constants_block(m["constants"], "constants")
    ranks = m.get("constants_ranks")
    if ws > 1:
        if not isinstance(ranks, list) or len(ranks) != ws:
            raise ManifestError("tp.world_size=%d needs constants_ranks with %d entries" % (ws, ws))
    if ranks is not None:
        if not isinstance(ranks, list):
            raise ManifestError("constants_ranks must be a list")
        for i, b in enumerate(ranks):
            _check_constants_block(b, "constants_ranks[%d]" % i)
        if ranks and ranks[0]["sha256"] != m["constants"]["sha256"]:
            raise ManifestError("constants_ranks[0] is not the rank-0 constants block")
    _check_file_block(m["probe"], "probe")
    _need(m["prepared_on"], PREPARED_KEYS, "prepared_on")
    if "report" in m and m["report"] is not None:
        _check_file_block(m["report"], "report")


def write_manifest(mdir: str, manifest: Dict[str, Any]) -> str:
    """Validate and write `<mdir>/lockstep.manifest.json` (sorted keys, indent 1).
    Returns the path."""
    validate_manifest(manifest)
    os.makedirs(mdir, exist_ok=True)
    path = os.path.join(mdir, MANIFEST_FILE)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return path


def read_manifest(mdir: str) -> Dict[str, Any]:
    """Read and validate `<mdir>/lockstep.manifest.json`.  Raises ManifestError."""
    path = os.path.join(mdir, MANIFEST_FILE)
    if not os.path.isfile(path):
        raise ManifestError("no %s in %s" % (MANIFEST_FILE, mdir))
    try:
        with open(path) as f:
            m = json.load(f)
    except Exception as e:
        raise ManifestError("%s is not valid JSON: %r" % (path, e))
    validate_manifest(m)
    return m


def file_blocks(manifest: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Every (label, {file, sha256}) block the manifest names: keq, constants (all ranks),
    probe, report."""
    out: List[Tuple[str, Dict[str, Any]]] = [("keq", manifest["keq"])]
    ranks = manifest.get("constants_ranks")
    if ranks:
        out += [("constants_ranks[%d]" % i, b) for i, b in enumerate(ranks)]
    else:
        out.append(("constants", manifest["constants"]))
    out.append(("probe", manifest["probe"]))
    if manifest.get("report"):
        out.append(("report", manifest["report"]))
    return out


def verify_files(mdir: str, manifest: Dict[str, Any], skip: Iterable[str] = ()
                 ) -> Dict[str, str]:
    """sha256-check every file the manifest names (labels in `skip` - e.g. 'constants',
    'constants_ranks[1]' - are not hashed; load_constants hashes those anyway).
    Returns {label: sha256}; raises ManifestError on the first missing/mismatching file."""
    skip = set(skip)
    out: Dict[str, str] = {}
    for label, b in file_blocks(manifest):
        if label in skip or (label.startswith("constants") and "constants" in skip):
            continue
        path = os.path.join(mdir, b["file"])
        if not os.path.isfile(path):
            raise ManifestError("%s: file %s missing" % (label, path))
        h = sha256_file(path)
        if h != b["sha256"]:
            raise ManifestError("%s: sha256 mismatch for %s (manifest %s, file %s)"
                                % (label, b["file"], b["sha256"], h))
        out[label] = h
    return out


# --------------------------------------------------------- HF config -> blocks
def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Attribute or dict lookup on an HF config (PretrainedConfig or plain dict)."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


ATTENTION_RULES = {
    "b8":   {"rule": "LSSA-B8", "sink": False, "v_planes": 1, "ks": 0, "ksr": "up"},
    "b9.1": {"rule": "LSSA-B9.1", "sink": True, "v_planes": 1, "ks": 0, "ksr": "up"},
    # B9.3: every key on its own value grid e_v + d, d = min(3, e_key - e_v), the u8 weight shifted right by
    # d in the chain (round half up); l on the unshifted weight
    "b9.3": {"rule": "LSSA-B9.3", "sink": False, "v_planes": 1, "ks": 3, "ksr": "up"},
}


def attention_block(rule: str = "b8", ks_round: str = "up") -> Dict[str, Any]:
    """The attention rule every row takes under contract v2.  Missing block == LSSA-B8 (every
    manifest before T16w).  B9.1 (SPEC 7.23): key 0 of every sequence is quantised with its own
    value exponent and folded as its own fp32 term before block 0; l is unchanged."""
    if rule not in ATTENTION_RULES:
        raise ManifestError("unknown attention rule %r (known: %s)" % (rule, ", ".join(ATTENTION_RULES)))
    b = dict(ATTENTION_RULES[rule], kb=128, seg=32, s_win=32, w_loc=9, n=7)
    if rule == "b9.3":
        if ks_round not in ("up", "even", "floor"):
            raise ManifestError("unknown B9.3 rounding %r" % (ks_round,))
        b["ksr"] = ks_round
    return b


def attention_sink(m: Dict[str, Any]) -> bool:
    """True when the manifest's rows take LSSA-B9.1 (key 0 alone)."""
    return bool((m.get("attention") or {}).get("sink", False))


def attention_ks(m: Dict[str, Any]):
    """(dmax, rounding) of LSSA-B9.3, or (0, "up") for the other rules."""
    a = m.get("attention") or {}
    return int(a.get("ks", 0) or 0), str(a.get("ksr", "up"))


def rotation_kind_for(n: int) -> str:
    """The R1 Hadamard construction t16l_stack.hadamard_for picks for hidden size n
    ('sylvester' | 'paley1' | 'sylvester_blockdiag<blk>'); asks t16l_stack when importable."""
    try:
        import t16l_stack as T
        return T.hadamard_for(n)[1]
    except Exception:
        pass
    if n & (n - 1) == 0:
        return "sylvester"
    q = n - 1

    def prime(v):
        if v < 2:
            return False
        i = 2
        while i * i <= v:
            if v % i == 0:
                return False
            i += 1
        return True
    if q % 4 == 3 and prime(q):
        return "paley1"
    return "sylvester_blockdiag%d" % (n & (-n))


def family_block_from_config(hf_config: Any, tp_world_size: int = 1) -> Dict[str, Any]:
    """Derive what the manifest can know from the HF config alone.

    Returns {"model": {...}, "family": {...}, "tp": {...}} with every key the schema
    requires; the prepare step overwrites the family entries that only the armed engine
    knows (rotation hashes, norm_folds, untied_lm_head, fused_layers).  `qk_norm` is
    known only from the architecture name (True for any 'Qwen3' architecture)."""
    arch = list(_cfg_get(hf_config, "architectures", None) or [])
    arch0 = arch[0] if arch else str(_cfg_get(hf_config, "model_type", "") or "")
    hidden = int(_cfg_get(hf_config, "hidden_size"))
    heads = int(_cfg_get(hf_config, "num_attention_heads"))
    kv = int(_cfg_get(hf_config, "num_key_value_heads", heads) or heads)
    head_dim = _cfg_get(hf_config, "head_dim", None)
    head_dim = int(head_dim) if head_dim else hidden // heads
    prf = _cfg_get(hf_config, "partial_rotary_factor", 1.0)
    try:
        prf = float(prf) if prf is not None else 1.0
    except Exception:
        prf = 1.0
    rotary_dim = int(head_dim * prf)
    use_sw = _cfg_get(hf_config, "use_sliding_window", None)
    sw = _cfg_get(hf_config, "sliding_window", None)
    if use_sw is False or not sw:
        sw = None
    else:
        sw = int(sw)
    vocab = int(_cfg_get(hf_config, "vocab_size"))
    qk_norm = "qwen3" in arch0.lower()
    model = {
        "hf_id": str(_cfg_get(hf_config, "_name_or_path", "") or ""),
        "config_sha256": "",
        "dtype": "bfloat16",
        "architecture": arch0,
        "hidden": hidden,
        "layers": int(_cfg_get(hf_config, "num_hidden_layers")),
        "heads": heads,
        "kv_heads": kv,
        "head_dim": head_dim,
        "intermediate": int(_cfg_get(hf_config, "intermediate_size")),
        "vocab": vocab,
        "org_vocab": vocab,
        "tied_embeddings": bool(_cfg_get(hf_config, "tie_word_embeddings", False)),
        "qk_norm": qk_norm,
        "rotary_dim": rotary_dim,
        "sliding_window": sw,
        "max_position": int(_cfg_get(hf_config, "max_position_embeddings", 0) or 0),
    }
    family = {
        "rotation": {"kind": rotation_kind_for(hidden), "R1": True, "R2": True, "R4": True,
                     "seed": 0, "H_sha256": "", "signs_sha256": ""},
        "untied_lm_head": False,
        "norm_folds": 2 * model["layers"] + 1,
        "fused_layers": 0 if qk_norm else model["layers"],
        "unfused_reason": "q_norm" if qk_norm else None,
        "lmhead_variant": "a15w15",
        "activation_bits": 8,
        "lut": "rsqrt_v1",
        "frac": "13/15",
    }
    tp = {"world_size": int(tp_world_size), "R4_pad": None}
    return {"model": model, "family": family, "tp": tp}
