#!/usr/bin/env python3
"""Verify the checked Lean artifact against the executable Python contract.

The positive path compares the complete JSON object. ``--negative`` mutates the
T2 digest in memory and succeeds only when the same verifier rejects it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import struct
from decimal import Decimal, getcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent

KB = 128
SEG = 32
NIDX = 7
BHI = 26
TOP = 255
W_LOC = 9
KTAB_DOUB = 17

SCHEMA = "lockstep-lean-artifacts-v2"
TOOLCHAIN = "leanprover/lean4:v4.33.1"

LOGICAL_OBLIGATIONS = [
    "row/head/channel Cartesian grid",
    "128-term block coverage",
    "32-block segment coverage",
    "CTA segment assignment coverage",
    "complete quotient/remainder ownership",
    "disjoint quotient/remainder ownership",
    "ascending segment and rank reconstruction",
]
HARDWARE_ASSUMPTIONS = [
    "WGMMA int8 products and int32 accumulation implement the documented operation",
    "PTX/SASS binary32 instructions round to nearest-even at each declared boundary",
    "compiler lowering preserves declared data dependencies and operation order",
    "GPU memory and synchronization primitives implement the CUDA memory model",
    "SHA-256 and BLAKE3 are collision resistant and binding",
]


def t2_table() -> list[int]:
    """Independent stdlib execution of the runtime contract's 40-digit Decimal generator."""
    getcontext().prec = 40
    ln2 = Decimal(2).ln()
    live = W_LOC * (1 << NIDX)
    table = [0] * (KTAB_DOUB * (1 << NIDX))
    for index in range(live):
        value = Decimal(TOP) * (-Decimal(index) / Decimal(1 << NIDX) * ln2).exp()
        table[index] = int(
            (value + Decimal("0.5")).to_integral_value(rounding="ROUND_FLOOR")
        )
    return table


def t2_digest(table: list[int]) -> str:
    payload = ",".join(str(value) for value in table).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def rescale_bits(word: int, distance: int) -> int:
    """Bit-level total version of the contract's exponent-field rescale."""
    exponent = (word >> 23) & 0xFF
    if exponent == 0xFF:
        return word
    if exponent == 0 or exponent <= distance:
        return 0
    return (word & ~(0xFF << 23)) | ((exponent - distance) << 23)

def f32_from_word(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word))[0]


def f32_word(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def binary32_vector(a: int, b: int) -> dict[str, int]:
    av, bv = f32_from_word(a), f32_from_word(b)
    return {
        "a": a,
        "b": b,
        "add": f32_word(av + bv),
        "divide": f32_word(av / bv),
    }


def bf16_rne(word: int) -> int:
    upper, lower = word >> 16, word & 0xFFFF
    return (upper + int(lower > 0x8000 or (lower == 0x8000 and upper & 1))) & 0xFFFF


def rounded_average(a: int, b: int) -> int:
    return (a + b) // 2


def rounded_fold(values: list[int]) -> int:
    result = 0
    for value in values:
        result = rounded_average(result, value)
    return result


def weighted_mean(items: list[tuple[int, int]]) -> int:
    return sum(weight * value for weight, value in items) // sum(weight for weight, _ in items)


def expected_artifact() -> dict[str, object]:
    table = t2_table()
    rescale_inputs = [(0x3F800000, 1), (0x80000000, 4), (0x00800000, 1), (0x7F800000, 3)]
    binary32_inputs = [
        (0x3F800000, 0x3F800000),
        (0x3F800000, 0x40000000),
        (0xBF800000, 0x3F800000),
        (0x00800000, 0x40000000),
        (0x00000001, 0x00000001),
        (0x3F800001, 0x3F7FFFFF),
    ]
    return {
        "schema": SCHEMA,
        "lean_toolchain": TOOLCHAIN,
        "contract": {
            "head_dim": 128,
            "block_size": KB,
            "segment_blocks": SEG,
            "score_frac_bits": NIDX,
            "score_hi_bits": BHI,
            "weight_bits": 8,
        },
        "bounds": {
            "score_abs": 128 * 127 * 127,
            "block_value_abs": TOP * 127 * KB,
            "block_weight": TOP * KB,
            "linear_acc_exclusive": 2**29,
        },
        "t2": {
            "csv": ",".join(str(value) for value in table),
            "length": len(table),
            "live_length": W_LOC * (1 << NIDX),
            "digest": t2_digest(table),
            "anchors": [
                {"index": index, "value": int(table[index])}
                for index in (0, 128, 1151, 1152)
            ],
        },
        "rescale_vectors": [
            {"input": word, "distance": distance, "output": rescale_bits(word, distance)}
            for word, distance in rescale_inputs
        ],
        "binary32": {
            "operations": [binary32_vector(a, b) for a, b in binary32_inputs],
            "bf16_ties": [bf16_rne(0x3F808000), bf16_rne(0x3F818000)],
            "single_token_row": [0x4000] + [0] * 127,
        },
        "negative_controls": {
            "descending_fold": [rounded_fold([8, 4, 2]), rounded_fold([2, 4, 8])],
            "nonaligned_chunk": [
                weighted_mean([(1, 0), (3, 10)]),
                (weighted_mean([(1, 0)]) + weighted_mean([(3, 10)])) // 2,
            ],
            "approx_division": [7 // 3, 7 // 4],
            "multiply_rescale": [3 // 2 + 3, (3 + 3) // 2],
        },
        "transcript": {
            "word": 0x4C535352,
            "little_endian": list(struct.pack("<I", 0x4C535352)),
        },
        "audit_example": {
            "total_rows": 10,
            "bad_rows": 2,
            "sample_size": 3,
            "clean_samples": math.comb(8, 3),
            "detecting_samples": math.comb(10, 3) - math.comb(8, 3),
        },
        "fidelity_certificate": {
            "value_contribution": -3,
            "weight_contribution": 5,
            "observed_error": 2,
            "value_bound": 3,
            "weight_bound": 5,
            "policy_limit": 8,
            "accepted": True,
        },
        "gpu": {
            "logical_obligations": LOGICAL_OBLIGATIONS,
            "hardware_assumptions": HARDWARE_ASSUMPTIONS,
        },
    }


def differences(actual: object, expected: object, path: str = "$.") -> list[str]:
    if type(actual) is not type(expected):
        return [f"{path}: type {type(actual).__name__} != {type(expected).__name__}"]
    if isinstance(expected, dict):
        issues: list[str] = []
        actual_keys, expected_keys = set(actual), set(expected)
        for key in sorted(expected_keys - actual_keys):
            issues.append(f"{path}{key}: missing")
        for key in sorted(actual_keys - expected_keys):
            issues.append(f"{path}{key}: unexpected")
        for key in sorted(actual_keys & expected_keys):
            issues.extend(differences(actual[key], expected[key], f"{path}{key}."))
        return issues
    if isinstance(expected, list):
        if len(actual) != len(expected):
            return [f"{path}: length {len(actual)} != {len(expected)}"]
        issues = []
        for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
            issues.extend(differences(got, want, f"{path}{index}."))
        return issues
    return [] if actual == expected else [f"{path[:-1]}: {actual!r} != {expected!r}"]


def verify(artifact: dict[str, object]) -> list[str]:
    return differences(artifact, expected_artifact())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact",
        nargs="?",
        type=Path,
        default=HERE.parent / "results" / "lean_contract_artifacts.json",
    )
    parser.add_argument("--negative", action="store_true")
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    if args.negative:
        artifact = copy.deepcopy(artifact)
        artifact["t2"]["digest"] = "0" * 64
        issues = verify(artifact)
        if not issues:
            print("LOCKSTEP-LEAN-BRIDGE FAIL: mutation was accepted")
            return 1
        print(f"LOCKSTEP-LEAN-BRIDGE PASS: mutation rejected ({issues[0]})")
        return 0

    issues = verify(artifact)
    if issues:
        print("LOCKSTEP-LEAN-BRIDGE FAIL")
        for issue in issues:
            print(f"  {issue}")
        return 1
    print("LOCKSTEP-LEAN-BRIDGE PASS: all formal artifacts match runtime values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
