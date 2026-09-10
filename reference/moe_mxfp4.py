"""Exact MXFP4 (OCP MX: FP4 E2M1 elements, E8M0 block-32 scales) dequantisation in torch, for the contract fold of
GPT-OSS experts. Every dequantised value is fp4_value * 2^(e - 127): a 3-significant-bit number times a power of two,
exactly representable in bf16 (and fp32) - so the dequant is exact and device-independent by construction.
Layout (the checkpoint's): packed uint8 [..., K / 2] with element 2i in the LOW nibble and 2i + 1 in the HIGH nibble;
scale uint8 [..., K / 32], one E8M0 exponent per 32 elements. The nibble order is verified against the stock model's
outputs in the M2-0 check (a wrong order would break routing agreement immediately)."""
import torch

# E2M1: sign | 2-bit exponent | 1-bit mantissa; codes 0..7 -> 0, 0.5, 1, 1.5, 2, 3, 4, 6 (exponent bias 1, subnormal 0.5)
_FP4_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def fp4_values(codes: torch.Tensor) -> torch.Tensor:
    """uint8 codes 0..15 -> fp32 values."""
    mag = _FP4_MAG.to(codes.device)[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -mag, mag)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/2] -> uint8 codes [..., K], low nibble first."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    return torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def dequant_mxfp4(packed: torch.Tensor, scale_e8m0: torch.Tensor, out_dtype=torch.bfloat16, block: int = 32) -> torch.Tensor:
    """packed uint8 [..., K/2], scale uint8 [..., K/block] -> [..., K] in out_dtype, exact."""
    codes = unpack_fp4(packed)
    vals = fp4_values(codes)                                                       # fp32, exact
    K = vals.shape[-1]
    assert scale_e8m0.shape[-1] * block == K, (scale_e8m0.shape, K)
    exp = scale_e8m0.to(torch.int32) - 127                                         # E8M0: 2^(e - 127); 255 = NaN (not expected)
    sc = torch.ldexp(torch.ones_like(exp, dtype=torch.float32), exp)               # exact powers of two
    sc = sc.unsqueeze(-1).expand(*sc.shape, block).reshape(*sc.shape[:-1], K)
    return (vals * sc).to(out_dtype)                                               # exact: 3 significant bits x 2^k


def self_test(device="cpu"):
    torch.manual_seed(0)
    E, N, K = 3, 8, 128
    codes = torch.randint(0, 16, (E, N, K), dtype=torch.uint8, device=device)
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)
    scale = torch.randint(100, 140, (E, N, K // 32), dtype=torch.uint8, device=device)
    y = dequant_mxfp4(packed, scale)
    ref = fp4_values(codes) * torch.pow(2.0, (scale.to(torch.float32) - 127.0)).repeat_interleave(32, dim=-1)
    assert torch.equal(y.to(torch.float32), ref), "mxfp4 dequant self-test failed"
    assert torch.equal(y.to(torch.float32).to(torch.bfloat16).to(torch.float32), ref), "not bf16-exact"
    return True


if __name__ == "__main__":
    print("MXFP4 self-test", self_test())
