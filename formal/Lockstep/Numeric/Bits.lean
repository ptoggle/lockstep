import Std.Tactic.BVDecide
import Lean.Elab.Tactic.Omega

namespace Lockstep
namespace Numeric

abbrev F32Bits := BitVec 32
abbrev BF16Bits := BitVec 16
abbrev ExpBits := BitVec 8

/-- IEEE-754 binary32 sign bit. -/
def sign (x : F32Bits) : BitVec 1 := BitVec.extractLsb 31 31 x

/-- IEEE-754 binary32 biased exponent. -/
def exponent (x : F32Bits) : ExpBits := BitVec.extractLsb 30 23 x

/-- IEEE-754 binary32 trailing significand field. -/
def fraction (x : F32Bits) : BitVec 23 := BitVec.extractLsb 22 0 x

def pack (s : BitVec 1) (e : ExpBits) (f : BitVec 23) : F32Bits :=
  (BitVec.zeroExtend 32 s <<< 31) |||
  (BitVec.zeroExtend 32 e <<< 23) |||
  BitVec.zeroExtend 32 f

def positiveZero : F32Bits := 0#32
def negativeZero : F32Bits := 0x80000000#32

def IsZero (x : F32Bits) : Prop := exponent x = 0#8 ∧ fraction x = 0#23
def IsSubnormal (x : F32Bits) : Prop := exponent x = 0#8 ∧ fraction x ≠ 0#23
def IsNormal (x : F32Bits) : Prop := exponent x ≠ 0#8 ∧ exponent x ≠ 255#8
def IsInfinite (x : F32Bits) : Prop := exponent x = 255#8 ∧ fraction x = 0#23
def IsNaN (x : F32Bits) : Prop := exponent x = 255#8 ∧ fraction x ≠ 0#23

/--
Contract exponent-field rescaling. Non-finite inputs are outside the admitted domain and are
left unchanged to make the operation total. Zero, subnormal, and underflowing normal inputs
map to positive zero. A surviving normal has its biased exponent reduced by `d`.
-/
def rescaleBits (x : F32Bits) (d : ExpBits) : F32Bits :=
  let e := exponent x
  if e = 255#8 then x
  else if e = 0#8 ∨ e ≤ d then positiveZero
  else pack (sign x) (e - d) (fraction x)

@[simp] theorem pack_fields (s : BitVec 1) (e : ExpBits) (f : BitVec 23) :
    sign (pack s e f) = s ∧ exponent (pack s e f) = e ∧ fraction (pack s e f) = f := by
  simp [sign, exponent, fraction, pack]
  bv_decide


@[simp] theorem pack_roundtrip (x : F32Bits) :
    pack (sign x) (exponent x) (fraction x) = x := by
  unfold pack sign exponent fraction
  bv_decide
@[simp] theorem rescale_positive_zero (d : ExpBits) :
    rescaleBits positiveZero d = positiveZero := by
  simp [rescaleBits, positiveZero, exponent]

@[simp] theorem rescale_negative_zero (d : ExpBits) :
    rescaleBits negativeZero d = positiveZero := by
  simp [rescaleBits, negativeZero, exponent]

/-- Both IEEE zero encodings are normalized to the contract's positive zero. -/
theorem rescale_zero {x : F32Bits} (h : IsZero x) (d : ExpBits) :
    rescaleBits x d = positiveZero := by
  rcases h with ⟨he, _⟩
  simp [rescaleBits, he]

/-- A normal value whose exponent does not survive the subtraction is flushed. -/
theorem rescale_flushes_small_exponent {x : F32Bits} {d : ExpBits}
    (hf : exponent x ≠ 255#8) (h : exponent x ≤ d) :
    rescaleBits x d = positiveZero := by
  simp [rescaleBits, hf, h]

/-- A surviving rescale preserves sign and trailing significand bits. -/
theorem rescale_preserves_significand {x : F32Bits} {d : ExpBits}
    (hn : IsNormal x) (h : d < exponent x) :
    sign (rescaleBits x d) = sign x ∧ fraction (rescaleBits x d) = fraction x := by
  rcases hn with ⟨h0, h255⟩
  have hnot : ¬ exponent x ≤ d := BitVec.not_le.mpr h
  simp [rescaleBits, h0, h255, hnot]

/-- A surviving rescale subtracts `d` from the biased exponent. -/
theorem rescale_subtracts_exponent {x : F32Bits} {d : ExpBits}
    (hn : IsNormal x) (h : d < exponent x) :
    exponent (rescaleBits x d) = exponent x - d := by
  rcases hn with ⟨h0, h255⟩
  have hnot : ¬ exponent x ≤ d := BitVec.not_le.mpr h
  simp [rescaleBits, h0, h255, hnot]

/-- Rescaling a normal finite value by zero is the identity. -/
theorem rescale_id {x : F32Bits} (hn : IsNormal x) :
    rescaleBits x 0#8 = x := by
  rcases hn with ⟨h0, h255⟩
  have hnot : ¬ exponent x ≤ 0#8 := by
    bv_decide
  simp [rescaleBits, h0, h255, hnot, pack_roundtrip]

/-- Mathematical predicate for an integer representable in a signed fixed-width word. -/
def FitsSigned (bits : Nat) (value : Int) : Prop :=
  0 < bits ∧ -(2 ^ (bits - 1) : Int) ≤ value ∧ value < (2 ^ (bits - 1) : Int)

theorem int8_contract_range_fits {value : Int}
    (hlo : -127 ≤ value) (hhi : value ≤ 127) : FitsSigned 8 value := by
  simp [FitsSigned]
  omega

theorem int32_score_range_fits {value : Int}
    (h : value.natAbs ≤ 2_064_512) : FitsSigned 32 value := by
  simp [FitsSigned]
  omega

/-- Integers in this range have at most 24 significant binary digits. -/
def IntExactlyRepresentableInBinary32 (z : Int) : Prop := z.natAbs ≤ 2^24

theorem int_block_partial_exact_binary32 {z : Int}
    (h : z.natAbs ≤ 4_145_280) : IntExactlyRepresentableInBinary32 z := by
  unfold IntExactlyRepresentableInBinary32
  omega

theorem int_block_weight_exact_binary32 {z : Nat}
    (h : z ≤ 32_640) : IntExactlyRepresentableInBinary32 (Int.ofNat z) := by
  unfold IntExactlyRepresentableInBinary32
  simp
  omega

end Numeric
end Lockstep
