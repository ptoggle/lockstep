import Lockstep.Numeric.Bits
import Lean.Elab.Tactic.Omega

namespace Lockstep
namespace Numeric

/-- Canonical quiet NaN used to make the executable arithmetic total outside the admitted domain. -/
def canonicalNaN : F32Bits := 0x7fc00000#32

def positiveInfinity : F32Bits := 0x7f800000#32
def negativeInfinity : F32Bits := 0xff800000#32

/-- Exact finite value `(-1)^significandSign * significand * 2^scale`. -/
structure FiniteDyadic where
  negative : Bool
  significand : Nat
  scale : Int
  deriving Repr, DecidableEq

private def signBool (x : F32Bits) : Bool := sign x = 1#1

/-- Decode a finite binary32 word into an exact dyadic value. -/
def decodeFinite (x : F32Bits) : Option FiniteDyadic :=
  let e := (exponent x).toNat
  let f := (fraction x).toNat
  if e = 255 then none
  else if e = 0 then
    some { negative := signBool x, significand := f, scale := -149 }
  else
    some {
      negative := signBool x
      significand := 2^23 + f
      scale := Int.ofNat e - 150
    }

private def packNat (negative : Bool) (biasedExponent fractionBits : Nat) : F32Bits :=
  BitVec.ofNat 32
    ((if negative then 2^31 else 0) + biasedExponent * 2^23 + fractionBits)

private def infinity (negative : Bool) : F32Bits :=
  if negative then negativeInfinity else positiveInfinity

/-- Round a nonnegative rational to the nearest natural number, ties to even. -/
def divRoundEven (numerator denominator : Nat) : Nat :=
  if denominator = 0 then 0
  else
    let quotient := numerator / denominator
    let remainder := numerator % denominator
    if denominator < 2 * remainder ∨
        (2 * remainder = denominator ∧ quotient % 2 = 1) then
      quotient + 1
    else quotient

/--
Normalize `numerator / denominator * 2^scale` until the rational significand lies in `[1,2)`.
The fuel bound exceeds every exponent distance reachable from binary32 inputs and contract ints.
-/
private def normalizeRatio : Nat → Nat → Nat → Int → Option (Nat × Nat × Int)
  | 0, _, _, _ => none
  | _ + 1, 0, _, _ => none
  | _ + 1, _, 0, _ => none
  | fuel + 1, numerator, denominator, scale =>
      if numerator < denominator then
        normalizeRatio fuel (2 * numerator) denominator (scale - 1)
      else if 2 * denominator ≤ numerator then
        normalizeRatio fuel numerator (2 * denominator) (scale + 1)
      else
        some (numerator, denominator, scale)

/-- Round an exact signed rational times a power of two to binary32, ties to even. -/
def roundRationalToF32 (negative : Bool) (numerator denominator : Nat)
    (scale : Int) : F32Bits :=
  if numerator = 0 then if negative then negativeZero else positiveZero
  else if denominator = 0 then infinity negative
  else
    match normalizeRatio 1024 numerator denominator scale with
    | none => canonicalNaN
    | some (normalizedNumerator, normalizedDenominator, exponentValue) =>
        if 127 < exponentValue then infinity negative
        else if -126 ≤ exponentValue then
          let rounded := divRoundEven (normalizedNumerator * 2^23) normalizedDenominator
          let carry := rounded = 2^24
          let finalExponent := if carry then exponentValue + 1 else exponentValue
          let significand := if carry then 2^23 else rounded
          if 127 < finalExponent then infinity negative
          else
            packNat negative (finalExponent + 127).toNat (significand - 2^23)
        else
          let subnormalShift := exponentValue + 149
          let rounded :=
            if 0 ≤ subnormalShift then
              divRoundEven (normalizedNumerator * 2^subnormalShift.toNat)
                normalizedDenominator
            else
              divRoundEven normalizedNumerator
                (normalizedDenominator * 2^(-subnormalShift).toNat)
          if rounded = 0 then if negative then negativeZero else positiveZero
          else if 2^23 ≤ rounded then packNat negative 1 0
          else packNat negative 0 rounded

/-- Round the exact dyadic integer `value * 2^scale` to binary32. -/
def scaledIntToF32 (value : Int) (scale : Int) : F32Bits :=
  roundRationalToF32 (value < 0) value.natAbs 1 scale

def scaledNatToF32 (value : Nat) (scale : Int) : F32Bits :=
  roundRationalToF32 false value 1 scale

private def signedAtScale (value : FiniteDyadic) (commonScale : Int) : Int :=
  let magnitude := value.significand * 2^(value.scale - commonScale).toNat
  if value.negative then -Int.ofNat magnitude else Int.ofNat magnitude

/-- Correctly rounded binary32 addition over finite inputs. Nonfinite inputs map to canonical NaN. -/
def addRNE (a b : F32Bits) : F32Bits :=
  match decodeFinite a, decodeFinite b with
  | some da, some db =>
      let commonScale := min da.scale db.scale
      scaledIntToF32 (signedAtScale da commonScale + signedAtScale db commonScale) commonScale
  | _, _ => canonicalNaN

/-- Correctly rounded binary32 division over finite inputs. -/
def divideRNE (numerator denominator : F32Bits) : F32Bits :=
  match decodeFinite numerator, decodeFinite denominator with
  | some n, some d =>
      if d.significand = 0 then
        if n.significand = 0 then canonicalNaN else infinity (n.negative != d.negative)
      else
        roundRationalToF32 (n.negative != d.negative) n.significand d.significand
          (n.scale - d.scale)
  | _, _ => canonicalNaN

/-- Contract rescaling with an unbounded logical distance. -/
def rescaleNat (x : F32Bits) (distance : Nat) : F32Bits :=
  if 255 < distance then positiveZero
  else rescaleBits x (BitVec.ofNat 8 distance)

/-- Round one binary32 word to its bfloat16 bit pattern, ties to even. -/
def toBF16RNE (x : F32Bits) : BF16Bits :=
  let word := x.toNat
  let upper := word / 2^16
  let lower := word % 2^16
  let increment := 2^15 < lower ∨ (lower = 2^15 ∧ upper % 2 = 1)
  BitVec.ofNat 16 (if increment then upper + 1 else upper)

@[simp] theorem decode_positive_zero :
    decodeFinite positiveZero = some { negative := false, significand := 0, scale := -149 } := by
  decide

@[simp] theorem scaled_one : scaledIntToF32 1 0 = 0x3f800000#32 := by decide
@[simp] theorem scaled_negative_one : scaledIntToF32 (-1) 0 = 0xbf800000#32 := by decide
@[simp] theorem add_one_one : addRNE 0x3f800000#32 0x3f800000#32 = 0x40000000#32 := by decide
@[simp] theorem divide_one_two : divideRNE 0x3f800000#32 0x40000000#32 = 0x3f000000#32 := by decide
@[simp] theorem bf16_tie_even_down : toBF16RNE 0x3f808000#32 = 0x3f80#16 := by decide
@[simp] theorem bf16_tie_even_up : toBF16RNE 0x3f818000#32 = 0x3f82#16 := by decide

end Numeric
end Lockstep
