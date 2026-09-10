import Lockstep.Numeric.Bounds
import Lockstep.Attention.Table
import Lean.Elab.Tactic.Omega

namespace Lockstep
namespace Decoder

open Numeric

theorem activation_quantization_range (value : Int) :
    -127 ≤ Attention.clampInt8 value ∧ Attention.clampInt8 value ≤ 127 :=
  Attention.quantized_in_int8_range value

theorem weight_quantization_range (value : Int) :
    -127 ≤ Attention.clampInt8 value ∧ Attention.clampInt8 value ≤ 127 :=
  Attention.quantized_in_int8_range value

/-- Maximum absolute value of an exact int8 product. -/
def int8ProductBound : Nat := 127 * 127

theorem linear_acc_abs_lt {terms : List Int}
    (hlen : terms.length ≤ 2 ^ 15)
    (hterms : TermsBounded int8ProductBound terms) :
    terms.sum.natAbs < 2 ^ 29 := by
  have hs := sum_natAbs_le_length_mul hterms
  have hm : terms.length * int8ProductBound ≤ (2 ^ 15) * int8ProductBound :=
    Nat.mul_le_mul_right _ hlen
  have hb : (2 ^ 15 : Nat) * int8ProductBound < 2 ^ 29 := by decide
  omega

/-- RMSNorm's square sum and capped epsilon term remain strictly below signed int64. -/
theorem rmsnorm_sum_lt_int64 {squareSum epsTerm : Nat}
    (hsquare : squareSum < 2 ^ 45)
    (heps : epsTerm ≤ 2 ^ 50) :
    squareSum + epsTerm < 2 ^ 63 := by omega

/-- Four signed int32 partials combined at shifts 14, 7, 7, and 0 fit int64. -/
theorem head_recomposition_lt_int64 {hh hl lh ll : Nat}
    (hhh : hh < 2 ^ 29) (hhl : hl < 2 ^ 29)
    (hlh : lh < 2 ^ 29) (hll : ll < 2 ^ 29) :
    hh * 2 ^ 14 + hl * 2 ^ 7 + lh * 2 ^ 7 + ll < 2 ^ 63 := by omega

/-- R4 pairs lane `i` with lane `i + 128` inside every 256-wide block. -/
def r4Pair (k : Nat) : Nat := if k < 128 then k else k - 128

def r4UpperLane (k : Nat) : Bool := decide (128 ≤ k)

def r4Reconstruct (pair : Nat) (upper : Bool) : Nat :=
  if upper then pair + 128 else pair

/-- Pair index and lane bit are a complete, disjoint coordinate system for a 256 block. -/
theorem r4_pairs_complete_disjoint {k : Nat} (hk : k < 256) :
    r4Pair k < 128 ∧ r4Reconstruct (r4Pair k) (r4UpperLane k) = k := by
  by_cases h : k < 128
  · simp [r4Pair, r4UpperLane, r4Reconstruct, h]
  · have hlo : 128 ≤ k := Nat.le_of_not_gt h
    simp [r4Pair, r4UpperLane, r4Reconstruct, h, hlo]
    omega

/-- Static envelope arithmetic used by all admitted decoder operators. -/
theorem decoder_static_safety :
    (2 ^ 15 : Nat) * int8ProductBound < 2 ^ 29 ∧
    (2 ^ 45 : Nat) + 2 ^ 50 < 2 ^ 63 ∧
    4 * ((2 ^ 29 : Nat) * 2 ^ 14) < 2 ^ 63 := by decide

end Decoder
end Lockstep
