import Lockstep.Attention.LSSA
import Lockstep.Numeric.Binary32

namespace Lockstep
namespace Attention

open Numeric

/-- Concrete bit-level implementation of the LSSA binary32 rounding boundary. -/
def binary32RoundOps : RoundOps F32Bits := {
  zero := positiveZero
  isZero := fun value => (exponent value == 0#8) && (fraction value == 0#23)
  fromScaledInt := scaledIntToF32
  fromScaledNat := scaledNatToF32
  add := addRNE
  rescale := rescaleNat
  divide := divideRNE
}

/-- Finish the row with the contract's zero-denominator rule and one bfloat16 RNE conversion. -/
def finishRowBF16 (state : RowState F32Bits) : BF16Bits :=
  toBF16RNE (finishRow binary32RoundOps state)

/-- Concrete bit-level evaluation of one output channel from materialized blocks. -/
def evalLSSAChannel (blocks : List Block) : BF16Bits :=
  finishRowBF16 (canonicalRow binary32RoundOps blocks)

/-- Concrete bit-level evaluation when semantic segment boundaries are already established. -/
def evalLSSAChannelSegments (segments : List (List Block)) : BF16Bits :=
  finishRowBF16 (canonicalTwoLevelRow binary32RoundOps segments)

/-- Canonical score-scale constant for head dimension 128 and 26 lattice bits. -/
def scoreScaleBase : Nat := 8_557_550

/-- Exact integer score-scale rule after the contract clamp to `[-7, 23]`. -/
def scoreScale (queryExponent keyExponent : Int) : Nat :=
  let shift := clampScoreScale (queryExponent + keyExponent)
  let shifted :=
    if 0 ≤ shift then
      (scoreScaleBase + if shift = 0 then 0 else 2^(shift - 1).toNat) / 2^shift.toNat
    else
      scoreScaleBase * 2^(-shift).toNat
  min (2^30) (max 1 shifted)

/-- Ceiling division by a positive natural denominator, returned as a signed integer. -/
def ceilDivInt (value : Int) (denominator : Nat) : Int :=
  if denominator = 0 then 0
  else if 0 ≤ value then
    Int.ofNat ((value.toNat + denominator - 1) / denominator)
  else
    -Int.ofNat (value.natAbs / denominator)

/-- Quantized key/value token within one absolute semantic block. -/
structure QuantizedToken where
  key : List Int
  value : List Int
  deriving Repr, DecidableEq

structure QuantizedBlock where
  keyExponent : Int
  valueExponent : Int
  tokens : List QuantizedToken
  deriving Repr, DecidableEq

structure QuantizedRow where
  query : List Int
  queryExponent : Int
  blocks : List QuantizedBlock
  deriving Repr, DecidableEq

/-- Exact integer dot product; unequal trailing dimensions are rejected by omission. -/
def quantizedScore (query key : List Int) : Int :=
  (List.zipWith (fun q k => q * k) query key).sum

private def maxInt : List Int → Int
  | [] => 0
  | value :: rest => rest.foldl max value

private def weightForScore (baseline score : Int) (scale : Nat) : Nat :=
  let distance := (baseline * Int.ofNat (2^26) - score * Int.ofNat scale).toNat
  t2 (tableIndex distance)

/-- Materialize the exact block metadata and one channel's weighted terms from q8/k8/v8. -/
def materializeBlock (query : List Int) (queryExponent : Int) (channel : Nat)
    (block : QuantizedBlock) : Block :=
  let scores := block.tokens.map fun token => quantizedScore query token.key
  let scale := scoreScale queryExponent block.keyExponent
  let maximum := maxInt scores
  let baseline := ceilDivInt (maximum * Int.ofNat scale) (2^26)
  let terms := (block.tokens.zip scores).map fun item => {
    weight := weightForScore baseline item.2 scale
    value := item.1.value.getD channel 0
  }
  { baseline, valueExponent := block.valueExponent, terms }

/-- End-to-end concrete LSSA-B8 evaluation from quantized row inputs to one bfloat16 channel. -/
def evalQuantizedRowChannel (row : QuantizedRow) (channel : Nat) : BF16Bits :=
  evalLSSAChannel (row.blocks.map (materializeBlock row.query row.queryExponent channel))

/-- Head-dimension-128 bit output of one quantized LSSA-B8 row. -/
def evalQuantizedRow (row : QuantizedRow) : List BF16Bits :=
  (List.range 128).map (evalQuantizedRowChannel row)

private def singleTokenRow : QuantizedRow := {
  query := [1]
  queryExponent := 0
  blocks := [{
    keyExponent := 0
    valueExponent := 0
    tokens := [{ key := [1], value := 2 :: List.replicate 127 0 }]
  }]
}

def singleTokenRowOutput : List BF16Bits :=
  evalQuantizedRow singleTokenRow

@[simp] theorem score_scale_base_vector : scoreScale 0 0 = scoreScaleBase := by decide

@[simp] theorem ceil_div_positive_vector : ceilDivInt 67_108_865 67_108_864 = 2 := by decide

@[simp] theorem ceil_div_negative_vector : ceilDivInt (-67_108_865) 67_108_864 = -1 := by decide

/-- Complete quantized-input-to-bfloat16 bridge vector. -/
theorem single_token_row_vector :
    singleTokenRowOutput = 0x4000#16 :: List.replicate 127 0#16 := by
  native_decide

end Attention
end Lockstep
