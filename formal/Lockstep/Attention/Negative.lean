import Lockstep.Attention.Schedule

set_option maxRecDepth 10000

namespace Lockstep
namespace Attention
namespace Negative

/-- Tiny deterministic rounding model used only to witness order sensitivity. -/
def roundedAverage (a b : Nat) : Nat := (a + b) / 2

def roundedFold (xs : List Nat) : Nat := xs.foldl roundedAverage 0

/-- A rounded fold cannot in general be traversed in descending order. -/
theorem descending_fold_counterexample :
    roundedFold [8, 4, 2] ≠ roundedFold [2, 4, 8] := by decide

private def averagingOps : RoundOps Nat := {
  zero := 0
  isZero := fun value => decide (value = 0)
  fromScaledInt := fun value _ => value.toNat
  fromScaledNat := fun value _ => value
  add := roundedAverage
  rescale := fun value _ => value
  divide := fun numerator _ => numerator
}

private def witnessPartial (value : Int) : BlockPartial := {
  baseline := 0
  value
  weight := 1
  valueExponent := 0
}

/-- An independently rounded two-level fold is not a flattened block fold in general. -/
theorem two_level_fold_is_not_flat :
    (foldSegmentPartials averagingOps
      [[witnessPartial 8, witnessPartial 4], [witnessPartial 2]]).output ≠
    (foldPartials averagingOps
      [witnessPartial 8, witnessPartial 4, witnessPartial 2]).output := by
  decide

def weightedMean (xs : List (Nat × Nat)) : Nat :=
  (xs.map fun x => x.1 * x.2).sum / (xs.map Prod.fst).sum

/-- Splitting inside a block and averaging independently changes the answer. -/
theorem nonaligned_chunk_counterexample :
    weightedMean [(1, 0), (3, 10)] ≠
      (weightedMean [(1, 0)] + weightedMean [(3, 10)]) / 2 := by decide

def truncatedT2 (limit k : Nat) : Nat := if k < limit then t2 k else 0

/-- The published last nonzero T2 entry detects a table truncated by one entry. -/
theorem table_truncation_counterexample :
    truncatedT2 1151 1151 ≠ t2 1151 := by decide

def exactDivision (numerator denominator : Nat) : Nat := numerator / denominator

def powerOfTwoDivision (numerator denominator : Nat) : Nat :=
  let roundedDenominator := if denominator ≤ 2 then 2 else 4
  numerator / roundedDenominator

/-- Replacing the mandated division with a nearby power-of-two divisor changes results. -/
theorem approx_division_counterexample :
    exactDivision 7 3 ≠ powerOfTwoDivision 7 3 := by decide

def rescaleThenAdd (old fresh : Nat) : Nat := old / 2 + fresh

def multiplyWholeAccumulator (old fresh : Nat) : Nat := (old + fresh) / 2

/-- Rescaling the old accumulator is observably different from scaling the whole sum. -/
theorem multiply_rescale_counterexample :
    rescaleThenAdd 3 3 ≠ multiplyWholeAccumulator 3 3 := by decide

end Negative
end Attention
end Lockstep
