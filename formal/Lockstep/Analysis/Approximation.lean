import Lockstep.Numeric.Bounds

namespace Lockstep
namespace Analysis

/-- Magnitude of the residual from an integer grid point. -/
def quantizationError (x q : Int) (step : Nat) : Nat :=
  (x - q * Int.ofNat step).natAbs

def RoundsToNearestGrid (x q : Int) (step : Nat) : Prop :=
  2 * quantizationError x q step ≤ step

/-- Nearest-grid quantization contributes at most half a step. -/
theorem quantization_error_le_half_step {x q : Int} {step : Nat}
    (h : RoundsToNearestGrid x q step) :
    2 * quantizationError x q step ≤ step := h

abbrev WeightedError := Nat × Int

def weightedErrorNumerator : List WeightedError → Int
  | [] => 0
  | (weight, error) :: rest => Int.ofNat weight * error + weightedErrorNumerator rest

def totalWeight : List WeightedError → Nat
  | [] => 0
  | (weight, _) :: rest => weight + totalWeight rest

/-- Value quantization changes a weighted-average numerator by at most total weight times error. -/
theorem weighted_average_value_error {errors : List WeightedError} {epsilon : Nat}
    (h : ∀ item ∈ errors, item.2.natAbs ≤ epsilon) :
    (weightedErrorNumerator errors).natAbs ≤ totalWeight errors * epsilon := by
  induction errors with
  | nil => simp [weightedErrorNumerator, totalWeight]
  | cons item rest ih =>
      have hitem : item.2.natAbs ≤ epsilon := h item (by simp)
      have hrest : ∀ x ∈ rest, x.2.natAbs ≤ epsilon := by
        intro x hx
        exact h x (by simp [hx])
      have hi := ih hrest
      rcases item with ⟨weight, error⟩
      simp only [weightedErrorNumerator, totalWeight]
      calc
        (Int.ofNat weight * error + weightedErrorNumerator rest).natAbs
            ≤ (Int.ofNat weight * error).natAbs + (weightedErrorNumerator rest).natAbs :=
              Int.natAbs_add_le _ _
        _ ≤ weight * epsilon + totalWeight rest * epsilon := by
              apply Nat.add_le_add
              · rw [Int.natAbs_mul]
                change weight * error.natAbs ≤ weight * epsilon
                exact Nat.mul_le_mul_left weight hitem
              · exact hi
        _ = (weight + totalWeight rest) * epsilon := by
              simp [Nat.add_mul]

abbrev WeightValueError := Int × Int

def weightValueErrorNumerator (terms : List WeightValueError) : Int :=
  (terms.map fun item => item.1 * item.2).sum

/-- Perturbed normalized weights contribute a bounded numerator error over bounded values. -/
theorem normalized_weight_perturbation_bound
    {terms : List WeightValueError} {count weightEpsilon valueBound : Nat}
    (hlen : terms.length ≤ count)
    (hweight : ∀ item ∈ terms, item.1.natAbs ≤ weightEpsilon)
    (hvalue : ∀ item ∈ terms, item.2.natAbs ≤ valueBound) :
    (weightValueErrorNumerator terms).natAbs ≤ count * (weightEpsilon * valueBound) := by
  have hb : Numeric.TermsBounded (weightEpsilon * valueBound)
      (terms.map fun item => item.1 * item.2) := by
    intro product hp
    simp only [List.mem_map] at hp
    rcases hp with ⟨item, hi, rfl⟩
    rw [Int.natAbs_mul]
    exact Nat.mul_le_mul (hweight item hi) (hvalue item hi)
  have hs := Numeric.sum_natAbs_le_length_mul hb
  unfold weightValueErrorNumerator
  simp only [List.length_map] at hs
  exact Nat.le_trans hs (Nat.mul_le_mul_right _ hlen)

/-- Local attention error is the sum of value-grid and normalized-weight contributions. -/
theorem local_attention_error_bound {valueContribution weightContribution : Int}
    {valueBound weightBound : Nat}
    (hv : valueContribution.natAbs ≤ valueBound)
    (hw : weightContribution.natAbs ≤ weightBound) :
    (valueContribution + weightContribution).natAbs ≤ valueBound + weightBound := by
  exact Nat.le_trans (Int.natAbs_add_le _ _) (Nat.add_le_add hv hw)

theorem local_attention_error_zero {valueContribution weightContribution : Int}
    (hv : valueContribution.natAbs ≤ 0)
    (hw : weightContribution.natAbs ≤ 0) :
    (valueContribution + weightContribution).natAbs = 0 := by
  have h := local_attention_error_bound hv hw
  omega


/-- Executable certificate for one signed fixed-point output error. The two
contributions must reconstruct the observed error and each satisfy its declared
bound; the deployment policy supplies the final limit. -/
structure FidelityCertificate where
  valueContribution : Int
  weightContribution : Int
  observedError : Int
  valueBound : Nat
  weightBound : Nat
  policyLimit : Nat
  deriving Repr, DecidableEq

def checkFidelityCertificate (certificate : FidelityCertificate) : Bool :=
  certificate.observedError ==
      certificate.valueContribution + certificate.weightContribution &&
  decide (certificate.valueContribution.natAbs ≤ certificate.valueBound) &&
  decide (certificate.weightContribution.natAbs ≤ certificate.weightBound) &&
  decide (certificate.valueBound + certificate.weightBound ≤ certificate.policyLimit)

/-- An accepted executable certificate bounds the observed fixed-point error by
the policy limit. -/
theorem fidelity_certificate_sound (certificate : FidelityCertificate)
    (h : checkFidelityCertificate certificate = true) :
    certificate.observedError.natAbs ≤ certificate.policyLimit := by
  simp only [checkFidelityCertificate, Bool.and_eq_true] at h
  rcases h with ⟨⟨⟨hobservation, hvalue⟩, hweight⟩, hlimit⟩
  have heq : certificate.observedError =
      certificate.valueContribution + certificate.weightContribution := by
    simpa using hobservation
  have hv : certificate.valueContribution.natAbs ≤ certificate.valueBound :=
    of_decide_eq_true hvalue
  have hw : certificate.weightContribution.natAbs ≤ certificate.weightBound :=
    of_decide_eq_true hweight
  have hl : certificate.valueBound + certificate.weightBound ≤ certificate.policyLimit :=
    of_decide_eq_true hlimit
  rw [heq]
  apply Nat.le_trans (Int.natAbs_add_le _ _)
  apply Nat.le_trans (Nat.add_le_add hv hw)
  exact hl

example : checkFidelityCertificate {
    valueContribution := -3
    weightContribution := 5
    observedError := 2
    valueBound := 3
    weightBound := 5
    policyLimit := 8 } = true := by native_decide

example : checkFidelityCertificate {
    valueContribution := -3
    weightContribution := 5
    observedError := 3
    valueBound := 3
    weightBound := 5
    policyLimit := 8 } = false := by native_decide
end Analysis
end Lockstep
