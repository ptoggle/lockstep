import Lockstep.Decoder.Operators

namespace Lockstep
namespace Decoder

/-- Absolute half-open suffix selected by a finite causal window. -/
def visibleStart (prefixLength window : Nat) : Nat :=
  prefixLength - min prefixLength window

def visibleRange (prefixLength : Nat) : Option Nat → Nat × Nat
  | none => (0, prefixLength)
  | some window => (visibleStart prefixLength window, prefixLength)

@[simp] theorem visible_range_full (length : Nat) :
    visibleRange length none = (0, length) := rfl

@[simp] theorem visible_range_finite (length window : Nat) :
    visibleRange length (some window) =
      (length - min length window, length) := rfl

inductive AttentionFamily where
  | groupedQuery
  | multiheadLatent (latentDim ropeDim valueDim : Nat)
  deriving Repr, DecidableEq

inductive FeedForwardFamily where
  | dense
  | sparseExperts (experts topK : Nat)
  deriving Repr, DecidableEq

/-- Family-specific structure around the shared decoder operator contract.

MLA is required to expand its latent cache to ordinary exact-integer K/V rows
before LSSA. MoE routing is required to order exact integer router scores by
`(score descending, expert index ascending)` and to combine in that order.
-/
structure FamilyContract where
  headDim : Nat
  window : Option Nat
  attention : AttentionFamily
  feedForward : FeedForwardFamily
  deriving Repr, DecidableEq

def FamilyContract.Valid (contract : FamilyContract) : Prop :=
  contract.headDim = 128 ∧
  (match contract.window with
    | none => True
    | some window => 0 < window) ∧
  (match contract.attention with
    | .groupedQuery => True
    | .multiheadLatent latentDim ropeDim valueDim =>
        0 < latentDim ∧ 0 < ropeDim ∧ 0 < valueDim) ∧
  (match contract.feedForward with
    | .dense => True
    | .sparseExperts experts topK =>
        0 < experts ∧ 0 < topK ∧ topK ≤ experts)

instance familyContractValidDecidable (contract : FamilyContract) :
    Decidable contract.Valid := by
  cases contract with
  | mk headDim window attention feedForward =>
      cases window <;> cases attention <;> cases feedForward <;>
        simp only [FamilyContract.Valid] <;> infer_instance

def checkFamilyContract (contract : FamilyContract) : Bool := decide contract.Valid

theorem check_family_contract_sound {contract : FamilyContract}
    (h : checkFamilyContract contract = true) : contract.Valid := by
  simpa [checkFamilyContract] using of_decide_eq_true h

private def mixtralContract : FamilyContract := {
  headDim := 128
  window := some 4096
  attention := .groupedQuery
  feedForward := .sparseExperts 8 2
}

private def deepseekContract : FamilyContract := {
  headDim := 128
  window := none
  attention := .multiheadLatent 512 64 128
  feedForward := .sparseExperts 256 8
}

@[simp] theorem mixtral_contract_accepts :
    checkFamilyContract mixtralContract = true := by decide

@[simp] theorem deepseek_contract_accepts :
    checkFamilyContract deepseekContract = true := by decide

@[simp] theorem zero_window_rejected :
    checkFamilyContract {
      headDim := 128
      window := some 0
      attention := .groupedQuery
      feedForward := .dense
    } = false := by decide

end Decoder
end Lockstep
