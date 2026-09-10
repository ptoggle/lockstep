import Lockstep.Attention.Table
import Lockstep.Numeric.Bounds
import Lean.Elab.Tactic.Omega

namespace Lockstep
namespace Attention

open Numeric

structure ScoreTerm where
  value : Int
  deriving Repr, DecidableEq

def ScoreTerm.Valid (t : ScoreTerm) : Prop := t.value.natAbs ≤ 127 * 127

def score (terms : List ScoreTerm) : Int := (terms.map (·.value)).sum

theorem score_abs_le {terms : List ScoreTerm}
    (hlen : terms.length ≤ 128)
    (hvalid : ∀ t ∈ terms, t.Valid) : (score terms).natAbs ≤ 2_064_512 := by
  have hb : TermsBounded (127 * 127) (terms.map (·.value)) := by
    intro x hx
    simp only [List.mem_map] at hx
    rcases hx with ⟨t, ht, rfl⟩
    exact hvalid t ht
  have hs := sum_natAbs_le_length_mul hb
  have hm : (terms.map (·.value)).length ≤ 128 := by simpa using hlen
  unfold score
  simp only [List.length_map] at hs
  omega

structure WeightedTerm where
  weight : Nat
  value : Int
  deriving Repr, DecidableEq

def WeightedTerm.Valid (t : WeightedTerm) : Prop :=
  t.weight ≤ 255 ∧ t.value.natAbs ≤ 127

def WeightedTerm.product (t : WeightedTerm) : Int :=
  Int.ofNat t.weight * t.value

structure Block where
  baseline : Int
  valueExponent : Int
  terms : List WeightedTerm
  deriving Repr, DecidableEq

def Block.Valid (b : Block) : Prop :=
  b.terms.length ≤ 128 ∧
  (∀ t ∈ b.terms, t.Valid) ∧
  -32 ≤ b.valueExponent ∧ b.valueExponent ≤ 40

def blockValue (b : Block) : Int := (b.terms.map WeightedTerm.product).sum
def blockWeight (b : Block) : Nat := (b.terms.map (·.weight)).sum

theorem weighted_term_abs_le {t : WeightedTerm} (h : t.Valid) :
    t.product.natAbs ≤ 255 * 127 := by
  rcases h with ⟨hw, hv⟩
  unfold WeightedTerm.product
  rw [Int.natAbs_mul]
  change t.weight * t.value.natAbs ≤ 255 * 127
  exact Nat.mul_le_mul hw hv

theorem block_value_abs_le {b : Block} (h : b.Valid) :
    (blockValue b).natAbs ≤ 4_145_280 := by
  have hb : TermsBounded (255 * 127) (b.terms.map WeightedTerm.product) := by
    intro x hx
    simp only [List.mem_map] at hx
    rcases hx with ⟨t, ht, rfl⟩
    exact weighted_term_abs_le (h.2.1 t ht)
  have hs := sum_natAbs_le_length_mul hb
  unfold blockValue
  simp only [List.length_map] at hs
  have hm : b.terms.length * (255 * 127) ≤ 128 * (255 * 127) :=
    Nat.mul_le_mul_right _ h.1
  omega

private theorem nat_sum_le_length_mul {bound : Nat} {xs : List Nat}
    (h : ∀ x ∈ xs, x ≤ bound) : xs.sum ≤ xs.length * bound := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      have hx : x ≤ bound := h x (by simp)
      have hxs : ∀ y ∈ xs, y ≤ bound := by
        intro y hy
        exact h y (by simp [hy])
      have hi := ih hxs
      simp only [List.sum_cons, List.length_cons]
      calc
        x + xs.sum ≤ bound + xs.length * bound := Nat.add_le_add hx hi
        _ = (xs.length + 1) * bound := by simp [Nat.add_mul, Nat.add_comm]

theorem block_weight_le {b : Block} (h : b.Valid) : blockWeight b ≤ 32_640 := by
  have hw : ∀ x ∈ b.terms.map (·.weight), x ≤ 255 := by
    intro x hx
    simp only [List.mem_map] at hx
    rcases hx with ⟨t, ht, rfl⟩
    exact (h.2.1 t ht).1
  have hs := nat_sum_le_length_mul hw
  unfold blockWeight
  simp only [List.length_map] at hs
  have hm : b.terms.length * 255 ≤ 128 * 255 := Nat.mul_le_mul_right _ h.1
  omega

/-- Abstract deterministic implementations of the explicitly ordered binary32 boundary. -/
structure RoundOps (α : Type) where
  zero : α
  isZero : α → Bool
  fromScaledInt : Int → Int → α
  fromScaledNat : Nat → Int → α
  add : α → α → α
  rescale : α → Nat → α
  divide : α → α → α

structure FoldState (α : Type) where
  output : α
  normalizer : α
  baseline : Option Int

abbrev SegmentState (α : Type) := FoldState α
abbrev RowState (α : Type) := FoldState α

structure BlockPartial where
  baseline : Int
  value : Int
  weight : Nat
  valueExponent : Int
  deriving Repr, DecidableEq

def partialOfBlock (b : Block) : BlockPartial := {
  baseline := b.baseline
  value := blockValue b
  weight := blockWeight b
  valueExponent := b.valueExponent
}

private def doublingDistance (older newer : Int) : Nat :=
  if newer ≤ older then (older - newer).toNat else 0

/-- One committed-order block transition. `baseline` is already expressed in doublings. -/
def foldBlock (ops : RoundOps α) (st : FoldState α) (p : BlockPartial) : FoldState α :=
  match st.baseline with
  | none => {
      output := ops.add ops.zero (ops.fromScaledInt p.value (-p.valueExponent))
      normalizer := ops.add ops.zero (ops.fromScaledNat p.weight 0)
      baseline := some p.baseline
    }
  | some current =>
      if current < p.baseline then
        let d := (p.baseline - current).toNat
        {
          output := ops.add (ops.rescale st.output d)
            (ops.fromScaledInt p.value (-p.valueExponent))
          normalizer := ops.add (ops.rescale st.normalizer d)
            (ops.fromScaledNat p.weight 0)
          baseline := some p.baseline
        }
      else
        let s := doublingDistance current p.baseline
        if 32 < s then st
        else {
          output := ops.add st.output (ops.fromScaledInt p.value (-(Int.ofNat s) - p.valueExponent))
          normalizer := ops.add st.normalizer (ops.fromScaledNat p.weight (-(Int.ofNat s)))
          baseline := some current
        }

def emptyFoldState (ops : RoundOps α) : FoldState α :=
  { output := ops.zero, normalizer := ops.zero, baseline := none }

def foldPartials (ops : RoundOps α) (ps : List BlockPartial) : FoldState α :=
  ps.foldl (foldBlock ops) (emptyFoldState ops)

/-- A segment is evaluated from a fresh state, independently of every other segment. -/
def foldSegment (ops : RoundOps α) (ps : List BlockPartial) : SegmentState α :=
  foldPartials ops ps

/--
Merge one completed segment into the row state.  This is the outer rounded recurrence from
the contract, not continuation of the inner block fold.
-/
def mergeSegment (ops : RoundOps α) (row : RowState α)
    (segment : SegmentState α) : RowState α :=
  match segment.baseline with
  | none => row
  | some segmentBaseline =>
      match row.baseline with
      | none => {
          output := ops.add row.output (ops.rescale segment.output 0)
          normalizer := ops.add row.normalizer (ops.rescale segment.normalizer 0)
          baseline := some segmentBaseline
        }
      | some rowBaseline =>
          if rowBaseline < segmentBaseline then
            let d := (segmentBaseline - rowBaseline).toNat
            {
              output := ops.add (ops.rescale row.output d) (ops.rescale segment.output 0)
              normalizer :=
                ops.add (ops.rescale row.normalizer d) (ops.rescale segment.normalizer 0)
              baseline := some segmentBaseline
            }
          else
            let s := (rowBaseline - segmentBaseline).toNat
            if 32 < s then row
            else {
              output := ops.add row.output (ops.rescale segment.output s)
              normalizer := ops.add row.normalizer (ops.rescale segment.normalizer s)
              baseline := some rowBaseline
            }

def foldSegmentPartials (ops : RoundOps α)
    (segments : List (List BlockPartial)) : RowState α :=
  (segments.map (foldSegment ops)).foldl (mergeSegment ops) (emptyFoldState ops)

private def segmentizeAux : Nat → List β → List (List β)
  | 0, _ => []
  | _ + 1, [] => []
  | fuel + 1, values =>
      values.take 32 :: segmentizeAux fuel (values.drop 32)

/-- Canonical consecutive 32-block segmentation. -/
def segmentize (values : List β) : List (List β) :=
  segmentizeAux values.length values

/-- Explicit two-level evaluator over already established semantic segments. -/
def canonicalTwoLevelRow (ops : RoundOps α) (segments : List (List Block)) : RowState α :=
  foldSegmentPartials ops (segments.map fun segment => segment.map partialOfBlock)

/-- Canonical row evaluator using consecutive 32-block semantic segments. -/
def canonicalRow (ops : RoundOps α) (blocks : List Block) : RowState α :=
  foldSegmentPartials ops (segmentize (blocks.map partialOfBlock))

def finishRow (ops : RoundOps α) (st : FoldState α) : α :=
  if ops.isZero st.normalizer then ops.zero else ops.divide st.output st.normalizer

def EvaluatesTo (ops : RoundOps α) (blocks : List Block) (output : α) : Prop :=
  finishRow ops (canonicalRow ops blocks) = output

/-- The canonical evaluation relation has at most one output. -/
theorem lssa_deterministic (ops : RoundOps α) (blocks : List Block) {a b : α}
    (ha : EvaluatesTo ops blocks a) (hb : EvaluatesTo ops blocks b) : a = b := by
  exact ha.symm.trans hb

end Attention
end Lockstep
