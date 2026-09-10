import Lockstep.Numeric.Bits
import Lean.Elab.Tactic.Omega

namespace Lockstep
namespace Numeric

/-- Every signed term has magnitude at most `bound`. -/
def TermsBounded (bound : Nat) (xs : List Int) : Prop :=
  ∀ x ∈ xs, x.natAbs ≤ bound

theorem sum_natAbs_le_length_mul {bound : Nat} {xs : List Int}
    (h : TermsBounded bound xs) : xs.sum.natAbs ≤ xs.length * bound := by
  induction xs with
  | nil => simp
  | cons x xs ih =>
      have hx : x.natAbs ≤ bound := h x (by simp)
      have hxs : TermsBounded bound xs := by
        intro y hy
        exact h y (by simp [hy])
      have hi := ih hxs
      calc
        (x :: xs).sum.natAbs = (x + xs.sum).natAbs := by rfl
        _ ≤ x.natAbs + xs.sum.natAbs := Int.natAbs_add_le _ _
        _ ≤ bound + xs.length * bound := Nat.add_le_add hx hi
        _ = (x :: xs).length * bound := by simp [Nat.succ_mul, Nat.add_comm]

/-- Every intermediate reduction over a sublist is safe under the same absolute-sum bound. -/
theorem sublist_sum_safe {bound limit : Nat} {xs ys : List Int}
    (hb : TermsBounded bound xs) (hsub : List.Sublist ys xs) (hlen : xs.length * bound ≤ limit) :
    ys.sum.natAbs ≤ limit := by
  have hy : TermsBounded bound ys := by
    intro y hym
    exact hb y (hsub.subset hym)
  have hyl := sum_natAbs_le_length_mul hy
  have hlength : ys.length ≤ xs.length := hsub.length_le
  have hmul : ys.length * bound ≤ xs.length * bound := Nat.mul_le_mul_right bound hlength
  omega

@[simp] theorem score_numeric_bound : 128 * 127 * 127 = 2_064_512 := by decide
@[simp] theorem block_value_numeric_bound : 255 * 127 * 128 = 4_145_280 := by decide
@[simp] theorem block_weight_numeric_bound : 255 * 128 = 32_640 := by decide

theorem score_fits_int32 : 128 * 127 * 127 < 2^31 := by omega
theorem score_exact_in_binary32 : 128 * 127 * 127 < 2^24 := by omega
theorem block_value_fits_int32 : 255 * 127 * 128 < 2^31 := by omega
theorem block_value_exact_in_binary32 : 255 * 127 * 128 < 2^24 := by omega
theorem block_weight_exact_in_binary32 : 255 * 128 < 2^24 := by omega

/-- Any partial sum of at most 128 weighted int8 values stays below the LSSA bound. -/
theorem all_partial_sums_safe {xs ys : List Int}
    (hterms : TermsBounded (255 * 127) xs)
    (hlen : xs.length ≤ 128)
    (hsub : List.Sublist ys xs) : ys.sum.natAbs ≤ 4_145_280 := by
  apply sublist_sum_safe hterms hsub
  calc
    xs.length * (255 * 127) ≤ 128 * (255 * 127) := Nat.mul_le_mul_right _ hlen
    _ = 4_145_280 := by decide

end Numeric
end Lockstep
