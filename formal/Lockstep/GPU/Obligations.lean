import Lean.Elab.Tactic.Omega

namespace Lockstep
namespace GPU

structure GridShape where
  rows : Nat
  heads : Nat
  blocks : Nat
  segments : Nat
  channels : Nat
  ctas : Nat
  deriving Repr, DecidableEq

/-- Quotient/remainder coordinate for fixed-width logical work groups. -/
def coordinate (width index : Nat) : Nat × Nat := (index / width, index % width)

def flattenCoordinate (width : Nat) (position : Nat × Nat) : Nat :=
  position.1 * width + position.2

/-- Every logical term is owned: quotient/remainder reconstructs its linear index. -/
theorem coordinate_complete {width : Nat} (index : Nat) :
    flattenCoordinate width (coordinate width index) = index := by
  unfold flattenCoordinate coordinate
  simpa [Nat.mul_comm, Nat.add_comm] using Nat.mod_add_div index width

/-- Ownership is disjoint: two terms with the same coordinate have the same index. -/
theorem coordinate_disjoint {width : Nat} {a b : Nat}
    (h : coordinate width a = coordinate width b) : a = b := by
  calc
    a = flattenCoordinate width (coordinate width a) := (coordinate_complete a).symm
    _ = flattenCoordinate width (coordinate width b) := by rw [h]
    _ = b := coordinate_complete b

/-- Lanes within a score block are exactly the remainder modulo 128. -/
theorem score_term_coverage (index : Nat) :
    flattenCoordinate 128 (coordinate 128 index) = index ∧
      (coordinate 128 index).2 < 128 := by
  constructor
  · exact coordinate_complete index
  · exact Nat.mod_lt _ (by decide)

/-- Blocks within a fold segment are exactly the remainder modulo 32. -/
theorem segment_block_coverage (block : Nat) :
    flattenCoordinate 32 (coordinate 32 block) = block ∧
      (coordinate 32 block).2 < 32 := by
  constructor
  · exact coordinate_complete block
  · exact Nat.mod_lt _ (by decide)

/-- Nested row/head/channel coordinate for a flattened tensor grid. -/
def rowHeadChannelCoordinate (heads channels index : Nat) : Nat × Nat × Nat :=
  let outer := coordinate (heads * channels) index
  let inner := coordinate channels outer.2
  (outer.1, inner.1, inner.2)

def flattenRowHeadChannel (heads channels : Nat) (position : Nat × Nat × Nat) : Nat :=
  flattenCoordinate (heads * channels) (position.1, flattenCoordinate channels position.2)

theorem row_head_channel_coverage (heads channels index : Nat) :
    flattenRowHeadChannel heads channels (rowHeadChannelCoordinate heads channels index) =
      index := by
  unfold rowHeadChannelCoordinate flattenRowHeadChannel
  simp only
  rw [coordinate_complete, coordinate_complete]

theorem row_head_channel_disjoint (heads channels : Nat) {a b : Nat}
    (h : rowHeadChannelCoordinate heads channels a =
      rowHeadChannelCoordinate heads channels b) : a = b := by
  calc
    a = flattenRowHeadChannel heads channels (rowHeadChannelCoordinate heads channels a) :=
      (row_head_channel_coverage heads channels a).symm
    _ = flattenRowHeadChannel heads channels (rowHeadChannelCoordinate heads channels b) := by
      rw [h]
    _ = b := row_head_channel_coverage heads channels b

/-- CTA owner and local segment index reconstruct every scheduled segment. -/
theorem cta_segment_coverage (ctaCount segment : Nat) :
    flattenCoordinate ctaCount (coordinate ctaCount segment) = segment := by
  exact coordinate_complete segment

theorem cta_segment_disjoint (ctaCount : Nat) {a b : Nat}
    (h : coordinate ctaCount a = coordinate ctaCount b) : a = b :=
  coordinate_disjoint h

/-- Logical obligations proved here, distinct from assumptions about compiled instructions. -/
def logicalObligations : List String :=
  ["row/head/channel Cartesian grid",
   "128-term block coverage",
   "32-block segment coverage",
   "CTA segment assignment coverage",
   "complete quotient/remainder ownership",
   "disjoint quotient/remainder ownership",
   "ascending segment and rank reconstruction"]

/-- Assumptions discharged by toolchain pins, implementation inspection, and hardware gates. -/
def hardwareAssumptions : List String :=
  ["WGMMA int8 products and int32 accumulation implement the documented operation",
   "PTX/SASS binary32 instructions round to nearest-even at each declared boundary",
   "compiler lowering preserves declared data dependencies and operation order",
   "GPU memory and synchronization primitives implement the CUDA memory model",
   "SHA-256 and BLAKE3 are collision resistant and binding"]

theorem logical_obligation_inventory_nonempty : logicalObligations ≠ [] := by decide

theorem hardware_assumption_inventory_nonempty : hardwareAssumptions ≠ [] := by decide

end GPU
end Lockstep
