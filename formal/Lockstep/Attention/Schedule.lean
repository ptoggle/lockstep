import Lockstep.Attention.LSSA

set_option maxRecDepth 10000

namespace Lockstep
namespace Attention

private theorem int_sum_eq_of_perm {xs ys : List Int}
    (h : xs.Perm ys) : xs.sum = ys.sum := by
  induction h with
  | nil => rfl
  | cons _ _ ih => simp [ih]
  | swap _ _ _ =>
      simp only [List.sum_cons]
      omega
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

private theorem nat_sum_eq_of_perm {xs ys : List Nat}
    (h : xs.Perm ys) : xs.sum = ys.sum := by
  induction h with
  | nil => rfl
  | cons _ _ ih => simp [ih]
  | swap _ _ _ =>
      simp only [List.sum_cons]
      omega
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

/-- A scheduled block may reorder lanes, but not metadata or membership. -/
def BlockRefines (scheduled canonical : Block) : Prop :=
  scheduled.baseline = canonical.baseline ∧
  scheduled.valueExponent = canonical.valueExponent ∧
  scheduled.terms.Perm canonical.terms

theorem block_reorder_invariant {scheduled canonical : Block}
    (h : BlockRefines scheduled canonical) :
    partialOfBlock scheduled = partialOfBlock canonical := by
  rcases h with ⟨hb, he, hp⟩
  have hv : blockValue scheduled = blockValue canonical := by
    unfold blockValue
    exact int_sum_eq_of_perm (hp.map WeightedTerm.product)
  have hw : blockWeight scheduled = blockWeight canonical := by
    unfold blockWeight
    exact nat_sum_eq_of_perm (hp.map (·.weight))
  cases scheduled
  cases canonical
  simp_all [partialOfBlock]

/-- The hardware schedule preserves block order and only permutes lanes within blocks. -/
inductive ScheduleRefines : List Block → List Block → Prop where
  | nil : ScheduleRefines [] []
  | cons {scheduled canonical scheduledTail canonicalTail} :
      BlockRefines scheduled canonical →
      ScheduleRefines scheduledTail canonicalTail →
      ScheduleRefines (scheduled :: scheduledTail) (canonical :: canonicalTail)

private theorem partials_eq_of_schedule {scheduled canonical : List Block}
    (h : ScheduleRefines scheduled canonical) :
    scheduled.map partialOfBlock = canonical.map partialOfBlock := by
  induction h with
  | nil => rfl
  | cons hblock htail ih =>
      simp only [List.map_cons]
      rw [block_reorder_invariant hblock, ih]

theorem scheduled_refines_canonical (ops : RoundOps α)
    {scheduled canonical : List Block}
    (h : ScheduleRefines scheduled canonical) :
    canonicalRow ops scheduled = canonicalRow ops canonical := by
  unfold canonicalRow
  rw [partials_eq_of_schedule h]

/-- Chunks are block-aligned because their element type is `Block`; no partial block is representable. -/
def AlignedChunks (chunks : List (List Block)) (canonical : List Block) : Prop :=
  chunks.flatten = canonical

theorem aligned_chunk_blocks_equal {chunks : List (List Block)} {canonical : List Block}
    (h : AlignedChunks chunks canonical) : chunks.flatten = canonical := h

def chunkedRow (ops : RoundOps α) (chunks : List (List Block)) : FoldState α :=
  canonicalRow ops chunks.flatten

theorem aligned_chunk_refines_canonical (ops : RoundOps α)
    {chunks : List (List Block)} {canonical : List Block}
    (h : AlignedChunks chunks canonical) :
    chunkedRow ops chunks = canonicalRow ops canonical := by
  unfold chunkedRow
  rw [h]

abbrev IndexedBlock := Nat × Block

def SegmentAssignment.Valid (total : Nat) (canonical : List (List Block))
    (segments : List (List IndexedBlock)) : Prop :=
  (segments.flatten.map Prod.fst = List.range total) ∧
  (segments.map fun segment => segment.map Prod.snd) = canonical

/--
Ascending global indices establish complete, disjoint ownership, while the nested value
equality preserves the contract's semantic segment boundaries.
-/
theorem segment_assignment_complete_disjoint {total : Nat}
    {canonical : List (List Block)} {segments : List (List IndexedBlock)}
    (h : SegmentAssignment.Valid total canonical segments) :
    (segments.map fun segment => segment.map Prod.snd) = canonical ∧
      (segments.flatten.map Prod.fst).Nodup := by
  constructor
  · exact h.2
  · rw [h.1]
    exact List.nodup_range

/-- Split-KV computes each segment independently and merges completed states in order. -/
def splitKVRow (ops : RoundOps α) (segments : List (List IndexedBlock)) : FoldState α :=
  canonicalTwoLevelRow ops (segments.map fun segment => segment.map Prod.snd)

theorem splitKV_refines_canonical (ops : RoundOps α)
    {total : Nat} {canonical : List (List Block)}
    {segments : List (List IndexedBlock)}
    (h : SegmentAssignment.Valid total canonical segments) :
    splitKVRow ops segments = canonicalTwoLevelRow ops canonical := by
  unfold splitKVRow
  rw [h.2]

end Attention
end Lockstep
