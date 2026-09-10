import Lockstep.Decoder.Operators

namespace Lockstep
namespace Decoder

/-- Arithmetic boundary for rank-ordered fp32 reduction followed by one bf16 rounding. -/
structure RankRoundOps (input widened output : Type) where
  zero : widened
  widen : input → widened
  add : widened → widened → widened
  roundOutput : widened → output

def orderedRankAccumulator (ops : RankRoundOps input widened output)
    (partials : List input) : widened :=
  partials.foldl (fun acc value => ops.add acc (ops.widen value)) ops.zero

def orderedRankFold (ops : RankRoundOps input widened output)
    (partials : List input) : output :=
  ops.roundOutput (orderedRankAccumulator ops partials)

abbrev IndexedPartial (input : Type) := Nat × input

def RankPartition.Valid (worldSize : Nat) (canonical : List input)
    (scheduled : List (IndexedPartial input)) : Prop :=
  scheduled.map Prod.fst = List.range worldSize ∧
  scheduled.map Prod.snd = canonical

/-- Exact ascending rank indices make the schedule complete, disjoint, and order preserving. -/
theorem rank_partition_complete_disjoint
    {worldSize : Nat} {canonical : List input}
    {scheduled : List (IndexedPartial input)}
    (h : RankPartition.Valid worldSize canonical scheduled) :
    scheduled.map Prod.snd = canonical ∧ (scheduled.map Prod.fst).Nodup := by
  constructor
  · exact h.2
  · rw [h.1]
    exact List.nodup_range

def scheduledRankFold (ops : RankRoundOps input widened output)
    (scheduled : List (IndexedPartial input)) : output :=
  orderedRankFold ops (scheduled.map Prod.snd)

theorem rank_partition_refines_ordered_fold
    (ops : RankRoundOps input widened output)
    {worldSize : Nat} {canonical : List input}
    {scheduled : List (IndexedPartial input)}
    (h : RankPartition.Valid worldSize canonical scheduled) :
    scheduledRankFold ops scheduled = orderedRankFold ops canonical := by
  unfold scheduledRankFold
  rw [h.2]

end Decoder
end Lockstep
