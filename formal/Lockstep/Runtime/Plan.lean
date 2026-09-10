import Lockstep.Attention.LSSA

namespace Lockstep
namespace Runtime

structure BlockRange where
  start : Nat
  stop : Nat
  deriving Repr, DecidableEq

structure BlockPlan where
  totalKeys : Nat
  blockSize : Nat
  blocks : List BlockRange
  deriving Repr, DecidableEq

private def ceilDiv (value divisor : Nat) : Nat :=
  if divisor = 0 then 0 else (value + divisor - 1) / divisor

def canonicalBlockPlan (totalKeys : Nat) : List BlockRange :=
  (List.range (ceilDiv totalKeys 128)).map fun index => {
    start := index * 128
    stop := min totalKeys ((index + 1) * 128)
  }

def BlockPlan.Valid (plan : BlockPlan) : Prop :=
  plan.blockSize = 128 ∧ plan.blocks = canonicalBlockPlan plan.totalKeys

instance blockPlanValidDecidable (plan : BlockPlan) : Decidable plan.Valid := by
  unfold BlockPlan.Valid
  infer_instance

def checkBlockPlan (plan : BlockPlan) : Bool := decide plan.Valid

theorem check_block_plan_sound {plan : BlockPlan} (h : checkBlockPlan plan = true) :
    plan.Valid := by
  simpa [checkBlockPlan] using of_decide_eq_true h

/-- Prompt chunks expressed as consecutive half-open absolute token ranges. -/
structure ChunkPlan where
  totalKeys : Nat
  blockSize : Nat
  chunks : List BlockRange
  deriving Repr, DecidableEq

private def RangesCoverFrom (blockSize totalKeys cursor : Nat) : List BlockRange → Prop
  | [] => cursor = totalKeys
  | range :: rest =>
      range.start = cursor ∧
      cursor < range.stop ∧
      range.stop ≤ totalKeys ∧
      (range.stop = totalKeys ∨ range.stop % blockSize = 0) ∧
      RangesCoverFrom blockSize totalKeys range.stop rest

private instance rangesCoverFromDecidable (blockSize totalKeys cursor : Nat)
    (ranges : List BlockRange) :
    Decidable (RangesCoverFrom blockSize totalKeys cursor ranges) :=
  match ranges with
  | [] => by
      simp only [RangesCoverFrom]
      infer_instance
  | range :: rest => by
      simp only [RangesCoverFrom]
      have : Decidable (RangesCoverFrom blockSize totalKeys range.stop rest) :=
        rangesCoverFromDecidable blockSize totalKeys range.stop rest
      infer_instance

def ChunkPlan.Valid (plan : ChunkPlan) : Prop :=
  plan.blockSize = 128 ∧ RangesCoverFrom plan.blockSize plan.totalKeys 0 plan.chunks

instance chunkPlanValidDecidable (plan : ChunkPlan) : Decidable plan.Valid := by
  unfold ChunkPlan.Valid
  infer_instance

def checkChunkPlan (plan : ChunkPlan) : Bool := decide plan.Valid

theorem check_chunk_plan_sound {plan : ChunkPlan} (h : checkChunkPlan plan = true) :
    plan.Valid := by
  simpa [checkChunkPlan] using of_decide_eq_true h

structure SegmentWork where
  owner : Nat
  blocks : List Nat
  deriving Repr, DecidableEq

structure SegmentPlan where
  totalBlocks : Nat
  blocksPerSegment : Nat
  ctaCount : Nat
  work : List SegmentWork
  deriving Repr, DecidableEq

private def segmentBlocks (plan : SegmentPlan) : List (List Nat) :=
  plan.work.map (·.blocks)

def SegmentPlan.Valid (plan : SegmentPlan) : Prop :=
  plan.blocksPerSegment = 32 ∧
  0 < plan.ctaCount ∧
  segmentBlocks plan = Attention.segmentize (List.range plan.totalBlocks) ∧
  plan.work.all (fun assignment => decide (assignment.owner < plan.ctaCount)) = true

instance segmentPlanValidDecidable (plan : SegmentPlan) : Decidable plan.Valid := by
  unfold SegmentPlan.Valid
  infer_instance

def checkSegmentPlan (plan : SegmentPlan) : Bool := decide plan.Valid

theorem check_segment_plan_sound {plan : SegmentPlan} (h : checkSegmentPlan plan = true) :
    plan.Valid := by
  simpa [checkSegmentPlan] using of_decide_eq_true h

/-- Rank order is semantic even when transport uses all-gather or all-to-all. -/
structure RankPlan where
  worldSize : Nat
  order : List Nat
  deriving Repr, DecidableEq


def RankPlan.Valid (plan : RankPlan) : Prop :=
  0 < plan.worldSize ∧ plan.order = List.range plan.worldSize

instance rankPlanValidDecidable (plan : RankPlan) : Decidable plan.Valid := by
  unfold RankPlan.Valid
  infer_instance

def checkRankPlan (plan : RankPlan) : Bool := decide plan.Valid

theorem check_rank_plan_sound {plan : RankPlan} (h : checkRankPlan plan = true) :
    plan.Valid := by
  simpa [checkRankPlan] using of_decide_eq_true h

private def validBlockVector : BlockPlan := {
  totalKeys := 257
  blockSize := 128
  blocks := [{ start := 0, stop := 128 }, { start := 128, stop := 256 },
    { start := 256, stop := 257 }]
}

private def validChunkVector : ChunkPlan := {
  totalKeys := 257
  blockSize := 128
  chunks := [{ start := 0, stop := 128 }, { start := 128, stop := 256 },
    { start := 256, stop := 257 }]
}

private def validSegmentVector : SegmentPlan := {
  totalBlocks := 33
  blocksPerSegment := 32
  ctaCount := 2
  work := [
    { owner := 0, blocks := List.range 32 },
    { owner := 1, blocks := [32] }
  ]
}

@[simp] theorem valid_block_vector_accepts : checkBlockPlan validBlockVector = true := by decide
@[simp] theorem valid_chunk_vector_accepts : checkChunkPlan validChunkVector = true := by decide
@[simp] theorem valid_segment_vector_accepts : checkSegmentPlan validSegmentVector = true := by
  native_decide
@[simp] theorem wrong_rank_order_rejected :
    checkRankPlan { worldSize := 3, order := [0, 2, 1] } = false := by decide

end Runtime
end Lockstep
