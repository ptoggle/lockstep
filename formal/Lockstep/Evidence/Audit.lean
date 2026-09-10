import Lockstep.Evidence.Trace

namespace Lockstep
namespace Evidence
namespace Audit

/-- Kernel-reducible binomial coefficient used for samples without replacement. -/
def choose : Nat → Nat → Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => choose n k + choose n (k + 1)

theorem choose_zero_of_lt {n k : Nat} (h : n < k) : choose n k = 0 := by
  induction n generalizing k with
  | zero =>
      cases k with
      | zero => omega
      | succ k => rfl
  | succ n ih =>
      cases k with
      | zero => omega
      | succ k =>
          unfold choose
          have hnk : n < k := by omega
          have hnks : n < k + 1 := by omega
          rw [ih hnk, ih hnks]
/-- A finite sample of row indices; validity enforces sampling without replacement. -/
abbrev Sample (totalRows : Nat) := List (Fin totalRows)

def Sample.Valid (sampleSize : Nat) (sample : Sample totalRows) : Prop :=
  sample.length = sampleSize ∧ sample.Nodup

def Sample.Clean (validRows : Nat) (sample : Sample totalRows) : Prop :=
  ∀ index ∈ sample, index.val < validRows


/-- Number of clean size-`sampleSize` subsets when `badRows` rows are invalid. -/
def cleanSampleCount (totalRows badRows sampleSize : Nat) : Nat :=
  choose (totalRows - badRows) sampleSize

/-- Number of subsets that detect at least one invalid row. -/
def detectionCount (totalRows badRows sampleSize : Nat) : Nat :=
  choose totalRows sampleSize - cleanSampleCount totalRows badRows sampleSize

theorem clean_sample_count (totalRows badRows sampleSize : Nat) :
    cleanSampleCount totalRows badRows sampleSize =
      choose (totalRows - badRows) sampleSize := rfl

theorem detection_count (totalRows badRows sampleSize : Nat) :
    detectionCount totalRows badRows sampleSize =
      choose totalRows sampleSize - choose (totalRows - badRows) sampleSize := rfl

/-- Sampling more rows than can all be valid makes detection certain. -/
theorem detection_certain_when_sample_exceeds_valid_rows
    {totalRows badRows sampleSize : Nat}
    (h : totalRows - badRows < sampleSize) :
    cleanSampleCount totalRows badRows sampleSize = 0 ∧
    detectionCount totalRows badRows sampleSize = choose totalRows sampleSize := by
  have hz := choose_zero_of_lt h
  constructor
  · exact hz
  · simp [detectionCount, cleanSampleCount, hz]

def openedRowConforms (expected opened : RowTranscript) : Bool :=
  decide (opened = expected)

/-- One differing opened transcript is a direct bit-level nonconformance witness. -/
theorem opened_difference_is_nonconformance {expected opened : RowTranscript}
    (h : opened ≠ expected) : openedRowConforms expected opened = false := by
  simp [openedRowConforms, h]


abbrev MerkleLeafHash := Nat → List Byte → Digest256
abbrev MerkleNodeHash := Digest256 → Digest256 → Digest256

structure MerkleStep where
  siblingOnLeft : Bool
  sibling : Digest256
  deriving Repr, DecidableEq

def foldMerklePath (node : MerkleNodeHash) :
    Nat → Digest256 → List MerkleStep → Option Digest256
  | index, current, [] => if index = 0 then some current else none
  | index, current, step :: rest =>
      let expectedLeft := index % 2 = 1
      if step.siblingOnLeft = expectedLeft then
        let parent :=
          if step.siblingOnLeft then node step.sibling current
          else node current step.sibling
        foldMerklePath node (index / 2) parent rest
      else none

structure RowOpening where
  index : Nat
  row : RowTranscript
  path : List MerkleStep
  deriving Repr, DecidableEq

def checkOpening (leaf : MerkleLeafHash) (node : MerkleNodeHash)
    (root : Digest256) (expectedIndex : Nat) (expectedRow : RowTranscript)
    (opening : RowOpening) : Bool :=
  opening.index == expectedIndex &&
  opening.row == expectedRow &&
  foldMerklePath node opening.index
    (leaf opening.index (rowEncode opening.row)) opening.path == some root

def OpeningValid (leaf : MerkleLeafHash) (node : MerkleNodeHash)
    (root : Digest256) (expectedIndex : Nat) (expectedRow : RowTranscript)
    (opening : RowOpening) : Prop :=
  opening.index = expectedIndex ∧
  opening.row = expectedRow ∧
  foldMerklePath node opening.index
    (leaf opening.index (rowEncode opening.row)) opening.path = some root

theorem check_opening_sound (leaf : MerkleLeafHash) (node : MerkleNodeHash)
    (root : Digest256) (expectedIndex : Nat) (expectedRow : RowTranscript)
    (opening : RowOpening)
    (h : checkOpening leaf node root expectedIndex expectedRow opening = true) :
    OpeningValid leaf node root expectedIndex expectedRow opening := by
  simp only [checkOpening, Bool.and_eq_true] at h
  rcases h with ⟨⟨hindex, hrow⟩, hpath⟩
  exact ⟨by simpa using hindex, by simpa using hrow, by simpa using hpath⟩

inductive OpeningsValid (leaf : MerkleLeafHash) (node : MerkleNodeHash)
    (root : Digest256) :
    List (Nat × RowTranscript) → List RowOpening → Prop
  | nil : OpeningsValid leaf node root [] []
  | cons (expectedIndex : Nat) (expectedRow : RowTranscript)
      (opening : RowOpening) (expected : List (Nat × RowTranscript))
      (openings : List RowOpening)
      (hopening : OpeningValid leaf node root expectedIndex expectedRow opening)
      (hrest : OpeningsValid leaf node root expected openings) :
      OpeningsValid leaf node root
        ((expectedIndex, expectedRow) :: expected) (opening :: openings)

def checkOpenings (leaf : MerkleLeafHash) (node : MerkleNodeHash)
    (root : Digest256) :
    List (Nat × RowTranscript) → List RowOpening → Bool
  | [], [] => true
  | (expectedIndex, expectedRow) :: expected, opening :: openings =>
      checkOpening leaf node root expectedIndex expectedRow opening &&
      checkOpenings leaf node root expected openings
  | _, _ => false

theorem check_openings_sound (leaf : MerkleLeafHash) (node : MerkleNodeHash)
    (root : Digest256) (expected : List (Nat × RowTranscript))
    (openings : List RowOpening)
    (h : checkOpenings leaf node root expected openings = true) :
    OpeningsValid leaf node root expected openings := by
  induction expected generalizing openings with
  | nil =>
      cases openings with
      | nil => exact OpeningsValid.nil
      | cons opening openings => simp [checkOpenings] at h
  | cons pair expected ih =>
      rcases pair with ⟨expectedIndex, expectedRow⟩
      cases openings with
      | nil => simp [checkOpenings] at h
      | cons opening openings =>
          simp only [checkOpenings, Bool.and_eq_true] at h
          exact OpeningsValid.cons expectedIndex expectedRow opening expected openings
            (check_opening_sound leaf node root expectedIndex expectedRow opening h.1)
            (ih openings h.2)
end Audit
end Evidence
end Lockstep
