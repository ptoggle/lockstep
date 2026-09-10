import Lockstep.Evidence.Transcript

namespace Lockstep
namespace Evidence

abbrev Digest256 := BitVec 256

/-- The hash primitive is abstract in Lean. Its four inputs bind the deployment
context, previous chain value, record sequence number, and canonical row bytes. -/
abbrev TraceHash := Digest256 → Digest256 → Word32 → List Byte → Digest256

structure AuthenticatedRow where
  sequence : Word32
  transcript : RowTranscript
  previous : Digest256
  digest : Digest256
  deriving Repr, DecidableEq

structure AuthenticatedTrace where
  context : Digest256
  rows : List AuthenticatedRow
  root : Digest256
  deriving Repr, DecidableEq

def rowDigest (hash : TraceHash) (context previous : Digest256)
    (sequence : Word32) (row : RowTranscript) : Digest256 :=
  hash context previous sequence (rowEncode row)

inductive ChainValid (hash : TraceHash) (context : Digest256) :
    Word32 → Digest256 → List AuthenticatedRow → Digest256 → Prop
  | nil (sequence previous) : ChainValid hash context sequence previous [] previous
  | cons (sequence previous) (row : AuthenticatedRow) (rest : List AuthenticatedRow)
      (hsequence : row.sequence = sequence)
      (hprevious : row.previous = previous)
      (hdigest : row.digest = rowDigest hash context previous sequence row.transcript)
      (hrest : ChainValid hash context (sequence + 1) row.digest rest root) :
      ChainValid hash context sequence previous (row :: rest) root

def checkChain (hash : TraceHash) (context : Digest256) :
    Word32 → Digest256 → List AuthenticatedRow → Digest256 → Bool
  | _, previous, [], root => previous == root
  | sequence, previous, row :: rest, root =>
      row.sequence == sequence &&
      row.previous == previous &&
      row.digest == rowDigest hash context previous sequence row.transcript &&
      checkChain hash context (sequence + 1) row.digest rest root

theorem check_chain_sound (hash : TraceHash) (context : Digest256)
    (sequence : Word32) (previous : Digest256) (rows : List AuthenticatedRow)
    (root : Digest256)
    (h : checkChain hash context sequence previous rows root = true) :
    ChainValid hash context sequence previous rows root := by
  induction rows generalizing sequence previous with
  | nil =>
      simp [checkChain] at h
      subst root
      exact ChainValid.nil sequence previous
  | cons row rest ih =>
      simp only [checkChain, Bool.and_eq_true] at h
      rcases h with ⟨⟨⟨hsequence, hprevious⟩, hdigest⟩, hrest⟩
      apply ChainValid.cons sequence previous row rest
      · simpa using hsequence
      · simpa using hprevious
      · simpa using hdigest
      · exact ih _ _ hrest

def checkTrace (hash : TraceHash) (expectedContext initial expectedRoot : Digest256)
    (trace : AuthenticatedTrace) : Bool :=
  trace.context == expectedContext &&
  trace.root == expectedRoot &&
  checkChain hash trace.context 0 initial trace.rows trace.root

def TraceValid (hash : TraceHash) (expectedContext initial expectedRoot : Digest256)
    (trace : AuthenticatedTrace) : Prop :=
  trace.context = expectedContext ∧
  trace.root = expectedRoot ∧
  ChainValid hash trace.context 0 initial trace.rows trace.root

/-- Acceptance by the executable checker implies every opened record is in
canonical sequence and authenticated to the externally committed root. -/
theorem check_trace_sound (hash : TraceHash)
    (expectedContext initial expectedRoot : Digest256) (trace : AuthenticatedTrace)
    (h : checkTrace hash expectedContext initial expectedRoot trace = true) :
    TraceValid hash expectedContext initial expectedRoot trace := by
  simp only [checkTrace, Bool.and_eq_true] at h
  rcases h with ⟨⟨hcontext, hroot⟩, hchain⟩
  refine ⟨?_, ?_, ?_⟩
  · simpa using hcontext
  · simpa using hroot
  · exact check_chain_sound hash trace.context 0 initial trace.rows trace.root hchain

private def sampleHash : TraceHash := fun context previous sequence bytes =>
  context ^^^ previous ^^^ BitVec.zeroExtend 256 sequence ^^^
    BitVec.ofNat 256 bytes.length

private def sampleTranscript : RowTranscript :=
  { layer := 2, row := 7, head := 3, payload := [0x3f800000#32, 0x40000000#32] }

private def sampleDigest : Digest256 :=
  rowDigest sampleHash 11 0 0 sampleTranscript

private def sampleRow : AuthenticatedRow :=
  { sequence := 0
    transcript := sampleTranscript
    previous := 0
    digest := sampleDigest }

private def sampleTrace : AuthenticatedTrace :=
  { context := 11
    rows := [sampleRow]
    root := sampleDigest }

example : checkTrace sampleHash 11 0 sampleDigest sampleTrace = true := by native_decide

example : checkTrace sampleHash 12 0 sampleDigest sampleTrace = false := by native_decide

end Evidence
end Lockstep
