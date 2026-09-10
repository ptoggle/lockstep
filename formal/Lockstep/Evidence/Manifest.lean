import Lockstep.Evidence.Transcript

namespace Lockstep
namespace Evidence

inductive PreimageAtom where
  | domain (value : String)
  | dtype (value : String)
  | shape (value : List Nat)
  | separator
  | byte (value : Byte)
  deriving Repr, DecidableEq

def tensorHeader (dtype : String) (shape : List Nat) : List PreimageAtom :=
  .domain "lockstep-tensor-v1" :: .separator :: .dtype dtype :: .separator ::
    [.shape shape, .separator]

def tensorPreimage (dtype : String) (shape : List Nat) (raw : List Byte) : List PreimageAtom :=
  tensorHeader dtype shape ++ raw.map PreimageAtom.byte

/-- Dtype is part of the digest preimage and cannot be reinterpreted silently. -/
theorem tensor_header_separates_dtype {a b : String} {shape : List Nat}
    (h : a ≠ b) : tensorHeader a shape ≠ tensorHeader b shape := by
  intro heq
  simp [tensorHeader] at heq
  exact h heq

/-- Shape is part of the digest preimage and cannot be reinterpreted silently. -/
theorem tensor_header_separates_shape {dtype : String} {a b : List Nat}
    (h : a ≠ b) : tensorHeader dtype a ≠ tensorHeader dtype b := by
  intro heq
  simp [tensorHeader] at heq
  exact h heq

structure TensorRecord where
  name : String
  dtype : String
  shape : List Nat
  digest : String
  deriving Repr, DecidableEq

structure Manifest where
  contractVersion : String
  model : String
  t2Digest : String
  tensorParallelWorldSize : Nat
  tensors : List TensorRecord
  deriving Repr, DecidableEq

/-- Validation is fail closed: the complete typed record must equal the expected record. -/
def validateManifest (expected candidate : Manifest) : Bool :=
  decide (candidate = expected)

theorem manifest_validation_sound {expected candidate : Manifest}
    (h : validateManifest expected candidate = true) : candidate = expected := by
  simpa [validateManifest] using h

def tamperT2 (candidate : Manifest) (replacement : String) : Manifest :=
  { candidate with t2Digest := replacement }

theorem manifest_tamper_rejected (expected : Manifest) {replacement : String}
    (h : replacement ≠ expected.t2Digest) :
    validateManifest expected (tamperT2 expected replacement) = false := by
  have hne : tamperT2 expected replacement ≠ expected := by
    intro heq
    have hd := congrArg Manifest.t2Digest heq
    exact h (by simpa [tamperT2] using hd)
  simp [validateManifest, hne]

theorem tampered_manifest_counterexample :
    validateManifest
      { contractVersion := "v2.4", model := "m", t2Digest := "good",
        tensorParallelWorldSize := 1, tensors := [] }
      { contractVersion := "v2.4", model := "m", t2Digest := "bad",
        tensorParallelWorldSize := 1, tensors := [] } = false := by decide

end Evidence
end Lockstep
