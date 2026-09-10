import Lean.Elab.Tactic.Omega

namespace Lockstep
namespace Contract

/-- Static constants whose values are part of contract v2.4/LSSA-B8. -/
structure ContractParams where
  headDim : Nat
  keyBlock : Nat
  segmentBlocks : Nat
  scoreBits : Nat
  indexBits : Nat
  weightTop : Nat
  weightWindow : Nat
  skipWindow : Nat
  expLow : Int
  expHigh : Int
  scoreScaleLow : Int
  scoreScaleHigh : Int
  deriving Repr, DecidableEq

/-- The unique parameter set mechanized by this development. -/
def canonicalParams : ContractParams := {
  headDim := 128
  keyBlock := 128
  segmentBlocks := 32
  scoreBits := 26
  indexBits := 7
  weightTop := 255
  weightWindow := 9
  skipWindow := 32
  expLow := -32
  expHigh := 40
  scoreScaleLow := -7
  scoreScaleHigh := 23
}

/-- Static parameter validity. Equalities make contract-version drift explicit. -/
def ParamsValid (p : ContractParams) : Prop :=
  p.headDim = 128 ∧
  p.keyBlock = 128 ∧
  p.segmentBlocks = 32 ∧
  p.scoreBits = 26 ∧
  p.indexBits = 7 ∧
  p.weightTop = 255 ∧
  p.weightWindow = 9 ∧
  p.skipWindow = 32 ∧
  p.expLow = -32 ∧
  p.expHigh = 40 ∧
  p.scoreScaleLow = -7 ∧
  p.scoreScaleHigh = 23

instance paramsValidDecidable (p : ContractParams) : Decidable (ParamsValid p) := by
  unfold ParamsValid
  infer_instance

inductive RejectReason where
  | contractParameters
  | zeroHeads
  | headGrouping
  | headDimension
  | hiddenAlignment
  | tensorParallel
  | mixtureOfExperts
  | multimodal
  | encoderDecoder
  | prequantized
  | slidingWindow
  deriving Repr, DecidableEq

/-- Manifest-visible model envelope relevant to the formalized static obligations. -/
structure ModelConfig where
  hidden : Nat
  heads : Nat
  kvHeads : Nat
  headDim : Nat
  intermediate : Nat
  tpWorld : Nat
  mixtureOfExperts : Bool := false
  multimodal : Bool := false
  encoderDecoder : Bool := false
  prequantized : Bool := false
  unsupportedSlidingWindow : Bool := false
  deriving Repr, DecidableEq

/-- Dense Llama-like v2.4 admission predicate. -/
def ModelValid (m : ModelConfig) : Prop :=
  0 < m.heads ∧
  0 < m.kvHeads ∧
  m.kvHeads ∣ m.heads ∧
  m.headDim = 128 ∧
  m.hidden = m.heads * m.headDim ∧
  m.hidden % 256 = 0 ∧
  0 < m.intermediate ∧
  0 < m.tpWorld ∧
  m.tpWorld ∣ m.heads ∧
  m.mixtureOfExperts = false ∧
  m.multimodal = false ∧
  m.encoderDecoder = false ∧
  m.prequantized = false ∧
  m.unsupportedSlidingWindow = false

/-- Full static admitted-domain predicate. Dynamic tensor bounds are modeled by later modules. -/
def Admitted (p : ContractParams) (m : ModelConfig) : Prop :=
  ParamsValid p ∧ ModelValid m

def validateParams (p : ContractParams) : Except RejectReason Unit :=
  if ParamsValid p then .ok () else .error .contractParameters

/-- Fail-closed executable envelope check with a stable first rejection reason. -/
def validateModel (p : ContractParams) (m : ModelConfig) : Except RejectReason Unit :=
  if ¬ ParamsValid p then .error .contractParameters
  else if m.heads = 0 ∨ m.kvHeads = 0 then .error .zeroHeads
  else if ¬ m.kvHeads ∣ m.heads then .error .headGrouping
  else if m.headDim ≠ 128 ∨ m.hidden ≠ m.heads * m.headDim then .error .headDimension
  else if m.hidden % 256 ≠ 0 then .error .hiddenAlignment
  else if m.intermediate = 0 ∨ m.tpWorld = 0 ∨ ¬ m.tpWorld ∣ m.heads then .error .tensorParallel
  else if m.mixtureOfExperts then .error .mixtureOfExperts
  else if m.multimodal then .error .multimodal
  else if m.encoderDecoder then .error .encoderDecoder
  else if m.prequantized then .error .prequantized
  else if m.unsupportedSlidingWindow then .error .slidingWindow
  else .ok ()

@[simp] theorem canonicalParams_valid : ParamsValid canonicalParams := by
  decide

theorem validateParams_sound {p : ContractParams}
    (h : validateParams p = .ok ()) : ParamsValid p := by
  simp only [validateParams] at h
  split at h
  · assumption
  · contradiction

theorem validateModel_sound {p : ContractParams} {m : ModelConfig}
    (h : validateModel p m = .ok ()) : Admitted p m := by
  simp only [validateModel] at h
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h <;> try contradiction
  split at h <;> try contradiction
  simp_all [Admitted, ParamsValid, ModelValid]
  omega

theorem admitted_head_group_divides {p : ContractParams} {m : ModelConfig}
    (h : Admitted p m) : m.kvHeads ∣ m.heads ∧ m.tpWorld ∣ m.heads := by
  exact ⟨h.2.2.2.1, h.2.2.2.2.2.2.2.2.2.1⟩

theorem admitted_lssa_constants {p : ContractParams} {m : ModelConfig}
    (h : Admitted p m) :
    p.headDim = 128 ∧ p.keyBlock = 128 ∧ p.segmentBlocks = 32 ∧
    p.scoreBits = 26 ∧ p.indexBits = 7 ∧ p.weightTop = 255 := by
  exact ⟨h.1.1, h.1.2.1, h.1.2.2.1, h.1.2.2.2.1,
    h.1.2.2.2.2.1, h.1.2.2.2.2.2.1⟩

/-- Representative admitted 32-head/8-KV-head configuration used by artifact checks. -/
def exampleModel : ModelConfig := {
  hidden := 4096
  heads := 32
  kvHeads := 8
  headDim := 128
  intermediate := 11008
  tpWorld := 4
}

example : validateModel canonicalParams exampleModel = .ok () := by rfl

end Contract
end Lockstep
