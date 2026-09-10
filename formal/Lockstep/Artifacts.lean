import Lockstep.Contract.Domain
import Lockstep.Numeric.Bits
import Lockstep.Numeric.Bounds
import Lockstep.Numeric.Binary32
import Lockstep.Attention.Table
import Lockstep.Attention.LSSA
import Lockstep.Attention.Schedule
import Lockstep.Attention.Concrete
import Lockstep.Attention.Negative
import Lockstep.Runtime.Plan
import Lockstep.Decoder.Operators
import Lockstep.Decoder.TensorParallel
import Lockstep.Evidence.Transcript
import Lockstep.Evidence.Trace
import Lockstep.Evidence.Manifest
import Lockstep.Evidence.Audit
import Lockstep.Analysis.Approximation
import Lockstep.GPU.Obligations

namespace Lockstep
namespace Artifacts

open Attention

private def quoted (value : String) : String := "\"" ++ value ++ "\""

private def stringArray (values : List String) : String :=
  "[" ++ String.intercalate "," (values.map quoted) ++ "]"

private def nat (value : Nat) : String := toString value
private def natArray (values : List Nat) : String :=
  "[" ++ String.intercalate "," (values.map nat) ++ "]"

private def binary32Vector (a b : Numeric.F32Bits) : String :=
  "{\"a\":" ++ nat a.toNat ++
    ",\"b\":" ++ nat b.toNat ++
    ",\"add\":" ++ nat (Numeric.addRNE a b).toNat ++
    ",\"divide\":" ++ nat (Numeric.divideRNE a b).toNat ++ "}"

private def binary32Vectors : String :=
  "[" ++ String.intercalate "," [
    binary32Vector 0x3f800000#32 0x3f800000#32,
    binary32Vector 0x3f800000#32 0x40000000#32,
    binary32Vector 0xbf800000#32 0x3f800000#32,
    binary32Vector 0x00800000#32 0x40000000#32,
    binary32Vector 0x00000001#32 0x00000001#32,
    binary32Vector 0x3f800001#32 0x3f7fffff#32] ++ "]"

private def singleTokenRowBits : String :=
  natArray (Attention.singleTokenRowOutput.map (·.toNat))
private def rescaleVector (input : Numeric.F32Bits) (distance : Numeric.ExpBits) : String :=
  "{\"input\":" ++ nat input.toNat ++
    ",\"distance\":" ++ nat distance.toNat ++
    ",\"output\":" ++ nat (Numeric.rescaleBits input distance).toNat ++ "}"

private def rescaleVectors : String :=
  "[" ++ String.intercalate "," [
    rescaleVector 0x3f800000#32 1#8,
    rescaleVector 0x80000000#32 4#8,
    rescaleVector 0x00800000#32 1#8,
    rescaleVector 0x7f800000#32 3#8] ++ "]"

private def t2Anchors : String :=
  "[" ++ String.intercalate "," [
    "{\"index\":0,\"value\":" ++ nat (t2 0) ++ "}",
    "{\"index\":128,\"value\":" ++ nat (t2 128) ++ "}",
    "{\"index\":1151,\"value\":" ++ nat (t2 1151) ++ "}",
    "{\"index\":1152,\"value\":" ++ nat (t2 1152) ++ "}"] ++ "]"

private def bool (value : Bool) : String := if value then "true" else "false"

private def fidelityExample : Analysis.FidelityCertificate :=
  { valueContribution := -3
    weightContribution := 5
    observedError := 2
    valueBound := 3
    weightBound := 5
    policyLimit := 8 }

private def auditExampleDetection : Nat :=
  Evidence.Audit.detectionCount 10 2 3

/-- Canonical byte-stable bridge artifact. Field order and whitespace are part of the output. -/
def json : String :=
  "{\n" ++
  "  \"schema\":\"lockstep-lean-artifacts-v2\",\n" ++
  "  \"lean_toolchain\":\"leanprover/lean4:v4.33.1\",\n" ++
  "  \"contract\":{\"head_dim\":128,\"block_size\":128,\"segment_blocks\":32," ++
    "\"score_frac_bits\":7,\"score_hi_bits\":26,\"weight_bits\":8},\n" ++
  "  \"bounds\":{\"score_abs\":2064512,\"block_value_abs\":4145280," ++
    "\"block_weight\":32640,\"linear_acc_exclusive\":536870912},\n" ++
  "  \"t2\":{\"length\":" ++ nat t2Length ++
    ",\"live_length\":" ++ nat t2LiveLength ++
    ",\"digest\":" ++ quoted t2Digest ++
    ",\"csv\":" ++ quoted t2Csv ++
    ",\"anchors\":" ++ t2Anchors ++ "},\n" ++
  "  \"rescale_vectors\":" ++ rescaleVectors ++ ",\n" ++
  "  \"binary32\":{\"operations\":" ++ binary32Vectors ++
    ",\"bf16_ties\":[" ++ nat (Numeric.toBF16RNE 0x3f808000#32).toNat ++ "," ++
      nat (Numeric.toBF16RNE 0x3f818000#32).toNat ++
    "],\"single_token_row\":" ++ singleTokenRowBits ++ "},\n" ++
  "  \"negative_controls\":{" ++
    "\"descending_fold\":[" ++ nat (Negative.roundedFold [8, 4, 2]) ++ "," ++
      nat (Negative.roundedFold [2, 4, 8]) ++ "]," ++
    "\"nonaligned_chunk\":[" ++ nat (Negative.weightedMean [(1, 0), (3, 10)]) ++ "," ++
      nat ((Negative.weightedMean [(1, 0)] + Negative.weightedMean [(3, 10)]) / 2) ++ "]," ++
    "\"approx_division\":[" ++ nat (Negative.exactDivision 7 3) ++ "," ++
      nat (Negative.powerOfTwoDivision 7 3) ++ "]," ++
    "\"multiply_rescale\":[" ++ nat (Negative.rescaleThenAdd 3 3) ++ "," ++
      nat (Negative.multiplyWholeAccumulator 3 3) ++ "]},\n" ++
  "  \"transcript\":{\"word\":1280529234,\"little_endian\":[82,83,83,76]},\n" ++
  "  \"audit_example\":{\"total_rows\":10,\"bad_rows\":2,\"sample_size\":3," ++
    "\"clean_samples\":56,\"detecting_samples\":" ++ nat auditExampleDetection ++ "},\n" ++
  "  \"fidelity_certificate\":{\"value_contribution\":-3,\"weight_contribution\":5," ++
    "\"observed_error\":2,\"value_bound\":3,\"weight_bound\":5,\"policy_limit\":8," ++
    "\"accepted\":" ++ bool (Analysis.checkFidelityCertificate fidelityExample) ++ "},\n" ++
  "  \"gpu\":{\"logical_obligations\":" ++ stringArray GPU.logicalObligations ++
    ",\"hardware_assumptions\":" ++ stringArray GPU.hardwareAssumptions ++ "}\n" ++
  "}\n"

def checkReport : String :=
  "LOCKSTEP-FORMAL PASS: contract, numeric, LSSA, decoder, evidence, audit, and GPU logical obligations built"

end Artifacts
end Lockstep
