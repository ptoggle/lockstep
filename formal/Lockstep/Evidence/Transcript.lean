import Std.Tactic.BVDecide

namespace Lockstep
namespace Evidence

abbrev Byte := BitVec 8
abbrev Word32 := BitVec 32

def wordEncode (w : Word32) : List Byte :=
  [BitVec.extractLsb 7 0 w, BitVec.extractLsb 15 8 w,
   BitVec.extractLsb 23 16 w, BitVec.extractLsb 31 24 w]

def wordDecode : List Byte → Option Word32
  | [a, b, c, d] => some (
      BitVec.zeroExtend 32 a |||
      (BitVec.zeroExtend 32 b <<< 8) |||
      (BitVec.zeroExtend 32 c <<< 16) |||
      (BitVec.zeroExtend 32 d <<< 24))
  | _ => none

theorem word_encode_decode (w : Word32) : wordDecode (wordEncode w) = some w := by
  simp [wordDecode, wordEncode]
  bv_decide

def wordsEncode : List Word32 → List Byte
  | [] => []
  | w :: ws => wordEncode w ++ wordsEncode ws

def wordsDecode : List Byte → Option (List Word32)
  | [] => some []
  | a :: b :: c :: d :: rest =>
      match wordDecode [a, b, c, d], wordsDecode rest with
      | some w, some ws => some (w :: ws)
      | _, _ => none
  | _ => none

private theorem wordsDecode_wordEncode_append (w : Word32) (bytes : List Byte) :
    wordsDecode (wordEncode w ++ bytes) =
      match wordsDecode bytes with
      | some ws => some (w :: ws)
      | none => none := by
  unfold wordEncode
  have hw :
      wordDecode
        [BitVec.extractLsb 7 0 w, BitVec.extractLsb 15 8 w,
         BitVec.extractLsb 23 16 w, BitVec.extractLsb 31 24 w] = some w := by
    simpa [wordEncode] using word_encode_decode w
  change
    (match
      wordDecode
        [BitVec.extractLsb 7 0 w, BitVec.extractLsb 15 8 w,
         BitVec.extractLsb 23 16 w, BitVec.extractLsb 31 24 w],
      wordsDecode bytes with
    | some decoded, some rest => some (decoded :: rest)
    | _, _ => none) =
      match wordsDecode bytes with
      | some rest => some (w :: rest)
      | none => none
  rw [hw]
  cases wordsDecode bytes <;> rfl

theorem words_encode_decode (ws : List Word32) : wordsDecode (wordsEncode ws) = some ws := by
  induction ws with
  | nil => rfl
  | cons w ws ih =>
      rw [wordsEncode, wordsDecode_wordEncode_append, ih]

structure RowTranscript where
  layer : Word32
  row : Word32
  head : Word32
  payload : List Word32
  deriving Repr, DecidableEq

def rowMagic : Word32 := 0x4c535352#32

def rowEncode (row : RowTranscript) : List Byte :=
  wordsEncode (rowMagic :: row.layer :: row.row :: row.head :: row.payload)

def rowDecode (bytes : List Byte) : Option RowTranscript := do
  let words ← wordsDecode bytes
  match words with
  | magic :: layer :: row :: head :: payload =>
      if magic = rowMagic then some { layer, row, head, payload } else none
  | _ => none

theorem row_encode_decode (row : RowTranscript) : rowDecode (rowEncode row) = some row := by
  simp [rowDecode, rowEncode, words_encode_decode, rowMagic]

theorem row_encoding_injective {a b : RowTranscript}
    (h : rowEncode a = rowEncode b) : a = b := by
  have hd : rowDecode (rowEncode a) = rowDecode (rowEncode b) := congrArg rowDecode h
  simpa [row_encode_decode] using hd

end Evidence
end Lockstep
