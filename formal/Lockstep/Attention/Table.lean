import Lockstep.Contract.Domain
import Lean.Elab.Tactic.Omega

set_option maxRecDepth 10000

namespace Lockstep
namespace Attention

open Contract

abbrev Int8Value := Int

def clampInt8 (x : Int) : Int :=
  if x < -127 then -127 else if 127 < x then 127 else x

theorem quantized_in_int8_range (x : Int) :
    -127 ≤ clampInt8 x ∧ clampInt8 x ≤ 127 := by
  unfold clampInt8
  split
  · omega
  · split <;> omega

def clampExponent (x : Int) : Int :=
  if x < -32 then -32 else if 40 < x then 40 else x

theorem quantized_exponent_in_range (x : Int) :
    -32 ≤ clampExponent x ∧ clampExponent x ≤ 40 := by
  unfold clampExponent
  split
  · omega
  · split <;> omega

/-- Finite proxy for the exponent selector after the binary32 magnitude analysis. -/
def selectExponentProxy (rawExponent : Int) : Int := clampExponent rawExponent

theorem selected_exponent_proxy_in_range (rawExponent : Int) :
    -32 ≤ selectExponentProxy rawExponent ∧ selectExponentProxy rawExponent ≤ 40 :=
  quantized_exponent_in_range rawExponent

def clampScoreScale (x : Int) : Int :=
  if x < -7 then -7 else if 23 < x then 23 else x

theorem score_scale_in_range (x : Int) :
    -7 ≤ clampScoreScale x ∧ clampScoreScale x ≤ 23 := by
  unfold clampScoreScale
  split
  · omega
  · split <;> omega

/-- Published nonzero region of the digest-pinned T2 table. -/
def t2LiveValues : List Nat := [
  255, 254, 252, 251, 250, 248, 247, 246, 244, 243, 242, 240, 239, 238, 236, 235, 234, 233, 231, 230, 229, 228, 226, 225,
  224, 223, 222, 220, 219, 218, 217, 216, 214, 213, 212, 211, 210, 209, 208, 206, 205, 204, 203, 202, 201, 200, 199, 198,
  197, 196, 195, 193, 192, 191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177, 176, 175, 175, 174,
  173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 164, 163, 162, 161, 160, 159, 158, 157, 157, 156, 155, 154, 153, 152,
  152, 151, 150, 149, 148, 148, 147, 146, 145, 144, 144, 143, 142, 141, 141, 140, 139, 138, 138, 137, 136, 135, 135, 134,
  133, 132, 132, 131, 130, 130, 129, 128, 128, 127, 126, 125, 125, 124, 123, 123, 122, 121, 121, 120, 119, 119, 118, 118,
  117, 116, 116, 115, 114, 114, 113, 113, 112, 111, 111, 110, 110, 109, 108, 108, 107, 107, 106, 105, 105, 104, 104, 103,
  103, 102, 102, 101, 100, 100, 99, 99, 98, 98, 97, 97, 96, 96, 95, 95, 94, 94, 93, 93, 92, 92, 91, 91,
  90, 90, 89, 89, 88, 88, 87, 87, 86, 86, 85, 85, 84, 84, 84, 83, 83, 82, 82, 81, 81, 80, 80, 80,
  79, 79, 78, 78, 77, 77, 77, 76, 76, 75, 75, 75, 74, 74, 73, 73, 73, 72, 72, 71, 71, 71, 70, 70,
  70, 69, 69, 68, 68, 68, 67, 67, 67, 66, 66, 65, 65, 65, 64, 64, 64, 63, 63, 63, 62, 62, 62, 61,
  61, 61, 60, 60, 60, 59, 59, 59, 58, 58, 58, 58, 57, 57, 57, 56, 56, 56, 55, 55, 55, 54, 54, 54,
  54, 53, 53, 53, 52, 52, 52, 52, 51, 51, 51, 51, 50, 50, 50, 49, 49, 49, 49, 48, 48, 48, 48, 47,
  47, 47, 47, 46, 46, 46, 46, 45, 45, 45, 45, 44, 44, 44, 44, 43, 43, 43, 43, 42, 42, 42, 42, 42,
  41, 41, 41, 41, 40, 40, 40, 40, 40, 39, 39, 39, 39, 39, 38, 38, 38, 38, 37, 37, 37, 37, 37, 36,
  36, 36, 36, 36, 36, 35, 35, 35, 35, 35, 34, 34, 34, 34, 34, 33, 33, 33, 33, 33, 33, 32, 32, 32,
  32, 32, 32, 31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 30, 29, 29, 29, 29, 29, 29, 28, 28, 28,
  28, 28, 28, 28, 27, 27, 27, 27, 27, 27, 27, 26, 26, 26, 26, 26, 26, 26, 25, 25, 25, 25, 25, 25,
  25, 24, 24, 24, 24, 24, 24, 24, 24, 23, 23, 23, 23, 23, 23, 23, 23, 22, 22, 22, 22, 22, 22, 22,
  22, 21, 21, 21, 21, 21, 21, 21, 21, 21, 20, 20, 20, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 19,
  19, 19, 19, 19, 19, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 17, 17, 17, 17, 17, 17, 17, 17, 17,
  17, 17, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
  15, 15, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13,
  13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11,
  11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
  10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
  9, 9, 9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
  8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
  7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
]

def t2LiveLength : Nat := 9 * 2^7
def t2Length : Nat := 17 * 2^7

def t2 (k : Nat) : Nat :=
  if k < t2LiveLength then t2LiveValues.getD k 0 else 0

/-- Canonical comma-separated serialization whose SHA-256 is the published digest. -/
def t2Csv : String :=
  String.intercalate "," ((List.range t2Length).map fun index => toString (t2 index))

def tableIndex (distance : Nat) : Nat :=
  min (distance / 2^(26 - 7)) (t2Length - 1)

@[simp] theorem t2LiveValues_size : t2LiveValues.length = 1152 := by decide
@[simp] theorem t2_live_length : t2LiveLength = 1152 := by decide
@[simp] theorem t2_length : t2Length = 2176 := by decide

theorem table_index_in_bounds (distance : Nat) : tableIndex distance < t2Length := by
  unfold tableIndex
  have h : t2Length - 1 < t2Length := by decide
  exact Nat.lt_of_le_of_lt (Nat.min_le_right _ _) h

theorem t2_zero_tail {k : Nat} (h : t2LiveLength ≤ k) : t2 k = 0 := by
  have hn : ¬ k < t2LiveLength := Nat.not_lt.mpr h
  simp only [t2, hn, ↓reduceIte]

/-- Contract anchor values, including the last nonzero entry. -/
theorem t2_anchor_values :
    t2 0 = 255 ∧ t2 128 = 128 ∧ t2 1151 = 1 ∧ t2 1152 = 0 := by
  decide

/-- The sole exact half tie used by the negative control. -/
def halfUpTie : Nat := (255 + 1) / 2
def truncTie : Nat := 255 / 2

@[simp] theorem half_up_tie_value : halfUpTie = 128 := by decide
@[simp] theorem trunc_tie_value : truncTie = 127 := by decide

/-- Digest of the canonical comma-separated 2,176-byte integer table. -/
def t2Digest : String :=
  "7a42a785940f557d5d5a4188fa5683ea00cf8b29e8d5213c75c581a5a0d92fbb"

end Attention
end Lockstep
