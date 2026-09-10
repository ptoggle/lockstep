import Lockstep

open Lockstep

private def usage : String :=
  "usage: lake exe lockstep-formal (check | export <path>)"

def main (args : List String) : IO UInt32 := do
  match args with
  | ["check"] =>
      IO.println Artifacts.checkReport
      pure 0
  | ["export", path] =>
      IO.FS.writeFile path Artifacts.json
      IO.println s!"LOCKSTEP-FORMAL wrote {path}"
      pure 0
  | _ =>
      IO.eprintln usage
      pure 2
