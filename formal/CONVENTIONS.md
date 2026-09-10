# Formal source conventions

- Every public theorem corresponds to a named obligation in `docs/LEAN_FORMALIZATION_PLAN.md`.
- `sorry`, `admit`, and unchecked axioms are prohibited.
- Contract semantics use raw bits or exact integers. Host floating-point execution is not a proof rule.
- Definitions distinguish contract semantics, scheduled logical implementations, and hardware assumptions.
- Concrete artifacts are deterministic and are checked by both Lean and the Python runtime bridge.
- A theorem name ending in `_counterexample` carries an explicit witness, not only a test assertion.
