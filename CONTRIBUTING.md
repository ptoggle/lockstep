# Contributing

**Sign-off.** Every commit carries a Developer Certificate of Origin sign-off (`git commit -s`); no contributor licence agreement is required. Contributions are accepted under the licence of the directory they touch (see `NOTICE`).

**The gate rule.** A change to a reference, a contract document or a conformance kit must keep every existing vector and selftest passing bit for bit, or must change the contract version, the changelog and the vectors together in one pull request with the reason. A performance change to an implementation (not in this repository) is measured paired against stock on the same node and gated bit-identical to the reference; those results are cited by commit hash here.

**Contract changes** go through an RFC: an issue that states the rule, its reference implementation, its conformance vectors and the gate that will enforce it. Precedence clause: on a conflict between prose and the reference, the reference is normative and the prose is corrected.

**Conformance digests from a new device** are welcome as pull requests to `conformance/` (see `CONFORMANCE.md`): the kit, the tag, the machine description and the digests file. A digest that differs is not a rejected contribution; it is the most valuable kind of report.

Run `make verify` before opening a pull request.
