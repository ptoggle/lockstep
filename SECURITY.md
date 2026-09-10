# Security policy

A differing bit is a bug. If you find an input on which the served engine and the reference disagree, a case on which two references disagree, a conformance digest that does not reproduce, or a way to make a manifest or fingerprint check pass on the wrong artifact, report it to security@codegree.de. Include the contract version, the command, the inputs (or a seed) and both outputs. We acknowledge within three working days and publish the case with the fix as a negative control in the gate suite.

Do not open a public issue for a finding that lets a provider pass verification while serving a different function; use the address above.
