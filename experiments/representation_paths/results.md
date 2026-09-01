# Native / Direct AST / NED-Tree Representation Diagnostic

Ten paired fixed-value constraints are evaluated through three representation paths. Every cell uses one `gpt-5.1` completion, permits at most one generated-program execution, and uses no repair.

| Representation Path | Constraint Preserved | Program Run | Type II Errors | Treatment Adherence |
|---|---:|---:|---:|---:|
| Native nonlinear expression | 10/10 | 10/10 | 0/10 | 10/10 |
| Direct AST expression | 10/10 | 9/10 | 1/10 | 10/10 |
| NED-Tree | 10/10 | 10/10 | 0/10 | 10/10 |

All four columns use the same ten cases. Constraint preservation is intentionally separated from Program Run: case c02 Direct AST identifies the correct fractional-power constraint but lowers it through an unsupported direct `NLExpr >= constant` call.

The paired outcome is one NED-Tree Program-Run/no-Type-II win over Direct AST (c02), with nine ties; NED-Tree and Native tie on all ten cases. Thus, this diagnostic supports the narrower claim that NED-Tree removes one direct-AST lowering ambiguity while retaining the Native path's robustness. It does not support universal superiority over Native nonlinear expressions.

This is a ten-case, fixed-value, one-shot mechanism diagnostic with no repair and provider-default temperature. Prompt lengths are not matched. The result is descriptive evidence and is not a replacement for the paper's main benchmark.

The observed median API latency is 7.10, 6.24, and 8.16 seconds for Native, Direct AST, and NED-Tree, respectively; median total tokens are 1,684.5, 1,670.0, and 1,926.5. These one-shot values are descriptive overhead statistics, not evidence of latency superiority. Direct AST's failed c02 program is not interpreted as faster solver execution.
