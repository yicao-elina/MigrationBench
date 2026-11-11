# Repair r01 gap report

| item | status | consequence |
|---|---|---|
| Corrected Conda bootstrap | verified | repair r01 entered MACE training on GPU nodes |
| Seed234 training completion | verified at scheduler/training-log level | compiled model and test metrics exist |
| Seed345 training completion | verified at scheduler/training-log level | compiled model and test metrics exist |
| Model hashes and data hashes | verified | exact downstream model/input identities recorded |
| GPU load and finite probe | verified | gate job `32455612` passed on `gpu03`; both models finite on train/valid/test probes |
| 20 ps MD smoke | verified failure | both tasks ran on `gpu18` and failed near 201 ps on the scientific stability threshold |
| 1 ns continuation | not submitted | correctly blocked because neither smoke gate passed |

The training failure in the original `32121744` attempt remains excluded. The
repair r01 models are verified as loadable finite models, but their MD smoke
results are retained as `FAILURE_MODE`; no 1 ns transport claim is supported.
