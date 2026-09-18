# Draft Response A2: Training/Evaluation Overlap

We thank the reviewer for raising the possibility that the reported FT-600K
performance could be affected by overlap between the fine-tuning data and the
NEB evaluation trajectories. We therefore audited all 1,110 configurations in
the current 1-6 and 1-7 QE histories against the declared train, validation,
and test files used by the evaluated fine-tuned checkpoint.

The audit used complementary whole-structure and Cr-centered local tests. We
found no exact global or local fingerprint matches. For every evaluation frame
and reference split, we also computed the nearest SOAP cosine distance; none
was at or below the pre-registered `1e-4` near-duplicate threshold. The minimum
distances were `2.79e-4`, `3.71e-4`, and `5.09e-4` for train, validation, and
test. Where Cr-local environments had compatible species cardinality, no
species-matched RMSD was at or below `0.05 A`; the corresponding minima were
`1.194 A`, `0.626 A`, and `0.591 A`. Environments without compatible local
species cardinality were recorded as non-comparable for RMSD rather than
silently discarded, while all configurations remained covered by SOAP.

We have added the full nearest-neighbor distance distributions, thresholds,
and comparable-sample counts to the Supporting Information. We have also made
the benchmark split group-aware at the system, endpoint-basin, and mechanism
levels so that future configurations from the same path lineage cannot be
assigned across training and evaluation partitions.

This conclusion is intentionally scoped to the declared fine-tuning files and
the evaluated checkpoint. The available artifacts do not expose the complete
pretraining corpus of the foundation model, so we do not make an absolute claim
about that unavailable corpus. Any change to the fine-tuning files, checkpoint,
benchmark frames, or mechanism grouping invalidates this pass and automatically
requires the audit to be repeated.
