# 639-candidate model grouping used by the family-grouped figure

The updated 639-candidate figure uses a nested visual encoding:

- **Colour:** the four manuscript-level groups: From Scratch, Foundation / zero-shot, FT-600K, and FT-MultiT.
- **Arm 1/2/3:** the three within-group entries, shown by the hatch/label encoding in panel b and by the ordered columns in panel a.

The twelve plotted checkpoints are not twelve exchangeable random-seed replicas:

| Major group | Arm 1 | Arm 2 | Arm 3 | Interpretation |
|---|---|---|---|---|
| From Scratch | Ablation Lmax2 / seed 123 | Ablation Lmax2 / seed 234 | Baseline / seed 345 | Two same-Lmax branches plus a distinct baseline branch |
| Foundation / zero-shot | OMAT legacy | MPA-0 medium | MP-0b3 medium | Three checkpoint variants; a common training seed is not applicable |
| FT-600K | seed 123 | seed 234 | seed 345 | Same-family training-seed replicas |
| FT-MultiT | seed 123 | seed 234 | seed 345 | Same-family training-seed replicas |

All 639-candidate evaluations use the same static input structures. The MD
velocity seed is not a variable in this static screen. The machine-readable
mapping, including model SHA256 values and lineage status, is
`MODEL_GROUPING.tsv`.

This distinction is required for interpretation: the FT groups support a
within-family training-seed comparison; the Foundation group measures
checkpoint heterogeneity; and the From Scratch group compares related but not
identical failure-mode branches. The grouping does not establish DFT
correctness or causal responsibility for the MD instability.
