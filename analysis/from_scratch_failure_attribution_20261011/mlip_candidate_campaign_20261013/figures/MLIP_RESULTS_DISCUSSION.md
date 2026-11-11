# Three-model MLIP candidate evaluation: audited Results and Discussion

Campaign status: `VERIFIED`. This text is generated from the compact MLIP result tables and must not be upgraded beyond the analysis manifest.

## Scope and protocol

The 639 saved 2050-atom fully periodic configurations were evaluated as static single points with the three registered checkpoint files: ablation_lmax_2 seed 123, ablation_lmax_2 seed 234, and baseline seed 345. The same candidate structure is therefore paired across models. No geometry relaxation or MD continuation was performed. Energies are retained for provenance but are not used for absolute cross-family ranking; the primary comparison is the force field, force tails, geometry, and paired cross-model disagreement.

## Coverage and numerical status

The compact result table contains 1917 model-candidate rows for 639 unique candidates and 3 model IDs. 1917 rows are finite successful evaluations; 0 rows remain in the failure ledger. These are numerical evaluation statuses, not evidence that any model is physically correct.

## Stage comparison

The following table reports the model-wise median static maximum force for each source stage/group. `stable_early` and `stable_pre` are source-trajectory labels; the MLIP evaluation itself does not redefine the onset gate.

| candidate group | window | model | n | success | median max |F| (eV Å^-1) | q95 max |F| (eV Å^-1) |
|---|---|---|---:|---:|---:|---:|
| precursor_near_onset | precursor | lmax2 / seed 123 | 208 | 208 | 3.74 | 116 |
| precursor_near_onset | precursor | lmax2 / seed 234 | 208 | 208 | 12.1 | 60.9 |
| precursor_near_onset | precursor | baseline / seed 345 | 208 | 208 | 3.16 | 78.2 |
| stable_early_control | stable_early | lmax2 / seed 123 | 207 | 207 | 2.58 | 3.14 |
| stable_early_control | stable_early | lmax2 / seed 234 | 207 | 207 | 5.31 | 19 |
| stable_early_control | stable_early | baseline / seed 345 | 207 | 207 | 2.56 | 3 |
| stable_pre_onset | stable_pre | lmax2 / seed 123 | 216 | 216 | 2.59 | 3.41 |
| stable_pre_onset | stable_pre | lmax2 / seed 234 | 216 | 216 | 5.61 | 20 |
| stable_pre_onset | stable_pre | baseline / seed 345 | 216 | 216 | 2.58 | 3.19 |
| terminal_stable_target | threshold | lmax2 / seed 123 | 1 | 1 | 2.56 | 2.56 |
| terminal_stable_target | threshold | lmax2 / seed 234 | 1 | 1 | 12.9 | 12.9 |
| terminal_stable_target | threshold | baseline / seed 345 | 1 | 1 | 2.66 | 2.66 |
| threshold_or_terminal | threshold | lmax2 / seed 123 | 7 | 7 | 133 | 1.54e+03 |
| threshold_or_terminal | threshold | lmax2 / seed 234 | 7 | 7 | 172 | 3.86e+03 |
| threshold_or_terminal | threshold | baseline / seed 345 | 7 | 7 | 359 | 2.05e+03 |

## Mechanistic interpretation and boundary

The paired same-configuration force spread is the direct MLIP diagnostic for this campaign. If the spread and model-specific force ranking increase from stable-pre to precursor/threshold, the result supports a model-dependent amplification of forces in the extrapolative region. If the three models remain close while all show the same geometric deterioration, the evidence instead supports a shared geometry-driven trigger. Neither pattern establishes which force is DFT-correct: the models share training lineage and the 2050-atom DFT comparison remains unavailable.

Descriptor novelty remains descriptive distributional evidence. A high force spread is not a calibrated uncertainty and is not a causal proof that the lmax reduction caused the runaway. Threshold-stopped MD branches remain `FAILURE_MODE`/`unstable — no Dxy`; this static campaign does not create a transport result.

## Reproducible claim status

- `VERIFIED` for: candidate/model row coverage, checkpoint hashes, finite MLIP energy/force outputs, paired force-spread calculations, and stage-wise summaries when the analysis manifest passes.
- `FAILURE_MODE` for: any model-candidate runtime or nonfinite rows, retained in `failure_ledger.csv`.
- `PROVENANCE_PARTIAL` for: physical correctness relative to DFT, DFT parity of the local reconstruction, and any causal statement about why the MD runaway occurs.

The next DFT step should select configurations where (i) the same candidate has a large paired force spread, (ii) the spread rises before the source onset, and (iii) the local periodic reconstruction passes the finite/minimum-distance/PBC gate. Those inputs remain DFT-ready candidates, not DFT results, until QE convergence and provenance gates pass.
