# FT 5% grouped campaign

## Existing evidence

- `Scratch5-grouped` already exists for the 600 K grouped dataset, with 89
  selected train configurations and three seeds. It is a scratch control, not
  an FT result, and it used the older global-random selection manifest.
- No accepted `FT-600K-5%` or `FT-MultiT-5%` family was found in the existing
  registry or Rockfish output roots.

## New campaign

- Build two frozen 5% releases from the full grouped datasets.
- Preserve all isolated E0 anchors and leave valid/test unchanged.
- Use trajectory-stratified random selection with selection seed 20261016.
- Train OMAT fine-tuning replicas with seeds 123/234/345 for both 600 K and
  Multi-T, changing only the seed within each family.
- Register the six models under new canonical IDs; never overwrite the old
  full-data or `Scratch5` IDs.

Submitted as Rockfish job `32069341` (`32069341_[0-5%6]`). The first status
check found all six tasks pending and no model output yet.

## After training

1. Verify generated dataset hashes, counts, source-group coverage, and no split
   leakage.
2. Verify six logs, compiled models, finite parameters, Cr/Sb/Te support, model
   hashes, and test RMSE.
3. Compare each 5% family against its matching full-data grouped family on the
   same 639 candidates and the unchanged grouped test split.
4. Report energy and force RMSE separately, with three-seed dispersion; do not
   claim equivalence from one seed or from cross-family absolute energies.
5. Update model registry, claim-evidence matrix, paper text, SI, and figures
   only after the acceptance gates pass.
