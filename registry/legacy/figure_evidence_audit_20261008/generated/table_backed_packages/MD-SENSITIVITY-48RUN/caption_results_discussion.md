# MD-SENSITIVITY-48RUN

## Article target
Results/SI: integrator/thermostat sensitivity and dynamic-stability boundary

## Caption
A–B, stability classification for 48 complete short-handoff runs across two input snapshots, four model families, and six integrator/thermostat settings; C, runaway fraction by snapshot; D, first divergence timing for Foundation-OMAT and from-scratch branches.

## Results
Foundation-OMAT and from-scratch are classified as runaway/nonfinite for all six settings at both snapshots. The smaller 0.5 fs timestep delays some first crossings but does not restore a finite bounded handoff. Naive-FT remains stable for 461800 and bounded/structurally shifted for 462000; Multi-T is stable for 461800 and bounded/structurally shifted for 462000.

## Discussion
The table-backed ablation supports a model-dependent PES-instability interpretation rather than a current Langevin-bath-only explanation. It is a short-range counterfactual stability result, not a diffusivity, DFT-validation, or long-trajectory chemical-mechanism result.

## Scientific boundary
- All 48 metrics files have 202 rows and the campaign README records input/model/cell hash checks.
- Do not infer thermostat/timestep irrelevance beyond the tested settings.
- Keep this package separate from the 1 ns transport figures.
