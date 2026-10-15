# 639-candidate, 12-MLIP static-screen: results and interpretation

## Scope and evidence boundary

The canonical 639 structures were evaluated with the same static MACE single-point protocol used for the three from-scratch checkpoints: identical `extxyz` inputs, no relaxation, no MD continuation, CUDA execution in the Rockfish MACE environment, and one energy/force evaluation per checkpoint. The nine additional checkpoints were evaluated in 32 chunks of at most 20 structures (array concurrency 2). The production array completed with 32/32 Slurm tasks successful; the remote collector then produced a verified 5,751-row matrix (639 structures × 9 additional checkpoints). After reconciliation with the existing 1,917 from-scratch rows, the paired matrix contains 7,668 rows (639 × 12), with zero failed rows and zero duplicate candidate–model pairs.

This is static MLIP evidence about model response and inter-model disagreement. It is not a DFT validation, a causal proof that any model force is wrong, or a new MD stability/transport result.

## Model coverage

The 12-checkpoint comparison contains:

- three existing from-scratch checkpoints: `ablation_lmax_2` seeds 123 and 234, and `baseline` seed 345;
- FT-600K seeds 123, 234 and 345;
- FT-MultiT seeds 123, 234 and 345;
- three foundation representatives: legacy compiled OMAT, MPA-0 medium and MP-0b3 medium.

The foundation checkpoints are model representatives rather than seed replicas. The OMAT checkpoint is explicitly labelled **legacy compiled OMAT** and is not silently relabelled as public MATPES-PBE OMAT-FT.

## Candidate provenance and representativeness

The 639 structures are not sampled from one trajectory. They are one structure per audited source snapshot and span nine trajectory/seed branches from three independently named source conditions:

| source condition | candidates |
|---|---:|
| original | 216 |
| reproduce-1 | 216 |
| reproduce-2 | 207 |

The nine run-key counts are 72, 72, 72, 72, 72, 72, 72, 64 and 71. The stage labels are intentionally enriched for failure attribution rather than drawn in proportion to MD time: 207 early-stable controls, 216 stable pre-onset frames, 208 near-onset precursor frames, 7 threshold/terminal frames and 1 1000-ps stable terminal target. Thus the set is representative of the **audited failure-attribution sampling frame** across multiple trajectories and seeds, but it is not representative of the equilibrium/time-weighted trajectory distribution and should not be used as an unbiased estimate of long-time state populations.

## Static force response across stages

All 12 checkpoints return finite forces for all 639 structures. The early-stable and stable pre-onset groups remain in a narrow static-force regime, with median maximum force approximately 2.4–3.2 eV Å−1 across model families. The precursor group exposes the first clear model-family divergence: median maximum forces are approximately 3.2 eV Å−1 for the baseline, 8.5 eV Å−1 across the two `ablation_lmax_2` checkpoints, 16.2–16.4 eV Å−1 for the two fine-tuned families, 17.0 eV Å−1 for legacy OMAT, and 29.9–30.8 eV Å−1 for the two foundation medium models. The threshold group amplifies this separation, with median maximum forces spanning approximately 133 eV Å−1 for scratch `seed123` to 3,295 eV Å−1 for FT-MultiT `seed234`; the corresponding 12-model median spread is 3,718 eV Å−1.

Across all 639 structures, the median 12-model maximum-force range is 8.92 eV Å−1 and the 95th percentile is 204.41 eV Å−1. By stage, the median range is 3.07 eV Å−1 for early-stable controls, 7.06 eV Å−1 for stable pre-onset frames, 27.29 eV Å−1 for precursors and 3,717.51 eV Å−1 for threshold/terminal frames. This pattern is consistent with a progressive loss of cross-model agreement as the sampled structures approach or cross the model-specific instability boundary, while remaining agnostic about which force is physically correct.

The 100 eV Å−1 diagnostic gate is crossed most often by MP-0b3 medium (51/639 structures), MPA-0 medium (49/639), FT-600K seed 234 (46/639), FT-600K seed 345 (43/639) and FT-MultiT seed 123 (41/639). These counts are screening diagnostics, not calibrated error probabilities.

## Distribution shift relative to the reference sets

The reference embedding contains 521 structures: 250 training, 124 test, 122 validation and 25 NEB structures. Under the existing SOAP descriptor and PCA diagnostic (Cr/Sb/Te, 5 Å cutoff, σ=0.5 Å, nmax=4, lmax=4, 20 PCA components), every candidate group has an outside-training fraction of 1.0 at both the training q95 and q99 thresholds. Candidate median descriptor distance to the training distribution is 11.0–12.7, whereas the candidate median distance to the NEB reference is 615.1–616.0 under the same stored diagnostic. Candidates occupy PCA1≈1.17–2.92 and PCA2≈−4.77 to −2.91; the training/test/validation references are centred near PCA2≈0, while NEB structures are centred near PCA2≈214.

The combined candidate-versus-reference two-dimensional histogram has Jensen–Shannon divergence 0.905 under the fixed 20×20 PCA binning used by the audit. This should be read as a descriptor-space distribution-shift diagnostic, not as a calibrated uncertainty or a statement that the structures are chemically invalid. The result supports a precise interpretation: these candidates are deliberately selected OOD/failure-attribution probes relative to the existing training, validation and NEB reference distributions, rather than representative samples of the original DFT training manifold.

## Nature-style manuscript text

> To disentangle model-specific response from trajectory-level stochasticity, we evaluated the same 639 failure-attribution structures with twelve MACE checkpoints spanning three from-scratch models, two fine-tuned families at three training seeds each, and three foundation-model representatives. The structures were drawn from nine trajectory/seed branches and were intentionally stratified across early-stable, pre-onset, precursor and threshold regimes; they therefore represent the audited failure-attribution frame rather than a time-weighted molecular-dynamics ensemble. All 7,668 paired evaluations were finite and complete. The models agreed closely on early-stable and pre-onset structures, whereas force disagreement increased markedly in the precursor regime and became extreme at the threshold. In descriptor space, all candidates lay beyond the training q99 envelope and occupied a region distinct from both the training/validation reference set and the NEB structures. These results identify a reproducible distribution-shift and model-divergence signature around the apparent instability, but do not by themselves establish which model force is physically correct; that question requires targeted reference calculations on reduced, well-defined periodic fragments.

## Reproducibility chain

- Canonical candidates: `candidate_manifest_639.csv` (SHA256 `65937709eed9562d1c16cadbf91e2d40c0769e777eef55bb912d6de71d0d486c`).
- New-model registry and file hashes: `MODEL_REGISTRY.tsv`, `MODEL_HASH_AUDIT.tsv`.
- Remote 9-model audit: `remote_analysis/analysis_manifest.json`, status `VERIFIED`, 639×9 rows, 0 failures, 0 duplicates.
- Combined 12-model audit: `analysis/combined_analysis_manifest.json`, status `VERIFIED`, 639×12 rows, 7,668 finite successful rows.
- Plot Atlas composite: `figures/fig_multimodel_639_overview.pdf` and `.png`; the figure manifest records the plot as a new reconstruction from verified analysis data.
- Rockfish campaign root: `/scratch16/pclancy3/yi/mace_multimodel_639_screen_20261014`.
