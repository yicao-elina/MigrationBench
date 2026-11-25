# MigrationBench integration map

This package is an endpoint/site-distribution and model-diagnostic evidence
module. Its role in the MigrationBench paper is to connect the DFT discovery
set to model robustness without turning an incomplete DFT endpoint ledger into
thermodynamic or transport evidence.

## One-sentence paper message

Across 47 Cr-containing source geometries, the discovery set spans three
geometric regions—within the van der Waals gap, at the gap/QL interface, and
within a QL—and the multi-model MACE response is most heterogeneous for
continuous vertical and local-coordination descriptors; this identifies where
model disagreement is expected, while a calibrated MACE–DFT error remains
pending matched final DFT geometries.

## Manuscript placement and dependency chain

| Result | Recommended location | What it contributes | Upstream evidence | Downstream handoff |
|---|---|---|---|---|
| 47-site lattice/region distribution | Main Results, discovery-set definition; FIG1 and Table 1 | Defines the support of the benchmark and makes the data distribution explicit | `processed/DFT_ENDPOINT_47_CLASSIFICATION.tsv`, source input hashes | Fixes the denominator for every later model comparison; cite before grouped-split or endpoint results | 
| 21 converged retry endpoints and 0–4.732 eV relative-energy span | Main Results or SI DFT audit; FIG2 and Table 1 | Gives a restricted same-campaign energetic reference and exposes three excluded time-limited endpoints | `DISCOVERY_PROGRESS_AUDIT_20261018.md` lineage, retry job records, BFGS final evidence | Supplies the DFT reference context; must not be phrased as a global stability ranking | 
| Four-model, three-start relaxation sensitivity | Main Results model-diagnostics paragraph or SI; FIG3 and Table 2 | Separates model-internal basin sensitivity from DFT error and shows Scratch is the least repeatable legacy cohort | `processed/MACE_FOUR_MODEL_RECORDS.tsv`, final relaxation summaries | Motivates seed/model ensembles and uncertainty-aware interpretation of MD/NEB outputs | 
| Twelve-model static source-geometry ensemble | Main Results uncertainty/coverage paragraph and SI methods; FIG4 and Table 3 | Tests foundation-family comparators plus FT/Scratch seeds on the same immutable 47 inputs | Slurm array 32373911, hash-verified registry, 564/564 predictions | Provides a structural diagnostic for model disagreement; does not validate absolute energy or force accuracy against DFT | 
| DFT–MACE error eligibility ledger | Methods/limitations and SI audit table | Prevents accidental subtraction of unmatched QE and MACE total energies | `processed/DFT_MACE_ERROR_ELIGIBILITY.tsv` | Opens a future extension once final DFT XYZ and same-geometry evaluation are archived | 

## How this connects to the rest of MigrationBench

1. **Dataset definition → this package.** The grouped-split and provenance
   narrative needs a transparent description of what the 47 discovery points
   actually cover. FIG1 is the bridge from “47 endpoints” to a geometric
   sampling statement.
2. **Endpoint/NEB evidence → this package.** Endpoint roles are multi-label
   pathway annotations, not mutually exclusive thermodynamic classes. The
   restricted DFT retry panel can accompany the endpoint/NEB section, while
   fixed-path and self-relaxed NEB claims remain separate.
3. **Model validation → this package.** FIG3 establishes local-basin
   sensitivity for the four legacy campaign labels. FIG4 expands the view to
   three foundation-family checkpoints and three seeds each for FT-600K,
   FT-MultiT, and Scratch. This is the natural handoff to uncertainty-aware
   model selection and to the existing grouped-split evaluation.
4. **MD/transport → this package.** The package can justify reporting model
   disagreement as a risk indicator for downstream trajectories. It cannot
   certify diffusion, conductivity, barriers, or 1-ns transport; those claims
   remain governed by their own trajectory and convergence gates.

## Required wording discipline

- Use “within-gap”, “gap/QL interfacial”, and “within-QL” for geometric zones.
- Use “lower/higher relative energy within the 21-endpoint retry subset”;
  avoid “stable”, “metastable”, and “unstable” unless a separate Hessian,
  phonon, or global-landscape analysis is added.
- Call the legacy model `OMAT-derived fine-tuned legacy Foundation`, not a
  zero-shot public Foundation model.
- Call FIG4 “source-geometry ensemble disagreement” or “model spread”, not
  calibrated MACE uncertainty or MACE–DFT error.

## Suggested figure/table labels for Overleaf

These are handoff labels, not a request to rename existing files:

- `fig:discovery-site-distribution` → `FIG1_ENDPOINT47_LATTICE_SITE_DISTRIBUTION.pdf`
- `fig:dft-retry-energy-structure` → `FIG2_RETRY21_DFT_ENERGY_STRUCTURE.pdf`
- `fig:legacy-model-basin-sensitivity` → `FIG3_FOUR_MODEL_BASIN_SENSITIVITY.pdf`
- `fig:source-ensemble-disagreement` → `FIG4_MACE12_SOURCE_GEOMETRY_DISAGREEMENT.pdf`
- `tab:dft47-geometry-energy-ensemble` → `TABLE1_DFT47_GEOMETRY_ENERGY_AND_ENSEMBLE_SUMMARY.tsv`
- `tab:legacy-four-model-sensitivity` → `TABLE2_LEGACY_FOUR_MODEL_SENSITIVITY.tsv`
- `tab:mace12-structural-associations` → `TABLE3_MACE12_STRUCTURAL_ASSOCIATIONS.tsv`

