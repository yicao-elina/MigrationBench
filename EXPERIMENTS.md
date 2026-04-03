# EXPERIMENTS.md — ICML 2026 Rebuttal Experiment Ledger

**Last updated:** 2026-04-02  
**Paper:** "DUAL-X: Interpretable Failure Diagnosis for ML Force Fields"  
**System:** Cr-doped Sb₂Te₃ (2-layer); 5 NEB paths; MACE models × 4

---

## Context

MACE FT-600K fails on Deep Penetration (OOD) NEB paths (1-2, 1-3): ~12 eV barrier error.
Performs well on In-gap (ID) paths (1-4, 1-5, 1-7): ~0.1 eV error.

**Central hypothesis:** This gap is mechanistically explained by SOAP-captured motif deviation in Cr local environment (and/or Sb backbone geometry).

**Reviewer challenge:** "correlation ≠ mechanism" and "N=5 NEB paths is too small."

---

## Motif Score Definitions

| Score | Species Family | Weighting | Hypothesis |
|-------|---------------|-----------|------------|
| **Score A** | Cr-Cr + Cr-Sb SOAP block | SHAP-weighted | Cr local environment drives error |
| **Score B** | Sb-Sb + Sb-Te SOAP block | L2 norm (unweighted) | Global backbone distortion drives error |

Note: `shap_top_features_NEB.csv` shows top features are Sb-Sb/Sb-Te globally — Score B is data-driven. Both computed in parallel.

---

## EXP-NEB1 — Motif co-variation along NEB path

**Status:** TODO  
**Script:** `xforce/src/xforce/neb_alignment/neb1_path_analysis.py`  
**Run:** `python -m xforce.neb_alignment.neb1_path_analysis` (or see runner notebook)

**Method:**  
Per NEB path: compute motif scores A+B per frame. Spearman ρ(motif, |MLFF error|).

**Expected outputs:**
- `results/neb_alignment/neb1_path_{1-2,1-3,1-4,1-5,1-7}.pdf`
- `results/neb_alignment/neb1_stats.csv`

**Expected result (hypothesis):**
- OOD paths (1-2, 1-3): |ρ| > 0.5 for at least one score
- ID paths (1-4, 1-5, 1-7): weaker |ρ|, demonstrating regime-specificity

**Actual result:** _TBD_

**Caveat:** Only path 1-4 has probe_features.csv; paths 1-2, 1-3, 1-5, 1-7 need `mace_probe_neb.py` re-run.

**Rebuttal template:**
> "Along the OOD Deep Penetration path, the SOAP-based motif score A co-varies with the per-frame MLFF error (Spearman ρ=X.XX, p=X.XX), while the ID In-gap paths show markedly weaker correlation (ρ=X.XX). This demonstrates that the geometric regime shift—not merely training distribution mismatch—is captured by our motif score."

---

## EXP-NEB2 — Cross-path Spearman(M_path, ΔEm)

**Status:** TODO  
**Script:** `xforce/src/xforce/neb_alignment/neb2_cross_path.py`

**Method:**  
M_path = max(motif_score) per path. Spearman ρ(M_path, ΔEm) across 5 paths.

**Expected outputs:**
- `results/neb_alignment/neb2_scatter_A_{model}.pdf`
- `results/neb_alignment/neb2_scatter_B_{model}.pdf`
- `results/neb_alignment/neb2_table.csv`

**Expected result:** ρ > 0.7; OOD paths cluster at high motif + high error corner.

**Actual result:** _TBD_

**N=5 caveat:** Explicitly reported; note that Spearman ρ > 0.7 with N=5 has p < 0.2 (one-tailed) which is indicative but not conclusive. Combined with EXP-NEB1 mechanistic evidence, this supports the claim.

**Rebuttal template:**
> "Across all 5 NEB paths, max motif score (M_path) correlates with barrier prediction error ΔEm (Spearman ρ=X.XX). We acknowledge N=5 limits statistical power; the cross-path result is presented as corroborating mechanistic evidence alongside the per-frame analysis in EXP-NEB1."

---

## EXP-AIMD1 — Motif vs |F_DFT| in AIMD trajectory

**Status:** TODO  
**Script:** `xforce/src/xforce/neb_alignment/aimd1_force_alignment.py`

**Method:**  
Per AIMD frame: motif scores A+B. Scatter vs per-frame mean |F_DFT| on Cr + nearest neighbors. Spearman ρ per temperature bin.

**Data:**
- Trajectory: `2-layer/aimd_merged_traj.xyz`
- Probe features: `archive/cleanup_2026_01_19/representation/analysis_Cr_Doped_600K/Cr_Doped_Sb2Te3_600K_probe_features.csv`

**Expected outputs:**
- `results/neb_alignment/aimd1_scatter.pdf`
- `results/neb_alignment/aimd1_stats.csv`

**Expected result:** ρ > 0.4 between motif score and force magnitude (structural stress ↔ motif deviation).

**Actual result:** _TBD_

**Rebuttal template:**
> "In the 600 K AIMD training trajectory, motif score B (Sb backbone) shows Spearman ρ=X.XX with DFT force magnitude on Cr neighbors (N=XXXX frames). This confirms that frames where the model struggles (high force magnitude = large atomic displacement from equilibrium) are precisely those where the Sb backbone geometry deviates—exactly the feature family identified by DUAL-X."

---

## EXP-AIMD2 — Geometric distortion index: ID vs OOD NEB frames

**Status:** TODO  
**Script:** `xforce/src/xforce/neb_alignment/aimd2_distortion.py`

**Method:**  
Per NEB frame: distortion index δ = mean |Cr–Te bond length − equilibrium|.
Box plots: ID (1-4, 1-5, 1-7) vs OOD (1-2, 1-3) NEB frames.
Mann-Whitney U test.

**Expected outputs:**
- `results/neb_alignment/aimd2_boxplot.pdf`
- `results/neb_alignment/aimd2_scatter.pdf`
- `results/neb_alignment/aimd2_stats.csv`

**Expected result:** OOD frames significantly higher δ (p < 0.05, Mann-Whitney U).

**Actual result:** _TBD_

**Rebuttal template:**
> "OOD Deep Penetration frames show significantly larger Cr–Te bond distortion (δ_OOD = X.XX ± X.XX Å vs δ_ID = X.XX ± X.XX Å, Mann-Whitney U p=X.XXX). This geometric evidence confirms that the training distribution—confined to moderate-temperature in-gap diffusion—provides no coverage of the heavily distorted geometries encountered during deep-layer penetration."

---

## EXP-CD — Cross-dataset generalization (placeholder)

**Status:** PLACEHOLDER (if time permits)  
**Concept:** Test whether a model trained on 600 K AIMD (Sb₂Te₃ only) can transfer to a related chalcogenide (e.g., GeTe) with similar layered structure. DUAL-X motif score used to predict where transfer will fail.

---

## Data Summary

| Path | Category | DFT Em (eV) | Model | ΔEm (eV) | probe_features.csv |
|------|----------|-------------|-------|-----------|-------------------|
| 1-2  | OOD      | ~0.3        | FT-600K | ~12 | MISSING — need re-run |
| 1-3  | OOD      | ~0.3        | FT-600K | ~12 | MISSING — need re-run |
| 1-4  | ID       | ~0.34       | FT-600K | ~0.16 | ✓ EXISTS |
| 1-5  | ID       | —           | —     | —   | MISSING — need re-run |
| 1-7  | ID       | —           | —     | —   | MISSING — need re-run |

**Action required:** Re-run `mace_probe_neb.py` (or `mace_neb_1-calc.py`) for paths 1-2, 1-3, 1-5, 1-7 to generate probe_features.csv before EXP-NEB1.

Script location: `neb_benchmark/scripts/mace_probe_neb.py`  
SLURM template: `neb_benchmark/slurm/submit_probe_neb.slurm`

---

## Running All Experiments

```bash
cd /data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/X-FORCE

# Phase 1: NEB experiments
python -c "
from xforce.shap.feature_extractor import SOAPFeatureInterpreter
from xforce.neb_alignment import neb1_path_analysis, neb2_cross_path

interp = SOAPFeatureInterpreter(
    species=['Cr', 'Sb', 'Te'], n_max=4, l_max=4, r_cut=5.0
)
shap_csv = 'xforce/results/2_Core_DUAL-X_Analysis/2.1_ID_vs_OOD_Analysis/dual_x_comprehensive_icml_publication/tables/shap_top_features_NEB.csv'
data_dir = '../../neb_benchmark/data'
out_dir  = 'xforce/results/neb_alignment'

neb1_path_analysis.run_neb1(data_dir, shap_csv, interp, out_dir)
neb2_cross_path.run_neb2(data_dir, shap_csv, interp, out_dir)
"

# Phase 2: AIMD experiments
# ... (see aimd1_force_alignment.py and aimd2_distortion.py)
```
