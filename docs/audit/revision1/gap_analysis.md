# Gap Analysis — Reviewer 1 (Round 1) — 2026-09-06

> **Post-LP new findings (beyond reviewer's list), all documented in audit_reports/:**
> - **SHAP**: SOAP label misalignment — 386/390 published feature labels wrong (l-innermost vs dscribe's
>   l-outermost); frame/feature misalignment in dataset builder; "R²>0.7" was in-sample; "500 structures"
>   unsupported (actual 150/30). Fixed + regenerated locally: CV R² 0.982/0.876/0.973; corrected FT-MultiT
>   top feature = Cr-Sb_n43_l0. Published consensus/non-consensus feature claims fail a CV check.
> - **MD**: published D for Foundation/FT-MultiT came from 41/140/162-frame truncated trajectories (8-32 ps);
>   full-length data show no diffusive regime for Foundation; all κ statistically consistent with zero.
> - **NEB/Fig 3**: Fig 3 in tex includes `figure4-v2.pdf` (structure snapshots) — caption/figure mismatch,
>   real composite is `Em_SI.pdf`; two conflicting local versions of the ΔEm bar plot; Fig S3b "0.41 eV"
>   not derivable from the plotted curve (≤0 everywhere); Scratch self-NEB trajectory locally proven diverged.
> - **Latent**: separability in original descriptor space is energy-scale-driven; z-scored silhouette ≈ -0.02
>   (overlap after scale control) — §3.4 wording must weaken (hedged replacement drafted).

Three parallel audits: (A) MigrationBench repo code verification, (B) manuscript/figure inventory,
(C) local data asset inventory. Verdicts below; full evidence inline in each todo item.

## Reviewer comment → gap → resolution route

| # | Reviewer point | Audit verdict | Route |
|---|---|---|---|
| C1 | Fig 3d (fixed-geom, Foundation +0.7 eV) vs Fig S3 (self-NEB, 0.41 eV) discrepancy; ref 0.34 vs ~0.3 eV | DFT ref confirmed locally: path **1-4 = 0.336 eV** (`examples/.../221_vdw_corr_DFT_D3/1-4/neb.out`). No standalone 0.3 eV in tex — tension is Fig3d-vs-S3 evaluation protocol. MACE self-NEB source: `neb_optm.xyz` (71 MB) + `visualization/neb_optm_1-scratch.xyz` exist in parent folder; full set on cluster | **Local**: DFT reference audit table + MEP profiles; **Cluster**: unified re-eval with one checkpoint, one reference |
| C2 | Train/test overlap FT-600K (SOAP distance or NEB-excluded retrain) | No SOAP descriptors, no overlap analysis locally; structures are cluster-side (`/data/pclancy3/yi/.../3.fine-tuning/`) | **Cluster** (JB-4) |
| C3 | ≥3 seeds for Scratch & FT-600K barrier errors | Only run-123 / run-234 single-seed evidence locally | **Cluster** (JB-2) |
| I4 | Rescope "generalizable standard" claims | Exact quotes located (sn-article-final.tex L153, L195) | **Local** text drafts (LP-7) |
| I5 | Scratch-5% control (~1,000 configs, random init) | Not run | **Cluster** (JB-3) |
| I6 | Soften t-SNE/PHATE; perplexity/seed sensitivity; separation metric in original descriptor space; SHAP surrogate R²; perturbation test | 404×6192 feature matrix cached locally (2 copies) → sensitivity + original-space silhouette fully **local**; SHAP pipeline + CSVs found in parent folder `0929-MultiT/`, `0929-multiT2/`, `SHAP/3-temperature/` → R² audit **local**; perturbation test on MACE inputs → **Cluster** (JB-8) |
| I7 | Cite arXiv:2510.05020, d5dd00534e, arXiv:2405.07105; delta vs arXiv:2509.00090 | Only cao2025migration (2509.00090) currently cited | **Local** (LP-7) |
| R1 (critical) | Fake zero forces in `neb_data_collector.py:442-443`, `neb_to_extended_xyz.py:278-282` | CONFIRMED verbatim. Generated extxyz absent locally; merged sets are broken symlinks to cluster. Whether FT runs consumed them = unconfirmed (merge config weights `data_2D_neb/ 1`, `data_Multi_T/ 0.1`) | **Local**: hard-disable fix (LP-5); **Cluster**: confirm/deny usage + conditional retrain (JB-1, JB-7) |
| R2 (critical) | Asymmetric MD protocol naive-FT: steps=2000/interval=5 vs 100000/200 | CONFIRMED verbatim; stale comment "sample every 200 steps" contradicts interval=5. FT-600K D/κ numbers potentially ~40× off | **Local**: script fix (LP-5) + error-bar re-analysis of existing valid conditions (LP-3); **Cluster**: re-run (JB-6) |
| R3 (major) | Ungrouped random split `split_data.py:52-66`; provenance `path_name`/`neb_image` never read downstream | CONFIRMED (grep-verified) | **Local**: grouped-split rewrite (LP-5); **Cluster**: retrain on new splits (folds into JB-2) |
| R4 (major) | Silhouette on t-SNE embedding (2 of 3 scripts 2D; `3d/tsne.py` is **3D**, not 2D) — methodologically invalid in all three | CONFIRMED w/ correction | **Local** recompute in original 6192-dim space (LP-2) |
| R5 (major) | Always-zero "force sensitivity" stub concatenated into features (3 sites); zero column verified present in shipped CSV | CONFIRMED (CSV last column sum = 0) | **Local**: drop dead column in re-analysis (LP-2) + remove stub (LP-5); real finite-diff sensitivity **Cluster** (later) |
| R6 (major) | SHAP code absent; 37-file `migrationbench` pkg, 13 scripts, templates, manuscript all 0-byte stubs | CONFIRMED; but working SHAP pipeline exists in parent folder (not in repo) | **Local**: audit + port into repo (LP-4, LP-5); **Cluster**: rerun with R² reporting (JB-8) |

## Missing items confirmed not present locally (must come from rockfish `/data/pclancy3/yi/`)
- MACE model checkpoints (.model/.pt), training extxyz sets (`data_600K`, `data_Multi_T`, `data_2D_neb`)
- MACE NEB per-image energies for all 14 paths; full multi-seed training logs for Table S1
- MD raw trajectories .xyz; SOAP descriptor caches; X-FORCE `shap_top_features_NEB.csv`
- Per-image QE pw.x outputs (needed for real forces in NEB extxyz export)
- `sn-article-SI.tex` SI source (only compiled SI PDFs exist) — **resolved 2026-09-06: SI will be
  re-assembled from scratch during revision; compiled PDFs serve as content reference.**
- Author confirmation (2026-09-06): zero-force NEB extxyz frames were **NOT** used in reported
  FT-600K/FT-MultiT training → JB-7 closed; JB-1 downgraded to evidence-documentation.

## Assets available locally beyond MigrationBench (parent folder `25NeurIPS-AI4MAT/`)
- `neb_optm.xyz` (71 MB), `1-2_sb2te3.xyz`, `visualization/neb_optm_1-scratch.xyz`, `visualization/neb_final_path.xyz` — NEB trajectory geometries
- `collected_png_NEB-PES/` — per-path barrier comparison panels (1-2…1-7, 2L-dislocation)
- `0929-MultiT/`, `0929-multiT2/` — SHAP pipeline (`1-shap-data.py`, `2-shap-plot-jhu.py`), prediction CSVs, slurm logs, JHU-styled figures
- `SHAP/3-temperature/` — earlier SHAP run
- `analysis_plots_comparison/`, `md_metrics_files_collected/` — MD figure assets
- `mace_l1_0802_run-123.log` — scratch training log (partial Table S1 source)
- `AI4MAT-fig-neb-diffusion.pdf`, `barrier_error_comparison_with_errorbars.png`
- `neurips2025-rebuttal/` — prior rebuttal reviewer comments (style precedent)
