# LP-6 — NEB PES / trajectory asset audit (Critical-C1, local side)

Date: 2026-09-06. Script: `Revision1/scripts/lp6_neb_pes_audit.py` (seed 0).
Reviewer issue: Fig 3d (fixed-geometry single-point MACE energies on DFT NEB geometries;
Foundation overestimates the in-gap barrier, DFT ref 0.336 eV, by ~0.7 eV) vs Fig S3
(Foundation self-run NEB converges to 0.41 eV).

## Verdict (one paragraph)

**Nothing quantitative behind Fig 3c–f or Fig S3 is locally recomputable.** The four
MACE models' fixed-geometry single-point energies on DFT NEB geometries (Fig 3c–f,
bar plot 3b) exist locally ONLY as PNG pixels; all `*.model` checkpoints and the
panel-generating scripts are absent locally (`MigrationBench/scripts/postprocessing/
energy_barrier_analysis.py` and `prediction_error.py` are 0-byte stubs). The only local
self-NEB trajectory (`neb_optm.xyz`) is the **Scratch** model's, and it is **diverged**
(interior images oscillate by 10^2–10^4 eV through the last step). The Foundation
"converged 0.41 eV" MEP exists locally only as `neb_energy_barrier.png` (identical to
Fig S3b) plus a geometry file (`neb_final_path.xyz`) whose energy field is a constant
placeholder. All numbers needed for a revised Fig 3 / Fig S3 are therefore
**[PENDING-CLUSTER]** (rockfish `3.neb/`, task JB-5), except the DFT reference
profiles (LP-1) and visual/approximate digitizations recorded here.

## File-by-file findings

### `P/neb_optm.xyz` == `P/visualization/neb_optm_1-scratch.xyz`
- **Byte-identical copies** (md5 `f91230462bcd8521977c90301997f20a`, both 68 MB). Two
  copies, one dataset. The "_1-scratch" name tags model=**Scratch**, path label "1".
- extended XYZ: 82 atoms (Te48 Sb32 Cr2), `energy`, `free_energy`, `forces`, full PBC.
- **8038 frames = 1339 optimizer steps x 6 images** (+ 4 trailing frames of a step 1340).
  Frame 6 k has **constant energy -179665.26139882064 eV in all 1339 steps** = fixed
  band endpoint (image 0). Images 1–5 fluctuate: per-image relative energies vs the
  endpoint span **-1208 to +47 122 eV** over the run; in the **final** step they are
  [0, +836, +384, +345, +306, +284] eV, and over the last 100 steps the band maximum
  never drops below **325 eV** (range 325–1742 eV). **This NEB never converged**
  — energies are real eV (training-set scale ≈ -2191 eV/atom, matching
  `03_neb/mlff_training_data/processing_report.json` per-atom energies and the Fig-S1
  parity plot range), so the oscillation is physical explosion, not a unit artifact.
- No generating script found locally (grep for `neb_optm` over P/ finds only the
  revision TODO/gap-analysis docs).
- Extracted: `data_processed/neb_optm_frame_energies_raw.csv` (8038 rows, raw),
  `data_processed/neb_selfneb_scratch_per_step.csv` (per step x image, relative to
  endpoint).

### `P/visualization/neb_final_path.xyz`
- 12 frames, same lattice as above (Te48 Sb32 Cr2), `energy` + `stress` metadata.
- **All 12 image energies are IDENTICAL (-309.0132814571389 eV)** — placeholder written
  by a lazy exporter (energy of one frame copied to all). -3.77 eV/atom is the
  MACE-MP-0/**Foundation** energy scale, unlike the Scratch-scale trajectory above.
- Interpretation: **12 converged self-NEB images (geometries) of the Foundation run**
  that produced Fig S3a/b; energies must be re-evaluated — they are NOT in this file.
- The two POV-Ray renders `neb_final_path_frame_0000/0011.png` = images 0 and 11
  (endpoints) of this file (renderer: `visualization/pov_ray_traj.py`).

### `P/1-2_sb2te3.xyz`
- 5 frames, 61 atoms (Te36 Sb24 Cr1), **plain xyz, empty comment lines — NO energy
  metadata**. Consistent with the 5-image DFT NEB cell of path 1-2 (LP-1:
  `03_neb/221_vdw_corr_DFT_D3/1-2`). Geometries only; MACE/DFT energies absent.

## Panel -> paper mapping

Paper figure files (all 1-page PDFs in `submission/Fig/`):

| Local asset | Paper panel | Contents / quantities shown |
|---|---|---|
| `Fig/NeurIPS2025-AI4MAT-figure4-v2.pdf` | included at tex L259 as Fig 3 (`fig:barrier`) | rows a–d of POV-Ray structure snapshots ONLY — **does not match its caption** (caption promises (a) schematic, (b) bar plot, (c–f) MEP profiles). Figure-file/caption mismatch to fix in revision. |
| `Fig/NeurIPS2025-AI4MAT-Em_SI.pdf` | actual composite matching the Fig-3 caption | (a) in-gap/deep-penetration schematics; (b) ΔEm bar plot {FT-600K 12.5, FT-MultiT 13.3, Foundation 10.9, Scratch 5.5 eV (deep/OOD); 0.1, 0.25, 0.4, 2.3 eV (in-gap)}; (c) MEP = path **1-7**; (d) MEP = path **1-4** (the reviewer's "Fig 3d"); (e) MEP = path **1-2**; (f) MEP = path **1-5** |
| `Fig/NeurIPS2025-AI4MAT-Em-1013.pdf` | bar-plot + schematic variant (older draft) | same ΔEm numbers as Em_SI(b) |
| `barrier_error_comparison_with_errorbars.png` | older/different draft of Fig 3b | **different numbers**: deep 9.3/10.1/7.9/2.2 eV, in-gap 0.18/0.35/0.32/2.3 eV — provenance conflict; do not reuse without recomputation |
| `AI4MAT-fig-neb-diffusion.pdf` | draft composite of Fig 3 | panels b=1-7?, c=1-4, d=1-2, e=1-5 (MEP) + renders |
| `collected_png_NEB-PES/1-4_..._final.png` | Fig 3d (Em_SI panel d) | path 1-4, 5 uniform-x markers/curve |
| `1-7_..._final.png` | Fig 3c | path 1-7, 5 markers |
| `1-2_..._final.png` | Fig 3e | path 1-2 (deep penetration, DFT peak 2.79 eV) |
| `1-5_..._final.png` | Fig 3f | path 1-5 |
| `1-3_..._final.png` | SI candidate (not in main Fig 3) | path 1-3 (DFT peak 4.00 eV) |
| `1-7_..._1-7.png` | superseded draft of 1-7 panel | ~10 markers/curve (earlier NEB discretization) |
| `2L-dislocation_1.bulk_..._final.png` | SI candidate (2D_neb family) | 10 markers/curve = 10-image 2D_neb runs |
| `2L-dislocation_2.cr-in-gap_..._final.png` | SI candidate | 5 markers (mixed discretization — flagged) |
| `Fig/NeurIPS2025-AI4MAT-SI2.pdf` | **Fig S3** | (a) initial/final state renders; (b) self-NEB MEP, annotated **Em = 0.41 eV**, 12 points, all values ≤ 0 (relative to image 0; curve: 0, -0.70, -0.74, -0.74, -0.73, -0.71, -0.60, -0.41, -0.32, -0.55, -1.04, -0.39 — the "barrier" 0.41 eV is not a simple max-min of this curve: max-initial = 0; likely max(-0.32)-min(-1.04)≈0.72 or endpoint-reference arithmetic — **the plotted MEP needs recomputation and explanation**); (c) energy + fmax convergence of all 4 models |
| `neb_energy_barrier.png` | = Fig S3b panel (standalone) | MEP curve above; 12 points match `neb_final_path.xyz`'s 12 images, but that file has no usable energies -> underlying data cluster-only |
| `convergence_plot_aligned.png` | = Fig S3c top (energy vs step) | FT-600K and FT-MultiT energies drift to +40–60 eV (not converged); Foundation/Scratch short runs |
| `fmax_convergence_plot.png` | = Fig S3c bottom (fmax vs time) | fmax spikes: Scratch ~1500 eV/A, FT-600K ~1200, FT-MultiT ~1100 → "explosive behavior" text claim (tex L307) |

Panel DFT curves were cross-checked against `MB/.../221_vdw_corr_DFT_D3/*/sb2te3.dat`
column 2: exact match for 1-2, 1-3, 1-4, 1-5, 1-7 (e.g. 1-4 = [0, 0.2895, 0.3360,
-0.1465, 0.3192] eV). Note panels use uniform x = 0/.25/.5/.75/1 while the true DFT
reaction coordinates (col 1) are non-uniform (0.272/0.422/0.703 for 1-4) — a cosmetic
inaccuracy worth fixing in regenerated panels.

## Digitized (approximate) values — path 1-4 panel
`data_processed/neb_pes_panel_1-4_digitized_approx.csv` (tick-mark-calibrated pixel
digitization; independent validation: digitized DFT markers vs exact sb2te3.dat values,
max |err| = 0.066 eV; ±0.07 eV):

| x | DFT | FT-600K | FT-MultiT | Foundation | Scratch |
|---|---|---|---|---|---|
| 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.25 | 0.325 (exact 0.290) | 0.423 | 0.553 | 0.990 | 1.063 |
| 0.50 | 0.402 (exact 0.336) | 0.516 | 0.783 | 0.888 | 1.568 |
| 0.75 | -0.152 (exact -0.146) | 0.100 | 0.734 | 1.042 | 4.534 |
| 1.00 | 0.316 (exact 0.319) | -0.127 | 0.470 | 0.602 | 1.860 |

**Critical-C1 number reproduced by digitization**: Foundation peak 1.04 eV vs DFT
0.336 eV → overestimate **+0.71 eV ≈ "+0.7 eV"** ✓.

## Verification log
- Frame count: neb_optm.xyz = 8038 = 1339 x 6 + 4; endpoint constancy checked exactly.
- DFT anchor: 1-4 sb2te3.dat col2 peak = 0.3360499 eV; LP-1 `dft_neb_barriers.csv` gives
  0.336050 ✓ (also confirms 5 images/path for the 221_vdw_corr_DFT_D3 family — matches
  the 5-marker panels; 2D_neb neb_1..5 have 10 images — matches the 10-marker panels).
- Digitization validated against exact DFT values (max error 0.066 eV).
- `processing_report.json` energy range (-177388 eV for 81-atom cells ≈ -2190 eV/atom)
  confirms neb_optm.xyz energies are on the same real-eV training scale.

## Completeness matrix — Fig 3 / Fig S3

| Panel | Status | Local basis | Remaining cluster need |
|---|---|---|---|
| Fig 3a schematic (migration pathway) | PARTIAL | POV-Ray renders exist locally (`neb_final_path_frame_*.png`, visualization/) | regenerate at will; no energy data needed |
| Fig 3b bar plot (ΔEm OOD vs in-gap) | PARTIAL | exists only as conflicting PNG/PDF versions (Em_SI/Em-1013 vs barrier_error_comparison_with_errorbars.png disagree) | **[PENDING-CLUSTER]**: recompute per-path MACE barriers with one checkpoint & canonical DFT reference, then aggregate mean±std |
| Fig 3c (= path 1-7 MEP) | PARTIAL | PNG only (+ digitization possible); DFT curve exact via sb2te3.dat | **[PENDING-CLUSTER]**: 4 models' single-point energies on the 5 DFT images |
| Fig 3d (= path 1-4 MEP) | PARTIAL | PNG + validated digitization (table above); DFT exact | **[PENDING-CLUSTER]**: same single-point re-eval |
| Fig 3e (= path 1-2 MEP) | PARTIAL | PNG only; DFT exact; `1-2_sb2te3.xyz` has the 5 geometries (no energies) | **[PENDING-CLUSTER]**: single-point re-eval |
| Fig 3f (= path 1-5 MEP) | PARTIAL | PNG only; DFT exact | **[PENDING-CLUSTER]**: single-point re-eval |
| Fig S3a (initial/final states) | LOCAL(ish) | `neb_final_path.xyz` images 0/11 geometries + renders | none essential |
| Fig S3b (Foundation self-NEB MEP, Em=0.41) | **PNG-ONLY** | `neb_energy_barrier.png`; geometries in `neb_final_path.xyz` (12 images) but **energy metadata is a constant placeholder** | **[PENDING-CLUSTER]**: the Foundation NEB optimizer log or per-image energies; also clarify the 0.41 eV definition (plotted curve is all ≤ 0) |
| Fig S3c (convergence energy + fmax, all 4 models) | **PNG-ONLY** | the two loose PNGs; only Scratch's raw trajectory is local (and diverged) | **[PENDING-CLUSTER]**: optimizer trajectories/logs for Foundation, FT-600K, FT-MultiT self-NEB runs |
| Scratch self-NEB (supporting C1 response) | LOCAL | `neb_optm.xyz` parsed, diverged (this audit + figS) | none |

## Deliverables
- `Revision1/scripts/lp6_neb_pes_audit.py` — regenerates everything below (seed 0).
- `Revision1/data_processed/neb_optm_frame_energies_raw.csv` — 8038 frame energies.
- `Revision1/data_processed/neb_selfneb_scratch_per_step.csv` — 1339 steps x 6 images, relative energies.
- `Revision1/data_processed/neb_pes_models_local.csv` — combined frame-level table for all four xyz assets, with per-row notes on duplicates / fake / absent energies.
- `Revision1/data_processed/neb_pes_panel_1-4_digitized_approx.csv` — digitized Fig 3d (approximate).
- `Revision1/figures/figS_neb_pes_local_audit.pdf` — audit figure: (a) Scratch self-NEB per-image relative energies (symlog), (b) band-max barrier proxy vs DFT 0.336 eV and the Fig S3 0.41 eV claim.

## Why no fixed-geometry comparison figure could be produced locally
The legitimate version of Fig 3c–f requires MACE single-point energies on the DFT NEB
geometries. No MACE checkpoint exists anywhere under `P/` or `MigrationBench/` (searched
for `*.model`, `*.pt`, `*.pth`, checkpoints), and the scripts that generated the panels
are not local, so the only "data" are PNG pixels. Digitizing them into a camera-ready
figure would violate the no-copied-from-PDF rule. Hence
`figS_neb_pes_local_audit.pdf` is an **audit artifact** (what is locally true: the
Scratch self-NEB diverged), not a replacement paper figure.

## Bottom line for JB-5 (cluster): the numbers still needed
1. Per-image single-point energies of all 4 MACE checkpoints on all DFT NEB image sets
   (221_vdw_corr_DFT_D3 paths 1-2…1-8, 5 images each; 2D_neb neb_1..5, 10 images each)
   with one pinned model version each → Fig 3b–f.
2. The Foundation self-NEB optimizer trajectory (per-step per-image energies, fmax) and
   its final 12-image energies → Fig S3b verification of "0.41 eV" (current local evidence
   contradicts a plain max-min reading of the published curve).
3. FT-600K / FT-MultiT self-NEB optimizer logs → Fig S3c (energy drift +40–60 eV suggests
   these did NOT simply "terminate early"; the claim needs the raw logs).
4. Canonical choice of bar-plot values (two conflicting local drafts: 12.5/13.3/10.9/5.5
   vs 9.3/10.1/7.9/2.2 eV deep-penetration ΔEm) must come from recomputation, not selection.
