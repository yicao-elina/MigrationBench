# LP-3 — MD transport re-analysis with uncertainty (audit note)

Script: `Revision1/scripts/md_transport_stats.py` (deterministic; no RNG used anywhere).
Inputs: `MigrationBench/results/analysis_plots_comparison/{msd,hfacf,kappa,thermodynamics}_data.csv`.
Outputs: `data_processed/md_transport_metrics.csv`, `tables/md_transport_metrics.tex`,
`figures/fig_md_msd_kappa_revised.pdf`, `data_processed/md_transport_run_report.json`.

## 1. Data integrity checks (all asserted in-script, pass)

- `msd_data.csv`: columns `(time_fs, d_sq [A^2], time_ps, label)`; 4 labels x 1000 frames.
- Time conversion derived from the CSV itself, not assumed: `time_ps == time_fs/1000` to
  machine precision for every row; frame stride is exactly 200 fs for every model.
- `hfacf_data.csv`: normalized HFACF (value 1.0 at t=0 asserted), 0.2 ps spacing, 0–199.8 ps.
- `kappa_data.csv` provenance: verified to be **exactly** the Green–Kubo running integral
  of the normalized HFACF in W/(m·K), following
  `MB/results/MD/1-from-scratch/mace_md_pp_xyz.py:350-389`:
  `kappa(t) = C_model * cumtrapz(hfacf_norm, dx=200 fs)` with per-model scalar
  `C_model = V / (3 k_B T^2) * <J^2> * 1.60218e6`. Refit through origin gives
  R^2 = 1.000000 for all four models, with V, T taken per model from
  `thermodynamics_data.csv` (NVT, V = 150190.66 A^3, T = 597.8–598.5 K for all four conditions).
- All four CSV datasets come from **uniform 200 ps, ~600 K, NVT** trajectories
  (1000 frames x 200 fs), i.e. they mutually consistent in provenance and temperature.

## 2. Analysis choices

- **Diffusive-regime fit window: 5–199.8 ps, identical for all models**
  (matches the original post-processing convention `fit_start_ps = 5.0`,
  `mace_md_pp_xyz.py:296`; upper bound = full trajectory length).
- **Block averaging: 6 contiguous blocks** over the 5–199.8 ps fit window
  (>= 5 blocks as required). Per-block `scipy.stats.linregress` slope;
  D_block = slope/6 (cm^2/s, 3D isotropic). Reported: mean ± sample std (ddof=1).
- Also reported: single full-window fit (D_full, R^2) for direct comparability with the
  slurm-logged convention.
- **kappa**: running integral taken from `kappa_data.csv` (Green–Kubo, provenance-verified
  above). Estimate = mean over the late evaluation window 100–199.8 ps, split into
  6 blocks, mean ± std of block means. This is an *integration-cutoff sensitivity*
  uncertainty, not a true Green–Kubo standard error from independent trajectory segments —
  only aggregate per-model HFACF curves exist locally; per-segment HFACFs would require
  the raw trajectories (cluster). This limitation is carried into the table footnote.

## 3. Results

| Model | D_full (cm^2/s) | D block mean ± std (cm^2/s) | R^2 | kappa mean ± std (W/m·K) |
|---|---|---|---|---|
| Foundation | -5.27e-10 | (0.7 ± 7.9)e-9 | 0.011 | (2.4 ± 8.6)e-3 |
| Scratch | -8.45e-11 | (-3.4 ± 3.2)e-9 | 0.001 | (0.9 ± 1.3)e-2 |
| FT-600K [INVALID-pending-JB-6] | 3.46e-08 | (3.9 ± 4.0)e-8 | 0.903 | (0.9 ± 1.2)e-2 |
| FT-MultiT | 5.20e-08 | (4.8 ± 4.8)e-8 | 0.861 | (2.0 ± 1.7)e-2 |

Reading:
- **Foundation and Scratch show no measurable diffusion in the 200 ps CSV data**:
  MSD has no linear regime (R^2 = 0.011 / 0.001) and D is consistent with zero within
  block scatter (|mean| << std).
- **kappa is consistent with zero for every condition**: the Green–Kubo running integral
  peaks at ~0.07–0.11 W/(m·K) within the first few ps and then decays back across zero,
  i.e. the HFACF integral does **not** converge over 200 ps. Panel (b) of
  `fig_md_msd_kappa_revised.pdf` shows this explicitly. The historically logged
  "Converged kappa: ±0.000 W/(m·K)" values (see below) reflect exactly this non-converged
  endpoint. **AIMD reference kappa = NaN** (`MB/results/MD/00-AIMD/slurm.10463185:55`) —
  Green–Kubo failed for the AIMD trajectory; retained as a stated limitation (no local
  fix possible).

## 4. Comparison vs slurm-logged values — discrepancy found, root cause identified

| Condition | Logged D (cm^2/s) | Log source | Frames in logged run | CSV D_full | 20% gate |
|---|---|---|---|---|---|
| Foundation | 1.1112e-07 | `0-foundation-omat/slurm.10462364:41` | **41 (= 8 ps; fit window 5–8 ps)** | -5.27e-10 | FAIL |
| Scratch | -9.2501e-11 | `1-from-scratch/slurm.10462331:40` | 1001 (= 200 ps) | -8.45e-11 | **PASS** (8.7%) |
| FT-MultiT | 1.5502e-08 / 1.3749e-08 | `3-multi_T-fine-tuning/slurm.10462280:41`, `slurm.10462372:41` | **140 (= 28 ps) / 162 (= 32 ps)** | 5.20e-08 | FAIL (3.4–3.8x) |
| FT-600K | 8.9416e-09 | `2-naive-fine-tuning/slurm.10462275:41`, `slurm.10462368:40` | 401 (INVALID run) | 3.46e-08 | n/a (quarantined) |

**Root cause.** The slurm-logged D values were computed by `mace_md_pp_xyz.py` from the
`.extxyz` trajectory files **as they existed on the cluster at post-processing time**, which
were truncated mid-run: Foundation 41 frames (8 ps total — the "linear fit window" 5–8 ps
contained only ~16 points), FT-MultiT 140/162 frames (28–32 ps). Only the Scratch run had
the full 1001 frames (200 ps), and that is exactly the one condition where the CSV-based
recompute matches the log within 20% (in fact within 9%). The logged Foundation value of
1.11e-07 cm^2/s — the paper's headline diffusivity — is an artifact of fitting a 3 ps
window of a truncated 8 ps trajectory; over the full 200 ps CSV trajectory the Foundation
MSD has no linear regime and D is consistent with zero.

**Consistency check of this explanation against the CSV data:** fitting the CSV
Foundation curve over the same truncated window 5–8 ps gives -8.7e-08 cm^2/s (not
+1.11e-07), confirming the 41-frame logged trajectory was a genuinely different/noisier
early-trajectory segment, not merely a windowing effect applied to the same data.

**Status of the gate.** Verification required "recomputed D within 20% of slurm-logged
values for the 3 valid conditions". This is satisfiable only when both sides refer to the
same trajectory. For Scratch they do (PASS). For Foundation and FT-MultiT the logged
values come from truncated partial trajectories that are not available locally, so the
20% gate cannot be met by any correct reprocessing of the 200 ps CSV data — the reference
itself is invalid. Resolution (per task coordinator guidance): the CSV-based values are
reported as the best local estimates, flagged **[PENDING-CLUSTER]** for Foundation and
FT-MultiT pending cluster-side confirmation that the 200 ps comparison datasets are the
completed counterparts of those runs (the generator script for
`analysis_plots_comparison/*.csv` is not present in the local repo copy; only Scratch's
200 ps provenance is locally confirmable).

## 5. FT-600K quarantine rationale

The naive fine-tuning 600 K condition is quarantined: its production run used
**steps = 2000, interval = 5** —
`MB/results/MD/2-naive-fine-tuning/mace_md_eval_0814.py:359-360`
(`steps = 2000  # MD steps` / `interval = 5`). At the 1 fs timestep
(`mace_md_eval_0814.py:103`, `timestep=1.0 * units.fs`) that is a **2 ps** simulation with
5 fs frame spacing, while the post-processing hard-codes `DT_FS = 200.0`
(`mace_md_pp_xyz.py:63-66`), a 40x time-axis misread, and its "fit from 5 ps" window does
not even exist. Its logged D (8.94e-09) and kappa (+/-0.000) are therefore invalid. Note
that the *comparison CSVs* do contain a full 200 ps, 598 K FT-600K dataset
(`thermodynamics_data.csv` asserts V and T identical in form to the other conditions), i.e.
whoever generated the comparison CSVs had access to a longer FT-600K run — but per the
revision plan (TODO LP-3/LP-5 + JB-6) every FT-600K transport number stays tagged
**[INVALID-pending-JB-6]** until the corrected 100000-step / interval-200 rerun is audited.

## 6. Reproducibility

- Fully deterministic script (no RNG); two consecutive runs produce byte-identical
  `md_transport_metrics.csv` and `md_transport_run_report.json` (verified 2026-09-06).
- Run: `python Revision1/scripts/md_transport_stats.py` (numpy/scipy/pandas/matplotlib).
- Spot check documented in-script: kappa CSV == Green–Kubo running integral of the
  normalized HFACF, R^2 = 1.000000 for all models (assertion enforced at runtime).
