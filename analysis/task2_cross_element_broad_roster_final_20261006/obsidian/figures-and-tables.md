# Figures 与 tables

## Scientific Plot Atlas figures

### QE/NEB overlay 与三 seed profiles

![[figures/task2_three_seed_profiles.png]]

数据：`results/validation/diagnostic_profiles.csv` 与 `results/broad_roster/qe_neb_path_profiles.csv`。

### Seed-wise diagnostic stability

![[figures/task2_three_seed_stability.png]]

数据：`results/broad_roster/table_3_diagnostic_profile_summary.csv`。

### Epoch-999 validation

![[figures/task2_three_seed_validation.png]]

数据：`results/broad_roster/seed_validation_metrics.csv`。

每张图均同时提供 `.png`、`.svg` 和 `.pdf`；绘图脚本为 `scripts/render_three_seed_task2_figures.py`。

## 主要数据文件

- `results/table_1_dft_quality.csv` — 逐路径 DFT gate 与 `<0.1 eV` screen
- `results/broad_roster/qe_neb_path_profiles.csv` — QE/NEB last-complete-table overlay
- `results/broad_roster/table_2_model_element_status.csv` — element probe
- `results/broad_roster/table_3_diagnostic_profile_summary.csv` — path/variant summary
- `results/broad_roster/table_4_seed_path_stability.csv` — path-level seed spread
- `results/broad_roster/table_5_imagewise_seed_stability.csv` — image-level seed spread
- `results/broad_roster/table_6_perturbation_sensitivity.csv` — two perturbation diagnostics
- `manuscript/RESULTS_DISCUSSION_EN_ZH.md` — 中英文结果与讨论草稿
