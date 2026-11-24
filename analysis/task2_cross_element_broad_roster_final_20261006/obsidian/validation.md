# Validation 与 seed-wise stability

## Held-out validation

| seed | energy RMSE (meV atom⁻¹) | force RMSE (meV Å⁻¹) |
|---:|---:|---:|
| 123 | 11.33 | 203.62 |
| 456 | 12.16 | 208.76 |
| 789 | 12.59 | 222.02 |

这些是训练分布上的 held-out validation diagnostics，不等同于 NEB 路径上的 DFT 误差。

## A100 path diagnostics

- 4 models × 10 paths × 10 images × 3 variants = 1,200 profile rows。
- 三种 variant：`base`、`perturb_a_dx020`、`perturb_b_tilt020`。
- 120 条 path/model/variant summaries。
- 0 calculation failures；所有记录 finite。
- 所有 path records 都保持 `scientific_success=false`，因为源 QE/NEB DFT gate 未通过。

## Seed variation

在排除 Foundation 的 20 个 path-level stability entries 中：

- 最大相对能量跨度的平均路径内标准差：`0.440 eV`。
- 最大 force 的平均路径内标准差：`0.438 eV Å⁻¹`。
- 最大 seed range：能量 `8.453 eV`，force `8.828 eV Å⁻¹`。

因此，训练 RMSE 的小幅变化不能被解释为 NEB-like 非平衡 snapshot 上的 seed-invariant transferability。

证据文件：

- `results/broad_roster/seed_validation_metrics.csv`
- `results/broad_roster/table_4_seed_path_stability.csv`
- `results/broad_roster/table_5_imagewise_seed_stability.csv`
- `results/validation/diagnostic_profiles.csv`
- `results/validation/diagnostic_summary.csv`
