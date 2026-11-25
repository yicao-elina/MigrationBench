# Task 2 — three-seed final-model Results and Discussion

## Results (English)

Three independent Cr-trained broad-roster fine-tuning runs were completed with seeds 123, 456, and 789. Each run reached the requested 1,000 epochs and produced a unique epoch-999 checkpoint. The exported `.model` was compared against the checkpoint state dictionary: all 90 tensor keys matched exactly and the maximum absolute tensor difference was 0.0 for all three seeds. The corresponding compiled models were also present. This resolves the earlier epoch-0 export problem and establishes that the Task 2 models used here are the actual epoch-999 training states.

The held-out validation diagnostics at epoch 999 were 11.33, 12.16, and 12.59 meV atom⁻¹ for energy RMSE and 203.62, 208.76, and 222.02 meV Å⁻¹ for force RMSE for seeds 123, 456, and 789, respectively. An A100 validation run loaded Foundation and all three final seed models, supported Cr, Cu, Li, Na, Sb, and Te for every model, and produced 1,200 finite path-profile rows across ten archived paths, ten images, three coordinate variants, and four models. No calculation failures were recorded.

The path profiles show clear seed-dependent variation. Across the 20 path-level seed-stability entries (three seeds, excluding Foundation), the mean within-path standard deviation was 0.440 eV for the maximum relative-energy span and 0.438 eV Å⁻¹ for the maximum force; the largest seed ranges were 8.453 eV and 8.828 eV Å⁻¹, respectively. These numbers are diagnostic measures of sensitivity across independently trained models, not estimates of DFT uncertainty. In particular, the large ranges on some paths prevent a claim of seed-invariant transferability.

The figure set overlays the three epoch-999 seed profiles with the last complete QE/NEB energy table for representative Cu, Li, and Na paths. The QE/NEB curves are included to preserve the requested path reference and provenance, but they are explicitly labelled provisional: all ten archived NEB paths retain SCF-convergence warnings and none passes the strict DFT scientific-success gate. Eight paths pass the penultimate/final-table energy-difference screen below 0.1 eV, but that screen does not replace all-image SCF, force, NEB, identity, and path-convergence acceptance.

## Discussion (English)

The corrected three-seed experiment supports two bounded conclusions. First, the broad Foundation roster is retained operationally: all three final models load on A100 and return finite probes for the six elements represented in Task 2. Second, the energy/force path diagnostics are not seed-invariant enough to support a publication-grade transfer or barrier claim. The appropriate interpretation is therefore a reproducible model-sensitivity diagnostic, not a validated cross-element migration barrier prediction.

The validation RMSE increases modestly from seed 123 to seed 789, especially for forces, while the path-level spread is much larger on selected non-equilibrium snapshots. This mismatch is scientifically informative: held-out training-distribution metrics do not by themselves guarantee stable behavior on archived NEB-like geometries, especially when the source path is not a converged DFT reference. Future publication claims should require a converged, all-image QE/DFT campaign and an explicitly designed out-of-distribution evaluation set before promoting these profiles to ground-truth error or transfer-gain analyses.

## 结果（中文）

我们使用随机种子 123、456 和 789 独立完成了三次 Cr-trained broad-roster fine-tuning。每个训练都完成了要求的 1000 个 epoch，并生成唯一的 epoch-999 checkpoint。随后将导出的 `.model` 与 checkpoint 的 state dictionary 逐 tensor 比较：三个 seed 均有 90 个 tensor key 完全匹配，最大绝对差值为 0.0，compiled model 也全部存在。因此，之前的 epoch-0 导出问题已经被修正，本 Task 2 使用的确实是 epoch-999 训练状态。

epoch-999 的 held-out validation diagnostics 中，三个 seed 的 energy RMSE 分别为 11.33、12.16 和 12.59 meV atom⁻¹，force RMSE 分别为 203.62、208.76 和 222.02 meV Å⁻¹。A100 validation 成功加载 Foundation 和三个最终 seed 模型；四个模型均支持 Cr、Cu、Li、Na、Sb、Te，并在十条归档路径、十个 image、三种坐标变体上产生 1200 条有限 profile，未记录 calculation failure。

路径 profile 显示出明显的 seed 依赖性。在排除 Foundation 后的 20 个 path-level stability entries 中，三个 seed 对最大相对能量跨度的平均路径内标准差为 0.440 eV，对最大 force 的平均路径内标准差为 0.438 eV Å⁻¹；最大 seed range 分别达到 8.453 eV 和 8.828 eV Å⁻¹。这些数值是独立模型之间的 diagnostic sensitivity，不是 DFT uncertainty；部分路径上的较大差异意味着目前不能宣称 seed-invariant transferability。

图中将三个 epoch-999 seed 的 profile 与 Cu、Li、Na 代表性路径的最后完整 QE/NEB energy table 叠加。QE/NEB 曲线被保留作为路径参考和 provenance reference，但明确标记为 provisional：十条归档 NEB 路径均保留 SCF convergence warning，因此没有任何一条通过严格的 DFT scientific-success gate。虽然其中八条路径的 penultimate/final-table energy difference 小于 0.1 eV，但这一筛选不能替代 all-image SCF、force、NEB、identity 和 path-convergence 验收。

## 证据边界与文件

本结果包的所有路径值都属于 non-equilibrium snapshot diagnostics；没有把它们包装成严格 DFT ground truth、DFT error、barrier accuracy 或 transfer gain。主要文件包括：

- `results/provenance/seed123_final_model_provenance.json`
- `results/provenance/seed456_final_model_provenance.json`
- `results/provenance/seed789_final_model_provenance.json`
- `results/broad_roster/table_4_seed_path_stability.csv`
- `results/broad_roster/table_5_imagewise_seed_stability.csv`
- `results/broad_roster/qe_neb_path_profiles.csv`
- `figures/task2_three_seed_stability.png/.svg/.pdf`
- `figures/task2_three_seed_profiles.png/.svg/.pdf`
- `figures/task2_three_seed_validation.png/.svg/.pdf`
