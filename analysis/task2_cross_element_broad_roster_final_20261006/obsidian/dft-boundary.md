# QE/NEB reference boundary

## What is retained

十条 Cu/Li/Na archived QE/NEB path 的 last-complete energy table 被保留，并叠加到三 seed model profiles 上。每条 path 都保留 10 个 image、endpoint identity/cell 审计和源文件 SHA256。

## What failed

十条路径全部满足：

- `dft_scientific_success = False`
- `explicit_neb_convergence = False`
- 源输出中保留 SCF convergence warning

因此这些 QE curves 的标签是 `provisional_reference_only`，不能称为严格 DFT ground truth。

## `<0.1 eV` screen

8/10 paths 的 penultimate/final-table maximum image energy difference 小于 `0.1 eV`。这只能说明最后两张表在该指标下接近，不能替代：

- all-image SCF convergence；
- forces 与 force convergence；
- explicit NEB convergence；
- endpoint identity/cell/PBC 检查；
- 完整 path-level scientific acceptance。

完整逐路径结果在 `results/table_1_dft_quality.csv`，QE overlay 数据在 `results/broad_roster/qe_neb_path_profiles.csv`，原始 provenance 在 `manifests/dft_neb_provenance_manifest.json`。
