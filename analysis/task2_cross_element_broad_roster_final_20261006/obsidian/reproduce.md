# Reproduce｜可复现运行顺序

本包将“训练完成的 epoch-999 模型审计”和“已有 A100 诊断结果的本地重打包/绘图”分开。模型文件不在 Git 中，运行前先按 `models/README.md` 和 `manifests/model_manifest.json` 放置并核对 SHA256。

## 1. Rockfish：验证 final checkpoint 与导出模型

在 MACE `0.3.12` 环境中，对每个 seed 运行：

```bash
python scripts/verify_final_model_remote.py \
  --model /scratch16/.../broad_roster_ft_seed123_final_epoch.model \
  --compiled-model /scratch16/.../broad_roster_ft_seed123_final_epoch_compiled.model \
  --checkpoint-dir /scratch16/.../broad_roster_ft_seed123_final_epoch \
  --expected-epoch 999 \
  --out results/provenance/seed123_final_model_provenance.json
```

将 `123` 替换为 `456`、`789`，并使用各自的 seed 目录。脚本会 fail-closed：不是唯一 epoch-999 checkpoint、tensor key 不同、max abs difference 非 0 或 compiled model 缺失时退出失败。

## 2. A100：重跑有限 profile diagnostics

在 A100 CUDA 环境中，使用 `manifests/a100_model_specs_three_seeds.json` 和
`manifests/dft_neb_provenance_manifest.json` 运行：

```bash
python scripts/run_a100_task2_profiles.py \
  --root . \
  --manifest manifests/dft_neb_provenance_manifest.json \
  --models-json manifests/a100_model_specs_three_seeds.json
```

脚本需要四个可加载的 model path，并写出 model probe、diagnostic profiles、summaries、failure ledger 和 run manifest。预期结果是 4 个模型、1,200 profile rows、120 summaries、0 failures。

## 3. 本地：打包、绘图、验证

当 A100 输出已经放入本包约定的 `results/validation/` 后，运行：

```bash
python scripts/package_three_seed_task2.py
python scripts/render_three_seed_task2_figures.py
python scripts/validate_three_seed_task2.py
```

最后一个命令应输出：

```json
{"status": "pass", "models": 4, "profiles": 1200, "qe_reference_rows": 100, "dft_paths": 10, "provenance_files": 3}
```

## 4. Scientific reading rule

任何复现结果都必须保留 `scientific_success=false`，直到源 QE/NEB 重新通过 SCF、force、NEB convergence、identity/cell 和 path-level gates。不要把非平衡 snapshot、finite probe 或 `<0.1 eV` screen 写成 DFT ground truth。
