# Cr interlayer grouped R04 数据来源决策

日期：2026-10-25

## 决策

用户要求补齐 Cr 体系中 FT-600K/FT-MultiT 的 interlayer self-relaxed 行。检查
SQLite asset registry 和当前 Rockfish 结果目录后，未发现这两类 FT 的现成
self-relaxed interlayer 记录，因此没有用空行、旧图读数或跨观测类型替代。采用
平行 Rockfish campaign 重新计算：

- campaign：`ft_interlayer_sliding_self_20261025_r01`
- Rockfish root：`/scratch16/pclancy3/yi/ft_interlayer_sliding_20261025_r01`
- Slurm array：job `32490060`
- 任务：6 个 FT checkpoint × 2 个端点体系 = 12 行
- 结果：collector `PASS`，12/12 rows 通过 campaign、schema、model SHA、endpoint、
  runtime、profile、iteration-history 和 scientific gates

## 计算定义

两类端点体系为 Bulk（80 atoms）和 Cr-in-gap（81 atoms）。每行使用固定端点、
ASE IDPP endpoint-only 初始化、10 images、CPU MACE、FIRE 500 steps、
`fmax=0.05 eV/A`、`dt=0.1`、`maxstep=0.2`、`k=0.1`、seed 42。输出观测量固定
标为 `ENDPOINT_ONLY_MLIP_NEB_PROXY`，不将其命名为 DFT MEP 或 DFT barrier。

模型文件和 SHA256 由 `MODEL_REGISTRY.tsv` 冻结，作业表由 `JOBS_SELF.tsv` 冻结，
验收边界由 `ACCEPTANCE_FREEZE_FT_SELF_R01.json` 冻结。12 行的逐行证据和失败
ledger 位于：

`/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/cr_interlayer_sliding_grouped_20261025/raw/rockfish/ft_interlayer_sliding_20261025_r01/`

## 图组中的合并规则

- Foundation self-relaxed：使用现有 Foundation 2×2 R13 endpoint-only MLIP NEB
  profile；不与 fixed-path 行合并成同一种 observable。
- FT-600K/FT-MultiT self-relaxed：使用本次 Rockfish 12 行结果。
- Foundation/FT fixed-path：使用已审核的 fixed R04 表。
- Scratch fixed/self：使用内部 checkpoint-matched 的 Rockfish Lmax1 fixed/self
  结果，使 Scratch 的 fixed 与 self 在同一组图中可配对；旧 fixed R04 Scratch
  checkpoint 的 SHA256 差异单独记录于 `SCRATCH_CHECKPOINT_AUDIT.tsv`。
- 组内阴影为三个模型 identity 间的 pointwise sample SD（ddof=1）；不把 fixed 和
  self-relaxed 行混成一条 barrier 排名。

## 结论边界

本次来源恢复完成了图组所需的 FT self-relaxed coverage，但它没有把 endpoint-only
MLIP NEB proxy 升格为 DFT MEP，也没有消除 DFT `COMPLETED_BUT_SCF_WARNING` 状态。
因此图 provenance manifest 保持 `PASS_WITH_SCIENTIFIC_BOUNDARIES` 和
`RECONSTRUCTION`，并保留全部原始 Rockfish evidence、manifest、runtime provenance
和 collector ledger。
