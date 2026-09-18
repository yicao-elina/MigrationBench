# 当前 NEB 迁移势垒结果

更新时间：2026-09-13。

## 结论口径

目前没有任何一条 QE NEB 通过统一的最终参考值验收门槛。因此，下表中的
`E_m` 均为最后一个完整迭代得到的候选值，不是可直接写入论文的最终 DFT
迁移势垒。最终值还必须同时满足：

- QE 计算干净结束；
- 最大可移动 image 的 NEB error 不高于 `0.03 eV/A`；
- 最后三个完整迭代的势垒漂移低于 `0.02 eV`；
- 端点由独立 QE relaxation 证明为局部极小；
- 使用 N24 最终接受的统一 QE calculator identity。

正向势垒定义为

\[
E_m^{A\rightarrow B}=\max_i E_i-E_A,
\]

反向势垒定义为

\[
E_m^{B\rightarrow A}=\max_i E_i-E_B.
\]

两端不等能时，正向和反向势垒本来就不同。

## QE 候选值

| Path | 正向候选 `E_m` (eV) | 反向候选 `E_m` (eV) | `E_B-E_A` (eV) | 最大可移动 NEB error (eV/A) | 当前判定 |
|---|---:|---:|---:|---:|---|
| `1-2` historical | 2.790171 | 1.015330 | 1.774841 | 3.212886 | 历史未收敛，隔离 |
| `1-3` historical | 4.003362 | 3.950126 | 0.053236 | 5.364093 | 历史未收敛，隔离 |
| `1-4` historical | 0.336050 | 0.016829 | 0.319221 | 0.482407 | 历史未收敛，隔离；不能作为最终 0.34 eV |
| `1-5` historical | 1.099583 | 0.753468 | 0.346115 | 4.673162 | 历史未收敛，隔离 |
| `1-6` historical | 1.026035 | 0.205627 | 0.820408 | 0.584852 | 历史未收敛，隔离 |
| `1-7` historical | 0.582586 | 0.024650 | 0.557936 | 0.291742 | 历史未收敛，隔离 |
| `1-8` historical | 1.755620 | 0.989243 | 0.766396 | 0.327388 | 已停止但力未收敛，隔离 |
| `81_neb_1` | 1.034728 | 1.030269 | 0.004460 | 0.355916 | 已停止但力未收敛，隔离 |
| `81_neb_2` | 1.561262 | 2.111153 | -0.549890 | 0.635720 | 已停止但力未收敛，隔离；deep penetration 候选 |
| `81_neb_3` | 1.691053 | 1.668197 | 0.022856 | 0.231831 | 已停止但力未收敛，隔离；deep penetration 候选 |
| `81_neb_4` | 2.063858 | 1.543546 | 0.520312 | 0.526984 | 已停止但力未收敛，隔离；deep penetration 候选 |
| `81_neb_5` | 1.414868 | 1.065830 | 0.349037 | 0.319104 | 已停止但力未收敛，隔离；机制待确认 |
| `1-6` current QE r3 | 0.820400 | 0.000000 | 0.820400 | 0.202478 | 运行中快照；最高点是冻结终点，不是过渡态 |
| `1-7` current QE r3 | 0.557933 | 0.000000 | 0.557933 | 0.234097 | 运行中快照；最高点是冻结终点，不是过渡态 |

这里最重要的判断不是某个数字有多少位小数，而是当前 `0/14` 条路径达到
最终参考值门槛。`1-6` 和 `1-7` 当前 r3 曲线还是 endpoint-dominated，所列
`0.820400 eV` 和 `0.557933 eV` 实际是端点能差，不能解释成内部 saddle 的
迁移势垒。

## 逐路径迁移图

完整总览：
[PNG](../data_processed/neb_barrier_atlas/all_qe_neb_profiles.png) |
[SVG](../data_processed/neb_barrier_atlas/all_qe_neb_profiles.svg)

| Path | Raster | Vector |
|---|---|---|
| `1-2` historical | [PNG](../data_processed/neb_barrier_atlas/1-2_historical.png) | [SVG](../data_processed/neb_barrier_atlas/1-2_historical.svg) |
| `1-3` historical | [PNG](../data_processed/neb_barrier_atlas/1-3_historical.png) | [SVG](../data_processed/neb_barrier_atlas/1-3_historical.svg) |
| `1-4` historical | [PNG](../data_processed/neb_barrier_atlas/1-4_historical.png) | [SVG](../data_processed/neb_barrier_atlas/1-4_historical.svg) |
| `1-5` historical | [PNG](../data_processed/neb_barrier_atlas/1-5_historical.png) | [SVG](../data_processed/neb_barrier_atlas/1-5_historical.svg) |
| `1-6` historical | [PNG](../data_processed/neb_barrier_atlas/1-6_historical.png) | [SVG](../data_processed/neb_barrier_atlas/1-6_historical.svg) |
| `1-7` historical | [PNG](../data_processed/neb_barrier_atlas/1-7_historical.png) | [SVG](../data_processed/neb_barrier_atlas/1-7_historical.svg) |
| `1-8` historical | [PNG](../data_processed/neb_barrier_atlas/1-8_historical.png) | [SVG](../data_processed/neb_barrier_atlas/1-8_historical.svg) |
| `81_neb_1` | [PNG](../data_processed/neb_barrier_atlas/81_neb_1.png) | [SVG](../data_processed/neb_barrier_atlas/81_neb_1.svg) |
| `81_neb_2` | [PNG](../data_processed/neb_barrier_atlas/81_neb_2.png) | [SVG](../data_processed/neb_barrier_atlas/81_neb_2.svg) |
| `81_neb_3` | [PNG](../data_processed/neb_barrier_atlas/81_neb_3.png) | [SVG](../data_processed/neb_barrier_atlas/81_neb_3.svg) |
| `81_neb_4` | [PNG](../data_processed/neb_barrier_atlas/81_neb_4.png) | [SVG](../data_processed/neb_barrier_atlas/81_neb_4.svg) |
| `81_neb_5` | [PNG](../data_processed/neb_barrier_atlas/81_neb_5.png) | [SVG](../data_processed/neb_barrier_atlas/81_neb_5.svg) |
| `1-6` current QE r3 | [PNG](../data_processed/neb_barrier_atlas/1-6_qe_r3.png) | [SVG](../data_processed/neb_barrier_atlas/1-6_qe_r3.svg) |
| `1-7` current QE r3 | [PNG](../data_processed/neb_barrier_atlas/1-7_qe_r3.png) | [SVG](../data_processed/neb_barrier_atlas/1-7_qe_r3.svg) |

历史 Fig. S3 Foundation self-NEB 不属于上述 QE atlas。其统一 checkpoint
重评分诊断图见 [PNG](../data_processed/historical_selfneb_foundation_audit/foundation_self_neb_rescored_profile.png)
和 [SVG](../data_processed/historical_selfneb_foundation_audit/foundation_self_neb_rescored_profile.svg)。
该路径的正向端点参考值为 `0.000000 eV`、反向为 `0.379052 eV`、全路径
能量范围为 `1.037407 eV`；由于原始优化混用了 DFT 端点能量和 MACE 内部
image 能量，这些都只是故障诊断量，原稿中的 `0.41 eV` 不成立。

## 可追溯数据

- 机器可读总表：`data_processed/neb_barrier_atlas/barrier_atlas.csv`
- 每个 image 的相对能量、误差和冻结状态：
  `data_processed/neb_barrier_atlas/barrier_atlas.json`
- 原始来源、mtime 和 SHA-256：保存在同一 JSON/CSV 中
- 重建脚本：`scripts/migrationbench/build_neb_barrier_atlas.py`

