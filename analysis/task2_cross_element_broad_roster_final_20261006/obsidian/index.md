# Task 2｜三 seed broad-roster 泛化进展

> [!summary] 当前状态
> 三个 epoch-999 final model 已完成一致性验证，A100 finite validation 已完成，Task 2 的三 seed 路径诊断、QE/NEB provisional overlay、表格和 Scientific Plot Atlas 图已打包。

## 一句话结论

这项工作已经形成可复现的三 seed **模型敏感性诊断**，但还不是严格的 DFT barrier/transferability 结论：10 条归档 QE/NEB 路径均保留 SCF convergence warning，因此 QE 曲线只能作为 provisional reference。

## 关键数字

| 项目 | 结果 |
|---|---:|
| 训练 seed | 123 / 456 / 789 |
| 最终 epoch | 999（每个 seed 唯一） |
| checkpoint 与导出模型 | 90 tensors，max abs diff = 0 |
| A100 finite profile rows | 1,200 |
| profile summaries | 120 |
| validation failures | 0 |
| QE/NEB paths | 10 |
| 通过 `<0.1 eV` 筛选的路径 | 8 / 10（仅 provisional screen） |

## 图表

![[figures/task2_three_seed_profiles.png]]

![[figures/task2_three_seed_stability.png]]

![[figures/task2_three_seed_validation.png]]

## 导航

- [[progress]] — 作业、训练和交付进展
- [[model-provenance]] — epoch-999、checkpoint、SHA256 和软件环境
- [[validation]] — finite probes、validation RMSE 和 seed-wise stability
- [[dft-boundary]] — QE/NEB 路径、SCF gate 与 `<0.1 eV` 边界
- [[figures-and-tables]] — 图表与数据文件索引
- [[reproduce]] — 可复现脚本与运行顺序
