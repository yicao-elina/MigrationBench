# 进展记录

## 已完成

1. 修正早期 epoch-0 导出问题，保留真正的 epoch-999 final model。
2. 完成三次独立 broad-roster fine-tuning：
   - seed 123：Rockfish job `31584299`
   - seed 456：array task `31585322_456`
   - seed 789：array task `31585322_789`
3. 对每个 seed 检查唯一 epoch-999 checkpoint，并把 checkpoint state dict 与导出 `.model` 逐 tensor 比较。
4. 完成 A100 validation retry `31658285`：Foundation + 三个 seed 模型均成功加载。
5. 完成十条 Cu/Li/Na 路径、十个 image、base 加两种扰动的 profile 诊断。
6. 生成 QE/NEB last-complete-table overlay、seed-wise stability、审计表、provenance 和中英文 Results/Discussion。

## 中间失败及修复

- 初次 validation job `31611100` 因包装脚本复制路径错误失败；没有加载模型，也没有科学结果。
- retry `31629587` 已加载模型并完成计算，但 CSV 写出时因 em dash 与 ASCII locale 冲突失败。
- retry `31658285` 切换 UTF-8 后完成并通过最终 package validation。

## 当前判断

训练/模型/finite 计算链已经闭合，且结果可以复现；但是路径层面的 seed variation 较大，不能据此发布 seed-invariant transferability 或 DFT barrier claim。下一步若要升级为论文级 DFT ground truth，必须重新获得通过 all-image SCF、force、NEB convergence、identity/cell 和 path-convergence gates 的 QE campaign。
