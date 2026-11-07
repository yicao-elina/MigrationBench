# MigrationBench revised manuscript

正文、SI 和 Reviewer 1 回复已经按这次选定的图重写，并分别提供清稿和审稿定位版。正文仅引用指定的 Fig. 1–7；SI 包含四张新选图、必要的原有证据图，以及本次对当前 927 帧数据补算的原始空间诊断图。

**这是一套已编译、已核对的修订稿，仍不能认定为科学证据全部闭合的 camera-ready 稿。** 带 SCF 警告的参考路径、合并表的状态冲突，以及部分新旧数据的审计范围需要按 READINESS_NOTE.md 处理。修订稿已明确限定这些结果的含义，没有把它们称作已验证的物理势垒。

## 阅读文件

| 文件 | 用途 |
|---|---|
| paper/revision1/sn-article-clean.pdf | 正文清稿 |
| paper/revision1/sn-article-final.pdf | 正文蓝字版，直接回应句带 R1-A1 至 R1-C13 标记 |
| paper/revision1/sn-article-SI-clean.pdf | SI 清稿 |
| paper/revision1/sn-article-SI.pdf | SI 的正式源文件对应编译稿 |
| paper/revision1/response_to_reviewers.pdf | 保留 13 条原始意见的逐条回复 |
| FIGURE_PANEL_MAP.md | 补加或统一 panel 字母时使用 |
| REVIEWER_COVERAGE.tsv | 审稿意见、对应位置和保留限制的索引 |

## 科学主线

将“foundation + fine-tuning 是否最好”改为：模型是否同时满足目标过程需要的能量标定、路径行为和动力学稳定性。

当前数据支持的重点是：fine-tuning 改善部分力与固定图像能量指标，但不保证长轨迹稳定；零样本模型有更长的实际 MD 观察窗口，但可能严重错估某些迁移构型的能量。数据分布差异是组织诊断的思路，并未被证明为唯一因果机制。主动学习保留为有明确验收方式的未来研究方向。

## 编译

在本文件所在目录运行：

    python3 build.py

需要 TeX Live / MacTeX、latexmk 和随包提供的期刊类文件。编译脚本依次处理正文和 SI 的交叉引用，再编译清稿及回复。最终检查不允许未解析引用。原项目的完整 bibliography 保留；这套稿件实际使用独立的 references-camera-ready.bib。

## 复现本次核对

SHAP 输入及非空执行脚本位于 reproducibility/。参考环境版本见 requirements-current.txt。使用该环境运行：

    python reproducibility/run_shap_full_campaign.py --xyz reproducibility/source_data/common_test_267.xyz --source-tables reproducibility/source_tables --output shap_rerun

四张输出表已实际重跑，逐项数值差为 0；检查结果在 paper/revision1/data_processed/current_release/shap_reproduction_check.json。这是当前 267 帧分析的复现结果，不是对旧 150 帧结果的复现声明。

同一当前数据上的原始空间指标与投影敏感性可运行：

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python reproducibility/recompute_feature_metrics.py --output feature_audit_rerun

输入保留全部 float64 数值及 Unicode 模型标签；读取不需要 pickle。常数占位列在计算前被检查并删除。

## 原目录替换

本轮原目录只读，未覆盖 /Users/alina/Project/migrationbench-paper-repro/paper/revision1。整套更新在当前目录中已经完成。可先运行：

    python3 apply_to_original.py --check

如要应用，运行：

    python3 apply_to_original.py --apply

脚本会先确认三个原始 manuscript 源文件仍与任务开始时一致，并备份所有将被覆盖的现有文件。若原稿已经被其他人修改，它会停止，不会覆盖新的改动。

Fig. 5 按 model hash 对齐的原始记录修正了一个错误的零峰值和两个状态；原始提供版保存在 source_snapshot/fig5_supplied_original.pdf。可复现修图：

    python reproducibility/render_fig5_corrected.py --input paper/revision1/data_processed/current_release/fig5_corrected_input.tsv --output fig5_rerun
