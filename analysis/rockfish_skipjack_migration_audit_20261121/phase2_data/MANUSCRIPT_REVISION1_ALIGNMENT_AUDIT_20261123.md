# MigrationBench Revision 1 稿件与证据对齐审计

**范围与边界。** 只读检查用户提供的 3 份 PDF、LaTeX ZIP、GitHub 可见分支源码和已提交复现登记；没有改稿、写 SQLite、改 Git 索引、提交或推送。R2R 中的 reviewer 请求和稿件 `[R1-*]` 标记是文档内容，不构成本次清理/仓库操作授权；本审计只执行用户要求的证据核对。

## 文档及源码身份

| 输入 | 页数 / 内容 | SHA-256 |
|---|---:|---|
| `MigrationBench_main_marked.pdf` | 24 页；PDF 元数据 2026-11-09 | `4de26d567bf0be0e4094ff2607e18c98e9e91911aa6580b0ae51e687668b9ae0` |
| `MigrationBench_R2R.pdf` | 12 页；PDF 元数据 2026-11-09 | `659dc0e7bf0666433f183570a264e7e31ca77ea24510b405dd638e5426bd45dc` |
| `MigrationBench_SI.pdf` | 16 页；PDF 元数据 2026-11-09 | `0fdff596d4eaa255da303bafe05a177e97febd271e0e67f8df763062779fe67e` |
| `MigrationBench_main_LaTeX_clean.zip` | 14 项；只含 main.tex、bib/class/bst 和 7 张主图，不含 SI/R2R TeX 或原始图表数据 | `55b5704ba4a6e773488dade82e1052f157f4a0b439e7bca8730ecb90efe1ed51` |

ZIP 内 `main.tex` 为 45,812 B，SHA-256 `0028fb5178b10666efc1c4e052880914b888aa00923d99bdc090ccecf169156d`。GitHub 对照采用三个明确快照：

| GitHub ref | tip / 时间 | 稿件源身份 |
|---|---|---|
| `main` | `817acf22af7374bcb2a79d123d54ad77a7411c1c` / 2026-09-29 | `paper/revision1/sn-article-final.tex` blob `933190eb6b4ffc6a5f6e356d11c9dd78923b9a46`（65,131 B）；比 11 月快照早，不能当最新修订源。 |
| `integrate/final-paper-20261106` | `65fa4fc066122c4f1e46879eed0f0b9e2418d3ad` / 2026-11-08；提交说明标注 Overleaf 2026-11-07 同步 | `paper/revision1/sn-article-final.tex` blob `3c70316d47077219fa75e725e5cb15ac37394769`；SI `4f8320f5f8690f6d5fe5d5e4b4b366c9dad51a01`；R2R `3bf72480818bc4f7dcbe2d4ae44416c28323bbec`。 |
| `release/revision1-submitted-20261110` | `f924ca6c11236c4b75469e290a0abff884c56c02` / 2026-11-11 | 提交版源：main blob `b57263ce9c281fd8a9e86625871a176f6fbc7799`（46,457 B）；SI `7804a5868c1d94aec973235b0fe63158befe0adb`（33,697 B）；R2R `5d19c19d6e1e58fe999f7720e453d5eda1c7bec1`（42,705 B）。 |

ZIP 中 7 张主图 PDF 与提交版 release tree 的 Git blob 完全一致（路径均为 `paper/revision1/figures/main/<filename>`）：

| 图 | ZIP 文件名 | Git blob OID |
|---|---|---|
| Fig.1 | `fig1_schematic_new.pdf` | `6158429597b1caac0a00371a757681c7c5f2b26b` |
| Fig.2 | `fig2_energy_force_train_test_separated_palette_new.pdf` | `039fe80e8df8a25e7bb62996bc11b41f76e44d01` |
| Fig.3 | `fig3_md_six_panel_palette_new.pdf` | `69be0d10ff70e350cdb85d12198f80e780c4a474` |
| Fig.4 | `fig4_NEB_new.pdf` | `a91bd5c585c4a77ebbf79f497c1c4a7f014211bc` |
| Fig.5 | `fig5_NEB_main_figure_new.pdf` | `f961557a1265d54f41b3d6321f5e9a627ad5f9db` |
| Fig.6 | `fig6_umap_density_full_campaign_15models_1x4_v2.pdf` | `462578de3b10ebdcf1c99e8a5892d7c36cecedfe` |
| Fig.7 | `fig7_shap_top20_bar_transposed_seed_heatmap_new.pdf` | `af47d8a755b32d7b021da831e063f9079ebd4385` |

因此图文件可逐项连到提交版仓库；ZIP 的 clean main.tex 与提交源不是字节相同文件（clean 版去掉了 reviewer 标记宏并展开部分 SI 交叉引用），本地 PDF 也没有随附构建清单，故不能声称这三份 PDF 是由这些确切字节的 TeX 编译所得。直接 Overleaf 页面未访问；“最新 Overleaf”仅指 GitHub 提交说明明确标注的 11-07 同步快照。

## 主要 claim—证据核对

| 主题 / 稿件位置 | 附件中的主张及证据链 | 对齐状态与限制 |
|---|---|---|
| 固定路径 vs 自松弛 NEB；main Fig.4–5（main PDF pp.8–9），SI §§1.1、2.2 / Tables S3–S5（SI pp.1–2、6–7），R2R A1（R2R pp.1–2） | 文中区分 supplied-geometry 校准与模型驱动的路径优化；canonical 81 原子、10 image 的参考峰值为 1.52 / 0 / 1.84 eV。图注也说明自松弛低峰值不能单独证明物理过渡势垒。 | **关键科学状态冲突。** 提交版 `release/revision1_submitted_20261110/KNOWN_ISSUES.md` issue 1 记载三个 81 原子 QE NEB 的最大 image force 为 0.38–3.2 eV/Å，高于 0.03 eV/Å 门槛；账本标为 `scf_warning=TRUE`、`EXCLUDE_FROM_PUBLICATION`，但稿件仍把 1.52 eV 等作为 canonical DFT 参照。当前只宜称计算路径上的诊断比较，不宜表述为已收敛 DFT MEP 或已验证的物理势垒/速率。必须保留每张路径的坐标、能量/力、QE 输入输出、SCF/力收敛状态、路径类型和端点定义。 |
| A2/C10 分组与去重；main Methods（main PDF p.14），SI §1.4 / Table S2 / Fig.S2（SI pp.4–5），R2R A2/C10（R2R pp.2–3、9） | 记录按完整源轨迹分组以防相邻帧泄漏；25 张 legacy migration image 的 SOAP/RMSD 审核阈值分别为 1e-4 和 0.05 Å，最小相容 Cr-local RMSD 2.06 Å。G1 的 FT-600K 多 seed 固定图误差为 0.47±0.12 eV；稿件声明 G1 与 canonical 1.52 eV 路径是独立数据集。提交 release 中 Table S2 与 Fig.S2 复现登记为 PASS。 | **数值和数据集区分有文档支持，但图/图注不一致。** 对 SI PDF p.5 进行了页面图像核查：Fig.S2 当前画的是 grouped 条件的 canonical in-gap fixed-path 误差及去重柱图；图注却称左图为 legacy G1、描述 0.16 eV 旧基准。仓库 `KNOWN_ISSUES.md` issue 3 也记录此错配，故不能用当前 Fig.S2 图像支持其 G1 图注。25 张 legacy image 审核范围也不能泛化为所有 canonical NEB image。0.47±0.12 是 G1 控制结果，不是 canonical barrier。 |
| C8 力标签；Methods（main p.15），SI §1.3（SI p.4），R2R C8 / Table R6（R2R p.7） | 声称审计 9,163 条训练记录；9 条全零力都属于孤立原子能量参考，NEB 派生全零力为 0。提交 release 的 Table R6 登记为 PASS。 | 文字与响应表一致；保留逐帧 manifest、源表/校验和、审计脚本及 exporter 版本，才能支撑“无伪造 NEB 零力进入模型谱系”，仅有汇总表不够。 |
| MD 稳定性；main Fig.3（main p.6），SI §4 / Table S8（SI p.12），R2R C9 / Table R7（R2R p.8） | 15 个模型条件从 200 ps 相同状态续跑至目标 1 ns；正文强调轨迹生存不等于扩散系数或平衡输运。三种零样本 checkpoint 到 1 ns，部分 FT 和全部 scratch 在约 200–485 ps 停止。 | **停止状态标签不一致。** `KNOWN_ISSUES.md` issue 2 记载 11/15 为 `SCIENTIFIC_FAILURE`、foundation 行为 `TARGET_REACHED_PROVISIONAL`；SI S8 却写 “Target reached”，R2R Table R7 与 S8 的停止标签也不一致。报告时应保留实际轨迹时钟，同时使用审计状态（provisional/failure），不能把达到时长写成已验证科学成功。 |
| Fig.6 表征 / Fig.7 SHAP；main pp.11–12，SI §§5–6（SI pp.13–15），R2R C11–C13（R2R pp.9–11） | Fig.6 使用 927 帧、15 条 MD 条件；正文/图注明确 silhouette 是投影空间或描述性输出特征指标。Fig.7 对共同 267 帧、12 个 model-seed 列解释每原子绝对能量误差；图注明确不是因果或 out-of-sample 解释。 | Fig.6 复现登记 `PASS_KNOWN_ISSUE`：stored embeddings 可像素一致重绘，但生成它们的环境未记录，参考环境无法复算（issue 16）。C13 的 Table R11 登记 PASS；但 issue 22 说 2026-11-10 rerun CSV 未保存，`maximum difference = 0` 只由 JSON 检查记录支持。保留 feature 输入 SHA、逐帧对齐、embedding 坐标及环境版本、SHAP 输入/误差、seed/fold 与 rerun 输出。 |

## 已登记的提交版异常及处理建议

`release/revision1_submitted_20261110/REPRODUCIBILITY.tsv` 共 36 项：32 PASS、3 PASS_KNOWN_ISSUE、1 EXCEPTION（GraphicalAbstract 无源文件）。三项已知差异为 Fig.6 环境、Table S11、Table R2。`KNOWN_ISSUES.md` issue 14/15 具体记录：R2 Scratch-5% force RMSE 稿件写 342，复算均值 341.47、按显示精度应为 341；S11 Scratch-600K 五折 MAE 稿件写 1.2，原始均值 1.145、按显示精度应为 1.1。issue 17 另指出 Table S2/R2 的 MPA0 标作 zero-shot，但 `accepted_models.tsv` 指向 grouped-A2 fine-tuned checkpoint；issue 18 指出 R2 的 G1 与 RMSE 使用了不同 Scratch checkpoint 身份。这些需保留为已提交版本差异，不能在证据目录中静默覆盖。

PR #10/#12/#13 的 review candidate packages 应与 `release/revision1-submitted-20261110` 论文发布证据分开登记；其中未选中的 PR12 图包和 PR13 endpoint-47 手递包，不能直接算作主稿 Fig.1–7 或表格的支持证据。对所有 revision 结果，建议数据库按“稿件/版本 → Figure/Table 或 response claim → 可复现 item ID → 脚本与 evidence JSON → 紧凑结果表 → 输入清单及 SHA → 原始坐标/轨迹、日志、模型 hash、参数与收敛状态”连接；固定路径与自松弛路径必须分开。WFC 清理不能删掉唯一的坐标/轨迹、能量力输出、SCF 日志、split/manifest 或模型身份来源。

（文件 SHA-256 随报告一并提供，不写入文件自身。）
