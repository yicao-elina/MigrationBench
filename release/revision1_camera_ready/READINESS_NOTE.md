# 投稿前尚需闭合的证据

正文和回复已经完成实质重写；五份 PDF 编译通过，引用自动对齐。下面列明已修正的问题和仍未闭合的证据。本文已按诊断研究收缩结论，没有伪造缺失的收敛或因果证据。

| 问题 | 当前证据与处理 | 保留边界或闭合条件 |
|---|---|---|
| Canonical migration 的 DFT 参考仍有 SCF 警告 | 81 原子、10 图像主图引用 1.5243 / 0 / 1.8449 eV；源记录状态为 COMPLETED_BUT_SCF_WARNING。正文和图注明确是 image-set diagnostics，不给激活能或速率。 | 若要保留“定量势垒/动力学准确性”主张，应取得同一 Hamiltonian 下电子与 NEB 都收敛的参考、原始能量和力记录；重算 Fig. 4–5 的全部相应数值。不能仅从曲线/PDF 提取值认定 DFT 势垒已验证。 |
| Fig. 5 的数值与状态错误已修正 | 9 个 Scratch-600K self 记录逐一匹配 model SHA、seed 和 path。s123 × within-QL 从错误 0 修正为 2.3507668938545976 eV；该记录与 s234 × in-gap 改为 nonconverged。修图标出全部 5 个未收敛点，原稿和 correction audit 保留。 | 这是源表错误的修正。其余 optimizer-converged 标签也不等于 DFT 路径已验证；定量动力学仍需先闭合 DFT 参考。 |
| Fig. 4 与 Fig. 5 并非一套完全配对的 checkpoint | Fig. 4 是旧四组 profile ensemble；Fig. 5 是后续 15 行 seed matrix。图注与 SI 给出成员数，未把二者当作完整配对干预实验。 | 若需要严密比较同一 checkpoint 的 fixed/self 变化，应固定全部模型 hash、端点、参考 images 和收敛标准，统一重算。 |
| 审计范围不能扩展到未检查的数据 | 25-image overlap 检查针对 legacy G/Q 图像；9,163-frame force audit 针对其列明的数据文件。正文明确没有声称这些检查覆盖全部后续路径与新训练文件。 | 对最终要作为独立验证的 canonical 图像及所有最终训练文件补做 hash-bound overlap/force-label 审计。 |
| Fig. 6 不支持原生 latent mechanism 的解释 | 当前 927 帧原始空间 silhouette = 0.06266，标准化后 −0.05919；t-SNE 图中为 0.117。SOAP PCA 原来是每条轨迹分别拟合，且能量零点不同。原生机制和单一因果解释已删除。 | 若保留“学到何种内部物理”的主张，需共同构型、共同可比坐标或原生 equivariant hidden states 的新分析与干预；当前描述性用途不需要该强主张。 |
| 主图 artwork 仍待补 panel 字母 | 除 Fig. 5 的上述必要修正外，其余六张正文 PDF 字节未改变。caption 使用最终 a/b/c… 编号；Fig. 1 的现有 projection “A”需与 caption 中 b 统一，Fig. 2 的 A–D 可统一为 a–d。少数原稿图内标签在最终缩放后较小。 | 按 FIGURE_PANEL_MAP.md 完成最终图内编号，并按目标期刊版心检查图内字号。 |

本次已经完成、无需再凭旧稿猜测的事项：

- 三个 foundation checkpoint 达到 1 ns clock；六个 FT 仅一个达到；scratch 的实际终点为 200.4–204.4 ps。这里是从共同 200 ps 状态继续到目标 clock，不是每个模型独立生成完整 1 ns。
- 原 0.41 eV 结果撤回；0.336 eV 与 0.34 eV 是同一个 legacy G1 值，1.5243 eV 是另一套 canonical 图像。
- Scratch-5% 实际是 89 帧，未再称为与 FT 等量。全量 600 K scratch/FT 都使用 1,786 target training frames；架构差异仍被保留为限制。
- 当前 SHAP 的 12 行 metrics、1,628 行 family features、4,884 行 seed features、3,204 行 aligned frames 已实际重跑，数值最大差为 0。
- 正文蓝字仅标直接回应句，13 个 reviewer index 全部覆盖；回复逐条给出自动解析的正文/SI 位置。

当前稿件可以用于作者审阅和收缩主张后的修订准备；不能据此宣称未收敛参考已经变成可发表的定量动力学 ground truth。
