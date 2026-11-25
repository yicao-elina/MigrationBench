# FROZEN_VISUAL_ACCEPTANCE_CRITERIA

## 0. 冻结状态

- 审计目录：`/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_fixed_path_trajectory_figure_20261023/visual_review/`
- 日期：2026-10-23
- `CRITERIA_FROZEN = YES`
- `FIGURE_ACCEPTED = NO`
- `PLOTTING_AUTHORIZED = NO`
- `DATA_MODIFIED = NO`
- `BLENDER_STARTED = NO`

本文件冻结的是“画图前验收合同”，不是对当前 DFT/NEB 科学结果的追认。任何一条必须项不通过，都不能把图标为正文最终图，也不能把诊断结果写成 ground truth。

## 1. 审计证据和边界

以下是本次只读审查实际使用的证据。科学有效性不能由文件名、目录名、图片存在或 scheduler/job 完成单独推出。

1. 参考视觉：`/Users/alina/.codex/attachments/c31191e4-e79f-47bc-bf77-d6eaca40fa6b/image-1.png`、`/Users/alina/.codex/attachments/c31191e4-e79f-47bc-bf77-d6eaca40fa6b/image-2.png`。仅用于构图、原子虚影、路径快照和曲线层次；其中的 synthetic PES/energy 不是本项目数据。
2. 数据资产：`/Users/alina/Library/CloudStorage/GoogleDrive-yicao.alina@gmail.com/My Drive/Obsidian-google/Projects/MigrationBench/MigrationBench-Data-Assets.md`。该文档规定了 FIG-168 palette、fixed/self 边界、seed 语义和数据缺口，但不是原始计算结果。
3. 三条实际 retained path geometry：
   - `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/task3_endpoint_only_neb/exploratory_energy_audit/rockfish_cr_neb_20261013/inputs/2L-Cu__neb-gap-diffuse.extxyz`
   - `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/task3_endpoint_only_neb/exploratory_energy_audit/rockfish_cr_neb_20261013/inputs/2L-Cu__neb-gap2QL.extxyz`
   - `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/task3_endpoint_only_neb/exploratory_energy_audit/rockfish_cr_neb_20261013/inputs/2L-Cu__neb-QLs.extxyz`
   当前核对到每条为 81 atoms、10 images、`pbc="T T T"`；这是几何证据，不等于 DFT 能量已收敛。
4. DFT/QE 状态：`/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/task3_endpoint_only_neb/exploratory_energy_audit/rockfish_cr_neb_20261013/rockfish_cr_qe_summary.csv`。三条记录均为 `COMPLETED_BUT_SCF_WARNING`，`scf_warning=True`。
5. canonical/paired registry：`/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/task3_endpoint_only_neb/neb_fixed_vs_self_matrix_20261019/SYSTEM_PATH_REGISTRY.tsv`、`CANONICAL_ENDPOINT_REGISTRY_r01.tsv`、`derived_r01/TIER1_RESULT_MATRIX.tsv`、`RELEASE_MANIFEST_r01.json`、`RELEASE_QA_REPORT_r01.json`。
6. fixed-path 数据缺口：`/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/task3_endpoint_only_neb/figure1_family_seed_audit_20261016/results/fixed_path_figure_status.json`、`FIXED_PATH_DATA_GAP.tsv`。该审计明确为 `BLOCKED_PENDING_NUMERIC_FIXED_PATH_MATRIX`；不能用 self-relaxed matrix 代替。
7. 模型 palette/marker：`/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/task3_endpoint_only_neb/figure1_family_seed_audit_20261016/results/FIG168_MODEL_STYLE.json`。
8. revision 边界：`/Users/alina/Project/migrationbench-paper-repro/paper/revision1/sn-article-SI.tex`、`/Users/alina/Project/migrationbench-paper-repro/paper/response_to_reviewers/reviewer_response_readiness_audit.md`。这些文件明确区分 fixed-path scoring、self-consistent MLFF-NEB、DFT warning 和未收敛 reference。
9. Plot Atlas：`/Users/alina/Documents/Codex/plot_library/docs/plot_catalog.json`、`panel_registry.json`、`examples/panels/render_steered_dynamics_nature.py`、`references/panel-contract.md`。当前没有已注册的 NEB 专用 recipe；trajectory recipe 只能提供排版语法，不能替代数据 contract。
10. 既有 Pov-ray 先例：`/Users/alina/Project/migrationbench-paper-repro/paper/response_to_reviewers/reviewer_a1_fig3_figs3_update/atom_renders_v2/` 和 `atom_renders_v4/`。只允许借鉴材质/渲染风格；不得借鉴或猜测坐标/能量。

## 2. 三列语义与不可变排序

从左到右必须为：**within gap → gap2QL → withinQL**。当前 r01 preview 的列顺序不同，不能照搬。

| 编号 | 正文显示 | canonical path id | 必须使用的 geometry | registry class |
|---|---|---|---|---|
| C-01 | 左：`within gap`（caption 可写 `within-gap`） | `cr_canonical_in_gap` | `2L-Cu__neb-gap-diffuse.extxyz` | `in-gap diffusion` |
| C-02 | 中：`gap2QL`，副标签统一为 `gap→QL` | `cr_canonical_gap_to_ql` | `2L-Cu__neb-gap2QL.extxyz` | `gap-to-QL` |
| C-03 | 右：`withinQL`（caption 可写 `within-QL`） | `cr_canonical_within_ql` | `2L-Cu__neb-QLs.extxyz` | `within-QL` |

每列上方原子图和下方曲线必须使用同一 `path_id`。不得把 static endpoint category、文件名或旧图列顺序当作 path semantics。

## 3. 第一层：视觉通过（VIS）

每一项均须逐条勾选。任何“不通过”都阻断视觉验收。

| 勾选 | 编号 | 视觉标准 | 通过判据 | 不通过判据 |
|---|---|---|---|---|
| ☐ | VIS-01 | 三列结构 | 三列等宽、顺序严格为 C-01/C-02/C-03；标题、caption、manifest 同名。 | 顺序错误、标题含义含混或上下图错列。 |
| ☐ | VIS-02 | 上下语义 | 每列上方是真实 fixed-path 原子迁移，下方是同一 path 的 fixed-path profile。 | 上方和下方来自不同 path/observable。 |
| ☐ | VIS-03 | 真实 atom ghost | 10 个真实 image 的坐标全部可见于 snapshot、ghost 或二者组合；Cr 高亮，host atoms 低 alpha。 | 只有抽象曲线、手工原子或未追溯的中间构型。 |
| ☐ | VIS-04 | 迁移方向 | image 0→9（或 1→10，全文只能选一种）和 endpoint/箭头清楚可辨。 | 方向依赖颜色猜测，或 image 编号不一致。 |
| ☐ | VIS-05 | ghost 语义 | ghost 的 alpha/层级只表达 path overlay/图像顺序，三列规则一致；不表达 uncertainty、DFT error 或 energy。 | 透明度被读成置信度/误差，或三列规则不同。 |
| ☐ | VIS-06 | 相机与尺度 | 三列相机方位、球半径、晶胞裁剪和关键 host layers 可比较；周期 unwrap 有记录。 | 相机/尺度改变了迁移方向或隐藏 endpoint。 |
| ☐ | VIS-07 | 禁止 fake PES | 原子图无 synthetic PES surface、伪造 energy height、人工过垒山峰或参考图手工轨迹。 | 使用 `pes_operator_schematic_v5` 或参考图 PES 作为真实数据。 |
| ☐ | VIS-08 | 曲线数据形态 | 真实 image 点按顺序连接；不使用会创造额外峰/谷的平滑 spline。 | 视觉曲线出现未计算的 image 或 artificial extrema。 |
| ☐ | VIS-09 | family 颜色 | Foundation `#CF4520`、From-scratch `#FF9F1B`、FT-600K `#002E71`、FT-MultiT `#67ACE5`。 | 自定义颜色、DFT 被涂成 family 色或颜色语义错位。 |
| ☐ | VIS-10 | reference 颜色/标签 | DFT/QE 用黑/石墨色，当前标签必须含 `SCF warning; diagnostic`。 | 黑线被标成无条件 ground truth。 |
| ☐ | VIS-11 | seed/variant marker | 只有 registry 确认的 training seed 才使用 123=圆、234=方、345=三角；Foundation 变体必须标 variant。 | Foundation variant 被写成 training seed，或 marker 语义不明。 |
| ☐ | VIS-12 | 状态非颜色编码 | accepted/diagnostic/nonconverged/failure 至少用线型、marker、边框之一额外区分。 | 只靠颜色区分状态，灰度下无法读。 |
| ☐ | VIS-13 | 失败可见 | `FAILURE_MODE` 无连接线并有 `×`/空心 marker；missing 不画假线；nonconverged 有状态线型。 | 失败变 0、静默删除、补线或被写成 converged。 |
| ☐ | VIS-14 | 图例和叙事 | family、reference、seed/variant、status、mean±SD 分组清楚，不遮挡原子/曲线。 | legend 覆盖数据或无法区分四族和状态。 |
| ☐ | VIS-15 | 轴语义可读 | x=`normalized reaction coordinate, s (0–1)`；y=`relative energy, ΔE (eV)`；10 image 可恢复。 | 写 time/MD step、轴单位缺失或 image 对应不可恢复。 |
| ☐ | VIS-16 | 无裁切/可读性 | endpoint、迁移 Cr、关键 host layer、标题、legend、刻度均无裁切；最终尺寸下可读。 | 超大画布可读但正文尺寸不可读。 |

## 4. 第二层：数据与物理通过（PHY/DATA）

这一级优先于美观。任何“不能确认”的项目都视为不通过，不得用文件名补足。

| 勾选 | 编号 | 数据/物理标准 | 通过判据 | 不通过判据 |
|---|---|---|---|---|
| ☐ | PHY-01 | path identity | 三列 path id、源文件 hash、cell/PBC、81 atoms、atom order、endpoint hash 与 registry 一致。 | 只凭文件名或目录名映射。 |
| ☐ | PHY-02 | image completeness | 每条 path 确认完整 10 images；无静默重排、重复、删除或 resampling。 | image 数不全或人工补 image。 |
| ☐ | PHY-03 | geometry authenticity | 所有 ghost/snapshot 来自真实 retained path；MIC/unwrap 算法和结果有记录。 | 未记录 unwrap 或使用 synthetic 坐标。 |
| ☐ | PHY-04 | energy normalization | 每条曲线明确 `ΔE_i=E_i−E_0`、单位 eV 和 zero；不跨模型比较未归一化绝对能量。 | zero/单位混用或把 absolute energy 当可比。 |
| ☐ | PHY-05 | fixed-path observable | 模型只在同一 retained 10-image path 上 single-point score；无 MLIP optimizer 改几何。 | 把 self-relaxed geometry/energy 回填 fixed 图。 |
| ☐ | PHY-06 | fixed/self/MIC 分离 | `FIXED_PATH_SINGLE_POINT`、`SELF_RELAXED_ENDPOINT_ONLY_NEB`、`MIC_LINEAR` 各自有独立 observable/行/manifest。 | 三者叠图、混合 ranking 或把 MIC 当 seed uncertainty。 |
| ☐ | PHY-07 | finite values | 进入曲线或统计的每个 energy finite，且逐点来源可回溯。 | NaN/Inf、缺失补零、人工填峰谷。 |
| ☐ | PHY-08 | force/error separation | `eV/Å` force error、SCF residual、path threshold residual 不进入 energy uncertainty band。 | 把 force/error 画成 ΔE band。 |
| ☐ | PHY-09 | family coverage | 四族实际 arm、可用数、失败数、excluded 数均可从 registry/manifest 复核。 | 部分覆盖称为四族完整综合图。 |
| ☐ | PHY-10 | Foundation semantics | Foundation 作为 checkpoint/model variants，不称三个 random training seeds，不画 seed band。 | variant 被伪装成 seed ensemble。 |
| ☐ | PHY-11 | From-scratch semantics | 当前 heterogeneous branches 只作离散分支；除非同质 lineage 另有证据，不画 training-seed band。 | 把异质分支的 spread 当 seed uncertainty。 |
| ☐ | PHY-12 | valid seed uncertainty | 仅 FT-600K/FT-MultiT 中同 path、同 observable、同 image count、同 protocol、同 DFT reference 的 verified training seeds可计算 mean±1 SD。 | 跨 path/fixed-self/protocol/缺失行合并。 |
| ☐ | PHY-13 | uncertainty definition | caption/legend 写 `sample SD across verified training seeds`；不写 CI、SEM、DFT error 或 replica uncertainty。 | uncertainty 来源含糊或与 NEB seed=42 混淆。 |
| ☐ | PHY-14 | no fabricated completeness | missing、load failure、nonconverged、reconstruction 不参与均值/SD，不插值、不补零。 | 用视觉完整性掩盖数据缺口。 |
| ☐ | PHY-15 | no historical mixing | 不混 61-atom、历史 81 chain、endpoint-only reconstruction、探索性路径或 47-site static endpoint package。 | 仅因输出存在就纳入。 |
| ☐ | PHY-16 | curve-level provenance | 每条线/带记录 path、model id、SHA256、family、seed/variant semantics、NEB seed、observable、source/output hash、处理脚本、状态。 | 只有 family 名或手工 legend。 |

### 4.1 DFT SCF warning：独立的不可隐瞒硬门槛

| 勾选 | 编号 | 硬门槛 | 通过条件 | 当前状态/失败判据 |
|---|---|---|---|---|
| ☐ | DFT-01 | reference 状态透明 | 三条 reference 在图例、caption 或 panel status 中明确写 `COMPLETED_BUT_SCF_WARNING`/`SCF warning; diagnostic`。 | 当前三条均有 warning；隐藏、缩小到不可读或改写为 converged 均失败。 |
| ☐ | DFT-02 | ground-truth 用词受限 | 在无 SCF warning、逐 image 收敛证据、有限 energy/force、endpoint/image identity 和 path-threshold 证据前，不使用无条件 `DFT ground truth`。 | 当前不能称无条件 ground truth；写该词即整图不通过。 |
| ☐ | DFT-03 | error 不冒充 energy | SCF warning/residual 只能作为状态/诊断信息，不能作为 energy uncertainty band。 | 当前 warning-bearing reference 不能被画成“精确黑色 ground-truth 曲线”。 |
| ☐ | DFT-04 | 修复升级条件 | 只有新 release 提供逐 image 收敛证据、finite values、输入/输出 hash 和更新 registry 后，才可申请升级 reference 标签。 | 仅 `JOB DONE`、scheduler 完成或图片生成成功不构成升级。 |

### 4.2 不确定种子语义：独立的不可隐瞒硬门槛

| 勾选 | 编号 | 硬门槛 | 通过条件 | 当前状态/失败判据 |
|---|---|---|---|---|
| ☐ | SEED-01 | seed 类型分开 | training seed、checkpoint variant、heterogeneous branch、NEB seed、MIC initialization 各有独立字段。 | 任意两种语义合成一个“seed uncertainty”即失败。 |
| ☐ | SEED-02 | Foundation 不伪装 seed | Foundation A/B/C 或其他 variants 逐条显示并明确 `checkpoint variant, not training seed`；不画训练 seed band。 | 当前 Foundation 不能按 3 个 random seeds 汇总。 |
| ☐ | SEED-03 | From-scratch 不伪装同质 ensemble | 123/234/345 若只是 heterogeneous branches，保留 discrete branch 语义；除非另有 lineage/protocol 证据。 | 当前不得把 branch spread 写成 training-seed SD。 |
| ☐ | SEED-04 | FT seed band 合法性 | FT-600K/FT-MultiT 仅在 3 个 verified training seeds 和 PHY-12 全通过时显示 mean±1 SD。 | 缺 specialist fixed-path matrix、arm 不齐或状态不一致时，必须阻断 band。 |
| ☐ | SEED-05 | seed 数和失败可见 | band 旁注明实际 seed 数、排除项和原因；失败 seed 不能静默丢弃。 | 只有漂亮 band、没有 n/排除信息即失败。 |

## 5. 第三层：发表门槛（PUB）

发表门槛不是“画面好看”门槛；它要求科学措辞、可复现性和导出质量同时成立。

| 勾选 | 编号 | 发表标准 | 通过判据 | 不通过判据 |
|---|---|---|---|---|
| ☐ | PUB-01 | claim 与证据一致 | caption 清楚区分 retained geometry、DFT/QE diagnostic reference、model fixed-path profile、self-relaxed proxy 和 uncertainty。 | 把 diagnostic/reference warning 或 model-native self barrier 写成验证过的 DFT truth。 |
| ☐ | PUB-02 | complete claim 有证据 | 只有逐 path、逐 family、逐 image、逐 arm 的 manifest 完整时，才可写 complete/综合图。 | 目前固定路径专用矩阵仍是 `BLOCKED_PENDING_NUMERIC_FIXED_PATH_MATRIX`。 |
| ☐ | PUB-03 | failure ledger 随图 | main figure/ caption/ SI 记录失败、nonconverged、load failure、excluded 和 missing 数量及处理。 | 只显示成功线或让读者猜删了什么。 |
| ☐ | PUB-04 | no unsupported ranking | fixed 与 self 不混作一个 barrier ranking；不从该图宣称 A2 leakage 已解决或 DFT ranking 已验证。 | 图形视觉被用来补足未完成审计。 |
| ☐ | PUB-05 | typography | 优先 Arial；最终正文放置尺寸下 axis/legend/ticks 至少约 11–12 pt，panel/column label 清楚加粗；fallback 已记录。 | 仅在大画布可读，正文尺寸不可读。 |
| ☐ | PUB-06 | layout | 三列等宽、上下对齐、legend 不遮挡；共享/独立 y-limit 的理由明确且不制造视觉比较误导。 | 不同尺度未标注，或小势垒被压到不可读。 |
| ☐ | PUB-07 | export | PDF、SVG、300–600 dpi PNG 均生成；字体/透明层正确；三种输出视觉一致且有 hash。 | 只有截图、裁切、字体丢失或格式不一致。 |
| ☐ | PUB-08 | final-size QA | 正文尺寸、100% 像素、灰度/色觉预览均无裁切、重叠、aliasing 误导或状态丢失。 | 只做单一分辨率或单一颜色模式检查。 |
| ☐ | PUB-09 | reproducible release | figure source、data manifest、环境/脚本版本、input/output hashes 与 caption 版本一同保存。 | 只能重现图片外观，不能重现每条曲线/ghost。 |
| ☐ | PUB-10 | final sign-off | VIS、PHY/DATA、DFT、SEED、PUB 全部勾选；视觉审查员记录日期、版本、证据路径和输出 hash。 | 任一硬门槛 pending/blocked，整图不得标 final。 |

## 6. 当前冻结结论

### 6.0 R04 哑光渲染增补

用户新增的 Blender 高级哑光质感要求已作为独立、可复查的增补标准冻结于 `visual_review/MATTE_BLENDER_STYLE_ADDENDUM_R04.md`。该文件只增加材质、光照、体积感、相机、抗锯齿、body-width 与独立视觉审查要求，不改变本文件既有科学和数据门槛；任一 MAT-01—MAT-24 未通过，都不能称为视觉验收通过。

### 已冻结的标准

1. 三列严格为 **within gap → gap2QL → withinQL**。
2. 上方必须是三条真实 81-atom、10-image retained fixed path 的原子迁移与 ghost；Pov-ray/Blender 只负责渲染，不负责创造科学数据。
3. 下方必须按 `s` 和 `ΔE (eV)` 展示真实 fixed-path reference 与四类模型，且 fixed/self/MIC 分离。
4. DFT SCF warning 必须显式显示，当前不得使用无条件 `ground truth`。
5. Foundation variant、From-scratch heterogeneous branch、FT training seed、NEB seed 和 MIC initialization 的语义必须分开；未经 gate 的 mean±SD 禁止出现。
6. failure、missing、nonconverged、reconstruction 和 provenance partial 状态必须可见，不得用补线或美术处理掩盖。

### 当前不可放宽的阻断项

- 三条 DFT/QE reference 当前均为 `COMPLETED_BUT_SCF_WARNING`；因此正文图目前只能使用 diagnostic reference 标签。
- fixed-path family/seed numeric matrix 当前被现有审计明确阻断；不能声称已经具备完整四族 fixed-path + 合法 seed uncertainty。
- Foundation/From-scratch 当前不能按普通同质 training-seed ensemble 汇总。

**最终决定：标准已冻结；图未制作、未验收、未批准发表。** 后续执行者若失败，必须返回对应编号（例如 `DFT-02`、`SEED-04`、`PHY-08` 或 `VIS-03`）、实际证据路径、失败原因和定向修改方案；不得自行放宽本标准。
