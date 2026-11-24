# REVIEW_R01 — Fixed-path NEB trajectory composite

## 0. 复核状态

- 复核标准：`/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_fixed_path_trajectory_figure_20261023/visual_review/FROZEN_VISUAL_ACCEPTANCE_CRITERIA.md`
- 候选合成图：`/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_fixed_path_trajectory_figure_20261023/integration/generated/fixed_path_trajectory_composite.png`
- 上方 Blender：`/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_fixed_path_trajectory_figure_20261023/blender/triptych_top.png`
- 下方 Python：`/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_fixed_path_trajectory_figure_20261023/python/generated/fixed_path_lower_panels.png`
- 复核方式：只读图像检查 + 读取已有 manifest/QA/interface 证据；未修改数据、脚本或候选输出。
- 候选图尺寸：4200 × 3255 px，PNG，600 dpi。
- 上方尺寸：4200 × 1750 px，PNG，72 dpi；下方尺寸：5869 × 2103 px，PNG，600 dpi。
- `QA_REPORT.json` 状态：`PROVISIONAL_DIAGNOSTIC_CANDIDATE`，`figure_accepted=false`，18 项 pass、14 项 blocked。
- `INTEGRATION_REPORT.json` 状态：`PROVISIONAL_UNREVIEWED_BLENDER_CANDIDATE`。

## 1. 总结决定

**正式结论：FAIL — 不通过视觉验收；不批准为正文最终图或 publication final。**

已经通过的部分不能抵消阻断项。候选图确实保持了左至右 `within gap → gap2QL → withinQL`，并且显式写出 `DFT/QE — SCF warning; diagnostic`、failure `×` 和 withheld seed band；但以下硬问题仍存在：

1. 上下三列没有可证明的严格 x 对齐。集成脚本只是把下方整张 PNG 缩放后直接粘贴，没有共享 panel bounding boxes 或 alignment metadata；下方 axes 的 `left/right/wspace` 内边距与上方 1/3 triptych 列不相同。
2. 上方真实 image 计数在 `scene_manifest.json` 中为每列 10，但 raster 中主要只看得到端点 `0`/`9` 和局部重叠 ghost；在正文尺寸下无法逐一恢复 image 1–8。
3. ghost 的透明度没有在图例或 caption 中明确写成“真实 path-image overlay，而非 uncertainty”；视觉上仍可能被误读为 uncertainty/trajectory band。
4. `SCF warning; diagnostic` 文字被放入每个 axes 内并压在上方曲线附近；这既遮挡曲线，也违反 Plot Atlas 对 plot 内自由说明文字的限制。
5. 下方绘图源码使用 `fontsize=6.4–8.7` 的 composite annotation/legend 字号；在候选 600 dpi、约 7 英寸宽的合成输出中，顶部小标题、subtitle、image label、状态文字不满足冻结的正文 11–12 pt 可读性门槛。
6. DFT 三条 reference 仍为 `COMPLETED_BUT_SCF_WARNING`；`FT-600K`/`FT-MultiT` 合法训练-seed band 仍被 withholding，且 fixed profile 存在 runtime fallback/provenance block。

## 2. VIS-01 到 VIS-16 逐条结果

状态只使用 `PASS`、`FAIL`、`PENDING`。`PENDING` 不是通过；它表示当前 raster/证据不足以安全判定，必须补证据后复核。

| 状态 | 编号 | 复核结果与确切证据 | 修改建议 |
|---|---|---|---|
| **FAIL** | VIS-01 | **顺序正确，但严格上下对齐失败。** 合成图标题顺序是 `within gap`、`gap2QL`、`withinQL`；然而 `/integration/compose_top_bottom.py:81–88` 仅将 5869 px 宽的下方图缩放到 4200 px 后粘贴，没有共享三列 x-boundary。下方 Python 源码还设置 `left=0.055, right=0.995, wspace=0.26`（`build_fixed_path_figure.py:1108`），与上方整幅 triptych 的等分列不是同一组 bounding boxes。 | 用同一份三列 panel geometry 重新合成；显式记录每列 top/bottom 的 x0、x1、center，逐列误差设为 0 或给出可接受像素阈值，并在 QA 中检查。 |
| **PASS** | VIS-02 | 三列上方 subtitle 与下方 profile label 一致：`within gap / in-gap diffusion`、`gap2QL / gap→QL`、`withinQL / within-QL`。`scene_manifest.json` 与 `fixed_path_long.tsv` 的 path id 也对应，未见上下错列。 | 保留映射；修复 VIS-01 后重新确认上下 panel bounding box。 |
| **PENDING** | VIS-03 | `scene_manifest.json` 对每条 path 记录 `natoms=81`、`image_count=10`、`cr_spheres_by_image=0…9`、`trajectory_polyline_points=10`，且 source hash match；但 raster 中 within gap/gap2QL 主要可见端点和重叠 ghost，不能在正文尺寸逐一辨识 10 个真实 image。 | 增加低干扰的 image 0–9 编号/小 marker 或少量真实 snapshot inset，使 10 个 image 可恢复；保持所有坐标来自现有 manifest，不添加插值构型。 |
| **PASS** | VIS-04 | 三个上方 panel 都显示 `image 0 → 9`；Cr 的箭头/端点环可读，方向没有依赖颜色猜测。 | 保留；若重排上方布局，不能删除 image direction。 |
| **FAIL** | VIS-05 | 橙色半透明 Cr ghost trail 可见，但图中没有明确说明 alpha 只表示真实 path-image overlay；legend 只有 `mean ± 1 sample SD withheld`，没有 ghost 语义，读者可能把 ghost 误读成 uncertainty。 | 在 caption/manifest 中加入明确 ghost 定义；如需图内提示，只用短标签 `real path-image overlay; not uncertainty`，不得新增自由说明段落。 |
| **PASS** | VIS-06 | 三列采用相同的晶体视角语言和相近比例；`scene_manifest.json` 记录了各 path 的正交相机、scale 与 MIC unwrap，未观察到结构被旋转成不可比较的方向。 | 保留同一 camera orientation/scale；任何 focus shift 必须继续登记。 |
| **PASS** | VIS-07 | 上方无 PES surface、energy height、人工山峰或 synthetic landscape；只是 ghosted lattice + Cr trajectory。 | 保留；不得引入参考图中的 PES。 |
| **PASS** | VIS-08 | 下方点位按 10 个 retained image 直接连接；源码 `build_fixed_path_figure.py` 绘制的是直接 line/marker，未见 spline 或额外中间点。 | 保留直线连接；不要为美观加入 smoothing。 |
| **PASS** | VIS-09 | 视觉上使用四类项目颜色：Foundation 红、From-scratch 橙、FT-600K 深蓝、FT-MultiT 浅蓝；`FIG168_MODEL_STYLE.json`/figure manifest 提供了 palette 来源。 | 保留并在最终 manifest 锁定 exact hex；不要改成近似 generic palette。 |
| **PASS** | VIS-10 | legend 明确写 `DFT/QE — SCF warning; diagnostic`，黑/石墨 reference 与模型族颜色分离；caption 也没有把它称为 unconditional ground truth。 | 保留该警示；不要把 label 简化为 `DFT ground truth`。 |
| **PASS** | VIS-11 | legend 同时显示 `training seed 123/234/345` 的圆/方/三角，以及 `Foundation checkpoint variant (not seed)` 的菱形，语义没有在视觉上混用。 | 保留；后续只允许 registry 证实的 seed 使用 seed marker。 |
| **FAIL** | VIS-12 | failure 有 `×`，但 dashed line 的 legend 项写成 `mean ± 1 sample SD withheld`，没有清楚解释 dashed curve 是 nonconverged/diagnostic/status 还是 SD 语法；diagnostic reference 又通过 axes 内文字表达，状态通道不够单义。 | 将 `uncertainty withheld` 与 `nonconverged/diagnostic` 分成两个独立 legend entries；为 status 使用明确 dash pattern + marker，避免 dashed line 被读成 uncertainty。 |
| **PASS** | VIS-13 | 每个下方 panel 顶部可见 `× FT-MultiT 123: MODEL_LOAD`；legend 另有 `failure / no numeric profile`，没有把 failure 画成 0 或假曲线。 | 保留 failure `×`；修复布局时不要把它裁掉或缩到不可读。 |
| **FAIL** | VIS-14 | legend 本身主要位于 axes 上方，但每个 axes 内右上角的 `SCF warning; diagnostic` 文本压在曲线区域上方（源码 `build_fixed_path_figure.py:1036–1040`）；左/中/右均可见文字与曲线空间冲突。底部还放置了方法性说明长句。 | 移除 axes 内自由说明文字；将 DFT warning 放入简洁 legend/caption/status strip；将 `ΔE_i=...`、image/observable 说明移到 caption 或图外审计记录。 |
| **PENDING** | VIS-15 | 下方可见 `s (0–1)`、`ΔE (eV)`，且底部说明写出 `ΔE_i = E_i − E_0`；但 composite 轴没有完整写出 `normalized reaction coordinate`，10 image 语义主要依赖外部 caption/manifest。 | 在不增加拥挤的情况下补齐轴/图注语义，并确认正文单独查看 PNG 时不需要打开外部文件才能理解 x/y。 |
| **FAIL** | VIS-16 | 4200×3255、600 dpi 不代表正文可读。源码明确使用 composite legend 6.7 pt、subtitle 8.7 pt、diagnostic 6.7 pt、failure text 6.4 pt（`build_fixed_path_figure.py:1034,1038,1049,1101,1106`）；上方 subtitle 和 `image 0 → 9` 在候选图中已经明显偏小。严格正文尺寸检查尚未通过。 | 以最终期刊物理宽度重排，不要只提高 PNG dpi；legend/annotation/tick 在最终 placement 均至少 11–12 pt，重新规划列宽、legend 行数和上下间距。 |

## 3. 数据/物理门槛（PHY/DATA）

| 状态 | 检查项 | 当前证据与结论 |
|---|---|---|
| **PASS（记录层）** | PHY-01 path identity | `scene_manifest.json` 对三条 path 记录 81 atoms、PBC TTT、endpoint consistency PASS、source SHA256 match；三列 mapping 与 frozen order 一致。 |
| **PASS** | PHY-02 image completeness | 三条 path manifest 均为 10 images；`dft_profiles.tsv` 也按 0–9 记录。视觉可辨性仍由 VIS-03/PUB-08 阻断。 |
| **PASS（数据层）** | PHY-03 geometry authenticity | `scene_manifest.json` 记录真实 extxyz 输入、MIC frame-to-frame unwrap、每 image 80 host + 1 Cr、10 trajectory points；这证明渲染接口使用真实 geometry，不证明 DFT energy 收敛。 |
| **PASS** | PHY-04 normalization | `fixed_path_long.tsv`/`dft_profiles.tsv` 使用 `relative_energy`，caption 明确 `ΔE_i=E_i−E_0`；force-error 单列。 |
| **PASS** | PHY-05 fixed-path observable | caption 和 data manifest 标明只进入 `FIXED_PATH_SINGLE_POINT`/`DFT_RETAINED_PATH`；self-relaxed/MIC_LINEAR 被排除。 |
| **PASS** | PHY-06 fixed/self/MIC separation | 没有发现 self-relaxed 或 MIC_LINEAR 曲线进入下方候选图；该 separation 在 manifest 中显式记录。 |
| **PASS** | PHY-07 finite/no fabricated values | QA 记录数值 energy finite，failure blocks 无 numeric profile；未见补零/人工填峰。 |
| **PASS** | PHY-08 force/error separation | DFT `error_eV_A` 保留在 `dft_profiles.tsv`，没有作为 ΔE band 绘出。 |
| **PASS（覆盖记录）** | PHY-09 family accounting | manifest 记录四族、target arms、excluded/failure；候选图也显示四族颜色。该项不等于四族 publication completeness。 |
| **PASS** | PHY-10 Foundation semantics | `checkpoint_variant_not_seed` 被保留；legend 明确 `not seed`。 |
| **PASS** | PHY-11 From-scratch semantics | heterogeneous branch 以离散 arm 表示，没有生成 branch SD band。 |
| **BLOCKED** | PHY-12 valid seed uncertainty | manifest/QA 明确没有 FT family 在 frozen gate 下具备三个 verified training seeds；band 被 withheld，这是正确的阻断状态。 |
| **PASS（withheld 语义）** | PHY-13 uncertainty definition | caption 明确写 sample SD band withheld，且没有把它称为 CI/DFT error；仍需保持图内 dashed status 不与 uncertainty 混淆（见 VIS-12）。 |
| **PASS** | PHY-14 no fabricated completeness | missing/load failure/nonconverged 不参与均值或补线，failure 保留。 |
| **PASS（排除记录）** | PHY-15 no historical mixing | 当前候选 manifest 标明 fixed canonical paths，未见 61-atom/history/reconstruction/47-site static endpoint 被画入三列。 |
| **BLOCKED** | PHY-16 curve-level provenance | `caption.md` 明确说明 runtime 读取 mirrored fixed `image_profile.csv` 被阻断，数值回退到 hashed `TIER1_IMAGE_PROFILES.tsv`；raw remote profile hash unavailable。这是 provenance blocker，不能用图形完整性掩盖。 |

**PHY 总状态：BLOCKED。** geometry/path identity/fixed-vs-self separation 记录较好，但 seed uncertainty gate 和曲线级 provenance gate 未通过。

## 4. DFT SCF warning 门槛（不可隐瞒）

| 状态 | 编号 | 结果 |
|---|---|---|
| **PASS** | DFT-01 | 三条 reference 的 status 在 `dft_profiles.tsv` 中为 `COMPLETED_BUT_SCF_WARNING`，图例/标题区域写 `SCF warning; diagnostic`。 |
| **BLOCKED** | DFT-02 | warning-bearing reference 不能称无条件 `DFT ground truth`；当前候选没有升级该措辞。该项仍是 publication hard block。 |
| **PASS** | DFT-03 | SCF/force-error 未被画为 energy uncertainty；seed band 也没有被伪造。 |
| **BLOCKED** | DFT-04 | 尚无三条 path 的无-warning、逐 image convergence、finite energy/force、path-threshold 和更新 registry 证据；job completion 不构成升级。 |

**DFT 总状态：BLOCKED。** 本次复核不把任何 DFT/QE 曲线升级为 ground truth。

## 5. 不确定种子门槛（不可隐瞒）

| 状态 | 编号 | 结果 |
|---|---|---|
| **PASS** | SEED-01 | training seed、checkpoint variant、heterogeneous branch、NEB seed 和 MIC initialization 在 manifest 中分列。 |
| **PASS** | SEED-02 | Foundation 以 checkpoint variant 表示，未被合成 random-seed band。 |
| **PASS** | SEED-03 | From-scratch heterogeneous branches 以离散线表示，未生成 branch SD。 |
| **BLOCKED** | SEED-04 | `FT-600K seed 123` 为 base-unproven，`FT-MultiT seed 123` 为 all-path `MODEL_LOAD` failure；三 verified training seed 的 fixed-path gate 不成立，band withheld 是必须行为。 |
| **PASS** | SEED-05 | seed presence/failure/withheld reason 在 caption/manifest 中可见；失败没有静默丢弃。 |

**SEED 总状态：BLOCKED。** 候选没有错误地画出 seed band，但因此也不能宣称完成了 seed uncertainty 综合图。

## 6. 发表门槛（PUB）

| 状态 | 编号 | 结果与定向建议 |
|---|---|---|
| **PASS（措辞层）** | PUB-01 | caption 使用 `diagnostic candidate`，区分 retained path、DFT warning、fixed model profile 和 withheld uncertainty；保留。 |
| **BLOCKED** | PUB-02 | `BLOCKED_PENDING_NUMERIC_FIXED_PATH_MATRIX` 和 PHY-16 fallback 仍存在；不能写 complete 四族 publication figure。补齐 verified fixed matrix 后再复核。 |
| **PENDING** | PUB-03 | failure × 可见，caption 说明 failure remains visible，但图/caption 尚未给出完整 failure/nonconverged/excluded 数字 ledger；补充 SI/manifest counts。 |
| **PASS（未作 unsupported claim）** | PUB-04 | 没有把 fixed/self 混作 ranking，也没有声称 A2 leakage 或 DFT ranking 已验证；保持这一措辞边界。 |
| **FAIL** | PUB-05 | 源码 composite 的 6.4–8.7 pt 字号明显低于 frozen 11–12 pt final-placement 门槛。按最终物理宽度重新排版。 |
| **FAIL** | PUB-06 | 上下 panel 的严格 alignment 未通过；axes 内诊断文字又侵入曲线空间。必须重做共享布局/legend。 |
| **FAIL** | PUB-07 | 下方有 PDF/SVG/PNG，但当前候选合成输出只有 PNG；没有与上方 Blender 合成后的 editable PDF/SVG。 |
| **FAIL** | PUB-08 | final-size、灰度/色觉和 100% 像素检查未通过；顶部小字和诊断标注在正文尺寸不可接受。 |
| **BLOCKED** | PUB-09 | runtime fixed-profile fallback 和 unavailable raw-profile hash 尚未修复/独立复核；当前 release 不能作为完整可复现实图。 |
| **BLOCKED** | PUB-10 | VIS-01/VIS-05/VIS-12/VIS-14/VIS-16、PHY-12/PHY-16、DFT-02/04、SEED-04 仍未通过；不得 final sign-off。 |

**PUB 总状态：BLOCKED。** 当前只能保留为 diagnostic candidate/visual QA candidate。

## 7. 必须定向修复后再复核

1. **先修集成几何，不改数据：** 用同一个三列布局 contract 生成 top/bottom panel bounding boxes；不要用“整张 lower PNG 缩放后粘贴”作为对齐方案。
2. **修 ghost 可读性：** 保持 scene manifest 中的 10 个真实 image 和 source hashes；只增加编号/可辨识的 path-image marker，不增加 synthetic image。
3. **删除 axes 内方法性文字：** 将 `SCF warning; diagnostic`、failure/status 和 uncertainty withheld 做成不遮挡曲线的 compact legend/status area；长说明移到 caption/SI。
4. **拆分 dashed 语义：** 明确 nonconverged/diagnostic 的 line style 与 `mean ± 1 sample SD withheld` 的语义不能共享一个 dashed sample。
5. **按正文物理尺寸重排字体：** 不能只提高 dpi；legend、subtitle、status、tick/axis 在最终 placement 达到冻结标准。
6. **完成 provenance 修复：** 处理 remote fixed `image_profile.csv` runtime read block，或完成独立可审计的 fallback revalidation，并保存每条曲线可用的 raw/profile hash。
7. **保持科学阻断：** 在三条 DFT SCF warning 未修复、三 verified training seeds 未满足 fixed-path gate 前，不得把图或标题升级为 `ground truth`、`final` 或 `publication accepted`。

## 8. 最终签字

- 视觉审查：**FAIL**
- 数据/物理：**BLOCKED**
- DFT reference：**BLOCKED — SCF warning retained**
- seed uncertainty：**BLOCKED — band correctly withheld**
- 发表门槛：**BLOCKED**
- 本次是否放宽冻结标准：**NO**
- 本次是否修改候选图/数据：**NO**
