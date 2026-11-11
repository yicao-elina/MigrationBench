# MATTE_BLENDER_STYLE_ADDENDUM_R04

## 0. 作用与边界

本文件是 `FROZEN_VISUAL_ACCEPTANCE_CRITERIA.md` 的 R04 additive addendum，不覆盖、不放宽 R01 已冻结的几何、数据、DFT、seed 和发表门槛。

用户提供的两张参考图只用于提炼视觉语法：高级哑光材质、柔和受控光照、原子体积感、清晰标注和路径快照层次。参考图中的示意坐标、能量曲线、PES、颜色所代表的化学体系和构图数据均不得复制到本项目。

`MATTE_VISUAL_GATE = PASS` 的必要条件是：MAT-01—MAT-24 全部 PASS，并由独立视觉审查员在 `REVIEW_Rxx_MATTE.md` 中逐项给出证据。Blender 的脚本 QA、渲染成功或文件存在都不能替代独立视觉验收。

## 1. 风格定义

目标是“科研图中的高端 Blender 哑光三维原子渲染”：表面有连续柔和的明暗渐变和少量受控高光，能看出圆润体积、前后层次和接触关系；整体克制、干净、可印刷，不是扁平色块、玻璃、金属球、霓虹塑料或照片级杂乱反射。

| 编号 | 冻结标准 | PASS 判据 | FAIL 判据与定向整改 |
|---|---|---|---|
| MAT-01 | 高级哑光材质响应 | host、Cr 和 ghost 均具有连续的球面渐变；高光宽、弱、受控；颜色饱和度克制，表面仍读作 matte。 | 若像纯色圆片、镜面球或塑料玩具：提高 roughness、降低尖锐 specular/metallic，保留宽而弱的高光，并重新渲染。 |
| MAT-02 | 光照层次 | 有明确但柔和的主光方向，配合宽补光/环境光；原子底部和相邻层有轻柔 contact/AO 阴影。 | 若无体积感：增加宽主光和轻 AO；若出现硬黑阴影、烧白斑或强烈反射：扩大光源、降低对比并重做 exposure。 |
| MAT-03 | 元素/角色分离 | host 为低对比冷灰/蓝灰哑光；迁移 Cr 为暖金/琥珀哑光且清楚突出；ghost 只是同一真实 Cr 材质按 image order 淡出。 | 若 host 平得像背景、Cr 与 host 混淆，或 ghost 像不确定性云带：调整材质对比与 alpha 顺序，禁止用 alpha 表示误差/置信度。 |
| MAT-04 | 真实三维深度 | 遮挡关系、前后层、球面边缘和接触关系在三列中连续且可比较；ghost 仍保留体积明暗。 | 若 ghost 只剩透明轮廓、层次错乱或原子像贴纸：保留实体球的 shading，再降低 opacity；检查 depth sort 和相机。 |
| MAT-05 | 晶格/键的克制 | 晶格或键为细、低对比石墨灰/灰蓝哑光线，能帮助读结构但不抢 Cr；无黑色杂乱网格。 | 若线条喧宾夺主、锯齿明显或完全消失：降低线宽/对比，保留必要结构连接并提升采样。 |
| MAT-06 | 几何真实性 | 每列来自对应真实 retained extxyz 的 10 个 image；Cr 与 host 坐标、PBC/unwrap 记录可追溯；不插值、不手画轨迹。 | 若出现中间补点、抽象路径、参考图手工球或 fake PES：立即退回数据/Blender 层，恢复真实 image 序列。 |
| MAT-07 | 相机与尺度一致 | 三列为正交或近正交可比较视图，球半径、相机方位、裁切和关键 host layers 一致；endpoint 不被裁掉。 | 若某列通过缩放、透视或裁切夸大迁移：统一 camera/scale，重新检查三列同一物理视角。 |
| MAT-08 | 背景与留白 | 白或近白干净背景，边缘有足够留白；无渐变背景、地面线、照片纹理和无关装饰；标注层清晰。 | 若背景抢注意力或边缘拥挤：移除装饰，扩大安全边界；不得用阴影制造一条“轨迹线”。 |
| MAT-09 | 标注独立性 | 0–9 image badge/leader rail 与原子材质分离，文字高对比、无遮挡、三列规则一致；不得把文字放在高光上。 | 若编号被 glare、ghost 或原子遮住：调整 annotation layer 与位置；不得改变坐标来迁就文字。 |
| MAT-10 | 抗锯齿与导出 | body-width proof（目标 7 in）在 100% 与 50% 检查无明显锯齿、破碎边缘、alpha halo 或透明排序伪影；主 PNG 300–600 dpi 等效。 | 若球边、线条或文字有锯齿/白边：提高 samples/分辨率，检查 premultiplied alpha 和合成背景后再渲染。 |
| MAT-11 | 发表尺寸可读性 | 正文目标尺寸下三列标题、0–9、Cr path、host 层次和下方曲线仍可读；灰度与常见色觉缺陷预览中，迁移角色和状态不丢失。 | 若只有大画布才可读：先锁定 body width，再调整字号、线宽和列内留白；不能靠最终缩放补救。 |
| MAT-12 | 负面检查与语义 | 不出现 synthetic PES、手工 energy mountain、伪影光带、摄影式镜面反射或“alpha=uncertainty”暗示；caption 明确 ghost 是真实 image overlay、alpha 只表达 image order。 | 任一负面现象存在即 FAIL；视觉审查员必须给出具体对象、列、image 或像素区域和整改动作。 |
| MAT-13 | 相机投影 | 使用正交或弱透视相机；三列晶格方向一致，无鱼眼、桶形畸变或明显透视收缩。 | 若不同列透视感/收缩不同：锁定相机投影与方位，重新渲染。 |
| MAT-14 | 路径辅助线 | 路径线/路径管只辅助真实原子迁移，线宽稳定、端点清楚；任何平滑不得制造新增 image。 | 若辅助线被读成额外构型：移除人工平滑，回到真实 image 连接。 |
| MAT-15 | ghost 单调性 | ghost 透明度随 image order 单调变化；无中间 ghost 异常变亮、跳层或脱离真实位置。 | 出现 alpha 跳变或层级错误：按 image index 重排 alpha/depth，并重新检查。 |
| MAT-16 | 边缘稳定性 | 球体轮廓、晶格线、leader rail 和边框抗锯齿；无 z-fighting、断边、闪烁边或不一致黑描边。 | 若有 aliasing/z-fighting：提高采样、调整深度偏移并统一 bevel/线宽。 |
| MAT-17 | 快照一致性 | 若使用 inset/snapshot，每个 inset 具有一致相机、比例、padding、边框和材质；不得混用视角或球半径。 | 若 inset 之间尺度/视角漂移：锁定模板并重新生成，不手工拉伸。 |
| MAT-18 | 标注位置 | 迁移原子与 ghost 标注位于留白或 leader rail；文字不压在关键原子、ghost 或路径线上。 | 若文字压图或与高光重叠：移动 annotation layer，不缩小到不可读。 |
| MAT-19 | 标注措辞 | 标注只说明 panel、image order、路径或状态；不得复制参考图中未经本数据支持的 PES/error 叙事。 | 出现 unsupported 词语：删除或改为本项目 manifest 支持的状态措辞。 |
| MAT-20 | 正文尺寸 | 先声明目标正文宽度；body-width 下 axis/legend/tick 约 11–12 pt，核心 image label 至少约 9–10 pt，标题清楚。 | 只有大画布可读：按目标宽度重新排版，不能只提高 dpi。 |
| MAT-21 | 100% 可辨识 | 目标正文尺寸 100% 下 ghost、迁移原子、0–9 和关键 host layer 均可辨识。 | 必须放大才看见：增大字号/marker/线宽或留白并复查。 |
| MAT-22 | 灰度可读 | 灰度下仍能区分 host、迁移原子、ghost、路径线和边框；关键语义不依赖色相。 | 灰度混淆：增加明度、alpha、线型、marker 或边框差异。 |
| MAT-23 | 色觉压力测试 | protanopia、deuteranopia、tritanopia 预览后，仍可依靠明度、透明度、线型、marker 或边框恢复语义。 | 若色觉预览下角色/状态丢失：重配明度与非颜色编码，并保存新 proof。 |
| MAT-24 | 矢量/位图边界 | 主 PNG 为 300–600 dpi；PDF/SVG 放大后文字、曲线、图例和标注保持 vector；Blender 顶部 raster layer 必须明确记录。 | 整张 composite 被低分辨率 PNG 包裹，或 vector layer 退化为位图：拒收并重做导出。 |

## 2. 独立审查流程

1. Blender 执行者提交渲染图、`.blend`、scene manifest 和机器 QA；机器 QA 只能证明“渲染/数据管线执行了检查”。
2. Python/整合执行者提交下方 vector curves、composite PNG/PDF/SVG、caption、hash 和 body-width、灰度、色觉预览。
3. 视觉审查员只基于上述输出、真实数据 manifest 和本 addendum 审查；必须逐项填写 PASS/FAIL/PENDING，不得以执行者自报 PASS 代替。
4. 任一 FAIL：审查员写出定向反馈，至少包括 `criterion_id + 证据位置 + 原因 + 修改动作 + 复查输出`；执行者只改对应层，不改科学数据来迎合视觉。
5. 只有全部 MAT 条件 PASS 且 R01 的 VIS/PHY/DFT/SEED/PUB 硬门槛也满足时，才能把整图标为 publication final。当前 DFT `SCF warning`、固定路径数值矩阵和合法 seed-band 状态仍须按主标准如实保留。

## 3. 参考图到本图的映射

| 参考图特征 | 本项目允许的对应实现 | 明确禁止的误读 |
|---|---|---|
| 圆润、哑光、柔和高光的原子 | Blender Principled/等效材质 + 宽主光 + 轻 AO | 直接用扁平 Matplotlib 圆代替上方真实原子渲染 |
| 多个快照形成迁移视觉 | 真实 image 0–9 的 ghost overlay，按 image order 淡出 | 由曲线或美术插值生成新坐标 |
| 清爽白底和清晰标签 | 独立 annotation layer、等宽列、body-width QA | 用阴影或装饰遮掩缺失 image/数据状态 |
| 参考图中的曲线和示意分区 | 只借鉴阅读层次；下方曲线必须由本项目固定路径数据生成 | 把参考图的 synthetic PES/能量值写进 NEB 图 |

## 4. 审查记录要求

视觉审查记录至少包含：审查员、日期、输出版本、输入图和 manifest 路径、MAT-01—MAT-24 状态、100%/50%/灰度/色觉检查、失败反馈、复查版本和最终 `MATTE_VISUAL_GATE`。`MATTE_VISUAL_GATE=PASS` 仍不等于科学 `FIGURE_ACCEPTED=YES`。

