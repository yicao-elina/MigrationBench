#!/usr/bin/env python3
"""Figure 1 visual specification, replacement contract and LaTeX assembly."""
from pathlib import Path
import json,csv
import pandas as pd
from matplotlib import get_data_path
from PIL import Image as PILImage
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
T=json.loads((ROOT/'Global_Visual_Tokens.json').read_text())
A=json.loads((ROOT/'data/geometry_audit.json').read_text())
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
font_root=Path('/usr/share/fonts/truetype/dejavu')
if not font_root.exists():font_root=Path(get_data_path())/'fonts/ttf'
pdfmetrics.registerFont(TTFont('DV',str(font_root/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DV-Bold',str(font_root/'DejaVuSans-Bold.ttf')))
INK=colors.HexColor(T['ui']['ink']);GREY=colors.HexColor(T['ui']['secondary']);RULE=colors.HexColor(T['ui']['rule'])
ZH=ParagraphStyle('ZH',fontName='STSong-Light',fontSize=10,leading=15,wordWrap='CJK',spaceAfter=6,textColor=INK)
S=ParagraphStyle('S',parent=ZH,fontSize=8.6,leading=12.8,spaceAfter=4)
H=ParagraphStyle('H',parent=ZH,fontSize=18,leading=24,spaceAfter=10)
H2=ParagraphStyle('H2',parent=ZH,fontSize=12,leading=17,spaceAfter=7,textColor=colors.HexColor('#235A83'))
EN=ParagraphStyle('EN',fontName='DV',fontSize=8.2,leading=11.7,spaceAfter=6,textColor=INK)
CAP=ParagraphStyle('CAP',parent=EN,fontSize=8.3,leading=12)
W=174*mm
def p(s,st=ZH):
 text=escape(s).replace('\n','<br/>').replace('Å','<font name="DV">Å</font>')
 return Paragraph(text,st)
def image(path,w=168*mm,h=160*mm):
 im=PILImage.open(path);fac=min(w/im.width,h/im.height)
 return Image(str(path),width=im.width*fac,height=im.height*fac,hAlign='CENTER')
def table(rows,widths):
 rr=[[p(str(v),S) for v in row] for row in rows]
 t=Table(rr,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#F1F5F7')),
  ('LINEBELOW',(0,0),(-1,0),.65,GREY),('LINEBELOW',(0,1),(-1,-1),.3,RULE),
  ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
  ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
 return t
def footer(c,doc):
 c.saveState();c.setStrokeColor(RULE);c.setLineWidth(.4);c.line(18*mm,14*mm,192*mm,14*mm)
 c.setFont('DV',7);c.setFillColor(GREY);c.drawString(18*mm,9.5*mm,'MigrationBench | Figure 1 production specification v2')
 c.drawRightString(192*mm,9.5*mm,str(doc.page));c.restoreState()

CAPTION='''Figure 1 | A task-conditioned design for comparing initialization and thermal target-data coverage. a, Schematic Cr-doped Sb2Te3, with Te, Sb and Cr represented in pink, cyan and gold. QL denotes a quintuple layer, with adjacent QLs separated by a vdW gap. The atomistic illustrations are visual references for later coordinate-based rendering, not optimized transition structures. b, The fitted comparison crosses random or foundation initialization with 600 K or multi-temperature target data. S600/SMT denote scratch models and FT600/FTMT foundation-initialized fine-tuning under the two regimes. Three symbols in each cell identify training seeds 123 (circle), 234 (triangle) and 345 (square); open/filled symbols denote 600 K/multi-temperature training. An independent zero-shot branch uses foundation weights without target fitting. All four fitted cells and the zero-shot branch return the same energy/force observables; matching of capacity and optimization budget is not implied by this schematic. c, Distinct probes measure pointwise residuals, dynamical rollout, local in-gap motion, entry and intralayer environments, and collective sliding. Their ordering is not a calibrated OOD coordinate. d, Fixed-image scoring and model-relaxed evaluation use distinct configuration operators. The drawn image paths are conceptual, not energy profiles. e, Cr displacements calculated from ten supplied images per input route after consecutive minimum-image unwrapping. The normal is defined by the first two cell vectors; lateral displacement is the norm of the orthogonal component. The short nominal entry input remains near the origin; its expanded view is supplied separately. These coordinates are not joined to the energy table and do not establish penetration, reference convergence or training-set coverage.'''
(ROOT/'Figure1_Caption_EN.txt').write_text(CAPTION+'\n')

story=[]
def add(x):story.append(x)
def para(t,st=ZH):add(p(t,st))
def page():add(PageBreak())

para('Figure 1：从结构骨架到可替换的视觉成稿',H)
para('这一版的目标是把“理想状态长什么样”落实到可见的图面、独立素材和明确的制作合同：原子场景给出视觉目标，workflow 与文字保留矢量，能由坐标计算的部分直接使用真数据。',S)
add(image(ROOT/'Figure1_Ideal_Visual.png',h=176*mm))
para('本图 183×198 mm。它是精细视觉目标版：3D 场景需要用真实结构的 Blender 渲染替换；流程、图例、公式、箭头和几何数据图已经可直接复用。此前六图总蓝图保留，本次只细化 Figure 1。',S)
page()

para('1 先明确 Figure 1 要回答什么',H)
para('不是在第一张图证明哪一个模型获胜，而是让读者一次看清材料、训练比较的两个因素、过程测试以及两种测量算子的差异。这样后文的排序反转才有可解释的上下文。')
add(table([['面板','核心内容与元素','最终制作方式'],
 ['a 材料场景','两组 QL、一个 gap Cr、独立原子图例；不把所有路径都挤进主材料图。','当前 A01 作为视觉目标；CIF/extxyz → Blender。标签和图例保持矢量。'],
 ['b 训练 workflow','AIMD 图标、2×2 模型训练格、三 seeds、独立 zero-shot bypass、共同 E/F 接口。','准确的逻辑、线条与字直接使用 SVG/PDF；A07/A02 换真实小渲染即可。'],
 ['c 任务 atlas','Static、MD、in-gap、entry、within-QL、sliding 各有独立原子场景和任务色细线。','六个可替换图像槽；动作箭头单独为矢量，避免烘进图片。'],
 ['d 测量算子','同一 reference images 与模型弛豫 images 的配置差异；两条公式。','全部精确矢量，可在 LaTeX 使用；示意路径没有数值坐标轴。'],
 ['e 真实输入几何','3 个 extxyz、每条 10 images 的 Cr lateral / normal displacement；独立图中附 entry 放大。','已经使用实际坐标生成 PDF/SVG/PNG 与 CSV，无需 Blender 或 imagegen。']],widths=[25*mm,83*mm,66*mm]))
para('视觉密度的分配',H2)
para('a 负责材料质感，b 负责逻辑精度，c 负责快速识别不同物理过程，d 负责定义比较方式，e 负责给出一处真实、可审计的数据锚点。无需在每个格子再堆一套网络结构和公式。')
para('主画布坐标写在 data/panel_layout_mm.json，原点为页面左上。局部细化时固定这一层，再替换图像和少量注释，避免每改一个元素都重排全图。',S)
page()

para('2 全文统一的颜色与图例规则',H)
para('你上传的旧生成参考图与实际 Blender 图采用了不同的元素配色。本规范明确采用后者：Te 粉紫、Sb 青、Cr 金。所有后续材料图都按这个映射。材质的 Hex 是 sRGB 基色目标，不要求所有受光像素都等于这个数值。',S)
add(table([['通道','元素','固定颜色／编码','约束'],
 ['原子','Te / Sb / Cr','#C88DB7 / #75BEC8 / #E4B44C','Cr 始终为金色，不按模型方法重新上色。'],
 ['方法','Zero-shot / FT / Scratch','#8FC7AB / #29A7B6 / #327FB1','只用于方法行、模型标记和方法图例。'],
 ['热采样','600 K / Multi-T','空心 / 实心','不占用另一套颜色；zero-shot 没有 thermal training cell。'],
 ['训练 seed','123 / 234 / 345','圆 / 三角 / 方形','显示全部 seeds。训练 seeds 与 MD velocity seeds 分开。'],
 ['基础 checkpoint','每个独立 checkpoint','菱形 + 独立名称','不把不同 checkpoint 当成训练重复计算均值误差条。'],
 ['任务','Static / MD','#73869A / #5C9B7C','任务标题、路径线或小图上方细线使用。'],
 ['局域任务','in-gap / entry / within-QL','#D28A32 / #CD6D62 / #7D5C99','方向箭头与相应真实几何曲线使用同一任务色。'],
 ['集体任务','Sliding','#287E8A','不把滑移箭头误作为 Cr 的单粒子轨迹。'],
 ['算子','Fixed / Self-relaxed','#F4BDC3 / #E978A4','只作算子标签细线。原先大粉色占位块由实际场景替换。'],
 ['参考／状态','Reference / failed / pending','#303840 / #BD6472 / #A6AFB7','参考为灰虚线；状态不能混进 energy 色标。'],
 ['数值色标','预测 energy；absolute deviation','顺序蓝色填充；误差控制点面积','适用于 Fig. 6 dot matrix；不是全文所有量都强行用同一色标。']],widths=[24*mm,42*mm,58*mm,50*mm]))
para('完整视觉 token 已提供 JSON 与 CSV，也有一页 Global_Visual_Standard.pdf。图例的标题必须说明通道：Atoms、Strategy、Thermal regime、Seed、Task、Operator 各自独立，避免“青色到底指 Sb 还是 FT”的无标题混用。',S)
page()

para('3 workflow 应该做到什么精度',H)
add(image(ROOT/'panels/Figure1b_Training_workflow.png',h=92*mm))
para('每一个箭头都有语义：目标数据定义列，初始化定义行，四个训练格分别接入共同的 E/F 接口。Zero-shot 从 foundation prior 走独立旁路，不能经过 target fitting，也不能被画成两个温度训练格。',S)
add(image(ROOT/'vector/MACE_Interface_Vector.png',h=49*mm))
para('补充的 MACE 接口图可作为 Figure 1b 的局部展开或方法图使用：局域结构 → 等变特征 → 原子能量 readout → E 与其梯度力。条带是特征示意符号，没有声称具体通道数、层数或 lmax。',S)
para('需要你用 LaTeX 替换什么',H2)
para('现在的公式、变量、箭头和标签均已是矢量，可直接使用，无需 OCR 或重画。若全文采用不同数学字族，只需在 LaTeX 中替换这层公式。算法逻辑不依赖 imagegen 的文字、连线或符号准确性。A02 局域环境图中的邻居数量也只是示意，不能当真实配位数。',S)
para('MACE 家族相同不等于网络容量、训练预算已匹配。图中写“共同 E/F 接口”，不把尚缺训练配置的模型画成已证实的 matched architecture。',S)
page()

para('4 材料与局域环境：给 Blender 的具体目标',H)
add(image(ROOT/'assets/concepts/A01_material_hero.png',h=95*mm))
add(table([['元素','视觉定义','真实渲染时的检查'],
 ['两个 QLs','侧向正交视图，小幅俯视；留出清晰 vdW gap；表面 Te 为粉色。','由 CIF / extxyz 导出实际层序，不按生成图的球数复制结构。'],
 ['Cr','金色，作为视觉焦点；单个场景中只有一个实际 Cr 身份。','核对 species、site 和 image index；不同路径不能被误画为同一原子的三段连续运动。'],
 ['球与键','satin 质感；soft key + fill；键细、灰、低反差；无景深模糊。','半径用于可读性；不要把装饰键当成经判定的化学键证据。'],
 ['背景','PNG RGBA，无文字、无图例、无已烘焙箭头。','白底检查透明边缘；不要截断球、键或边缘原子。']],widths=[22*mm,76*mm,76*mm]))
para('A02：局域环境与特征编码插图',H2)
para('中心 Cr 为金色，邻居 Te / Sb 遵循相同粉／青基色。它适合 workflow 小图标，而不是在这里报告某个固定配位壳。若用于真实局域结构分析，应按所选 image、cutoff 和真实邻居集合重新渲染。',S)
page()

para('5 六类任务如何填满 atlas',H)
add(image(ROOT/'panels/Figure1c_Task_atlas.png',h=62*mm))
add(table([['任务／素材 ID','图像里必须看见什么','动作层与替换要求'],
 ['Static / A02','一组局域原子环境，清楚辨认 Cr 与不同 host species。','没有假造的 E/F 散点或力矢量数值；标签是 pointwise residual。'],
 ['MD / A07','同类材料的多个配置缩略图，代表采样和 rollout。','当前三张小晶体不是三条实际轨迹，也不是温度样本数；有真实 MD 后按可比时刻取帧。'],
 ['In-gap / A03','完整 gap 与其中的单个 Cr，横向有可用空间。','方向箭头橙色，原子与键由真实输入替换；不把箭头长度当位移数值。'],
 ['Entry / A04','Cr 接近上侧 QL 的内表面，显示 gap / layer 界面。','珊瑚色上行箭头只解释测试意图；实际 entry 输入很短，不能暗示已验证穿层。'],
 ['Within-QL / A05','单个 QL 的局部 cutaway，让层内 Cr 可见。','紫色动作线独立；真实 crop 不能为了露出 Cr 任意删改物理结构，需说明 cutaway。'],
 ['Sliding / A06','两个完整平行 QL 的相对平移，Cr 留在 gap。','上下两层的反向箭头表示相对滑移；不是 Cr 的迁移路径。']],widths=[32*mm,73*mm,69*mm]))
para('七个独立 PNG 都已给出透明底。A01 是材料主场景，A02/A07 同时供 workflow 与任务 atlas 使用，A03–A06 是四种过程环境。这些资产可以单独讨论和替换；不需要重新生成整个页面。',S)
para('小图统一相机、球半径比例和照明。局部 cutaway 与透明 ghost 只作为展示手法，并在 caption 中说明；不能混入真实计算几何。',S)
page()

para('6 这一处直接用真实坐标作图',H)
add(image(ROOT/'panels/Figure1e_Actual_geometry.png',w=125*mm,h=110*mm))
rows=[['输入','images','Cr 净位移（Å）','采样路径长（Å）']]
for a in A:
 rows.append([a['route'],str(a['frames']),f"{a['net_displacement_A']:.6f}",f"{a['sampled_length_A']:.6f}"])
add(table(rows,[45*mm,22*mm,53.5*mm,53.5*mm]))
para('计算口径',H2)
para('逐相邻 image 用 cell 的 minimum-image convention 展开 Cr 位移；层法向 n = (a×b)/|a×b|。纵轴为 Δr·n，横轴为剔除法向分量后的位移模长。所有单位为 Å，三条线各连接 10 个输入 images，原坐标与结果 CSV 一起提供。',S)
para('这张图可以直接保留，但它回答的是“输入几何改变了多少”，不回答“训练集覆盖多广”或“路径是否确实进入另一环境”。nominal entry 文件的净位移仅约 0.148 Å。没有 geometry hash 对齐时，不把这些坐标与 NEB 能量表自动配对。',S)
para('因此不生成一个看似漂亮但没有原数组的 PCA / UMAP。后续若提供真实 embedding 和 sampling manifest，可将 e 替换为训练集／目标路径 coverage 图，其余版式与颜色不用动。',S)
page()

para('7 算子区与完整 Figure 1 caption',H)
add(image(ROOT/'panels/Figure1d_Operators.png',h=70*mm))
para('Fixed 使用同一组 reference images，比较能量／力；self-relaxed 允许模型参与优化，必须同时保留优化失败、端点变化和路径验收状态。示意中的弯线不是结果曲线，没有人为插入能垒形状。',S)
para('完整英文 caption',H2)
para(CAPTION,CAP)
para('将真实 Blender 渲染全部替换完成后，caption 第一段可改为“Atomistic views rendered from the stated structural inputs”，并补 image ID 与显示的 crop / cutaway。在此之前保持“visual references”限定。',S)
page()

para('8 逐元素替换合同',H)
add(table([['对象','当前交付','最终替换／保留','位置'],
 ['A01','两 QL 主场景艺术参考 PNG','Blender → 同名 PNG RGBA','Figure 1a'],
 ['A02','Cr 局域邻域艺术参考 PNG','若需真实配位，Blender 重做；否则保留为 schematic','1b / 1c / MACE 接口'],
 ['A03','in-gap 原子环境 PNG','真实路径代表 image → Blender','1c'],
 ['A04','entry 界面 PNG','核验真实入口几何后再 Blender；不得假装已穿层','1c'],
 ['A05','within-QL cutaway PNG','按实际结构制作并说明 cutaway','1c'],
 ['A06','相对滑移 PNG','同一滑移 reference 的代表几何 → Blender','1c'],
 ['A07','三配置示意 PNG','真实 AIMD / MD 快照可替换；当前不是实际轨迹','1b / 1c'],
 ['Workflow / 箭头','准确 SVG/PDF，含完整旁路','直接保留；必要时改 SVG 文本或 source','1b / 1c'],
 ['文字、公式、图例','矢量，可编辑','保留；若全文字族不同，用 LaTeX 替换数学层','全图'],
 ['Actual geometry','真实 CSV + plot PDF/SVG','可直接保留；有真正的 coverage 数组时升级','1e']],widths=[26*mm,48*mm,75*mm,25*mm]))
para('操作方法',H2)
para('把最终 Blender 输出放入 assets/production/，使用与当前素材相同的文件名，例如 A03_in_gap.png。执行 python source/build_figure1.py，程序优先读取 production，缺少时才读取 concepts。这样局部替换不会污染旧图或其他五张主图。',S)
para('任务箭头保持独立，并用金色 Cr 的图像位置注册起点；data/route_arrow_anchors.json 记录了它们。这个注册只用于排版，不是从图片恢复真实轨迹。真实 Cr 没有被正确检测或构图改变时，需要按实际 image 手动核对箭头起止点。',S)
page()

para('9 Blender：按统一镜头与材质重建',H)
add(table([['项目','本项目约定','原因／注意点'],
 ['相机','Orthographic；起始 azimuth 18°、elevation 12°、roll 0°','作为视觉起点，并非从参考图反演的精确相机；所有 atlas 小图保持同镜头。'],
 ['基色','Te #C88DB7；Sb #75BEC8；Cr #E4B44C','Blender_Handoff.json 同时提供 sRGB 与 linear RGB，避免色彩空间误用。'],
 ['材质','metallic 0；roughness 0.32；柔和高光','不使用铬金属镜面、强 bloom 或景深。Cr 的金色是身份编码，不表示材质为金属金。'],
 ['相对尺寸','Te 1.00、Sb 0.95、Cr 0.90；键半径约 host 半径 0.10','是图示可读性的起点，不表示真实原子半径测定。'],
 ['照明','左上 broad key；fill/key 0.35；轻 rim 0.12','灰影只用于结构体积，不在透明背景上加地面或大投影。'],
 ['输出','主场景建议 3600×2600；task 1600×1400；local 1600×1600','PNG RGBA；主体留约 8% 边距；不烘焙文字、图例、箭头。'],
 ['结构验收','species 逐层核对；Cr 身份与路径 image 对齐；明确周期副本与裁剪','Te-Sb-Te-Sb-Te 层序必须来自实际输入，不按 imagegen 的球数仿造。'],
 ['叠加检查','在最终白底、最终物理尺寸检查；核对箭头与金色 Cr 的位置','材质基色与高光像素不必相同；所有 task 图的亮度和比例需一致。']],widths=[23*mm,78*mm,73*mm]))
para('本次没有运行 Blender。另已导出 9 个不改坐标的代表帧 XYZ（inputs/blender_frames），覆盖 A01/A03/A04/A05，并提供原路径、image index 与 SHA 索引。A02 可以从真实结构裁取；A06 仍需实际 sliding geometry，A07 需 AIMD/MD 快照，不能将 NEB 帧冒充 MD 帧。',S)
para('若需要玻璃 ghost、剖面或夸张的表层偏移，请放在独立 collection 并注明“display only”；能量计算对应的原结构必须保留为不修改的 collection。',S)
page()

para('10 数据图的统一标准',H)
add(table([['内容','统一画法','不能混用'],
 ['Energy / force RMSE','E 用 meV/atom；F 用 meV/Å；注明 component 或 vector magnitude','训练点、力分量和训练 seed 是不同统计单位。'],
 ['Migration barrier','路径总能量的 eV；明确 forward / reverse；reference 资格随图给出','不能把 per-atom energy RMSE 直接与 barrier eV 作同轴精度比较。'],
 ['Fixed-path profiles','同 image、同能量零点；最好给全路径 residual 与 endpoint ΔE','低 forward barrier 不说明路径形状或终点能量正确。'],
 ['训练重复','n=3 展示所有 dots；需要汇总时 mean ± sample SD','不做平滑 violin；没有重复统计基础就不画 CI / p-value。'],
 ['Foundation baseline','逐 checkpoint 显示和命名，保留 SHA / lineage','不同基础 checkpoint 不当作同一方法的随机 seed。'],
 ['不收敛 / 缺失','NC 用 crossed-open marker，无定量填色或面积；NA 留空并标明','零值、零误差、NC、缺数据不可共用同一种小圆点。'],
 ['MD','保留真实时间轴、失败时刻、目标时长和 velocity seed','未完成或失败的轨迹不能输出 1 ns transport；不得从绝对步数盲目换算时间。'],
 ['色标','顺序蓝用于非负量；有符号残差另用以 0 为中心的发散色标','绝对量、signed deviation 和类别不可复用一个无标题色条。']],widths=[30*mm,78*mm,66*mm]))
para('物理尺寸、线条与文字',H2)
para('183 mm 双栏宽；面板字母约 10 pt；标题约 8.4 pt；正文和轴标签约 7.1 pt；刻度约 6.5 pt；细线 0.6 pt；reference line 0.7 pt。以 PDF/SVG 保留原生文本；PNG 450 dpi 供预览与合成。这里是本项目的设计规范，不是对所有 Nature 期刊的官方尺寸声明。',S)
para('本图真实 plot 的 source/CSV、所有颜色和形状定义随包提供，图像生成不参与任何数据坐标、统计量或误差条。',S)
page()

para('11 LaTeX 合成与下一轮局部精修',H)
para('已提供三种可编辑层级',H2)
add(table([['层级','文件','修改内容'],
 ['完整图','Figure1_Ideal_Visual.pdf / .svg / .png','整体查看；SVG 内文字、线、点与嵌入图像分开。'],
 ['图片层','vector/Figure1_Artwork_Only.pdf','只有原子与晶体 cutouts；保持 183×198 mm 相同画布。'],
 ['矢量层','vector/Figure1_Vector_Overlay.pdf / .svg','所有文字、流程线、图例、公式、方向箭头及真实数据图。'],
 ['局部面板','panels/Figure1a–e_*.pdf / .svg / .png','单独讨论每一部分，无需重做全图。'],
 ['编译入口','latex/Figure1_LaTeX_Assembly.tex','原子图片层在下，准确矢量层在上；保持整体尺寸。'],
 ['重建入口','source/build_figure1.py','更换 production 同名资产，重建全部图层与图像。']],widths=[27*mm,79*mm,68*mm]))
para('编译示例',H2)
para('cd latex\npdflatex -interaction=nonstopmode -halt-on-error Figure1_LaTeX_Assembly.tex',EN)
para('先替换一个原子槽 → 核对相机、配色与尺寸 → 检查箭头注册 → 重建 full figure → 最终尺寸查看。一次优先改结构内容或版式其中一个，避免无法判断改善来自哪一层。具体修改可以指向“b 的 zero-shot 旁路”“c 的 entry 原子场景”或“e 的 entry 放大图”，而不用重新生成整页。',S)
para('提交前的 Figure 1 验收',H2)
para('材料身份和层序正确；两个训练因素与 zero-shot 旁路正确；所有素材使用一致物种配色；任务动作箭头不暗示已验证机制；a–d 为 schematic，e 为真实输入坐标；无未声明的 PCA、能量曲线或统计结果。',S)

doc=SimpleDocTemplate(str(ROOT/'Figure1_Production_Guide.pdf'),pagesize=(210*mm,297*mm),
 leftMargin=18*mm,rightMargin=18*mm,topMargin=17*mm,bottomMargin=19*mm,
 title='MigrationBench Figure 1 | production design, assets and replacement specification')
doc.build(story,onFirstPage=footer,onLaterPages=footer)

tex=r'''\documentclass{article}
\usepackage[paperwidth=183mm,paperheight=198mm,margin=0mm]{geometry}
\usepackage{graphicx,tikz}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\begin{document}
\begin{tikzpicture}[remember picture,overlay]
\node[anchor=north west,inner sep=0pt] at (current page.north west)
  {\includegraphics[width=183mm,height=198mm]{../vector/Figure1_Artwork_Only.pdf}};
\node[anchor=north west,inner sep=0pt] at (current page.north west)
  {\includegraphics[width=183mm,height=198mm]{../vector/Figure1_Vector_Overlay.pdf}};
\end{tikzpicture}
\null
\end{document}
'''
(ROOT/'latex/Figure1_LaTeX_Assembly.tex').write_text(tex)
(ROOT/'latex/README.txt').write_text('The two PDFs share a 183 x 198 mm canvas. Artwork below, vectors above.\nRun pdflatex -interaction=nonstopmode -halt-on-error Figure1_LaTeX_Assembly.tex from this directory.\nReplace the PNG slots in assets/production first and rebuild source/build_figure1.py to update both layers.\n')
(ROOT/'README.txt').write_text('''Figure 1 production kit v2

Start with Figure1_Ideal_Visual.pdf and Figure1_Production_Guide.pdf.
The seven generated atomic cutouts are conceptual visual references. They are
not claimed to be actual relaxed coordinates or atomistic simulation results.
Te=pink, Sb=cyan, Cr=gold follows the supplied actual Blender image legend.

Already exact: vector workflow/legend/text/formula layers and panel e, which
is computed from 30 supplied extxyz images with an explicit MIC convention.
No source-less PCA/UMAP, RMSE, force vectors or energy curves were invented.

To replace 3D art, place the final same-named PNG RGBA assets in
assets/production/. Rebuild with:
  python source/build_figure1.py
  python source/build_production_guide.py

Blender_Handoff.json and Global_Visual_Tokens.json define rendering/visual roles.
The PDF/SVG vector overlay and the artwork plate have exactly the same canvas.
latex/Figure1_LaTeX_Assembly.tex demonstrates assembly without rasterizing text.

The extra MACE_Interface_Vector illustrates the local-to-E/F interface and can
be used as a methods expansion. Its feature glyphs do not assert layer/channel
counts or matching of capacities across the compared checkpoints.

Source image-generation prompts are in source/imagegen_prompt_set.json;
the built-in image_gen tool was used for the seven art references.
Blender was not run in this environment. Its JSON values are art-direction
starting points, not inverse-recovered settings or a tested .blend scene.
''')
(ROOT/'requirements.txt').write_text('numpy\npandas\nmatplotlib\npillow\nPyMuPDF\nreportlab\n')
manifest=[['asset_id','role','replace_in','production_filename','current_type'],
 ['A01_material_hero','two-QL main scene','a','A01_material_hero.png','imagegen concept'],
 ['A02_local_environment','local neighborhood','b/c/interface','A02_local_environment.png','imagegen concept'],
 ['A03_in_gap','gap migration environment','c','A03_in_gap.png','imagegen concept'],
 ['A04_gap_to_ql','entry interface','c','A04_gap_to_ql.png','imagegen concept'],
 ['A05_within_ql','intralayer cutaway','c','A05_within_ql.png','imagegen concept'],
 ['A06_collective_sliding','relative layer translation','c','A06_collective_sliding.png','imagegen concept'],
 ['A07_thermal_sampling','illustrative configuration ensemble','b/c','A07_thermal_sampling.png','imagegen concept'],
 ['E01','real input-coordinate displacement','e','replace only with verified source arrays','data plot'],
 ['V01','training flow and common outputs','b','edit SVG / source','native vector'],
 ['V02','operator definitions and formulas','d','edit SVG / source / LaTeX','native vector']]
with open(ROOT/'Element_Replacement_Map.csv','w') as f:csv.writer(f).writerows(manifest)
print('Created Figure 1 production guide, caption, replacement map and LaTeX assembly source.')
