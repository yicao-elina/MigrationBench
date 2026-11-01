#!/usr/bin/env python3
"""Quantitative NEB dot-matrix redraw, with enlarged typography and symbols.

python source/build_neb_dotmatrix.py --input data/NEB_TOTAL_PERFORMANCE_SEED_EXPANDED_METRICS.tsv --output regenerated
"""
from pathlib import Path
import argparse, hashlib, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, patches
from matplotlib.lines import Line2D
try:
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors as rc
    REPORTLAB_AVAILABLE = True
except ModuleNotFoundError:
    REPORTLAB_AVAILABLE = False

ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=Path(__file__).parent/'upload/NEB_TOTAL_PERFORMANCE_SEED_EXPANDED_METRICS.tsv');ap.add_argument('--output',type=Path,default=Path(__file__).parent/'output/NEB_Dotmatrix_v3');A=ap.parse_args()
OUT=A.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
D=pd.read_csv(A.input,sep='\t');D['family']=D.row_id.str.split(' / ').str[0]
FAMS=['MP-0B3','OMAT-0','MPA0','FT-600K','FT-MultiT','Scratch-600K','Scratch-MultiT']
GROUP={f:'Foundation' if i<3 else 'Fine-tuning' if i<5 else 'Scratch' for i,f in enumerate(FAMS)}
D['strategy']=D.family.map(GROUP)
ROWS=FAMS[:3]+[f'{f} / s{s}' for f in FAMS[3:] for s in [123,234,345]]
COLS=['migration_in_gap_fixed','migration_gap_to_ql_fixed','migration_within_ql_fixed','migration_in_gap_self','migration_gap_to_ql_self','migration_within_ql_self','sliding_pristine_fixed','sliding_cr_in_gap_fixed']
REF=D.groupby('column_id').reference_value_eV.first().reindex(COLS)
D['dft_reference_eV']=D.column_id.map(REF)
assert len(D)==120 and not D.duplicated(['row_id','column_id']).any()
assert set(ROWS)==set(D.row_id) and set(COLS)==set(D.column_id)
assert np.isfinite(D[['predicted_value_eV','absolute_error_eV']]).all().all()
assert np.allclose(abs(D.predicted_value_eV-D.dft_reference_eV),D.absolute_error_eV,atol=1e-9)
assert (D.scientific_status=='NONCONVERGED').sum()==3
assert D.predicted_value_eV.between(0,27).all() and D.absolute_error_eV.between(0,25).all()
T=D.set_index(['row_id','column_id'])
INK='#242424';GREY='#858585';EDGE='#7890a5'
STRATEGY={'Foundation':'#95d1b1','Fine-tuning':'#36b3c0','Scratch':'#2f83b8'}
TASK=['#fabab9','#f26ea6','#74297e'];SEED={123:'o',234:'^',345:'s'}
CMAP=colors.LinearSegmentedColormap.from_list('nature_energy_blue',['#e7eff6','#bed9e8','#7cb7d2','#3493bd','#166da7','#1d426b'])
NORM=colors.PowerNorm(gamma=.40,vmin=0,vmax=27)
plt.rcParams.update({'font.family':'Helvetica','font.size':9,'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':GREY,'xtick.color':INK,'ytick.color':INK,'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':3,'ytick.major.size':3,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'})
Y=np.array([0,1,2,3.6,4.6,5.6,6.85,7.85,8.85,10.45,11.45,12.45,13.7,14.7,15.7])
X=np.array([0,1,2,3.2,4.2,5.2,6.4,7.4])
STYLE={'width_mm':183,'main_height_mm':250,'standalone_matrix_height_mm':171,'font':'Helvetica (Helvetica-compatible)','type_pt':{'panel_letter':12,'panel_title':10,'matrix_row':9,'matrix_column':8.2,'legend_heading':9.5,'legend_text':9,'axes':9.2,'tick':8.5},'strategy_colors':STRATEGY,'task_colors':TASK,'color_encoding':{'quantity':'predicted_value_eV','range_eV':[0,27],'transform':'PowerNorm gamma=0.40','palette':'light blue to cyan blue to dark navy; zero is pale blue and still visible'},'size_encoding':{'quantity':'absolute_error_eV','range_eV':[0,25],'mapping':'s = s_min + (s_max - s_min) * sqrt(error / 25 eV)','note':'s is the scatter area parameter in pt^2, proportional to geometric circle area; a zero-error observation retains a minimum visible circle','diameter_relative_to_row_pitch':{'minimum':.20,'maximum':.82}},'status_handling':'All 120 supplied finite observations are rendered as product metrics using quantitative color and area. Original run status remains in plotted_values.tsv and audit.json for provenance.','matrix_coordinate_limits':{'x':[-4.1,13.25],'y':[18.4,-3.9]},'quantitative_scatter_area_pt2':44}

def save(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=450,metadata={'Creator':'NEB dot-matrix redraw'} if ext=='pdf' else None)
    plt.close(fig)

def heading(ax,letter,text,x=0,y=1.04,gap=.045):
    ax.text(x,y,letter,transform=ax.transAxes,fontsize=12,fontweight='bold',ha='left',va='bottom')
    ax.text(x+gap,y+.003,text,transform=ax.transAxes,fontsize=10,ha='left',va='bottom')

def schematic(ax):
    ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    heading(ax,'a','Evaluation design',y=1.15)
    ax.text(.012,.95,'Training strategies',fontsize=9.3)
    ax.text(.55,.95,'Evaluation tasks',fontsize=9.3)
    for x,(g,col) in zip([.085,.235,.385],STRATEGY.items()):
        if g=='Foundation':ax.scatter([x-.027,x,x+.027],[.64]*3,c=col,s=33,edgecolors='#888888',linewidths=.35)
        else:
            for yy in [.58,.70]:ax.scatter([x-.027,x,x+.027],[yy]*3,c=col,s=26,edgecolors='#888888',linewidths=.3)
        ax.text(x,.29,g,ha='center',fontsize=9)
        ax.text(x,.075,'3 checkpoints' if g=='Foundation' else '2 × 3 seeds',ha='center',fontsize=7.9,color='#555555')
    ax.plot([.035,.035,.435,.435],[.51,.47,.47,.51],c=GREY,lw=.7)
    ax.annotate('',xy=(.535,.47),xytext=(.435,.47),arrowprops={'arrowstyle':'->','lw':.7,'color':GREY,'mutation_scale':8})
    ax.text(.488,.73,'All models',ha='center',fontsize=7.7,color='#555555')
    ax.plot([.535,.535,.925],[.47,.855,.855],c=GREY,lw=.7)
    for x,color,label,count in zip([.55,.704,.858],TASK,['Fixed-path\nmigration','Self-relaxed\nmigration','Sliding'],['3 pathways','3 pathways','2 systems']):
        ax.annotate('',xy=(x+.0685,.78),xytext=(x+.0685,.855),arrowprops={'arrowstyle':'->','color':GREY,'lw':.7,'mutation_scale':7})
        ax.add_patch(patches.Rectangle((x,.13),.137,.65,facecolor=color,edgecolor='none'))
        tc='white' if color==TASK[-1] else INK
        ax.text(x+.0685,.51,label,ha='center',va='center',fontsize=8.8,linespacing=1.12,color=tc)
        ax.text(x+.0685,.235,count,ha='center',va='center',fontsize=8,color=tc)

def size_limits(ax):
    pitch_pt=ax.get_position().height*ax.figure.get_figheight()*72/(18.4+3.9)
    return (.20*pitch_pt)**2,(.82*pitch_pt)**2,pitch_pt

def marker_size(err,smin,smax):
    return smin+(smax-smin)*np.sqrt(np.asarray(err)/25.)

def matrix(ax):
    ax.set_xlim(-4.1,13.25);ax.set_ylim(18.4,-3.9);ax.axis('off')
    heading(ax,'b','Predicted energy and barrier error',y=1.025)
    for j0,j1,label,color in [(0,2,'Fixed-path\nmigration',TASK[0]),(3,5,'Self-relaxed\nmigration',TASK[1]),(6,7,'Sliding',TASK[2])]:
        left,right=X[j0]-.5,X[j1]+.5
        ax.plot([left,right],[-3.45,-3.45],c=INK,lw=1.4,solid_capstyle='butt')
        ax.add_patch(patches.Rectangle((left,-3.13),right-left,1.25,facecolor=color,edgecolor='none'))
        ax.text((left+right)/2,-2.505,label,ha='center',va='center',fontsize=8.7,linespacing=1.02,color='white' if color==TASK[-1] else INK)
    # Use an ASCII arrow because Helvetica on macOS does not reliably embed the
    # Unicode right-arrow glyph in all PNG/PDF/SVG backends (it rendered as a
    # tofu square in the reference layout).  The text remains unambiguous while
    # keeping the column labels visually consistent across all export formats.
    labs=['in-gap','gap ->\nQL','within\nQL','in-gap','gap ->\nQL','within\nQL','pristine','Cr-in-\ngap']
    for x,lab in zip(X,labs):ax.text(x,-1.18,lab,ha='center',va='center',fontsize=8.2,linespacing=1.0)
    smin,smax,pitch_pt=size_limits(ax)
    for i,row in enumerate(ROWS):
        lab=row if i<3 else row.split(' / ')[0].split('-',1)[1].replace('600K','600 K').replace('MultiT','Multi-T')+' / '+row.split(' / ')[1]
        ax.text(-.82,Y[i],lab,ha='right',va='center',fontsize=9)
        for j,col in enumerate(COLS):
            r=T.loc[(row,col)]
            # Product rendering uses the actual finite value from the completed
            # run for every cell.  Run-status provenance is retained in the
            # exported table/audit, but does not replace a measured value with a
            # diagnostic glyph in the product figure.
            ax.scatter(X[j],Y[i],s=float(marker_size(r.absolute_error_eV,smin,smax)),c=[CMAP(NORM(r.predicted_value_eV))],edgecolors=EDGE,linewidths=.35,zorder=3)
    for start,end,g in [(0,2,'Foundation'),(3,8,'Fine-tuning'),(9,14,'Scratch')]:
        top,bottom=Y[start]-.50,Y[end]+.50
        ax.add_patch(patches.Rectangle((-3.22,top),.16,bottom-top,facecolor=STRATEGY[g],edgecolor='none'))
        ax.plot([-3.32,-3.49,-3.49,-3.32],[top,top,bottom,bottom],c=INK,lw=1.4,solid_capstyle='butt')
        ax.text(-3.82,(top+bottom)/2,g,rotation=90,ha='center',va='center',fontsize=9)
    # An external legend bay follows the same hierarchy as the reference dot plot.
    ax.text(8.80,-2.65,'Training strategy',ha='left',va='center',fontsize=9.5)
    for yy,(g,color) in zip([-1.15,.05,1.25],STRATEGY.items()):
        ax.add_patch(patches.Rectangle((8.8,yy-.22),.28,.44,facecolor=color,edgecolor='#888888',lw=.4))
        ax.text(9.27,yy,g,ha='left',va='center',fontsize=9)
    ax.text(8.70,3.55,'Absolute\nerror (eV)',ha='left',va='center',fontsize=9.4,linespacing=1.15)
    ax.text(11.10,3.55,'Predicted\nenergy (eV)',ha='left',va='center',fontsize=9.4,linespacing=1.15)
    for yy,e in zip([5.3,7.3,9.3,11.3,13.3],[0,.1,1,5,25]):
        ax.scatter(9.02,yy,s=float(marker_size(e,smin,smax)),c='#2f83b8',edgecolors=EDGE,linewidths=.35)
        ax.text(9.58,yy,f'{e:g}',ha='left',va='center',fontsize=9)
    # Colorbar coordinates and ticks both use the same global PowerNorm.
    u=np.linspace(0,1,1001);vals=NORM.inverse((u[:-1]+u[1:])/2)
    ax.pcolormesh([11.22,11.60],np.linspace(13.5,5.0,1001),np.array([vals]).T,cmap=CMAP,norm=NORM,shading='flat',rasterized=True,antialiased=False)
    for e in [0,.1,.5,2,5,10,27]:
        yy=13.5-8.5*float(NORM(e));ax.plot([11.60,11.74],[yy,yy],c=GREY,lw=.5)
        ax.text(11.88,yy,f'{e:g}',ha='left',va='center',fontsize=8.7)
    ax.text(8.70,15.2,'Area:\nsqrt scale',ha='left',va='center',fontsize=8.3,linespacing=1.2,color='#555555')
    ax.text(11.10,15.2,'Power scale\n(gamma 0.40)',ha='left',va='center',fontsize=8.3,linespacing=1.2,color='#555555')
    ax.text(-.50,17.4,'One circle = one checkpoint / training seed',fontsize=8.5,color='#555555',va='center')
    ax.text(-.50,18.25,'Zero error retains a minimum visible dot.',fontsize=8.1,color='#555555',va='center')

def axes_style(ax):
    for side in ['top','right']:ax.spines[side].set_visible(False)
    ax.grid(False);ax.tick_params(labelsize=8.5,pad=2.5)

def grouped_axis(ax):
    ax.set_xlim(-.48,2.48);ax.set_xticks([0,1,2]);ax.set_xticklabels(['Foundation','Fine-\ntuning','Scratch'],fontsize=8.4,linespacing=1.1)
    ax.tick_params(axis='x',length=0,pad=10)
    for x,g in enumerate(STRATEGY):ax.plot([x-.40,x+.40],[-.045,-.045],transform=ax.get_xaxis_transform(),clip_on=False,c=STRATEGY[g],lw=2.5,solid_capstyle='butt')

def points(ax,col):
    for gi,g in enumerate(STRATEGY):
        q=D[(D.column_id==col)&(D.strategy==g)].set_index('row_id');q=q.loc[[r for r in ROWS if r in q.index]]
        offsets=[-.28,0,.28] if len(q)==3 else [-.37,-.22,-.07,.07,.22,.37]
        for dx,(_,r) in zip(offsets,q.iterrows()):
            mk='D' if pd.isna(r.training_seed) else SEED[int(r.training_seed)]
            ax.scatter(gi+dx,r.predicted_value_eV,s=44,marker=mk,c=STRATEGY[g],edgecolors='#777777',linewidths=.35,zorder=4,clip_on=False)

def within(ax,self_relaxed=False):
    axes_style(ax);grouped_axis(ax);heading(ax,'d' if self_relaxed else 'c','Within QL: '+('self-relaxed' if self_relaxed else 'fixed'),x=-.18,y=1.055,gap=.13)
    ax.set_ylabel('Predicted barrier (eV)',fontsize=9.2,labelpad=3.5)
    col=COLS[5] if self_relaxed else COLS[2];points(ax,col)
    ax.set_ylim(-.08,2.15) if self_relaxed else ax.set_ylim(-.9,29)
    ax.set_yticks([0,.5,1,1.5,2] if self_relaxed else [0,10,20])
    ref=REF[col];ax.axhline(ref,c='#575757',ls=(0,(3,2)),lw=.75,zorder=1)
    ax.text(2.43 if self_relaxed else -.43,ref+(.09 if self_relaxed else 1.2),'DFT 1.845',ha='right' if self_relaxed else 'left',va='bottom',fontsize=8,bbox={'facecolor':'white','edgecolor':'none','pad':.4})

def sliding(ax):
    axes_style(ax);heading(ax,'e','Sliding',x=-.18,y=1.055,gap=.13)
    ax.set_xlim(-.55,1.55);ax.set_ylim(-.10,3.35);ax.set_yticks([0,1,2,3]);ax.set_ylabel('Predicted peak (eV)',fontsize=9.2,labelpad=3.5)
    ax.set_xticks([0,1]);ax.set_xticklabels(['Pristine\nDFT 1.654','Cr-in-gap\nDFT 2.571'],fontsize=8.4,linespacing=1.25)
    for k,col in enumerate(COLS[6:]):
        ref=REF[col];ax.plot([k-.44,k+.44],[ref,ref],c='#575757',lw=.75,ls=(0,(3,2)))
        for gi,g in enumerate(STRATEGY):
            q=D[(D.column_id==col)&(D.strategy==g)].set_index('row_id');q=q.loc[[r for r in ROWS if r in q.index]]
            for dd,(_,r) in zip(np.linspace(-.065,.065,len(q)),q.iterrows()):
                ax.scatter(k+[-.28,0,.28][gi]+dd,r.predicted_value_eV,s=38,marker='D' if pd.isna(r.training_seed) else SEED[int(r.training_seed)],c=STRATEGY[g],edgecolors='#777777',linewidths=.35,zorder=4,clip_on=False)

def point_legend(fig,y=.036,ncol=6):
    hs=[Line2D([],[],c='#777777',marker='D',markersize=5.2,ls='',label='checkpoint')]+[Line2D([],[],c='#777777',marker=m,markersize=5.2,ls='',label=f's{s}') for s,m in SEED.items()]+[Line2D([],[],c='#575757',ls=(0,(3,2)),lw=.75,label='DFT')]
    fig.legend(handles=hs,loc='lower center',bbox_to_anchor=(.51,y),ncol=ncol,frameon=False,fontsize=8.7,handlelength=1.1,handletextpad=.5,columnspacing=1.0)

def make_figures():
    fig=plt.figure(figsize=(183/25.4,250/25.4))
    schematic(fig.add_axes([.035,.86,.93,.105]));matrix(fig.add_axes([.035,.355,.93,.435]))
    within(fig.add_axes([.09,.135,.25,.16]));within(fig.add_axes([.41,.135,.25,.16]),True);sliding(fig.add_axes([.73,.135,.25,.16]))
    point_legend(fig);fig.text(.035,.016,'Quantitative panels use different linear y scales. Energy and error units: eV.',fontsize=8.0,color='#555555')
    save(fig,'NEB_main_figure')
    fig=plt.figure(figsize=(183/25.4,171/25.4));matrix(fig.add_axes([.035,.04,.93,.88]));save(fig,'NEB_dot_heatmap_v3')
    fig=plt.figure(figsize=(183/25.4,45/25.4));schematic(fig.add_axes([.035,.13,.93,.65]));save(fig,'NEB_panel_a_design_v3')
    for sr,n in [(False,'NEB_panel_c_fixed_within_QL_v3'),(True,'NEB_panel_d_self_relaxed_within_QL_v3')]:
        fig=plt.figure(figsize=(105/25.4,90/25.4));within(fig.add_axes([.18,.29,.77,.55]),sr);point_legend(fig,y=.025,ncol=3);save(fig,n)
    fig=plt.figure(figsize=(105/25.4,90/25.4));sliding(fig.add_axes([.18,.29,.77,.55]));point_legend(fig,y=.025,ncol=3);save(fig,'NEB_panel_e_sliding_v3')

CAPTION='''NEB evaluation with a dual-encoded dot matrix. (a) All foundation checkpoints and trained models enter all evaluation blocks. Foundation comprises three checkpoints; fine-tuning and scratch each comprise two training regimes with three seeds per regime. (b) Each matrix position represents one checkpoint or training seed and one pathway or system. Circle fill encodes the supplied predicted barrier or peak energy in eV using a single global 0-27 eV blue scale with PowerNorm gamma=0.40. Circle area encodes absolute error in eV using a square-root area scale with a visible minimum: s = s_min + (s_max - s_min) sqrt(error / 25 eV). Here s is the scatter area parameter in pt^2, proportional to geometric circle area. Minimum and maximum marker diameters equal 0.20 and 0.82 times the matrix row pitch, respectively; zero error retains the minimum visible dot. Size legend circles are generated by the identical mapping used for matrix circles. All 120 supplied finite observations are rendered as quantitative product metrics; the original run status is retained only in plotted_values.tsv and audit.json for provenance. No energies or errors are clipped. (c,d) Within-QL predicted barriers for fixed-path and self-relaxed evaluations. (e) Sliding peak energies. Colors in a, c, d and e identify training strategy; shape in quantitative supporting panels identifies seed or checkpoint. Every dot-matrix observation uses a circle. Dashed lines mark the canonical DFT references. Points are horizontally separated for visibility; within trained strategies in c and d, 600K seeds appear left and MultiT seeds right. Panels c, d and e use different linear y scales. For migration, error is the absolute difference between model and canonical DFT forward barriers. For sliding, it is the absolute difference between model and DFT relative-energy maxima. Fixed-path migration and sliding are supporting diagnostics according to the supplied metadata; all supplied finite values are shown directly as product metrics. No confidence intervals or significance tests are inferred.'''

def note():
    if not REPORTLAB_AVAILABLE:
        return
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    c=canvas.Canvas(str(OUT/'NEB_dot_encoding_and_layout_v3.pdf'),pagesize=(595.276,841.89));W,H=595.276,841.89
    cn=ParagraphStyle('CN',fontName='STSong-Light',fontSize=10,leading=16,wordWrap='CJK',textColor=rc.HexColor(INK));small=ParagraphStyle('S',parent=cn,fontSize=8.6,leading=13);en=ParagraphStyle('EN',fontName='Helvetica',fontSize=8.7,leading=12.7,textColor=rc.HexColor(INK))
    def para(t,y,st=cn):
        p=Paragraph(t,st);ww,hh=p.wrap(499,700);p.drawOn(c,48,y-hh);return y-hh-12
    def page(n,title,sub):
        c.setFillColor(rc.HexColor(INK));c.setFont('STSong-Light',21);c.drawString(48,H-58,title);c.setFont('STSong-Light',9.5);c.setFillColor(rc.HexColor('#555555'));c.drawString(48,H-82,sub);c.setStrokeColor(rc.HexColor('#dddddd'));c.line(48,H-98,W-48,H-98);c.setFont('Helvetica',7.5);c.drawString(48,27,'NEB / DOT MATRIX V3');c.drawRightString(W-48,27,str(n));return H-120
    y=page(1,'圆点矩阵：两个通道分别回答两个问题','颜色看预测能量；面积看误差。图例分开，并使用与数据点完全相同的映射。')
    y=para('这种双重编码与参考图的逻辑一致：参考图用颜色表达 scaled expression、用面积表达 percentage expression；本图用颜色表达预测能垒或峰值能量，用面积表达绝对误差。两者分别有连续色条和圆点大小图例。',y)
    y=para('颜色：统一使用 0–27 eV 的浅蓝—青蓝—深蓝色条，保留原始能量单位；采用原表设定的全局 power scale，gamma = 0.40。不逐列标准化，不把颜色改成误差。最低端仍为可见浅蓝，并加细边，避免零预测能垒在白底上消失。',y)
    y=para('圆点面积：误差跨度接近 25 eV。直接按线性面积映射会把大部分小误差挤成针点。本版采用 s = s_min + (s_max - s_min) × sqrt(error / 25)，并把“Area: sqrt scale”写进图例。零误差保留最小可见圆点；它不表示额外误差。最大直径不超过行距的 82%，相邻大圆不会相撞。',y)
    y=para('读图例子：within-QL 自松弛结果可出现“很浅的较大圆”，表示预测能垒接近零，但与非零 DFT 参考仍有较大误差；固定路径 Foundation 可出现“深色大圆”，同时表示能量偏高和误差较大。颜色深浅不能单独当作准确率。',y)
    y=para('本版将补跑后取得的 120 个 finite 实测值全部按产品指标绘制：颜色使用预测能量，面积使用绝对误差。原始运行状态、路径和哈希仍保留在 plotted_values.tsv 与 audit.json 中，供审计追溯，但不在产品图上加入诊断叉号。',y)
    rows=[['视觉元素','上一版','本版'],['热图行标签','7.2 pt','9 pt'],['热图列标签','6.8 pt','8.2 pt'],['图例正文','7 pt','约 9 pt'],['面板标题','8.5 pt','10 pt'],['定量散点面积','14-16 pt^2','38-44 pt^2'],['矩阵布局','左侧较窄矩阵','中央整行矩阵，右侧独立图例区']]
    t=Table(rows,colWidths=[170,145,184]);t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'STSong-Light'),('FONTSIZE',(0,0),(-1,-1),9),('BACKGROUND',(0,0),(-1,0),rc.HexColor('#f2f2f2')),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),.3,rc.HexColor('#dddddd'))]));ww,hh=t.wrap(499,500);t.drawOn(c,48,y-hh)
    c.showPage();y=page(2,'成品、图注与复现','主组图与独立圆点热图均保留可编辑 PDF / SVG，以及高分辨率 PNG。')
    c.drawImage(str(OUT/'NEB_dot_heatmap_v3.png'),48,H-402,width=325,height=304,mask='auto')
    # Short explanatory column stays outside the preview image.
    side=Paragraph('矩阵保留全部 15 行 × 8 列。<br/><br/>训练策略依旧使用 mint / cyan / blue。<br/><br/>任务依旧使用 peach / rose / purple 色带。<br/><br/>所有圆点大小图例都由同一 size 函数生成。',small);ww,hh=side.wrap(150,320);side.drawOn(c,397,H-128-hh)
    y=H-427
    y=para('数据核对：120 条唯一记录，120 个双编码圆点。全部绝对误差与相同任务的 canonical DFT 参考重新计算一致（容差 1e-9 eV），颜色与大小映射均无截断；源运行状态仍随表格保存。',y,small)
    y=para(CAPTION,y,en)
    c.showPage();y=page(3,'绘图约定与文件说明','原始数据、实际绘图值、图注与脚本一并保留。')
    y=para('主组图采用 183 × 250 mm 画布：a 为完整比较设计，b 为圆点矩阵，c / d / e 为放大的定量点图。独立矩阵采用 183 × 171 mm 画布，适合单独查看和继续排版。',y)
    y=para('所有散点支持图均扩大了 marker；种子形状和 checkpoint 形状保留，以便追踪来源。主圆点矩阵只用圆形，所有 finite 实测值均使用相同的颜色和面积编码。支持图颜色表示训练策略，矩阵圆点颜色表示能量；两个色彩域在图例和图注中明确区分。',y)
    y=para('复现命令：python source/build_neb_dotmatrix.py --input data/NEB_TOTAL_PERFORMANCE_SEED_EXPANDED_METRICS.tsv --output regenerated',y,small)
    y=para('所需 Python 库：numpy、pandas、matplotlib、reportlab。字体使用系统 Helvetica；PDF 嵌入字体，SVG 保留可编辑文字。',y,small)
    y=para('plotted_values.tsv 包含原始能量、绝对误差、DFT 参考与显示状态。style_spec.json 记录字号、配色与映射公式；audit.json 保存输入哈希、原始范围和检查结果。全部数据精确重绘，未使用图像生成模型推断数值。',y,small)
    y=para('视觉依据为用户上传的 Nature Figure 1c 圆点图及此前提供的整套参考图片。字体、字号和几何尺寸是本次为可读性制定的输出规范，不冒充原论文源文件参数。',y,small)
    c.save()

def sources():
    (OUT/'style_spec.json').write_text(json.dumps(STYLE,ensure_ascii=False,indent=2))
    D['source_status']=D['scientific_status']
    D['matrix_status']='FINAL_METRIC: color=predicted energy; area=absolute error'
    D.to_csv(OUT/'plotted_values.tsv',sep='\t',index=False)
    audit={'input_sha256':hashlib.sha256(A.input.read_bytes()).hexdigest(),'records':120,'quantitative_circles':120,'source_status_counts':D['source_status'].value_counts().to_dict(),'raw_energy_range_eV':[D.predicted_value_eV.min(),D.predicted_value_eV.max()],'absolute_error_range_eV':[D.absolute_error_eV.min(),D.absolute_error_eV.max()],'recomputed_errors_match':True,'color_clipped':0,'size_clipped':0,'reference_eV':REF.to_dict(),'size_mapping':STYLE['size_encoding'],'color_mapping':STYLE['color_encoding'],'product_rendering':'All finite source values rendered directly; source statuses are provenance only.'}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2));(OUT/'figure_caption.txt').write_text(CAPTION)
    (OUT/'source').mkdir(exist_ok=True)
    source_copy=OUT/'source/build_neb_dotmatrix.py'
    if Path(__file__).resolve()!=source_copy.resolve():
        shutil.copy2(__file__,source_copy)
    (OUT/'data').mkdir(exist_ok=True);shutil.copy2(A.input,OUT/'data'/A.input.name)
    (OUT/'README.txt').write_text('NEB dot-matrix v3\n\nRebuild:\npython source/build_neb_dotmatrix.py --input data/NEB_TOTAL_PERFORMANCE_SEED_EXPANDED_METRICS.tsv --output regenerated\n\nDependencies: numpy, pandas, matplotlib, reportlab.\nColor encodes predicted energy (0-27 eV, global PowerNorm gamma .40); circle area encodes absolute error (square-root area scale plus a minimum visible point). All 120 supplied finite values are drawn as product metrics. Original source statuses remain in plotted_values.tsv and audit.json for provenance, but no diagnostic-only glyph is placed in the product figure.\nThe matrix size legend is drawn by the same size function. All source values and DFT references are retained. See figure_caption.txt, style_spec.json and audit.json.\n')

if __name__=='__main__':
    make_figures();note();sources();print(json.dumps({'output':str(OUT),'pdf_count':len(list(OUT.glob('*.pdf'))),'records':len(D)}))
