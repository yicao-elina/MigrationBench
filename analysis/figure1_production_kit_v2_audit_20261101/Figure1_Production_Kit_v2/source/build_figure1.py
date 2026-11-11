#!/usr/bin/env python3
"""Production-oriented Figure 1: concept cutouts + exact vector + real geometry.

python source/build_figure1.py
The resolver first uses assets/production/<ID>.png, then assets/concepts/<ID>.png.
Generated illustrations never provide plotted coordinates or fitted values.
"""
from pathlib import Path
import json,re,csv,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches, colors
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage
from PIL import Image
import fitz

ROOT=Path(__file__).resolve().parents[1]
TOK=json.loads((ROOT/'Global_Visual_Tokens.json').read_text())
INK=TOK['ui']['ink'];SECOND=TOK['ui']['secondary'];RULE=TOK['ui']['rule'];SURF=TOK['ui']['surface']
ATOM=TOK['atoms'];METHOD=TOK['methods'];TASK=TOK['tasks'];OP=TOK['operators']
plt.rcParams.update({'font.family':'Nimbus Sans','font.size':7.2,'text.color':INK,
 'axes.labelcolor':INK,'axes.edgecolor':SECOND,'axes.linewidth':.55,
 'xtick.color':INK,'ytick.color':INK,'xtick.major.width':.5,'ytick.major.width':.5,
 'xtick.labelsize':6.4,'ytick.labelsize':6.4,'xtick.major.size':2.3,'ytick.major.size':2.3,
 'pdf.fonttype':42,'svg.fonttype':'none','figure.facecolor':'white','savefig.facecolor':'white'})
for x in ['panels','vector','data','review','latex']: (ROOT/x).mkdir(exist_ok=True)
ASSETS=['A01_material_hero','A02_local_environment','A03_in_gap','A04_gap_to_ql',
 'A05_within_ql','A06_collective_sliding','A07_thermal_sampling']
ART_RESOLUTION={}
ROUTE_ANCHORS={}

def asset_path(name):
 for folder,kind in [('production','coordinate_render'),('concepts','generated_visual_reference')]:
  p=ROOT/'assets'/folder/(name+'.png')
  if p.exists():
   ART_RESOLUTION[name]={'path':str(p.relative_to(ROOT)),'kind':kind,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
   return p
 raise FileNotFoundError('Missing required image asset: '+name)

def off(ax):
 ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)

def add_rect_ax(parent,rect):
 p=parent.get_position();return parent.figure.add_axes([p.x0+rect[0]*p.width,p.y0+rect[1]*p.height,rect[2]*p.width,rect[3]*p.height])

def cutout(ax,name,rect=(0,0,1,1)):
 sub=add_rect_ax(ax,rect);off(sub)
 im=Image.open(asset_path(name)).convert('RGBA')
 bbox=im.getchannel('A').getbbox()
 if bbox:im=im.crop(bbox)
 artist=sub.imshow(im,extent=[0,1,0,1],aspect='auto',interpolation='lanczos')
 artist.set_gid(name+'_'+str(len(ax.figure.axes)))
 return sub

def keep_aspect_rect(parent,name,rect):
 p=parent.get_position();im=Image.open(asset_path(name)).convert('RGBA');bb=im.getchannel('A').getbbox()
 ratio=((bb[2]-bb[0])/(bb[3]-bb[1])) if bb else im.width/im.height
 f=parent.figure
 rw=rect[2]*p.width*f.get_figwidth();rh=rect[3]*p.height*f.get_figheight()
 factor=min(rw/(ratio*rh),1)
 if rw/rh>ratio:
  w=rect[2]*(ratio*rh/rw);return (rect[0]+(rect[2]-w)/2,rect[1],w,rect[3])
 h=rect[3]*(rw/(ratio*rh));return (rect[0],rect[1]+(rect[3]-h)/2,rect[2],h)

def art(ax,name,rect=(0,0,1,1)):
 return cutout(ax,name,keep_aspect_rect(ax,name,rect))

def gold_center(name):
 im=Image.open(asset_path(name)).convert('RGBA');bb=im.getchannel('A').getbbox()
 if bb:im=im.crop(bb)
 a=np.asarray(im).astype(float);r,g,b,alpha=[a[:,:,i] for i in range(4)]
 mask=(r>150)&(g>95)&(g>1.20*b)&(r>1.10*b)&(g<r*1.10)&(alpha>128)
 yy,xx=np.where(mask)
 if len(xx)<10:return (.5,.5)
 return (float(np.median(xx)/(im.width-1)),float(1-np.median(yy)/(im.height-1)))

def head(ax,letter,title,y=1.02):
 ax.text(0,y,letter,transform=ax.transAxes,fontsize=10.2,fontweight='bold',va='bottom')
 ax.text(.055,y+.003,title,transform=ax.transAxes,fontsize=8.4,va='bottom')

def arrow(ax,start,end,color=SECOND,lw=.75,style='->',rad=0,**kw):
 ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle=style,color=color,lw=lw,
   mutation_scale=7,connectionstyle=f'arc3,rad={rad}',shrinkA=0,shrinkB=0),**kw)

def export(fig,stem,folder='panels'):
 for ext in ['pdf','svg','png']:
  fig.savefig(ROOT/folder/f'{stem}.{ext}',dpi=450)
 plt.close(fig)

def figmm(w,h):return plt.figure(figsize=(w/25.4,h/25.4))

def read_extxyz(path):
 lines=path.read_text().splitlines();frames=[];i=0
 while i<len(lines):
  if not lines[i].strip():i+=1;continue
  n=int(lines[i]);comment=lines[i+1];chunk=lines[i+2:i+n+2]
  latt=re.search(r'Lattice="([^"]+)"',comment)
  if not latt:raise ValueError('Explicit cell required: '+path.name)
  cell=np.array(list(map(float,latt.group(1).split()))).reshape(3,3)
  syms=[x.split()[0] for x in chunk];pos=np.array([list(map(float,x.split()[1:4])) for x in chunk])
  ids=[j for j,s in enumerate(syms) if s=='Cr']
  if len(ids)!=1:raise ValueError(f'Expected one Cr in {path.name}: {ids}')
  frames.append(dict(cell=cell,species=syms,pos=pos,cr_index=ids[0]));i+=n+2
 return frames

def geometry_data():
 configs=[('in_gap','Within gap','within_gap__ideal2_cr_canonical_in_gap.extxyz'),
  ('gap_to_ql','Entry input','gap_to_ql__ideal1_cr_canonical_gap_to_ql.extxyz'),
  ('within_ql','Within QL','within_ql__neb_3_sb2te3.extxyz')]
 rows=[];audit=[]
 for key,label,fn in configs:
  path=ROOT/'inputs'/fn;frames=read_extxyz(path);cell=frames[0]['cell']
  assert all(np.allclose(fr['cell'],cell,atol=1e-7) for fr in frames)
  norm=np.cross(cell[0],cell[1]);norm=norm/np.linalg.norm(norm)
  cr=np.array([fr['pos'][fr['cr_index']] for fr in frames]);inv=np.linalg.inv(cell)
  delta=np.diff(cr,axis=0);frac=delta@inv;frac-=np.rint(frac);steps=frac@cell
  unwrapped=np.vstack([np.zeros(3),np.cumsum(steps,axis=0)])
  dz=unwrapped@norm;para=unwrapped-np.outer(dz,norm);lateral=np.linalg.norm(para,axis=1)
  arc=np.r_[0,np.cumsum(np.linalg.norm(steps,axis=1))]
  for j in range(len(frames)):
   rows.append(dict(route=key,label=label,image_index=j,dx_A=unwrapped[j,0],dy_A=unwrapped[j,1],
    dz_cartesian_A=unwrapped[j,2],normal_displacement_A=dz[j],lateral_displacement_A=lateral[j],
    sampled_arc_length_A=arc[j],source_file=fn,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  audit.append(dict(route=key,file=fn,frames=len(frames),cr_count_per_frame=1,
    cr_index=frames[0]['cr_index'],atoms_per_frame=len(frames[0]['species']),
    net_displacement_A=float(np.linalg.norm(unwrapped[-1])),sampled_length_A=float(arc[-1]),
    method='consecutive-image minimum-image unwrapping; normal=a cross b',
    energy_rows_joined=False,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 out=pd.DataFrame(rows);out.to_csv(ROOT/'data/Cr_input_geometry.csv',index=False)
 (ROOT/'data/geometry_audit.json').write_text(json.dumps(audit,indent=2))
 return out

GEO=geometry_data()

def material(ax,letter='a',title='Layered target system'):
 off(ax);head(ax,letter,title)
 aa=art(ax,'A01_material_hero',(.07,.19,.84,.72))
 for y0,y1 in [(.59,.96),(.04,.42)]:
  aa.plot([-.025,-.045,-.045,-.025],[y0,y0,y1,y1],c=SECOND,lw=.5,clip_on=False)
  aa.text(-.085,(y0+y1)/2,'QL',rotation=90,ha='center',va='center',fontsize=6.2,color=SECOND,clip_on=False)
 aa.annotate('',xy=(1.025,.56),xytext=(1.025,.44),arrowprops=dict(arrowstyle='<->',color=SECOND,lw=.55,mutation_scale=5),annotation_clip=False)
 aa.text(1.09,.50,'vdW gap',rotation=90,ha='center',va='center',fontsize=6.2,color=SECOND,clip_on=False)
 ax.text(.5,.105,r'Cr-doped Sb$_2$Te$_3$',ha='center',fontsize=8.2)
 for x,s in zip([.20,.48,.76],['Te','Sb','Cr']):
  ax.scatter(x,.015,s=23,c=ATOM[s],ec='white',lw=.25,clip_on=False)
  ax.text(x+.06,.015,s,fontsize=7,va='center')

def weight_glyph(ax,x,y,c):
 for j in range(3):
  for k in range(2):
   ax.add_patch(patches.Rectangle((x+j*.016,y+k*.020),.012,.014,fc=c,ec='none',alpha=.45+.18*k))

def training_workflow(ax,letter='b',title='Target adaptation and inference'):
 off(ax);head(ax,letter,title)
 # Images are illustrative; all arrows, model identities and labels are exact vector objects.
 art(ax,'A07_thermal_sampling',(.005,.64,.20,.24))
 ax.text(.105,.91,'Target AIMD',ha='center',fontsize=7.1)
 ax.text(.105,.59,'E / F labels',ha='center',fontsize=6.8,color=SECOND)
 weight_glyph(ax,.075,.335,METHOD['zero_shot'])
 ax.text(.105,.26,'Foundation prior',ha='center',fontsize=6.6,color=TOK['method_edges']['zero_shot'])
 ax.text(.105,.20,r'$\theta_{\rm FM}$',ha='center',fontsize=10,color=TOK['method_edges']['zero_shot'])
 # Column labels are target-data regimes, not zero-shot regimes.
 for x,txt,filled in [(.405,'600 K',False),(.655,'Multi-T',True)]:
  ax.scatter(x-.050,.87,s=17,fc=SECOND if filled else 'white',ec=SECOND,lw=.7)
  ax.text(x-.018,.87,txt,fontsize=7.0,va='center')
 ax.plot([.29,.79],[.79,.79],c=RULE,lw=.7)
 ax.plot([.29,.79],[.17,.17],c=RULE,lw=.7)
 ax.text(.27,.60,'Random\ninit.',fontsize=6.7,ha='right',va='center',linespacing=1.1,color=METHOD['scratch'])
 ax.text(.27,.34,'FM\ninit.',fontsize=6.7,ha='right',va='center',linespacing=1.1,color=METHOD['fine_tuning'])
 for y,cc,prefix in [(.60,METHOD['scratch'],'S'),(.34,METHOD['fine_tuning'],'FT')]:
  for x,suffix,filled in [(.405,'600',False),(.655,'MT',True)]:
   ax.add_patch(patches.FancyBboxPatch((x-.10,y-.095),.20,.19,boxstyle='round,pad=0,rounding_size=.012',fc='white',ec=cc,lw=.75))
   ax.text(x,y+.027,prefix+suffix,ha='center',va='center',fontsize=8.1,fontweight='bold',color=cc)
   for dx,m in zip([-.035,0,.035],['o','^','s']):ax.scatter(x+dx,y-.045,s=12,marker=m,fc=cc if filled else 'white',ec=cc,lw=.65)
 arrow(ax,(.195,.75),(.405,.78),SECOND,lw=.6,rad=-.10)
 arrow(ax,(.195,.75),(.655,.78),SECOND,lw=.6,rad=-.10)
 arrow(ax,(.195,.35),(.285,.35),TOK['method_edges']['zero_shot'],lw=.7)
 # Shared inference interface, no equality of architecture or capacity asserted.
 art(ax,'A02_local_environment',(.845,.69,.14,.20))
 ax.text(.915,.60,'E / F',ha='center',fontsize=9)
 ax.text(.915,.51,'interface',ha='center',fontsize=6.7,color=SECOND)
 for y,cc in [(.60,METHOD['scratch']),(.34,METHOD['fine_tuning'])]:
  yy=y-.117
  for xx in [.405,.655]:
   ax.plot([xx,xx],[y-.095,yy],c=cc,lw=.65)
  ax.plot([.405,.80],[yy,yy],c=cc,lw=.65)
  ax.plot([.80,.83,.83],[yy,yy,.56],c=cc,lw=.65)
 arrow(ax,(.83,.56),(.88,.56),SECOND,lw=.7)
 # Independent bypass: no target fitting is applied to the zero-shot reference.
 ax.plot([.105,.105,.88],[.15,.055,.055],c=METHOD['zero_shot'],lw=.9)
 arrow(ax,(.88,.055),(.96,.055),METHOD['zero_shot'],lw=.9)
 ax.text(.33,.09,'Zero-shot: no target fitting',fontsize=6.7,color=TOK['method_edges']['zero_shot'])
 ax.text(.92,.12,'E / F',fontsize=7.5,ha='center')

def task_atlas(ax,letter='c',title='Process probes across distinct configuration changes'):
 off(ax);head(ax,letter,title)
 items=[('Static','A02_local_environment','static','Pointwise E / F'),
 ('Dynamics','A07_thermal_sampling','md','Rollout stability'),
 ('In-gap','A03_in_gap','in_gap','Local motion'),
 ('Gap-to-QL','A04_gap_to_ql','gap_to_ql','Entry environment'),
 ('Within-QL','A05_within_ql','within_ql','Intralayer motion'),
 ('Sliding','A06_collective_sliding','sliding','Collective motion')]
 for j,(label,aid,key,desc) in enumerate(items):
  x=j/6;w=1/6
  ax.plot([x+.008,x+w-.010],[.91,.91],c=TASK[key],lw=1.5,solid_capstyle='butt')
  ax.text(x+w/2,.98,label,fontsize=7.1,ha='center',va='center',color=TASK[key])
  a=art(ax,aid,(x+.012,.20,w-.024,.64))
  # Purely schematic action arrows are separate editable vector elements.
  if j in [2,3,4]:
   gx,gy=gold_center(aid)
   end=(min(.93,gx+.30),gy) if j==2 else (min(.93,gx+.08),min(.93,gy+.27)) if j==3 else (min(.93,gx+.29),min(.93,gy+.08))
   arrow(a,(gx,gy),end,TASK[key],lw=1.2,rad=-.10)
   ROUTE_ANCHORS[aid]={'start_normalized':[gx,gy],'end_normalized':list(end),'meaning':'schematic direction only; not a fitted MEP'}
  if j==5:
   arrow(a,(.26,.95),(.77,.95),TASK[key],lw=1.0)
   arrow(a,(.75,.05),(.24,.05),TASK[key],lw=1.0)
  ax.text(x+w/2,.095,desc,fontsize=6.5,ha='center',color=SECOND)
 ax.text(0,-.07,'Task labels define probes; they are not a calibrated scalar of distribution shift.',fontsize=6.5,color=SECOND)

def path_glyph(ax,x0,x1,y,cc,curved=False):
 xs=np.linspace(x0,x1,4);ys=y+(.075*np.sin(np.linspace(0,np.pi,4)) if curved else np.zeros(4))
 ax.plot(xs,ys,c=cc,lw=.8)
 ax.scatter(xs,ys,s=13,c=ATOM['Cr'],ec='white',lw=.3,zorder=3)

def operators(ax,letter='d',title='Two measurement operators'):
 off(ax);head(ax,letter,title)
 # No energy curves: only configurations and model operations.
 for left,sub,formula,col in [(0,'Fixed reference images',r'$E_\theta(q_s^{\rm ref})$',OP['fixed']),
                              (.535,'Model-relaxed images',r'$E_\theta(q_s^{\theta})$',OP['self_relaxed'])]:
  ax.plot([left,left+.46],[.91,.91],c=col,lw=2.5,solid_capstyle='butt')
  ax.text(left+.23,.79,sub,fontsize=7.2,ha='center')
  path_glyph(ax,left+.05,left+.41,.57,SECOND,curved=left>0)
  if left>0:
   ax.plot([left+.05,left+.41],[.50,.50],c=RULE,lw=.65,ls=(0,(2,2)))
   arrow(ax,(left+.24,.49),(left+.24,.62),SECOND,lw=.65)
  ax.text(left+.23,.33,formula,fontsize=12,ha='center')
  ax.text(left+.23,.13,'Same geometries; score E / F' if left==0 else 'Optimize geometry; retain NC',
          fontsize=6.5,ha='center',color=SECOND)
 ax.text(0,-.04,'Schematic image paths. Optimizer convergence and path validity are separate checks.',fontsize=6.3,color=SECOND)

def geometry(ax,letter='e',title='Geometry of supplied inputs',mini=False):
 if letter:
  ax.text(-.20,1.19,letter,transform=ax.transAxes,fontsize=10.2,fontweight='bold',va='bottom')
  ax.text(-.09,1.193,title,transform=ax.transAxes,fontsize=8.2,va='bottom')
 for key,style,label in [('in_gap','o','Within gap'),('gap_to_ql','^','Entry input'),('within_ql','s','Within QL')]:
  q=GEO[GEO.route==key]
  ax.plot(q.lateral_displacement_A,q.normal_displacement_A,c=TASK[key],lw=.8,
       marker=style,markersize=2.4,markeredgewidth=.3,label=label)
 ax.set_xlim(-.08,2.75);ax.set_ylim(-.10,3.75);ax.set_xticks([0,1,2]);ax.set_yticks([0,1,2,3])
 ax.set_xlabel(r'$\Delta r_\parallel$ (Å)',fontsize=7.1,labelpad=1.5)
 ax.set_ylabel(r'$\Delta r_\perp$ (Å)',fontsize=7.1,labelpad=1.5)
 for s in ['top','right']:ax.spines[s].set_visible(False)
 ax.tick_params(pad=1.5)
 ax.legend(frameon=False,fontsize=6.1,loc='upper left',handlelength=1.1,handletextpad=.4,labelspacing=.25)

def geometry_panel():
 # An adjacent zoom panel keeps every primary data point unobscured.
 f=figmm(150,80);geometry(f.add_axes([.11,.18,.50,.64]))
 ax=f.add_axes([.73,.26,.23,.44]);q=GEO[GEO.route=='gap_to_ql']
 ax.plot(q.lateral_displacement_A,q.normal_displacement_A,c=TASK['gap_to_ql'],lw=.8,marker='^',ms=3)
 ax.set_title('Entry input (zoom)',fontsize=7.5,pad=5)
 ax.set_xlim(-.002,.028);ax.set_ylim(-.01,.16);ax.set_xticks([0,.01,.02]);ax.set_yticks([0,.05,.10,.15])
 ax.set_xlabel(r'$\Delta r_\parallel$ (Å)',fontsize=7.1,labelpad=2)
 ax.set_ylabel(r'$\Delta r_\perp$ (Å)',fontsize=7.1,labelpad=2)
 for s in ['top','right']:ax.spines[s].set_visible(False)
 ax.tick_params(labelsize=6.5,pad=2)
 export(f,'Figure1e_Actual_geometry')

def algorithm_interface(ax):
 off(ax);head(ax,'','Local geometry to a conservative energy / force interface')
 art(ax,'A02_local_environment',(.00,.20,.18,.68))
 ax.text(.09,.09,'Local environment',ha='center',fontsize=7.5)
 arrow(ax,(.19,.56),(.26,.56),SECOND)
 for r in range(3):
  for c in range(7):ax.add_patch(patches.Rectangle((.275+c*.019,.38+r*.085),.014,.055,fc='#98AFC1',ec='none',alpha=.55+r*.15))
 ax.text(.34,.81,'Equivariant features',ha='center',fontsize=8)
 ax.text(.34,.11,r'$h_i^{(\ell+1)}=U_\theta(h_i^{(\ell)},m_i^{(\ell)})$',ha='center',fontsize=10)
 arrow(ax,(.43,.56),(.50,.56),SECOND)
 for k in range(5):ax.add_patch(patches.Rectangle((.52+k*.021,.40),.015,.28,fc='#9BB2C3',ec='none'))
 ax.text(.57,.81,'Atomic readout',ha='center',fontsize=8)
 ax.text(.57,.11,r'$E_\theta=\sum_i\varepsilon_\theta(h_i)$',ha='center',fontsize=10)
 arrow(ax,(.66,.56),(.73,.56),SECOND)
 ax.text(.85,.68,r'$E_\theta(q)$',ha='center',fontsize=15)
 ax.text(.85,.38,r'$\mathbf{F}_{i,\theta}=-\nabla_{\mathbf{r}_i}E_\theta$',ha='center',fontsize=13)
 ax.text(.85,.11,'Shared observable definitions',ha='center',fontsize=7,color=SECOND)

def make_main():
 f=figmm(183,198)
 # Exact rectangles in mm, y measured from the top; panel labels are separate.
 layout={'a':[4,8,58,58],'b':[68,8,111,58],'c':[4,81,175,45],
         'd':[4,143,101,40],'e':[125,148,51,33]}
 def axmm(r):
  x,y,w,h=r;return f.add_axes([x/183,1-(y+h)/198,w/183,h/198])
 aa=axmm(layout['a']);aa.set_gid('panel_a');material(aa)
 bb=axmm(layout['b']);bb.set_gid('panel_b');training_workflow(bb)
 cc=axmm(layout['c']);cc.set_gid('panel_c');task_atlas(cc)
 dd=axmm(layout['d']);dd.set_gid('panel_d');operators(dd)
 ee=axmm(layout['e']);ee.set_gid('panel_e');geometry(ee)
 f.text(.022,.016,'Visual target: replace the 3D reference cutouts with coordinate-based renders. Panel e uses supplied coordinates.',fontsize=6.2,color=SECOND)
 (ROOT/'data/panel_layout_mm.json').write_text(json.dumps({'canvas_mm':[183,198],'rectangles_xywh_from_top':layout},indent=2))
 # Exact same-canvas art / annotation plates for native LaTeX composition.
 original=[]
 for ax in f.axes:
  for item in ax.get_children():
   original.append((item,item.get_visible()))
   if not isinstance(item,AxesImage):item.set_visible(False)
 for t in f.texts:original.append((t,t.get_visible()));t.set_visible(False)
 f.savefig(ROOT/'vector/Figure1_Artwork_Only.pdf',transparent=True,dpi=450)
 for item,visible in original:item.set_visible(visible)
 images=[item for ax in f.axes for item in ax.get_children() if isinstance(item,AxesImage)]
 for item in images:item.set_visible(False)
 f.savefig(ROOT/'vector/Figure1_Vector_Overlay.pdf',transparent=True,dpi=450)
 f.savefig(ROOT/'vector/Figure1_Vector_Overlay.svg',transparent=True)
 for item in images:item.set_visible(True)
 export(f,'Figure1_Ideal_Visual',folder='.')

def standalone_panels():
 f=figmm(84,84);material(f.add_axes([.04,.08,.92,.80]));export(f,'Figure1a_Material')
 f=figmm(150,84);training_workflow(f.add_axes([.035,.07,.93,.80]));export(f,'Figure1b_Training_workflow')
 f=figmm(183,60);task_atlas(f.add_axes([.025,.18,.95,.64]));export(f,'Figure1c_Task_atlas')
 f=figmm(132,64);operators(f.add_axes([.035,.16,.93,.68]));export(f,'Figure1d_Operators')
 geometry_panel()
 f=figmm(183,54);algorithm_interface(f.add_axes([.025,.12,.95,.70]));export(f,'MACE_Interface_Vector',folder='vector')

def contact_sheet():
 f=figmm(275,190)
 labels=['A01  Material / two QLs','A02  Local environment','A07  Thermal ensemble',
         'A03  In-gap','A04  Entry interface','A05  Within QL','A06  Collective sliding']
 ids=[ASSETS[i] for i in [0,1,6,2,3,4,5]]
 for i,(name,label) in enumerate(zip(ids,labels)):
  x=i%4;y=i//4;a=f.add_axes([.025+x*.245,.51-y*.455,.225,.39]);off(a)
  a.add_patch(patches.Rectangle((0,0),1,1,fc='white',ec=RULE,lw=.5))
  art(a,name,(.03,.08,.94,.80));a.text(.04,.96,label,ha='left',va='top',fontsize=8)
 a=f.add_axes([.025+3*.245,.51-1*.455,.225,.39]);off(a)
 a.text(.02,.87,'Replacement rule',fontsize=10,fontweight='bold')
 a.text(.02,.70,'Keep the same asset ID.\nRender from actual geometry.\nExport clean PNG RGBA.\nKeep text and arrows in vector.',fontsize=8,linespacing=1.6,va='top')
 a.text(.02,.24,'Te  pink   Sb  cyan   Cr  gold',fontsize=8,color=SECOND)
 f.text(.025,.955,'Figure 1 | generated visual references and replacement slots',fontsize=15,fontweight='bold')
 f.text(.025,.016,'All 3D illustrations on this sheet are conceptual art references; no plotted value is derived from them.',fontsize=7,color=SECOND)
 f.savefig(ROOT/'Figure1_Elements_Contact.png',dpi=200);plt.close(f)

def palette():
 f=figmm(183,153);ax=f.add_axes([.04,.05,.92,.88]);off(ax)
 ax.text(0,1.03,'MigrationBench | shared visual vocabulary',fontsize=13,fontweight='bold')
 groups=[('Atoms',ATOM),('Model strategies',METHOD),('Task identity',TASK),('Measurement operator',OP),('Evidence state',TOK['evidence'])]
 yy=.92
 for label,vals in groups:
  ax.text(0,yy,label,fontsize=9,fontweight='bold',va='center')
  for j,(name,cc) in enumerate(vals.items()):
   col=j%3;row=j//3;x=.27+col*.25;y=yy-row*.075
   ax.add_patch(patches.Rectangle((x,y-.019),.030,.038,fc=cc,ec='none'))
   ax.text(x+.044,y,name.replace('_',' ')+'\n'+cc,fontsize=7,va='center',linespacing=1.25)
  yy-=.125+((len(vals)-1)//3)*.075
 ax.text(0,.07,'Thermal regime: open = 600 K; filled = Multi-T.\nSeed: circle = 123; triangle = 234; square = 345.  Checkpoint: diamond.\nEnergy color is sequential; evidence status is never encoded as energy.',fontsize=8,linespacing=1.5,color=SECOND)
 f.savefig(ROOT/'Global_Visual_Standard.pdf');f.savefig(ROOT/'Global_Visual_Standard.png',dpi=220);plt.close(f)
 with open(ROOT/'Global_Visual_Tokens.csv','w') as out:
  w=csv.writer(out);w.writerow(['channel','element','hex_sRGB'])
  for channel in ['atoms','methods','method_edges','tasks','operators','evidence','ui']:
   for k,v in TOK[channel].items():w.writerow([channel,k,v])

def audit():
 (ROOT/'data/asset_resolution.json').write_text(json.dumps(ART_RESOLUTION,indent=2))
 (ROOT/'data/route_arrow_anchors.json').write_text(json.dumps(ROUTE_ANCHORS,indent=2))
 a={'concept_assets':list(ART_RESOLUTION),'coordinates_from_imagegen':False,
    'geometry_rows':len(GEO),'geometry_source_files':GEO.source_file.nunique(),
    'legend_seed_counts':'three training seeds per fitted cell; zero-shot is independent',
    'no_force_energy_or_PCA_values_imputed':True,'plot_units':'Å',
    'input_geometry_is_not_a_measured_OOD_distance':True}
 (ROOT/'audit.json').write_text(json.dumps(a,indent=2))
 d=fitz.open(ROOT/'Figure1_Ideal_Visual.pdf')
 d[0].get_pixmap(matrix=fitz.Matrix(190/72,190/72),alpha=False).save(ROOT/'review/Figure1_main_190dpi.png')
 for name in ['Figure1b_Training_workflow','Figure1c_Task_atlas','Figure1d_Operators','Figure1e_Actual_geometry']:
  d=fitz.open(ROOT/'panels'/f'{name}.pdf')
  d[0].get_pixmap(matrix=fitz.Matrix(140/72,140/72),alpha=False).save(ROOT/'review'/f'{name}.png')
 print(json.dumps(a,indent=2))

if __name__=='__main__':
 make_main();standalone_panels();contact_sheet();palette();audit()
