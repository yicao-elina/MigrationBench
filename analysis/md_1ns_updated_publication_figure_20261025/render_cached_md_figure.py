#!/usr/bin/env python3
"""Render the evidence-bounded updated 1 ns MD publication figure."""
import csv, hashlib, json, math
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
RES = OUT / "results"
CACHE = Path("/tmp/md_data")
COLORS = {"Foundation":"#CF4520", "FT-600K":"#002E71", "FT-MultiT":"#67ACE5", "Scratch-600K":"#FF9F1B", "Scratch-MultiT":"#F4C95D"}
MODELS = {
    "FT-600K": [("seed123", CACHE/"ft600_123"), ("seed234", CACHE/"ft600_234"), ("seed345", CACHE/"ft600_345")],
    "FT-MultiT": [("seed123", CACHE/"ftmt_123"), ("seed234", CACHE/"ftmt_234"), ("seed345", CACHE/"ftmt_345")],
    "Foundation*": [("OMAT0-medium", CACHE/"fnd_omat"), ("MP-0b3-medium", CACHE/"fnd_mp")],
}
SCRATCH = {"Scratch-600K":{"seed123":204.4,"seed234":200.8,"seed345":200.4}, "Scratch-MultiT":{"seed123":201.2,"seed234":200.4,"seed345":200.4}}

def rows(path):
    with open(path, newline="") as f: return list(csv.DictReader(f))

def series(folder, kind):
    suffix = {"temperature":"timeseries.csv", "msd":"msd.csv", "vacf":"vacf.csv"}[kind]
    path = next(iter(folder.glob("*" + suffix)), None)
    if path is None: raise FileNotFoundError(folder / suffix)
    names = (path.name, "time_ps", "temperature_K") if kind=="temperature" else (path.name, "lag_ps", "msd_A2" if kind=="msd" else "vacf_normalized")
    rr = rows(folder/names[0]); x=[]; y=[]
    for r in rr:
        try: a,b=float(r[names[1]]),float(r[names[2]])
        except (KeyError,ValueError): continue
        if math.isfinite(a) and math.isfinite(b): x.append(a); y.append(b)
    return np.asarray(x), np.asarray(y)

def aggregate(family, kind):
    grid=np.linspace(0,1000,1001) if kind=="temperature" else np.linspace(0,800,801); vals=[]
    for _,folder in MODELS[family]:
        x,y=series(folder,kind); vals.append(np.interp(grid,x,y))
    a=np.vstack(vals); return grid,a.mean(0),a.std(0),vals

def dxy(folder, lo, hi):
    x,y=series(folder,"msd"); m=(x>=lo)&(x<=hi); return np.polyfit(x[m],y[m],1)[0]/4e4

def stats(family,lo,hi):
    v=np.asarray([dxy(folder,lo,hi) for _,folder in MODELS[family]])
    return v.mean(), (v.std(ddof=1) if len(v)>1 else 0), v

def axes_style(ax,title,letter,xlabel=None,ylabel=None):
    ax.set_title(title,loc="left",fontsize=12,fontweight="bold",pad=8)
    ax.text(-.12,1.06,letter,transform=ax.transAxes,fontsize=18,fontweight="bold",va="top")
    ax.tick_params(which="major",labelsize=9,width=1.2,length=5)
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontweight("bold")
    for s in ax.spines.values(): s.set_linewidth(1.2); s.set_color("#333333")
    ax.grid(False)
    if xlabel: ax.set_xlabel(xlabel,fontsize=10,fontweight="bold")
    if ylabel: ax.set_ylabel(ylabel,fontsize=10,fontweight="bold")

def band(ax,x,m,s,c,label):
    ax.fill_between(x,m-s,m+s,color=c,alpha=.18,linewidth=0); ax.plot(x,m,color=c,lw=2,label=label)

def build():
    mpl.rcParams.update({"font.family":"Arial","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})
    fig,aa=plt.subplots(2,3,figsize=(13.2,7.7),dpi=400); aa=aa.ravel()
    for fam in ["FT-600K","FT-MultiT","Foundation*"]:
        label="Foundation" if fam=="Foundation*" else fam; x,m,s,_=aggregate(fam,"temperature"); band(aa[0],x,m,s,COLORS[label],label)
    aa[0].axhline(600,color="#666",ls=(0,(3,2)),lw=1); aa[0].set_xlim(0,1000); aa[0].set_ylim(200,720); axes_style(aa[0],"Temperature stability","A","Time (ps)","Temperature (K)")
    for fam in ["FT-600K","FT-MultiT","Foundation*"]:
        label="Foundation" if fam=="Foundation*" else fam; x,m,s,_=aggregate(fam,"msd"); band(aa[1],x,m,s,COLORS[label],label)
    aa[1].set_xlim(0,800); aa[1].set_ylim(bottom=0); axes_style(aa[1],"Mean-square displacement","B","Lag time (ps)",r"MSD ($\AA^2$)")
    for fam in ["FT-600K","FT-MultiT","Foundation*"]:
        label="Foundation" if fam=="Foundation*" else fam; x,m,s,_=aggregate(fam,"vacf"); band(aa[2],x,m,s,COLORS[label],label)
    aa[2].axhline(0,color="#666",ls=(0,(3,2)),lw=1); aa[2].set_xlim(0,800); aa[2].set_ylim(-.25,1.05); axes_style(aa[2],"Velocity autocorrelation","C","Lag time (ps)","Normalized VACF")
    fams=["FT-600K","FT-MultiT","Foundation*"]; pos=np.arange(3)
    for i,fam in enumerate(fams):
        label="Foundation" if fam=="Foundation*" else fam; m1,s1,_=stats(fam,200,500); m2,s2,_=stats(fam,500,900)
        aa[3].errorbar(i-.1,m1,yerr=s1,fmt="o",ms=6,color=COLORS[label],mfc="white",mew=1.5,capsize=3,lw=1.2,label="200–500 ps" if i==0 else None)
        aa[3].errorbar(i+.1,m2,yerr=s2,fmt="D",ms=5,color=COLORS[label],mfc=COLORS[label],mew=1.2,capsize=3,lw=1.2,label="500–900 ps" if i==0 else None)
    aa[3].set_xticks(pos,["FT-600K","FT-MultiT","Foundation*"]); aa[3].set_yscale("log"); aa[3].set_ylim(1e-8,1e-2); axes_style(aa[3],r"$D_{xy}$ window comparison","D",None,r"$D_{xy}$ (cm$^2$ s$^{-1}$)"); aa[3].legend(frameon=False,fontsize=8,loc="upper left")
    horizon=[("FT-600K","seed123",1000,1),("FT-600K","seed234",1000,1),("FT-600K","seed345",1000,1),("FT-MultiT","seed123",1000,1),("FT-MultiT","seed234",1000,1),("FT-MultiT","seed345",1000,1),("Foundation","OMAT0-medium",1000,0),("Foundation","MP-0b3-medium",1000,0)]
    horizon += [(fam,seed,t,0) for fam,dd in SCRATCH.items() for seed,t in dd.items()]
    for i,(fam,seed,t,ok) in enumerate(horizon):
        aa[4].plot([0,t],[i,i],color=COLORS[fam],lw=2.2); aa[4].plot(t,i,marker="o" if ok else "x",color=COLORS[fam],ms=6,mew=1.8,mfc="white" if ok else None)
    aa[4].axvline(1000,color="#222",ls=(0,(4,2)),lw=1); aa[4].set_yticks([1,4,6.5,9.5,12],["FT-600K","FT-MultiT","Foundation*","Scratch-600K","Scratch-MultiT"]); aa[4].set_xlim(0,1040); aa[4].set_ylim(-1,len(horizon)); axes_style(aa[4],"Observed trajectory horizon","E","Time (ps)","Model family"); aa[4].tick_params(axis="y",length=0)
    for i,fam in enumerate(fams):
        label="Foundation" if fam=="Foundation*" else fam; m1,s1,_=stats(fam,200,500); m2,s2,_=stats(fam,500,900); r=m2/m1; er=r*math.sqrt((s1/m1)**2+(s2/m2)**2)
        aa[5].errorbar(i,r,yerr=er,fmt="o",color=COLORS[label],ms=7,mfc="white" if fam=="Foundation*" else COLORS[label],mew=1.5,capsize=3,lw=1.2)
    aa[5].axhline(1,color="#666",ls=(0,(3,2)),lw=1); aa[5].set_xticks(pos,["FT-600K","FT-MultiT","Foundation*"]); aa[5].set_yscale("log"); aa[5].set_ylim(.05,20); axes_style(aa[5],r"$D_{xy}$ window ratio","F",None,r"500–900 / 200–500 ps")
    hs=[]; ls=[]
    for label in ["Foundation","FT-600K","FT-MultiT","Scratch-600K","Scratch-MultiT"]:
        h,=aa[0].plot([],[],color=COLORS[label],lw=2.4,label=label); hs.append(h); ls.append(label)
    fig.legend(hs,ls,loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(.5,.008),fontsize=10,handlelength=2.4,columnspacing=1.6); fig.subplots_adjust(left=.075,right=.985,top=.965,bottom=.15,wspace=.32,hspace=.38)
    RES.mkdir(exist_ok=True); outs=[RES/"fig_md_1ns_updated_A_to_F.pdf",RES/"fig_md_1ns_updated_A_to_F.png",RES/"fig_md_1ns_updated_A_to_F.svg"]
    fig.savefig(outs[0],bbox_inches="tight",facecolor="white"); fig.savefig(outs[1],dpi=400,bbox_inches="tight",facecolor="white"); fig.savefig(outs[2],bbox_inches="tight",facecolor="white"); plt.close(fig)
    outrows=[]
    for fam in ["FT-600K","FT-MultiT","Foundation*"]:
        for seed,folder in MODELS[fam]: outrows.append({"family":fam,"seed_or_variant":seed,"trajectory_horizon_ps":1000,"status":"CONVERGED_1NS" if fam!="Foundation*" else "PROVISIONAL_1NS","dxy_200_500_cm2_s":dxy(folder,200,500),"dxy_500_900_cm2_s":dxy(folder,500,900)})
    for fam,dd in SCRATCH.items():
        for seed,t in dd.items(): outrows.append({"family":fam,"seed_or_variant":seed,"trajectory_horizon_ps":t,"status":"SCIENTIFIC_FAILURE_NO_DXY","dxy_200_500_cm2_s":"","dxy_500_900_cm2_s":""})
    csvout=OUT/"md_1ns_family_summary.csv"; fields=list(outrows[0]);
    with csvout.open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(outrows)
    manifest={"figure":"fig_md_1ns_updated_A_to_F","style_reference":"FIG168_MODEL_STYLE_YELLOW_SCRATCH_R04.json","palette":COLORS,"scientific_boundaries":{"FT":"accepted 1 ns three-seed trajectories","Foundation":"provisional comparator; current zero-shot continuation remains pending","Scratch":"six 600K/MultiT smoke runs fail around 200 ps; no 1 ns Dxy reported"},"cached_source_directory":str(CACHE),"outputs":[]}
    for p in outs+[csvout]: manifest["outputs"].append({"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    (OUT/"FIGURE_MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n")

if __name__ == "__main__": build()
