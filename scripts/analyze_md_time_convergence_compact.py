#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def fit(x, y):
    if len(x) < 5 or np.ptp(x) <= 0: return {"n": int(len(x)), "slope_A2_per_ps": None, "Dxy_cm2_s": None, "r2": None}
    slope, intercept = np.polyfit(x, y, 1); pred = slope*x + intercept; sst=np.sum((y-y.mean())**2)
    return {"n": int(len(x)), "slope_A2_per_ps": float(slope), "Dxy_cm2_s": float(slope/4*1e-4), "r2": float(1-np.sum((y-pred)**2)/sst) if sst>0 else None}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--original",type=Path,required=True); p.add_argument("--segment",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--name",required=True); p.add_argument("--start-step",type=int,default=200000); p.add_argument("--cell",required=True); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    cell=np.asarray([float(x) for x in a.cell.split()],float).reshape(3,3); rows=[]
    for path in (a.original,a.segment):
        data=np.loadtxt(path); data=np.atleast_2d(data); rows.append(data)
    data=np.vstack(rows); order=np.argsort(data[:,0],kind="stable"); data=data[order]; data=data[np.r_[True,np.diff(data[:,0])>0]]
    steps=data[:,0].astype(int); xyz=data[:,1:].reshape(len(data),-1,3); mask=steps>=a.start_step; steps=steps[mask]; xyz=xyz[mask]
    frac=np.einsum("nij,jk->nik",xyz,np.linalg.inv(cell)); uw=np.empty_like(frac); uw[0]=frac[0]
    for i in range(1,len(frac)): d=frac[i]-frac[i-1]; d-=np.rint(d); uw[i]=uw[i-1]+d
    pos=np.einsum("nij,jk->nik",uw,cell)[:,:,:2]; t=(steps-steps[0])*.001; msd=np.mean(np.sum((pos-pos[0])**2,axis=2),axis=1)
    cum=[]
    for w in (50.,100.,200.,500.): s=t<=w+1e-9; row={"window_ps":w,"actual_end_ps":float(t[s][-1]) if s.any() else 0.}; row.update(fit(t[s],msd[s])); cum.append(row)
    blocks=[]
    for i in range(int(np.floor(t[-1]/50))):
        lo=i*50.; hi=(i+1)*50.; s=(t>=lo-1e-9)&(t<=hi+1e-9); local=np.mean(np.sum((pos[s]-pos[s][0])**2,axis=2),axis=1); row={"block_index":i+1,"start_ps":lo,"end_ps":hi}; row.update(fit(t[s]-lo,local)); blocks.append(row)
    summary={"model":a.name,"original":str(a.original),"segment":str(a.segment),"dopant":"Cr","dopant_count":int(xyz.shape[1]),"natoms":2050,"pbc":[True,True,True],"frames":int(len(steps)),"start_step":int(steps[0]),"last_step":int(steps[-1]),"duration_ps":float(t[-1]),"time_axis_reconstructed_for_legacy_original":True,"cumulative_windows":cum,"blocks_50ps":blocks}
    (a.out/"time_convergence.json").write_text(json.dumps(summary,indent=2)+"\n")
    for name,rs in (("cumulative_windows.csv",cum),("blocks_50ps.csv",blocks)):
        with (a.out/name).open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rs[0]) if rs else ["empty"]); w.writeheader(); w.writerows(rs)
    plt.rcParams.update({"font.family":"Arial","font.size":11,"axes.linewidth":1.4}); fig,ax=plt.subplots(1,2,figsize=(10,3.9),constrained_layout=True); ax[0].plot(t,msd,color="#245a9a",lw=1.7); ax[0].set(xlabel="Time after 200 ps (ps)",ylabel="MSD$_{xy}$ (Å$^2$");
    if blocks: ax[1].plot([f"{int(x['start_ps'])}–{int(x['end_ps'])}" for x in blocks],[x['Dxy_cm2_s'] for x in blocks],marker="o",color="#b24d35",lw=1.5); ax[1].tick_params(axis="x",rotation=55)
    ax[1].set(xlabel="Chronological 50 ps block",ylabel="D$_{xy}$ (cm$^2$ s$^{-1}$)"); [x.tick_params(width=1.2,length=4) for x in ax]
    for ext in ("png","pdf","svg"): fig.savefig(a.out/f"time_convergence.{ext}",dpi=300 if ext=="png" else None)
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
