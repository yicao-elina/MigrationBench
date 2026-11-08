#!/usr/bin/env python3
"""Phase-3 diagnostics, using the Plot Atlas time-course/scatter grammar."""
import csv, math, re
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.linewidth": 1.5,
                     "xtick.major.width": 1.3, "ytick.major.width": 1.3})
BLUE="#2F5D8C"; ORANGE="#D17A22"; RED="#B33A3A"; GRAY="#4C566A"

rows=list(csv.DictReader((ROOT/"ordered_block_audit.csv").open()))
r81=sorted((r for r in rows if r["model"]=="81"), key=lambda r:int(r["step"]))
steps=[int(r["step"]) for r in r81]; glob=[float(r["max_force_eV_A"]) for r in r81]; cr=[float(r["cr_max_force_eV_A"]) for r in r81]
fig, ax = plt.subplots(figsize=(5.8,3.6))
ax.plot(steps, glob, "o-", color=RED, lw=2.0, ms=5, label="all atoms")
ax.plot(steps, cr, "o-", color=BLUE, lw=2.0, ms=5, label="Cr only")
ax.set_xlabel("MD step"); ax.set_ylabel("MACE force norm (eV Å$^{-1}$)")
ax.set_yscale("log"); ax.set_xticks(steps); ax.tick_params(axis="x", rotation=30)
ax.legend(frameon=False, loc="upper left"); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig(FIG/"mace_force_progression_81.png", dpi=220); fig.savefig(FIG/"mace_force_progression_81.svg"); plt.close(fig)

def read_xyz(p):
    ls=p.read_text().splitlines(); n=int(ls[0]); out=[]
    for line in ls[2:2+n]:
        f=line.split(); out.append((f[0], math.sqrt(sum(float(x)**2 for x in f[7:10]))))
    return out
xyz=read_xyz(ROOT/"inputs/ordered_block7/snapshot_step_462000_same_site81_ordered.extxyz")
idx=list(range(1,len(xyz)+1)); vals=[x[1] for x in xyz]; colors=[ORANGE if s=="Cr" else GRAY for s,_ in xyz]
fig, ax=plt.subplots(figsize=(6.0,3.5)); ax.scatter(idx, vals, s=13, c=colors, alpha=.85, linewidths=0)
ax.set_xlabel("atom index in ordered 81-atom block"); ax.set_ylabel("MACE force norm (eV Å$^{-1}$)")
ax.set_yscale("log"); ax.set_title("Step 462000: local block force landscape", loc="left", fontsize=10, pad=8)
ax.text(.99,.97,"orange = Cr", transform=ax.transAxes, ha="right", va="top", color=ORANGE, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); fig.tight_layout()
fig.savefig(FIG/"mace_force_landscape_step_462000_81.png", dpi=220); fig.savefig(FIG/"mace_force_landscape_step_462000_81.svg"); plt.close(fig)
