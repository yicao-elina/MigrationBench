"""Publication-style DFT-vs-MACE NEB profile for the grouped benchmark."""
from pathlib import Path
import os
import csv, json, glob
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/revision1/figures/foundation_grouped"
OUT.mkdir(exist_ok=True)
DFT = ROOT / "data/processed/foundation_grouped/dft_1-4_neb_energies.csv"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 11,
    "axes.labelsize": 14, "axes.labelweight": "bold",
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 9, "svg.fonttype": "none",
})

def read_dft():
    rows = list(csv.DictReader(DFT.open()))
    e = np.array([float(r["energy"]) for r in rows])
    # The DFT CSV energies are in the same arbitrary total-energy scale;
    # report the physically relevant relative profile.
    return np.linspace(0, 1, len(e)), (e - e[0])

def read_models():
    out = {}
    nebdir = ROOT / "data/processed/foundation_grouped/neb"
    names = ["agnesi_medium", "agnesi_stress_large", "agnesi_stress_medium",
             "agnesi_stress_small", "mh0", "mh1", "mp_0b3_medium",
             "mpa0_medium", "omat0_medium", "omat0_small"]
    for name in names:
        p = nebdir / f"{name}__1-4.json"
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
            y = np.asarray(d["relative_energies_eV"], float)
            if np.isfinite(y).all():
                out[d["model_id"]] = (np.linspace(0, 1, len(y)), y)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return out

def label(model_id):
    return {
        "mp_0b3_medium": "MP-0b3",
        "mpa0_medium": "MPA0",
        "omat0_medium": "OMAT-0",
        "mace_mp_0": "MPA0",
        "mace_mp_0_medium": "MPA0",
    }.get(model_id, model_id.replace("_", " "))

x_dft, y_dft = read_dft()
models = read_models()
fig, ax = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)
ax.plot(x_dft, y_dft, color="black", lw=2.5, marker="o", ms=6,
        label="DFT (NEB trajectory)", zorder=5)
colors = plt.get_cmap("tab10").colors
for i, (mid, (x, y)) in enumerate(models.items()):
    ax.plot(x, y, lw=1.5, ls="--", marker="^", ms=4.5,
            color=colors[i % len(colors)], label=label(mid), alpha=.9)
ax.set_xlabel("Reaction coordinate")
ax.set_ylabel("Relative energy (eV)")
ax.set_xlim(-.03, 1.03)
ax.axhline(0, color="0.65", lw=.8, zorder=0)
ax.set_title("Path 1–4: DFT NEB trajectory and MACE profiles")
ax.legend(frameon=True, ncol=2, loc="best")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for t in ax.get_xticklabels() + ax.get_yticklabels(): t.set_fontweight("bold")
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"path_1-4_energy_profiles_corrected.{ext}", dpi=300)
plt.close(fig)
print(f"DFT images: {len(y_dft)}; MACE profiles: {len(models)}")

# Family-level view: mean profile with a transparent +/-1 standard-deviation band.
families = {
    "Agnesi": [k for k in models if k.startswith("agnesi")],
    "MH": [k for k in models if k.startswith("mh")],
    "MP": [k for k in models if k.startswith("mp")],
    "OMAT": [k for k in models if k.startswith("omat")],
}
family_colors = {"Agnesi": "#4C78A8", "MH": "#F58518", "MP": "#54A24B", "OMAT": "#B279A2"}
fig, ax = plt.subplots(figsize=(7.8, 5.2), constrained_layout=True)
ax.plot(x_dft, y_dft, color="black", lw=2.5, marker="o", ms=6,
        label="DFT (NEB trajectory)", zorder=5)
for fam, ids in families.items():
    if not ids:
        continue
    ys = np.array([models[k][1] for k in ids])
    mean, std = ys.mean(axis=0), ys.std(axis=0)
    c = family_colors[fam]
    ax.plot(x_dft, mean, color=c, lw=2.2, marker="o", ms=4, label=f"{fam} (n={len(ids)})")
    ax.fill_between(x_dft, mean - std, mean + std, color=c, alpha=0.20, linewidth=0)
ax.set_xlabel("Reaction coordinate")
ax.set_ylabel("Relative energy (eV)")
ax.set_xlim(-.03, 1.03)
ax.axhline(0, color="0.65", lw=.8, zorder=0)
ax.set_title("Path 1–4: model-family mean ± 1 SD")
ax.legend(frameon=True, ncol=2, loc="best")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for t in ax.get_xticklabels() + ax.get_yticklabels(): t.set_fontweight("bold")
for ext in ("png", "svg", "pdf"):
    fig.savefig(OUT / f"path_1-4_energy_profiles_family_aggregated.{ext}", dpi=300)
plt.close(fig)
