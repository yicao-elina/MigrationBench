#!/usr/bin/env python
"""Render the isolated analysis release into manuscript-ready figures/tables."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

COLORS = {
    "2-naive-fine-tuning": "#3B6EA8",
    "3-multi_T-fine-tuning": "#B34F3C",
    "0-foundation-omat": "#2F8F83",
    "1-from-scratch": "#6B6B6B",
}
LABELS = {
    "0-foundation-omat": "Foundation-OMAT",
    "1-from-scratch": "From-scratch",
    "2-naive-fine-tuning": "FT-600K",
    "3-multi_T-fine-tuning": "FT-MultiT",
}


def style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for s in ax.spines.values(): s.set_linewidth(1.4)
    ax.tick_params(width=1.2, length=4, labelsize=10)
    for t in ax.get_xticklabels() + ax.get_yticklabels(): t.set_fontweight("bold")


def load_metrics(root, name):
    p = Path(root) / name / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def read_csv(path):
    with open(path, newline="") as f: return list(csv.DictReader(f))


def make(root, out):
    root, out = Path(root), Path(out); out.mkdir(parents=True, exist_ok=True)
    name_map = {
        "policy": "Manuscript-facing names are canonical; raw campaign IDs and paths are retained in audit/provenance files.",
        "models": [
            {"raw_id": raw, "display_name": LABELS[raw], "raw_campaign_directory": raw}
            for raw in ["0-foundation-omat", "1-from-scratch", "2-naive-fine-tuning", "3-multi_T-fine-tuning"]
        ],
    }
    encoded_name_map = json.dumps(name_map, indent=2) + "\n"
    (root / "model_name_map.json").write_text(encoded_name_map)
    (out / "model_name_map.json").write_text(encoded_name_map)
    stable = [n for n in ["0-foundation-omat", "2-naive-fine-tuning", "3-multi_T-fine-tuning"] if load_metrics(root, n)]
    metrics = {n: load_metrics(root, n) for n in stable}

    fig, axs = plt.subplots(2, 2, figsize=(8.2, 6.3), constrained_layout=True)
    for name in stable:
        rows = read_csv(root / name / "timeseries.csv")
        t = np.array([float(r["time_ps"]) for r in rows])
        T = np.array([float(r["temperature_K"]) for r in rows])
        F = np.array([float(r["max_force_eV_A"]) for r in rows])
        D = np.array([float(r["min_distance_A"]) for r in rows])
        E = np.array([float(r["energy_eV"]) for r in rows])
        c = COLORS[name]; lab = LABELS[name]
        axs[0,0].plot(t, T, lw=1.0, color=c, label=lab)
        axs[0,1].plot(t, F, lw=1.0, color=c, label=lab)
        axs[1,0].plot(t, D, lw=1.0, color=c, label=lab)
        axs[1,1].plot(t, E - E[0], lw=1.0, color=c, label=lab)
    axs[0,0].axhline(600, ls="--", lw=1, color="0.35"); axs[0,0].set_ylabel("Temperature (K)")
    axs[0,1].set_yscale("log"); axs[0,1].set_ylabel("Max force (eV/Å)")
    axs[1,0].axhline(1.0, ls="--", lw=1, color="0.35"); axs[1,0].set_ylabel("Min distance (Å)")
    axs[1,1].set_ylabel("Δ energy (eV)")
    for ax in axs.flat: ax.set_xlabel("Time (ps)"); style(ax)
    axs[0,0].legend(frameon=False, fontsize=9, loc="best")
    for i, ax in enumerate(axs.flat): ax.text(-0.14, 1.05, "abcd"[i], transform=ax.transAxes, fontweight="bold", fontsize=12)
    fig.savefig(out / "fig_stability.pdf"); fig.savefig(out / "fig_stability.svg"); fig.savefig(out / "fig_stability.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for name in stable:
        rows = read_csv(root / name / "msd.csv")
        t = np.array([float(r["lag_ps"]) for r in rows]); y = np.array([float(r["msd_A2"]) for r in rows])
        ax.plot(t, y, lw=1.8, color=COLORS[name], label=LABELS[name])
    ax.axvspan(200, 500, color="0.85", alpha=0.35, lw=0)
    ax.axvspan(500, 900, color="0.92", alpha=0.5, lw=0)
    ax.set(xlabel="Lag time (ps)", ylabel="Cr MSD (Å²)"); ax.legend(frameon=False, fontsize=9); style(ax)
    fig.tight_layout(); fig.savefig(out / "fig_msd.pdf"); fig.savefig(out / "fig_msd.svg"); fig.savefig(out / "fig_msd.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for name in stable:
        rows = read_csv(root / name / "vacf.csv")
        t = np.array([float(r["lag_ps"]) for r in rows]); y = np.array([float(r["vacf_normalized"]) for r in rows])
        ax.plot(t, y, lw=1.5, color=COLORS[name], label=LABELS[name])
    ax.axhline(0, ls="--", lw=1, color="0.35"); ax.set(xlabel="Lag time (ps)", ylabel="Normalized VACF")
    ax.legend(frameon=False, fontsize=9); style(ax); fig.tight_layout()
    fig.savefig(out / "fig_vacf.pdf"); fig.savefig(out / "fig_vacf.svg"); fig.savefig(out / "fig_vacf.png", dpi=300); plt.close(fig)

    failure = root / "1-from-scratch" / "failure_scan.csv"
    if failure.exists():
        rows = read_csv(failure); t = np.array([float(r["time_ps"]) for r in rows]);
        fig, axs = plt.subplots(3, 1, figsize=(6.2, 7.1), sharex=True, constrained_layout=True)
        vals = [np.array([float(r["temperature_K"]) for r in rows]), np.array([float(r["max_force_eV_A"]) for r in rows]), np.array([float(r["min_distance_A"]) for r in rows])]
        labels = ["Temperature (K)", "Max force (eV/Å)", "Min distance (Å)"]
        thresholds = [5000, 100, 0.5]
        for i, ax in enumerate(axs):
            ax.plot(t, vals[i], color=COLORS["1-from-scratch"], lw=1.5)
            ax.axhline(thresholds[i], color="#B34F3C", ls="--", lw=1.1)
            ax.set_ylabel(labels[i]); style(ax)
            if i < 2: ax.set_yscale("log")
        first = json.loads((root / "1-from-scratch" / "failure_scan.json").read_text())["first_instability"]
        if first:
            for ax in axs: ax.axvline(first["time_ps"], color="#B34F3C", lw=1.2)
            axs[0].annotate("unstable — no Dxy reported", xy=(first["time_ps"], thresholds[0]), xytext=(0.55, 0.72), textcoords="axes fraction", arrowprops={"arrowstyle":"->", "color":"#B34F3C"}, color="#B34F3C", fontweight="bold")
        axs[-1].set_xlabel("Time (ps)")
        fig.savefig(out / "fig_from_scratch_failure.pdf"); fig.savefig(out / "fig_from_scratch_failure.svg"); fig.savefig(out / "fig_from_scratch_failure.png", dpi=300); plt.close(fig)

    rows = []
    for name, m in metrics.items():
        rows.append({"model": LABELS[name], "last_step": m["last_step"], "T_mean_K": m["temperature_K"]["mean"], "T_min_K": m["temperature_K"]["min"], "T_max_K": m["temperature_K"]["max"], "max_force_eV_A": m["max_force_eV_A"], "min_distance_A": m["min_distance_A"], "Dxy_200_500": m["Dxy"]["200_500_ps"]["D_cm2_s"], "Dxy_500_900": m["Dxy"]["500_900_ps"]["D_cm2_s"]})
    with (out / "table_stability_diffusion.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with (out / "table_stability_diffusion.tex").open("w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\nModel & $T_{mean}$ & $T_{min}$ & $T_{max}$ & $F_{max}$ & $d_{min}$ & $D_{xy}$\\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['model']} & {r['T_mean_K']:.1f} & {r['T_min_K']:.1f} & {r['T_max_K']:.1f} & {r['max_force_eV_A']:.3g} & {r['min_distance_A']:.3f} & {r['Dxy_500_900']:.3g}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); make(a.root, a.out)
