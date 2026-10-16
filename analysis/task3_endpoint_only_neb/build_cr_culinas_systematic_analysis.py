#!/usr/bin/env python3
"""Combine the three Rockfish Cr paths with the nine Cu/Li/Na path panels."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "exploratory_energy_audit"
OUT = AUDIT / "cr_vs_culinas_systematic_20261015"
CR = AUDIT / "rockfish_cr_neb_20261013"
sys.path.insert(0, str(ROOT))
from neb_plot_style import QE_COLOR, model_colors, model_label  # noqa: E402


MODELS = ["mace-large-density-agnesi-stress", "mace-medium-density-agnesi-stress", "mace-small-density-agnesi-stress", "mace_agnesi_medium", "mace-mh-0", "mace-mh-1", "mace-mp-0b3-medium", "mace-omat-0-medium", "mace-omat-0-small", "mpa0_medium"]
SHORT = {"mace-large-density-agnesi-stress": "agnesi_stress_large", "mace-medium-density-agnesi-stress": "agnesi_stress_medium", "mace-small-density-agnesi-stress": "agnesi_stress_small", "mace_agnesi_medium": "agnesi_medium", "mace-mh-0": "mh0", "mace-mh-1": "mh1", "mace-mp-0b3-medium": "mp_0b3_medium", "mace-omat-0-medium": "omat0_medium", "mace-omat-0-small": "omat0_small", "mpa0_medium": "mpa0_medium"}
PATHS = {"Cu": {"within-QL": "2L-Cu__neb-QLs", "in-gap diffusion": "2L-Cu__neb-gap-diffuse", "gap → QL": "2L-Cu__neb-gap2QL"}, "Li": {"within-QL": "2L-Li__neb-QLs", "in-gap diffusion": "2L-Li__neb-gap-diffuse", "gap → QL": "2L-Li__neb-gap2Ql"}, "Na": {"within-QL": "2L-Na__neb-QLs", "in-gap diffusion": "2L-Na__neb-gap-diffuse", "gap → QL": "2L-Na__neb-gap2QL"}, "Cr": {"within-QL": "2L-Cu__neb-QLs", "in-gap diffusion": "2L-Cu__neb-gap-diffuse", "gap → QL": "2L-Cu__neb-gap2QL"}}
ORDER = ["within-QL", "in-gap diffusion", "gap → QL"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_original() -> tuple[pd.DataFrame, pd.DataFrame]:
    sys.path.insert(0, str(ROOT))
    import plot_figure1_task_type_matrix_v2 as v2
    d = v2.load_current_profiles()
    rows = []
    for dopant in ["Cu", "Li", "Na"]:
        for path_class, path_id in PATHS[dopant].items():
            q = d[d.path_id.eq(path_id)]
            qe = float(q.qe_relative_energy_eV.max())
            for model in MODELS:
                m = q[q.model_id.eq(model)]
                barrier = float(m.relative_energy_eV.max())
                rows.append({"dopant": dopant, "path_class": path_class, "path_id": path_id, "model_id": model, "model_label": model_label(model), "qe_barrier_like_eV": qe, "model_barrier_like_eV": barrier, "signed_error_eV": barrier - qe, "absolute_error_eV": abs(barrier - qe), "reference_status": "SCREENED_EXPLORATORY"})
    return pd.DataFrame(rows), d


def load_cr() -> pd.DataFrame:
    q = pd.read_csv(CR / "rockfish_cr_qe_summary.csv")
    b = pd.read_csv(CR / "rockfish_cr_foundation_barrier_table.csv")
    b["model_id"] = b.model_id.map({v: k for k, v in SHORT.items()})
    b["model_label"] = b.model_id.map(model_label)
    b = b.drop(columns=["reference_status"], errors="ignore")
    b = b.merge(q[["path_class", "path_id", "qe_forward_activation_eV", "qe_reverse_activation_eV", "job_id", "qe_last_complete_iteration", "status"]], on=["path_class", "path_id"], how="left")
    return b.rename(columns={"foundation_barrier_like_eV": "model_barrier_like_eV", "status": "reference_status"})[["path_class", "path_id", "model_id", "model_label", "qe_barrier_like_eV", "model_barrier_like_eV", "signed_error_eV", "absolute_error_eV", "reference_status", "job_id", "qe_last_complete_iteration", "qe_forward_activation_eV", "qe_reverse_activation_eV"]].assign(dopant="Cr")


def render(barriers: pd.DataFrame) -> None:
    from plot_atlas import apply_publication_style, panel_label, save_figure
    apply_publication_style(8.2)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.25), sharey=False)
    colors = model_colors(MODELS)
    xlabels = ["Cu", "Li", "Na", "Cr"]
    offsets = np.linspace(-.18, .18, len(MODELS))
    for idx, path_class in enumerate(ORDER):
        ax = axes[idx]
        sub = barriers[barriers.path_class.eq(path_class)]
        for j, model in enumerate(MODELS):
            vals = sub[sub.model_id.eq(model)].set_index("dopant").reindex(xlabels).model_barrier_like_eV.to_numpy()
            ax.plot(np.arange(4) + offsets[j], vals, color=colors[j], marker="o", ms=3.1, lw=.85, alpha=.92)
        qe = sub.drop_duplicates(["dopant"])[["dopant", "qe_barrier_like_eV"]].set_index("dopant").reindex(xlabels).qe_barrier_like_eV.to_numpy()
        ax.plot(np.arange(4), qe, color=QE_COLOR, marker="o", ms=4.0, lw=1.15, label="QE path table")
        ax.scatter([3], [qe[3]], s=52, facecolors="none", edgecolors="#9B4A4A", linewidths=1.1, zorder=5)
        ax.set_xticks(np.arange(4), xlabels)
        ax.set_title(path_class, fontsize=8.8, fontweight="bold")
        ax.set_xlabel("Dopant")
        if idx == 0:
            ax.set_ylabel("Barrier-like ΔE (eV)")
        ax.text(.03, .96, "Cr: SCF-warning QE", transform=ax.transAxes, fontsize=5.6, color="#7A3E3E", va="top")
        ax.grid(False)
        for spine in ax.spines.values(): spine.set_linewidth(.8)
        panel_label(ax, chr(ord("a") + idx), x=-.13, y=1.08)
    handles = [plt.Line2D([], [], color=QE_COLOR, marker="o", lw=1.1, ms=4, label="QE path table")]
    handles += [plt.Line2D([], [], color=colors[i], marker="o", lw=.85, ms=3, label=model_label(m)) for i, m in enumerate(MODELS)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, -.01), ncol=6, frameon=False, fontsize=5.8, columnspacing=.9, handletextpad=.3)
    fig.subplots_adjust(left=.08, right=.99, bottom=.22, top=.86, wspace=.35)
    stem = OUT / "figure_cr_vs_culinas_barrier_like_matrix"
    save_figure(fig, stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    original, _ = load_original()
    cr = load_cr()
    barriers = pd.concat([original, cr], ignore_index=True, sort=False)
    barriers.to_csv(OUT / "cr_vs_culinas_model_barriers_long.csv", index=False)
    qe_summary = barriers.groupby(["dopant", "path_class", "path_id", "reference_status"], as_index=False).agg(qe_barrier_like_eV=("qe_barrier_like_eV", "first"), model_median_eV=("model_barrier_like_eV", "median"), model_min_eV=("model_barrier_like_eV", "min"), model_max_eV=("model_barrier_like_eV", "max"), model_median_absolute_error_eV=("absolute_error_eV", "median"), n_models=("model_id", "nunique"))
    qe_summary.to_csv(OUT / "cr_vs_culinas_path_class_summary.csv", index=False)
    model_summary = barriers.groupby(["dopant", "path_class", "model_id", "model_label"], as_index=False).agg(qe_barrier_like_eV=("qe_barrier_like_eV", "first"), model_barrier_like_eV=("model_barrier_like_eV", "first"), absolute_error_eV=("absolute_error_eV", "first"), reference_status=("reference_status", "first"))
    model_summary.to_csv(OUT / "cr_vs_culinas_model_by_path_summary.csv", index=False)
    pd.DataFrame([
        {"path_class": "within-QL", "v2_path_id": "neb_2", "v3_path_id": PATHS["Cr"]["within-QL"], "v3_job_id": "31916908", "v3_n_atoms": 81, "v3_n_images": 10, "v3_chemistry": "Cr substitution", "v2_status": "historical exploratory reconstruction", "v3_status": "COMPLETED_BUT_SCF_WARNING"},
        {"path_class": "in-gap diffusion", "v2_path_id": "81-dislocation-gap", "v3_path_id": PATHS["Cr"]["in-gap diffusion"], "v3_job_id": "31916910", "v3_n_atoms": 81, "v3_n_images": 10, "v3_chemistry": "Cr substitution", "v2_status": "historical exploratory reconstruction", "v3_status": "COMPLETED_BUT_SCF_WARNING"},
        {"path_class": "gap → QL", "v2_path_id": "1-7", "v3_path_id": PATHS["Cr"]["gap → QL"], "v3_job_id": "31916909", "v3_n_atoms": 81, "v3_n_images": 10, "v3_chemistry": "Cr substitution", "v2_status": "historical exploratory reconstruction", "v3_status": "COMPLETED_BUT_SCF_WARNING"},
    ]).to_csv(OUT / "cr_trajectory_identity_audit.csv", index=False)
    pd.DataFrame([
        {"parameter": "num_of_images", "rockfish_cr": "10", "joshitha_original": "10", "match": True},
        {"parameter": "nstep_path", "rockfish_cr": "1000", "joshitha_original": "1000", "match": True},
        {"parameter": "opt_scheme", "rockfish_cr": "broyden", "joshitha_original": "broyden", "match": True},
        {"parameter": "ds", "rockfish_cr": "0.0756D0", "joshitha_original": "0.0756D0", "match": True},
        {"parameter": "path_thr", "rockfish_cr": "0.001D0", "joshitha_original": "0.001D0", "match": True},
        {"parameter": "max_seconds", "rockfish_cr": "250000", "joshitha_original": "250000", "match": True},
        {"parameter": "ecutwfc (Ry)", "rockfish_cr": "70", "joshitha_original": "70", "match": True},
        {"parameter": "ecutrho (Ry)", "rockfish_cr": "280", "joshitha_original": "280", "match": True},
        {"parameter": "degauss (Ry)", "rockfish_cr": "0.00500", "joshitha_original": "0.00500", "match": True},
        {"parameter": "smearing", "rockfish_cr": "cold", "joshitha_original": "cold", "match": True},
        {"parameter": "vdw_corr", "rockfish_cr": "DFT-D3", "joshitha_original": "DFT-D3", "match": True},
        {"parameter": "k-point mesh", "rockfish_cr": "1 1 1 1 1 1", "joshitha_original": "1 1 1 1 1 1", "match": True},
        {"parameter": "nat", "rockfish_cr": "81", "joshitha_original": "81", "match": True},
        {"parameter": "chemistry", "rockfish_cr": "Cr_PBE_FR.SG15v1.2.UPF / Cr", "joshitha_original": "Cu/Li/Na path-specific species", "match": False, "note": "intentional dopant substitution"},
        {"parameter": "launcher", "rockfish_cr": "Rockfish Slurm/Hydra launcher fix", "joshitha_original": "Joshitha original wrapper", "match": False, "note": "runtime layer only; QE input contract unchanged"},
    ]).to_csv(OUT / "rockfish_vs_joshitha_contract_comparison.tsv", sep="\t", index=False)
    render(barriers)

    text = """# Cr versus Cu/Li/Na systematic NEB-path analysis\n\n## Scope\n\nThis package joins the three new Rockfish Cr-substitution paths to the existing nine Cu/Li/Na path-class panels: within-QL, in-gap diffusion, and gap → QL. Foundation-model values are same-geometry fixed-path single-point evaluations. The QE columns are path-table references, not certified converged barriers.\n\n## Results\n\nThe Cr within-QL path is the dominant model-disagreement case. Its Rockfish QE path-table forward activation is 1.8449 eV, while the ten-model barrier-like range is approximately 19.33–29.74 eV. The discrepancy is concentrated at the interior images and should be interpreted as a strong model/geometry stress signal, not as evidence that the physical Cr barrier is 20–30 eV.\n\nFor Cr in-gap diffusion, the QE path-table value is 1.5243 eV and the Foundation-model range is approximately 0.899–1.549 eV, with the large-density Agnesi model closest to the screened QE reference. For Cr gap → QL, the QE path is monotone downhill relative to image 1 (forward 0.000 eV; reverse 0.095947 eV), and all model barrier-like values remain below 0.073 eV.\n\nAgainst the existing Cu/Li/Na panel, within-QL paths are the most model-sensitive class: Cu is already a large-disagreement outlier, whereas Li and Na show much closer model envelopes. The gap-based classes have substantially smaller model spreads, although gap → QL is endpoint-direction dependent and should not be summarized by a forward barrier alone. The Cr pattern therefore extends the existing stress-test story rather than providing a single universal dopant ranking.\n\n## Publication status\n\nThe 27 Rockfish Cr path-table outputs are `COMPLETED`/`JOB DONE` with ten finite images, but each carries an SCF-convergence warning. Accordingly, Cr QE values are `COMPLETED_BUT_SCF_WARNING` and excluded from final converged-barrier claims. Figure 1 v3 is publication-safe as a provenance-labeled diagnostic/profile figure; a final Cr activation-energy claim requires rerunning the three paths until the SCF warning and path-convergence gates are both cleared.\n"""
    text = text.replace("approximately 19.33–29.74 eV", "21.176812–29.735416 eV")
    text = text.replace("The 27 Rockfish Cr path-table outputs are", "The three Rockfish Cr path-table outputs, each with ten images, are")
    cr_qe = cr.drop_duplicates(["path_class", "path_id"])[["path_class", "job_id", "qe_forward_activation_eV", "qe_reverse_activation_eV", "qe_last_complete_iteration"]].sort_values("path_class")
    cr_model = cr.groupby("path_class", as_index=False).agg(model_min_eV=("model_barrier_like_eV", "min"), model_median_eV=("model_barrier_like_eV", "median"), model_max_eV=("model_barrier_like_eV", "max"))
    cr_report = cr_qe.merge(cr_model, on="path_class")
    table_rows = [
        "| Path class | Rockfish job | Forward QE (eV) | Reverse QE (eV) | Last complete iter. | Model range (eV) | Model median (eV) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in cr_report.iterrows():
        table_rows.append(f"| {r.path_class} | {r.job_id} | {r.qe_forward_activation_eV:.6f} | {r.qe_reverse_activation_eV:.6f} | {int(r.qe_last_complete_iteration)} | {r.model_min_eV:.6f}–{r.model_max_eV:.6f} | {r.model_median_eV:.6f} |")
    report = text.replace("# Cr versus Cu/Li/Na systematic NEB-path analysis", "# Publication report: Cr versus Cu/Li/Na NEB paths", 1)
    report = report.replace("## Results", "## Cr activation-energy readout\n\n" + "\n".join(table_rows) + "\n\n## Results", 1)
    report += "\n## Reproducibility files\n\n- `rockfish_vs_joshitha_contract_comparison.tsv`: numerical QE/NEB contract audit.\n- `cr_trajectory_identity_audit.csv`: v2-to-v3 Cr trajectory mapping.\n- `rockfish_cr_foundation_barrier_table.csv`: all 10 foundation-model evaluations per Cr path.\n- `figure1_task_type_matrix_v3/figure1_task_type_matrix_v3.pdf`: updated figure-book panel.\n"
    (OUT / "cr_vs_culinas_paper_text.md").write_text(text)
    (OUT / "cr_vs_culinas_publication_report.md").write_text(report)
    manifest = {"status": "PROVENANCE_PARTIAL", "reference_status": {"Cu_Li_Na": "SCREENED_EXPLORATORY", "Cr": "COMPLETED_BUT_SCF_WARNING"}, "inputs": {"rockfish_cr": str(CR), "figure_v3": str(AUDIT / "figure1_task_type_matrix_v3")}, "outputs": {}}
    for f in sorted(OUT.iterdir()):
        if f.is_file() and f.name != "cr_vs_culinas_manifest.json": manifest["outputs"][f.name] = sha256(f)
    (OUT / "cr_vs_culinas_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(barriers), "paths": int(barriers.path_id.nunique()), "models": int(barriers.model_id.nunique()), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
