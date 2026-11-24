"""Render FIG-057 v3 with the completed Rockfish Cr trajectories.

v2 is preserved.  v3 replaces the older illustrative Cr row with the three
10-image Cr paths from the 2026-10-13 Rockfish campaign while retaining the
explicit provisional/nonconverged status in the companion provenance files.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "exploratory_energy_audit"
V2_SCRIPT = ROOT / "plot_figure1_task_type_matrix_v2.py"
V3 = AUDIT / "figure1_task_type_matrix_v3"
CR = AUDIT / "rockfish_cr_neb_20261013"


def load_v2():
    spec = importlib.util.spec_from_file_location("figure1_v2", V2_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_labels(v2) -> dict[str, str]:
    return {
        "mp_0b3_medium": "MP-0B3", "mpa0_medium": "MPA0", "omat0_small": "OMAT-0-S",
        "omat0_medium": "OMAT-0-M", "mh0": "MH-0", "mh1": "MH-1",
        "agnesi_medium": "Agnesi-M", "agnesi_stress_small": "Agnesi-S-dens",
        "agnesi_stress_medium": "Agnesi-M-dens", "agnesi_stress_large": "Agnesi-L-dens",
    }


def normalize_cr_sources(v2) -> tuple[dict[str, Path], dict[str, Path]]:
    labels = model_labels(v2)
    raw_profile = pd.read_csv(CR / "rockfish_cr_foundation_image_profiles.csv")
    raw_profile["model_label"] = raw_profile["model_id"].map(labels).fillna(raw_profile["model_label"])
    profile_files = {}
    barrier_files = {}
    for path_class in ["within-QL", "in-gap diffusion", "gap → QL"]:
        sub = raw_profile[raw_profile.path_class.eq(path_class)].copy()
        sub = sub[["path_class", "path_id", "image_index", "model_id", "model_label", "relative_energy_eV", "reaction_coordinate"]]
        profile_file = V3 / f"cr_{path_class.replace(' ', '_').replace('→', 'to')}_image_profiles.csv"
        sub.to_csv(profile_file, index=False)
        profile_files[path_class] = profile_file

    barriers = pd.read_csv(CR / "rockfish_cr_foundation_barrier_table.csv")
    barriers["model_label"] = barriers["model_id"].map(labels)
    for path_class in ["within-QL", "in-gap diffusion", "gap → QL"]:
        sub = barriers[barriers.path_class.eq(path_class)].copy()
        barrier_file = V3 / f"cr_{path_class.replace(' ', '_').replace('→', 'to')}_barrier_table.csv"
        sub.to_csv(barrier_file, index=False)
        barrier_files[path_class] = barrier_file
    return profile_files, barrier_files


def render(v2, current: pd.DataFrame, cr: dict[str, pd.DataFrame]) -> None:
    from plot_atlas import apply_publication_style, panel_label, save_figure
    from neb_plot_style import QE_COLOR, model_colors

    apply_publication_style(7.8)
    colors = model_colors(v2.CURRENT_MODELS)
    cr_colors = model_colors(v2.CR_MODELS)
    fig, axes = plt.subplots(5, 3, figsize=(10.7, 11.1), squeeze=False)
    fig.subplots_adjust(left=.12, right=.985, bottom=.10, top=.91, wspace=.30, hspace=.52)
    kinds = [("within", "within-QL"), ("gap", "in-gap diffusion"), ("gap2ql", "gap → QL")]
    for col, (kind, label) in enumerate(kinds):
        v2.draw_schematic(axes[0, col], kind, label)
        panel_label(axes[0, col], chr(ord("a") + col), x=-.14, y=1.08)
    row_names = ["Cu", "Li", "Na", "Cr"]
    for r, dopant in enumerate(row_names, start=1):
        for c, (_, path_class) in enumerate(kinds):
            ax = axes[r, c]
            if dopant != "Cr":
                v2.draw_panel(ax, current, v2.PATH_SPECS[dopant][path_class], v2.CURRENT_MODELS, colors, True)
            else:
                v2.draw_panel(ax, cr[path_class], v2.PATH_SPECS[dopant][path_class], v2.CR_MODELS, cr_colors, True)
            if r == 4:
                ax.set_xlabel("Path coordinate", fontweight="bold")
            if c == 0:
                ax.set_ylabel("ΔE (eV)", fontweight="bold")
            panel_label(ax, chr(ord("d") + (r - 1) * 3 + c), x=-.14, y=1.08)
    # Place each dopant label in axes coordinates at the true vertical center
    # of its row, just outside the first panel. This remains aligned when the
    # figure height or subplot spacing changes.
    for r, dopant in enumerate(row_names, start=1):
        axes[r, 0].text(-.28, .50, dopant, transform=axes[r, 0].transAxes,
                        ha="right", va="center", rotation=0, fontsize=10,
                        fontweight="bold", color="#303538", clip_on=False)
    legend_handles = [plt.Line2D([], [], color=QE_COLOR, marker="o", lw=1.0, ms=3.0, label="QE path table (screened)")]
    legend_handles += [plt.Line2D([], [], color=col, marker="o", lw=1.0, ms=2.8, label=v2.MODEL_LABELS[m]) for m, col in zip(v2.CURRENT_MODELS, colors)]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(.55, .025), ncol=6, frameon=False, fontsize=6.3, handletextpad=.32, columnspacing=1.05)
    stem = V3 / "figure1_task_type_matrix_v3"
    save_figure(fig, stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def build_error_tables_v3(v2):
    frames = []
    for path_class, source in v2.CR_BARRIER_FILES.items():
        d = pd.read_csv(source).copy()
        d["path_class"] = path_class
        frames.append(d)
    barriers = pd.concat(frames, ignore_index=True)
    selected = {(path_class, path_id) for path_class, path_id in v2.PATH_SPECS["Cr"].items()}
    barriers["selected_for_main_matrix"] = [(r.path_class, r.path_id) in selected for r in barriers.itertuples()]
    barriers["public_path_label"] = barriers.path_class
    barriers["model_label_public"] = barriers.model_label
    matrix = barriers.pivot_table(index=["path_class", "path_id", "qe_barrier_like_eV", "selected_for_main_matrix"], columns="model_label_public", values="absolute_error_eV", aggfunc="first").reset_index()
    matrix.columns.name = None
    model_cols = [v2.MODEL_LABELS[m] for m in v2.CR_MODELS]
    matrix = matrix.rename(columns={"qe_barrier_like_eV": "QE barrier (eV)"})
    matrix = matrix[["path_class", "path_id", "selected_for_main_matrix", "QE barrier (eV)"] + model_cols]
    extra = barriers[~barriers.selected_for_main_matrix].copy()
    summary_columns = ["Path class", "QE barrier (eV)", "Median |error| (eV)", "Mean |error| (eV)", "Best model", "Best |error| (eV)", "Worst model", "Worst |error| (eV)"]
    summary_rows = []
    for (path_class, path_id), g in extra.groupby(["path_class", "path_id"], sort=False):
        best = g.loc[g.absolute_error_eV.idxmin()]
        worst = g.loc[g.absolute_error_eV.idxmax()]
        summary_rows.append({"Path class": path_class, "QE barrier (eV)": g.qe_barrier_like_eV.iloc[0], "Median |error| (eV)": g.absolute_error_eV.median(), "Mean |error| (eV)": g.absolute_error_eV.mean(), "Best model": best.model_label, "Best |error| (eV)": best.absolute_error_eV, "Worst model": worst.model_label, "Worst |error| (eV)": worst.absolute_error_eV})
    summary = pd.DataFrame(summary_rows, columns=summary_columns)
    if not summary.empty:
        summary = summary.sort_values(["Path class", "QE barrier (eV)"]).reset_index(drop=True)
    return barriers, matrix, {"summary": summary, "extra": extra}


def main() -> None:
    V3.mkdir(parents=True, exist_ok=True)
    v2 = load_v2()
    v2.OUT = V3
    labels = model_labels(v2)
    v2.CR_MODELS = list(labels)
    v2.MODEL_LABELS.update(labels)
    v2.PATH_SPECS = {**v2.PATH_SPECS, "Cr": {"within-QL": "2L-Cu__neb-QLs", "in-gap diffusion": "2L-Cu__neb-gap-diffuse", "gap → QL": "2L-Cu__neb-gap2QL"}}
    profile_files, barrier_files = normalize_cr_sources(v2)
    v2.CR_PROFILE_FILES = profile_files
    v2.CR_BARRIER_FILES = barrier_files
    current = v2.load_current_profiles()
    cr = v2.load_cr_profiles()
    render(v2, current, cr)

    _, matrix, tables = build_error_tables_v3(v2)
    tables["summary"].to_csv(V3 / "cr_path_model_error_summary.csv", index=False)
    tables["extra"].to_csv(V3 / "cr_path_model_error_long.csv", index=False)
    matrix.to_csv(V3 / "cr_path_model_error_matrix.csv", index=False)
    tables["summary"].to_latex(V3 / "cr_path_model_error_summary.tex", index=False, escape=False, float_format=lambda x: f"{x:.3f}")

    mapping = []
    panel = 1
    for dopant in ["Cu", "Li", "Na", "Cr"]:
        for _, path_class in [("within", "within-QL"), ("gap", "in-gap diffusion"), ("gap2ql", "gap → QL")]:
            source = str(profile_files[path_class]) if dopant == "Cr" else str(v2.CURRENT_PROFILE if not (dopant == "Na" and path_class == "within-QL") else v2.ALL_PATH_PROFILE)
            mapping.append({"panel_label": chr(ord("d") + panel - 1), "row_dopant": dopant, "column_path_class": path_class, "source_path_id": v2.PATH_SPECS[dopant][path_class], "source_artifact": source, "reference_status": "COMPLETED_BUT_SCF_WARNING" if dopant == "Cr" else "screened_existing_path_table"})
            panel += 1
    pd.DataFrame(mapping).to_csv(V3 / "panel_mapping.csv", index=False)

    caption = """# FIG-057 v3 | Dopant-by-path-class task matrix\n\n**Figure 1 | Foundation-model profiles across the three migration classes.** Rows show Cu, Li, Na, and Cr; columns show within-QL, in-gap diffusion, and gap → QL. Colored curves are same-geometry single-point evaluations from ten local Foundation/MPA0 checkpoints. The black curve is the corresponding screened QE path table. The new Cr row uses the three completed Rockfish Cr-substitution trajectories (jobs 31916908–31916910), with 10 images and 81 atoms per path.\n\nThe Cr Rockfish jobs completed and produced finite path tables, but all three contain a final SCF-convergence warning; their QE traces and barrier-like values are therefore provisional diagnostic references and are not publication-grade converged activation energies. Exact input, job, model, and hash provenance is retained in the companion Rockfish package. The earlier exploratory 61/81-atom Cr family used in v2 is preserved and is not mixed into this v3 matrix.\n"""
    (V3 / "figure1_task_type_matrix_v3_caption.md").write_text(caption)
    (V3 / "figure1_task_type_matrix_v3.tex").write_text("""% FIG-057 v3\n% Cr panels use Rockfish jobs 31916908, 31916909, and 31916910.\n% All three are COMPLETED with JOB DONE and finite values but retain SCF warnings; do not interpret as final converged barriers.\n\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=0.98\\textwidth]{figure1_task_type_matrix_v3.pdf}\n\\caption{Foundation-model energy profiles stratified by dopant and migration-path class. The Cr row uses the three same-geometry Cr-substitution paths completed on Rockfish. These Cr QE traces are screened, completed-but-SCF-warning path tables and are not publication-grade converged activation energies.}\n\\label{fig:task-type-matrix-v3}\n\\end{figure*}\n""")

    manifest = {"figure_id": "FIG-057", "version": "v3", "status": "PROVENANCE_PARTIAL", "scientific_status": "diagnostic figure; Cr QE reference gate open", "representative_cr_paths": v2.PATH_SPECS["Cr"], "source_packages": {"rockfish_cr": str(CR), "previous_v2": str(AUDIT / "figure1_task_type_matrix_v2")}, "inputs": {}, "outputs": {}}
    for file in sorted(CR.rglob("*")):
        if file.is_file() and file.stat().st_size < 20_000_000:
            manifest["inputs"][str(file.relative_to(CR))] = sha256(file)
    for file in sorted(V3.iterdir()):
        if file.is_file() and file.name != "FIG-057_v3_manifest.json":
            manifest["outputs"][file.name] = sha256(file)
    (V3 / "FIG-057_v3_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figure": str(V3 / "figure1_task_type_matrix_v3.pdf"), "n_cr_profiles": sum(len(x) for x in cr.values()), "n_error_rows": len(matrix)}, indent=2))


if __name__ == "__main__":
    main()
