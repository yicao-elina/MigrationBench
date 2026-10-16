#!/usr/bin/env python3
"""Build bidirectional activation tables/figure and audit gap→QL geometry."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import find_mic
from ase.io import read

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "exploratory_energy_audit"
CR = AUDIT / "rockfish_cr_neb_20261013"
OUT = AUDIT / "gap2ql_bidirectional_20261016"
V2_SCRIPT = ROOT / "plot_figure1_task_type_matrix_v2.py"
PATH_CLASSES = ["within-QL", "in-gap diffusion", "gap → QL"]
DOPE = ["Cu", "Li", "Na", "Cr"]
MODELS = [
    "mace-large-density-agnesi-stress", "mace-medium-density-agnesi-stress",
    "mace-small-density-agnesi-stress", "mace_agnesi_medium", "mace-mh-0",
    "mace-mh-1", "mace-mp-0b3-medium", "mace-omat-0-medium",
    "mace-omat-0-small", "mpa0_medium",
]
CR_MODEL_MAP = {
    "mp_0b3_medium": "mace-mp-0b3-medium", "mpa0_medium": "mpa0_medium",
    "omat0_small": "mace-omat-0-small", "omat0_medium": "mace-omat-0-medium",
    "mh0": "mace-mh-0", "mh1": "mace-mh-1", "agnesi_medium": "mace_agnesi_medium",
    "agnesi_stress_small": "mace-small-density-agnesi-stress",
    "agnesi_stress_medium": "mace-medium-density-agnesi-stress",
    "agnesi_stress_large": "mace-large-density-agnesi-stress",
}


def load_v2():
    spec = importlib.util.spec_from_file_location("figure1_v2_bidirectional", V2_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directional(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    peak = float(values.max())
    return peak, peak - float(values[-1])


def add_profile_row(rows, dopant, path_class, path_id, model_id, model_label, values, status, ref_type):
    fwd, rev = directional(values)
    rows.append({"dopant": dopant, "path_class": path_class, "path_id": path_id,
                 "reference_type": ref_type, "model_id": model_id, "model_label": model_label,
                 "forward_activation_eV": fwd, "reverse_activation_eV": rev,
                 "reference_status": status})


def build_energy_table() -> pd.DataFrame:
    v2 = load_v2()
    current = v2.load_current_profiles()
    rows = []
    for dopant in ["Cu", "Li", "Na"]:
        for path_class in PATH_CLASSES:
            path_id = v2.PATH_SPECS[dopant][path_class]
            sub = current[current.path_id.eq(path_id)]
            for model in MODELS:
                s = sub[sub.model_id.eq(model)].sort_values("image_index")
                if not s.empty:
                    add_profile_row(rows, dopant, path_class, path_id, model, s.model_label.iloc[0], s.relative_energy_eV.to_numpy(), "SCREENED_EXPLORATORY", "Foundation/MPA0")
            q = sub.sort_values("image_index").drop_duplicates("image_index")
            add_profile_row(rows, dopant, path_class, path_id, "QE", "QE path table", q.qe_relative_energy_eV.to_numpy(), "SCREENED_EXPLORATORY", "QE")
    cr = pd.read_csv(CR / "rockfish_cr_foundation_image_profiles.csv")
    for path_class in PATH_CLASSES:
        sub = cr[cr.path_class.eq(path_class)]
        for raw_model, model in CR_MODEL_MAP.items():
            s = sub[sub.model_id.eq(raw_model)].sort_values("image_index")
            add_profile_row(rows, "Cr", path_class, s.path_id.iloc[0], model, s.model_label.iloc[0], s.relative_energy_eV.to_numpy(), "COMPLETED_BUT_SCF_WARNING", "Foundation/MPA0")
        q = sub[sub.model_id.eq("QE")].sort_values("image_index")
        add_profile_row(rows, "Cr", path_class, q.path_id.iloc[0], "QE", "QE path table", q.relative_energy_eV.to_numpy(), "COMPLETED_BUT_SCF_WARNING", "QE")
    return pd.DataFrame(rows)


def parse_path(path: Path) -> dict[int, np.ndarray]:
    lines, result, i = path.read_text().splitlines(), {}, 0
    while i < len(lines):
        if lines[i].startswith("Image:"):
            image = int(lines[i].split(":")[1])
            result[image] = np.array([[float(x) for x in lines[j].split()[:3]] for j in range(i + 2, i + 2 + 81)])
            i += 83
        else:
            i += 1
    return result


def geometry_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = {
        "Cu": (ROOT / "inputs/2L-Cu__neb-gap2QL__initial.extxyz", ROOT / "inputs/2L-Cu__neb-gap2QL__final.extxyz"),
        "Li": (ROOT / "inputs/2L-Li__neb-gap2Ql__initial.extxyz", ROOT / "inputs/2L-Li__neb-gap2Ql__final.extxyz"),
        "Na": (ROOT / "inputs/2L-Na__neb-gap2QL__initial.extxyz", ROOT / "inputs/2L-Na__neb-gap2QL__final.extxyz"),
        "Cr": (ROOT / "inputs/2L-Cu__neb-gap2QL__initial.extxyz", ROOT / "inputs/2L-Cu__neb-gap2QL__final.extxyz"),
    }
    cu_initial, cu_final = read(sources["Cu"][0]), read(sources["Cu"][1])
    rows = []
    for dopant, (initial_path, final_path) in sources.items():
        initial, final = read(initial_path), read(final_path)
        if dopant == "Cr":
            initial[-1].symbol, final[-1].symbol = "Cr", "Cr"
        d0 = np.linalg.norm(initial.positions - cu_initial.positions, axis=1)
        d1 = np.linalg.norm(final.positions - cu_final.positions, axis=1)
        def coordination(at):
            vec, _ = find_mic(at[:-1].positions - at[-1].position, cell=at.cell, pbc=at.pbc)
            distances = np.linalg.norm(vec, axis=1)
            te = distances[[a.symbol == "Te" for a in at[:-1]]]
            sb = distances[[a.symbol == "Sb" for a in at[:-1]]]
            return float(np.sort(te)[:4].mean()), float(np.sort(sb)[:3].mean())
        te0, sb0 = coordination(initial)
        te1, sb1 = coordination(final)
        endpoint_disp = float(np.linalg.norm(final[-1].position - initial[-1].position))
        rows.extend([
            {"dopant": dopant, "endpoint": "initial", "source": str(initial_path), "n_atoms": len(initial),
             "cell_max_delta_vs_Cu_A": float(np.abs(initial.cell.array - cu_initial.cell.array).max()),
             "coordinate_rms_vs_Cu_A": float(np.sqrt(np.mean(d0 * d0))), "coordinate_max_vs_Cu_A": float(d0.max()),
             "covalent_radius_A": float(covalent_radii[atomic_numbers[dopant]]), "nearest_4_Te_mean_A": te0,
             "nearest_3_Sb_mean_A": sb0, "dopant_initial_to_final_displacement_A": endpoint_disp},
            {"dopant": dopant, "endpoint": "final", "source": str(final_path), "n_atoms": len(final),
             "cell_max_delta_vs_Cu_A": float(np.abs(final.cell.array - cu_final.cell.array).max()),
             "coordinate_rms_vs_Cu_A": float(np.sqrt(np.mean(d1 * d1))), "coordinate_max_vs_Cu_A": float(d1.max()),
             "covalent_radius_A": float(covalent_radii[atomic_numbers[dopant]]), "nearest_4_Te_mean_A": te1,
             "nearest_3_Sb_mean_A": sb1, "dopant_initial_to_final_displacement_A": endpoint_disp},
        ])
    path_sources = {
        "Cu": ROOT / "remote_snapshot/2L-Cu/NEB/neb-gap2QL/sb2te3.path",
        "Li": ROOT / "remote_snapshot/2L-Li/NEB/neb-gap2Ql/sb2te3.path",
        "Na": ROOT / "remote_snapshot/2L-Na/NEB/neb-gap2QL/sb2te3.path",
        "Cr": CR / "raw/2L-Cu__neb-gap2QL/sb2te3.path",
    }
    ref = parse_path(path_sources["Cu"])
    path_rows = []
    for dopant, path in path_sources.items():
        series = parse_path(path)
        rms, max_atom = [], []
        for image in sorted(ref):
            delta = np.linalg.norm(series[image] - ref[image], axis=1)
            rms.append(float(np.sqrt(np.mean(delta * delta))))
            max_atom.append(float(delta.max()))
        path_rows.append({"dopant": dopant, "path_class": "gap → QL", "source": str(path), "n_images": len(series),
                          "mean_image_rms_vs_Cu_A": float(np.mean(rms)), "max_image_rms_vs_Cu_A": float(max(rms)),
                          "max_atom_displacement_vs_Cu_A": float(max(max_atom))})
    return pd.DataFrame(rows), pd.DataFrame(path_rows)


def qe_input_contract_audit() -> pd.DataFrame:
    """Compare raw QE endpoint inputs before NEB iterations move images."""
    sources = {
        "Cu": ROOT / "remote_snapshot/2L-Cu/NEB/neb-gap2QL",
        "Li": ROOT / "remote_snapshot/2L-Li/NEB/neb-gap2Ql",
        "Na": ROOT / "remote_snapshot/2L-Na/NEB/neb-gap2QL",
        "Cr": CR / "raw/2L-Cu__neb-gap2QL",
    }

    def parse_qe(path: Path):
        lines = path.read_text().splitlines()
        species, coords, cell = [], [], []
        apos = next(i for i, line in enumerate(lines) if line.strip().startswith("ATOMIC_POSITIONS")) + 1
        for line in lines[apos:]:
            parts = line.split()
            if len(parts) >= 4 and parts[0] in {"Te", "Sb", "Cu", "Li", "Na", "Cr"}:
                species.append(parts[0])
                coords.append([float(x) for x in parts[1:4]])
            elif species and (not line.strip() or parts[0] in {"K_POINTS", "CELL_PARAMETERS", "ATOMIC_SPECIES"}):
                break
        cstart = next((i for i, line in enumerate(lines) if line.strip().startswith("CELL_PARAMETERS")), None)
        if cstart is not None:
            cell = np.array([[float(x) for x in lines[cstart + j].split()[:3]] for j in range(1, 4)])
        return species, np.asarray(coords), cell

    rows = []
    for endpoint in ["pw_1.in", "pw_2.in"]:
        ref_species, ref_coords, ref_cell = parse_qe(sources["Cu"] / endpoint)
        for dopant, directory in sources.items():
            species, coords, cell = parse_qe(directory / endpoint)
            delta = np.linalg.norm(coords - ref_coords, axis=1)
            mismatches = [i for i, (a, b) in enumerate(zip(ref_species, species)) if a != b]
            rows.append({"endpoint_input": endpoint, "dopant": dopant, "n_atoms": len(coords),
                         "dopant_species": species[-1], "species_mismatch_count_vs_Cu": len(mismatches),
                         "species_mismatch_indices_vs_Cu": ",".join(map(str, mismatches)),
                         "coordinate_rms_vs_Cu_A": float(np.sqrt(np.mean(delta * delta))),
                         "coordinate_max_vs_Cu_A": float(delta.max()),
                         "cell_max_delta_vs_Cu_A": float(np.abs(cell - ref_cell).max()),
                         "source": str(directory / endpoint)})
    return pd.DataFrame(rows)


def render(table: pd.DataFrame) -> None:
    from plot_atlas import apply_publication_style, panel_label, save_figure
    from neb_plot_style import QE_COLOR, model_colors
    v2 = load_v2()
    apply_publication_style(8.0)
    colors = model_colors(MODELS)
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.2), squeeze=False)
    fig.subplots_adjust(left=.12, right=.985, bottom=.18, top=.88, wspace=.30, hspace=.42)
    for r, direction in enumerate(["forward_activation_eV", "reverse_activation_eV"]):
        for c, path_class in enumerate(PATH_CLASSES):
            ax = axes[r, c]
            sub = table[table.path_class.eq(path_class)]
            x = np.arange(4)
            for model, color in zip(MODELS, colors):
                vals = sub[(sub.model_id == model) & sub.reference_type.eq("Foundation/MPA0")].set_index("dopant").reindex(DOPE)[direction].to_numpy()
                ax.plot(x, vals, color=color, marker="o", ms=3.0, lw=.9, alpha=.9)
            qe = sub[sub.reference_type.eq("QE")].set_index("dopant").reindex(DOPE)[direction].to_numpy()
            ax.plot(x, qe, color=QE_COLOR, marker="D", ms=4.0, lw=1.25, label="QE path table")
            ax.scatter([3], [qe[3]], s=48, facecolors="none", edgecolors="#9B4A4A", linewidths=1.1, zorder=5)
            ax.set_xticks(x, DOPE)
            ax.set_xlabel("Dopant", fontweight="bold")
            ax.set_title(path_class, fontsize=8.8, fontweight="bold")
            if c == 0:
                ax.set_ylabel("Activation energy (eV)", fontweight="bold")
            ax.grid(False)
            ax.margins(x=.08)
            panel_label(ax, chr(ord("a") + r * 3 + c), x=-.14, y=1.08)
    fig.canvas.draw()
    for r, label in enumerate(["Forward", "Reverse"]):
        pos = axes[r, 0].get_position()
        fig.text(.038, pos.y0 + pos.height / 2, label, ha="right", va="center", fontsize=10, fontweight="bold")
    legend = [plt.Line2D([], [], color=QE_COLOR, marker="D", lw=1.2, ms=4, label="QE path table")]
    legend += [plt.Line2D([], [], color=color, marker="o", lw=.9, ms=3, label=v2.MODEL_LABELS[model]) for model, color in zip(MODELS, colors)]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.56, .02), ncol=6, frameon=False, fontsize=6.2, columnspacing=.9, handletextpad=.3)
    stem = OUT / "figure_gap2ql_bidirectional_activation"
    save_figure(fig, stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = build_energy_table()
    table.to_csv(OUT / "bidirectional_activation_energy_long.csv", index=False)
    summary = table.groupby(["dopant", "path_class", "path_id", "reference_type", "reference_status"], as_index=False).agg(forward_activation_eV=("forward_activation_eV", "first"), reverse_activation_eV=("reverse_activation_eV", "first"), model_count=("model_id", lambda s: int((s != "QE").sum())))
    summary.to_csv(OUT / "bidirectional_activation_energy_summary.csv", index=False)
    model_summary = table[table.reference_type.eq("Foundation/MPA0")].groupby(["dopant", "path_class"], as_index=False).agg(
        n_models=("model_id", "nunique"), forward_median_eV=("forward_activation_eV", "median"),
        forward_min_eV=("forward_activation_eV", "min"), forward_max_eV=("forward_activation_eV", "max"),
        reverse_median_eV=("reverse_activation_eV", "median"), reverse_min_eV=("reverse_activation_eV", "min"),
        reverse_max_eV=("reverse_activation_eV", "max"))
    model_summary.to_csv(OUT / "bidirectional_activation_energy_model_summary.csv", index=False)
    geom, paths = geometry_audit()
    geom.to_csv(OUT / "gap2ql_endpoint_geometry_audit.csv", index=False)
    paths.to_csv(OUT / "gap2ql_neb_path_geometry_audit.csv", index=False)
    qe_contract = qe_input_contract_audit()
    qe_contract.to_csv(OUT / "gap2ql_raw_qe_input_contract_audit.csv", index=False)
    render(table)
    qe_report = summary[summary.reference_type.eq("QE")][["dopant", "path_class", "forward_activation_eV", "reverse_activation_eV"]].sort_values(["path_class", "dopant"]).copy()
    qe_report["endpoint_delta_eV"] = qe_report.forward_activation_eV - qe_report.reverse_activation_eV
    qe_md = qe_report.to_markdown(index=False, floatfmt=".6f")
    model_md = model_summary.to_markdown(index=False, floatfmt=".6f")
    report = f"""# Gap→QL bidirectional activation-energy and geometry audit

## Geometry finding

All four canonical inputs use the same 81-atom cell and atom ordering. The raw Cr `pw_1.in` and `pw_2.in` are exact Cu-coordinate substitutions at both endpoints: only the final dopant species changes from Cu to Cr. Li and Na are not coordinate-identical to Cu; their endpoint structures contain element-specific dopant and host relaxation. The saved NEB paths also move differently after iteration, so final image positions are not pointwise identical across dopants.

## Energy definition

For every ordered image profile, `forward = max(E_i − E_1)` and `reverse = max(E_i − E_10)`. These are two directional readouts of one profile. The Foundation models evaluate the fixed DFT images and do not constitute separate ML NEB optimizations.

## QE directional values

`endpoint_delta_eV = forward − reverse`; positive values mean the final endpoint is higher than the initial endpoint, and negative values mean the process is downhill in the forward direction.

{qe_md}

## Foundation-model directional ranges

{model_md}

## Interpretation

Cr gap→QL is downhill in the forward direction in the retained path table, whereas Cu, Li, and Na retain nonzero forward rises. This is consistent with a chemistry-dependent coordination/size response, but the present data do not isolate covalent radius from charge transfer, d-state bonding, and endpoint relaxation. The radius/coordination explanation is therefore a mechanistic hypothesis, not a demonstrated causal decomposition.

The full directional table and geometry audit are stored beside this report. Cr QE values remain `COMPLETED_BUT_SCF_WARNING`; they are diagnostic path-table references, not final converged barriers.
"""
    (OUT / "gap2ql_bidirectional_analysis.md").write_text(report)
    manifest = {"status": "PROVENANCE_PARTIAL", "definition": "forward=max(E-E1); reverse=max(E-E10)", "inputs": {}, "outputs": {}}
    for f in sorted(OUT.iterdir()):
        if f.is_file() and f.name != "gap2ql_bidirectional_manifest.json":
            manifest["outputs"][f.name] = sha256(f)
    (OUT / "gap2ql_bidirectional_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": len(table), "model_rows": int((table.reference_type == "Foundation/MPA0").sum()), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
