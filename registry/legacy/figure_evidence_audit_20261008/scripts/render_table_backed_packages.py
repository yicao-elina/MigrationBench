#!/usr/bin/env python3
"""Render high-value table-backed analyses with the reusable Plot Atlas.

The packages are intentionally additive under the audit directory. They do
not overwrite campaign outputs and they label screening/gate-only results so
that a readable figure is never mistaken for a scientific acceptance result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


ATLAS_ROOT = Path("/Users/alina/Documents/Codex/plot_library")
if str(ATLAS_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ATLAS_ROOT / "src"))

from plot_atlas import (  # noqa: E402
    apply_publication_style,
    bar_with_points,
    clustered_heatmap,
    get_palette,
    grouped_bar,
    joint_scatter,
    panel_label,
    save_figure,
    timecourse,
)


PROJECT_ROOT = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench")
AUDIT_ROOT = PROJECT_ROOT / "analysis/figure_evidence_audit_20261008"
OUT_ROOT = AUDIT_ROOT / "generated/table_backed_packages"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(value: str | None) -> float:
    try:
        result = float(value) if value not in (None, "") else math.nan
    except (TypeError, ValueError):
        result = math.nan
    return result if math.isfinite(result) else math.nan


def save_render(fig, output_stem: Path) -> list[str]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    save_figure(fig, png)
    fig.savefig(pdf, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    return [str(png), str(output_stem.with_suffix(".svg")), str(pdf)]


def style_heatmap_labels(ax, labels: list[list[str]], *, title: str) -> None:
    ax.set_title(title, pad=8, fontweight="bold")
    for i, row in enumerate(labels):
        for j, text in enumerate(row):
            ax.text(j, i, text, ha="center", va="center", fontsize=6.5,
                    fontweight="bold", color="#222222")


def write_table(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_package(
    package_dir: Path,
    package_id: str,
    source_files: list[Path],
    figure_files: list[str],
    processed_files: list[Path],
    article_target: str,
    scientific_status: str,
    caption: str,
    results: str,
    discussion: str,
    caveats: list[str],
) -> None:
    manifest = {
        "package_id": package_id,
        "generated_by": str(Path(__file__).resolve()),
        "plot_atlas_root": str(ATLAS_ROOT),
        "plot_atlas_panels": ["timecourse", "grouped_bar", "bar_with_points", "clustered_heatmap"],
        "source_files": [{"path": str(path), "sha256": sha256(path)} for path in source_files],
        "processed_files": [{"path": str(path), "sha256": sha256(path)} for path in processed_files],
        "figure_files": [{"path": path, "sha256": sha256(Path(path))} for path in figure_files],
        "article_target": article_target,
        "scientific_status": scientific_status,
        "caption": caption,
        "results": results,
        "discussion": discussion,
        "caveats": caveats,
    }
    (package_dir / "evidence_package_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (package_dir / "caption_results_discussion.md").write_text(
        "# " + package_id + "\n\n"
        + "## Article target\n" + article_target + "\n\n"
        + "## Caption\n" + caption + "\n\n"
        + "## Results\n" + results + "\n\n"
        + "## Discussion\n" + discussion + "\n\n"
        + "## Scientific boundary\n"
        + "\n".join(f"- {item}" for item in caveats) + "\n",
        encoding="utf-8",
    )


def render_phase2_sensitivity() -> None:
    package = OUT_ROOT / "MD-SENSITIVITY-48RUN"
    package.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT / "analysis/phase2_integrator_sensitivity_20261003/audit_20261004_final/sensitivity_summary.csv"
    readme = PROJECT_ROOT / "analysis/phase2_integrator_sensitivity_20261003/SENSITIVITY_RESULTS.md"
    rows = read_csv(source)
    models = ["foundation-omat", "from-scratch", "naive-fine-tuning", "multi_T-fine-tuning"]
    model_labels = ["Foundation-OMAT", "From-scratch", "Naive-FT", "Multi-T"]
    settings = [
        "nve_dt0p5", "nve_dt1p0", "nve_dt2p0", "langevin_f0_dt1p0",
        "langevin_f0p001_dt1p0", "langevin_f0p01_dt1p0",
    ]
    setting_labels = ["NVE\ndt=0.5", "NVE\ndt=1", "NVE\ndt=2", "Langevin\nf=0", "Langevin\nf=.001", "Langevin\nf=.01"]
    code = {"stable_short_handoff": 0, "bounded_with_structural_shift": 1, "runaway_or_nonfinite": 2}
    text = {0: "stable", 1: "bounded", 2: "runaway"}
    matrix_rows: list[dict] = []
    for row in rows:
        matrix_rows.append({key: row.get(key, "") for key in ["snapshot", "model", "setting", "classification", "first_divergence_time_ps", "max_force_eV_A"]})
    write_table(package / "processed_sensitivity_summary.csv", matrix_rows, list(matrix_rows[0]))

    palette = get_palette("immuno_soft")
    colors = palette["categorical"]
    render_files: list[str] = []

    def matrix_panel(snapshot: str, ax) -> None:
        matrix = np.array([[code[next(row["classification"] for row in rows if row["snapshot"] == snapshot and row["model"] == model and row["setting"] == setting)] for setting in settings] for model in models], dtype=float)
        clustered_heatmap(ax, matrix, model_labels, setting_labels, low="#D8F0EA", mid="#F2C14E", high="#A61C3C", cluster=False, colorbar=False)
        label_grid = [[text[int(value)] for value in matrix_row] for matrix_row in matrix]
        style_heatmap_labels(ax, label_grid, title=f"Input snapshot {snapshot}")
        ax.tick_params(axis="x", labelrotation=0)
        ax.set_xlabel("Integrator / timestep / bath")
        ax.set_ylabel("Model")
        ax.legend(handles=[Patch(color="#D8F0EA", label="stable"), Patch(color="#F2C14E", label="bounded / shift"), Patch(color="#A61C3C", label="runaway / nonfinite")], frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(0, -0.28), ncol=3)

    def summary_panel(ax) -> None:
        matrix = []
        for model in models:
            values = []
            for snapshot in ["461800", "462000"]:
                subset = [row for row in rows if row["model"] == model and row["snapshot"] == snapshot]
                values.append(sum(row["classification"] == "runaway_or_nonfinite" for row in subset) / len(subset))
            matrix.append(values)
        grouped_bar(ax, model_labels, np.asarray(matrix), ["461800", "462000"], colors[:2], ylabel="Runaway fraction", legend=True)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, .25, .5, .75, 1.0], ["0", ".25", ".5", ".75", "1"])
        ax.set_title("Runaway fraction across the six settings", pad=8, fontweight="bold")

    def divergence_panel(ax) -> None:
        x = np.arange(len(settings))
        ys = []
        labels = []
        for model, label in zip(models[:2], model_labels[:2]):
            for snapshot in ["461800", "462000"]:
                ys.append([numeric(next(row["first_divergence_time_ps"] for row in rows if row["snapshot"] == snapshot and row["model"] == model and row["setting"] == setting)) for setting in settings])
                labels.append(f"{label} / {snapshot}")
        timecourse(ax, x, ys, labels, colors[:4], xlabel="Setting", ylabel="First divergence time (ps)", legend=True, markers=True)
        ax.set_xticks(x, setting_labels, rotation=30, ha="right")
        ax.set_title("First-gate timing for unstable model families", pad=8, fontweight="bold")
        ax.set_ylim(bottom=0)

    fig = plt.figure(figsize=(12, 8))
    grid = fig.add_gridspec(2, 2, hspace=.70, wspace=.32)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    matrix_panel("461800", axes[0]); matrix_panel("462000", axes[1]); summary_panel(axes[2]); divergence_panel(axes[3])
    for ax, label in zip(axes, "ABCD"):
        panel_label(ax, label)
    fig.suptitle("MD-SENSITIVITY-48RUN | integrator / thermostat ablation", fontsize=12, fontweight="bold")
    render_files.extend(save_render(fig, package / "phase2_sensitivity_ablation_atlas"))

    caption = "A–B, stability classification for 48 complete short-handoff runs across two input snapshots, four model families, and six integrator/thermostat settings; C, runaway fraction by snapshot; D, first divergence timing for Foundation-OMAT and from-scratch branches."
    results = "Foundation-OMAT and from-scratch are classified as runaway/nonfinite for all six settings at both snapshots. The smaller 0.5 fs timestep delays some first crossings but does not restore a finite bounded handoff. Naive-FT remains stable for 461800 and bounded/structurally shifted for 462000; Multi-T is stable for 461800 and bounded/structurally shifted for 462000."
    discussion = "The table-backed ablation supports a model-dependent PES-instability interpretation rather than a current Langevin-bath-only explanation. It is a short-range counterfactual stability result, not a diffusivity, DFT-validation, or long-trajectory chemical-mechanism result."
    write_package(package, "MD-SENSITIVITY-48RUN", [source, readme], render_files, [package / "processed_sensitivity_summary.csv"], "Results/SI: integrator/thermostat sensitivity and dynamic-stability boundary", "complete short-handoff ablation; not a transport result", caption, results, discussion, ["All 48 metrics files have 202 rows and the campaign README records input/model/cell hash checks.", "Do not infer thermostat/timestep irrelevance beyond the tested settings.", "Keep this package separate from the 1 ns transport figures."])


def render_candidate_eval() -> None:
    package = OUT_ROOT / "MACE-CANDIDATE-EVAL"
    package.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT / "analysis/mace_candidate_eval_20261001/remote_results/mace_single_point_results.csv"
    coverage = PROJECT_ROOT / "analysis/mace_candidate_eval_20261001/coverage_summary.json"
    rows = read_csv(source)
    family_order = ["Foundation", "FT-600K", "MPA0", "FT-MultiT", "Scratch", "FT-600K-grouped", "MPA0-grouped", "Scratch-grouped", "Scratch5-grouped"]
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_family[row["model_family"]].append(row)
    summary_rows = []
    for family in family_order:
        subset = by_family[family]
        n_success = sum(row["status"] == "success" for row in subset)
        summary_rows.append({"model_family": family, "n_total": len(subset), "n_success": n_success, "n_runtime_failure": len(subset) - n_success, "success_fraction": n_success / len(subset) if subset else 0.0})
    force_fields = ["candidate_id", "model_label", "model_family", "category", "max_force_eV_A", "rms_force_eV_A", "min_interatomic_distance_A", "status"]
    force_rows = [
        {key: row.get(key, "") for key in force_fields}
        for row in rows
        if row["status"] == "success" and math.isfinite(numeric(row["max_force_eV_A"]))
    ]
    write_table(package / "processed_candidate_eval_summary.csv", summary_rows, list(summary_rows[0]))
    write_table(package / "processed_success_force_rows.csv", force_rows, force_fields)

    palette = get_palette("isotope_sites")
    colors = palette["categorical"]
    render_files: list[str] = []
    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 2, hspace=.58, wspace=.28)
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    means = [row["success_fraction"] for row in summary_rows]
    bar_with_points(axes[0], family_order, means, [0.0] * len(means), colors * 5, points=[[value] for value in means], ylabel="Successful model evaluations / all evaluations", rotate=35)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Runtime coverage by model family", pad=8, fontweight="bold")
    axes[0].text(.01, .97, "195/975 rows are successful; 780 rows are runtime failures", transform=axes[0].transAxes, va="top", fontsize=7)

    categories = ["stable endpoint", "NEB intermediate", "near saddle"]
    category_rows = []
    for category in ["stable_endpoint", "neb_intermediate", "near_saddle"]:
        subset = [row for row in rows if row["category"] == category]
        category_rows.append([sum(row["status"] == "success" for row in subset), sum(row["status"] != "success" for row in subset)])
    grouped_bar(axes[1], categories, np.asarray(category_rows), ["success", "runtime failure"], colors[:2], ylabel="Rows", legend=True)
    axes[1].set_title("Candidate-category coverage", pad=8, fontweight="bold")

    force_models = ["Foundation", "FT-600K", "MPA0"]
    points = [[numeric(row["max_force_eV_A"]) for row in force_rows if row["model_label"] == model] for model in force_models]
    medians = [float(np.nanmedian(values)) for values in points]
    errors = [float((np.nanpercentile(values, 75) - np.nanpercentile(values, 25)) / 2) for values in points]
    bar_with_points(axes[2], force_models, medians, errors, colors[:3], points=points, ylabel="Maximum force (eV Å$^{-1}$)")
    axes[2].set_yscale("log")
    axes[2].set_title("Successful-row force scale", pad=8, fontweight="bold")
    axes[2].text(.02, .03, "successful rows only; not a cross-family accuracy ranking", transform=axes[2].transAxes, fontsize=6.5)
    for ax, label in zip(axes, "ABC"):
        panel_label(ax, label)
    fig.suptitle("MACE-CANDIDATE-EVAL | 975 single-point rows", fontsize=12, fontweight="bold")
    render_files.extend(save_render(fig, package / "mace_candidate_eval_atlas"))

    caption = "A, runtime coverage across the nine registered model families; B, successful versus runtime-failure rows by candidate type; C, maximum-force distribution for the three model labels with successful rows."
    results = "The table contains 975 rows: 195 successful evaluations and 780 runtime failures. The successful rows are limited to Foundation, FT-600K, and MPA0 labels; the grouped-seed, Scratch, and FT-MultiT branches are not scientifically failed model evaluations here because their recorded runtime error is the missing NVIDIA driver."
    discussion = "This figure is a campaign/runtime coverage audit and a force-scale diagnostic. It should be used to explain why the current candidate-evaluation ablation cannot support a complete model-family ranking or grouped-seed conclusion."
    write_package(package, "MACE-CANDIDATE-EVAL", [source, coverage], render_files, [package / "processed_candidate_eval_summary.csv", package / "processed_success_force_rows.csv"], "Results/SI: candidate selection and model-evaluation coverage audit", "runtime-coverage diagnostic; not a model accuracy benchmark", caption, results, discussion, ["The runtime failure traceback identifies a missing NVIDIA driver; do not relabel those rows as model-quality failures.", "Absolute energies are not comparable across model families; this package uses status and force-scale fields only.", "The 65-candidate manifest and selection provenance remain the source of candidate identity."])


def render_phase3_input_audit() -> None:
    package = OUT_ROOT / "DFT-PHASE3-INPUT-AUDIT"
    package.mkdir(parents=True, exist_ok=True)
    full_source = PROJECT_ROOT / "analysis/dft_validation_phase3_20261002/snapshot_audit.csv"
    block_source = PROJECT_ROOT / "analysis/dft_validation_phase3_20261002/ordered_block_audit.csv"
    full_rows = read_csv(full_source)
    block_rows = read_csv(block_source)
    rows = []
    for row in full_rows:
        rows.append({"series": "full_2050", "step": row["step"], "force_max_eV_A": row["force_max_eV_A"], "cr_force_max_eV_A": row["cr_force_max_eV_A"], "natoms": row["natoms"]})
    for row in block_rows:
        name = Path(row["file"]).name
        series = "ordered_82" if "ordered82" in name else "same_site_81"
        rows.append({"series": series, "step": row["step"], "force_max_eV_A": row["force_max_eV_A"], "cr_force_max_eV_A": row["cr_force_max_eV_A"], "natoms": row["natoms"]})
    write_table(package / "processed_phase3_input_force_audit.csv", rows, list(rows[0]))
    palette = get_palette("immuno_soft")
    labels = ["Full 2050", "Ordered 82", "Same-site 81"]
    series_names = ["full_2050", "ordered_82", "same_site_81"]
    steps = np.array([460000, 461800, 462000, 462200])
    render_files: list[str] = []
    fig = plt.figure(figsize=(10, 4.4))
    grid = fig.add_gridspec(1, 2, wspace=.30)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    for ax, field, ylabel, title in [(axes[0], "force_max_eV_A", "Maximum force (eV Å$^{-1}$)", "All-atom force audit"), (axes[1], "cr_force_max_eV_A", "Cr-local maximum force (eV Å$^{-1}$)", "Cr-local force audit")]:
        ys = []
        for series in series_names:
            ys.append([numeric(next(row[field] for row in rows if row["series"] == series and int(row["step"]) == int(step))) for step in steps])
        timecourse(ax, steps, ys, labels, palette["categorical"][:3], xlabel="Source trajectory step", ylabel=ylabel, legend=True, markers=True)
        ax.set_yscale("log")
        ax.set_title(title, pad=8, fontweight="bold")
    panel_label(axes[0], "A"); panel_label(axes[1], "B")
    fig.suptitle("DFT-PHASE3-INPUT-AUDIT | MACE snapshot and reduced-block force diagnostics", fontsize=11, fontweight="bold")
    render_files.extend(save_render(fig, package / "phase3_input_force_audit_atlas"))
    caption = "A, maximum all-atom force; B, Cr-local maximum force for the full 2,050-atom source snapshot and two reduced ordered-block representations across four source steps."
    results = "The full source snapshot rises from 2.83 eV Å⁻¹ at step 460000 to 229.97 eV Å⁻¹ at step 462200. The reduced 82- and 81-atom representations retain the same source-step progression but are different validation units."
    discussion = "This is a pre-DFT input-integrity and stress-test figure. It documents why steps 462000 and 462200 are OOD/stress geometries and why a successful reduced-cell QE single point would still be a local comparison, not a converged migration barrier."
    write_package(package, "DFT-PHASE3-INPUT-AUDIT", [full_source, block_source], render_files, [package / "processed_phase3_input_force_audit.csv"], "Results/SI: reduced-cell DFT validation boundary", "input audit only; no accepted target DFT output", caption, results, discussion, ["These are MACE forces stored in the source extxyz audits, not QE forces.", "The 81-atom model removes one Cr and is explicitly a same-site composition approximation.", "No plot panel may be used to claim DFT convergence or a barrier."])


def render_dft_gate_status() -> None:
    package = OUT_ROOT / "DFT-SINGLE-POINT-GATE"
    package.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT / "analysis/dft_single_point_validation_20261001/dft_single_point_results.csv"
    rows = read_csv(source)
    order = ["stable_endpoint", "neb_intermediate", "near_saddle", "failure_evidence"]
    labels = ["Stable endpoint", "NEB intermediate", "Near saddle", "Failure evidence"]
    summary = []
    for category, label in zip(order, labels):
        subset = [row for row in rows if row["category"] == category]
        summary.append({"category": category, "label": label, "n_total": len(subset), "n_accepted": sum(row["dft_status"] == "accepted" for row in subset), "n_not_accepted": sum(row["dft_status"] != "accepted" for row in subset)})
    write_table(package / "processed_dft_gate_summary.csv", summary, list(summary[0]))
    palette = get_palette("isotope_sites")
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    grouped_bar(ax, labels, np.asarray([[item["n_accepted"], item["n_not_accepted"]] for item in summary]), ["accepted", "not accepted"], palette["categorical"][:2], ylabel="Candidate count", legend=True)
    ax.set_title("Independent DFT single-point gate status", pad=8, fontweight="bold")
    ax.text(.01, .97, "0/20 accepted; no finite energy/force table is available for a scientific comparison", transform=ax.transAxes, va="top", fontsize=7)
    panel_label(ax, "A")
    render_files = save_render(fig, package / "dft_single_point_gate_status")
    caption = "Candidate-level status of the independent DFT single-point wave, separated into stable endpoints, NEB intermediates, near-saddle structures, and failure-evidence frames."
    results = "All 20 audited candidates are marked not accepted: 5 stable endpoints, 5 NEB intermediates, 5 near-saddle structures, and 5 failure-evidence frames. The current table contains no accepted finite DFT energy/force values."
    discussion = "This figure is a transparent negative/gate result. It explains why MACE candidate tables cannot yet be paired with an accepted DFT target, and it prevents a missing-output campaign from being mistaken for a zero-error benchmark."
    write_package(package, "DFT-SINGLE-POINT-GATE", [source], render_files, [package / "processed_dft_gate_summary.csv"], "Results/SI: DFT validation coverage and acceptance boundary", "gate-status diagnostic; no accepted DFT result", caption, results, discussion, ["Pending/missing QE output is not treated as a scientific zero.", "The MD snapshot subset also retains an explicit missing-provenance gate.", "This figure should not be used as a DFT energy/force plot."])


def render_foundation_gate() -> None:
    package = OUT_ROOT / "FOUNDATION-PATH-GATE"
    package.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT / "analysis/task1_foundation_path/foundation_model_path_gate.csv"
    schema = PROJECT_ROOT / "analysis/task1_foundation_path/figure_data_dictionary.json"
    rows = read_csv(source)
    dopants = sorted({row["dopant"] for row in rows})
    paths = sorted({row["path_name"] for row in rows})
    summary = [{"dopant": dopant, "path_name": path, "n_rows": sum(row["dopant"] == dopant and row["path_name"] == path for row in rows), "n_scientific_success": sum(row["dopant"] == dopant and row["path_name"] == path and row["scientific_success"].lower() == "true" for row in rows)} for dopant in dopants for path in paths]
    write_table(package / "processed_foundation_gate_summary.csv", summary, list(summary[0]))
    matrix = np.array([[sum(row["dopant"] == dopant and row["path_name"] == path and row["scientific_success"].lower() == "true" for row in rows) for path in paths] for dopant in dopants], dtype=float)
    labels_grid = [["blocked" if value == 0 else "pass" for value in row] for row in matrix]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    clustered_heatmap(ax, matrix, dopants, paths, low="#F2C14E", mid="#F2C14E", high="#2A9D8F", cluster=False, colorbar=False)
    style_heatmap_labels(ax, labels_grid, title="Foundation path gate status")
    ax.set_xlabel("Path family"); ax.set_ylabel("Dopant")
    ax.text(.01, -.28, "All rows are blocked by the explicit DFT reference gate; this is not a path-accuracy result.", transform=ax.transAxes, fontsize=7)
    panel_label(ax, "A")
    render_files = save_render(fig, package / "foundation_path_gate_status")
    caption = "Explicit Foundation path-gate status by dopant and path family. A cell is marked pass only when the DFT reference path and model-path evidence satisfy the declared gate."
    results = "Every audited cell is blocked because no DFT path passed the explicit convergence, finite-image energy/error, identity, cell/PBC, and provenance requirements."
    discussion = "The table supports a transparent decision not to fabricate Foundation-vs-DFT profiles, barrier scatter, or perturbation stability plots. A gate-status panel is appropriate for the audit; scientific profile panels remain schema-only until accepted numerical rows exist."
    write_package(package, "FOUNDATION-PATH-GATE", [source, schema], render_files, [package / "processed_foundation_gate_summary.csv"], "Results/SI: Foundation path validation gate", "gate-only diagnostic; required numerical profiles unavailable", caption, results, discussion, ["The figure_data_dictionary explicitly prohibits fabricated scientific renders.", "No Foundation path accuracy or barrier claim is made.", "When accepted DFT/model rows arrive, regenerate the planned profile and force-error panels from the same package."])


def render_neb_topology_audit() -> None:
    package = OUT_ROOT / "NEB-TOPOLOGY-AUDIT"
    package.mkdir(parents=True, exist_ok=True)
    source = Path("/Users/alina/Project/migrationbench-paper-repro/data/published/path_topology_audit/path_topology_summary.csv")
    segmented = Path("/Users/alina/Project/migrationbench-paper-repro/data/published/path_topology_audit/segmented_barriers.csv")
    readme = Path("/Users/alina/Project/migrationbench-paper-repro/data/published/path_topology_audit/README.md")
    rows = read_csv(source)
    selected = [{key: row[key] for key in ["path_id", "n_atoms", "n_images", "activation_forward_qe_eV", "max_image_error_eV_A", "internal_basin_candidate_count", "topology_status", "minimum_pair_distance_A", "geometry_gate_pass"]} for row in rows]
    write_table(package / "processed_topology_summary.csv", selected, list(selected[0]))
    palette = get_palette("agro_joint")
    x = np.asarray([numeric(row["activation_forward_qe_eV"]) for row in rows])
    y = np.asarray([numeric(row["max_image_error_eV_A"]) for row in rows])
    groups = np.asarray(["geometry gate pass" if row["geometry_gate_pass"].lower() == "true" else "geometry gate fail" for row in rows])
    order = np.argsort(x)
    fig = plt.figure(figsize=(12, 5.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.2], wspace=.34)
    scatter = joint_scatter(fig, grid[0], x, y, palette, groups=groups, regression=True, xlabel="Historical QE forward barrier (eV)", ylabel="Maximum image error (eV Å$^{-1}$)", title="Barrier / image-error relation", marginal_axes=True, annotate=True)
    ax_bar = fig.add_subplot(grid[1])
    labels = [rows[index]["path_id"].replace("_historical", "") for index in order]
    values = [x[index] for index in order]
    bar_with_points(ax_bar, labels, values, [0.0] * len(values), palette["categorical"] * 6, rotate=65, ylabel="Historical QE forward barrier (eV)")
    ax_bar.set_title("Path-level forward-barrier inventory", pad=8, fontweight="bold")
    ax_bar.text(.01, .98, "endpoint stability remains unverified", transform=ax_bar.transAxes, va="top", fontsize=7)
    panel_label(scatter["main"], "A"); panel_label(ax_bar, "B")
    fig.suptitle("NEB-TOPOLOGY-AUDIT | hidden-basin and endpoint-dominated path screening", fontsize=11, fontweight="bold")
    render_files = save_render(fig, package / "neb_topology_audit_atlas")
    caption = "A, historical QE forward-barrier values versus maximum image force error, colored by the geometry-gate flag; B, sorted path-level forward-barrier inventory."
    results = "The table contains historical and 81-atom path entries with hidden-basin candidates and endpoint-dominated segments. Several paths have geometry-gate failure or large image errors, while the 0.336 eV 1-4 value is retained as a candidate path value rather than an accepted barrier."
    discussion = "This figure is a topology/provenance audit for deciding which path segments require independent endpoint relaxation. It does not certify endpoint stability, NEB convergence, or a publication barrier ranking."
    write_package(package, "NEB-TOPOLOGY-AUDIT", [source, segmented, readme], render_files, [package / "processed_topology_summary.csv"], "SI Detailed Migration Pathway Analysis: historical path topology and hidden-basin audit", "historical/reconstruction audit; not an accepted DFT barrier figure", caption, results, discussion, ["Endpoint stability is explicitly unverified because inspected endpoints are frozen.", "The geometry flag is not a convergence certificate.", "Keep this family separate from current Cr-61/Cr-81 exploratory path figures."])


def render_dft_recovery_reconstruction() -> None:
    package = OUT_ROOT / "DFT-RECOVERY-RECONSTRUCTION"
    package.mkdir(parents=True, exist_ok=True)
    source = Path("/Users/alina/Project/migrationbench-paper-repro/analysis/phase3_dft_recovery_20261004/results_reconstructed/dft_vs_mace_within_method.csv")
    gate = Path("/Users/alina/Project/migrationbench-paper-repro/analysis/phase3_dft_recovery_20261004/results_reconstructed/qe_gate_audit.json")
    rows = read_csv(source)
    selected = [{key: row[key] for key in ["case", "model", "dft_max_force_eV_A", "mace_max_force_eV_A", "force_mae_eV_A", "force_rmse_eV_A", "scope"]} for row in rows]
    write_table(package / "processed_dft_vs_mace_reconstruction.csv", selected, list(selected[0]))
    palette = get_palette("isotope_sites")
    models = ["foundation-omat", "from-scratch", "naive-fine-tuning", "multi_T-fine-tuning"]
    model_labels = ["Foundation", "From-scratch", "Naive-FT", "Multi-T"]
    cases = list(dict.fromkeys(row["case"] for row in rows))
    case_labels = [case.replace("_", " ").replace("normal", "normal").replace("bounded", "bounded").replace("runaway", "runaway") for case in cases]
    fig = plt.figure(figsize=(12, 5.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=.35)
    ax_rmse = fig.add_subplot(grid[0])
    matrix = []
    for case in cases:
        matrix.append([numeric(next(row["force_rmse_eV_A"] for row in rows if row["case"] == case and row["model"] == model)) for model in models])
    grouped_bar(ax_rmse, case_labels, np.asarray(matrix), model_labels, palette["categorical"][:4], ylabel="Force RMSE (eV Å$^{-1}$)", legend=True)
    ax_rmse.tick_params(axis="x", labelrotation=35)
    ax_rmse.set_title("Reconstructed force-error comparison", pad=8, fontweight="bold")
    ax_scatter = fig.add_subplot(grid[1])
    dft = np.asarray([numeric(row["dft_max_force_eV_A"]) for row in rows])
    mace = np.asarray([numeric(row["mace_max_force_eV_A"]) for row in rows])
    groups = np.asarray([row["model"] for row in rows])
    # Use the atlas scatter grammar on a local GridSpec slot, then hide the
    # marginal axes only where the composite does not need their tick labels.
    inner = grid[1].subgridspec(1, 1)
    ax_scatter.remove()
    scatter = joint_scatter(fig, inner[0], dft, mace, palette, groups=groups, regression=True, xlabel="DFT max force (eV Å$^{-1}$)", ylabel="MACE max force (eV Å$^{-1}$)", title="Within-method force-scale reconstruction", marginal_axes=False, annotate=True)
    scatter["main"].set_xscale("log"); scatter["main"].set_yscale("log")
    panel_label(ax_rmse, "A"); panel_label(scatter["main"], "B")
    fig.suptitle("DFT-RECOVERY-RECONSTRUCTION | within-method force audit", fontsize=11, fontweight="bold")
    render_files = save_render(fig, package / "dft_recovery_reconstruction_atlas")
    caption = "A, reconstructed DFT-versus-MACE force RMSE by recovery case and model; B, maximum-force relation for the same reconstructed rows."
    results = "The reconstruction table contains normal, bounded, and runaway cases with finite-looking DFT/MACE comparison fields, but every row is explicitly labeled RECONSTRUCTION / REPRODUCED_SCOPE_LIMITED in the source table. Model force errors increase strongly in the runaway cases."
    discussion = "This package is useful for auditing the DFT-recovery boundary and for deciding whether a later accepted QE release can replace the reconstruction. It must not be presented as a converged DFT validation benchmark."
    write_package(package, "DFT-RECOVERY-RECONSTRUCTION", [source, gate], render_files, [package / "processed_dft_vs_mace_reconstruction.csv"], "Results/SI: DFT recovery and MACE force-validation boundary", "reconstruction-only diagnostic; source scope limited", caption, results, discussion, ["The source scope is RECONSTRUCTION / REPRODUCED_SCOPE_LIMITED.", "A finite reconstructed row is not equivalent to a normal, accepted QE target.", "Re-run or replace this package after raw QE outputs and gate manifests are independently closed."])


def main() -> None:
    apply_publication_style(8.2)
    render_phase2_sensitivity()
    render_candidate_eval()
    render_phase3_input_audit()
    render_dft_gate_status()
    render_foundation_gate()
    render_neb_topology_audit()
    render_dft_recovery_reconstruction()
    print(f"wrote table-backed packages under {OUT_ROOT}")


if __name__ == "__main__":
    main()
