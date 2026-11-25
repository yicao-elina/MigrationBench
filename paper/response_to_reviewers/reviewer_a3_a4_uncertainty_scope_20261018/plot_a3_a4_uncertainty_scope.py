"""Build reproducible A3/A4 reviewer-response evidence assets.

The script is intentionally additive. It reads the existing grouped-retraining
package, the current NEB protocol matrix, and the archived broad-roster
cross-element diagnostic; it never edits those source assets.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

PLOT_LIBRARY = Path("/Users/alina/Documents/Codex/plot_library/src")
if str(PLOT_LIBRARY) not in sys.path:
    sys.path.insert(0, str(PLOT_LIBRARY))

from plot_atlas.charts import bar_with_points, forest_effects
from plot_atlas.palettes import get_palette
from plot_atlas.style import apply_publication_style, panel_label, save_figure


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
DATA = PACKAGE / "data"
FIGURES = PACKAGE / "figures"
TABLES = PACKAGE / "tables"

GROUPED = ROOT / "paper/response_to_reviewers/reviewer_a2_grouped_retraining/data/grouped_neb_barrier_summary.csv"
MATRIX = ROOT / "analysis/neb_protocol_matrix_20261018/derived/neb_matrix_status.csv"
CROSS = Path(
    "/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/"
    "analysis/task2_cross_element_broad_roster_final_20261006"
)
CROSS_DFT = CROSS / "results/table_1_dft_quality.csv"
CROSS_VALIDATION = CROSS / "results/broad_roster/seed_validation_metrics.csv"
CROSS_STABILITY = CROSS / "results/broad_roster/table_4_seed_path_stability.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def tex_number(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def tex_model(name: str) -> str:
    return {
        "FT-600K": "FT-600K",
        "Scratch": "Scratch",
        "Scratch-5%": "Scratch-5\\%",
        "Scratch-MultiT": "Scratch-Multi$_T$",
    }.get(name, name)


def tex_path(path: str) -> str:
    return path.replace("_", "\\_")


def build_a3_data() -> tuple[list[dict], list[dict]]:
    source = read_csv(GROUPED)
    model_prefix = {
        "scratch_grouped": "Scratch",
        "ft600k_omatft_grouped": "FT-600K",
        "scratch5_grouped": "Scratch-5%",
    }
    path_class = {
        "1-2": "deep-penetration candidate",
        "1-3": "deep-penetration candidate",
        "1-4": "in-gap controlled probe",
        "1-5": "deep-penetration candidate; geometry recheck",
        "1-7": "in-gap candidate; auxiliary",
    }
    rows = []
    for row in source:
        match = next((prefix for prefix in model_prefix if row["model_id"].startswith(prefix)), None)
        if match is None:
            continue
        rows.append(
            {
                "pathway": row["pathway"],
                "model": model_prefix[match],
                "seed": row["model_id"].rsplit("seed", 1)[-1],
                "absolute_error_eV": float(row["absolute_error_ev"]),
                "path_class": path_class[row["pathway"]],
                "reference_status": "provisional DFT candidate",
            }
        )
    write_csv(DATA / "a3_grouped_path_errors_per_seed.csv", rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["pathway"], row["model"])].append(row["absolute_error_eV"])
    summary = []
    order = ["1-2", "1-3", "1-4", "1-5", "1-7"]
    for pathway in order:
        for model in ["FT-600K", "Scratch", "Scratch-5%"]:
            values = grouped.get((pathway, model))
            if not values:
                continue
            avg, sd = mean_sd(values)
            summary.append(
                {
                    "pathway": pathway,
                    "model": model,
                    "seed_values_eV": ";".join(f"{v:.9f}" for v in values),
                    "mean_eV": avg,
                    "sample_sd_eV": sd,
                    "n_seeds": len(values),
                    "path_class": path_class[pathway],
                    "reference_status": "provisional DFT candidate",
                }
            )
    write_csv(DATA / "a3_grouped_path_errors_summary.csv", summary)
    return rows, summary


def build_matrix_data() -> list[dict]:
    source = read_csv(MATRIX)
    selected = []
    for row in source:
        if (
            row["protocol"] == "fixed_path"
            and row["cell_atoms"] == "61"
            and row["family"] in {"FT-600K", "Scratch-MultiT"}
            and row["path_id"] in {"1-2", "1-3", "1-4", "1-5", "1-7"}
        ):
            selected.append(
                {
                    "family": row["family"],
                    "pathway": row["path_id"],
                    "seed": row["seed"],
                    "absolute_error_eV": abs(float(row["barrier_error_eV"])),
                    "publication_status": row["publication_status"],
                    "failure_flag": row["failure_flag"],
                    "reference_status": row["reference_status"],
                }
            )
    write_csv(DATA / "a3_protocol_matrix_61_fixed_path_per_seed.csv", selected)

    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["family"], row["pathway"])].append(row["absolute_error_eV"])
    summary = []
    for family in ["FT-600K", "Scratch-MultiT"]:
        for pathway in ["1-2", "1-3", "1-4", "1-5", "1-7"]:
            values = grouped[(family, pathway)]
            avg, sd = mean_sd(values)
            failure_count = sum(
                1
                for r in selected
                if r["family"] == family and r["pathway"] == pathway and r["failure_flag"]
            )
            summary.append(
                {
                    "family": family,
                    "pathway": pathway,
                    "mean_eV": avg,
                    "sample_sd_eV": sd,
                    "n_seeds": len(values),
                    "force_pathology_cells": failure_count,
                    "status": "supporting matrix; not pooled with strict grouped lineage",
                }
            )
    write_csv(DATA / "a3_protocol_matrix_61_fixed_path_summary.csv", summary)
    return summary


def build_a4_data() -> list[dict]:
    dft = read_csv(CROSS_DFT)
    validation = read_csv(CROSS_VALIDATION)
    stability = read_csv(CROSS_STABILITY)
    write_csv(DATA / "a4_cross_element_dft_quality.csv", dft)
    write_csv(DATA / "a4_cross_element_seed_validation.csv", validation)
    write_csv(DATA / "a4_cross_element_path_stability.csv", stability)

    stable_energy = {
        row["path_id"]: row
        for row in stability
        if row["metric"] == "max_relative_energy_eV"
    }
    compact = []
    for row in dft:
        stability_row = stable_energy[row["path_id"]]
        compact.append(
            {
                "element": row["element"],
                "path_id": row["path_id"],
                "dft_scientific_success": row["dft_scientific_success"],
                "explicit_neb_convergence": row["explicit_neb_convergence"],
                "screen_delta_lt_0p1_eV": row["screen_delta_lt_0p1_eV"],
                "last_two_max_abs_image_delta_eV": float(row["last_two_max_abs_image_delta_eV"]),
                "max_relative_energy_mean_eV": float(stability_row["seed_mean"]),
                "max_relative_energy_sd_eV": float(stability_row["seed_std"]),
                "seed_range_eV": float(stability_row["seed_range"]),
                "interpretation": "diagnostic snapshot only; DFT scientific gate failed",
            }
        )
    write_csv(DATA / "a4_cross_element_compact_summary.csv", compact)
    return compact


def write_tables(a3_summary: list[dict], matrix_summary: list[dict], cross_summary: list[dict]) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Class-resolved kinetic errors.} Entries are absolute fixed-path barrier errors relative to the QE barrier of each path (mean $\pm$ sample standard deviation over three independent training initializations); path names are defined in the SI path registry. G1 uses the 0.336~eV QE barrier; for the other paths the reference is the QE energy profile of the same images (SI NEB barrier table). Errors are fixed-path errors on identical images, reported per path and not pooled.}",
        r"\label{tab:a3_strict_path_errors}",
        r"\begin{tabular}{llc}",
        r"\toprule",
        r"Pathway class & Model & Mean $\pm$ SD (eV) \\",
        r"\midrule",
    ]
    selected = [
        ("1-2", "Scratch"),
        ("1-2", "FT-600K"),
        ("1-3", "Scratch"),
        ("1-3", "FT-600K"),
        ("1-4", "Scratch"),
        ("1-4", "FT-600K"),
        ("1-4", "Scratch-5%"),
        ("1-5", "Scratch"),
        ("1-5", "FT-600K"),
        ("1-7", "Scratch"),
        ("1-7", "FT-600K"),
    ]
    class_label = {
        "1-2": "Q1 (gap-to-QL/into-QL)",
        "1-3": "Q2 (gap-to-QL/into-QL)",
        "1-4": "G1 (within-gap)",
        "1-5": "Q3 (gap-to-QL/into-QL)",
        "1-7": "G2 (within-gap)",
    }
    by_key = {(r["pathway"], r["model"]): r for r in a3_summary}
    for pathway, model in selected:
        row = by_key[(pathway, model)]
        mean_text = "$" + tex_number(row["mean_eV"]) + r" \pm " + tex_number(row["sample_sd_eV"]) + "$"
        lines.append(f"{class_label[pathway]} & {tex_model(model)} & {mean_text} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "a3_strict_path_errors.tex").write_text("\n".join(lines) + "\n")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{\textbf{Supporting 61-atom fixed-path protocol matrix.} Values are absolute errors from the current matrix and are shown as mean $\pm$ sample SD across seeds 123/234/345. Scratch-Multi$_T$ is a separate lineage from the strict grouped Scratch arm. Force-pathology cells are retained and are not silently treated as accurate barriers.}",
        r"\label{tab:a3_protocol_matrix}",
        r"\begin{tabular}{llcccr}",
        r"\toprule",
        r"Family & Audit ID (SI only) & Mean (eV) & SD (eV) & Runs & Force-pathology cells \\",
        r"\midrule",
    ]
    for row in matrix_summary:
        lines.append(
            f"{tex_model(row['family'])} & {row['pathway']} & "
            f"{tex_number(row['mean_eV'])} & {tex_number(row['sample_sd_eV'])} & "
            f"{row['n_seeds']} & {row['force_pathology_cells']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "a3_protocol_matrix.tex").write_text("\n".join(lines) + "\n")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\tiny",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\caption{\textbf{A4 cross-element diagnostic.} The Cr-trained broad-roster models were evaluated on archived Cu/Li/Na paths in the same host. All ten DFT records retain SCF-convergence warnings and therefore fail the scientific DFT gate. The final-column diagnostic is the mean $\pm$ sample SD of the maximum MLIP relative energy across three checkpoints; it is not a DFT barrier error.}",
        r"\label{tab:a4_cross_element_diagnostic}",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Element & Path & DFT gate & $<0.1$ eV screen & Max relative energy (eV) & Seed range (eV) \\",
        r"\midrule",
    ]
    for row in cross_summary:
        gate = "failed" if row["dft_scientific_success"] == "False" else "passed"
        screen = "yes" if row["screen_delta_lt_0p1_eV"] == "True" else "no"
        mean_text = "$" + tex_number(row["max_relative_energy_mean_eV"]) + r" \pm " + tex_number(row["max_relative_energy_sd_eV"]) + "$"
        lines.append(
            f"{row['element']} & \\texttt{{{tex_path(row['path_id'])}}} & {gate} & {screen} & "
            f"{mean_text} & {tex_number(row['seed_range_eV'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABLES / "a4_cross_element_diagnostic.tex").write_text("\n".join(lines) + "\n")


def make_a3_figure(a3_summary: list[dict]) -> None:
    apply_publication_style(8.0)
    palette = get_palette("jhu_mace_models")
    model_colors = {
        "FT-600K": palette["roles"]["ft_600k"],
        "Scratch": palette["roles"]["from_scratch"],
        "Scratch-5%": "#F4B65D",
    }
    by_key = {(r["pathway"], r["model"]): r for r in a3_summary}

    fig = plt.figure(figsize=(7.25, 3.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.36], wspace=0.36)

    ax = fig.add_subplot(gs[0, 0])
    models = ["FT-600K", "Scratch", "Scratch-5%"]
    points = [
        [float(v) for v in by_key[("1-4", model)]["seed_values_eV"].split(";")]
        for model in models
    ]
    means = [float(by_key[("1-4", model)]["mean_eV"]) for model in models]
    errors = [float(by_key[("1-4", model)]["sample_sd_eV"]) for model in models]
    bar_with_points(
        ax,
        ["FT-600K", "Scratch", "Scratch-5\\%"],
        means,
        errors,
        [model_colors[m] for m in models],
        points=points,
        ylabel="Within-gap $|$barrier error$|$ (eV)",
        rotate=28,
    )
    ax.axhline(0.16, color="#666666", linestyle=(0, (3, 2)), linewidth=0.8)
    ax.set_ylim(0, 1.65)
    ax.set_title("Within-gap path G1")
    panel_label(ax, "a", x=-0.18, y=1.10)

    ax = fig.add_subplot(gs[0, 1])
    paths = ["Q1", "Q2", "Q3"]
    source_paths = ["1-2", "1-3", "1-5"]
    effects = []
    lows = []
    highs = []
    for model in ["Scratch", "FT-600K"]:
        vals = [float(by_key[(p, model)]["mean_eV"]) for p in source_paths]
        sds = [float(by_key[(p, model)]["sample_sd_eV"]) for p in source_paths]
        effects.append(vals)
        lows.append([v - sd for v, sd in zip(vals, sds)])
        highs.append([v + sd for v, sd in zip(vals, sds)])
    forest_effects(
        ax,
        paths,
        effects,
        lows,
        highs,
        ["Scratch", "FT-600K"],
        [model_colors["Scratch"], model_colors["FT-600K"]],
        xlabel="Absolute barrier error (eV)",
        legend=True,
        xlim=(0, 11.5),
    )
    ax.set_title("Gap-to-QL / into-QL paths")
    panel_label(ax, "b", x=-0.15, y=1.10)

    FIGURES.mkdir(parents=True, exist_ok=True)
    save_figure(fig, FIGURES / "a3_seed_uncertainty.pdf")
    fig.savefig(FIGURES / "a3_seed_uncertainty.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_a4_figure(cross_summary: list[dict]) -> None:
    apply_publication_style(8.0)
    palette = get_palette("jhu_mace_models")
    neutral = ["#6F7F91", "#8FA0B3", "#B0BFCE"]

    validation = read_csv(CROSS_VALIDATION)
    seed_labels = [f"Run {chr(65 + i)}" for i, _ in enumerate(validation)]
    energy = [float(row["validation_rmse_energy_meV_atom"]) for row in validation]
    force = [float(row["validation_rmse_force_meV_A"]) for row in validation]

    selected_ids = [
        "Cu_neb-gap-diffuse",
        "Cu_neb-gap2QL",
        "Li_neb-gap-diffuse",
        "Li_neb-gap2Ql",
        "Na_neb-gap-diffuse",
        "Na_neb-gap2QL",
    ]
    compact = {row["path_id"]: row for row in cross_summary}
    categories = ["Cu\n gap", "Cu\n gap--QL", "Li\n gap", "Li\n gap--QL", "Na\n gap", "Na\n gap--QL"]
    means = [compact[path]["max_relative_energy_mean_eV"] for path in selected_ids]
    sds = [compact[path]["max_relative_energy_sd_eV"] for path in selected_ids]

    fig = plt.figure(figsize=(7.6, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.82, 0.82, 1.55], wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    bar_with_points(
        ax,
        seed_labels,
        energy,
        [0.0] * len(energy),
        neutral,
        points=[[v] for v in energy],
        ylabel="Energy RMSE (meV/atom)",
    )
    ax.set_ylim(0, 14)
    ax.set_title("Validation")
    panel_label(ax, "a", x=-0.22, y=1.10)

    ax = fig.add_subplot(gs[0, 1])
    bar_with_points(
        ax,
        seed_labels,
        force,
        [0.0] * len(force),
        neutral,
        points=[[v] for v in force],
        ylabel="Force RMSE (meV/\\AA)",
    )
    ax.set_ylim(0, 245)
    ax.set_title("Validation")
    panel_label(ax, "b", x=-0.22, y=1.10)

    ax = fig.add_subplot(gs[0, 2])
    forest_effects(
        ax,
        categories,
        [means],
        [[m - s for m, s in zip(means, sds)]],
        [[m + s for m, s in zip(means, sds)]],
        ["Cr-trained broad roster"],
        [palette["roles"]["ft_600k"]],
        xlabel="Max relative energy (eV)",
        legend=False,
        xlim=(0, 0.38),
    )
    ax.axvline(0.1, color="#777777", linestyle=(0, (3, 2)), linewidth=0.8)
    ax.set_title("Cross-element diagnostic")
    panel_label(ax, "c", x=-0.16, y=1.10)

    fig.suptitle(
        "Three-checkpoint broad-roster reproducibility and cross-element stress test",
        y=1.03,
        fontsize=9,
        fontweight="bold",
    )
    FIGURES.mkdir(parents=True, exist_ok=True)
    save_figure(fig, FIGURES / "a4_cross_element_diagnostic.pdf")
    fig.savefig(FIGURES / "a4_cross_element_diagnostic.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_manifest() -> None:
    sources = [GROUPED, MATRIX, CROSS_DFT, CROSS_VALIDATION, CROSS_STABILITY]
    outputs = sorted(
        [path for path in DATA.glob("*.csv")]
        + [path for path in TABLES.glob("*.tex")]
        + [path for path in FIGURES.glob("a3_*")]
        + [path for path in FIGURES.glob("a4_*")]
    )
    try:
        atlas_commit = subprocess.run(
            ["git", "-C", str(PLOT_LIBRARY.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        atlas_commit = "unavailable"
    script_path = Path(__file__).resolve()
    manifest = {
        "created": "2026-10-18",
        "package": str(PACKAGE),
        "processing_script": {
            "path": str(script_path),
            "sha256": sha256(script_path),
        },
        "sources": [
            {"path": str(path), "sha256": sha256(path), "exists": path.exists()}
            for path in sources
        ],
        "outputs": [
            {"path": str(path), "sha256": sha256(path)}
            for path in outputs
        ],
        "environment": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "plot_atlas_root": str(PLOT_LIBRARY.parent),
            "plot_atlas_git_commit": atlas_commit,
        },
        "evidence_boundary": {
            "strict_grouped": "Primary A3 evidence; seeds 123/234/345; provisional DFT candidate references.",
            "protocol_matrix": "Supporting descriptive evidence; separate model lineages and explicit force-pathology/status flags.",
            "cross_element": "A4 diagnostic only; all ten Cu/Li/Na DFT records retain SCF warnings and fail the scientific gate.",
            "self_relaxed": "Not a DFT barrier ranking; MLIP-relaxed images are not DFT-rescored.",
        },
    }
    (PACKAGE / "a3_a4_evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    for path in [GROUPED, MATRIX, CROSS_DFT, CROSS_VALIDATION, CROSS_STABILITY]:
        if not path.exists():
            raise FileNotFoundError(path)
    DATA.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    a3_rows, a3_summary = build_a3_data()
    matrix_summary = build_matrix_data()
    cross_summary = build_a4_data()
    write_tables(a3_summary, matrix_summary, cross_summary)
    make_a3_figure(a3_summary)
    make_a4_figure(cross_summary)
    build_manifest()
    print(f"Wrote A3/A4 assets under {PACKAGE}")
    print(f"Strict grouped rows: {len(a3_rows)}; A3 summary rows: {len(a3_summary)}")
    print(f"Cross-element paths: {len(cross_summary)}")


if __name__ == "__main__":
    main()
