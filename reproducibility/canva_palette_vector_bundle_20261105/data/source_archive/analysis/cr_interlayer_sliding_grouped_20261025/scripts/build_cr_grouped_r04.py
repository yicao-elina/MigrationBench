#!/usr/bin/env python3
"""Build the Cr interlayer-sliding grouped 2×2 Plot Atlas package.

This is an additive reconstruction package.  Fixed-path Foundation/FT rows
come from the reviewed local R04 table; Scratch fixed/self rows come from the
paired Rockfish Lmax1 campaign; Foundation self rows come from the current
 three-model 2×2 package; FT self-relaxed rows come from the parallel FT
 Rockfish campaign recorded in this package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
# The external Plot Atlas library is only needed by draw_panel()/main(); the data
# loaders (normalize_rows, aggregate, reference) work without it.  Point
# PLOT_ATLAS_ROOT at a checkout of plot_library_recovered to rebuild the R04 package.
PLOT_ATLAS_ENV = "PLOT_ATLAS_ROOT"


def plot_profile_ensemble(*args, **kwargs):
    import os

    root = os.environ.get(PLOT_ATLAS_ENV)
    if not root:
        raise RuntimeError(f"set {PLOT_ATLAS_ENV} to the plot_library_recovered checkout to draw R04 panels")
    if str(Path(root) / "src") not in sys.path:
        sys.path.insert(0, str(Path(root) / "src"))
    from plot_atlas import plot_profile_ensemble as plot_atlas_profile_ensemble

    return plot_atlas_profile_ensemble(*args, **kwargs)


GROUP_ORDER = ("Foundation", "FT-600K", "FT-MultiT", "Scratch-600K", "Scratch-MultiT")
COLORS = {
    "Foundation": "#CF4520",
    "FT-600K": "#002E71",
    "FT-MultiT": "#67ACE5",
    "Scratch-600K": "#FF9F1B",
    "Scratch-MultiT": "#F4C95D",
}
METHOD_LABELS = {"fixed": "fixed-path", "self": "self-relaxed path"}
SYSTEM_LABELS = {"bulk": "Bulk", "cr_in_gap": "Cr-in-gap"}

FIXED_LOCAL = ROOT / "source_tables/fixed_profiles_R04.csv"
FOUNDATION_PROFILES = ROOT / "source_tables/foundation_2x2_IMAGE_PROFILES.tsv"
FOUNDATION_MATRIX = ROOT / "source_tables/foundation_2x2_RESULT_MATRIX.tsv"
FOUNDATION_REGISTRY = ROOT / "source_tables/foundation_2x2_MODEL_REGISTRY.tsv"
QE_REFERENCE = ROOT / "source_tables/foundation_2x2_QE_REFERENCE.tsv"
REMOTE_FIXED = ROOT / "raw/rockfish/scratch_lmax1_interlayer_sliding_20261025_r01"
REMOTE_SELF = ROOT / "raw/rockfish/scratch_lmax1_g4_self_20261025_r02"
REMOTE_FT_SELF = ROOT / "raw/rockfish/ft_interlayer_sliding_20261025_r01"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relpath(path: Path) -> str:
    return str(path.relative_to(ROOT))


def write_tsv(path: Path, rows: Iterable[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def model_group_from_family(family: str) -> str:
    return {
        "foundation": "Foundation",
        "ft_600k": "FT-600K",
        "ft_multit": "FT-MultiT",
    }[family]


def model_group_from_label(label: str) -> str:
    normalized = label.lower().replace(" ", "-")
    if "600k" in normalized:
        return "Scratch-600K"
    if "multit" in normalized or "multi-t" in normalized:
        return "Scratch-MultiT"
    raise ValueError(f"cannot infer Scratch group from label: {label!r}")


def ft_model_group_from_label(label: str) -> str:
    normalized = label.lower().replace(" ", "-")
    if "600k" in normalized:
        return "FT-600K"
    if "multit" in normalized or "multi-t" in normalized:
        return "FT-MultiT"
    raise ValueError(f"cannot infer FT group from label: {label!r}")


def foundation_status_by_record() -> dict[str, dict[str, str]]:
    return {row["record_id"]: row for row in read_tsv(FOUNDATION_MATRIX)}


def foundation_sha_by_model() -> dict[str, str]:
    return {row["model_id"]: row["sha256"] for row in read_tsv(FOUNDATION_REGISTRY) if row.get("included") == "true"}


def normalize_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    # Reviewed local fixed-path R04 rows, excluding the old Scratch
    # checkpoints because the paired Rockfish self campaign uses Lmax1 repair
    # checkpoints with different SHA256 identities.
    for source in read_csv(FIXED_LOCAL):
        family = source["model_family"]
        if family not in {"foundation", "ft_600k", "ft_multit"}:
            continue
        system = "bulk" if source["system_id"] == "pristine" else source["system_id"]
        group = model_group_from_family(family)
        rows.append(
            {
                "observable": "FIXED_PATH_SINGLE_POINT",
                "method": "fixed",
                "system": system,
                "model_group": group,
                "model_id": source["model_id"],
                "replicate": source["model_id"],
                "model_label": group,
                "image_index": int(source["image_index"]),
                "image_fraction": float(source["image_index"]) / 9.0,
                "energy_eV": float(source["energy_eV"]),
                "relative_energy_eV": float(source["relative_energy_eV"]),
                "model_sha256": source["model_sha256"],
                "scientific_status": source["status"],
                "publication_status": source["publication_status"],
                "source_kind": "local_reviewed_fixed_R04",
                "source_path": relpath(FIXED_LOCAL),
            }
        )

    # Paired Rockfish Scratch fixed rows selected from the full collect ledger,
    # which removes the duplicate smoke-attempt profile for Cr-in-gap seed123.
    fixed_collect = json.loads((REMOTE_FIXED / "collect/G4_FIXED_FULL_COLLECT.json").read_text())
    for record in fixed_collect["records"]:
        remote_path = record["profile_path"]
        relative_after_runs = remote_path.split("/runs/", 1)[1]
        profile_path = REMOTE_FIXED / "runs" / relative_after_runs
        if not profile_path.is_file():
            raise FileNotFoundError(profile_path)
        profile_rows = read_csv(profile_path)
        if len(profile_rows) != 10:
            raise ValueError(f"expected 10 fixed Scratch images: {profile_path}")
        group = model_group_from_label(profile_rows[0]["model_name"])
        system = record["system"]
        for source in profile_rows:
            rows.append(
                {
                    "observable": "FIXED_PATH_SINGLE_POINT",
                    "method": "fixed",
                    "system": system,
                    "model_group": group,
                    "model_id": record["model_id"],
                    "replicate": record["model_id"],
                    "model_label": group,
                    "image_index": int(source["image_index"]),
                    "image_fraction": float(source["image_fraction"]),
                    "energy_eV": float(source["predicted_energy_eV"]),
                    "relative_energy_eV": float(source["relative_energy_eV"]),
                    "model_sha256": record["model_sha256"],
                    "scientific_status": record["scientific_status"],
                    "publication_status": record["publication_status"],
                    "source_kind": "rockfish_fixed_lmax1_candidate",
                    "source_path": relpath(profile_path),
                }
            )

    # Exact three-model Foundation self-relaxed rows from the current 2×2 R13
    # package.  The result matrix supplies the status boundary.
    status_by_record = foundation_status_by_record()
    sha_by_model = foundation_sha_by_model()
    for source in read_tsv(FOUNDATION_PROFILES):
        if source["observable"] != "SELF_RELAXED_ENDPOINT_ONLY_NEB":
            continue
        status = status_by_record[source["record_id"]]
        rows.append(
            {
                "observable": source["observable"],
                "method": "self",
                "system": source["system"],
                "model_group": "Foundation",
                "model_id": source["model_id"],
                "replicate": source["model_id"],
                "model_label": "Foundation",
                "image_index": int(source["image_index"]),
                "image_fraction": float(source["image_fraction"]),
                "energy_eV": float(source["energy_eV"]),
                "relative_energy_eV": float(source["relative_energy_eV"]),
                "model_sha256": sha_by_model[source["model_id"]],
                "scientific_status": status["scientific_status"],
                "publication_status": status["publication_status"],
                "source_kind": "foundation_2x2_R13_self",
                "source_path": relpath(FOUNDATION_PROFILES),
            }
        )

    # Rockfish Scratch endpoint-only self-relaxed profiles.  Each manifest is
    # one model/system replicate and carries the 10-image profile directly.
    self_manifests = sorted(REMOTE_SELF.glob("runs/self/*/*/*/endpoint_only_manifest.json"))
    if len(self_manifests) != 12:
        raise ValueError(f"expected 12 Scratch self manifests, got {len(self_manifests)}")
    for manifest_path in self_manifests:
        manifest = json.loads(manifest_path.read_text())
        profile = manifest["engine"]["profile"]
        if len(profile) != 10:
            raise ValueError(f"expected 10 self Scratch images: {manifest_path}")
        group = model_group_from_label(manifest["model_label"])
        for source in profile:
            image_index = int(source["image_index"])
            rows.append(
                {
                    "observable": manifest["observable"],
                    "method": "self",
                    "system": manifest["system"],
                    "model_group": group,
                    "model_id": manifest["model_id"],
                    "replicate": manifest["model_id"],
                    "model_label": group,
                    "image_index": image_index,
                    "image_fraction": image_index / 9.0,
                    "energy_eV": float(source["energy_eV"]),
                    "relative_energy_eV": float(source["rel_energy_eV"]),
                    "model_sha256": manifest["model_sha256_registry"],
                    "scientific_status": manifest["scientific_status"],
                    "publication_status": manifest["publication_status"],
                    "source_kind": "rockfish_self_lmax1_R02",
                    "source_path": relpath(manifest_path),
                }
            )

    # Parallel Rockfish FT-600K / FT-MultiT endpoint-only self-relaxed rows.
    # The campaign collector has already passed all 12 rows; the per-row
    # manifests remain the source of the plotted profiles and model hashes.
    ft_manifests = sorted(REMOTE_FT_SELF.glob("runs/self/*/*/*/endpoint_only_manifest.json"))
    if len(ft_manifests) != 12:
        raise ValueError(f"expected 12 FT self manifests, got {len(ft_manifests)}")
    for manifest_path in ft_manifests:
        manifest = json.loads(manifest_path.read_text())
        profile = manifest["engine"]["profile"]
        if len(profile) != 10:
            raise ValueError(f"expected 10 self FT images: {manifest_path}")
        group = ft_model_group_from_label(manifest["model_label"])
        for source in profile:
            image_index = int(source["image_index"])
            rows.append(
                {
                    "observable": manifest["observable"],
                    "method": "self",
                    "system": manifest["system"],
                    "model_group": group,
                    "model_id": manifest["model_id"],
                    "replicate": manifest["model_id"],
                    "model_label": group,
                    "image_index": image_index,
                    "image_fraction": image_index / 9.0,
                    "energy_eV": float(source["energy_eV"]),
                    "relative_energy_eV": float(source["rel_energy_eV"]),
                    "model_sha256": manifest["model_sha256_registry"],
                    "scientific_status": manifest["scientific_status"],
                    "publication_status": manifest["publication_status"],
                    "source_kind": "rockfish_ft_self_R01",
                    "source_path": relpath(manifest_path),
                }
            )

    keys = [(row["observable"], row["system"], row["model_id"], row["image_index"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate normalized profile rows detected")
    rows.sort(key=lambda row: (row["method"], row["system"], GROUP_ORDER.index(row["model_group"]), row["model_id"], row["image_index"]))
    return rows


def write_audits(rows: list[dict[str, object]]) -> None:
    table_dir = ROOT / "results/tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    counts: list[dict[str, object]] = []
    for method in ("fixed", "self"):
        for system in ("bulk", "cr_in_gap"):
            for group in GROUP_ORDER:
                # The normalized method field is the stable coverage key.  The
                # Rockfish self rows intentionally retain their more specific
                # ENDPOINT_ONLY_MLIP_NEB_PROXY observable label.
                selected = [r for r in rows if r["method"] == method and r["system"] == system and r["model_group"] == group]
                replicates = sorted({str(r["replicate"]) for r in selected})
                counts.append(
                    {
                        "method": method,
                        "system": system,
                        "model_group": group,
                        "n_replicates": len(replicates),
                        "n_profile_rows": len(selected),
                        "scientific_statuses": ";".join(sorted({str(r["scientific_status"]) for r in selected})) or "MISSING",
                        "publication_statuses": ";".join(sorted({str(r["publication_status"]) for r in selected})) or "MISSING",
                        "plot_included": bool(selected),
                        "notes": "" if selected else ("FT self-relaxed interlayer source not found" if method == "self" and group.startswith("FT-") else "missing source rows"),
                    }
                )
    write_tsv(
        table_dir / "COVERAGE_AUDIT.tsv",
        counts,
        ["method", "system", "model_group", "n_replicates", "n_profile_rows", "scientific_statuses", "publication_statuses", "plot_included", "notes"],
    )

    existing: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_csv(FIXED_LOCAL):
        family = row["model_family"]
        if family != "scratch":
            continue
        group = "Scratch-600K" if row["scratch_subfamily"] == "scratch_600k" else "Scratch-MultiT"
        existing[(group, row["training_seed"])] = row

    paired: dict[tuple[str, str], dict[str, str]] = {}
    for record in fixed_collect_records():
        group = model_group_from_label(record["model_label"])
        seed = record["model_id"].rsplit("_seed", 1)[-1]
        paired[(group, seed)] = record
    mismatch_rows = []
    for key in sorted(set(existing) | set(paired)):
        old = existing.get(key, {})
        new = paired.get(key, {})
        mismatch_rows.append(
            {
                "model_group": key[0],
                "seed": key[1],
                "existing_fixed_R04_model_id": old.get("model_id", "MISSING"),
                "existing_fixed_R04_sha256": old.get("model_sha256", "MISSING"),
                "paired_Rockfish_model_id": new.get("model_id", "MISSING"),
                "paired_Rockfish_sha256": new.get("model_sha256", "MISSING"),
                "sha256_match": old.get("model_sha256") == new.get("model_sha256") if old and new else False,
                "plot_decision": "use paired Rockfish rows" if old.get("model_sha256") != new.get("model_sha256") else "same identity",
            }
        )
    write_tsv(
        table_dir / "SCRATCH_CHECKPOINT_AUDIT.tsv",
        mismatch_rows,
        ["model_group", "seed", "existing_fixed_R04_model_id", "existing_fixed_R04_sha256", "paired_Rockfish_model_id", "paired_Rockfish_sha256", "sha256_match", "plot_decision"],
    )


def fixed_collect_records() -> list[dict[str, str]]:
    collect = json.loads((REMOTE_FIXED / "collect/G4_FIXED_FULL_COLLECT.json").read_text())
    records = []
    for record in collect["records"]:
        first_profile = REMOTE_FIXED / "runs" / record["profile_path"].split("/runs/", 1)[1]
        first_profile_row = read_csv(first_profile)[0]
        records.append({"model_id": record["model_id"], "model_sha256": record["model_sha256"], "model_label": first_profile_row["model_name"]})
    return records


def aggregate(rows: list[dict[str, object]], method: str, system: str) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    selected = [row for row in rows if row["method"] == method and row["system"] == system]
    by_group: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in selected:
        by_group[str(row["model_group"])][str(row["replicate"])][int(row["image_index"])] = float(row["relative_energy_eV"])
    if not by_group:
        raise ValueError(f"no normalized rows for {method}/{system}")
    result: dict[str, dict[str, np.ndarray]] = {}
    x = np.linspace(0.0, 1.0, 10)
    for group in GROUP_ORDER:
        if group not in by_group:
            continue
        replicate_arrays = []
        for replicate, values in sorted(by_group[group].items()):
            if sorted(values) != list(range(10)):
                raise ValueError(f"incomplete profile for {method}/{system}/{group}/{replicate}")
            replicate_arrays.append([values[index] for index in range(10)])
        array = np.asarray(replicate_arrays, dtype=float)
        result[group] = {
            "mean": array.mean(axis=0),
            "sd": array.std(axis=0, ddof=1) if len(array) > 1 else np.zeros(10),
        }
    return x, result


def reference(system: str) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted((row for row in read_tsv(QE_REFERENCE) if row["system"] == system), key=lambda row: int(row["image_index"]))
    return (
        np.asarray([float(row["image_fraction"]) for row in rows]),
        np.asarray([float(row["relative_energy_eV"]) for row in rows]),
    )


def y_limits(profiles: dict[str, dict[str, np.ndarray]], reference_y: np.ndarray) -> tuple[float, float]:
    values = [reference_y]
    for profile in profiles.values():
        values.extend((profile["mean"] - profile["sd"], profile["mean"] + profile["sd"]))
    low = float(np.min(np.concatenate(values)))
    high = float(np.max(np.concatenate(values)))
    return math.floor(low * 2.0) / 2.0, math.ceil(high * 2.0) / 2.0


def plot_style() -> dict[str, object]:
    return {
        "font_family": "Arial",
        "font_size": 16.0,
        "title_size": 18.0,
        "x_label_size": 16.8,
        "y_label_size": 17.0,
        "axes_linewidth": 1.5,
        "tick_major_length": 5.0,
        "tick_major_width": 1.4,
        "tick_minor_length": 3.0,
        "tick_minor_width": 1.1,
        "tick_pad": 3.0,
        "line_width": 2.0,
        "marker_size": 4.3,
        "marker_edge_width": 0.95,
        "band_alpha": 0.20,
        "reference_color": "#222222",
        "reference_line_width": 2.2,
        "reference_dash": (0, (4, 2)),
        "reference_marker_size": 3.9,
        "reference_marker_edge_width": 1.2,
        "tick_labels_bold": True,
        "panel_label_x": -0.16,
        "panel_label_y": 1.06,
    }


def palette() -> dict[str, object]:
    return {"categorical": [COLORS[group] for group in GROUP_ORDER], "roles": COLORS, "reference": "#222222"}


def draw_panel(ax, rows: list[dict[str, object]], method: str, system: str, panel_label: str) -> dict[str, object]:
    x, profiles = aggregate(rows, method, system)
    reference_x, reference_y = reference(system)
    output = plot_profile_ensemble(
        ax,
        profiles,
        x=x,
        series_order=list(profiles),
        palette=palette(),
        style=plot_style(),
        reference={
            "x": reference_x,
            "y": reference_y,
            "color": "#222222",
            "linewidth": 2.2,
            "linestyle": (0, (4, 2)),
            "marker": "D",
            "markersize": 3.9,
            "markerfacecolor": "white",
            "markeredgecolor": "#222222",
            "markeredgewidth": 1.2,
            "label": "DFT-NEB",
        },
        title=f"{SYSTEM_LABELS[system]} · {METHOD_LABELS[method]}",
        xlabel="Reaction Coordinate",
        ylabel=r"$\Delta E$ (eV)",
        panel_label=panel_label,
    )
    ax.xaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_fontfamily("Arial")
    ax.set_ylim(*y_limits(profiles, reference_y))
    return {"plot": output, "profiles": profiles, "x": x, "reference_y": reference_y}


def legend_handles(groups: list[str]) -> list[Line2D]:
    handles = [
        Line2D([], [], color=COLORS[group], lw=2.0, marker="o", markersize=4.3, markerfacecolor="white", markeredgecolor=COLORS[group], markeredgewidth=0.95, label=group)
        for group in groups
    ]
    handles.append(Line2D([], [], color="#222222", lw=2.2, ls=(0, (4, 2)), marker="D", markersize=3.9, markerfacecolor="white", markeredgecolor="#222222", markeredgewidth=1.2, label="DFT-NEB"))
    return handles


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [stem.with_suffix(ext) for ext in (".png", ".svg", ".pdf")]
    for path in paths:
        fig.savefig(path, dpi=400 if path.suffix == ".png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def render(rows: list[dict[str, object]]) -> tuple[list[Path], dict[str, object]]:
    out = ROOT / "results/figures/plot_atlas_grouped_cr_r04"
    out.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.0), sharex=False, sharey=False)
    first = draw_panel(axes[0, 0], rows, "fixed", "bulk", "a")
    draw_panel(axes[0, 1], rows, "fixed", "cr_in_gap", "b")
    draw_panel(axes[1, 0], rows, "self", "bulk", "c")
    draw_panel(axes[1, 1], rows, "self", "cr_in_gap", "d")
    handles = legend_handles(GROUP_ORDER)
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=6, frameon=False, fontsize=14.0, handlelength=2.0, columnspacing=0.95, handletextpad=0.35)
    for text in legend.get_texts():
        text.set_fontfamily("Arial")
        text.set_fontweight("bold")
    fig.subplots_adjust(left=0.095, right=0.985, top=0.86, bottom=0.085, wspace=0.34, hspace=0.48)
    outputs.extend(save_figure(fig, out / "cr_interlayer_sliding_grouped_2x2_plot_atlas_r04"))

    panel_specs = (
        ("fixed", "bulk", "a", "fixed_bulk_grouped_plot_atlas_r04"),
        ("fixed", "cr_in_gap", "b", "fixed_cr_in_gap_grouped_plot_atlas_r04"),
        ("self", "bulk", "c", "self_bulk_grouped_plot_atlas_r04"),
        ("self", "cr_in_gap", "d", "self_cr_in_gap_grouped_plot_atlas_r04"),
    )
    for method, system, label, name in panel_specs:
        fig, ax = plt.subplots(figsize=(7.2, 5.6))
        panel = draw_panel(ax, rows, method, system, label)
        visible_groups = list(panel["profiles"])
        legend = ax.legend(handles=legend_handles(visible_groups), frameon=False, fontsize=17.0, ncol=2, handlelength=2.2, columnspacing=1.25, handletextpad=0.45, loc="upper center", bbox_to_anchor=(0.5, -0.23))
        for text in legend.get_texts():
            text.set_fontfamily("Arial")
            text.set_fontweight("bold")
        fig.subplots_adjust(left=0.15, right=0.98, top=0.90, bottom=0.31)
        outputs.extend(save_figure(fig, out / name))

    return outputs, {"out_dir": str(out), "main_figure": str(out / "cr_interlayer_sliding_grouped_2x2_plot_atlas_r04.png")}


def source_records() -> list[dict[str, object]]:
    paths = [
        FIXED_LOCAL,
        ROOT / "source_tables/FIG168_MODEL_STYLE_YELLOW_SCRATCH_R04.json",
        ROOT / "source_tables/foundation_2x2_IMAGE_PROFILES.tsv",
        ROOT / "source_tables/foundation_2x2_RESULT_MATRIX.tsv",
        ROOT / "source_tables/foundation_2x2_MODEL_REGISTRY.tsv",
        ROOT / "source_tables/foundation_2x2_QE_REFERENCE.tsv",
        REMOTE_FIXED / "collect/G4_FIXED_FULL_COLLECT.json",
        REMOTE_FIXED / "collect/G4_FIXED_FULL_HASH_LEDGER.tsv",
        REMOTE_FIXED / "MODEL_REGISTRY.tsv",
        REMOTE_FIXED / "PATH_REGISTRY.tsv",
        REMOTE_FIXED / "ACCEPTANCE_FREEZE_R01.json",
        REMOTE_SELF / "G4_SELF_R02_COLLECT.json",
        REMOTE_SELF / "G4_SELF_R02_COLLECT.tsv",
        REMOTE_SELF / "G4_SELF_R02_FAILURE_LEDGER.tsv",
        REMOTE_SELF / "MODEL_REGISTRY.tsv",
        REMOTE_SELF / "jobs/JOBS_SELF.tsv",
        REMOTE_SELF / "ACCEPTANCE_FREEZE_R02.json",
        REMOTE_FT_SELF / "MODEL_REGISTRY.tsv",
        REMOTE_FT_SELF / "jobs/JOBS_SELF.tsv",
        REMOTE_FT_SELF / "ACCEPTANCE_FREEZE_FT_SELF_R01.json",
        REMOTE_FT_SELF / "FT_SELF_R01_COLLECT.json",
        REMOTE_FT_SELF / "FT_SELF_R01_COLLECT.tsv",
        REMOTE_FT_SELF / "FT_SELF_R01_FAILURE_LEDGER.tsv",
    ]
    paths.extend(sorted(REMOTE_FIXED.glob("runs/fixed_smoke/*/*/*/fixed_profile.csv")))
    paths.extend(sorted(REMOTE_FIXED.glob("runs/fixed_smoke/*/*/*/fixed_path_manifest.json")))
    paths.extend(sorted(REMOTE_SELF.glob("runs/self/*/*/*/endpoint_only_manifest.json")))
    paths.extend(sorted(REMOTE_FT_SELF.glob("runs/self/*/*/*/endpoint_only_manifest.json")))
    paths.extend(sorted(REMOTE_FT_SELF.glob("runs/self/*/*/*/endpoint_only_command.json")))
    paths.extend(sorted(REMOTE_FT_SELF.glob("runs/self/*/*/*/runtime_provenance.json")))
    paths.extend(sorted(REMOTE_FT_SELF.glob("runs/self/*/*/*/mlff_neb_manifest.json")))
    paths.extend(sorted(REMOTE_FT_SELF.glob("runs/self/*/*/*/mlff_neb_iteration_history.csv")))
    paths.extend(sorted(REMOTE_FT_SELF.glob("runs/self/*/*/*/mlff_neb_iteration_history_manifest.json")))
    paths.extend(sorted(REMOTE_FT_SELF.glob("runs/self/*/*/*/mlff_neb_dual_potential_history_manifest.json")))
    seen: set[Path] = set()
    records = []
    for path in paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        records.append({"path": relpath(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return records


def build_manifest(rows: list[dict[str, object]], outputs: list[Path], render_meta: dict[str, object]) -> dict[str, object]:
    coverage = read_tsv(ROOT / "results/tables/COVERAGE_AUDIT.tsv")
    return {
        "schema_version": "cr_interlayer_grouped_profile_plot_atlas_manifest_r04_v1",
        "campaign_id": "cr_interlayer_sliding_grouped_20261025",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_SCIENTIFIC_BOUNDARIES",
        "render_status": "RECONSTRUCTION",
        "layout": "2 systems × 2 observables; grouped model classes as series",
        "plot_atlas_panel": "profile_ensemble",
        "group_order": list(GROUP_ORDER),
        "group_colors": COLORS,
        "uncertainty": "pointwise sample SD across model identities within each class; ddof=1",
        "normalized_profile_table": {"path": relpath(ROOT / "results/tables/NORMALIZED_PROFILES.tsv"), "sha256": sha256_file(ROOT / "results/tables/NORMALIZED_PROFILES.tsv"), "rows": len(rows)},
        "coverage_audit": {"path": relpath(ROOT / "results/tables/COVERAGE_AUDIT.tsv"), "sha256": sha256_file(ROOT / "results/tables/COVERAGE_AUDIT.tsv"), "rows": len(coverage)},
        "checkpoint_audit": {"path": relpath(ROOT / "results/tables/SCRATCH_CHECKPOINT_AUDIT.tsv"), "sha256": sha256_file(ROOT / "results/tables/SCRATCH_CHECKPOINT_AUDIT.tsv")},
        "source_files": source_records(),
        "outputs": [{"path": relpath(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in outputs],
        "scientific_boundaries": {
            "fixed_path": "FINITE_DIAGNOSTIC; fixed-path single-point scoring",
            "self_relaxed": "CONVERGED endpoint-only MLIP NEB proxy; not a DFT MEP barrier",
            "dft_reference": "COMPLETED_BUT_SCF_WARNING; retained as screened reference context",
            "ft_self_relaxed": "CONVERGED endpoint-only MLIP NEB proxy; Rockfish collector PASS 12/12",
            "scratch_pairing": "Rockfish Lmax1 fixed/self rows are internally checkpoint-matched but differ from the Scratch rows in the existing fixed R04 figure",
        },
        "publication_note": "Complete model-family coverage is present. Scratch fixed/self profiles use an internally paired Rockfish Lmax1 checkpoint set whose hashes differ from the old fixed R04 Scratch rows; see SCRATCH_CHECKPOINT_AUDIT.tsv before publication.",
        "render_metadata": render_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.root.resolve() != ROOT.resolve():
        raise SystemExit("this reconstruction script is bound to its package root")

    rows = normalize_rows()
    fields = [
        "observable", "method", "system", "model_group", "model_id", "replicate", "model_label",
        "image_index", "image_fraction", "energy_eV", "relative_energy_eV", "model_sha256",
        "scientific_status", "publication_status", "source_kind", "source_path",
    ]
    write_tsv(ROOT / "results/tables/NORMALIZED_PROFILES.tsv", rows, fields)
    write_audits(rows)
    outputs, render_meta = render(rows)
    manifest = build_manifest(rows, outputs, render_meta)
    manifest_path = ROOT / "manifests/FIGURE_PROVENANCE_CR_INTERLAYER_GROUPED_R04.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": relpath(manifest_path), "outputs": [relpath(path) for path in outputs], "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
