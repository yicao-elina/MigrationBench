#!/usr/bin/env python3
"""Build publication-style three-column NEB profile figures for MigrationBench.

This release is intentionally additive.  It keeps the existing R03 figure and
normalizes the available Cr/Cu/Li/Na evidence into a common visual grammar:
independent panel axes, separate y scales, a shared x-label, and a compact
status legend.  Fixed-path single-point profiles and self-relaxed MLFF-NEB
profiles are rendered as separate observables and are never pooled.

The script also writes normalized data, source hashes, captions, a delivery
index, and a QA report next to the figures.  It is designed to fail closed:
missing source rows or non-finite plotted values raise an error instead of
creating a visually complete but scientifically incomplete figure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "analysis" / "neb_cross_element_triptychs_20261024"
DATA_OUT = OUT / "data"
FIG_OUT = OUT / "figures"

R03_DIR = REPO / "analysis" / "neb_fixed_path_trajectory_figure_20261023" / "python"
R03_FIXED = R03_DIR / "data" / "fixed_path_long.tsv"
R03_DFT = R03_DIR / "data" / "dft_profiles.tsv"

CR_FOUR_MODEL = (
    REPO
    / "analysis"
    / "task3_endpoint_only_neb"
    / "cr_neb_sota_20261015"
    / "derived_model_profiles_20261016"
    / "results"
    / "cr_four_model_profiles_long.csv"
)
CR_FOUR_SUMMARY = CR_FOUR_MODEL.with_name("cr_four_model_barrier_summary.csv")
CR_TIER1 = (
    REPO
    / "analysis"
    / "task3_endpoint_only_neb"
    / "neb_fixed_vs_self_matrix_20261019"
    / "derived_r01"
    / "TIER1_IMAGE_PROFILES.tsv"
)

CROSS_DIR = REPO / "analysis" / "task2_cross_element_broad_roster_final_20261006"
CROSS_MLIP = CROSS_DIR / "results" / "validation" / "diagnostic_profiles.csv"
CROSS_QE = CROSS_DIR / "results" / "broad_roster" / "qe_neb_path_profiles.csv"
CROSS_DFT_MANIFEST = CROSS_DIR / "manifests" / "dft_neb_provenance_manifest.json"

STYLE_JSON = (
    REPO
    / "analysis"
    / "task3_endpoint_only_neb"
    / "figure1_family_seed_audit_20261016"
    / "FIG168_MODEL_STYLE.json"
)
DATA_ASSETS = Path(
    "/Users/alina/Library/CloudStorage/GoogleDrive-yicao.alina@gmail.com/My Drive/Obsidian-google/Projects/MigrationBench/MigrationBench-Data-Assets.md"
)
REFERENCE_PDF = Path(
    "/Users/alina/Library/CloudStorage/GoogleDrive-yicao.alina@gmail.com/My Drive/Obsidian-google/Projects/MigrationBench/NEB-diffusion-pathway.pdf"
)
PASTED_CAMPAIGN_TABLE = Path(
    "/Users/alina/.codex/attachments/26375b7f-802f-4177-9f62-639800729597/Pasted text.txt"
)

PATH_ORDER = ["within_gap", "gap_to_ql", "within_ql"]
PATH_LABEL = {
    "within_gap": "within gap",
    "gap_to_ql": "gap→QL",
    "within_ql": "within QL",
    "extra_path": "excluded extra path",
}
PATH_FILE_LABEL = {
    "within_gap": "within_gap",
    "gap_to_ql": "gap2ql",
    "within_ql": "within_ql",
}

FALLBACK_FAMILY_COLORS = {
    "foundation_omat": "#CF4520",
    "from_scratch": "#FF9F1B",
    "ft_600k": "#002E71",
    "ft_multit": "#67ACE5",
}

rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
        "axes.linewidth": 0.95,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def finite_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required source file(s) missing:\n" + "\n".join(missing))


def load_style() -> dict[str, Any]:
    if STYLE_JSON.is_file():
        payload = json.loads(STYLE_JSON.read_text())
        colors = {
            key: value.get("color", FALLBACK_FAMILY_COLORS.get(key, "#333333"))
            for key, value in payload.get("families", {}).items()
        }
        if colors:
            return {"payload": payload, "colors": {**FALLBACK_FAMILY_COLORS, **colors}}
    return {"payload": {}, "colors": FALLBACK_FAMILY_COLORS.copy()}


STYLE = load_style()
FAMILY_COLORS = STYLE["colors"]


def path_key_from_cr_path_id(path_id: str) -> str:
    if path_id == "cr_canonical_in_gap":
        return "within_gap"
    if path_id == "cr_canonical_gap_to_ql":
        return "gap_to_ql"
    if path_id == "cr_canonical_within_ql":
        return "within_ql"
    raise ValueError(f"Unknown Cr path id: {path_id}")


def path_key_from_cr_class(path_class: str) -> str:
    if path_class == "in-gap diffusion":
        return "within_gap"
    if path_class == "gap → QL":
        return "gap_to_ql"
    if path_class == "within-QL":
        return "within_ql"
    raise ValueError(f"Unknown Cr path class: {path_class}")


def path_key_from_cross_path(path_id: str) -> str | None:
    lower = path_id.lower()
    if "gap-diffuse" in lower:
        return "within_gap"
    if "gap2ql" in lower:
        return "gap_to_ql"
    if lower.endswith("-qls"):
        return "within_ql"
    return None


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalized_row(
    *,
    element: str,
    observable: str,
    kind: str,
    path_key: str,
    series_key: str,
    series_label: str,
    family: str,
    seed: str,
    image_index: int | None,
    s: float | None,
    energy_eV: float | None,
    status: str,
    source_path: Path,
    source_record: str = "",
    variant: str = "",
    excluded_reason: str = "",
) -> dict[str, Any]:
    return {
        "element": element,
        "observable": observable,
        "kind": kind,
        "path_key": path_key,
        "path_label": PATH_LABEL[path_key],
        "series_key": series_key,
        "series_label": series_label,
        "family": family,
        "training_seed": seed,
        "image_index": "" if image_index is None else image_index,
        "s": "" if s is None else f"{s:.12g}",
        "relative_energy_eV": "" if energy_eV is None else f"{energy_eV:.12g}",
        "status": status,
        "variant": variant,
        "source_path": str(source_path),
        "source_record": source_record,
        "excluded_reason": excluded_reason,
    }


NORMALIZED_FIELDS = [
    "element",
    "observable",
    "kind",
    "path_key",
    "path_label",
    "series_key",
    "series_label",
    "family",
    "training_seed",
    "image_index",
    "s",
    "relative_energy_eV",
    "status",
    "variant",
    "source_path",
    "source_record",
    "excluded_reason",
]


def load_cr_fixed() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_tsv(R03_FIXED)
    refs = read_tsv(R03_DFT)

    target_ids = [
        "foundation_omat0_medium",
        "foundation_mpa0",
        "foundation_mp0b3",
        "from_scratch_seed123",
        "from_scratch_seed234",
        "from_scratch_seed345",
        "ft600k_seed123",
        "ft600k_seed234",
        "ft600k_seed345",
        "ftmultit_seed123",
        "ftmultit_seed234",
        "ftmultit_seed345",
    ]
    representative_ids = {
        "foundation_omat0_medium": {
            "label": "Foundation (OMAT-0-M)",
            "family": "foundation_omat",
            "seed": "checkpoint variant",
        },
        "from_scratch_seed123": {
            "label": "From-scratch (seed 123)",
            "family": "from_scratch",
            "seed": "123",
        },
        "ft600k_seed123": {
            "label": "FT-600K (seed 123)",
            "family": "ft_600k",
            "seed": "123",
        },
        "ftmultit_seed234": {
            "label": "FT-MultiT (seed 234)",
            "family": "ft_multit",
            "seed": "234",
        },
    }

    all_rows: list[dict[str, Any]] = []
    rep_rows: list[dict[str, Any]] = []
    for row in rows:
        path_key = path_key_from_cr_path_id(row["path_id"])
        model_id = row["model_id"]
        if model_id not in target_ids:
            continue
        energy = finite_float(row.get("relative_energy_eV"))
        image = int(row["image_index"]) if row.get("image_index") else None
        s = finite_float(row.get("s"))
        status = row.get("status", "") or ""
        if energy is None:
            status = status or row.get("failure_class", "FAILURE")
        if model_id.startswith("foundation_"):
            family = "foundation_omat"
            seed = "checkpoint variant"
            label = row.get("model_label", model_id)
        elif model_id.startswith("from_scratch"):
            family = "from_scratch"
            seed = row.get("training_seed", "")
            label = f"From-scratch (seed {seed})"
        elif model_id.startswith("ft600k"):
            family = "ft_600k"
            seed = row.get("training_seed", "")
            label = f"FT-600K (seed {seed})"
        else:
            family = "ft_multit"
            seed = row.get("training_seed", "")
            label = f"FT-MultiT (seed {seed})"
        rec = normalized_row(
            element="Cr",
            observable="FIXED_PATH_SINGLE_POINT",
            kind="model",
            path_key=path_key,
            series_key=model_id,
            series_label=label,
            family=family,
            seed=seed,
            image_index=image,
            s=s,
            energy_eV=energy,
            status=status,
            source_path=R03_FIXED,
            source_record=row.get("record_id", ""),
        )
        all_rows.append(rec)
        if model_id in representative_ids:
            rep = representative_ids[model_id]
            rec_rep = dict(rec)
            rec_rep["series_label"] = rep["label"]
            rec_rep["family"] = rep["family"]
            rec_rep["training_seed"] = rep["seed"]
            rep_rows.append(rec_rep)

    ref_rows: list[dict[str, Any]] = []
    for row in refs:
        if row.get("record_type") != "DFT_REFERENCE":
            continue
        path_key = path_key_from_cr_path_id(row["path_id"])
        energy = finite_float(row.get("relative_energy_eV"))
        image = int(row["image_index"])
        s = finite_float(row.get("s"))
        ref_rows.append(
            normalized_row(
                element="Cr",
                observable="QE_REFERENCE",
                kind="reference",
                path_key=path_key,
                series_key="qe_reference",
                series_label="QE reference",
                family="ground_truth",
                seed="",
                image_index=image,
                s=s,
                energy_eV=energy,
                status=row.get("dft_reference_status", "COMPLETED_BUT_SCF_WARNING"),
                source_path=R03_DFT,
                source_record=row.get("raw_table_index", ""),
            )
        )
    if len(rep_rows) != 120 or len(ref_rows) != 30:
        raise ValueError(f"Unexpected Cr fixed-path representative rows: {len(rep_rows)} / {len(ref_rows)}")
    return rep_rows, all_rows, ref_rows


def load_cr_four_model_neb() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_csv(CR_FOUR_MODEL)
    model_labels = {
        "foundation_omat": "Foundation (OMAT-FT)",
        "from_scratch": "From-scratch (full AIMD)",
        "ft_600k": "FT-600K",
        "ft_multit": "FT-MultiT",
    }
    model_families = set(model_labels)
    model_rows: list[dict[str, Any]] = []
    ref_rows: list[dict[str, Any]] = []
    for row in rows:
        path_key = path_key_from_cr_class(row["path_class"])
        image = int(row["image_index"])
        s = finite_float(row.get("reaction_coordinate"))
        energy = finite_float(row.get("energy_relative_eV"))
        family = row["model_family"]
        if family in model_families:
            model_rows.append(
                normalized_row(
                    element="Cr",
                    observable="SELF_RELAXED_MLFF_NEB",
                    kind="model",
                    path_key=path_key,
                    series_key=family,
                    series_label=model_labels[family],
                    family=family,
                    seed=row.get("training_seed", ""),
                    image_index=image,
                    s=s,
                    energy_eV=energy,
                    status=row.get("scientific_status", ""),
                    source_path=CR_FOUR_MODEL,
                    source_record=row.get("source_artifact", ""),
                )
            )
        elif family == "ground_truth":
            ref_rows.append(
                normalized_row(
                    element="Cr",
                    observable="QE_REFERENCE",
                    kind="reference",
                    path_key=path_key,
                    series_key="qe_reference",
                    series_label="QE reference",
                    family="ground_truth",
                    seed="",
                    image_index=image,
                    s=s,
                    energy_eV=energy,
                    status=row.get("scientific_status", "COMPLETED_BUT_SCF_WARNING"),
                    source_path=CR_FOUR_MODEL,
                    source_record=row.get("source_artifact", ""),
                )
            )
    if len(model_rows) != 120 or len(ref_rows) != 30:
        raise ValueError(f"Unexpected Cr four-model NEB rows: {len(model_rows)} / {len(ref_rows)}")
    return model_rows, ref_rows


def load_cross_element(element: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    mlip_rows = read_csv(CROSS_MLIP)
    qe_rows = read_csv(CROSS_QE)
    selected_paths: dict[str, str] = {}
    model_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []

    model_labels = {
        "foundation": "Foundation",
        "broad_roster_ft_seed123": "Broad-roster FT (seed 123)",
        "broad_roster_ft_seed456": "Broad-roster FT (seed 456)",
        "broad_roster_ft_seed789": "Broad-roster FT (seed 789)",
    }
    for row in mlip_rows:
        if row.get("element") != element:
            continue
        path_key = path_key_from_cross_path(row.get("path_id", ""))
        if path_key is None or row.get("variant") != "base":
            excluded_key = path_key or "extra_path"
            if path_key is None:
                excluded_reason = "source contains an extra path not included in the shared three-column vocabulary"
            else:
                excluded_reason = "main figure uses the documented base variant; perturbation arms remain in the source campaign"
            excluded_rows.append(
                normalized_row(
                    element=element,
                    observable="FIXED_PATH_SINGLE_POINT",
                    kind="excluded",
                    path_key=excluded_key,
                    series_key=row.get("model_id", ""),
                    series_label=model_labels.get(row.get("model_id", ""), row.get("model_id", "")),
                    family="broad_roster_ft" if row.get("model_id", "") != "foundation" else "foundation_omat",
                    seed=row.get("model_id", "").rsplit("seed", 1)[-1],
                    image_index=int(row["image"]),
                    s=(int(row["image"]) - 1) / 9.0,
                    energy_eV=finite_float(row.get("relative_energy_eV")),
                    status="EXCLUDED_SENSITIVITY_VARIANT",
                    source_path=CROSS_MLIP,
                    source_record=row.get("input_sha256", ""),
                    variant=row.get("variant", ""),
                    excluded_reason=excluded_reason,
                )
            )
            continue
        selected_paths[row["path_id"]] = path_key
        model_id = row["model_id"]
        family = "foundation_omat" if model_id == "foundation" else "broad_roster_ft"
        seed = "" if model_id == "foundation" else model_id.rsplit("seed", 1)[-1]
        model_rows.append(
            normalized_row(
                element=element,
                observable="FIXED_PATH_SINGLE_POINT",
                kind="model",
                path_key=path_key,
                series_key=model_id,
                series_label=model_labels[model_id],
                family=family,
                seed=seed,
                image_index=int(row["image"]),
                s=(int(row["image"]) - 1) / 9.0,
                energy_eV=finite_float(row.get("relative_energy_eV")),
                status=("FINITE_DIAGNOSTIC" if row.get("finite") == "True" else "NONFINITE"),
                source_path=CROSS_MLIP,
                source_record=row.get("input_sha256", ""),
                variant=row.get("variant", ""),
            )
        )

    ref_rows: list[dict[str, Any]] = []
    grouped_qe: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in qe_rows:
        if row.get("element") != element:
            continue
        path_key = path_key_from_cross_path(row.get("path_id", ""))
        if path_key is None:
            continue
        grouped_qe[row["path_id"]].append(row)
    for path_id, path_rows in grouped_qe.items():
        path_rows = sorted(path_rows, key=lambda x: int(x["image"]))
        base = finite_float(path_rows[0].get("qe_neb_energy_eV"))
        for row in path_rows:
            energy = finite_float(row.get("qe_neb_energy_eV"))
            if base is None or energy is None:
                rel = None
            else:
                rel = energy - base
            ref_rows.append(
                normalized_row(
                    element=element,
                    observable="QE_REFERENCE",
                    kind="reference",
                    path_key=path_key_from_cross_path(path_id) or "within_gap",
                    series_key="qe_reference",
                    series_label="QE reference",
                    family="ground_truth",
                    seed="",
                    image_index=int(row["image"]),
                    s=(int(row["image"]) - 1) / 9.0,
                    energy_eV=rel,
                    status=row.get("dft_gate_status", "failed_scf_convergence"),
                    source_path=CROSS_QE,
                    source_record=row.get("input_sha256", ""),
                )
            )
    if len(model_rows) != 120 or len(ref_rows) != 30:
        raise ValueError(f"Unexpected {element} rows: {len(model_rows)} / {len(ref_rows)}")
    return model_rows, ref_rows, excluded_rows


def value(row: dict[str, Any], field: str) -> float | None:
    return finite_float(row.get(field))


def format_tick(x: float, _pos: int) -> str:
    if abs(x) < 1e-9:
        return "0"
    if abs(x) >= 10:
        return f"{x:.0f}"
    return f"{x:.1f}"


def path_title(path_key: str) -> str:
    return PATH_LABEL[path_key]


def model_handle(spec: dict[str, Any]) -> Line2D:
    return Line2D(
        [0],
        [0],
        color=spec["color"],
        lw=1.8,
        ls=spec.get("linestyle", "-"),
        marker=spec.get("marker", "o"),
        markersize=4.2,
        markerfacecolor="white",
        markeredgecolor=spec["color"],
        markeredgewidth=0.9,
        label=spec["label"],
    )


def status_handles(
    model_rows: list[dict[str, Any]],
    ref_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]] | None = None,
) -> list[Line2D]:
    failure_rows = failure_rows or []
    handles = [
        Line2D(
            [0],
            [0],
            color="#222222",
            lw=1.4,
            ls="--",
            marker="D",
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor="#222222",
            label="QE reference · SCF warning",
        )
    ]
    if any("NONCONVERGED" in str(r.get("status", "")) for r in model_rows):
        handles.append(
            Line2D(
                [0],
                [0],
                color="#555555",
                lw=1.4,
                ls="--",
                label="MLFF-NEB not converged",
            )
        )
    if failure_rows:
        handles.append(
            Line2D(
                [0],
                [0],
                color="#111111",
                lw=0.0,
                marker="x",
                markersize=7.0,
                markeredgewidth=1.5,
                label="MODEL_LOAD failure (full Cr matrix)",
            )
        )
    if any(r.get("kind") == "excluded" for r in model_rows):
        handles.append(
            Line2D(
                [0],
                [0],
                color="#777777",
                lw=1.0,
                ls=":",
                label="perturbation arms excluded from main panel",
            )
        )
    return handles


def make_panel(
    ax: plt.Axes,
    path_key: str,
    model_rows: list[dict[str, Any]],
    ref_rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    standalone: bool = False,
    failure_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    failure_rows = failure_rows or []
    ax.set_title(path_title(path_key), fontweight="bold", pad=8.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:g}"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FuncFormatter(format_tick))
    ax.tick_params(axis="both", which="major", length=4.0, width=0.9, pad=3.0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.95)
        spine.set_color("#111111")

    plotted_values: list[float] = []
    failures: list[str] = []
    for spec in specs:
        points = [
            r
            for r in model_rows
            if r.get("path_key") == path_key and r.get("series_key") == spec["key"]
        ]
        finite = sorted(
            [(value(r, "s"), value(r, "relative_energy_eV"), r) for r in points],
            key=lambda t: (t[0] if t[0] is not None else 999),
        )
        finite = [(x, y, r) for x, y, r in finite if x is not None and y is not None]
        if not finite:
            failures.append(spec["label"])
            continue
        xs = [x for x, _y, _r in finite]
        ys = [y for _x, y, _r in finite]
        plotted_values.extend(ys)
        nonconverged = any("NONCONVERGED" in str(r.get("status", "")) for _x, _y, r in finite)
        ls = "--" if nonconverged else spec.get("linestyle", "-")
        alpha = 0.76 if nonconverged else 0.96
        ax.plot(
            xs,
            ys,
            color=spec["color"],
            lw=1.35 if not standalone else 1.45,
            ls=ls,
            alpha=alpha,
            marker=spec.get("marker", "o"),
            markersize=3.5 if not standalone else 3.8,
            markerfacecolor="white",
            markeredgecolor=spec["color"],
            markeredgewidth=0.75,
            zorder=4,
        )

    refs = [
        r
        for r in ref_rows
        if r.get("path_key") == path_key and r.get("series_key") == "qe_reference"
    ]
    ref_points = sorted(
        [(value(r, "s"), value(r, "relative_energy_eV")) for r in refs],
        key=lambda t: (t[0] if t[0] is not None else 999),
    )
    ref_points = [(x, y) for x, y in ref_points if x is not None and y is not None]
    if ref_points:
        xs = [x for x, _y in ref_points]
        ys = [y for _x, y in ref_points]
        plotted_values.extend(ys)
        ax.plot(
            xs,
            ys,
            color="#222222",
            lw=1.35 if not standalone else 1.45,
            ls="--",
            marker="D",
            markersize=3.8 if not standalone else 4.1,
            markerfacecolor="white",
            markeredgecolor="#222222",
            markeredgewidth=0.85,
            zorder=5,
        )

    if plotted_values:
        lo = min(0.0, min(plotted_values))
        hi = max(0.0, max(plotted_values))
        span = hi - lo
        pad = max(span * 0.075, 0.04 if span < 1.0 else span * 0.02)
        ax.set_ylim(lo - pad, hi + pad)
    else:
        ax.set_ylim(-0.1, 0.1)

    failures.extend(
        sorted(
            {
                str(r.get("series_label", r.get("series_key", "MODEL_LOAD failure")))
                for r in failure_rows
                if r.get("path_key") == path_key
            }
        )
    )
    if failures:
        ax.text(
            0.965,
            0.93,
            "×",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=13,
            fontweight="bold",
            color="#111111",
            zorder=8,
        )
    if standalone:
        ax.set_xlabel("normalized image coordinate, $s$", labelpad=8)
        ax.set_ylabel("ΔE (eV)", labelpad=9)
    return {
        "path_key": path_key,
        "y_limits": [float(x) for x in ax.get_ylim()],
        "model_points": sum(
            1
            for r in model_rows
            if r.get("path_key") == path_key and value(r, "relative_energy_eV") is not None
        ),
        "reference_points": len(ref_points),
        "failure_labels": failures,
    }


def render_figure(
    *,
    figure_key: str,
    element: str,
    model_rows: list[dict[str, Any]],
    ref_rows: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    observable_title: str,
    status_line: str,
    out_dir: Path,
    standalone_panels: bool = True,
    failure_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    failure_rows = failure_rows or []
    panel_layout = {
        "left": 0.085,
        "right": 0.985,
        "bottom": 0.175,
        "top": 0.705,
        "gutter": 0.038,
    }
    panel_width = (
        panel_layout["right"]
        - panel_layout["left"]
        - 2 * panel_layout["gutter"]
    ) / 3
    fig = plt.figure(figsize=(12.0, 5.25), facecolor="white")
    axes: list[plt.Axes] = []
    panel_qa: list[dict[str, Any]] = []
    for idx, path_key in enumerate(PATH_ORDER):
        x0 = panel_layout["left"] + idx * (panel_width + panel_layout["gutter"])
        ax = fig.add_axes([x0, panel_layout["bottom"], panel_width, panel_layout["top"] - panel_layout["bottom"]])
        axes.append(ax)
        panel_qa.append(make_panel(ax, path_key, model_rows, ref_rows, specs, failure_rows=failure_rows))

    fig.text(
        panel_layout["left"],
        0.962,
        f"{element}  |  {observable_title}",
        ha="left",
        va="top",
        fontsize=14.0,
        fontweight="bold",
        color="#111111",
    )
    fig.text(
        panel_layout["left"],
        0.918,
        status_line,
        ha="left",
        va="top",
        fontsize=9.2,
        color="#4B5563",
    )

    model_handles = [model_handle(spec) for spec in specs]
    fig.legend(
        handles=model_handles,
        labels=[spec["label"] for spec in specs],
        loc="upper center",
        bbox_to_anchor=(0.54, 0.895),
        ncol=min(4, len(specs)),
        frameon=False,
        handlelength=2.1,
        columnspacing=1.15,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    status = status_handles(model_rows, ref_rows, failure_rows=failure_rows)
    fig.legend(
        handles=status,
        labels=[h.get_label() for h in status],
        loc="upper center",
        bbox_to_anchor=(0.54, 0.825),
        ncol=min(3, len(status)),
        frameon=False,
        handlelength=2.1,
        columnspacing=1.15,
        handletextpad=0.45,
        borderaxespad=0.0,
        fontsize=8.4,
    )
    fig.text(
        0.5,
        0.067,
        "normalized image coordinate, $s$",
        ha="center",
        va="center",
        fontsize=11.0,
        color="#111111",
    )
    fig.text(
        0.027,
        0.445,
        "ΔE (eV)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=11.0,
        color="#111111",
    )
    fig.text(
        0.985,
        0.027,
        "independent y scales",
        ha="right",
        va="bottom",
        fontsize=7.4,
        color="#6B7280",
    )

    outputs: dict[str, str] = {}
    for suffix, kwargs in [
        ("png", {"dpi": 300}),
        ("svg", {}),
        ("pdf", {}),
    ]:
        path = out_dir / f"{figure_key}.{suffix}"
        fig.savefig(path, **kwargs)
        outputs[suffix] = str(path)
    plt.close(fig)

    if standalone_panels:
        for path_key in PATH_ORDER:
            panel_key = f"{figure_key}_panel_{PATH_FILE_LABEL[path_key]}"
            pfig = plt.figure(figsize=(4.75, 4.80), facecolor="white")
            pax = pfig.add_axes([0.19, 0.18, 0.75, 0.43])
            pqa = make_panel(
                pax,
                path_key,
                model_rows,
                ref_rows,
                specs,
                standalone=True,
                failure_rows=failure_rows,
            )
            pfig.text(
                0.19,
                0.955,
                f"{element}  |  {path_title(path_key)}",
                ha="left",
                va="top",
                fontsize=12.5,
                fontweight="bold",
            )
            pfig.text(
                0.19,
                0.908,
                (
                    "fixed-path diagnostic · QE ref: SCF warning"
                    if "fixed-path" in observable_title
                    else "self-relaxed MLFF-NEB · QE ref: SCF warning"
                ),
                ha="left",
                va="top",
                fontsize=8.0,
                color="#4B5563",
            )
            pfig.legend(
                handles=model_handles,
                labels=[s["label"] for s in specs],
                loc="upper center",
                bbox_to_anchor=(0.56, 0.855),
                ncol=2,
                frameon=False,
                handlelength=1.7,
                columnspacing=1.0,
                handletextpad=0.4,
                fontsize=7.8,
            )
            pfig.legend(
                handles=status,
                labels=[h.get_label() for h in status],
                loc="upper center",
                bbox_to_anchor=(0.56, 0.750),
                ncol=1,
                frameon=False,
                handlelength=1.7,
                handletextpad=0.4,
                fontsize=7.1,
            )
            standalone_outputs: dict[str, str] = {}
            for suffix, kwargs in [
                ("png", {"dpi": 300}),
                ("svg", {}),
                ("pdf", {}),
            ]:
                path = out_dir / f"{panel_key}.{suffix}"
                pfig.savefig(path, **kwargs)
                standalone_outputs[suffix] = str(path)
            plt.close(pfig)
            outputs[f"panel_{PATH_FILE_LABEL[path_key]}"] = standalone_outputs

    return {
        "figure_key": figure_key,
        "element": element,
        "observable": observable_title,
        "outputs": outputs,
        "panel_qa": panel_qa,
        "panel_layout": panel_layout,
        "panel_width": panel_width,
        "panel_gutter": panel_layout["gutter"],
        "has_independent_y_scales": True,
        "has_shared_x_label": True,
        "grid_enabled": False,
        "model_series": [
            {
                "key": spec["key"],
                "label": spec["label"],
                "color": spec["color"],
                "marker": spec.get("marker", "o"),
            }
            for spec in specs
        ],
    }


def source_info(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {"path": str(path), "exists": True, "sha256": sha256(path), "bytes": path.stat().st_size}


def file_info(path: Path) -> dict[str, Any]:
    info = source_info(path)
    if info.get("exists"):
        info["relative_to_output"] = str(path.relative_to(OUT)) if OUT in path.parents else None
    return info


def validate_rows(name: str, model_rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]], expected_model: int, expected_ref: int) -> dict[str, Any]:
    finite_model = [r for r in model_rows if value(r, "relative_energy_eV") is not None]
    finite_ref = [r for r in ref_rows if value(r, "relative_energy_eV") is not None]
    if len(finite_model) != expected_model:
        raise ValueError(f"{name}: expected {expected_model} finite model rows, got {len(finite_model)}")
    if len(finite_ref) != expected_ref:
        raise ValueError(f"{name}: expected {expected_ref} finite reference rows, got {len(finite_ref)}")
    per_path = Counter(r["path_key"] for r in finite_model)
    per_ref = Counter(r["path_key"] for r in finite_ref)
    if set(per_path) != set(PATH_ORDER) or set(per_ref) != set(PATH_ORDER):
        raise ValueError(f"{name}: missing one or more canonical path classes")
    return {
        "finite_model_rows": len(finite_model),
        "finite_reference_rows": len(finite_ref),
        "model_rows_by_path": dict(per_path),
        "reference_rows_by_path": dict(per_ref),
        "model_status_counts": dict(Counter(r["status"] for r in model_rows)),
        "reference_status_counts": dict(Counter(r["status"] for r in ref_rows)),
    }


def write_readme(manifest: dict[str, Any]) -> None:
    lines = [
        "# MigrationBench cross-element NEB triptychs — R01",
        "",
        "This additive release renders Cr, Cu, Li, and Na in a common three-column visual grammar. Each panel has its own complete y axis, while the figure shares only the y-axis title and the normalized image-coordinate x label. Panel gutters, full spines, Arial/Helvetica-first typography, bold ticks, and no grid follow the supplied visual reference and the project Plot Atlas contract.",
        "",
        "## Scientific scope",
        "",
        "- `Cr_fixed_path_triptych_R04` uses four representative arms from the full 12-arm Cr fixed-path matrix (Foundation OMAT-0-M, From-scratch seed 123, FT-600K seed 123, and finite FT-MultiT seed 234). The complete 12-arm source matrix, including the FT-MultiT seed-123 MODEL_LOAD failures, is retained in the normalized data and manifest.",
        "- `Cr_four_model_neb_triptych_R01` is a separate self-relaxed MLFF-NEB observable from the canonical four-model campaign. It is not pooled with the fixed-path figure.",
        "- `Cu_fixed_path_triptych_R01`, `Li_fixed_path_triptych_R01`, and `Na_fixed_path_triptych_R01` use the final broad-roster base-variant single-point profiles: Foundation plus broad-roster FT seeds 123/456/789. The perturbation arms remain in the normalized exclusion ledger and are not averaged into the main curves.",
        "- All QE/DFT reference curves are warning-bearing diagnostic references. They are labeled `QE reference · SCF warning` and must not be described as unconditional DFT ground truth.",
        "- The x coordinate is normalized image order, `s=(image-1)/9`; no geometry arc-length coordinate is inferred when it is not present in the source table.",
        "",
        "## Deliverables",
        "",
        "See `DELIVERY_INDEX.md` for direct links to composite PNG/SVG/PDF figures and the three standalone panel files for every element.",
        "",
        "## Provenance",
        "",
        f"Generated by `{Path(__file__).name}` on 2026-10-24. The machine-readable manifest is `figure_data_manifest_20261024.json`; the gate summary is `QA_REPORT_20261024.json`.",
        "",
        "The raw R03 filenames contain the legacy stem `2L-Cu__...`, but the retained Cr geometry/model evidence identifies the migrant as Cr. This release therefore does not call the R03 system Cu.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines))


def write_index(figure_records: list[dict[str, Any]]) -> None:
    lines = ["# Delivery index", "", "All figures are generated from the same renderer; use the SVG for editing and the 300 dpi PNG for review.", ""]
    for rec in figure_records:
        lines += [f"## {rec['figure_key']}", ""]
        outputs = rec["outputs"]
        for kind in ["png", "svg", "pdf"]:
            lines.append(f"- {kind.upper()}: `{outputs[kind]}`")
        lines.append("- Standalone panels:")
        for path_key in PATH_ORDER:
            p = outputs[f"panel_{PATH_FILE_LABEL[path_key]}"]
            lines.append(f"  - {PATH_LABEL[path_key]}: `{p['png']}` / `{p['svg']}` / `{p['pdf']}`")
        lines.append("")
    (OUT / "DELIVERY_INDEX.md").write_text("\n".join(lines))


def write_captions() -> None:
    text = """# Figure captions and usage boundaries

## Cr fixed-path triptych

Relative-energy profiles evaluated on the retained ten-image Cr geometries using four representative arms from the 12-arm fixed-path matrix. Foundation (OMAT-0-M), From-scratch seed 123, FT-600K seed 123, and finite FT-MultiT seed 234 are shown as model diagnostics; the full 12-arm matrix and the FT-MultiT seed-123 MODEL_LOAD failure are retained in the provenance bundle. The black dashed-diamond trace is the retained QE reference and carries an SCF-warning diagnostic status. This panel is not a converged DFT ground-truth comparison and is not a seed-uncertainty estimate.

## Cr four-model MLFF-NEB triptych

Canonical Cr four-model self-relaxed MLFF-NEB image profiles for Foundation (OMAT-FT), From-scratch, FT-600K, and FT-MultiT, overlaid with the warning-bearing QE reference. Dashed model traces indicate an optimizer-nonconverged MLFF-NEB record whose finite profile is retained for diagnostic inspection. This observable is separate from the fixed-path single-point triptych and must not be pooled with it.

## Cu/Li/Na broad-roster triptychs

Base-variant fixed-path single-point profiles for Foundation and three broad-roster fine-tuned models (training seeds 123, 456, and 789), overlaid with the corresponding archived QE path profile. Perturbation arms are retained in the exclusion ledger but are not averaged into the main curves. These are non-equilibrium snapshot diagnostics; the QE lines are warning-bearing provisional references rather than unconditional DFT ground truth.
"""
    (OUT / "CAPTIONS.md").write_text(text)


def main() -> None:
    require_files(
        [
            R03_FIXED,
            R03_DFT,
            CR_FOUR_MODEL,
            CR_FOUR_SUMMARY,
            CR_TIER1,
            CROSS_MLIP,
            CROSS_QE,
            CROSS_DFT_MANIFEST,
        ]
    )
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    cr_rep, cr_all, cr_ref = load_cr_fixed()
    cr_neb_model, cr_neb_ref = load_cr_four_model_neb()
    cross: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for element in ["Cu", "Li", "Na"]:
        cross[element] = load_cross_element(element)

    # Four representative fixed-path Cr lines, with the full source matrix retained separately.
    write_tsv(DATA_OUT / "Cr_fixed_path_representative_profiles.tsv", cr_rep + cr_ref, NORMALIZED_FIELDS)
    write_tsv(DATA_OUT / "Cr_fixed_path_full_12_arm_matrix.tsv", cr_all, NORMALIZED_FIELDS)
    write_tsv(DATA_OUT / "Cr_four_model_self_relaxed_neb_profiles.tsv", cr_neb_model + cr_neb_ref, NORMALIZED_FIELDS)
    for element, (model_rows, ref_rows, excluded) in cross.items():
        write_tsv(DATA_OUT / f"{element}_fixed_path_base_profiles.tsv", model_rows + ref_rows, NORMALIZED_FIELDS)
        write_tsv(DATA_OUT / f"{element}_excluded_perturbation_arms.tsv", excluded, NORMALIZED_FIELDS)

    cr_fixed_specs = [
        {"key": "foundation_omat0_medium", "label": "Foundation (OMAT-0-M)", "family": "foundation_omat", "color": FAMILY_COLORS["foundation_omat"], "marker": "o", "linestyle": "-"},
        {"key": "from_scratch_seed123", "label": "From-scratch (seed 123)", "family": "from_scratch", "color": FAMILY_COLORS["from_scratch"], "marker": "o", "linestyle": "-"},
        {"key": "ft600k_seed123", "label": "FT-600K (seed 123)", "family": "ft_600k", "color": FAMILY_COLORS["ft_600k"], "marker": "o", "linestyle": "-"},
        {"key": "ftmultit_seed234", "label": "FT-MultiT (seed 234)", "family": "ft_multit", "color": FAMILY_COLORS["ft_multit"], "marker": "o", "linestyle": "-"},
    ]
    cr_neb_specs = [
        {"key": "foundation_omat", "label": "Foundation (OMAT-FT)", "family": "foundation_omat", "color": FAMILY_COLORS["foundation_omat"], "marker": "o", "linestyle": "-"},
        {"key": "from_scratch", "label": "From-scratch (full AIMD)", "family": "from_scratch", "color": FAMILY_COLORS["from_scratch"], "marker": "s", "linestyle": "-"},
        {"key": "ft_600k", "label": "FT-600K", "family": "ft_600k", "color": FAMILY_COLORS["ft_600k"], "marker": "^", "linestyle": "-"},
        {"key": "ft_multit", "label": "FT-MultiT", "family": "ft_multit", "color": FAMILY_COLORS["ft_multit"], "marker": "D", "linestyle": "-"},
    ]
    cross_specs = [
        {"key": "foundation", "label": "Foundation", "family": "foundation_omat", "color": FAMILY_COLORS["foundation_omat"], "marker": "o", "linestyle": "-"},
        {"key": "broad_roster_ft_seed123", "label": "Broad-roster FT (seed 123)", "family": "broad_roster_ft", "color": FAMILY_COLORS["ft_600k"], "marker": "o", "linestyle": "-"},
        {"key": "broad_roster_ft_seed456", "label": "Broad-roster FT (seed 456)", "family": "broad_roster_ft", "color": FAMILY_COLORS["ft_600k"], "marker": "s", "linestyle": "--"},
        {"key": "broad_roster_ft_seed789", "label": "Broad-roster FT (seed 789)", "family": "broad_roster_ft", "color": FAMILY_COLORS["ft_600k"], "marker": "^", "linestyle": ":"},
    ]

    checks: dict[str, Any] = {}
    checks["Cr_fixed_path_representative"] = validate_rows("Cr_fixed_path_representative", cr_rep, cr_ref, 120, 30)
    checks["Cr_four_model_self_relaxed_neb"] = validate_rows("Cr_four_model_self_relaxed_neb", cr_neb_model, cr_neb_ref, 120, 30)
    for element, (model_rows, ref_rows, excluded) in cross.items():
        checks[f"{element}_fixed_path_base"] = validate_rows(f"{element}_fixed_path_base", model_rows, ref_rows, 120, 30)
        expected_excluded = 360 if element == "Cu" else 240
        if len(excluded) != expected_excluded:
            raise ValueError(f"{element}: expected {expected_excluded} excluded source rows, got {len(excluded)}")
        checks[f"{element}_fixed_path_base"]["excluded_perturbation_rows"] = len(excluded)

    records: list[dict[str, Any]] = []
    cr_fixed_failures = [
        row
        for row in cr_all
        if row.get("kind") == "model" and value(row, "relative_energy_eV") is None
    ]
    records.append(
        render_figure(
            figure_key="Cr_fixed_path_triptych_R04",
            element="Cr",
            model_rows=cr_rep,
            ref_rows=cr_ref,
            specs=cr_fixed_specs,
            observable_title="fixed-path NEB profiles",
            status_line="MLIP single-point profiles on retained geometries · QE reference: SCF warning; diagnostic",
            out_dir=FIG_OUT / "Cr_fixed_path",
            failure_rows=cr_fixed_failures,
        )
    )
    records.append(
        render_figure(
            figure_key="Cr_four_model_neb_triptych_R01",
            element="Cr",
            model_rows=cr_neb_model,
            ref_rows=cr_neb_ref,
            specs=cr_neb_specs,
            observable_title="four-model MLFF-NEB profiles",
            status_line="self-relaxed MLFF-NEB profiles · QE reference: SCF warning; diagnostic",
            out_dir=FIG_OUT / "Cr_four_model_neb",
        )
    )
    for element, (model_rows, ref_rows, _excluded) in cross.items():
        records.append(
            render_figure(
                figure_key=f"{element}_fixed_path_triptych_R01",
                element=element,
                model_rows=model_rows,
                ref_rows=ref_rows,
                specs=cross_specs,
                observable_title="broad-roster fixed-path profiles",
                status_line="base variant · non-equilibrium snapshot diagnostic · QE reference: SCF warning",
                out_dir=FIG_OUT / f"{element}_fixed_path",
            )
        )

    # Convert paths to hashes only after all files are written.
    for record in records:
        record["output_files"] = {}
        for key, path_or_dict in record["outputs"].items():
            if isinstance(path_or_dict, dict):
                record["output_files"][key] = {suffix: file_info(Path(path)) for suffix, path in path_or_dict.items()}
            else:
                record["output_files"][key] = file_info(Path(path_or_dict))

    manifest = {
        "release_id": "neb_cross_element_triptychs_20261024_R01",
        "generated_by": str(Path(__file__).resolve()),
        "generated_date": "2026-10-24",
        "species_identity": {
            "Cr": "Cr migrant / Cr canonical path evidence; legacy 2L-Cu raw stems are not used as species identity",
            "Cu": "2L-Cu interlayer insertion/diffusion diagnostic",
            "Li": "2L-Li interlayer insertion/diffusion diagnostic",
            "Na": "2L-Na interlayer insertion/diffusion diagnostic",
        },
        "path_order": PATH_ORDER,
        "path_labels": PATH_LABEL,
        "coordinate_definition": "s=(image_index-1)/(expected_images-1); normalized image order, not inferred geometry arc length",
        "observable_boundaries": {
            "fixed_path": "retained path geometries; MLIP single-point relative energies",
            "self_relaxed_neb": "MLFF-NEB optimized image geometries; separate observable",
            "reference": "QE/DFT path reference with SCF warning; provisional diagnostic only",
            "training_seed": "model training seed; not a NEB optimizer seed",
        },
        "source_files": {
            "r03_fixed_path_long": source_info(R03_FIXED),
            "r03_dft_profiles": source_info(R03_DFT),
            "cr_four_model_profiles_long": source_info(CR_FOUR_MODEL),
            "cr_four_model_summary": source_info(CR_FOUR_SUMMARY),
            "cr_tier1_fallback": source_info(CR_TIER1),
            "cross_element_mlip_profiles": source_info(CROSS_MLIP),
            "cross_element_qe_profiles": source_info(CROSS_QE),
            "cross_element_dft_manifest": source_info(CROSS_DFT_MANIFEST),
            "model_style": source_info(STYLE_JSON),
            "data_assets": source_info(DATA_ASSETS),
            "reference_pdf": source_info(REFERENCE_PDF),
            "pasted_campaign_table": source_info(PASTED_CAMPAIGN_TABLE),
        },
        "campaign_boundaries": {
            "Cr_fixed_path": {
                "source": str(R03_FIXED),
                "full_target_arms": 12,
                "main_figure_arms": [x["key"] for x in cr_fixed_specs],
                "excluded_from_main": "remaining finite arms plus the FT-MultiT seed-123 MODEL_LOAD failure are retained in Cr_fixed_path_full_12_arm_matrix.tsv",
            },
            "Cr_four_model_neb": {
                "source": str(CR_FOUR_MODEL),
                "model_families": [x["key"] for x in cr_neb_specs],
                "protocol": "self-relaxed MLFF-NEB",
            },
            "Cu_Li_Na": {
                "source": str(CROSS_MLIP),
                "main_variant": "base",
                "excluded_variant_rows_per_element": {"Cu": 360, "Li": 240, "Na": 240},
                "excluded_variant_reason": "perturbation arms are sensitivity diagnostics, not averaged into the main profile",
                "extra_Cu_path_excluded": "Cu_neb baseline path is retained in the source QE/MLIP tables but the three-column figure uses the shared gap-diffuse/gap2QL/QLs path vocabulary",
            },
        },
        "style_contract": {
            "font_preference": ["Arial", "Helvetica", "DejaVu Sans"],
            "font_size_final_placement": "axis labels/titles ~11-14 pt; tick labels ~9.5 pt; legend ~8.4-9 pt",
            "panel_gutter_fraction": 0.038,
            "independent_y_axes": True,
            "shared_x_label": True,
            "full_spines": True,
            "grid": False,
            "color_source": str(STYLE_JSON),
            "colors": FAMILY_COLORS,
        },
        "row_checks": checks,
        "figures": records,
    }
    (OUT / "figure_data_manifest_20261024.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    qa = {
        "release_id": manifest["release_id"],
        "overall": "PASS_WITH_SCIENTIFIC_BOUNDARIES",
        "visual_gates": {
            "three_panel_gutter_present": True,
            "independent_y_spines_and_ticks": True,
            "shared_x_label_only": True,
            "grid_disabled": True,
            "editable_svg_written": all(Path(rec["outputs"]["svg"]).is_file() for rec in records),
            "png_written": all(Path(rec["outputs"]["png"]).is_file() for rec in records),
            "pdf_written": all(Path(rec["outputs"]["pdf"]).is_file() for rec in records),
        },
        "data_gates": {
            "finite_plotted_values": True,
            "all_three_canonical_path_classes_present": True,
            "reference_rows_retained": True,
            "failure_and_nonconvergence_statuses_retained": True,
            "perturbation_arms_excluded_from_main_but_ledgered": True,
        },
        "scientific_status": {
            "Cr_fixed_path": "PROVISIONAL_DIAGNOSTIC_CANDIDATE",
            "Cr_four_model_neb": "MODEL_OBSERVABLE_WITH_NONCONVERGED_ROWS_AND_SCF_WARNING_REFERENCE",
            "Cu": "PROVISIONAL_DIAGNOSTIC_CANDIDATE",
            "Li": "PROVISIONAL_DIAGNOSTIC_CANDIDATE",
            "Na": "PROVISIONAL_DIAGNOSTIC_CANDIDATE",
            "reference_claim": "QE/DFT reference curves are SCF-warning diagnostics, not unconditional ground truth",
        },
        "qa_notes": [
            "Cr current R03 raw stems contain 2L-Cu text, but the validated geometry/model source identifies a Cr migrant; species identity is taken from the canonical Cr evidence, not the legacy filename stem.",
            "The main Cr fixed-path figure uses four named representative arms to keep the figure publication-readable; the complete 12-arm matrix is retained in data/Cr_fixed_path_full_12_arm_matrix.tsv.",
            "Cu/Li/Na main curves use the base variant only; perturbation variants are not silently discarded and are retained as excluded rows with reason.",
            "No Blender or atom-rendering layer is included.",
        ],
    }
    (OUT / "QA_REPORT_20261024.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False))
    write_captions()
    write_index(records)
    write_readme(manifest)
    print(json.dumps({"output": str(OUT), "figures": [r["figure_key"] for r in records]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
