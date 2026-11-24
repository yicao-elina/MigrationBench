#!/usr/bin/env python3
"""Render the frozen 2x2 interlayer-sliding Foundation figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


MODELS = [
    ("mp0b3_medium", "MACE-MP-0b3-medium", "#d5a238"),
    ("omat0_medium", "MACE-OMAT-0-medium", "#d85a45"),
    ("mpa0_medium", "MACE-MPA-0-medium", "#5d8b56"),
]
PANELS = [
    ("A", "FIXED_PATH_SINGLE_POINT", "bulk", "Bulk", "fixed path · single point"),
    ("B", "FIXED_PATH_SINGLE_POINT", "cr_in_gap", "Cr-in-gap", "fixed path · single point"),
    ("C", "SELF_RELAXED_ENDPOINT_ONLY_NEB", "bulk", "Bulk", "self-relaxed · endpoint-only NEB"),
    ("D", "SELF_RELAXED_ENDPOINT_ONLY_NEB", "cr_in_gap", "Cr-in-gap", "self-relaxed · endpoint-only NEB"),
]
REQUIRED_MATRIX_PROVENANCE_FIELDS = (
    "input_sha256", "model_sha256_observed", "model_sha256_registry", "model_registry_sha256",
    "runner_sha256", "wrapper_sha256", "job_table_sha256", "job_id", "environment_sha256", "output_sha256",
)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def finite_values(rows: list[dict], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def expected_record_metadata() -> dict[str, dict[str, str]]:
    return {
        f"{'fixed' if observable == 'FIXED_PATH_SINGLE_POINT' else 'self'}__{system}__{model_id}": {
            "observable": observable,
            "system": system,
            "plot_column": column,
            "model_id": model_id,
            "model_label": label,
        }
        for _letter, observable, system, column, _title in PANELS
        for model_id, label, _color in MODELS
    }


def expected_record_ids() -> set[str]:
    return set(expected_record_metadata())


def validate_data_gate(root: Path, matrix: list[dict], profiles: list[dict], qe: list[dict]) -> tuple[bool, list[str]]:
    """Validate the exact collected release shape before any final rendering."""
    reasons: list[str] = []
    expected_metadata = expected_record_metadata()
    expected_ids = set(expected_metadata)
    observed_ids = {row.get("record_id", "") for row in matrix}
    if len(matrix) != 12 or observed_ids != expected_ids:
        reasons.append(f"expected exactly 12 primary record IDs, found {len(matrix)} rows and {len(observed_ids - expected_ids)} unexpected IDs")
    allowed_status = {"FINITE_DIAGNOSTIC", "CONVERGED"}
    for row in matrix:
        record_id = row.get("record_id", "")
        metadata = expected_metadata.get(record_id)
        if metadata is None:
            reasons.append(f"{record_id}: unexpected primary record ID")
        else:
            for field, expected_value in metadata.items():
                if row.get(field) != expected_value:
                    reasons.append(f"{record_id}: {field}={row.get(field)!r}, expected {expected_value!r}")
            expected_atoms = "80" if metadata["system"] == "bulk" else "81"
            if row.get("atoms") != expected_atoms:
                reasons.append(f"{record_id}: atoms={row.get('atoms')!r}, expected {expected_atoms!r}")
        if row.get("scientific_status") not in allowed_status:
            reasons.append(f"{record_id}: scientific_status={row.get('scientific_status')}")
        if row.get("publication_status") != "PRIMARY_EVIDENCE":
            reasons.append(f"{record_id}: publication_status={row.get('publication_status')}")
        for field in REQUIRED_MATRIX_PROVENANCE_FIELDS:
            if row.get(field, "") in (None, ""):
                reasons.append(f"{record_id}: missing {field}")
        try:
            if int(row["n_images"]) != 10 or not math.isfinite(float(row["barrier_eV"])):
                reasons.append(f"{record_id}: image/barrier field invalid")
        except (KeyError, TypeError, ValueError):
            reasons.append(f"{record_id}: image/barrier field missing or nonnumeric")
    profile_counts: dict[str, int] = {}
    for row in profiles:
        record_id = row.get("record_id", "")
        profile_counts[record_id] = profile_counts.get(record_id, 0) + 1
        metadata = expected_metadata.get(record_id)
        if metadata is None:
            reasons.append(f"{record_id}: unexpected profile record ID")
        else:
            for field, expected_value in metadata.items():
                if row.get(field) != expected_value:
                    reasons.append(f"{record_id}: profile {field}={row.get(field)!r}, expected {expected_value!r}")
        try:
            index = int(row["image_index"])
            x = float(row["image_fraction"])
            energy = float(row["energy_eV"])
            relative = float(row["relative_energy_eV"])
            if index not in range(10) or abs(x - index / 9.0) > 1.0e-12 or not all(math.isfinite(v) for v in (x, energy, relative)):
                reasons.append(f"{record_id}: invalid image coordinate or nonfinite profile value")
        except (KeyError, TypeError, ValueError):
            reasons.append(f"{record_id}: malformed profile row")
    if len(profiles) != 120:
        reasons.append(f"expected 120 model profile rows, found {len(profiles)}")
    for record_id in expected_ids:
        rows = [row for row in profiles if row.get("record_id") == record_id]
        if len(rows) != 10:
            reasons.append(f"{record_id}: expected 10 profile rows, found {len(rows)}")
            continue
        try:
            indices = [int(row["image_index"]) for row in rows]
            if set(indices) != set(range(10)) or len(indices) != len(set(indices)):
                reasons.append(f"{record_id}: image indices are not the exact set 0..9: {indices}")
            rows.sort(key=lambda row: int(row["image_index"]))
            if abs(float(rows[0]["relative_energy_eV"])) > 1.0e-10:
                reasons.append(f"{record_id}: image-0 relative energy is not zero")
        except (KeyError, TypeError, ValueError):
            reasons.append(f"{record_id}: image-0 baseline unavailable")
    if len(qe) != 20:
        reasons.append(f"expected 20 QE reference rows, found {len(qe)}")
    for system in ("bulk", "cr_in_gap"):
        rows = [row for row in qe if row.get("system") == system]
        if len(rows) != 10:
            reasons.append(f"QE {system}: expected 10 rows, found {len(rows)}")
            continue
        try:
            indices = [int(row["image_index"]) for row in rows]
            if set(indices) != set(range(10)) or len(indices) != len(set(indices)):
                reasons.append(f"QE {system}: image indices are not the exact set 0..9: {indices}")
            rows.sort(key=lambda row: int(row["image_index"]))
            values = [float(row["relative_energy_eV"]) for row in rows]
            x_values = [float(row["image_fraction"]) for row in rows]
            if not all(math.isfinite(value) for value in values + x_values):
                reasons.append(f"QE {system}: nonfinite value")
            if any(abs(x - index / 9.0) > 1.0e-12 for index, x in enumerate(x_values)):
                reasons.append(f"QE {system}: image coordinate is not normalized 0..1")
            if abs(values[0]) > 1.0e-10:
                reasons.append(f"QE {system}: image-0 baseline is not zero")
        except (KeyError, TypeError, ValueError):
            reasons.append(f"QE {system}: malformed reference rows")
    return not reasons, reasons


def panel_limits(values: list[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    span = high - low
    pad = max(0.08, 0.08 * span)
    if span == 0:
        pad = 0.15
    return low - pad, high + pad


def draw_panel(ax, letter: str, observable: str, system: str, title: str, profiles: list[dict], qe: list[dict], standalone: bool = False):
    model_handles = []
    values = []
    for model_id, label, color in MODELS:
        rows = [row for row in profiles if row["observable"] == observable and row["system"] == system and row["model_id"] == model_id]
        rows.sort(key=lambda row: int(row["image_index"]))
        x = [float(row["image_fraction"]) for row in rows]
        y = [float(row["relative_energy_eV"]) for row in rows]
        if rows:
            ax.plot(x, y, color=color, lw=1.9, marker="o", ms=4.2, mfc="none", mec=color, mew=1.1, label=label, zorder=3)
            values.extend(y)
        model_handles.append(Line2D([0], [0], color=color, lw=1.9, marker="o", ms=4.2, mfc="none", mec=color, mew=1.1, label=label))

    qe_rows = [row for row in qe if row["system"] == system]
    qe_rows.sort(key=lambda row: int(row["image_index"]))
    qe_color = "#222222" if observable == "FIXED_PATH_SINGLE_POINT" else "#777777"
    qe_label = "QE b-path2 retained reference (SCF-warning)" if observable == "FIXED_PATH_SINGLE_POINT" else "QE retained path (context only)"
    if qe_rows:
        x = [float(row["image_fraction"]) for row in qe_rows]
        y = [float(row["relative_energy_eV"]) for row in qe_rows]
        ax.plot(x, y, color=qe_color, lw=1.45, linestyle=(0, (4, 2)), marker="D", ms=4.2, mfc="none", mec=qe_color, mew=1.0, label=qe_label, zorder=2)
        values.extend(y)

    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    if values:
        ax.set_ylim(*panel_limits(values))
    ax.set_title(f"{letter}  {system.replace('_', '-') } · {title}", loc="left", fontsize=11.2, fontweight="bold", pad=7)
    ax.tick_params(axis="both", labelsize=10.5, length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.grid(False)
    if standalone:
        ax.set_xlabel("normalized image coordinate (0 → 1)", fontsize=11.0)
        ax.set_ylabel("ΔE from image 0 (eV)", fontsize=11.0)
        handles = model_handles + [Line2D([0], [0], color=qe_color, lw=1.45, linestyle=(0, (4, 2)), marker="D", ms=4.2, mfc="none", mec=qe_color, label=qe_label)]
        ax.legend(handles=handles, loc="best", frameon=False, fontsize=10.5, handlelength=2.5)
    return model_handles, Line2D([0], [0], color=qe_color, lw=1.45, linestyle=(0, (4, 2)), marker="D", ms=4.2, mfc="none", mec=qe_color, label=qe_label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--diagnostic", action="store_true", help="Allow rendering when the 12-row data gate is incomplete.")
    args = parser.parse_args()
    root = args.root.resolve()
    result_matrix = root / "results/tables/RESULT_MATRIX.tsv"
    profiles_path = root / "results/tables/IMAGE_PROFILES.tsv"
    qe_path = root / "results/tables/QE_REFERENCE.tsv"
    missing = [str(path) for path in (result_matrix, profiles_path, qe_path) if not path.is_file()]
    if missing:
        message = "12-row scientific gate is incomplete: missing collected table(s): " + ", ".join(missing)
        if args.diagnostic:
            raise SystemExit(message + "; diagnostic rendering requires the collector tables to exist")
        raise SystemExit(message)
    matrix = read_tsv(result_matrix)
    profiles = read_tsv(profiles_path)
    qe = read_tsv(qe_path)
    complete, gate_reasons = validate_data_gate(root, matrix, profiles, qe)
    if not complete and not args.diagnostic:
        raise SystemExit("12-row scientific gate is incomplete: " + "; ".join(gate_reasons[:12]))

    outdir = root / "results/figures/foundation_interlayer_sliding_2x2_R13"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "pdf.fonttype": 42, "svg.fonttype": "none"})

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.5), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.12, top=0.79, wspace=0.22, hspace=0.32)
    all_model_handles = None
    qe_fixed_handle = None
    qe_self_handle = None
    for index, (letter, observable, system, _column, title) in enumerate(PANELS):
        ax = axes.flat[index]
        model_handles, qe_line = draw_panel(ax, letter, observable, system, title, profiles, qe)
        if all_model_handles is None:
            all_model_handles = model_handles
        if observable == "FIXED_PATH_SINGLE_POINT" and qe_fixed_handle is None:
            qe_fixed_handle = qe_line
        if observable == "SELF_RELAXED_ENDPOINT_ONLY_NEB" and qe_self_handle is None:
            qe_self_handle = qe_line
        if index in (0, 1):
            ax.set_xlabel("")
        if index in (1, 3):
            ax.set_ylabel("")
    fig.legend(handles=all_model_handles + [qe_fixed_handle, qe_self_handle], loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False, fontsize=10.5, handlelength=2.5, columnspacing=1.3)
    fig.supxlabel("normalized image coordinate (0 → 1)", x=0.54, y=0.035, fontsize=11.0)
    fig.supylabel("ΔE from image 0 (eV)", x=0.018, y=0.47, fontsize=11.0)
    composite_paths = []
    for suffix in ("pdf", "svg", "png"):
        path = outdir / f"foundation_interlayer_sliding_2x2_R13.{suffix}"
        fig.savefig(path, dpi=320 if suffix == "png" else None, bbox_inches="tight", pad_inches=0.04)
        composite_paths.append(str(path))
    plt.close(fig)

    panel_paths = []
    for letter, observable, system, _column, title in PANELS:
        fig, ax = plt.subplots(figsize=(4.25, 3.35))
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.18, top=0.88)
        draw_panel(ax, letter, observable, system, title, profiles, qe, standalone=True)
        stem = f"panel_{letter}_{observable.lower()}_{system}"
        for suffix in ("pdf", "svg", "png"):
            path = outdir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=320 if suffix == "png" else None, bbox_inches="tight", pad_inches=0.04)
            panel_paths.append(str(path))
        plt.close(fig)

    provenance = {
        "schema_version": "foundation_interlayer_sliding_2x2_figure_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": "foundation_interlayer_sliding_2x2_20261024",
        "figure_status": "ACCEPTED_DATA_GATE_PENDING_REVIEWS" if complete else "DIAGNOSTIC_INCOMPLETE_DATA_GATE",
        "acceptance_freeze": str(root / "ACCEPTANCE_FREEZE_R13.json"),
        "acceptance_freeze_sha256": sha256(root / "ACCEPTANCE_FREEZE_R13.json"),
        "style_registry": str(root / "STYLE_REGISTRY_R13.json"),
        "style_registry_sha256": sha256(root / "STYLE_REGISTRY_R13.json"),
        "reference_pdf": str(root / "references/foundation_model_class_3x3_R03.pdf"),
        "reference_pdf_sha256": sha256(root / "references/foundation_model_class_3x3_R03.pdf"),
        "result_matrix": str(result_matrix),
        "result_matrix_sha256": sha256(result_matrix),
        "image_profiles": str(profiles_path),
        "image_profiles_sha256": sha256(profiles_path),
        "qe_reference": str(qe_path),
        "qe_reference_sha256": sha256(qe_path),
        "plot_script": str(Path(__file__).resolve()),
        "plot_script_sha256": sha256(Path(__file__).resolve()),
        "composite_outputs": composite_paths,
        "panel_outputs": panel_paths,
        "model_count": 3,
        "panel_count": 4,
        "data_rows": len(matrix),
        "profile_rows": len(profiles),
        "data_gate_reasons": gate_reasons,
    }
    (outdir / "FIGURE_PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
