#!/usr/bin/env python3
"""Build an auditable atlas of the canonical, locally synced QE NEB profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "migrationbench"))

from parse_qe_neb_output import parse_neb_out  # noqa: E402


HISTORICAL_PATHS = {
    **{
        f"1-{index}_historical": ROOT / "data_raw" / "historical_neb" / "legacy_61" / f"1-{index}" / "neb.out"
        for index in range(2, 9)
    },
    **{
        f"81_neb_{index}": ROOT / "data_raw" / "historical_neb" / "layer_81" / f"neb_{index}" / "neb.out"
        for index in range(1, 6)
    },
}

CURRENT_PATHS = {
    "1-6_qe_r3": ROOT / "cluster" / "mb_qe16_direct_r3_s42_30812496" / "neb.out",
    "1-7_qe_r3": ROOT / "cluster" / "mb_qe17_direct_r3_s42_30812497" / "neb.out",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    return value is not None and math.isfinite(float(value))


def collect_record(path_id: str, source: Path, source_class: str) -> dict:
    parsed = parse_neb_out(source)
    images = parsed.get("last_images") or []
    energies = [float(row["energy_eV"]) for row in images if finite(row.get("energy_eV"))]
    if len(energies) != len(images) or not energies:
        profile = []
        forward = reverse = endpoint_delta = None
        peak_image = None
    else:
        profile = [energy - energies[0] for energy in energies]
        peak_offset = max(range(len(energies)), key=energies.__getitem__)
        peak_image = int(images[peak_offset]["image_index"])
        forward = max(energies) - energies[0]
        reverse = max(energies) - energies[-1]
        endpoint_delta = energies[-1] - energies[0]

    converged = bool(parsed.get("converged_by_default_gate"))
    if converged:
        status = "accepted_converged"
    elif source_class == "historical":
        status = "historical_candidate_quarantine"
    elif parsed.get("job_done"):
        status = "stopped_unconverged"
    else:
        status = "running_unconverged_synced_snapshot"

    if peak_image is None:
        barrier_shape = "unavailable"
    elif peak_image in {int(images[0]["image_index"]), int(images[-1]["image_index"])}:
        barrier_shape = "endpoint_dominated"
    else:
        barrier_shape = "interior_peak_candidate"

    return {
        "path_id": path_id,
        "source_class": source_class,
        "status": status,
        "barrier_shape": barrier_shape,
        "qe_reported_forward_candidate_eV": parsed.get("activation_forward_eV"),
        "qe_reported_reverse_candidate_eV": parsed.get("activation_reverse_eV"),
        "discrete_forward_candidate_eV": forward,
        "discrete_reverse_candidate_eV": reverse,
        "endpoint_delta_eV": endpoint_delta,
        "peak_image": peak_image,
        "max_movable_error_eV_A": parsed.get("max_image_error_eV_A"),
        "max_all_image_error_eV_A": parsed.get("max_image_error_all_eV_A"),
        "barrier_drift_last_three_eV": parsed.get("barrier_drift_last_three_eV"),
        "last_complete_iteration": parsed.get("last_complete_iteration"),
        "job_done": parsed.get("job_done"),
        "converged_by_default_gate": converged,
        "n_images": len(images),
        "image_indices": [int(row["image_index"]) for row in images],
        "relative_energies_eV": profile,
        "image_errors_eV_A": [row.get("error_eV_A") for row in images],
        "frozen_images": [bool(row.get("frozen")) for row in images],
        "source_file": str(source),
        "source_sha256": sha256_file(source),
        "source_mtime_utc": datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat(),
    }


def display_name(path_id: str) -> str:
    return path_id.replace("_historical", " (historical)").replace("_qe_r3", " (QE r3)")


def draw_profile(ax, record: dict, compact: bool = False) -> None:
    xs = record["image_indices"]
    ys = record["relative_energies_eV"]
    color = "#2563eb" if record["source_class"] == "current" else "#b45309"
    ax.axhline(0, color="#9ca3af", linewidth=0.8, linestyle="--", zorder=0)
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4)
    for x, y, frozen in zip(xs, ys, record["frozen_images"]):
        if frozen:
            ax.scatter([x], [y], facecolor="white", edgecolor=color, linewidth=1.5, s=32, zorder=3)
    title = display_name(record["path_id"])
    candidate = record["qe_reported_forward_candidate_eV"]
    if finite(candidate):
        title += f" | forward candidate {candidate:.3f} eV"
    ax.set_title(title, fontsize=9 if compact else 12, loc="left", pad=6)
    ax.set_xlabel("NEB image index", fontsize=8 if compact else 10)
    ax.set_ylabel("E - E(initial) (eV)", fontsize=8 if compact else 10)
    ax.set_xticks(xs)
    ax.tick_params(labelsize=7 if compact else 9)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if not compact:
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        note = (
            f"{record['status']} | {record['barrier_shape']} | "
            f"max movable error={record['max_movable_error_eV_A']!s} eV/A"
        )
        ax.text(0, -0.24, note, transform=ax.transAxes, fontsize=8, color="#4b5563")


def write_csv(records: list[dict], path: Path) -> None:
    fields = [
        "path_id", "source_class", "status", "barrier_shape", "qe_reported_forward_candidate_eV",
        "qe_reported_reverse_candidate_eV", "discrete_forward_candidate_eV",
        "discrete_reverse_candidate_eV", "endpoint_delta_eV", "peak_image", "max_movable_error_eV_A",
        "max_all_image_error_eV_A", "barrier_drift_last_three_eV", "last_complete_iteration",
        "job_done", "converged_by_default_gate", "n_images", "source_mtime_utc", "source_sha256",
        "source_file",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})


def write_markdown(records: list[dict], path: Path, created_at: str) -> None:
    lines = [
        "# QE NEB Barrier Atlas",
        "",
        f"Generated: `{created_at}` from locally synced outputs.",
        "",
        "> No row currently passes the unified final-reference gate. Values below are candidate energy spans, not manuscript-ready migration barriers. `endpoint_dominated` means the maximum is a frozen endpoint rather than an interior saddle image.",
        "",
        "| Path | QE forward candidate (eV) | QE reverse candidate (eV) | Endpoint delta (eV) | Peak | Max movable error (eV/A) | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in records:
        fmt = lambda value: "" if not finite(value) else f"{float(value):.6f}"
        lines.append(
            f"| `{row['path_id']}` | {fmt(row['qe_reported_forward_candidate_eV'])} | {fmt(row['qe_reported_reverse_candidate_eV'])} | "
            f"{fmt(row['endpoint_delta_eV'])} | {row['peak_image'] or ''} ({row['barrier_shape']}) | "
            f"{fmt(row['max_movable_error_eV_A'])} | `{row['status']}` |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `accepted_converged` requires the parser's force and barrier-drift gates; there are currently zero accepted rows.",
        "- `historical_candidate_quarantine` preserves old numerical evidence but does not promote it to a final reference.",
        "- Current r3 rows are the latest locally synced snapshots, not a fresh scheduler poll.",
        "- The table reports QE's interpolated forward/reverse activation energies from the last complete iteration.",
        "- The JSON/CSV also retain discrete `max(E)-E_initial` and `max(E)-E_final` values; those actual image energies are used to draw the curves.",
        "",
    ])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data_processed" / "neb_barrier_atlas")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path_id, source in {**HISTORICAL_PATHS, **CURRENT_PATHS}.items():
        if not source.exists():
            raise FileNotFoundError(source)
        records.append(collect_record(path_id, source, "current" if path_id in CURRENT_PATHS else "historical"))

    created_at = datetime.now(timezone.utc).isoformat()
    payload = {"created_at_utc": created_at, "records": records}
    (args.out_dir / "barrier_atlas.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(records, args.out_dir / "barrier_atlas.csv")
    write_markdown(records, args.out_dir / "README.md", created_at)

    for record in records:
        fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
        draw_profile(ax, record)
        stem = record["path_id"]
        fig.savefig(args.out_dir / f"{stem}.png", dpi=180, facecolor="white")
        fig.savefig(args.out_dir / f"{stem}.svg", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(16, 13), constrained_layout=True)
    for ax, record in zip(axes.flat, records):
        draw_profile(ax, record, compact=True)
    for ax in list(axes.flat)[len(records):]:
        ax.axis("off")
    fig.suptitle("QE NEB energy-profile atlas: all values provisional until acceptance gates pass", fontsize=16)
    fig.savefig(args.out_dir / "all_qe_neb_profiles.png", dpi=180, facecolor="white")
    fig.savefig(args.out_dir / "all_qe_neb_profiles.svg", facecolor="white")
    plt.close(fig)
    print(json.dumps({"records": len(records), "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
