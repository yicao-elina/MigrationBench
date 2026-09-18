#!/usr/bin/env python3
"""Plot checkpoint-scoped train/evaluation nearest-neighbor distances."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "data_processed"
    / "training_leakage_audit"
    / "checkpoint_finetuned_MACE_multihead0804_s42"
    / "nearest_neighbors.csv"
)
SOAP_THRESHOLD = 1.0e-4
RMSD_THRESHOLD_A = 0.05


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def quantile(values: list[float], fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["reference_split"]].append(row)

    result = []
    for split in ("train", "valid", "test"):
        split_rows = grouped.get(split, [])
        soap = [value for row in split_rows if (value := finite_float(row.get("soap_cosine_distance"))) is not None]
        rmsd = [value for row in split_rows if (value := finite_float(row.get("cr_local_species_rmsd_A"))) is not None]
        result.append(
            {
                "reference_split": split,
                "evaluation_rows": len(split_rows),
                "soap_comparable_rows": len(soap),
                "soap_min_cosine_distance": min(soap) if soap else None,
                "soap_q05_cosine_distance": quantile(soap, 0.05),
                "soap_median_cosine_distance": quantile(soap, 0.5),
                "soap_lte_threshold": sum(value <= SOAP_THRESHOLD for value in soap),
                "soap_min_over_threshold": min(soap) / SOAP_THRESHOLD if soap else None,
                "rmsd_comparable_rows": len(rmsd),
                "rmsd_incompatible_species_cardinality_rows": len(split_rows) - len(rmsd),
                "rmsd_min_A": min(rmsd) if rmsd else None,
                "rmsd_q05_A": quantile(rmsd, 0.05),
                "rmsd_median_A": quantile(rmsd, 0.5),
                "rmsd_lte_threshold": sum(value <= RMSD_THRESHOLD_A for value in rmsd),
                "rmsd_min_over_threshold": min(rmsd) / RMSD_THRESHOLD_A if rmsd else None,
            }
        )
    return result


def empirical_cdf(values: list[float]):
    ordered = sorted(values)
    return ordered, [(index + 1) / len(ordered) for index in range(len(ordered))]


def draw(rows: list[dict], output_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"train": "#2A6F97", "valid": "#C44536", "test": "#2A9D8F"}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["reference_split"]].append(row)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    specs = [
        ("soap_cosine_distance", SOAP_THRESHOLD, "SOAP cosine distance", True),
        ("cr_local_species_rmsd_A", RMSD_THRESHOLD_A, "Cr-local species-matched RMSD (A)", False),
    ]
    for ax, (column, threshold, xlabel, log_x) in zip(axes, specs):
        for split in ("train", "valid", "test"):
            values = [
                value
                for row in grouped.get(split, [])
                if (value := finite_float(row.get(column))) is not None
            ]
            xs, ys = empirical_cdf(values)
            ax.step(xs, ys, where="post", linewidth=2, color=colors[split], label=f"{split} (n={len(values)})")
        ax.axvline(threshold, color="#222222", linewidth=1.2, linestyle="--", label=f"overlap threshold = {threshold:g}")
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Empirical cumulative fraction")
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.7)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_title("All Cr-centered environments")
    axes[1].set_title("Only species-cardinality-compatible environments")
    fig.suptitle("Checkpoint-scoped NEB/train nearest-neighbor audit", fontsize=14)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(output_stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INPUT.parent)
    args = parser.parse_args()

    with args.input.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"reference_split", "soap_cosine_distance", "cr_local_species_rmsd_A"}
    missing = required.difference(rows[0] if rows else {})
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "leakage_distance_distributions"
    draw(rows, stem)

    payload = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "declared fine-tuning checkpoint files versus current 1-6/1-7 QE evaluation histories",
        "scope_limit": "unknown foundation-model pretraining data are not auditable from available files",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "thresholds": {"soap_cosine_distance": SOAP_THRESHOLD, "cr_local_species_rmsd_A": RMSD_THRESHOLD_A},
        "splits": summary,
    }
    (args.output_dir / "leakage_distance_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    fields = list(summary[0])
    with (args.output_dir / "leakage_distance_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
