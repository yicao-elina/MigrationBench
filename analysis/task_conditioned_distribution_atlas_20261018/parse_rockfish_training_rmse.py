#!/usr/bin/env python3
"""Extract original Rockfish validation RMSE checkpoints with provenance.

The source files are JSONL training logs copied additively from the original
Rockfish training directory.  Only ``mode == eval`` records with the default
head are extracted.  The resulting table is descriptive historical training
evidence, not a replacement for grouped held-out multi-seed test statistics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw_cache" / "rockfish_original_training_20261019"
DERIVED = HERE / "derived"

SOURCES = [
    {
        "family": "Foundation",
        "run_label": "OMAT-FT run 123",
        "log_name": "foundation_omat_run123_train.txt",
        "remote_path": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/finetuned_MACE_run-123_train.txt",
        "native_png": "foundation_omat_run123_stage_two.png",
        "lineage_status": "LINEAGE_SUPPORTED",
    },
    {
        "family": "Scratch",
        "run_label": "MACE_models/mace01 run 123",
        "log_name": "scratch_mace01_run123_train.txt",
        "remote_path": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE_models/mace01_run-123_train.txt",
        "native_png": "scratch_mace01_run123_stage_two.png",
        "lineage_status": "PROVENANCE_PARTIAL",
    },
    {
        "family": "FT-600K",
        "run_label": "600K fine-tuning run 123",
        "log_name": "ft600k_run123_train.txt",
        "remote_path": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_600K/finetuned_MACE_multihead0804_run-123_train.txt",
        "native_png": "ft600k_run123_stage_two.png",
        "lineage_status": "LINEAGE_SUPPORTED",
    },
    {
        "family": "FT-MultiT",
        "run_label": "MultiT fine-tuning run 123",
        "log_name": "ft_multit_run123_train.txt",
        "remote_path": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_Multi_T/finetuned_MACE_multihead0804_run-123_train.txt",
        "native_png": "ft_multit_run123_stage_two.png",
        "lineage_status": "LINEAGE_SUPPORTED",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract(source: dict) -> tuple[list[dict], dict]:
    log_path = RAW / "logs" / source["log_name"]
    png_path = RAW / "figures" / source["native_png"]
    log_hash = sha256(log_path)
    png_hash = sha256(png_path)
    rows: list[dict] = []
    seen_epochs: set[int | None] = set()
    record_index = 0
    with log_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("mode") != "eval" or record.get("head") != "default":
                continue
            epoch = record.get("epoch")
            is_initial = epoch is None
            epoch_key = None if is_initial else int(epoch)
            duplicate_record = epoch_key in seen_epochs
            duplicate_initial = is_initial and duplicate_record
            seen_epochs.add(epoch_key)
            rows.append(
                {
                    "family": source["family"],
                    "run_label": source["run_label"],
                    "run_seed": 123,
                    "record_index": record_index,
                    "source_line": line_number,
                    "epoch": -50 if is_initial else int(epoch),
                    "phase": "pre_epoch" if is_initial else "eval",
                    "duplicate_initial": duplicate_initial,
                    "duplicate_record": duplicate_record,
                    "plot_include": not duplicate_record,
                    "rmse_energy_meV_atom": float(record["rmse_e_per_atom"]) * 1000.0,
                    "rmse_force_meV_A": float(record["rmse_f"]) * 1000.0,
                    "mae_energy_meV_atom": float(record["mae_e_per_atom"]) * 1000.0,
                    "mae_force_meV_A": float(record["mae_f"]) * 1000.0,
                    "source_log": str(log_path),
                    "source_log_sha256": log_hash,
                }
            )
            record_index += 1
    manifest_row = {
        **source,
        "local_log": str(log_path),
        "local_log_bytes": log_path.stat().st_size,
        "local_log_sha256": log_hash,
        "local_native_png": str(png_path),
        "local_native_png_bytes": png_path.stat().st_size,
        "local_native_png_sha256": png_hash,
        "eval_records": len(rows),
        "plot_records": sum(row["plot_include"] for row in rows),
        "metric_semantics": "original mode=eval, head=default validation RMSE checkpoints",
        "uncertainty_status": "DESCRIPTIVE_SINGLE_RUN",
    }
    return rows, manifest_row


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    manifest_rows: list[dict] = []
    for source in SOURCES:
        source_rows, manifest_row = extract(source)
        rows.extend(source_rows)
        manifest_rows.append(manifest_row)
    table = pd.DataFrame(rows)
    table = table.sort_values(["family", "record_index"]).reset_index(drop=True)
    table.to_csv(DERIVED / "rockfish_training_rmse_eval.csv", index=False)
    manifest = {
        "package": "task_conditioned_distribution_atlas_20261018",
        "created": "2026-10-19",
        "source_root": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer",
        "selection": "JSONL records with mode=eval and head=default; duplicate log checkpoints retained but excluded from plotting",
        "records": manifest_rows,
        "claim_boundary": "The reconstructed curves are descriptive single-run validation-log evidence. They do not establish seed uncertainty or grouped held-out test performance.",
    }
    (DERIVED / "rockfish_training_rmse_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(table)} eval records to {DERIVED / 'rockfish_training_rmse_eval.csv'}")


if __name__ == "__main__":
    main()
