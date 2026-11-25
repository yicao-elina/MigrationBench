#!/usr/bin/env python3
"""Build the durable Phase 2 provenance manifest from copied remote artifacts."""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = ROOT / "remote_artifacts"
summary = json.loads((REMOTE / "runs/retry2/stability_summary.json").read_text())
jobs = {}
for name in ("submission_manifest.tsv", "retry1_submission_manifest.tsv", "retry2_submission_manifest.tsv"):
    path = REMOTE / name
    if path.exists():
        with path.open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                jobs[(row["snapshot"], row["model"], row.get("attempt", "0"))] = row
model_hashes = {
    "foundation-omat": "dc41a98908dd25b150fb5c8dd0a28d95f5feeac37022682a7c389c475444d08e",
    "from-scratch": "da17ab17a98241c9ba1ad75f79ef051af4c273879a85e2371213d0b51257f144",
    "naive-fine-tuning": "6829d16e68bdd1238ee4226363f76ceac936d9e7325c9282283d72093e2bb88a",
    "multi_T-fine-tuning": "0c132dd8b4b953240ff1cc935b251dfd76df0211869888962a3f0d1e7d1d1522",
}
model_paths = {
    "foundation-omat": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/finetuned_MACE_compiled.model",
    "from-scratch": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE_models_l1_0802/mace_l1_0802_compiled.model",
    "naive-fine-tuning": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_600K/finetuned_MACE_multihead0804_compiled.model",
    "multi_T-fine-tuning": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_Multi_T/finetuned_MACE_multihead0804.model",
}
for row in summary:
    row["job_id_attempt2"] = jobs.get((row["snapshot"].replace("snapshot_", ""), row["model"], "2"), {}).get("job_id")
    row["model_path"] = model_paths[row["model"]]
    row["model_sha256"] = model_hashes[row["model"]]
payload = {
    "campaign": "phase2_handoff_md_20261002",
    "status": "complete_for_requested_8_runs",
    "remote_root": "/scratch16/pclancy3/yi/phase2_handoff_md_20261002",
    "source": {
        "trajectory": "/scratch16/pclancy3/yi/mace_four_model_MD_continuation_20260930/1-from-scratch/segment_from_000200000.extxyz",
        "sha256": "38140e3252961d7e2b470232a179ffbeb65d29f2dca4eb0e1d42c9d94c0b45c7",
        "size_bytes": 1283842154,
        "mtime": "2026-10-01 17:02:38.6320757860",
        "natoms": 2050, "pbc": [True, True, True], "frame_stride_steps": 200,
        "properties": "species:S:1:pos:R:3:momenta:R:3:forces:R:3",
        "source_role": "from-scratch continuation; stable prefix before the known step-489000 runaway",
    },
    "snapshots": json.loads((REMOTE / "snapshots/from_scratch_source/snapshot_manifest.json").read_text()),
    "model_registry": [{"model": k, "path": model_paths[k], "sha256": model_hashes[k]} for k in model_paths],
    "protocol": {"requested_snapshots": [461800, 462000], "context_snapshots": [460000, 462200], "steps_per_run": 2000, "duration_ps": 2.0, "write_interval_steps": 10, "timestep_fs": 1.0, "target_temperature_K": 600.0, "friction_1_per_fs": 0.01, "device": "cuda", "velocity_resampling": False, "nve": False},
    "runs": summary,
    "failure_and_retry": {"initial_attempt_jobs": [31511868,31511869,31511870,31511871,31511872,31511873,31511874,31511875], "retry1_jobs": [31511881,31511882,31511883,31511884,31511885,31511886,31511887,31511888], "accepted_attempt2_jobs": [31511895,31511896,31511897,31511898,31511899,31511900,31511901,31511902], "initial_failure": "pre-created output directory violated no-overwrite guard", "retry1_failure": "early retry jobs used the pre-fix pair-NaN finite check; later retry1 jobs are retained but retry2 is canonical", "trajectory_policy": "raw remote trajectory files retained remotely; local release includes exact input snapshots, metrics, summaries, hashes, and paths"},
    "scientific_boundary": "short-range model-stability/handoff evidence only; not DFT truth, not an independent replica ensemble, and not a diffusivity result",
}
(ROOT / "phase2_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
