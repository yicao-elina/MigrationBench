#!/usr/bin/env python3
"""Build an explicit curated rerun record from the read-only WSe2 audit TSVs."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE / "wse2_run_lineage.tsv"
CANDIDATES = HERE / "candidate_wfc_manifest.tsv"
OUT = HERE / "WSE2_REPRODUCTION_MANIFEST.json"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def main() -> None:
    runs = read_tsv(RUNS)
    candidates = read_tsv(CANDIDATES)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    payload = {
        "manifest_id": "ROCKFISH-SCRATCH4-WSE2-CURATED-RERUN-20261122",
        "manifest_type": "CURATED_REPRODUCTION_RECORD",
        "record_status": "RECONSTRUCTION",
        "created_utc": created,
        "original_run_manifest_present": False,
        "source_root": "/scratch4/pclancy3/yi/3.WSe2/1.pwrelax",
        "project_id": "WSE2",
        "workflow_type": "QE variable-cell relaxation",
        "scientific_status": "PROVENANCE_PARTIAL",
        "publication_status": "UNADJUDICATED",
        "scope_note": "This is a newly curated run/reproduction record, not a recovered historical manifest. It captures enough surviving inputs and runtime evidence to rerun the relaxations; no manuscript claim or figure link has been established.",
        "reproduction": {
            "staging_rule": "Use a fresh empty working directory per run; copy the retained QE input and exact submitted Slurm script there before submission, so QE writes a new WSe2_vc_relax_z.save tree without overwriting retained output.",
            "command_template": "mkdir -p <fresh_run_directory> && cp <retained_input> <fresh_run_directory>/ && cp <retained_slurm_script> <fresh_run_directory>/ && cd <fresh_run_directory> && sbatch sbatch-relax+nscf+bands.slurm",
            "runtime_setup": ["#!/bin/bash -l", "module reset", "source /data/apps/go.sh", "ml intel", "ml qe/7.2"],
            "executable": "/data/apps/extern/qe/7.2/bin/pw.x",
            "mpi_ranks": 48,
            "active_calculation_command": "mpirun /data/apps/extern/qe/7.2/bin/pw.x < <run_input>.in > relax.out",
            "inactive_script_lines": "The NSCF and bands commands in the submitted script are commented out; the recorded jobs ran vc-relax only.",
            "initial_wavefunctions": "The surviving QE outputs say `Starting wfcs are ... randomized atomic wfcs`; the preserved inputs specify outdir='./' and do not set startingwfc or restart_mode. Use a fresh empty directory for a from-scratch rerun.",
            "runtime_limit": "The exact QE binary path and module version were recorded. The shared `/data/apps/go.sh` setup script was referenced but not hashed; bitwise identity is not claimed.",
        },
        "runs": runs,
        "deletion_candidates": candidates,
        "retained_local_evidence": {
            "run_lineage_tsv": str(RUNS),
            "candidate_manifest_tsv": str(CANDIDATES),
            "predelete_hashes": str(HERE / "wse2_wfc_sha256_predelete.txt"),
            "retained_evidence_hashes": str(HERE / "wse2_reproduction_evidence_sha256.txt"),
            "copied_submitted_scripts": str(HERE / "sbatch_scripts"),
        },
        "cleanup_preconditions": [
            "Before deletion, refresh current squeue and confirm none of job instances 19076231@2024-12-20, 19075950@2024-12-20, or 19075952@2024-12-20 is live; Slurm numeric IDs were reused in 2026.",
            "Recompute every candidate SHA256 and size against candidate_wfc_manifest.tsv immediately before deletion.",
            "Keep all QE inputs, submitted scripts, Slurm logs, relax.out, XML, charge density, pseudopotentials, and this curated record.",
            "Update the SQL artifact locations and cleanup receipt after deletion, then refresh GPFS group quota.",
        ],
        "limitations": [
            "The downstream search in SUMMARY.md covered only /scratch4/pclancy3/yi and cannot prove there are no references elsewhere on Rockfish.",
            "No original per-run manifest or paper-claim mapping was found.",
            "Scheduler COMPLETED 0:0 is not scientific or publication acceptance by itself.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
