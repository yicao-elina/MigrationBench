#!/usr/bin/env python3
"""Bootstrap an isolated canonical Scratch-600K three-path NEB campaign.

The script is intended to run on Rockfish.  It copies the already audited
single-point and endpoint-only runners from the 20261019 campaign, copies the
canonical 81-atom path assets, and creates a new registry-backed 3-seed job
matrix without modifying the parent campaign.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/scratch16/pclancy3/yi/scratch_600k_migration_20261101")
SOURCE = Path("/scratch16/pclancy3/yi/neb_fixed_vs_self_matrix_20261019")
MODEL_ROOT = Path("/scratch16/pclancy3/yi/scratch_lmax1_grouped_20261019_repair_r01/models")
CAMPAIGN = "scratch_600k_migration_canonical_20261101"
SEEDS = (123, 234, 345)
PATHS = (
    ("cr_canonical_in_gap", "in-gap diffusion", "in_gap"),
    ("cr_canonical_gap_to_ql", "gap-to-QL", "gap_to_ql"),
    ("cr_canonical_within_ql", "within-QL", "within_ql"),
)
CODE_FILES = (
    "capture_runtime_provenance.py",
    "run_fixed_job.py",
    "run_fixed_single_point.py",
    "run_self_job.py",
    "run_endpoint_only_neb.py",
    "run_mlff_neb.py",
    "export_mlff_neb_iteration_history.py",
    "run_model_load_smoke.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def slurm_script(kind: str, count: int) -> str:
    if kind == "fixed":
        job_name = "mbscratchfixed"
        table = "fixed_jobs.tsv"
        runner = "run_fixed_job.py"
        stem = "fixed"
        memory = "32G"
        time = "02:00:00"
    elif kind == "idpp":
        job_name = "mbscratchidpp"
        table = "idpp_jobs.tsv"
        runner = "run_self_job.py"
        stem = "idpp"
        memory = "48G"
        time = "08:00:00"
    elif kind == "smoke":
        job_name = "mbscratchload"
        table = "MODEL_REGISTRY.tsv"
        runner = "run_model_load_smoke.py"
        stem = "load"
        memory = "16G"
        time = "00:30:00"
    else:
        raise ValueError(kind)
    array = f"0-{count - 1}%3"
    if kind == "smoke":
        command = (
            '"${PYTHON}" "${ROOT}/code/run_model_load_smoke.py" '
            '--model-table "${ROOT}/MODEL_REGISTRY.tsv" '
            '--index "${SLURM_ARRAY_TASK_ID}" '
            '--input "${ROOT}/inputs/endpoints_r01/cr_canonical_in_gap__initial.extxyz" '
            '--out-root "${ROOT}/smoke" --device cpu'
        )
    else:
        command = (
            f'"${{PYTHON}}" "${{ROOT}}/code/{runner}" '
            f'--root "${{ROOT}}" --job-table "${{ROOT}}/jobs_r01/{table}" '
            '--index "${SLURM_ARRAY_TASK_ID}"'
        )
    return f'''#!/bin/bash -l
# {CAMPAIGN}: {kind} array; one row per model-seed/path pair.
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --partition=parallel
#SBATCH -A pclancy3
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={memory}
#SBATCH --array={array}
#SBATCH --output={ROOT}/logs/{stem}-%A_%a.out
#SBATCH --error={ROOT}/logs/{stem}-%A_%a.err

set -euo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export PATH=/cm/shared/apps/slurm/current/bin:$PATH
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK}}"
export PYTHON=/data/pclancy3/yi/.conda/envs/mace/bin/python
ROOT={ROOT}
{command}
'''


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for relative in ("code", "inputs/fixed_r01", "inputs/endpoints_r01", "jobs_r01", "runs", "logs", "smoke"):
        (ROOT / relative).mkdir(parents=True, exist_ok=True)

    for name in CODE_FILES:
        shutil.copy2(SOURCE / "code" / name, ROOT / "code" / name)

    # The old endpoint-only wrapper embeds the old campaign identifier.  Patch
    # only the copied child, leaving the source campaign untouched.
    endpoint_wrapper = ROOT / "code" / "run_endpoint_only_neb.py"
    text = endpoint_wrapper.read_text()
    old = '"campaign_id": "neb_fixed_vs_self_matrix_20261019"'
    new = f'"campaign_id": "{CAMPAIGN}"'
    if text.count(old) != 1:
        raise RuntimeError("unexpected endpoint-only wrapper campaign identifier")
    endpoint_wrapper.write_text(text.replace(old, new))
    fixed_runner = ROOT / "code" / "run_fixed_single_point.py"
    text = fixed_runner.read_text()
    old = '"campaign_id": "neb_fixed_vs_self_matrix_20261019"'
    if text.count(old) != 1:
        raise RuntimeError("unexpected fixed-path runner campaign identifier")
    fixed_runner.write_text(text.replace(old, new))

    for subdir in ("fixed_r01", "endpoints_r01"):
        source_dir = SOURCE / "inputs" / subdir
        target_dir = ROOT / "inputs" / subdir
        for path in source_dir.glob("*.extxyz"):
            shutil.copy2(path, target_dir / path.name)

    source_rows = list(csv.DictReader((SOURCE / "jobs_r01" / "fixed_jobs.tsv").open(), delimiter="\t"))
    path_metadata = {}
    for row in source_rows:
        path_metadata.setdefault(row["path_id"], row)
    if set(path_metadata) != {path[0] for path in PATHS}:
        raise RuntimeError(f"source path table does not cover canonical paths: {sorted(path_metadata)}")

    model_rows = []
    for seed in SEEDS:
        model_id = f"scratch_600k_lmax1_grouped_repair_r01_seed{seed}"
        model_path = MODEL_ROOT / model_id / f"{model_id}_stagetwo_compiled.model"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        model_rows.append(
            {
                "model_id": model_id,
                "model_family": "Scratch-600K",
                "model_label": f"Scratch-600K / seed {seed}",
                "training_seed": seed,
                "seed_semantics": "registered_training_seed",
                "model_path": str(model_path),
                "model_sha256": sha256(model_path),
                "model_head": "default",
                "model_lineage_status": "CANONICAL_SCRATCH_LMAX1",
            }
        )
    write_tsv(ROOT / "MODEL_REGISTRY.tsv", model_rows)

    path_rows = []
    for path_id, path_class, short in PATHS:
        source = path_metadata[path_id]
        fixed = ROOT / "inputs" / "fixed_r01" / f"{path_id}.extxyz"
        initial = ROOT / "inputs" / "endpoints_r01" / f"{path_id}__initial.extxyz"
        final = ROOT / "inputs" / "endpoints_r01" / f"{path_id}__final.extxyz"
        path_rows.append(
            {
                "path_id": path_id,
                "short_path_id": short,
                "path_class": path_class,
                "atoms": 81,
                "image_count": 10,
                "fixed_path_file": f"inputs/fixed_r01/{path_id}.extxyz",
                "fixed_path_sha256": sha256(fixed),
                "initial_endpoint_file": f"inputs/endpoints_r01/{path_id}__initial.extxyz",
                "final_endpoint_file": f"inputs/endpoints_r01/{path_id}__final.extxyz",
                "initial_endpoint_sha256": sha256(initial),
                "final_endpoint_sha256": sha256(final),
                "cell_sha256": source["cell_sha256"],
                "dft_reference_id": path_id,
                "dft_reference_status": "COMPLETED_BUT_SCF_WARNING",
            }
        )
    write_tsv(ROOT / "PATH_REGISTRY.tsv", path_rows)

    fixed_rows = []
    self_rows = []
    record_index = 1
    for path_id, path_class, _ in PATHS:
        metadata = next(row for row in path_rows if row["path_id"] == path_id)
        for seed in SEEDS:
            model = next(row for row in model_rows if row["training_seed"] == seed)
            common = {
                "scope_tier": "T1_A1_CORE",
                "system_id": "CR81_CANONICAL",
                "system_family": "Cr@Sb2Te3_2L",
                "atoms": 81,
                "path_id": path_id,
                "path_class": path_class,
                "image_count": 10,
                "model_id": model["model_id"],
                "model_family": "Scratch-600K",
                "model_label": model["model_label"],
                "training_seed": seed,
                "seed_semantics": "registered_training_seed",
                "neb_seed": 42,
                "perturbation_seed": "NA",
                "model_path": model["model_path"],
                "model_sha256": model["model_sha256"],
                "model_head": "default",
                "model_lineage_status": "CANONICAL_SCRATCH_LMAX1",
                "initial_endpoint_file": metadata["initial_endpoint_file"],
                "final_endpoint_file": metadata["final_endpoint_file"],
                "initial_endpoint_sha256": metadata["initial_endpoint_sha256"],
                "final_endpoint_sha256": metadata["final_endpoint_sha256"],
                "cell_sha256": metadata["cell_sha256"],
                "fixed_path_file": metadata["fixed_path_file"],
                "fixed_path_sha256": metadata["fixed_path_sha256"],
                "dft_reference_id": metadata["dft_reference_id"],
                "dft_reference_status": metadata["dft_reference_status"],
            }
            fixed_rows.append(
                {
                    "record_id": f"SCRATCH600K-20261101-F-{record_index:03d}",
                    "observable": "FIXED_PATH_SINGLE_POINT",
                    "initialization_method": "DFT_RETAINED_PATH",
                    "seed_semantics": common["seed_semantics"],
                    **common,
                }
            )
            self_rows.append(
                {
                    "record_id": f"SCRATCH600K-20261101-S-{record_index:03d}",
                    "observable": "SELF_RELAXED_ENDPOINT_ONLY_NEB",
                    "initialization_method": "IDPP",
                    "initialization_source": "ENDPOINTS_ONLY",
                    "endpoint_policy": "FIXED_ENDPOINTS",
                    **common,
                }
            )
            record_index += 1
    write_tsv(ROOT / "jobs_r01" / "fixed_jobs.tsv", fixed_rows)
    write_tsv(ROOT / "jobs_r01" / "idpp_jobs.tsv", self_rows)

    manifest = {
        "schema_version": "migrationbench_scratch_600k_migration_v1",
        "campaign": CAMPAIGN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "source_campaign": str(SOURCE),
        "scope": {
            "model_family": "Scratch-600K",
            "training_seeds": list(SEEDS),
            "paths": [path_id for path_id, _, _ in PATHS],
            "fixed_jobs": len(fixed_rows),
            "self_relaxed_idpp_jobs": len(self_rows),
            "atoms": 81,
            "images": 10,
        },
        "models": model_rows,
        "paths": path_rows,
        "protocol": {
            "fixed": "single-point MACE energies/forces on retained canonical 10-image DFT path",
            "self_relaxed": "endpoint-only IDPP interpolation, FIRE, 500 steps, fmax 0.05 eV/A",
            "neb_seed": 42,
            "device": "cpu",
            "migrant_element": "Cr",
        },
        "scientific_boundary": "DFT reference carries COMPLETED_BUT_SCF_WARNING; fixed scores and self-relaxed endpoint-only barriers remain separate.",
        "status": "SUBMITTED_PENDING_SCIENTIFIC_GATES",
        "code_sha256": {name: sha256(ROOT / "code" / name) for name in CODE_FILES},
    }
    (ROOT / "CAMPAIGN_MANIFEST_r01.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (ROOT / "submit_fixed_array.slurm").write_text(slurm_script("fixed", len(fixed_rows)) + "\n")
    (ROOT / "submit_idpp_array.slurm").write_text(slurm_script("idpp", len(self_rows)) + "\n")
    (ROOT / "submit_model_load_smoke.slurm").write_text(slurm_script("smoke", len(model_rows)) + "\n")
    print(json.dumps({"root": str(ROOT), "fixed_jobs": len(fixed_rows), "idpp_jobs": len(self_rows), "models": model_rows, "paths": path_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
