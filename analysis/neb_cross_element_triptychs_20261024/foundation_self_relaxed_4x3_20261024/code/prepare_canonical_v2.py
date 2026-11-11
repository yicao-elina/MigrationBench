#!/usr/bin/env python3
"""Create a hash-bound canonical endpoint layer and isolated R02 campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
PARENT = BASE.parent
V2 = PARENT / "foundation_self_relaxed_4x3_R02_20261024"
REPO_ROOT = BASE.parents[2]
OLD_NEB_ROOT = REPO_ROOT / "analysis/task3_endpoint_only_neb/neb_fixed_vs_self_matrix_20261019"
REMOTE_ROOT = "/scratch16/pclancy3/yi/foundation_self_relaxed_4x3_20261024_r02"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_frame(path: Path) -> tuple[str, list[str]]:
    lines = path.read_text().splitlines()
    natoms = int(lines[0])
    if len(lines) != natoms + 2:
        raise ValueError(f"{path}: expected one extxyz frame, got {len(lines)} lines")
    return lines[1], lines[2:]


def write_canonical(path: Path, header: str, atoms: list[str]) -> None:
    payload = (str(len(atoms)) + "\n" + header + "\n" + "\n".join(atoms) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    if V2.exists() and any(V2.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty R02 root: {V2}")
    (V2 / "inputs/endpoints").mkdir(parents=True, exist_ok=True)
    (V2 / "code").mkdir(parents=True, exist_ok=True)
    (V2 / "data").mkdir(parents=True, exist_ok=True)
    (V2 / "figures").mkdir(parents=True, exist_ok=True)
    (V2 / "review").mkdir(parents=True, exist_ok=True)
    (V2 / "data_review").mkdir(parents=True, exist_ok=True)
    (V2 / "visual_review").mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASE / "ACCEPTANCE_CRITERIA.md", V2 / "ACCEPTANCE_CRITERIA.md")
    shutil.copy2(BASE / "CAPTION_FOUNDATION_SELF_RELAXED_4x3_R01.md", V2 / "CAPTION_FOUNDATION_SELF_RELAXED_4x3_R02.md")
    old_paths = read_tsv(BASE / "PATH_REGISTRY.tsv")
    new_paths: list[dict] = []
    path_lookup = {}
    for path_row in old_paths:
        initial_source = BASE / path_row["initial_endpoint_file"]
        final_source = BASE / path_row["final_endpoint_file"]
        initial_header, initial_atoms = read_frame(initial_source)
        final_header, final_atoms = read_frame(final_source)
        element = path_row["element"]
        path_id = path_row["path_id"]
        canonical_id = "foundation_self_relaxed_4x3_20261024_r02"
        entries = []
        for role, source_header, atoms, source_frame_index in (
            ("initial", initial_header, initial_atoms, "0"),
            ("final", final_header, final_atoms, "9"),
        ):
            geometry_hash = sha256_bytes(("\n".join(atoms) + "\n").encode())
            lattice = source_header.split('Lattice="', 1)[1].split('"', 1)[0]
            pbc = path_row["pbc"]
            if path_row["element"] == "Cr":
                source_file = OLD_NEB_ROOT / "inputs/endpoints_r01" / f"cr_canonical_{path_row['path_class']}__{role}.extxyz"
                source_file_sha256 = sha256_file(source_file)
                source_frame_index = "0"
            else:
                source_file = Path(path_row["source_linear_file"])
                source_file_sha256 = path_row["source_linear_sha256"]
            header = (
                f'Lattice="{lattice}" Properties=species:S:1:pos:R:3 '
                f'campaign_id={canonical_id} path_id={path_id} element={element} role={role} '
                f'source_kind={path_row["source_kind"]} source_path={source_file} '
                f'source_sha256={source_file_sha256} source_frame_index={source_frame_index} '
                f'legacy_source_stem={path_row["legacy_source_stem"]} endpoint_only_input=T '
                f'endpoint_geometry_sha256={geometry_hash} cell_sha256={path_row["cell_sha256"]} pbc="{pbc}"'
            )
            destination = V2 / "inputs/endpoints" / f"{path_id}__{role}.extxyz"
            write_canonical(destination, header, atoms)
            entries.append((role, destination, sha256_file(destination), geometry_hash, header))
        new_row = dict(path_row)
        new_row.update(
            {
                "campaign_id": canonical_id,
                "header_schema": "canonical_endpoint_v2",
                "initial_endpoint_file": str(entries[0][1].relative_to(V2)),
                "final_endpoint_file": str(entries[1][1].relative_to(V2)),
                "initial_endpoint_sha256": entries[0][2],
                "final_endpoint_sha256": entries[1][2],
                "initial_endpoint_geometry_sha256": entries[0][3],
                "final_endpoint_geometry_sha256": entries[1][3],
                "endpoint_header_campaign_id": canonical_id,
                "endpoint_header_path_id": path_id,
                "endpoint_header_roles": "initial,final",
                "endpoint_header_endpoint_only_input": "T",
            }
        )
        new_paths.append(new_row)
        path_lookup[path_id] = new_row
    write_tsv(V2 / "PATH_REGISTRY.tsv", new_paths)

    models = read_tsv(BASE / "MODEL_REGISTRY.tsv")
    write_tsv(V2 / "MODEL_REGISTRY.tsv", models)
    old_jobs = read_tsv(BASE / "JOB_TABLE_IDPP.tsv")
    new_jobs: list[dict] = []
    for index, old_job in enumerate(old_jobs, start=1):
        path = path_lookup[old_job["path_id"]]
        row = dict(old_job)
        row.update(
            {
                "record_id": f"FSELF-R02-20261024-{index:06d}",
                "campaign_id": canonical_id,
                "execution_status": "PLANNED",
                "run_action": "SUBMIT_CANONICAL_FORMAL_ARRAY",
                "initial_endpoint_file": path["initial_endpoint_file"],
                "final_endpoint_file": path["final_endpoint_file"],
                "initial_endpoint_sha256": path["initial_endpoint_sha256"],
                "final_endpoint_sha256": path["final_endpoint_sha256"],
                "execution_role": "FORMAL_ARRAY",
                "manifest_role": "FORMAL_LAUNCH",
                "remote_root": REMOTE_ROOT,
            }
        )
        new_jobs.append(row)
    write_tsv(V2 / "JOB_TABLE_IDPP.tsv", new_jobs)

    runtime_files = [
        "run_mlff_neb.py",
        "export_mlff_neb_iteration_history.py",
        "capture_runtime_provenance.py",
        "run_one_job.py",
        "collect_results.py",
        "build_class_profiles.py",
        "render_foundation_self_relaxed_4x3.py",
    ]
    for name in runtime_files:
        shutil.copy2(BASE / "code" / name, V2 / "code" / name)
    array_script = (BASE / "code/run_foundation_self_array.slurm").read_text()
    array_script = array_script.replace("foundation_self_relaxed_4x3_20261024", "foundation_self_relaxed_4x3_20261024_r02")
    (V2 / "code/run_foundation_self_array.slurm").write_text(array_script)

    manifest = {
        "schema_version": "foundation_self_relaxed_4x3_campaign_v2",
        "campaign_id": canonical_id,
        "status": "READY_FOR_FORMAL_ARRAY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "acceptance_criteria": "ACCEPTANCE_CRITERIA.md",
        "acceptance_criteria_sha256": sha256_file(V2 / "ACCEPTANCE_CRITERIA.md"),
        "attached_document_role": "visual_reference_only",
        "endpoint_layer": "canonical_endpoint_v2_header_bound",
        "canonical_elements": ["Cr", "Cu", "Li", "Na"],
        "canonical_path_classes": ["in_gap", "gap_to_ql", "within_ql"],
        "model_count": 9,
        "path_count": 12,
        "target_job_count": 108,
        "target_shape": "4 elements x 3 path classes x 9 Foundation models = 108",
        "initialization": {"method": "IDPP", "endpoint_only": True, "n_images": 10, "seed": 42, "fmax_eV_A": 0.05, "steps_max": 500},
        "execution_environment": {"environment": "automatbench", "python": "/data/pclancy3/yi/.conda/envs/automatbench/bin/python", "mace_version_expected": "0.3.16"},
        "remote_root": REMOTE_ROOT,
        "formal_launch_manifest_immutable": True,
        "formal_launch_job_id": None,
        "prior_precannonical_campaign": "../foundation_self_relaxed_4x3_20261024",
        "prior_outputs_reused": False,
        "fixed_path_data_excluded": True,
        "blender_excluded": True,
        "registries": {"model": "MODEL_REGISTRY.tsv", "path": "PATH_REGISTRY.tsv", "jobs": "JOB_TABLE_IDPP.tsv"},
    }
    (V2 / "CAMPAIGN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"root": str(V2), "paths": len(new_paths), "models": len(models), "jobs": len(new_jobs), "manifest_sha256": sha256_file(V2 / "CAMPAIGN_MANIFEST.json")}, indent=2))


if __name__ == "__main__":
    main()

