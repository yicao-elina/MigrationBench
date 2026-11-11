#!/usr/bin/env python3
"""Prepare canonical endpoint-only inputs and the 108-row Foundation target matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


CAMPAIGN_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CAMPAIGN_ROOT.parents[2]
OLD_NEB_ROOT = REPO_ROOT / "analysis/task3_endpoint_only_neb/neb_fixed_vs_self_matrix_20261019"
INTERP_ROOT = REPO_ROOT / "analysis/task3_endpoint_only_neb/interpolations"

REMOTE_ROOT = "/scratch16/pclancy3/yi/foundation_self_relaxed_4x3_20261024"
REMOTE_PYTHON = "/data/pclancy3/yi/.conda/envs/automatbench/bin/python"
MODEL_ROOT = "/scratch16/pclancy3/yi/foundation_grouped_benchmark_20261001/models"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    try:
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(fd)
    return digest.hexdigest()


def parse_frames(path: Path) -> list[tuple[str, list[str]]]:
    # The project may sit on a mixed local/cloud-backed volume; byte reads are
    # deterministic here and avoid an intermittent text-stream stall observed
    # on the small canonical endpoint files.
    fd = os.open(path, os.O_RDONLY)
    try:
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(fd)
    lines = b"".join(chunks).decode().splitlines()
    frames: list[tuple[str, list[str]]] = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        try:
            natoms = int(lines[cursor].strip())
        except ValueError as exc:
            raise ValueError(f"{path}: invalid frame atom count at line {cursor + 1}") from exc
        if cursor + 1 + natoms >= len(lines):
            raise ValueError(f"{path}: truncated frame at line {cursor + 1}")
        header = lines[cursor + 1]
        atoms = lines[cursor + 2 : cursor + 2 + natoms]
        if len(atoms) != natoms:
            raise ValueError(f"{path}: expected {natoms} atoms, got {len(atoms)}")
        frames.append((header, atoms))
        cursor += 2 + natoms
    return frames


def frame_text(frame: tuple[str, list[str]]) -> bytes:
    header, atoms = frame
    return (str(len(atoms)) + "\n" + header + "\n" + "\n".join(atoms) + "\n").encode()


def lattice_and_pbc(header: str) -> tuple[str, str]:
    lattice = re.search(r'Lattice="([^"]+)"', header)
    pbc = re.search(r'pbc="([^"]+)"', header)
    if not lattice or not pbc:
        raise ValueError(f"missing Lattice/pbc in extxyz header: {header}")
    return lattice.group(1), pbc.group(1)


def cell_hash(header: str) -> str:
    lattice, pbc = lattice_and_pbc(header)
    return sha256_bytes(f"{lattice}|{pbc}".encode())


def symbols(frame: tuple[str, list[str]]) -> list[str]:
    return [line.split()[0] for line in frame[1]]


def write_frame(path: Path, frame: tuple[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != frame_text(frame):
        raise RuntimeError(f"refusing to overwrite a different endpoint: {path}")
    path.write_bytes(frame_text(frame))


def rows_to_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


MODELS = [
    {
        "model_id": "foundation_agnesi_large",
        "model_label": "MACE-MP · large",
        "model_family": "MACE-MP",
        "model_path": f"{MODEL_ROOT}/mace-large-density-agnesi-stress.model",
        "model_file": "mace-large-density-agnesi-stress.model",
        "model_sha256": "348390e758e1c90011c7675e864850f8e9c5b3e7c217f79a2b4bad1baaa657ad",
        "model_head": "default",
    },
    {
        "model_id": "foundation_agnesi_medium_density",
        "model_label": "MACE-MP · medium-density",
        "model_family": "MACE-MP",
        "model_path": f"{MODEL_ROOT}/mace-medium-density-agnesi-stress.model",
        "model_file": "mace-medium-density-agnesi-stress.model",
        "model_sha256": "a90be07c8aa6623c390fcc4653d3e319c4a356f8b4a53d323f96b2012d375caf",
        "model_head": "default",
    },
    {
        "model_id": "foundation_agnesi_small_density",
        "model_label": "MACE-MP · small-density",
        "model_family": "MACE-MP",
        "model_path": f"{MODEL_ROOT}/mace-small-density-agnesi-stress.model",
        "model_file": "mace-small-density-agnesi-stress.model",
        "model_sha256": "d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9",
        "model_head": "default",
    },
    {
        "model_id": "foundation_agnesi_medium",
        "model_label": "MACE-MP · medium",
        "model_family": "MACE-MP",
        "model_path": f"{MODEL_ROOT}/mace_agnesi_medium.model",
        "model_file": "mace_agnesi_medium.model",
        "model_sha256": "ab8baff639a8f295f3eccad3d3ccf574efb6fb63220bd52cd88664211569e521",
        "model_head": "default",
    },
    {
        "model_id": "foundation_mp0b3",
        "model_label": "MACE-MP · 0b3-medium",
        "model_family": "MACE-MP",
        "model_path": f"{MODEL_ROOT}/mace-mp-0b3-medium.model",
        "model_file": "mace-mp-0b3-medium.model",
        "model_sha256": "2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012",
        "model_head": "default",
    },
    {
        "model_id": "foundation_omat0_medium",
        "model_label": "MACE-OMAT-0 · medium",
        "model_family": "MACE-OMAT-0",
        "model_path": f"{MODEL_ROOT}/mace-omat-0-medium.model",
        "model_file": "mace-omat-0-medium.model",
        "model_sha256": "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a",
        "model_head": "default",
    },
    {
        "model_id": "foundation_omat0_small",
        "model_label": "MACE-OMAT-0 · small",
        "model_family": "MACE-OMAT-0",
        "model_path": f"{MODEL_ROOT}/mace-omat-0-small.model",
        "model_file": "mace-omat-0-small.model",
        "model_sha256": "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5",
        "model_head": "default",
    },
    {
        "model_id": "foundation_mh0",
        "model_label": "MACE-MH · 0",
        "model_family": "MACE-MH",
        "model_path": f"{MODEL_ROOT}/mace-mh-0.model",
        "model_file": "mace-mh-0.model",
        "model_sha256": "d62ff8f293664e6556cfa49364b28ee72a70cf1e6150f3c111a578397fed609d",
        "model_head": "omat_pbe",
    },
    {
        "model_id": "foundation_mh1",
        "model_label": "MACE-MH · 1",
        "model_family": "MACE-MH",
        "model_path": f"{MODEL_ROOT}/mace-mh-1.model",
        "model_file": "mace-mh-1.model",
        "model_sha256": "a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47",
        "model_head": "omat_pbe",
    },
]


PATH_SPECS = [
    ("in_gap", "in-gap diffusion", "gap-diffuse"),
    ("gap_to_ql", "gap→QL", "gap2QL"),
    ("within_ql", "within-QL", "QLs"),
]


def prepare_paths() -> list[dict]:
    endpoint_root = CAMPAIGN_ROOT / "inputs/endpoints"
    rows: list[dict] = []
    for element in ("Cr", "Cu", "Li", "Na"):
        for path_class, path_label, stem_suffix in PATH_SPECS:
            if element == "Cr":
                source_stem = f"2L-Cu__neb-{stem_suffix}"
                old_prefix = OLD_NEB_ROOT / "inputs/endpoints_r01" / f"cr_canonical_{path_class}"
                initial = parse_frames(old_prefix.with_name(old_prefix.name + "__initial.extxyz"))
                final = parse_frames(old_prefix.with_name(old_prefix.name + "__final.extxyz"))
                if len(initial) != 1 or len(final) != 1:
                    raise ValueError(f"Cr canonical endpoint files must have one frame: {old_prefix}")
                source_file = old_prefix.with_name(old_prefix.name + "__initial.extxyz")
                source_kind = "CANONICAL_ENDPOINT_INPUT"
                source_frame_count = 1
                source_sha = sha256_file(source_file)
                legacy_element_hint = "Cr"
            else:
                source_stem = f"2L-{element}__neb-{stem_suffix}"
                source_file = INTERP_ROOT / source_stem / "linear.extxyz"
                frames = parse_frames(source_file)
                if len(frames) != 10:
                    raise ValueError(f"{source_file}: expected 10 source frames, got {len(frames)}")
                initial, final = [frames[0]], [frames[-1]]
                initial = initial[0]
                final = final[0]
                source_kind = "LEGACY_LINEAR_SOURCE_ENDPOINTS_ONLY"
                source_frame_count = len(frames)
                source_sha = sha256_file(source_file)
                legacy_element_hint = element

            initial_frame = initial[0] if isinstance(initial, list) else initial
            final_frame = final[0] if isinstance(final, list) else final
            for frame, name in ((initial_frame, "initial"), (final_frame, "final")):
                syms = symbols(frame)
                if len(syms) != 81 or syms.count(element) != 1:
                    raise ValueError(f"{element} {path_class} {name}: expected 81 atoms and one {element}; got {len(syms)} / {syms.count(element)}")
            if symbols(initial_frame) != symbols(final_frame):
                raise ValueError(f"{element} {path_class}: endpoint atom order/symbols differ")
            initial_path = endpoint_root / f"{element}__{path_class}__initial.extxyz"
            final_path = endpoint_root / f"{element}__{path_class}__final.extxyz"
            write_frame(initial_path, initial_frame)
            write_frame(final_path, final_frame)
            lattice, pbc = lattice_and_pbc(initial_frame[0])
            path_id = f"{element}__{path_class}"
            rows.append(
                {
                    "path_id": path_id,
                    "element": element,
                    "path_class": path_class,
                    "path_label": path_label,
                    "legacy_source_stem": source_stem,
                    "legacy_source_element_hint": legacy_element_hint,
                    "source_kind": source_kind,
                    "source_linear_file": str(source_file),
                    "source_linear_sha256": source_sha,
                    "source_frame_count": source_frame_count,
                    "intermediate_source_frames_used_by_runner": "NO",
                    "initial_endpoint_file": str(initial_path.relative_to(CAMPAIGN_ROOT)),
                    "final_endpoint_file": str(final_path.relative_to(CAMPAIGN_ROOT)),
                    "initial_endpoint_sha256": sha256_file(initial_path),
                    "final_endpoint_sha256": sha256_file(final_path),
                    "cell_sha256": cell_hash(initial_frame[0]),
                    "lattice": lattice,
                    "pbc": pbc,
                    "atoms": len(initial_frame[1]),
                }
            )
    return rows


def main() -> None:
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    path_rows = prepare_paths()
    rows_to_tsv(CAMPAIGN_ROOT / "PATH_REGISTRY.tsv", path_rows)
    rows_to_tsv(CAMPAIGN_ROOT / "MODEL_REGISTRY.tsv", MODELS)

    job_rows: list[dict] = []
    index = 0
    for element in ("Cr", "Cu", "Li", "Na"):
        for path in [row for row in path_rows if row["element"] == element]:
            for model in MODELS:
                index += 1
                job_rows.append(
                    {
                        "record_id": f"FSELF-20261024-{index:06d}",
                        "campaign_id": "foundation_self_relaxed_4x3_20261024",
                        "scope_tier": "FOUNDATION_SELF_RELAXED",
                        "observable": "SELF_RELAXED_ENDPOINT_ONLY_NEB",
                        "execution_status": "PLANNED",
                        "run_action": "SUBMIT_FULL_CAMPAIGN",
                        "element": element,
                        "atoms": path["atoms"],
                        "path_id": path["path_id"],
                        "path_class": path["path_class"],
                        "path_label": path["path_label"],
                        "legacy_source_stem": path["legacy_source_stem"],
                        "initial_endpoint_file": path["initial_endpoint_file"],
                        "final_endpoint_file": path["final_endpoint_file"],
                        "initial_endpoint_sha256": path["initial_endpoint_sha256"],
                        "final_endpoint_sha256": path["final_endpoint_sha256"],
                        "cell_sha256": path["cell_sha256"],
                        "model_id": model["model_id"],
                        "model_label": model["model_label"],
                        "model_family": model["model_family"],
                        "model_path": model["model_path"],
                        "model_sha256": model["model_sha256"],
                        "model_head": model["model_head"],
                        "initialization_method": "IDPP",
                        "initialization_source": "ENDPOINTS_ONLY",
                        "n_images": 10,
                        "neb_seed": 42,
                        "fmax_target_eV_A": 0.05,
                        "steps_max": 500,
                        "dft_reference_id": f"DIAGNOSTIC_REFERENCE_{element}_{path['path_class']}",
                        "dft_reference_status": "NOT_USED_SELF_PROFILE",
                        "remote_python": REMOTE_PYTHON,
                        "remote_root": REMOTE_ROOT,
                    }
                )
    rows_to_tsv(CAMPAIGN_ROOT / "JOB_TABLE_IDPP.tsv", job_rows)

    criteria = CAMPAIGN_ROOT / "ACCEPTANCE_CRITERIA.md"
    manifest = {
        "schema_version": "foundation_self_relaxed_4x3_campaign_v1",
        "campaign_id": "foundation_self_relaxed_4x3_20261024",
        "status": "PLANNED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "acceptance_criteria": str(criteria),
        "acceptance_criteria_sha256": sha256_file(criteria),
        "attached_document_role": "visual_reference_only",
        "canonical_elements": ["Cr", "Cu", "Li", "Na"],
        "canonical_path_classes": [row[0] for row in PATH_SPECS],
        "model_count": len(MODELS),
        "path_count": len(path_rows),
        "target_job_count": len(job_rows),
        "target_shape": "4 elements x 3 path classes x 9 Foundation models = 108",
        "initialization": {
            "method": "IDPP",
            "endpoint_only": True,
            "n_images": 10,
            "seed": 42,
            "fmax_eV_A": 0.05,
            "steps_max": 500,
        },
        "execution_environment": {
            "python": REMOTE_PYTHON,
            "environment": "automatbench",
            "selection_reason": "MACE-MH-1 loads under MACE 0.3.16 here; the older mace environment did not provide a valid load path",
        },
        "fixed_path_data_excluded": True,
        "blender_excluded": True,
        "prior_outputs_reused": False,
        "registries": {
            "model": "MODEL_REGISTRY.tsv",
            "path": "PATH_REGISTRY.tsv",
            "jobs": "JOB_TABLE_IDPP.tsv",
        },
    }
    (CAMPAIGN_ROOT / "CAMPAIGN_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"paths": len(path_rows), "models": len(MODELS), "jobs": len(job_rows), "root": str(CAMPAIGN_ROOT)}, indent=2))


if __name__ == "__main__":
    main()

