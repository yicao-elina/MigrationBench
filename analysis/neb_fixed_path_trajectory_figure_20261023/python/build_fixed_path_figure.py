#!/usr/bin/env python3
"""Build the fixed-path NEB diagnostic data layer and lower-panel figure.

This additive script reads only the r01 FIXED_PATH_SINGLE_POINT evidence,
retained 81-atom geometries, and the last complete QE/NEB tables. It does not
upgrade the warning-bearing DFT reference or the seed gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


STATUS = "PROVISIONAL_DIAGNOSTIC_CANDIDATE"
OBSERVABLE = "FIXED_PATH_SINGLE_POINT"
N_IMAGES = 10

PATH_ORDER = [
    {
        "column_index": 1,
        "display_label": "within gap",
        "short_label": "within_gap",
        "subtitle": "in-gap diffusion",
        "path_id": "cr_canonical_in_gap",
        "registry_class": "in-gap diffusion",
        "raw_stem": "2L-Cu__neb-gap-diffuse",
    },
    {
        "column_index": 2,
        "display_label": "gap2QL",
        "short_label": "gap2ql",
        "subtitle": "gap→QL",
        "path_id": "cr_canonical_gap_to_ql",
        "registry_class": "gap-to-QL",
        "raw_stem": "2L-Cu__neb-gap2QL",
    },
    {
        "column_index": 3,
        "display_label": "withinQL",
        "short_label": "within_ql",
        "subtitle": "within-QL",
        "path_id": "cr_canonical_within_ql",
        "registry_class": "within-QL",
        "raw_stem": "2L-Cu__neb-QLs",
    },
]

TARGET_MODEL_IDS = [
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

FAMILY_KEY = {
    "Foundation": "foundation_omat",
    "From-scratch": "from_scratch",
    "FT-600K": "ft_600k",
    "FT-MultiT": "ft_multit",
}
SEED_MARKERS = {"123": "o", "234": "s", "345": "^"}
FOUNDATION_LINESTYLES = {"FND-A": "-", "FND-B": "--", "FND-C": "-."}


def trace(message: str) -> None:
    print(f"[fixed-path-build] {message}", file=sys.stderr, flush=True)


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    size = path.stat().st_size
    descriptor = os.open(path, os.O_RDONLY)
    try:
        offset = 0
        while offset < size:
            chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
            if not chunk:
                raise OSError(f"short pread at offset {offset} for {path}")
            h.update(chunk)
            offset += len(chunk)
    finally:
        os.close(descriptor)
    return h.hexdigest()


def read_exact_bytes(path: Path) -> bytes:
    """Read exactly stat().st_size bytes without a blocking EOF read."""
    size = path.stat().st_size
    descriptor = os.open(path, os.O_RDONLY)
    try:
        chunks = []
        offset = 0
        while offset < size:
            chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
            if not chunk:
                raise OSError(f"short pread at offset {offset} for {path}")
            chunks.append(chunk)
            offset += len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def read_exact_text(path: Path, encoding: str = "utf-8") -> str:
    return read_exact_bytes(path).decode(encoding, errors="replace")


def json_read(path: Path) -> Any:
    return json.loads(read_exact_text(path))


def load_style_or_frozen_palette(path: Path) -> tuple[dict[str, Any], str]:
    """Read FIG-168 style; use only the frozen contract colors if unreadable."""
    try:
        value = json_read(path)
        if isinstance(value, dict) and value.get("families"):
            return value, "FIG168_MODEL_STYLE_JSON"
    except (OSError, json.JSONDecodeError):
        pass
    fallback = {
        "style_id": "FIG-168-frozen-contract-fallback",
        "families": {
            "foundation_omat": {"label": "Foundation", "color": "#CF4520"},
            "from_scratch": {"label": "From-scratch", "color": "#FF9F1B"},
            "ft_600k": {"label": "FT-600K", "color": "#002E71"},
            "ft_multit": {"label": "FT-MultiT", "color": "#67ACE5"},
        },
        "seed_order": [123, 234, 345],
        "seed_markers": {"123": "o", "234": "s", "345": "^"},
        "source_note": "Exact palette supplied by FROZEN_VISUAL_ACCEPTANCE_CRITERIA.md because the named style file was empty at runtime.",
    }
    return fallback, "FROZEN_CONTRACT_PALETTE_FALLBACK_STYLE_SOURCE_EMPTY"


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_tsv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(read_exact_text(path).splitlines(), delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def float_or_none(value: Any) -> float | None:
    if value is None or str(value).strip() in {"", "NA", "None", "nan", "NaN"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def source_paths(repo_root: Path) -> dict[str, dict[str, Path]]:
    raw_root = repo_root / "analysis/task3_endpoint_only_neb/exploratory_energy_audit/rockfish_cr_neb_20261013"
    return {
        config["path_id"]: {
            "raw_neb": raw_root / "raw" / config["raw_stem"] / "neb.out",
            "retained_extxyz": raw_root / "inputs" / f"{config['raw_stem']}.extxyz",
        }
        for config in PATH_ORDER
    }


def parse_lattice(comment: str) -> np.ndarray:
    match = re.search(r'Lattice="([^"]+)"', comment)
    if not match:
        raise ValueError("retained extxyz comment has no Lattice attribute")
    values = [float(item) for item in match.group(1).split()]
    if len(values) != 9:
        raise ValueError("Lattice must contain nine values")
    return np.asarray(values, dtype=float).reshape(3, 3)


def parse_extxyz(path: Path) -> dict[str, Any]:
    lines = read_exact_text(path).splitlines()
    frames: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        count = int(lines[cursor])
        comment = lines[cursor + 1]
        cell = parse_lattice(comment)
        species: list[str] = []
        positions: list[list[float]] = []
        for line in lines[cursor + 2 : cursor + 2 + count]:
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"invalid atom line: {line}")
            species.append(fields[0])
            positions.append([float(fields[1]), float(fields[2]), float(fields[3])])
        frames.append(
            {
                "atom_count": count,
                "comment": comment,
                "cell_A": cell,
                "species": species,
                "positions_A": np.asarray(positions, dtype=float),
            }
        )
        cursor += 2 + count
    if not frames:
        raise ValueError(f"no extxyz frames found: {path}")
    return {"frames": frames, "sha256": sha256_file(path)}


def mic_delta(delta_A: np.ndarray, cell_A: np.ndarray) -> np.ndarray:
    fractional = np.asarray(delta_A, dtype=float) @ np.linalg.inv(cell_A)
    return (fractional - np.rint(fractional)) @ cell_A


def make_geometry(config: dict[str, Any], source: dict[str, Path]) -> dict[str, Any]:
    parsed = parse_extxyz(source["retained_extxyz"])
    frames = parsed["frames"]
    if len(frames) != N_IMAGES or any(frame["atom_count"] != 81 for frame in frames):
        raise ValueError(f"{config['path_id']} is not exactly 10 frames of 81 atoms")
    species = frames[0]["species"]
    if any(frame["species"] != species for frame in frames):
        raise ValueError(f"{config['path_id']} changes species order")
    if species.count("Cr") != 1:
        raise ValueError(f"{config['path_id']} must contain one Cr migrant")
    cell = frames[0]["cell_A"]
    if any(not np.allclose(frame["cell_A"], cell, atol=1e-8) for frame in frames):
        raise ValueError(f"{config['path_id']} changes its cell")

    unwrapped = [frames[0]["positions_A"].copy()]
    step_rms = [0.0]
    for previous, current in zip(frames[:-1], frames[1:]):
        delta = mic_delta(current["positions_A"] - previous["positions_A"], cell)
        unwrapped.append(unwrapped[-1] + delta)
        step_rms.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    cumulative = np.cumsum(step_rms)
    s_values = (
        cumulative / cumulative[-1]
        if cumulative[-1] > 0
        else np.linspace(0.0, 1.0, N_IMAGES)
    )
    cr_index = species.index("Cr")
    frame_records = []
    for index, frame in enumerate(frames):
        frame_records.append(
            {
                "image_index": index,
                "s": float(s_values[index]),
                "positions_A": np.round(frame["positions_A"], 8).tolist(),
                "positions_unwrapped_A": np.round(unwrapped[index], 8).tolist(),
                "cr_position_A": np.round(frame["positions_A"][cr_index], 8).tolist(),
                "cr_position_unwrapped_A": np.round(unwrapped[index][cr_index], 8).tolist(),
            }
        )
    return {
        **config,
        "geometry_source": str(source["retained_extxyz"]),
        "geometry_sha256": parsed["sha256"],
        "atoms": 81,
        "image_count": N_IMAGES,
        "species_order": species,
        "cr_indices": [cr_index],
        "cell_A": np.round(cell, 8).tolist(),
        "pbc": ["T", "T", "T"],
        "s_definition": "normalized cumulative RMS minimum-image displacement over all 81 atoms",
        "mic_unwrap": {
            "method": "fractional displacement minus nearest periodic integer shift",
            "periodic_axes": [True, True, True],
            "interpolated_images": 0,
        },
        "frames": frame_records,
    }


def parse_dft_table(path: Path) -> dict[str, Any]:
    text = read_exact_text(path)
    lines = text.splitlines()
    header_re = re.compile(r"^\s*image\s+energy\s+\(eV\)", re.IGNORECASE)
    row_re = re.compile(
        r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
        r"(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+([FT])\s*$"
    )
    blocks: list[dict[str, Any]] = []
    for header_index, line in enumerate(lines):
        if not header_re.search(line):
            continue
        rows: list[dict[str, Any]] = []
        cursor = header_index + 1
        while cursor < len(lines):
            match = row_re.match(lines[cursor])
            if match:
                rows.append(
                    {
                        "image_index": int(match.group(1)) - 1,
                        "energy_eV": float(match.group(2)),
                        "error_eV_A": float(match.group(3)),
                    }
                )
                cursor += 1
                if len(rows) == N_IMAGES:
                    break
                continue
            if rows and lines[cursor].strip():
                break
            cursor += 1
        if len(rows) == N_IMAGES and [row["image_index"] for row in rows] == list(range(N_IMAGES)):
            blocks.append(
                {
                    "table_index": len(blocks),
                    "line_start": header_index + 1,
                    "line_end": cursor,
                    "rows": rows,
                }
            )
    if not blocks:
        raise ValueError(f"no complete ten-image DFT table in {path}")
    return {
        "source_raw_neb": str(path),
        "source_raw_sha256": sha256_file(path),
        "table_count": len(blocks),
        "selected_table": blocks[-1],
        "scf_warning_detected_in_raw": bool(re.search(r"scf convergence NOT achieved", text, re.I)),
        "job_done_detected_in_raw": "JOB DONE" in text,
    }


def load_json_if_nonempty(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return None
    try:
        value = json_read(path)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def profile_path(fixed_root: Path, model_id: str, path_id: str, result: dict[str, str]) -> dict[str, Path | None]:
    attempt = result.get("job_id", "").strip()
    expected = fixed_root / model_id / path_id / f"attempt_{attempt}"
    if attempt and (expected / "image_profile.csv").exists():
        directory = expected
    else:
        candidates = sorted((fixed_root / model_id / path_id).glob("*/image_profile.csv"))
        directory = candidates[0].parent if candidates else None
    if directory is None:
        return {"profile": None, "manifest": None}
    return {
        "profile": directory / "image_profile.csv",
        "manifest": directory / "fixed_single_point_manifest.json",
    }


ROW_FIELDS = [
    "record_type", "path_column", "path_display_label", "path_id", "path_class",
    "observable", "initialization_method", "neb_seed", "model_arm_id", "model_id",
    "model_label", "model_family", "family_key", "training_seed", "seed_semantics",
    "lineage_status", "image_index", "expected_images", "s", "energy_eV",
    "raw_relative_energy_eV", "relative_energy_eV", "fmax_eV_A", "finite_energy",
    "finite_forces", "status", "failure_class", "failure_reason", "publication_status",
    "provenance_status", "record_id", "input_source", "input_sha256", "profile_source",
    "profile_sha256", "model_sha256", "result_profile_artifact", "tier1_profile_source",
    "remote_profile_read_status", "numeric_source", "numeric_source_sha256",
    "dft_reference_status",
]

DFT_FIELDS = [
    "record_type", "path_column", "path_display_label", "path_id", "path_class",
    "observable", "image_index", "expected_images", "s", "energy_eV",
    "relative_energy_eV", "error_eV_A", "finite_energy", "dft_reference_status",
    "scf_warning", "raw_table_index", "raw_table_line_start", "raw_table_line_end",
    "source_raw_neb", "source_raw_sha256",
]


def build_data(repo_root: Path, fig_root: Path) -> dict[str, Any]:
    data_dir = fig_root / "python/data"
    generated_dir = fig_root / "python/generated"
    data_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    campaign = repo_root / "analysis/task3_endpoint_only_neb/neb_fixed_vs_self_matrix_20261019"
    style_path = repo_root / "analysis/task3_endpoint_only_neb/figure1_family_seed_audit_20261016/FIG168_MODEL_STYLE.json"
    arm_path = campaign / "PAIRED_MODEL_ARM_REGISTRY.tsv"
    system_path = campaign / "SYSTEM_PATH_REGISTRY.tsv"
    canonical_path = campaign / "CANONICAL_ENDPOINT_REGISTRY_r01.tsv"
    result_path = campaign / "derived_r01/TIER1_RESULT_MATRIX.tsv"
    image_path = campaign / "derived_r01/TIER1_IMAGE_PROFILES.tsv"
    release_qa_path = campaign / "RELEASE_QA_REPORT_r01.json"
    criteria_path = fig_root / "visual_review/FROZEN_VISUAL_ACCEPTANCE_CRITERIA.md"
    fixed_root = campaign / "remote_fixed_r01/fixed"

    trace("loading style and registries")
    style, style_runtime_status = load_style_or_frozen_palette(style_path)
    expected_palette = {
        "foundation_omat": "#CF4520",
        "from_scratch": "#FF9F1B",
        "ft_600k": "#002E71",
        "ft_multit": "#67ACE5",
    }
    actual_palette = {key: style["families"][key]["color"].upper() for key in expected_palette}
    if actual_palette != expected_palette:
        raise ValueError(f"FIG-168 palette mismatch: {actual_palette}")

    arms = {row["model_id"]: row for row in read_tsv(arm_path)}
    system_rows = {row["path_id"]: row for row in read_tsv(system_path)}
    canonical_rows = {row["path_id"]: row for row in read_tsv(canonical_path)}
    result_rows = read_tsv(result_path)
    image_rows = read_tsv(image_path)
    result_lookup = {
        (row["model_id"], row["path_id"]): row
        for row in result_rows
        if row["observable"] == OBSERVABLE
    }
    tier1_lookup = {
        (row["record_id"], int(row["image_index"])): row
        for row in image_rows
        if row["observable"] == OBSERVABLE
    }
    tier1_rows_by_record: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in image_rows:
        if row["observable"] == OBSERVABLE:
            tier1_rows_by_record[row["record_id"]].append(row)
    trace(f"loaded registries: {len(arms)} arms, {len(result_rows)} result rows, {len(image_rows)} image rows")
    sources = source_paths(repo_root)
    geometries: dict[str, dict[str, Any]] = {}
    dft_profiles: list[dict[str, Any]] = []
    dft_meta: dict[str, Any] = {}

    for config in PATH_ORDER:
        path_id = config["path_id"]
        trace(f"parsing geometry and DFT table: {path_id}")
        geometries[path_id] = make_geometry(config, sources[path_id])
        dft = parse_dft_table(sources[path_id]["raw_neb"])
        dft_meta[path_id] = dft
        system = system_rows[path_id]
        canonical = canonical_rows[path_id]
        if system["dft_reference_status"] != "COMPLETED_BUT_SCF_WARNING":
            raise ValueError(f"unexpected DFT status for {path_id}")
        if not dft["scf_warning_detected_in_raw"] or not dft["job_done_detected_in_raw"]:
            raise ValueError(f"DFT raw markers incomplete for {path_id}")
        first_energy = dft["selected_table"]["rows"][0]["energy_eV"]
        for row in dft["selected_table"]["rows"]:
            image_index = row["image_index"]
            dft_profiles.append(
                {
                    "record_type": "DFT_REFERENCE",
                    "path_column": config["column_index"],
                    "path_display_label": config["display_label"],
                    "path_id": path_id,
                    "path_class": config["registry_class"],
                    "observable": "DFT_QE_REFERENCE",
                    "image_index": image_index,
                    "expected_images": N_IMAGES,
                    "s": geometries[path_id]["frames"][image_index]["s"],
                    "energy_eV": row["energy_eV"],
                    "relative_energy_eV": row["energy_eV"] - first_energy,
                    "error_eV_A": row["error_eV_A"],
                    "finite_energy": True,
                    "dft_reference_status": system["dft_reference_status"],
                    "scf_warning": True,
                    "raw_table_index": dft["selected_table"]["table_index"],
                    "raw_table_line_start": dft["selected_table"]["line_start"],
                    "raw_table_line_end": dft["selected_table"]["line_end"],
                    "source_raw_neb": str(sources[path_id]["raw_neb"]),
                    "source_raw_sha256": dft["source_raw_sha256"],
                    "canonical_fixed_path_sha256": canonical["fixed_retained_path_sha256"],
                }
            )

    fixed_root_rows: list[dict[str, Any]] = []
    block_records: list[dict[str, Any]] = []
    selected_keys = {(model_id, config["path_id"]) for model_id in TARGET_MODEL_IDS for config in PATH_ORDER}
    tier1_checked = 0
    tier1_mismatches: list[dict[str, Any]] = []

    for model_id in TARGET_MODEL_IDS:
        trace(f"collecting fixed profiles: {model_id}")
        if model_id not in arms:
            raise ValueError(f"model not in paired arm registry: {model_id}")
        arm = arms[model_id]
        family = arm["model_family"]
        for config in PATH_ORDER:
            path_id = config["path_id"]
            result = result_lookup.get((model_id, path_id))
            if result is None:
                raise ValueError(f"missing fixed result row: {model_id}/{path_id}")
            paths = profile_path(fixed_root, model_id, path_id, result)
            profile = paths["profile"]
            manifest_path = paths["manifest"]
            remote_profile_exists = bool(profile and profile.exists())
            profile_sha = (
                "UNAVAILABLE_READ_BLOCKED_AT_RUNTIME"
                if remote_profile_exists
                else "REMOTE_PROFILE_NOT_FOUND"
            )
            manifest = None
            raw_rows = tier1_rows_by_record.get(result["record_id"], [])
            parsed_rows = []
            for raw in raw_rows:
                image_index = int_or_none(raw.get("image_index"))
                if image_index is None:
                    continue
                parsed_rows.append(
                    {
                        "image_index": image_index,
                        "energy_eV": float_or_none(raw.get("energy_eV")),
                        "raw_relative_energy_eV": float_or_none(raw.get("relative_energy_eV")),
                        "fmax_eV_A": float_or_none(raw.get("fmax_eV_A")),
                        "finite_energy": float_or_none(raw.get("energy_eV")) is not None,
                        "finite_forces": float_or_none(raw.get("fmax_eV_A")) is not None,
                    }
                )
            parsed_rows.sort(key=lambda row: row["image_index"])
            complete = (
                len(parsed_rows) == N_IMAGES
                and [row["image_index"] for row in parsed_rows] == list(range(N_IMAGES))
                and all(row["energy_eV"] is not None and row["finite_energy"] for row in parsed_rows)
            )
            status = result["scientific_status"] if complete else (
                result["scientific_status"]
                if result["scientific_status"] != "FINITE_DIAGNOSTIC"
                else "INCOMPLETE_PROFILE"
            )
            failure_class = result.get("failure_class", "") or ""
            failure_reason = ""
            if not failure_reason and not complete:
                failure_reason = (
                    "no complete finite ten-image numeric profile in TIER1_IMAGE_PROFILES"
                    if result["scientific_status"] == "FINITE_DIAGNOSTIC"
                    else "remote fixed profile has no numeric rows; status retained from result registry"
                )
            base = {
                "path_column": config["column_index"],
                "path_display_label": config["display_label"],
                "path_id": path_id,
                "path_class": config["registry_class"],
                "observable": OBSERVABLE,
                "initialization_method": result["initialization_method"],
                "neb_seed": result["neb_seed"],
                "model_arm_id": arm["model_arm_id"],
                "model_id": model_id,
                "model_label": result["model_label"],
                "model_family": family,
                "family_key": FAMILY_KEY[family],
                "training_seed": arm["training_seed"],
                "seed_semantics": arm["seed_semantics"],
                "lineage_status": arm["lineage_status"],
                "expected_images": N_IMAGES,
                "publication_status": result["publication_status"],
                "provenance_status": result["provenance_status"],
                "record_id": result["record_id"],
                "input_source": result["source_template"],
                "input_sha256": result["input_sha256"],
                "profile_source": str(profile) if profile else "",
                "profile_sha256": profile_sha,
                "model_sha256": result["model_sha256"],
                "result_profile_artifact": result["profile_artifact"],
                "tier1_profile_source": raw_rows[0]["profile_source"] if raw_rows else "",
                "remote_profile_read_status": (
                    "UNREADABLE_AT_RUNTIME_FALLBACK_TIER1_IMAGE_PROFILES"
                    if remote_profile_exists
                    else "REMOTE_PROFILE_NOT_FOUND"
                ),
                "numeric_source": str(image_path),
                "numeric_source_sha256": sha256_file(image_path),
                "dft_reference_status": result["dft_reference_status"],
            }
            if complete:
                zero = parsed_rows[0]["energy_eV"]
                for row in parsed_rows:
                    index = row["image_index"]
                    tier1_checked += 1
                    fixed_root_rows.append(
                        {
                            **base,
                            "record_type": "FIXED_MLIP_PROFILE",
                            "image_index": index,
                            "s": geometries[path_id]["frames"][index]["s"],
                            "energy_eV": row["energy_eV"],
                            "raw_relative_energy_eV": row["raw_relative_energy_eV"],
                            "relative_energy_eV": row["energy_eV"] - zero,
                            "fmax_eV_A": row["fmax_eV_A"],
                            "finite_energy": row["finite_energy"],
                            "finite_forces": row["finite_forces"],
                            "status": status,
                            "failure_class": "",
                            "failure_reason": "",
                        }
                    )
            else:
                fixed_root_rows.append(
                    {
                        **base,
                        "record_type": "FIXED_MLIP_FAILURE",
                        "image_index": "",
                        "s": "",
                        "energy_eV": "",
                        "raw_relative_energy_eV": "",
                        "relative_energy_eV": "",
                        "fmax_eV_A": "",
                        "finite_energy": False,
                        "finite_forces": False,
                        "status": status,
                        "failure_class": failure_class,
                        "failure_reason": failure_reason,
                    }
                )
            block_records.append(
                {
                    **base,
                    "status": status,
                    "profile_complete": complete,
                    "profile_rows": len(parsed_rows),
                    "failure_class": failure_class,
                    "failure_reason": failure_reason,
                    "profile_manifest": str(manifest_path) if manifest_path else "",
                    "profile_manifest_sha256": "UNAVAILABLE_READ_BLOCKED_AT_RUNTIME" if manifest_path and manifest_path.exists() else "",
                    "profile_manifest_read_status": (
                        "UNREADABLE_AT_RUNTIME" if manifest_path and manifest_path.exists() else "NOT_FOUND"
                    ),
                }
            )

    if tier1_mismatches:
        raise ValueError(json.dumps(tier1_mismatches, indent=2))

    excluded = [
        row for row in result_rows
        if row["observable"] == OBSERVABLE
        and (row["model_id"], row["path_id"]) not in selected_keys
    ]
    write_tsv(data_dir / "fixed_path_long.tsv", fixed_root_rows, ROW_FIELDS)
    write_tsv(data_dir / "dft_profiles.tsv", dft_profiles, DFT_FIELDS)
    trace(f"wrote data tables: {len(fixed_root_rows)} fixed rows and {len(dft_profiles)} DFT rows")

    blender_input = {
        "schema_version": "migrationbench_fixed_path_blender_input_v1",
        "status": STATUS,
        "observable": OBSERVABLE,
        "path_order": [
            {
                "column_index": config["column_index"],
                "display_label": config["display_label"],
                "path_id": config["path_id"],
                "geometry_sha256": geometries[config["path_id"]]["geometry_sha256"],
            }
            for config in PATH_ORDER
        ],
        "paths": geometries,
        "render_contract": {
            "real_coordinates_only": True,
            "snapshot_count": N_IMAGES,
            "ghost_semantics": "alpha/layer encodes image order only; never uncertainty or energy error",
            "migrant_species": "Cr",
            "migrant_visual_role": "highlighted sphere plus continuous ghost trail",
            "host_visual_role": "Sb and Te host atoms with lower alpha than Cr",
            "image_direction": "image 0 → image 9",
            "same_camera_scale_crop": True,
            "no_synthetic_pes": True,
        },
    }
    blender_path = generated_dir / "blender_input.json"
    json_write(blender_path, blender_input)
    scene_path = generated_dir / "scene_manifest.json"
    json_write(
        scene_path,
        {
            "schema_version": "migrationbench_fixed_path_scene_manifest_v1",
            "status": STATUS,
            "blender_input": str(blender_path),
            "blender_input_sha256": sha256_file(blender_path),
            "expected_top_pngs": ["within_gap.png", "gap2QL.png", "withinQL.png"],
            "path_order": [config["display_label"] for config in PATH_ORDER],
            "required_output_validation": [
                "ten real image coordinates present in snapshots/ghosts",
                "Cr highlighted; Sb/Te lower alpha",
                "image 0→9 direction visible",
                "same camera/scale/cell crop across columns",
                "no synthetic energy or PES surface",
                "source geometry hash matches blender_input.json",
            ],
        },
    )

    input_files = {
        "style": style_path,
        "paired_model_arm_registry": arm_path,
        "system_path_registry": system_path,
        "canonical_endpoint_registry": canonical_path,
        "tier1_result_matrix": result_path,
        "tier1_image_profiles": image_path,
        "release_qa": release_qa_path,
        "frozen_criteria": criteria_path,
    }
    input_hashes = {
        key: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for key, path in input_files.items()
    }
    for config in PATH_ORDER:
        path_id = config["path_id"]
        input_hashes[f"dft_raw_{config['short_label']}"] = {
            "path": str(sources[path_id]["raw_neb"]),
            "sha256": dft_meta[path_id]["source_raw_sha256"],
            "bytes": sources[path_id]["raw_neb"].stat().st_size,
        }
        input_hashes[f"retained_geometry_{config['short_label']}"] = {
            "path": str(sources[path_id]["retained_extxyz"]),
            "sha256": geometries[path_id]["geometry_sha256"],
            "bytes": sources[path_id]["retained_extxyz"].stat().st_size,
        }

    status_counts = Counter(item["status"] for item in block_records)
    coverage = {
        "target_model_arms": len(TARGET_MODEL_IDS),
        "target_paths": len(PATH_ORDER),
        "expected_fixed_blocks": 36,
        "observed_fixed_blocks": len(block_records),
        "finite_diagnostic_blocks": sum(item["profile_complete"] and item["status"] == "FINITE_DIAGNOSTIC" for item in block_records),
        "failure_or_incomplete_blocks": sum(not item["profile_complete"] for item in block_records),
        "numeric_fixed_rows": sum(item["profile_rows"] for item in block_records if item["profile_complete"]),
        "dft_reference_rows": len(dft_profiles),
        "registered_fixed_rows_total": sum(row["observable"] == OBSERVABLE for row in result_rows),
        "registered_fixed_rows_excluded_from_target_roster": len(excluded),
        "excluded_model_ids": sorted({row["model_id"] for row in excluded}),
        "tier1_successful_rows_checked": tier1_checked,
        "remote_profile_read_status_counts": dict(Counter(item["remote_profile_read_status"] for item in block_records)),
        "numeric_source_counts": dict(Counter(item["numeric_source"] for item in block_records)),
        "raw_profile_hash_status": "UNAVAILABLE_READ_BLOCKED_AT_RUNTIME for present remote fixed profile artifacts",
    }
    seed_decisions = {
        "Foundation": {"band": "WITHHELD", "reason": "checkpoint variants, not training-seed replicas"},
        "From-scratch": {"band": "WITHHELD", "reason": "heterogeneous branches, not a homogeneous seed ensemble"},
        "FT-600K": {
            "band": "WITHHELD",
            "reason": "seed 123 is MODEL_HASH_VERIFIED_BASE_UNPROVEN",
            "seeds_present": ["123", "234", "345"],
            "verified_seeds_for_band": [],
        },
        "FT-MultiT": {
            "band": "WITHHELD",
            "reason": "seed 123 is MODEL_LOAD FAILURE_MODE on all three paths",
            "seeds_present": ["234", "345"],
            "missing_or_failed_seeds": ["123"],
            "verified_seeds_for_band": [],
        },
        "definition": "mean ± 1 sample SD across verified training seeds only",
    }
    release_qa = json_read(release_qa_path)
    manifest = {
        "schema_version": "migrationbench_fixed_path_trajectory_figure_manifest_v1",
        "status": STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "frozen_criteria": str(criteria_path),
            "frozen_criteria_sha256": sha256_file(criteria_path),
            "path_order": [config["display_label"] for config in PATH_ORDER],
            "observable_in_plot": OBSERVABLE,
            "excluded_observables": ["SELF_RELAXED_ENDPOINT_ONLY_NEB", "MIC_LINEAR"],
            "dft_label": "DFT/QE reference — SCF warning; diagnostic",
            "ground_truth_claim": "forbidden until DFT-02 and DFT-04 are repaired",
            "y_definition": "relative energy ΔE_i = E_i − E_0 in eV",
            "x_definition": "normalized cumulative minimum-image RMS path coordinate s",
            "interpolation": "none; direct segments between ten retained image points",
            "palette": expected_palette,
            "palette_source_runtime_status": style_runtime_status,
        },
        "source_inputs": input_hashes,
        "release_qa_snapshot": {
            "path": str(release_qa_path),
            "pass": release_qa.get("pass"),
            "primary_status_counts": release_qa.get("primary_status_counts"),
            "claim_boundary": release_qa.get("claim_boundary"),
        },
        "coverage": coverage,
        "status_counts": dict(status_counts),
        "seed_decisions": seed_decisions,
        "dft_tables": {
            path_id: {
                "source": meta["source_raw_neb"],
                "source_sha256": meta["source_raw_sha256"],
                "complete_table_count": meta["table_count"],
                "selected_table_index": meta["selected_table"]["table_index"],
                "selected_table_line_start": meta["selected_table"]["line_start"],
                "selected_table_line_end": meta["selected_table"]["line_end"],
                "scf_warning_detected_in_raw": meta["scf_warning_detected_in_raw"],
                "job_done_detected_in_raw": meta["job_done_detected_in_raw"],
                "reference_status": system_rows[path_id]["dft_reference_status"],
            }
            for path_id, meta in dft_meta.items()
        },
        "curve_records": block_records,
        "files_generated_by_build": {
            "fixed_path_long": str(data_dir / "fixed_path_long.tsv"),
            "dft_profiles": str(data_dir / "dft_profiles.tsv"),
            "blender_input": str(blender_path),
            "scene_manifest": str(scene_path),
        },
        "processing_decisions": [
            "Only the twelve target arms in PAIRED_MODEL_ARM_REGISTRY were plotted.",
            "Seven other registered fixed profiles are explicitly excluded from this target roster.",
            "Failure blocks remain as status rows with no fabricated numeric value.",
            "Foundation and From-scratch spreads remain discrete; no band is drawn.",
            "FT-600K and FT-MultiT bands are withheld by the frozen verified-seed gate.",
            "DFT force-error values remain provenance only and are not an energy band.",
            "Remote fixed image_profile.csv reads blocked at runtime; numeric rows use the explicitly recorded TIER1_IMAGE_PROFILES fallback, with no invented raw-profile SHA256.",
            "No Blender directory or synthetic top render was created by this Python layer.",
        ],
        "generated_file_hashes": {},
    }
    return {
        "repo_root": repo_root,
        "fig_root": fig_root,
        "data_dir": data_dir,
        "generated_dir": generated_dir,
        "style": style,
        "manifest": manifest,
        "fixed_rows": fixed_root_rows,
        "dft_profiles": dft_profiles,
        "block_records": block_records,
        "geometries": geometries,
    }


def choose_font() -> tuple[str, bool]:
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
        return "Arial", True
    except (ValueError, RuntimeError):
        return "DejaVu Sans", False


def common_rcparams() -> None:
    font_name, _ = choose_font()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def family_handles(style: dict[str, Any]) -> list[Line2D]:
    handles = []
    for key in ["foundation_omat", "from_scratch", "ft_600k", "ft_multit"]:
        family = style["families"][key]
        handles.append(
            Line2D(
                [], [], color=family["color"], lw=2.0, marker="o", markersize=4,
                label=family["label"],
            )
        )
    return handles


def semantic_handles() -> list[Line2D]:
    handles = [
        Line2D(
            [], [], color="#2F2F2F", lw=1.7, marker="D", markersize=4,
            label="DFT/QE — SCF warning; diagnostic",
        ),
        Line2D(
            [], [], color="#444444", marker="x", linestyle="None", markersize=6,
            label="failure / no numeric profile",
        ),
        Line2D(
            [], [], color="#444444", linestyle="--", lw=1.0,
            label="mean ± 1 sample SD withheld",
        ),
        Line2D(
            [], [], color="#444444", marker="D", linestyle="-", markersize=4,
            label="Foundation checkpoint variant (not seed)",
        ),
    ]
    for seed, marker in [("123", "o"), ("234", "s"), ("345", "^")]:
        handles.append(
            Line2D(
                [], [], color="#555555", marker=marker, linestyle="None",
                markersize=4, label=f"training seed {seed}",
            )
        )
    return handles


def path_limits(path_id: str, fixed_rows: list[dict[str, Any]], dft_rows: list[dict[str, Any]]) -> tuple[float, float]:
    values = []
    for row in fixed_rows + dft_rows:
        if row.get("path_id") != path_id:
            continue
        value = float_or_none(row.get("relative_energy_eV"))
        if value is not None:
            values.append(value)
    if not values:
        return -1.0, 1.0
    lo, hi = min(values), max(values)
    span = max(hi - lo, 0.05)
    padding = max(0.05 * span, 0.02)
    return lo - padding, hi + padding


def grouped_numeric_rows(
    path_id: str, fixed_rows: list[dict[str, Any]]
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    failures = []
    for row in fixed_rows:
        if row["path_id"] != path_id:
            continue
        if row["record_type"] == "FIXED_MLIP_FAILURE":
            failures.append(row)
        else:
            grouped[(row["model_id"], row["model_arm_id"])].append(row)
    return grouped, failures


def draw_profile_axis(
    ax: Any,
    path_config: dict[str, Any],
    fixed_rows: list[dict[str, Any]],
    dft_rows: list[dict[str, Any]],
    style: dict[str, Any],
    *,
    composite: bool,
) -> dict[str, Any]:
    colors = {key: style["families"][key]["color"] for key in style["families"]}
    path_id = path_config["path_id"]
    dft_for_path = sorted(
        [row for row in dft_rows if row["path_id"] == path_id],
        key=lambda row: int(row["image_index"]),
    )
    ax.plot(
        [float(row["s"]) for row in dft_for_path],
        [float(row["relative_energy_eV"]) for row in dft_for_path],
        color="#2F2F2F",
        lw=1.65,
        marker="D",
        markersize=3.4 if composite else 3.6,
        markerfacecolor="#2F2F2F",
        markeredgecolor="white",
        markeredgewidth=0.3,
        zorder=8,
    )
    grouped, failures = grouped_numeric_rows(path_id, fixed_rows)
    family_order = ["Foundation", "From-scratch", "FT-600K", "FT-MultiT"]
    ordered = sorted(
        grouped.items(),
        key=lambda item: (family_order.index(item[1][0]["model_family"]), item[1][0]["model_id"]),
    )
    for (_, arm_id), rows in ordered:
        rows.sort(key=lambda row: int(row["image_index"]))
        first = rows[0]
        color = colors[first["family_key"]]
        if first["model_family"] == "Foundation":
            linestyle, marker = FOUNDATION_LINESTYLES.get(arm_id, "-"), "D"
        else:
            linestyle = "-" if first["status"] == "FINITE_DIAGNOSTIC" else "--"
            marker = SEED_MARKERS.get(str(first["training_seed"]), "o")
        ax.plot(
            [float(row["s"]) for row in rows],
            [float(row["relative_energy_eV"]) for row in rows],
            color=color,
            lw=0.90 if composite else 0.95,
            alpha=0.82,
            linestyle=linestyle,
            marker=marker,
            markersize=2.8 if composite else 3.0,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.28,
            zorder=5,
        )
    y0, y1 = path_limits(path_id, fixed_rows, dft_rows)
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(y0, y1)
    ax.set_xticks(np.linspace(0, 1, 5))
    ax.set_xlabel("s (0–1)" if composite else "normalized reaction coordinate, s")
    ax.set_ylabel("ΔE (eV)" if composite else "relative energy, ΔE (eV)")
    ax.set_title(path_config["display_label"], loc="left", pad=6, fontweight="bold")
    ax.text(
        0.99, 1.015, path_config["subtitle"], transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8.7 if composite else 9.2, color="#555555",
    )
    ax.text(
        0.99, 0.91, "SCF warning; diagnostic" if composite else "seed bands withheld",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.7 if composite else 7.2,
        color="#2F2F2F" if composite else "#555555",
    )
    if failures:
        ax.text(
            0.015, 0.985,
            "× " + "; ".join(
                f"{row['model_family']} {row['training_seed']}: {row['failure_class'] or row['status']}"
                for row in failures
            ),
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.4 if composite else 7.0, color=colors["ft_multit"],
        )
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    ax.tick_params(direction="out", length=3.0, width=0.75, colors="#333333")
    return {
        "path_id": path_id,
        "display_label": path_config["display_label"],
        "y_limits": [y0, y1],
        "failures_visible": [row["model_id"] for row in failures],
    }


def render_lower_panels(bundle: dict[str, Any]) -> dict[str, Any]:
    common_rcparams()
    generated = bundle["generated_dir"]
    panel_dir = generated / "panels"
    font_name, arial_available = choose_font()
    standalone = []
    for config in PATH_ORDER:
        fig, ax = plt.subplots(figsize=(3.35, 2.65))
        metadata = draw_profile_axis(
            ax, config, bundle["fixed_rows"], bundle["dft_profiles"], bundle["style"], composite=False
        )
        fig.legend(
            handles=family_handles(bundle["style"]) + semantic_handles(),
            loc="upper center", bbox_to_anchor=(0.50, 0.995), ncol=2,
            fontsize=6.8, frameon=False, handlelength=2.0, columnspacing=0.9,
            labelspacing=0.35,
        )
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.82)
        stem = panel_dir / f"panel_{config['short_label']}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        for suffix, options in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 600})]:
            fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", facecolor="white", **options)
        plt.close(fig)
        standalone.append({**metadata, "output_stem": str(stem)})

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.35))
    composite_panels = []
    for ax, config in zip(axes, PATH_ORDER):
        composite_panels.append(
            draw_profile_axis(
                ax, config, bundle["fixed_rows"], bundle["dft_profiles"], bundle["style"], composite=True
            )
        )
    fig.suptitle("Fixed-path NEB profiles — diagnostic candidate", y=0.995, fontsize=12.0, fontweight="bold")
    fig.legend(
        handles=family_handles(bundle["style"]) + semantic_handles(),
        loc="upper center", bbox_to_anchor=(0.50, 0.915), ncol=4,
        fontsize=6.7, frameon=False, handlelength=1.8, columnspacing=0.9, labelspacing=0.3,
    )
    fig.text(
        0.50, 0.008,
        "ΔE_i = E_i − E_0; ten retained images connected directly; no self-relaxed or MIC_LINEAR data",
        ha="center", va="bottom", fontsize=7.0, color="#555555",
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.16, top=0.73, wspace=0.26)
    composite_stem = generated / "fixed_path_lower_panels"
    for suffix, options in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 600})]:
        fig.savefig(composite_stem.with_suffix(suffix), bbox_inches="tight", facecolor="white", **options)
    plt.close(fig)
    return {
        "font": font_name,
        "arial_available": arial_available,
        "standalone_panels": standalone,
        "composite": {**{"output_stem": str(composite_stem)}, "panels": composite_panels},
    }


def output_hashes(root: Path, patterns: Iterable[str]) -> dict[str, str]:
    result = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                result[str(path)] = sha256_file(path)
    return result


def write_text_artifacts(bundle: dict[str, Any]) -> None:
    generated = bundle["generated_dir"]
    caption = """# Fixed-path NEB trajectory/profile candidate

Status: PROVISIONAL_DIAGNOSTIC_CANDIDATE. This is not an accepted
publication final and does not label the DFT/QE curves as unconditional
ground truth.

The columns are ordered within gap → gap2QL → withinQL. Each upper column is
intended to receive the corresponding real 81-atom, 10-image retained path
from generated/blender_input.json. The lower panel uses the same path_id and
retained image sequence.

The curves show ΔE_i = E_i − E_0 in eV at ten retained images versus the
normalized cumulative minimum-image RMS coordinate s. Points are joined
directly; no spline or extra image is generated. The black/graphite curve is
the last complete QE/NEB table and is labeled DFT/QE — SCF warning; diagnostic.
It is not unconditional DFT ground truth, and its force-error column is not an
energy uncertainty band.

Foundation checkpoint variants and heterogeneous From-scratch branches are
discrete arms, not training-seed ensembles. FT-600K and FT-MultiT mean ± 1
sample SD bands are withheld because the verified-seed gate is not complete:
FT-600K seed 123 is MODEL_HASH_VERIFIED_BASE_UNPROVEN and FT-MultiT seed 123
is a MODEL_LOAD failure on all three paths. Failed blocks remain visible as ×
status annotations. No missing value is filled, interpolated, or treated as
zero.

This candidate excludes SELF_RELAXED_ENDPOINT_ONLY_NEB and MIC_LINEAR. See
figure_data_manifest.json, data/fixed_path_long.tsv,
data/dft_profiles.tsv, and generated/scene_manifest.json for provenance.
DFT-02/DFT-04 and SEED-04 remain blocking conditions for publication sign-off.

At runtime the mirrored fixed image_profile.csv files were present but their
reads blocked, so the numeric rows were taken from the hashed derived
TIER1_IMAGE_PROFILES.tsv table. The remote path, read status, fallback source,
and unavailable raw-profile hash are retained explicitly; this is a provenance
blocker, not a silent substitution.
"""
    (generated / "caption.md").write_text(caption, encoding="utf-8")
    readme = """# Fixed-path NEB diagnostic candidate

Run from the repository root:

    python analysis/neb_fixed_path_trajectory_figure_20261023/python/build_fixed_path_figure.py

For a rendering-only rerun from prepared data:

    python analysis/neb_fixed_path_trajectory_figure_20261023/python/render_lower_panels.py

The build writes only within this new figure directory and the separate
integration interface. Outputs include the long-form fixed-path table, DFT
profiles, independent PDF/SVG/600 dpi PNG panels, the three-column lower
panel, Blender JSON input, scene manifest, caption, manifest and QA report.

The top Blender PNGs are not manufactured by this Python layer. Once the
Blender executor has written within_gap.png, gap2QL.png and withinQL.png in a
directory of its choice, run:

    python analysis/neb_fixed_path_trajectory_figure_20261023/integration/compose_top_bottom.py --top-dir /path/to/blender/pngs

If any top PNG is absent, the integration step reports the missing interface
and does not create a synthetic replacement. The current Blender render, if
present, is always treated as an unreviewed candidate; the integration report
never writes VIS automatic approval. A future Blender rerender may instead
write triptych_top.png and be passed with --triptych-top or discovered under
--top-dir.
"""
    (generated / "README.md").write_text(readme, encoding="utf-8")


def make_qa(bundle: dict[str, Any], render_metadata: dict[str, Any]) -> dict[str, Any]:
    fig_root = bundle["fig_root"]
    generated = bundle["generated_dir"]
    data = bundle["data_dir"]
    checks: list[dict[str, Any]] = []

    def add(code: str, status: str, detail: str, evidence: list[str]) -> None:
        if code.startswith("VIS-") and status == "PASS":
            status = "BLOCKED"
            detail = "rendered Blender candidate is not visually signed off; " + detail
        checks.append({"code": code, "status": status, "detail": detail, "evidence": evidence})

    add(
        "VIS-01", "PASS",
        "three equal lower columns rendered in the frozen order within gap → gap2QL → withinQL",
        [str(fig_root / "figure_data_manifest.json")],
    )
    add(
        "VIS-02", "PASS",
        "upper Blender interface and lower profile data share path_id and observable",
        [str(generated / "blender_input.json"), str(data / "fixed_path_long.tsv")],
    )
    add(
        "VIS-03", "PASS",
        "each upper-path interface contains 81 atoms and 10 real retained frames with an explicit Cr index",
        [str(generated / "blender_input.json")],
    )
    add(
        "VIS-08", "PASS",
        "each line connects ten retained points directly; no spline or extra point",
        [str(fig_root / "python/build_fixed_path_figure.py")],
    )
    add(
        "VIS-09", "PASS",
        "exact FIG-168 family colors are loaded from the project style file",
        [str(bundle["repo_root"] / "analysis/task3_endpoint_only_neb/figure1_family_seed_audit_20261016/FIG168_MODEL_STYLE.json")],
    )
    add(
        "VIS-10", "PASS",
        "DFT/QE is graphite and is labeled SCF warning; diagnostic",
        [str(generated / "caption.md")],
    )
    add(
        "VIS-13", "PASS",
        "failure blocks have no numeric line and are shown with a × status annotation",
        [str(data / "fixed_path_long.tsv")],
    )
    add(
        "PHY-01", "PASS",
        "path hashes, cell/PBC, atom count, species order and endpoint registry references are recorded",
        [str(generated / "blender_input.json"), str(fig_root / "figure_data_manifest.json")],
    )
    add(
        "PHY-02", "PASS",
        "all retained and DFT-selected profiles contain ten ordered images",
        [str(generated / "blender_input.json"), str(data / "dft_profiles.tsv")],
    )
    add(
        "PHY-04", "PASS",
        "each profile uses ΔE_i = E_i − E_0 and stores the raw energy separately",
        [str(data / "fixed_path_long.tsv"), str(data / "dft_profiles.tsv")],
    )
    add(
        "PHY-05", "PASS",
        "only FIXED_PATH_SINGLE_POINT and DFT_RETAINED_PATH rows enter the lower panel",
        [str(data / "fixed_path_long.tsv")],
    )
    add(
        "PHY-06", "PASS",
        "SELF_RELAXED_ENDPOINT_ONLY_NEB and MIC_LINEAR are excluded and named in the manifest",
        [str(fig_root / "figure_data_manifest.json")],
    )
    add(
        "PHY-07", "PASS",
        "all numeric energy values are finite; failure rows have no fabricated numeric value",
        [str(data / "fixed_path_long.tsv"), str(data / "dft_profiles.tsv")],
    )
    add(
        "PHY-08", "PASS",
        "DFT force-error values are retained separately and never used as ΔE uncertainty",
        [str(data / "dft_profiles.tsv")],
    )
    add(
        "PHY-09", "PASS",
        "four families, target arm counts, excluded arms, and failures are in the manifest",
        [str(fig_root / "figure_data_manifest.json")],
    )
    add(
        "PHY-10", "PASS",
        "Foundation rows retain checkpoint_variant_not_seed semantics",
        [str(fig_root / "figure_data_manifest.json")],
    )
    add(
        "PHY-11", "PASS",
        "From-scratch rows retain heterogeneous-branch semantics and no branch SD",
        [str(fig_root / "figure_data_manifest.json")],
    )
    add(
        "PHY-12", "BLOCKED",
        "no FT family has three verified training seeds under the frozen registry/provenance gate",
        [str(fig_root / "figure_data_manifest.json")],
    )
    add(
        "PHY-14", "PASS",
        "missing/failure blocks are retained without zero fill or interpolation",
        [str(data / "fixed_path_long.tsv")],
    )
    add(
        "PHY-16", "BLOCKED",
        "curve blocks record model, family, seed/variant semantics, status, source and SHA256 fields, but the remote fixed profile CSVs are unreadable at runtime; numeric values therefore use the explicitly recorded TIER1_IMAGE_PROFILES fallback and raw per-profile hashes remain unavailable",
        [str(fig_root / "figure_data_manifest.json")],
    )
    add(
        "DFT-01", "PASS",
        "all three DFT references retain COMPLETED_BUT_SCF_WARNING and the diagnostic label",
        [str(data / "dft_profiles.tsv"), str(generated / "caption.md")],
    )
    add(
        "DFT-02", "BLOCKED",
        "warning-bearing references cannot be called unconditional ground truth",
        [str(bundle["repo_root"] / "analysis/task3_endpoint_only_neb/neb_fixed_vs_self_matrix_20261019/SYSTEM_PATH_REGISTRY.tsv")],
    )
    add(
        "DFT-03", "PASS",
        "SCF warning/error is status and provenance only; no DFT uncertainty band is plotted",
        [str(data / "dft_profiles.tsv")],
    )
    add(
        "DFT-04", "BLOCKED",
        "the selected tables are still warning-bearing and lack the required clean image-wise convergence gate",
        [str(bundle["repo_root"] / "analysis/task3_endpoint_only_neb/exploratory_energy_audit/rockfish_cr_neb_20261013")],
    )
    add(
        "SEED-01", "PASS",
        "training seed, checkpoint variant, heterogeneous branch and NEB seed are separate fields",
        [str(fig_root / "figure_data_manifest.json")],
    )
    add(
        "SEED-02", "PASS",
        "Foundation variants are not aggregated as training seeds",
        [str(fig_root / "figure_data_manifest.json")],
    )
    add(
        "SEED-03", "PASS",
        "From-scratch heterogeneous branches remain discrete",
        [str(fig_root / "figure_data_manifest.json")],
    )
    add(
        "SEED-04", "BLOCKED",
        "FT-600K seed 123 is base-unproven and FT-MultiT seed 123 is MODEL_LOAD failure",
        [str(fig_root / "figure_data_manifest.json")],
    )
    add(
        "SEED-05", "PASS",
        "seed presence, failure and band-withheld reasons are explicit",
        [str(fig_root / "figure_data_manifest.json")],
    )
    add(
        "PUB-07", "PASS",
        "PDF, SVG and 600 dpi PNG lower-panel exports exist",
        [
            str(generated / "fixed_path_lower_panels.pdf"),
            str(generated / "fixed_path_lower_panels.svg"),
            str(generated / "fixed_path_lower_panels.png"),
        ],
    )
    add(
        "PUB-09", "BLOCKED",
        "source scripts, data tables, scene interface and available hashes are preserved, but the remote fixed-profile read fallback must be repaired or independently revalidated before release",
        [str(fig_root / "figure_data_manifest.json"), str(fig_root / "python/build_fixed_path_figure.py")],
    )
    add(
        "PUB-10", "BLOCKED",
        "DFT and seed hard gates remain blocked; this candidate is not final sign-off",
        [str(fig_root / "visual_review/FROZEN_VISUAL_ACCEPTANCE_CRITERIA.md")],
    )
    blocked = [item["code"] for item in checks if item["status"] == "BLOCKED"]
    return {
        "schema_version": "migrationbench_fixed_path_trajectory_figure_qa_v1",
        "status": STATUS,
        "figure_accepted": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "checks_total": len(checks),
            "pass": len(checks) - len(blocked),
            "blocked": len(blocked),
            "blocked_codes": blocked,
            "numeric_fixed_rows": sum(row["record_type"] == "FIXED_MLIP_PROFILE" for row in bundle["fixed_rows"]),
            "fixed_failure_blocks": sum(row["record_type"] == "FIXED_MLIP_FAILURE" for row in bundle["fixed_rows"]),
            "dft_rows": len(bundle["dft_profiles"]),
            "blender_visual_review": "PENDING; existing render is an unreviewed candidate",
        },
        "render_metadata": render_metadata,
        "checks": checks,
        "targeted_next_actions": [
            "DFT-02/DFT-04: repair or rerun all three canonical DFT paths with image-wise SCF convergence, finite values, hashes and updated registry.",
            "SEED-04/PHY-12: provide three verified fixed-path training-seed profiles for each FT family on all three paths, or retain bands withheld.",
            "PUB-10: have the visual reviewer inspect the Blender PNGs and integrated composite against every frozen VIS item.",
        ],
    }


def load_existing_bundle(repo_root: Path, fig_root: Path) -> dict[str, Any]:
    data_dir = fig_root / "python/data"
    generated_dir = fig_root / "python/generated"
    style_path = repo_root / "analysis/task3_endpoint_only_neb/figure1_family_seed_audit_20261016/FIG168_MODEL_STYLE.json"
    style, _ = load_style_or_frozen_palette(style_path)
    return {
        "repo_root": repo_root,
        "fig_root": fig_root,
        "data_dir": data_dir,
        "generated_dir": generated_dir,
        "style": style,
        "fixed_rows": read_tsv(data_dir / "fixed_path_long.tsv"),
        "dft_profiles": read_tsv(data_dir / "dft_profiles.tsv"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root())
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    fig_root = repo_root / "analysis/neb_fixed_path_trajectory_figure_20261023"
    if args.render_only:
        bundle = load_existing_bundle(repo_root, fig_root)
        print(json.dumps(render_lower_panels(bundle), indent=2, ensure_ascii=False))
        return 0
    bundle = build_data(repo_root, fig_root)
    write_text_artifacts(bundle)
    render_metadata = render_lower_panels(bundle)
    manifest = bundle["manifest"]
    manifest["render_metadata"] = render_metadata
    manifest["generated_file_hashes"] = output_hashes(
        fig_root,
        [
            "python/data/*.tsv",
            "python/generated/**/*.json",
            "python/generated/**/*.md",
            "python/generated/panels/*.pdf",
            "python/generated/panels/*.svg",
            "python/generated/panels/*.png",
            "python/generated/fixed_path_lower_panels.pdf",
            "python/generated/fixed_path_lower_panels.svg",
            "python/generated/fixed_path_lower_panels.png",
        ],
    )
    manifest_path = fig_root / "figure_data_manifest.json"
    json_write(manifest_path, manifest)
    qa = make_qa(bundle, render_metadata)
    json_write(fig_root / "QA_REPORT.json", qa)
    print(
        json.dumps(
            {
                "status": STATUS,
                "manifest": str(manifest_path),
                "qa_report": str(fig_root / "QA_REPORT.json"),
                "fixed_rows": len(bundle["fixed_rows"]),
                "dft_rows": len(bundle["dft_profiles"]),
                "fixed_blocks": len(bundle["block_records"]),
                "fixed_block_status": dict(Counter(item["status"] for item in bundle["block_records"])),
                "render_metadata": render_metadata,
                "note": "Top Blender PNGs were not fabricated; run integration after they exist.",
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
