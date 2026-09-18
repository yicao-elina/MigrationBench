#!/usr/bin/env python3
"""Prepare auditable standalone QE relaxations for NEB endpoint/basin images."""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_neb_path_topology import read_qe_image, read_xyz  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def set_namelist_value(text, namelist, key, value):
    match = re.search(
        rf"(?ims)^\s*&{re.escape(namelist)}\b.*?^\s*/\s*$",
        text,
    )
    if not match:
        raise ValueError(f"Missing &{namelist} namelist")
    block = match.group(0)
    key_pattern = re.compile(rf"(?im)^\s*{re.escape(key)}\s*=.*$")
    replacement = f"  {key} = {value}"
    if key_pattern.search(block):
        block = key_pattern.sub(replacement, block, count=1)
    else:
        block = re.sub(r"(?m)^\s*/\s*$", replacement + "\n/", block, count=1)
    return text[: match.start()] + block + text[match.end() :]


def replace_positions(text, symbols, positions):
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.strip().upper().startswith("ATOMIC_POSITIONS")),
        None,
    )
    if start is None or len(lines) < start + 1 + len(symbols):
        raise ValueError("Incomplete or missing ATOMIC_POSITIONS block")
    lines[start] = "ATOMIC_POSITIONS angstrom"
    for offset, (symbol, position) in enumerate(zip(symbols, positions), start=1):
        fields = lines[start + offset].split()
        if not fields or fields[0] != symbol:
            raise ValueError(
                f"Atom order mismatch at atom {offset}: expected {symbol}, found {fields[:1]}"
            )
        suffix = fields[4:]
        lines[start + offset] = "{}  {:.12f}  {:.12f}  {:.12f}".format(symbol, *position)
        if suffix:
            lines[start + offset] += "  " + " ".join(suffix)
    return "\n".join(lines) + "\n"


def uses_multiframe_coordinates(source):
    return str(source.get("image_format", "")).startswith("multi_frame_xyz")


def may_use_numbered_qe_coordinates(source, path):
    return path.exists() and not uses_multiframe_coordinates(source)


def transform_pw(text, prefix, max_seconds, nstep=200):
    if not re.search(r"(?im)^\s*&CONTROL\s*$", text):
        raise ValueError("Missing &CONTROL namelist")
    for key, value in (
        ("calculation", "'relax'"),
        ("restart_mode", "'from_scratch'"),
        ("tprnfor", ".true."),
        ("max_seconds", str(max_seconds)),
        ("nstep", str(nstep)),
        ("prefix", f"'{prefix}'"),
    ):
        text = set_namelist_value(text, "CONTROL", key, value)
    # Endpoint validation is fixed-cell by definition. A legacy &CELL block
    # must not silently turn this into a different optimization protocol.
    text = re.sub(r"(?ims)^\s*&CELL\b.*?^\s*/\s*\n?", "", text)
    if not re.search(r"(?im)^\s*&IONS\s*$", text):
        marker = re.search(r"(?im)^\s*ATOMIC_SPECIES\s*$", text)
        if not marker:
            raise ValueError("Missing ATOMIC_SPECIES section")
        ions = "&IONS\n  ion_dynamics = 'bfgs'\n/\n"
        text = text[: marker.start()] + ions + text[marker.start() :]
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology-manifest", type=Path, required=True)
    parser.add_argument("--select", action="append", required=True, help="PATH_ID:IMAGE_INDEX_QE")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--walltime", default="24:00:00")
    parser.add_argument("--max-seconds", type=int, default=84600)
    parser.add_argument("--nstep", type=int, default=200)
    parser.add_argument("--ntasks", type=int, default=2)
    parser.add_argument("--memory", default="160G")
    args = parser.parse_args()

    topology = json.loads(args.topology_manifest.read_text())
    by_id = {row["path_id"]: row for row in topology["results"]}
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for selection in args.select:
        path_id, raw_index = selection.rsplit(":", 1)
        image_index = int(raw_index)
        source = by_id[path_id]
        image_sources = [Path(row["path"]) for row in source["image_sources"]]
        # In historical multi-frame layouts pw_1.in is only the calculator/cell
        # template. Coordinates for every image, including image 1, must come
        # from the same trajectory to avoid mixing path states.
        multi_frame_source = uses_multiframe_coordinates(source)
        numbered = [] if multi_frame_source else [
            path for path in image_sources if path.name == f"pw_{image_index}.in"
        ]
        coordinate_source = None
        coordinate_source_type = None
        coordinate_source_frame = None
        if numbered:
            pw_source = numbered[0]
            template_text = pw_source.read_text()
            coordinate_source = pw_source
            coordinate_source_type = "numbered_qe_input"
        else:
            run_dir = Path(source["run_dir"])
            candidate = run_dir / f"pw_{image_index}.in"
            if may_use_numbered_qe_coordinates(source, candidate):
                pw_source = candidate
                template_text = pw_source.read_text()
                coordinate_source = pw_source
                coordinate_source_type = "numbered_qe_input"
            else:
                xyz_sources = [path for path in image_sources if path.suffix.lower() == ".xyz"]
                template_sources = [path for path in image_sources if path.name == "pw_1.in"]
                if not xyz_sources or not template_sources:
                    raise ValueError(
                        f"No numbered QE input or multi-frame XYZ + pw_1.in fallback for {selection}"
                    )
                coordinate_source = xyz_sources[0]
                coordinate_source_type = "multi_frame_xyz"
                coordinate_source_frame = image_index
                frames = read_xyz(coordinate_source)
                if not 1 <= image_index <= len(frames):
                    raise ValueError(
                        f"Image {image_index} outside 1..{len(frames)} in {coordinate_source}"
                    )
                frame = frames[image_index - 1]
                pw_source = template_sources[0]
                template = read_qe_image(pw_source)
                if frame["symbols"] != template["symbols"]:
                    raise ValueError(f"Atom order mismatch between {coordinate_source} and {pw_source}")
                template_text = replace_positions(
                    pw_source.read_text(), frame["symbols"], frame["positions"]
                )
        safe_path = re.sub(r"[^A-Za-z0-9]+", "", path_id).lower()
        branch_id = f"{path_id}_img{image_index}_endpoint_relax_s{args.seed}"
        job_name = f"mb_ep_{safe_path}_i{image_index}_s{args.seed}"
        branch_dir = out_dir / branch_id
        branch_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"ep_{safe_path}_i{image_index}_s{args.seed}"
        relax_input = branch_dir / "relax.in"
        relax_input.write_text(transform_pw(template_text, prefix, args.max_seconds, args.nstep))
        energy_row = next(
            (row for row in source["qe_summary"]["last_images"] if row["image_index"] == image_index),
            None,
        )
        manifest = {
            "schema_version": "1.1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "branch_id": branch_id,
            "job_name": job_name,
            "seed": args.seed,
            "scientific_role": "standalone_local_minimum_validation",
            "source_path_id": path_id,
            "source_image_index_qe": image_index,
            "source_pw_input": str(pw_source.resolve()),
            "source_pw_sha256": sha256_file(pw_source),
            "coordinate_source": str(coordinate_source.resolve()),
            "coordinate_source_sha256": sha256_file(coordinate_source),
            "coordinate_source_type": coordinate_source_type,
            "coordinate_source_frame_qe": coordinate_source_frame,
            "source_neb_out": source["neb_out"],
            "source_neb_out_sha256": source["neb_out_sha256"],
            "source_neb_image_state": energy_row,
            "endpoint_acceptance_gate": {
                "qe_job_done": True,
                "bfgs_converged": True,
                "final_max_force_eV_A_lte": 0.05,
                "no_geometry_gate_failure": True,
            },
            "resources": {
                "walltime": args.walltime,
                "pw_max_seconds": args.max_seconds,
                "nstep": args.nstep,
                "ntasks": args.ntasks,
                "memory": args.memory,
            },
            "fixed_cell": True,
            "relax_input": str(relax_input),
            "relax_input_sha256": sha256_file(relax_input),
        }
        manifest_path = branch_dir / "endpoint_relax_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        jobs.append({**manifest, "manifest_path": str(manifest_path)})

    batch = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "topology_manifest": str(args.topology_manifest.resolve()),
        "topology_manifest_sha256": sha256_file(args.topology_manifest),
        "jobs": jobs,
    }
    (out_dir / "endpoint_relax_batch_manifest.json").write_text(json.dumps(batch, indent=2) + "\n")
    print(json.dumps({"prepared": len(jobs), "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
