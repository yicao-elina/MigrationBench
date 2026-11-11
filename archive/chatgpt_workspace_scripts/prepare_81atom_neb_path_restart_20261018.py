#!/usr/bin/env python3
"""Prepare an additive QE NEB restart from the last saved historical path.

The authoritative historical tree does not contain the original QE
``sb2te3.path`` checkpoint.  It does contain a ten-image ``sb2te3.xyz`` and
the matching ``sb2te3.dat``/``neb.out`` final profile.  This script rebuilds a
QE-readable path checkpoint from those files, preserving the provenance gap:
the historical gradients/Broyden state are unavailable and are represented by
zero placeholders for QE to replace after fresh SCF evaluations.
"""

import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


HISTORICAL_ROOT = Path(
    "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/2.2-layer-diffusion/2D_421_Diffusion_traj"
)
TEMPLATE_ROOT = Path(
    "/scratch16/pclancy3/yi/revision1_migrationbench_inputs"
)
DEFAULT_CAMPAIGN_ROOT = Path(
    "/scratch16/pclancy3/yi/cr_81atom_neb_path_restart_20261018"
)
CAMPAIGN_ID = os.environ.get(
    "MIGRATIONBENCH_CAMPAIGN_ID", "cr_81atom_neb_path_restart_20261018"
)
PIPELINE_ROOT = Path("/scratch16/pclancy3/yi/revision1_migrationbench_pipeline")
BOHR_PER_ANGSTROM = 0.529177210903
HARTREE_PER_EV = 27.211386245988
KPOINT_CARD = os.environ.get("MIGRATIONBENCH_KPOINT_CARD", "2 2 2 1 1 1")
PREVIOUS_CAMPAIGN_ROOT = os.environ.get("MIGRATIONBENCH_PREVIOUS_CAMPAIGN_ROOT", "")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_tree_files(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    copied = {}
    for item in sorted(source.iterdir()):
        if item.is_file():
            target = destination / item.name
            shutil.copy2(str(item), str(target))
            copied[item.name] = sha256(target)
    return copied


def read_xyz(path):
    lines = path.read_text().splitlines()
    frames = []
    cursor = 0
    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            break
        try:
            nat = int(lines[cursor].strip())
        except ValueError:
            raise RuntimeError("invalid XYZ atom count in %s at line %d" % (path, cursor + 1))
        if cursor + 1 + nat >= len(lines):
            raise RuntimeError("truncated XYZ frame in %s" % path)
        comment = lines[cursor + 1]
        atoms = []
        for line in lines[cursor + 2:cursor + 2 + nat]:
            fields = line.split()
            if len(fields) != 4:
                raise RuntimeError("invalid XYZ atom line in %s: %s" % (path, line))
            atoms.append((fields[0], float(fields[1]), float(fields[2]), float(fields[3])))
        frames.append({"nat": nat, "comment": comment, "atoms": atoms})
        cursor += 2 + nat
    return frames


def read_profile(path):
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 3:
            try:
                rows.append([float(value) for value in fields])
            except ValueError:
                pass
    return rows


def read_last_complete_table(path):
    text = path.read_text()
    headers = list(re.finditer(r"^\s*-+ iteration\s+(\d+)\s*-+\s*$", text, re.MULTILINE))
    row_pattern = re.compile(
        r"^\s*(\d+)\s+"
        r"([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
        r"([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
        r"([TF])\s*$",
        re.MULTILINE,
    )
    complete = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        block = text[header.end():end]
        marker = re.search(r"image\s+energy \(eV\)\s+error \(eV/A\)\s+frozen", block)
        if not marker:
            continue
        rows = []
        for match in row_pattern.finditer(block[marker.end():]):
            rows.append(
                {
                    "image": int(match.group(1)),
                    "energy_eV": float(match.group(2)),
                    "error_eV_per_A": float(match.group(3)),
                    "frozen": match.group(4),
                }
            )
        if len(rows) == 10 and [row["image"] for row in rows] == list(range(1, 11)):
            complete.append({"iteration": int(header.group(1)), "rows": rows})
    if not complete:
        raise RuntimeError("no complete 10-image energy table found in %s" % path)
    return complete[-1]


def format_path(frames, table):
    lines = [
        "RESTART INFORMATION",
        "%8d" % table["iteration"],
        "%8d" % 1000,
        "%8d" % 0,
        "NUMBER OF IMAGES",
        "%3d" % 10,
        "APPLY CONSTANT BIAS",
        "F",
        "ENERGIES, POSITIONS AND GRADIENTS",
    ]
    for image_index, frame in enumerate(frames, 1):
        energy_hartree = table["rows"][image_index - 1]["energy_eV"] / HARTREE_PER_EV
        lines.append("Image: %4d" % image_index)
        lines.append("%17.10f" % energy_hartree)
        for symbol, x, y, z in frame["atoms"]:
            lines.append(
                "%20.12f %20.12f %20.12f %20.12f %20.12f %20.12f  1  1  1"
                % (
                    x / BOHR_PER_ANGSTROM,
                    y / BOHR_PER_ANGSTROM,
                    z / BOHR_PER_ANGSTROM,
                    0.0,
                    0.0,
                    0.0,
                )
            )
    return "\n".join(lines) + "\n"


def render_endpoint(frame):
    lines = ["ATOMIC_POSITIONS angstrom"]
    for symbol, x, y, z in frame["atoms"]:
        lines.append("  %-2s % .10f % .10f % .10f" % (symbol, x, y, z))
    return "\n".join(lines)


def patch_input(template, first_frame, last_frame):
    text = template.read_text()
    if not re.search(r"\bnat\s*=\s*81\b", text):
        raise RuntimeError("template does not declare nat=81: %s" % template)
    path_block = """BEGIN_PATH_INPUT
&PATH
  restart_mode      = 'restart'
  string_method     = 'neb'
  nstep_path        = 1000
  ds                = 0.0756D0
  opt_scheme        = 'broyden'
  num_of_images     = 10
  k_max             = 0.0131D0
  k_min             = 0.0050D0
  CI_scheme         = 'no-CI'
  path_thr          = 0.001D0
  first_last_opt    = .true.
/
END_PATH_INPUT"""
    text, count = re.subn(
        r"BEGIN_PATH_INPUT\n&PATH.*?/\nEND_PATH_INPUT",
        path_block,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("template has no unique BEGIN_PATH_INPUT block: %s" % template)
    text, count = re.subn(r"max_seconds\s*=\s*250000", "max_seconds = 171000", text, count=1)
    if count != 1:
        raise RuntimeError("template has no unique max_seconds=250000: %s" % template)
    text, count = re.subn(
        r"(?m)^\s*2 2 2 1 1 1\s*$",
        KPOINT_CARD,
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("template has no unique 2x2x2 K_POINTS card: %s" % template)
    positions = "BEGIN_POSITIONS\nFIRST_IMAGE\n%s\nLAST_IMAGE\n%s\nEND_POSITIONS" % (
        render_endpoint(first_frame),
        render_endpoint(last_frame),
    )
    text, count = re.subn(
        r"BEGIN_POSITIONS.*?END_POSITIONS",
        positions,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("template has no unique BEGIN_POSITIONS block: %s" % template)
    return text


def main():
    campaign_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CAMPAIGN_ROOT
    if campaign_root.exists():
        raise SystemExit("refusing to overwrite existing campaign root: %s" % campaign_root)
    campaign_root.mkdir(parents=True)
    (campaign_root / "archive" / "historical").mkdir(parents=True)
    (campaign_root / "archive" / "calculator_templates").mkdir(parents=True)
    (campaign_root / "inputs").mkdir()
    (campaign_root / "runs").mkdir()
    all_paths = []

    for number in range(1, 6):
        historical = HISTORICAL_ROOT / ("neb_%d" % number)
        template_dir = TEMPLATE_ROOT / ("mb_qe81_neb_%d_mace_r1_s42" % number)
        for required in (historical / "neb.out", historical / "sb2te3.dat", historical / "sb2te3.xyz", template_dir / "neb.in"):
            if not required.is_file():
                raise SystemExit("missing required source: %s" % required)

        historical_archive = campaign_root / "archive" / "historical" / ("neb_%d" % number)
        template_archive = campaign_root / "archive" / "calculator_templates" / ("neb_%d" % number)
        historical_hashes = copy_tree_files(historical, historical_archive)
        template_hashes = copy_tree_files(template_dir, template_archive)

        xyz_path = historical / "sb2te3.xyz"
        dat_path = historical / "sb2te3.dat"
        out_path = historical / "neb.out"
        frames = read_xyz(xyz_path)
        profile = read_profile(dat_path)
        table = read_last_complete_table(out_path)
        if len(frames) != 10 or any(frame["nat"] != 81 for frame in frames):
            raise SystemExit("neb_%d XYZ is not exactly ten 81-atom frames" % number)
        if len(profile) != 10 or len(table["rows"]) != 10:
            raise SystemExit("neb_%d profile/table does not contain ten images" % number)
        if [atom[0] for atom in frames[0]["atoms"]] != [atom[0] for atom in frames[-1]["atoms"]]:
            raise SystemExit("neb_%d endpoint atom ordering differs" % number)

        input_dir = campaign_root / "inputs" / ("neb_%d" % number)
        input_dir.mkdir()
        reconstructed_path = format_path(frames, table)
        restart_path = input_dir / "sb2te3.path"
        restart_path.write_text(reconstructed_path)
        shutil.copy2(str(xyz_path), str(input_dir / "historical_final_path.xyz"))
        shutil.copy2(str(dat_path), str(input_dir / "historical_final_profile.dat"))

        neb_input = input_dir / "neb.in"
        neb_input.write_text(patch_input(template_dir / "neb.in", frames[0], frames[-1]))

        reconstruction = {
            "path_id": "81_neb_%d" % number,
            "historical_neb_out": str(out_path),
            "historical_xyz": str(xyz_path),
            "historical_profile": str(dat_path),
            "historical_sha256": {
                "neb.out": sha256(out_path),
                "sb2te3.xyz": sha256(xyz_path),
                "sb2te3.dat": sha256(dat_path),
            },
            "last_complete_iteration": table["iteration"],
            "images": 10,
            "atoms_per_image": 81,
            "energy_table_eV": table["rows"],
            "profile_rows": profile,
            "restart_path_sha256": sha256(restart_path),
            "gradient_source": "unavailable in historical artifacts; zero placeholders written",
            "optimizer_state_source": "unavailable; old QE Broyden/checkpoint files were not found",
            "qe_action": "fresh SCF force/gradient evaluation is required before path optimization continues",
        }
        (input_dir / "path_reconstruction.json").write_text(
            json.dumps(reconstruction, indent=2, sort_keys=True) + "\n"
        )

        launch_manifest = {
            "schema_version": "2.0",
            "campaign_id": CAMPAIGN_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "path_id": "81_neb_%d" % number,
            "scientific_role": "qe_neb_restart_from_historical_last_saved_path",
            "restart_lineage": {
                "qe_restart_mode": "restart",
                "path_geometry_source": str(xyz_path),
                "path_profile_source": str(dat_path),
                "last_complete_neb_output_source": str(out_path),
                "last_complete_iteration": table["iteration"],
                "old_qe_path_checkpoint_reused": False,
                "old_qe_broyden_or_gradient_state_reused": False,
                "reconstruction_note": (
                    "The original QE sb2te3.path checkpoint is absent. sb2te3.path was reconstructed "
                    "from the historical final ten-image XYZ plus the last complete neb.out energy table; "
                    "gradients are zero placeholders and must be recomputed by QE."
                ),
            },
            "expected_geometry": {
                "images": 10,
                "atoms": 81,
                "mobile_species": "Cr",
                "host_species": "Sb32Te48",
                "coordinate_source_is_mace_preconditioned": False,
            },
            "qe": {
                "version_module": "qe/7.3.1-cpu",
                "ntasks": 4,
                "kpoint_pools": 4,
                "memory": "180G",
                "walltime": "48:00:00",
                "max_seconds": 171000,
                "path_thr_eV_per_A": 0.001,
                "k_points_card": KPOINT_CARD,
                "ds_bohr": 0.0756,
                "k_max": 0.0131,
                "k_min": 0.005,
                "ci_scheme": "no-CI",
                "first_last_opt": True,
            },
            "sources": {
                "historical_dir": str(historical),
                "historical_archive": str(historical_archive),
                "calculator_template_dir": str(template_dir),
                "calculator_template_archive": str(template_archive),
                "historical_sha256": historical_hashes,
                "calculator_template_sha256": template_hashes,
                "restart_path_sha256": sha256(restart_path),
                "patched_input_sha256": sha256(neb_input),
            },
            "wrapper": str(campaign_root / "submit_81atom_neb_path_restart_20261018.slurm"),
            "code_root": str(PIPELINE_ROOT),
        }
        (input_dir / "launch_manifest.json").write_text(
            json.dumps(launch_manifest, indent=2, sort_keys=True) + "\n"
        )
        all_paths.append({
            "path_id": "81_neb_%d" % number,
            "last_complete_iteration": table["iteration"],
            "historical_archive": str(historical_archive),
            "template_archive": str(template_archive),
            "input_dir": str(input_dir),
            "restart_path_sha256": sha256(restart_path),
            "patched_input_sha256": sha256(neb_input),
        })

    historical_relax = sorted(HISTORICAL_ROOT.rglob("relax.out"))
    (campaign_root / "archive" / "NO_relax.out_FOUND.txt").write_text(
        "Preflight evidence for the authoritative historical 81-atom tree.\n"
        "searched_root=%s\npattern=relax.out\nmatches=%d\n"
        "No relax.out existed; historical neb.out/sb2te3.xyz/sb2te3.dat were copied instead.\n"
        % (HISTORICAL_ROOT, len(historical_relax))
    )
    (campaign_root / "archive" / "README.txt").write_text(
        "Additive, copy-only archive for the corrected historical-path QE restart.\n"
        "historical/ preserves the full five source directories.\n"
        "calculator_templates/ preserves only the per-path QE calculator templates.\n"
        "The production coordinates come from historical sb2te3.xyz, never from MACE-preconditioned coordinates.\n"
        "The original QE sb2te3.path/Broyden state was absent, so the generated path files are reconstructed and use zero gradient placeholders.\n"
    )
    manifest = {
        "schema_version": "2.0",
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_root": str(HISTORICAL_ROOT),
        "template_root": str(TEMPLATE_ROOT),
        "pipeline_root": str(PIPELINE_ROOT),
        "previous_campaign_root": PREVIOUS_CAMPAIGN_ROOT,
        "k_points_card": KPOINT_CARD,
        "relax_out_matches_in_historical_root": len(historical_relax),
        "restart_type": "path-level QE restart from reconstructed historical final path",
        "exact_original_qe_checkpoint_available": False,
        "archive_policy": "copy-only; no overwrite of old campaign roots",
        "paths": all_paths,
    }
    (campaign_root / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"campaign_root": str(campaign_root), "paths": all_paths}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
