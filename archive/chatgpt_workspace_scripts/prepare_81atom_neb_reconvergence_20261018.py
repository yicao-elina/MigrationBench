#!/usr/bin/env python3
"""Create an additive, provenance-preserving Rockfish campaign root."""

import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


HISTORICAL_ROOT = Path(
    "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/2.2-layer-diffusion/2D_421_Diffusion_traj"
)
PRECONDITIONED_ROOT = Path(
    "/scratch16/pclancy3/yi/revision1_migrationbench_inputs"
)
DEFAULT_CAMPAIGN_ROOT = Path(
    "/scratch16/pclancy3/yi/cr_81atom_neb_reconvergence_20261018"
)
PIPELINE_ROOT = Path("/scratch16/pclancy3/yi/revision1_migrationbench_pipeline")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_files(source: Path, destination: Path) -> Dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied = {}  # type: Dict[str, str]
    for item in sorted(source.iterdir()):
        if item.is_file():
            target = destination / item.name
            shutil.copy2(item, target)
            copied[item.name] = sha256(target)
    return copied


def main() -> int:
    campaign_root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CAMPAIGN_ROOT
    if campaign_root.exists():
        raise SystemExit(f"refusing to overwrite existing campaign root: {campaign_root}")

    for required in (HISTORICAL_ROOT, PRECONDITIONED_ROOT):
        if not required.is_dir():
            raise SystemExit(f"missing required source directory: {required}")

    historical_relax = sorted(HISTORICAL_ROOT.rglob("relax.out"))
    campaign_root.mkdir(parents=True)
    (campaign_root / "archive" / "historical").mkdir(parents=True)
    (campaign_root / "archive" / "preconditioned").mkdir(parents=True)
    (campaign_root / "inputs").mkdir()
    (campaign_root / "runs").mkdir()

    paths = []  # type: List[Dict[str, object]]
    for number in range(1, 6):
        historical = HISTORICAL_ROOT / f"neb_{number}"
        preconditioned = PRECONDITIONED_ROOT / f"mb_qe81_neb_{number}_mace_r1_s42"
        if not historical.is_dir() or not preconditioned.is_dir():
            raise SystemExit(f"missing path source for neb_{number}")
        if not (historical / "neb.out").is_file():
            raise SystemExit(f"missing historical neb.out for neb_{number}")
        if not (preconditioned / "neb.in").is_file():
            raise SystemExit(f"missing preconditioned neb.in for neb_{number}")

        historical_archive = campaign_root / "archive" / "historical" / f"neb_{number}"
        preconditioned_archive = campaign_root / "archive" / "preconditioned" / f"neb_{number}"
        historical_hashes = copy_files(historical, historical_archive)
        preconditioned_hashes = copy_files(preconditioned, preconditioned_archive)

        input_dir = campaign_root / "inputs" / f"neb_{number}"
        input_dir.mkdir()
        for item in sorted(preconditioned.iterdir()):
            if item.is_file():
                shutil.copy2(item, input_dir / item.name)

        neb_input = input_dir / "neb.in"
        text = neb_input.read_text()
        checks = (
            "restart_mode = 'from_scratch'",
            "nstep_path = 1000",
            "path_thr = 0.03",
            "nat = 81",
        )
        missing = [needle for needle in checks if needle not in text]
        if missing:
            raise SystemExit(f"neb_{number} input failed preflight: {missing}")
        patched, replacements = re.subn(
            r"max_seconds\s*=\s*250000",
            "max_seconds = 171000",
            text,
            count=1,
        )
        if replacements != 1:
            raise SystemExit(f"neb_{number} input has no unique max_seconds=250000")
        neb_input.write_text(patched)

        source_manifest = json.loads((preconditioned / "neb.manifest.json").read_text())
        launch_manifest = {
            "schema_version": "1.0",
            "campaign_id": "cr_81atom_neb_reconvergence_20261018",
            "path_id": f"81_neb_{number}",
            "scientific_role": "qe_refinement_from_mace_preconditioned_explicit_image_band",
            "restart_lineage": {
                "qe_restart_mode": "from_scratch",
                "initial_band_source": str(preconditioned / "neb.in"),
                "historical_neb_output": str(historical / "neb.out"),
                "note": "Prior MACE-relaxed geometry is the new QE band initial condition; no old QE checkpoint is reused.",
            },
            "expected_geometry": {"images": 10, "atoms": 81, "mobile_species": "Cr", "host_species": "Sb32Te48"},
            "qe": {
                "version_module": "qe/7.3.1-cpu",
                "ntasks": 4,
                "kpoint_pools": 4,
                "memory": "180G",
                "walltime": "48:00:00",
                "max_seconds": 171000,
                "path_thr_eV_per_A": 0.03,
                "production_degauss_Ry": 0.005,
            },
            "sources": {
                "historical_dir": str(historical),
                "preconditioned_dir": str(preconditioned),
                "historical_archive": str(historical_archive),
                "preconditioned_archive": str(preconditioned_archive),
                "historical_sha256": historical_hashes,
                "preconditioned_sha256": preconditioned_hashes,
                "patched_input_sha256": sha256(neb_input),
            },
            "source_manifest_snapshot": source_manifest,
            "wrapper": str(campaign_root / "submit_81atom_neb_reconvergence_20261018.slurm"),
            "code_root": str(PIPELINE_ROOT),
        }
        (input_dir / "launch_manifest.json").write_text(json.dumps(launch_manifest, indent=2, sort_keys=True) + "\n")
        paths.append({
            "path_id": f"81_neb_{number}",
            "historical_dir": str(historical),
            "preconditioned_dir": str(preconditioned),
            "historical_archive": str(historical_archive),
            "preconditioned_archive": str(preconditioned_archive),
            "input_dir": str(input_dir),
            "historical_sha256": historical_hashes,
            "preconditioned_sha256": preconditioned_hashes,
            "patched_input_sha256": sha256(neb_input),
        })

    no_relax = campaign_root / "archive" / "NO_relax.out_FOUND.txt"
    no_relax.write_text(
        "Preflight evidence for the authoritative historical 81-atom tree.\n"
        f"searched_root={HISTORICAL_ROOT}\n"
        "pattern=relax.out\n"
        f"matches={len(historical_relax)}\n"
        "Therefore no relax.out was renamed or fabricated. Historical neb.out files and source paths were archived instead.\n"
    )
    (campaign_root / "archive" / "README.txt").write_text(
        "This archive is additive and immutable for this campaign.\n"
        "historical/ preserves the five prior neb.out/path records.\n"
        "preconditioned/ preserves the five original MACE-preconditioned QE inputs and manifests.\n"
        "NO_relax.out_FOUND.txt records that no relax.out existed in the authoritative tree.\n"
    )
    manifest = {
        "schema_version": "1.0",
        "campaign_id": "cr_81atom_neb_reconvergence_20261018",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_root": str(HISTORICAL_ROOT),
        "preconditioned_root": str(PRECONDITIONED_ROOT),
        "pipeline_root": str(PIPELINE_ROOT),
        "relax_out_matches_in_historical_root": len(historical_relax),
        "archive_policy": "copy-only; no overwrite/cancel/reuse of old campaign roots",
        "paths": paths,
    }
    (campaign_root / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"campaign_root": str(campaign_root), "paths": 5, "relax_out_matches": len(historical_relax)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
