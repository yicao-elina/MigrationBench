#!/usr/bin/env python3
"""Smoke-test the Revision1 MigrationBench pipeline end to end."""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

from ase import Atoms
from ase.io import write


ROOT = Path(__file__).resolve().parents[2]
SMOKE = Path(os.environ.get("MIGRATIONBENCH_OUTPUT_ROOT", ROOT / "data_processed" / "pipeline_smoke")).resolve()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def make_fixture() -> tuple[Path, Path]:
    SMOKE.mkdir(parents=True, exist_ok=True)
    images = []
    for i in range(5):
        x = 1.8 + 0.15 * i
        atoms = Atoms(
            "Cu2",
            positions=[(0.0, 0.0, 0.0), (x, 0.0, 0.0)],
            cell=[8.0, 8.0, 8.0],
            pbc=[True, True, True],
        )
        atoms.info["energy"] = float(i * (4 - i)) * 0.01
        atoms.info["reaction_coordinate"] = i / 4
        atoms.info["path_name"] = "smoke_cu"
        atoms.info["neb_image"] = i
        images.append(atoms)
    image_path = SMOKE / "smoke_images.extxyz"
    write(image_path, images)

    engine = SMOKE / "qe_engine_template.in"
    engine.write_text(
        """&CONTROL
  calculation = 'scf',
  prefix = 'smoke',
  pseudo_dir = './pseudo',
  outdir = './out'
/
&SYSTEM
  ibrav = 0,
  nat = 2,
  ntyp = 1,
  ecutwfc = 30,
  ecutrho = 240
/
&ELECTRONS
  conv_thr = 1.0d-6
/
ATOMIC_SPECIES
Cu 63.546 Cu.pbe-dn-kjpaw_psl.1.0.0.UPF
K_POINTS gamma
"""
    )
    return image_path, engine


def main() -> None:
    image_path, engine = make_fixture()
    mlff_dir = SMOKE / "mlff_emt"
    run(
        [
            sys.executable,
            "scripts/migrationbench/run_mlff_neb.py",
            "--images",
            str(image_path),
            "--out-dir",
            str(mlff_dir),
            "--calculator",
            "emt",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--fmax",
            "10.0",
        ]
    )
    qe_in = SMOKE / "qe_from_mlff.neb.in"
    run(
        [
            sys.executable,
            "scripts/migrationbench/qe_neb_from_images.py",
            "--images",
            str(mlff_dir / "mlff_neb_images.extxyz"),
            "--engine-template",
            str(engine),
            "--output",
            str(qe_in),
            "--nstep-path",
            "5",
            "--path-thr",
            "0.5",
        ]
    )
    dataset_dir = SMOKE / "hf_dataset"
    run(
        [
            sys.executable,
            "scripts/migrationbench/migrationbench_dataset.py",
            "build",
            "--images",
            str(mlff_dir / "mlff_neb_images.extxyz"),
            "--out-dir",
            str(dataset_dir),
            "--pathway-id",
            "smoke_Cu2",
            "--path-label",
            "smoke",
            "--family",
            "smoke",
            "--protocol",
            "mlff_pre_neb",
            "--calculator-family",
            "MLFF",
            "--calculator-label",
            "EMT",
            "--method",
            "EMT",
            "--convergence-status",
            "converged",
            "--acceptance-notes",
            "smoke test only",
        ]
    )

    text = qe_in.read_text()
    assert "BEGIN_POSITIONS" in text
    assert text.count("INTERMEDIATE_IMAGE") == 3
    for table in ["configurations", "calculations", "neb_paths", "derived_barriers"]:
        assert (dataset_dir / f"{table}.jsonl").exists()
        assert (dataset_dir / f"{table}.csv").exists()
        assert (dataset_dir / f"{table}.parquet").exists() or (dataset_dir / f"{table}.parquet.error.txt").exists()
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
