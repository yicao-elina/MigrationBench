#!/usr/bin/env python3
"""Generate isolated QE SCF inputs from fixed local extxyz snapshots."""

import argparse
import hashlib
import json
import re
from pathlib import Path


LATTICE_RE = re.compile(r'Lattice="([^"]+)"')
STEP_RE = re.compile(r"(?:^|\s)step=(-?\d+)(?:\s|$)")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_extxyz(path):
    lines = path.read_text().splitlines()
    nat = int(lines[0])
    comment = lines[1]
    rows = [line.split() for line in lines[2:2 + nat]]
    lattice = [float(x) for x in LATTICE_RE.search(comment).group(1).split()]
    positions = [(row[0], [float(x) for x in row[1:4]]) for row in rows]
    return nat, comment, lattice, positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": {
            "code": "Quantum ESPRESSO pw.x 7.3.1-cpu",
            "calculation": "scf",
            "fixed_positions_and_cell": True,
            "functional": "PBE + DFT-D3",
            "pseudopotential_dir": "/data/pclancy3/yi/ONCVPseudoPack/SG15_v1.2/UPF/PBE",
            "pseudopotentials": {
                "Cr": "Cr_PBE_FR.SG15v1.2.UPF",
                "Sb": "Sb_PBE_FR.SG15v1.2.UPF",
                "Te": "Te_PBE_FR.SG15v1.2.UPF",
            },
            "ecutwfc_Ry": 70,
            "ecutrho_Ry": 280,
            "kpoints": "2 2 2 1 1 1",
            "occupations": "smearing",
            "smearing": "cold",
            "degauss_Ry": 0.0146997236,
            "nspin": 2,
            "soc": False,
            "conv_thr_Ry": 1e-6,
            "electron_maxstep": 400,
            "tprnfor": True,
            "tstress": True,
        },
        "inputs": [],
    }
    for source in sorted(args.input_dir.glob("*_same_site81_ordered.extxyz")):
        nat, comment, lattice, positions = read_extxyz(source)
        step = int(STEP_RE.search(comment).group(1))
        run_dir = args.output_dir / ("step_%06d" % step)
        run_dir.mkdir(parents=True, exist_ok=True)
        input_path = run_dir / "scf.in"
        by_symbol = {symbol: [] for symbol in ("Cr", "Sb", "Te")}
        for symbol, position in positions:
            by_symbol[symbol].append(position)
        lines = [
            "&CONTROL",
            "  calculation = 'scf'",
            "  prefix = 'mb_phase3_step_%06d'" % step,
            "  outdir = './out'",
            "  pseudo_dir = '/data/pclancy3/yi/ONCVPseudoPack/SG15_v1.2/UPF/PBE'",
            "  verbosity = 'high'",
            "/",
            "&SYSTEM",
            "  ibrav = 0",
            "  nat = %d" % nat,
            "  ntyp = 3",
            "  ecutwfc = 70",
            "  ecutrho = 280",
            "  occupations = 'smearing'",
            "  smearing = 'cold'",
            "  degauss = 0.0146997236",
            "  vdw_corr = 'DFT-D3'",
            "  nspin = 2",
            "  starting_magnetization(1) = 0.5",
            "/",
            "&ELECTRONS",
            "  conv_thr = 1.0d-6",
            "  electron_maxstep = 400",
            "  mixing_beta = 0.4",
            "/",
            "ATOMIC_SPECIES",
            "Cr 51.99610 Cr_PBE_FR.SG15v1.2.UPF",
            "Sb 121.76000 Sb_PBE_FR.SG15v1.2.UPF",
            "Te 127.60000 Te_PBE_FR.SG15v1.2.UPF",
            "K_POINTS automatic",
            "2 2 2 1 1 1",
            "CELL_PARAMETERS angstrom",
        ]
        for i in range(0, 9, 3):
            lines.append("  %.14f %.14f %.14f" % tuple(lattice[i:i + 3]))
        lines.append("ATOMIC_POSITIONS angstrom")
        for symbol in ("Cr", "Sb", "Te"):
            for position in by_symbol[symbol]:
                lines.append("  %s %.12f %.12f %.12f" % (symbol, *position))
        lines.append("")
        input_path.write_text("\n".join(lines))
        manifest["inputs"].append({
            "step": step,
            "source_extxyz": str(source),
            "source_sha256": sha256(source),
            "qe_input": str(input_path),
            "qe_input_sha256": sha256(input_path),
            "natoms": nat,
            "source_comment": comment,
        })
    (args.output_dir / "qe_input_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
