from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list


PSEUDOS = {
    "Cr": (51.99610, "Cr_PBE_FR.SG15v1.2.UPF"),
    "Sb": (121.76000, "Sb_PBE_FR.SG15v1.2.UPF"),
    "Te": (127.60000, "Te_PBE_FR.SG15v1.2.UPF"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def min_distance(atoms) -> tuple[float, str]:
    i, j, d = neighbor_list("ijd", atoms, cutoff=4.0, self_interaction=False)
    if len(d) == 0:
        return float("nan"), "none"
    k = int(np.argmin(d))
    symbols = atoms.get_chemical_symbols()
    return float(d[k]), "-".join(sorted((symbols[int(i[k])], symbols[int(j[k])])))


def species_counts(atoms) -> dict[str, int]:
    return {symbol: int(atoms.get_chemical_symbols().count(symbol)) for symbol in sorted(set(atoms.get_chemical_symbols()))}


def make_tile(atoms, repeat: np.ndarray, center_index: int | None):
    frac = atoms.get_scaled_positions(wrap=True)
    tile_id = np.floor(frac * repeat).astype(int)
    tile_id = np.minimum(np.maximum(tile_id, 0), repeat - 1)
    if center_index is None:
        chosen = np.array([repeat[0] // 2, repeat[1] // 2, 0], dtype=int)
    else:
        chosen = tile_id[center_index].copy()
    keep = np.all(tile_id == chosen[None, :], axis=1)
    if not np.any(keep):
        raise ValueError(f"no atoms in tile {chosen.tolist()}")
    selected = atoms[keep]
    new_frac = frac[keep] * repeat[None, :] - chosen[None, :]
    new_cell = np.diag(1.0 / repeat) @ atoms.cell.array
    selected.set_cell(new_cell, scale_atoms=False)
    selected.set_positions(new_frac @ new_cell)
    selected.set_pbc([True, True, True])
    selected.wrap()
    return selected, {"tile_repeat": repeat.tolist(), "tile_index": chosen.tolist(),
                      "source_selected_count": int(np.sum(keep)), "reconstruction_status": "RECONSTRUCTION"}


def make_shell(atoms, target_atoms: int, center_index: int | None, margin_A: float):
    symbols = atoms.get_chemical_symbols()
    if center_index is None:
        center_index = int(np.where(np.asarray(symbols) == "Cr")[0][0])
    indices = np.arange(len(atoms), dtype=int)
    distances = atoms.get_distances(center_index, indices, mic=True)
    order = np.argsort(distances)
    keep = order[:target_atoms]
    vectors = atoms.get_distances(center_index, keep, mic=True, vector=True)
    lo = np.min(vectors, axis=0)
    hi = np.max(vectors, axis=0)
    span = hi - lo
    cell_lengths = span + 2.0 * margin_A
    if np.any(cell_lengths <= 2.0 * margin_A):
        raise ValueError("local shell cell is degenerate")
    positions = vectors - lo[None, :] + margin_A
    selected = atoms[keep]
    selected.set_cell(np.diag(cell_lengths))
    selected.set_positions(positions)
    selected.set_pbc([True, True, True])
    return selected, {"center_index": int(center_index), "target_atoms": int(target_atoms),
                      "selected_count": int(len(selected)), "margin_A": float(margin_A),
                      "reconstruction_status": "RECONSTRUCTION",
                      "boundary_model": "orthorhombic local periodic shell; not exact parity with the 2050-atom cell"}


def qe_input(atoms, prefix: str, pseudo_dir: str) -> str:
    lines = [
        f"! DFT-ready local reconstruction; prefix={prefix}",
        "! This input is a smoke-tested template; convergence and physical parity remain pending.",
        "&CONTROL", "  calculation = 'scf'", "  restart_mode = 'from_scratch'", "  outdir = './out/'",
        f"  prefix = '{prefix}'", f"  pseudo_dir = '{pseudo_dir}'", "  verbosity = 'high'",
        "  tprnfor = .true.", "  tstress = .true.", "  max_seconds = 171000", "/", "",
        "&SYSTEM", "  ibrav = 0", f"  nat = {len(atoms)}", "  ntyp = 3", "  ecutwfc = 70",
        "  ecutrho = 280", "  occupations = 'smearing'", "  smearing = 'cold'",
        "  degauss = 1.4699723600d-02", "  vdw_corr = 'DFT-D3'", "/", "",
        "&ELECTRONS", "  conv_thr = 1.0000000000d-06", "  electron_maxstep = 400", "  mixing_beta = 0.4", "/", "",
        "ATOMIC_SPECIES",
    ]
    for symbol in ("Cr", "Sb", "Te"):
        mass, pseudo = PSEUDOS[symbol]
        lines.append(f"  {symbol:<2} {mass:10.5f}  {pseudo}")
    lines += ["", "K_POINTS gamma", "", "CELL_PARAMETERS angstrom"]
    lines += ["  " + " ".join(f"{x:16.10f}" for x in row) for row in atoms.cell.array]
    lines += ["", "ATOMIC_POSITIONS angstrom"]
    for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions):
        lines.append(f"{symbol:<2} " + " ".join(f"{x:16.10f}" for x in position))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", choices=["primitive_tile", "local_periodic_shell"], required=True)
    ap.add_argument("--repeat", nargs=3, type=int, default=[5, 5, 1])
    ap.add_argument("--target-atoms", type=int, default=120)
    ap.add_argument("--center-index", type=int)
    ap.add_argument("--margin-A", type=float, default=5.0)
    ap.add_argument("--pseudo-dir", default="/data/pclancy3/yi/ONCVPseudoPack/SG15_v1.2/UPF/PBE/")
    args = ap.parse_args()
    source = Path(args.input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atoms = read(source, index=-1)
    if len(atoms) != 2050 or not bool(np.all(atoms.pbc)):
        raise SystemExit(f"source gate failed: natoms={len(atoms)} pbc={atoms.pbc.tolist()}")
    if args.mode == "primitive_tile":
        reduced, mode_meta = make_tile(atoms, np.asarray(args.repeat, dtype=int), args.center_index)
    else:
        reduced, mode_meta = make_shell(atoms, args.target_atoms, args.center_index, args.margin_A)
    if len(reduced) < 50 or len(reduced) > 200:
        raise SystemExit(f"output atom-count gate failed: {len(reduced)}")
    min_d, pair = min_distance(reduced)
    if not np.isfinite(min_d) or min_d < 0.8:
        raise SystemExit(f"output geometry gate failed: min_distance={min_d} pair={pair}")
    if not np.isfinite(reduced.positions).all() or not np.isfinite(reduced.cell.array).all():
        raise SystemExit("output finite-value gate failed")
    prefix = f"dft_ready_{args.mode}"
    extxyz = output / f"{prefix}.extxyz"
    qe = output / f"{prefix}.scf.in"
    write(extxyz, reduced, format="extxyz")
    qe.write_text(qe_input(reduced, prefix, args.pseudo_dir))
    meta = {
        "schema": "migrationbench_dft_ready_local_reconstruction_v1",
        "status": "RECONSTRUCTION",
        "source": str(source), "source_sha256": sha256(source),
        "mode": args.mode, "natoms": len(reduced), "pbc": reduced.pbc.tolist(),
        "cell_A": reduced.cell.array.tolist(), "species": species_counts(reduced),
        "min_distance_A": min_d, "min_distance_pair": pair,
        "mode_metadata": mode_meta, "extxyz": str(extxyz), "qe_input": str(qe),
        "extxyz_sha256": sha256(extxyz), "qe_input_sha256": sha256(qe),
        "pseudo_dir": args.pseudo_dir,
        "scientific_gate": "DFT convergence, pseudopotential, k-point, and parity checks are pending; no DFT claim is made",
    }
    (output / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
