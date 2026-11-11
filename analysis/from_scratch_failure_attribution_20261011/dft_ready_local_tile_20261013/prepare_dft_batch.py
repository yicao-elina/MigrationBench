from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_source(row: pd.Series) -> Path:
    for column in ("local_extxyz", "prepared_extxyz"):
        value = str(row.get(column, ""))
        if value and value != "nan":
            path = Path(value)
            if path.is_file():
                return path
    raise FileNotFoundError(f"no local candidate source for {row['candidate_id']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--builder", default=str(Path(__file__).with_name("build_dft_ready_local.py")))
    ap.add_argument("--mode", choices=["local_periodic_shell", "primitive_tile"], default="local_periodic_shell")
    ap.add_argument("--target-atoms", type=int, default=120)
    ap.add_argument("--repeat", nargs=3, type=int, default=[5, 5, 1])
    ap.add_argument("--margin-A", type=float, default=5.0)
    args = ap.parse_args()

    selection_path = Path(args.selection)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    qe_inputs = output / "qe_inputs"
    source_inputs = output / "source_inputs"
    qe_inputs.mkdir(exist_ok=True)
    source_inputs.mkdir(exist_ok=True)
    selection = pd.read_csv(selection_path)
    if len(selection) == 0 or len(selection) > 20:
        raise SystemExit(f"selection must contain 1-20 candidates, found {len(selection)}")
    if selection["candidate_id"].duplicated().any():
        raise SystemExit("duplicate candidate_id in selection")

    index = []
    rows = []
    for _, row in selection.sort_values("batch_order").iterrows():
        candidate_id = str(row["candidate_id"])
        source = resolve_source(row)
        copied_source = source_inputs / f"{candidate_id}.extxyz"
        shutil.copy2(source, copied_source)
        candidate_dir = output / "built" / candidate_id
        command = [
            sys.executable,
            str(Path(args.builder).resolve()),
            "--input", str(copied_source),
            "--output-dir", str(candidate_dir),
            "--mode", args.mode,
            "--target-atoms", str(args.target_atoms),
            "--margin-A", str(args.margin_A),
        ]
        if args.mode == "primitive_tile":
            command += ["--repeat", *[str(x) for x in args.repeat]]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            raise SystemExit(f"{candidate_id}: DFT-ready geometry gate failed\n{completed.stdout}\n{completed.stderr}")
        manifest = json.loads((candidate_dir / "manifest.json").read_text())
        qe = Path(manifest["qe_input"])
        final_qe = qe_inputs / f"{candidate_id}.scf.in"
        shutil.copy2(qe, final_qe)
        source_sha = sha256(copied_source)
        if source_sha != str(manifest["source_sha256"]):
            raise SystemExit(f"{candidate_id}: source hash changed during preparation")
        record = row.to_dict()
        record.update({
            "candidate_id": candidate_id,
            "source_input": str(copied_source),
            "source_input_sha256": source_sha,
            "dft_ready_extxyz": manifest["extxyz"],
            "dft_ready_extxyz_sha256": manifest["extxyz_sha256"],
            "qe_input": str(final_qe),
            "qe_input_sha256": sha256(final_qe),
            "dft_ready_mode": manifest["mode"],
            "dft_ready_natoms": manifest["natoms"],
            "dft_ready_pbc": json.dumps(manifest["pbc"]),
            "dft_ready_min_distance_A": manifest["min_distance_A"],
            "dft_ready_species": json.dumps(manifest["species"], sort_keys=True),
            "dft_ready_status": manifest["status"],
            "dft_scientific_status": "PREPARED_ONLY",
        })
        rows.append(record)
        index.append({"candidate_id": candidate_id, "batch_order": int(row["batch_order"]), "qe_input": str(final_qe)})

    pd.DataFrame(rows).to_csv(output / "dft_ready_batch_manifest.csv", index=False)
    (output / "selected_input_index.json").write_text(json.dumps(index, indent=2) + "\n")
    meta = {
        "schema": "migrationbench_dft_ready_selected_batch_v1",
        "status": "PREPARED_ONLY",
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "n_selected": len(rows),
        "mode": args.mode,
        "target_atoms": args.target_atoms if args.mode == "local_periodic_shell" else None,
        "repeat": args.repeat if args.mode == "primitive_tile" else None,
        "pbc_preserved": True,
        "reconstruction_boundary": "local periodic shell or selected primitive tile is a PBC-preserving reconstruction; it is not exact full-cell parity until convergence and parity checks pass",
        "no_dft_submitted": True,
    }
    (output / "dft_ready_batch_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
