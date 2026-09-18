#!/usr/bin/env python3
"""Build auditable MigrationBench dataset tables from NEB artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from ase.io import read


def stable_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def hash_obj(value) -> str:
    return hashlib.sha256(stable_json(value).encode()).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atoms_record(atoms, source_uri: str, source_sha: str | None, system_id: str, pathway_id: str | None, image_index: int | None) -> dict:
    symbols = atoms.get_chemical_symbols()
    base = {
        "symbols": symbols,
        "cell": atoms.get_cell().array.round(10).tolist(),
        "pbc": [bool(x) for x in atoms.get_pbc()],
        "positions": atoms.get_positions().round(10).tolist(),
    }
    structure_hash = hash_obj(base)
    config_hash = hash_obj({**base, "pathway_id": pathway_id, "image_index": image_index})
    return {
        "configuration_id": f"CO_{config_hash[:32]}",
        "structure_hash": f"ST_{structure_hash[:32]}",
        "system_id": system_id,
        "composition": atoms.get_chemical_formula(),
        "elements": sorted(set(symbols)),
        "nsites": len(atoms),
        "cell": base["cell"],
        "pbc": base["pbc"],
        "positions": base["positions"],
        "source_uri": source_uri,
        "source_sha256": source_sha,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_json": stable_json({"atoms_info": dict(atoms.info), "arrays": list(atoms.arrays.keys())}),
    }


def calc_record(atoms, config_id: str, args: argparse.Namespace, source_sha: str | None, image_index: int) -> dict:
    info = dict(atoms.info)
    energy = info.get("energy")
    if energy is None:
        try:
            energy = atoms.get_potential_energy()
        except Exception:
            energy = None
    forces = None
    force_status = "absent"
    if "forces" in atoms.arrays:
        forces = atoms.arrays["forces"].tolist()
        flat = [abs(v) for row in forces for v in row]
        force_status = "fabricated_zero_quarantined" if flat and max(flat) == 0 else "real"

    payload = {
        "configuration_id": config_id,
        "pathway_id": args.pathway_id,
        "image_index": image_index,
        "calculator_label": args.calculator_label,
        "protocol": args.protocol,
        "source_sha256": source_sha,
    }
    calc_id = f"CA_{hash_obj(payload)[:32]}"
    return {
        "calculation_id": calc_id,
        "configuration_id": config_id,
        "pathway_id": args.pathway_id,
        "image_index": image_index,
        "reaction_coordinate": info.get("reaction_coordinate"),
        "calculator_family": args.calculator_family,
        "calculator_label": args.calculator_label,
        "model_checkpoint_uri": args.model_checkpoint_uri,
        "method": args.method,
        "energy_eV": energy,
        "forces_eV_A": forces,
        "stress": info.get("stress"),
        "force_label_status": force_status,
        "convergence_status": args.convergence_status,
        "run_id": args.run_id,
        "raw_log_uri": args.raw_log_uri,
        "raw_log_sha256": sha256_file(Path(args.raw_log_uri)) if args.raw_log_uri and Path(args.raw_log_uri).exists() else None,
        "software": args.software,
        "metadata_json": stable_json({"atoms_info": info, "source_sha256": source_sha}),
    }


def read_acceptance(path):
    if path is None:
        return None
    resolved = Path(path).resolve()
    return json.loads(resolved.read_text()) if resolved.is_file() else None


def manuscript_gate(args):
    reasons = []
    path_acceptance = read_acceptance(args.path_acceptance)
    endpoint_acceptance = read_acceptance(args.endpoint_acceptance)
    calculator_acceptance = read_acceptance(args.calculator_acceptance)
    runtime_provenance = read_acceptance(getattr(args, "runtime_provenance", None))
    if args.protocol != "dft_neb":
        reasons.append("protocol_is_not_dft_neb")
    if args.convergence_status != "converged":
        reasons.append("path_is_not_converged")
    if not path_acceptance or path_acceptance.get("status") != "accepted_final_reference":
        reasons.append("path_acceptance_missing_or_failed")
    elif path_acceptance.get("pathway_id") != args.pathway_id:
        reasons.append("path_acceptance_pathway_mismatch")
    if not endpoint_acceptance or endpoint_acceptance.get("status") != "accepted":
        reasons.append("endpoint_acceptance_missing_or_failed")
    if not calculator_acceptance or calculator_acceptance.get("status") != "pass":
        reasons.append("calculator_acceptance_missing_or_failed")
    elif calculator_acceptance.get("accepted_calculator_identity") != args.calculator_identity:
        reasons.append("calculator_identity_mismatch")
    runtime_binaries = (runtime_provenance or {}).get("binaries", {})
    runtime_qe_available = any(
        runtime_binaries.get(name, {}).get("exists") for name in ("pw.x", "neb.x")
    )
    if not runtime_provenance or not runtime_qe_available:
        reasons.append("runtime_provenance_missing_or_failed")
    return not reasons, reasons, {
        "path_acceptance_sha256": sha256_file(Path(args.path_acceptance).resolve()) if args.path_acceptance else None,
        "endpoint_acceptance_sha256": sha256_file(Path(args.endpoint_acceptance).resolve()) if args.endpoint_acceptance else None,
        "calculator_acceptance_sha256": sha256_file(Path(args.calculator_acceptance).resolve()) if args.calculator_acceptance else None,
        "runtime_provenance_sha256": sha256_file(Path(args.runtime_provenance).resolve()) if getattr(args, "runtime_provenance", None) else None,
    }


def derive_barrier(calcs: list[dict], args: argparse.Namespace) -> dict:
    energies = [row["energy_eV"] for row in calcs]
    if any(e is None for e in energies):
        barrier = None
        energy_reasons = ["missing_energy"]
    else:
        rel = [float(e) - float(energies[0]) for e in energies]
        barrier = max(rel)
        energy_reasons = []
    gate_pass, gate_reasons, acceptance_hashes = manuscript_gate(args)
    exclusion_reasons = energy_reasons + gate_reasons
    manuscript_allowed = barrier is not None and gate_pass
    exclusion = None if manuscript_allowed else ";".join(exclusion_reasons)

    payload = {
        "pathway_id": args.pathway_id,
        "calculator_label": args.calculator_label,
        "protocol": args.protocol,
        "calculation_ids": [row["calculation_id"] for row in calcs],
    }
    reference = args.reference_barrier_eV
    return {
        "barrier_result_id": f"BR_{hash_obj(payload)[:32]}",
        "pathway_id": args.pathway_id,
        "protocol": args.protocol,
        "calculator_label": args.calculator_label,
        "barrier_eV": barrier,
        "reference_barrier_eV": reference,
        "error_eV": None if barrier is None or reference is None else barrier - reference,
        "source_calculation_ids": [row["calculation_id"] for row in calcs],
        "convergence_status": args.convergence_status,
        "manuscript_allowed": manuscript_allowed,
        "exclusion_reason": exclusion,
        "metadata_json": stable_json({
            "barrier_convention": "peak relative to first image",
            "calculator_identity": args.calculator_identity,
            "acceptance_artifact_hashes": acceptance_hashes,
            "manuscript_gate_reasons": exclusion_reasons,
        }),
    }


def build(args: argparse.Namespace) -> dict:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.images).resolve()
    source_sha = sha256_file(source)
    images = read(source, index=":")
    configs, calcs = [], []
    for i, atoms in enumerate(images):
        cfg = atoms_record(atoms, str(source), source_sha, args.system_id, args.pathway_id, i)
        configs.append(cfg)
        calcs.append(calc_record(atoms, cfg["configuration_id"], args, source_sha, i))

    neb_path = {
        "pathway_id": args.pathway_id,
        "system_id": args.system_id,
        "path_label": args.path_label,
        "family": args.family,
        "n_images": len(images),
        "endpoint_configuration_ids": [configs[0]["configuration_id"], configs[-1]["configuration_id"]],
        "reference_calculation_ids": [row["calculation_id"] for row in calcs] if args.protocol == "dft_neb" else [],
        "barrier_eV": None,
        "reverse_barrier_eV": None,
        "barrier_convention": "peak relative to first image",
        "status": args.convergence_status,
        "acceptance_notes": args.acceptance_notes,
        "metadata_json": stable_json({"source": str(source), "protocol": args.protocol}),
    }
    barrier = derive_barrier(calcs, args)
    neb_path["barrier_eV"] = barrier["barrier_eV"]

    tables = {
        "configurations": pd.DataFrame(configs),
        "calculations": pd.DataFrame(calcs),
        "neb_paths": pd.DataFrame([neb_path]),
        "derived_barriers": pd.DataFrame([barrier]),
    }
    parquet_status = {}
    for name, df in tables.items():
        df.to_json(out_dir / f"{name}.jsonl", orient="records", lines=True)
        df.to_csv(out_dir / f"{name}.csv", index=False)
        try:
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
            parquet_status[name] = "ok"
        except Exception as exc:
            parquet_status[name] = f"skipped: {type(exc).__name__}: {exc}"
            (out_dir / f"{name}.parquet.error.txt").write_text(parquet_status[name])

    card = f"""# MigrationBench

This dataset stores migration-barrier benchmark artifacts with lossless provenance.

## Tables

- `configurations`: structures and source hashes.
- `calculations`: energies, forces when real, convergence status, raw-log provenance.
- `neb_paths`: ordered NEB paths and canonical barrier metadata.
- `derived_barriers`: protocol-specific barriers and manuscript inclusion gates.

## Current Build Note

Generated from `{source}` with source sha256 `{source_sha}`.
Protocol: `{args.protocol}`. Convergence status: `{args.convergence_status}`.

Rows with `force_label_status=fabricated_zero_quarantined`, placeholder energies, digitized PNG values,
or unconverged/diverged status are retained for audit but excluded from manuscript-allowed derived barriers.
"""
    (out_dir / "README.md").write_text(card)
    manifest = {
        "out_dir": str(out_dir),
        "tables": {name: len(df) for name, df in tables.items()},
        "parquet_status": parquet_status,
        "source_sha256": source_sha,
        "barrier_eV": barrier["barrier_eV"],
        "manuscript_allowed": barrier["manuscript_allowed"],
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("build", nargs="?")
    p.add_argument("--images", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--system-id", default="Cr-doped Sb2Te3")
    p.add_argument("--pathway-id", required=True)
    p.add_argument("--path-label", required=True)
    p.add_argument("--family", default="unknown")
    p.add_argument("--protocol", default="dft_neb")
    p.add_argument("--calculator-family", default="DFT")
    p.add_argument("--calculator-label", default="Quantum ESPRESSO")
    p.add_argument("--model-checkpoint-uri")
    p.add_argument("--method", default="PBE-D3")
    p.add_argument("--convergence-status", default="unconverged")
    p.add_argument("--run-id", default="local")
    p.add_argument("--raw-log-uri")
    p.add_argument("--software", default="Quantum ESPRESSO")
    p.add_argument("--reference-barrier-eV", type=float)
    p.add_argument("--acceptance-notes", default="")
    p.add_argument("--calculator-identity")
    p.add_argument("--path-acceptance")
    p.add_argument("--endpoint-acceptance")
    p.add_argument("--calculator-acceptance")
    p.add_argument("--runtime-provenance")
    return p


def main() -> None:
    build(build_parser().parse_args())


if __name__ == "__main__":
    main()
