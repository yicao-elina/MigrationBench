"""Write immutable campaign metadata and checksums for the DFT validation run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    df = pd.read_csv(ROOT / "candidate_manifest.csv").fillna("")
    selected = df[df.dft_status.eq("selected_wave1") & df.input_path.ne("")]
    ledger = []
    for _, row in df[df.input_path.ne("")].iterrows():
        path = Path(row.input_path)
        ledger.append({
            "candidate_id": row.candidate_id,
            "artifact_type": "candidate_extxyz",
            "path": str(path),
            "sha256": sha256(path),
            "status": row.dft_status,
        })
    for path in sorted((ROOT / "qe_inputs").glob("cand_*.scf.in")):
        ledger.append({"candidate_id": path.stem.split(".")[0], "artifact_type": "qe_scf_input", "path": str(path), "sha256": sha256(path), "status": "prepared"})
    pd.DataFrame(ledger).to_csv(ROOT / "sha256_ledger.csv", index=False)
    coverage = {
        category: int((selected.category == category).sum())
        for category in ["stable_endpoint", "neb_intermediate", "near_saddle", "failure_evidence"]
    }
    metadata = {
        "campaign_id": "dft_single_point_validation_20261001",
        "created": "2026-10-01",
        "local_root": str(ROOT),
        "remote_root": "/scratch16/pclancy3/yi/dft_single_point_validation_20261001",
        "scope": "independent QE 7.2 non-magnetic SCF single-point validation; no ionic relaxation",
        "source_campaign_47": {
            "status": "separate_existing_campaign_not_touched",
            "input_root": "/scratch16/pclancy3/yi/revision1_migrationbench_inputs/endpoint_dft_qe72_nonmag_campaign_20260930_all/",
            "result_scope": "will be combined only after independent scientific quality gates",
        },
        "method": {
            "qe_executable": "/data/apps/extern/qe/7.2/bin/pw.x",
            "pseudo_dir": "/data/pclancy3/yi/ONCVPseudoPack/SG15_v1.2/UPF/PBE/",
            "pseudopotentials": {
                "Cr_PBE_FR.SG15v1.2.UPF": "a9352047039238c12614fb6c3ed93acb6e84b2cd9011db2f62eda1858045c5e4",
                "Sb_PBE_FR.SG15v1.2.UPF": "998415c8a49d852c7376645c0a896110258aa7bb40b5c331a1e2874ed34e8132",
                "Te_PBE_FR.SG15v1.2.UPF": "1b6d3177b82988cecabebf7da1636825613b7270bc269795a1fb6c12743e12fa",
            },
            "ecutwfc_Ry": 70,
            "ecutrho_Ry": 280,
            "kpoints": "2 2 2 1 1 1",
            "smearing": "cold",
            "degauss_Ry": 0.0146997236,
            "vdw_corr": "DFT-D3",
            "spin": "QE default nspin=1; magnetic keywords absent",
            "calculation": "scf",
        },
        "selection": {
            "candidate_inventory": int(len(df)),
            "selected_wave1": int(len(selected)),
            "coverage": coverage,
            "md_high_energy_snapshots": 5,
            "md_status": "provenance_incomplete_and_dft_blocked",
            "scratch_seed234_runaway_15": "not_located_in_authoritative_current_inventory; no structures fabricated",
        },
        "quality_gate": [
            "explicit QE JOB DONE",
            "finite total energy",
            "finite total force",
            "QE-reported nat matches candidate manifest",
            "no QE error markers",
            "input hash and non-magnetic SCF contract retained",
        ],
    }
    (ROOT / "campaign_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"selected": len(selected), "inventory": len(df), "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
