from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import iread, read, write


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "source_candidate_manifest_65.csv"
INPUTS = ROOT / "inputs"
EVIDENCE = ROOT / "source_evidence"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def failure_frames(path: Path):
    frames = []
    for i, atoms in enumerate(iread(str(path), format="extxyz")):
        forces = np.asarray(atoms.get_forces(), float)
        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        energy = float(atoms.info.get("energy_eV", atoms.info.get("energy", np.nan)))
        max_force = float(np.linalg.norm(forces, axis=1).max())
        min_distance = float(np.nanmin(distances))
        frames.append((i, atoms, energy, max_force, min_distance))
    anomalies = [x for x in frames if not np.isfinite(x[2]) or not np.isfinite(x[3]) or x[3] > 5.0 or x[4] < 1.5]
    first_anomaly = anomalies[0] if anomalies else None
    idx = frames.index(first_anomaly) if first_anomaly else len(frames)
    last_physical = frames[idx - 1] if idx else frames[0]
    return last_physical, first_anomaly, frames


def base_rows():
    df = pd.read_csv(BASE).copy()
    df["source_manifest"] = str(BASE)
    df["dft_status"] = "pending_selection"
    df["dft_eligibility"] = "eligible_candidate"
    df["candidate_role"] = "raw_structure"
    df["pending_reason"] = ""
    for col in ["selection_priority", "failure_class", "evidence_frame", "mace_energy_eV", "mace_max_force_eV_A", "mace_min_distance_A"]:
        df[col] = ""
    # Keep the new campaign self-contained while retaining original source paths in source_path.
    df["input_path"] = df["candidate_id"].map(lambda x: str(INPUTS / f"{x}.extxyz"))
    return df.to_dict("records")


def add_failure_rows(rows):
    specs = [
        ("epmlff_034_unique_065_base", "unique_065", "base"),
        ("epmlff_035_unique_065_perturb_a_dx020", "unique_065", "perturb_a_dx020"),
        ("epmlff_067_unique_080_base", "unique_080", "base"),
        ("epmlff_068_unique_080_perturb_a_dx020", "unique_080", "perturb_a_dx020"),
        ("epmlff_069_unique_080_perturb_b_tilt020", "unique_080", "perturb_b_tilt020"),
    ]
    next_id = len(rows) + 1
    for name, endpoint, variant in specs:
        source = EVIDENCE / f"{name}.trajectory.extxyz"
        last_physical, first_anomaly, frames = failure_frames(source)
        for role, item in [("last_physical", last_physical), ("first_anomaly_evidence", first_anomaly)]:
            if item is None:
                continue
            frame, atoms, energy, max_force, min_distance = item
            cid = f"cand_{next_id:04d}"
            next_id += 1
            out = INPUTS / f"{cid}.extxyz"
            write(out, atoms, format="extxyz")
            rows.append({
                "candidate_id": cid,
                "category": "failure_evidence",
                "mechanism_group": "failure-risk",
                "endpoint_id": endpoint,
                "endpoint_type": "",
                "path_id": "",
                "cov_id": "",
                "image_index": "",
                "reaction_coordinate": "",
                "barrier_eV": "",
                "candidate_reason": "MPA0_unique_065_080_failure_variant",
                "source_path": str(source),
                "source_hash": sha256(source),
                "source_identity": f"ft600k_mpa0_grouped_seed123:{name}:frame_{frame}",
                "input_path": str(out),
                "input_sha256": sha256(out),
                "n_atoms": len(atoms),
                "pbc": json.dumps(atoms.pbc.tolist()),
                "cell_A": json.dumps(np.asarray(atoms.cell).round(8).tolist()),
                "md_status": "not_applicable",
                "selection_score": max_force,
                "source_manifest": str(source),
                "dft_status": "pending_selection" if role == "last_physical" else "evidence_only",
                "dft_eligibility": "eligible_candidate" if role == "last_physical" else "evidence_only",
                "candidate_role": role,
                "pending_reason": "" if role == "last_physical" else "runaway_structure_not_a_valid_DFT_configuration",
                "selection_priority": "P1",
                "failure_class": "mace_runaway_or_nonconverged",
                "evidence_frame": frame,
                "mace_energy_eV": energy,
                "mace_max_force_eV_A": max_force,
                "mace_min_distance_A": min_distance,
            })
    return rows


def add_md_gap_rows(rows):
    # Five real high-energy frames were extracted from the 1223-frame
    # multi-temperature trajectory.  They remain DFT-blocked because the
    # exported directory still lacks the original run manifest/checkpoint
    # chain; the rows are retained so coverage is explicit and countable.
    md_dir = EVIDENCE / "md_high_energy_snapshots"
    extracted = sorted(md_dir.glob("md_multiT_high_energy_rank*_frame*.extxyz"))
    source = "/scratch16/pclancy3/yi/md_time_convergence_20261001_snapshots/3-multi_T-fine-tuning.segment.extxyz"
    source_hash = "b19bbd5c40e68568717ebbac0c0c011a9724a7e655ac82dd78a18e7b273b53aa"
    next_id = len(rows) + 1
    if len(extracted) == 5:
        for path in extracted:
            atoms = read(path, format="extxyz")
            match = path.stem.split("_frame")[-1]
            frame = int(match)
            header = path.read_text(errors="replace").splitlines()[1]
            energy_match = re.search(r"(?:^| )energy=([^ ]+)", header)
            energy = float(energy_match.group(1)) if energy_match else float("nan")
            rows.append({
                "candidate_id": f"cand_{next_id:04d}", "category": "md_high_energy_or_large_displacement",
                "mechanism_group": "MD high-energy snapshot", "endpoint_id": "", "endpoint_type": "",
                "path_id": "", "cov_id": "", "image_index": "", "reaction_coordinate": "",
                "barrier_eV": "", "candidate_reason": "md_high_energy_snapshot_frame",
                "source_path": source, "source_hash": source_hash,
                "source_identity": f"md_time_convergence_20261001_snapshots:frame_{frame}",
                "input_path": str(path), "input_sha256": sha256(path), "n_atoms": len(atoms),
                "pbc": json.dumps(atoms.pbc.tolist()), "cell_A": json.dumps(np.asarray(atoms.cell).round(8).tolist()),
                "md_status": "provenance_incomplete", "selection_score": energy,
                "source_manifest": source, "dft_status": "pending_md_provenance",
                "dft_eligibility": "blocked_missing_provenance", "candidate_role": "md_high_energy_snapshot",
                "pending_reason": "Original MD run manifest/checkpoint chain not located; 2050-atom DFT resource gate",
                "selection_priority": "P3_mechanism_coverage", "failure_class": "md_provenance_incomplete",
                "evidence_frame": frame, "mace_energy_eV": energy, "mace_max_force_eV_A": "", "mace_min_distance_A": "",
            })
            next_id += 1
        return rows

    # Fallback retains the previously located source paths if the extracted
    # frames are not present on a fresh checkout.
    sources = [
        "/scratch16/pclancy3/yi/md_time_convergence_20261001_snapshots/1-from-scratch.segment.extxyz",
        source,
        "/scratch16/pclancy3/yi/md_continuation_400ps_rockfish/segments/0-foundation-omat_to_400ps_gpu.xyz",
        "/scratch16/pclancy3/yi/md_continuation_400ps_rockfish/segments/1-from-scratch_to_400ps_gpu.xyz",
        "/scratch16/pclancy3/yi/md_continuation_400ps_rockfish/segments/2-naive-fine-tuning_to_400ps_gpu.xyz",
    ]
    for source in sources:
        rows.append({
            "candidate_id": f"cand_{next_id:04d}", "category": "md_high_energy_or_large_displacement",
            "mechanism_group": "MD provenance gap", "endpoint_id": "", "endpoint_type": "",
            "path_id": "", "cov_id": "", "image_index": "", "reaction_coordinate": "",
            "barrier_eV": "", "candidate_reason": "md_missing_no_complete_provenance_chain",
            "source_path": source, "source_hash": "not_obtained", "source_identity": "MD source located remotely",
            "input_path": "", "input_sha256": "not_available", "n_atoms": "2050", "pbc": "[true, true, true]",
            "cell_A": "", "md_status": "md_missing", "selection_score": "",
            "source_manifest": source, "dft_status": "pending_md_provenance",
            "dft_eligibility": "blocked_missing_provenance", "candidate_role": "coverage_gap",
            "pending_reason": "No auditable run manifest/input hash plus 2050-atom single-point resource gate",
            "selection_priority": "P3", "failure_class": "md_missing", "evidence_frame": "",
            "mace_energy_eV": "", "mace_max_force_eV_A": "", "mace_min_distance_A": "",
        })
        next_id += 1
    return rows


def main():
    rows = add_md_gap_rows(add_failure_rows(base_rows()))
    df = pd.DataFrame(rows)
    # Preserve the full inventory, but make the first wave deterministic and
    # auditable.  The mechanism-coverage gate is one representative per
    # selected endpoint/path family, not simply the first five CSV rows.
    df["dft_status"] = df["dft_status"].replace({"pending_selection": "pending"})
    def require_one(column, values, label):
        indices = []
        for value in values:
            matches = df.index[df[column].fillna("").eq(value)].tolist()
            if not matches:
                raise RuntimeError(f"missing {label} candidate for {column}={value}")
            indices.append(matches[0])
        return indices

    stable = require_one(
        "endpoint_id",
        ["unique_044", "unique_096", "unique_067", "unique_109", "unique_097"],
        "stable",
    )
    inter = require_one("cov_id", ["cov_020", "cov_035", "cov_002", "cov_027", "cov_031"], "intermediate")
    saddle = []
    for cov in ["cov_020", "cov_035", "cov_002", "cov_027", "cov_031"]:
        matches = df.index[df.category.eq("near_saddle") & df.cov_id.fillna("").eq(cov)].tolist()
        if not matches:
            raise RuntimeError(f"missing near-saddle candidate for cov_id={cov}")
        saddle.append(matches[0])
    failure = df.index[(df.category.eq("failure_evidence")) & (df.candidate_role.eq("last_physical"))].tolist()
    selected = stable + inter + saddle + failure
    df.loc[selected, "dft_status"] = "selected_wave1"
    df.loc[stable + inter + saddle, "selection_priority"] = "P3_mechanism_coverage"
    df.loc[failure, "selection_priority"] = "P1_failure_evidence"
    df.loc[df.category.eq("failure_evidence") & df.candidate_role.eq("first_anomaly_evidence"), "dft_status"] = "evidence_only"
    df.loc[df.category.eq("failure_evidence") & df.candidate_role.eq("first_anomaly_evidence"), "dft_eligibility"] = "evidence_only"
    df["campaign_input_sha256"] = df["input_path"].map(lambda p: sha256(Path(p)) if str(p) else "not_available")
    df["input_sha256"] = df["campaign_input_sha256"]
    cols = ["candidate_id", "category", "mechanism_group", "endpoint_id", "endpoint_type", "path_id", "cov_id", "image_index", "reaction_coordinate", "barrier_eV", "candidate_reason", "source_path", "source_hash", "source_identity", "input_path", "input_sha256", "n_atoms", "pbc", "cell_A", "md_status", "selection_score", "source_manifest", "dft_status", "dft_eligibility", "candidate_role", "pending_reason", "selection_priority", "failure_class", "evidence_frame", "mace_energy_eV", "mace_max_force_eV_A", "mace_min_distance_A"]
    df = df[cols]
    df.to_csv(ROOT / "candidate_manifest.csv", index=False)
    (ROOT / "candidate_manifest.json").write_text(json.dumps({"schema_version": "2.0", "created": "2026-10-01", "n_candidates": len(df), "md_status": "pending_audit", "selected_wave1": int((df.dft_status == "selected_wave1").sum()), "rows": df.to_dict("records")}, indent=2) + "\n")
    print(df.groupby(["category", "dft_status"], dropna=False).size().to_string())
    print("selected_wave1", int((df.dft_status == "selected_wave1").sum()))


if __name__ == "__main__":
    main()
