#!/usr/bin/env python3
"""Build a read-only provenance and DFT gate audit for Joshitha's NEB tree.

The remote tree is never modified.  Only compact metadata, hashes, and failure
evidence are written into this task's isolated namespace.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/scratch4/pclancy3/joshitha/vdW-sratch/sb2te3"
REMOTE = "rockfish"
SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=20",
    "-o", "ControlMaster=auto",
    "-o", "ControlPersist=120",
    "-o", "ControlPath=/tmp/codex-rockfish-%r-%h-%p",
]

PATHS = [
    ("Cu", "neb", "2L-Cu/NEB/neb"),
    ("Cu", "neb-QLs", "2L-Cu/NEB/neb-QLs"),
    ("Cu", "neb-gap-diffuse", "2L-Cu/NEB/neb-gap-diffuse"),
    ("Cu", "neb-gap2QL", "2L-Cu/NEB/neb-gap2QL"),
    ("Li", "neb-QLs", "2L-Li/NEB/neb-QLs"),
    ("Li", "neb-gap-diffuse", "2L-Li/NEB/neb-gap-diffuse"),
    ("Li", "neb-gap2Ql", "2L-Li/NEB/neb-gap2Ql"),
    ("Na", "neb-QLs", "2L-Na/NEB/neb-QLs"),
    ("Na", "neb-gap-diffuse", "2L-Na/NEB/neb-gap-diffuse"),
    ("Na", "neb-gap2QL", "2L-Na/NEB/neb-gap2QL"),
]


def remote_text(path: str) -> str:
    p = path.replace("'", "'\\''")
    out = subprocess.run(
        ["ssh", *SSH_OPTS, REMOTE, "cat", p],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout


def remote_hash(path: str) -> str | None:
    out = subprocess.run(
        ["ssh", *SSH_OPTS, REMOTE, "sha256sum", path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return out.split()[0] if out else None


def parse_float(value: str) -> float | None:
    try:
        result = float(value.replace("D", "E").replace("d", "e"))
        return result if result == result and abs(result) != float("inf") else None
    except ValueError:
        return None


def parse_neb_input(text: str) -> dict:
    def get(pattern: str, default=None):
        match = re.search(pattern, text, flags=re.I | re.M)
        return match.group(1) if match else default

    nat = int(get(r"\bnat\s*=\s*(\d+)", "0"))
    nimages = int(get(r"\bnum_of_images\s*=\s*(\d+)", "0"))
    path_thr = parse_float(get(r"\bpath_thr\s*=\s*([0-9.EeDd+-]+)", "nan"))
    species = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and re.fullmatch(r"[A-Z][a-z]?", parts[0]) and re.fullmatch(r"[0-9.]+", parts[1]):
            species[parts[0]] = parts[2]
    sections = {}
    for label in ("FIRST_IMAGE", "LAST_IMAGE"):
        match = re.search(rf"{label}\s+ATOMIC_POSITIONS[^\n]*\n(.*?)(?=\n(?:LAST_IMAGE|END_POSITIONS|END_PATH_INPUT|$))", text, flags=re.S)
        if match:
            atoms = []
            comments = []
            for line in match.group(1).splitlines():
                parts = line.split()
                if len(parts) >= 4 and re.fullmatch(r"[A-Z][a-z]?", parts[0]):
                    atoms.append(parts[0])
                    if "!" in line:
                        comments.append(line.split("!", 1)[1].strip())
            sections[label.lower()] = {"n_atoms": len(atoms), "elements": atoms, "comments": comments}
    return {"nat": nat, "num_of_images": nimages, "path_threshold_eV_A": path_thr, "species": species, "sections": sections}


def parse_cell(text: str) -> list[list[float]] | None:
    match = re.search(r"CELL_PARAMETERS[^\n]*\n(.*?)(?=\n\s*(?:K_POINTS|ATOMIC|$))", text, flags=re.S | re.I)
    if not match:
        return None
    rows = []
    for line in match.group(1).splitlines():
        parts = line.split()
        if len(parts) >= 3:
            vals = [parse_float(x) for x in parts[:3]]
            if all(v is not None for v in vals):
                rows.append(vals)
    return rows if len(rows) == 3 else None


def parse_output(text: str, expected_images: int) -> dict:
    tail = text[-12000:]
    warning_images = [int(x) for x in re.findall(r"SCF convergence NOT achieved on image\s+(\d+)", text, flags=re.I)]
    tables = list(re.finditer(r"image\s+energy \(eV\)\s+error \(eV/A\).*?\n((?:\s*\d+\s+[-+0-9.Ee]+\s+[-+0-9.Ee]+\s+\S+\s*\n){1,20})", text, flags=re.I))
    rows = []
    if tables:
        for line in tables[-1].group(1).splitlines():
            parts = line.split()
            if len(parts) >= 4:
                rows.append({"image": int(parts[0]), "energy_eV": parse_float(parts[1]), "error_eV_A": parse_float(parts[2]), "frozen": parts[3]})
    explicit_convergence = bool(re.search(r"(?:converged|path convergence|optimization converged|convergence achieved)", text, flags=re.I)) and not bool(warning_images)
    fatal = re.findall(r"(?:MPI_Abort|Disk quota exceeded|TIME LIMIT|CANCELLED|segmentation fault|error during write|SCF convergence NOT achieved)", text, flags=re.I)
    activation = re.findall(r"activation energy \((?:->|<-)\)\s*=\s*([-+0-9.Ee]+)\s*eV", text, flags=re.I)
    return {
        "job_done": bool(re.search(r"JOB DONE", text, flags=re.I)),
        "explicit_neb_converged": explicit_convergence,
        "scf_warning_images": sorted(set(warning_images)),
        "final_image_rows": rows,
        "all_final_energy_force_finite": len(rows) == expected_images and all(r["energy_eV"] is not None and r["error_eV_A"] is not None for r in rows),
        "image_count_ok": len(rows) == expected_images,
        "activation_energies_eV": [parse_float(x) for x in activation],
        "fatal_or_failure_signatures": sorted(set(x.lower() for x in fatal)),
        "tail_sha256": hashlib.sha256(tail.encode()).hexdigest(),
    }


def audit_one(element: str, label: str, rel: str) -> dict:
    remote_dir = f"{REMOTE_ROOT}/{rel}"
    neb_in = remote_text(f"{remote_dir}/neb.in")
    pw1 = remote_text(f"{remote_dir}/pw_1.in")
    pw2 = remote_text(f"{remote_dir}/pw_2.in")
    out_name = "neb.out"
    try:
        output = remote_text(f"{remote_dir}/{out_name}")
    except subprocess.CalledProcessError:
        out_name = "neb1.out"
        output = remote_text(f"{remote_dir}/{out_name}")
    parsed = parse_neb_input(neb_in)
    endpoint = parsed["sections"]
    input_elements = endpoint.get("first_image", {}).get("elements", [])
    last_elements = endpoint.get("last_image", {}).get("elements", [])
    path_text = remote_text(f"{remote_dir}/sb2te3.path")
    dat_text = remote_text(f"{remote_dir}/sb2te3.dat")
    path_header = re.search(r"NUMBER OF IMAGES\s+(\d+)", path_text, flags=re.I)
    path_images = int(path_header.group(1)) if path_header else None
    out = parse_output(output, parsed["num_of_images"])
    out["dft_scientific_success"] = bool(
        out["job_done"] and out["explicit_neb_converged"] and out["all_final_energy_force_finite"]
        and out["image_count_ok"] and input_elements == last_elements
    )
    reasons = []
    if not out["job_done"]:
        reasons.append("missing_JOB_DONE")
    if not out["explicit_neb_converged"]:
        reasons.append("no_explicit_NEb_convergence_and_or_SCF_warning")
    if not out["all_final_energy_force_finite"]:
        reasons.append("missing_or_nonfinite_final_image_energy_force_table")
    if not out["image_count_ok"] or path_images != parsed["num_of_images"]:
        reasons.append("image_count_mismatch_or_unparsed")
    if input_elements != last_elements:
        reasons.append("endpoint_element_order_mismatch")
    if out["fatal_or_failure_signatures"]:
        reasons.append("failure_retry_signature_present")
    return {
        "path_id": f"{element}_{label}",
        "element": element,
        "system": "Sb2Te3",
        "remote_dir": remote_dir,
        "neb_output_path": f"{remote_dir}/{out_name}",
        "input_paths": {"neb_in": f"{remote_dir}/neb.in", "pw_1": f"{remote_dir}/pw_1.in", "pw_2": f"{remote_dir}/pw_2.in", "path": f"{remote_dir}/sb2te3.path", "profile": f"{remote_dir}/sb2te3.dat"},
        "artifact_sha256": {name: remote_hash(path) for name, path in {
            "neb_in": f"{remote_dir}/neb.in", "pw_1": f"{remote_dir}/pw_1.in", "pw_2": f"{remote_dir}/pw_2.in", "path": f"{remote_dir}/sb2te3.path", "profile": f"{remote_dir}/sb2te3.dat", "neb_output": f"{remote_dir}/{out_name}"
        }.items()},
        "input_sha256_local_derivation": {"neb_in": hashlib.sha256(neb_in.encode()).hexdigest(), "pw_1": hashlib.sha256(pw1.encode()).hexdigest(), "pw_2": hashlib.sha256(pw2.encode()).hexdigest(), "path": hashlib.sha256(path_text.encode()).hexdigest(), "profile": hashlib.sha256(dat_text.encode()).hexdigest()},
        "images": {"neb_input": parsed["num_of_images"], "path_file": path_images, "final_table": len(out["final_image_rows"])},
        "atoms": {"neb_input": parsed["nat"], "first_image": endpoint.get("first_image", {}).get("n_atoms"), "last_image": endpoint.get("last_image", {}).get("n_atoms")},
        "element_order": {"species": sorted(parsed["species"]), "first_image": input_elements, "last_image": last_elements, "same_order": input_elements == last_elements},
        "cell_A": parse_cell(pw1),
        "cell_A_last": parse_cell(pw2),
        "pbc": [True, True, True],
        "endpoint_identity": {"first_comments": endpoint.get("first_image", {}).get("comments", []), "last_comments": endpoint.get("last_image", {}).get("comments", [])},
        "method": {"path_threshold_eV_A": parsed["path_threshold_eV_A"], "vdw": "DFT-D3", "ecutwfc_Ry": 70.0, "ecutrho_Ry": 280.0, "scf_conv_thr": 1e-6, "forc_conv_thr": 1e-3, "cell_relaxation_input": "bfgs but ignored by NEB according to output"},
        "output_audit": out,
        "scientific_success": out["dft_scientific_success"],
        "failure_reason": ";".join(reasons) if reasons else None,
        "retry_evidence": {"slurm_files": sorted(re.findall(r"slurm-[0-9]+\.out", subprocess.run(["ssh", *SSH_OPTS, REMOTE, "find", remote_dir, "-maxdepth", "1", "-type", "f", "-printf", "%f\\n"], check=True, capture_output=True, text=True).stdout)), "output_tail_sha256": out["tail_sha256"]},
    }


def main() -> None:
    records = [audit_one(*item) for item in PATHS]
    now = datetime.now(timezone.utc).isoformat()
    manifest = {"schema_version": "task2-neb-provenance-1.0", "created_utc": now, "campaign_namespace": "task2_cross_element", "remote_read_only_root": REMOTE_ROOT, "source_host": REMOTE, "records": records}
    out = ROOT / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dft_neb_provenance_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    fields = ["path_id", "element", "remote_dir", "neb_output_path", "images_neb_input", "images_final_table", "natoms", "species", "pbc", "scientific_success", "failure_reason", "slurm_files", "neb_in_sha256", "neb_output_sha256"]
    with (out / "dft_neb_provenance_manifest.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({"path_id": r["path_id"], "element": r["element"], "remote_dir": r["remote_dir"], "neb_output_path": r["neb_output_path"], "images_neb_input": r["images"]["neb_input"], "images_final_table": r["images"]["final_table"], "natoms": r["atoms"]["neb_input"], "species": ";".join(r["element_order"]["species"]), "pbc": ";".join(map(str, r["pbc"])), "scientific_success": r["scientific_success"], "failure_reason": r["failure_reason"], "slurm_files": ";".join(r["retry_evidence"]["slurm_files"]), "neb_in_sha256": r["artifact_sha256"]["neb_in"], "neb_output_sha256": r["artifact_sha256"]["neb_output"]})
    ledger = []
    for r in records:
        ledger.append({"path_id": r["path_id"], "scope": "DFT_gate", "status": "scientific_success" if r["scientific_success"] else "failure_or_unaccepted", "failure_reason": r["failure_reason"], "retry_evidence": r["retry_evidence"], "do_not_use_as_reference": not r["scientific_success"]})
    (out / "failure_retry_ledger.json").write_text(json.dumps({"created_utc": now, "records": ledger}, indent=2) + "\n")
    print(json.dumps({"records": len(records), "scientific_success": sum(r["scientific_success"] for r in records), "manifest": str(out / "dft_neb_provenance_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
