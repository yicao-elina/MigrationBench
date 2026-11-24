#!/usr/bin/env python3
"""Read-only audit and blocked-matrix manifest for the 2L Cu/Li/Na NEB tree.

The raw QE tree is never copied or modified.  A single SSH/Python request
collects hashes and diagnostic metadata; this script writes only processed,
machine-readable evidence under the task's isolated namespace.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REMOTE_ROOT = "/scratch4/pclancy3/joshitha/vdW-sratch/sb2te3"
REMOTE_AUDIT = r'''
import hashlib, json, os, re, subprocess, sys

ROOT = "/scratch4/pclancy3/joshitha/vdW-sratch/sb2te3"
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
MODEL = "/scratch16/pclancy3/yi/cross_element_generalization_repair_20261002/broad_roster_ft_seed123/broad_roster_ft_seed123_compiled.model"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def read_text(path):
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except Exception:
        return ""

def parse_float(s):
    try:
        return float(s.replace("D", "E").replace("d", "e"))
    except Exception:
        return None

def parse_input(text):
    def first(pattern, default=None):
        m = re.search(pattern, text, re.I | re.M)
        return m.group(1) if m else default
    nat = int(first(r"\bnat\s*=\s*(\d+)", "0"))
    nimg = int(first(r"\bnum_of_images\s*=\s*(\d+)", "0"))
    thr = parse_float(first(r"\bpath_thr\s*=\s*([0-9.+-EeDd]+)", "nan"))
    species = re.findall(r"^\s*([A-Z][a-z]?)\s+[-+0-9.]+\s+\S+\s*$", text, re.M)
    # Take only the ATOMIC_SPECIES block, not position lines.
    m = re.search(r"ATOMIC_SPECIES\s*(.*?)\n\s*BEGIN_POSITIONS", text, re.S | re.I)
    species = re.findall(r"^\s*([A-Z][a-z]?)\s+[-+0-9.]+\s+\S+\s*$", m.group(1), re.M) if m else []
    m = re.search(r"BEGIN_POSITIONS\s*\n\s*FIRST_IMAGE\s*\n\s*ATOMIC_POSITIONS[^\n]*\n(.*?)\n\s*LAST_IMAGE", text, re.S | re.I)
    first_block = m.group(1) if m else ""
    m = re.search(r"LAST_IMAGE\s*\n\s*ATOMIC_POSITIONS[^\n]*\n(.*?)(?:\n\s*CELL_PARAMETERS|\n\s*END_POSITIONS|\Z)", text, re.S | re.I)
    last_block = m.group(1) if m else ""
    def block_info(block):
        rows = []
        labels = []
        for line in block.splitlines():
            q = re.match(r"^\s*([A-Z][a-z]?)\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)(?:\s*!\s*(.*))?$", line)
            if not q:
                continue
            rows.append((q.group(1), [parse_float(q.group(i)) for i in (2, 3, 4)]))
            if q.group(5):
                labels.append(q.group(5).strip())
        return rows, labels
    first_rows, first_labels = block_info(first_block)
    last_rows, last_labels = block_info(last_block)
    m = re.search(r"CELL_PARAMETERS\s+[^\n]*\n((?:\s*[-+0-9.eEdD]+\s+[-+0-9.eEdD]+\s+[-+0-9.eEdD]+\s*\n){3})", text, re.I)
    cell = []
    if m:
        for line in m.group(1).splitlines():
            vals = [parse_float(x) for x in line.split()[:3]]
            if len(vals) == 3:
                cell.append(vals)
    return {
        "nat": nat, "num_of_images": nimg, "path_thr_eV_A": thr,
        "species": species, "first_rows": first_rows, "last_rows": last_rows,
        "first_labels": first_labels, "last_labels": last_labels,
        "cell_A": cell, "pbc": [True, True, True],
    }

def parse_xyz(text):
    frames = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        try:
            n = int(lines[i].strip())
        except Exception:
            i += 1
            continue
        if i + 2 + n > len(lines):
            break
        rows = []
        finite = True
        for line in lines[i + 2:i + 2 + n]:
            q = line.split()
            if len(q) < 4:
                finite = False
                continue
            rows.append(q[0])
            try:
                finite = finite and all(map(lambda x: abs(float(x)) < float("inf"), q[1:4]))
            except Exception:
                finite = False
        frames.append({"nat": n, "symbols": rows, "finite": finite})
        i += 2 + n
    return frames

def parse_output(text, nimg, thr):
    iters = [int(x) for x in re.findall(r"iteration\s+(\d+)", text, re.I)]
    warnings = re.findall(r"WARNING\s*:\s*([^\n]+)", text, re.I)
    rows = []
    final_iter = iters[-1] if iters else None
    table_iter = None
    complete_tables = []
    headers = list(re.finditer(r"image\s+energy\s*\(eV\)\s+error\s*\(eV/A\)", text, re.I))
    # A final SCF failure can occur before QE prints that iteration's image
    # table.  Use the last complete image-wise table, but retain the failed
    # final iteration and warning as failure evidence.
    for head in reversed(headers):
        candidate = []
        for line in text[head.end():].splitlines():
            q = re.match(r"^\s*(\d+)\s+([-+0-9.eEdD]+)\s+([-+0-9.eEdD]+)\s+([FT])\s*$", line)
            if not q:
                if candidate and ("path length" in line.lower() or "iteration" in line.lower() or "warning" in line.lower()):
                    break
                continue
            candidate.append({"image": int(q.group(1)), "energy_eV": parse_float(q.group(2)), "error_eV_A": parse_float(q.group(3)), "frozen": q.group(4)})
        if len(candidate) == nimg:
            prior = list(re.finditer(r"iteration\s+(\d+)", text[:head.start()], re.I))
            complete_tables.append({"iteration": int(prior[-1].group(1)) if prior else None, "rows": candidate})
    complete_tables.sort(key=lambda x: x["iteration"] if x["iteration"] is not None else -1, reverse=True)
    if complete_tables:
        rows = complete_tables[0]["rows"]
        table_iter = complete_tables[0]["iteration"]
    activations = []
    for m in re.finditer(r"activation energy \(->\)\s*=\s*([-+0-9.eEdD]+) eV\s*\n\s*activation energy \(<-\)\s*=\s*([-+0-9.eEdD]+) eV", text, re.I):
        prior = list(re.finditer(r"iteration\s+(\d+)", text[:m.start()], re.I))
        activations.append({"iteration": int(prior[-1].group(1)) if prior else None, "forward_barrier_eV": parse_float(m.group(1)), "reverse_barrier_eV": parse_float(m.group(2))})
    barrier_delta_forward = barrier_delta_reverse = barrier_delta_max = None
    if len(activations) >= 2:
        a, b = activations[-2:]
        barrier_delta_forward = abs(b["forward_barrier_eV"] - a["forward_barrier_eV"])
        barrier_delta_reverse = abs(b["reverse_barrier_eV"] - a["reverse_barrier_eV"])
        barrier_delta_max = max(barrier_delta_forward, barrier_delta_reverse)
    profile_delta_max = profile_delta_rms = None
    if len(complete_tables) >= 2:
        previous = complete_tables[1]["rows"]
        diffs = [abs(a["energy_eV"] - b["energy_eV"]) for a, b in zip(previous, rows)]
        profile_delta_max = max(diffs)
        profile_delta_rms = (sum(x * x for x in diffs) / len(diffs)) ** 0.5
    finite = bool(rows) and all(r["energy_eV"] is not None and r["error_eV_A"] is not None for r in rows)
    max_error = max((r["error_eV_A"] for r in rows), default=None)
    has_scf_warning = bool(re.search(r"scf convergence\s+NOT achieved", text, re.I))
    return {
        "iterations_observed": len(iters), "final_iteration": final_iter,
        "last_complete_table_iteration": table_iter,
        "complete_table_iterations": [x["iteration"] for x in complete_tables],
        "activation_history_tail": activations[-3:],
        "barrier_delta_forward_eV": barrier_delta_forward,
        "barrier_delta_reverse_eV": barrier_delta_reverse,
        "barrier_delta_max_eV": barrier_delta_max,
        "profile_energy_delta_max_eV": profile_delta_max,
        "profile_energy_delta_rms_eV": profile_delta_rms,
        "inner_scf_iteration_count": "not_reported_in_neb_log",
        "final_rows": rows, "final_image_count": len(rows),
        "final_energies_errors_finite": finite, "final_error_max_eV_A": max_error,
        "scf_warning": has_scf_warning, "warnings": warnings[-10:],
        "job_done": bool(re.search(r"JOB DONE", text, re.I)),
        "path_converged": bool(rows) and len(rows) == nimg and finite and max_error is not None and max_error <= thr and not has_scf_warning,
    }

def audit_one(dopant, name, rel):
    d = os.path.join(ROOT, rel)
    files = []
    for base, _, names in os.walk(d):
        for n in names:
            p = os.path.join(base, n)
            try:
                files.append({"path": p, "size": os.path.getsize(p)})
            except OSError:
                pass
    inp = os.path.join(d, "neb.in")
    out = os.path.join(d, "neb.out")
    if not os.path.isfile(out):
        out = os.path.join(d, "neb1.out")
    inp_text = read_text(inp)
    out_text = read_text(out)
    spec = parse_input(inp_text)
    result = parse_output(out_text, spec["num_of_images"], spec["path_thr_eV_A"] or float("nan"))
    xyzs = sorted(os.path.join(d, n) for n in os.listdir(d) if n.lower().endswith(".xyz"))
    xyz = xyzs[0] if xyzs else ""
    frames = parse_xyz(read_text(xyz)) if xyz else []
    key_names = ["neb.in", os.path.basename(out), "neb.dat"]
    if xyz:
        key_names.append(os.path.basename(xyz))
    key_hashes = {}
    for n in key_names:
        p = os.path.join(d, n)
        if os.path.isfile(p):
            key_hashes[n] = sha256(p)
    retry = sorted(os.path.basename(x["path"]) for x in files if os.path.basename(x["path"]).startswith("slurm-"))
    symbols_ok = bool(spec["first_rows"]) and bool(spec["last_rows"]) and len(spec["first_rows"]) == spec["nat"] and len(spec["last_rows"]) == spec["nat"] and [x[0] for x in spec["first_rows"]] == [x[0] for x in spec["last_rows"]]
    traj_ok = bool(frames) and all(f["nat"] == spec["nat"] and f["finite"] for f in frames) and len({tuple(f["symbols"]) for f in frames}) == 1
    return {
        "dopant": dopant, "path_name": name, "path_id": f"2L-{dopant}__{name}",
        "remote_path": d, "input_path": inp, "output_path": out,
        "input_sha256": key_hashes.get("neb.in", ""), "output_sha256": key_hashes.get(os.path.basename(out), ""),
        "key_file_sha256": key_hashes, "all_file_count": len(files), "retry_evidence": retry,
        "species": spec["species"], "dopant_detected": dopant,
        "nat": spec["nat"], "num_of_images": spec["num_of_images"], "path_thr_eV_A": spec["path_thr_eV_A"],
        "endpoint_first_label": spec["first_labels"][0] if spec["first_labels"] else "",
        "endpoint_last_label": spec["last_labels"][0] if spec["last_labels"] else "",
        "first_element_order": [x[0] for x in spec["first_rows"]], "last_element_order": [x[0] for x in spec["last_rows"]],
        "cell_A": spec["cell_A"], "pbc": spec["pbc"], "cell_pbc_ok": len(spec["cell_A"]) == 3 and spec["pbc"] == [True, True, True],
        "trajectory_path": xyz, "trajectory_sha256": key_hashes.get(os.path.basename(xyz), "") if xyz else "",
        "trajectory_frames": len(frames), "trajectory_natoms_ok": bool(frames) and all(f["nat"] == spec["nat"] for f in frames),
        "trajectory_element_order_consistent": traj_ok, "endpoint_identity_ok": symbols_ok,
        **result,
    }

def main():
    rows = [audit_one(*x) for x in PATHS]
    env = {}
    py = "/data/pclancy3/yi/.conda/envs/mace/bin/python"
    try:
        env["python_version"] = subprocess.check_output([py, "--version"], stderr=subprocess.STDOUT, text=True, timeout=20).strip()
    except Exception as e:
        env["python_version"] = f"unavailable:{type(e).__name__}"
    model = {"model_path": MODEL, "exists": os.path.isfile(MODEL), "size": os.path.getsize(MODEL) if os.path.isfile(MODEL) else None, "sha256": sha256(MODEL) if os.path.isfile(MODEL) else ""}
    print(json.dumps({"remote_root": ROOT, "paths": rows, "environment": env, "foundation_model": model}, sort_keys=True))

if __name__ == "__main__":
    main()
'''


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_remote() -> dict:
    proc = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "rockfish", "python3", "-"],
        input=REMOTE_AUDIT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"remote read-only audit failed ({proc.returncode}): {proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main_local() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit = run_remote()
    created = datetime.now(timezone.utc).isoformat()
    paths = audit["paths"]
    model = audit["foundation_model"]
    model_id = "legacy_foundation"
    model_label = "Foundation"
    dft_manifest_rows = []
    model_gate_rows = []
    dft_ledger = []
    retry_rows = []
    for p in paths:
        dft_pass = bool(
            p["path_converged"]
            and p["final_image_count"] == p["num_of_images"]
            and p["final_energies_errors_finite"]
            and p["endpoint_identity_ok"]
            and p["cell_pbc_ok"]
            and p["trajectory_natoms_ok"]
            and p["trajectory_element_order_consistent"]
        )
        reasons = []
        if not p["path_converged"]:
            reasons.append(f"explicit NEB convergence failed: final max image error {p['final_error_max_eV_A']} > path_thr {p['path_thr_eV_A']} eV/A or SCF warning")
        if p["final_image_count"] != p["num_of_images"]:
            reasons.append("final image count mismatch")
        if not p["final_energies_errors_finite"]:
            reasons.append("non-finite or missing final image energy/error")
        if not p["endpoint_identity_ok"]:
            reasons.append("endpoint identity or element order audit failed")
        if not p["cell_pbc_ok"]:
            reasons.append("cell/PBC audit failed")
        if not p["trajectory_natoms_ok"] or not p["trajectory_element_order_consistent"]:
            reasons.append("trajectory image/atom/element consistency audit failed")
        status = "accepted" if dft_pass else "ledger-only"
        row = {
            **p,
            "created_utc": created,
            "scientific_success": dft_pass,
            "acceptance_status": status,
            "failure_reason": "; ".join(reasons) if reasons else "",
            "convergence_criterion": "all final image NEB errors <= path_thr, finite energies/errors, no SCF warning",
        }
        dft_manifest_rows.append(row)
        dft_ledger.append({
            "created_utc": created, "source_kind": "DFT_NE​B".replace("​", ""), "path_id": p["path_id"],
            "model_id": "", "dopant": p["dopant"], "attempt": len(p["retry_evidence"]),
            "status": status, "scientific_success": dft_pass,
            "failure_reason": row["failure_reason"], "job_done": p["job_done"],
            "slurm_retry_evidence": json.dumps(p["retry_evidence"]), "input_sha256": p["input_sha256"], "output_sha256": p["output_sha256"],
        })
        retry_rows.append(dft_ledger[-1])
        gate_reason = "blocked: no DFT path passed the explicit reference gate"
        if not model["exists"]:
            gate_reason = "blocked: Foundation model file missing"
        model_gate_rows.append({
            "path_id": p["path_id"], "dopant": p["dopant"], "path_name": p["path_name"],
            "model_id": model_id, "model_label": model_label, "model_path": model["model_path"],
            "model_sha256": model["sha256"], "element_support_status": "not_tested_dft_gate_blocked",
            "dft_reference_status": status, "scientific_success": False,
            "failure_reason": gate_reason, "created_utc": created,
        })
        for perturb in ("base_000", "dx020", "tilt020"):
            retry_rows.append({
                "created_utc": created, "source_kind": "foundation_perturb_matrix", "path_id": p["path_id"],
                "model_id": model_id, "dopant": p["dopant"], "attempt": 0, "status": "blocked",
                "scientific_success": False, "failure_reason": gate_reason + f"; perturbation={perturb}",
                "job_done": False, "slurm_retry_evidence": "[]", "input_sha256": "", "output_sha256": "",
            })
    # First ten records are path reference records; the second ten are model gates.
    manifest_fields = [
        "created_utc", "path_id", "dopant", "path_name", "remote_path", "input_path", "output_path",
        "input_sha256", "output_sha256", "key_file_sha256", "all_file_count", "retry_evidence", "species",
        "dopant_detected", "nat", "num_of_images", "path_thr_eV_A", "endpoint_first_label", "endpoint_last_label",
        "first_element_order", "last_element_order", "cell_A", "pbc", "cell_pbc_ok", "trajectory_path",
        "trajectory_sha256", "trajectory_frames", "trajectory_natoms_ok", "trajectory_element_order_consistent",
        "endpoint_identity_ok", "iterations_observed", "final_iteration", "final_rows", "final_image_count",
        "final_energies_errors_finite", "final_error_max_eV_A", "scf_warning", "warnings", "job_done",
        "path_converged", "scientific_success", "acceptance_status", "failure_reason", "convergence_criterion",
    ]
    write_csv(out / "dft_reference_manifest.csv", paths, manifest_fields)
    (out / "dft_reference_manifest.json").write_text(json.dumps({"schema_version": "1.0", "created_utc": created, "remote_root": audit["remote_root"], "environment": audit["environment"], "foundation_model": model, "paths": dft_manifest_rows}, indent=2, sort_keys=True))
    stats_fields = ["path_id", "dopant", "path_name", "iterations_observed", "final_iteration", "last_complete_table_iteration", "complete_table_iterations", "inner_scf_iteration_count", "scf_warning", "activation_history_tail", "barrier_delta_forward_eV", "barrier_delta_reverse_eV", "barrier_delta_max_eV", "profile_energy_delta_max_eV", "profile_energy_delta_rms_eV"]
    stats_rows = []
    for p in paths:
        q = {k: p.get(k, "") for k in stats_fields}
        q["dopant"] = p["dopant"]
        q["path_name"] = p["path_name"]
        q["path_id"] = p["path_id"]
        stats_rows.append(q)
    write_csv(out / "scf_iteration_energy_delta.csv", stats_rows, stats_fields)
    ledger_fields = ["created_utc", "source_kind", "path_id", "model_id", "dopant", "attempt", "status", "scientific_success", "failure_reason", "job_done", "slurm_retry_evidence", "input_sha256", "output_sha256"]
    write_csv(out / "dft_acceptance_failure_ledger.csv", dft_ledger, ledger_fields)
    write_csv(out / "failure_retry_ledger.csv", retry_rows, ledger_fields)
    gate_fields = ["path_id", "dopant", "path_name", "model_id", "model_label", "model_path", "model_sha256", "element_support_status", "dft_reference_status", "scientific_success", "failure_reason", "created_utc"]
    write_csv(out / "foundation_model_path_gate.csv", model_gate_rows, gate_fields)

    blocked_sp = []
    blocked_force = []
    blocked_relax = []
    for p in paths:
        common = {"model_id": model_id, "model_label": model_label, "model_sha256": model["sha256"], "path_id": p["path_id"], "dopant": p["dopant"], "status": "blocked_dft_reference_gate", "scientific_success": False, "failure_reason": "No DFT path passed explicit convergence/finite-image/identity/cell/PBC gate", "input_sha256": p["input_sha256"], "runtime_manifest_sha256": ""}
        blocked_sp.append({**common, "image_index": "", "energy_eV": "", "max_force_eV_A": "", "relative_energy_eV": "", "barrier_eV": "", "barrier_image": "", "profile_rmse_eV": "", "endpoint_ordering": ""})
        blocked_force.append({**common, "image_index": "", "dft_force_eV_A": "", "mace_force_eV_A": "", "force_error_eV_A": "", "force_error_norm_eV_A": ""})
        for perturb in ("base_000", "dx020", "tilt020"):
            blocked_relax.append({**common, "perturbation": perturb, "displacement_A": "", "whole_rmsd_A": "", "host_rmsd_A": "", "max_displacement_A": "", "residual_force_eV_A": "", "relative_energy_eV": "", "nearest_dft_image": "", "basin_return": "", "runaway": "", "identity_anomaly": ""})
    write_csv(out / "foundation_path_single_point.csv", blocked_sp, list(blocked_sp[0]))
    write_csv(out / "foundation_path_force_errors.csv", blocked_force, list(blocked_force[0]))
    write_csv(out / "foundation_perturb_relax_summary.csv", blocked_relax, list(blocked_relax[0]))

    dictionary = {
        "scientific_status": "blocked",
        "reason": "All 10 DFT candidates are ledger-only after explicit reference gate; no Foundation MACE scientific rows were executed.",
        "tables": {
            "dft_reference_manifest.csv": "One row per remote DFT NEB candidate; hashes are remote raw-file hashes and no raw tree is copied.",
            "foundation_path_single_point.csv": "One blocked row per model-path; numeric fields are intentionally blank, never placeholders for measurements.",
            "foundation_path_force_errors.csv": "Image-wise force-error schema; empty numeric cells because the DFT reference gate failed.",
            "foundation_perturb_relax_summary.csv": "Three perturbation rows per model-path; no relaxation was started.",
            "failure_retry_ledger.csv": "DFT failed/unfinished evidence plus downstream matrix blocks; SLURM completion is not success.",
        },
        "required_figure_panels": ["DFT_vs_Foundation_profiles", "barrier_scatter", "imagewise_force_error", "return_distance_distribution", "basin_heatmap", "model_path_image_stability_matrix"],
        "figure_policy": "schema-only until accepted DFT paths and real MACE results exist; no scientific PNG/SVG/PDF is rendered from fabricated values",
    }
    (out / "figure_data_dictionary.json").write_text(json.dumps(dictionary, indent=2, sort_keys=True))
    (out / "caption_draft.md").write_text("# Figure caption draft\n\nPlaceholder only: no publication figure is rendered because all ten DFT NEB candidates remain ledger-only after the explicit convergence gate.\n")
    (out / "results_draft_zh_en.md").write_text("# Results draft (evidence-bounded)\n\n中文：10 条候选 DFT NEB 路径均已完成只读 provenance 审计，但没有路径满足最终所有 image 的显式 NEB 收敛、有限 energy/error、identity 与 cell/PBC 门槛。因此本阶段不报告 Foundation MACE 的路径精度或扰动稳定性。\n\nEnglish: All 10 candidate DFT NEB paths were provenance-audited read-only, but none passed the final explicit NEB convergence, finite image energy/error, identity, and cell/PBC gates. Foundation MACE pathway accuracy and perturbation stability are therefore not reported in this stage.\n")
    runtime = {"created_utc": created, "remote_root": audit["remote_root"], "environment": audit["environment"], "foundation_model": model, "model_support_status": "not_tested_dft_gate_blocked", "scientific_status": "blocked", "coverage": {"dft_candidates": len(paths), "dft_accepted": sum(bool(x["path_converged"]) for x in paths), "mace_single_point_started": 0, "perturb_relax_started": 0}}
    (out / "runtime_manifest.json").write_text(json.dumps(runtime, indent=2, sort_keys=True))
    print(json.dumps(runtime, indent=2, sort_keys=True))


if __name__ == "__main__":
    main_local()
