#!/usr/bin/env python3
"""Audit NEB outer iterations and last-two-iteration changes.

This deliberately does not promote any path to DFT scientific success.  It
quantifies the provisional snapshot quality requested for exploratory MACE
sampling.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "rockfish"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ControlMaster=auto", "-o", "ControlPersist=120", "-o", "ControlPath=/tmp/codex-rockfish-%r-%h-%p"]


def fetch(path: str) -> str:
    return subprocess.run(SSH + [REMOTE, "cat", path], check=True, capture_output=True, text=True).stdout


def f(x: str):
    try:
        return float(x.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def parse_blocks(text: str):
    starts = list(re.finditer(r"-{10,}\s*iteration\s+(\d+)\s*-{10,}", text, re.I))
    blocks = []
    for i, match in enumerate(starts):
        chunk = text[match.start() : starts[i + 1].start() if i + 1 < len(starts) else len(text)]
        table = re.search(r"image\s+energy \(eV\)\s+error \(eV/A\).*?\n((?:\s*\d+\s+[-+0-9.Ee]+\s+[-+0-9.Ee]+\s+\S+\s*\n){1,20})", chunk, re.I)
        rows = []
        if table:
            for line in table.group(1).splitlines():
                p = line.split()
                if len(p) >= 4:
                    rows.append({"image": int(p[0]), "energy_eV": f(p[1]), "error_eV_A": f(p[2]), "frozen": p[3]})
        scf_images = [int(x) for x in re.findall(r"self-consistency for image\s+(\d+)", chunk, re.I)]
        scf_warning_images = [int(x) for x in re.findall(r"SCF convergence NOT achieved on image\s+(\d+)", chunk, re.I)]
        blocks.append({"iteration": int(match.group(1)), "rows": rows, "scf_image_calls": scf_images, "scf_warning_images": scf_warning_images})
    return blocks


def main():
    manifest = json.loads((ROOT / "manifests/dft_neb_provenance_manifest.json").read_text())
    records = []
    for record in manifest["records"]:
        text = fetch(record["neb_output_path"])
        blocks = parse_blocks(text)
        completed_blocks = [b for b in blocks if len(b.get("rows", [])) == record["images"]["neb_input"]]
        last = completed_blocks[-1] if completed_blocks else {"rows": []}
        prev = completed_blocks[-2] if len(completed_blocks) >= 2 else {"rows": []}
        prev_by = {r["image"]: r for r in prev["rows"]}
        deltas = []
        for row in last["rows"]:
            if row["image"] in prev_by and row["energy_eV"] is not None and prev_by[row["image"]]["energy_eV"] is not None:
                deltas.append({"image": row["image"], "delta_energy_eV": row["energy_eV"] - prev_by[row["image"]]["energy_eV"], "abs_delta_energy_eV": abs(row["energy_eV"] - prev_by[row["image"]]["energy_eV"]), "last_energy_eV": row["energy_eV"], "prev_energy_eV": prev_by[row["image"]]["energy_eV"], "last_error_eV_A": row["error_eV_A"]})
        max_delta = max((x["abs_delta_energy_eV"] for x in deltas), default=None)
        mean_delta = sum(x["abs_delta_energy_eV"] for x in deltas) / len(deltas) if deltas else None
        barrier_last = max((x["last_energy_eV"] for x in deltas), default=None)
        barrier_prev = max((x["prev_energy_eV"] for x in deltas), default=None)
        records.append({"path_id": record["path_id"], "element": record["element"], "neb_output_path": record["neb_output_path"], "outer_neb_iterations": len(blocks), "first_iteration": blocks[0]["iteration"] if blocks else None, "last_outer_iteration": blocks[-1].get("iteration") if blocks else None, "last_outer_iteration_scf_image_calls": len(blocks[-1].get("scf_image_calls", [])) if blocks else 0, "last_outer_iteration_scf_warning_images": sorted(set(blocks[-1].get("scf_warning_images", []))) if blocks else [], "last_completed_iteration": last.get("iteration"), "last_completed_iteration_scf_image_calls": len(last.get("scf_image_calls", [])), "last_completed_iteration_unique_scf_images": sorted(set(last.get("scf_image_calls", []))), "last_completed_iteration_scf_warning_images": sorted(set(last.get("scf_warning_images", []))), "last_table_image_count": len(last.get("rows", [])), "previous_table_image_count": len(prev.get("rows", [])), "last_two_energy_deltas": deltas, "max_abs_image_energy_delta_eV": max_delta, "mean_abs_image_energy_delta_eV": mean_delta, "previous_max_image_energy_eV": barrier_prev, "last_max_image_energy_eV": barrier_last, "barrier_delta_eV": (barrier_last - barrier_prev) if barrier_last is not None and barrier_prev is not None else None, "provisional_snapshot_usable": bool(len(last.get("rows", [])) == record["images"]["neb_input"] and all(x["last_energy_eV"] is not None for x in deltas)), "scientific_success": False, "status": "provisional_non_equilibrium_snapshot_only"})
    out = ROOT / "results/provisional_neb_iteration_audit.json"
    out.write_text(json.dumps({"schema_version": "task2-provisional-neb-audit-1.0", "created_utc": datetime.now(timezone.utc).isoformat(), "scientific_success_policy": "always_false_until_explicit_SCF_and_NEB_convergence", "records": records}, indent=2) + "\n")
    with (ROOT / "results/provisional_neb_iteration_audit.csv").open("w", newline="") as fh:
        fields = ["path_id", "element", "outer_neb_iterations", "last_outer_iteration", "last_outer_iteration_scf_image_calls", "last_outer_iteration_scf_warning_images", "last_completed_iteration", "last_completed_iteration_scf_image_calls", "last_completed_iteration_scf_warning_images", "last_table_image_count", "max_abs_image_energy_delta_eV", "mean_abs_image_energy_delta_eV", "barrier_delta_eV", "provisional_snapshot_usable", "scientific_success", "status"]
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k) for k in fields})
    print(json.dumps({"paths": len(records), "provisional_snapshot_usable": sum(r["provisional_snapshot_usable"] for r in records), "scientific_success": 0}, indent=2))


if __name__ == "__main__":
    main()
