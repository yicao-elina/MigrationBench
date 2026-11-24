#!/usr/bin/env python3
"""Audit the retained 81-atom branch of the grouped Foundation benchmark."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from audit_historical_grouped_foundation_benchmark import CAMPAIGN, MODELS, OUT, extract_qe, sha256


PATHS = [f"neb_{i}" for i in range(1, 6)]


def dat_barrier(path: Path) -> tuple[int, float, bool]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 3:
            try:
                rows.append((float(fields[0]), float(fields[1]), float(fields[2])))
            except ValueError:
                continue
    energy = np.asarray([row[1] for row in rows], dtype=float)
    return len(rows), float(np.max(energy)) if len(energy) else math.nan, bool(len(energy) and np.isfinite(energy).all())


def audit_models() -> tuple[list[dict], dict]:
    rows = []
    counts = {"n_json": 0, "n_finite": 0, "n_profile_identity": 0}
    for model in MODELS:
        for path_id in PATHS:
            p = CAMPAIGN / "results/neb_81" / f"{model}__{path_id}.json"
            rec = {"model_id": model, "path_id": path_id, "json": str(p), "json_exists": p.exists()}
            if p.exists():
                counts["n_json"] += 1
                rec["json_sha256"] = sha256(p)
                data = json.loads(p.read_text())
                energies = np.asarray(data.get("energies_eV", []), dtype=float)
                rel = np.asarray(data.get("relative_energies_eV", []), dtype=float)
                finite = bool(data.get("finite", False)) and bool(np.isfinite(energies).all()) and bool(np.isfinite(rel).all())
                identity = len(energies) == len(rel) and len(rel) > 0 and bool(np.allclose(rel, energies - energies[0], atol=1e-10))
                counts["n_finite"] += int(finite)
                counts["n_profile_identity"] += int(identity)
                rec.update({
                    "n_images": int(data.get("n_images", len(rel))),
                    "finite": finite,
                    "profile_identity": identity,
                    "barrier_eV": float(data.get("barrier_eV", math.nan)),
                    "max_force_eV_A": float(np.max(data.get("force_max_eV_per_A", [math.nan]))),
                    "endpoint_delta_eV": float(data.get("endpoint_delta_eV", math.nan)),
                })
            rows.append(rec)
    return rows, counts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qe_rows = []
    for path_id in PATHS:
        out = CAMPAIGN / "data/expanded_81_paths" / path_id / "neb.out"
        dat = CAMPAIGN / "data/expanded_81_paths" / path_id / "sb2te3.dat"
        row = extract_qe(out)
        n_dat, b_dat, finite_dat = dat_barrier(dat)
        row.update({
            "path_id": path_id,
            "dat_file": str(dat),
            "dat_sha256": sha256(dat),
            "dat_n_images": n_dat,
            "dat_barrier_eV": b_dat,
            "dat_finite_energy": finite_dat,
            "output_barrier_minus_dat_eV": row["barrier_eV"] - b_dat if np.isfinite(row["barrier_eV"]) and np.isfinite(b_dat) else math.nan,
        })
        qe_rows.append(row)
    model_rows, counts = audit_models()

    for filename, rows in [("qe_81_path_audit.csv", qe_rows), ("foundation_81_profile_audit.csv", model_rows)]:
        with (OUT / filename).open("w", newline="") as fh:
            fields = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    old_rows = list(csv.DictReader((CAMPAIGN / "analysis/combined_all_paths_barrier_results.csv").open()))
    old_dft = {row["path"]: float(row["dft_barrier_eV"]) for row in old_rows if row["system"] == "81-atom"}
    summary = {
        "system": "81-atom",
        "paths": PATHS,
        "n_qe_paths": len(qe_rows),
        "n_foundation_models": len(MODELS),
        "n_foundation_profiles_expected": len(MODELS) * len(PATHS),
        "foundation_profile_counts": counts,
        "scf_warning_paths": [row["path_id"] for row in qe_rows if row["scf_warning"]],
        "job_done_paths": [row["path_id"] for row in qe_rows if row["job_done"]],
        "formal_reference_candidates": sum(int(row["finite_energy"] and row["finite_error"] and row["job_done"] and row["explicit_convergence_marker"] and not row["scf_warning"]) for row in qe_rows),
        "historical_dft_quality_label": "SCF convergence warning",
        "barrier_reconciliation": {
            row["path_id"]: {
                "reextracted_neb_out_eV": row["barrier_eV"],
                "sb2te3_dat_eV": row["dat_barrier_eV"],
                "historical_combined_table_eV": old_dft.get(row["path_id"]),
            }
            for row in qe_rows
        },
    }
    (OUT / "audit_81_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Historical grouped Foundation-vs-NEB audit: 81-atom branch (2026-10-03)",
        "",
        "The five 81-atom fixed-path profiles are finite and have retained `JOB DONE` markers, but every QE output contains an explicit `SCF convergence NOT achieved` warning. They are therefore stress-test/exploratory references, not formal converged DFT reference answers.",
        "",
        "| Path | Images in `.dat` | `.dat` barrier (eV) | NEB-output barrier (eV) | Difference (eV) | JOB DONE | SCF warning |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in qe_rows:
        lines.append(f"| {row['path_id']} | {row['dat_n_images']} | {row['dat_barrier_eV']:.6f} | {row['barrier_eV']:.6f} | {row['output_barrier_minus_dat_eV']:.3e} | {row['job_done']} | {row['scf_warning']} |")
    lines += [
        "",
        f"Foundation profile audit: {counts['n_json']}/{len(MODELS) * len(PATHS)} JSON profiles found; {counts['n_finite']}/{counts['n_json']} finite; {counts['n_profile_identity']}/{counts['n_json']} satisfy the stored relative-energy identity.",
        "",
        "The old `SCF convergence warning` label is directionally correct, but it must propagate into every barrier comparison and figure caption: these are diagnostic stress-test references, not uniformly converged DFT barriers.",
    ]
    (OUT / "historical_grouped_81_path_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
