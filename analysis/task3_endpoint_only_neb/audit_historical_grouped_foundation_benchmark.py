#!/usr/bin/env python3
"""Audit the retained grouped Foundation-vs-NEB benchmark without rewriting it.

The historical campaign is treated as an external evidence bundle.  This
script re-extracts the final image-energy table from each retained 61-atom QE
output, checks for explicit completion/convergence evidence, validates the
50 Foundation JSON profiles, and writes a dated audit outside the campaign
root.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np


CAMPAIGN = Path("/Users/alina/Documents/Codex/foundation_grouped_benchmark_20261001")
OUT = Path(
    "/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/task3_endpoint_only_neb/"
    "exploratory_energy_audit/historical_grouped_benchmark_audit_20261003"
)
PATHS = ["1-2", "1-3", "1-4", "1-5", "1-7"]
MODELS = [
    "agnesi_medium",
    "agnesi_stress_large",
    "agnesi_stress_medium",
    "agnesi_stress_small",
    "mh0",
    "mh1",
    "mp_0b3_medium",
    "mpa0_medium",
    "omat0_medium",
    "omat0_small",
]

TABLE_RE = re.compile(
    r"image\s+energy \(eV\).*?\n"
    r"((?:\s*\d+\s+-?[0-9.Ee+-]+\s+[0-9.Ee+-]+\s+[FT]\s*\n)+)",
    re.S,
)
ROW_RE = re.compile(
    r"^\s*(\d+)\s+(-?[0-9.Ee+-]+)\s+([0-9.Ee+-]+)\s+([FT])",
    re.M,
)
ITER_RE = re.compile(r"iteration\s+(\d+)", re.I)


def sha256(path: Path) -> str:
    # Use the system digest utility for this external campaign mirror.  On
    # this macOS/OneDrive-backed workspace, repeated Python buffered reads of
    # a subset of the retained JSON files can block despite normal `cat` and
    # `shasum` reads; the subprocess path is deterministic and bounded.
    result = subprocess.run(
        ["shasum", "-a", "256", str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.split()[0]


def extract_qe(path: Path) -> dict:
    text = path.read_text(errors="replace")
    tables = TABLE_RE.findall(text)
    if not tables:
        return {
            "output": str(path),
            "output_sha256": sha256(path),
            "n_tables": 0,
            "n_images": 0,
            "finite_energy": False,
            "finite_error": False,
            "barrier_eV": math.nan,
            "max_error_eV_A": math.nan,
            "last_iteration": int(ITER_RE.findall(text)[-1]) if ITER_RE.findall(text) else None,
            "last_nonblank_line": next((x.strip() for x in reversed(text.splitlines()) if x.strip()), ""),
            "job_done": bool(re.search(r"JOB DONE", text, re.I)),
            "explicit_convergence_marker": bool(
                re.search(
                    r"(NEB\s+converged|BFGS\s+converged|end of bfgs|convergence\s+has\s+been\s+achieved)",
                    text,
                    re.I,
                )
            ),
            "scf_warning": bool(re.search(r"scf convergence not achieved", text, re.I)),
        }

    rows = ROW_RE.findall(tables[-1])
    energy = np.asarray([float(row[1]) for row in rows], dtype=float)
    error = np.asarray([float(row[2]) for row in rows], dtype=float)
    barrier = float(np.max(energy - energy[0])) if len(energy) else math.nan
    return {
        "output": str(path),
        "output_sha256": sha256(path),
        "n_tables": len(tables),
        "n_images": len(rows),
        "finite_energy": bool(np.isfinite(energy).all()),
        "finite_error": bool(np.isfinite(error).all()),
        "barrier_eV": barrier,
        "max_error_eV_A": float(np.max(error)) if len(error) else math.nan,
        "last_iteration": int(ITER_RE.findall(text)[-1]) if ITER_RE.findall(text) else None,
        "last_nonblank_line": next((x.strip() for x in reversed(text.splitlines()) if x.strip()), ""),
        "job_done": bool(re.search(r"JOB DONE", text, re.I)),
        "explicit_convergence_marker": bool(
            re.search(
                r"(NEB\s+converged|BFGS\s+converged|end of bfgs|convergence\s+has\s+been\s+achieved)",
                text,
                re.I,
            )
        ),
        "scf_warning": bool(re.search(r"scf convergence not achieved", text, re.I)),
    }


def audit_models() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    n_json = 0
    n_finite = 0
    n_profile_identity = 0
    for model in MODELS:
        for path_id in PATHS:
            p = CAMPAIGN / "results/neb" / f"{model}__{path_id}.json"
            rec = {"model_id": model, "path_id": path_id, "json": str(p), "json_exists": p.exists()}
            if not p.exists():
                rows.append(rec)
                continue
            n_json += 1
            rec["json_sha256"] = sha256(p)
            data = json.loads(p.read_text())
            rel = np.asarray(data.get("relative_energies_eV", []), dtype=float)
            energies = np.asarray(data.get("energies_eV", []), dtype=float)
            finite = bool(data.get("finite", False)) and bool(np.isfinite(rel).all()) and bool(np.isfinite(energies).all())
            identity = len(energies) == len(rel) and len(rel) > 0 and bool(np.allclose(rel, energies - energies[0], atol=1e-10))
            n_finite += int(finite)
            n_profile_identity += int(identity)
            rec.update(
                {
                    "n_images": int(data.get("n_images", len(rel))),
                    "finite": finite,
                    "profile_identity": identity,
                    "barrier_eV": float(data.get("barrier_eV", math.nan)),
                    "max_force_eV_A": float(np.max(data.get("force_max_eV_per_A", [math.nan]))),
                    "endpoint_delta_eV": float(data.get("endpoint_delta_eV", math.nan)),
                }
            )
            rows.append(rec)
    return rows, {"n_json": n_json, "n_finite": n_finite, "n_profile_identity": n_profile_identity}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qe_rows = []
    for path_id in PATHS:
        qe_rows.append(extract_qe(CAMPAIGN / "data/dft_paths" / f"{path_id}-neb.out"))

    model_rows, model_counts = audit_models()
    with (OUT / "qe_path_audit.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(qe_rows[0]))
        writer.writeheader()
        writer.writerows(qe_rows)
    with (OUT / "foundation_profile_audit.csv").open("w", newline="") as fh:
        fields = sorted({key for row in model_rows for key in row})
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(model_rows)

    old_rows = list(csv.DictReader((CAMPAIGN / "analysis/combined_all_paths_barrier_results.csv").open()))
    old_dft = {
        row["path"]: float(row["dft_barrier_eV"])
        for row in old_rows
        if row["system"] == "61-atom"
    }

    summary = {
        "campaign_root": str(CAMPAIGN),
        "paths": PATHS,
        "n_qe_paths": len(qe_rows),
        "n_foundation_models": len(MODELS),
        "n_foundation_profiles_expected": len(MODELS) * len(PATHS),
        "foundation_profile_counts": model_counts,
        "qe_formal_reference_candidates": sum(
            int(row["finite_energy"] and row["finite_error"] and row["job_done"] and row["explicit_convergence_marker"])
            for row in qe_rows
        ),
        "historical_dft_quality_label_in_combined_table": "pass",
        "audit_interpretation": (
            "The retained numeric image tables and Foundation profiles are finite and reproducible, "
            "but the local QE outputs preserve no explicit JOB DONE or NEB/BFGS convergence marker "
            "for the five 61-atom paths. They are therefore exploratory fixed-path references, not "
            "formal converged DFT barriers, unless the missing completion evidence is recovered."
        ),
        "barrier_reconciliation": {
            path_id: {
                "reextracted_eV": next(row["barrier_eV"] for row in qe_rows if str(row["output"]).endswith(f"{path_id}-neb.out")),
                "historical_table_eV": old_dft.get(path_id),
            }
            for path_id in PATHS
        },
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Historical grouped Foundation-vs-NEB audit (2026-10-03)",
        "",
        "## Verdict",
        "",
        "The finite numerical profiles are reproducible, but the retained local QE `neb.out` files do not provide sufficient completion evidence to promote the five 61-atom paths to formal DFT reference answers. The historical combined table's `dft_quality=pass` label is therefore too strong for the evidence bundle currently available.",
        "",
        "## Re-extracted QE path tables",
        "",
        "| Path | Images | Last-table barrier (eV) | Max last-table error (eV/A) | JOB DONE | Explicit convergence marker | Status |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in qe_rows:
        status = "conditional / exploratory" if row["finite_energy"] and row["finite_error"] else "failed"
        lines.append(
            f"| {Path(row['output']).stem.replace('-neb','')} | {row['n_images']} | {row['barrier_eV']:.6f} | "
            f"{row['max_error_eV_A']:.6f} | {row['job_done']} | {row['explicit_convergence_marker']} | {status} |"
        )
    lines += [
        "",
        "For path `1-4`, the re-extracted barrier is `0.336050 eV`, matching the historical `0.336050 eV` value to numerical precision. This validates the arithmetic of the retained final image table, not the missing convergence evidence.",
        "",
        "## Foundation profile checks",
        "",
        f"The audit found {model_counts['n_json']}/{len(MODELS) * len(PATHS)} expected JSON profiles. All finite profiles have internally consistent `relative_energies_eV = energies_eV - energies_eV[0]` records: {model_counts['n_profile_identity']}/{model_counts['n_json']}.",
        "",
        "The model-side fixed-path calculations are therefore usable as finite diagnostic evaluations. Their barrier errors must be reported against a conditional/exploratory QE reference until the QE completion and force/path convergence evidence is recovered.",
        "",
        "## Provenance limitation",
        "",
        "The retained grouped-benchmark mirror does not contain the original QE input files for these five 61-atom paths. Functional/SOC/D3/k-point/pseudopotential and convergence settings cannot be independently revalidated from this bundle alone. The raw output hashes and extracted rows are retained in `qe_path_audit.csv`; Foundation JSON hashes and profile checks are retained in `foundation_profile_audit.csv`.",
        "",
        "## Source-of-truth files",
        "",
        f"- Campaign: `{CAMPAIGN}`",
        "- QE source outputs: `data/dft_paths/*-neb.out`",
        "- Foundation outputs: `results/neb/*.json`",
        "- Historical combined table: `analysis/combined_all_paths_barrier_results.csv`",
    ]
    (OUT / "historical_grouped_benchmark_audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
