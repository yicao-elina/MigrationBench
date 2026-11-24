#!/usr/bin/env python3
"""Build additive, provenance-first products for terminal 1 ns MD failures.

The products intentionally retain finite diagnostics while preventing incomplete
or instability-terminated trajectories from being promoted to transport results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def read_run(run_dir: Path, manifest_dir: Path) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    name = run_dir.name
    manifest_path = manifest_dir / f"{name}.run_manifest.json"
    status_path = manifest_dir / f"{name}.status.json"
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    final = metrics["status"]
    temp = metrics["temperature_K"]
    force = metrics["max_force_eV_A"]
    pressure = metrics["pressure_GPa"]
    hop = metrics["hop_count"]
    heat_flux = metrics["heat_flux"]
    dxy = metrics["msd"]
    stop_step = int(final["last_step"])
    target_step = int(final["target_step"])
    row = {
        "run_id": name,
        "campaign": manifest["campaign"],
        "slurm_job_id": manifest["slurm_job_id"],
        "slurm_array_task_id": manifest["slurm_array_task_id"],
        "family": manifest["family"],
        "training_seed": manifest["training_seed"],
        "md_seed": manifest["md_seed"],
        "model_path": manifest["model"],
        "model_sha256": manifest["model_sha256"],
        "input_path": manifest["input"],
        "input_sha256": manifest["input_sha256"],
        "species_sha256": manifest["species_sha256"],
        "natoms": manifest["natoms"],
        "pbc": ";".join(str(bool(x)).lower() for x in manifest["pbc"]),
        "cell_A": json.dumps(manifest["cell_A"], separators=(",", ":")),
        "timestep_fs": manifest["timestep_fs"],
        "write_interval_steps": manifest["write_interval_steps"],
        "temperature_target_K": manifest["temperature_target_K"],
        "friction_fs^-1": manifest["friction_fs^-1"],
        "software": json.dumps(manifest["software"], sort_keys=True, separators=(",", ":")),
        "host": manifest["host"],
        "status": final["status"],
        "stop_reason": final["stop_reason"],
        "target_step": target_step,
        "last_step": stop_step,
        "absolute_time_ps": number(metrics["duration_ps"]),
        "completion_fraction": number(stop_step / target_step),
        "frames": metrics["frames"],
        "finite_gate": metrics["finite_gate"],
        "complete_step_gate": metrics["complete_step_gate"],
        "scientific_gate": metrics["scientific_gate"],
        "temperature_min_K": number(temp["min"]),
        "temperature_mean_K": number(temp["mean"]),
        "temperature_max_K": number(temp["max"]),
        "max_force_max_eV_A": number(force["max"]),
        "max_force_mean_eV_A": number(force["mean"]),
        "pressure_finite_frames": pressure["finite_frames"],
        "pressure_mean_GPa": number(pressure["mean"]),
        "hop_status": hop["status"],
        "candidate_hop_events": hop["events_between_saved_frames"],
        "heat_flux_status": heat_flux["status"],
        "heat_flux_formula": heat_flux["formula"],
        "kappa_final_W_mK_diagnostic": number(heat_flux["kappa_final_W_mK"]),
        "dxy_200_500_cm2_s_diagnostic": number(dxy["Dxy_200_500_ps"].get("D_cm2_s")),
        "dxy_200_500_r2": number(dxy["Dxy_200_500_ps"].get("r2")),
        "dxy_500_900_cm2_s_diagnostic": number(dxy["Dxy_500_900_ps"].get("D_cm2_s")),
        "dxy_500_900_r2": number(dxy["Dxy_500_900_ps"].get("r2")),
        "trajectory_path": metrics["trajectory"],
        "trajectory_sha256": metrics["trajectory_sha256"],
        "manifest_sha256": sha256(manifest_path),
        "status_sha256": sha256(status_path),
        "publication_status": "FAILURE_MODE;EXCLUDE_FROM_PUBLICATION",
    }
    # Cross-check the separately copied status file against metrics and manifest.
    row["status_crosscheck"] = "PASS" if status == final else "MISMATCH"
    return row


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--postprocess-root", type=Path, default=ROOT / "remote_postprocess_terminal_20261006")
    ap.add_argument("--manifest-root", type=Path, default=ROOT / "remote_terminal_manifests_20261006")
    ap.add_argument("--output-root", type=Path, default=ROOT / "terminal_failure_bundle_20261006")
    args = ap.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = [read_run(p.parent, args.manifest_root) for p in sorted(args.postprocess_root.glob("*/metrics.json"))]
    rows.sort(key=lambda row: (row["family"], int(row["training_seed"])))
    fields = list(rows[0])
    write_csv(args.output_root / "terminal_run_inventory.csv", rows, fields)
    write_tsv(args.output_root / "terminal_run_inventory.tsv", rows, fields)
    (args.output_root / "terminal_run_inventory.json").write_text(json.dumps(rows, indent=2) + "\n")

    registry_fields = [
        "canonical_model_id", "family", "training_seed", "md_seed", "model_path", "model_sha256",
        "input_path", "input_sha256", "species_sha256", "run_id", "job_id", "scientific_status",
    ]
    registry = []
    for row in rows:
        registry.append({
            "canonical_model_id": f"{row['family']}__training_seed_{row['training_seed']}",
            "family": row["family"], "training_seed": row["training_seed"], "md_seed": row["md_seed"],
            "model_path": row["model_path"], "model_sha256": row["model_sha256"],
            "input_path": row["input_path"], "input_sha256": row["input_sha256"],
            "species_sha256": row["species_sha256"], "run_id": row["run_id"],
            "job_id": f"{row['slurm_job_id']}_{row['slurm_array_task_id']}",
            "scientific_status": row["publication_status"],
        })
    write_tsv(args.output_root / "MODEL_REGISTRY.tsv", registry, registry_fields)

    claim_fields = [
        "claim_id", "manuscript_file", "section_figure_table", "exact_claim_or_value",
        "canonical_model_id", "input_id", "run_job_id", "raw_artifact", "derived_artifact",
        "processing_script", "environment", "hashes", "scientific_status", "publication_action",
    ]
    claims = []
    for row in rows:
        model_id = f"{row['family']}__training_seed_{row['training_seed']}"
        claims.extend([
            {
                "claim_id": f"FAIL-{row['training_seed']}-01", "manuscript_file": "discussion_draft.md",
                "section_figure_table": "terminal failure section", "exact_claim_or_value": f"{row['family']} seed {row['training_seed']} terminated at absolute step {row['last_step']} ({row['absolute_time_ps']} ps) with instability_threshold; target was {row['target_step']} steps.",
                "canonical_model_id": model_id, "input_id": row["input_sha256"],
                "run_job_id": f"{row['slurm_job_id']}_{row['slurm_array_task_id']}",
                "raw_artifact": row["trajectory_path"], "derived_artifact": str(args.output_root / "terminal_run_inventory.tsv"),
                "processing_script": "build_terminal_failure_bundle.py", "environment": row["software"],
                "hashes": f"model={row['model_sha256']};input={row['input_sha256']};trajectory={row['trajectory_sha256']};manifest={row['manifest_sha256']}",
                "scientific_status": "FAILURE_MODE", "publication_action": "retain as failure evidence; exclude from formal transport",
            },
            {
                "claim_id": f"FAIL-{row['training_seed']}-02", "manuscript_file": "discussion_draft.md",
                "section_figure_table": "terminal failure section", "exact_claim_or_value": f"Finite diagnostics reached T_max={row['temperature_max_K']} K, max|F|={row['max_force_max_eV_A']} eV/A; these values diagnose instability and are not converged observables.",
                "canonical_model_id": model_id, "input_id": row["input_sha256"],
                "run_job_id": f"{row['slurm_job_id']}_{row['slurm_array_task_id']}",
                "raw_artifact": row["trajectory_path"], "derived_artifact": str(args.output_root / "terminal_run_inventory.tsv"),
                "processing_script": "build_terminal_failure_bundle.py", "environment": row["software"],
                "hashes": f"trajectory={row['trajectory_sha256']};metrics={sha256(args.postprocess_root / row['run_id'] / 'metrics.json')}",
                "scientific_status": "FAILURE_MODE", "publication_action": "diagnostic only; do not use for Dxy/kappa claims",
            },
        ])
    write_tsv(args.output_root / "CLAIM_EVIDENCE_MATRIX.tsv", claims, claim_fields)

    gap = [
        "# From-scratch ablation MD gap report (terminal 1 ns campaign)",
        "",
        "Status: BLOCKED for formal transport comparison; all three selected 1 ns branches terminated by the scientific instability threshold.",
        "",
        "## Verified gates",
        "",
        "- All runs used the same 2050-atom periodic input, the same MD seed, 1 fs timestep, 600 K target, 0.01 fs^-1 friction, and the recorded model/input/species hashes.",
        "- All three runs have finite postprocessed energy/force/temperature/pressure fields and reproducible trajectory hashes.",
        "- None reached the requested 1,000,000 absolute steps; therefore no run passes the complete-step gate.",
        "- Each branch is `FAILURE_MODE` and `EXCLUDE_FROM_PUBLICATION` for formal Dxy, hop-rate, replica uncertainty, or transport claims.",
        "",
        "## Scientific consequences",
        "",
        "- The longer seed-123 lmax-2 branch provides a diagnostic 200–500 ps Dxy fit and candidate hop count only; its later 500–900 ps fit overlaps a time window beyond the recorded terminal time and is not accepted.",
        "- Stress-based heat-flux and running-kappa products are retained as repository diagnostics. They are not converged HFACF/transport results because the trajectory is unstable and incomplete.",
        "- No block/replica uncertainty estimate is scientifically valid for this three-branch set because all selected replicas are incomplete.",
        "- No automatic continuation was performed: the terminal condition is scientific instability, not a wall-time interruption.",
        "",
        "## Required before formal publication",
        "",
        "1. Resolve the force/geometry instability with a validated model/protocol change or a user-approved replacement branch.",
        "2. Re-run all approved replicas to the complete absolute-step target with checkpoint, identity, finite-state, geometry, and continuity gates passing.",
        "3. Recompute block/replica uncertainty, mapped hop events, MSD/Dxy/VACF and, when a physical heat-flux observable is available, HFACF/running integral.",
    ]
    (args.output_root / "GAP_REPORT.md").write_text("\n".join(gap) + "\n")
    release = {
        "bundle": "terminal_failure_bundle_20261006",
        "status": "BLOCKED",
        "scientific_boundary": "finite diagnostics retained; incomplete instability-terminated trajectories excluded from formal transport",
        "runs": [row["run_id"] for row in rows],
        "generated_by": "build_terminal_failure_bundle.py",
        "source_postprocess_root": str(args.postprocess_root),
        "source_manifest_root": str(args.manifest_root),
    }
    (args.output_root / "release_manifest.json").write_text(json.dumps(release, indent=2) + "\n")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "builder_script": str(Path(__file__).resolve()),
        "builder_script_sha256": sha256(Path(__file__).resolve()),
        "plot_atlas_source": "/Users/alina/Documents/Codex/plot_library/src",
        "plot_atlas_recipe_family": "molecular_dynamics / nature_steered_dynamics",
        "scientific_status": "diagnostic bundle only; no formal transport estimates",
    }
    (args.output_root / "environment_manifest.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "output_root": str(args.output_root), "status": "BLOCKED"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
