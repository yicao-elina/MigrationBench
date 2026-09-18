#!/usr/bin/env python3
"""Compare paired direct and geometry-transformed QE endpoint relaxations."""

import argparse
import csv
import hashlib
import html
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from ase.io import read as ase_read

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_neb_path_topology import displacement, norm, read_qe_image  # noqa: E402
from audit_qe_calculator_identity import identity as calculator_identity  # noqa: E402
from parse_qe_relax_output import parse_relax_out  # noqa: E402
from validate_runtime_provenance import validate_runtime_provenance  # noqa: E402


def elapsed_seconds(value):
    if not value:
        return None
    days = 0
    if "-" in value:
        day, value = value.split("-", 1)
        days = int(day)
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    else:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def relaxation_data(job):
    local_dir = Path(job["local_dir"])
    lineage_path = local_dir / "relax_lineage.json"
    segment_dirs = [local_dir]
    if lineage_path.exists():
        lineage = json.loads(lineage_path.read_text())
        segment_dirs = [Path(row["local_dir"]) for row in lineage.get("segments", [])]
    combined = {"steps": []}
    offset = 0
    scf_offset = 0
    for segment_index, segment_dir in enumerate(segment_dirs, start=1):
        output = segment_dir / "relax.out"
        if not output.exists():
            continue
        parsed = parse_relax_out(output, segment_dir / "relax.in")
        for raw_step in parsed["steps"]:
            step = dict(raw_step)
            step["segment_index"] = segment_index
            step["segment_ionic_step"] = raw_step["ionic_step"]
            step["ionic_step"] = offset + raw_step["ionic_step"]
            segment_scf = raw_step.get("cumulative_scf_iterations")
            step["cumulative_scf_iterations"] = (
                scf_offset + segment_scf if segment_scf is not None else None
            )
            combined["steps"].append(step)
        offset += parsed["n_ionic_steps"]
        scf_offset += parsed["total_scf_iterations_seen"]
    return combined


def energy_metrics(parsed):
    steps = parsed.get("steps", [])
    if not steps:
        return {"initial_energy_eV": None, "final_energy_eV": None, "relaxation_energy_drop_eV": None}
    initial = steps[0]["energy_eV"]
    final = steps[-1]["energy_eV"]
    return {
        "initial_energy_eV": initial,
        "final_energy_eV": final,
        "relaxation_energy_drop_eV": initial - final,
    }


def first_force_threshold_step(parsed, threshold):
    for step in parsed.get("steps", []):
        force = step.get("max_atom_force_eV_A")
        if force is not None and force <= threshold:
            return step["ionic_step"]
    return None


def first_sustained_force_threshold(parsed, threshold):
    """Return the first evaluated step after which no observed force rebounds."""
    complete = [
        step for step in parsed.get("steps", [])
        if step.get("force_complete", True)
        and step.get("max_atom_force_eV_A") is not None
    ]
    for index, step in enumerate(complete):
        if all(
            later["max_atom_force_eV_A"] <= threshold
            for later in complete[index:]
        ):
            return {
                "ionic_step": step["ionic_step"],
                "cumulative_scf_iterations": step.get("cumulative_scf_iterations"),
            }
    return {"ionic_step": None, "cumulative_scf_iterations": None}


def finite_ratio(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / denominator


def display_number(value, digits=3):
    return "pending" if value is None else f"{value:.{digits}f}"


def transform_shift_A(manifest_row):
    transform = manifest_row.get("transform", {})
    metrics = transform.get("metrics", {})
    value = metrics.get("cr_displacement_A", transform.get("cr_displacement_A"))
    if value is None:
        raise ValueError("Transform manifest lacks Cr displacement")
    return float(value)


def local_signature(step, cr_index, fallback_cell=None):
    positions = step["positions_A"]
    cell = step.get("cell_A") or fallback_cell
    distances = [
        norm(displacement(positions[cr_index], position, cell))
        for index, position in enumerate(positions)
        if index != cr_index
    ]
    return sorted(distances)[:12]


def basin_comparison(direct_step, transformed_step, source_input, args, final=False):
    if direct_step is None or transformed_step is None:
        return {
            "basin_equivalence": "pending",
            "diagnostic_current_geometry_equivalence": "unavailable",
        }
    source = read_qe_image(source_input)
    cr_indices = [i for i, symbol in enumerate(source["symbols"]) if symbol == "Cr"]
    if len(cr_indices) != 1:
        return {"basin_equivalence": "invalid_migrant_count"}
    cr_index = cr_indices[0]
    cell = direct_step.get("cell_A") or transformed_step.get("cell_A") or source["cell"]
    vectors = [
        displacement(a, b, cell)
        for a, b in zip(direct_step["positions_A"], transformed_step["positions_A"])
    ]
    host = [vector for index, vector in enumerate(vectors) if index != cr_index]
    host_rmsd = math.sqrt(sum(norm(vector) ** 2 for vector in host) / len(host))
    cr_distance = norm(vectors[cr_index])
    direct_signature = local_signature(direct_step, cr_index, cell)
    transformed_signature = local_signature(transformed_step, cr_index, cell)
    signature_rms = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(direct_signature, transformed_signature))
        / len(direct_signature)
    )
    same = (
        cr_distance <= args.basin_cr_distance_A
        and host_rmsd <= args.basin_host_rmsd_A
        and signature_rms <= args.basin_local_signature_rms_A
    )
    return {
        "basin_equivalence": ("same" if same else "different") if final else "pending",
        "diagnostic_current_geometry_equivalence": "same" if same else "different",
        "final_cr_distance_A": cr_distance,
        "final_host_rmsd_A": host_rmsd,
        "final_local_signature_rms_A": signature_rms,
    }


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_acceptance_artifact(acceptance_path, recorded):
    if not recorded:
        return None
    path = Path(recorded)
    if path.is_file():
        return path
    candidate = acceptance_path.parent / path.name
    return candidate if candidate.is_file() else None


def endpoint_repeat_record(job, acceptance_path=None):
    if acceptance_path is None:
        return {"state": "pending", "record": None, "accepted_step": None}
    payload = json.loads(acceptance_path.read_text())
    records = payload.get("endpoint_records", payload.get("records", []))
    matches = [
        row for row in records
        if str(row.get("parent_job_id")) == str(job["job_id"])
        and row.get("path_id") == job["path_id"]
        and int(row.get("image_index_qe", -1)) == int(job["image_index_qe"])
    ]
    if len(matches) != 1:
        return {
            "state": "pending" if not matches else "invalid_duplicate",
            "record": None,
            "accepted_step": None,
        }
    record = matches[0]
    if record.get("status") != "accepted":
        return {"state": "rejected", "record": record, "accepted_step": None}
    structure = local_acceptance_artifact(acceptance_path, record.get("accepted_structure"))
    expected_hash = record.get("accepted_structure_sha256")
    if not structure or not expected_hash or sha256_file(structure) != expected_hash:
        return {"state": "invalid_structure_provenance", "record": record, "accepted_step": None}
    atoms = ase_read(structure)
    return {
        "state": "accepted",
        "record": record,
        "accepted_step": {
            "positions_A": atoms.positions.tolist(),
            "cell_A": atoms.cell.array.tolist(),
        },
    }


def resolve_acceptance_paths(
    combined_acceptance=None,
    direct_acceptance=None,
    transformed_acceptance=None,
):
    if combined_acceptance is not None and (
        direct_acceptance is not None or transformed_acceptance is not None
    ):
        raise ValueError(
            "Use either --endpoint-acceptance or the two arm-specific acceptance files, not both"
        )
    if (direct_acceptance is None) != (transformed_acceptance is None):
        raise ValueError(
            "--direct-endpoint-acceptance and --transformed-endpoint-acceptance must be provided together"
        )
    if combined_acceptance is not None:
        return combined_acceptance, combined_acceptance
    return direct_acceptance, transformed_acceptance


def lineage_runtime_provenance(job):
    local_dir = Path(job["local_dir"]).resolve()
    lineage_path = local_dir / "relax_lineage.json"
    if lineage_path.exists():
        segments = [Path(row["local_dir"]).resolve() for row in json.loads(lineage_path.read_text()).get("segments", [])]
    else:
        segments = [local_dir]
    records = []
    for segment_dir in segments:
        input_path = segment_dir / "relax.in"
        manifest_path = segment_dir / "endpoint_relax_manifest.json"
        expected_job_id = str(job["job_id"]) if segment_dir == local_dir else None
        if expected_job_id is None and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            expected_job_id = manifest.get("submission", {}).get("job_id")
        if not input_path.exists() or expected_job_id is None:
            records.append({
                "segment_dir": str(segment_dir),
                "passed": False,
                "failed_checks": ["input_or_expected_job_id_missing"],
            })
            continue
        validation = validate_runtime_provenance(
            segment_dir / "runtime_provenance.json",
            {"relax_input": sha256_file(input_path)},
            ["pw.x"],
            str(expected_job_id),
        )
        records.append({"segment_dir": str(segment_dir), **validation})
    return {
        "passed": bool(records) and all(record["passed"] for record in records),
        "segments": records,
        "segment_count": len(records),
    }


def calculator_comparison(direct_job, transformed_job, acceptance_path=None):
    direct = calculator_identity(Path(direct_job["local_dir"]) / "relax.in", "ab_direct")
    transformed = calculator_identity(
        Path(transformed_job["local_dir"]) / "relax.in", "ab_transformed"
    )
    direct_key = direct["calculator_identity"]
    transformed_key = transformed["calculator_identity"]
    matched = (
        direct["identity_complete"]
        and transformed["identity_complete"]
        and direct_key == transformed_key
    )
    acceptance = None
    if acceptance_path is not None and acceptance_path.exists():
        acceptance = json.loads(acceptance_path.read_text())
    accepted = bool(
        matched
        and acceptance
        and acceptance.get("accepted") is True
        and acceptance.get("calculator_identity") == direct_key
    )
    direct_runtime = lineage_runtime_provenance(direct_job)
    transformed_runtime = lineage_runtime_provenance(transformed_job)
    return {
        "direct_calculator_identity": direct_key,
        "transformed_calculator_identity": transformed_key,
        "calculator_identities_match": matched,
        "calculator_identity_accepted_by_n24": accepted,
        "calculator_acceptance_artifact": str(acceptance_path.resolve()) if acceptance_path and acceptance_path.exists() else None,
        "direct_lineage_runtime_provenance": direct_runtime,
        "transformed_lineage_runtime_provenance": transformed_runtime,
        "lineage_runtime_provenance_complete": direct_runtime["passed"] and transformed_runtime["passed"],
    }


def comparison_decision(
    both_converged, basin_equivalence, calculator, repeat_acceptance_state="accepted"
):
    if not calculator["calculator_identities_match"]:
        return "calculator_identity_mismatch"
    if not both_converged:
        return "pending_not_converged"
    if not calculator.get("lineage_runtime_provenance_complete", False):
        return "same_basin_runtime_provenance_pending"
    if repeat_acceptance_state == "rejected":
        return "same_basin_repeat_acceptance_rejected"
    if repeat_acceptance_state != "accepted":
        return "same_basin_repeat_acceptance_pending"
    if basin_equivalence == "different":
        return "basin_changed_no_speedup"
    if basin_equivalence != "same":
        return "invalid_basin_classification"
    if not calculator["calculator_identity_accepted_by_n24"]:
        return "same_basin_calculator_gate_pending"
    return "comparable_same_basin_accepted_calculator"


def status_index(path):
    payload = json.loads(path.read_text())
    return {(row["path_id"], int(row["image_index_qe"])): row for row in payload["jobs"]}


def svg_polyline(points, x0, y0, width, height, x_max, y_min, y_max):
    if not points:
        return ""
    x_scale = width / max(x_max, 1)
    y_span = max(y_max - y_min, 1.0e-12)
    coordinates = []
    for x_value, y_value in points:
        x = x0 + x_value * x_scale
        y = y0 + height - (y_value - y_min) * height / y_span
        coordinates.append(f"{x:.2f},{y:.2f}")
    return " ".join(coordinates)


def write_curve_svg(path, rows):
    pairs = sorted({(row["path_id"], int(row["image_index_qe"])) for row in rows})
    colors = {"direct": "#1f77b4", "transformed": "#d1495b"}
    panel_width, panel_height = 520, 240
    margin_x, margin_y = 68, 52
    width = margin_x * 2 + panel_width
    height = 72 + len(pairs) * (panel_height + 48)
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="68" y="30" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#202124">Direct vs geometry-transformed QE relaxation</text>',
        '<line x1="390" y1="25" x2="418" y2="25" stroke="#1f77b4" stroke-width="3"/><text x="426" y="30" font-family="Arial, sans-serif" font-size="12">direct</text>',
        '<line x1="490" y1="25" x2="518" y2="25" stroke="#d1495b" stroke-width="3"/><text x="526" y="30" font-family="Arial, sans-serif" font-size="12">transformed</text>',
    ]
    for panel, key in enumerate(pairs):
        subset = [row for row in rows if (row["path_id"], int(row["image_index_qe"])) == key]
        x_max = max((int(row["ionic_step"]) for row in subset), default=1)
        forces = [float(row["max_atom_force_eV_A"]) for row in subset if row["max_atom_force_eV_A"] not in (None, "")]
        y_min = 0.0
        y_max = max(forces + [0.05]) * 1.08
        x0 = margin_x
        y0 = 62 + panel * (panel_height + 48)
        chunks.extend([
            f'<text x="{x0}" y="{y0 - 12}" font-family="Arial, sans-serif" font-size="14" font-weight="700">{html.escape(key[0])}, image {key[1]}</text>',
            f'<rect x="{x0}" y="{y0}" width="{panel_width}" height="{panel_height}" fill="#fafafa" stroke="#c9cdd2"/>',
        ])
        for threshold in (0.5, 0.2, 0.1, 0.05):
            if threshold > y_max:
                continue
            y = y0 + panel_height - threshold * panel_height / max(y_max, 1.0e-12)
            chunks.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + panel_width}" y2="{y:.2f}" stroke="#d6d9dc" stroke-dasharray="4 4"/>')
            chunks.append(f'<text x="{x0 - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="10">{threshold:g}</text>')
        for variant in ("direct", "transformed"):
            points = sorted(
                (int(row["ionic_step"]), float(row["max_atom_force_eV_A"]))
                for row in subset
                if row["variant"] == variant and row["max_atom_force_eV_A"] not in (None, "")
            )
            polyline = svg_polyline(points, x0, y0, panel_width, panel_height, x_max, y_min, y_max)
            if polyline:
                chunks.append(f'<polyline points="{polyline}" fill="none" stroke="{colors[variant]}" stroke-width="2.5"/>')
        chunks.extend([
            f'<text x="{x0 + panel_width / 2}" y="{y0 + panel_height + 30}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">Ionic step</text>',
            f'<text x="20" y="{y0 + panel_height / 2}" transform="rotate(-90 20 {y0 + panel_height / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">Max atomic force (eV/A)</text>',
        ])
    chunks.append("</svg>")
    path.write_text("\n".join(chunks) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-status", type=Path, required=True)
    parser.add_argument("--transformed-status", type=Path, required=True)
    parser.add_argument("--transform-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--basin-cr-distance-A", type=float, default=0.25)
    parser.add_argument("--basin-host-rmsd-A", type=float, default=0.15)
    parser.add_argument("--basin-local-signature-rms-A", type=float, default=0.15)
    parser.add_argument(
        "--calculator-acceptance", type=Path,
        help="Final N24 JSON with accepted=true and the exact calculator_identity",
    )
    parser.add_argument(
        "--endpoint-acceptance", type=Path,
        help="Legacy combined repeat-acceptance batch covering both A/B arms",
    )
    parser.add_argument(
        "--direct-endpoint-acceptance", type=Path,
        help="Repeat-acceptance batch for the direct arm; requires the transformed-arm file",
    )
    parser.add_argument(
        "--transformed-endpoint-acceptance", type=Path,
        help="Repeat-acceptance batch for the transformed arm; requires the direct-arm file",
    )
    args = parser.parse_args()

    direct_acceptance, transformed_acceptance = resolve_acceptance_paths(
        args.endpoint_acceptance,
        args.direct_endpoint_acceptance,
        args.transformed_endpoint_acceptance,
    )

    direct = status_index(args.direct_status)
    transformed = status_index(args.transformed_status)
    manifest = json.loads(args.transform_manifest.read_text())
    manifest_jobs = manifest.get("jobs") or manifest.get("transformed_jobs")
    if not manifest_jobs:
        raise ValueError(
            "Transform manifest must contain jobs or transformed_jobs"
        )
    manifest_index = {
        (row["source_path_id"], int(row["source_image_index_qe"])): row
        for row in manifest_jobs
    }
    rows = []
    curve_rows = []
    selected_keys = sorted(set(transformed) & set(direct) & set(manifest_index))
    for key in selected_keys:
        direct_job = direct[key]
        transformed_job = transformed[key]
        transform = manifest_index[key]
        direct_parsed = direct_job["parsed"]
        transformed_parsed = transformed_job["parsed"]
        direct_full = relaxation_data(direct_job)
        transformed_full = relaxation_data(transformed_job)
        direct_energy = energy_metrics(direct_full)
        transformed_energy = energy_metrics(transformed_full)
        direct_repeat = endpoint_repeat_record(direct_job, direct_acceptance)
        transformed_repeat = endpoint_repeat_record(
            transformed_job, transformed_acceptance
        )
        repeat_states = {direct_repeat["state"], transformed_repeat["state"]}
        repeat_acceptance_state = (
            "accepted" if repeat_states == {"accepted"}
            else "rejected" if "rejected" in repeat_states
            else "pending"
        )
        both_converged = (
            direct_job["classification"] == "accepted_local_minimum"
            and transformed_job["classification"] == "accepted_local_minimum"
        )
        direct_basin_step = direct_repeat["accepted_step"] or (
            direct_full["steps"][-1] if direct_full["steps"] else None
        )
        transformed_basin_step = transformed_repeat["accepted_step"] or (
            transformed_full["steps"][-1] if transformed_full["steps"] else None
        )
        basin = basin_comparison(
            direct_basin_step,
            transformed_basin_step,
            Path(direct_job["local_dir"]) / "relax.in",
            args,
            final=both_converged and repeat_acceptance_state == "accepted",
        )
        calculator = calculator_comparison(
            direct_job, transformed_job, args.calculator_acceptance
        )
        comparison_status = comparison_decision(
            both_converged,
            basin["basin_equivalence"],
            calculator,
            repeat_acceptance_state,
        )
        comparable = comparison_status == "comparable_same_basin_accepted_calculator"
        direct_scf = direct_parsed.get("lineage_total_scf_iterations", direct_parsed.get("total_scf_iterations_seen"))
        transformed_scf = transformed_parsed.get("lineage_total_scf_iterations", transformed_parsed.get("total_scf_iterations_seen"))
        direct_steps = direct_parsed.get("lineage_total_ionic_steps", direct_parsed.get("n_ionic_steps"))
        transformed_steps = transformed_parsed.get("lineage_total_ionic_steps", transformed_parsed.get("n_ionic_steps"))
        direct_elapsed = elapsed_seconds(direct_job.get("sacct_main", {}).get("Elapsed") or direct_job.get("elapsed"))
        transformed_elapsed = elapsed_seconds(transformed_job.get("sacct_main", {}).get("Elapsed") or transformed_job.get("elapsed"))
        row = {
                "path_id": key[0],
                "image_index_qe": key[1],
                "direct_job_id": direct_job["job_id"],
                "transformed_job_id": transformed_job["job_id"],
                "transform_shift_A": transform_shift_A(transform),
                "direct_repeat_acceptance": direct_repeat["state"],
                "transformed_repeat_acceptance": transformed_repeat["state"],
                "direct_repeat_acceptance_artifact": (
                    str(direct_acceptance.resolve()) if direct_acceptance else None
                ),
                "direct_repeat_acceptance_artifact_sha256": (
                    sha256_file(direct_acceptance) if direct_acceptance else None
                ),
                "transformed_repeat_acceptance_artifact": (
                    str(transformed_acceptance.resolve())
                    if transformed_acceptance else None
                ),
                "transformed_repeat_acceptance_artifact_sha256": (
                    sha256_file(transformed_acceptance)
                    if transformed_acceptance else None
                ),
                "repeat_acceptance_artifact": (
                    str(args.endpoint_acceptance.resolve())
                    if args.endpoint_acceptance else None
                ),
                "direct_status": direct_job["classification"],
                "transformed_status": transformed_job["classification"],
                **basin,
                **calculator,
                "direct_ionic_steps": direct_steps,
                "transformed_ionic_steps": transformed_steps,
                "direct_scf_iterations": direct_scf,
                "transformed_scf_iterations": transformed_scf,
                "direct_elapsed_seconds": direct_elapsed,
                "transformed_elapsed_seconds": transformed_elapsed,
                "direct_current_max_force_eV_A": direct_parsed.get("final_max_atom_force_eV_A"),
                "transformed_current_max_force_eV_A": transformed_parsed.get("final_max_atom_force_eV_A"),
                "diagnostic_prefix_ionic_step_ratio": finite_ratio(direct_steps, transformed_steps),
                "diagnostic_prefix_scf_iteration_ratio": finite_ratio(direct_scf, transformed_scf),
                "direct_initial_energy_eV": direct_energy["initial_energy_eV"],
                "transformed_initial_energy_eV": transformed_energy["initial_energy_eV"],
                "transformed_minus_direct_initial_energy_eV": (
                    transformed_energy["initial_energy_eV"] - direct_energy["initial_energy_eV"]
                    if direct_energy["initial_energy_eV"] is not None and transformed_energy["initial_energy_eV"] is not None
                    else None
                ),
                "direct_final_energy_eV": direct_energy["final_energy_eV"],
                "transformed_final_energy_eV": transformed_energy["final_energy_eV"],
                "absolute_final_energy_difference_eV": (
                    abs(transformed_energy["final_energy_eV"] - direct_energy["final_energy_eV"])
                    if direct_energy["final_energy_eV"] is not None and transformed_energy["final_energy_eV"] is not None
                    else None
                ),
                "direct_relaxation_energy_drop_eV": direct_energy["relaxation_energy_drop_eV"],
                "transformed_relaxation_energy_drop_eV": transformed_energy["relaxation_energy_drop_eV"],
                "ionic_step_speedup": (float(direct_steps) / transformed_steps) if comparable and transformed_steps else None,
                "scf_iteration_speedup": (float(direct_scf) / transformed_scf) if comparable and transformed_scf else None,
                "walltime_speedup": (float(direct_elapsed) / transformed_elapsed) if comparable and transformed_elapsed else None,
                "comparison_status": comparison_status,
            }
        for threshold in (0.5, 0.2, 0.1, 0.05):
            label = str(threshold).replace(".", "p")
            direct_hit = first_force_threshold_step(direct_full, threshold)
            transformed_hit = first_force_threshold_step(transformed_full, threshold)
            row[f"direct_first_step_fmax_lte_{label}"] = direct_hit
            row[f"transformed_first_step_fmax_lte_{label}"] = transformed_hit
            row[f"diagnostic_threshold_step_ratio_{label}"] = finite_ratio(direct_hit, transformed_hit)
            direct_sustained = first_sustained_force_threshold(direct_full, threshold)
            transformed_sustained = first_sustained_force_threshold(
                transformed_full, threshold
            )
            row[f"direct_first_sustained_step_fmax_lte_{label}"] = (
                direct_sustained["ionic_step"]
            )
            row[f"transformed_first_sustained_step_fmax_lte_{label}"] = (
                transformed_sustained["ionic_step"]
            )
            row[f"direct_scf_to_sustained_fmax_lte_{label}"] = (
                direct_sustained["cumulative_scf_iterations"]
            )
            row[f"transformed_scf_to_sustained_fmax_lte_{label}"] = (
                transformed_sustained["cumulative_scf_iterations"]
            )
            row[f"diagnostic_prefix_sustained_step_ratio_{label}"] = finite_ratio(
                direct_sustained["ionic_step"], transformed_sustained["ionic_step"]
            )
            row[f"diagnostic_prefix_sustained_scf_ratio_{label}"] = finite_ratio(
                direct_sustained["cumulative_scf_iterations"],
                transformed_sustained["cumulative_scf_iterations"],
            )
            row[f"formal_sustained_step_speedup_{label}"] = (
                finite_ratio(
                    direct_sustained["ionic_step"],
                    transformed_sustained["ionic_step"],
                ) if comparable else None
            )
            row[f"formal_sustained_scf_speedup_{label}"] = (
                finite_ratio(
                    direct_sustained["cumulative_scf_iterations"],
                    transformed_sustained["cumulative_scf_iterations"],
                ) if comparable else None
            )
        rows.append(row)
        for variant, job, parsed in (
            ("direct", direct_job, direct_full),
            ("transformed", transformed_job, transformed_full),
        ):
            final_energy = parsed["steps"][-1]["energy_eV"] if parsed["steps"] else None
            for step in parsed["steps"]:
                curve_rows.append(
                    {
                        "path_id": key[0],
                        "image_index_qe": key[1],
                        "variant": variant,
                        "job_id": job["job_id"],
                        "ionic_step": step["ionic_step"],
                        "energy_eV": step["energy_eV"],
                        "energy_above_current_final_eV": step["energy_eV"] - final_energy,
                        "max_atom_force_eV_A": step["max_atom_force_eV_A"],
                        "total_force_eV_A": step["total_force_eV_A"],
                        "cumulative_scf_iterations": step.get(
                            "cumulative_scf_iterations"
                        ),
                    }
                )

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (out_dir / "paired_relaxation_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if curve_rows:
        with (out_dir / "relaxation_curves.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
            writer.writeheader()
            writer.writerows(curve_rows)
        write_curve_svg(out_dir / "force_relaxation_curves.svg", curve_rows)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "basin_equivalence_thresholds": {
            "cr_distance_A": args.basin_cr_distance_A,
            "host_rmsd_A": args.basin_host_rmsd_A,
            "local_signature_rms_A": args.basin_local_signature_rms_A,
        },
        "rows": rows,
        "relaxation_curve_rows": len(curve_rows),
    }
    (out_dir / "paired_relaxation_comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    report = [
        "# Direct Versus Geometry-Transformed QE Relaxation",
        "",
        "Speedup is reported only when both relaxations pass the force gate, converge to the same basin, every lineage segment passes input/job/pw.x runtime provenance, calculator identities match, and that identity is accepted by N24. Prefix ratios, current-geometry basin diagnostics, and first or sustained threshold crossings from a running prefix are not final speedups.",
        "",
        "| Path | Image | Shift (A) | Direct steps / fmax | Transformed steps / fmax | Prefix step ratio | Basin | Comparison |",
        "|---|---:|---:|---|---|---:|---|---|",
    ]
    for row in rows:
        report.append(
            "| {path} | {image} | {shift} | {direct_steps} / {direct_force} | "
            "{transformed_steps} / {transformed_force} | {ratio} | {basin} | {status} |".format(
                path=row["path_id"],
                image=row["image_index_qe"],
                shift=display_number(row["transform_shift_A"]),
                direct_steps=row["direct_ionic_steps"],
                direct_force=display_number(row["direct_current_max_force_eV_A"]),
                transformed_steps=row["transformed_ionic_steps"],
                transformed_force=display_number(row["transformed_current_max_force_eV_A"]),
                ratio=display_number(row["diagnostic_prefix_ionic_step_ratio"]),
                basin=row["basin_equivalence"],
                status=row["comparison_status"],
            )
        )
    (out_dir / "README.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"pairs": len(rows), "comparable": sum(row["comparison_status"] == "comparable_same_basin_accepted_calculator" for row in rows), "output": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
