#!/usr/bin/env python3
"""Compare MLFF NEB preconditioners using path-optimization observables."""

import argparse
import csv
import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trapezoid_auc(iterations, field):
    return sum(
        (right["optimizer_iteration"] - left["optimizer_iteration"])
        * (left[field] + right[field]) / 2
        for left, right in zip(iterations, iterations[1:])
    )


def first_step_below(iterations, field, threshold):
    return next(
        (point["optimizer_iteration"] for point in iterations if point[field] <= threshold),
        None,
    )


def metrics(run):
    main_manifest_path = Path(run["main_manifest"]).resolve()
    history_path = Path(run["history_manifest"]).resolve()
    main = json.loads(main_manifest_path.read_text())
    history = json.loads(history_path.read_text())
    iterations = history["iteration_summaries"]
    if not iterations:
        raise ValueError(f"No iteration summaries in {history_path}")
    if main["seed"] != run["seed"]:
        raise ValueError(f"Seed mismatch for {run['label']}")
    if main["input_images_sha256"] != run["input_images_sha256"]:
        raise ValueError(f"Input image hash mismatch for {run['label']}")
    first, last = iterations[0], iterations[-1]
    steps = max(last["optimizer_iteration"] - first["optimizer_iteration"], 0)
    residual_ratio = (
        last["max_neb_residual_eV_A"] / first["max_neb_residual_eV_A"]
        if first["max_neb_residual_eV_A"] > 0 else None
    )
    log_slope = (
        math.log(residual_ratio) / steps
        if residual_ratio is not None and residual_ratio > 0 and steps > 0 else None
    )
    return {
        "label": run["label"],
        "run_role": main["run_role"],
        "calculator": main["calculator"],
        "model_path": run.get("model_path") or main.get("model_path") or main["calculator_label"],
        "model_path_sha256": run.get("model_path_sha256") or main.get("model_path_sha256"),
        "seed": main["seed"],
        "input_images_sha256": main["input_images_sha256"],
        "optimizer_steps_completed": main.get("optimizer_steps_completed", steps),
        "optimizer_converged": bool(main.get("optimizer_converged", False)),
        "initial_neb_residual_eV_A": first["max_neb_residual_eV_A"],
        "final_neb_residual_eV_A": last["max_neb_residual_eV_A"],
        "residual_final_over_initial": residual_ratio,
        "log_residual_slope_per_step": log_slope,
        "residual_auc_eV_A_step": trapezoid_auc(iterations, "max_neb_residual_eV_A"),
        "steps_to_1_eV_A": first_step_below(iterations, "max_neb_residual_eV_A", 1.0),
        "steps_to_0.5_eV_A": first_step_below(iterations, "max_neb_residual_eV_A", 0.5),
        "steps_to_0.1_eV_A": first_step_below(iterations, "max_neb_residual_eV_A", 0.1),
        "initial_barrier_proxy_eV": first["barrier_proxy_eV"],
        "final_barrier_proxy_eV": last["barrier_proxy_eV"],
        "barrier_proxy_change_eV": last["barrier_proxy_eV"] - first["barrier_proxy_eV"],
        "final_max_internal_true_force_eV_A": last["max_internal_true_force_eV_A"],
        "final_max_internal_neb_force_eV_A": last["max_internal_neb_force_eV_A"],
        "manuscript_barrier_eligible": False,
        "main_manifest": str(main_manifest_path),
        "main_manifest_sha256": sha256(main_manifest_path),
        "history_manifest": str(history_path),
        "history_manifest_sha256": sha256(history_path),
        "iterations": iterations,
    }


def write_svg(path, records):
    width, height = 700, 360
    left, top, plot_width, plot_height = 70, 45, 560, 240
    values = [point["max_neb_residual_eV_A"] for record in records for point in record["iterations"]]
    y_max = max(values) * 1.08
    x_max = max(point["optimizer_iteration"] for record in records for point in record["iterations"])
    colors = ["#1f77b4", "#d1495b", "#2a9d8f", "#6a4c93"]
    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="70" y="25" font-family="Arial, sans-serif" font-size="18" font-weight="700">MLFF pre-NEB residual reduction</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#fafafa" stroke="#c9cdd2"/>',
    ]
    for index, record in enumerate(records):
        color = colors[index % len(colors)]
        points = []
        for point in record["iterations"]:
            x = left + point["optimizer_iteration"] * plot_width / max(x_max, 1)
            y = top + plot_height - point["max_neb_residual_eV_A"] * plot_height / y_max
            points.append(f"{x:.2f},{y:.2f}")
        chunks.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        label = f"{record['label']} s{record['seed']}"
        legend_y = 310 + 18 * index
        chunks.append(f'<line x1="80" y1="{legend_y}" x2="108" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        chunks.append(f'<text x="116" y="{legend_y + 4}" font-family="Arial, sans-serif" font-size="12">{html.escape(label)}</text>')
    chunks.extend([
        f'<text x="{left + plot_width / 2}" y="{top + plot_height + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">Optimizer iteration</text>',
        f'<text x="20" y="{top + plot_height / 2}" transform="rotate(-90 20 {top + plot_height / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="12">Max NEB residual (eV/A)</text>',
        '</svg>',
    ])
    path.write_text("\n".join(chunks) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-config", type=Path, required=True,
        help="JSON containing explicit main/history manifest pairs and frozen input hashes",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.run_config.resolve()
    config = json.loads(config_path.read_text())
    records = [metrics(run) for run in config["runs"]]
    if {record["iterations"][0]["optimizer_iteration"] for record in records} != {0}:
        raise ValueError("Every comparison history must include optimizer iteration 0")
    input_hashes = {record["input_images_sha256"] for record in records}
    if input_hashes != {config["shared_input_images_sha256"]}:
        raise ValueError("All comparison runs must use the frozen shared input images")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    table_rows = [{key: value for key, value in row.items() if key != "iterations"} for row in records]
    with (out_dir / "mlff_preconditioner_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_config": str(config_path),
        "run_config_sha256": sha256(config_path),
        "records": records,
        "interpretation_gate": "Diagnostic only unless endpoints are accepted, each run reaches its NEB force target, and downstream DFT handoff quality is measured.",
        "task_relevant_metrics": [
            "NEB residual trajectory", "steps to force threshold", "path deformation",
            "barrier-proxy drift", "DFT initial residual", "DFT steps to convergence",
            "final DFT mechanism and barrier",
        ],
    }
    (out_dir / "mlff_preconditioner_comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_svg(out_dir / "mlff_preconditioner_residual.svg", records)
    print(json.dumps({"runs": len(records), "output": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
