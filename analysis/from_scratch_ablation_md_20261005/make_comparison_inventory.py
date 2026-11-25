#!/usr/bin/env python3
"""Build a merge-ready inventory without claiming protocol equivalence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def row(**kwargs):
    defaults = {
        "model": "",
        "family": "",
        "training_seed": "",
        "source": "",
        "protocol_class": "",
        "status": "",
        "duration_ps": "",
        "T_mean_K": "",
        "T_min_K": "",
        "T_max_K": "",
        "max_force_eV_A": "",
        "min_distance_A": "",
        "Dxy_200_500_cm2_s": "",
        "Dxy_500_900_cm2_s": "",
        "HFACF_status": "",
        "notes": "",
    }
    defaults.update(kwargs)
    return defaults


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screening", type=Path, required=True)
    ap.add_argument("--unified-screening", type=Path)
    ap.add_argument("--old-table", type=Path, required=True)
    ap.add_argument("--new-metrics-root", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    if args.unified_screening and args.unified_screening.exists():
        with args.unified_screening.open(newline="") as handle:
            for item in csv.DictReader(handle):
                stable = item.get("scientific_gate") == "screening_or_production_complete"
                rows.append(row(
                    model=item.get("run", ""),
                    family=item.get("family", ""),
                    training_seed=item.get("training_seed", ""),
                    source="from_scratch_ablation_20ps_unified",
                    protocol_class="SCREENING_ONLY" if stable else "UNSTABLE_NO_DXY",
                    status=item.get("status", ""),
                    duration_ps=item.get("duration_ps", ""),
                    T_mean_K=item.get("temperature_mean_K", ""),
                    T_min_K=item.get("temperature_min_K", ""),
                    T_max_K=item.get("temperature_max_K", ""),
                    max_force_eV_A=item.get("max_force_eV_A", ""),
                    min_distance_A=item.get("min_distance_A", ""),
                    Dxy_200_500_cm2_s=item.get("Dxy_screening_cm2_s", "") if stable else "",
                    HFACF_status=item.get("HFACF_status", ""),
                    notes="unified pressure/stress/HFACF diagnostics; candidate hops only; unstable runs intentionally have no Dxy",
                ))
    else:
        screening = json.loads(args.screening.read_text())
        for item in screening:
            temp = item.get("temperature_K", {})
            force = item.get("max_force_eV_A", {})
            dxy = item.get("dxy_screening") or {}
            status = item.get("status", {}).get("status", item.get("scientific_gate", "unknown"))
            rows.append(row(
                model=f"{item.get('family')}__seed{item.get('training_seed')}",
                family=item.get("family", ""),
                training_seed=item.get("training_seed", ""),
                source="from_scratch_ablation_20ps",
                protocol_class="SCREENING_ONLY" if item.get("scientific_gate") != "blocked" else "UNSTABLE_NO_DXY",
                status=status,
                duration_ps=item.get("duration_ps", ""),
                T_mean_K=temp.get("mean", ""),
                T_min_K=temp.get("min", ""),
                T_max_K=temp.get("max", ""),
                max_force_eV_A=force.get("max", ""),
                Dxy_200_500_cm2_s=dxy.get("Dxy_cm2_s", "") if item.get("scientific_gate") != "blocked" else "",
                HFACF_status="not_recorded_in_20ps_summary",
                notes="candidate hop count is diagnostic only; unstable runs intentionally have no Dxy",
            ))

    if args.new_metrics_root and args.new_metrics_root.exists():
        for metrics_path in sorted(args.new_metrics_root.glob("*/metrics.json")):
            item = json.loads(metrics_path.read_text())
            msd = item.get("msd", {})
            temp = item.get("temperature_K", {})
            force = item.get("max_force_eV_A", {})
            pressure = item.get("pressure_GPa", {})
            hf = item.get("heat_flux", {})
            rows.append(row(
                model=metrics_path.parent.name,
                source="from_scratch_ablation_1ns",
                protocol_class="FORMAL_SAME_PROTOCOL" if item.get("complete_step_gate") and item.get("finite_gate") else "INCOMPLETE_OR_BLOCKED",
                status=item.get("scientific_gate", ""),
                duration_ps=item.get("duration_ps", ""),
                T_mean_K=temp.get("mean", ""),
                T_min_K=temp.get("min", ""),
                T_max_K=temp.get("max", ""),
                max_force_eV_A=force.get("max", ""),
                min_distance_A=item.get("min_distance_A", ""),
                Dxy_200_500_cm2_s=msd.get("Dxy_200_500_ps", {}).get("D_cm2_s", ""),
                Dxy_500_900_cm2_s=msd.get("Dxy_500_900_ps", {}).get("D_cm2_s", ""),
                HFACF_status=hf.get("status", ""),
                notes=f"pressure finite frames={pressure.get('finite_frames', '')}; transport convergence remains a separate gate",
            ))

    with args.old_table.open(newline="") as handle:
        for item in csv.DictReader(handle):
            rows.append(row(
                model=item.get("model", ""),
                source="old_continuation_processed",
                protocol_class="REFERENCE_PROTOCOL_DIFFERENT",
                status="complete_in_old_campaign",
                T_mean_K=item.get("T_mean_K", ""),
                T_min_K=item.get("T_min_K", ""),
                T_max_K=item.get("T_max_K", ""),
                max_force_eV_A=item.get("max_force_eV_A", ""),
                min_distance_A=item.get("min_distance_A", ""),
                Dxy_200_500_cm2_s=item.get("Dxy_200_500", ""),
                Dxy_500_900_cm2_s=item.get("Dxy_500_900", ""),
                HFACF_status="available_in_old_processed_release",
                notes="different input/start-step/model provenance; not a same-protocol replica",
            ))

    fields = list(row().keys())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
