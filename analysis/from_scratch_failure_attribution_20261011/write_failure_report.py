#!/usr/bin/env python3
"""Write the evidence-bounded Results/Discussion for the failure audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def fmt(value: float, digits: int = 1) -> str:
    return "NA" if pd.isna(value) else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = pd.read_csv(args.analysis / "run_summary.csv")
    lines = [
        "# From-scratch failure attribution: audited Results and Discussion",
        "",
        "## Results",
        "",
        "The exact-protocol reproduction used the original 2050-atom periodic input, the original model files, the original training-seed/model pairing, `md_seed=20261005`, 600 K Langevin NVT, 1 fs time step, friction 0.01 fs^-1, and the original threshold gate. The original campaign was not modified; reproduction outputs were written to a separate root.",
        "",
        "The failure times below are threshold-onset times, not inferred physical transition times. They are the first saved frames at which the recorded temperature, maximum force, or minimum pair-distance gate failed.",
        "",
        "| source | family | training seed | onset (ps) | trigger | maximum temperature (K) | maximum force (eV/Å) | minimum distance (Å) |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in summary.sort_values(["source", "training_seed"]).itertuples(index=False):
        lines.append(f"| {row.source} | {row.family} | {int(row.training_seed)} | {fmt(row.onset_ps)} | {row.trigger} | {fmt(row.max_temperature_K)} | {fmt(row.max_force_eV_A)} | {fmt(row.min_distance_A, 3)} |")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "A reproduced onset at the same step within the saved-frame resolution would support a deterministic model/input/protocol failure under the fixed MD seed. It would not, by itself, prove that the threshold frame is a physical phase transition. A shifted onset with the same model and seed would instead identify sensitivity to hardware, software, floating-point ordering, or an incompletely locked initial state.",
        "",
        "The attribution analysis therefore compares stable pre-onset and precursor/threshold windows using global SOAP descriptors, Cr-centered coordination and bond-length statistics, local angle distributions, force/temperature/minimum-distance traces, and nearest-neighbour distances to the training, validation, test, and NEB descriptor clouds. Descriptor-space distance is a novelty diagnostic, not calibrated epistemic uncertainty and not a causal proof.",
        "",
        "## Planned causal test",
        "",
        "The strongest evidence for a data-distribution explanation will be a concordant pattern in which (i) all three reproductions recur at the same onset, (ii) a precursor window moves outside the training descriptor support before the force runaway, (iii) the nearest test/NEB reference is materially closer than the nearest training configuration, and (iv) force disagreement among the three trained models rises before the threshold. If only the final frame is novel while the force disagreement remains low, the evidence instead favors a local dynamical runaway rather than a clean distribution boundary.",
        "",
        "HFACF and thermal conductivity are not used for this failure diagnosis unless a heat-flux time series is explicitly present. Failed trajectories are never assigned a formal diffusion coefficient.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
