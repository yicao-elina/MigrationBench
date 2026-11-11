#!/usr/bin/env python3
"""Load each Foundation checkpoint and check finite energy/forces on four endpoints."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--element", default="Cr", choices=["Cr", "Cu", "Li", "Na"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    models = list(csv.DictReader((root / "MODEL_REGISTRY.tsv").open(), delimiter="\t"))
    paths = list(csv.DictReader((root / "PATH_REGISTRY.tsv").open(), delimiter="\t"))
    output = root / "data/MODEL_SMOKE.tsv"
    output.parent.mkdir(exist_ok=True)
    rows: list[dict[str, str]] = []
    for model in models:
        try:
            kwargs = {
                "model_paths": model["model_path"],
                "device": "cpu",
                "default_dtype": "float64",
            }
            if model["model_head"] != "default":
                kwargs["head"] = model["model_head"]
            calculator = MACECalculator(**kwargs)
        except Exception as exc:  # preserve load failures as data
            rows.append(
                {
                    "model_id": model["model_id"],
                    "element": "*",
                    "load_status": "LOAD_FAIL",
                    "finite_force_status": "NOT_RUN",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        for element in (args.element,):
            path = next(
                item
                for item in paths
                if item["element"] == element and item["path_class"] == "in_gap"
            )
            atoms = read(root / path["initial_endpoint_file"])
            atoms.calc = calculator
            try:
                energy = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces(), dtype=float)
                finite = bool(np.isfinite(energy) and np.isfinite(forces).all())
                rows.append(
                    {
                        "model_id": model["model_id"],
                        "element": element,
                        "load_status": "LOAD_OK",
                        "finite_force_status": "PASS" if finite else "NONFINITE",
                        "error": "" if finite else "non-finite energy/forces",
                    }
                )
            except Exception as exc:  # preserve force failures as data
                rows.append(
                    {
                        "model_id": model["model_id"],
                        "element": element,
                        "load_status": "LOAD_OK",
                        "finite_force_status": "FORCE_FAIL",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_id", "element", "load_status", "finite_force_status", "error"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} ({len(rows)} rows)")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

