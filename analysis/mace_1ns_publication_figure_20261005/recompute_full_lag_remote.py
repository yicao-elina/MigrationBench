#!/usr/bin/env python3
"""Emit a full-lag MSD CSV from one remote four-model trajectory."""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    root, model, module_path = sys.argv[1:]
    spec = importlib.util.spec_from_file_location("publication_metrics", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    run_dir = Path(root) / model
    manifest, _ = module.manifest_for(run_dir)
    data = module.load_trajectory(manifest["input"], manifest["segment"])
    pos = module.unwrap(
        np.einsum("nij,jk->nik", data["xyz"], np.linalg.inv(data["cell"])),
        data["cell"],
    )
    lag, msd = module.msd_series(pos, max_lag=5000, origin_stride=10)
    writer = csv.writer(sys.stdout)
    writer.writerow(["lag_ps", "msd_A2"])
    writer.writerows((float(k) * 0.2, float(value)) for k, value in zip(lag, msd))


if __name__ == "__main__":
    main()
