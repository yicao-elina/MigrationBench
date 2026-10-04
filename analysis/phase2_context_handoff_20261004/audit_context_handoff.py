#!/usr/bin/env python3
"""Audit the additive 460000/462200 context handoff outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODELS = ["foundation-omat", "from-scratch", "naive-fine-tuning", "multi_T-fine-tuning"]
SNAPS = [460000, 462200]
GATES = {
    "max_force_eV_A": 100.0,
    "max_velocity_A_ps": 1000.0,
    "min_distance_A": 1.5,
    "temperature_K": 2000.0,
    "max_unwrapped_displacement_A": 50.0,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_run(path: Path, snap: int) -> dict:
    metrics = path / "metrics.csv"
    completion = path / "completion.json"
    handoff = path / "handoff_input.json"
    if not (metrics.exists() and completion.exists() and handoff.exists()):
        return {"status": "MISSING_ARTIFACT", "path": str(path)}
    rows = list(csv.DictReader(metrics.open()))
    finite_rows = []
    first_gate = None
    for row in rows:
        vals = {key: float(row[key]) for key in GATES}
        finite = all(math.isfinite(v) for v in vals.values())
        finite_rows.append(finite)
        if first_gate is None and (not finite or any(vals[key] >= threshold if key != "min_distance_A" else vals[key] < threshold for key, threshold in GATES.items())):
            first_gate = int(row["absolute_step"])
    classification = "runaway_or_nonfinite" if first_gate is not None else "stable_short_handoff"
    return {
        "status": "COMPLETE",
        "snapshot": snap,
        "model": path.name.removeprefix("model_"),
        "frames": len(rows),
        "first_gate_step": first_gate,
        "classification": classification,
        "input_sha256": json.loads(handoff.read_text())["input_sha256"],
        "model_sha256": json.loads(handoff.read_text())["model_sha256"],
        "metrics_sha256": sha256(metrics),
        "completion": json.loads(completion.read_text()),
    }


def main() -> None:
    rows = []
    for snap in SNAPS:
        for model in MODELS:
            rows.append(audit_run(ROOT / "runs" / f"snapshot_{snap}" / f"model_{model}", snap))
    out = ROOT / "audit_results.json"
    out.write_text(json.dumps({"runs": rows}, indent=2, sort_keys=True) + "\n")
    for row in rows:
        print(row.get("snapshot"), row.get("model"), row["status"], row.get("classification", ""), row.get("first_gate_step", ""))


if __name__ == "__main__":
    main()
