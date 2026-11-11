#!/usr/bin/env python3
"""Promote only models with an explicit compiled-model gate result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--acceptance", required=True)
    parser.add_argument("--gate-audit", required=True)
    parser.add_argument("--output-registry", required=True)
    parser.add_argument("--output-acceptance", required=True)
    args = parser.parse_args()

    registry_path = Path(args.registry)
    acceptance_path = Path(args.acceptance)
    gate_path = Path(args.gate_audit)
    registry = pd.read_csv(registry_path, sep="\t", dtype=str).fillna("")
    acceptance = pd.read_csv(acceptance_path, sep="\t", dtype=str).fillna("")
    gate = json.loads(gate_path.read_text())
    gate_rows = {str(row["canonical_model_id"]): row for row in gate.get("results", [])}

    registry["model_gate_status"] = registry["canonical_model_id"].map(
        lambda model_id: str(gate_rows.get(str(model_id), {}).get("status", "NOT_AUDITED"))
    )
    registry["model_gate_failure_reasons"] = registry["canonical_model_id"].map(
        lambda model_id: ";".join(gate_rows.get(str(model_id), {}).get("failure_reasons", []))
    )
    registry["acceptance_decision"] = "HELD"
    for index, row in registry.iterrows():
        gate_status = row["model_gate_status"]
        if gate_status == "PASS" and row["computational_status"] == "COMPLETED_UNACCEPTED":
            registry.at[index, "computational_status"] = "COMPLETED_ACCEPTED"
            registry.at[index, "acceptance_decision"] = "ACCEPTED_FOR_STATIC_SCREEN"
        elif gate_status == "PASS" and row["computational_status"] == "REGISTERED":
            registry.at[index, "acceptance_decision"] = "REGISTERED_COMPARATOR"
        elif gate_status == "FAIL":
            registry.at[index, "acceptance_decision"] = "REJECTED_MODEL_GATE"

    for column in ("model_gate_status", "model_gate_failure_reasons"):
        if column not in acceptance:
            acceptance[column] = acceptance["canonical_model_id"].map(registry.set_index("canonical_model_id")[column])
    acceptance["computational_status"] = acceptance["canonical_model_id"].map(registry.set_index("canonical_model_id")["computational_status"])
    acceptance["provenance_status"] = acceptance["canonical_model_id"].map(registry.set_index("canonical_model_id")["provenance_status"])
    acceptance["static_639_status"] = acceptance.apply(
        lambda row: "READY_FOR_SUBMISSION"
        if row["model_gate_status"] == "PASS" and row["publication_status"] != "EXCLUDE_FROM_PUBLICATION"
        else row["static_639_status"],
        axis=1,
    )
    acceptance["publication_status"] = acceptance.apply(
        lambda row: "HELD_PENDING_STATIC_SCREEN"
        if row["model_gate_status"] == "PASS" and row["publication_status"] == "EXCLUDE_FROM_PUBLICATION"
        else row["publication_status"],
        axis=1,
    )
    acceptance["acceptance_decision"] = acceptance["canonical_model_id"].map(registry.set_index("canonical_model_id")["acceptance_decision"])

    output_registry = Path(args.output_registry)
    output_acceptance = Path(args.output_acceptance)
    output_registry.parent.mkdir(parents=True, exist_ok=True)
    output_acceptance.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(output_registry, sep="\t", index=False)
    acceptance.to_csv(output_acceptance, sep="\t", index=False)
    summary = {
        "registry": str(output_registry),
        "acceptance": str(output_acceptance),
        "gate_audit": str(gate_path),
        "gate_status": gate.get("status", "UNKNOWN"),
        "models_in_registry": len(registry),
        "models_gate_pass": int((registry["model_gate_status"] == "PASS").sum()),
        "models_gate_fail": int((registry["model_gate_status"] == "FAIL").sum()),
        "models_held": int((registry["acceptance_decision"] == "HELD").sum()),
    }
    Path(output_registry.parent / "ACCEPTANCE_FINALIZATION_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if gate.get("status") != "PASS":
        raise SystemExit("gate audit is not globally PASS; no final acceptance should be used")


if __name__ == "__main__":
    main()
