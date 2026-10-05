#!/usr/bin/env python3
"""Audit the additive QE reconstruction array without promoting it to formal DFT."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/scratch16/pclancy3/yi/phase3_dft_recovery_20261004")
# Force-enabled derived inputs preserve the canonical geometry/cell and all
# DFT settings, adding only CONTROL tprnfor=.true. so QE emits 81 forces.
INPUT_ROOT = ROOT / "qe_inputs_force"
RESULT_ROOT = ROOT / "results_reconstructed"
RY_BOHR_TO_EV_A = 13.605693009 / 0.529177210903


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(value: float) -> bool:
    return math.isfinite(value)


def parse_output(path: Path) -> dict:
    text = path.read_text(errors="replace")
    energy_matches = re.findall(r"!\s+total energy\s*=\s*([-+0-9.EeDd]+)\s+Ry", text)
    energy_ry = float(energy_matches[-1].replace("D", "E").replace("d", "e")) if energy_matches else None
    force_vectors = []
    for match in re.finditer(
        r"force\s*=\s*([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)",
        text,
        flags=re.IGNORECASE,
    ):
        force_vectors.append([float(x.replace("D", "E").replace("d", "e")) for x in match.groups()])
    force_norms = [RY_BOHR_TO_EV_A * sum(x * x for x in vector) ** 0.5 for vector in force_vectors]
    force_vectors_eV_A = [[RY_BOHR_TO_EV_A * component for component in vector]
                          for vector in force_vectors]
    convergence = bool(re.search(r"convergence has been achieved", text, flags=re.IGNORECASE))
    job_done = "JOB DONE" in text
    errors = re.findall(r"(?:Error in routine|convergence NOT achieved|stopping\s+due|%%%%%%%%%%%%)", text, flags=re.IGNORECASE)
    return {
        "output": str(path),
        "total_energy_ry": energy_ry,
        "total_energy_finite": energy_ry is not None and finite(energy_ry),
        "force_count": len(force_vectors),
        "force_finite": bool(force_norms) and all(finite(x) for x in force_norms),
        "force_vectors_eV_A": force_vectors_eV_A,
        "max_force_eV_A": max(force_norms) if force_norms else None,
        "scf_converged": convergence,
        "job_done": job_done,
        "error_markers": errors,
    }


def main() -> None:
    RESULT_ROOT.mkdir(exist_ok=True)
    manifest = json.loads((INPUT_ROOT / "reconstruction_manifest.json").read_text())
    rows = []
    for case, metadata in manifest["cases"].items():
        input_path = INPUT_ROOT / case / "scf.in"
        outputs = sorted((RESULT_ROOT / case).glob("**/scf.out")) if (RESULT_ROOT / case).exists() else []
        if not outputs:
            rows.append({"case": case, "status": "MISSING_OUTPUT", "input_sha256": sha256(input_path)})
            continue
        for output in outputs:
            parsed = parse_output(output)
            parsed.update(
                {
                    "case": case,
                    "input_sha256": sha256(input_path),
                    "input_sha256_recorded": metadata["input_sha256"],
                    "scope": metadata["scope"],
                    "formal_dft_gate": False,
                }
            )
            parsed["status"] = (
                "DIAGNOSTIC_ACCEPTED"
                if parsed["total_energy_finite"]
                and parsed["force_finite"]
                and parsed["force_count"] >= metadata["nat"]
                and parsed["scf_converged"]
                and parsed["job_done"]
                and not parsed["error_markers"]
                and parsed["input_sha256"] == parsed["input_sha256_recorded"]
                else "REJECTED"
            )
            rows.append(parsed)
    (RESULT_ROOT / "qe_gate_audit.json").write_text(
        json.dumps({"job": "force-enabled-reconstruction", "rows": rows}, indent=2) + "\n"
    )
    for row in rows:
        print(row["case"], row["status"], row.get("output", ""))


if __name__ == "__main__":
    main()
