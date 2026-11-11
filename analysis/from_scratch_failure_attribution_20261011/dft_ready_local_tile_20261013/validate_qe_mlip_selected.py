from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


ENERGY_RE = re.compile(r"!\s+total energy\s+=\s+([-+0-9.eEdD]+)\s+Ry")
FORCE_RE = re.compile(r"Total force\s*=\s*([-+0-9.eEdD]+)\s+Total SCF correction force")


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value.replace("D", "E").replace("d", "e")))
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--selected-index", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    selected = json.loads(Path(args.selected_index).read_text())
    rows = []
    for record in selected:
        candidate_id = record["candidate_id"]
        run_dir = root / "results" / candidate_id
        out = run_dir / "scf.out"
        err = run_dir / "scf.err"
        text = out.read_text(errors="replace") if out.is_file() else ""
        energies = [x.group(1) for x in ENERGY_RE.finditer(text)]
        forces = [x.group(1) for x in FORCE_RE.finditer(text)]
        exit_code = (run_dir / "exit_code.txt").read_text().strip() if (run_dir / "exit_code.txt").is_file() else "missing"
        row = {**record, "run_dir": str(run_dir), "exit_code": exit_code,
               "job_done": "JOB DONE" in text, "n_energy_lines": len(energies),
               "total_energy_Ry": energies[-1] if energies else "",
               "total_energy_finite": finite(energies[-1]) if energies else False,
               "n_total_force_lines": len(forces),
               "total_force_finite": finite(forces[-1]) if forces else False,
               "scf_error_tail": err.read_text(errors="replace")[-500:] if err.is_file() else ""}
        row["scientific_status"] = "VERIFIED" if row["job_done"] and exit_code == "0" and row["total_energy_finite"] and row["total_force_finite"] else "PENDING_OR_FAILED"
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows)[["candidate_id", "scientific_status", "job_done", "total_energy_finite", "total_force_finite"]].to_string(index=False))


if __name__ == "__main__":
    main()
