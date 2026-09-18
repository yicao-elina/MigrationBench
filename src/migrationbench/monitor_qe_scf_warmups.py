#!/usr/bin/env python3
"""Sync and classify standalone QE image-SCF warmup jobs on Rockfish."""

import argparse
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ENERGY_RE = re.compile(r"!\s+total energy\s+=\s+([-+0-9.Ee]+)\s+Ry")
ACCURACY_RE = re.compile(r"estimated scf accuracy\s+<\s+([-+0-9.Ee]+)\s+Ry")
ITERATION_RE = re.compile(r"iteration #\s+(\d+)")
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
TOTAL_MAG_RE = re.compile(r"total magnetization\s*=\s*([^\n]+?)\s+Bohr mag/cell", re.I)
ABS_MAG_RE = re.compile(rf"absolute magnetization\s*=\s*({FLOAT})\s+Bohr mag/cell", re.I)
SITE_MAG_RE = re.compile(
    rf"atom:\s*(\d+)\s+charge:\s*({FLOAT})\s+magn:\s*({FLOAT})"
    rf"(?:\s+({FLOAT})\s+({FLOAT}))?",
    re.I,
)


def run(command, check=True):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and result.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{result.stderr}")
    return result


def queue_state(job_id, host):
    result = run(["ssh", host, f"squeue -j {job_id} -h -o '%T|%M|%R'"], check=False)
    if not result.stdout.strip():
        return {"queue_state": "not_in_squeue", "elapsed": None, "reason": None}
    state, elapsed, *reason = result.stdout.strip().splitlines()[0].split("|", 2)
    return {"queue_state": state, "elapsed": elapsed, "reason": reason[0] if reason else None}


def accounting_state(job_id, host):
    result = run(
        ["ssh", host, f"sacct -X -S now-7days -j {job_id} --format=JobID,JobName%40,State,Elapsed,ExitCode,MaxRSS,ReqMem -P"],
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return {}
    header = lines[0].split("|")
    rows = [dict(zip(header, line.split("|"))) for line in lines[1:]]
    return next((row for row in rows if row.get("JobID") == str(job_id)), rows[0])


def input_symbols(path):
    if path is None or not path.exists():
        return []
    lines = path.read_text(errors="replace").splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.strip().upper().startswith("ATOMIC_POSITIONS")),
        None,
    )
    if start is None:
        return []
    symbols = []
    for line in lines[start + 1:]:
        fields = line.split()
        if len(fields) < 4 or not re.fullmatch(FLOAT, fields[1]):
            break
        symbols.append(fields[0])
    return symbols


def magnetic_observables(text, symbols=None):
    symbols = symbols or []
    total = [
        [float(value.replace("d", "e").replace("D", "E")) for value in re.findall(FLOAT, raw)]
        for raw in TOTAL_MAG_RE.findall(text)
    ]
    absolute = [float(value.replace("d", "e").replace("D", "E")) for value in ABS_MAG_RE.findall(text)]
    latest_sites = {}
    for match in SITE_MAG_RE.finditer(text):
        atom_index = int(match.group(1))
        components = [
            float(value.replace("d", "e").replace("D", "E"))
            for value in match.groups()[2:] if value is not None
        ]
        latest_sites[atom_index] = {
            "atom_index_qe": atom_index,
            "species": symbols[atom_index - 1] if atom_index <= len(symbols) else None,
            "charge": float(match.group(2).replace("d", "e").replace("D", "E")),
            "magnetization_components_Bohr_magneton": components,
        }
    return {
        "total_magnetization_Bohr_magneton_cell": total[-1][0] if total and len(total[-1]) == 1 else None,
        "total_magnetization_components_Bohr_magneton_cell": total[-1] if total else [],
        "absolute_magnetization_Bohr_magneton_cell": absolute[-1] if absolute else None,
        "site_magnetic_moments_last": [latest_sites[index] for index in sorted(latest_sites)],
    }


def parse_output(path, input_path=None):
    if not path.exists():
        return {"output_present": False}
    text = path.read_text(errors="replace")
    energies = [float(value) for value in ENERGY_RE.findall(text)]
    accuracies = [float(value) for value in ACCURACY_RE.findall(text)]
    iterations = [int(value) for value in ITERATION_RE.findall(text)]
    parsed = {
        "output_present": True,
        "job_done": "JOB DONE" in text,
        "scf_converged": "convergence has been achieved" in text,
        "scf_not_converged": "convergence NOT achieved" in text,
        "maximum_cpu_time_exceeded": "Maximum CPU time exceeded" in text,
        "scf_iteration_records": len(iterations),
        "last_scf_iteration": iterations[-1] if iterations else None,
        "last_estimated_accuracy_Ry": accuracies[-1] if accuracies else None,
        "final_energy_Ry": energies[-1] if energies else None,
    }
    parsed.update(magnetic_observables(text, input_symbols(input_path)))
    return parsed


def restart_inventory(host, remote_dir):
    command = (
        f"d='{remote_dir}/out/sb2te3.save'; "
        "test -s \"$d/charge-density.dat\" && charge=1 || charge=0; "
        "test -s \"$d/data-file-schema.xml\" && xml=1 || xml=0; "
        "wfc=$(find \"$d\" -maxdepth 1 -type f -name 'wfc*.dat' -size +0c 2>/dev/null | wc -l); "
        "printf '%s|%s|%s' \"$charge\" \"$xml\" \"$wfc\""
    )
    result = run(["ssh", host, command], check=False)
    fields = result.stdout.strip().split("|")
    if len(fields) != 3:
        return {"charge_density": False, "xml": False, "wavefunction_files": 0}
    return {
        "charge_density": fields[0] == "1",
        "xml": fields[1] == "1",
        "wavefunction_files": int(fields[2]),
    }


def classify(queue, parsed, restart):
    complete_restart = restart["charge_density"] and restart["xml"]
    energy = parsed.get("final_energy_Ry")
    finite_energy = isinstance(energy, (int, float)) and math.isfinite(energy)
    if parsed.get("job_done") and parsed.get("scf_converged") and finite_energy and complete_restart:
        return "accepted_scf_warmstart"
    if queue == "RUNNING":
        return "scf_running"
    if queue == "PENDING":
        return "pending"
    if parsed.get("maximum_cpu_time_exceeded"):
        return "clean_timeout_needs_restart"
    return "terminal_needs_review"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-file", type=Path, required=True)
    parser.add_argument("--ssh-host", default="rockfish")
    parser.add_argument("--local-cluster-root", type=Path, default=Path("./cluster"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    jobs = json.loads(args.jobs_file.read_text())["jobs"]
    rows = []
    for job in jobs:
        queue = queue_state(job["job_id"], args.ssh_host)
        accounting = accounting_state(job["job_id"], args.ssh_host)
        accounting_state_name = accounting.get("State", "").split("+", 1)[0]
        if queue["queue_state"] == "not_in_squeue" and accounting_state_name:
            queue["queue_state"] = accounting_state_name
            queue["elapsed"] = accounting.get("Elapsed")
        local_dir = args.local_cluster_root / Path(job["remote_run_dir"]).name
        local_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                "rsync", "-a", "--include", "scf.out", "--include", "scf.in",
                "--include", "*.json", "--include", "slurm-*.out", "--include", "slurm-*.err",
                "--exclude", "*", f"{args.ssh_host}:{job['remote_run_dir'].rstrip('/')}/", str(local_dir) + "/",
            ],
            check=False,
        )
        parsed = parse_output(local_dir / "scf.out", local_dir / "scf.in")
        restart = restart_inventory(args.ssh_host, job["remote_run_dir"])
        row = dict(job)
        row.update(queue)
        row["accounting"] = accounting
        row["parsed"] = parsed
        row["restart_artifacts"] = restart
        row["classification"] = classify(queue["queue_state"], parsed, restart)
        row["local_dir"] = str(local_dir)
        rows.append(row)
    payload = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "jobs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
