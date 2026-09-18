#!/usr/bin/env python3
"""Parse a Quantum ESPRESSO neb.x output into an auditable JSON summary."""

import argparse
import json
import re
from pathlib import Path


ITER_RE = re.compile(r"-+ iteration\s+(\d+)\s+-+")
ACT_FWD_RE = re.compile(r"activation energy \(->\)\s*=\s*([-+0-9.Ee*]+)\s*eV")
ACT_REV_RE = re.compile(r"activation energy \(<-\)\s*=\s*([-+0-9.Ee*]+)\s*eV")
IMAGE_ROW_RE = re.compile(r"^\s*(\d+)\s+([-+0-9.Ee*]+)\s+([-+0-9.Ee*]+)\s+([FT])\s*$")
PATH_LENGTH_RE = re.compile(r"initial path length\s*=\s*([-+0-9.Ee*]+)\s*bohr")
INTER_IMAGE_RE = re.compile(r"initial inter-image distance\s*=\s*([-+0-9.Ee*]+)\s*bohr")
TCPU_RE = re.compile(r"\btcpu\s*=\s*([-+0-9.Ee]+)")


def parse_qe_float(token):
    return None if "*" in token else float(token)


def has_overflow_token(token):
    return "*" in token


def parse_neb_out(path):
    lines = path.read_text(errors="replace").splitlines()
    iterations = []
    current = None
    in_image_table = False
    overflow_flags = []
    initial_path_length_bohr = None
    initial_inter_image_distance_bohr = None
    tcpu_values = []
    scf_not_converged_count = 0

    for line in lines:
        tcpu = TCPU_RE.search(line)
        if tcpu:
            tcpu_values.append(float(tcpu.group(1)))
        if "scf convergence NOT achieved" in line:
            scf_not_converged_count += 1
        path_len = PATH_LENGTH_RE.search(line)
        if path_len:
            token = path_len.group(1)
            if has_overflow_token(token):
                overflow_flags.append("initial_path_length_overflow")
            else:
                initial_path_length_bohr = float(token)
            continue

        inter_img = INTER_IMAGE_RE.search(line)
        if inter_img:
            token = inter_img.group(1)
            if has_overflow_token(token):
                overflow_flags.append("initial_inter_image_distance_overflow")
            else:
                initial_inter_image_distance_bohr = float(token)
            continue

        iter_match = ITER_RE.search(line)
        if iter_match:
            current = {"iteration": int(iter_match.group(1)), "images": []}
            iterations.append(current)
            in_image_table = False
            continue

        fwd = ACT_FWD_RE.search(line)
        if fwd and current is not None:
            token = fwd.group(1)
            if has_overflow_token(token):
                current["activation_forward_overflow"] = True
                overflow_flags.append("activation_forward_overflow")
            else:
                current["activation_forward_eV"] = float(token)
            continue

        rev = ACT_REV_RE.search(line)
        if rev and current is not None:
            token = rev.group(1)
            if has_overflow_token(token):
                current["activation_reverse_overflow"] = True
                overflow_flags.append("activation_reverse_overflow")
            else:
                current["activation_reverse_eV"] = float(token)
            continue

        if line.strip().startswith("image") and "energy" in line and "error" in line:
            in_image_table = True
            continue

        if in_image_table and current is not None:
            row = IMAGE_ROW_RE.match(line)
            if row:
                current["images"].append(
                    {
                        "image_index": int(row.group(1)),
                        "energy_eV": parse_qe_float(row.group(2)),
                        "error_eV_A": parse_qe_float(row.group(3)),
                        "energy_overflow": has_overflow_token(row.group(2)),
                        "error_overflow": has_overflow_token(row.group(3)),
                        "frozen": row.group(4) == "T",
                    }
                )
            elif current.get("images") and not line.strip():
                in_image_table = False

    complete_iterations = [
        row
        for row in iterations
        if row.get("images") and ("activation_forward_eV" in row or row.get("activation_forward_overflow"))
    ]
    last = complete_iterations[-1] if complete_iterations else {}
    errors_all = [float(row["error_eV_A"]) for row in last.get("images", []) if row.get("error_eV_A") is not None]
    errors_movable = [
        float(row["error_eV_A"])
        for row in last.get("images", [])
        if row.get("error_eV_A") is not None and not row.get("frozen")
    ]
    if any(row.get("error_overflow") for row in last.get("images", [])):
        overflow_flags.append("image_error_overflow")
    fwd_values = [float(row["activation_forward_eV"]) for row in complete_iterations if "activation_forward_eV" in row]
    recent = fwd_values[-3:]
    drift = max(recent) - min(recent) if len(recent) >= 2 else None
    job_done = any("JOB DONE" in line for line in lines)
    last_iteration = iterations[-1].get("iteration") if iterations else None
    last_complete_iteration = last.get("iteration")
    complete_final_iteration = last_iteration is not None and last_iteration == last_complete_iteration
    force_gate = bool(errors_movable) and max(errors_movable) <= 0.03
    drift_gate = drift is not None and drift <= 0.02
    converged = job_done and complete_final_iteration and force_gate and drift_gate

    return {
        "neb_out": str(path),
        "job_done": job_done,
        "initial_path_length_bohr": initial_path_length_bohr,
        "initial_inter_image_distance_bohr": initial_inter_image_distance_bohr,
        "overflow_flags": sorted(set(overflow_flags)),
        "has_overflow": bool(overflow_flags),
        "converged_by_default_gate": converged,
        "convergence_force_gate_passed": force_gate,
        "convergence_drift_gate_passed": drift_gate,
        "last_iteration_complete": complete_final_iteration,
        "last_iteration": last_iteration,
        "last_complete_iteration": last_complete_iteration,
        "activation_forward_eV": last.get("activation_forward_eV"),
        "activation_reverse_eV": last.get("activation_reverse_eV"),
        "max_image_error_eV_A": max(errors_movable) if errors_movable else None,
        "max_image_error_movable_eV_A": max(errors_movable) if errors_movable else None,
        "max_image_error_all_eV_A": max(errors_all) if errors_all else None,
        "barrier_drift_last_three_eV": drift,
        "n_iterations_parsed": len(iterations),
        "n_complete_iterations_parsed": len(complete_iterations),
        "n_images_last_iteration": len(last.get("images", [])),
        "last_tcpu_seconds": max(tcpu_values) if tcpu_values else None,
        "scf_not_converged_warning_count": scf_not_converged_count,
        "last_images": last.get("images", []),
        "complete_iterations": complete_iterations,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("neb_out", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = parse_neb_out(args.neb_out)
    text = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
