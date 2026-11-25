#!/usr/bin/env python3
"""Build an endpoint-only, provenance-first audit for the collaborator NEBs.

The remote tree is read through ssh.  Only pw_1.in and pw_2.in are materialized
as local endpoint inputs.  Existing NEB path/checkpoint files are inspected for
the DFT gate, but are never used to construct an endpoint-only interpolation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REMOTE_ROOT = "/scratch4/pclancy3/joshitha/vdW-sratch/sb2te3"
BOHR_TO_A = 0.529177210903


def ssh(host: str, command: str) -> bytes:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, command],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_remote(host: str, path: str) -> str:
    out = ssh(host, f"sha256sum -- {shlex.quote(path)}")
    return out.decode().split()[0]


def remote_file(host: str, path: str) -> tuple[bytes | None, str]:
    try:
        data = ssh(host, f"cat -- {shlex.quote(path)}")
    except subprocess.CalledProcessError:
        return None, "missing_or_unreadable"
    return data, sha256_bytes(data)


def remote_inventory(host: str, root: str) -> list[str]:
    out = ssh(
        host,
        "find "
        + shlex.quote(root)
        + " -maxdepth 1 -type f -printf '%f\\n' | sort",
    )
    return [x for x in out.decode().splitlines() if x]


def parse_float(text: str, key: str) -> float | None:
    m = re.search(rf"^\s*{re.escape(key)}\s*=\s*([-+0-9.eEdD]+)", text, re.M)
    if not m:
        return None
    return float(m.group(1).replace("D", "E").replace("d", "e"))


def parse_int(text: str, key: str) -> int | None:
    value = parse_float(text, key)
    return int(value) if value is not None else None


def parse_endpoint(text: str) -> dict:
    nat = parse_int(text, "nat")
    m = re.search(r"ATOMIC_POSITIONS\s+(\S+)\s*\n(.*?)(?=\n(?:K_POINTS|CELL_PARAMETERS)\b|\Z)", text, re.S)
    if not m:
        raise ValueError("ATOMIC_POSITIONS card not found")
    unit = m.group(1).lower()
    atoms = []
    for line in m.group(2).splitlines():
        fields = line.split()
        if len(fields) >= 4 and re.match(r"^[A-Z][a-z]?$", fields[0]):
            atoms.append((fields[0], *(float(x.replace("D", "E")) for x in fields[1:4])))
    if nat is not None and len(atoms) != nat:
        raise ValueError(f"nat={nat} but parsed {len(atoms)} atoms")
    c = re.search(r"CELL_PARAMETERS\s+(\S+)\s*\n([^\n]+)\n([^\n]+)\n([^\n]+)", text)
    if not c:
        raise ValueError("CELL_PARAMETERS card not found")
    cell_unit = c.group(1).lower()
    cell = np.array([[float(x.replace("D", "E")) for x in c.group(i).split()] for i in (2, 3, 4)])
    if unit.startswith("bohr"):
        coords = np.array([[a[1], a[2], a[3]] for a in atoms]) * BOHR_TO_A
    else:
        coords = np.array([[a[1], a[2], a[3]] for a in atoms])
    if cell_unit.startswith("bohr"):
        cell = cell * BOHR_TO_A
    symbols = [a[0] for a in atoms]
    return {"nat": len(atoms), "symbols": symbols, "positions_A": coords, "cell_A": cell, "pbc": [True, True, True], "coordinate_unit": unit, "cell_unit": cell_unit}


def endpoint_extxyz(ep: dict, path_id: str, role: str, source_path: str, source_sha: str) -> str:
    lattice = " ".join(f"{x:.12f}" for x in ep["cell_A"].reshape(-1))
    lines = [str(ep["nat"]), f'path_id="{path_id}" role="{role}" source_path="{source_path}" source_sha256="{source_sha}" pbc="T T T" Lattice="{lattice}" Properties=species:S:1:pos:R:3']
    lines += [f"{s} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}" for s, p in zip(ep["symbols"], ep["positions_A"])]
    return "\n".join(lines) + "\n"


def parse_path_file(text: str, nat: int) -> dict:
    blocks = re.split(r"\nImage:\s*", text)
    images = []
    for block in blocks[1:]:
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        try:
            energy = float(lines[1].strip().split()[0].replace("D", "E"))
        except (ValueError, IndexError):
            continue
        rows = []
        for line in lines[2:]:
            vals = line.split()
            if len(vals) >= 6:
                try:
                    rows.append([float(x.replace("D", "E")) for x in vals[:9]])
                except ValueError:
                    pass
            if len(rows) == nat:
                break
        images.append({"energy_eV": energy, "n_rows": len(rows), "rows": rows})
    finite = bool(images) and all(
        math.isfinite(im["energy_eV"])
        and im["n_rows"] == nat
        and all(math.isfinite(v) for row in im["rows"] for v in row)
        for im in images
    )
    max_grad = None
    if finite:
        max_grad = max(float(np.linalg.norm(np.asarray(row[3:6]))) for im in images for row in im["rows"])
    return {"n_images": len(images), "all_images_finite": finite, "max_gradient_raw": max_grad}


def min_image_max_diff(a: np.ndarray, b: np.ndarray, cell: np.ndarray) -> float:
    inv = np.linalg.inv(cell)
    fa = a @ inv
    fb = b @ inv
    df = fa - fb
    df -= np.round(df)
    return float(np.max(np.linalg.norm(df @ cell, axis=1)))


def parse_output(text: str, expected_images: int) -> dict:
    warnings = re.findall(r"WARNING\s*:\s*scf convergence NOT achieved on image\s+(\d+)", text, re.I)
    explicit = bool(re.search(r"path\s+(?:has\s+)?converged|converged\s+path|path_thr.*reached", text, re.I))
    return {
        "job_done": bool(re.search(r"JOB DONE", text)),
        "explicit_path_convergence": explicit,
        "scf_warning_images": ",".join(warnings),
        "n_scf_warning_images": len(warnings),
        "expected_images": expected_images,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="rockfish")
    ap.add_argument("--remote-root", default=REMOTE_ROOT)
    ap.add_argument("--output", type=Path, default=Path("analysis/task3_endpoint_only_neb"))
    ap.add_argument("--snapshot", type=Path, default=None, help="Local read-only tar snapshot of audited remote files")
    args = ap.parse_args()
    out = args.output.resolve()
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "interpolations").mkdir(parents=True, exist_ok=True)
    snapshot = args.snapshot.resolve() if args.snapshot else None
    remote_hashes = {}
    hash_file = out / "remote_sha256.tsv"
    if hash_file.exists():
        for line in hash_file.read_text().splitlines():
            if "  " in line:
                h, p = line.split("  ", 1)
                remote_hashes[p] = h

    def get_file(path: str) -> tuple[bytes | None, str]:
        if snapshot:
            rel = path.removeprefix(args.remote_root).lstrip("/")
            local = snapshot / rel
            if not local.is_file():
                return None, "missing_or_unreadable"
            data = local.read_bytes()
            return data, remote_hashes.get(path, sha256_bytes(data))
        return remote_file(args.host, path)

    def list_files(remote_dir: str) -> list[str]:
        if snapshot:
            rel = remote_dir.removeprefix(args.remote_root).lstrip("/")
            local_dir = snapshot / rel
            return sorted(p.name for p in local_dir.iterdir() if p.is_file())
        return remote_inventory(args.host, remote_dir)

    if snapshot:
        paths = [str(p) for p in sorted(snapshot.glob("2L-*/NEB/*")) if p.is_dir()]
        paths = [args.remote_root + "/" + p.removeprefix(str(snapshot) + "/") for p in paths]
    else:
        paths = []
        for system in sorted(x.decode() for x in ssh(args.host, f"find {shlex.quote(args.remote_root)} -mindepth 2 -maxdepth 2 -type d -name NEB -print").splitlines()):
            neb_dirs = ssh(args.host, f"find {shlex.quote(system)} -mindepth 1 -maxdepth 1 -type d -print | sort").decode().splitlines()
            for d in neb_dirs:
                paths.append(d)
    manifest_rows = []
    leakage_rows = []
    endpoint_objects = {}
    for remote_dir in paths:
        parts = remote_dir.rstrip("/").split("/")
        system = parts[-3]
        neb_name = parts[-1]
        dopant = system.split("-")[1].split("+")[0]
        path_id = f"{system}__{neb_name}".replace(" ", "_")
        neb_in_path = f"{remote_dir}/neb.in"
        pw1_path = f"{remote_dir}/pw_1.in"
        pw2_path = f"{remote_dir}/pw_2.in"
        neb_bytes, neb_sha = get_file(neb_in_path)
        pw1_bytes, pw1_sha = get_file(pw1_path)
        pw2_bytes, pw2_sha = get_file(pw2_path)
        if not neb_bytes or not pw1_bytes or not pw2_bytes:
            continue
        neb_text = neb_bytes.decode(errors="replace")
        ep1_text = pw1_bytes.decode(errors="replace")
        ep2_text = pw2_bytes.decode(errors="replace")
        ep1, ep2 = parse_endpoint(ep1_text), parse_endpoint(ep2_text)
        endpoint_objects[path_id] = (ep1, ep2, pw1_path, pw2_path, pw1_sha, pw2_sha)
        endpoint_consistent = ep1["symbols"] == ep2["symbols"] and np.allclose(ep1["cell_A"], ep2["cell_A"], atol=1e-8) and ep1["nat"] == ep2["nat"]
        n_images = parse_int(neb_text, "num_of_images")
        path_thr = parse_float(neb_text, "path_thr")
        all_files = list_files(remote_dir)
        output_candidates = sorted([x for x in all_files if re.fullmatch(r"neb(?:\d+)?\.out", x)])
        output_name = output_candidates[-1] if output_candidates else ""
        out_bytes, out_sha = get_file(f"{remote_dir}/{output_name}") if output_name else (None, "missing")
        path_bytes, path_sha = get_file(f"{remote_dir}/sb2te3.path")
        path_info = parse_path_file(path_bytes.decode(errors="replace"), ep1["nat"]) if path_bytes else {"n_images": 0, "all_images_finite": False, "max_gradient_raw": None}
        output_info = parse_output(out_bytes.decode(errors="replace"), n_images or 0) if out_bytes else {"job_done": False, "explicit_path_convergence": False, "scf_warning_images": "", "n_scf_warning_images": 0, "expected_images": n_images}
        endpoint_path_diff = None
        endpoint_path_consistent = False
        if path_info["all_images_finite"] and path_info["n_images"] == (n_images or 0):
            # The path artifact is deliberately used only for audit.  It stores bohr coordinates.
            blocks = re.split(r"\nImage:\s*", path_bytes.decode(errors="replace"))[1:]
            first_rows, last_rows = [], []
            for block in (blocks[0], blocks[-1]):
                rows = []
                for line in block.splitlines()[2:]:
                    vals = line.split()
                    if len(vals) >= 6:
                        rows.append([float(x.replace("D", "E")) for x in vals[:9]])
                    if len(rows) == ep1["nat"]:
                        break
                (first_rows if not first_rows else last_rows).extend(rows)
            if len(first_rows) == ep1["nat"] and len(last_rows) == ep1["nat"]:
                d1 = min_image_max_diff(np.array(first_rows)[:, :3] * BOHR_TO_A, ep1["positions_A"], ep1["cell_A"])
                d2 = min_image_max_diff(np.array(last_rows)[:, :3] * BOHR_TO_A, ep2["positions_A"], ep2["cell_A"])
                endpoint_path_diff = max(d1, d2)
                endpoint_path_consistent = endpoint_path_diff < 0.25
        failure_reasons = []
        if not output_info["explicit_path_convergence"]:
            failure_reasons.append("no_explicit_path_convergence_marker")
        if output_info["n_scf_warning_images"]:
            failure_reasons.append(f"scf_not_converged_images:{output_info['scf_warning_images']}")
        if not path_info["all_images_finite"]:
            failure_reasons.append("missing_or_nonfinite_path_image_energy_force")
        if path_info["n_images"] != (n_images or 0):
            failure_reasons.append(f"image_count_mismatch:{path_info['n_images']}!=declared_{n_images}")
        if not endpoint_consistent:
            failure_reasons.append("endpoint_species_order_or_cell_mismatch")
        if not endpoint_path_consistent:
            failure_reasons.append("path_endpoint_identity_not_verified")
        accepted = not failure_reasons
        manifest_rows.append({
            "path_id": path_id, "system": system, "dopant": dopant, "remote_neb_dir": remote_dir,
            "neb_input_path": neb_in_path, "neb_input_sha256": neb_sha,
            "initial_endpoint_path": pw1_path, "initial_endpoint_sha256": pw1_sha,
            "final_endpoint_path": pw2_path, "final_endpoint_sha256": pw2_sha,
            "dft_output_path": f"{remote_dir}/{output_name}", "dft_output_sha256": out_sha,
            "dft_path_artifact": f"{remote_dir}/sb2te3.path", "dft_path_sha256": path_sha,
            "nat": ep1["nat"], "species_order_sha256": sha256_bytes("\n".join(ep1["symbols"]).encode()),
            "species_order": " ".join(ep1["symbols"]), "dopant_identity": dopant,
            "cell_A": json.dumps(np.asarray(ep1["cell_A"]).round(10).tolist()), "pbc": "T T T",
            "declared_n_images": n_images, "observed_path_images": path_info["n_images"], "path_thr_eV_A": path_thr,
            "all_path_images_finite": path_info["all_images_finite"], "max_path_gradient_raw": path_info["max_gradient_raw"],
            "job_done": output_info["job_done"], "explicit_path_convergence": output_info["explicit_path_convergence"],
            "scf_warning_images": output_info["scf_warning_images"], "endpoint_path_max_diff_A": endpoint_path_diff,
            "failure_reason": ";".join(failure_reasons), "scientific_success": accepted,
            "formal_mace_eligible": accepted,
        })
        leakage_rows.append({
            "path_id": path_id, "endpoint_source_only": True, "initial_source": pw1_path,
            "final_source": pw2_path, "initial_sha256": pw1_sha, "final_sha256": pw2_sha,
            "intermediate_files_seen_for_audit": ";".join(x for x in all_files if ".path" in x or ".xyz" in x or ".axsf" in x),
            "intermediate_files_used_as_initial_guess": "",
            "intermediate_input_leakage": False,
            "audit_note": "DFT path/checkpoint artifacts were inspected only for gate/identity diagnostics; endpoint inputs alone feed interpolation.",
        })
        pdir = out / "inputs"
        (pdir / f"{path_id}__initial.extxyz").write_text(endpoint_extxyz(ep1, path_id, "initial", pw1_path, pw1_sha))
        (pdir / f"{path_id}__final.extxyz").write_text(endpoint_extxyz(ep2, path_id, "final", pw2_path, pw2_sha))

    write_csv(out / "dft_gate_manifest.csv", manifest_rows)
    write_csv(out / "endpoint_only_leakage_audit.csv", leakage_rows)
    (out / "endpoint_only_provenance.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "remote_root": args.remote_root,
        "remote_host": args.host, "n_paths": len(manifest_rows), "endpoint_inputs_only": True,
        "intermediate_paths_not_used_as_initial_guess": True, "manifest_sha256": sha256_bytes((out / "dft_gate_manifest.csv").read_bytes()),
    }, indent=2) + "\n")
    print(json.dumps({"n_paths": len(manifest_rows), "dft_gate_pass": sum(bool(r["scientific_success"]) for r in manifest_rows), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
