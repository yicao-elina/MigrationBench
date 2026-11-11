#!/usr/bin/env python3
"""Prepare and optionally submit the isolated third exact-protocol replicate.

The command is intentionally fail-closed.  It refuses to create the target
root unless every current reproduction is terminal, scientifically classified,
and provenance-validated.  It never copies a checkpoint or trajectory and it
never modifies either the protected original or the current reproduction root.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


EXPECTED = {
    ("baseline", 345): "9cadf6fa3c3212c512bc06ad1bb9392bd9b664c40949157ee6184e9c0b05811d",
    ("ablation_lmax_2", 123): "a26db3575de8484ca3c7fcf7621165e38ce0162a580cde022bccd16135c61351",
    ("ablation_lmax_2", 234): "52fccd0994664618256ec5e986dc5e962050c845d3f05f3d70862ff064009af5",
}
EXPECTED_INPUT_SHA = "bb42c493fb0c997af3957a89dfde03e3e5a67ebcfd37f5eced8f722dad546161"
RUNS = (
    ("baseline", 345),
    ("ablation_lmax_2", 123),
    ("ablation_lmax_2", 234),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    submission_path = args.target_root / "third_replicate_submission_manifest.json"
    resume = False
    existing_submission = None
    if args.target_root.exists():
        # Support a deliberate two-phase workflow: the first invocation creates
        # the isolated root and records a completed preflight; a second
        # invocation may submit only that exact, still-unsubmitted root.
        if not args.submit or not submission_path.exists():
            raise SystemExit(f"refusing to use existing target root: {args.target_root}")
        existing_submission = load(submission_path)
        if existing_submission.get("submitted") is not False:
            raise SystemExit(f"refusing to resubmit existing target root: {args.target_root}")
        resume = True

    rows = []
    for family, seed in RUNS:
        run_name = f"1ns__{family}__seed{seed}__mdseed20261005__r00"
        run = args.current_root / "runs" / run_name
        manifest_path = run / "run_manifest.json"
        status_path = run / "status.json"
        metrics_path = run / "metrics.csv"
        trajectory_path = run / "trajectory.extxyz"
        for path in (manifest_path, status_path, metrics_path, trajectory_path):
            if not path.exists():
                raise SystemExit(f"missing required terminal artifact: {path}")
        manifest = load(manifest_path)
        status = load(status_path)
        if manifest.get("family") != family or int(manifest.get("training_seed", -1)) != seed:
            raise SystemExit(f"manifest mapping mismatch: {manifest_path}")
        if manifest.get("model_sha256") != EXPECTED[(family, seed)]:
            raise SystemExit(f"model hash mismatch: {manifest_path}")
        model_path = Path(str(manifest.get("model", "")))
        if not model_path.exists() or sha256_file(model_path) != EXPECTED[(family, seed)]:
            raise SystemExit(f"live model bytes do not match locked hash: {model_path}")
        if manifest.get("input_sha256") != EXPECTED_INPUT_SHA:
            raise SystemExit(f"input hash mismatch: {manifest_path}")
        if int(manifest.get("md_seed", -1)) != 20261005:
            raise SystemExit(f"MD seed mismatch: {manifest_path}")
        if int(manifest.get("natoms", -1)) != 2050 or tuple(manifest.get("pbc", [])) != (True, True, True):
            raise SystemExit(f"atom/PBC manifest mismatch: {manifest_path}")
        if status.get("status") not in {"failed", "complete"}:
            raise SystemExit(f"current replicate is not terminal: {status_path}: {status}")
        if status.get("status") == "failed" and status.get("stop_reason") != "instability_threshold":
            raise SystemExit(f"non-scientific failure cannot be cloned: {status_path}: {status}")
        if int(status.get("last_step", -1)) <= 0:
            raise SystemExit(f"invalid terminal step: {status_path}")
        rows.append({
            "family": family,
            "training_seed": seed,
            "source_run": str(run),
            "status": status.get("status"),
            "stop_reason": status.get("stop_reason"),
            "last_step": status.get("last_step"),
            "model_sha256": manifest.get("model_sha256"),
            "input_sha256": manifest.get("input_sha256"),
        })

    registry = args.current_root / "registry" / "selected_1ns.tsv"
    input_path = args.current_root / "inputs" / "common_initial.xyz"
    if sha256_file(input_path) != EXPECTED_INPUT_SHA:
        raise SystemExit(f"current input bytes do not match locked hash: {input_path}")
    for path in (args.current_root / "code" / "run_ablation_md.py", args.current_root / "code" / "submit_1ns.slurm", registry):
        if not path.exists():
            raise SystemExit(f"missing source campaign file: {path}")

    if resume:
        for relative in (
            "code/run_ablation_md.py",
            "code/submit_1ns.slurm",
            "inputs/common_initial.xyz",
            "registry/selected_1ns.tsv",
            "third_replicate_submission_manifest.json",
        ):
            if not (args.target_root / relative).is_file():
                raise SystemExit(f"incomplete preflight target root: {args.target_root / relative}")
        submission = existing_submission
        submission["preflight_rows"] = rows
    else:
        args.target_root.mkdir(parents=True)
        for name in ("code", "inputs", "registry", "logs", "runs"):
            (args.target_root / name).mkdir()
        for relative in (
            "code/PROTOCOL_AND_COMPARISON.md",
            "code/README.md",
            "code/run_ablation_md.py",
            "code/submit_1ns.slurm",
            "inputs/common_initial.xyz",
            "registry/selected_1ns.tsv",
            "registry/usable_models.tsv",
            "source_campaign_manifest.json",
        ):
            source = args.current_root / relative
            destination = args.target_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        submission = {
            "schema": "migrationbench_from_scratch_third_replicate_submission_v1",
            "target_root": str(args.target_root),
            "source_root": str(args.current_root),
            "array": "0-2%2",
            "mapping": [
                {"array_task": 0, "family": "baseline", "training_seed": 345},
                {"array_task": 1, "family": "ablation_lmax_2", "training_seed": 123},
                {"array_task": 2, "family": "ablation_lmax_2", "training_seed": 234},
            ],
            "preflight_rows": rows,
            "input_sha256": EXPECTED_INPUT_SHA,
            "protocol": {"md_seed": 20261005, "steps": 1000000, "write_interval_steps": 200, "temperature_K": 600.0, "friction_fs^-1": 0.01, "timestep_fs": 1.0},
            "submitted": False,
        }
    if args.submit:
        command = ["sbatch", str(args.target_root / "code" / "submit_1ns.slurm")]
        result = subprocess.run(command, cwd=args.target_root, env={**os.environ, "CAMPAIGN_ROOT": str(args.target_root)}, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not match:
            raise SystemExit(f"could not parse sbatch output: {result.stdout!r}")
        submission["job_id"] = match.group(1)
        submission["submitted"] = True
        submission["sbatch_stdout"] = result.stdout.strip()
    submission_path.write_text(json.dumps(submission, indent=2, sort_keys=True) + "\n")
    print(json.dumps(submission, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
