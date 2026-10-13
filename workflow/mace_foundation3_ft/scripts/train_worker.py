#!/usr/bin/env python3
"""Run one additive foundation fine-tuning task with strict provenance gates."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    requested = os.environ.get("CAMPAIGN_FAMILIES")
    family_ids = [item.strip() for item in requested.split(",")] if requested else list(cfg["families"])
    family_ids = [item for item in family_ids if item]
    seeds = [int(value) for value in cfg["training"]["seeds"]]
    jobs = [(family_id, seed) for family_id in family_ids for seed in seeds]
    if not 0 <= args.index < len(jobs):
        raise SystemExit(f"invalid array index {args.index}; expected 0..{len(jobs) - 1}")

    family_id, seed = jobs[args.index]
    family = cfg["families"][family_id]
    foundation = cfg["foundation_models"][family["foundation_model"]]
    if os.environ.get("MACE_RUNTIME") != family["runtime"]:
        raise SystemExit(f"runtime mismatch for {family_id}: expected {family['runtime']}")
    if foundation["model_type"] == "PolarMACE" and os.environ.get("POLAR_DATA_CONTRACT_APPROVED") != "1":
        raise SystemExit("POLAR_DATA_CONTRACT_NOT_APPROVED: no charge/spin/external_field adapter was approved")

    dataset = cfg["datasets"][family["dataset"]]
    campaign_root = Path(os.environ.get("CAMPAIGN_OUTPUT_ROOT", cfg["remote_root"]))
    if args.smoke:
        campaign_root = campaign_root / "smoke"
    root = campaign_root / family["output_dir_template"].format(seed=seed)
    if root.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)

    model = Path(foundation["source"])
    if not model.is_file() or sha256(model) != foundation["sha256"]:
        raise SystemExit(f"foundation model SHA256 mismatch: {model}")
    files = [Path(dataset["root"]) / name for name in ("train.xyz", "valid.xyz", "test.xyz")]
    for path in files:
        expected = dataset["files"][path.name]
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"dataset preflight failed: {path}")

    name = f"{family_id}_seed{seed}"
    train = dict(cfg["training"])
    if args.smoke:
        train["max_num_epochs"] = int(cfg["training"]["smoke_max_num_epochs"])
    model_type = foundation["model_type"]
    command = [
        "mace_run_train", f"--name={name}", f"--model_dir={root}", f"--log_dir={root}",
        f"--checkpoints_dir={root}", f"--results_dir={root}",
        f"--train_file={files[0]}", f"--valid_file={files[1]}", f"--test_file={files[2]}",
        f"--model={model_type}", f"--num_channels={train['num_channels']}", f"--max_L={train['max_L']}",
        f"--r_max={train['r_max']}", f"--hidden_irreps={train['hidden_irreps']}",
        f"--batch_size={train['batch_size']}", f"--valid_batch_size={train['valid_batch_size']}",
        f"--max_num_epochs={train['max_num_epochs']}", f"--eval_interval={train['eval_interval']}",
        f"--patience={train['patience']}", f"--forces_weight={train['forces_weight']}",
        f"--energy_weight={train['energy_weight']}", f"--stress_weight={train['stress_weight']}",
        f"--energy_key={train['energy_key']}", f"--forces_key={train['forces_key']}",
        f"--device={train['device']}", f"--seed={seed}", f"--E0s={family['e0s']}",
        f"--foundation_model={model}", f"--multiheads_finetuning={family['multiheads_finetuning']}",
    ]
    if train.get("swa"):
        command += ["--swa", f"--start_swa={train['start_swa']}"]
    if train.get("ema"):
        command += ["--ema", f"--ema_decay={train['ema_decay']}"]

    payload = {
        "schema_version": "1.0",
        "campaign_id": cfg["campaign_id"],
        "foundation_name": foundation["canonical_name"],
        "foundation_id": family["foundation_model"],
        "foundation_path": str(model),
        "foundation_sha256": foundation["sha256"],
        "foundation_public_url": foundation["public_url"],
        "foundation_training_lineage": foundation["training_lineage"],
        "model_type": model_type,
        "model_family": foundation["model_family"],
        "comparability": foundation["comparability"],
        "family": family_id,
        "dataset_id": family["dataset"],
        "dataset_paths": [str(path) for path in files],
        "dataset_sha256": {str(path): sha256(path) for path in files},
        "training_seed": seed,
        "md_velocity_seed": int(cfg["md"]["md_velocity_seed_by_training_seed"][str(seed)]),
        "command": command,
        "code_commit": os.environ.get("CAMPAIGN_GIT_COMMIT", "3218331"),
        "runtime": {
            "python": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "mace_torch": package_version("mace-torch"),
            "torch": package_version("torch"),
            "ase": package_version("ase"),
            "graph_longrange": package_version("graph_longrange"),
        },
        "environment_manifest": str(root / "environment_manifest.json"),
        "artifact_path": None,
        "artifact_sha256": None,
        "training_log": str(
            campaign_root
            / "logs"
            / (
                f"train-{os.environ.get('SLURM_ARRAY_JOB_ID', os.environ.get('SLURM_JOB_ID', 'local'))}_"
                f"{os.environ.get('SLURM_ARRAY_TASK_ID', args.index)}-{family['runtime']}.out"
            )
        ),
        "training_completion_state": "started",
        "smoke": bool(args.smoke),
        "output_phase": "smoke" if args.smoke else "production",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.uname().nodename,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    (root / "environment_manifest.json").write_text(
        json.dumps(
            {
                "campaign_id": cfg["campaign_id"],
                "family": family_id,
                "training_seed": seed,
                "python": payload["runtime"],
                "pip_freeze": freeze.stdout.splitlines(),
                "pip_freeze_returncode": freeze.returncode,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (root / "preflight.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        payload.update(
            {
                "training_completion_state": "failed",
                "failure_returncode": exc.returncode,
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        (root / "training_failure.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        raise

    artifact = None
    for pattern in train["artifact_candidates"]:
        candidate = root / pattern.format(name=name)
        if candidate.is_file() and candidate.stat().st_size > 0:
            artifact = candidate
            break
    if artifact is None:
        raise SystemExit(f"training completed but no model artifact matched {train['artifact_candidates']}")
    payload.update({
        "artifact_path": str(artifact),
        "artifact_sha256": sha256(artifact),
        "training_completion_state": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_metrics": sorted(str(path) for path in root.rglob("*") if path.is_file() and "test" in path.name.lower()),
    })
    (root / "training_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
