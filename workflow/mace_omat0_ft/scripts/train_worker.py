#!/usr/bin/env python3
"""Run one configuration-defined MACE fine-tuning task and write its manifest."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--index", required=True, type=int)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    seeds = [int(x) for x in cfg["training"]["seeds"]]
    jobs = [(family_id, seed) for family_id in cfg["families"] for seed in seeds]
    if not 0 <= args.index < len(jobs):
        raise SystemExit(f"invalid array index {args.index}; expected 0..{len(jobs)-1}")
    family_id, seed = jobs[args.index]
    if args.smoke:
        family_id = cfg["training"]["smoke_family"]
        seed = int(cfg["training"]["smoke_seed"])
    family = cfg["families"][family_id]
    dataset = cfg["datasets"][family["dataset"]]
    foundation = cfg["foundation_models"][cfg["selected_foundation_model"]]
    campaign_root = Path(os.environ.get("CAMPAIGN_OUTPUT_ROOT", cfg["remote_root"]))
    root = campaign_root / "training" / family_id / f"seed{seed}"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "training_manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite existing training manifest: {manifest_path}")
    model = Path(foundation["source"])
    if sha256(model) != foundation["sha256"]:
        raise SystemExit("foundation model SHA256 mismatch")
    files = [Path(dataset["root"]) / name for name in ("train.xyz", "valid.xyz", "test.xyz")]
    for path, expected in zip(files, [dataset["files"][p.name] for p in files]):
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"dataset preflight failed: {path}")
    name = f"{family_id}_seed{seed}"
    t = dict(cfg["training"])
    if args.smoke:
        t["max_num_epochs"] = cfg["training"]["smoke_max_num_epochs"]
    cmd = [
        "mace_run_train", f"--name={name}", f"--model_dir={root}", f"--log_dir={root}",
        f"--checkpoints_dir={root}", f"--results_dir={root}",
        f"--train_file={files[0]}", f"--valid_file={files[1]}", f"--test_file={files[2]}",
        f"--model={t['model']}", f"--num_channels={t['num_channels']}", f"--max_L={t['max_L']}",
        f"--r_max={t['r_max']}", f"--hidden_irreps={t['hidden_irreps']}",
        f"--batch_size={t['batch_size']}", f"--valid_batch_size={t['valid_batch_size']}",
        f"--max_num_epochs={t['max_num_epochs']}", f"--eval_interval={t['eval_interval']}",
        f"--patience={t['patience']}", f"--forces_weight={t['forces_weight']}",
        f"--energy_weight={t['energy_weight']}", f"--stress_weight={t['stress_weight']}",
        f"--energy_key={t['energy_key']}", f"--forces_key={t['forces_key']}",
        f"--device={t['device']}", f"--seed={seed}", f"--E0s={family['e0s']}",
        f"--foundation_model={model}", f"--multiheads_finetuning={family['multiheads_finetuning']}",
    ]
    if t.get("swa"):
        cmd += ["--swa", f"--start_swa={t['start_swa']}"]
    if t.get("ema"):
        cmd += ["--ema", f"--ema_decay={t['ema_decay']}"]
    payload = {
        "schema_version": "1.0", "campaign_id": cfg["campaign_id"], "family_id": family_id,
        "family_label": family["label"], "training_seed": seed,
        "foundation_model_id": cfg["selected_foundation_model"], "foundation_model": str(model),
        "foundation_model_sha256": foundation["sha256"], "dataset_id": family["dataset"],
        "dataset_files": {str(p): sha256(p) for p in files}, "command": cmd,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "hostname": os.uname().nodename,
        "smoke": bool(args.smoke),
    }
    (root / "preflight.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    subprocess.run(cmd, check=True)
    artifact = None
    for pattern in t["artifact_candidates"]:
        candidate = root / pattern.format(name=name)
        if candidate.is_file() and candidate.stat().st_size > 0:
            artifact = candidate
            break
    if artifact is None:
        raise SystemExit(f"training completed but no model artifact matched {t['artifact_candidates']}")
    payload.update({"status": "completed", "artifact": str(artifact), "artifact_sha256": sha256(artifact)})
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
