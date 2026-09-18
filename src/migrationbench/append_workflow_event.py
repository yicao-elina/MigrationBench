#!/usr/bin/env python3
"""Append a machine-readable event to the MigrationBench workflow log.

Events are JSON objects with enough provenance for comparing repair, MLFF,
and QE branches. This intentionally uses only the Python standard library so
it can run on login nodes as well as local machines.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REQUIRED = [
    "event_id",
    "timestamp_utc",
    "event_type",
    "path_id",
    "branch_id",
    "status",
    "inputs",
    "outputs",
    "config",
    "metrics",
    "decision",
]


def sha256_file(path):
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_arg(value, default):
    if value is None:
        return default
    stripped = value.lstrip()
    if stripped.startswith(("{", "[")):
        return json.loads(value)
    p = Path(value)
    if p.exists():
        return json.loads(p.read_text())
    return json.loads(value)


def stable_event_id(event):
    basis = json.dumps(
        {
            "event_type": event.get("event_type"),
            "path_id": event.get("path_id"),
            "branch_id": event.get("branch_id"),
            "status": event.get("status"),
            "inputs": event.get("inputs"),
            "outputs": event.get("outputs"),
            "config": event.get("config"),
            "metrics": event.get("metrics"),
            "decision": event.get("decision"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(basis.encode()).hexdigest()[:16]


def validate(event):
    missing = [key for key in REQUIRED if key not in event]
    if missing:
        raise ValueError(f"missing required event fields: {', '.join(missing)}")
    for key in ["inputs", "outputs", "config", "metrics", "decision"]:
        if not isinstance(event[key], dict):
            raise ValueError(f"{key} must be an object")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="JSONL workflow log path")
    parser.add_argument("--event-type", required=True)
    parser.add_argument("--path-id", required=True)
    parser.add_argument("--branch-id", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--inputs", help="JSON string or JSON file")
    parser.add_argument("--outputs", help="JSON string or JSON file")
    parser.add_argument("--config", help="JSON string or JSON file")
    parser.add_argument("--metrics", help="JSON string or JSON file")
    parser.add_argument("--decision", help="JSON string or JSON file")
    parser.add_argument("--links", help="JSON string or JSON file")
    parser.add_argument("--notes", default="")
    parser.add_argument("--checksum-file", action="append", default=[], help="Input/output file to hash and record")
    args = parser.parse_args()

    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_type": args.event_type,
        "path_id": args.path_id,
        "branch_id": args.branch_id,
        "status": args.status,
        "inputs": load_json_arg(args.inputs, {}),
        "outputs": load_json_arg(args.outputs, {}),
        "config": load_json_arg(args.config, {}),
        "metrics": load_json_arg(args.metrics, {}),
        "decision": load_json_arg(args.decision, {}),
        "links": load_json_arg(args.links, {}),
        "notes": args.notes,
    }
    checksums = {path: sha256_file(path) for path in args.checksum_file}
    checksums = {k: v for k, v in checksums.items() if v}
    if checksums:
        event.setdefault("provenance", {})["sha256"] = checksums
    event["event_id"] = stable_event_id(event)
    validate(event)

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    print(json.dumps(event, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"append_workflow_event.py: {exc}", file=sys.stderr)
        raise
