#!/usr/bin/env python3
"""Load a raw GPFS `mmlsquota -Y` capture without rounding its counters."""

from __future__ import annotations

import hashlib
import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDIT_DIR = HERE.parent
DB = AUDIT_DIR / "provenance_catalog.sqlite"
DEFAULT_RAW = AUDIT_DIR / "phase2_data" / "rockfish_quota_baseline_20261122.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_utc(value: str) -> str:
    dt = datetime.fromisoformat(value)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_capture(raw: Path) -> tuple[str, list[dict[str, str]]]:
    lines = raw.read_text(encoding="utf-8").splitlines()
    observed = to_utc(lines[0].strip())
    header: dict[str, list[str]] = {}
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if not line.startswith("mmlsquota:"):
            continue
        parts = line.split(":")
        scope = parts[1]
        kind = parts[2]
        if kind == "HEADER":
            header[scope] = parts[3:]
            continue
        names = header.get(scope)
        if not names:
            raise RuntimeError(f"Quota header missing for {scope}")
        row = dict(zip(names, parts[3:]))
        if row.get("fid") != "0" or row.get("filesetname") not in ("", "root"):
            continue
        rows.append(row)
    return observed, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--source-key", default="rockfish_quota_baseline_20261122")
    parser.add_argument("--source-command", default=(
        "date -Is; /usr/lpp/mmfs/bin/mmlsquota -u ycao73 -Y data; "
        "/usr/lpp/mmfs/bin/mmlsquota -g pclancy3 -Y data scratch4"))
    args = parser.parse_args()
    raw = args.raw.resolve()
    observed, rows = parse_capture(raw)
    command = args.source_command
    local_path = str(raw)
    digest = sha256(raw)
    with sqlite3.connect(DB) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        prior = connection.execute(
            "SELECT source_path,sha256 FROM registry_source WHERE source_key=?",
            (args.source_key,),
        ).fetchone()
        if prior and prior != (local_path, digest):
            raise SystemExit("Quota evidence changed after registration; use a new source key.")
        connection.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
                immutable_snapshot,authority_note)
               VALUES (?, ?, 'GPFS_QUOTA_CAPTURE',?,?,?,1,?)""",
            (args.source_key, local_path, digest, observed, observed,
             "Raw machine-readable GPFS quota capture; block counters are 1KiB units and file counts are separate."),
        )
        for row in rows:
            filesystem = row["filesystemName"]
            quota_type = row["quotaType"]
            if filesystem == "data" and quota_type == "USR" and row["name"] == "ycao73":
                root_id = "RF-DATA-FS"
                subject_kind = "USER"
            elif filesystem == "data" and quota_type == "GRP" and row["name"] == "pclancy3":
                root_id = "RF-DATA-FS"
                subject_kind = "GROUP"
            elif filesystem == "scratch4" and quota_type == "GRP" and row["name"] == "pclancy3":
                root_id = "RF-SCRATCH4-FS"
                subject_kind = "GROUP"
            else:
                continue
            values = (
                root_id, observed, subject_kind, row["name"],
                int(row["blockUsage"]) * 1024,
                int(row["blockQuota"]) * 1024 if int(row["blockQuota"]) else None,
                int(row["blockLimit"]) * 1024 if int(row["blockLimit"]) else None,
                int(row["blockInDoubt"]) * 1024,
                row["blockGrace"], command, local_path,
                int(row["filesUsage"]),
                int(row["filesQuota"]) if int(row["filesQuota"]) else None,
                int(row["filesLimit"]) if int(row["filesLimit"]) else None,
                int(row["filesInDoubt"]),
                "Raw GPFS -Y block counters converted from 1KiB units; zero quotas mean no configured limit. Fileset root row only.",
            )
            existing = connection.execute(
                """SELECT quota_observation_id FROM quota_observation
                   WHERE root_id=? AND observed_utc=? AND quota_subject_kind=? AND quota_subject=?""",
                values[:4],
            ).fetchone()
            if existing:
                continue
            connection.execute(
                """INSERT INTO quota_observation
                   (root_id,observed_utc,quota_subject_kind,quota_subject,used_bytes,
                    quota_bytes,hard_limit_bytes,in_doubt_bytes,grace_state,source_command,
                    raw_output_path,files_used,files_quota,files_limit,files_in_doubt,notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                values,
            )
        connection.commit()
    with sqlite3.connect(DB) as connection:
        for row in connection.execute(
            """SELECT root_id,observed_utc,quota_subject_kind,quota_subject,
                      used_bytes,quota_bytes,in_doubt_bytes,grace_state,
                      files_used,files_quota,files_in_doubt
               FROM quota_observation ORDER BY quota_observation_id"""
        ):
            print("\t".join(str(x) if x is not None else "NULL" for x in row))


if __name__ == "__main__":
    main()
