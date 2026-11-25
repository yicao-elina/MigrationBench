#!/usr/bin/env python3
"""Import raw GPFS quota output using its capture wrapper timestamp sidecar."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: import_quota_capture_with_receipt_status.py RAW_OUTPUT STATUS_JSON")
    raw = Path(sys.argv[1]).resolve(strict=True)
    status_path = Path(sys.argv[2]).resolve(strict=True)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    captured = status.get("local_captured_utc") or status.get("local_observed_utc")
    if not captured:
        raise SystemExit("capture status has no wrapper observation timestamp")
    observed = datetime.fromisoformat(captured.replace("Z", "+00:00")).astimezone(timezone.utc)
    observed_utc = observed.isoformat(timespec="seconds").replace("+00:00", "Z")
    lines = raw.read_text(encoding="utf-8").splitlines()
    headers: dict[str, list[str]] = {}
    rows: list[dict[str, str]] = []
    for line in lines:
        if not line.startswith("mmlsquota:"):
            continue
        parts = line.split(":")
        scope, kind = parts[1], parts[2]
        if kind == "HEADER":
            headers[scope] = parts[3:]
        elif headers.get(scope):
            row = dict(zip(headers[scope], parts[3:]))
            if row.get("fid") == "0" and row.get("filesetname") in ("", "root"):
                rows.append(row)
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    source_key = "rockfish_quota_wrapper_capture_" + observed.strftime("%Y%m%dT%H%M%SZ")
    command = status.get("remote_command") or status.get("command") or "UNRECORDED"
    con = sqlite3.connect(DB, timeout=180)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=180000")
    try:
        con.execute("BEGIN IMMEDIATE")
        prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
        expected = (str(raw), digest)
        if prior and tuple(prior) != expected:
            raise RuntimeError("quota source key already points at different bytes")
        con.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
               VALUES(?,?,'GPFS_QUOTA_CAPTURE',?,?,?,1,?,?)""",
            (source_key, str(raw), digest, observed_utc, observed_utc,
             "Raw GPFS quota command output; observation time comes from the local SSH wrapper status because the remote timestamp line was not present in stdout.",
             "Raw bytes are not modified or prefixed. The local wrapper timestamp and exact command are in the sidecar status JSON."),
        )
        inserted = 0
        for row in rows:
            fs, kind, name = row.get("filesystemName"), row.get("quotaType"), row.get("name")
            if fs == "data" and kind == "USR" and name == "ycao73":
                root, subject_kind = "RF-DATA-FS", "USER"
            elif fs == "scratch4" and kind == "USR" and name == "ycao73":
                root, subject_kind = "RF-SCRATCH4-FS", "USER"
            elif fs == "data" and kind == "GRP" and name == "pclancy3":
                root, subject_kind = "RF-DATA-FS", "GROUP"
            elif fs == "scratch4" and kind == "GRP" and name == "pclancy3":
                root, subject_kind = "RF-SCRATCH4-FS", "GROUP"
            else:
                continue
            if con.execute("""SELECT 1 FROM quota_observation WHERE root_id=? AND observed_utc=?
                              AND quota_subject_kind=? AND quota_subject=?""",
                           (root, observed_utc, subject_kind, name)).fetchone():
                continue
            con.execute(
                """INSERT INTO quota_observation
                   (root_id,observed_utc,quota_subject_kind,quota_subject,used_bytes,quota_bytes,hard_limit_bytes,
                    in_doubt_bytes,grace_state,source_command,raw_output_path,files_used,files_quota,files_limit,
                    files_in_doubt,notes)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (root, observed_utc, subject_kind, name, int(row["blockUsage"])*1024,
                 int(row["blockQuota"])*1024 if int(row["blockQuota"]) else None,
                 int(row["blockLimit"])*1024 if int(row["blockLimit"]) else None,
                 int(row["blockInDoubt"])*1024, row["blockGrace"], command, str(raw),
                 int(row["filesUsage"]), int(row["filesQuota"]) if int(row["filesQuota"]) else None,
                 int(row["filesLimit"]) if int(row["filesLimit"]) else None, int(row["filesInDoubt"]),
                 "Group/user fileset total; quota change is not localized beyond filesystem/root and concurrent activity can affect the bracket."),
            )
            inserted += 1
        con.commit()
        print(json.dumps({"source_key": source_key, "source_sha256": digest, "observed_utc": observed_utc,
                          "inserted_observations": inserted, "raw_path": str(raw)}, sort_keys=True))
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


if __name__ == "__main__":
    main()
