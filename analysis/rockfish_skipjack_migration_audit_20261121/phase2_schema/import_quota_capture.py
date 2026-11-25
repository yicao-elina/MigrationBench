#!/usr/bin/env python3
"""Import a dated GPFS mmlsquota -Y capture as immutable evidence."""

import hashlib
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"


def main():
    raw = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else AUDIT / "phase2_data/rockfish_quota_after_wse2_20261122.txt"
    lines = raw.read_text().splitlines()
    timestamp_line = next((line.strip() for line in lines if line.strip() and
                           (line.strip()[0:4].isdigit() and "T" in line.strip())), None)
    if timestamp_line is None:
        raise SystemExit("quota capture has no ISO timestamp line")
    if timestamp_line.startswith("capture_utc="):
        timestamp_line = timestamp_line.split("=", 1)[1]
    dt = datetime.fromisoformat(timestamp_line)
    observed = dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    headers = {}
    rows = []
    for line in lines[1:]:
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
    h = hashlib.sha256(raw.read_bytes()).hexdigest()
    source_key = (sys.argv[3] if len(sys.argv) > 3 else
                  "rockfish_quota_capture_" + raw.parent.name + "_" + raw.stem)
    source_path = str(raw)
    command = (sys.argv[2] if len(sys.argv) > 2 else
               "date -Is; /usr/lpp/mmfs/bin/mmlsquota -u ycao73 -Y data; /usr/lpp/mmfs/bin/mmlsquota -g pclancy3 -Y data scratch4")
    con = sqlite3.connect(DB)
    try:
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("BEGIN IMMEDIATE")
        prior = con.execute("SELECT source_path,sha256 FROM registry_source WHERE source_key=?", (source_key,)).fetchone()
        if prior and prior != (source_path, h):
            raise SystemExit("same quota source key is already bound to different bytes")
        con.execute(
            """INSERT OR IGNORE INTO registry_source
               (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note)
               VALUES (?,?, 'GPFS_QUOTA_CAPTURE',?,?,?,1,?)""",
            (source_key, source_path, h, observed, observed,
             "Raw machine-readable GPFS mmlsquota output; block units are 1 KiB and file counts are separate."),
        )
        inserted = 0
        for row in rows:
            fs, typ, name = row["filesystemName"], row["quotaType"], row["name"]
            if fs == "data" and typ == "USR" and name == "ycao73":
                root, subject_kind = "RF-DATA-FS", "USER"
            elif fs == "scratch4" and typ == "USR" and name == "ycao73":
                root, subject_kind = "RF-SCRATCH4-FS", "USER"
            elif fs == "data" and typ == "GRP" and name == "pclancy3":
                root, subject_kind = "RF-DATA-FS", "GROUP"
            elif fs == "scratch4" and typ == "GRP" and name == "pclancy3":
                root, subject_kind = "RF-SCRATCH4-FS", "GROUP"
            else:
                continue
            exists = con.execute(
                "SELECT 1 FROM quota_observation WHERE root_id=? AND observed_utc=? AND quota_subject_kind=? AND quota_subject=?",
                (root, observed, subject_kind, name),
            ).fetchone()
            if exists:
                continue
            con.execute(
                """INSERT INTO quota_observation
                   (root_id,observed_utc,quota_subject_kind,quota_subject,used_bytes,quota_bytes,hard_limit_bytes,
                    in_doubt_bytes,grace_state,source_command,raw_output_path,files_used,files_quota,files_limit,
                    files_in_doubt,notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (root, observed, subject_kind, name, int(row["blockUsage"])*1024,
                 int(row["blockQuota"])*1024 if int(row["blockQuota"]) else None,
                 int(row["blockLimit"])*1024 if int(row["blockLimit"]) else None,
                 int(row["blockInDoubt"])*1024, row["blockGrace"], command, source_path,
                 int(row["filesUsage"]), int(row["filesQuota"]) if int(row["filesQuota"]) else None,
                 int(row["filesLimit"]) if int(row["filesLimit"]) else None, int(row["filesInDoubt"]),
                 ("Time-stamped user-fileset quota total; it is not localized to /data/pclancy3/yi or /scratch4/pclancy3/yi." if subject_kind == "USER" else
                  "Time-stamped group-level quota total; it spans more than the scanned user tree and is not allocated to this user's paths.")),
            )
            inserted += 1
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    with sqlite3.connect(DB) as check:
        print("source_key=" + source_key)
        print("sha256=" + h)
        print("inserted_observations=" + str(inserted))
        print("integrity_check=" + check.execute("PRAGMA integrity_check").fetchone()[0])
        print("foreign_key_violations=" + str(len(check.execute("PRAGMA foreign_key_check").fetchall())))
        for item in check.execute("SELECT root_id,observed_utc,quota_subject_kind,quota_subject,used_bytes,hard_limit_bytes,in_doubt_bytes,files_used,grace_state FROM quota_observation ORDER BY quota_observation_id"):
            print("\t".join(str(x) if x is not None else "NULL" for x in item))


if __name__ == "__main__":
    main()
