#!/usr/bin/env python3
"""Build a no-payload SQLite index for the cross-element checkpoint audit."""
from __future__ import print_function
import argparse
import csv
import hashlib
import json
import os
import sqlite3
import stat
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BRIDGE = HERE.parent
COLUMNS = ["model_dir", "seed", "job_id", "file_class", "epoch", "path", "size_bytes",
           "allocated_bytes", "mtime_utc", "inode", "hardlink_count", "disposition", "sha256", "sha256_status"]
FILES = ["CROSS_ELEMENT_CHECKPOINT_FILE_MAP_20261125.tsv",
         "CROSS_ELEMENT_CHECKPOINT_RETENTION_AUDIT_20261125.md",
         "FINAL_TRAINING_METADATA_SHA256SUMS_20261125.txt",
         "SLURM_STATE_20261125_1934UTC.txt"]
EXPECTED_HASHES = {
    "CROSS_ELEMENT_CHECKPOINT_FILE_MAP_20261125.tsv": "3fac5002398bd4b196377cb4e7fc265e3af063143c711628d9cd218ff8b4f9a5",
    "CROSS_ELEMENT_CHECKPOINT_RETENTION_AUDIT_20261125.md": "07eb0de7f711fa6d91d890891dec044be11e590529008cfe1b9ca535c24e5df6",
    "FINAL_TRAINING_METADATA_SHA256SUMS_20261125.txt": "00a06ec8d670e20c282f6a9ff4196aa4a40e48989acd0b9ed31686e975c3a917",
    "SLURM_STATE_20261125_1934UTC.txt": "df1998d5deacd70409edf707c770f4dea9c943f00ec10a7f423ea832d5df1ad9",
}
RUNS = [
    {"training_run_key":"cross_element_seed123_job31584299", "seed":123, "job_id":"31584299", "scheduler_state":"COMPLETED", "exit_code":"0:0", "ended_utc":"2026-10-04T20:36:59"},
    {"training_run_key":"cross_element_seed456_job31585322_456", "seed":456, "job_id":"31585322_456", "scheduler_state":"COMPLETED", "exit_code":"0:0", "ended_utc":"2026-10-04T20:41:37"},
    {"training_run_key":"cross_element_seed789_job31585322_789", "seed":789, "job_id":"31585322_789", "scheduler_state":"COMPLETED", "exit_code":"0:0", "ended_utc":"2026-10-04T20:39:26"},
]
SNAPSHOT_UTC = "2026-11-25T19:34:21Z"


def sha256(path):
    h=hashlib.sha256()
    with open(str(path),"rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="directory containing the pinned audit bundle")
    parser.add_argument("--output", required=True, help="new SQLite path; refuses to replace existing files")
    parser.add_argument("--archive-path", help="stable deployment path recorded in the snapshot manifest")
    args=parser.parse_args()
    source=Path(args.source_dir).expanduser().resolve()
    output=Path(args.output).expanduser().resolve()
    archive_path=args.archive_path or str(output)
    if output.exists(): raise ValueError("refusing to replace existing output: {}".format(output))
    if not source.is_dir(): raise IOError("source directory is missing: {}".format(source))

    bundle=source/"BUNDLE_SHA256SUMS_20261125.txt"
    pinned={}
    for line in bundle.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        expected,name=line.split(None,1)
        name=name.strip().lstrip("*")
        pinned[name]=expected
    if set(pinned)!=set(FILES): raise ValueError("bundle checksum file list differs from expected source files")
    actual={}
    for name in FILES:
        path=source/name
        if not path.is_file(): raise IOError("missing source file: {}".format(path))
        digest=sha256(path)
        if digest.lower()!=EXPECTED_HASHES[name] or digest.lower()!=pinned[name].lower():
            raise ValueError("source SHA-256 mismatch for {}".format(name))
        actual[name]=digest

    map_path=source/FILES[0]
    with open(str(map_path),"r",encoding="utf-8",newline="") as f:
        reader=csv.DictReader(f,delimiter="\t")
        if reader.fieldnames!=COLUMNS: raise ValueError("inventory TSV columns mismatch")
        rows=list(reader)
    if len(rows)!=3027: raise ValueError("expected 3027 inventory rows, found {}".format(len(rows)))
    if len(set(r["path"] for r in rows))!=len(rows): raise ValueError("inventory path is not unique")
    class_counts=Counter(r["file_class"] for r in rows)
    disposition_counts=Counter(r["disposition"] for r in rows)
    hash_counts=Counter(r["sha256_status"] for r in rows)
    if class_counts["checkpoint"]!=3000: raise ValueError("expected 3000 checkpoint path rows")
    if disposition_counts["HOLD_PENDING_USER_POLICY"]!=2997: raise ValueError("expected 2997 held intermediate checkpoint rows")
    if disposition_counts["KEEP_DESIGNATED_FINAL"]!=3: raise ValueError("expected 3 designated final checkpoint rows")
    by_seed=defaultdict(list)
    for r in rows:
        by_seed[int(r["seed"])].append(r)
    if set(by_seed)!={123,456,789}: raise ValueError("unexpected seed set")
    for seed,seed_rows in by_seed.items():
        checkpoints=[r for r in seed_rows if r["file_class"]=="checkpoint"]
        epochs=sorted(int(r["epoch"]) for r in checkpoints if r["epoch"]!="")
        if len(epochs)!=1000 or epochs!=list(range(1000)):
            raise ValueError("seed {} checkpoint epochs are incomplete or duplicated".format(seed))
    state=(source/FILES[3]).read_text(encoding="utf-8")
    if "Snapshot UTC: {}".format(SNAPSHOT_UTC) not in state: raise ValueError("scheduler capture timestamp not found")
    for run in RUNS:
        marker="Job {} (seed {}): COMPLETED, ExitCode 0:0, ended {}.".format(run["job_id"],run["seed"],run["ended_utc"])
        if run["seed"]==123:
            pass
        elif run["seed"] in (456,789):
            marker="Array task {} (seed {}): COMPLETED, ExitCode 0:0, ended {}.".format(run["job_id"],run["seed"],run["ended_utc"])
        if marker not in state: raise ValueError("scheduler result not found: {}".format(marker))

    output.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(str(output))
    try:
        db.executescript((HERE/"schema.sql").read_text(encoding="utf-8"))
        slurm_sha=actual[FILES[3]]
        db.executemany("INSERT INTO training_run_snapshot VALUES(?,?,?,?,?,?,?,?)",[
            (r["training_run_key"],r["seed"],r["job_id"],r["scheduler_state"],r["exit_code"],r["ended_utc"],SNAPSHOT_UTC,slurm_sha) for r in RUNS])
        values=[]
        for r in rows:
            epoch=int(r["epoch"]) if r["epoch"]!="" else None
            values.append((r["path"],r["model_dir"],int(r["seed"]),r["job_id"],r["file_class"],epoch,
                           int(r["size_bytes"]),int(r["allocated_bytes"]),r["mtime_utc"],r["inode"],
                           int(r["hardlink_count"]),r["disposition"],r["sha256"],r["sha256_status"]))
        db.executemany("INSERT INTO artifact_inventory VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",values)
        logical=sum(int(r["size_bytes"]) for r in rows)
        allocated=sum(int(r["allocated_bytes"]) for r in rows)
        metadata={
          "snapshot_id":"cross_element_checkpoint_inventory_r1_20261125",
          "source_inventory_path":"/scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/scratch16_root_lstat_refresh_20261125/cross_element_checkpoint_retention_20261125/CROSS_ELEMENT_CHECKPOINT_FILE_MAP_20261125.tsv",
          "source_inventory_sha256":actual[FILES[0]],
          "retention_audit_sha256":actual[FILES[1]],
          "final_training_metadata_sums_sha256":actual[FILES[2]],
          "scheduler_state_path":"/scratch16/pclancy3/yi/MigrationBench_data_archive/storage_audits/scratch16_root_lstat_refresh_20261125/cross_element_checkpoint_retention_20261125/SLURM_STATE_20261125_1934UTC.txt",
          "scheduler_state_sha256":slurm_sha,
          "scheduler_snapshot_utc":SNAPSHOT_UTC,
          "artifact_rows":str(len(rows)),
          "file_class_counts_json":json.dumps(dict(sorted(class_counts.items())),sort_keys=True),
          "disposition_counts_json":json.dumps(dict(sorted(disposition_counts.items())),sort_keys=True),
          "sha256_status_counts_json":json.dumps(dict(sorted(hash_counts.items())),sort_keys=True),
          "logical_path_bytes_sum":str(logical),
          "allocated_path_bytes_sum":str(allocated),
          "intermediate_content_hash_policy":"2997 epochs 0-998 are NOT_COMPUTED; no payload hashes were performed",
          "identity_boundary":"path/size/inode metadata is not content identity; only nonblank source SHA values are exact hashes",
          "retention_boundary":"HOLD_PENDING_USER_POLICY is preserved as source status; no delete/move recommendation is encoded",
          "source_scope":"three user-owned cross-element training output roots under /scratch16/pclancy3/yi",
          "bundle_sha256_file_sha256":sha256(bundle)
        }
        db.executemany("INSERT INTO snapshot_metadata VALUES(?,?)",sorted(metadata.items()))
        db.commit()
        if db.execute("PRAGMA quick_check").fetchone()[0]!="ok": raise ValueError("new snapshot quick_check failed")
        if db.execute("PRAGMA foreign_key_check").fetchall(): raise ValueError("new snapshot has FK violations")
        count=db.execute("SELECT COUNT(*) FROM artifact_inventory").fetchone()[0]
        if count!=len(rows): raise ValueError("database row count mismatch")
        db.close()
    except Exception:
        db.close()
        try: output.unlink()
        except OSError: pass
        raise
    os.chmod(str(output),stat.S_IRUSR|stat.S_IWUSR)
    manifest={
      "snapshot":"Cross-element checkpoint metadata inventory; no payload bytes",
      "user_version":1,"database_path":archive_path,"bytes":output.stat().st_size,"sha256":sha256(output),
      "quick_check":"ok","foreign_key_violations":0,"artifact_rows":len(rows),
      "training_run_rows":len(RUNS),"file_class_counts":dict(sorted(class_counts.items())),
      "disposition_counts":dict(sorted(disposition_counts.items())),"sha256_status_counts":dict(sorted(hash_counts.items())),
      "logical_path_bytes_sum":logical,"allocated_path_bytes_sum":allocated,
      "source_hashes":actual,"scheduler_snapshot_utc":SNAPSHOT_UTC,
      "limitations":["No hashes were computed for 2,997 intermediate checkpoint payloads.","The scheduler record is a historical capture, not live queue state.","Retention remains HOLD_PENDING_USER_POLICY for the intermediate checkpoints."]
    }
    mp=output.with_suffix(output.suffix+".manifest.json")
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.chmod(str(mp),stat.S_IRUSR|stat.S_IWUSR)
    print(json.dumps(manifest,indent=2,sort_keys=True))

if __name__=="__main__":
    try: main()
    except (IOError,OSError,sqlite3.Error,ValueError,KeyError,json.JSONDecodeError) as e:
        sys.stderr.write("checkpoint inventory import failed: {}\n".format(e));sys.exit(2)
