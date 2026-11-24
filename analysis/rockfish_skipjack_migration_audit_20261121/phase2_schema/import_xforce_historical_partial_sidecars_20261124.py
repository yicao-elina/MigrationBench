#!/usr/bin/env python3
"""Prepare an idempotent, review-first catalog import for X-FORCE historical sidecars.

The default mode is read-only preflight. Pass --apply only after reviewing the
remote manifest/content checks and the row-count summary. This importer creates
no copies and makes no claims about an original generator, scientific validity,
or manuscript selection.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

AUDIT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = AUDIT_ROOT / "provenance_catalog.sqlite"
REMOTE_HOST_DEFAULT = "rockfish"
REMOTE_REPO = "/scratch16/pclancy3/yi/X-FORCE"
MANIFEST_REL = "xforce/results/provenance/historical_partial_sidecars_20261124.tsv"
MANIFEST_ABS = REMOTE_REPO + "/" + MANIFEST_REL
EXPECTED_MANIFEST_SHA256 = "c17e8bd5d3c8aa6465371ef532e42c2128792aa76c86a5efd285680536b8d402"
EXPECTED_MANIFEST_BYTES = 313301
EXPECTED_MANIFEST_GIT_BLOB = "78a8f593b01210eb69134f17e1cedcb148915cbb"
EXPECTED_ROWS = 680
CANDIDATE_COMMIT = "7e466f0e291603323725ae6f832c38534f95cfff"
DECLARED_SOURCE_COMMIT = "df3781985da6447a24b19f74c3fc416ecb19eb18"
DECLARED_SOURCE_BRANCH = "feat/dualx-ground-truth-experiments"
PROJECT_ID = "DUALX"
STORAGE_ROOT_ID = "RF-SCRATCH16-YI"
REPOSITORY_LOCAL_PATH = REMOTE_REPO
RUN_ID = "DUALX-XFORCE-HISTORICAL-PARTIAL-SIDECARS-REVIEW-DRAFT-20261124"
REGISTRY_SOURCE_KEY = "DUALX-XFORCE-HISTORICAL-PARTIAL-SIDECARS-20261124"
MANIFEST_ARTIFACT_KEY = "DUALX-XFORCE-HISTORICAL-SIDECAR:20261124:MANIFEST"

EXPECTED_COLUMNS = [
    "artifact_path", "sidecar_path", "source_branch", "source_commit",
    "git_blob_oid", "byte_size", "sha256",
]

# Executed only on the remote host for a read-only manifest and Git/content audit.
# It returns per-file metadata for the importer; it does not write to the repo.
REMOTE_VERIFY_PY = r'''import csv,hashlib,json,os,re,subprocess,sys
from pathlib import Path
REPO=Path("/scratch16/pclancy3/yi/X-FORCE")
MAN_REL="xforce/results/provenance/historical_partial_sidecars_20261124.tsv"
MAN_PATH=REPO/MAN_REL
EXPECTED_MANIFEST_SHA="c17e8bd5d3c8aa6465371ef532e42c2128792aa76c86a5efd285680536b8d402"
EXPECTED_ROWS=680
CANDIDATE_COMMIT="7e466f0e291603323725ae6f832c38534f95cfff"
SOURCE_COMMIT="df3781985da6447a24b19f74c3fc416ecb19eb18"
SOURCE_BRANCH="feat/dualx-ground-truth-experiments"
def sha_file(p):
 h=hashlib.sha256()
 with open(str(p),"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def run_git(git,args,input=None):
 p=subprocess.run([git,"-C",str(REPO)]+args,input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if p.returncode:raise RuntimeError(p.stderr.decode("utf-8","replace"))
 return p.stdout
def tree(git,commit):
 raw=run_git(git,["ls-tree","-r","-z","--full-tree",commit,"--","xforce/results/"])
 result={}
 for rec in raw.split(b"\0"):
  if not rec:continue
  meta,path=rec.split(b"\t",1);parts=meta.decode("ascii").split()
  if len(parts)==3 and parts[1]=="blob":result[path.decode("utf-8")]=parts[2]
 return result
def object_sizes(git,oids):
 unique=list(dict.fromkeys(oids))
 data=("\n".join(unique)+"\n").encode("ascii")
 raw=run_git(git,["cat-file","--batch-check=%(objectname) %(objecttype) %(objectsize)"],data)
 out={}
 for line in raw.decode("ascii").splitlines():
  oid,kind,size=line.split()
  if kind!="blob":raise RuntimeError("manifest path is not a Git blob: "+oid)
  out[oid]=int(size)
 return out
with MAN_PATH.open(newline="",encoding="utf-8") as f:
 reader=csv.DictReader(f,delimiter="\t");rows=list(reader);columns=reader.fieldnames
if columns != ["artifact_path","sidecar_path","source_branch","source_commit","git_blob_oid","byte_size","sha256"]:
 raise RuntimeError("unexpected sidecar manifest header: "+repr(columns))
raw_manifest=MAN_PATH.read_bytes();manifest_sha=hashlib.sha256(raw_manifest).hexdigest()
if manifest_sha!=EXPECTED_MANIFEST_SHA or len(rows)!=EXPECTED_ROWS:raise RuntimeError("manifest hash or row count changed")
if len(set(r["artifact_path"] for r in rows))!=EXPECTED_ROWS or len(set(r["sidecar_path"] for r in rows))!=EXPECTED_ROWS:
 raise RuntimeError("artifact or sidecar paths are not unique")
for r in rows:
 if r["source_branch"]!=SOURCE_BRANCH or r["source_commit"]!=SOURCE_COMMIT:raise RuntimeError("declared origin branch/commit changed")
 if not re.fullmatch(r"[0-9a-f]{40}",r["git_blob_oid"]):raise RuntimeError("invalid Git blob OID")
 if not re.fullmatch(r"[0-9a-f]{64}",r["sha256"]):raise RuntimeError("invalid SHA-256")
 if int(r["byte_size"])<0:raise RuntimeError("invalid byte size")
 for k in ("artifact_path","sidecar_path"):
  p=Path(r[k])
  if p.is_absolute() or ".." in p.parts or not r[k].startswith("xforce/results/"):
   raise RuntimeError("unsafe or unexpected repository-relative path: "+r[k])
# Git can be available through Rockfish's login environment without being on the
# non-login SSH PATH. Resolve it once, then use the absolute executable.
which=subprocess.run(["bash","-lc","command -v git"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
git=which.stdout.decode().strip()
if which.returncode or not git:raise RuntimeError("could not locate Git in Rockfish login PATH")
head=run_git(git,["rev-parse","HEAD"]).decode().strip()
ancestor=subprocess.run([git,"-C",str(REPO),"merge-base","--is-ancestor",CANDIDATE_COMMIT,"HEAD"],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if ancestor.returncode:raise RuntimeError("pinned candidate commit is not an ancestor of the current checkout HEAD")
branch=run_git(git,["branch","--show-current"]).decode().strip()
commit_date=run_git(git,["show","-s","--format=%cI",CANDIDATE_COMMIT]).decode().strip()
candidate_tree=tree(git,CANDIDATE_COMMIT)
source_tree=tree(git,SOURCE_COMMIT)
manifest_oid=candidate_tree.get(MAN_REL)
if not manifest_oid:raise RuntimeError("manifest is not in pinned candidate Git tree")
all_oids=list(candidate_tree.values())+list(source_tree.values())
sizes=object_sizes(git,all_oids)
if sizes.get(manifest_oid)!=len(raw_manifest):raise RuntimeError("committed manifest Git size mismatch")
verified=[]
for r in rows:
 ap=r["artifact_path"];sp=r["sidecar_path"]
 source_oid=source_tree.get(ap);candidate_artifact_oid=candidate_tree.get(ap);sidecar_oid=candidate_tree.get(sp)
 if source_oid!=r["git_blob_oid"] or candidate_artifact_oid!=r["git_blob_oid"]:
  raise RuntimeError("manifest Git blob differs from source/candidate tree: "+ap)
 artifact_size=int(r["byte_size"])
 if sizes.get(source_oid)!=artifact_size or sizes.get(candidate_artifact_oid)!=artifact_size:
  raise RuntimeError("artifact Git byte size mismatch: "+ap)
 artifact_file=REPO/ap
 sidecar_file=REPO/sp
 if not artifact_file.is_file() or not sidecar_file.is_file():raise RuntimeError("current checkout file is missing: "+ap+" or "+sp)
 if artifact_file.stat().st_size!=artifact_size or sha_file(artifact_file)!=r["sha256"]:
  raise RuntimeError("working-tree artifact identity mismatch: "+ap)
 if not sidecar_oid:raise RuntimeError("sidecar is absent from candidate Git tree: "+sp)
 sidecar_size=sizes[sidecar_oid]
 if sidecar_file.stat().st_size!=sidecar_size:raise RuntimeError("sidecar Git byte size mismatch: "+sp)
 sidecar_sha=sha_file(sidecar_file)
 try:data=json.loads(sidecar_file.read_text(encoding="utf-8"))
 except Exception as e:raise RuntimeError("invalid provenance JSON "+sp+": "+str(e))
 q=data.get("quality_checks",{})
 if data.get("artifact_path")!=ap or data.get("source_branch")!=SOURCE_BRANCH or data.get("git_commit")!=SOURCE_COMMIT:
  raise RuntimeError("sidecar manifest metadata mismatch: "+sp)
 if data.get("provenance_status")!="PROVENANCE_PARTIAL" or data.get("generated_by_script") is not None:
  raise RuntimeError("sidecar partial status/generator field differs from policy: "+sp)
 if q.get("byte_size")!=artifact_size or q.get("sha256")!=r["sha256"] or q.get("git_blob_oid")!=r["git_blob_oid"]:
  raise RuntimeError("sidecar quality_checks identity mismatch: "+sp)
 if not data.get("lineage_gap") or "exact Git blob identity only" not in q.get("verification",""):
  raise RuntimeError("sidecar lacks explicit lineage limitation: "+sp)
 verified.append({
  "artifact_path":ap,"sidecar_path":sp,"source_branch":r["source_branch"],
  "source_commit":r["source_commit"],"git_blob_oid":r["git_blob_oid"],
  "byte_size":artifact_size,"sha256":r["sha256"],
  "sidecar_git_blob_oid":sidecar_oid,"sidecar_byte_size":sidecar_size,
  "sidecar_sha256":sidecar_sha,
 })
result={"manifest_sha256":manifest_sha,"manifest_size":len(raw_manifest),"manifest_git_blob_oid":manifest_oid,
 "candidate_commit":CANDIDATE_COMMIT,"candidate_is_ancestor_of_head":True,"candidate_branch":branch,"candidate_commit_date":commit_date,
 "head":head,"source_commit":SOURCE_COMMIT,"source_branch":SOURCE_BRANCH,"rows":verified}
sys.stdout.write(json.dumps(result,separators=(",",":")))
'''


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def run_capture(command, *, input_bytes=None):
    result = subprocess.run(
        command,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "command failed (exit " + str(result.returncode) + "): "
            + result.stderr.decode("utf-8", "replace").strip()
        )
    return result.stdout


def fetch_manifest(remote_host):
    raw = run_capture(["ssh", "-o", "BatchMode=yes", remote_host, "cat", MANIFEST_ABS])
    if len(raw) != EXPECTED_MANIFEST_BYTES or sha256_bytes(raw) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("remote manifest size/SHA-256 differs from the committed input")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError("manifest is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if reader.fieldnames != EXPECTED_COLUMNS:
        raise RuntimeError("unexpected manifest columns: " + repr(reader.fieldnames))
    rows = list(reader)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError("expected 680 manifest rows; found " + str(len(rows)))
    if len({r["artifact_path"] for r in rows}) != EXPECTED_ROWS:
        raise RuntimeError("artifact_path values are not unique")
    if len({r["sidecar_path"] for r in rows}) != EXPECTED_ROWS:
        raise RuntimeError("sidecar_path values are not unique")
    for row in rows:
        if row["source_branch"] != DECLARED_SOURCE_BRANCH:
            raise RuntimeError("unexpected manifest source branch in " + row["artifact_path"])
        if row["source_commit"] != DECLARED_SOURCE_COMMIT:
            raise RuntimeError("unexpected manifest source commit in " + row["artifact_path"])
        if not re.fullmatch(r"[0-9a-f]{40}", row["git_blob_oid"]):
            raise RuntimeError("invalid Git blob OID for " + row["artifact_path"])
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise RuntimeError("invalid SHA-256 for " + row["artifact_path"])
        if int(row["byte_size"]) < 0:
            raise RuntimeError("invalid byte size for " + row["artifact_path"])
        for key in ("artifact_path", "sidecar_path"):
            path = PurePosixPath(row[key])
            if path.is_absolute() or ".." in path.parts or not row[key].startswith("xforce/results/"):
                raise RuntimeError("unsafe or unexpected repository path: " + row[key])
    return raw, rows


def verify_remote(remote_host):
    raw = run_capture(
        ["ssh", "-o", "BatchMode=yes", remote_host, "python3", "-"],
        input_bytes=REMOTE_VERIFY_PY.encode("utf-8"),
    )
    try:
        result = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError("remote verifier returned invalid JSON: " + repr(exc)) from exc
    if result.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("remote verifier saw a different manifest SHA-256")
    if result.get("manifest_size") != EXPECTED_MANIFEST_BYTES:
        raise RuntimeError("remote verifier saw a different manifest size")
    if result.get("manifest_git_blob_oid") != EXPECTED_MANIFEST_GIT_BLOB:
        raise RuntimeError("manifest Git blob OID differs from the reviewed commit")
    if result.get("candidate_commit") != CANDIDATE_COMMIT or result.get("candidate_is_ancestor_of_head") is not True:
        raise RuntimeError("Rockfish checkout no longer descends from the pinned reviewed candidate commit")
    if result.get("source_commit") != DECLARED_SOURCE_COMMIT:
        raise RuntimeError("remote verifier saw a different manifest-declared source commit")
    if len(result.get("rows", [])) != EXPECTED_ROWS:
        raise RuntimeError("remote verifier did not return 680 validated file pairs")
    return result


def artifact_keys(row):
    return (
        "DUALX-XFORCE-HISTORICAL-SIDECAR:20261124:ARTIFACT:" + row["artifact_path"],
        "DUALX-XFORCE-HISTORICAL-SIDECAR:20261124:SIDECAR:" + row["sidecar_path"],
    )


def make_records(rows, remote_result):
    verified = {r["artifact_path"]: r for r in remote_result["rows"]}
    records = []
    for row in rows:
        v = verified.get(row["artifact_path"])
        if v is None:
            raise RuntimeError("remote verification omitted " + row["artifact_path"])
        if any(v[k] != row[k] for k in ("artifact_path", "sidecar_path", "source_branch", "source_commit", "git_blob_oid", "sha256")):
            raise RuntimeError("local/remote manifest row disagreement: " + row["artifact_path"])
        main_key, sidecar_key = artifact_keys(row)
        extension = PurePosixPath(row["artifact_path"]).suffix.lower() or "[no extension]"
        common = (
            "manifest_source_branch=" + row["source_branch"]
            + "; source_commit=" + row["source_commit"]
            + "; candidate_tree_commit=" + CANDIDATE_COMMIT
            + "; manifest_sha256=" + EXPECTED_MANIFEST_SHA256
            + "; provenance_status=PROVENANCE_PARTIAL"
        )
        main_notes = (
            "artifact_path=" + row["artifact_path"]
            + "; sidecar_path=" + row["sidecar_path"]
            + "; git_blob_oid=" + row["git_blob_oid"]
            + "; byte_size=" + str(int(row["byte_size"]))
            + "; sha256=" + row["sha256"]
            + "; " + common
            + "; lineage_gap=original generator, command, inputs, and environment not recovered"
            + "; no scientific validation or manuscript selection inferred"
        )
        sidecar_notes = (
            "artifact_path=" + row["artifact_path"]
            + "; sidecar_path=" + row["sidecar_path"]
            + "; git_blob_oid=" + v["sidecar_git_blob_oid"]
            + "; byte_size=" + str(v["sidecar_byte_size"])
            + "; sha256=" + v["sidecar_sha256"]
            + "; manifest_source_commit=" + row["source_commit"]
            + "; candidate_tree_commit=" + CANDIDATE_COMMIT
            + "; provenance_status=PROVENANCE_PARTIAL"
            + "; this JSON records partial lineage, not an artifact generator or scientific validation"
        )
        records.append({
            "key": main_key, "kind": "RESULT",
            "role": "historical result artifact; partial provenance; format " + extension,
            "sha": row["sha256"], "size": int(row["byte_size"]),
            "retention": "PERMANENT_SOURCE", "notes": main_notes,
            "relative_path": "X-FORCE/" + row["artifact_path"],
            "package_role": "historical_result_artifact", "direction": "INPUT",
            "registry_entity_type": "historical_sidecar_manifest_row",
            "registry_entity_key": row["artifact_path"],
            "registry_relationship": "MANIFEST_DESCRIBES_ARTIFACT",
            "registry_notes": "git_blob_oid=" + row["git_blob_oid"]
                + "; source_commit=" + row["source_commit"]
                + "; candidate_tree_commit=" + CANDIDATE_COMMIT
                + "; byte_size=" + str(int(row["byte_size"]))
                + "; sha256=" + row["sha256"],
        })
        records.append({
            "key": sidecar_key, "kind": "PROVENANCE_FILE",
            "role": "historical partial-provenance sidecar JSON",
            "sha": v["sidecar_sha256"], "size": int(v["sidecar_byte_size"]),
            "retention": "PERMANENT_SOURCE", "notes": sidecar_notes,
            "relative_path": "X-FORCE/" + row["sidecar_path"],
            "package_role": "partial_provenance_sidecar", "direction": "PROVENANCE",
            "registry_entity_type": "historical_sidecar_manifest_row",
            "registry_entity_key": row["artifact_path"],
            "registry_relationship": "MANIFEST_LINKS_PROVENANCE_SIDECAR",
            "registry_notes": "git_blob_oid=" + v["sidecar_git_blob_oid"]
                + "; manifest_source_commit=" + row["source_commit"]
                + "; candidate_tree_commit=" + CANDIDATE_COMMIT
                + "; byte_size=" + str(v["sidecar_byte_size"])
                + "; sha256=" + v["sidecar_sha256"],
        })
    manifest_record = {
        "key": MANIFEST_ARTIFACT_KEY, "kind": "PROVENANCE_MANIFEST",
        "role": "manifest for historical partial-provenance result artifacts and sidecars",
        "sha": EXPECTED_MANIFEST_SHA256, "size": EXPECTED_MANIFEST_BYTES,
        "retention": "PERMANENT_SOURCE",
        "notes": "source_path=" + MANIFEST_ABS
            + "; git_blob_oid=" + EXPECTED_MANIFEST_GIT_BLOB
            + "; candidate_tree_commit=" + CANDIDATE_COMMIT
            + "; row_count=680; each row links one result asset to one provenance sidecar"
            + "; provenance_status=PROVENANCE_PARTIAL; no archive copy or manuscript selection inferred",
        "relative_path": "X-FORCE/" + MANIFEST_REL,
        "package_role": "source_manifest", "direction": "PROVENANCE",
        "registry_entity_type": "registry_manifest",
        "registry_entity_key": REGISTRY_SOURCE_KEY,
        "registry_relationship": "REGISTRY_SOURCE_FILE",
        "registry_notes": "git_blob_oid=" + EXPECTED_MANIFEST_GIT_BLOB
            + "; candidate_tree_commit=" + CANDIDATE_COMMIT
            + "; sha256=" + EXPECTED_MANIFEST_SHA256
            + "; byte_size=" + str(EXPECTED_MANIFEST_BYTES),
    }
    records.append(manifest_record)
    return records


def chunks(values, size=300):
    for i in range(0, len(values), size):
        yield values[i:i + size]


def count_existing(db_path, records):
    uri = "file:" + str(db_path) + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        paths = [r["relative_path"] for r in records]
        keys = [r["key"] for r in records]
        path_count = 0
        key_count = 0
        for group in chunks(paths):
            marks = ",".join("?" for _ in group)
            path_count += conn.execute(
                "SELECT count(*) FROM storage_path WHERE root_id=? AND relative_path IN (" + marks + ")",
                [STORAGE_ROOT_ID] + group,
            ).fetchone()[0]
        for group in chunks(keys):
            marks = ",".join("?" for _ in group)
            key_count += conn.execute(
                "SELECT count(*) FROM artifact WHERE artifact_key IN (" + marks + ")", group
            ).fetchone()[0]
        project = conn.execute("SELECT count(*) FROM project WHERE project_id=?", (PROJECT_ID,)).fetchone()[0]
        root = conn.execute("SELECT count(*) FROM storage_root WHERE root_id=?", (STORAGE_ROOT_ID,)).fetchone()[0]
        repo = conn.execute("SELECT count(*) FROM git_repository WHERE local_path=?", (REPOSITORY_LOCAL_PATH,)).fetchone()[0]
        source = conn.execute("SELECT count(*) FROM registry_source WHERE source_key=?", (REGISTRY_SOURCE_KEY,)).fetchone()[0]
        package = conn.execute("SELECT count(*) FROM workflow_run WHERE run_id=?", (RUN_ID,)).fetchone()[0]
        return {
            "existing_exact_storage_paths": path_count,
            "existing_artifact_keys": key_count,
            "project_rows": project,
            "root_rows": root,
            "repository_rows": repo,
            "registry_source_key_rows": source,
            "package_run_id_rows": package,
        }
    finally:
        conn.close()


def ensure_storage_path(conn, relpath, now):
    conn.execute(
        "INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",
        (STORAGE_ROOT_ID, relpath, now),
    )
    row = conn.execute(
        "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",
        (STORAGE_ROOT_ID, relpath),
    ).fetchone()
    if row is None:
        raise RuntimeError("storage path insert did not resolve: " + relpath)
    return row[0]


def ensure_artifact(conn, record):
    row = conn.execute("SELECT * FROM artifact WHERE artifact_key=?", (record["key"],)).fetchone()
    expected = (
        PROJECT_ID, record["kind"], record["role"], "UNKNOWN", record["sha"],
        record["size"], "PROVENANCE_PARTIAL", record["retention"], record["notes"],
    )
    if row is not None:
        actual = tuple(row[k] for k in (
            "project_id", "artifact_kind", "semantic_role", "owner_status", "sha256",
            "byte_size", "scientific_status", "retention_class", "notes",
        ))
        if actual != expected:
            raise RuntimeError("existing artifact key conflicts with verified import: " + record["key"])
        return row["artifact_id"], False
    cur = conn.execute(
        """INSERT INTO artifact
           (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,
            scientific_status,retention_class,notes)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (PROJECT_ID, record["key"], record["kind"], record["role"], "UNKNOWN",
         record["sha"], record["size"], "PROVENANCE_PARTIAL", record["retention"], record["notes"]),
    )
    return cur.lastrowid, True


def ensure_location(conn, artifact_id, path_id, now):
    row = conn.execute(
        "SELECT availability FROM artifact_location WHERE artifact_id=? AND path_id=?",
        (artifact_id, path_id),
    ).fetchone()
    if row is not None:
        if row[0] != "PRESENT":
            raise RuntimeError("existing artifact location is not PRESENT; refusing to overwrite")
        return False
    conn.execute(
        """INSERT INTO artifact_location
           (artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
           VALUES(?,?,'PRESENT',?,?)""",
        (artifact_id, path_id, now, now),
    )
    return True


def ensure_run(conn, now):
    fields = (
        "project_id", "campaign_id", "run_label", "workflow_type", "method", "system_key",
        "source_root_id", "source_relative_path", "scheduler_state", "scientific_status", "notes",
    )
    values = (
        PROJECT_ID, None,
        "Historical X-FORCE partial-provenance sidecars — REVIEW/DRAFT package",
        "RESULT_PACKAGE_REVIEW", "historical provenance indexing", None,
        STORAGE_ROOT_ID, "X-FORCE/" + MANIFEST_REL, "UNKNOWN", "PROVENANCE_PARTIAL",
        "Review/draft index of 680 historical result files and 680 companion provenance JSONs. "
        "This is not a scientific compute run; no original generator, validation, archive copy, "
        "or manuscript/figure selection is asserted. Artifact source identity follows the manifest-declared "
        "branch/commit and exact Git blob, SHA-256, and size; candidate-tree commit is bound separately.",
    )
    row = conn.execute("SELECT * FROM workflow_run WHERE run_id=?", (RUN_ID,)).fetchone()
    if row is not None:
        actual = tuple(row[f] for f in fields)
        if actual != values:
            raise RuntimeError("existing review/draft package run id has conflicting metadata")
        return False
    conn.execute(
        """INSERT INTO workflow_run
           (run_id,project_id,campaign_id,run_label,workflow_type,method,system_key,
            source_root_id,source_relative_path,scheduler_state,scientific_status,notes)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
        (RUN_ID,) + values,
    )
    return True


def ensure_run_artifact(conn, artifact_id, record):
    role = record["package_role"]
    desired = (record["direction"], 0, "Member of REVIEW/DRAFT sidecar package; no generation or validation claim.")
    row = conn.execute(
        "SELECT direction,required_for_reproduction,notes FROM run_artifact WHERE run_id=? AND artifact_id=? AND role=?",
        (RUN_ID, artifact_id, role),
    ).fetchone()
    if row is not None:
        if tuple(row) != desired:
            raise RuntimeError("existing run-artifact link conflicts for " + record["key"])
        return False
    conn.execute(
        """INSERT INTO run_artifact
           (run_id,artifact_id,role,direction,required_for_reproduction,notes)
           VALUES(?,?,?,?,?,?)""",
        (RUN_ID, artifact_id, role, record["direction"], 0, desired[2]),
    )
    return True


def ensure_registry_source(conn, now, as_of):
    path = MANIFEST_ABS
    kind = "XFORCE_HISTORICAL_PARTIAL_SIDECAR_MANIFEST"
    authority = (
        "Committed X-FORCE manifest; artifact hashes/sizes/Git blobs verified against its declared source commit "
        "and candidate tree. Identity verification only; no scientific validation or manuscript selection."
    )
    notes = (
        "rows=680; candidate_commit=" + CANDIDATE_COMMIT
        + "; declared_source_branch=" + DECLARED_SOURCE_BRANCH
        + "; declared_source_commit=" + DECLARED_SOURCE_COMMIT
        + "; companion provenance JSONs are registered as separate PROVENANCE_PARTIAL files"
        + "; no archive copy is created or inferred"
    )
    row = conn.execute("SELECT * FROM registry_source WHERE source_key=?", (REGISTRY_SOURCE_KEY,)).fetchone()
    expected = (path, kind, EXPECTED_MANIFEST_SHA256, 1, authority, notes)
    if row is not None:
        actual = (row["source_path"], row["registry_kind"], row["sha256"], row["immutable_snapshot"], row["authority_note"], row["notes"])
        if actual != expected:
            raise RuntimeError("existing registry source key conflicts with manifest")
        return row["source_id"], False
    cur = conn.execute(
        """INSERT INTO registry_source
           (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,
            immutable_snapshot,authority_note,notes)
           VALUES(?,?,?,?,?,?,1,?,?)""",
        (REGISTRY_SOURCE_KEY, path, kind, EXPECTED_MANIFEST_SHA256, now, as_of, authority, notes),
    )
    return cur.lastrowid, True


def ensure_registry_ref(conn, source_id, entity_type, entity_key, catalog_key, relationship, notes):
    row = conn.execute(
        """SELECT notes FROM registry_entity_ref WHERE source_id=? AND entity_type=? AND entity_key=?
           AND catalog_entity_type='artifact' AND catalog_entity_key=? AND relationship=?""",
        (source_id, entity_type, entity_key, catalog_key, relationship),
    ).fetchone()
    if row is not None:
        if row[0] != notes:
            raise RuntimeError("existing manifest entity reference has different provenance notes")
        return False
    conn.execute(
        """INSERT INTO registry_entity_ref
           (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
           VALUES(?,?,?,'artifact',?,?,?)""",
        (source_id, entity_type, entity_key, catalog_key, relationship, notes),
    )
    return True


def ensure_git_binding(conn, repo_id, manifest_artifact_id):
    role = "CANDIDATE_TREE_CONTENT"
    row = conn.execute(
        """SELECT evidence_artifact_id,provenance_status FROM run_git_binding
           WHERE run_id=? AND repository_id=? AND commit_sha=? AND binding_role=?""",
        (RUN_ID, repo_id, CANDIDATE_COMMIT, role),
    ).fetchone()
    expected = (manifest_artifact_id, "PROVENANCE_PARTIAL")
    if row is not None:
        if tuple(row) != expected:
            raise RuntimeError("existing candidate commit binding conflicts")
        return False
    conn.execute(
        """INSERT INTO run_git_binding
           (run_id,repository_id,commit_sha,binding_role,evidence_artifact_id,provenance_status)
           VALUES(?,?,?,?,?,'PROVENANCE_PARTIAL')""",
        (RUN_ID, repo_id, CANDIDATE_COMMIT, role, manifest_artifact_id),
    )
    return True


def apply_import(db_path, records, remote_result):
    conn = sqlite3.connect(str(db_path), timeout=120)
    conn.row_factory = sqlite3.Row
    counts = {"artifacts_inserted": 0, "paths_inserted": 0, "locations_inserted": 0,
              "run_artifact_links_inserted": 0, "registry_refs_inserted": 0}
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT count(*) FROM project WHERE project_id=?", (PROJECT_ID,)).fetchone()[0] != 1:
            raise RuntimeError("required DUALX project row is missing")
        if conn.execute("SELECT count(*) FROM storage_root WHERE root_id=?", (STORAGE_ROOT_ID,)).fetchone()[0] != 1:
            raise RuntimeError("required RF-SCRATCH16-YI storage root is missing")
        repo = conn.execute("SELECT repository_id FROM git_repository WHERE local_path=?", (REPOSITORY_LOCAL_PATH,)).fetchone()
        if repo is None:
            raise RuntimeError("X-FORCE Git repository row is missing")
        repo_id = repo[0]
        now = utc_now()
        as_of = remote_result["candidate_commit_date"]
        # Register each file by path, never coalescing equal content hashes.
        artifact_ids = {}
        for record in records:
            artifact_id, inserted = ensure_artifact(conn, record)
            counts["artifacts_inserted"] += int(inserted)
            artifact_ids[record["key"]] = artifact_id
            path_before = conn.execute(
                "SELECT 1 FROM storage_path WHERE root_id=? AND relative_path=?",
                (STORAGE_ROOT_ID, record["relative_path"]),
            ).fetchone()
            path_id = ensure_storage_path(conn, record["relative_path"], now)
            counts["paths_inserted"] += int(path_before is None)
            counts["locations_inserted"] += int(ensure_location(conn, artifact_id, path_id, now))
        ensure_run(conn, now)
        source_id, _ = ensure_registry_source(conn, now, as_of)
        manifest_id = artifact_ids[MANIFEST_ARTIFACT_KEY]
        counts["git_binding_inserted"] = int(ensure_git_binding(conn, repo_id, manifest_id))
        for record in records:
            artifact_id = artifact_ids[record["key"]]
            counts["run_artifact_links_inserted"] += int(ensure_run_artifact(conn, artifact_id, record))
            if record["key"] == MANIFEST_ARTIFACT_KEY:
                continue
            counts["registry_refs_inserted"] += int(ensure_registry_ref(
                conn, source_id, record["registry_entity_type"], record["registry_entity_key"],
                record["key"], record["registry_relationship"], record["registry_notes"],
            ))
        # One manifest-to-package file reference, in addition to two references per data row.
        manifest_record = next(r for r in records if r["key"] == MANIFEST_ARTIFACT_KEY)
        counts["registry_refs_inserted"] += int(ensure_registry_ref(
            conn, source_id, manifest_record["registry_entity_type"], manifest_record["registry_entity_key"],
            MANIFEST_ARTIFACT_KEY, manifest_record["registry_relationship"], manifest_record["registry_notes"],
        ))
        conn.commit()
        return counts
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="shared provenance catalog path")
    parser.add_argument("--remote-host", default=REMOTE_HOST_DEFAULT, help="SSH host alias for Rockfish")
    parser.add_argument("--apply", action="store_true", help="write the validated rows; default is read-only preflight")
    args = parser.parse_args()
    if not args.db.is_file():
        raise RuntimeError("catalog database not found: " + str(args.db))
    _, rows = fetch_manifest(args.remote_host)
    remote_result = verify_remote(args.remote_host)
    records = make_records(rows, remote_result)
    preflight = count_existing(args.db, records)
    if preflight["project_rows"] != 1 or preflight["root_rows"] != 1 or preflight["repository_rows"] != 1:
        raise RuntimeError("catalog prerequisite rows missing or ambiguous: " + json.dumps(preflight, sort_keys=True))
    report = {
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "candidate_commit": CANDIDATE_COMMIT,
        "declared_source_commit": DECLARED_SOURCE_COMMIT,
        "artifact_rows": 680,
        "sidecar_json_rows": 680,
        "manifest_artifact_rows": 1,
        "new_artifact_keys_expected": len(records),
        "new_storage_paths_expected": len(records),
        "new_artifact_locations_expected": len(records),
        "new_run_artifact_links_expected": len(records),
        "new_registry_entity_refs_expected": len(records),
        "new_registry_source_rows_expected": 1,
        "new_review_draft_package_run_rows_expected": 1,
        "new_run_git_binding_rows_expected": 1,
        "unique_result_sha256": len(set(r["sha256"] for r in rows)),
        "preflight": preflight,
        "apply_requested": args.apply,
    }
    if not args.apply:
        print(json.dumps(report, indent=2, sort_keys=True))
        print("Read-only preflight only; no SQLite changes made. Pass --apply after review to import.")
        return 0
    counts = apply_import(args.db, records, remote_result)
    report["inserted_rows"] = counts
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("IMPORT ABORTED:", str(exc), file=sys.stderr)
        sys.exit(2)
