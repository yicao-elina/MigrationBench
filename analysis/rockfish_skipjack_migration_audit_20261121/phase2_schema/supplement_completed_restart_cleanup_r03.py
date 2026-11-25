#!/usr/bin/env python3
"""Read-only preflight and transactional R03 provenance supplement.

The default invocation only reads the catalog and emits a JSON plan.  Use
--apply only after reviewing that plan.  This script never deletes remote
files and deliberately does not populate cleanup_item.deleted_utc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
REVIEW = AUDIT / "phase2_data/restart_coordinate_audit_20261123T1044Z"
SUPPORT = REVIEW / "restart_recipe_support_capture_20261123_r03.json"
STATUS = REVIEW / "restart_recipe_support_capture_20261123_r03.status.json"
CAPTURE_SCRIPT = REVIEW / "capture_restart_recipe_support_r03_20261123.py"
CROSSWALK = REVIEW / "run_coordinate_crosswalk_20261123T1129Z.tsv"
SACCT = REVIEW / "sacct_restart_ids_with_submitline_20261123T1114Z.psv"
PAIR_CAPTURE = AUDIT / "phase2_data/yi_mix_repro_review_20261123/two_neb_mix_pair_predelete_hashes_20261123.json"
BATCH = "RF-YI-QE-RESTART-CLEANUP-20261123"
CLUSTER = "Rockfish"
RECIPE_VERSION = "r03-sacct-captured-2026-11-23-v1"
ROOT_PREFIX = "RF-QE-RESTART-ROOT-"
JOB_PREFIX = "RF-ROCKFISH-QE-JOB-"
_ROOT_CACHE: list[dict] | None = None


class AuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def root_id(run_root: str) -> str:
    return ROOT_PREFIX + hashlib.sha256(run_root.encode()).hexdigest()[:20]


def job_run_id(job_id: str, workdir: str) -> str:
    suffix = hashlib.sha256((job_id + "\0" + workdir).encode()).hexdigest()[:12]
    return JOB_PREFIX + re.sub(r"[^A-Za-z0-9_-]", "_", job_id) + "-" + suffix


def normalize_scheduler_state(raw: str | None) -> str:
    """Map Slurm's annotated raw state to the catalog enum; raw state stays in its observation row."""
    token = (raw or "").strip().upper().split(maxsplit=1)[0] if (raw or "").strip() else ""
    token = token.rstrip("+")
    allowed = {"PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "UNKNOWN"}
    if token in allowed:
        return token
    if token in {"OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "PREEMPTED", "DEADLINE"}:
        return "FAILED"
    return "UNKNOWN"


def load_inputs() -> tuple[dict, dict, dict]:
    for p in (DB, SUPPORT, STATUS, CAPTURE_SCRIPT, CROSSWALK, SACCT, PAIR_CAPTURE):
        if not p.is_file():
            raise AuditError(f"required evidence file is missing: {p}")
    support = json.loads(SUPPORT.read_text(encoding="utf-8"))
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    pair_capture = json.loads(PAIR_CAPTURE.read_text(encoding="utf-8"))
    if status.get("output_sha256") != sha256(SUPPORT):
        raise AuditError("R03 support JSON hash does not match its status receipt")
    if status.get("capture_script_sha256") != sha256(CAPTURE_SCRIPT):
        raise AuditError("R03 capture script hash does not match its status receipt")
    if status.get("crosswalk_sha256") != sha256(CROSSWALK):
        raise AuditError("R03 crosswalk hash does not match its status receipt")
    if status.get("sacct_sha256") != sha256(SACCT):
        raise AuditError("R03 sacct file hash does not match its status receipt")
    if len(support.get("roots", [])) != 48 or status.get("root_count") != 48:
        raise AuditError("R03 root count is not 48")
    return support, status, pair_capture


def connect(readonly: bool) -> sqlite3.Connection:
    if readonly:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    else:
        con = sqlite3.connect(DB, timeout=180)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=180000")
    if readonly:
        con.execute("PRAGMA query_only=ON")
    return con


def q1(con: sqlite3.Connection, sql: str, args: tuple = ()):
    return con.execute(sql, args).fetchone()


def long_root(con: sqlite3.Connection, path: str):
    # READ_ONLY and archived roots still describe exact provenance paths; status
    # controls mutation policy, not whether a path can be cataloged.
    global _ROOT_CACHE
    if _ROOT_CACHE is None:
        _ROOT_CACHE = [dict(r) for r in con.execute("SELECT root_id,scoped_path,project_id,status FROM storage_root")]
    matches = [r for r in _ROOT_CACHE if path == r["scoped_path"] or path.startswith(r["scoped_path"].rstrip("/") + "/")]
    return max(matches, key=lambda r: len(r["scoped_path"])) if matches else None


def relative_path(scoped: str, path: str) -> str:
    if path == scoped:
        return ""
    return path[len(scoped.rstrip("/")) + 1 :]


def ensure_project(con: sqlite3.Connection) -> None:
    p = q1(con, "SELECT project_id FROM project WHERE project_id='OTHER_RESEARCH'")
    if p is None:
        raise AuditError("OTHER_RESEARCH project is missing; refusing implicit project creation")


def make_root_index(con: sqlite3.Connection, support: dict) -> dict[str, dict]:
    roots = {}
    for item in support["roots"]:
        path = item["run_root"]
        rid = root_id(path)
        run = q1(con, "SELECT * FROM workflow_run WHERE run_id=?", (rid,))
        if run is None:
            raise AuditError(f"staged root workflow missing from catalog: {path}")
        roots[path] = {"data": item, "run": dict(run), "run_id": rid}
    return roots


def support_file_records(root: dict) -> list[tuple[dict, str]]:
    keys = (("inputs", "QE_INPUT"), ("script_candidates", "SCHEDULER_SCRIPT"),
            ("manifests", "RUN_MANIFEST"), ("runtime_context", "RUNTIME_CONTEXT"),
            ("logs", "QE_OUTPUT_LOG"), ("pseudo_dependencies", "PSEUDOPOTENTIAL"))
    out = []
    for key, role in keys:
        for entry in root.get(key, []):
            out.append((entry, role))
    return out


def captured_local_records() -> list[tuple[Path, str, str]]:
    return [(SUPPORT, "R03_SUPPORT_CAPTURE_JSON", "R03 support capture data, SHA-256 checked against status receipt."),
            (STATUS, "R03_SUPPORT_CAPTURE_STATUS_JSON", "R03 support capture status and evidence hashes."),
            (CAPTURE_SCRIPT, "R03_SUPPORT_CAPTURE_SCRIPT", "Script that produced the R03 support capture."),
            (CROSSWALK, "R03_RUN_COORDINATE_CROSSWALK_TSV", "Run-root/job crosswalk; exact command strings only where captured."),
            (SACCT, "R03_SACCT_SUBMITLINE_PSV", "Scheduler accounting and captured submit-line evidence.")]


def inspect_plan(con: sqlite3.Connection, support: dict, status: dict, pair_capture: dict) -> dict:
    ensure_project(con)
    roots = make_root_index(con, support)
    issues: list[str] = []
    support_paths: dict[str, dict] = {}
    support_counts = Counter()
    for runroot, info in roots.items():
        for entry, role in support_file_records(info["data"]):
            path = entry["path"]
            rec = support_paths.setdefault(path, {"entry": entry, "roles": set(), "roots": set()})
            if rec["entry"].get("exists") != entry.get("exists"):
                issues.append(f"conflicting existence observation for {path}")
            if entry.get("sha256") and rec["entry"].get("sha256") not in (None, entry["sha256"]):
                issues.append(f"conflicting full hash for {path}")
            if rec["entry"].get("sha256") is None and entry.get("sha256"):
                rec["entry"] = entry
            rec["roles"].add(role)
            rec["roots"].add(runroot)
            support_counts[role] += 1
            sr = long_root(con, path)
            if not sr:
                issues.append(f"no catalog storage_root covers R03 path: {path}")

    support_path_aliases = []
    for path, rec in support_paths.items():
        sr = long_root(con, path)
        if not sr:
            continue
        rel = relative_path(sr["scoped_path"], path)
        sp = q1(con, "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (sr["root_id"], rel))
        if not sp:
            continue
        existing = con.execute("SELECT a.artifact_id,a.artifact_key,a.project_id,a.sha256,a.byte_size FROM artifact_location al JOIN artifact a USING(artifact_id) WHERE al.path_id=?", (sp["path_id"],)).fetchall()
        if len(existing) <= 1:
            continue
        entry = rec["entry"]
        sha = entry.get("sha256") if entry.get("hash_scope") == "FULL_FILE" else None
        size = entry.get("size_bytes")
        compatible = [a for a in existing if (not sha or not a["sha256"] or a["sha256"] == sha) and (size is None or a["byte_size"] is None or a["byte_size"] == size)]
        root_key = f"{sr['root_id']}:{rel}"
        root_matches = [a for a in compatible if a["artifact_key"] == root_key]
        pseudos = [a for a in compatible if a["project_id"] == "PSEUDOPOTENTIALS"]
        preserved_aliases = [a for a in compatible if a["artifact_key"].startswith("RF-YI-QE-PRESERVED-EVIDENCE:")]
        if len(compatible) != len(existing):
            issues.append(f"conflicting same-path catalog artifacts for R03 support: {path}")
        elif len(pseudos) == 1 and len(existing) == 2:
            alias = next(a for a in existing if a["artifact_id"] != pseudos[0]["artifact_id"])
            support_path_aliases.append({"path": path, "alias_artifact_id": alias["artifact_id"],
                                         "canonical_artifact_id": pseudos[0]["artifact_id"], "path_id": sp["path_id"],
                                         "sha256": sha, "size_bytes": size, "relation": "PSEUDOPOTENTIALS_CANONICAL"})
        elif len(root_matches) == 1 and len(existing) == 2 and len(preserved_aliases) == 1:
            # The R03 preserved-evidence reconciliation below records this alias.
            pass
        else:
            issues.append(f"ambiguous catalog artifact identities for R03 support path: {path}")

    jobs: dict[tuple[str, str], dict] = {}
    for runroot, info in roots.items():
        for job in info["data"].get("scheduler_jobs", []):
            key = (str(job.get("job_id", "")), job.get("workdir", ""))
            if not key[0] or not key[1]:
                issues.append(f"scheduler row lacks exact job_id/WorkDir in {runroot}")
                continue
            if key in jobs:
                old = jobs[key]
                for fld in ("state", "exit_code", "submit_line", "submit_time_raw", "start_time_raw", "end_time_raw", "job_name"):
                    if old["job"].get(fld) != job.get(fld):
                        issues.append(f"duplicate job/workdir has conflicting {fld}: {key}")
                old["roots"].append(runroot)
            else:
                jobs[key] = {"job": job, "roots": [runroot]}

    # Exact pair deletion aliases: full hashes and byte sizes come from the separate pre-delete capture.
    pair_files = [x for x in pair_capture["files"] if x.get("role") == "QE_MIXING_RESTART"]
    if len(pair_files) != 192:
        issues.append(f"expected 192 captured pair mixing restart files; found {len(pair_files)}")
    pair_aliases = []
    for f in pair_files:
        path = f["path"]
        sr = long_root(con, path)
        if not sr:
            issues.append(f"pair artifact path has no storage root: {path}")
            continue
        sp = q1(con, "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (sr["root_id"], relative_path(sr["scoped_path"], path)))
        if not sp:
            issues.append(f"pair artifact path absent from storage_path: {path}")
            continue
        arts = con.execute("SELECT a.artifact_id,a.artifact_key,a.sha256,a.byte_size FROM artifact_location al JOIN artifact a USING(artifact_id) WHERE al.path_id=?", (sp["path_id"],)).fetchall()
        aliases = [a for a in arts if a["artifact_key"].startswith("RF-YI-QE-RESTART:")]
        canon = [a for a in arts if a["artifact_key"].startswith("RF-DATA-YI:")]
        if len(aliases) != 1 or len(canon) != 1:
            issues.append(f"pair path needs one restart alias and one RF-DATA-YI canonical artifact: {path}")
            continue
        a, c = aliases[0], canon[0]
        if c["sha256"] != f["sha256"] or c["byte_size"] != f["size_bytes"]:
            issues.append(f"pair canonical differs from pre-delete full hash or size: {path}")
        pair_aliases.append({"alias_id": a["artifact_id"], "canonical_id": c["artifact_id"], "run_id": f["run_id"], "path": path})
    if len(pair_aliases) != 192:
        issues.append(f"resolved {len(pair_aliases)}/192 pair aliases")

    # Preserved XML/trajectory aliases are redirected only when a unique same-path canonical artifact exists.
    evidence = con.execute("SELECT a.artifact_id,a.artifact_key,a.sha256,a.byte_size,al.path_id,sp.root_id,sp.relative_path FROM artifact a JOIN artifact_location al USING(artifact_id) JOIN storage_path sp USING(path_id) WHERE a.artifact_key LIKE 'RF-YI-QE-PRESERVED-EVIDENCE:%'").fetchall()
    ev_by_path: dict[int, list] = defaultdict(list)
    for row in evidence:
        ev_by_path[row["path_id"]].append(row)
    evidence_aliases = []
    for path_id, aliases in ev_by_path.items():
        canon = con.execute("SELECT a.artifact_id,a.artifact_key,a.sha256,a.byte_size FROM artifact_location al JOIN artifact a USING(artifact_id) WHERE al.path_id=? AND a.artifact_key NOT LIKE 'RF-YI-QE-PRESERVED-EVIDENCE:%'", (path_id,)).fetchall()
        if not canon:
            continue
        if len(aliases) != 1 or len(canon) != 1:
            issues.append(f"preserved-evidence path has ambiguous aliases/canonicals: {path_id}")
            continue
        a, c = aliases[0], canon[0]
        if a["sha256"] and c["sha256"] and a["sha256"] != c["sha256"]:
            issues.append(f"preserved evidence/canonical SHA conflict at path_id={path_id}")
            continue
        obs = q1(con, "SELECT logical_bytes,sha256,hash_scope FROM inventory_observation WHERE path_id=? AND observation_status='PRESENT' ORDER BY observation_id DESC LIMIT 1", (path_id,))
        size_evidence = obs["logical_bytes"] if obs else None
        if size_evidence is None:
            # The actual file sizes are included in the R03 capture when one of these paths is a captured support file.
            root_row = q1(con, "SELECT scoped_path FROM storage_root WHERE root_id=?", (a["root_id"],))
            full_path = (root_row["scoped_path"].rstrip("/") + "/" + a["relative_path"]) if root_row else None
            same = support_paths.get(full_path) if full_path else None
            if same:
                size_evidence = same["entry"].get("size_bytes")
        if c["byte_size"] is not None and size_evidence is not None and c["byte_size"] != size_evidence:
            issues.append(f"preserved evidence canonical size disagrees with inventory at path_id={path_id}")
            continue
        if c["byte_size"] is None and size_evidence is None:
            issues.append(f"cannot prove canonical byte size for preserved evidence path_id={path_id}")
            continue
        evidence_aliases.append({"alias_id": a["artifact_id"], "canonical_id": c["artifact_id"],
                                 "path_id": path_id, "sha256": a["sha256"], "size_bytes": size_evidence or c["byte_size"]})
    if len(evidence) != 193 or len(evidence_aliases) != 28:
        issues.append(f"preserved evidence reconciliation expected 193 aliases / 28 overlaps; found {len(evidence)} / {len(evidence_aliases)}")

    # Ensure aliases can be explicitly resolved and no unexplained FK references prevent retirement of their locations.
    artifact_fk_columns = []
    for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        for fk in con.execute(f'PRAGMA foreign_key_list("{table}")'):
            if fk["table"] == "artifact":
                artifact_fk_columns.append((table, fk["from"]))
    handled = {
        ("artifact_location", "artifact_id"),
        ("run_artifact", "artifact_id"),
        ("recipe_input", "artifact_id"),
        ("cleanup_item", "artifact_id"),
        ("cleanup_candidate_context", "artifact_id"),
        ("derivation_edge", "parent_artifact_id"),
        ("derivation_edge", "child_artifact_id"),
        ("quality_gate", "evidence_artifact_id"),
        ("claim_evidence", "artifact_id"),
        ("paper_figure_asset", "artifact_id"),
        ("paper_release_item_evidence", "artifact_id"),
        ("duplicate_member", "artifact_id"),
    }
    alias_ids = [x["alias_id"] for x in pair_aliases] + [x["alias_id"] for x in evidence_aliases]
    alias_set = set(alias_ids)
    for table, col in artifact_fk_columns:
        # artifact_alias is the supported, durable redirect relation itself.
        if table == "artifact_alias" or (table, col) in handled:
            continue
        if alias_set and q1(con, f'SELECT 1 FROM "{table}" WHERE "{col}" IN ({",".join("?" for _ in alias_set)}) LIMIT 1', tuple(alias_set)):
            issues.append(f"unsupported artifact alias references exist in {table}.{col}")

    receipt = q1(con, "SELECT execution_state,deleted_file_count,actual_logical_bytes,quota_before_observation_id,quota_after_observation_id FROM cleanup_execution_receipt WHERE cleanup_batch_id=?", (BATCH,))
    if not receipt or receipt["execution_state"] != "COMPLETED" or receipt["deleted_file_count"] != 7581:
        issues.append("completed restart cleanup receipt is missing or has unexpected totals")
    held = q1(con, "SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id=? AND disposition='HOLD' AND deletion_state='HELD'", (BATCH,))[0]
    deleted = q1(con, "SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id=? AND disposition='DELETE' AND deletion_state='VERIFIED'", (BATCH,))[0]
    deleted_utc = q1(con, "SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id=? AND deleted_utc IS NOT NULL", (BATCH,))[0]
    if held != 8 or deleted != 7581 or deleted_utc:
        issues.append(f"cleanup invariant changed: held={held}, verified_delete={deleted}, deleted_utc_set={deleted_utc}")
    reviewed_pair = q1(con, "SELECT count(*) FROM reproduction_recipe WHERE run_id IN ('RF-DATA-YI-QE-NEB-25090135','RF-DATA-YI-QE-NEB-25090139') AND recipe_status='REVIEWED' AND recipe_version='fresh-start-stage-submit-2026-11-23-v1'")[0]
    if reviewed_pair != 2:
        issues.append("the two existing reviewed pair recipes are not both present")

    # Per-job recipe readiness is conservative: exact sbatch string, uniquely matched fully hashed script,
    # plus full hashes for every captured input and pseudo in the associated root group.
    job_status_counts = Counter()
    state_normalizations = Counter()
    job_plan = []
    support_by_root = {p: info["data"] for p, info in roots.items()}
    for (jid, wd), entry in jobs.items():
        job = entry["job"]
        chosen_root, match_basis = resolve_root(entry["roots"], wd, job.get("submit_line", ""), support_by_root)
        rdata = support_by_root.get(chosen_root, {}) if chosen_root else {}
        script = match_script(job.get("submit_line", ""), wd, rdata.get("script_candidates", [])) if chosen_root else None
        all_deps = rdata.get("inputs", []) + rdata.get("pseudo_dependencies", [])
        deps_ok = bool(all_deps) and all(x.get("exists") is True and x.get("hash_scope") == "FULL_FILE" and x.get("sha256") for x in all_deps)
        exact = is_exact_sbatch(job.get("submit_line", ""))
        normalized_state = normalize_scheduler_state(job.get("state"))
        if normalized_state != (job.get("state") or "UNKNOWN"):
            state_normalizations[f"{job.get('state')} -> {normalized_state}"] += 1
        ready = bool(chosen_root and exact and script and deps_ok)
        state = "REVIEWED" if ready else "INCOMPLETE"
        job_status_counts[state] += 1
        job_plan.append({"job_id": jid, "workdir": wd, "run_id": job_run_id(jid, wd), "root": chosen_root,
                         "root_match_basis": match_basis,
                         "recipe_status": state, "exact_command_captured": exact,
                         "workflow_scheduler_state": normalized_state,
                         "hashed_script_match": bool(script), "hashed_inputs_and_pseudos": deps_ok,
                         "job_sources": len(entry["roots"])})

    migration = q1(con, "SELECT migration_file FROM schema_migration WHERE version=16")
    migration_applied = bool(migration and migration["migration_file"] == "016_artifact_alias_redirects.sql")
    return {"ok": not issues, "issues": issues, "batch": BATCH,
            "required_schema_migration": "016_artifact_alias_redirects.sql", "migration_016_applied": migration_applied,
            "support_capture_utc": support.get("captured_utc"), "support_file_counts": dict(support_counts),
            "unique_support_paths": len(support_paths), "support_path_aliases": support_path_aliases,
            "root_count": len(roots),
            "scheduler_entries": sum(len(x["data"].get("scheduler_jobs", [])) for x in roots.values()),
            "unique_job_workdirs": len(jobs), "job_recipe_status_counts": dict(job_status_counts),
            "scheduler_state_normalizations": dict(state_normalizations),
            "job_runs": job_plan,
            "job_root_match_counts": dict(Counter(x["root_match_basis"] for x in job_plan)),
            "pair_restart_aliases": len(pair_aliases),
            "preserved_evidence_aliases": len(evidence), "preserved_evidence_overlaps_redirectable": len(evidence_aliases),
            "preserved_evidence_unique_to_keep": len(evidence) - len(evidence_aliases),
            "cleanup_invariants": {"delete_verified": deleted, "hold_held": held, "deleted_utc_populated": deleted_utc},
            "receipt_quota_refs_current": [receipt["quota_before_observation_id"], receipt["quota_after_observation_id"]],
            "reviewed_pair_recipes": reviewed_pair}


def resolve_root(candidate_roots: list[str], workdir: str, submit_line: str, support_by_root: dict[str, dict]) -> tuple[str | None, str]:
    """Resolve a root only from WorkDir ancestry or unique captured script/environment evidence."""
    ancestors = [r for r in candidate_roots if workdir == r or workdir.startswith(r.rstrip("/") + "/")]
    if ancestors:
        return max(ancestors, key=len), "WORKDIR_ANCESTRY"
    env_values = re.findall(r"MIGRATIONBENCH_(CAMPAIGN_ROOT|RUN_DIR)=([^\s,]+)", submit_line or "")
    env_hits = []
    for kind, value in env_values:
        value = value.strip("'\"\\")
        if kind == "RUN_DIR":
            candidates = [r for r in candidate_roots if value == r or value.startswith(r.rstrip("/") + "/")]
        else:
            candidates = [r for r in candidate_roots if value == r or r.startswith(value.rstrip("/") + "/")]
        if len(candidates) == 1:
            env_hits.append(candidates[0])
        elif len(candidates) > 1:
            return None, "AMBIGUOUS_CAPTURED_CAMPAIGN_PATH"
    if len(set(env_hits)) == 1:
        return env_hits[0], "EXACT_CAPTURED_CAMPAIGN_PATH"
    if len(set(env_hits)) > 1:
        return None, "AMBIGUOUS_CAPTURED_CAMPAIGN_PATH"
    token = submitted_script_token(submit_line or "")
    if token:
        expected = resolve_script_path(token, workdir)
        hits = []
        for root in candidate_roots:
            for entry in support_by_root.get(root, {}).get("script_candidates", []):
                if entry.get("exists") is True and entry.get("hash_scope") == "FULL_FILE" and entry.get("sha256") and str(Path(entry["path"])) == expected:
                    hits.append(root)
        if len(set(hits)) == 1:
            return hits[0], "EXACT_HASHED_SUBMIT_SCRIPT_PATH"
        if len(set(hits)) > 1:
            return None, "AMBIGUOUS_HASHED_SUBMIT_SCRIPT_PATH"
    return None, "UNMATCHED_WORKDIR_AND_SUBMIT_EVIDENCE"


def is_exact_sbatch(line: str) -> bool:
    if not line or line.startswith("submit="):
        return False
    try:
        tokens = shlex.split(line)
    except ValueError:
        return False
    return any(t == "sbatch" or t.endswith("/sbatch") for t in tokens)


def match_script(line: str, workdir: str, records: list[dict]) -> dict | None:
    if not is_exact_sbatch(line):
        return None
    script_token = submitted_script_token(line)
    if not script_token:
        return None
    eligible = [r for r in records if r.get("exists") is True and r.get("hash_scope") == "FULL_FILE" and r.get("sha256")]
    expected = resolve_script_path(script_token, workdir)
    hits = [r for r in eligible if str(Path(r["path"])) == expected]
    return hits[0] if len(hits) == 1 else None


def submitted_script_token(line: str) -> str | None:
    if not is_exact_sbatch(line):
        return None
    try:
        tokens = shlex.split(line)
    except ValueError:
        return None
    positions = [i for i, tok in enumerate(tokens) if tok == "sbatch" or tok.endswith("/sbatch")]
    if not positions:
        return None
    args = [t for t in tokens[positions[-1] + 1:] if not t.startswith("-") and ("." in Path(t).name or "/" in t)]
    return args[-1] if args else None


def resolve_script_path(token: str, workdir: str) -> str:
    return str(Path(token)) if token.startswith("/") else str(Path(workdir) / token)


def apply_plan(con: sqlite3.Connection, support: dict, status: dict, pair_capture: dict, plan: dict) -> dict:
    if not plan["ok"]:
        raise AuditError("preflight has conflicts; refusing --apply")
    if not plan.get("migration_016_applied"):
        raise AuditError("apply requires migration 016_artifact_alias_redirects.sql to be applied and recorded first")
    con.execute("BEGIN IMMEDIATE")
    roots = make_root_index(con, support)
    captured_utc = support["captured_utc"]
    source_ids = {}

    def register_local(path: Path, kind: str, note: str) -> int:
        digest, size = sha256(path), path.stat().st_size
        source_key = "qe_restart_r03:" + kind + ":" + digest[:20]
        con.execute("INSERT OR IGNORE INTO registry_source(source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes) VALUES(?,?,?,?,?,?,1,?,?)",
                    (source_key, str(path.resolve()), kind, digest, captured_utc, captured_utc,
                     "R03 Rockfish restart provenance capture; values remain evidence-bounded.", note))
        src = q1(con, "SELECT source_id,sha256 FROM registry_source WHERE source_key=?", (source_key,))
        if not src or src["sha256"] != digest:
            raise AuditError(f"registry source conflict: {source_key}")
        sr = long_root(con, str(path.resolve()))
        if not sr:
            raise AuditError(f"no local storage root for evidence: {path}")
        rel = relative_path(sr["scoped_path"], str(path.resolve()))
        con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (sr["root_id"], rel, captured_utc))
        path_id = q1(con, "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (sr["root_id"], rel))[0]
        akey = "RF-R03-EVIDENCE:" + kind + ":" + digest[:20]
        con.execute("INSERT OR IGNORE INTO artifact(project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes) VALUES('ROCKFISH_STORAGE_AUDIT',?,?,?,'USER_CONFIRMED',?,?,'LINEAGE_SUPPORTED','PERMANENT_SOURCE',?)",
                    (akey, kind, kind, digest, size, note))
        aid = q1(con, "SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (akey,))
        if not aid or (aid["sha256"], aid["byte_size"]) != (digest, size):
            raise AuditError(f"local evidence artifact conflict: {path}")
        con.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)", (aid["artifact_id"], path_id, captured_utc, captured_utc))
        source_ids[kind] = (src["source_id"], aid["artifact_id"])
        return aid["artifact_id"]

    evidence_artifacts = {}
    for path, kind, note in captured_local_records():
        evidence_artifacts[kind] = register_local(path, kind, note)

    # Support artifact identity is the exact path. Reuse a unique existing same-path artifact and enrich
    # only NULL fields; otherwise create a deterministic R03 evidence artifact.
    support_artifacts: dict[str, int] = {}
    support_roles: dict[str, set[str]] = defaultdict(set)
    support_duplicate_aliases = []
    for runroot, info in roots.items():
        for entry, role in support_file_records(info["data"]):
            path = entry["path"]
            sr = long_root(con, path)
            rel = relative_path(sr["scoped_path"], path)
            con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)", (sr["root_id"], rel, captured_utc))
            path_id = q1(con, "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (sr["root_id"], rel))[0]
            existing = con.execute("SELECT a.artifact_id,a.artifact_key,a.project_id,a.sha256,a.byte_size FROM artifact_location al JOIN artifact a USING(artifact_id) WHERE al.path_id=?", (path_id,)).fetchall()
            sha = entry.get("sha256") if entry.get("hash_scope") == "FULL_FILE" else None
            size = entry.get("size_bytes")
            compatible = [a for a in existing if (not sha or not a["sha256"] or a["sha256"] == sha) and (size is None or a["byte_size"] is None or a["byte_size"] == size)]
            root_key = f"{sr['root_id']}:{rel}"
            root_matches = [a for a in compatible if a["artifact_key"] == root_key]
            pseudos = [a for a in compatible if a["project_id"] == "PSEUDOPOTENTIALS"]
            preserved_aliases = [a for a in compatible if a["artifact_key"].startswith("RF-YI-QE-PRESERVED-EVIDENCE:")]
            if len(existing) == 2 and len(pseudos) == 1:
                chosen_existing = pseudos
                alias = next(a for a in existing if a["artifact_id"] != pseudos[0]["artifact_id"])
                support_duplicate_aliases.append((alias["artifact_id"], pseudos[0]["artifact_id"], path_id))
            elif len(existing) == 2 and len(preserved_aliases) == 1:
                chosen_existing = [a for a in compatible if a["artifact_id"] != preserved_aliases[0]["artifact_id"]]
            elif len(root_matches) == 1:
                chosen_existing = root_matches
            else:
                chosen_existing = compatible if len(compatible) == 1 else []
            if existing and len(chosen_existing) == 1:
                aid = chosen_existing[0]["artifact_id"]
                old = chosen_existing[0]
                if (old["sha256"] and sha and old["sha256"] != sha) or (old["byte_size"] is not None and size is not None and old["byte_size"] != size):
                    raise AuditError(f"support artifact hash/size conflict: {path}")
                con.execute("UPDATE artifact SET sha256=COALESCE(sha256,?),byte_size=COALESCE(byte_size,?) WHERE artifact_id=?", (sha, size, aid))
            elif len(existing) == 0:
                digest_key = hashlib.sha256(path.encode()).hexdigest()[:24]
                akey = "RF-R03-SUPPORT:" + digest_key
                con.execute("INSERT OR IGNORE INTO artifact(project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes) VALUES(?,?,?,?, 'USER_CONFIRMED',?,?, 'LINEAGE_SUPPORTED',?,?)",
                            ("OTHER_RESEARCH" if "Replican-ONCV" in path else info["run"]["project_id"], akey, role, role, sha, size,
                             "REPRODUCIBLE_INPUT" if role in ("QE_INPUT", "SCHEDULER_SCRIPT", "PSEUDOPOTENTIAL") else "COMPACT_RESULT" if role == "QE_OUTPUT_LOG" else "PERMANENT_SOURCE",
                             f"R03 captured {role}; original path={path}; payload not opened unless capture records a full-file hash."))
                aidrow = q1(con, "SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?", (akey,))
                if not aidrow or (aidrow["sha256"], aidrow["byte_size"]) != (sha, size):
                    raise AuditError(f"R03 artifact identity conflict: {path}")
                aid = aidrow["artifact_id"]
            else:
                raise AuditError(f"ambiguous support path artifacts: {path}")
            if "Replican-ONCV" in path:
                con.execute("UPDATE artifact SET project_id='OTHER_RESEARCH' WHERE artifact_id=?", (aid,))
            avail = "PRESENT" if entry.get("exists") is True else "MISSING"
            con.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,?,?,?)", (aid, path_id, avail, captured_utc, captured_utc))
            con.execute("UPDATE artifact_location SET availability=?,last_verified_utc=? WHERE artifact_id=? AND path_id=?", (avail, captured_utc, aid, path_id))
            support_artifacts[path] = aid
            support_roles[runroot].add(role)

    # Attach captured evidence to root workflows and retain missing pseudos as required, explicit inputs.
    for runroot, info in roots.items():
        project = "OTHER_RESEARCH" if "Replican-ONCV" in runroot else info["run"]["project_id"]
        system = "CsPbBr3" if "Replican-ONCV" in runroot else info["run"]["system_key"]
        con.execute("UPDATE workflow_run SET project_id=?,system_key=?,notes=CASE WHEN instr(notes,'R03 support capture registered')=0 THEN notes||? ELSE notes END WHERE run_id=?",
                    (project, system, " R03 support capture registered; run remains PROVENANCE_PARTIAL.", info["run_id"]))
        for entry, role in support_file_records(info["data"]):
            aid = support_artifacts[entry["path"]]
            direction = "LOG" if role == "QE_OUTPUT_LOG" else "ENVIRONMENT" if role == "RUNTIME_CONTEXT" else "PROVENANCE" if role == "RUN_MANIFEST" else "INPUT"
            required = 1 if role in ("QE_INPUT", "SCHEDULER_SCRIPT", "PSEUDOPOTENTIAL") else 0
            con.execute("INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,?,?,?)",
                        (info["run_id"], aid, role.lower(), direction, required, "R03 captured support path; status and hash scope are stored on artifact/location."))
        con.execute("INSERT OR IGNORE INTO registry_entity_ref(source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes) VALUES(?,?,?,?,?,'SUPPORTS_RUN','R03 exact run-root crosswalk.')",
                    (source_ids["R03_SUPPORT_CAPTURE_JSON"][0], "RUN_ROOT", runroot, "WORKFLOW_RUN", info["run_id"]))

    # Create exact WorkDir-level workflow runs and observations (unique by job id + exact WorkDir).
    jobs: dict[tuple[str, str], dict] = {}
    for runroot, info in roots.items():
        for job in info["data"].get("scheduler_jobs", []):
            key = (str(job["job_id"]), job["workdir"])
            if key not in jobs:
                jobs[key] = {"job": job, "roots": [runroot]}
            elif runroot not in jobs[key]["roots"]:
                jobs[key]["roots"].append(runroot)
    sacct_aid = evidence_artifacts["R03_SACCT_SUBMITLINE_PSV"]
    run_count = reviewed_count = incomplete_count = observation_count = 0
    job_to_root = {}
    support_by_root = {p: info["data"] for p, info in roots.items()}
    for (jid, wd), record in sorted(jobs.items()):
        job = record["job"]
        chosen, match_basis = resolve_root(record["roots"], wd, job.get("submit_line", ""), support_by_root)
        parent = roots[chosen] if chosen else None
        rid = job_run_id(jid, wd)
        if chosen:
            job_to_root[(jid, wd)] = chosen
        src = long_root(con, wd)
        srcroot = src["root_id"] if src else (parent["run"]["source_root_id"] if parent else None)
        srcrel = relative_path(src["scoped_path"], wd) if src else (parent["run"]["source_relative_path"] if parent else None)
        is_replican = bool(chosen and "Replican-ONCV" in chosen)
        project = "OTHER_RESEARCH" if is_replican or not parent else parent["run"]["project_id"]
        system = "CsPbBr3" if is_replican else (parent["run"]["system_key"] if parent else None)
        workflow_type = parent["run"]["workflow_type"] if parent else "QE_JOB"
        method = parent["run"]["method"] if parent else "Quantum ESPRESSO"
        con.execute("INSERT INTO workflow_run(run_id,project_id,campaign_id,run_label,workflow_type,method,system_key,source_root_id,source_relative_path,scheduler_state,scientific_status,notes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id) DO NOTHING",
                    (rid, project, parent["run"]["campaign_id"] if parent else None, f"Slurm {jid} @ {wd}", workflow_type, method, system,
                     srcroot, srcrel, normalize_scheduler_state(job.get("state")), "PROVENANCE_PARTIAL",
                     f"Exact WorkDir-level run from R03 evidence; root_match_basis={match_basis}; "
                     + ("project/campaign/system unresolved; OTHER_RESEARCH used as an administrative bucket. " if not parent else "")
                     + "scheduler completion is not scientific validation."))
        oldrun = q1(con, "SELECT project_id,system_key,source_root_id,source_relative_path FROM workflow_run WHERE run_id=?", (rid,))
        if not oldrun or (oldrun["project_id"], oldrun["system_key"], oldrun["source_root_id"], oldrun["source_relative_path"]) != (project, system, srcroot, srcrel):
            raise AuditError(f"job run id conflicts with existing run: {rid}")
        if parent:
            con.execute("INSERT OR IGNORE INTO run_parent(child_run_id,parent_run_id,relationship,notes) VALUES(?,?,'JOB_EXECUTION_OF','Root link is supported by exact WorkDir, hashed submit script, or captured campaign path.')", (rid, parent["run_id"]))
        con.execute("INSERT INTO scheduler_job_source_observation(run_id,cluster_name,job_id,job_name,sacct_state,exit_code,submit_time_raw,start_time_raw,end_time_raw,workdir,observed_utc,source_artifact_id,notes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(run_id,cluster_name,job_id,observed_utc) DO NOTHING",
                    (rid, CLUSTER, jid, job.get("job_name") or "", job.get("state") or "", job.get("exit_code") or "", job.get("submit_time_raw") or "", job.get("start_time_raw") or "", job.get("end_time_raw") or "", wd,
                     support["captured_utc"], sacct_aid, f"R03 source={job.get('source','sacct')}; raw scheduler fields retained; duplicate crosswalk memberships={len(record['roots'])}."))
        saved_observation = q1(con, "SELECT workdir,source_artifact_id,sacct_state,exit_code,submit_time_raw,start_time_raw,end_time_raw,job_name FROM scheduler_job_source_observation WHERE run_id=? AND cluster_name=? AND job_id=? AND observed_utc=?", (rid, CLUSTER, jid, support["captured_utc"]))
        expected_observation = (wd, sacct_aid, job.get("state") or "", job.get("exit_code") or "", job.get("submit_time_raw") or "", job.get("start_time_raw") or "", job.get("end_time_raw") or "", job.get("job_name") or "")
        if not saved_observation or tuple(saved_observation) != expected_observation:
            raise AuditError(f"scheduler observation failed to persist for job/workdir {jid} @ {wd}")
        observation_count += 1
        con.execute("INSERT OR IGNORE INTO registry_entity_ref(source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes) VALUES(?,?,?,?,?,'OBSERVES_RUN','Exact job_id + WorkDir association from R03.')",
                    (source_ids["R03_SACCT_SUBMITLINE_PSV"][0], "SLURM_JOB", jid + " @ " + wd, "WORKFLOW_RUN", rid))
        rdata = parent["data"] if parent else {}
        for entry, role in support_file_records(rdata):
            aid = support_artifacts[entry["path"]]
            direction = "LOG" if role == "QE_OUTPUT_LOG" else "ENVIRONMENT" if role == "RUNTIME_CONTEXT" else "PROVENANCE" if role == "RUN_MANIFEST" else "INPUT"
            required = 1 if role in ("QE_INPUT", "SCHEDULER_SCRIPT", "PSEUDOPOTENTIAL") else 0
            con.execute("INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,?,?,?)",
                        (rid, aid, role.lower(), direction, required, "R03 run-root evidence inherited by the exact WorkDir job."))
        script_record = match_script(job.get("submit_line", ""), wd, rdata.get("script_candidates", []))
        script_id = support_artifacts[script_record["path"]] if script_record else None
        deps = rdata.get("inputs", []) + rdata.get("pseudo_dependencies", [])
        manifests = rdata.get("manifests", [])
        manifest_id = support_artifacts[manifests[0]["path"]] if len(manifests) == 1 else None
        deps_ok = bool(deps) and all(x.get("exists") is True and x.get("hash_scope") == "FULL_FILE" and x.get("sha256") for x in deps)
        exact = is_exact_sbatch(job.get("submit_line", ""))
        recipe_status = "REVIEWED" if chosen and exact and script_id is not None and deps_ok else "INCOMPLETE"
        if recipe_status == "REVIEWED":
            reviewed_count += 1
        else:
            incomplete_count += 1
        con.execute("INSERT OR IGNORE INTO reproduction_recipe(run_id,recipe_version,exact_command,working_directory,environment_description,module_list,script_artifact_id,scheduler_script_artifact_id,input_manifest_artifact_id,recipe_status,validation_note) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (rid, RECIPE_VERSION, job.get("submit_line") if exact else None, wd,
                     "R03 capture only. External QE/MPI modules and binaries are not content-pinned.",
                     "Module names only where visible in hashed script; exact environment not asserted.", script_id, script_id, manifest_id, recipe_status,
                     "Static fresh-start evidence only; never marked REPRODUCED. REVIEWED requires exact root association, exact captured sbatch, unique FULL_FILE hashed script, and FULL_FILE hashed inputs+pseudopotentials."
                     if recipe_status == "REVIEWED" else f"INCOMPLETE: root association={match_basis}; missing exact root/script/input evidence remains unresolved; no command was inferred."))
        recipe = q1(con, "SELECT recipe_id,recipe_status,exact_command FROM reproduction_recipe WHERE run_id=? AND recipe_version=?", (rid, RECIPE_VERSION))
        if not recipe or recipe["recipe_status"] != recipe_status or recipe["exact_command"] != (job.get("submit_line") if exact else None):
            raise AuditError(f"recipe conflict for job run {rid}")
        for entry in deps:
            aid = support_artifacts[entry["path"]]
            con.execute("INSERT OR IGNORE INTO recipe_input(recipe_id,artifact_id,input_role) VALUES(?,?,?)", (recipe["recipe_id"], aid, "PSEUDOPOTENTIAL" if entry in rdata.get("pseudo_dependencies", []) else "QE_INPUT"))
        if manifest_id:
            con.execute("INSERT OR IGNORE INTO recipe_input(recipe_id,artifact_id,input_role) VALUES(?,?,'INPUT_MANIFEST')", (recipe["recipe_id"], manifest_id))
        if script_id:
            con.execute("INSERT OR IGNORE INTO recipe_input(recipe_id,artifact_id,input_role) VALUES(?,?,'SCHEDULER_SCRIPT')", (recipe["recipe_id"], script_id))
        run_count += 1

    # Roots without jobs get an explicit incomplete root recipe; Replican also records its absent pseudos.
    jobs_by_root = Counter(job_to_root.values())
    for runroot, info in roots.items():
        if jobs_by_root[runroot]:
            continue
        rid = info["run_id"]
        con.execute("INSERT OR IGNORE INTO reproduction_recipe(run_id,recipe_version,exact_command,working_directory,environment_description,recipe_status,validation_note) VALUES(?,?,NULL,?,?, 'INCOMPLETE',?)",
                    (rid, RECIPE_VERSION, runroot, "No exact scheduler submit line was captured for this root.",
                     "Root-level recipe is incomplete; captured inputs, scripts, and missing dependencies remain linked as artifacts."))
        recipe = q1(con, "SELECT recipe_id FROM reproduction_recipe WHERE run_id=? AND recipe_version=?", (rid, RECIPE_VERSION))[0]
        for entry, role in support_file_records(info["data"]):
            if role in ("QE_INPUT", "SCHEDULER_SCRIPT", "PSEUDOPOTENTIAL"):
                con.execute("INSERT OR IGNORE INTO recipe_input(recipe_id,artifact_id,input_role) VALUES(?,?,?)", (recipe, support_artifacts[entry["path"]], role))

    # Migration 016 creates artifact_alias. Historical artifact rows and locations remain intact for audit.
    if not q1(con, "SELECT 1 FROM sqlite_master WHERE type='table' AND name='artifact_alias'"):
        raise AuditError("artifact_alias table missing; apply schema migration 016 first")
    pair_map = {}
    for f in pair_capture["files"]:
        if f.get("role") != "QE_MIXING_RESTART":
            continue
        sr = long_root(con, f["path"])
        rel = relative_path(sr["scoped_path"], f["path"])
        path_id = q1(con, "SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?", (sr["root_id"], rel))[0]
        a = q1(con, "SELECT a.artifact_id FROM artifact a JOIN artifact_location al USING(artifact_id) WHERE al.path_id=? AND a.artifact_key LIKE 'RF-YI-QE-RESTART:%'", (path_id,))[0]
        c = q1(con, "SELECT a.artifact_id FROM artifact a JOIN artifact_location al USING(artifact_id) WHERE al.path_id=? AND a.artifact_key LIKE 'RF-DATA-YI:%'", (path_id,))[0]
        pair_map[a] = (c, f["run_id"], path_id)
    ev_map = {r["alias_id"]: (r["canonical_id"], None, r["path_id"]) for r in plan_evidence_aliases(con)}
    support_duplicate_aliases = sorted(set(support_duplicate_aliases))
    support_map = {alias_id: (canonical_id, None, path_id) for alias_id, canonical_id, path_id in support_duplicate_aliases}
    all_alias_map = {**support_map, **{a: (c, run, path) for a, (c, run, path) in pair_map.items()}, **ev_map}
    for old, (new, mapped_run, path_id) in all_alias_map.items():
        if mapped_run:
            evidence_aid = evidence_artifacts["R03_SUPPORT_CAPTURE_JSON"]
            reason = "192-file pair restart metadata alias; full SHA/size verified against pre-delete capture."
        elif old in support_map:
            evidence_aid = evidence_artifacts["R03_SUPPORT_CAPTURE_JSON"]
            reason = "Same-path pseudopotential catalog alias; R03 full SHA/size agrees with the PSEUDOPOTENTIALS canonical artifact."
        else:
            evidence_aid = evidence_artifacts["R03_RUN_COORDINATE_CROSSWALK_TSV"]
            reason = "Preserved-evidence alias shares an exact storage path with a canonical artifact; SHA/size checked before redirect."
        con.execute("INSERT OR IGNORE INTO artifact_alias(alias_artifact_id,canonical_artifact_id,alias_reason,evidence_artifact_id,recorded_utc) VALUES(?,?,?,?,?)", (old, new, reason, evidence_aid, captured_utc))
        alias = q1(con, "SELECT canonical_artifact_id FROM artifact_alias WHERE alias_artifact_id=?", (old,))
        if not alias or alias["canonical_artifact_id"] != new:
            raise AuditError(f"artifact alias already points to a different canonical id: {old}")
        # Cleanup item/context rows move together. Pair context is rebound to the real historical run.
        item = q1(con, "SELECT * FROM cleanup_item WHERE cleanup_batch_id=? AND artifact_id=?", (BATCH, old))
        if item:
            existing = q1(con, "SELECT 1 FROM cleanup_item WHERE cleanup_batch_id=? AND artifact_id=?", (BATCH, new))
            if existing:
                raise AuditError(f"cleanup item collision while redirecting alias {old}")
            con.execute("UPDATE cleanup_item SET artifact_id=? WHERE cleanup_batch_id=? AND artifact_id=?", (new, BATCH, old))
        ctx = q1(con, "SELECT * FROM cleanup_candidate_context WHERE cleanup_batch_id=? AND artifact_id=?", (BATCH, old))
        if ctx:
            existing = q1(con, "SELECT 1 FROM cleanup_candidate_context WHERE cleanup_batch_id=? AND artifact_id=?", (BATCH, new))
            if existing:
                raise AuditError(f"cleanup context collision while redirecting alias {old}")
            if mapped_run:
                con.execute("UPDATE cleanup_candidate_context SET artifact_id=?,run_id=? WHERE cleanup_batch_id=? AND artifact_id=?", (new, mapped_run, BATCH, old))
            else:
                con.execute("UPDATE cleanup_candidate_context SET artifact_id=? WHERE cleanup_batch_id=? AND artifact_id=?", (new, BATCH, old))
        # Preserve run-artifact identity, but bind pair restarts to the two known real runs.
        links = con.execute("SELECT run_id,role,direction,required_for_reproduction,notes FROM run_artifact WHERE artifact_id=?", (old,)).fetchall()
        for link in links:
            target_run = mapped_run or link["run_id"]
            con.execute("INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes) VALUES(?,?,?,?,?,?)",
                        (target_run, new, link["role"], link["direction"], link["required_for_reproduction"], link["notes"] + " Redirected from metadata alias."))
            con.execute("DELETE FROM run_artifact WHERE run_id=? AND artifact_id=? AND role=?", (link["run_id"], old, link["role"]))
        # Redirect other direct artifact FKs only when a corresponding canonical row can be preserved safely.
        redirect_other_fk_refs(con, old, new)
        if mapped_run:
            # These are the 192 physically deleted mixing restart files; mark both historical alias and
            # canonical path entries REMOVED while retaining their location rows.
            con.execute("UPDATE artifact_location SET availability='REMOVED',last_verified_utc=? WHERE artifact_id IN (?,?) AND path_id=?", (captured_utc, old, new, path_id))
            con.execute("UPDATE cleanup_candidate_context SET run_id=? WHERE cleanup_batch_id=? AND artifact_id=?", (mapped_run, BATCH, new))
    # Canonical evidence receives the full hash only if size is proven and no conflicting identity exists.
    for old, (new, run, path_id) in ev_map.items():
        alias = q1(con, "SELECT sha256 FROM artifact WHERE artifact_id=?", (old,))
        canon = q1(con, "SELECT sha256,byte_size FROM artifact WHERE artifact_id=?", (new,))
        obs = q1(con, "SELECT logical_bytes,sha256 FROM inventory_observation WHERE path_id=? AND observation_status='PRESENT' ORDER BY observation_id DESC LIMIT 1", (path_id,))
        size = obs["logical_bytes"] if obs else canon["byte_size"]
        if canon["sha256"] and alias["sha256"] and canon["sha256"] != alias["sha256"]:
            raise AuditError(f"preserved-evidence canonical hash conflict for artifact {new}")
        if canon["byte_size"] is not None and size is not None and canon["byte_size"] != size:
            raise AuditError(f"preserved-evidence canonical size conflict for artifact {new}")
        if not size:
            raise AuditError(f"no exact size proof for canonical preserved evidence artifact {new}")
        con.execute("UPDATE artifact SET sha256=COALESCE(sha256,?),byte_size=COALESCE(byte_size,?) WHERE artifact_id=?", (alias["sha256"], size, new))

    # Idempotently bind all alias redirects to canonical identities; never set deleted_utc.
    if con.execute("SELECT count(*) FROM artifact_alias WHERE alias_artifact_id IN (SELECT artifact_id FROM artifact WHERE artifact_key LIKE 'RF-YI-QE-RESTART:%')").fetchone()[0] != 192:
        raise AuditError("pair alias redirect count did not persist as 192")
    if con.execute("SELECT count(*) FROM artifact_alias WHERE alias_artifact_id IN (SELECT artifact_id FROM artifact WHERE artifact_key LIKE 'RF-YI-QE-PRESERVED-EVIDENCE:%')").fetchone()[0] != 28:
        raise AuditError("preserved evidence alias redirect count did not persist as 28")
    if q1(con, "SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id=? AND disposition='HOLD' AND deletion_state='HELD'", (BATCH,))[0] != 8:
        raise AuditError("held rows changed during supplement")
    if q1(con, "SELECT count(*) FROM cleanup_item WHERE cleanup_batch_id=? AND deleted_utc IS NOT NULL", (BATCH,))[0] != 0:
        raise AuditError("cleanup item deleted_utc must remain NULL")
    if q1(con, "SELECT count(*) FROM reproduction_recipe WHERE run_id IN ('RF-DATA-YI-QE-NEB-25090135','RF-DATA-YI-QE-NEB-25090139') AND recipe_status='REVIEWED' AND recipe_version='fresh-start-stage-submit-2026-11-23-v1'", ())[0] != 2:
        raise AuditError("existing pair REVIEWED recipes were changed")
    fk = con.execute("PRAGMA foreign_key_check").fetchall()
    if fk:
        raise AuditError(f"foreign_key_check failed: {len(fk)} rows")
    con.commit()
    return {"job_runs_imported": run_count, "scheduler_observations_imported": observation_count,
            "job_recipes_reviewed": reviewed_count, "job_recipes_incomplete": incomplete_count,
            "pair_alias_redirects": len(pair_map), "preserved_evidence_redirects": len(ev_map),
            "same_path_pseudopotential_aliases": len(support_duplicate_aliases),
            "preserved_evidence_unique_aliases_kept": 193 - len(ev_map), "artifact_alias_table": "artifact_alias"}


def plan_evidence_aliases(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute("SELECT a.artifact_id,a.artifact_key,a.sha256,a.byte_size,al.path_id FROM artifact a JOIN artifact_location al USING(artifact_id) WHERE a.artifact_key LIKE 'RF-YI-QE-PRESERVED-EVIDENCE:%'").fetchall()
    groups: dict[int, list] = defaultdict(list)
    for r in rows:
        groups[r["path_id"]].append(r)
    out = []
    for path_id, aliases in groups.items():
        canonical = con.execute("SELECT a.artifact_id,a.artifact_key,a.sha256,a.byte_size FROM artifact a JOIN artifact_location al USING(artifact_id) WHERE al.path_id=? AND a.artifact_key NOT LIKE 'RF-YI-QE-PRESERVED-EVIDENCE:%'", (path_id,)).fetchall()
        if len(aliases) != 1 or len(canonical) != 1:
            continue
        alias, canon = aliases[0], canonical[0]
        obs = q1(con, "SELECT logical_bytes FROM inventory_observation WHERE path_id=? AND observation_status='PRESENT' ORDER BY observation_id DESC LIMIT 1", (path_id,))
        size = obs["logical_bytes"] if obs else canon["byte_size"]
        if alias["sha256"] and canon["sha256"] and alias["sha256"] != canon["sha256"]:
            raise AuditError(f"preserved-evidence hash conflict at path_id={path_id}")
        if canon["byte_size"] is not None and size is not None and canon["byte_size"] != size:
            raise AuditError(f"preserved-evidence size conflict at path_id={path_id}")
        if not size:
            raise AuditError(f"preserved-evidence size unproven at path_id={path_id}")
        out.append({"alias_id": alias["artifact_id"], "canonical_id": canon["artifact_id"], "path_id": path_id,
                    "sha256": alias["sha256"], "size_bytes": size})
    return out


def redirect_other_fk_refs(con: sqlite3.Connection, old: int, new: int) -> None:
    handled_here = {("artifact_location", "artifact_id"), ("run_artifact", "artifact_id"),
                    ("cleanup_item", "artifact_id"), ("cleanup_candidate_context", "artifact_id")}
    tables = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
    for (table,) in tables:
        for fk in con.execute(f'PRAGMA foreign_key_list("{table}")'):
            col = fk["from"]
            if fk["table"] != "artifact" or table == "artifact_alias" or (table, col) in handled_here:
                continue
            rows = con.execute(f'SELECT rowid,* FROM "{table}" WHERE "{col}"=?', (old,)).fetchall()
            for row in rows:
                try:
                    con.execute(f'UPDATE OR IGNORE "{table}" SET "{col}"=? WHERE rowid=?', (new, row["rowid"]))
                except sqlite3.OperationalError as exc:
                    raise AuditError(f"cannot redirect {table}.{col}: {exc}") from exc
                remain = con.execute(f'SELECT rowid,* FROM "{table}" WHERE rowid=?', (row["rowid"],)).fetchone()
                if remain and remain[col] == old:
                    values = dict(remain)
                    values[col] = new
                    clauses = " AND ".join(f'"{k}" IS ?' for k in values if k != "rowid")
                    params = [values[k] for k in values if k != "rowid"]
                    dupe = con.execute(f'SELECT 1 FROM "{table}" WHERE {clauses} LIMIT 1', params).fetchone()
                    if dupe:
                        con.execute(f'DELETE FROM "{table}" WHERE rowid=?', (row["rowid"],))
                    else:
                        raise AuditError(f"non-identical unique-reference conflict in {table}.{col} for alias {old}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="apply the validated supplement in one SQLite transaction")
    args = parser.parse_args()
    try:
        support, status, pair_capture = load_inputs()
        con = connect(readonly=not args.apply)
        try:
            plan = inspect_plan(con, support, status, pair_capture)
            if args.apply and not plan["ok"]:
                raise AuditError("preflight failed; no changes applied")
            report = {"mode": "APPLY" if args.apply else "READ_ONLY_PREFLIGHT", **plan}
            if args.apply:
                report["applied"] = apply_plan(con, support, status, pair_capture, plan)
                report["ok"] = True
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if plan["ok"] else 2
        finally:
            con.close()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc), "mode": "APPLY" if args.apply else "READ_ONLY_PREFLIGHT"}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
