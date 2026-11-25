#!/usr/bin/env python3
"""Import the hashed Revision 1 retained-file crosswalk and explicit incomplete recipes."""
from __future__ import annotations
import collections, csv, hashlib, json, sqlite3
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

AUDIT = Path(__file__).resolve().parent.parent
REPO = AUDIT.parents[1]
DB = AUDIT / "provenance_catalog.sqlite"
CAPTURE = AUDIT / "phase2_data/revision1_run_evidence_capture_20261123T110231Z"
ROWS = CAPTURE / "run_artifact_bindings.jsonl"
SUMMARY = CAPTURE / "capture_summary.json"
INPUT_MAP = CAPTURE / "inputs_by_run_id.json"
CROSSWALK = AUDIT / "phase2_data/revision1_restart_audit_20261123T081721Z/run_group_crosswalk.tsv"
RECIPE_VERSION = "rev1-historical-rerun-evidence-20261123-v1"
RUN_PREFIX = "RF-MB-REV1-RESTART-JOB-"
SNAPSHOT_SUFFIX = "REV1-RUN-ARTIFACTS-20261123T110231Z"


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")


def root_for(con: sqlite3.Connection, absolute: str) -> tuple[str,str,str]:
    p=PurePosixPath(absolute)
    matches=[]
    for root_id,scoped in con.execute("SELECT root_id,scoped_path FROM storage_root WHERE status IN ('ACTIVE','READ_ONLY')"):
        r=PurePosixPath(scoped)
        try: rel=str(p.relative_to(r))
        except ValueError: continue
        matches.append((len(str(r)),root_id,str(r),rel))
    if not matches: raise RuntimeError("No registered storage root covers "+absolute)
    _,root_id,root_path,rel=max(matches)
    return root_id,root_path,rel


def add_source(con: sqlite3.Connection, path: Path, kind: str, note: str, captured: str) -> tuple[int,int]:
    path=path.resolve(strict=True); digest=sha256(path); size=path.stat().st_size
    local_root=con.execute("SELECT scoped_path FROM storage_root WHERE root_id='LOCAL-MAIN'").fetchone()
    if not local_root or Path(local_root[0]).resolve()!=REPO.resolve():
        raise RuntimeError("LOCAL-MAIN does not resolve to this unified worktree")
    rel=path.relative_to(REPO).as_posix()
    key=f"rev1_run_capture:{kind}:{digest[:20]}"
    con.execute("""INSERT OR IGNORE INTO registry_source
      (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
      VALUES(?,?,?,?,?,?,1,?,?)""",
      (key,str(path),kind,digest,captured,captured,
       "Read-only capture; user-authorized storage provenance and restart cleanup.",note))
    source=con.execute("SELECT source_id,source_path,sha256 FROM registry_source WHERE source_key=?",(key,)).fetchone()
    if not source or source[1:]!=(str(path),digest): raise RuntimeError("source identity conflict "+key)
    con.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES('LOCAL-MAIN',?,?)",(rel,captured))
    path_id=con.execute("SELECT path_id FROM storage_path WHERE root_id='LOCAL-MAIN' AND relative_path=?",(rel,)).fetchone()[0]
    artifact_key=f"RF-AUDIT:{rel}:{digest[:16]}"
    con.execute("""INSERT OR IGNORE INTO artifact
      (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
      VALUES('ROCKFISH_STORAGE_AUDIT',?,'AUDIT_EVIDENCE',?,'USER_CONFIRMED',?,?,'LINEAGE_SUPPORTED','PERMANENT_SOURCE',?)""",
      (artifact_key,kind,digest,size,note))
    art=con.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?",(artifact_key,)).fetchone()
    if not art or art[1:]!=(digest,size): raise RuntimeError("evidence artifact conflict "+artifact_key)
    con.execute("""INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
      VALUES(?,?,'PRESENT',?,?)""",(art[0],path_id,captured,captured))
    con.execute("UPDATE artifact_location SET availability='PRESENT',last_verified_utc=? WHERE artifact_id=? AND path_id=?",(captured,art[0],path_id))
    return int(source[0]),int(art[0])


def add_ref(con, source_id:int, run_id:str, note:str):
    con.execute("""INSERT OR IGNORE INTO registry_entity_ref
      (source_id,entity_type,entity_key,catalog_entity_type,catalog_entity_key,relationship,notes)
      VALUES(?,'WORKFLOW_RUN',?,'WORKFLOW_RUN',?,'EVIDENCE_FOR',?)""",(source_id,run_id,run_id,note))


def classify(paths:list[str])->str:
    names=[PurePosixPath(p).name.lower() for p in paths]
    has_scf=any("scf" in n for n in names)
    has_relax=any("relax" in n for n in names)
    if any(n=="neb.in" or n.startswith("neb.") for n in names): return "QE_NEB"
    if has_scf and has_relax: return "QE_SCF_RELAX_CHAIN"
    if has_scf: return "QE_SCF"
    if any(".cont" in n for n in names): return "QE_RELAX_CONTINUATION"
    if has_relax: return "QE_RELAX"
    raise RuntimeError("Cannot classify QE inputs without ambiguity: "+repr(paths))


def main():
    for p in (ROWS,SUMMARY,INPUT_MAP,CROSSWALK): p.resolve(strict=True)
    summary=json.loads(SUMMARY.read_text())
    if sha256(ROWS)!=summary["jsonl_sha256"]: raise SystemExit("capture JSONL checksum mismatch")
    rows=[json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    if len(rows)!=1669 or any(x.get("status")!="PRESENT" or x.get("error") for x in rows):
        raise SystemExit("capture row count/presence check failed")
    if any(len(x.get("sha256",""))!=64 for x in rows): raise SystemExit("capture row lacks sha256")
    mapping=json.loads(INPUT_MAP.read_text())
    input_by_run=mapping["inputs_by_run_id"]
    if len(input_by_run)!=113 or sum(len(x["input_paths"]) for x in input_by_run.values())!=275:
        raise SystemExit("input map run/path count mismatch")
    with CROSSWALK.open(newline="",encoding="utf-8") as f:
        crosswalk=list(csv.DictReader(f,delimiter="\t"))
    run_by_id={RUN_PREFIX+x["run_job_id"]:x for x in crosswalk}
    if len(run_by_id)!=113 or set(run_by_id)!=set(input_by_run):
        raise SystemExit("crosswalk and capture run ids differ")
    types=collections.Counter(classify([x["path"] for x in input_by_run[r]["input_paths"]]) for r in input_by_run)
    expected={"QE_RELAX":74,"QE_RELAX_CONTINUATION":4,"QE_SCF":5,"QE_SCF_RELAX_CHAIN":2,"QE_NEB":28}
    if dict(types)!=expected: raise SystemExit("workflow type audit differs: "+repr(dict(types)))

    # Current XML payloads were not reread; only those hashes have prior-crosswalk provenance.
    for x in rows:
        if x["category"]=="QE_COORDINATE_XML":
            if x.get("sha256_scope")!="PRIOR_AUDIT_CROSSWALK" or not x.get("known_sha256"):
                raise SystemExit("coordinate XML hash provenance missing")
        elif x.get("sha256_scope")!="CURRENT_CONTENT":
            raise SystemExit("non-XML file lacks current content hash")

    now=utc_now(); captured=summary["capture_utc_remote"]
    con=sqlite3.connect(DB,timeout=180); con.row_factory=sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON"); con.execute("PRAGMA busy_timeout=180000")
    try:
        if con.execute("SELECT count(*) FROM schema_migration WHERE version=14").fetchone()[0]!=1:
            raise SystemExit("schema migration 014 is required for null exact_command")
        if con.execute("SELECT count(*) FROM workflow_run WHERE run_id LIKE ?",(RUN_PREFIX+"%",)).fetchone()[0]!=113:
            raise SystemExit("expected 113 staged workflow_run rows")
        con.execute("BEGIN IMMEDIATE")

        source_info={}
        for p,kind,note in [
            (ROWS,"REV1_RUN_ARTIFACT_BINDINGS_CAPTURE","1,669 run-to-file bindings; full hashes for non-XML, prior-audit hashes and current stat for XML."),
            (SUMMARY,"REV1_RUN_ARTIFACT_CAPTURE_SUMMARY","Run-level capture completeness and explicitly enumerated missing evidence."),
            (INPUT_MAP,"REV1_RUN_INPUT_PATH_MAPPING","275 ordered QE input paths and their current SHA-256/metadata by historical run ID."),
            (CAPTURE/"README.md","REV1_RUN_ARTIFACT_CAPTURE_README","Read-only capture method and scope boundary."),
        ]:
            source_info[kind]=add_source(con,p,kind,note,captured)

        # Map each physical path to one canonical artifact row and its root-local path ID.
        artifact_by_path={}
        records_by_run=collections.defaultdict(list)
        for rec in rows:
            run_id=rec["run_id"]
            if run_id not in run_by_id: raise RuntimeError("unrecognized captured run "+run_id)
            root_id,root_path,rel=root_for(con,rec["path"])
            path_id_row=con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",(root_id,rel)).fetchone()
            if path_id_row: path_id=int(path_id_row[0])
            else:
                con.execute("INSERT INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",(root_id,rel,captured))
                path_id=int(con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",(root_id,rel)).fetchone()[0])
            digest=rec["sha256"]; size=int(rec["size_bytes"])
            if rec["path"] not in artifact_by_path:
                located=con.execute("""SELECT a.artifact_id,a.sha256,a.byte_size FROM artifact_location l
                    JOIN artifact a USING(artifact_id) WHERE l.path_id=? ORDER BY a.artifact_id""",(path_id,)).fetchall()
                matching=[a for a in located if a["sha256"] in (None,digest) and a["byte_size"] in (None,size)]
                if matching:
                    aid=int(matching[0]["artifact_id"])
                else:
                    conflicting=[a for a in located if a["sha256"] not in (None,digest) or a["byte_size"] not in (None,size)]
                    if conflicting: raise RuntimeError("existing path artifact conflicts with capture: "+rec["path"])
                    key=root_id+":"+rel
                    con.execute("""INSERT OR IGNORE INTO artifact
                      (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
                      VALUES('MIGRATIONBENCH',?,? ,?,'USER_CONFIRMED',?,?,'PROVENANCE_PARTIAL',?,?)""",
                      (key,rec["category"],rec["category"],digest,size,
                       "REPRODUCIBLE_INPUT" if rec["category"] in ("QE_INPUT","QE_PSEUDOPOTENTIAL") else "PERMANENT_SOURCE",
                       "Current verified artifact capture; historical association is kept separate from scientific acceptance."))
                    aid=int(con.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?",(key,)).fetchone()[0])
                row=con.execute("SELECT sha256,byte_size FROM artifact WHERE artifact_id=?",(aid,)).fetchone()
                if row["sha256"] not in (None,digest) or row["byte_size"] not in (None,size):
                    raise RuntimeError("artifact identity conflict: "+rec["path"])
                con.execute("""UPDATE artifact SET project_id=COALESCE(project_id,'MIGRATIONBENCH'),
                    sha256=COALESCE(sha256,?),byte_size=COALESCE(byte_size,?),
                    owner_status=CASE WHEN owner_status='UNKNOWN' THEN 'USER_CONFIRMED' ELSE owner_status END,
                    semantic_role=CASE WHEN semantic_role IN ('UNCLASSIFIED','UNSPECIFIED','') THEN ? ELSE semantic_role END,
                    retention_class=CASE WHEN retention_class='UNCLASSIFIED' THEN ? ELSE retention_class END,
                    scientific_status=CASE WHEN scientific_status='UNKNOWN' THEN 'PROVENANCE_PARTIAL' ELSE scientific_status END
                    WHERE artifact_id=?""",
                    (digest,size,rec["category"],
                     "REPRODUCIBLE_INPUT" if rec["category"] in ("QE_INPUT","QE_PSEUDOPOTENTIAL") else "PERMANENT_SOURCE",aid))
                con.execute("""INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                    VALUES(?,?,'PRESENT',?,?)""",(aid,path_id,captured,captured))
                con.execute("UPDATE artifact_location SET availability='PRESENT',last_verified_utc=? WHERE artifact_id=? AND path_id=?",(captured,aid,path_id))
                artifact_by_path[rec["path"]]=aid
            else:
                aid=artifact_by_path[rec["path"]]
            # One stat row per physical file in each root-targeted snapshot.
            records_by_run[run_id].append((rec,aid))
            root_snap=f"{root_id}-{SNAPSHOT_SUFFIX}"
            snapshot_notes="Targeted hash/metadata capture for retained Revision 1 run inputs, outputs and dependencies; this is not a complete tree inventory. XML payloads were not reread; their hashes are from the prior coordinate crosswalk."
            con.execute("""INSERT OR IGNORE INTO inventory_snapshot
              (snapshot_id,root_id,started_utc,finished_utc,method,tool_version,completion_state,files_seen,unreadable_count,logical_bytes,allocated_bytes,manifest_path,manifest_sha256,notes)
              VALUES(?,?,?,?,'Targeted retained QE run artifact hash/metadata capture','revision1-run-artifact-capture-v1','PARTIAL',0,0,0,0,?,?,?)""",
              (root_snap,root_id,captured,captured,str(ROWS),sha256(ROWS),snapshot_notes))
        # Snapshot totals and observations are deduplicated by root/path, not by run binding.
        unique={x["path"]:x for x in rows}
        by_root=collections.defaultdict(list)
        for rec in unique.values():
            rid,_,rel=root_for(con,rec["path"]);by_root[rid].append(rec)
        for root_id,items in by_root.items():
            snap=f"{root_id}-{SNAPSHOT_SUFFIX}"
            expected=(len(items),sum(int(x["size_bytes"]) for x in items),sum(int(x["allocated_bytes"]) for x in items))
            cur=con.execute("SELECT files_seen,logical_bytes,allocated_bytes FROM inventory_snapshot WHERE snapshot_id=?",(snap,)).fetchone()
            if tuple(cur)!=expected:
                con.execute("UPDATE inventory_snapshot SET files_seen=?,logical_bytes=?,allocated_bytes=? WHERE snapshot_id=?",(*expected,snap))
            for rec in items:
                _,_,rel=root_for(con,rec["path"])
                path_id=int(con.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",(root_id,rel)).fetchone()[0])
                scope="MANIFEST_ONLY" if rec["category"]=="QE_COORDINATE_XML" else "FULL_FILE"
                vals=(snap,path_id,int(rec["uid"]),int(rec["gid"]),rec["mode_octal"],int(rec["size_bytes"]),
                      int(rec["allocated_bytes"]),int(rec["mtime_ns"]),str(rec["inode"]),str(rec["device"]),
                      rec["sha256"],scope,int(rec.get("nlink",1)))
                con.execute("""INSERT OR IGNORE INTO inventory_observation
                  (snapshot_id,path_id,entry_type,uid,gid,mode_octal,logical_bytes,allocated_bytes,mtime_ns,inode,device,sha256,hash_scope,observation_status,link_count)
                  VALUES(?,?,'FILE',?,?,?,?,?,?,?,?,?,?,'PRESENT',?)""",vals)

        # Correct run taxonomy and link every captured file to the run(s) it actually supports.
        path_owner_runs=collections.defaultdict(set)
        for rec in rows:
            rel=rec.get("relationship",[])
            rel=rel if isinstance(rel,list) else [rel]
            if rec["category"]=="QE_COORDINATE_XML" and any(x in rel for x in ("OWN_RUN_PARSED_XML","OWN_RUN_COORDINATE_EVIDENCE")):
                path_owner_runs[rec["path"]].add(rec["run_id"])

        role_map={
          "QE_INPUT":("QE_INPUT","INPUT",1),
          "QE_OUTPUT_LOG":("QE_OUTPUT_LOG","LOG",0),
          "QE_PSEUDOPOTENTIAL":("QE_PSEUDOPOTENTIAL","INPUT",1),
          "RUN_MANIFEST":("RUN_MANIFEST","PROVENANCE",1),
          "RUNTIME_PROVENANCE":("RUNTIME_PROVENANCE","ENVIRONMENT",1),
          "SLURM_STDOUT":("SLURM_STDOUT","LOG",0),
          "SLURM_STDERR":("SLURM_STDERR","LOG",0),
          "SCHEDULER_OBSERVATION_SOURCE":("SCHEDULER_OBSERVATION_SOURCE","PROVENANCE",1),
        }
        per_run_categories=collections.defaultdict(lambda:collections.defaultdict(list))
        for rec in rows:
            run_id=rec["run_id"]; aid=artifact_by_path[rec["path"]]; cat=rec["category"]
            if cat=="QE_COORDINATE_XML":
                rel=rec.get("relationship",[]); rel=rel if isinstance(rel,list) else [rel]
                if rec["run_id"] in path_owner_runs[rec["path"]]:
                    role,direction,required=("ATOMIC_COORDINATE_XML","OUTPUT",0)
                else:
                    role,direction,required=("LATER_RUN_COORDINATE_EVIDENCE_SUPPORT","PROVENANCE",0)
                note="Current metadata plus prior-audit XML SHA. QE XML coordinates support result provenance; they do not establish convergence."
            else:
                role,direction,required=role_map[cat]
                note=f"Exact {cat} path and current SHA-256 captured at {captured}; historical equivalence is governed by run manifest and scheduler evidence."
            con.execute("""INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes)
              VALUES(?,?,?,?,?,?)""",(run_id,aid,role,direction,required,note))
            per_run_categories[run_id][cat].append(aid)

        # The two reused later-run XMLs were previously mislabelled as outputs of the earlier runs.
        later_to_producer={
          RUN_PREFIX+"30999374":RUN_PREFIX+"31006555",
          RUN_PREFIX+"31003909":RUN_PREFIX+"31010149",
        }
        for earlier,producer in later_to_producer.items():
            cw=run_by_id[earlier]
            xml_path=cw["coordinate_xml_paths_parse_nat_hash"]
            # The crosswalk stores an array of path/parse/hash evidence; resolve exact file by capture relationship.
            recs=[x for x in rows if x["run_id"]==earlier and x["category"]=="QE_COORDINATE_XML" and x.get("relationship")=="LATER_RUN_COORDINATE_EVIDENCE_SUPPORTING_CLEANUP"]
            if not recs: raise RuntimeError("later-run coordinate mapping missing for "+earlier)
            for rec in recs:
                aid=artifact_by_path[rec["path"]]
                con.execute("DELETE FROM run_artifact WHERE run_id=? AND artifact_id=? AND role='ATOMIC_COORDINATE_XML'",(earlier,aid))
                con.execute("""INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes)
                  VALUES(?,?,?,'PROVENANCE',0,?)""",
                  (earlier,aid,"LATER_RUN_COORDINATE_EVIDENCE_SUPPORT",
                   "This XML was generated by "+producer+" and supports the restart-file cleanup coordinate check; it is not an output of this run."))
                con.execute("""INSERT OR IGNORE INTO run_artifact(run_id,artifact_id,role,direction,required_for_reproduction,notes)
                  VALUES(?,?, 'ATOMIC_COORDINATE_XML','OUTPUT',0,?)""",
                  (producer,aid,"Own-run coordinate XML output; prior coordinate crosswalk SHA retained."))

        # One explicit INCOMPLETE recipe per historical run; absent commands stay NULL, never invented.
        per_run_cov=summary["per_run_coverage"]
        for run_id,inputs in input_by_run.items():
            run=con.execute("SELECT source_root_id,source_relative_path,notes FROM workflow_run WHERE run_id=?",(run_id,)).fetchone()
            root=con.execute("SELECT scoped_path FROM storage_root WHERE root_id=?",(run["source_root_id"],)).fetchone()[0]
            working=f"{root.rstrip('/')}/{run['source_relative_path']}"
            categories=per_run_cov[run_id]
            note=("Historical recipe is INCOMPLETE: exact per-job submission script/command and a verified historical Git commit are absent. "
                  "The retained input, output/log, manifest/runtime (when present), scheduler and pseudopotential paths are linked below. "
                  "This row is an evidence-gap record; it does not claim the deleted restart state or scientific result can be exactly reproduced. "
                  f"Group coverage: manifest_files={categories['manifest_file_count']}, runtime_files={categories['runtime_provenance_file_count']}, "
                  f"Slurm stdout/stderr={categories['slurm_stdout_count']}/{categories['slurm_stderr_count']}; no actual rerun was performed.")
            manifests=per_run_categories[run_id]["RUN_MANIFEST"]
            manifest_aid=manifests[0] if manifests else None
            con.execute("""INSERT OR IGNORE INTO reproduction_recipe
              (run_id,recipe_version,exact_command,working_directory,executable_version,environment_description,module_list,
               script_artifact_id,scheduler_script_artifact_id,input_manifest_artifact_id,recipe_status,last_reproduced_utc,validation_note)
              VALUES(?,?,NULL,?,NULL,?,NULL,NULL,NULL,?,'INCOMPLETE',NULL,?)""",
              (run_id,RECIPE_VERSION,working,
               "Runtime evidence exists but has not been normalized into a reviewed rerun recipe." if categories["runtime_provenance_file_count"] else "No per-run runtime provenance file was captured.",
               manifest_aid,note))
            recipe_id=con.execute("SELECT recipe_id FROM reproduction_recipe WHERE run_id=? AND recipe_version=?",(run_id,RECIPE_VERSION)).fetchone()[0]
            for idx,path in enumerate(inputs["input_paths"]):
                aid=artifact_by_path[path["path"]]
                con.execute("INSERT OR IGNORE INTO recipe_input(recipe_id,artifact_id,input_role) VALUES(?,?,?)",(recipe_id,aid,f"QE_INPUT_{idx}"))
            for aid in sorted(set(per_run_categories[run_id]["QE_PSEUDOPOTENTIAL"])):
                con.execute("INSERT OR IGNORE INTO recipe_input(recipe_id,artifact_id,input_role) VALUES(?,?,?)",(recipe_id,aid,"QE_PSEUDOPOTENTIAL"))
            for aid in sorted(set(manifests)):
                con.execute("INSERT OR IGNORE INTO recipe_input(recipe_id,artifact_id,input_role) VALUES(?,?,?)",(recipe_id,aid,"RUN_MANIFEST"))
            marker=" Retained QE input/output hash capture imported 2026-11-23; exact script/Git provenance remains open."
            if marker.strip() not in run["notes"]:
                con.execute("UPDATE workflow_run SET workflow_type=?,notes=notes||? WHERE run_id=?",(classify([x["path"] for x in inputs["input_paths"]]),marker,run_id))
            else:
                con.execute("UPDATE workflow_run SET workflow_type=? WHERE run_id=?",(classify([x["path"] for x in inputs["input_paths"]]),run_id))
            add_ref(con,source_info["REV1_RUN_ARTIFACT_BINDINGS_CAPTURE"][0],run_id,"Hashed current file/path and run-role mapping; XML digest is explicitly sourced from the prior crosswalk.")

        con.commit()
        print(json.dumps({
          "capture_rows":len(rows),"unique_paths":len(unique),"workflow_types":dict(types),
          "incomplete_recipes_added":con.execute("SELECT count(*) FROM reproduction_recipe WHERE recipe_version=?",(RECIPE_VERSION,)).fetchone()[0],
          "run_artifact_links_for_113":con.execute("SELECT count(*) FROM run_artifact WHERE run_id LIKE ?",(RUN_PREFIX+"%",)).fetchone()[0],
          "foreign_key_errors":len(con.execute("PRAGMA foreign_key_check").fetchall()),
          "snapshots":con.execute("SELECT count(*) FROM inventory_snapshot").fetchone()[0],
          "inventory_observations":con.execute("SELECT count(*) FROM inventory_observation").fetchone()[0],
        },indent=2))
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

if __name__=="__main__":main()
