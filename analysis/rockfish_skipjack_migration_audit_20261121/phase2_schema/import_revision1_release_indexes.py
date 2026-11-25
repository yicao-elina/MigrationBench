#!/usr/bin/env python3
"""Import submitted Revision 1 result, reproducibility, claim, and asset indexes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
SOURCE_DIR = AUDIT / "phase2_data/revision1_release_sources_20261110"
RELEASE_ID = "MB-REVISION1-SUBMITTED-20261110"
ROOT_ID = "RF-SCRATCH16-YI"
LOCAL_ROOT_ID = "LOCAL-DOWNLOADS"
REPO_ROOT = Path("/scratch16/pclancy3/yi/MigrationBench")
LOCAL_PDFS = {
    "MigrationBench_main_marked.pdf": Path("/Users/alina/Downloads/MigrationBench_main_marked.pdf"),
    "MigrationBench_R2R.pdf": Path("/Users/alina/Downloads/MigrationBench_R2R.pdf"),
    "MigrationBench_SI.pdf": Path("/Users/alina/Downloads/MigrationBench_SI.pdf"),
    "MigrationBench_main_LaTeX_clean.zip": Path("/Users/alina/Downloads/MigrationBench_main_LaTeX_clean.zip"),
}
PATH_TOKEN = re.compile(r"(?<![A-Za-z0-9_.-])(?:paper|release|analysis|docs|reproducibility|items|data|reviewer_[A-Za-z0-9_-]+)/[A-Za-z0-9_./@+=,-]+")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def tokens(text: str) -> list[str]:
    out = []
    for raw in PATH_TOKEN.findall(text or ""):
        value = raw.rstrip(".,;:)")
        if "*" not in value and "{" not in value and "}" not in value and "..." not in value:
            out.append(value)
    return list(dict.fromkeys(out))


def safe_status(source: str) -> str:
    v = (source or "").upper()
    if v == "FAILURE_MODE":
        return "FAILURE_MODE"
    if v == "EXCLUDE_FROM_PUBLICATION":
        return "EXCLUDE_FROM_PUBLICATION"
    if v.startswith("RECONSTRUCTION") or v == "MANUAL_SCHEMATIC":
        return "RECONSTRUCTION"
    if v.startswith("VERIFIED") and "PROVISIONAL" not in v and "LIMITATION" not in v and "GAP" not in v:
        return "LINEAGE_SUPPORTED"
    if v.startswith("FULL") or v.startswith("PASS") or v in ("PROVISIONAL_REFERENCE_WITH_AUDITED_MODEL_VALUES", "VALIDATION_GAP"):
        return "PROVENANCE_PARTIAL"
    return "UNKNOWN"


def assessment_action(source: str) -> str:
    v = (source or "").upper()
    if v == "EXCLUDE_FROM_PUBLICATION": return "EXCLUDE"
    if v == "FAILURE_MODE": return "QUALIFY"
    if "VALIDATION_GAP" in v or "PROVISIONAL" in v or "LIMITATION" in v or "WITH_KNOWN" in v or "STALE" in v or "MISMATCH" in v:
        return "QUALIFY"
    return "PENDING"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", required=True, type=Path)
    parser.add_argument("--asset-status", required=True, type=Path)
    args = parser.parse_args()
    for name in ("RESULTS_INDEX.tsv", "REPRODUCIBILITY.tsv", "claim_evidence_matrix.tsv", "KNOWN_ISSUES.md", "canonical_dft_barriers.tsv"):
        (SOURCE_DIR / name).resolve(strict=True)
    args.asset_manifest.resolve(strict=True)
    args.asset_status.resolve(strict=True)
    result_rows = read_tsv(SOURCE_DIR / "RESULTS_INDEX.tsv")
    repro_rows = read_tsv(SOURCE_DIR / "REPRODUCIBILITY.tsv")
    claim_rows = read_tsv(SOURCE_DIR / "claim_evidence_matrix.tsv")
    asset_rows = [json.loads(x) for x in args.asset_manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    asset_status = json.loads(args.asset_status.read_text(encoding="utf-8"))
    if len(result_rows) != 52 or len(repro_rows) != 36 or len(claim_rows) != 22:
        raise SystemExit("revision1 source index row count does not match the audited release")
    if asset_status.get("results_index_rows") != 52 or asset_status.get("reproducibility_rows") != 36 or asset_status.get("claim_matrix_rows") != 22:
        raise SystemExit("remote asset manifest is not tied to the expected release source counts")
    asset_by_path = {x["repository_relative_path"]: x for x in asset_rows}

    db = sqlite3.connect(DB, timeout=180)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=180000")
    try:
        if db.execute("SELECT count(*) FROM schema_migration WHERE version=12").fetchone()[0] == 0:
            db.executescript((AUDIT / "phase2_schema/012_revision1_release_and_scheduler_evidence.sql").read_text(encoding="utf-8"))
        if db.execute("SELECT count(*) FROM schema_migration WHERE version=13").fetchone()[0] == 0:
            db.executescript((AUDIT / "phase2_schema/013_duplicate_source_claim_rows.sql").read_text(encoding="utf-8"))
        now = utc()

        def source_and_artifact(path: Path, kind: str, role: str, note: str, remote_relative: str | None = None,
                                digest: str | None = None, size: int | None = None, status: str = "LINEAGE_SUPPORTED",
                                retention: str = "PUBLICATION_EVIDENCE", owner: str = "USER_CONFIRMED") -> tuple[int, int]:
            path = path.resolve(strict=True)
            actual_sha = digest or sha256(path)
            actual_size = size if size is not None else path.stat().st_size
            try:
                rel = path.relative_to(AUDIT.parents[1]).as_posix()
            except ValueError:
                rel = path.name
            source_key = "mb_r1_release:" + role + ":" + actual_sha[:20]
            db.execute("""INSERT OR IGNORE INTO registry_source
                (source_key,source_path,registry_kind,sha256,captured_utc,as_of_utc,immutable_snapshot,authority_note,notes)
                VALUES(?,?,?,?,?,?,1,?,?)""",(source_key,str(path),kind,actual_sha,now,now,
                 "Frozen Revision 1 submitted source branch and attached manuscript artifacts.",note))
            src = db.execute("SELECT source_id,source_path,sha256 FROM registry_source WHERE source_key=?",(source_key,)).fetchone()
            if not src or src[1]!=str(path) or src[2]!=actual_sha: raise RuntimeError("registry-source identity conflict: "+source_key)
            akey = "MB-R1-RELEASE-ASSET:"+role+":"+actual_sha[:20]
            db.execute("""INSERT OR IGNORE INTO artifact
                (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
                VALUES('MIGRATIONBENCH',?,?,?,?,?,?,?,?,?)""",
                (akey,kind,role,owner,actual_sha,actual_size,status,retention,note))
            art = db.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?",(akey,)).fetchone()
            if not art or art[1]!=actual_sha or art[2]!=actual_size: raise RuntimeError("artifact identity conflict: "+akey)
            if remote_relative:
                db.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",(ROOT_ID,remote_relative,now))
                pid=db.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",(ROOT_ID,remote_relative)).fetchone()[0]
                db.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)",(art[0],pid,now,now))
            return int(src[0]),int(art[0])

        def pointer_artifact(rel: str, kind: str, role: str, note: str, asset_meta: dict | None = None) -> int | None:
            meta=asset_meta or asset_by_path.get(rel,{})
            avail=meta.get("availability")
            if avail not in ("HASHED","PRESENT_NOT_HASHED_GT_100MIB"):
                return None
            digest=meta.get("sha256") or None
            size=meta.get("byte_size")
            if not digest:
                akey="MB-R1-UNHASHED-PATH:"+hashlib.sha256(rel.encode()).hexdigest()[:24]
                db.execute("""INSERT OR IGNORE INTO artifact
                    (project_id,artifact_key,artifact_kind,semantic_role,owner_status,byte_size,scientific_status,retention_class,notes)
                    VALUES('MIGRATIONBENCH',?,?,?,'USER_CONFIRMED',?,'PROVENANCE_PARTIAL','PUBLICATION_EVIDENCE',?)""",
                    (akey,kind,role,size,note+"; present but not hashed by the <=100 MiB manifest policy."))
                aid=db.execute("SELECT artifact_id FROM artifact WHERE artifact_key=?",(akey,)).fetchone()[0]
            else:
                akey="MB-R1-RELEASE-FILE:"+hashlib.sha256(rel.encode()).hexdigest()[:24]
                db.execute("""INSERT OR IGNORE INTO artifact
                    (project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
                    VALUES('MIGRATIONBENCH',?,?,?,'USER_CONFIRMED',?,?,'LINEAGE_SUPPORTED','PUBLICATION_EVIDENCE',?)""",
                    (akey,kind,role,digest,size,note))
                aid=db.execute("SELECT artifact_id,sha256,byte_size FROM artifact WHERE artifact_key=?",(akey,)).fetchone()[0]
                artifact=db.execute("SELECT sha256,byte_size FROM artifact WHERE artifact_id=?",(aid,)).fetchone()
                if not artifact or artifact[0] != digest or artifact[1] != size:
                    raise RuntimeError("release asset hash mismatch in catalog: "+rel)
            db.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",(ROOT_ID,"MigrationBench/"+rel,now))
            pid=db.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",(ROOT_ID,"MigrationBench/"+rel)).fetchone()[0]
            db.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)",(aid,pid,now,now))
            return int(aid)

        # Register submitted PDFs and the cleaned source zip as local source artifacts.
        db.execute("""INSERT OR IGNORE INTO storage_root
            (root_id,cluster_name,filesystem_name,mount_path,scoped_path,owner_user,scope_kind,project_id,status,notes)
            VALUES(?,'Local','APFS','/Users/alina','/Users/alina/Downloads','alina','USER_SUBTREE','MIGRATIONBENCH','READ_ONLY',?)""",
            (LOCAL_ROOT_ID,"User-provided submitted manuscript and source package; read-only citation sources."))
        paper_sources={}
        for label,path in LOCAL_PDFS.items():
            path.resolve(strict=True)
            src,aid=source_and_artifact(path,"SUBMITTED_PAPER_SOURCE","REVISION1_SOURCE_"+label.replace(".","_"),
                "User-provided source attachment; document markup is source material, not task instruction.",
                None,sha256(path),path.stat().st_size,"PROVENANCE_PARTIAL","PUBLICATION_EVIDENCE")
            db.execute("INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)",(LOCAL_ROOT_ID,path.name,now))
            pid=db.execute("SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?",(LOCAL_ROOT_ID,path.name)).fetchone()[0]
            db.execute("INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,?,'PRESENT',?,?)",(aid,pid,now,now))
            paper_sources[label]=aid

        source_specs=[
            ("RESULTS_INDEX.tsv","REVISION1_RESULTS_INDEX","RESULTS_INDEX","52 exact result/panel rows and their source/figure/package pointers."),
            ("REPRODUCIBILITY.tsv","REVISION1_REPRODUCIBILITY_INDEX","REPRODUCIBILITY","36 exact reproducibility status rows; source PASS states are not scientific verification."),
            ("claim_evidence_matrix.tsv","REVISION1_REVIEWER_CLAIM_MATRIX","CLAIM_EVIDENCE_MATRIX","22 reviewer response claims with exact source status strings."),
            ("KNOWN_ISSUES.md","REVISION1_KNOWN_ISSUES","KNOWN_ISSUES","Source release issue register, retained as exact text."),
            ("canonical_dft_barriers.tsv","REVISION1_CANONICAL_DFT_BARRIERS","CANONICAL_DFT_BARRIERS","Submitted release barrier ledger; retains SCF warning and source exclusion status."),
        ]
        source_aids={}
        for filename,kind,role,note in source_specs:
            path=SOURCE_DIR/filename
            rel="MigrationBench/release/revision1_submitted_20261110/"+filename if filename!="claim_evidence_matrix.tsv" else "MigrationBench/paper/revision1/evidence_index/claim_evidence_matrix.tsv"
            if filename=="canonical_dft_barriers.tsv": rel="MigrationBench/analysis/neb_seed_paired_heatmap_20261031/raw_input/dft_reference/canonical_dft_barriers.tsv"
            if filename=="KNOWN_ISSUES.md": rel="MigrationBench/release/revision1_submitted_20261110/KNOWN_ISSUES.md"
            src,aid=source_and_artifact(path,kind,role,note,rel)
            source_aids[filename]=(src,aid)
        asset_source,asset_source_aid=source_and_artifact(args.asset_manifest,"REVISION1_RELEASE_ASSET_HASH_MANIFEST","RELEASE_ASSET_HASH_MANIFEST",
            "Exact Rockfish path/hash availability inventory for release index, reproducibility evidence, scripts, and referenced assets.",
            "MigrationBench/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/revision1_release_sources_20261110/release_asset_inventory.jsonl")
        status_source,status_aid=source_and_artifact(args.asset_status,"REVISION1_RELEASE_ASSET_HASH_SUMMARY","RELEASE_ASSET_HASH_STATUS",
            "Count/status summary for the release source file availability and <=100 MiB hashing pass.",
            "MigrationBench/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/revision1_release_sources_20261110/release_asset_inventory_status.json")

        results_aid=source_aids["RESULTS_INDEX.tsv"][1]
        db.execute("""INSERT OR IGNORE INTO git_repository
            (repository_key,local_path,remote_url,project_id,ownership_status,canonical_branch,notes)
            VALUES('rockfish:/scratch16/pclancy3/yi/MigrationBench',?, 'https://github.com/yicao-elina/MigrationBench.git', 'MIGRATIONBENCH','USER','main',?)""",
            (str(REPO_ROOT),"Fresh GitHub clone in the dedicated scratch16 project area. Release branch f924ca6c was fetched and merged into a local codex integration candidate; this is not yet pushed."))
        repo_id=db.execute("SELECT repository_id FROM git_repository WHERE repository_key='rockfish:/scratch16/pclancy3/yi/MigrationBench'").fetchone()[0]
        db.execute("""INSERT OR IGNORE INTO paper_release
            (release_id,project_id,release_label,manuscript_commit,repository_id,overleaf_version,released_utc,status,manifest_artifact_id,notes)
            VALUES(?,'MIGRATIONBENCH',?,'334260cdda3869d55999081cd610b34bfba53257',?,NULL,NULL,'REVIEW',?,?)""",
            (RELEASE_ID,"Revision 1 submitted package (2026-11-10)",repo_id,results_aid,
             "Source paper export commit 334260c, frozen release branch commit f924ca6c. The attached marked main/R2R/SI PDFs are user-provided source artifacts; the clean LaTeX zip is not byte-identical to the frozen sn-article-final.tex. No Overleaf version or journal acceptance is asserted. Source status rows remain distinct from catalog scientific assessment."))

        # Make all referenced small/medium files queryable by canonical Rockfish path.
        for rel,meta in asset_by_path.items():
            if meta.get("availability") in ("HASHED","PRESENT_NOT_HASHED_GT_100MIB"):
                pointer_artifact(rel,"PAPER_RELEASE_FILE","RELEASE_EVIDENCE_FILE","Source-referenced Revision 1 release asset.",meta)

        result_id_map={}
        for r in result_rows:
            section=r.get("section",""); item=r.get("item_id",""); panel=r.get("panel","")
            db.execute("""INSERT OR IGNORE INTO paper_release_item
                (release_id,source_item_id,section,panel,content,final_asset_path,final_asset_sha256,generator,plotted_data,upstream_data,raw_location,model_refs,calculation_type,source_status,notes,item_dir,reproduce_script,reproduction_status)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (RELEASE_ID,item,section,panel,r.get("content","") or "",r.get("final_asset") or None,r.get("asset_sha256") or None,
                 r.get("generator") or None,r.get("plotted_data") or None,r.get("upstream_data") or None,r.get("raw_location") or None,
                 r.get("models") or None,r.get("calc_type") or None,r.get("status","") or "",r.get("note","") or "",
                 r.get("item_dir") or None,r.get("reproduce_script") or None,r.get("reproduction") or None))
            item_row=db.execute("SELECT release_item_id FROM paper_release_item WHERE release_id=? AND source_item_id=? AND section=? AND panel=?",
                                (RELEASE_ID,item,section,panel)).fetchone()
            rid=int(item_row[0]); result_id_map[(item,section,panel)]=rid
            def link_path(path_text,role):
                if not path_text: return
                meta=asset_by_path.get(path_text,{})
                if meta:
                    availability={"HASHED":"PRESENT","PRESENT_NOT_HASHED_GT_100MIB":"PRESENT","MISSING":"MISSING"}.get(meta.get("availability"),"EXTERNAL_POINTER")
                    aid=pointer_artifact(path_text,"PAPER_RELEASE_FILE",role,"Source-indexed Revision 1 evidence path.",meta) if availability=="PRESENT" else None
                else:
                    availability="EXTERNAL_POINTER" if path_text.startswith(("rockfish:","/")) else "NOT_MATERIALIZED"
                    aid=None
                source_note="Exact source index field; paths containing globs or external cluster pointers remain pointers."
                if role=="FINAL_ASSET":
                    declared=r.get("asset_sha256","")
                    actual=(meta.get("sha256") or "")
                    source_note += " Declared asset SHA-256="+(declared or "NONE")+"; observed="+(actual or "NOT_HASHED_OR_MISSING")+"; match="+str(bool(declared and actual and declared==actual))+"."
                db.execute("""INSERT OR IGNORE INTO paper_release_item_evidence
                    (release_item_id,artifact_id,source_path,evidence_role,availability,relationship,notes)
                    VALUES(?,?,?,?,?,?,?)""",(rid,aid,path_text,role,availability,"SOURCE_INDEX_REFERENCE",
                    source_note))
            link_path(r.get("final_asset",""),"FINAL_ASSET")
            link_path(r.get("reproduce_script",""),"REPRODUCTION_SCRIPT")
            for field,role in (("generator","GENERATOR"),("plotted_data","PLOTTED_DATA"),("upstream_data","UPSTREAM_DATA")):
                for rel in tokens(r.get(field,"")):
                    link_path(rel,role)

        for r in repro_rows:
            item=r.get("item_id",""); section=r.get("section",""); item_dir=r.get("item_dir","")
            linked=db.execute("SELECT release_item_id FROM paper_release_item WHERE release_id=? AND source_item_id=? AND section=? ORDER BY panel LIMIT 1",(RELEASE_ID,item,section)).fetchone()
            result=(r.get("result","") or "").upper()
            assessment="LINEAGE_SUPPORTED" if result=="PASS" else ("PROVENANCE_PARTIAL" if result in ("PASS_KNOWN_ISSUE","EXCEPTION") else "UNKNOWN")
            db.execute("""INSERT OR IGNORE INTO paper_reproducibility_record
                (release_id,source_item_id,section,item_dir,source_result,scripts,evidence,checks_count,inputs_count,problems,linked_release_item_id,assessment_status,notes)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (RELEASE_ID,item,section,item_dir,r.get("result","") or "",r.get("scripts","") or "",r.get("evidence","") or "",
                 int(r["checks"]) if r.get("checks","").isdigit() else None,int(r["inputs"]) if r.get("inputs","").isdigit() else None,
                 r.get("problems","") or "",linked[0] if linked else None,assessment,
                 "Source reproduction status is preserved verbatim; PASS means the declared computational checks passed, not that scientific claims are independently validated."))
            if linked:
                rid=int(linked[0])
                for field,role in (("scripts","REPRODUCTION_SCRIPT"),("evidence","REPRODUCTION_EVIDENCE")):
                    for rel in tokens(r.get(field,"")):
                        meta=asset_by_path.get(rel,{})
                        avail={"HASHED":"PRESENT","PRESENT_NOT_HASHED_GT_100MIB":"PRESENT","MISSING":"MISSING"}.get(meta.get("availability"),"EXTERNAL_POINTER")
                        aid=pointer_artifact(rel,"PAPER_RELEASE_FILE",role,"Reproduction status evidence file.",meta) if avail=="PRESENT" else None
                        db.execute("INSERT OR IGNORE INTO paper_release_item_evidence(release_item_id,artifact_id,source_path,evidence_role,availability,relationship,notes) VALUES(?,?,?,?,?,?,?)",
                                   (rid,aid,rel,role,avail,"REPRODUCIBILITY_TSV_REFERENCE","Referenced by the release reproducibility index."))

        # Reviewer response matrix rows remain separate source observations and conservative claims.
        matrix_source_id=source_aids["claim_evidence_matrix.tsv"][0]
        claim_occurrences={}
        for r in claim_rows:
            source_claim=r.get("claim_id","")
            claim_occurrences[source_claim]=claim_occurrences.get(source_claim,0)+1
            source_row_key=f"{source_claim}#{claim_occurrences[source_claim]:02d}"
            catalog_claim="MB-R1-REVIEW-"+source_row_key
            source_status=r.get("scientific_status","")
            catalog_status=safe_status(source_status)
            action=assessment_action(source_status)
            db.execute("""INSERT OR IGNORE INTO paper_claim
                (claim_id,release_id,manuscript_path,section_or_panel,claim_text,scientific_status,publication_action,notes)
                VALUES(?,?,?,?,?,?,?,?)""",
                (catalog_claim,RELEASE_ID,r.get("manuscript_location","") or None,r.get("claim_id","") or None,
                 r.get("question","") or "",catalog_status,action,
                 "Imported from reviewer response matrix. Exact source status is preserved in paper_release_claim_observation; normalized status is conservative and is not an independent scientific re-evaluation."))
            db.execute("""INSERT OR IGNORE INTO paper_release_claim_observation
                (release_id,source_row_key,source_claim_id,reviewer,question,source_scientific_status,primary_artifact,source_or_table,rebuild_script,manuscript_location,response_location,boundary,catalog_claim_id,notes)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (RELEASE_ID,source_row_key,source_claim,r.get("reviewer","") or "",r.get("question","") or "",source_status,
                 r.get("primary_artifact","") or "",r.get("source_or_table","") or "",r.get("rebuild_script","") or "",
                 r.get("manuscript_location","") or "",r.get("response_location","") or "",r.get("boundary","") or "",catalog_claim,
                 "Source status is a response-matrix assertion, not proof. Artifact paths remain explicit and can be queried even when unavailable."))
            refs=[]
            primary=r.get("primary_artifact","")
            for rel in tokens(primary):
                meta=asset_by_path.get(rel,{})
                availability=meta.get("availability")
                aid=pointer_artifact(rel,"PAPER_CLAIM_EVIDENCE","REVIEWER_CLAIM_PRIMARY_ARTIFACT","Primary artifact cited by source reviewer claim.",meta) if availability in ("HASHED","PRESENT_NOT_HASHED_GT_100MIB") else None
                if aid:
                    support="EXCLUDED" if source_status=="EXCLUDE_FROM_PUBLICATION" else ("QUALIFIES" if catalog_status in ("PROVENANCE_PARTIAL","RECONSTRUCTION","FAILURE_MODE") else "SUPPORTS")
                    db.execute("INSERT OR IGNORE INTO claim_evidence(claim_id,artifact_id,run_id,evidence_role,support_status,notes) VALUES(?,?,NULL,'SOURCE_MATRIX_PRIMARY_ARTIFACT',?,?)",
                               (catalog_claim,aid,support,"Path and asset are imported from the source matrix; relation status reflects source wording only."))
                    refs.append({"path":rel,"artifact_id":aid,"availability":availability})
                else:
                    refs.append({"path":rel,"artifact_id":None,"availability":availability or "EXTERNAL_POINTER"})
            db.execute("""INSERT OR IGNORE INTO paper_claim_source_observation
                (source_id,source_claim_key,claim_id,source_status,source_publication_action,source_job_or_run,source_target,source_evidence_refs,observed_utc)
                VALUES(?,?,?,?,?,?,?,?,?)""",
                (matrix_source_id,source_row_key,catalog_claim,source_status,action,"",r.get("manuscript_location","") or "",
                 json.dumps(refs,sort_keys=True),now))

        # Specific attached-manuscript statement whose referenced DFT anchor is excluded by the release's own gate.
        main_claim="MB-R1-MAIN-IN-GAP-DFT-BARRIER-1P52EV"
        db.execute("""INSERT OR IGNORE INTO paper_claim
            (claim_id,release_id,manuscript_path,section_or_panel,claim_text,scientific_status,publication_action,notes)
            VALUES(?,?,?,?,?,'PROVENANCE_PARTIAL','QUALIFY',?)""",
            (main_claim,RELEASE_ID,"MigrationBench_main_marked.pdf","Results; Fig. 4/Table S3",
             "The submitted manuscript reports an approximately 1.52 eV canonical in-gap DFT-NEB barrier for the 81-atom, 10-image path.",
             "The source release lists 1.524299 eV, completed Slurm state, SCF warning TRUE, and scientific_status EXCLUDE_FROM_PUBLICATION. The QE source trail is traceable, but the stated acceptance gate is not met; do not normalize the value to a verified reference."))
        dft_aid=source_aids["canonical_dft_barriers.tsv"][1]
        db.execute("INSERT OR IGNORE INTO claim_evidence(claim_id,artifact_id,run_id,evidence_role,support_status,notes) VALUES(?,?,NULL,'CANONICAL_DFT_BARRIER_LEDGER','QUALIFIES',?)",
                   (main_claim,dft_aid,"Source ledger supports the numeric value and contradicts an unconditional accepted-reference interpretation via SCF warning and EXCLUDE_FROM_PUBLICATION status."))
        for label,aid in paper_sources.items():
            if label in ("MigrationBench_main_marked.pdf","MigrationBench_SI.pdf"):
                db.execute("INSERT OR IGNORE INTO claim_evidence(claim_id,artifact_id,run_id,evidence_role,support_status,notes) VALUES(?,?,NULL,'SUBMITTED_PAPER_SOURCE','QUALIFIES',?)",
                           (main_claim,aid,"The attached submitted source carries the reported barrier; the file is treated as a claim source, not a computational validation."))

        counts={
            "release_items":db.execute("SELECT count(*) FROM paper_release_item WHERE release_id=?",(RELEASE_ID,)).fetchone()[0],
            "reproducibility_records":db.execute("SELECT count(*) FROM paper_reproducibility_record WHERE release_id=?",(RELEASE_ID,)).fetchone()[0],
            "review_claims":db.execute("SELECT count(*) FROM paper_release_claim_observation WHERE release_id=?",(RELEASE_ID,)).fetchone()[0],
            "release_evidence_links":db.execute("SELECT count(*) FROM paper_release_item_evidence WHERE release_item_id IN (SELECT release_item_id FROM paper_release_item WHERE release_id=?)",(RELEASE_ID,)).fetchone()[0],
            "main_dft_claim_status":db.execute("SELECT scientific_status||'/'||publication_action FROM paper_claim WHERE claim_id=?",(main_claim,)).fetchone()[0],
        }
        if counts["release_items"] != 52 or counts["reproducibility_records"] != 36 or counts["review_claims"] != 22:
            raise RuntimeError("release index import count mismatch: "+json.dumps(counts))
        if db.execute("PRAGMA foreign_key_check").fetchall(): raise RuntimeError("foreign_key_check failed")
        db.commit()
        print(json.dumps({"release_id":RELEASE_ID,"state":"IMPORTED","source_commit":"334260cdda3869d55999081cd610b34bfba53257",
                          "frozen_release_branch_commit":"f924ca6c11236c4b75469e290a0abff884c56c02",
                          "asset_manifest_status":asset_status.get("availability_counts"),"counts":counts},sort_keys=True))
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
