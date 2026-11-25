#!/usr/bin/env python3
"""Create an additive, conservative R1 lineage overlay r2 from audited maps."""
import csv
import argparse
import hashlib
import json
import mimetypes
import shutil
import sqlite3
from pathlib import Path

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument('--bundle-dir', type=Path, required=True,
                    help='Directory containing source/, maps/, and the target overlay bundle')
PARSER.add_argument('--rockfish-yi-root', type=Path, required=True,
                    help='Personal Rockfish root containing the mapped R1 and MD folders')
ARGS = PARSER.parse_args()
BASE = ARGS.bundle_dir.resolve()
MAPS = BASE / 'maps'
SRC_DB = BASE / 'source' / 'migrationbench_r1_lineage_overlay_20261125.sqlite'
OUT_DB = BASE / 'migrationbench_r1_lineage_overlay_20261125_r2.sqlite'
ROOT = ARGS.rockfish_yi_root.resolve()
# Freeze the capture timestamp so the same source DB and TSV inputs reproduce
# byte-identical logical rows on a clean rerun.
TS = '2026-11-25T23:11:55+00:00'

def rows(path):
    raw = path.read_text()
    # One generated TSV contains literal backslash-t separators; normalize only
    # that file format quirk before parsing, without changing field contents.
    if '\\t' in raw.splitlines()[0] and '\t' not in raw.splitlines()[0]:
        raw = raw.replace('\\t', '\t')
    return list(csv.DictReader(raw.splitlines(), delimiter='\t'))

def ident(prefix, key):
    return prefix + hashlib.sha256(key.encode()).hexdigest()[:24]

def mime(path):
    return mimetypes.guess_type(path)[0] or 'application/octet-stream'

def add_artifact(db, key, role, path, scope, sha, size, availability, notes):
    aid = ident('artifact_', key)
    db.execute('''INSERT INTO lineage_artifacts
      (artifact_id,artifact_key,artifact_role,source_path,path_scope,sha256,byte_size,mime_type,
       availability_status,scientific_status,captured_at,notes)
      VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''',
      (aid,key,role,path,scope,sha,size,mime(path),availability,'PROVENANCE_PARTIAL',TS,notes))
    return aid

if OUT_DB.exists():
    raise SystemExit(f'refusing to overwrite {OUT_DB}')
src = sqlite3.connect(SRC_DB)
dst = sqlite3.connect(OUT_DB)
src.backup(dst)
dst.execute('PRAGMA foreign_keys=ON')
dst.execute('BEGIN IMMEDIATE')

campaigns = [
 ('revision1_manifest_inventory_20261125','Revision 1 input/run manifest inventory 2026-11-25',
  str(ROOT/'revision1_migrationbench_inputs')+'; '+str(ROOT/'revision1_migrationbench_runs'), 'PARTIAL',
  '73 input roots and 540 run roots. 494 input and 1,372 run metadata files under strict basename rule; headline count discrepancy unresolved. Inventory is not a claim/run closure.'),
 ('revision1_legacy_qe_61_neb_20261125','Revision 1 legacy 61-atom QE NEB outputs',
  str(ROOT/'revision1_migrationbench_pipeline/data_raw/historical_neb/legacy_61'), 'NEEDS_AUDIT',
  'Six outputs located by legacy path and owner. No crosswalk expected SHA; no convergence or scientific acceptance inferred.'),
 ('md_continuation_400ps_rockfish','MD continuation 400 ps Rockfish',
  str(ROOT/'md_continuation_400ps_rockfish'), 'PARTIAL',
  'Read-only mapping; model/input hashes and scientific acceptance are unresolved.'),
 ('md_continuation_1ns_rockfish','MD continuation 1 ns Rockfish',
  str(ROOT/'md_continuation_1ns_rockfish'), 'PARTIAL',
  'Read-only mapping; model/input hashes, launch script and output-generation chronology are unresolved.'),
]
for cid,title,path,status,notes in campaigns:
    if dst.execute('SELECT 1 FROM campaigns WHERE campaign_id=?',(cid,)).fetchone():
        raise SystemExit(f'campaign ID already exists: {cid}')
    dst.execute('INSERT INTO campaigns(campaign_id,title,campaign_path,status,notes) VALUES(?,?,?,?,?)',
                (cid,title,path,status,notes))

artifact_ids = {}
counts = {'manifest_artifacts':0,'legacy_qe_artifacts':0,'md_artifacts':0,'audit_package_artifacts':0,
          'legacy_result_links':0,'md_runs':0,'md_run_artifact_links':0}

# Hash-verified manifest/provenance files only; payload references remain separate and unresolved.
index_path = MAPS/'revision1_migrationbench_manifest_index_20261125.tsv'
for r in rows(index_path):
    root = ROOT/('revision1_migrationbench_inputs' if r['kind']=='input' else 'revision1_migrationbench_runs')
    path = root/r['manifest_relpath']
    if len(r['manifest_sha256']) != 64:
        continue
    key = f"r1-manifest:{r['kind']}:{r['manifest_relpath']}:{r['manifest_sha256']}"
    artifact_ids[str(path)] = add_artifact(dst,key,'MANIFEST',str(path),'SCRATCH16',r['manifest_sha256'],
       int(r['manifest_bytes']),'PRESENT',
       f"R1 manifest audit 2026-11-25; kind={r['kind']}; read_status={r['read_status']}; "
       f"path_refs={r['path_reference_count']}; sha_refs={r['embedded_sha256_count']}; no run acceptance inferred.")
    counts['manifest_artifacts'] += 1

# Legacy QE outputs: exact discovered paths, but their old expected digests were not recorded.
qe_tsv = MAPS/'MIGRATIONBENCH_LEGACY_QE_RAW_NEBS_SCOPED_SEARCH_20261125.tsv'
qe_rows = rows(qe_tsv)
for r in qe_rows:
    path = r['found_path']
    key = f"r1-legacy-qe:{r['path_id']}:{r['sha256']}"
    aid = add_artifact(dst,key,'OUTPUT',path,'SCRATCH16',r['sha256'],int(r['size_bytes']),'PRESENT',
      f"Located by legacy path ID {r['path_id']}; owner={r['owner']}:{r['group']}; "
      f"mtime={r['mtime_iso']}; crosswalk_expected_sha={r['crosswalk_expected_sha']}; "
      'location/hash recorded; NEB convergence and scientific validity not assessed.')
    artifact_ids[path] = aid
    counts['legacy_qe_artifacts'] += 1
    for direction in ('forward','reverse'):
        result_key=f"R1X:Table S5:table:{r['path_id']} / {direction}"
        result=dst.execute('SELECT result_id FROM lineage_results WHERE result_key=?',(result_key,)).fetchone()
        if not result:
            raise SystemExit(f'missing expected existing result row {result_key}')
        dst.execute('''INSERT INTO result_artifact_links
          (result_id,artifact_id,artifact_role,relation,evidence_status,note)
          VALUES(?,?,?,?,?,?)''',
          (result[0],aid,'SOURCE_ROW','DERIVED_FROM','PARTIAL',
           'Path ID and legacy path resolve to this file; no expected SHA was recorded in the crosswalk, so exact historical identity and producing run remain unverified.'))
        counts['legacy_result_links'] += 1

# MD continuation file payload map (37 files), keeping observed workflow labels in notes.
md_tsv = MAPS/'md_continuation'/'ARTIFACT_FILE_MAP.tsv'
md_roles={'trajectory_xyz':'OUTPUT','slurm_log_err':'LOG','slurm_log_out':'LOG','launch_script':'CONFIG'}
md_rows=rows(md_tsv)
for r in md_rows:
    role=md_roles.get(r['artifact_kind'],'OTHER')
    key=f"md-continuation:{r['campaign_id']}:{r['relative_path']}:{r['sha256']}"
    aid=add_artifact(dst,key,role,r['path'],'SCRATCH16',r['sha256'],int(r['size_bytes']),'PRESENT',
      f"Campaign={r['campaign_id']}; mapped artifact_kind={r['artifact_kind']}; "
      f"observed_status={r['artifact_status']}; source_checkpoint_sha={r['source_checkpoint_sha256']}; "
      f"source_input_sha={r['source_input_sha256']}; paper_link={r['paper_link_status']}; "
      'mapping observation only; no scientific acceptance inferred.')
    artifact_ids[r['path']]=aid
    counts['md_artifacts'] += 1

run_tsv = MAPS/'md_continuation'/'RUN_ATTEMPT_MAP.tsv'
md_runs = rows(run_tsv)
run_ids = {}
for r in md_runs:
    run_key=f"md-continuation:{r['run_attempt_id']}"
    rid=ident('run_',run_key)
    cid=r['campaign_id']
    dst.execute('''INSERT INTO lineage_runs
      (run_id,run_key,campaign_id,run_kind,protocol_key,model_id,model_checkpoint_sha256,seed,sample_key,
       attempt_number,parent_run_id,scheduler_job_id,scheduler_status,run_status,scientific_status,
       started_at,finished_at,notes)
      VALUES(?,?,?,'MD','',NULL,'','','',0,NULL,?,'UNKNOWN','UNKNOWN','PROVENANCE_PARTIAL',NULL,NULL,?)''',
      (rid,run_key,cid,r['slurm_job_task_id'],
       f"Observed status={r['observed_status']}; summary={r['evidence_summary']}; "
       f"input={r['input_path']} [{r['input_sha256']}]; model={r['model_path']} [{r['model_sha256']}]; "
       f"trajectory={r['trajectory_output_path']} [{r['trajectory_output_sha256']}]; "
       f"scientific_acceptance={r['scientific_acceptance_status']}; paper_claim_link={r['paper_claim_link']}. "
       'Database run_status and scheduler_status deliberately UNKNOWN: map records observations, not live scheduler or scientific validation.'))
    run_ids[r['run_attempt_id']]=rid
    counts['md_runs'] += 1
    # Link only artifacts with exact path evidence from the attempt map.
    candidates=[]
    for path in (r['trajectory_output_path'],):
        if path and path in artifact_ids:
            candidates.append((artifact_ids[path],'OUTPUT','attempt map names exact trajectory path; output generation chronology/acceptance may remain unresolved.'))
    job=r['slurm_job_task_id']
    for a in md_rows:
        if a['campaign_id']==cid and a['artifact_kind'] in ('slurm_log_out','slurm_log_err') and job and job in Path(a['path']).name:
            candidates.append((artifact_ids[a['path']],'LOG','job/task identifier matches log basename; log contents retain observed status.'))
    for aid,role,note in candidates:
        dst.execute('INSERT INTO run_artifact_links(run_id,artifact_id,artifact_role,evidence_status,note) VALUES(?,?,?,?,?)',
                    (rid,aid,role,'PARTIAL',note))
        counts['md_run_artifact_links'] += 1

# Register immutable audit sidecars themselves so their evidence maps can be found from SQL.
sidecars = [
 (MAPS/'MIGRATIONBENCH_R1_MANIFEST_RUN_LINEAGE_AUDIT_20261125.md','r1-manifest-audit-report'),
 (MAPS/'MIGRATIONBENCH_LEGACY_QE_RAW_NEBS_SCOPED_SEARCH_20261125.md','legacy-qe-discovery-report'),
 (qe_tsv,'legacy-qe-discovery-tsv'),
 (MAPS/'md_continuation'/'README.md','md-continuation-map-readme'),
 (md_tsv,'md-continuation-artifact-map'),
 (run_tsv,'md-continuation-run-attempt-map'),
 (MAPS/'SCRATCH16_LMAX1_G3_PROVENANCE_MAP_20261125.md','lmax1-g3-map-report'),
 (MAPS/'SCRATCH16_LMAX1_G3_PROVENANCE_MAP_20261125.tsv','lmax1-g3-map-tsv'),
]
for f,key in sidecars:
    sha=hashlib.sha256(f.read_bytes()).hexdigest()
    # These are archived in the audited Rockfish data archive.
    if key.startswith('r1-manifest'):
        p=str(ROOT/'MigrationBench_data_archive/revision1_lineage_overlays/20261125'/f.name)
    elif key.startswith('legacy-qe'):
        p=str(ROOT/'MigrationBench_data_archive/revision1_lineage_overlays/20261125/legacy_qe_outputs_discovery'/f.name)
    elif key.startswith('md-continuation'):
        p=str(ROOT/'MigrationBench_data_archive/storage_audits/mlip_md_payload_map_20261125'/f.name)
    else:
        p=str(ROOT/'MigrationBench_data_archive/storage_audits/lmax1_g3_provenance_map_20261125'/f.name)
    artifact_ids[p]=add_artifact(dst,'audit-sidecar:'+key+':'+sha,'OTHER',p,'SCRATCH16',sha,f.stat().st_size,'ARCHIVED',
       'Immutable audit map/report; checksum was locally computed and the corresponding Rockfish package was separately checksum-verified.')
    counts['audit_package_artifacts'] += 1

# Add Lmax1 diagnostic campaign records only; no synthetic run rows are invented.
lmax_rows=rows(MAPS/'SCRATCH16_LMAX1_G3_PROVENANCE_MAP_20261125.tsv')
for r in lmax_rows:
    cid=r['campaign_id']
    if dst.execute('SELECT 1 FROM campaigns WHERE campaign_id=?',(cid,)).fetchone():
        continue
    dst.execute('INSERT INTO campaigns(campaign_id,title,campaign_path,status,notes) VALUES(?,?,?,?,?)',
      (cid,r['project_group']+' — '+cid,r['root'],'PARTIAL',
       f"Files={r['file_count']}; logical_bytes={r['logical_bytes']}; observed={r['observed_status_summary']}; "
       f"publication_status={r['publication_status']}; cleanup={r['cleanup_disposition']}. "
       f"Parent/source={r['parent_source']}; row-level attempt/payload lineage remains incomplete."))

dst.commit()
dst.execute('PRAGMA foreign_keys=ON')
quick=dst.execute('PRAGMA quick_check').fetchone()[0]
fk=list(dst.execute('PRAGMA foreign_key_check'))
if quick!='ok' or fk:
    raise SystemExit(f'integrity failed quick_check={quick!r} fk={len(fk)}')

counts.update({
 'campaigns_total':dst.execute('SELECT count(*) FROM campaigns').fetchone()[0],
 'lineage_artifacts_total':dst.execute('SELECT count(*) FROM lineage_artifacts').fetchone()[0],
 'lineage_runs_total':dst.execute('SELECT count(*) FROM lineage_runs').fetchone()[0],
 'lineage_results_total':dst.execute('SELECT count(*) FROM lineage_results').fetchone()[0],
 'result_artifact_links_total':dst.execute('SELECT count(*) FROM result_artifact_links').fetchone()[0],
 'run_artifact_links_total':dst.execute('SELECT count(*) FROM run_artifact_links').fetchone()[0],
})
dst.close(); src.close()
manifest={
 'created_at_utc':TS,'source_overlay':str(SRC_DB),'source_overlay_sha256':hashlib.sha256(SRC_DB.read_bytes()).hexdigest(),
 'output_overlay':str(OUT_DB),'output_overlay_sha256':hashlib.sha256(OUT_DB.read_bytes()).hexdigest(),
 'output_bytes':OUT_DB.stat().st_size,'quick_check':quick,'foreign_key_violations':len(fk),
 'counts':counts,'boundary':'Additive evidence inventory only; no claim statuses changed and no uncertain result-to-run PRODUCED_BY links created.'
}
(BASE/'migrationbench_r1_lineage_overlay_20261125_r2_import_receipt.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps(manifest,indent=2,sort_keys=True))
