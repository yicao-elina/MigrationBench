#!/usr/bin/env python3
"""Build an additive R4 lineage snapshot with organized archive copies and symlink correction."""
import argparse, csv, hashlib, json, mimetypes, platform, shutil, sqlite3
from pathlib import Path

CAMPAIGN_ID='mace_from_scratch_ablation_md_20261005'
CAPTURED_AT='2026-11-25'
EXPECTED_R3_SHA256='a3d62faf39f54e66fca3cfe37616c9d804c20ed8f503bef6eb1e5dfa8700140d'
ARCHIVE_REL='campaigns/cr_sb2te3/mace_from_scratch_ablation_md_20261005'
TYPE_ROLE={'run_manifest':('MANIFEST','MANIFEST'),'status':('LOG','LOG'),'trajectory':('OUTPUT','OUTPUT'),
           'shared_input':('INPUT','INPUT'),'model_checkpoint':('CHECKPOINT','CHECKPOINT'),
           'postprocess_csv':('DERIVED_DATA','METRIC'),'postprocess_json':('METRIC','METRIC')}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def keyhash(*parts): return hashlib.sha256('\0'.join(parts).encode()).hexdigest()[:24]
def strip_prefix(value,prefix): return value[len(prefix):] if value.startswith(prefix) else value
def safe_unlink(path):
    try: path.unlink()
    except OSError: pass

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--r3-db',type=Path,required=True)
    ap.add_argument('--archive-copy-manifest',type=Path,required=True)
    ap.add_argument('--symlink-observation',type=Path,required=True)
    ap.add_argument('--archive-receipt',type=Path,required=True)
    ap.add_argument('--archive-sha256sums',type=Path,required=True)
    ap.add_argument('--output-db',type=Path,required=True)
    ap.add_argument('--archive-root',type=Path,help='Optional live archive root; if supplied, hash every archived payload.')
    a=ap.parse_args()
    base=a.r3_db.resolve(); out=a.output_db.resolve(); manp=a.archive_copy_manifest.resolve()
    symp=a.symlink_observation.resolve(); recp=a.archive_receipt.resolve(); sumsp=a.archive_sha256sums.resolve()
    if out.exists(): raise SystemExit(f'refusing to overwrite {out}')
    if sha256(base)!=EXPECTED_R3_SHA256:
        raise SystemExit(f'R3 base snapshot SHA mismatch: {base}')
    manifest=json.loads(manp.read_text()); symlink=json.loads(symp.read_text()); receipt=json.loads(recp.read_text())
    files=manifest['files']
    if manifest['campaign_id']!=CAMPAIGN_ID or len(files)!=37 or manifest['payload_bytes']!=1819048604:
        raise SystemExit('unexpected frozen campaign archive manifest identity')
    if receipt['sha256sum_check']!='PASS' or receipt['payload_files']!=37 or receipt['payload_bytes']!=manifest['payload_bytes']:
        raise SystemExit('archive copy receipt does not verify expected campaign payload')
    if receipt['manifest_sha256']!=sha256(manp) or receipt['symlink_sidecar_sha256']!=sha256(symp):
        raise SystemExit('archive receipt manifest/sidecar SHA mismatch')
    if symlink['symlink_lstat_bytes']!=145 or symlink['target_stat_bytes']!=320651971 or symlink['target_sha256']!='bb42c493fb0c997af3957a89dfde03e3e5a67ebcfd37f5eced8f722dad546161':
        raise SystemExit('unexpected input symlink/target observation')
    sum_map={}
    for line in sumsp.read_text().splitlines():
        h,name=line.split(None,1);sum_map[name.strip()]=h
    for p in [manp,symp,recp]:
        # These are archive-root metadata files; validate by their basename.
        if p.name not in sum_map or sha256(p)!=sum_map[p.name]:raise SystemExit(f'checksum mismatch: {p.name}')
    # SHA256SUMS cannot contain its own digest. Its content is instead pinned
    # by the archive-copy receipt and compared byte-for-byte at archive-root.
    sums_sha=sha256(sumsp)
    for r in files:
        rel=r['archive_relative_path']
        if rel not in sum_map or sum_map[rel]!=r['verified_sha256']:
            raise SystemExit(f'payload hash is not pinned in SHA256SUMS: {rel}')
    if a.archive_root:
        root=a.archive_root.resolve()
        for r in files:
            p=root/r['archive_relative_path']
            if p.stat().st_size!=int(r['verified_bytes']) or sha256(p)!=r['verified_sha256']:
                raise SystemExit(f'archive payload mismatch: {p}')
        for name in ['ARCHIVE_COPY_MANIFEST.json','SOURCE_INPUT_SYMLINK.json','ARCHIVE_COPY_RECEIPT.json','README.md']:
            p=root/name
            if p.name not in sum_map or sha256(p)!=sum_map[p.name]:raise SystemExit(f'archive control mismatch: {p}')
        if (root/'SHA256SUMS').read_bytes()!=sumsp.read_bytes():raise SystemExit('archive SHA256SUMS differs from frozen local integrity list')
        payload_verification='ALL_37_PAYLOADS_HASH_VERIFIED'
    else:
        payload_verification='REMOTE_ARCHIVE_RECEIPT_AND_MANIFEST_VERIFIED'
    shutil.copy2(base,out)
    con=sqlite3.connect(str(out));con.execute('PRAGMA foreign_keys=ON')
    new_ids=[]
    try:
      with con:
        link_path=symlink['symlink_path'];target_path=symlink['dereferenced_target_path']
        wrong=con.execute("SELECT artifact_id FROM lineage_artifacts WHERE source_path=? AND sha256=? AND byte_size=? AND artifact_key LIKE 'mace_terminal_failure_r3:%'",
                          (link_path,symlink['target_sha256'],symlink['symlink_lstat_bytes'])).fetchall()
        if len(wrong)!=1:raise SystemExit(f'expected one R3 combined symlink/content row, found {len(wrong)}')
        symlink_artifact_id=wrong[0][0]
        con.execute("UPDATE lineage_artifacts SET sha256='',byte_size=?,mime_type='inode/symlink',availability_status='NOT_CHECKED',scientific_status='NOT_ASSESSED',notes=notes||? WHERE artifact_id=?",
                    (symlink['symlink_lstat_bytes'],f"; R4 correction: this path is a symbolic link to {target_path}; lstat size is {symlink['symlink_lstat_bytes']} bytes, not target payload size. Original R3 row combined link size and target SHA.",symlink_artifact_id))
        con.execute("DELETE FROM run_artifact_links WHERE artifact_id=? AND artifact_role='INPUT' AND run_id IN (SELECT run_id FROM lineage_runs WHERE campaign_id=?)",
                    (symlink_artifact_id,CAMPAIGN_ID))
        target_id='mbr4_'+keyhash(target_path,symlink['target_sha256'])
        con.execute("""INSERT INTO lineage_artifacts(artifact_id,artifact_key,artifact_role,source_path,path_scope,sha256,byte_size,mime_type,
          availability_status,scientific_status,captured_at,notes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
          (target_id,'mace_terminal_failure_r4:input_target:'+keyhash(target_path,symlink['target_sha256']),'INPUT',target_path,'DATA',
           symlink['target_sha256'],symlink['target_stat_bytes'],'chemical/x-xyz','PRESENT','EXCLUDED',CAPTURED_AT,
           f"Dereferenced target of {link_path}; independently identified content input. Original symlink metadata is a separate artifact."))
        new_ids.append(target_id)
        control_paths={
          'ARCHIVE_COPY_MANIFEST.json':('MANIFEST','MANIFEST'),
          'SOURCE_INPUT_SYMLINK.json':('OTHER','OTHER'),
          'ARCHIVE_COPY_RECEIPT.json':('MANIFEST','MANIFEST'),
          'README.md':('OTHER','OTHER'),
          'SHA256SUMS':('MANIFEST','MANIFEST')}
        control_ids={}
        for name,(role,linkrole) in control_paths.items():
          h=sums_sha if name=='SHA256SUMS' else sum_map.get(name,'')
          if not h:raise SystemExit(f'archive checksum list missing {name}')
          path=f"/scratch16/pclancy3/yi/MigrationBench_data_archive/{ARCHIVE_REL}/{name}"
          aid='mbr4_'+keyhash(path,h);control_ids[name]=aid
          con.execute("""INSERT INTO lineage_artifacts(artifact_id,artifact_key,artifact_role,source_path,path_scope,sha256,byte_size,mime_type,
            availability_status,scientific_status,captured_at,notes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (aid,'mace_terminal_failure_r4:archive_control:'+keyhash(path,h),role,path,'SCRATCH16',h,None,
             mimetypes.guess_type(name)[0] or 'text/plain','ARCHIVED','EXCLUDED',CAPTURED_AT,
             'Control artifact for the organized SHA-verified campaign archive; payload statuses remain FAILURE_MODE/EXCLUDED.'))
          new_ids.append(aid)
        archive_ids={}
        for r in files:
          rel=r['archive_relative_path'];path=f"/scratch16/pclancy3/yi/MigrationBench_data_archive/{ARCHIVE_REL}/{rel}"
          typ=r['artifact_type'];role,_=TYPE_ROLE[typ];h=r['verified_sha256'];aid='mbr4_'+keyhash(path,h)
          archive_ids[(r['run_id'],typ,rel)]=aid
          note=f"SHA-verified independent archive copy; source_path={r['source_path']}; archive_relative_path={rel}; captured 2026-11-25."
          if typ=='shared_input':note+=f" Original run input symlink={symlink['symlink_path']} -> {symlink['symlink_target']}; lstat bytes=145; target bytes=320651971."
          con.execute("""INSERT INTO lineage_artifacts(artifact_id,artifact_key,artifact_role,source_path,path_scope,sha256,byte_size,mime_type,
            availability_status,scientific_status,captured_at,notes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (aid,'mace_terminal_failure_r4:archive_copy:'+keyhash(path,h),role,path,'SCRATCH16',h,int(r['verified_bytes']),
             mimetypes.guess_type(path)[0] or '', 'ARCHIVED','EXCLUDED',CAPTURED_AT,note))
          new_ids.append(aid)
        # Path alias record is preserved as OTHER; actual content input is the /data target plus archive copy.
        run_rows=con.execute('SELECT run_id,model_id FROM lineage_runs WHERE campaign_id=? ORDER BY run_id',(CAMPAIGN_ID,)).fetchall()
        if len(run_rows)!=3:raise SystemExit('unexpected campaign run count')
        for run_id,model_id in run_rows:
          con.execute('INSERT INTO run_artifact_links(run_id,artifact_id,artifact_role,evidence_status,note) VALUES(?,?,?,?,?)',
            (run_id,symlink_artifact_id,'OTHER','TRACEABLE','Original input path is a symlink alias; see separate target file artifact.'))
          con.execute('INSERT INTO run_artifact_links(run_id,artifact_id,artifact_role,evidence_status,note) VALUES(?,?,?,?,?)',
            (run_id,target_id,'INPUT','TRACEABLE','Exact dereferenced input content SHA and size are recorded separately from symlink metadata.'))
          con.execute('INSERT INTO run_artifact_links(run_id,artifact_id,artifact_role,evidence_status,note) VALUES(?,?,?,?,?)',
            (run_id,control_ids['ARCHIVE_COPY_MANIFEST.json'],'MANIFEST','TRACEABLE','Archive copy manifest records exact source and organized archive paths, sizes, and SHA256.'))
          con.execute('INSERT INTO run_artifact_links(run_id,artifact_id,artifact_role,evidence_status,note) VALUES(?,?,?,?,?)',
            (run_id,control_ids['ARCHIVE_COPY_RECEIPT.json'],'MANIFEST','TRACEABLE','Remote archive SHA256SUMS check passed for all payloads and control files.'))
          for r in files:
            if r['artifact_type']=='shared_input' or r['run_id'] in ('','None'):continue
            if r['run_id'] != strip_prefix(run_id,'mbr3_'):continue
            aid=archive_ids[(r['run_id'],r['artifact_type'],r['archive_relative_path'])]
            role=TYPE_ROLE[r['artifact_type']][1]
            con.execute('INSERT INTO run_artifact_links(run_id,artifact_id,artifact_role,evidence_status,note) VALUES(?,?,?,?,?)',
              (run_id,aid,role,'TRACEABLE','Independent archive copy SHA matches the observed Rockfish source artifact.'))
          shared=next(r for r in files if r['artifact_type']=='shared_input')
          aid=archive_ids[(shared['run_id'],shared['artifact_type'],shared['archive_relative_path'])]
          con.execute('INSERT INTO run_artifact_links(run_id,artifact_id,artifact_role,evidence_status,note) VALUES(?,?,?,?,?)',
            (run_id,aid,'INPUT','TRACEABLE','Organized archive copy of the dereferenced common initial structure.'))
          result=con.execute("SELECT result_id FROM lineage_results WHERE source_row_key=? AND result_key LIKE 'mace_terminal_failure_r3:%'",
                             (strip_prefix(run_id,'mbr3_'),)).fetchone()
          if not result:raise SystemExit(f'missing terminal result for {run_id}')
          result_id=result[0]
          source_artifacts=con.execute("SELECT artifact_id,source_path FROM lineage_artifacts WHERE source_path LIKE ? AND artifact_key LIKE 'mace_terminal_failure_r3:%'",
                                       (f'%/{strip_prefix(run_id,"mbr3_")}/%',)).fetchall()
          picks={}
          for aid,path in source_artifacts:
            if path.endswith('/status.json'):picks['status']=aid
            elif path.endswith('/run_manifest.json'):picks['manifest']=aid
            elif path.endswith('/trajectory.extxyz'):picks['trajectory']=aid
          if set(picks)!={'status','manifest','trajectory'}:raise SystemExit(f'incomplete direct result evidence for {run_id}: {picks}')
          for aid,role,note in [(picks['status'],'SOURCE_ROW','Terminal step and stop reason observed in run status record.'),
                                (picks['manifest'],'SOURCE_ROW','Target and actual step lineage recorded in run manifest.'),
                                (picks['trajectory'],'DERIVED_OUTPUT','Raw terminal trajectory is preserved as failure evidence.'),
                                (target_id,'INPUT','Exact initial structure target content identity.')]:
            con.execute('INSERT INTO result_artifact_links(result_id,artifact_id,artifact_role,relation,evidence_status,note) VALUES(?,?,?,?,?,?)',
              (result_id,aid,role,'DERIVED_FROM','TRACEABLE',note))
        con.execute("INSERT INTO audit_log(checked_at,check_name,result,details) VALUES(?,?,?,?)",
          (CAPTURED_AT,'mace_archive_lineage_r4_import','PASS','Corrected the common_initial symlink/target byte identity; registered organized archive copies for 37 payloads; result states remain excluded failure observations.'))
      qc=con.execute('PRAGMA quick_check').fetchone()[0];fk=con.execute('PRAGMA foreign_key_check').fetchall()
      if qc!='ok' or fk:raise SystemExit(f'integrity error quick_check={qc}; fk={fk[:5]}')
      counts={t:con.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in ['campaigns','lineage_artifacts','lineage_runs','lineage_results','run_artifact_links','result_run_links','result_artifact_links']}
      con.close()
      outmanifest={'version':'R4','base_snapshot':'MigrationBench R1 lineage overlay R3','base_sha256':sha256(base),
        'campaign_id':CAMPAIGN_ID,'archive_relative_path':ARCHIVE_REL,'archive_payload_files':len(files),
        'archive_payload_bytes':manifest['payload_bytes'],'archive_manifest_sha256':sha256(manp),
        'symlink_observation_sha256':sha256(symp),'archive_receipt_sha256':sha256(recp),
        'input_identity_correction':{'symlink_bytes':145,'target_bytes':320651971,'target_sha256':symlink['target_sha256']},
        'archive_payload_verification':payload_verification,'result_database':out.name,'result_sha256':sha256(out),
        'result_bytes':out.stat().st_size,'counts':counts,'quick_check':qc,'foreign_key_violations':len(fk),
        'build_runtime':{'python_version':platform.python_version(),'sqlite_version':sqlite3.sqlite_version},
        'scientific_boundary':'3 terminal instability failures; FAILURE_MODE; EXCLUDE_FROM_PUBLICATION',
        'built_at':CAPTURED_AT}
      (out.parent/'R4_MANIFEST.json').write_text(json.dumps(outmanifest,indent=2,sort_keys=True)+'\n')
      print(json.dumps(outmanifest,indent=2,sort_keys=True))
    except Exception:
      con.rollback();con.close();safe_unlink(out);raise
if __name__=='__main__':main()
