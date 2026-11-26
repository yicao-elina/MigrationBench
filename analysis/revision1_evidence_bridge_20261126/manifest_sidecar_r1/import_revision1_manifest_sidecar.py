#!/usr/bin/env python3
"""Build an isolated Revision 1 manifest provenance staging sidecar.

This importer never opens the central 4 GB provenance catalog. It consumes the
frozen Nov 25 audit exports and a read-only Rockfish revalidation receipt.
Compatible with Python 3.6+.
"""
from __future__ import print_function
import argparse, collections, csv, hashlib, json, os, re, sqlite3
from pathlib import Path

INPUT_ROOT='/scratch16/pclancy3/yi/revision1_migrationbench_inputs'
RUN_ROOT='/scratch16/pclancy3/yi/revision1_migrationbench_runs'
ROOTS={INPUT_ROOT:'MBR1-R1-INPUTS',RUN_ROOT:'MBR1-R1-RUNS'}
CONTROL_RE=re.compile(r'job|retry|attempt|slurm|scheduler',re.I)
EXPECTED_COUNTS={'input_manifest_files':494,'run_manifest_files':1372,
                 'path_references':9825,'manifest_control_observations':1174,
                 'manifest_control_fields':3367}
CANONICAL_MAIN='cf6c2b1649ba993d079d0eadf8478775a81736d2'

def sha256(path):
    h=hashlib.sha256()
    with open(str(path),'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def read_tsv(path):
    with open(str(path),'r') as f:
        return list(csv.DictReader(f,delimiter='\t'))

def path_scope(value):
    for root,root_id in ROOTS.items():
        if value.startswith(root+'/'):
            parts=value[len(root)+1:].split('/')
            if not parts or any(x in ('','..') for x in parts):
                return 'INVALID_PATH',None,None
            return ('PERSONAL_INPUT_ROOT' if root==INPUT_ROOT else 'PERSONAL_RUN_ROOT',root_id,'/'.join(parts))
    if not value or not value.startswith('/'):
        return 'UNRESOLVED_RELATIVE',None,None
    return 'OUTSIDE_PERSONAL_ROOTS',None,None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-dir',type=Path,required=True)
    ap.add_argument('--targets',type=Path,required=True)
    ap.add_argument('--revalidation-results',type=Path,required=True)
    ap.add_argument('--revalidation-receipt',type=Path,required=True)
    ap.add_argument('--schema',type=Path,required=True)
    ap.add_argument('--output-db',type=Path,required=True)
    ap.add_argument('--output-manifest',type=Path,required=True)
    a=ap.parse_args()
    out=a.output_db.resolve()
    if out.exists(): raise SystemExit('refusing to overwrite '+str(out))
    index_path=a.source_dir/'manifest_index.tsv'
    refs_path=a.source_dir/'manifest_references.tsv'
    counts_path=a.source_dir/'audit_counts.json'
    for p in (index_path,refs_path,counts_path,a.targets,a.revalidation_results,a.revalidation_receipt,a.schema):
        p.resolve(strict=True)
    source_hashes={p.name:sha256(p) for p in (index_path,refs_path,counts_path,a.targets,a.revalidation_results,a.revalidation_receipt,a.schema)}
    counts=json.loads(counts_path.read_text())
    if counts.get('source_scan_sha256')!='9ec3d8272eb60e2afbc1661cf6d0b92ce0697764bc79e74c645929809c64f486':
        raise SystemExit('unexpected frozen source scan hash')
    if source_hashes['manifest_index.tsv']!='d5a8fb50e89814db8c2fe98378bf2f058a8abfad72cb7d4a139b7be868730c0e':
        raise SystemExit('manifest index export hash mismatch')
    if source_hashes['manifest_references.tsv']!='c8cdb42e27ca8afdf9fb7305adfc51e3a5ccfcb63758b23a3f0b1edf3da096e5':
        raise SystemExit('manifest reference export hash mismatch')

    index=read_tsv(index_path); refs=read_tsv(refs_path); targets=read_tsv(a.targets); results=read_tsv(a.revalidation_results)
    receipt=json.loads(a.revalidation_receipt.read_text())
    input_n=sum(1 for r in index if r['kind']=='input'); run_n=sum(1 for r in index if r['kind']=='run')
    path_refs=[r for r in refs if r['is_path_candidate']=='True']
    control_fields=[r for r in refs if CONTROL_RE.search(r['field_path'] or '')]
    controls_by_manifest=collections.defaultdict(list)
    for r in control_fields:
        controls_by_manifest[(r['kind'],r['manifest_relpath'])].append(r)
    if (input_n,run_n,len(path_refs),len(controls_by_manifest),len(control_fields)) != tuple(EXPECTED_COUNTS[k] for k in ('input_manifest_files','run_manifest_files','path_references','manifest_control_observations','manifest_control_fields')):
        raise SystemExit('frozen source manifest/control/reference counts differ')
    if len(index)!=1866 or len(path_refs)!=9825 or len(targets)!=2322 or len(results)!=2322:
        raise SystemExit('frozen input or remote revalidation row count differs')
    if receipt.get('target_rows')!=len(targets) or receipt.get('targets_sha256')!=source_hashes['REVALIDATION_TARGETS.tsv']:
        raise SystemExit('Rockfish target receipt does not match target TSV')
    if receipt.get('results_tsv_sha256')!=source_hashes['REVALIDATION_RESULTS.tsv']:
        raise SystemExit('Rockfish receipt does not match revalidation results')
    if receipt.get('read_only') is not True or set(receipt.get('declared_roots',[]))!=set(ROOTS.keys()):
        raise SystemExit('unexpected Rockfish scope or read-only status')

    target_by_path={r['path']:r for r in targets}
    result_by_path={r['path']:r for r in results}
    if len(target_by_path)!=len(targets) or len(result_by_path)!=len(results) or set(target_by_path)!=set(result_by_path):
        raise SystemExit('target and revalidation path sets differ')
    for p,r in result_by_path.items():
        t=target_by_path[p]
        if r['target_kind']!=t['target_kind'] or r['expected_bytes']!=t['expected_bytes'] or r['expected_sha256']!=t['expected_sha256']:
            raise SystemExit('target row and remote result identity differ: '+p)
    if sum(1 for r in results if r['verification_status']=='METADATA_SIZE_SHA_MATCH')!=1866:
        raise SystemExit('not all 1866 metadata files revalidated by exact size/SHA')

    captured=receipt['captured_utc']
    snapshot_id='MBR1-MANIFEST-RECHECK-20261126-R1'
    con=sqlite3.connect(str(out)); con.execute('PRAGMA foreign_keys=ON')
    try:
        con.executescript(a.schema.read_text())
        with con:
            con.execute("INSERT INTO project(project_id,project_name,owner_scope,description,status) VALUES('MIGRATIONBENCH','Wiley MigrationBench','SHARED','Staging sidecar for Revision 1 input/run manifest evidence; copied central project owner_scope, no claims or workflow_run rows imported.','ACTIVE')")
            for root,root_id in ROOTS.items():
                con.execute("""INSERT INTO storage_root(root_id,cluster_name,filesystem_name,mount_path,scoped_path,owner_user,owner_group,scope_kind,project_id,status,notes)
                    VALUES(?, 'Rockfish','Scratch16','/scratch16',?,'ycao73','pclancy3','USER_SUBTREE','MIGRATIONBENCH','ACTIVE',?)""",
                    (root_id,root,'Personal Revision 1 inputs/run subtree only; no other-owner paths are included.'))
            src_hash_map={
                'manifest_index.tsv':('manifest-index-export',index_path,'MANIFEST_INDEX_TSV'),
                'manifest_references.tsv':('manifest-reference-export',refs_path,'MANIFEST_REFERENCE_TSV'),
                'audit_counts.json':('manifest-audit-counts',counts_path,'AUDIT_COUNTS_JSON'),
                'REVALIDATION_TARGETS.tsv':('rockfish-revalidation-targets',a.targets,'REVALIDATION_TARGET_TSV'),
                'REVALIDATION_RESULTS.tsv':('rockfish-revalidation-results',a.revalidation_results,'REVALIDATION_RESULT_TSV'),
                'REVALIDATION_RECEIPT.json':('rockfish-revalidation-receipt',a.revalidation_receipt,'REVALIDATION_RECEIPT_JSON'),
                'schema.sql':('staging-schema',a.schema,'STAGING_SCHEMA_SQL'),
            }
            for name,(key,path,kind) in src_hash_map.items():
                con.execute('INSERT INTO sidecar_source(source_key,local_path,sha256,byte_size,source_kind,notes) VALUES(?,?,?,?,?,?)',
                            (key,str(path),source_hashes[name],path.stat().st_size,kind,'Frozen sidecar input; this source is not imported as a scientific result.'))
            con.execute("""INSERT INTO revalidation_snapshot(snapshot_id,targets_sha256,results_sha256,receipt_sha256,captured_utc,host_name,python_version,target_count,method,scope_note)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (snapshot_id,source_hashes['REVALIDATION_TARGETS.tsv'],source_hashes['REVALIDATION_RESULTS.tsv'],source_hashes['REVALIDATION_RECEIPT.json'],captured,
                 receipt['host'],receipt['python_version'],len(results),'Read-only lstat/stat with full hash for all 1,866 manifest files; candidate hash policy pinned in receipt.',
                 'Only the two user-owned Scratch16 Revision 1 roots; final symlinks not followed; no other-owner directory scanned.'))

            path_id_by_abs={}
            for r in results:
                scope,root_id,rel=path_scope(r['path'])
                if root_id is None: raise SystemExit('remote revalidation returned out-of-scope path '+r['path'])
                con.execute('INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)',(root_id,rel,captured))
                pid=con.execute('SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?',(root_id,rel)).fetchone()[0]
                path_id_by_abs[r['path']]=int(pid)
                exp_bytes=int(r['expected_bytes']) if r['expected_bytes'] else None
                cur_bytes=int(r['current_bytes']) if r['current_bytes'] else None
                cur_sha=r['current_sha256'] or None
                con.execute("""INSERT INTO path_revalidation(snapshot_id,path_id,target_kind,observed_type,current_bytes,current_sha256,expected_bytes,expected_sha256,mtime_ns,verification_status,symlink_target,error_text)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (snapshot_id,pid,r['target_kind'],r['observed_type'],cur_bytes,cur_sha,exp_bytes,r['expected_sha256'] or None,
                     r['mtime_ns'] or None,r['verification_status'],r['symlink_target'] or '',r.get('verification_error','')))

            # Artifact per observed manifest location + recorded scan digest. This preserves all 1,866 file observations.
            metadata_by_path={}; current_artifact_by_path={}
            selector='basename contains manifest or provenance; regular files only; symlinks not followed'
            for r in index:
                root=INPUT_ROOT if r['kind']=='input' else RUN_ROOT
                abs_path=root+'/'+r['manifest_relpath']
                scope,root_id,rel=path_scope(abs_path)
                if root_id is None: raise SystemExit('manifest index produced out-of-scope path '+abs_path)
                verify=result_by_path.get(abs_path)
                if not verify or verify['target_kind']!='METADATA': raise SystemExit('manifest file missing revalidation row '+abs_path)
                recorded_sha=r['manifest_sha256']; recorded_bytes=int(r['manifest_bytes'])
                key='MBR1-META:'+root_id+':'+rel+':'+recorded_sha
                con.execute("""INSERT INTO artifact(project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
                    VALUES('MIGRATIONBENCH',?,'MANIFEST_OR_PROVENANCE','REV1_MANIFEST_METADATA','USER_CONFIRMED',?,?,'PROVENANCE_PARTIAL','PUBLICATION_EVIDENCE',?)""",
                    (key,recorded_sha,recorded_bytes,'Nov 25 scan content identity. Current location was rechecked on Rockfish; no run/claim association or scientific acceptance is asserted.'))
                aid=int(con.execute('SELECT artifact_id FROM artifact WHERE artifact_key=?',(key,)).fetchone()[0])
                pid=path_id_by_abs[abs_path]
                exact=(verify['verification_status']=='METADATA_SIZE_SHA_MATCH')
                con.execute("""INSERT INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc)
                    VALUES(?,?,?,?,?)""",(aid,pid,'PRESENT' if exact else 'MISSING',captured if exact else None,captured))
                current_aid=None
                if verify['observed_type']=='REGULAR_FILE' and verify['current_sha256']:
                    cur_sha=verify['current_sha256'];cur_size=int(verify['current_bytes'])
                    ckey='MBR1-CURRENT:'+root_id+':'+rel+':'+cur_sha
                    if exact:
                        # Current bytes are exactly the scanned identity; do not create a duplicate artifact.
                        current_aid=aid
                    else:
                        con.execute("""INSERT OR IGNORE INTO artifact(project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
                            VALUES('MIGRATIONBENCH',?,'CURRENT_MANIFEST_FILE','CURRENT_REV1_METADATA','USER_CONFIRMED',?,?,'PROVENANCE_PARTIAL','PUBLICATION_EVIDENCE',?)""",
                            (ckey,cur_sha,cur_size,'Current revalidation content; distinct from the Nov 25 scanned artifact identity.'))
                        current_aid=int(con.execute('SELECT artifact_id FROM artifact WHERE artifact_key=?',(ckey,)).fetchone()[0])
                        con.execute('INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,? ,\'PRESENT\',?,?)',(current_aid,pid,captured,captured))
                else:
                    current_aid=aid if exact else None
                meta_id=int(con.execute("""INSERT INTO manifest_observation(root_id,path_id,artifact_id,manifest_kind,top_level_dir,manifest_relpath,selector,recorded_scan_sha256,recorded_scan_bytes,scan_read_status,current_verification_status,current_artifact_id,revalidation_snapshot_id)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (root_id,pid,aid,r['kind'],r['top_level_dir'],r['manifest_relpath'],selector,recorded_sha,recorded_bytes,r['read_status'],
                     verify['verification_status'],current_aid,snapshot_id)).lastrowid)
                metadata_by_path[abs_path]=(meta_id,aid,current_aid,pid)

            # Current referenced targets are registered only under the two declared personal roots.
            target_artifact_by_path={}
            for p,r in result_by_path.items():
                if r['target_kind']!='REFERENCE': continue
                if r['observed_type']!='REGULAR_FILE' or not r['current_bytes']:
                    continue
                scope,root_id,rel=path_scope(p); pid=path_id_by_abs[p]
                digest=r['current_sha256'] or None;size=int(r['current_bytes'])
                suffix=digest if digest else 'NO_SHA_'+str(size)
                key='MBR1-REFTARGET:'+root_id+':'+rel+':'+suffix
                con.execute("""INSERT OR IGNORE INTO artifact(project_id,artifact_key,artifact_kind,semantic_role,owner_status,sha256,byte_size,scientific_status,retention_class,notes)
                    VALUES('MIGRATIONBENCH',?,'MANIFEST_REFERENCE_TARGET','UNRESOLVED_REV1_TARGET','USER_CONFIRMED',?,?,'PROVENANCE_PARTIAL','UNCLASSIFIED',?)""",
                    (key,digest,size,'Current personal-root reference target identity only; source run/claim and scientific role remain unverified.'))
                aid=int(con.execute('SELECT artifact_id FROM artifact WHERE artifact_key=?',(key,)).fetchone()[0])
                con.execute('INSERT OR IGNORE INTO artifact_location(artifact_id,path_id,availability,first_verified_utc,last_verified_utc) VALUES(?,? ,\'PRESENT\',?,?)',(aid,pid,captured,captured))
                target_artifact_by_path[p]=aid

            # One control observation per manifest record; child rows retain each matched field value.
            for (kind,rel),fields in controls_by_manifest.items():
                root=INPUT_ROOT if kind=='input' else RUN_ROOT;abs_path=root+'/'+rel
                meta_id=metadata_by_path[abs_path][0]
                obs_id=int(con.execute('INSERT INTO manifest_control_observation(manifest_observation_id,field_count,field_key_count,validation_status,notes) VALUES(?,?,?,\'UNVERIFIED\',?)',
                    (meta_id,len(fields),len(set(x['field_path'] for x in fields)),'Field values are source-manifest strings; no live Slurm state or run/result relation is asserted.')).lastrowid)
                for x in fields:
                    con.execute('INSERT INTO manifest_control_field(control_observation_id,field_path,raw_value,value_type,interpretation_status) VALUES(?,?,?,\'TEXT_CAPTURE\',\'UNVERIFIED\')',
                        (obs_id,x['field_path'],x['value']))

            # Preserve every path-candidate occurrence. Outside/relative values stay literal and unverified.
            for x in path_refs:
                root=INPUT_ROOT if x['kind']=='input' else RUN_ROOT;source_abs=root+'/'+x['manifest_relpath']
                meta_id=metadata_by_path[source_abs][0]
                scope,root_id,rel=path_scope(x['value'])
                target_pid=None;target_aid=None;current_status='NOT_RECHECKED_OUTSIDE_SCOPE'
                notes='Literal manifest value only; no run or claim binding is inferred.'
                if root_id:
                    target_abs=x['value'];target_pid=path_id_by_abs.get(target_abs)
                    if target_pid is None:
                        # Candidate may be in-scope but wasn't previously resolved; only add a path pointer.
                        con.execute('INSERT OR IGNORE INTO storage_path(root_id,relative_path,first_seen_utc) VALUES(?,?,?)',(root_id,rel,captured))
                        target_pid=int(con.execute('SELECT path_id FROM storage_path WHERE root_id=? AND relative_path=?',(root_id,rel)).fetchone()[0])
                    recheck=result_by_path.get(target_abs)
                    if recheck:
                        current_status=recheck['verification_status']
                        target_aid=target_artifact_by_path.get(target_abs)
                        if target_aid is None and target_abs in metadata_by_path:
                            target_aid=metadata_by_path[target_abs][1] if recheck['verification_status']=='METADATA_SIZE_SHA_MATCH' else metadata_by_path[target_abs][2]
                        notes='Personal-root path was revalidated using the recorded lstat/stat/hash policy; status remains provenance-only.'
                    else:
                        current_status='NOT_IN_REVALIDATION_TARGET_SET'
                elif scope=='UNRESOLVED_RELATIVE':
                    current_status='NOT_RECHECKED_RELATIVE_PATH'
                else:
                    notes='Outside the two authorized Scratch16 input/run roots; no external path was accessed.'
                con.execute("""INSERT INTO manifest_path_reference(manifest_observation_id,reference_index,field_path,literal_value,previous_resolution,previous_bytes,previous_target_sha256,previous_hash_status,scope_class,target_path_id,target_artifact_id,current_verification_status,notes)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (meta_id,int(x['reference_index']),x['field_path'],x['value'],x['resolution'],int(x['bytes']) if x['bytes'] else None,
                     x['target_sha256'] or None,x['hash_status'],scope,target_pid,target_aid,current_status,notes))

            con.execute("""INSERT INTO sidecar_import(import_id,source_scan_sha256,importer_sha256,schema_sha256,canonical_main_commit,imported_utc,metadata_observation_count,control_observation_count,control_field_count,path_reference_count,revalidation_path_count,quick_check,foreign_key_violations,scientific_boundary)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ('REV1-MANIFEST-SIDECAR-R1-20261126',counts['source_scan_sha256'],sha256(Path(__file__)),sha256(a.schema),CANONICAL_MAIN,captured,
                 len(index),len(controls_by_manifest),len(control_fields),len(path_refs),len(results),'PENDING',-1,
                 'All manifest and target artifacts are PROVENANCE_PARTIAL; control-field observations UNVERIFIED; no workflow_run, slurm_job, result, claim, or scientific status was imported.'))
        qc=con.execute('PRAGMA quick_check').fetchone()[0]
        fk=con.execute('PRAGMA foreign_key_check').fetchall()
        if qc!='ok' or fk: raise SystemExit('sidecar integrity failed: '+str(qc)+' fk='+repr(fk[:5]))
        table_counts={t:con.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in ('manifest_observation','manifest_control_observation','manifest_control_field','manifest_path_reference','path_revalidation','artifact','artifact_location')}
        expected={'manifest_observation':1866,'manifest_control_observation':1174,'manifest_control_field':3367,'manifest_path_reference':9825,'path_revalidation':2322}
        for t,n in expected.items():
            if table_counts[t]!=n: raise SystemExit('unexpected '+t+' count '+str(table_counts[t])+' expected '+str(n))
        con.execute("UPDATE sidecar_import SET quick_check=?,foreign_key_violations=? WHERE import_id='REV1-MANIFEST-SIDECAR-R1-20261126'",(qc,len(fk)))
        con.commit()
        manifest={'schema_version':'1','sidecar_id':'REV1-MANIFEST-SIDECAR-R1-20261126','database_name':out.name,
          'database_sha256':sha256(out),'database_bytes':out.stat().st_size,'canonical_main_commit':CANONICAL_MAIN,
          'source_scan_sha256':counts['source_scan_sha256'],'rockfish_revalidation_receipt_sha256':source_hashes['REVALIDATION_RECEIPT.json'],
          'rockfish_revalidation_host':receipt['host'],'rockfish_python_version':receipt['python_version'],
          'revalidation_counts':receipt['verification_counts'],'table_counts':table_counts,
          'quick_check':qc,'foreign_key_violations':len(fk),
          'boundary':'Staging sidecar only; no central catalog is opened for writes; all scientific records remain PROVENANCE_PARTIAL/UNVERIFIED.'}
        a.output_manifest.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        print(json.dumps(manifest,indent=2,sort_keys=True))
    except Exception:
        con.rollback();con.close()
        try: out.unlink()
        except OSError: pass
        raise
    finally:
        try: con.close()
        except Exception: pass

if __name__=='__main__': main()
