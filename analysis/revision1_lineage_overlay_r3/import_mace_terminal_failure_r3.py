#!/usr/bin/env python3
"""Build an additive Revision 1 lineage R3 snapshot from R2 and frozen evidence."""
import argparse, csv, hashlib, json, shutil, sqlite3
from pathlib import Path

CAMPAIGN_ID = 'mace_from_scratch_ablation_md_20261005'
CAMPAIGN_ROOT = '/scratch16/pclancy3/yi/mace_from_scratch_ablation_MD_20261005'
CAPTURED_AT = '2026-11-25'
TYPE_ROLE = {
    'run_manifest': ('MANIFEST', 'MANIFEST'), 'status': ('LOG', 'LOG'),
    'trajectory': ('OUTPUT', 'OUTPUT'), 'shared_input': ('INPUT', 'INPUT'),
    'model_checkpoint': ('CHECKPOINT', 'CHECKPOINT'),
    'postprocess_csv': ('DERIVED_DATA', 'METRIC'), 'postprocess_json': ('METRIC', 'METRIC'),
}

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def ident(*parts): return hashlib.sha256('\0'.join(parts).encode()).hexdigest()[:24]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--r2-db',type=Path,required=True)
    ap.add_argument('--evidence-dir',type=Path,required=True)
    ap.add_argument('--output-db',type=Path,required=True)
    a=ap.parse_args()
    src=a.r2_db.resolve(); ev=a.evidence_dir.resolve(); out=a.output_db.resolve()
    if out.exists(): raise SystemExit(f'refusing to overwrite {out}')
    run_rows=list(csv.DictReader((ev/'runs.tsv').open(),delimiter='\t'))
    art_rows=list(csv.DictReader((ev/'lineage_artifacts.tsv').open(),delimiter='\t'))
    if len(run_rows)!=3 or len(art_rows)!=61: raise SystemExit('unexpected frozen input counts')
    sums={}
    for line in (ev/'SHA256SUMS').read_text().splitlines():
        expected,name=line.split(None,1); name=name.strip()
        if sha256(ev/name)!=expected: raise SystemExit(f'evidence checksum mismatch: {name}')
        sums[name]=expected
    shutil.copy2(src,out)
    con=sqlite3.connect(out)
    con.execute('PRAGMA foreign_keys=ON')
    try:
      with con:
        con.execute("INSERT INTO campaigns(campaign_id,title,campaign_path,source_id,campaign_date,status,notes) VALUES(?,?,?,?,?,?,?)",
          (CAMPAIGN_ID,'MACE from-scratch ablation 1 ns terminal instability failures',CAMPAIGN_ROOT,None,'2026-10-05','PARTIAL',
           'Three observed runs terminated at instability_threshold before 1,000,000 steps. Preserved as FAILURE_MODE; EXCLUDE_FROM_PUBLICATION. Not a complete transport campaign.'))
        model_by_run={}
        for rr in run_rows:
          model_id='checkpoint_sha256_'+rr['model_sha256'][:16]
          model_by_run[rr['run_id']]=model_id
          if not con.execute('SELECT 1 FROM model_families WHERE family_id=?',('scratch_historical',)).fetchone():
            raise SystemExit('required scratch_historical family missing')
          con.execute("""INSERT INTO models(model_id,display_name,family_id,identity_kind,training_regime,training_data_label,
            training_temperature_label,training_seed,representation_label,checkpoint_sha256,artifact_path,lineage_status,
            release_status,scientific_status,active_scope,notes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (model_id,rr['family']+' training seed '+rr['training_seed']+' checkpoint '+rr['model_sha256'][:12],
             'scratch_historical','CHECKPOINT_SHA256','Train from Scratch','ablation study',None,rr['training_seed'],
             'MACE Lmax2' if rr['family']=='ablation_lmax_2' else 'MACE baseline',rr['model_sha256'],rr['model_path'],
             'PROVENANCE_PARTIAL','HISTORICAL','EXCLUDED','Historical terminal-failure audit',
             'Exact checkpoint SHA/path linked to an instability-threshold terminal failure; excluded from publication claims.'))
        artifact_ids={}
        for ar in art_rows:
          kind,role=TYPE_ROLE[ar['artifact_type']]
          path=ar['path']; side=ar['side']; digest=ar['sha256']
          if side=='git_main':
            marker='/analysis/from_scratch_ablation_md_20261005/remote_postprocess_terminal_20261006/'
            if marker not in path: raise SystemExit(f'unexpected Git path: {path}')
            rel='analysis/from_scratch_ablation_md_20261005/remote_postprocess_terminal_20261006/'+path.split(marker,1)[1]
            source_path='git+https://github.com/yicao-elina/MigrationBench.git@feb328843932e436f0d75e7f407b9b425319d96c/'+rel
            scope='GIT_BLOB'; availability='ARCHIVED'
          else:
            source_path=path
            scope='DATA' if path.startswith('/data/') else 'SCRATCH16'
            availability='PRESENT' if scope=='DATA' else 'ARCHIVED'
          aid='mbr3_'+ident(source_path,digest)
          artifact_ids[(ar['run_id'],ar['artifact_type'],side,path)]=aid
          note=f"Frozen 2026-11-25 audit; original observed path={path}; side={side}; type={ar['artifact_type']}"
          if ar['data_rows']: note+=f"; data_rows={ar['data_rows']} (header excluded)"
          if ar['artifact_type'] in ('postprocess_csv','postprocess_json') and ar['side']=='git_main':
            note+='; Git historical copy, keep distinct from live Rockfish output even when similar'
          if ar['artifact_type']=='trajectory': note+='; raw terminal trajectory retained in Rockfish; scientific status EXCLUDED'
          con.execute("""INSERT INTO lineage_artifacts(artifact_id,artifact_key,artifact_role,source_path,path_scope,sha256,
            byte_size,mime_type,availability_status,scientific_status,captured_at,notes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (aid,'mace_terminal_failure_r3:'+ident(ar['artifact_type'],ar['run_id'],side,path,digest),kind,
             source_path,scope,digest,int(ar['bytes']),'',availability,'EXCLUDED',CAPTURED_AT,note))
        for rr in run_rows:
          rid=rr['run_id']; model_id=model_by_run[rid]
          note=(f"MACE 1 ns terminal failure; stop_reason={rr['stop_reason']}; actual_step={rr['actual_steps']}/"
                f"{rr['target_steps']} ({rr['completion_fraction']}); training_seed={rr['training_seed']}; "
                f"md_seed={rr['md_seed']}; Slurm array={rr['slurm_job_id']}_{rr['slurm_array_task_id']}; "
                f"finite_gate={rr['finite_gate']}; complete_step_gate={rr['complete_step_gate']}; "
                'FAILURE_MODE; EXCLUDE_FROM_PUBLICATION. Scheduler state not independently refreshed.')
          con.execute("""INSERT INTO lineage_runs(run_id,run_key,campaign_id,run_kind,protocol_key,model_id,
            model_checkpoint_sha256,seed,sample_key,attempt_number,parent_run_id,scheduler_job_id,scheduler_status,
            run_status,scientific_status,started_at,finished_at,notes) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            ('mbr3_'+rid,'mace_terminal_failure_r3:'+rid,CAMPAIGN_ID,'MD','md_1ns',model_id,rr['model_sha256'],
             rr['training_seed'],'mdseed'+rr['md_seed'],0,None,rr['slurm_job_id']+'_'+rr['slurm_array_task_id'],
             'UNKNOWN','FAILED','EXCLUDED',None,None,note))
          value={'stop_reason':rr['stop_reason'],'actual_step':int(rr['actual_steps']),'target_steps':int(rr['target_steps']),
                 'absolute_time_ps':float(rr['absolute_time_ps']),'completion_fraction':float(rr['completion_fraction']),
                 'training_seed':int(rr['training_seed']),'md_seed':int(rr['md_seed']),
                 'finite_gate':rr['finite_gate']=='true','complete_step_gate':rr['complete_step_gate']=='true'}
          result_id='mbr3_terminal_'+ident(rid)
          con.execute("""INSERT INTO lineage_results(result_id,result_key,result_kind,source_row_key,result_grain,task_id,
            system_id,model_id,model_checkpoint_sha256,protocol_key,value_json,unit,evidence_status,scientific_status,notes)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (result_id,'mace_terminal_failure_r3:'+rid,'SUMMARY',rid,'run_terminal_state','md_1ns','cr_md_bulk',model_id,
             rr['model_sha256'],'md_1ns',json.dumps(value,sort_keys=True,separators=(',',':')),'step; ps; fraction','TRACEABLE','EXCLUDED',
             'Observed terminal status only; not a completed 1 ns transport result; FAILURE_MODE; EXCLUDE_FROM_PUBLICATION.'))
          con.execute("INSERT INTO result_run_links(result_id,run_id,relation,evidence_status,note) VALUES(?,?,?,?,?)",
            (result_id,'mbr3_'+rid,'PRODUCED_BY','TRACEABLE','Run status/manifest tie terminal step and stop reason to this attempt; publication status remains excluded.'))
        for ar in art_rows:
          if not ar['run_id']: continue
          rid=ar['run_id']; aid=artifact_ids[(rid,ar['artifact_type'],ar['side'],ar['path'])]
          role=TYPE_ROLE[ar['artifact_type']][1]
          status='CONFLICT' if ar['artifact_type']=='postprocess_csv' and ar['side']=='git_main' else 'TRACEABLE'
          note='Exact frozen artifact identity and source path captured in R3 package.'
          if status=='CONFLICT': note+=' Same CSV semantic row counts but SHA/byte differences from Rockfish version; preserve as distinct historical artifact.'
          con.execute('INSERT INTO run_artifact_links(run_id,artifact_id,artifact_role,evidence_status,note) VALUES(?,?,?,?,?)',
            ('mbr3_'+rid,aid,role,status,note))
        shared=next(ar for ar in art_rows if ar['artifact_type']=='shared_input')
        shared_id=artifact_ids[('',shared['artifact_type'],shared['side'],shared['path'])]
        for rr in run_rows:
          con.execute('INSERT INTO run_artifact_links(run_id,artifact_id,artifact_role,evidence_status,note) VALUES(?,?,?,?,?)',
            ('mbr3_'+rr['run_id'],shared_id,'INPUT','TRACEABLE','Shared initial structure SHA is recorded in frozen run manifest inventory.'))
        con.execute("INSERT INTO audit_log(checked_at,check_name,result,details) VALUES(?,?,?,?)",
          (CAPTURED_AT,'mace_terminal_failure_r3_import','PASS','Imported 3 failed/excluded runs, 3 terminal-state results, 61 artifacts, and run-artifact links from immutable R2 base.'))
      qc=con.execute('PRAGMA quick_check').fetchone()[0]
      fk=con.execute('PRAGMA foreign_key_check').fetchall()
      if qc!='ok' or fk: raise SystemExit(f'integrity failure quick_check={qc}, FK={fk[:5]}')
      counts={t:con.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in ['campaigns','models','lineage_runs','lineage_results','lineage_artifacts','run_artifact_links','result_run_links']}
      con.close()
      manifest={'version':'R3','base_snapshot':'MigrationBench R1 lineage overlay R2','base_sha256':sha256(src),
        'campaign_id':CAMPAIGN_ID,'evidence_package_sha256':sums,'result_database':out.name,'result_sha256':sha256(out),
        'result_bytes':out.stat().st_size,'counts':counts,'quick_check':qc,'foreign_key_violations':len(fk),
        'scientific_boundary':'3 terminal instability failures; FAILURE_MODE; EXCLUDE_FROM_PUBLICATION',
        'postprocess_note':'All 15 Rockfish/Git CSV pairs retain their differing SHA256 and byte sizes; row counts agree.',
        'built_at':CAPTURED_AT}
      (out.parent/'R3_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
      print(json.dumps(manifest,indent=2,sort_keys=True))
    except Exception:
      con.rollback(); con.close(); out.unlink(missing_ok=True); raise
if __name__=='__main__': main()
