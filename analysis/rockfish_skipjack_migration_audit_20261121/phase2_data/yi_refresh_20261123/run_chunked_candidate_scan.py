#!/usr/bin/env python3
"""Run one metadata-only candidate scan per immediate directory; persist root receipts."""
import csv, gzip, hashlib, json, os, subprocess, sys, time, datetime
BASE='/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/yi_refresh_20261123'
CHILDREN=BASE+'/immediate_children_20261123.tsv'
REMOTE=BASE+'/find_large_candidates_remote.py'
REMOTE_BYTES=open(REMOTE,'rb').read()
rows=[]
with open(CHILDREN,encoding='utf-8') as f:
    for line in f:
        cols=line.rstrip('\n').split('\t')
        if len(cols)>=2: rows.append((cols[0],cols[1]))
summary={'started_local_epoch':time.time(),'roots':[],'ssh_options':['BatchMode=yes','ConnectTimeout=15','ServerAliveInterval=30','ServerAliveCountMax=4']}
for root,kind in rows:
    if root=='/data/pclancy3/yi/.conda':
        item={'root':root,'status':'skipped_existing_conda_evidence','started_local_epoch':time.time(),'completed_local_epoch':time.time(),'exit_code':0,'candidate_count':0,'reason':'Reuse prior fresh Conda package and >500MiB hash/hardlink evidence; avoid rewalking the large cache tree.'}
        summary['roots'].append(item); print(root,'SKIP using existing Conda evidence',flush=True); continue
    if kind!='d':
        item={'root':root,'status':'skipped_not_directory','started_local_epoch':time.time(),'completed_local_epoch':time.time(),'exit_code':0,'candidate_count':0}
        summary['roots'].append(item); print(root,'SKIP not directory',flush=True); continue
    slug=root.rsplit('/',1)[-1].replace('/','_') or 'root'
    out=BASE+'/root_'+slug+'_large_candidates_20261123.jsonl.gz'
    err_path=BASE+'/root_'+slug+'_large_candidates_20261123.stderr'
    status_path=BASE+'/root_'+slug+'_large_candidates_20261123.status.json'
    started=time.time()
    argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=4','rockfish','python3','-',root]
    p=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    timed_out=False
    try:
        stdout,stderr=p.communicate(REMOTE_BYTES,timeout=600)
    except subprocess.TimeoutExpired:
        timed_out=True
        p.kill()
        stdout,stderr=p.communicate()
    completed=time.time()
    with gzip.open(out,'wb',compresslevel=6) as z: z.write(stdout)
    with open(err_path,'wb') as ef: ef.write(stderr)
    records=[]; complete=False; count=0
    for line in stdout.splitlines():
        try:
            d=json.loads(line); records.append(d)
            if d.get('record_type')=='candidate': count+=1
            if d.get('record_type')=='root_complete': complete=(d.get('exit_code')==0)
        except Exception: pass
    item={'root':root,'status':'complete' if p.returncode==0 and complete else 'incomplete_or_error','started_local_epoch':started,'completed_local_epoch':completed,'elapsed_seconds':completed-started,'exit_code':p.returncode,'timed_out':timed_out,'candidate_count':count,'root_complete_marker':complete,'stdout_bytes':len(stdout),'stdout_sha256':hashlib.sha256(stdout).hexdigest(),'stderr_path':os.path.basename(err_path),'raw_path':os.path.basename(out),'status_path':os.path.basename(status_path)}
    with open(status_path,'w') as sf: json.dump(item,sf,indent=2,sort_keys=True); sf.write('\n')
    summary['roots'].append(item)
    print(root,item['status'],'candidates='+str(count),'exit='+str(p.returncode),flush=True)
    if p.returncode not in (0,None): continue
summary['completed_local_epoch']=time.time()
summary['complete_roots']=sum(1 for x in summary['roots'] if x['status']=='complete')
summary['incomplete_roots']=sum(1 for x in summary['roots'] if x['status']=='incomplete_or_error')
summary['total_candidate_records']=sum(x.get('candidate_count',0) for x in summary['roots'])
with open(BASE+'/candidate_scan_run_summary_20261123.json','w') as f: json.dump(summary,f,indent=2,sort_keys=True); f.write('\n')
print('SUMMARY',summary['complete_roots'],'complete roots,',summary['incomplete_roots'],'incomplete,',summary['total_candidate_records'],'candidate records',flush=True)
