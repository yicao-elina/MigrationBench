#!/usr/bin/env python3
import glob,gzip,json,os,subprocess,time,hashlib
BASE='/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/yi_refresh_20261123'
rows=[]
for p in glob.glob(BASE+'/root_*_large_candidates_20261123.jsonl.gz'):
 with gzip.open(p,'rt') as f:
  for l in f:
   d=json.loads(l)
   if d.get('record_type')=='candidate' and 'qe_wfc_or_mix_gt_100MiB' in d.get('categories',[]): rows.append(d)
groups={}
for d in rows: groups.setdefault(os.path.dirname(d['path']),[]).append(d['path'])
remote_template=r'''import os,sys,json,stat,time,hashlib
paths=__PATHS__
start=time.time()
def emit(d): sys.stdout.write(json.dumps(d,sort_keys=True,ensure_ascii=True,separators=(',',':'))+'\\n'); sys.stdout.flush()
def fields(st):
 return {'type':'file' if stat.S_ISREG(st.st_mode) else 'other','uid':st.st_uid,'gid':st.st_gid,'mode_octal':format(stat.S_IMODE(st.st_mode),'04o'),'size_bytes':st.st_size,'allocated_bytes':getattr(st,'st_blocks',0)*512,'mtime_ns':getattr(st,'st_mtime_ns',int(st.st_mtime*1e9)),'device':st.st_dev,'inode':st.st_ino,'nlink':st.st_nlink}
emit({'record_type':'hash_group_start','group_root':os.path.dirname(paths[0]) if paths else None,'file_count':len(paths),'started_epoch':start,'method':'open O_RDONLY|O_NOFOLLOW; chunked SHA-256; pre/post fstat; no writes'})
for path in paths:
 try:
  flags=os.O_RDONLY
  if hasattr(os,'O_NOFOLLOW'): flags |= os.O_NOFOLLOW
  fd=os.open(path,flags)
  with os.fdopen(fd,'rb',closefd=True) as f:
   before=os.fstat(f.fileno()); ctx=hashlib.sha256(); total=0
   while True:
    b=f.read(16*1024*1024)
    if not b: break
    total+=len(b);ctx.update(b)
   after=os.fstat(f.fileno())
  bfields=fields(before); afields=fields(after)
  stable=all(bfields[k]==afields[k] for k in ('type','uid','gid','mode_octal','size_bytes','allocated_bytes','mtime_ns','device','inode','nlink')) and total==before.st_size
  emit({'record_type':'hash_candidate','path':path,'pre':bfields,'post':afields,'bytes_hashed':total,'sha256':ctx.hexdigest(),'stable_during_hash':stable,'ended_epoch':time.time()})
 except Exception as e:
  emit({'record_type':'hash_error','path':path,'error_type':type(e).__name__,'error':str(e)})
emit({'record_type':'hash_group_complete','group_root':os.path.dirname(paths[0]) if paths else None,'ended_epoch':time.time(),'elapsed_seconds':time.time()-start,'file_count':len(paths)})
'''
status=[]
for index,(root,paths) in enumerate(sorted(groups.items()),1):
 slug='mix_group_'+str(index)
 remote=remote_template.replace('__PATHS__',repr(paths))
 out=BASE+'/'+slug+'_sha256_20261123.jsonl'; err=BASE+'/'+slug+'_sha256_20261123.stderr'; status_path=BASE+'/'+slug+'_sha256_20261123.status.json'
 started=time.time(); p=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=4','rockfish','python3','-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 timed_out=False
 try: stdout,stderr=p.communicate(remote.encode('utf-8'),timeout=900)
 except subprocess.TimeoutExpired:
  timed_out=True; p.kill(); stdout,stderr=p.communicate()
 ended=time.time()
 with open(out,'wb') as f:f.write(stdout)
 with open(err,'wb') as f:f.write(stderr)
 records=[]
 for l in stdout.splitlines():
  try:records.append(json.loads(l))
  except Exception:pass
 hashes=sum(1 for x in records if x.get('record_type')=='hash_candidate')
 errors=sum(1 for x in records if x.get('record_type')=='hash_error')
 complete=any(x.get('record_type')=='hash_group_complete' for x in records)
 item={'group_root':root,'file_count_expected':len(paths),'hash_records':hashes,'error_records':errors,'status':'complete' if p.returncode==0 and complete and not errors else 'incomplete_or_error','exit_code':p.returncode,'timed_out':timed_out,'started_local_epoch':started,'completed_local_epoch':ended,'elapsed_seconds':ended-started,'stdout_bytes':len(stdout),'stdout_sha256':hashlib.sha256(stdout).hexdigest(),'output_path':os.path.basename(out),'stderr_path':os.path.basename(err)}
 with open(status_path,'w') as f:json.dump(item,f,indent=2,sort_keys=True);f.write('\n')
 status.append(item); print(json.dumps(item),flush=True)
with open(BASE+'/qe_restart_hash_run_summary_20261123.json','w') as f:json.dump({'groups':status,'total_candidate_files':sum(x['file_count_expected'] for x in status),'total_hash_records':sum(x['hash_records'] for x in status)},f,indent=2,sort_keys=True);f.write('\n')
