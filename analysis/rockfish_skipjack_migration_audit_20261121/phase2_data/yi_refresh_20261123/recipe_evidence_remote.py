#!/usr/bin/env python3
import os,sys,json,stat,time,collections
BASE='/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/1.pwrelax/3-2.2-layer_CrSb2Te3/2.Cr-in-gap'
runs={
 'base_spin':{'input':BASE+'/spin/relax.in','script':BASE+'/spin/sbatch-relax.slurm','log':BASE+'/spin/relax.out','scheduler_log':None},
 'mag3':{'input':BASE+'/spin/mag3/relax.in','script':BASE+'/spin/mag3/sbatch-relax.slurm','log':BASE+'/spin/mag3/relax.out','scheduler_log':None},
 'charged3_spin':{'input':BASE+'/charged_3/spin/relax.in','script':BASE+'/charged_3/spin/sbatch-relax.slurm','alt_script':BASE+'/charged_3/spin/sbatch-pp.slurm','log':BASE+'/charged_3/spin/relax.out','scheduler_log':None},
}
patterns=('job done','convergence has been achieved','convergence not achieved','error in routine','error:','stopping because','maximum number of iterations')

def emit(d): sys.stdout.write(json.dumps(d,sort_keys=True,ensure_ascii=True,separators=(',',':'))+'\n');sys.stdout.flush()

def statrow(p):
 s=os.lstat(p); return {'path':p,'type':'symlink' if stat.S_ISLNK(s.st_mode) else 'file' if stat.S_ISREG(s.st_mode) else 'other','uid':s.st_uid,'gid':s.st_gid,'mode_octal':format(stat.S_IMODE(s.st_mode),'04o'),'size_bytes':s.st_size,'allocated_bytes':getattr(s,'st_blocks',0)*512,'mtime_ns':getattr(s,'st_mtime_ns',int(s.st_mtime*1e9)),'device':s.st_dev,'inode':s.st_ino,'nlink':s.st_nlink}

start=time.time(); emit({'record_type':'evidence_start','scope_root':BASE,'started_epoch':start,'reads':'QE input and Slurm script contents; log marker lines and tail only'})
for run,files in runs.items():
 emit({'record_type':'run_start','run':run})
 for kind,path in files.items():
  if not path: continue
  try:
   before=statrow(path)
   if kind in ('input','script','alt_script'):
    with open(path,'rb') as f: raw=f.read()
    after=statrow(path)
    before['content_utf8']=raw.decode('utf-8','replace')
    before['stable_during_read']=(before['size_bytes']==after['size_bytes'] and before['mtime_ns']==after['mtime_ns'] and before['inode']==after['inode'] and before['device']==after['device'])
    before['read_status']='complete'
    emit({'record_type':'recipe_file','run':run,'kind':kind,**before})
   elif kind=='log':
    last=collections.deque(maxlen=30); matched=[]; line_count=0
    with open(path,'r',errors='replace') as f:
     for line in f:
      line_count+=1; last.append(line.rstrip('\n'))
      low=line.lower()
      if any(p in low for p in patterns): matched.append(line.rstrip('\n'))
    after=statrow(path)
    emit({'record_type':'log_evidence','run':run,'kind':kind,**before,'line_count':line_count,'marker_lines':matched[-20:],'tail_lines':list(last),'stable_during_read':(before['size_bytes']==after['size_bytes'] and before['mtime_ns']==after['mtime_ns'] and before['inode']==after['inode'] and before['device']==after['device'])})
  except Exception as e: emit({'record_type':'evidence_error','run':run,'kind':kind,'path':path,'error_type':type(e).__name__,'error':str(e)})
 emit({'record_type':'run_complete','run':run,'completed_epoch':time.time()})
emit({'record_type':'evidence_complete','ended_epoch':time.time(),'elapsed_seconds':time.time()-start})
