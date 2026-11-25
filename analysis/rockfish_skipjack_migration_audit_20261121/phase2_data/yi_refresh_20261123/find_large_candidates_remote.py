#!/usr/bin/env python3
"""Bounded read-only find of large paths and QE restart name candidates under one root."""
import os, sys, json, stat, time, subprocess, pwd, grp

root = sys.argv[1]
start = time.time()

def emit(d):
    sys.stdout.write(json.dumps(d, sort_keys=True, ensure_ascii=True, separators=(',', ':')) + '\n')
    sys.stdout.flush()

def name_for(fn, n, attr):
    try: return getattr(fn(n), attr)
    except Exception: return str(n)

try:
    st = os.lstat(root)
except Exception as e:
    emit({'record_type':'root_start','root':root,'started_epoch':start,'status':'stat_error','error_type':type(e).__name__,'error':str(e)})
    emit({'record_type':'root_complete','root':root,'ended_epoch':time.time(),'elapsed_seconds':time.time()-start,'exit_code':2,'candidate_count':0})
    sys.exit(2)
if not stat.S_ISDIR(st.st_mode):
    emit({'record_type':'root_start','root':root,'started_epoch':start,'status':'not_directory','type':'symlink' if stat.S_ISLNK(st.st_mode) else 'other'})
    emit({'record_type':'root_complete','root':root,'ended_epoch':time.time(),'elapsed_seconds':time.time()-start,'exit_code':0,'candidate_count':0})
    sys.exit(0)

emit({'record_type':'root_start','root':root,'started_epoch':start,'root_uid':st.st_uid,'root_gid':st.st_gid,'root_mode':format(stat.S_IMODE(st.st_mode),'04o'),'root_dev':st.st_dev,'root_ino':st.st_ino,'method':'GNU find -P -xdev, metadata only; no symlink traversal or file reads'})
fmt='%p\\0%U\\0%G\\0%m\\0%s\\0%b\\0%T@\\0%D\\0%i\\0%n\\0%y\\0'
patterns=['-iname','wfc*.dat','-o','-iname','*.wfc*','-o','-iname','mix*','-o','-iname','*.mix*']
find_expr=['-type','f','(', '-size','+500M','-o','(', '-size','+100M','(']+patterns+[')',')',')']
cmd=['find','-P',root,'-xdev']+find_expr+['-printf',fmt]
proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
raw,err=proc.communicate()
fields=raw.split(b'\x00')
if fields and fields[-1]==b'': fields.pop()
count=0
for i in range(0,len(fields),11):
    chunk=fields[i:i+11]
    if len(chunk)!=11: break
    vals=[os.fsdecode(x) for x in chunk]
    path,uid,gid,mode,size,blocks,mtime,dev,ino,nlink,typ=vals
    size=int(size); blocks=int(blocks); uid=int(uid); gid=int(gid); dev=int(dev); ino=int(ino); nlink=int(nlink)
    base=os.path.basename(path).lower()
    qe_restart = (base.startswith('wfc') and base.endswith('.dat')) or '.wfc' in base or base.startswith('mix') or '.mix' in base
    categories=[]
    if size>500*1024*1024: categories.append('large_gt_500MiB')
    if qe_restart and size>100*1024*1024: categories.append('qe_wfc_or_mix_gt_100MiB')
    emit({'record_type':'candidate','root':root,'path':path,'type':typ,'uid':uid,'owner':name_for(pwd.getpwuid,uid,'pw_name'),'gid':gid,'group':name_for(grp.getgrgid,gid,'gr_name'),'mode_octal':mode,'size_bytes':size,'allocated_bytes':blocks*512,'mtime_epoch':float(mtime),'device':dev,'inode':ino,'nlink':nlink,'categories':categories})
    count+=1
emit({'record_type':'root_complete','root':root,'ended_epoch':time.time(),'elapsed_seconds':time.time()-start,'exit_code':proc.returncode,'candidate_count':count,'stderr':os.fsdecode(err) if err else ''})
sys.exit(proc.returncode if proc.returncode is not None else 2)
