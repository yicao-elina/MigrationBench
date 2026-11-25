#!/usr/bin/env python3
import os,json,subprocess,time,hashlib
base='/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/yi_refresh_20261123'
paths=[
'/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/1.2_bands/1.vcrelax/Sb2Te3/atomic_proj.xml',
'/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/1.pwrelax/3.Cr-Sb2Te3/*new_vdw_relaxed/antisite/1-Sb10/bands/atomic_proj.xml',
'/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/1.pwrelax/2.Sb2Te3/5.soc+optb86b/bands/atomic_proj.xml',
'/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/1.pwrelax/2.Sb2Te3/2.vdw_corr/222/bands_before/atomic_proj.xml',
'/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/2.aimd/2.Sb2Te3/650K-vc_pbe/aimd.out',
'/data/pclancy3/yi/.conda/envs/deepmd2/lib/libcublasLt.so.11.11.3.6',
'/data/pclancy3/yi/.conda/envs/mace/lib/python3.9/site-packages/nvidia/cublas/lib/libcublasLt.so.11',
'/data/pclancy3/yi/.conda/pkgs/cudatoolkit-11.8.0-h4ba93d1_13/lib/libcublasLt.so.11.11.3.6',
]
remote='''import os,sys,json,time,stat\npaths='''+repr(paths)+'''\ndef emit(d): print(json.dumps(d,sort_keys=True,separators=(',',':')))\nprint(json.dumps({'record_type':'stat_start','started_epoch':time.time(),'path_count':len(paths)},sort_keys=True,separators=(',',':')))\nfor p in paths:\n try:\n  s=os.stat(p,follow_symlinks=False)\n  emit({'record_type':'stat','path':p,'type':'file' if stat.S_ISREG(s.st_mode) else 'directory' if stat.S_ISDIR(s.st_mode) else 'symlink' if stat.S_ISLNK(s.st_mode) else 'other','uid':s.st_uid,'gid':s.st_gid,'mode_octal':format(stat.S_IMODE(s.st_mode),'04o'),'size_bytes':s.st_size,'allocated_bytes':getattr(s,'st_blocks',0)*512,'mtime_ns':getattr(s,'st_mtime_ns',int(s.st_mtime*1e9)),'device':s.st_dev,'inode':s.st_ino,'nlink':s.st_nlink})\n except Exception as e: emit({'record_type':'stat_error','path':p,'error_type':type(e).__name__,'error':str(e)})\nprint(json.dumps({'record_type':'stat_complete','ended_epoch':time.time()},sort_keys=True,separators=(',',':')))\n'''
started=time.time(); p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=4','rockfish','python3','-'],input=remote.encode(),capture_output=True); ended=time.time()
out=base+'/fresh_duplicate_stat_20261123.jsonl';err=base+'/fresh_duplicate_stat_20261123.stderr'
open(out,'wb').write(p.stdout);open(err,'wb').write(p.stderr)
d={'started_local_epoch':started,'ended_local_epoch':ended,'elapsed_seconds':ended-started,'exit_code':p.returncode,'path_count':len(paths),'stdout_bytes':len(p.stdout),'sha256':hashlib.sha256(p.stdout).hexdigest(),'stderr_path':os.path.basename(err)}
with open(base+'/fresh_duplicate_stat_20261123.status.json','w') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(d,indent=2))
