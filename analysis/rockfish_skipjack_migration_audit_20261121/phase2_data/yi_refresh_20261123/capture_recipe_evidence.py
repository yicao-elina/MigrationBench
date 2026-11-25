#!/usr/bin/env python3
import os,subprocess,time,hashlib,json
base='/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/yi_refresh_20261123'
script=open(base+'/recipe_evidence_remote.py','rb').read()
out=base+'/recipe_evidence_20261123.jsonl';err=base+'/recipe_evidence_20261123.stderr'
started=time.time()
with open(out,'wb') as fo,open(err,'wb') as fe:
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=4','rockfish','python3','-'],input=script,stdout=fo,stderr=fe)
ended=time.time()
d={'started_local_epoch':started,'ended_local_epoch':ended,'elapsed_seconds':ended-started,'exit_code':p.returncode,'output_bytes':os.path.getsize(out),'sha256':hashlib.sha256(open(out,'rb').read()).hexdigest(),'stderr_path':os.path.basename(err)}
with open(base+'/recipe_evidence_20261123.status.json','w') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(d,indent=2))
