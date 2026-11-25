#!/usr/bin/env python3
import os,subprocess,time,json,hashlib,shlex
base='/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/yi_refresh_20261123'
root='/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/1.pwrelax/3-2.2-layer_CrSb2Te3/2.Cr-in-gap'
outs={'base_spin':root+'/spin/out','mag3':root+'/spin/mag3/out','charged3_spin':root+'/charged_3/spin/out'}
all_status=[]
for key,path in outs.items():
 fmt='%p\\t%y\\t%U\\t%G\\t%m\\t%s\\t%b\\t%T@\\t%D\\t%i\\t%n\\n'
 args=['find','-P',path,'-xdev','-maxdepth','3','-printf',fmt]
 remote='date -Is; '+' '.join(shlex.quote(x) for x in args)+'; rc=$?; printf "EXIT\\t%s\\t" "$rc"; date -Is; exit "$rc"'
 out=base+'/qe_'+key+'_out_listing_20261123.tsv';err=base+'/qe_'+key+'_out_listing_20261123.stderr'
 started=time.time()
 with open(out,'wb') as fo,open(err,'wb') as fe:p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=4','rockfish',remote],stdout=fo,stderr=fe)
 ended=time.time(); raw=open(out,'rb').read()
 s={'root':path,'started_local_epoch':started,'ended_local_epoch':ended,'elapsed_seconds':ended-started,'exit_code':p.returncode,'output_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'raw_path':os.path.basename(out),'stderr_path':os.path.basename(err)}
 with open(base+'/qe_'+key+'_out_listing_20261123.status.json','w') as f:json.dump(s,f,indent=2,sort_keys=True);f.write('\n')
 all_status.append(s);print(json.dumps(s),flush=True)
with open(base+'/qe_out_listing_status_20261123.json','w') as f:json.dump(all_status,f,indent=2,sort_keys=True);f.write('\n')
