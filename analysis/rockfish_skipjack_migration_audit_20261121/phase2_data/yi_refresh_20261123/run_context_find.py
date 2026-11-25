#!/usr/bin/env python3
import os,subprocess,time,json,shlex,hashlib
base='/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data/yi_refresh_20261123'
root='/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/1.pwrelax/3-2.2-layer_CrSb2Te3/2.Cr-in-gap'
out=base+'/cr_in_gap_reproduction_context_20261123.tsv'
err=base+'/cr_in_gap_reproduction_context_20261123.stderr'
fmt='%p\\t%U\\t%G\\t%m\\t%s\\t%b\\t%T@\\t%D\\t%i\\t%n\\t%y\\n'
args=['find','-P',root,'-xdev','-maxdepth','6','-type','f','(',
      '-iname','*.in','-o','-iname','*.out','-o','-iname','*.xml','-o','-iname','*.slurm',
      '-o','-iname','*.sh','-o','-iname','*.py','-o','-iname','*manifest*','-o','-name','charge-density.dat',
      '-o','-iname','wfc*.dat','-o','-iname','*.mix*','-o','-iname','mix*','-o','-iname','submit*','-o','-iname','run*',')',
      '-printf',fmt]
remote='date -Is; '+' '.join(shlex.quote(x) for x in args)+'; rc=$?; printf "EXIT\\t%s\\t" "$rc"; date -Is; exit "$rc"'
started=time.time()
with open(out,'wb') as fo,open(err,'wb') as fe:
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=4','rockfish',remote],stdout=fo,stderr=fe)
completed=time.time()
d={'scope_root':root,'maxdepth':6,'path_types':'regular files only; default no symlink following','patterns':['*.in','*.out','*.xml','*.slurm','*.sh','*.py','*manifest*','charge-density.dat','wfc*.dat','*.mix*','mix*','submit*','run*'],'started_local_epoch':started,'completed_local_epoch':completed,'elapsed_seconds':completed-started,'exit_code':p.returncode,'output_bytes':os.path.getsize(out),'output_sha256':hashlib.sha256(open(out,'rb').read()).hexdigest(),'stderr_path':os.path.basename(err)}
with open(base+'/cr_in_gap_reproduction_context_20261123.status.json','w') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(d,indent=2))
