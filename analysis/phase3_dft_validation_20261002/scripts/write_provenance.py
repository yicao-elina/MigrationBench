#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
files=[]
for p in sorted(ROOT.rglob('*')):
    if p.is_file() and p.name not in {'provenance_manifest.json'}:
        files.append({'path':str(p.relative_to(ROOT)),'size':p.stat().st_size,'sha256':sha(p)})
obj={'campaign':'phase3_dft_validation_20261002','source_remote':'/scratch16/pclancy3/yi/mace_four_model_MD_continuation_20260930/1-from-scratch/segment_from_000200000.extxyz','source_description':'2050-atom MACE trajectory; exact frame extraction by step metadata','reduced_model':'ordered block index 7 of 25; full cell (a,b,c) reduced to (a/5,b/5,c); 82 atoms; 81 model removes local Cr 81 and retains local Cr 82','qe_method':'QE 7.3.1-cpu; PBE+D3; nspin=2; cold smearing degauss 0.0146997236 Ry; ecutwfc=70 Ry; ecutrho=280 Ry; 2x2x2; fixed SCF; no relaxation/NEB/SOC claim','files':files}
(ROOT/'provenance_manifest.json').write_text(json.dumps(obj,indent=2)+'\n')
print(len(files))
