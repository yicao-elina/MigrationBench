#!/usr/bin/env python3
from pathlib import Path
import re, sys
ROOT=Path(__file__).resolve().parents[1]
fails=[]
for d in sorted((ROOT/'inputs').iterdir()):
    if not d.is_dir(): continue
    for p in sorted(d.glob('*.in')):
        s=p.read_text()
        if 'Cr_PBE_FR.SG15v1.2.UPF' not in s: fails.append((p,'missing Cr pseudo'))
        if re.search(r"restart_mode\s*=\s*'restart'",s): fails.append((p,'restart mode is restart'))
        if re.search(r'^\s*(Cu|Li|Na)\s+[-+0-9.eEdD]+\s+',s,re.M): fails.append((p,'old dopant remains'))
        m=re.search(r'\bnat\s*=\s*(\d+)',s,re.I)
        if not m or m.group(1)!='81': fails.append((p,'nat is not 81'))
        if p.name == 'neb.in' and not re.search(r'num_of_images\s*=\s*10',s): fails.append((p,'not 10 images'))
if fails:
    for p,msg in fails: print(f'FAIL {p}: {msg}')
    sys.exit(1)
print('PASS: all generated inputs have Cr pseudo, no old dopant, from_scratch, nat=81, 10 images')
