#!/usr/bin/env python3
"""Parse only explicit QE acceptance markers; never infer success from Slurm state."""
import csv, math, re, sys
from pathlib import Path

def parse(path):
    text=path.read_text(errors='replace')
    energies=re.findall(r'!\s+total energy\s+=\s+([-+0-9.Ee]+)\s+Ry',text)
    forces=[]; in_force=False
    for line in text.splitlines():
        if 'Forces acting on atoms' in line: in_force=True; continue
        if in_force:
            m=re.search(r'force\s*=\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)',line)
            if m: forces.append([float(x) for x in m.groups()])
            elif forces and ('Total force' in line): in_force=False
    norms=[math.sqrt(sum(x*x for x in f))*13.605693009 for f in forces]
    return {'output':str(path),'energy_Ry':float(energies[-1]) if energies else None,'n_energy_lines':len(energies),'n_forces':len(forces),'max_force_eV_A':max(norms) if norms else None,'scf_converged':'convergence has been achieved' in text,'job_done':'JOB DONE' in text,'finite':all(math.isfinite(x) for x in norms) and bool(energies),'accepted':bool(energies and forces and 'convergence has been achieved' in text and 'JOB DONE' in text and all(math.isfinite(x) for x in norms))}

paths=[Path(x) for x in sys.argv[1:]]
if not paths: raise SystemExit('usage: parse_qe_outputs.py OUTPUT...')
rows=[parse(p) for p in paths]
out=paths[0].parent/'qe_results.csv'
with out.open('w',newline='') as h:
    w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('\n'.join(str(r) for r in rows)); print('wrote',out)
