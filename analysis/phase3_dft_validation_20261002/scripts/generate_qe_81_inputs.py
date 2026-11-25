#!/usr/bin/env python3
import hashlib, json, re
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "inputs" / "ordered_block7"
OUT = ROOT / "qe_inputs_81"
OUT.mkdir(exist_ok=True)
PSEUDO = "/data/pclancy3/yi/ONCVPseudoPack/SG15_v1.2/UPF/PBE"

def parse(p):
    lines = p.read_text().splitlines(); n = int(lines[0]); c = lines[1]
    m = re.search(r'Lattice="([^"]+)"', c); cell = [float(x) for x in m.group(1).split()]
    rows = [line.split() for line in lines[2:2+n]]
    return cell, rows

manifest=[]
for p in sorted(INP.glob("snapshot_step_*_same_site81_ordered.extxyz")):
    step = int(re.search(r"step_(\d+)", p.name).group(1)); cell, rows = parse(p)
    if len(rows) != 81: raise RuntimeError(p)
    rows.sort(key=lambda f: {"Cr": 0, "Sb": 1, "Te": 2}[f[0]])
    def vec(i): return "  %.14f %.14f %.14f" % tuple(cell[i:i+3])
    text = """&CONTROL
  calculation = 'scf'
  prefix = 'mb_phase3_step_%06d'
  outdir = './out'
  pseudo_dir = '%s'
  verbosity = 'high'
/
&SYSTEM
  ibrav = 0
  nat = 81
  ntyp = 3
  ecutwfc = 70
  ecutrho = 280
  occupations = 'smearing'
  smearing = 'cold'
  degauss = 0.0146997236
  vdw_corr = 'DFT-D3'
/
&ELECTRONS
  conv_thr = 1.0d-6
  electron_maxstep = 400
  mixing_beta = 0.4
/
ATOMIC_SPECIES
Cr 51.99610 Cr_PBE_FR.SG15v1.2.UPF
Sb 121.76000 Sb_PBE_FR.SG15v1.2.UPF
Te 127.60000 Te_PBE_FR.SG15v1.2.UPF
K_POINTS automatic
2 2 2 1 1 1
CELL_PARAMETERS angstrom
%s
%s
%s
ATOMIC_POSITIONS angstrom
""" % (step, PSEUDO, vec(0), vec(3), vec(6))
    for f in rows:
        text += "  %-2s %.12f %.12f %.12f\n" % (f[0], float(f[1]), float(f[2]), float(f[3]))
    dest = OUT / ("step_%06d" % step); dest.mkdir(exist_ok=True); out=dest/"scf.in"; out.write_text(text)
    manifest.append({"step":step,"input":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"nat":81,"method":"PBE+D3, non-spin-polarized (nspin=1 default), cold smearing, 70/280 Ry, 2x2x2, fixed SCF"})
(OUT/"qe_input_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps(manifest,indent=2))
