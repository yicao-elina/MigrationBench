#!/usr/bin/env python3
"""Generate geometry-matched Cr substitutions from the nine clean NEB inputs."""
from __future__ import annotations
import csv, hashlib, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path('/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/task3_endpoint_only_neb/remote_snapshot')
CASES = [
 ('2L-Cu__neb-QLs', '2L-Cu/NEB/neb-QLs', 'Cu'),
 ('2L-Cu__neb-gap2QL', '2L-Cu/NEB/neb-gap2QL', 'Cu'),
 ('2L-Cu__neb-gap-diffuse', '2L-Cu/NEB/neb-gap-diffuse', 'Cu'),
 ('2L-Li__neb-QLs', '2L-Li/NEB/neb-QLs', 'Li'),
 ('2L-Li__neb-gap2Ql', '2L-Li/NEB/neb-gap2Ql', 'Li'),
 ('2L-Li__neb-gap-diffuse', '2L-Li/NEB/neb-gap-diffuse', 'Li'),
 ('2L-Na__neb-QLs', '2L-Na/NEB/neb-QLs', 'Na'),
 ('2L-Na__neb-gap2QL', '2L-Na/NEB/neb-gap2QL', 'Na'),
 ('2L-Na__neb-gap-diffuse', '2L-Na/NEB/neb-gap-diffuse', 'Na'),
]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def species_line(line): return bool(re.match(r'^\s*(Cu|Li|Na)\s+[-+0-9.eEdD]+\s+\S+\.UPF\s*$', line))
def position_line(line, old):
    return bool(re.match(r'^\s*'+old+r'\s+[-+0-9.eEdD]+\s+[-+0-9.eEdD]+\s+[-+0-9.eEdD]+(?:\s*!.*)?\s*$', line))

def transform(text, old):
    out=[]; in_pos=False; species_changes=0; coord_changes=0
    for raw in text.splitlines(keepends=True):
        line=raw
        if re.match(r'^\s*ATOMIC_POSITIONS\b', line, re.I): in_pos=True
        if species_line(line):
            line=re.sub(r'^\s*'+old+r'(?=\s)', lambda m: m.group(0).replace(old,'Cr'), line, count=1)
            line=line.replace(old+'_PBE_FR.SG15v1.2.UPF','Cr_PBE_FR.SG15v1.2.UPF')
            species_changes += 1
        elif in_pos and position_line(line, old):
            line=re.sub(r'^\s*'+old+r'(?=\s)', lambda m: m.group(0).replace(old,'Cr'), line, count=1)
            coord_changes += 1
        if in_pos and re.match(r'^\s*(?:K_POINTS|CELL|BEGIN_POSITIONS|END_POSITIONS|BEGIN_ENGINE_INPUT|END_ENGINE_INPUT|BEGIN_PATH_INPUT|END_PATH_INPUT|&|/)', line, re.I):
            if re.match(r'^\s*(?:K_POINTS|CELL|BEGIN_ENGINE_INPUT|END_ENGINE_INPUT|BEGIN_PATH_INPUT|END_PATH_INPUT|&|/)', line, re.I): in_pos=False
        if "restart_mode" in line: line=re.sub(r"restart_mode\s*=\s*'[^']+'", "restart_mode = 'from_scratch'", line)
        out.append(line)
    return ''.join(out), species_changes, coord_changes

rows=[]
for case, rel, old in CASES:
    srcdir=SRC/rel; outdir=ROOT/'inputs'/case; outdir.mkdir(parents=True, exist_ok=True)
    for src in sorted(srcdir.glob('*.in')):
        dst=outdir/src.name; text=src.read_text(); new, ns, nc=transform(text, old)
        if ns != 1: raise SystemExit(f'{src}: expected one species replacement, got {ns}')
        if nc < 1: raise SystemExit(f'{src}: no coordinate replacement')
        dst.write_text(new)
        rows.append(dict(case=case, source=str(src), output=str(dst), source_sha256=sha(src), output_sha256=sha(dst), dopant_before=old, dopant_after='Cr', species_replacements=ns, coordinate_replacements=nc, restart_mode='from_scratch'))
with (ROOT/'manifests'/'cr_substitution_manifest.tsv').open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter='\t'); w.writeheader(); w.writerows(rows)
print(f'generated {len(rows)} input files for {len(CASES)} paths')
