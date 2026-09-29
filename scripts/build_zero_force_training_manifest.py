#!/usr/bin/env python3
"""Create a hash-bound, frame-level audit manifest for training extxyz files."""
import argparse, csv, hashlib, json, re
from pathlib import Path

PROP = re.compile(r'Properties=("[^"]+"|\S+)')

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def prop_start(header: str):
    m = PROP.search(header)
    if not m: return None
    fields = m.group(1).strip('"').split(':')
    col = 0
    for i in range(0, len(fields) - 2, 3):
        try: n = int(fields[i + 2])
        except ValueError: return None
        if fields[i] == 'forces': return col, n
        col += n
    return None

def audit(path: Path, dataset: str, source_script: str):
    rows, zeros, consecutive = [], [], 0
    file_hash = sha256(path)
    with path.open(errors='replace') as f:
        frame = 0
        while True:
            line = f.readline()
            if not line: break
            if not line.strip(): continue
            natoms = int(line)
            header = f.readline()
            loc = prop_start(header)
            vals = []
            for _ in range(natoms):
                parts = f.readline().split()
                if loc: vals.extend(float(x) for x in parts[loc[0]:loc[0]+loc[1]])
            frame += 1
            is_zero = bool(vals and all(v == 0.0 for v in vals))
            consecutive = consecutive + 1 if is_zero else 0
            if is_zero: zeros.append(frame)
            is_isolated = 'config_type=IsolatedAtom' in header
            zero_class = 'isolated_atom_reference' if is_zero and is_isolated else ('unclassified_zero_force' if is_zero else 'nonzero')
            rows.append({'dataset':dataset,'frame_index':frame,'source_file':str(path),
                         'source_script':source_script,'source_sha256':file_hash,
                         'natoms':natoms,'has_force_array':bool(loc),'all_zero_force':is_zero,
                         'zero_force_class':zero_class,
                         'max_abs_force_eV_A':max([abs(v) for v in vals]) if vals else None,
                         'consecutive_zero_run_ending_here':consecutive,
                         'source_header_prefix':header.strip()[:240]})
    return rows, zeros

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('--source-script',required=True); ap.add_argument('files',nargs='+'); a=ap.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True); all_rows=[]; summary=[]
    for spec in a.files:
        dataset, raw = spec.split('=',1) if '=' in spec else (Path(spec).stem, spec)
        p=Path(raw); rows, zeros=audit(p,dataset,a.source_script); all_rows += rows
        summary.append({'dataset':dataset,'path':str(p),'sha256':sha256(p),'frames':len(rows),'frames_with_force_array':sum(r['has_force_array'] for r in rows),'all_zero_force_frames':len(zeros),'unclassified_zero_force_frames':sum(r['zero_force_class']=='unclassified_zero_force' for r in rows),'zero_frame_indices':zeros})
    with (a.output_dir/'frame_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=all_rows[0].keys()); w.writeheader(); w.writerows(all_rows)
    (a.output_dir/'summary.json').write_text(json.dumps({'schema_version':'1.0','source_script':a.source_script,'files':summary,'total_frames':len(all_rows),'total_all_zero_force_frames':sum(x['all_zero_force_frames'] for x in summary),'total_unclassified_zero_force_frames':sum(x['unclassified_zero_force_frames'] for x in summary),'acceptance':'PASS' if not any(x['unclassified_zero_force_frames'] for x in summary) else 'FAIL'},indent=2)+'\n')
if __name__=='__main__': main()
