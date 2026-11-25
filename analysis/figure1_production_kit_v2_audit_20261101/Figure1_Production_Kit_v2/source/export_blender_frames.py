#!/usr/bin/env python3
"""Copy selected real coordinate frames to XYZ and a metadata index; no rendering."""
from pathlib import Path
import csv,hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'inputs/blender_frames';OUT.mkdir(exist_ok=True)
requests=[
 ('A01_material','within_gap__ideal2_cr_canonical_in_gap.extxyz',[0]),
 ('A03_in_gap','within_gap__ideal2_cr_canonical_in_gap.extxyz',[0,4,9]),
 ('A04_entry','gap_to_ql__ideal1_cr_canonical_gap_to_ql.extxyz',[0,9]),
 ('A05_within_ql','within_ql__neb_3_sb2te3.extxyz',[0,4,9])]
records=[]
for slot,fn,indices in requests:
 path=ROOT/'inputs'/fn;ls=path.read_text().splitlines();frames=[];i=0
 while i<len(ls):
  if not ls[i].strip():i+=1;continue
  n=int(ls[i]);meta=ls[i+1];body=ls[i+2:i+n+2];frames.append((n,meta,body));i+=n+2
 sha=hashlib.sha256(path.read_bytes()).hexdigest()
 for idx in indices:
  n,meta,body=frames[idx]
  dst=OUT/f'{slot}_image{idx:02d}.xyz'
  dst.write_text(str(n)+'\n'+f'Units=angstrom Source={fn} image_index={idx} SHA256={sha}\n'+'\n'.join(body)+'\n')
  cell=re.search(r'Lattice="([^"]+)"',meta).group(1)
  records.append({'slot':slot,'file':str(dst.relative_to(ROOT)),'source_file':fn,'source_image_index':idx,
   'source_sha256':sha,'atoms':n,'cell_vectors_A':[float(x) for x in cell.split()],
   'purpose':'actual structural input only; no energy-table join or convergence acceptance implied'})
(ROOT/'data/Blender_Frame_Index.json').write_text(json.dumps(records,indent=2))
with open(ROOT/'data/Blender_Frame_Index.csv','w') as f:
 w=csv.writer(f);w.writerow(['slot','file','source_file','zero_based_image_index','source_sha256','atoms'])
 for r in records:w.writerow([r['slot'],r['file'],r['source_file'],r['source_image_index'],r['source_sha256'],r['atoms']])
print('Exported',len(records),'unchanged real frames; MD and sliding coordinates were not fabricated.')
