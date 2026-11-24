#!/usr/bin/env python3
"""Apply the completed replacement, with source guards and reversible backups."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--target',type=Path,default=Path('/Users/alina/Project/migrationbench-paper-repro/paper/revision1'))
group=parser.add_mutually_exclusive_group()
group.add_argument('--apply',action='store_true')
group.add_argument('--check',action='store_true')
args=parser.parse_args()
manifest=json.loads((HERE/'replacement_manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,wanted in manifest['original_source_hashes'].items():
    path=args.target/name
    if not path.is_file() or digest(path)!=wanted:
        raise SystemExit(f'Refusing to overwrite a changed original source: {path}')
planned=[]
for name in manifest['updated_revision_files']:
    src=HERE/'paper/revision1'/name
    dst=args.target/name
    if not src.is_file():raise SystemExit(f'Missing replacement file: {src}')
    if not dst.exists() or digest(src)!=digest(dst):planned.append((name,src,dst))
print(f'Original-source guards: PASS. Planned changed/new files: {len(planned)}.')
if not args.apply:
    print('Check only; no target file was written.')
    raise SystemExit(0)
backup=args.target/('camera_ready_backup_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
backup.mkdir(parents=True,exist_ok=False)
for name,src,dst in planned:
    if dst.exists():
        saved=backup/name
        saved.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(dst,saved)
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
print(f'Applied {len(planned)} files. Previous versions are preserved in {backup}.')
