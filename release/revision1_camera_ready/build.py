#!/usr/bin/env python3
"""Compile all reader and review copies, including cross-document references."""
from pathlib import Path
import subprocess
import re
ROOT = Path(__file__).resolve().parent
paper = ROOT.parents[1]/'paper'  # repo-level paper/ tree
logs = ROOT/'build_logs'
logs.mkdir(exist_ok=True)
order = ['sn-article-final','sn-article-SI','sn-article-final','sn-article-SI',
         'sn-article-clean','sn-article-SI-clean','response_to_reviewers']
for i,name in enumerate(order):
    log=logs/f'{i+1:02d}_{name}.txt'
    with log.open('w') as stream:
        proc=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode',
            '-halt-on-error','-outdir=revision1',f'revision1/{name}.tex'],
            cwd=paper,stdout=stream,stderr=subprocess.STDOUT)
    if proc.returncode:
        print(log.read_text()[-4500:],flush=True)
        raise SystemExit(f'Build failed: {name}; see {log}')
    print(f'Compiled {name}',flush=True)
for name in ['sn-article-final','sn-article-SI','sn-article-clean','sn-article-SI-clean','response_to_reviewers']:
    s=(paper/'revision1'/f'{name}.log').read_text()
    unresolved=re.findall(r"(?:Reference|Citation) .([^']+)'[^\n]*undefined",s)
    bad=[l for l in s.splitlines() if 'Overfull' in l]
    print(name,'unresolved=',unresolved,'overfull=',bad,flush=True)
    if unresolved: raise SystemExit(f'Unresolved reference in {name}')
