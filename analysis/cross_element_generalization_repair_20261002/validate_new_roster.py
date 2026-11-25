#!/usr/bin/env python3
"""Validate the atomic-number roster of a newly produced checkpoint."""
import argparse, json
import torch
EXPECTED={3,11,23,24,25,29,35,42,51,52,55,82}
ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
try:
    m=torch.jit.load(a.model,map_location='cpu')
except Exception:
    m=torch.load(a.model,map_location='cpu',weights_only=False)
z={int(x) for x in m.atomic_numbers.detach().cpu().tolist()}
r={'model':a.model,'atomic_numbers':sorted(z),'expected_atomic_numbers':sorted(EXPECTED),'roster_contains_target':EXPECTED.issubset(z),'missing':sorted(EXPECTED-z),'unexpected':sorted(z-EXPECTED)}
open(a.out,'w').write(json.dumps(r,indent=2)+'\n'); print(json.dumps(r,indent=2))
