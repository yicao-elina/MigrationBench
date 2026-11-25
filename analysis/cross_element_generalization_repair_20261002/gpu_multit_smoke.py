#!/usr/bin/env python3
"""GPU-only FT-MultiT element and path smoke; never modifies the model."""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
from ase import Atoms

ELEMENTS={"Cr":24,"Cu":29,"Li":3,"Na":11,"Sb":51,"Te":52}

def probe_atoms(symbol):
    return Atoms([symbol,"Sb","Te"], positions=[[1,1,1],[3,3,3],[5,5,5]], cell=np.diag([12.,12.,12.]), pbc=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True); ap.add_argument("--out",required=True); ap.add_argument("--device",default="cuda"); args=ap.parse_args()
    p=Path(args.model); result={"model":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"device":args.device,"python":platform.python_version(),"status":"started","elements":{}}
    try:
        from mace.calculators import MACECalculator
        calc=MACECalculator(model_paths=str(p),device=args.device,default_dtype="float64")
        result["load_status"]="loaded"
        model=calc.models[0]
        z=getattr(model,"atomic_numbers",None)
        result["atomic_numbers"]=[int(x) for x in z.detach().cpu().tolist()] if z is not None else None
        for symbol, atomic_number in ELEMENTS.items():
            if result["atomic_numbers"] is not None and atomic_number not in result["atomic_numbers"]:
                result["elements"][symbol]={"status":"unsupported_by_metadata"}; continue
            try:
                atoms=probe_atoms(symbol); atoms.calc=calc
                e=float(atoms.get_potential_energy()); f=np.asarray(atoms.get_forces())
                result["elements"][symbol]={"status":"supported","finite":bool(np.isfinite(e) and np.isfinite(f).all()),"energy_eV":e,"max_force_eV_A":float(np.linalg.norm(f,axis=1).max())}
            except Exception as exc:
                result["elements"][symbol]={"status":"runtime_failure","error":f"{type(exc).__name__}: {str(exc).replace(chr(10),' ')[:500]}"}
        result["status"]="complete"
    except Exception as exc:
        result.update({"status":"runtime_failure","load_status":"failed","error":f"{type(exc).__name__}: {str(exc).replace(chr(10),' ')[:1000]}"})
    Path(args.out).write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))

if __name__ == "__main__": main()
