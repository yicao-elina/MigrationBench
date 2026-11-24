#!/usr/bin/env python3
"""Read-only checkpoint metadata audit for isolated generalization testing."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
MODELS = {
    "foundation": Path("../task2_cross_element/models/Foundation-MACE-matpes-pbe-omat-ft.model"),
    "ft600k": Path("../task2_cross_element/models/FT-600K.model"),
    "ft_multit": Path("../task2_cross_element/models/FT_Multi_T.model"),
}

def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1<<20), b""): h.update(block)
    return h.hexdigest()

def main():
    rows=[]
    for name, rel in MODELS.items():
        p=(ROOT/rel).resolve(); row={"model_id":name,"path":str(p),"exists":p.exists()}
        if not p.exists(): rows.append(row); continue
        row.update({"sha256":sha256(p),"bytes":p.stat().st_size})
        try:
            if name == "ft600k":
                m=torch.jit.load(str(p),map_location="cpu")
            else:
                m=torch.load(str(p),map_location="cpu",weights_only=False)
            z=getattr(m,"atomic_numbers",None)
            row["load_status"]="loaded"
            row["atomic_numbers"] = [int(x) for x in z.detach().cpu().tolist()] if z is not None else None
            row["load_note"]="read-only metadata inspection"
        except Exception as e:
            row.update({"load_status":"failed","error":f"{type(e).__name__}: {str(e).replace(chr(10),' ')[:800]}"})
        rows.append(row)
    out=ROOT/"checkpoint_audit.json"; out.write_text(json.dumps({"schema_version":"1.0","rows":rows},indent=2)+"\n")
    print(json.dumps(rows,indent=2))

if __name__ == "__main__": main()
