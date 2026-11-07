#!/usr/bin/env python3
"""Build the exact 82-atom ordered periodic block and the 81-atom same-site approximation."""
import csv, json, math, re
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "inputs" / "from_scratch"
OUT = ROOT / "inputs" / "ordered_block7"
OUT.mkdir(parents=True, exist_ok=True)
BLOCK = 7
BLOCK_N = 82

def read_extxyz(path):
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    comment = lines[1]
    rows = []
    for line in lines[2:2+n]:
        f = line.split()
        rows.append({"symbol": f[0], "pos": [float(x) for x in f[1:4]],
                     "mom": [float(x) for x in f[4:7]], "force": [float(x) for x in f[7:10]]})
    return n, comment, rows

def localize(rows, comment):
    m = re.search(r'Lattice="([^"]+)"', comment)
    full = np.array([float(x) for x in m.group(1).split()]).reshape(3, 3)
    small = full.copy(); small[0] /= 5.0; small[1] /= 5.0
    inv = np.linalg.inv(full)
    out = []
    for r in rows:
        frac = np.array(r["pos"]) @ inv
        frac[:2] = (frac[:2] * 5.0) % 1.0
        q = dict(r); q["pos"] = (frac @ small).tolist(); out.append(q)
    lat = " ".join("%.14f" % x for x in small.reshape(-1))
    return out, re.sub(r'Lattice="[^"]+"', 'Lattice="%s"' % lat, comment)

def write_extxyz(path, rows, comment):
    with path.open("w") as h:
        h.write(str(len(rows)) + "\n")
        h.write(comment + "\n")
        for r in rows:
            vals = r["pos"] + r["mom"] + r["force"]
            h.write(r["symbol"] + " " + " ".join("%.12f" % x for x in vals) + "\n")

def lattice(comment):
    m = re.search(r'Lattice="([^"]+)"', comment)
    if not m: return None
    return [float(x) for x in m.group(1).split()]

audit = []
manifest = {"source": str(SRC), "block_index_zero_based": BLOCK, "block_size": BLOCK_N,
            "retained_cr_local_index_one_based": 82, "removed_cr_local_index_one_based": 81,
            "models": []}
for src in sorted(SRC.glob("snapshot_step_*.extxyz")):
    step = int(re.search(r"step_(\d+)", src.name).group(1))
    n, comment, rows = read_extxyz(src)
    start = BLOCK * BLOCK_N
    block = rows[start:start+BLOCK_N]
    if len(block) != BLOCK_N or [r["symbol"] for r in block].count("Cr") != 2:
        raise RuntimeError("unexpected ordered block at %s" % src)
    block, local_comment = localize(block, comment)
    kept = [r for i, r in enumerate(block) if i != 80]  # remove local Cr 81; retain local Cr 82
    out_comment = local_comment + " reduced_model=ordered_block7 source_natoms=2050 source_block_atoms=575-656"
    out82 = OUT / ("snapshot_step_%06d_ordered82.extxyz" % step)
    out81 = OUT / ("snapshot_step_%06d_same_site81_ordered.extxyz" % step)
    write_extxyz(out82, block, out_comment + " model=exact_local_periodic_82")
    write_extxyz(out81, kept, out_comment + " model=same_site_81_one_Cr_removed")
    for label, arr in [("82", block), ("81", kept)]:
        norms = [math.sqrt(sum(x*x for x in r["force"])) for r in arr]
        cr_norms = [v for v, r in zip(norms, arr) if r["symbol"] == "Cr"]
        audit.append({"step": step, "model": label, "natoms": len(arr),
                      "composition": "Cr%d_Sb%d_Te%d" % (sum(r["symbol"]=="Cr" for r in arr), sum(r["symbol"]=="Sb" for r in arr), sum(r["symbol"]=="Te" for r in arr)),
                      "max_force_eV_A": max(norms), "cr_max_force_eV_A": max(cr_norms),
                      "finite": all(math.isfinite(v) for r in arr for key in ("pos", "mom", "force") for v in r[key]),
                      "source_sha256": __import__("hashlib").sha256(src.read_bytes()).hexdigest(),
                      "source": str(src), "output": str(out82 if label == "82" else out81)})
    manifest["models"].append({"step": step, "exact82": str(out82), "same_site81": str(out81),
                               "source_sha256": audit[-2]["source_sha256"], "block_max_force_eV_A": audit[-2]["max_force_eV_A"]})
(OUT / "ordered_block_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
with (ROOT / "ordered_block_audit.csv").open("w", newline="") as h:
    w = csv.DictWriter(h, fieldnames=list(audit[0])); w.writeheader(); w.writerows(audit)
(ROOT / "ordered_block_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps({"models": len(audit), "output": str(OUT)}))
