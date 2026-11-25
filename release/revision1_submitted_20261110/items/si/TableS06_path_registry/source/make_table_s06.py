#!/usr/bin/env python3
"""SI Table S6 (tab:path_registry): path names <-> audit IDs with classification basis.

Labels (name, class, audit ID, basis wording) come from the REGISTRY constant of the existing
generator paper/revision1/scripts/build_path_registry.py (imported, not run); atom counts are read
from the first line of each in-repo QE image file (sb2te3.xyz). The committed
path_registry.csv is cross-checked against both.

Known wording drift (recorded in an informational check):
  * G1: the generator says "image-set reference path"; the committed path_registry.csv was not
    regenerated and still says "barrier reference path".
  * L1: generator and CSV say "probable deep penetration" (AUDIT_REPORT.md:
    probable_deep_penetration, canonical mapping "ambiguous"); the submitted text, edited on
    Overleaf, says Cr is within the quintuple layer, matching the registry class "within-QL".
    Exact-coordinate rows are therefore worded from the registry class (BASIS_BY_CLASS).
Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import Evidence  # noqa: E402

GENERATOR = "paper/revision1/scripts/build_path_registry.py"
REGISTRY_CSV = "paper/revision1/data_processed/path_registry.csv"
LEGACY_XYZ = "data/raw_compact/legacy_61_qe_neb/{audit_id}/sb2te3.xyz"
CHAIN_XYZ = "data/raw_compact/revision1/81_atom_2D_421_Diffusion_traj/{audit_id}/sb2te3.xyz"
EXACT_COORDINATE_EVIDENCE = "paper/revision1/figures/si_path_audit/AUDIT_REPORT.md"
BASIS_BY_CLASS = {  # wording of the exact-coordinate rows, keyed by the registry class
    "mixed": "Exact image coordinates give a mixed gap/QL transition",
    "within-QL": "Exact image coordinates place Cr within the quintuple layer",
}


def load_generator_registry(ev: Evidence) -> list[dict]:
    spec = importlib.util.spec_from_file_location("build_path_registry", ev.input(GENERATOR))
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True  # never leave a __pycache__ in the frozen paper/ tree
    spec.loader.exec_module(module)  # module level only defines constants; main() is not called
    keys = ["name", "class", "atoms", "audit_id", "label", "basis", "evidence"]
    return [dict(zip(keys, entry)) for entry in module.REGISTRY]


def xyz_atoms(ev: Evidence, audit_id: str) -> int:
    template = CHAIN_XYZ if audit_id.startswith("neb_") else LEGACY_XYZ
    with ev.input(template.format(audit_id=audit_id)).open() as handle:
        return int(handle.readline().split()[0])


def basis(entry: dict) -> str:
    return BASIS_BY_CLASS[entry["class"]] if entry["evidence"] == EXACT_COORDINATE_EVIDENCE else entry["basis"]


def main() -> int:
    ev = Evidence(__file__, "tab:path_registry")
    registry = load_generator_registry(ev)
    with ev.input(REGISTRY_CSV).open(newline="") as handle:
        committed = {r["name"]: r for r in csv.DictReader(handle)}
    rows, problems, wording = [], [], []
    for entry in registry:
        atoms = xyz_atoms(ev, entry["audit_id"])
        csv_row = committed[entry["name"]]
        for field, value in (("class", entry["class"]), ("audit_id", entry["audit_id"]), ("atoms", str(atoms))):
            if csv_row[field] != value:
                problems.append(f"{entry['name']} {field}: data {value} vs path_registry.csv {csv_row[field]}")
        if atoms != entry["atoms"]:
            problems.append(f"{entry['name']}: xyz {atoms} atoms vs generator constant {entry['atoms']}")
        text = basis(entry)
        if text != csv_row["classification_basis"] or text != entry["basis"]:
            wording.append({"name": entry["name"], "rendered": text, "generator": entry["basis"],
                            "path_registry_csv": csv_row["classification_basis"]})
        rows.append([entry["name"], entry["class"], str(atoms),
                     f"\\texttt{{{entry['audit_id'].replace('_', chr(92) + '_')}}}", text])
    ev.check("atoms (sb2te3.xyz), class and audit ID agree with path_registry.csv and the generator",
             not problems, problems or f"{len(rows)} paths agree")
    ev.check("basis wording differing from generator or path_registry.csv (informational)", True, wording)
    ev.compare_table(rows, "sn-article-SI.tex")
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
