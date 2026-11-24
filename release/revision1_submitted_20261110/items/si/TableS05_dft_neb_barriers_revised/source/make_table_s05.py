#!/usr/bin/env python3
"""SI Table S5 (tab:dft_neb_barriers_revised): QE forward/reverse references per registry path.

Same arithmetic as paper/revision1/scripts/build_path_registry.py (which also wrote a Status
column that was deleted on Overleaf): names and classes from the path registry, image counts and
barriers from dft_neb_barriers.csv. Each barrier is cross-checked against the last
"activation energy" pair and num_of_images of the in-repo QE neb.out.
Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import Evidence, fmt_num  # noqa: E402

DATA = "paper/revision1/data_processed"
REGISTRY = f"{DATA}/path_registry.csv"
BARRIERS = f"{DATA}/dft_neb_barriers.csv"
LEGACY_NEB = "data/raw_compact/legacy_61_qe_neb/{audit_id}/neb.out"
CHAIN_NEB = "data/raw_compact/revision1/81_atom_2D_421_Diffusion_traj/{audit_id}/neb.out"
DECIMALS = 2
TOLERANCE_EV = 1e-6
FORWARD = re.compile(r"activation energy \(->\)\s*=\s*([-0-9.]+)\s*eV")
REVERSE = re.compile(r"activation energy \(<-\)\s*=\s*([-0-9.]+)\s*eV")
IMAGES = re.compile(r"num_of_images\s*=\s*(\d+)")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def neb_out(audit_id: str) -> str:
    template = CHAIN_NEB if audit_id.startswith("neb_") else LEGACY_NEB
    return template.format(audit_id=audit_id)


def raw_mismatches(ev: Evidence, audit_id: str, barrier: dict) -> list[str]:
    text = ev.input(neb_out(audit_id)).read_text(errors="replace")
    forward, reverse = float(FORWARD.findall(text)[-1]), float(REVERSE.findall(text)[-1])
    images = int(IMAGES.search(text).group(1))
    problems = []
    if abs(forward - float(barrier["E_act_fwd_eV"])) > TOLERANCE_EV:
        problems.append(f"{audit_id}: forward {forward} vs csv {barrier['E_act_fwd_eV']}")
    if abs(reverse - float(barrier["E_act_rev_eV"])) > TOLERANCE_EV:
        problems.append(f"{audit_id}: reverse {reverse} vs csv {barrier['E_act_rev_eV']}")
    if images != int(barrier["n_images"]):
        problems.append(f"{audit_id}: num_of_images {images} vs csv {barrier['n_images']}")
    return problems


def main() -> int:
    ev = Evidence(__file__, "tab:dft_neb_barriers_revised")
    registry = read_csv(ev.input(REGISTRY))
    barriers = {r["path"]: r for r in read_csv(ev.input(BARRIERS))}
    rows, problems = [], []
    for entry in registry:
        barrier = barriers[entry["audit_id"]]
        problems += raw_mismatches(ev, entry["audit_id"], barrier)
        rows.append([entry["name"], entry["class"], str(int(barrier["n_images"])),
                     fmt_num(barrier["E_act_fwd_eV"], DECIMALS), fmt_num(barrier["E_act_rev_eV"], DECIMALS)])
    ev.check("dft_neb_barriers.csv equals last-iteration activation energies and num_of_images in neb.out",
             not problems, problems or f"{len(registry)} paths agree to {TOLERANCE_EV} eV")
    ev.compare_table(rows, "sn-article-SI.tex")
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
