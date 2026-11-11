#!/usr/bin/env python3
"""Fig. S3: foundation-model fixed-path responses for Cr/Cu/Li/Na (4 x 3 panels).

The generator plots two committed tables in source_data/: the R03 MACE foundation-class
aggregate profiles and the screened QE reference profiles.  The QE table is checked here against
its upstream sources (Cu/Li/Na: upstream/qe_neb_path_profiles.csv, energies relative to image 1;
Cr: the 81-atom QE neb.out files parsed by src/migrationbench/dft_neb_reference.py).  The class
aggregate comes from 1080 per-model fixed-image evaluations kept on Rockfish and is taken as given.
The generator rendered the submitted SI asset directly, so that asset is the reference.

Writes rerun_output/ and evidence/FigS3.json.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import REPO, Evidence, lint_script  # noqa: E402

ITEM = "release/revision1_submitted_20261110/items/si/FigS3_four_dopant_fixed_path"
GENERATOR = f"{ITEM}/source/render_four_dopant_fixed_path.py"
CLASS_PROFILES = f"{ITEM}/source/source_data/foundation_model_class_profiles_4x3_R03.tsv"
QE_PROFILES = f"{ITEM}/source/source_data/foundation_model_qe_reference_profiles_4x3_R03.tsv"
QE_UPSTREAM = f"{ITEM}/upstream/qe_neb_path_profiles.csv"
DFT_HELPER = "src/migrationbench/dft_neb_reference.py"
DFT_RAW = "data/raw_compact/cr81_dft_neb_reference_20261013/raw"
PALETTE = "reproducibility/canva_palette_vector_bundle_20261105/MIGRATIONBENCH_FIGURE_PALETTE.json"
FINAL_ASSET = "paper/revision1/figures/main/SI/fig_NEB-fixed-path_Cu-Li-Na-Cr.pdf"
CR_PATH_KEY = {"within_gap": "in_gap", "gap_to_ql": "gap_to_ql", "within_ql": "within_ql"}
ENERGY_TOLERANCE_EV = 1e-6


def read(path: Path, delimiter: str) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def upstream_key(path_id: str) -> str:
    """'2L-Cu__neb-gap2QL' (plotted table) -> 'Cu_neb-gap2QL' (upstream CSV)."""
    return re.sub(r"^2L-(\w+?)__", r"\1_", path_id)


def check_dopant_references(ev: Evidence, rows: list[dict]) -> None:
    upstream: dict[str, dict[int, float]] = {}
    for row in read(ev.input(QE_UPSTREAM), ","):
        upstream.setdefault(row["path_id"], {})[int(row["image"])] = float(row["qe_neb_energy_eV"])
    errors, compared = [], 0
    for row in rows:
        if row["element"] == "Cr":
            continue
        energies = upstream[upstream_key(row["path_id"])]
        expected = energies[int(row["image_index"])] - energies[min(energies)]
        compared += 1
        if abs(expected - float(row["relative_energy_eV"])) > ENERGY_TOLERANCE_EV:
            errors.append(f"{row['path_id']} image {row['image_index']}: {row['relative_energy_eV']} vs {expected}")
    ev.check("Cu/Li/Na QE references equal upstream neb energies relative to image 1", not errors,
             errors[:10] or f"{compared} points within {ENERGY_TOLERANCE_EV} eV")


def check_cr_reference(ev: Evidence, rows: list[dict]) -> None:
    ev.input(DFT_HELPER)
    for neb_out in sorted((REPO / DFT_RAW).glob("2L-Cu__neb-*/neb.out")):
        ev.input(neb_out.relative_to(REPO).as_posix())
    spec = importlib.util.spec_from_file_location("dft_neb_reference", REPO / DFT_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profiles = module.profiles_for_plot()
    cr_rows = [row for row in rows if row["element"] == "Cr"]
    first = {key: min(int(r["image_index"]) for r in cr_rows if r["path_key"] == key) for key in CR_PATH_KEY}
    errors, compared = [], 0
    for row in cr_rows:
        _, energies = profiles[CR_PATH_KEY[row["path_key"]]]
        compared += 1
        position = int(row["image_index"]) - first[row["path_key"]]  # Cr rows count images from 0
        if abs(energies[position] - float(row["relative_energy_eV"])) > ENERGY_TOLERANCE_EV:
            errors.append(f"{row['path_key']} image {row['image_index']}")
    ev.check("Cr QE references equal the parsed neb.out profiles", not errors,
             errors[:10] or f"{compared} points within {ENERGY_TOLERANCE_EV} eV")


def main() -> int:
    ev = Evidence(__file__, "FigS3")
    for path in (GENERATOR, CLASS_PROFILES, PALETTE, FINAL_ASSET):
        ev.input(path)
    findings = lint_script(REPO / GENERATOR)
    ev.check(f"hardcode lint {GENERATOR}", not findings, findings or "no findings")
    qe_rows = read(ev.input(QE_PROFILES), "\t")
    check_dopant_references(ev, qe_rows)
    check_cr_reference(ev, qe_rows)

    stem = ev.output("fig_NEB-fixed-path_Cu-Li-Na-Cr_palette_updated")
    run = subprocess.run([sys.executable, str(REPO / GENERATOR)], cwd=REPO, capture_output=True, text=True,
                         env={**os.environ, "FOUR_DOPANT_OUT": str(stem)})
    if ev.check("generator exits 0", run.returncode == 0, run.stderr[-2000:] or "ok"):
        ev.compare_pdf(stem.with_suffix(".pdf"), FINAL_ASSET)
    ev.check("known limitation (not used for PASS)", True,
             "foundation-class aggregate (mean/SD over n_models per class) is plotted as committed; its 1080 "
             "per-model fixed-image evaluations stay on Rockfish (foundation_self_relaxed_4x3_20261024_r02/runs, "
             "see PROVENANCE.md raw location) and were not re-aggregated here")
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
