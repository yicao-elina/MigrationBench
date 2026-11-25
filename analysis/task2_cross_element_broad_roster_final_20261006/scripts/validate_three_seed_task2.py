#!/usr/bin/env python3
"""Fail-closed validation for the final three-seed Task 2 package."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = {"broad_roster_ft_seed123", "broad_roster_ft_seed456", "broad_roster_ft_seed789"}
ELEMENTS = {"Cu", "Li", "Na", "Sb", "Te", "Cr"}
PATHS = {"Cu_neb", "Cu_neb-QLs", "Cu_neb-gap-diffuse", "Cu_neb-gap2QL", "Li_neb-QLs", "Li_neb-gap-diffuse", "Li_neb-gap2Ql", "Na_neb-QLs", "Na_neb-gap-diffuse", "Na_neb-gap2QL"}


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    probe = json.loads((ROOT / "results/validation/model_element_probe.json").read_text(encoding="utf-8"))
    assert len(probe["records"]) == 4
    assert all(r["load_status"] == "loaded" for r in probe["records"])
    for record in probe["records"]:
        assert set(record["model_supported_elements"]) >= ELEMENTS
        assert all(v["status"] == "supported" and v["finite"] for v in record["element_probe"].values())

    profiles = rows(ROOT / "results/validation/diagnostic_profiles.csv")
    assert len(profiles) == 1200
    assert all(r["finite"] == "True" and r["scientific_success"] == "False" for r in profiles)
    assert {r["model_id"] for r in profiles} == {"foundation", *SEEDS}
    assert {r["path_id"] for r in profiles} == PATHS
    assert {r["variant"] for r in profiles} == {"base", "perturb_a_dx020", "perturb_b_tilt020"}
    assert all(math.isfinite(float(r["energy_eV"])) and math.isfinite(float(r["max_force_eV_A"])) for r in profiles)

    qe = rows(ROOT / "results/broad_roster/qe_neb_path_profiles.csv")
    assert len(qe) == 100
    assert all(r["ground_truth_status"] == "provisional_reference_only" for r in qe)
    dft = rows(ROOT / "results/table_1_dft_quality.csv")
    assert len(dft) == 10 and all(r["dft_scientific_success"] == "False" for r in dft)

    provenance = list((ROOT / "results/provenance").glob("seed*_final_model_provenance.json"))
    assert len(provenance) == 3
    for path in provenance:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["status"] == "complete"
        assert data["expected_epoch"] == 999
        assert data["state_dict_exact_match"] is True
        assert data["state_dict_max_abs_difference"] == 0.0
        assert len(data["checkpoint_candidates"]) == 1

    for figure in ("task2_three_seed_stability", "task2_three_seed_profiles", "task2_three_seed_validation"):
        for suffix in ("png", "svg", "pdf"):
            assert (ROOT / "figures" / f"{figure}.{suffix}").exists()
    print(json.dumps({"status": "pass", "models": 4, "profiles": 1200, "qe_reference_rows": 100, "dft_paths": 10, "provenance_files": 3}, indent=2))


if __name__ == "__main__":
    main()
