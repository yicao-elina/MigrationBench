import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "migrationbench"))

from plot_training_leakage_audit import quantile, summarize  # noqa: E402


def test_quantile_interpolates_sorted_values():
    assert quantile([4.0, 1.0, 3.0, 2.0], 0.5) == pytest.approx(2.5)


def test_summary_reports_threshold_margins_and_incompatible_rmsd():
    rows = [
        {"reference_split": "train", "soap_cosine_distance": "0.0002", "cr_local_species_rmsd_A": "0.1"},
        {"reference_split": "train", "soap_cosine_distance": "0.0003", "cr_local_species_rmsd_A": ""},
        {"reference_split": "valid", "soap_cosine_distance": "0.0004", "cr_local_species_rmsd_A": "0.2"},
        {"reference_split": "test", "soap_cosine_distance": "0.0005", "cr_local_species_rmsd_A": "0.3"},
    ]
    by_split = {row["reference_split"]: row for row in summarize(rows)}
    assert by_split["train"]["soap_lte_threshold"] == 0
    assert by_split["train"]["soap_min_over_threshold"] == pytest.approx(2.0)
    assert by_split["train"]["rmsd_comparable_rows"] == 1
    assert by_split["train"]["rmsd_incompatible_species_cardinality_rows"] == 1
    assert by_split["train"]["rmsd_min_over_threshold"] == pytest.approx(2.0)
