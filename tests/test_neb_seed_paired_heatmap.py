import subprocess
import sys
from pathlib import Path

import pandas as pd

from migrationbench.neb_seed_paired_heatmap import derive_tables, read_and_validate, render_bundle

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "neb_seed_paired_heatmap_20261031"


def _make_demo(path: Path) -> None:
    subprocess.run(
        [sys.executable, str(ANALYSIS / "scripts" / "make_synthetic_demo.py"), "--output", str(path)],
        check=True,
    )


def test_demo_has_complete_4x3x3_design(tmp_path: Path) -> None:
    fixture = tmp_path / "demo.tsv"
    _make_demo(fixture)
    frame, audit = read_and_validate(
        fixture,
        expected_families=4,
        expected_seeds_per_family=3,
        expected_paths=3,
        allow_synthetic_demo=True,
    )
    assert audit.model_seed_units == 12
    assert audit.rows == 360
    assert set(frame["observable"]) == {"FIXED_PATH_SINGLE_POINT"}
    errors, metrics, effects = derive_tables(frame, baseline_family="Foundation")
    assert len(errors) == 360
    assert len(metrics) == 36
    assert len(effects) == 27
    assert metrics["mae_eV"].gt(0).all()


def test_render_bundle_exports_editable_and_processed_products(tmp_path: Path) -> None:
    fixture = tmp_path / "demo.tsv"
    _make_demo(fixture)
    figures = tmp_path / "figures"
    processed = tmp_path / "processed"
    manifest = render_bundle(fixture, figures, processed, allow_synthetic_demo=True)
    assert manifest["input_audit"]["model_seed_units"] == 12
    assert (figures / "fig_neb_seed_performance_heatmap.svg").is_file()
    assert (figures / "fig_neb_seed_error_lineage_clustermap.pdf").is_file()
    assert (figures / "fig_reference1_bubble_matrix_synthetic.png").is_file()
    assert (processed / "NEB_MODEL_SEED_ROUTE_METRICS.tsv").is_file()
    metrics = pd.read_csv(processed / "NEB_MODEL_SEED_ROUTE_METRICS.tsv", sep="\t")
    assert metrics.shape[0] == 36


def test_geometry_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = tmp_path / "demo.tsv"
    _make_demo(fixture)
    frame = pd.read_csv(fixture, sep="\t")
    frame.loc[0, "dft_path_geometry_sha256"] = "different"
    frame.to_csv(fixture, sep="\t", index=False)
    try:
        read_and_validate(fixture, allow_synthetic_demo=True)
    except ValueError as error:
        assert "geometry hashes" in str(error)
    else:
        raise AssertionError("mismatched model/DFT path hashes must fail validation")


def test_dft_energy_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = tmp_path / "demo.tsv"
    _make_demo(fixture)
    frame = pd.read_csv(fixture, sep="\t")
    frame.loc[0, "dft_relative_energy_eV"] = 999.0
    frame.to_csv(fixture, sep="\t", index=False)
    try:
        read_and_validate(fixture, allow_synthetic_demo=True)
    except ValueError as error:
        assert "dft_relative_energy_eV" in str(error)
    else:
        raise AssertionError("a fixed DFT image must have one reference energy across all models")
