import importlib.util
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from migrationbench.neb_seed_paired_heatmap import (
    _apply_typography,
    derive_tables,
    read_and_validate,
    render_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "neb_seed_paired_heatmap_20261031"


def _load_dot_builder():
    path = ANALYSIS / "scripts" / "build_total_performance_dot_metrics.py"
    spec = importlib.util.spec_from_file_location("build_total_performance_dot_metrics", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_export_typography_bumps_and_bolds_axis_text() -> None:
    fig, axis = plt.subplots()
    axis.set_xticks([0, 1], ["route A", "route B"])
    axis.set_yticks([0, 1], ["model A", "model B"])
    axis.set_xlabel("X axis")
    axis.set_ylabel("Y axis")
    baseline = axis.get_xticklabels()[0].get_fontsize()
    _apply_typography(fig)
    assert axis.get_xticklabels()[0].get_fontsize() == baseline + 2.0
    assert axis.get_yticklabels()[0].get_fontweight() == "bold"
    assert axis.xaxis.label.get_fontweight() == "bold"
    assert axis.yaxis.label.get_fontweight() == "bold"
    plt.close(fig)


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


def test_total_dot_table_has_frozen_order_and_scientific_coverage() -> None:
    builder = _load_dot_builder()
    table, expanded = builder.build_dot_tables(
        ANALYSIS / "processed/NEB_TOTAL_PERFORMANCE_METRICS.tsv",
        ANALYSIS / "raw_input/foundation_migration_20261019/TIER1_RESULT_MATRIX.tsv",
        ANALYSIS / "raw_input/sliding_foundation_20261024/fixed_metrics_per_model.csv",
        ANALYSIS / "raw_input/dft_reference/canonical_dft_barriers.tsv",
    )
    assert table["row_id"].drop_duplicates().tolist() == [
        "Scratch-600K",
        "Scratch-MultiT",
        "FT-600K",
        "FT-MultiT",
        "MP-0B3",
        "OMAT-0",
        "MPA0",
    ]
    assert table["column_id"].drop_duplicates().tolist() == [
        "migration_in_gap_fixed",
        "migration_gap_to_ql_fixed",
        "migration_within_ql_fixed",
        "migration_in_gap_self",
        "migration_gap_to_ql_self",
        "migration_within_ql_self",
        "sliding_pristine_fixed",
        "sliding_cr_in_gap_fixed",
    ]
    assert len(table) == 56
    assert table["color_value_eV"].equals(table["predicted_value_eV"])
    assert table["color_vmin_eV"].dropna().eq(0.0).all()
    assert table["color_vmax_eV"].dropna().gt(table["color_value_eV"].dropna().max()).all()
    assert set(table["metric_type"]) == {"barrier_error"}
    assert table["error_eV"].dropna().ge(0.0).all()
    assert table["color_transform"].dropna().unique().tolist() == ["global_powernorm_gamma_0.40"]
    scratch_fixed = table.loc[
        table["row_id"].eq("Scratch-600K") & table["column_id"].eq("migration_in_gap_fixed")
    ].iloc[0]
    assert scratch_fixed["raw_value_eV"] > 0.0
    assert scratch_fixed["predicted_value_eV"] == scratch_fixed["raw_value_eV"]
    assert scratch_fixed["error_eV"] >= 0.0
    assert scratch_fixed["aggregation"] == "mean_over_training_seeds"
    assert {"color_value_seed_123_eV", "color_value_seed_234_eV", "color_value_seed_345_eV"}.issubset(table.columns)
    assert expanded["row_id"].drop_duplicates().tolist() == [
        *(f"Scratch-600K / s{seed}" for seed in (123, 234, 345)),
        *(f"Scratch-MultiT / s{seed}" for seed in (123, 234, 345)),
        *(f"FT-600K / s{seed}" for seed in (123, 234, 345)),
        *(f"FT-MultiT / s{seed}" for seed in (123, 234, 345)),
        "MP-0B3",
        "OMAT-0",
        "MPA0",
    ]
    assert expanded["color_value_eV"].equals(expanded["predicted_value_eV"])
    scratch = table.loc[table["row_id"].eq("Scratch-600K")]
    # The 2026-11-01 extension campaign recovered the formerly missing
    # Scratch-600K s123 within-QL self-relaxed value.
    assert scratch.loc[scratch["column_id"].eq("migration_within_ql_self"), "n_available"].item() == 3
    omat = table.loc[
        table["row_id"].eq("OMAT-0") & table["column_id"].eq("migration_within_ql_self")
    ].iloc[0]
    assert pd.isna(omat["raw_value_eV"])
    assert omat["n_available"] == 0
