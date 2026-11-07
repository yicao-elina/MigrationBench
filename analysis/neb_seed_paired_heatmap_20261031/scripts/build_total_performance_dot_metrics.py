"""Build seed-expanded and seed-aggregated bubble-heatmap tables.

Color is computed from the unnormalized predicted energy value in eV using
one global, nonlinear display transform.  The transform only allocates visual
resolution across the strongly right-skewed values; it does not change the
stored energy value or compare each column on a separate scale.
Bubble size is the absolute predicted-energy error in eV. Trained-family
cells in the aggregate table are arithmetic means over available seeds; the
three seed color values and errors are retained for the split-bubble renderer.
Foundation checkpoints remain single rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DOT_COLUMNS = [
    ("migration_in_gap_fixed", "in-gap", "Migration / fixed pathway", "barrier_error"),
    ("migration_gap_to_ql_fixed", "gap→QL", "Migration / fixed pathway", "barrier_error"),
    ("migration_within_ql_fixed", "within QL", "Migration / fixed pathway", "barrier_error"),
    ("migration_in_gap_self", "in-gap", "Migration / self-relaxed", "barrier_error"),
    ("migration_gap_to_ql_self", "gap→QL", "Migration / self-relaxed", "barrier_error"),
    ("migration_within_ql_self", "within QL", "Migration / self-relaxed", "barrier_error"),
    ("sliding_pristine_fixed", "pristine", "Sliding", "barrier_error"),
    ("sliding_cr_in_gap_fixed", "Cr-in-gap", "Sliding", "barrier_error"),
]
MODEL_ROWS = ["Scratch-600K", "Scratch-MultiT", "FT-600K", "FT-MultiT", "MP-0B3", "OMAT-0", "MPA0"]
TRAINED_ROWS = MODEL_ROWS[:4]
SEEDS = [123, 234, 345]
FAMILY_ALIAS = {"From-scratch": "Scratch-MultiT"}
FOUNDATION_MODELS = {
    "foundation_mp0b3": "MP-0B3",
    "foundation_omat0_medium": "OMAT-0",
    "foundation_mpa0": "MPA0",
}
FOUNDATION_SLIDING_MODELS = {
    "foundation_mp0b3_medium": "MP-0B3",
    "foundation_omat0_medium": "OMAT-0",
    "foundation_mpa0_medium": "MPA0",
}
PATH_MAP = {
    "cr_canonical_in_gap": "in_gap",
    "cr_canonical_gap_to_ql": "gap_to_ql",
    "cr_canonical_within_ql": "within_ql",
}


def _finite(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) and np.isfinite(float(number)) else np.nan


def _write_tsv(frame: pd.DataFrame, path: Path) -> None:
    text = frame.to_csv(sep="\t", index=False)
    path.write_text("\n".join(line.rstrip("\t") for line in text.splitlines()) + "\n")


def _unique_join(values: pd.Series) -> str:
    unique = list(dict.fromkeys(str(value) for value in values.dropna() if str(value) and str(value) != "nan"))
    return ";".join(unique)


def _status_text(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "NOT_AVAILABLE/DIAGNOSTIC_ONLY"
    fields = []
    for scientific, publication in frame[["scientific_status", "publication_status"]].drop_duplicates().itertuples(index=False):
        fields.append(f"{scientific}/{publication}")
    return ";".join(fields)


def _base(
    *,
    row_id: str,
    row_label: str,
    model_kind: str,
    column_id: str,
    column_label: str,
    task_block: str,
    metric_type: str,
    predicted_value: float,
    error_eV: float,
    n_available: int,
    n_expected: int,
    aggregation: str,
    status_summary: str,
    source_artifact: str,
    model_sha256: str,
    training_seed: int | str | None = None,
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "row_label": row_label,
        "model_kind": model_kind,
        "training_seed": training_seed if training_seed is not None else "",
        "column_id": column_id,
        "column_label": column_label,
        "task_block": task_block,
        "metric_type": metric_type,
        "raw_value_eV": predicted_value,
        "predicted_value_eV": predicted_value,
        "error_eV": error_eV,
        "absolute_error_eV": error_eV,
        "n_available": n_available,
        "n_expected": n_expected,
        "coverage_fraction": n_available / n_expected if n_expected else np.nan,
        "aggregation": aggregation,
        "status_summary": status_summary,
        "source_artifact": source_artifact,
        "model_sha256": model_sha256,
    }


def _load_dft_reference(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path, sep="\t")
    return dict(zip(frame["ground_truth_id"].astype(str), pd.to_numeric(frame["forward_barrier_eV"], errors="coerce")))


def _foundation_migration_rows(
    path: Path,
    dft_reference: Path,
    *,
    include_diagnostic_self: bool = False,
) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {
        "model_id",
        "path_id",
        "observable",
        "initialization_method",
        "scientific_status",
        "publication_status",
        "model_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"foundation migration source lacks required columns: {', '.join(missing)}")
    dft = _load_dft_reference(dft_reference)
    frame = frame.loc[frame["model_id"].astype(str).isin(FOUNDATION_MODELS)].copy()
    frame["route"] = frame["path_id"].astype(str).map(PATH_MAP)
    frame = frame.loc[frame["route"].notna()].copy()
    frame = frame.loc[
        (frame["observable"].eq("FIXED_PATH_SINGLE_POINT"))
        | (
            frame["observable"].eq("SELF_RELAXED_ENDPOINT_ONLY_NEB")
            & frame["initialization_method"].eq("IDPP")
        )
    ].copy()
    frame["column_id"] = [
        f"migration_{route}_{'fixed' if obs == 'FIXED_PATH_SINGLE_POINT' else 'self'}"
        for route, obs in zip(frame["route"], frame["observable"])
    ]
    if "forward_barrier_eV" not in frame:
        frame["forward_barrier_eV"] = np.nan
    if "barrier_proxy_eV" not in frame:
        frame["barrier_proxy_eV"] = np.nan
    frame["model_barrier_eV"] = np.where(
        frame["observable"].eq("FIXED_PATH_SINGLE_POINT"),
        pd.to_numeric(frame["forward_barrier_eV"], errors="coerce"),
        pd.to_numeric(frame["barrier_proxy_eV"], errors="coerce"),
    )
    frame["dft_reference_barrier_eV"] = frame["path_id"].map(dft)
    frame["raw_error_eV"] = (frame["model_barrier_eV"] - frame["dft_reference_barrier_eV"]).abs()
    if not include_diagnostic_self:
        frame.loc[
            ~(
                frame["scientific_status"].eq("CONVERGED")
                & frame["publication_status"].eq("PRIMARY_EVIDENCE")
            )
            & frame["observable"].eq("SELF_RELAXED_ENDPOINT_ONLY_NEB"),
            "raw_error_eV",
        ] = np.nan
    return frame


def _foundation_rows(
    migration_path: Path,
    sliding_path: Path,
    dft_reference: Path,
    *,
    include_diagnostic_self: bool = False,
) -> list[dict[str, object]]:
    migration = _foundation_migration_rows(
        migration_path,
        dft_reference,
        include_diagnostic_self=include_diagnostic_self,
    )
    sliding = pd.read_csv(sliding_path)
    sliding = sliding.loc[sliding["model_id"].astype(str).isin(FOUNDATION_SLIDING_MODELS)].copy()
    rows: list[dict[str, object]] = []
    for label in MODEL_ROWS[4:]:
        migration_ids = [model_id for model_id, model_label in FOUNDATION_MODELS.items() if model_label == label]
        sliding_ids = [model_id for model_id, model_label in FOUNDATION_SLIDING_MODELS.items() if model_label == label]
        mig = migration.loc[migration["model_id"].astype(str).isin(migration_ids)].copy()
        slide = sliding.loc[sliding["model_id"].astype(str).isin(sliding_ids)].copy()
        for column_id, column_label, task_block, metric_type in DOT_COLUMNS:
            if column_id.startswith("migration_"):
                selected = mig.loc[mig["column_id"].eq(column_id)].copy()
                predicted = pd.to_numeric(selected["model_barrier_eV"], errors="coerce")
                errors = pd.to_numeric(selected["raw_error_eV"], errors="coerce")
                error = float(errors.iloc[0]) if len(errors) == 1 and np.isfinite(errors.iloc[0]) else np.nan
                predicted_value = (
                    float(predicted.iloc[0])
                    if len(predicted) == 1 and np.isfinite(predicted.iloc[0]) and np.isfinite(error)
                    else np.nan
                )
                n_available = int((np.isfinite(predicted) & np.isfinite(errors)).sum())
                source = f"{migration_path};{dft_reference}"
                model_sha = _unique_join(selected["model_sha256"])
                status = _status_text(selected)
                reference = float(selected["dft_reference_barrier_eV"].iloc[0]) if len(selected) == 1 else np.nan
            else:
                system_id = "pristine" if column_id == "sliding_pristine_fixed" else "cr_in_gap"
                selected = slide.loc[slide["system_id"].astype(str).eq(system_id)].copy()
                predicted = pd.to_numeric(selected["max_relative_energy_eV"], errors="coerce")
                reference_values = pd.to_numeric(selected["dft_peak_eV"], errors="coerce")
                errors = (predicted - reference_values).abs()
                reference = float(reference_values.iloc[0]) if len(reference_values) == 1 and np.isfinite(reference_values.iloc[0]) else np.nan
                error = float(errors.iloc[0]) if len(errors) == 1 and np.isfinite(errors.iloc[0]) else np.nan
                predicted_value = (
                    float(predicted.iloc[0])
                    if len(predicted) == 1 and np.isfinite(predicted.iloc[0]) and np.isfinite(reference) and np.isfinite(error)
                    else np.nan
                )
                n_available = int((np.isfinite(predicted) & np.isfinite(reference_values) & np.isfinite(errors)).sum())
                source = str(sliding_path)
                model_sha = _unique_join(selected["model_sha256"])
                status = _status_text(selected)
            row = _base(
                    row_id=label,
                    row_label=label,
                    model_kind="foundation_checkpoint",
                    column_id=column_id,
                    column_label=column_label,
                    task_block=task_block,
                    metric_type=metric_type,
                    predicted_value=predicted_value,
                    error_eV=error,
                    n_available=n_available,
                    n_expected=1,
                    aggregation="single_checkpoint",
                    status_summary=status,
                    source_artifact=source,
                    model_sha256=model_sha,
                )
            row["scientific_status"] = str(selected["scientific_status"].iloc[0]) if not selected.empty else "NOT_AVAILABLE"
            row["publication_status"] = str(selected["publication_status"].iloc[0]) if not selected.empty else "DIAGNOSTIC_ONLY"
            row["reference_value_eV"] = reference
            row["error_definition"] = (
                "abs(model barrier - canonical DFT forward barrier)"
                if column_id.startswith("migration_")
                else "abs(max model relative energy - max DFT relative energy)"
            )
            rows.append(row)
    return rows


def _trained_seed_rows(total: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    total = total.copy()
    total["display_family"] = total["model_family"].astype(str).replace(FAMILY_ALIAS)
    total["training_seed"] = pd.to_numeric(total["training_seed"], errors="coerce").astype("Int64")
    for family in TRAINED_ROWS:
        for seed in SEEDS:
            selected = total.loc[
                total["display_family"].eq(family) & total["training_seed"].eq(seed)
            ].copy()
            for column_id, column_label, task_block, metric_type in DOT_COLUMNS:
                cell = selected.loc[selected["column_id"].eq(column_id)].copy()
                cell_values = pd.to_numeric(cell["predicted_value_eV"], errors="coerce")
                cell_errors = pd.to_numeric(cell["absolute_error_eV"], errors="coerce")
                finite_cell = cell.loc[np.isfinite(cell_values) & np.isfinite(cell_errors)].copy()
                predicted_value = _finite(finite_cell["predicted_value_eV"].iloc[0]) if not finite_cell.empty else np.nan
                error = _finite(finite_cell["absolute_error_eV"].iloc[0]) if not finite_cell.empty else np.nan
                finite = int(np.isfinite(predicted_value) and np.isfinite(error))
                source_cell = finite_cell if not finite_cell.empty else cell
                row = _base(
                        row_id=f"{family} / s{seed}",
                        row_label=f"{family} / s{seed}",
                        model_kind="trained_seed",
                        training_seed=seed,
                        column_id=column_id,
                        column_label=column_label,
                        task_block=task_block,
                        metric_type=metric_type,
                        predicted_value=predicted_value,
                        error_eV=error,
                        n_available=finite,
                        n_expected=1,
                        aggregation="single_training_seed",
                        status_summary=_status_text(source_cell) if finite else "NOT_AVAILABLE/DIAGNOSTIC_ONLY",
                        source_artifact=_unique_join(source_cell["source_artifact"]) if not source_cell.empty else "",
                        model_sha256=_unique_join(source_cell["model_sha256"]) if not source_cell.empty else "",
                    )
                if not source_cell.empty:
                    row["scientific_status"] = str(source_cell["scientific_status"].iloc[0])
                    row["publication_status"] = str(source_cell["publication_status"].iloc[0])
                else:
                    row["scientific_status"] = "NOT_AVAILABLE"
                    row["publication_status"] = "DIAGNOSTIC_ONLY"
                rows.append(row)
    return rows


def _add_energy_color_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Record the raw-energy color channel and one global display transform."""
    frame = frame.copy()
    values = pd.to_numeric(frame["predicted_value_eV"], errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        raise ValueError("no finite predicted energies are available for the color scale")
    if (finite < 0).any():
        raise ValueError("predicted barrier/peak energies must be non-negative")
    frame["color_value_eV"] = values
    frame["color_vmin_eV"] = 0.0
    frame["color_vmax_eV"] = float(np.ceil(float(finite.max())))
    frame["color_transform"] = "global_powernorm_gamma_0.40"
    frame["color_scale"] = "raw_predicted_energy_eV_global"
    return frame


def build_dot_tables(
    total_metrics: Path,
    foundation_migration: Path,
    foundation_sliding: Path,
    dft_reference: Path,
    include_diagnostic_self: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = pd.read_csv(total_metrics, sep="\t")
    required = {
        "model_family", "column_id", "predicted_value_eV", "absolute_error_eV",
        "source_artifact", "model_sha256",
    }
    missing = sorted(required - set(total.columns))
    if missing:
        raise ValueError(f"total performance table lacks required columns: {', '.join(missing)}")
    expanded = _trained_seed_rows(total)
    expanded.extend(
        _foundation_rows(
            foundation_migration,
            foundation_sliding,
            dft_reference,
            include_diagnostic_self=include_diagnostic_self,
        )
    )
    expanded = _add_energy_color_values(pd.DataFrame(expanded))
    expanded_order = [
        f"{family} / s{seed}" for family in TRAINED_ROWS for seed in SEEDS
    ] + MODEL_ROWS[4:]
    expanded["row_id"] = pd.Categorical(expanded["row_id"], categories=expanded_order, ordered=True)
    expanded["column_id"] = pd.Categorical(
        expanded["column_id"], categories=[column_id for column_id, *_ in DOT_COLUMNS], ordered=True
    )
    expanded = expanded.sort_values(["row_id", "column_id"]).reset_index(drop=True)
    expanded["row_id"] = expanded["row_id"].astype(str)
    expanded["column_id"] = expanded["column_id"].astype(str)

    aggregate_rows: list[dict[str, object]] = []
    for family in TRAINED_ROWS:
        family_rows = expanded.loc[expanded["row_id"].str.startswith(f"{family} / s")].copy()
        for column_id, column_label, task_block, metric_type in DOT_COLUMNS:
            selected = family_rows.loc[family_rows["column_id"].eq(column_id)].copy()
            values = pd.to_numeric(selected["predicted_value_eV"], errors="coerce")
            errors = pd.to_numeric(selected["absolute_error_eV"], errors="coerce")
            finite = values[np.isfinite(values) & np.isfinite(errors)]
            finite_errors = errors[np.isfinite(values) & np.isfinite(errors)]
            seed_color = {
                f"color_value_seed_{seed}_eV": float(selected.loc[selected["training_seed"].astype(str).eq(str(seed)), "predicted_value_eV"].iloc[0])
                if not selected.loc[selected["training_seed"].astype(str).eq(str(seed)), "predicted_value_eV"].empty
                else np.nan
                for seed in SEEDS
            }
            seed_error = {
                f"error_seed_{seed}": float(selected.loc[selected["training_seed"].astype(str).eq(str(seed)), "absolute_error_eV"].iloc[0])
                if not selected.loc[selected["training_seed"].astype(str).eq(str(seed)), "absolute_error_eV"].empty
                else np.nan
                for seed in SEEDS
            }
            row = _base(
                row_id=family,
                row_label=family,
                model_kind="trained_family",
                column_id=column_id,
                column_label=column_label,
                task_block=task_block,
                metric_type=metric_type,
                predicted_value=float(finite.mean()) if len(finite) else np.nan,
                error_eV=float(finite_errors.mean()) if len(finite_errors) else np.nan,
                n_available=int(len(finite)),
                n_expected=3,
                aggregation="mean_over_training_seeds",
            status_summary=_status_text(selected.loc[np.isfinite(values) & np.isfinite(errors)]) if len(finite) else "NOT_AVAILABLE/DIAGNOSTIC_ONLY",
                source_artifact=_unique_join(selected["source_artifact"]),
                model_sha256=_unique_join(selected["model_sha256"]),
            )
            row.update(seed_color)
            row.update(seed_error)
            row["color_value_eV"] = row["predicted_value_eV"]
            row["color_vmin_eV"] = 0.0
            row["color_vmax_eV"] = float(expanded["color_vmax_eV"].iloc[0])
            row["color_transform"] = "global_powernorm_gamma_0.40"
            row["color_scale"] = "raw_predicted_energy_eV_global"
            row["scientific_status"] = "AGGREGATED_FROM_SEED_ROWS"
            row["publication_status"] = "DERIVED_SUMMARY"
            aggregate_rows.append(row)
    for label in MODEL_ROWS[4:]:
        selected = expanded.loc[expanded["row_id"].eq(label)].copy()
        for column_id, column_label, task_block, metric_type in DOT_COLUMNS:
            cell = selected.loc[selected["column_id"].eq(column_id)].iloc[0]
            row = cell.to_dict()
            row.update({"row_id": label, "row_label": label, "aggregation": "single_checkpoint"})
            row["color_value_seed_123_eV"] = np.nan
            row["color_value_seed_234_eV"] = np.nan
            row["color_value_seed_345_eV"] = np.nan
            row["error_seed_123"] = np.nan
            row["error_seed_234"] = np.nan
            row["error_seed_345"] = np.nan
            aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate["row_id"] = pd.Categorical(aggregate["row_id"], categories=MODEL_ROWS, ordered=True)
    aggregate["column_id"] = pd.Categorical(
        aggregate["column_id"], categories=[column_id for column_id, *_ in DOT_COLUMNS], ordered=True
    )
    aggregate = aggregate.sort_values(["row_id", "column_id"]).reset_index(drop=True)
    aggregate["row_id"] = aggregate["row_id"].astype(str)
    aggregate["column_id"] = aggregate["column_id"].astype(str)
    return aggregate, expanded


def build_dot_table(
    total_metrics: Path,
    foundation_migration: Path,
    foundation_sliding: Path,
    dft_reference: Path | None = None,
    include_diagnostic_self: bool = False,
) -> pd.DataFrame:
    """Backward-compatible aggregate-table entry point."""
    if dft_reference is None:
        dft_reference = foundation_migration.parents[1] / "dft_reference" / "canonical_dft_barriers.tsv"
    return build_dot_tables(
        total_metrics,
        foundation_migration,
        foundation_sliding,
        dft_reference,
        include_diagnostic_self=include_diagnostic_self,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-metrics", type=Path, required=True)
    parser.add_argument("--foundation-migration", type=Path, required=True)
    parser.add_argument("--foundation-sliding", type=Path, required=True)
    parser.add_argument("--dft-reference", type=Path, required=True)
    parser.add_argument(
        "--include-diagnostic-self",
        action="store_true",
        help=(
            "retain finite terminal values for non-converged Foundation self-relaxed retries; "
            "scientific/publication status remains diagnostic"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-expanded-output", type=Path, required=True)
    args = parser.parse_args()
    aggregate, expanded = build_dot_tables(
        args.total_metrics,
        args.foundation_migration,
        args.foundation_sliding,
        args.dft_reference,
        args.include_diagnostic_self,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_tsv(aggregate, args.output)
    _write_tsv(expanded, args.seed_expanded_output)
    print(f"wrote aggregate={len(aggregate)} rows and seed_expanded={len(expanded)} rows")


if __name__ == "__main__":
    main()
