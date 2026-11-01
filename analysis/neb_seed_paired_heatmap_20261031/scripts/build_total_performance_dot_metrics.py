"""Build seed-expanded and seed-aggregated bubble-heatmap tables.

The score is computed from the per-seed/checkpoint error table, column by
column, with lower error mapped to a higher score in [0, 1].  Trained-family
cells in the aggregate table are arithmetic means over available seeds; the
three seed scores are retained in separate columns for the split-bubble
renderer.  Foundation checkpoints remain single rows.
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
    ("sliding_pristine_fixed", "pristine", "Sliding", "MAE"),
    ("sliding_cr_in_gap_fixed", "Cr-in-gap", "Sliding", "MAE"),
]
MODEL_ROWS = ["Scratch-MultiT", "Scratch-600K", "FT-600K", "FT-MultiT", "MP-0B3", "OMAT-0", "MPA0"]
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
    value: float,
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
        "raw_value_eV": value,
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


def _foundation_migration_rows(path: Path, dft_reference: Path) -> pd.DataFrame:
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
) -> list[dict[str, object]]:
    migration = _foundation_migration_rows(migration_path, dft_reference)
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
                finite = pd.to_numeric(selected["raw_error_eV"], errors="coerce")
                value = float(finite.iloc[0]) if len(finite) == 1 and np.isfinite(finite.iloc[0]) else np.nan
                n_available = int(np.isfinite(finite).sum())
                source = f"{migration_path};{dft_reference}"
                model_sha = _unique_join(selected["model_sha256"])
                status = _status_text(selected)
            else:
                system_id = "pristine" if column_id == "sliding_pristine_fixed" else "cr_in_gap"
                selected = slide.loc[slide["system_id"].astype(str).eq(system_id)].copy()
                finite = pd.to_numeric(selected["mae_dft_eV"], errors="coerce")
                value = float(finite.iloc[0]) if len(finite) == 1 and np.isfinite(finite.iloc[0]) else np.nan
                n_available = int(np.isfinite(finite).sum())
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
                    value=value,
                    n_available=n_available,
                    n_expected=1,
                    aggregation="single_checkpoint",
                    status_summary=status,
                    source_artifact=source,
                    model_sha256=model_sha,
                )
            row["scientific_status"] = str(selected["scientific_status"].iloc[0]) if not selected.empty else "NOT_AVAILABLE"
            row["publication_status"] = str(selected["publication_status"].iloc[0]) if not selected.empty else "DIAGNOSTIC_ONLY"
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
                cell_values = pd.to_numeric(cell["value_eV"], errors="coerce")
                finite_cell = cell.loc[np.isfinite(cell_values)].copy()
                value = _finite(finite_cell["value_eV"].iloc[0]) if not finite_cell.empty else np.nan
                finite = int(np.isfinite(value))
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
                        value=value,
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


def _add_seed_scores(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    scores = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby("column_id", sort=False):
        values = pd.to_numeric(group["raw_value_eV"], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        minimum, maximum = float(finite.min()), float(finite.max())
        if maximum > minimum:
            scores.loc[group.index] = 1.0 - (values - minimum) / (maximum - minimum)
        else:
            scores.loc[group.index] = 1.0
    frame["score_01"] = scores
    frame["score_direction"] = "lower_is_better"
    frame["color_scale"] = "column_minmax_score_01_across_15_units"
    return frame


def build_dot_tables(
    total_metrics: Path,
    foundation_migration: Path,
    foundation_sliding: Path,
    dft_reference: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = pd.read_csv(total_metrics, sep="\t")
    required = {"model_family", "column_id", "value_eV", "source_artifact", "model_sha256"}
    missing = sorted(required - set(total.columns))
    if missing:
        raise ValueError(f"total performance table lacks required columns: {', '.join(missing)}")
    expanded = _trained_seed_rows(total)
    expanded.extend(_foundation_rows(foundation_migration, foundation_sliding, dft_reference))
    expanded = _add_seed_scores(pd.DataFrame(expanded))
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
            values = pd.to_numeric(selected["raw_value_eV"], errors="coerce")
            scores = pd.to_numeric(selected["score_01"], errors="coerce")
            finite = values[np.isfinite(values)]
            finite_scores = scores[np.isfinite(scores)]
            seed_score = {
                f"score_seed_{seed}": float(selected.loc[selected["training_seed"].astype(str).eq(str(seed)), "score_01"].iloc[0])
                if not selected.loc[selected["training_seed"].astype(str).eq(str(seed)), "score_01"].empty
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
                value=float(finite.mean()) if len(finite) else np.nan,
                n_available=int(len(finite)),
                n_expected=3,
                aggregation="mean_over_training_seeds",
                status_summary=_status_text(selected.loc[np.isfinite(values)]) if len(finite) else "NOT_AVAILABLE/DIAGNOSTIC_ONLY",
                source_artifact=_unique_join(selected["source_artifact"]),
                model_sha256=_unique_join(selected["model_sha256"]),
            )
            row.update(seed_score)
            row["score_01"] = float(finite_scores.mean()) if len(finite_scores) else np.nan
            row["score_direction"] = "lower_is_better"
            row["color_scale"] = "column_minmax_score_01_across_15_units"
            row["scientific_status"] = "AGGREGATED_FROM_SEED_ROWS"
            row["publication_status"] = "DERIVED_SUMMARY"
            aggregate_rows.append(row)
    for label in MODEL_ROWS[4:]:
        selected = expanded.loc[expanded["row_id"].eq(label)].copy()
        for column_id, column_label, task_block, metric_type in DOT_COLUMNS:
            cell = selected.loc[selected["column_id"].eq(column_id)].iloc[0]
            row = cell.to_dict()
            row.update({"row_id": label, "row_label": label, "aggregation": "single_checkpoint"})
            row["score_seed_123"] = np.nan
            row["score_seed_234"] = np.nan
            row["score_seed_345"] = np.nan
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
) -> pd.DataFrame:
    """Backward-compatible aggregate-table entry point."""
    if dft_reference is None:
        dft_reference = foundation_migration.parents[1] / "dft_reference" / "canonical_dft_barriers.tsv"
    return build_dot_tables(total_metrics, foundation_migration, foundation_sliding, dft_reference)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-metrics", type=Path, required=True)
    parser.add_argument("--foundation-migration", type=Path, required=True)
    parser.add_argument("--foundation-sliding", type=Path, required=True)
    parser.add_argument("--dft-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-expanded-output", type=Path, required=True)
    args = parser.parse_args()
    aggregate, expanded = build_dot_tables(
        args.total_metrics, args.foundation_migration, args.foundation_sliding, args.dft_reference
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.output, sep="\t", index=False)
    expanded.to_csv(args.seed_expanded_output, sep="\t", index=False)
    print(f"wrote aggregate={len(aggregate)} rows and seed_expanded={len(expanded)} rows")


if __name__ == "__main__":
    main()
