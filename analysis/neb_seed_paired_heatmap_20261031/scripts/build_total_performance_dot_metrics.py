"""Aggregate the total NEB/sliding matrix into a score-and-coverage bubble table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DOT_COLUMNS = [
    ("migration_in_gap_fixed", "in-gap", "Migration / fixed pathway", "barrier_proxy"),
    ("migration_gap_to_ql_fixed", "gap→QL", "Migration / fixed pathway", "barrier_proxy"),
    ("migration_within_ql_fixed", "within QL", "Migration / fixed pathway", "barrier_proxy"),
    ("migration_in_gap_self", "in-gap", "Migration / self-relaxed", "barrier_proxy"),
    ("migration_gap_to_ql_self", "gap→QL", "Migration / self-relaxed", "barrier_proxy"),
    ("migration_within_ql_self", "within QL", "Migration / self-relaxed", "barrier_proxy"),
    ("sliding_pristine_fixed", "pristine", "Sliding", "MAE"),
    ("sliding_cr_in_gap_fixed", "Cr-in-gap", "Sliding", "MAE"),
]
MODEL_ROWS = ["Scratch-MultiT", "Scratch-600K", "FT-600K", "FT-MultiT", "MP-0B3", "OMAT-0", "MPA0"]
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
    unique = list(dict.fromkeys(str(value) for value in values.dropna() if str(value)))
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
) -> dict[str, object]:
    return {
        "row_id": row_id,
        "row_label": row_label,
        "model_kind": model_kind,
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


def _aggregate_main(total: pd.DataFrame) -> list[dict[str, object]]:
    family_alias = {"From-scratch": "Scratch-MultiT"}
    rows: list[dict[str, object]] = []
    for family in MODEL_ROWS[:4]:
        source_families = [key for key, value in family_alias.items() if value == family] + [family]
        family_rows = total.loc[total["model_family"].astype(str).isin(source_families)].copy()
        for column_id, column_label, task_block, metric_type in DOT_COLUMNS:
            selected = family_rows.loc[family_rows["column_id"].astype(str) == column_id].copy()
            values = pd.to_numeric(selected["value_eV"], errors="coerce")
            finite = values[np.isfinite(values)]
            source = _unique_join(selected["source_artifact"])
            model_sha = _unique_join(selected["model_sha256"])
            rows.append(
                _base(
                    row_id=family,
                    row_label=family,
                    model_kind="trained_family",
                    column_id=column_id,
                    column_label=column_label,
                    task_block=task_block,
                    metric_type=metric_type,
                    value=float(np.median(finite)) if len(finite) else np.nan,
                    n_available=int(len(finite)),
                    n_expected=3,
                    aggregation="median_over_training_seeds",
                    status_summary=_status_text(selected.loc[np.isfinite(values)]),
                    source_artifact=source,
                    model_sha256=model_sha,
                )
            )
    return rows


def _foundation_migration_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {
        "model_id",
        "path_id",
        "observable",
        "initialization_method",
        "barrier_proxy_eV",
        "scientific_status",
        "publication_status",
        "model_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"foundation migration source lacks required columns: {', '.join(missing)}")
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
    frame["column_prefix"] = np.where(
        frame["observable"].eq("FIXED_PATH_SINGLE_POINT"), "migration_{}_fixed", "migration_{}_self"
    )
    frame["column_id"] = [prefix.format(route) for prefix, route in zip(frame["column_prefix"], frame["route"])]
    return frame


def _aggregate_foundation(migration_path: Path, sliding_path: Path) -> list[dict[str, object]]:
    migration = _foundation_migration_rows(migration_path)
    sliding = pd.read_csv(sliding_path)
    sliding = sliding.loc[sliding["model_id"].astype(str).isin(FOUNDATION_SLIDING_MODELS)].copy()
    rows: list[dict[str, object]] = []
    for label in MODEL_ROWS[4:]:
        migration_model_ids = [model_id for model_id, model_label in FOUNDATION_MODELS.items() if model_label == label]
        sliding_model_ids = [model_id for model_id, model_label in FOUNDATION_SLIDING_MODELS.items() if model_label == label]
        mig = migration.loc[migration["model_id"].astype(str).isin(migration_model_ids)].copy()
        slide = sliding.loc[sliding["model_id"].astype(str).isin(sliding_model_ids)].copy()
        for column_id, column_label, task_block, metric_type in DOT_COLUMNS:
            if column_id.startswith("migration_"):
                selected = mig.loc[mig["column_id"].eq(column_id)].copy()
                values = pd.to_numeric(selected["barrier_proxy_eV"], errors="coerce")
                if column_id.endswith("_self"):
                    accepted = selected.loc[
                        selected["scientific_status"].eq("CONVERGED")
                        & selected["publication_status"].eq("PRIMARY_EVIDENCE")
                    ]
                else:
                    accepted = selected.loc[np.isfinite(values)]
                accepted_values = pd.to_numeric(accepted["barrier_proxy_eV"], errors="coerce")
                source = str(migration_path)
                model_sha = _unique_join(selected["model_sha256"])
                n_available = int(np.isfinite(accepted_values).sum())
                value = float(accepted_values.iloc[0]) if n_available == 1 else np.nan
                status = _status_text(selected)
            else:
                system_id = "pristine" if column_id == "sliding_pristine_fixed" else "cr_in_gap"
                selected = slide.loc[slide["system_id"].astype(str).eq(system_id)].copy()
                values = pd.to_numeric(selected["mae_dft_eV"], errors="coerce")
                finite = values[np.isfinite(values)]
                source = str(sliding_path)
                model_sha = _unique_join(selected["model_sha256"])
                n_available = int(len(finite))
                value = float(finite.iloc[0]) if n_available == 1 else np.nan
                status = _status_text(selected)
            rows.append(
                _base(
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
            )
    return rows


def _add_scores(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    scores = []
    for column_id, group in frame.groupby("column_id", sort=False):
        values = pd.to_numeric(group["raw_value_eV"], errors="coerce")
        finite = values[np.isfinite(values)]
        minimum = float(finite.min()) if len(finite) else np.nan
        maximum = float(finite.max()) if len(finite) else np.nan
        if len(finite) and maximum > minimum:
            score = 1.0 - (values - minimum) / (maximum - minimum)
        elif len(finite):
            score = values * 0.0 + 1.0
        else:
            score = values * np.nan
        scores.append(score.rename("score_01").to_frame().assign(_index=group.index))
    score_frame = pd.concat(scores).set_index("_index")["score_01"] if scores else pd.Series(dtype=float)
    frame["score_01"] = score_frame.reindex(frame.index).to_numpy()
    frame["score_direction"] = "lower_is_better"
    frame["color_scale"] = "column_minmax_score_01"
    return frame


def build_dot_table(total_metrics: Path, foundation_migration: Path, foundation_sliding: Path) -> pd.DataFrame:
    total = pd.read_csv(total_metrics, sep="\t")
    required = {"model_family", "column_id", "value_eV", "source_artifact", "model_sha256"}
    missing = sorted(required - set(total.columns))
    if missing:
        raise ValueError(f"total performance table lacks required columns: {', '.join(missing)}")
    rows = _aggregate_main(total)
    rows.extend(_aggregate_foundation(foundation_migration, foundation_sliding))
    result = _add_scores(pd.DataFrame(rows))
    result["row_id"] = pd.Categorical(result["row_id"], categories=MODEL_ROWS, ordered=True)
    result["column_id"] = pd.Categorical(
        result["column_id"], categories=[column_id for column_id, *_ in DOT_COLUMNS], ordered=True
    )
    result = result.sort_values(["row_id", "column_id"]).reset_index(drop=True)
    result["row_id"] = result["row_id"].astype(str)
    result["column_id"] = result["column_id"].astype(str)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-metrics", type=Path, required=True)
    parser.add_argument("--foundation-migration", type=Path, required=True)
    parser.add_argument("--foundation-sliding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_dot_table(args.total_metrics, args.foundation_migration, args.foundation_sliding)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    print(f"wrote {len(result)} rows for {result['row_id'].nunique()} model rows to {args.output}")


if __name__ == "__main__":
    main()
