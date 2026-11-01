"""Combine sliding MAE and migration barrier-proxy scores without pooling campaigns."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SLIDING_FAMILIES = ["Scratch-MultiT", "Scratch-600K", "FT-600K", "FT-MultiT"]
MIGRATION_FAMILIES = ["From-scratch", "Scratch-600K", "FT-600K", "FT-MultiT"]
SEEDS = [123, 234, 345]
COLUMNS = [
    ("sliding_pristine_fixed", "Sliding / pristine\nfixed (MAE)", "sliding", "MAE"),
    ("sliding_cr_in_gap_fixed", "Sliding / Cr-in-gap\nfixed (MAE)", "sliding", "MAE"),
    ("migration_in_gap_fixed", "Migration / within gap\nfixed (barrier)", "migration", "barrier_proxy"),
    ("migration_gap_to_ql_fixed", "Migration / gap→QL\nfixed (barrier)", "migration", "barrier_proxy"),
    ("migration_within_ql_fixed", "Migration / within QL\nfixed (barrier)", "migration", "barrier_proxy"),
    ("migration_in_gap_self", "Migration / within gap\nself-relaxed (barrier)", "migration", "barrier_proxy"),
    ("migration_gap_to_ql_self", "Migration / gap→QL\nself-relaxed (barrier)", "migration", "barrier_proxy"),
    ("migration_within_ql_self", "Migration / within QL\nself-relaxed (barrier)", "migration", "barrier_proxy"),
]
PATH_MAP = {
    "in-gap diffusion": "in_gap",
    "gap-to-QL": "gap_to_ql",
    "within-QL": "within_ql",
}


def _finite_or_nan(value: object) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(number) if pd.notna(number) and np.isfinite(float(number)) else np.nan


def _base_row(
    *,
    unit_id: str,
    row_label: str,
    campaign: str,
    model_family: str,
    model_id: str | None,
    training_seed: int,
    source_artifact: str,
    model_sha256: str | None,
    dataset_status: str,
) -> dict[str, object]:
    return {
        "unit_id": unit_id,
        "row_label": row_label,
        "campaign": campaign,
        "model_family": model_family,
        "model_id": model_id or "",
        "training_seed": training_seed,
        "source_artifact": source_artifact,
        "model_sha256": model_sha256 or "",
        "dataset_status": dataset_status,
    }


def _append_unit_rows(
    rows: list[dict[str, object]],
    *,
    base: dict[str, object],
    values: dict[str, float],
    status_by_column: dict[str, tuple[str, str]],
) -> None:
    for column_id, column_label, _, metric_type in COLUMNS:
        scientific_status, publication_status = status_by_column.get(
            column_id, ("NOT_AVAILABLE", "DIAGNOSTIC_ONLY")
        )
        rows.append(
            {
                **base,
                "column_id": column_id,
                "column_label": column_label,
                "metric_type": metric_type,
                "value_eV": values.get(column_id, np.nan),
                "scientific_status": scientific_status,
                "publication_status": publication_status,
            }
        )


def build_total_table(sliding_metrics: Path, migration_matrix: Path) -> pd.DataFrame:
    sliding = pd.read_csv(sliding_metrics, sep="\t")
    migration = pd.read_csv(migration_matrix, sep="\t")
    migration = migration.loc[migration["model_family"].astype(str) != "Foundation"].copy()
    migration["training_seed"] = pd.to_numeric(migration["training_seed"], errors="coerce").astype("Int64")
    migration = migration.loc[migration["training_seed"].isin(SEEDS)].copy()

    rows: list[dict[str, object]] = []
    sliding_order = {(family, seed): f"SLIDING::{family}::s{seed}" for family in SLIDING_FAMILIES for seed in SEEDS}
    for family in SLIDING_FAMILIES:
        for seed in SEEDS:
            unit_id = sliding_order[(family, seed)]
            sub = sliding.loc[
                (sliding["model_family"].astype(str) == family)
                & (pd.to_numeric(sliding["training_seed"], errors="coerce") == seed)
            ]
            values = {
                "sliding_pristine_fixed": _finite_or_nan(sub.loc[sub["path_id"] == "pristine", "mae_eV"].iloc[0])
                if not sub.loc[sub["path_id"] == "pristine"].empty
                else np.nan,
                "sliding_cr_in_gap_fixed": _finite_or_nan(sub.loc[sub["path_id"] == "cr_in_gap", "mae_eV"].iloc[0])
                if not sub.loc[sub["path_id"] == "cr_in_gap"].empty
                else np.nan,
            }
            model_id = str(sub["model_id"].iloc[0]) if not sub.empty else None
            base = _base_row(
                unit_id=unit_id,
                row_label=f"Sliding · {family} · s{seed}",
                campaign="sliding",
                model_family=family,
                model_id=model_id,
                training_seed=seed,
                source_artifact=str(sliding_metrics),
                model_sha256=None,
                dataset_status="PROVENANCE_PARTIAL",
            )
            status = {column_id: ("FINITE_DIAGNOSTIC", "SUPPORTING_EVIDENCE") for column_id, _, _, _ in COLUMNS[:2]}
            _append_unit_rows(rows, base=base, values=values, status_by_column=status)

    migration_lookup: dict[tuple[str, int, str, str], pd.Series] = {}
    for record in migration.itertuples(index=False):
        route = PATH_MAP.get(str(record.path_class))
        if route is None:
            raise ValueError(f"unrecognized migration path class: {record.path_class}")
        if record.observable == "FIXED_PATH_SINGLE_POINT":
            observable_key = "fixed"
        elif record.observable == "SELF_RELAXED_ENDPOINT_ONLY_NEB" and record.initialization_method == "IDPP":
            observable_key = "self"
        else:
            continue
        key = (str(record.model_family), int(record.training_seed), route, observable_key)
        if key in migration_lookup:
            raise ValueError(f"duplicate selected migration row: {key}")
        migration_lookup[key] = pd.Series(record._asdict())

    for family in MIGRATION_FAMILIES:
        for seed in SEEDS:
            unit_id = f"MIGRATION::{family}::s{seed}"
            available = migration.loc[
                (migration["model_family"].astype(str) == family)
                & (migration["training_seed"] == seed)
            ]
            model_id = str(available["model_id"].iloc[0]) if not available.empty else None
            model_sha = str(available["model_sha256"].iloc[0]) if not available.empty else None
            dataset_status = "PROVENANCE_PARTIAL" if not available.empty else "NOT_AVAILABLE_IN_SOURCE_CAMPAIGN"
            base = _base_row(
                unit_id=unit_id,
                row_label=f"Migration · {family} · s{seed}",
                campaign="migration",
                model_family=family,
                model_id=model_id,
                training_seed=seed,
                source_artifact=str(migration_matrix),
                model_sha256=model_sha,
                dataset_status=dataset_status,
            )
            values: dict[str, float] = {}
            statuses: dict[str, tuple[str, str]] = {}
            for column_id, _, _, _ in COLUMNS[2:]:
                is_self = column_id.endswith("_self")
                route = column_id.removeprefix("migration_").removesuffix("_fixed").removesuffix("_self")
                key = (family, seed, route, "self" if is_self else "fixed")
                record = migration_lookup.get(key)
                values[column_id] = _finite_or_nan(record.get("barrier_proxy_eV")) if record is not None else np.nan
                if record is None:
                    statuses[column_id] = ("NOT_AVAILABLE", "DIAGNOSTIC_ONLY")
                else:
                    statuses[column_id] = (str(record.get("scientific_status")), str(record.get("publication_status")))
            _append_unit_rows(rows, base=base, values=values, status_by_column=statuses)

    result = pd.DataFrame(rows)
    result["training_seed"] = result["training_seed"].astype(int)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sliding-metrics", type=Path, required=True)
    parser.add_argument("--migration-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_total_table(args.sliding_metrics, args.migration_matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    print(f"wrote {len(result)} rows for {result['unit_id'].nunique()} units to {args.output}")


if __name__ == "__main__":
    main()
