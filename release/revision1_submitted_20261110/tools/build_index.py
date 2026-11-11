#!/usr/bin/env python3
"""Build the Revision 1 results index, per-item provenance pages and SQL tables.

Inputs (all inside this package):
  sources/main_figures.tsv, sources/si_and_tables.tsv   per-item/panel survey
  sources/status_overrides.tsv                          statuses after fixes
  sources/rockfish_inventory.tsv, sources/rockfish_models.tsv, sources/foundation_verify.tsv
  imports/import_plan.tsv, imports/EXTERNAL_STORE_MANIFEST.tsv
Outputs:
  RESULTS_INDEX.tsv            one row per figure panel / table (the maintenance list)
  STORAGE_LOCATIONS.tsv        every raw-data location, with migration target
  MODEL_REGISTRY.tsv           every trained / foundation checkpoint, used or not
  items/<section>/<item>/PROVENANCE.md
  analysis/migrationbench_asset_registry_20261025/migrations/002_revision1_submission_trace.sql
  (applied to migrationbench_asset_registry.sqlite unless --no-sql)
"""
from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
REGISTRY_DIR = REPO / "analysis/migrationbench_asset_registry_20261025"
MIGRATION = REGISTRY_DIR / "migrations/002_revision1_submission_trace.sql"
DB = REGISTRY_DIR / "migrationbench_asset_registry.sqlite"
PACKAGE_REL = PACKAGE.relative_to(REPO).as_posix()

SLUGS = {
    "Fig1": "Fig1_overview", "Fig2": "Fig2_parity", "Fig3": "Fig3_md", "Fig4": "Fig4_neb_profiles",
    "Fig5": "Fig5_task_matrix", "Fig6": "Fig6_representation", "Fig7": "Fig7_shap",
    "GraphicalAbstract": "GraphicalAbstract", "HiResFigureFiles": "HiResFigureFiles",
    "FigS1": "FigS1_training_design", "FigS2": "FigS2_grouped_retraining_overlap",
    "FigS3": "FigS3_four_dopant_fixed_path", "FigS4": "FigS4_sliding", "FigS5": "FigS5_feature_audit",
}
INDEX_FIELDS = [
    "item_id", "section", "panel", "content", "final_asset", "asset_sha256", "generator",
    "plotted_data", "upstream_data", "raw_location", "models", "calc_type", "status", "note", "item_dir",
]
# Downloads / workspace paths whose content now lives in the repo or archive store.
EXTRA_REWRITES = [
    ("/Users/alina/Project/migrationbench-paper-repro/", ""),
    ("/Users/alina/Project/migrationbench-integrate/", ""),
    ("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/foundation_interlayer_sliding_10model_20261024",
     "reproducibility/canva_palette_vector_bundle_20261105/data/source_archive/analysis/foundation_interlayer_sliding_10model_20261024"),
    ("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/output/revision1_camera_ready/reproducibility",
     "release/revision1_camera_ready/reproducibility"),
    # stale pointer recorded in the bundle; the checkpoints are in ~/Downloads/mace-models (MODEL_REGISTRY.tsv)
    ("/Users/alina/Downloads/mace-foundation-models", "local:/Users/alina/Project/migrationbench-archive/downloads_20261110/mace-models"),
    ("/Users/alina/Downloads/mace-models", "local:/Users/alina/Project/migrationbench-archive/downloads_20261110/mace-models"),
]


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def path_rewrites() -> list[tuple[str, str]]:
    plan = read_tsv(PACKAGE / "imports/import_plan.tsv")
    pairs = [(row["source"].rstrip("/"), row["dest"].rstrip("/")) for row in plan] + EXTRA_REWRITES
    return sorted(pairs, key=lambda pair: len(pair[0]), reverse=True)


def rewrite(text: str, rules: list[tuple[str, str]]) -> str:
    for old, new in rules:
        text = text.replace(old, new)
    return text.replace("C/analysis/", "analysis/")


def item_key(item_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", item_id)


def item_dir(item_id: str, label: str) -> str:
    key = item_key(item_id)
    if key.startswith("Table"):
        section = "response" if key.startswith("TableR") else "si"
        number = int(re.sub(r"\D", "", key))
        slug = f"{key[:6]}{number:02d}_{label.split(':', 1)[-1]}"
        return f"items/{section}/{slug}"
    section = "si" if key.startswith("FigS") else "main"
    return f"items/{section}/{SLUGS.get(key, key)}"


def main_rows(rules) -> list[dict]:
    rows = []
    for row in read_tsv(PACKAGE / "sources/main_figures.tsv"):
        rows.append({
            "item_id": row["item_id"], "section": "main", "panel": row["panel"], "content": row["content"],
            "final_asset": row["final_asset"],
            "asset_sha256": row["asset_sha256"] if re.fullmatch(r"[0-9a-f]{64}", row["asset_sha256"]) else "",
            "generator": rewrite(row["plot_script"], rules), "plotted_data": rewrite(row["plotted_data"], rules),
            "upstream_data": rewrite(row["upstream_data"], rules), "raw_location": rewrite(row["raw_location"], rules),
            "models": rewrite(row["models"], rules), "calc_type": row["calc_type"], "status": row["status"],
            "note": rewrite(" ".join(filter(None, [row["composition"], row["notes"]])), rules),
            "item_dir": item_dir(row["item_id"], ""),
        })
    return rows


def si_rows(rules) -> list[dict]:
    rows = []
    for row in read_tsv(PACKAGE / "sources/si_and_tables.tsv"):
        section = "response" if row["item_id"].startswith("Table R") else "si"
        rows.append({
            "item_id": row["item_id"], "section": section, "panel": row["label"], "content": row["content"],
            "final_asset": row["source_tex"], "asset_sha256": "",
            "generator": rewrite(row["generator_script"], rules), "plotted_data": rewrite(row["backing_data"], rules),
            "upstream_data": rewrite(row["upstream_data"], rules), "raw_location": rewrite(row["raw_location"], rules),
            "models": rewrite(row["models"], rules), "calc_type": row["calc_type"], "status": row["status"],
            "note": rewrite(" ".join(filter(None, [row["value_check"], row["notes"]])), rules),
            "item_dir": item_dir(row["item_id"], row["label"]),
        })
    return rows


def apply_overrides(rows: list[dict]) -> list[dict]:
    overrides = {(o["item_id"], o["panel"]): o for o in read_tsv(PACKAGE / "sources/status_overrides.tsv")}
    updated = []
    for row in rows:
        override = overrides.get((row["item_id"], row["panel"]))
        if override:
            row = {**row, "status": override["status"], "note": f"{override['note']} | survey: {row['note']}"}
        updated.append(row)
    return updated


def storage_rows() -> list[dict]:
    rows = []
    for row in read_tsv(PACKAGE / "sources/rockfish_inventory.tsv"):
        keep = row["used_in_paper"] not in {"junk", "no"} or row["category"] != "other_project"
        rows.append({
            "location": f"rockfish:{row['rockfish_path']}", "host": "rockfish", "size": row["size"],
            "n_files": row["n_files"], "mtime": row["mtime"], "category": row["category"],
            "used_in_paper": row["used_in_paper"], "paper_items": row["paper_items"],
            "local_copy": row["local_copy"], "proposed_archive_path": row["proposed_archive_path"],
            "migrate_to": "skipjack + backup" if keep and row["used_in_paper"] != "junk" else "do not migrate",
            "notes": row["notes"],
        })
    rows += [
        {"location": "local:/Users/alina/Project/migrationbench-archive/repo_mirror", "host": "laptop",
         "category": "imported heavy binaries (extxyz/traj/blend/npz)", "used_in_paper": "yes",
         "paper_items": "Fig1 Fig6 Fig4 FigS3 FigS4", "migrate_to": "skipjack + backup",
         "notes": f"files listed with sha256 in {PACKAGE_REL}/imports/EXTERNAL_STORE_MANIFEST.tsv"},
        {"location": "local:/Users/alina/Project/migrationbench-archive/downloads_20261110/mace-models", "host": "laptop", "category": "model",
         "used_in_paper": "yes", "paper_items": "foundation rows of Fig2-Fig5, Table S8",
         "migrate_to": "skipjack + backup", "notes": "public MACE foundation checkpoints moved from ~/Downloads/mace-models (2026-11-10, imports/DOWNLOADS_ARCHIVE.tsv); Rockfish copies in MODEL_REGISTRY.tsv"},
        {"location": "local:/Users/alina/Project/migrationbench-archive/downloads_20261110", "host": "laptop",
         "category": "Downloads archive (Overleaf zip, Canva backups, Figure 1 kit, drafts)", "used_in_paper": "partial",
         "paper_items": "all (lineage)", "migrate_to": "skipjack + backup", "notes": "per-file sha256 in imports/DOWNLOADS_ARCHIVE.tsv"},
        {"location": "icloud:/Users/alina/Documents/ChatGPT/Wiley-MigrationBench", "host": "laptop (iCloud, partly evicted)",
         "category": "workspace", "used_in_paper": "partial", "migrate_to": "skipjack (cold archive) after imports verified",
         "notes": "trace-relevant files imported; see imports/IMPORT_MANIFEST.tsv and ICLOUD_DATALESS.tsv"},
    ]
    return rows


STORAGE_FIELDS = ["location", "host", "size", "n_files", "mtime", "category", "used_in_paper", "paper_items",
                  "local_copy", "proposed_archive_path", "migrate_to", "notes"]
MODEL_FIELDS = ["model_id", "family", "condition", "train_temp", "seed", "location", "sha256_recorded",
                "sha256_verified", "config_path", "train_data_path", "used_in_paper", "paper_items",
                "proposed_archive_path", "migrate_to"]


def model_rows() -> list[dict]:
    rows = []
    for row in read_tsv(PACKAGE / "sources/rockfish_models.tsv"):
        rows.append({**row, "location": f"rockfish:{row['rockfish_path']}", "migrate_to": "skipjack + backup"})
    with (PACKAGE / "sources/foundation_verify.tsv").open() as handle:
        for line in handle:
            path, size, sha = line.rstrip("\n").split("\t")[:3]
            rows.append({"model_id": Path(path).stem, "family": "MACE foundation (public)", "location": f"rockfish:{path}",
                         "sha256_recorded": sha, "sha256_verified": "yes", "used_in_paper": "see paper_items",
                         "migrate_to": "skipjack + backup"})
    return rows


def provenance_page(item: str, rows: list[dict]) -> str:
    head = rows[0]
    lines = [f"# {item} — provenance (Revision 1, submitted 2026-11-10)", "",
             "Generated by `tools/build_index.py` from `RESULTS_INDEX.tsv`; edit the sources, not this file.", ""]
    for row in rows:
        lines += [f"## Panel / label: {row['panel']}", "", f"- **Content:** {row['content']}",
                  f"- **Status:** `{row['status']}`", f"- **Final asset:** `{row['final_asset']}` {row['asset_sha256'][:12]}",
                  f"- **Generator:** {row['generator'] or '—'}", f"- **Plotted data:** {row['plotted_data'] or '—'}",
                  f"- **Upstream data:** {row['upstream_data'] or '—'}", f"- **Raw location:** {row['raw_location'] or '—'}",
                  f"- **Models:** {row['models'] or '—'}", f"- **Calculation type:** {row['calc_type'] or '—'}",
                  f"- **Notes:** {row['note'] or '—'}", ""]
    return "\n".join(lines) if head else ""


def sql_literal(value) -> str:
    return "'" + str(value if value is not None else "").replace("'", "''") + "'"


def write_migration(tables: dict[str, tuple[list[str], list[dict]]]) -> str:
    out = ["-- Generated by release/revision1_submitted_20261110/tools/build_index.py; do not hand-edit.",
           "INSERT OR REPLACE INTO metadata (key, value) VALUES ('revision1_submission_trace', '002 (2026-11-10)');"]
    for table, (fields, rows) in tables.items():
        out.append(f"DROP TABLE IF EXISTS {table};")
        out.append(f"CREATE TABLE {table} ({', '.join(f'{f} TEXT' for f in fields)});")
        for row in rows:
            out.append(f"INSERT INTO {table} VALUES ({', '.join(sql_literal(row.get(f)) for f in fields)});")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-sql", action="store_true", help="do not apply the migration to the SQLite DB")
    args = parser.parse_args()
    rules = path_rewrites()
    index = apply_overrides(main_rows(rules) + si_rows(rules))
    storage, models = storage_rows(), model_rows()
    imports = read_tsv(PACKAGE / "imports/IMPORT_MANIFEST.tsv")
    external = read_tsv(PACKAGE / "imports/EXTERNAL_STORE_MANIFEST.tsv")
    write_tsv(PACKAGE / "RESULTS_INDEX.tsv", INDEX_FIELDS, index)
    write_tsv(PACKAGE / "STORAGE_LOCATIONS.tsv", STORAGE_FIELDS, storage)
    write_tsv(PACKAGE / "MODEL_REGISTRY.tsv", MODEL_FIELDS, models)

    by_item: dict[str, list[dict]] = {}
    for row in index:
        by_item.setdefault(row["item_dir"], []).append(row)
    for directory, rows in by_item.items():
        target = PACKAGE / directory
        target.mkdir(parents=True, exist_ok=True)
        (target / "PROVENANCE.md").write_text(provenance_page(rows[0]["item_id"], rows))

    sql = write_migration({
        "rev1_results_index": (INDEX_FIELDS, index),
        "rev1_storage_locations": (STORAGE_FIELDS, storage),
        "rev1_model_registry": (MODEL_FIELDS, models),
        "rev1_imported_files": (["source", "dest", "bytes", "sha256", "reason"], imports),
        "rev1_external_store": (["repo_path", "stored_at", "bytes", "sha256", "original_source", "migration"], external),
    })
    MIGRATION.write_text(sql)
    if not args.no_sql:
        with sqlite3.connect(DB) as conn:
            conn.executescript(sql)
    print(f"{len(index)} index rows, {len(by_item)} item pages, {len(storage)} storage rows, "
          f"{len(models)} models, {len(imports)} imported files; migration -> {MIGRATION.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
