"""SQL schema of the Revision 1 trace tables (migration 002), built from the package TSVs.

Every table has a primary key, so a duplicate row in a TSV fails the migration instead
of entering the registry twice. All text goes through path_roots.portable().
"""
from __future__ import annotations

import re
from path_roots import PACKAGE, portable, read_tsv, root_rows

VERSION = "002 (2026-11-10, keyed tables + issues/pipeline/migration plan)"


def _tsv(rel: str) -> tuple[list[str], list[dict]]:
    rows = read_tsv(PACKAGE / rel)
    return list(rows[0]), rows


def known_issue_rows() -> list[dict]:
    """Numbered items of KNOWN_ISSUES.md, with the section they sit under."""
    rows, group, current = [], "", None
    for line in (PACKAGE / "KNOWN_ISSUES.md").read_text().splitlines():
        if line.startswith("## "):
            group = line[3:].strip()
            continue
        match = re.match(r"(\d+)\.\s+(.*)", line)
        if match:
            current = {"issue_no": match.group(1), "issue_group": group, "text": match.group(2)}
            rows.append(current)
        elif current and line.startswith("   "):
            current["text"] += " " + line.strip()
        elif not line.strip():
            current = None
    for row in rows:
        title = re.match(r"\*\*(.+?)\*\*", row["text"])
        row["title"] = title.group(1).rstrip(".:") if title else row["text"][:80]
        row["resolved"] = "yes" if "(resolved)" in row["text"] else "no"
    return rows


def pipeline_rows() -> list[dict]:
    rows = []
    for line in (PACKAGE / "PIPELINE.md").read_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0].isdigit():
            rows.append(dict(zip(["stage", "stage_name", "flow", "storage", "feeds"], cells)))
    return rows


def item_rows(index: list[dict]) -> list[dict]:
    items: dict[str, dict] = {}
    for row in index:
        item = items.setdefault(row["item_id"], {
            "item_id": row["item_id"], "section": row["section"], "item_dir": row["item_dir"],
            "title": row["content"], "panels": [], "statuses": []})
        item["panels"].append(row["panel"])
        if row["status"] not in item["statuses"]:
            item["statuses"].append(row["status"])
    return [{**item, "n_rows": str(len(item["panels"])), "panels": " ".join(item["panels"]),
             "statuses": " ".join(item["statuses"])} for item in items.values()]


def table_specs(index, storage, models, index_fields, storage_fields, model_fields) -> dict:
    """name -> (fields, primary key, rows)."""
    specs = {
        "rev1_path_roots": (["label", "absolute_root", "host", "meaning"], ["label"], root_rows()),
        "rev1_results_index": (index_fields, ["item_id", "panel"], index),
        "rev1_items": (["item_id", "section", "item_dir", "title", "n_rows", "panels", "statuses"], ["item_id"],
                       item_rows(index)),
        "rev1_storage_locations": (storage_fields, ["location"], storage),
        "rev1_model_registry": (model_fields, ["location"], models),
        "rev1_pipeline_stages": (["stage", "stage_name", "flow", "storage", "feeds"], ["stage"], pipeline_rows()),
        "rev1_known_issues": (["issue_no", "issue_group", "title", "resolved", "text"], ["issue_no"],
                              known_issue_rows()),
    }
    for name, rel, key in [
        ("rev1_item_calc_layers", "sources/item_calc_layers.tsv", ["item_id"]),
        ("rev1_rockfish_migration_plan", "ROCKFISH_MIGRATION_PLAN.tsv", ["rockfish_path"]),
        ("rev1_import_plan", "imports/import_plan.tsv", ["source"]),
        ("rev1_imported_files", "imports/IMPORT_MANIFEST.tsv", ["dest"]),
        ("rev1_external_store", "imports/EXTERNAL_STORE_MANIFEST.tsv", ["repo_path"]),
        ("rev1_archived_untracked_raw", "imports/ARCHIVED_UNTRACKED_RAW.tsv", ["repo_path"]),
        ("rev1_downloads_archive", "imports/DOWNLOADS_ARCHIVE.tsv", ["original_path"]),
        ("rev1_icloud_dataless", "imports/ICLOUD_DATALESS.tsv", ["source"]),
        ("rev1_deferred_large_files", "imports/DEFERRED_LARGE_FILES.tsv", ["source"]),
        ("rev1_import_conflicts", "imports/IMPORT_CONFLICTS.tsv", ["source"]),
    ]:
        fields, rows = _tsv(rel)
        specs[name] = (fields, key, rows)
    return specs


VIEWS = {
    "v_rev1_item_status": """
        SELECT i.section, i.item_id, i.n_rows, i.statuses, i.item_dir,
               SUM(r.status NOT LIKE 'FULL%') AS rows_not_full
        FROM rev1_items i JOIN rev1_results_index r USING (item_id)
        GROUP BY i.item_id ORDER BY i.section, i.item_id""",
    "v_rev1_models_used": """
        SELECT model_id, family, condition, train_temp, seed, sha256_verified, paper_items, location
        FROM rev1_model_registry WHERE used_in_paper = 'yes' ORDER BY family, model_id""",
    "v_rev1_rockfish_to_migrate": """
        SELECT priority, used_in_paper, size, category, rockfish_path, skipjack_target, paper_items
        FROM rev1_rockfish_migration_plan ORDER BY CAST(priority AS INTEGER), rockfish_path""",
    "v_rev1_open_issues": """
        SELECT issue_no, issue_group, title FROM rev1_known_issues WHERE resolved = 'no'
        ORDER BY CAST(issue_no AS INTEGER)""",
}


def sql_literal(value, keep_paths: bool = False) -> str:
    text = str(value if value is not None else "")
    return "'" + (text if keep_paths else portable(text)).replace("'", "''") + "'"


def write_migration(specs: dict) -> str:
    out = ["-- Generated by release/revision1_submitted_20261110/tools/build_index.py; do not hand-edit.",
           "-- Paths are portable: '<label>:rel' resolves through rev1_path_roots.",
           "BEGIN;",
           f"INSERT OR REPLACE INTO metadata (key, value) VALUES ('revision1_submission_trace', {sql_literal(VERSION)});"]
    out += [f"DROP VIEW IF EXISTS {name};" for name in VIEWS]
    for name, (fields, key, rows) in specs.items():
        columns = ", ".join(f"{field} TEXT NOT NULL DEFAULT ''" for field in fields)
        out.append(f"DROP TABLE IF EXISTS {name};")
        out.append(f"CREATE TABLE {name} ({columns}, PRIMARY KEY ({', '.join(key)}));")
        for row in rows:
            keep = name == "rev1_path_roots"  # the root table holds the absolute roots themselves
            values = ", ".join(sql_literal(row.get(field), keep) for field in fields)
            out.append(f"INSERT INTO {name} VALUES ({values});")
    out += [f"CREATE VIEW {name} AS {body.strip()};" for name, body in VIEWS.items()]
    out.append("COMMIT;")
    return "\n".join(out) + "\n"


def counts(specs: dict) -> dict[str, int]:
    return {name: len(rows) for name, (_f, _k, rows) in specs.items()}

