"""Tests for the Revision 1 trace tools: portable paths, SQL tables, overview helpers."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "release/revision1_submitted_20261110/tools"
sys.path.insert(0, str(TOOLS))

import check_sql  # noqa: E402
import overview_spec  # noqa: E402
import path_roots  # noqa: E402
import sql_tables  # noqa: E402

HOME = path_roots.HOME


def test_portable_strips_repo_root_and_labels_external_roots():
    text = (f"{HOME}/Project/migrationbench-paper-repro/analysis/x.csv; "
            f"local:{HOME}/Project/migrationbench-archive/repo_mirror/a.traj; "
            f"{HOME}/Documents/ChatGPT/Wiley-MigrationBench/output/b.json")
    assert path_roots.portable(text) == "analysis/x.csv; archive:repo_mirror/a.traj; chatgpt:output/b.json"


def test_portable_is_idempotent_and_leaves_rockfish_paths():
    text = "rockfish:/scratch16/pclancy3/yi/run archive:downloads_20261110/m.model"
    assert path_roots.portable(text) == text
    assert path_roots.portable(path_roots.portable(f"{HOME}/Downloads/a.pdf")) == "downloads:a.pdf"


def test_localize_prefers_the_repo_copy_of_an_imported_source():
    rule = (f"{HOME}/Documents/ChatGPT/Wiley-MigrationBench/analysis/foo", "analysis/foo")
    assert path_roots.localize(f"see {rule[0]}/bar.py", [rule]) == "see analysis/foo/bar.py"


def test_resolve_maps_local_labels_and_skips_remote_ones():
    assert path_roots.resolve("archive:repo_mirror/a") == Path(f"{HOME}/Project/migrationbench-archive/repo_mirror/a")
    assert path_roots.resolve("rockfish:/scratch16/x") is None


def test_known_issues_are_parsed_with_groups_and_resolution():
    issues = sql_tables.known_issue_rows()
    numbers = [int(row["issue_no"]) for row in issues]
    assert numbers == sorted(numbers) and numbers[0] == 1
    assert all(row["issue_group"] and row["title"] for row in issues)
    assert any(row["resolved"] == "yes" for row in issues)


def test_pipeline_has_ten_ordered_stages():
    stages = sql_tables.pipeline_rows()
    assert [row["stage"] for row in stages] == [str(n) for n in range(10)]


def test_migration_keys_every_table_and_rejects_duplicates():
    specs = {"t": (["a", "b"], ["a"], [{"a": "1", "b": f"{HOME}/Downloads/x"}, {"a": "1", "b": "y"}])}
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.executescript(sql_tables.write_migration(specs))


def test_migration_writes_portable_values():
    specs = {"t": (["a"], ["a"], [{"a": f"{HOME}/Downloads/x.pdf"}])}
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
    conn.executescript(sql_tables.write_migration(specs))
    assert conn.execute("SELECT a FROM t").fetchone()[0] == "downloads:x.pdf"


def test_registry_passes_sql_checks():
    conn = sqlite3.connect(check_sql.DB)
    assert check_sql.run_checks(conn) == []


def test_tiers_detects_storage_and_falls_back_to_referenced_item():
    assert overview_spec.tiers("rockfish:/scratch16/a and analysis/b.csv") == "git, Rockfish"
    assert overview_spec.tiers("see S9") == "as S9"
    assert overview_spec.tiers("n/a") == "—"
