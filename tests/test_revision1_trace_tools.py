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


# ---------------------------------------------------------------- repro_lib
import re  # noqa: E402

import repro_lib  # noqa: E402


def _labels(tex: str) -> list[str]:
    return re.findall(r"\\label\{(tab:[^}]+)\}", (repro_lib.PAPER / tex).read_text())


@pytest.mark.parametrize("tex", ["sn-article-SI.tex", "response_to_reviewers.tex"])
def test_every_submitted_table_has_body_rows(tex):
    for label in _labels(tex):
        assert repro_lib.submitted_table(tex, label), label


def test_compare_rows_accepts_submitted_precision_and_rejects_wrong_values():
    submitted = [["FT-600K", "1,786", r"$0.47\pm0.12$", r"$6.3\times10^{-2}$"]]
    assert repro_lib.compare_rows(submitted, [["FT-600K", "1786", r"$0.4712\pm0.1234$", "0.0631"]]) == []
    errors = repro_lib.compare_rows(submitted, [["FT-600K", "1787", r"$0.49\pm0.12$", "0.0631"]])
    assert len(errors) == 2
    assert repro_lib.compare_rows(submitted, [["Scratch", "1786", r"$0.47\pm0.12$", "0.063"]])  # text differs


def test_formatters_match_submitted_style():
    assert repro_lib.fmt_int(1786.0) == "1,786"
    assert repro_lib.fmt_pm(0.4712, 0.1234) == r"$0.47\pm0.12$"
    assert repro_lib.fmt_sci(0.0631) == r"$6.3\times10^{-2}$"


def test_lint_flags_absolute_paths_and_data_arrays_but_not_layout(tmp_path):
    script = tmp_path / "s.py"
    script.write_text('"""doc /Users/x"""\nP = "/Users/alina/Downloads/a.pdf"\n'
                      "DATA = [1.5243, 0.3361, 2.1187, 0.9912, 1.0047, 3.2215]\n"
                      "TICKS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]\n")
    original = repro_lib.REPO
    try:
        repro_lib.REPO = tmp_path
        findings = repro_lib.lint_script(script)
    finally:
        repro_lib.REPO = original
    assert any("absolute path" in f for f in findings)
    assert any("literal float array" in f for f in findings)
    assert len(findings) == 2
