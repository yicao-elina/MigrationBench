#!/usr/bin/env python3
"""Create R2 by copying claim-matrix R1 and adding its Fig. 4/R05 crosswalk."""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BRIDGE = HERE.parent
R05_ROOT = REPO / "paper" / "revision1"
PROFILE_TSV = R05_ROOT / "data_processed" / "current_release" / "cr_three_path_plotted_profiles_R05.tsv"
COVERAGE_TSV = R05_ROOT / "data_processed" / "current_release" / "cr_three_path_coverage_audit_R05.tsv"
R05_MANIFEST = R05_ROOT / "data_processed" / "current_release" / "cr_three_path_fixed_vs_self_2x3_R05.manifest.json"
FIGURE_PDF = R05_ROOT / "figures" / "main" / "fig4_NEB_new.pdf"
SCHEMA = HERE / "schema.sql"
SOURCE_MAIN_COMMIT = "296ef562a5bc28151a824d9201c383766cfa116f"
EXPECTED = {
    "base_r1_sha256": "cdc723b93143b6778fa1b3d6915a798e0cfd6fc6f31a6dce9e969a580e766578",
    "r4_sha256": "d033f47913eaf33d61a2ce7b0b0281b90b385d23190890a97f477e2f901896d7",
    "profile_sha256": "705643bbad0220de074853cf11b94eacbe739f6dc8ca5d5c16921374d2e6d398",
    "coverage_sha256": "753de574b2e73912500385f3edfbbdd088fae95623975a10c3d8ecab93db3cc3",
    "manifest_sha256": "d4221a019c3a047da022404e6c31a1f3c5896cf6d1d1cb5d1a48ff3f5f590071",
    "figure_sha256": "7b1016535481017801e07694d86e0d3943613c739d20f821dec83663fcf2cc93",
}
R4_PATH_ON_ROCKFISH = "/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_lineage_overlays/20261125_r4/migrationbench_r1_lineage_overlay_20261125_r4.sqlite"
R1_PATH_ON_ROCKFISH = "/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_claim_matrix_snapshots/20261126/claim_matrix_r1.sqlite"
R2_PATH_ON_ROCKFISH = "/scratch16/pclancy3/yi/MigrationBench_data_archive/revision1_claim_matrix_snapshots/20261126_r2/claim_matrix_r2.sqlite"

PANEL_PATH = {"a": "in_gap", "b": "gap_to_ql", "c": "within_ql",
              "d": "in_gap", "e": "gap_to_ql", "f": "within_ql"}
PANEL_OBSERVABLE = {panel: ("FIXED_PATH_SINGLE_POINT" if panel in "abc"
                            else "SELF_RELAXED_ENDPOINT_ONLY_NEB") for panel in PANEL_PATH}
PATH_CLASS = {"in_gap": "in-gap diffusion", "gap_to_ql": "gap→QL", "within_ql": "within-QL"}
GROUPS = ("foundation", "ft_600k", "ft_multit", "scratch_600k", "scratch_multit")
R4_SHA_RELPATH = "../revision1_lineage_overlay_r4/SNAPSHOT_MANIFEST.json"


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_file(label, path, expected):
    if not path.is_file():
        raise IOError("{} input is missing: {}".format(label, path))
    actual = sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError("{} SHA-256 mismatch: expected {}, found {}".format(label, expected, actual))
    return actual


def read_tsv(path, expected_columns):
    with open(str(path), "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != expected_columns:
            raise ValueError("{} has unexpected columns: {}".format(path, reader.fieldnames))
        return list(reader)


def check_sqlite(path, label):
    con = sqlite3.connect(str(path.resolve().as_uri()) + "?mode=ro", uri=True)
    try:
        quick = con.execute("PRAGMA quick_check").fetchone()[0]
        foreign = con.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        con.close()
    if quick != "ok" or foreign:
        raise ValueError("{} integrity failure: quick_check={}, FK violations={}".format(label, quick, len(foreign)))


def build_rows(r4_path, profiles, coverage, hashes):
    profile_index = {}
    profile_line = {}
    for line, row in enumerate(profiles, start=2):
        key = (row["observable"], row["path_key"], row["group"])
        if key in profile_index:
            raise ValueError("duplicate R05 profile key: {}".format(key))
        profile_index[key] = row
        profile_line[key] = line

    coverage_index = {}
    coverage_line = {}
    for line, row in enumerate(coverage, start=2):
        path_key = next((key for key, value in PATH_CLASS.items() if value == row["path_class"]), None)
        if path_key is None:
            raise ValueError("unrecognized R05 path class: {}".format(row["path_class"]))
        key = (row["observable"], path_key, row["group"])
        if key in coverage_index:
            raise ValueError("duplicate R05 coverage key: {}".format(key))
        coverage_index[key] = row
        coverage_line[key] = line

    if len(profiles) != 29 or len(coverage) != 30:
        raise ValueError("unexpected R05 row counts: profiles={}, coverage={}".format(len(profiles), len(coverage)))
    if set(coverage_index) != {
        (PANEL_OBSERVABLE[panel], PANEL_PATH[panel], group)
        for panel in PANEL_PATH for group in GROUPS
    }:
        raise ValueError("R05 coverage does not enumerate the expected 6-panel × 5-group grid")

    r4 = sqlite3.connect(str(r4_path.resolve().as_uri()) + "?mode=ro", uri=True)
    r4.row_factory = sqlite3.Row
    try:
        raw_results = r4.execute(
            "SELECT result_id, result_key, source_row_key, protocol_key, evidence_status, scientific_status "
            "FROM lineage_results WHERE result_key LIKE 'R1X:Fig4:%' ORDER BY result_key"
        ).fetchall()
    finally:
        r4.close()
    if len(raw_results) != 30:
        raise ValueError("expected 30 R4 Fig. 4 summary rows, found {}".format(len(raw_results)))

    rows = []
    seen_keys = set()
    for result in raw_results:
        pieces = result["result_key"].split(":")
        if len(pieces) != 4 or pieces[:2] != ["R1X", "Fig4"]:
            raise ValueError("unexpected R4 Fig. 4 result key: {}".format(result["result_key"]))
        panel, group = pieces[2], pieces[3]
        if panel not in PANEL_PATH or group not in GROUPS:
            raise ValueError("unrecognized R4 Fig. 4 panel/group: {}".format(result["result_key"]))
        observable = PANEL_OBSERVABLE[panel]
        path_key = PANEL_PATH[panel]
        key = (observable, path_key, group)
        if key in seen_keys:
            raise ValueError("duplicate R4 result summary key: {}".format(key))
        seen_keys.add(key)
        cov = coverage_index.get(key)
        if cov is None:
            raise ValueError("R4 result has no R05 coverage row: {}".format(key))

        profile = profile_index.get(key)
        match_status = cov["status"]
        if match_status == "PLOTTED":
            if profile is None:
                raise ValueError("R05 plotted coverage has no profile row: {}".format(key))
            for field in ("n_models", "n_profile_rows"):
                if profile[field] != cov[field]:
                    raise ValueError("R05 profile/coverage mismatch for {} field {}".format(key, field))
            if result["protocol_key"] != observable:
                raise ValueError("R4 protocol differs from panel-derived observable: {}".format(result["result_key"]))
            rows.append((
                result["result_key"], result["result_id"], result["result_key"], panel, group,
                observable, path_key, PATH_CLASS[path_key], profile_line[key], coverage_line[key],
                "PRESENT", int(cov["n_models"]), int(cov["n_profile_rows"]),
                profile["scientific_statuses"], profile["publication_statuses"], profile["source_boundary"],
                result["protocol_key"], result["evidence_status"], result["scientific_status"],
                "SUMMARY_TO_R05_GROUP_ROW", "NOT_ESTABLISHED", "NOT_ASSESSED",
                "{}:{}:{}".format(observable, path_key, group),
            ))
        elif match_status == "MISSING_OR_EXCLUDED":
            if profile is not None or (panel, path_key, group) != ("f", "within_ql", "ft_multit"):
                raise ValueError("unexpected missing R05 group: {}".format(key))
            if int(cov["n_models"]) != 0 or int(cov["n_profile_rows"]) != 0:
                raise ValueError("explicit missing row has nonzero R05 counts")
            if result["protocol_key"] != observable:
                raise ValueError("R4 protocol differs from panel-derived observable: {}".format(result["result_key"]))
            rows.append((
                result["result_key"], result["result_id"], result["result_key"], panel, group,
                observable, path_key, PATH_CLASS[path_key], None, coverage_line[key],
                "MISSING_OR_EXCLUDED", 0, 0, "NOT_RECORDED", "NOT_RECORDED",
                "R05_COVERAGE_AUDIT_ONLY", result["protocol_key"],
                result["evidence_status"], result["scientific_status"],
                "SUMMARY_TO_R05_GROUP_ROW", "NOT_ESTABLISHED", "NOT_ASSESSED",
                "{}:{}:{}".format(observable, path_key, group),
            ))
        else:
            raise ValueError("unexpected R05 coverage status {} for {}".format(match_status, key))

    if len(rows) != 30 or sum(row[10] == "PRESENT" for row in rows) != 29:
        raise ValueError("R4/R05 join cardinality failed")
    if sum(row[10] == "MISSING_OR_EXCLUDED" for row in rows) != 1:
        raise ValueError("expected one explicit missing/excluded R05 slot")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-r1-path", required=True, help="exact immutable claim_matrix_r1.sqlite")
    parser.add_argument("--r4-path", required=True, help="exact R4 lineage overlay SQLite snapshot")
    parser.add_argument("--output", required=True, help="new R2 SQLite output path; refuses overwrite")
    parser.add_argument("--archive-path", help="stable Rockfish R2 path recorded in the manifest")
    parser.add_argument("--manifest-output", help="optional versioned snapshot-manifest output path")
    args = parser.parse_args()

    base = Path(args.base_r1_path).expanduser().resolve()
    r4_path = Path(args.r4_path).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    archive_path = args.archive_path or R2_PATH_ON_ROCKFISH
    if output.exists():
        raise ValueError("refusing to replace existing output: {}".format(output))
    r1_sha = check_file("claim-matrix R1", base, EXPECTED["base_r1_sha256"])
    r4_sha = check_file("R4 lineage overlay", r4_path, EXPECTED["r4_sha256"])
    r05_sha = {
        "profile_tsv": check_file("R05 plotted profiles", PROFILE_TSV, EXPECTED["profile_sha256"]),
        "coverage_tsv": check_file("R05 coverage audit", COVERAGE_TSV, EXPECTED["coverage_sha256"]),
        "r05_manifest": check_file("R05 manifest", R05_MANIFEST, EXPECTED["manifest_sha256"]),
        "fig4_pdf": check_file("Fig. 4 PDF", FIGURE_PDF, EXPECTED["figure_sha256"]),
        "sqlite_schema": sha256(SCHEMA),
        "importer": sha256(Path(__file__)),
    }
    check_sqlite(base, "claim-matrix R1")
    check_sqlite(r4_path, "R4 lineage overlay")

    profiles = read_tsv(PROFILE_TSV, ["observable", "path_key", "path_class", "group", "n_models",
                                       "n_profile_rows", "scientific_statuses", "publication_statuses",
                                       "source_boundary"])
    coverage = read_tsv(COVERAGE_TSV, ["observable", "path_class", "group", "n_models",
                                        "n_profile_rows", "status"])
    rows = build_rows(r4_path, profiles, coverage, r05_sha)

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(base), str(output))
    db = sqlite3.connect(str(output))
    try:
        # QCE04 reads snapshot_metadata, which is copied from R1 and intentionally unchanged.
        db.executescript(SCHEMA.read_text(encoding="utf-8"))
        metadata = {
            "crosswalk_version": "R2_FIG4_R05_SUMMARY_CROSSWALK",
            "base_claim_matrix_r1_sha256": r1_sha,
            "base_claim_matrix_r1_archive_path": R1_PATH_ON_ROCKFISH,
            "r4_snapshot_sha256": r4_sha,
            "r4_snapshot_path": R4_PATH_ON_ROCKFISH,
            "r05_profile_tsv_sha256": r05_sha["profile_tsv"],
            "r05_coverage_tsv_sha256": r05_sha["coverage_tsv"],
            "r05_manifest_sha256": r05_sha["r05_manifest"],
            "fig4_pdf_sha256": r05_sha["fig4_pdf"],
            "source_main_commit": SOURCE_MAIN_COMMIT,
            "crosswalk_rows": "30",
            "present_profile_rows": "29",
            "missing_or_excluded_rows": "1",
            "mapping_key": "observable,path_key,group",
            "grain_boundary": "Fig. 4 R4 profile summaries mapped to aggregate R05 group rows; no run or attempt joins",
            "acceptance_boundary": "No figure validation or scientific acceptance claim; R05 statuses are preserved as source values",
        }
        db.executemany(
            "INSERT INTO fig4_crosswalk_metadata(metadata_key,metadata_value) VALUES(?,?)",
            sorted(metadata.items()),
        )
        columns = ["crosswalk_key", "r4_result_id", "r4_result_key", "panel", "group_key",
                   "observable", "path_key", "path_class", "profile_data_line", "coverage_data_line",
                   "match_status", "n_models", "n_profile_rows", "r05_scientific_statuses",
                   "r05_publication_statuses", "source_boundary", "r4_protocol_key",
                   "r4_evidence_status", "r4_scientific_status", "mapping_grain",
                   "run_attempt_linkage", "scientific_acceptance", "source_r05_key"]
        placeholders = ",".join("?" for _ in columns)
        db.executemany(
            "INSERT INTO fig4_r05_summary_crosswalk({}) VALUES({})".format(",".join(columns), placeholders),
            rows,
        )
        db.commit()
        quick = db.execute("PRAGMA quick_check").fetchone()[0]
        foreign = db.execute("PRAGMA foreign_key_check").fetchall()
        r1_rows = db.execute("SELECT COUNT(*) FROM claim_evidence_row").fetchone()[0]
        r1_meta = db.execute("SELECT COUNT(*) FROM snapshot_metadata").fetchone()[0]
        crosswalk_counts = dict(db.execute(
            "SELECT match_status,COUNT(*) FROM fig4_r05_summary_crosswalk GROUP BY match_status"
        ).fetchall())
    finally:
        db.close()
    if quick != "ok" or foreign or r1_rows != 639 or r1_meta != 20:
        raise ValueError("R2 integrity/R1 preservation failed: quick_check={}, FK={}, claims={}, metadata={}".format(
            quick, len(foreign), r1_rows, r1_meta))
    if crosswalk_counts != {"MISSING_OR_EXCLUDED": 1, "PRESENT": 29}:
        raise ValueError("R2 crosswalk counts differ: {}".format(crosswalk_counts))
    os.chmod(str(output), stat.S_IRUSR | stat.S_IWUSR)

    manifest = {
        "snapshot": "MigrationBench Revision 1 claim matrix R2 with Fig. 4/R05 summary crosswalk",
        "snapshot_version": "claim_matrix_r2",
        "user_version": 1,
        "database_path": archive_path,
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
        "quick_check": quick,
        "foreign_key_violations": len(foreign),
        "claim_evidence_row_count": r1_rows,
        "snapshot_metadata_row_count": r1_meta,
        "fig4_r05_crosswalk_row_count": len(rows),
        "fig4_r05_match_status_counts": dict(sorted(crosswalk_counts.items())),
        "base_r1": {"path": R1_PATH_ON_ROCKFISH, "bytes": base.stat().st_size, "sha256": r1_sha},
        "r4": {"path": R4_PATH_ON_ROCKFISH, "bytes": r4_path.stat().st_size, "sha256": r4_sha},
        "r05_sources": {
            "profile_tsv": {"path": "paper/revision1/data_processed/current_release/cr_three_path_plotted_profiles_R05.tsv", "sha256": r05_sha["profile_tsv"], "rows": len(profiles)},
            "coverage_tsv": {"path": "paper/revision1/data_processed/current_release/cr_three_path_coverage_audit_R05.tsv", "sha256": r05_sha["coverage_tsv"], "rows": len(coverage)},
            "manifest": {"path": "paper/revision1/data_processed/current_release/cr_three_path_fixed_vs_self_2x3_R05.manifest.json", "sha256": r05_sha["r05_manifest"]},
            "fig4_pdf": {"path": "paper/revision1/figures/main/fig4_NEB_new.pdf", "sha256": r05_sha["fig4_pdf"]},
        },
        "mapping_key": ["observable", "path_key", "group"],
        "mapping_grain": "summary row to R05 group row; no run/attempt linkage",
        "source_main_commit": SOURCE_MAIN_COMMIT,
        "builder_sha256": {"importer": r05_sha["importer"], "schema": r05_sha["sqlite_schema"]},
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "QCE01-QCE04 tables and snapshot_metadata are copied unchanged from the pinned R1 database.",
            "The 30-row table maps aggregate Fig. 4 summaries to R05 aggregate rows only.",
            "The explicit missing row is not evidence of zero attempts.",
            "Hash identity does not establish run provenance, plotted-output validity, or scientific acceptance.",
            "Fixed-path single-point and endpoint-only self-relaxed MLIP-NEB are different protocols.",
        ],
    }
    manifest_path = Path(args.manifest_output).expanduser().resolve() if args.manifest_output else HERE / "snapshot_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (IOError, OSError, sqlite3.Error, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.stderr.write("Fig. 4/R05 crosswalk import failed: {}\n".format(exc))
        sys.exit(2)
