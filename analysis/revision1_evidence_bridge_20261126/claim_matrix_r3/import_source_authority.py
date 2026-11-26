#!/usr/bin/env python3
"""Create additive claim-matrix R3 manuscript source and ledger-delta tables."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3


HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(HERE, "schema_additions.sql")
EXPECTED_R2_SHA256 = "bbc13c16855d54760944ba9b98287ef91e578558a4a9e63bfd580296c8e075d2"
SOURCE_FILES = [
    ("SUBMITTED_REVISION_1", "sn-article-final.tex", "main_manuscript"),
    ("SUBMITTED_REVISION_1", "sn-article-SI.tex", "supplementary_information"),
    ("SUBMITTED_REVISION_1", "data_processed/manuscript_source_ledger.csv", "claim_source_ledger"),
    ("PR10_OVERLEAF_CANDIDATE", "sn-article-final.tex", "main_manuscript"),
    ("PR10_OVERLEAF_CANDIDATE", "sn-article-SI.tex", "supplementary_information"),
    ("PR10_OVERLEAF_CANDIDATE", "data_processed/manuscript_source_ledger.csv", "claim_source_ledger"),
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ledger(path):
    with open(path, "r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        key = row["claim_id"]
        if key in result:
            raise ValueError("duplicate claim_id in ledger: " + key)
        result[key] = row
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("r2_database")
    parser.add_argument("submitted_root")
    parser.add_argument("candidate_root")
    parser.add_argument("candidate_readme")
    parser.add_argument("pr10_reconciliation_json")
    parser.add_argument("output_database")
    args = parser.parse_args()

    if sha256_file(args.r2_database) != EXPECTED_R2_SHA256:
        raise ValueError("base claim-matrix R2 SHA-256 mismatch")
    if os.path.exists(args.output_database):
        os.unlink(args.output_database)
    shutil.copyfile(args.r2_database, args.output_database)

    connection = sqlite3.connect(args.output_database)
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        connection.executescript(open(SCHEMA, "r").read())
        versions = [
            ("SUBMITTED_REVISION_1", "SUBMITTED_RELEASE_CANONICAL",
             "334260cdda3869d55999081cd610b34bfba53257", None, None,
             "HASHED_FROM_CANONICAL_SUBMITTED_TREE",
             "Release README identifies these paper/revision1 TeX/ledger sources as the submitted Revision 1 files, synced from the 2026-11-09 Overleaf export. The original export ZIP was not independently recovered in this audit."),
            ("PR10_OVERLEAF_CANDIDATE", "CLOSED_UNMERGED_CANDIDATE",
             "51cb06f71732e46cd10eb92cee1f0c5f8f4833c9", 10, "CLOSED_UNMERGED",
             "HASHED_FROM_PRESERVED_CANDIDATE_SNAPSHOT",
             "PR #10 is closed with no merge commit. Its README says the three files are comparison-only candidate copies and were not compiled as a standalone tree. This does not establish that this snapshot equals the live Overleaf tip."),
            ("LIVE_OVERLEAF_TIP", "LIVE_OVERLEAF_NOT_VERIFIED",
             None, None, None, "NOT_CAPTURED",
             "Current server-side Overleaf source was not captured or hash-compared in this audit; no current-tip identity is asserted."),
        ]
        connection.executemany(
            "INSERT INTO manuscript_source_version VALUES (?, ?, ?, ?, ?, ?, ?)", versions)

        version_roots = {
            "SUBMITTED_REVISION_1": args.submitted_root,
            "PR10_OVERLEAF_CANDIDATE": args.candidate_root,
        }
        repository_prefixes = {
            "SUBMITTED_REVISION_1": "paper/revision1",
            "PR10_OVERLEAF_CANDIDATE": "paper/revision1_candidates/pr10_overleaf_candidate_20261123/source/paper/revision1",
        }
        files = []
        for version_id, relative_path, role in SOURCE_FILES:
            absolute_path = os.path.join(version_roots[version_id], relative_path)
            with open(absolute_path, "rb") as handle:
                byte_count = os.fstat(handle.fileno()).st_size
            note = "Exact file identity in the named Git source tree; text differences do not validate cited evidence."
            repo_path = repository_prefixes[version_id] + "/" + relative_path
            files.append((version_id, repo_path, role, byte_count,
                          sha256_file(absolute_path), note))
        candidate_readme_path = args.candidate_readme
        files.append(("PR10_OVERLEAF_CANDIDATE",
                      "paper/revision1_candidates/pr10_overleaf_candidate_20261123/README.md",
                      "candidate_source_context", os.path.getsize(candidate_readme_path),
                      sha256_file(candidate_readme_path),
                      "States the candidate is comparison-only and not the submitted release."))
        connection.executemany(
            "INSERT INTO manuscript_source_file VALUES (?, ?, ?, ?, ?, ?)", files)

        submitted = load_ledger(os.path.join(args.submitted_root,
                                             "data_processed/manuscript_source_ledger.csv"))
        candidate = load_ledger(os.path.join(args.candidate_root,
                                             "data_processed/manuscript_source_ledger.csv"))
        deltas = []
        for claim_id in sorted(set(submitted) | set(candidate)):
            old, new = submitted.get(claim_id), candidate.get(claim_id)
            if old is None:
                delta_type = "CANDIDATE_ONLY"
                fields = "claim_id"
            elif new is None:
                delta_type = "SUBMITTED_ONLY"
                fields = "claim_id"
            elif old != new:
                delta_type = "ALTERED"
                fields = ",".join(sorted(k for k in old if old[k] != new[k]))
            else:
                continue
            deltas.append((claim_id, delta_type,
                           old.get("status") if old else None,
                           new.get("status") if new else None,
                           fields,
                           json.dumps(old, sort_keys=True) if old else None,
                           json.dumps(new, sort_keys=True) if new else None,
                           "Versioned ledger-text comparison only; no source artifact was revalidated by this delta row."))
        connection.executemany(
            "INSERT INTO manuscript_ledger_delta VALUES (?, ?, ?, ?, ?, ?, ?, ?)", deltas)
        with open(args.pr10_reconciliation_json, "r") as handle:
            reconciliation = json.load(handle)
        if (reconciliation["pr_head"] != "51cb06f71732e46cd10eb92cee1f0c5f8f4833c9"
                or reconciliation["current_main"] != "19708d74d14103832e4f86b51b8b5d8875175cc3"
                or reconciliation["changed_file_count"] != 47):
            raise ValueError("PR #10 full-tree reconciliation identity mismatch")
        rows = [(
            item["pull_request"], item["pr_head_commit"], item["source_path"],
            item["pr_blob_oid"], item["pr_byte_count"], item["pr_sha256"],
            item["current_path"], item["current_blob_oid"],
            item["current_byte_count"], item["current_sha256"],
            item["reconciliation_status"], item["reconciliation_note"]
        ) for item in reconciliation["rows"]]
        connection.executemany(
            "INSERT INTO pr_file_reconciliation VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows)
        connection.commit()
        connection.execute("PRAGMA user_version = 3")
        connection.commit()
        claim_rows = connection.execute("SELECT COUNT(*) FROM claim_evidence_row").fetchone()[0]
        metadata_rows = connection.execute("SELECT COUNT(*) FROM snapshot_metadata").fetchone()[0]
        fig4_rows = connection.execute("SELECT COUNT(*) FROM fig4_r05_summary_crosswalk").fetchone()[0]
        source_rows = connection.execute("SELECT COUNT(*) FROM manuscript_source_version").fetchone()[0]
        file_rows = connection.execute("SELECT COUNT(*) FROM manuscript_source_file").fetchone()[0]
        delta_rows = connection.execute("SELECT COUNT(*) FROM manuscript_ledger_delta").fetchone()[0]
        pr_rows = connection.execute("SELECT COUNT(*) FROM pr_file_reconciliation").fetchone()[0]
        quick = connection.execute("PRAGMA quick_check").fetchone()[0]
        fk = connection.execute("PRAGMA foreign_key_check").fetchall()
        if (claim_rows, metadata_rows, fig4_rows, source_rows, file_rows, pr_rows) != (639, 20, 30, 3, 7, 47):
            raise ValueError("unexpected preserved or added row counts")
        if quick != "ok" or fk:
            raise ValueError("snapshot integrity failure")
        print("claim_rows={} metadata_rows={} fig4_rows={} source_versions={} files={} ledger_deltas={} pr_files={} quick_check={} fk={}".format(
            claim_rows, metadata_rows, fig4_rows, source_rows, file_rows, delta_rows, pr_rows, quick, len(fk)))
        print("delta_classes={}".format(connection.execute(
            "SELECT delta_type, COUNT(*) FROM manuscript_ledger_delta GROUP BY delta_type ORDER BY delta_type").fetchall()))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
