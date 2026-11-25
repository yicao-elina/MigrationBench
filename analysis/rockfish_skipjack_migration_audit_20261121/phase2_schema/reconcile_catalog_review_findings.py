#!/usr/bin/env python3
"""Reconcile reviewed cleanup, recipe, and merge-event semantics in SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"


def main() -> None:
    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        con.execute("BEGIN IMMEDIATE")
        batch = con.execute(
            "SELECT status,expected_bytes,actual_bytes FROM cleanup_batch WHERE cleanup_batch_id='RF-DATA-CONDA-INDEX-CACHE-20261122'"
        ).fetchone()
        items = con.execute(
            "SELECT count(*),sum(pre_delete_size_bytes),sum(actual_bytes),sum(pre_delete_sha256 IS NOT NULL),sum(deleted_utc IS NOT NULL) FROM cleanup_item WHERE cleanup_batch_id='RF-DATA-CONDA-INDEX-CACHE-20261122'"
        ).fetchone()
        if batch is None or items != (42, 1123596223, 1124405248, 0, 0):
            raise RuntimeError(f"Unexpected post-hoc Conda batch evidence: batch={batch}; items={items}")
        con.execute(
            """UPDATE cleanup_item SET actual_bytes=pre_delete_size_bytes
               WHERE cleanup_batch_id='RF-DATA-CONDA-INDEX-CACHE-20261122'"""
        )
        con.execute(
            """UPDATE cleanup_batch SET status='PARTIAL',actual_bytes=expected_bytes,
                      authorization_note=authorization_note || ' Physical cache removal was verified after the event, but the evidence is partial: no pre-delete content hashes or contemporaneous queue check.'
               WHERE cleanup_batch_id='RF-DATA-CONDA-INDEX-CACHE-20261122'"""
        )
        con.execute(
            """UPDATE git_merge_observation SET pre_target_sha=resulting_sha
               WHERE repository_id=27 AND source_ref='dev' AND merge_result='ALREADY_CONTAINS'
                 AND resulting_sha='817acf22af7374bcb2a79d123d54ad77a7411c1c'"""
        )
        con.execute(
            """UPDATE reproduction_recipe SET validation_note=?
               WHERE run_id='RF-LAMMPS-FLARE-BUILD-B620C52-20261122-R01' AND recipe_version='2026-11-22-r03'""",
            ("Remote rebuild script SHA verified; KokkosKernels moved and full file/hash/path verification passed at the versioned sibling path. No rebuild job was submitted. The guarded cleanup of 1,876 generated build files completed at 2026-11-22T22:54:41Z; the verified per-file receipt and before/after quota observations are linked to the cleanup batch.",),
        )
        if con.execute("SELECT changes()").fetchone()[0] != 1:
            raise RuntimeError("LAMMPS recipe row was not updated")
        if con.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("Foreign-key check failed")
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    with sqlite3.connect(DB) as verify:
        report = {
            "conda_batch": verify.execute("SELECT status,expected_bytes,actual_bytes FROM cleanup_batch WHERE cleanup_batch_id='RF-DATA-CONDA-INDEX-CACHE-20261122'").fetchone(),
            "conda_items": verify.execute("SELECT count(*),sum(actual_bytes),sum(pre_delete_sha256 IS NOT NULL),sum(deleted_utc IS NOT NULL) FROM cleanup_item WHERE cleanup_batch_id='RF-DATA-CONDA-INDEX-CACHE-20261122'").fetchone(),
            "git_dev_observation": verify.execute("SELECT pre_target_sha,resulting_sha FROM git_merge_observation WHERE repository_id=27 AND source_ref='dev'").fetchone(),
            "lammps_validation_note": verify.execute("SELECT validation_note FROM reproduction_recipe WHERE run_id='RF-LAMMPS-FLARE-BUILD-B620C52-20261122-R01' AND recipe_version='2026-11-22-r03'").fetchone()[0],
            "foreign_key_violations": len(verify.execute("PRAGMA foreign_key_check").fetchall()),
        }
        print(json.dumps(report, default=list, sort_keys=True))


if __name__ == "__main__":
    main()
