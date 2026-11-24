#!/usr/bin/env python3
"""Move the guarded Conda package cleanup batch from EVIDENCE_READY to EXECUTING."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

AUDIT = Path(__file__).resolve().parent.parent
DB = AUDIT / "provenance_catalog.sqlite"
PREFLIGHT = AUDIT / "phase2_data" / "conda_package_cleanup_preflight_20261122_r02.json"
BATCH = "RF-DATA-CONDA-PKG-LIBSTDCXX-UNUSED-20261122-R01"


def main() -> None:
    evidence = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if evidence.get("event") != "preflight_passed" or evidence.get("queue", {}).get("overlap_count") != 0:
        raise RuntimeError("Latest exact-manifest guard preflight did not pass")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    con = sqlite3.connect(DB, timeout=120)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=120000")
    try:
        con.execute("BEGIN IMMEDIATE")
        batch = con.execute("SELECT status,expected_bytes FROM cleanup_batch WHERE cleanup_batch_id=?", (BATCH,)).fetchone()
        items = con.execute(
            "SELECT count(*),sum(pre_delete_size_bytes),sum(deletion_state='EVIDENCE_READY') FROM cleanup_item WHERE cleanup_batch_id=? AND disposition='DELETE'",
            (BATCH,),
        ).fetchone()
        if batch != ("EVIDENCE_READY", 14223129) or items != (90, 14223129, 90):
            raise RuntimeError("Exact cleanup batch or all 90 pre-delete evidence rows are not ready")
        receipt = con.execute(
            """SELECT 1 FROM cleanup_execution_receipt cer JOIN registry_source s ON s.source_id=cer.receipt_source_id
               WHERE cer.cleanup_batch_id=? AND cer.execution_state='PRECHECKED'
                 AND s.source_key='conda_package_live_preflight_20261122_r02'""",
            (BATCH,),
        ).fetchone()
        if not receipt:
            raise RuntimeError("Current R02 preflight is not linked in the SQL cleanup receipt table")
        con.execute("UPDATE cleanup_batch SET status='EXECUTING' WHERE cleanup_batch_id=?", (BATCH,))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    print(json.dumps({"batch": BATCH, "status": "EXECUTING", "started_utc": now, "delete_items": 90}, sort_keys=True))


if __name__ == "__main__":
    main()
