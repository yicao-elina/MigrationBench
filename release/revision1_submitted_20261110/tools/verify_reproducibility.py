#!/usr/bin/env python3
"""Check that every figure and table has a reproduction script and current, passing evidence.

For each item in RESULTS_INDEX.tsv this requires at least one items/<...>/evidence/*.json that
  * says PASS, with no hardcode-lint findings,
  * was written by a script that still exists and still has the recorded sha256,
  * read inputs that still exist with the recorded sha256.
Items that cannot have a script are listed, with the reason, in sources/reproduce_exceptions.tsv.
A failing check is accepted only when sources/known_discrepancies.tsv names that exact check or
every one of its mismatch details (an error in the submitted version, or an unrecoverable
environment); the item is then PASS_KNOWN_ISSUE and the issue number is reported.
Writes REPRODUCIBILITY.tsv (one row per item). With --rerun, runs every evidence command first.
Exit status 1 if any item fails.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

from repro_lib import PACKAGE, REPO, lint_script, sha256

FIELDS = ["item_id", "section", "item_dir", "result", "scripts", "evidence", "checks", "inputs", "problems"]
THREADS = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}


def read_tsv(rel: str) -> list[dict]:
    with (PACKAGE / rel).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def items() -> dict[str, dict]:
    out = {}
    for row in read_tsv("RESULTS_INDEX.tsv"):
        out.setdefault(row["item_id"], {"item_id": row["item_id"], "section": row["section"], "item_dir": row["item_dir"]})
    return out


def explained(check: dict, known: list[dict]) -> list[dict]:
    """Known-discrepancy rows that account for a failing check (all its details, or its name)."""
    details = check["details"] if isinstance(check["details"], list) else [str(check["details"])]
    by_name = [k for k in known if k["match"] == check["check"]]
    if by_name:
        return by_name
    rows = [next((k for k in known if k["match"] == d), None) for d in details]
    return rows if rows and all(rows) else []


def evidence_problems(record: dict, known: list[dict] | None = None, notes: list[str] | None = None) -> list[str]:
    problems = []
    for check in (c for c in record["checks"] if not c["passed"]):
        rows = explained(check, known or [])
        if rows and notes is not None:
            notes.extend(f"known issue {r['issue_no']}: {r['note']}" for r in rows)
        elif not rows:
            problems.append(f"{record['name']}: {check['check']}")
    if not record["checks"]:
        problems.append("no checks")
    script = REPO / record["command"].split()[-1]
    if not script.is_file():
        problems.append(f"script missing: {record['command']}")
    else:
        if sha256(script) != record["script_sha256"]:
            problems.append(f"stale evidence: {script.name} changed since the evidence was written")
        problems += lint_script(script)
    for path, digest in record["inputs"].items():
        target = REPO / path
        if not target.exists():
            problems.append(f"input missing: {path}")
        elif digest != "directory" and sha256(target) != digest:
            problems.append(f"input changed: {path}")
    return problems


def rerun(records: list[dict]) -> None:
    for command in sorted({r["command"] for r in records}):
        script = command.split()[-1]
        print(f"rerun {script}")
        subprocess.run([sys.executable, script], cwd=REPO, env={**os.environ, **THREADS}, check=False)


def load(item: dict) -> list[dict]:
    folder = PACKAGE / item["item_dir"] / "evidence"
    return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))] if folder.is_dir() else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rerun", action="store_true", help="run every evidence command before checking")
    args = parser.parse_args()
    catalogue = items()
    exceptions = {row["item_id"]: row for row in read_tsv("sources/reproduce_exceptions.tsv")}
    discrepancies = read_tsv("sources/known_discrepancies.tsv")
    if args.rerun:
        rerun([r for item in catalogue.values() for r in load(item)])
    rows, failed = [], 0
    for item in catalogue.values():
        records = load(item)
        known = [k for k in discrepancies if k["item_id"] == item["item_id"]]
        notes: list[str] = []
        problems = [p for r in records for p in evidence_problems(r, known, notes)]
        exception = exceptions.get(item["item_id"])
        if not records and not exception:
            problems.append("no reproduction script / evidence")
        if records and not problems:
            result = "PASS_KNOWN_ISSUE" if notes else "PASS"
        else:
            result = "EXCEPTION" if exception and not records else "FAIL"
        failed += result == "FAIL"
        rows.append({**item, "result": result,
                     "scripts": " ".join(sorted({r["command"].split()[-1] for r in records})),
                     "evidence": " ".join(f"{item['item_dir']}/evidence/{n}" for n in
                                          sorted(p.name for p in (PACKAGE / item["item_dir"] / "evidence").glob("*.json")))
                     if records else "",
                     "checks": str(sum(len(r["checks"]) for r in records)),
                     "inputs": str(len({p for r in records for p in r["inputs"]})),
                     "problems": "; ".join(problems + sorted(set(notes)))
                     or (exception["reason"] if exception and not records else "")})
    with (PACKAGE / "REPRODUCIBILITY.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        mark = {"PASS": "ok  ", "PASS_KNOWN_ISSUE": "ok* ", "EXCEPTION": "exc ", "FAIL": "FAIL"}[row["result"]]
        print(f"{mark} {row['item_id']:<18} {row['problems'][:150]}")
    print(f"{len(rows)} items: {sum(r['result'] == 'PASS' for r in rows)} pass, "
          f"{sum(r['result'] == 'PASS_KNOWN_ISSUE' for r in rows)} pass with a known issue, "
          f"{sum(r['result'] == 'EXCEPTION' for r in rows)} exceptions, {failed} fail")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
