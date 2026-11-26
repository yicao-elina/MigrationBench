#!/usr/bin/env python3
"""Build a compact, source-pinned SQLite index of Lmax1 diagnostic records."""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import stat
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "sources"
MANIFEST = HERE / "source_manifest.json"
EXPECTED_MANIFEST_SHA256 = "7141da48ad46cf453ffb1ab92edbfff8b3c4d7b8974aaae1672bac63773e86e3"

CAMPAIGNS = [
    {
        "campaign_id": "scratch_lmax1_g3_tier1_20261025_r01",
        "stage": "TIER1_BASELINE",
        "collector": "G3_TIER1_COLLECT.tsv",
        "scope": "ACCEPTANCE_SCOPE_R01.json",
        "source_ledger": "G3_TIER1_FAILURE_LEDGER.tsv",
        "parent": None,
        "relation": None,
        "expected_rows": 54,
    },
    {
        "campaign_id": "scratch_lmax1_g3_tier1_retry_20261025_r01",
        "stage": "TIER1_RETRY",
        "collector": "G3_TIER1_RETRY_COLLECT.tsv",
        "scope": "ACCEPTANCE_SCOPE_R01.json",
        "source_ledger": "SOURCE_G3_TIER1_FAILURE_LEDGER.tsv",
        "parent": "scratch_lmax1_g3_tier1_20261025_r01",
        "relation": "RETRY_OF_TIER1_FAILURE",
        "expected_rows": 30,
    },
    {
        "campaign_id": "scratch_lmax1_g3_perturb_20261025_r01",
        "stage": "TIER1_PERTURBATION",
        "collector": "G3_PERTURB_COLLECT.tsv",
        "scope": "ACCEPTANCE_SCOPE_R01.json",
        "source_ledger": "SOURCE_G3_TIER1_FAILURE_LEDGER.tsv",
        "parent": "scratch_lmax1_g3_tier1_20261025_r01",
        "relation": "PERTURBATION_OF_TIER1_FAILURE",
        "expected_rows": 20,
    },
    {
        "campaign_id": "scratch_lmax1_g3_tier2_20261025_r01",
        "stage": "TIER2_DIAGNOSTIC",
        "collector": "G3_TIER2_COLLECT.tsv",
        "scope": "ACCEPTANCE_SCOPE_R01.json",
        "source_ledger": "G3_TIER2_COLLECT_FAILURE_LEDGER.tsv",
        "parent": None,
        "relation": None,
        "expected_rows": 90,
    },
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path):
    with open(str(path), "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError("missing TSV header: {}".format(path))
        return list(reader), list(reader.fieldnames)


def clean(value):
    return "" if value is None else str(value)


def verify_sources(source_dir, manifest_path):
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("source manifest SHA-256 differs from pinned package manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified = {}
    for entry in manifest["source_files"]:
        path = source_dir / entry["relative_path"]
        if not path.is_file():
            raise IOError("pinned source file is missing: {}".format(path))
        if path.stat().st_size != int(entry["bytes"]):
            raise ValueError("source size mismatch: {}".format(path))
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError("source SHA-256 mismatch: {}".format(path))
        verified[entry["relative_path"]] = dict(entry)
    actual = {p.relative_to(source_dir).as_posix() for p in source_dir.rglob("*") if p.is_file()}
    if actual != set(verified):
        raise ValueError("source bundle file set differs from pinned manifest")
    return manifest, verified


def role_for(name):
    if name.endswith("ACCEPTANCE_SCOPE_R01.json"):
        return "acceptance_scope"
    if name.endswith("MODEL_REGISTRY.tsv"):
        return "model_registry"
    if name.endswith("PATH_REGISTRY.tsv"):
        return "path_registry"
    if "FAILURE_LEDGER" in name:
        return "failure_ledger"
    if name.endswith("_COLLECT.tsv"):
        return "collector"
    if "JOBS" in name:
        return "planned_jobs"
    return "source_metadata"


def build_snapshot(source_dir, manifest_path, output_path, archive_path):
    source_dir = Path(source_dir).resolve()
    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise IOError("refusing to replace existing output: {}".format(output_path))
    manifest, pinned_files = verify_sources(source_dir, manifest_path)
    by_relative = {item["relative_path"]: item for item in pinned_files.values()}
    stages = []
    all_records = {}
    records_by_campaign = {}

    for config in CAMPAIGNS:
        prefix = config["campaign_id"]
        scope_rel = prefix + "/" + config["scope"]
        scope_path = source_dir / scope_rel
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
        scope_digest = by_relative[scope_rel]["sha256"]
        model_rel = prefix + "/MODEL_REGISTRY.tsv"
        path_rel = prefix + "/PATH_REGISTRY.tsv"
        model_sha = by_relative[model_rel]["sha256"]
        path_sha = by_relative[path_rel]["sha256"]
        expected_model = scope.get("model_registry_sha256") or scope.get("artifact_hashes", {}).get("model_registry")
        expected_path = scope.get("path_registry_sha256") or scope.get("artifact_hashes", {}).get("path_registry")
        if expected_model != model_sha or expected_path != path_sha:
            raise ValueError("registry pin mismatch in {}".format(prefix))
        if scope.get("campaign_id") != prefix:
            raise ValueError("campaign id mismatch in {}".format(scope_rel))

        source_rel = prefix + "/" + config["source_ledger"]
        collector_rel = prefix + "/" + config["collector"]
        rows, columns = read_tsv(source_dir / collector_rel)
        if len(rows) != config["expected_rows"]:
            raise ValueError("{} expected {} rows, found {}".format(prefix, config["expected_rows"], len(rows)))
        seen_ids = set()
        current = []
        for row_number, row in enumerate(rows, 1):
            record_id = clean(row.get("record_id"))
            if not record_id or record_id in seen_ids:
                raise ValueError("missing or duplicate record_id in {} row {}".format(prefix, row_number))
            seen_ids.add(record_id)
            status = clean(row.get("run_status"))
            manifest_path_value = clean(row.get("manifest_path"))
            materialization = "MISSING" if status.upper() == "MISSING" or not manifest_path_value else "PRESENT_IN_COLLECTOR"
            current.append({
                "campaign_id": prefix,
                "record_id": record_id,
                "stage": config["stage"],
                "source_row_number": row_number,
                "source_record_id": clean(row.get("source_record_id")),
                "source_job_index": clean(row.get("source_job_index") or row.get("job_index")),
                "operator": clean(row.get("operator")),
                "path_id": clean(row.get("path_id")),
                "model_id": clean(row.get("model_id")),
                "materialization_status": materialization,
                "run_status": status,
                "scientific_status": clean(row.get("scientific_status")),
                "failure_class": clean(row.get("failure_class")),
                "leakage_gate": clean(row.get("leakage_gate")),
                "manifest_path": manifest_path_value,
                "manifest_sha256": clean(row.get("manifest_sha256")),
                "publication_status": clean(row.get("publication_status")),
                "raw_row_json": json.dumps(row, sort_keys=True, ensure_ascii=False),
            })
        records_by_campaign[prefix] = current
        all_records[prefix] = {r["record_id"]: r for r in current}
        stages.append({
            "campaign_id": prefix,
            "stage": config["stage"],
            "root_path": "/scratch16/pclancy3/yi/" + prefix,
            "parent_campaign_id": config["parent"],
            "model_registry_sha256": model_sha,
            "path_registry_sha256": path_sha,
            "source_ledger_sha256": by_relative[source_rel]["sha256"],
            "acceptance_scope_sha256": scope_digest,
            "acceptance_scope_json": json.dumps(scope, sort_keys=True, ensure_ascii=False),
            "relationship_boundary": "explicit_source_record_id_edges_only" if config["parent"] else "no_parent_attempt_link_inferred",
        })

    tier1_id = CAMPAIGNS[0]["campaign_id"]
    failure_ids = {r["record_id"] for r in records_by_campaign[tier1_id] if r["scientific_status"] == "NOT_CONVERGED"}
    if len(failure_ids) != 10:
        raise ValueError("expected 10 Tier-1 NOT_CONVERGED source rows, found {}".format(len(failure_ids)))
    if len(records_by_campaign[tier1_id]) != 54:
        raise ValueError("Tier-1 baseline must retain all 54 collector rows")

    links = []
    for config in CAMPAIGNS[1:3]:
        child_rows = records_by_campaign[config["campaign_id"]]
        counts = Counter()
        for row in child_rows:
            source_id = row["source_record_id"]
            if source_id not in failure_ids:
                raise ValueError("{} source_record_id is not a Tier-1 failure: {}".format(config["campaign_id"], source_id))
            counts[source_id] += 1
            links.append((tier1_id, source_id, config["campaign_id"], row["record_id"], config["relation"], "source_record_id"))
        expected_per_source = 3 if config["stage"] == "TIER1_RETRY" else 2
        if set(counts) != failure_ids or any(count != expected_per_source for count in counts.values()):
            raise ValueError("{} explicit source edge multiplicity is incomplete".format(config["campaign_id"]))

    retry = records_by_campaign[CAMPAIGNS[1]["campaign_id"]]
    retry_counts = Counter(r["materialization_status"] for r in retry)
    if retry_counts != Counter({"PRESENT_IN_COLLECTOR": 20, "MISSING": 10}):
        raise ValueError("retry materialization counts differ from pinned audit: {}".format(dict(retry_counts)))
    tier2 = records_by_campaign[CAMPAIGNS[3]["campaign_id"]]
    pending = [r for r in tier2 if r["leakage_gate"] == "PENDING_G3_11"]
    if len(pending) != 90:
        raise ValueError("expected all 90 Tier-2 leakage gates pending, found {}".format(len(pending)))
    if any(r["materialization_status"] != "PRESENT_IN_COLLECTOR" for r in tier2):
        raise ValueError("Tier-2 collector includes an unmaterialized row")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(output_path))
    try:
        db.execute("PRAGMA journal_mode=DELETE")
        db.executescript((HERE / "schema.sql").read_text(encoding="utf-8"))
        # Parents must precede child campaign rows due to the explicit stage FK.
        for stage in stages:
            db.execute("INSERT INTO campaign_stage VALUES (?,?,?,?,?,?,?,?,?,?)", tuple(stage[k] for k in (
                "campaign_id", "stage", "root_path", "parent_campaign_id", "model_registry_sha256", "path_registry_sha256",
                "source_ledger_sha256", "acceptance_scope_sha256", "acceptance_scope_json", "relationship_boundary")))
        for relative, entry in sorted(pinned_files.items()):
            campaign_id = relative.split("/", 1)[0]
            db.execute("INSERT INTO source_file VALUES (?,?,?,?,?,?)", (
                relative, campaign_id, entry["rockfish_path"], int(entry["bytes"]), entry["sha256"], role_for(relative)))
        for config in CAMPAIGNS:
            for row in records_by_campaign[config["campaign_id"]]:
                db.execute("INSERT INTO campaign_record VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", tuple(row[k] for k in (
                    "campaign_id", "record_id", "stage", "source_row_number", "source_record_id", "source_job_index", "operator",
                    "path_id", "model_id", "materialization_status", "run_status", "scientific_status", "failure_class",
                    "leakage_gate", "manifest_path", "manifest_sha256", "publication_status", "raw_row_json")))
        db.executemany("INSERT INTO record_link VALUES (?,?,?,?,?,?)", links)
        status_counts = {}
        for campaign_id, records in records_by_campaign.items():
            status_counts[campaign_id] = dict(sorted(Counter(r["scientific_status"] for r in records).items()))
        metadata = {
            "snapshot_id": "lmax1_failure_retry_event_index_r1_20261126",
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_file_count": str(len(pinned_files)),
            "source_file_bytes_sum": str(sum(int(item["bytes"]) for item in pinned_files.values())),
            "campaign_record_count": str(sum(len(rows) for rows in records_by_campaign.values())),
            "campaign_record_counts_json": json.dumps({k: len(v) for k, v in records_by_campaign.items()}, sort_keys=True),
            "scientific_status_counts_json": json.dumps(status_counts, sort_keys=True),
            "explicit_source_link_count": str(len(links)),
            "retry_materialization_counts_json": json.dumps(dict(sorted(retry_counts.items())), sort_keys=True),
            "tier2_pending_g3_11_count": str(len(pending)),
            "scope_boundary": "diagnostic_only_candidate_only; no decision, training-selection, or measured-gain evidence",
            "missing_retry_boundary": "10 planned climb_long rows explicitly remain MISSING, not scientific failures",
            "tier2_boundary": "90 records retain PENDING_G3_11; no leakage pass or publication acceptance inferred",
        }
        db.executemany("INSERT INTO snapshot_metadata VALUES (?,?)", sorted(metadata.items()))
        db.commit()
        db.execute("VACUUM")
        if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("snapshot quick_check failed")
        if db.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("snapshot has foreign-key violations")
    finally:
        db.close()
    os.chmod(str(output_path), stat.S_IRUSR | stat.S_IWUSR)
    counts = {campaign_id: len(records) for campaign_id, records in records_by_campaign.items()}
    result = {
        "snapshot": "Lmax1 G3 diagnostic source and explicit retry/perturbation lineage index",
        "database_path": archive_path,
        "bytes": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_file_count": len(pinned_files),
        "source_file_bytes_sum": sum(int(item["bytes"]) for item in pinned_files.values()),
        "campaign_record_counts": counts,
        "campaign_record_total": sum(counts.values()),
        "explicit_source_links": len(links),
        "retry_materialization_counts": dict(sorted(retry_counts.items())),
        "tier2_pending_g3_11": len(pending),
        "quick_check": "ok",
        "foreign_key_violations": 0,
        "user_version": 1,
        "mode": "0600",
        "limitations": [
            "Records are diagnostic summaries from collectors; run manifests and large raw outputs were not rehashed or imported.",
            "Missing planned retry variants are preserved as MISSING and are not optimizer failures.",
            "Tier 2 remains candidate-only with all G3_11 leakage gates pending.",
            "No detector decision, selected training data, next checkpoint, or measured active-learning gain is recorded.",
        ],
    }
    manifest_out = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(str(manifest_out), stat.S_IRUSR | stat.S_IWUSR)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=str(SOURCE_DIR), help="directory containing the pinned compact source files")
    parser.add_argument("--source-manifest", default=str(MANIFEST), help="pinned source file manifest")
    parser.add_argument("--output", required=True, help="new SQLite snapshot path; existing path is never replaced")
    parser.add_argument("--archive-path", help="stable Rockfish deployment path stored in output manifest")
    args = parser.parse_args(argv)
    build_snapshot(args.source_dir, args.source_manifest, args.output, args.archive_path or args.output)


if __name__ == "__main__":
    try:
        main()
    except (IOError, OSError, sqlite3.Error, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.stderr.write("Lmax1 import failed: {}\n".format(exc))
        sys.exit(2)
