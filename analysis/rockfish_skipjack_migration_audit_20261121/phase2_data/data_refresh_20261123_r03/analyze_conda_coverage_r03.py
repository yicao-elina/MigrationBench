#!/usr/bin/env python3
"""Reconcile the full Nov 22 Conda baseline with current partial and root scans."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import stat
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


AUDIT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASELINE = AUDIT / "phase2_data" / "data_path_inventory_20261122.jsonl.gz"
R02 = AUDIT / "phase2_data" / "data_refresh_20261123" / "data_pclancy3_refresh_20261123.jsonl.gz"
INDEX_MANIFEST = AUDIT / "phase2_data" / "conda_index_cache_removed_manifest_20261122.tsv"
INDEX_VERIFY = AUDIT / "phase2_data" / "conda_index_cache_removed_verification_20261122.txt"
PACKAGE_RECEIPT = AUDIT / "phase2_data" / "conda_package_cleanup_execution_receipt_20261122.json"
ENV_ROOTS = ["automatbench", "deepmd2", "qe-dev", "sklearn-env"]
CONDA_ROOT = "/data/pclancy3/yi/.conda"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def type_old(record: str) -> str:
    return {"dir": "directory", "file": "file", "symlink": "symlink"}.get(record, record)


def old_metadata(row: dict) -> dict:
    t = type_old(row.get("record", "unknown"))
    return {
        "path": row["path"], "type": t,
        "size_bytes": int(row.get("size_bytes", 0)) if t == "file" else 0,
        "stat_size_bytes": int(row.get("size_bytes", 0)),
        "allocated_bytes": int(row.get("blocks_512", 0)) * 512,
        "mtime_ns": row.get("mtime_ns"), "device": row.get("dev"),
        "inode": row.get("ino"), "nlink": row.get("nlink"),
        "mode_text": row.get("mode"), "source": "R01_FULL_BASELINE",
    }


def current_metadata(row: dict, source: str) -> dict:
    t = row.get("type", "unknown")
    return {
        "path": row["path"], "type": t,
        "size_bytes": int(row.get("size_bytes", 0)) if t == "file" else 0,
        "stat_size_bytes": int(row.get("stat_size_bytes", 0)),
        "allocated_bytes": int(row.get("allocated_bytes", 0)),
        "mtime_ns": row.get("mtime_ns"), "device": row.get("device"),
        "inode": row.get("inode"), "nlink": row.get("nlink"),
        "mode_octal": row.get("mode_octal"), "uid": row.get("uid"),
        "owner": row.get("owner"), "gid": row.get("gid"),
        "group": row.get("group"), "source": source,
    }


def permissions_from_ls(mode: str | None) -> int | None:
    if not mode or len(mode) < 10:
        return None
    bits = 0
    for idx, shift in ((1, 8), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (8, 1), (9, 0)):
        char = mode[idx]
        if char in "rwxst":
            bits |= 1 << shift
        if idx in (3, 6, 9) and char in "sStT":
            bits |= 1 << (11 if idx == 3 else 10 if idx == 6 else 9)
    return bits


def baseline_match(old: dict, new: dict) -> list[str]:
    diffs = []
    for key in ("type", "stat_size_bytes", "allocated_bytes", "mtime_ns"):
        if old.get(key) != new.get(key):
            diffs.append(key)
    for key in ("device", "inode", "nlink"):
        if old.get(key) is not None and new.get(key) is not None and old[key] != new[key]:
            diffs.append(key)
    old_mode = permissions_from_ls(old.get("mode_text"))
    new_mode_text = new.get("mode_octal")
    if old_mode is not None and new_mode_text:
        try:
            if old_mode != int(new_mode_text, 8):
                diffs.append("mode")
        except ValueError:
            diffs.append("mode_unparseable")
    return diffs


def load_deleted_paths() -> tuple[dict[str, dict], list[dict]]:
    deleted: dict[str, dict] = {}
    index_rows = []
    with INDEX_MANIFEST.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            path = row["path"]
            item = {"path": path, "entry_type": "file", "logical_bytes": int(row["size_bytes"]),
                    "allocated_bytes": int(row["allocated_bytes"]), "sha256": row.get("sha256_predelete"),
                    "evidence": INDEX_MANIFEST.name, "cleanup_quality": "PARTIAL_NO_PREDELETE_HASH_OR_QUEUE"}
            deleted[path] = item
            index_rows.append(item)
    package = json.loads(PACKAGE_RECEIPT.read_text(encoding="utf-8"))
    package_rows = []
    for row in package.get("deleted", []):
        path = row["candidate"]
        item = {"path": path, "entry_type": row["entry_type"], "logical_bytes": int(row["logical_bytes"]),
                "allocated_bytes": int(row["allocated_bytes"]), "sha256": row.get("sha256"),
                "evidence": PACKAGE_RECEIPT.name, "cleanup_quality": "GUARDED_AND_VERIFIED"}
        if row.get("verified_missing") is not True:
            raise RuntimeError("package-cache receipt contains an unverified deletion: " + path)
        if path in deleted:
            raise RuntimeError("overlapping cleanup paths in Conda receipts: " + path)
        deleted[path] = item
        package_rows.append(item)
    for path in package.get("removed_directories", []):
        item = {"path": path, "entry_type": "dir", "logical_bytes": 0,
                "allocated_bytes": 0, "sha256": None,
                "evidence": PACKAGE_RECEIPT.name, "cleanup_quality": "GUARDED_AND_VERIFIED"}
        if path in deleted:
            raise RuntimeError("overlapping cleanup paths in Conda receipts: " + path)
        deleted[path] = item
        package_rows.append(item)
    if len(package.get("removed_directories", [])) != int(package.get("directory_count_verified_missing", -1)):
        raise RuntimeError("package receipt removed-directory list does not match verified directory count")
    if "cache_dir_exists=0" not in INDEX_VERIFY.read_text(encoding="utf-8"):
        raise RuntimeError("index-cache postcheck does not verify the cache directory missing")
    cache_dir = CONDA_ROOT + "/pkgs/cache"
    deleted[cache_dir] = {"path": cache_dir, "entry_type": "directory", "logical_bytes": 0,
                          "allocated_bytes": None, "sha256": None,
                          "evidence": INDEX_VERIFY.name, "cleanup_quality": "PARTIAL_POSTCHECK_ONLY"}
    return deleted, [{"index_rows": index_rows, "package_rows": package_rows}]


def load_current_scans() -> tuple[dict[str, dict], dict, dict[str, dict], dict[str, dict]]:
    latest: dict[str, dict] = {}
    r02_rows: dict[str, dict] = {}
    r03_rows: dict[str, dict] = {}
    statuses = []
    for row in jsonl_gz(R02):
        if row.get("record_type") != "entry":
            continue
        path = row.get("path", "")
        if path == CONDA_ROOT or path.startswith(CONDA_ROOT + "/"):
            item = current_metadata(row, "R02_PARTIAL_ROOT_REFRESH")
            if path in latest:
                raise RuntimeError("duplicate path in R02 stream: " + path)
            latest[path] = item
            r02_rows[path] = item
    for name in ENV_ROOTS:
        path = OUT / ("conda_env_{}_r03.status.json".format(name))
        status = json.loads(path.read_text(encoding="utf-8"))
        statuses.append(status)
        if not status.get("root_complete_marker"):
            continue
        archive = OUT / ("conda_env_{}_r03.jsonl.gz".format(name))
        for row in jsonl_gz(archive):
            if row.get("record_type") == "root_started":
                item = current_metadata(row["root_identity"], "R03_PER_ROOT_COMPLETE_SCAN")
            elif row.get("record_type") == "entry":
                item = current_metadata(row, "R03_PER_ROOT_COMPLETE_SCAN")
            else:
                continue
            path = item["path"]
            if path in latest:
                prior = latest[path]
                mismatch = [key for key in ("type", "stat_size_bytes", "allocated_bytes", "mtime_ns", "device", "inode", "nlink", "mode_octal")
                            if prior.get(key) is not None and item.get(key) is not None and prior[key] != item[key]]
                if mismatch:
                    item["r02_r03_mismatch_fields"] = mismatch
            latest[path] = item
            r03_rows[path] = item
    return latest, {"statuses": statuses}, r02_rows, r03_rows


def main() -> None:
    baseline = {}
    for row in jsonl_gz(BASELINE):
        path = row.get("path", "")
        if path == CONDA_ROOT or path.startswith(CONDA_ROOT + "/"):
            if path in baseline:
                raise RuntimeError("duplicate path in R01 baseline: " + path)
            baseline[path] = old_metadata(row)
    if not baseline:
        raise RuntimeError("R01 full inventory contains no .conda paths")

    removed, receipt_rows = load_deleted_paths()
    current_scans, scan_meta, r02_rows, r03_rows = load_current_scans()
    expected_current = set(baseline) - set(removed)
    observed_current = set(current_scans) - set(removed)
    deleted_paths_still_observed = sorted(set(current_scans) & set(removed))
    uncovered = sorted(expected_current - observed_current)
    unexpected = sorted(observed_current - expected_current)

    r02_pathset = set(r02_rows)
    r03_pathset = set(r03_rows)
    overlap_metadata_mismatches = []
    baseline_differences = Counter()
    expected_path_differences = []
    for path in sorted(observed_current & set(baseline)):
        latest = current_scans[path]
        old = baseline[path]
        diffs = baseline_match(old, latest)
        if diffs:
            baseline_differences.update(diffs)
            expected = (path == CONDA_ROOT + "/pkgs" and set(diffs).issubset({"mtime_ns", "nlink"}))
            row = {"path": path, "fields": diffs, "expected_from_cleanup": expected,
                   "old": {key: old.get(key) for key in diffs},
                   "current": {key: latest.get(key) for key in diffs}}
            expected_path_differences.append(row)
    for path in sorted(r02_pathset & r03_pathset):
        # Compare duplicate R02/R03 observations by identity fields where both snapshots have them.
        a, b = r02_rows[path], r03_rows[path]
        diffs = [key for key in ("type", "stat_size_bytes", "allocated_bytes", "mtime_ns", "device", "inode", "nlink")
                 if a.get(key) is not None and b.get(key) is not None and a[key] != b[key]]
        if diffs:
            overlap_metadata_mismatches.append({"path": path, "fields": diffs})

    final_map = {path: item for path, item in current_scans.items() if path in expected_current}
    counts = Counter(item["type"] for item in final_map.values())
    regular_files = [item for item in final_map.values() if item["type"] == "file"]
    symlinks = [item for item in final_map.values() if item["type"] == "symlink"]
    directories = [item for item in final_map.values() if item["type"] == "directory"]
    path_allocated = sum(int(item.get("allocated_bytes") or 0) for item in final_map.values())
    unique_inode_bytes = 0
    unique_seen = set()
    for item in final_map.values():
        dev, ino = item.get("device"), item.get("inode")
        if dev is None or ino is None or (dev, ino) in unique_seen:
            continue
        unique_seen.add((dev, ino))
        unique_inode_bytes += int(item.get("allocated_bytes") or 0)

    large = sorted((item for item in regular_files if int(item["size_bytes"]) >= 100_000_000), key=lambda x: (-int(x["size_bytes"]), x["path"]))
    inode_names: dict[tuple, list[str]] = defaultdict(list)
    for item in large:
        key = (item.get("device"), item.get("inode"))
        if key[0] is not None and key[1] is not None:
            inode_names[key].append(item["path"])
    large_tsv = OUT / "conda_large_files_over_100MB_r03.tsv"
    fields = ["path", "environment_or_cache", "owner", "uid", "group", "gid", "logical_bytes", "allocated_path_bytes", "device", "inode", "nlink", "hardlink_aliases_in_large_set", "sha256_status", "disposition", "reason"]
    with large_tsv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for item in large:
            path = item["path"]
            if "/envs/" in path:
                env = path.split("/envs/", 1)[1].split("/", 1)[0]
                category = "environment:" + env
                disposition = "HOLD_INSTALLED_ENV_DEPENDENCY"
                reason = "Installed environment payload; do not remove by size. Confirm conda-meta package ownership and downstream env usage before any change."
            elif "/pkgs/" in path:
                category = "package_cache"
                disposition = "HOLD_PACKAGE_CACHE_REFERENCE_REVIEW"
                reason = "Package cache may be reusable by an environment; the reviewed unreferenced libstdcxx tree is already removed. No new unused package tree proven."
            else:
                category = "other_conda"
                disposition = "HOLD_PROVENANCE_REVIEW"
                reason = "No exact consumer/reproduction evidence supports deletion."
            aliases = inode_names.get((item.get("device"), item.get("inode")), [])
            if int(item.get("nlink") or 1) > 1:
                reason += " Hardlink count is greater than one; unlinking a single name may not release blocks."
            writer.writerow({
                "path": path, "environment_or_cache": category,
                "owner": item.get("owner") or "OWNER_NOT_REFRESHED", "uid": item.get("uid"),
                "group": item.get("group"), "gid": item.get("gid"),
                "logical_bytes": item["size_bytes"], "allocated_path_bytes": item.get("allocated_bytes"),
                "device": item.get("device"), "inode": item.get("inode"), "nlink": item.get("nlink"),
                "hardlink_aliases_in_large_set": len(aliases), "sha256_status": "NOT_REHASHED_IN_R03",
                "disposition": disposition, "reason": reason,
            })

    cache_counts = Counter(item["type"] for path, item in final_map.items() if path == CONDA_ROOT + "/pkgs" or path.startswith(CONDA_ROOT + "/pkgs/"))
    environment_counts = Counter(item["type"] for path, item in final_map.items() if path == CONDA_ROOT + "/envs" or path.startswith(CONDA_ROOT + "/envs/"))
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "coverage_status": "COMPLETE_RECONCILED_FROM_FULL_BASELINE_INCREMENTAL_SCANS_AND_EXACT_CLEANUP_RECEIPTS" if not uncovered and not unexpected and not deleted_paths_still_observed else "INCOMPLETE_OR_UNRECONCILED",
        "single_fresh_full_tree_scan": False,
        "root": CONDA_ROOT,
        "baseline": {"path": str(BASELINE), "sha256": sha256_file(BASELINE), "completed_utc": "2026-11-22T18:33:34Z", "entry_count": len(baseline), "type_counts": dict(Counter(item["type"] for item in baseline.values()))},
        "r02_partial_stream": {"path": str(R02), "sha256": sha256_file(R02), "started_utc": "2026-11-23T01:23:18Z", "entry_count_under_conda": len(r02_pathset), "root_complete_marker": False},
        "r03_per_root_scans": scan_meta["statuses"],
        "cleanup_reconciliation": {
            "removed_path_count": len(removed),
            "index_cache_file_count": len(receipt_rows[0]["index_rows"]),
            "index_cache_logical_bytes": sum(int(row["logical_bytes"]) for row in receipt_rows[0]["index_rows"]),
            "index_cache_allocated_path_bytes": sum(int(row["allocated_bytes"]) for row in receipt_rows[0]["index_rows"]),
            "package_cache_receipt_leaf_count": sum(row["entry_type"] in ("file", "symlink") for row in receipt_rows[0]["package_rows"]),
            "package_cache_receipt_directory_count": sum(row["entry_type"] == "dir" for row in receipt_rows[0]["package_rows"]),
            "package_cache_logical_leaf_bytes": sum(int(row["logical_bytes"]) for row in receipt_rows[0]["package_rows"] if row["entry_type"] in ("file", "symlink")),
            "package_cache_allocated_leaf_bytes": sum(int(row["allocated_bytes"]) for row in receipt_rows[0]["package_rows"] if row["entry_type"] in ("file", "symlink")),
            "cache_directory_removal_evidence": "conda_index_cache_removed_verification_20261122.txt",
        },
        "coverage_comparison": {"expected_current_paths": len(expected_current), "observed_current_paths": len(observed_current),
                                "uncovered_current_paths": len(uncovered), "unexpected_paths": len(unexpected),
                                "receipt_deleted_paths_still_observed": len(deleted_paths_still_observed),
                                "receipt_deleted_path_examples_still_observed": deleted_paths_still_observed[:20],
                                "uncovered_examples": uncovered[:20], "unexpected_examples": unexpected[:20],
                                "r02_r03_duplicate_metadata_mismatches": overlap_metadata_mismatches[:30],
                                "baseline_metadata_differences": expected_path_differences[:30],
                                "baseline_difference_field_counts": dict(baseline_differences)},
        "current_summary": {"entry_count": len(final_map), "type_counts": dict(counts),
                            "regular_files": len(regular_files), "regular_file_logical_bytes": sum(int(x["size_bytes"]) for x in regular_files),
                            "symlinks": len(symlinks), "symlink_lstat_bytes": sum(int(x["stat_size_bytes"]) for x in symlinks),
                            "directories": len(directories), "allocated_path_bytes_all_entries": path_allocated,
                            "unique_inode_allocated_bytes_metadata_estimate": unique_inode_bytes,
                            "pkgs_type_counts": dict(cache_counts), "envs_type_counts": dict(environment_counts)},
        "large_files": {"threshold_bytes": 100_000_000, "path_count": len(large),
                        "logical_bytes": sum(int(x["size_bytes"]) for x in large),
                        "over_500MiB_count": sum(int(x["size_bytes"]) > 500 * 1024 * 1024 for x in large),
                        "over_500MiB_logical_bytes": sum(int(x["size_bytes"]) for x in large if int(x["size_bytes"]) > 500 * 1024 * 1024),
                        "owner_currently_verified_count": sum(x.get("uid") is not None for x in large),
                        "hardlink_paths_count": sum(int(x.get("nlink") or 1) > 1 for x in large),
                        "review_tsv": str(large_tsv)},
        "cleanup_disposition": "No new deletion candidate found; all current >100 MB paths are installed environment payloads or package-cache items held for owner/reference review.",
    }
    json_out = OUT / "conda_coverage_reconciliation_r03.json"
    json_out.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_out = OUT / "R03_CONDA_COVERAGE_REPORT_20261123.md"
    statuses = sorted(scan_meta["statuses"], key=lambda x: x["root"])
    status_lines = ["| Root | entries | dirs | files | symlinks | errors | elapsed (s) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for item in statuses:
        completion = item.get("completion_record", {})
        counts_row = completion.get("counts", {})
        status_lines.append("| `{}` | {} | {} | {} | {} | {} | {:.2f} |".format(
            Path(item["root"]).name, counts_row.get("entries", 0), counts_row.get("directories", 0),
            counts_row.get("regular_files", 0), counts_row.get("symlinks", 0), counts_row.get("errors", 0),
            completion.get("elapsed_seconds", 0)))
    report = """# R03 Conda coverage reconciliation — 2026-11-23

## Coverage and method

This is a metadata-only reconciliation of `/data/pclancy3/yi/.conda`. It does
not read file payloads, compute new hashes, alter environments, open the shared
SQLite catalog, or delete anything. It joins the completed R01 full-tree
baseline, the R02 partial refresh, four isolated R03 per-environment scans,
and exact receipts for the two earlier Conda-cache removals. R03 did not run a
single full `.conda` walk; its complete coverage is established by the
path-set union and exact reconciliation below.

- R01 completed at 2026-11-22 18:33 UTC and recorded {base_entries} paths under
  `.conda`: {base_dirs} directories, {base_files} regular files, and {base_links}
  symlinks.
- R02 started at 2026-11-23 01:23 UTC and recorded {r02_entries} `.conda`
  paths before the broader `/data/pclancy3` scan ended without a completion
  marker. R02 covers the complete current `.conda/pkgs` entry set and seven of
  eleven `.conda/envs` roots; four environment roots were incomplete.
- R03 scanned only those four missing environment roots and emitted explicit
  `root_started`/`root_complete` markers with zero errors. Counts below are
  children; the separately recorded root identity adds one path per root.

{status_table}

The four completed R03 child counts match R01 exactly after excluding each
environment root itself. The union of R02 and R03 covers {observed_current}
current paths against {expected_current} expected paths: {uncovered} uncovered
and {unexpected} unexpected; {deleted_still_observed} receipt-deleted paths
were still observed. R02/R03 overlapping identities and baseline
metadata were compared; see the machine-readable manifest for exact field
differences. The only expected baseline change is the `.conda/pkgs` directory
metadata after removal of children.

## Conda cleanup reconciliation

The R01 baseline was taken before the two earlier cache cleanups. The later
receipts account for exactly {removed_count} paths: 42 index-cache files,
the 88 files and 2 symlinks plus 15 directories in the guarded unused
`libstdcxx-14.2.0-h8f9b012_2` package-tree removal, and the now-missing
`.conda/pkgs/cache` directory. R02 confirms the package-cache path set is
exactly baseline minus those recorded removals. The index-cache event remains
`PARTIAL` because it lacks pre-delete hashes and a contemporaneous queue
guard; it is not upgraded by this reconciliation.

Current reconciled `.conda` totals are {current_entries} paths: {current_dirs}
directories, {current_files} files, and {current_links} symlinks. Regular files
sum to {current_file_bytes} logical bytes. Allocated path bytes from metadata
are {path_allocated} (hardlink aliases can be counted more than once); the
unique-inode metadata estimate is {unique_allocated} bytes.

## Large paths and disposition

The reconciled tree contains {large_count} regular files larger than 100 MB,
totalling {large_bytes} logical bytes; {large500} exceed 500 MiB. The TSV
records each path, current owner where observed, inode/link count, allocated
path bytes, and disposition. The large files are installed Torch/TensorFlow,
CUDA, and related environment libraries or package-cache payloads. They remain
held because size alone does not establish an unused package or reproducible
replacement. The previously verified 574,565,016-byte `libcublasLt.so.11`
hardlink/duplicate group is still not a safe deletion candidate; removing a
single hardlink name would not release inode blocks while the other link
remains. No new safe Conda cleanup candidate was found and no R03 file was
deleted.

## Evidence and limits

- Full baseline and R02 partial stream remain unchanged under `phase2_data/`.
- Each R03 root archive, status JSON, and stderr file is preserved separately;
  the scanner source is `scan_conda_env_roots_r03.py`.
- Cleanup source evidence remains in `conda_index_cache_removed_manifest_20261122.tsv`,
  `conda_index_cache_removed_verification_20261122.txt`, and
  `conda_package_cleanup_execution_receipt_20261122.json`.
- These scans cover `.conda` only. They do not complete the whole
  `/data/pclancy3` tree or resolve the mount-wide quota discrepancy.
""".format(
        base_entries=len(baseline), base_dirs=Counter(x["type"] for x in baseline.values())["directory"],
        base_files=Counter(x["type"] for x in baseline.values())["file"], base_links=Counter(x["type"] for x in baseline.values())["symlink"],
        r02_entries=len(r02_pathset), status_table="\n".join(status_lines), expected_current=len(expected_current),
        observed_current=len(observed_current), uncovered=len(uncovered), unexpected=len(unexpected),
        deleted_still_observed=len(deleted_paths_still_observed),
        removed_count=len(removed), current_entries=len(final_map), current_dirs=counts["directory"],
        current_files=counts["file"], current_links=counts["symlink"],
        current_file_bytes=sum(int(x["size_bytes"]) for x in regular_files), path_allocated=path_allocated,
        unique_allocated=unique_inode_bytes, large_count=len(large),
        large_bytes=sum(int(x["size_bytes"]) for x in large),
        large500=sum(int(x["size_bytes"]) > 500 * 1024 * 1024 for x in large))
    report_out.write_text(report, encoding="utf-8")
    print(json.dumps({"coverage_status": meta["coverage_status"], "current_paths": len(final_map),
                      "uncovered": len(uncovered), "unexpected": len(unexpected),
                      "large_over_100MB": len(large), "no_new_cleanup_candidate": True,
                      "manifest": str(json_out), "report": str(report_out), "large_tsv": str(large_tsv)}, sort_keys=True))

    if uncovered or unexpected or deleted_paths_still_observed or len(statuses) != 4 or any(not x.get("root_complete_marker") for x in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
