#!/usr/bin/env python3
"""Capture byte-level PR #10 file reconciliation against current Git main."""

import argparse
import hashlib
import json
import os
import subprocess


PR_HEAD = "51cb06f71732e46cd10eb92cee1f0c5f8f4833c9"
PR_BASE = "817acf22af7374bcb2a79d123d54ad77a7411c1c"
EXPECTED_CURRENT_HEAD = "19708d74d14103832e4f86b51b8b5d8875175cc3"
PRESERVED_PREFIX = "paper/revision1_candidates/pr10_overleaf_candidate_20261123/source/paper/revision1/"


def git(*args):
    return subprocess.check_output(["git", *args]).decode().strip()


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def read_blob(rev, path):
    oid = git("rev-parse", "{}:{}".format(rev, path))
    payload = subprocess.check_output(["git", "show", "{}:{}".format(rev, path)])
    return oid, payload


def has_blob(rev, path):
    return subprocess.run(["git", "cat-file", "-e", "{}:{}".format(rev, path)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_json")
    args = parser.parse_args()
    current_head = git("rev-parse", "HEAD")
    if current_head != EXPECTED_CURRENT_HEAD:
        raise ValueError("capture expects canonical current main {}; got {}".format(
            EXPECTED_CURRENT_HEAD, current_head))
    subprocess.check_call(["git", "merge-base", "--is-ancestor", PR_HEAD, current_head],
                          stdout=subprocess.DEVNULL)
    paths = git("diff-tree", "--no-commit-id", "--name-only", "-r", PR_BASE, PR_HEAD).splitlines()
    if len(paths) != 47:
        raise ValueError("expected 47 PR #10 files, found {}".format(len(paths)))
    rows = []
    for source_path in paths:
        pr_oid, pr_bytes = read_blob(PR_HEAD, source_path)
        pr_sha = sha256_bytes(pr_bytes)
        current_path = source_path if has_blob(current_head, source_path) else None
        current_oid = current_sha = None
        current_bytes = None
        if current_path:
            current_oid, current_bytes = read_blob(current_head, current_path)
            current_sha = sha256_bytes(current_bytes)
        if current_bytes == pr_bytes:
            status = "SAME_PATH_IDENTICAL"
            note = "PR #10 payload is present at the same path and byte-identical on current main."
        else:
            candidate_path = None
            candidate_rel = None
            if source_path.startswith("paper/revision1/"):
                candidate_rel = source_path[len("paper/revision1/"):]
                candidate_path = PRESERVED_PREFIX + candidate_rel
            if candidate_path and has_blob(current_head, candidate_path):
                preserved_oid, preserved_bytes = read_blob(current_head, candidate_path)
                if preserved_bytes == pr_bytes:
                    current_path = candidate_path
                    current_oid = preserved_oid
                    current_bytes = preserved_bytes
                    current_sha = sha256_bytes(preserved_bytes)
                    status = "PRESERVED_CANDIDATE_IDENTICAL"
                    note = "Submitted/current paper path differs, but exact PR #10 bytes are retained under the candidate snapshot path."
                else:
                    status = "CONTENT_CHANGED_UNRESOLVED"
                    note = "Neither current path nor preserved candidate path matches PR #10 bytes."
            else:
                status = "MISSING_UNRESOLVED" if current_bytes is None else "CONTENT_CHANGED_UNRESOLVED"
                note = "PR #10 bytes were not found at the current or preserved candidate path."
        rows.append({
            "pull_request": 10,
            "pr_head_commit": PR_HEAD,
            "source_path": source_path,
            "pr_blob_oid": pr_oid,
            "pr_byte_count": len(pr_bytes),
            "pr_sha256": pr_sha,
            "current_path": current_path,
            "current_blob_oid": current_oid,
            "current_byte_count": len(current_bytes) if current_bytes is not None else None,
            "current_sha256": current_sha,
            "reconciliation_status": status,
            "reconciliation_note": note,
        })
    counts = {}
    for row in rows:
        key = row["reconciliation_status"]
        counts[key] = counts.get(key, 0) + 1
    if counts != {"SAME_PATH_IDENTICAL": 44, "PRESERVED_CANDIDATE_IDENTICAL": 3}:
        raise ValueError("unexpected PR #10 reconciliation counts: {}".format(counts))
    payload = {
        "capture_scope": "PR #10 changed file blobs compared with current MigrationBench main",
        "pr_base": PR_BASE,
        "pr_head": PR_HEAD,
        "current_main": current_head,
        "pr_is_ancestor_of_current_main": True,
        "changed_file_count": len(rows),
        "status_counts": counts,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("files={} counts={} current_main={} output={}".format(
        len(rows), counts, current_head, args.output_json))


if __name__ == "__main__":
    main()
