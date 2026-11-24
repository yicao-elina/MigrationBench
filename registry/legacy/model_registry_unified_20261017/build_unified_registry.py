#!/usr/bin/env python3
"""Build an additive, provenance-preserving MLIP registry from prior registries."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[2]
OUT = Path(__file__).parent

REGISTRIES = [
    "analysis/canonical_model_matrix_20261016/MODEL_REGISTRY_CANONICAL_CURRENT_20261017.tsv",
    "analysis/canonical_model_matrix_20261016/MODEL_REGISTRY_ACCEPTED_GATE_32114531.tsv",
    "analysis/mace_endpoint_grouped_retrain_10model/models.tsv",
    "analysis/multimodel_639_screen_20261014/MODEL_REGISTRY.tsv",
    "analysis/multimodel_639_screen_20261014/models_rockfish.tsv",
    "analysis/paper_provenance_audit_20261013/MODEL_REGISTRY.tsv",
    "analysis/elementa_20261015/MODEL_REGISTRY.tsv",
    "analysis/from_scratch_failure_attribution_20261011/mlip_candidate_campaign_20261013/MODEL_REGISTRY.tsv",
    "analysis/from_scratch_failure_attribution_20261011/mlip_candidate_campaign_20261013/models_rockfish.tsv",
    "analysis/mace_candidate_eval_20261001/models_rockfish.tsv",
]

MANIFESTS = [
    "analysis/canonical_model_matrix_20261016/EXPERIMENT_MANIFEST.json",
    "analysis/canonical_model_matrix_20261016/SCREEN_SOURCE_MANIFEST.json",
    "analysis/canonical_model_matrix_20261016/scratch_600k_grouped_v1/EXPERIMENT_MANIFEST.json",
    "analysis/canonical_model_matrix_20261016/screen_inputs_gate_32114531_v1/SCREEN_INPUT_MANIFEST.json",
    "analysis/elementa_20261015/EXPERIMENT_MANIFEST.json",
    "analysis/from_scratch_failure_attribution_20261011/mlip_candidate_campaign_20261013/RELEASE_MANIFEST.json",
    "analysis/ft_5pct_grouped_20261016/EXPERIMENT_MANIFEST.json",
    "analysis/ft_600k_random_reproduction_20261016/EXPERIMENT_MANIFEST.json",
    "analysis/multimodel_639_screen_20261014/CAMPAIGN_MANIFEST.json",
    "analysis/multimodel_639_screen_20261014/ENVIRONMENT_MANIFEST.json",
    "analysis/multimodel_639_screen_20261014/PAPER_RELEASE_MANIFEST.json",
    "analysis/scratch_lmax1_grouped_20261016/EXPERIMENT_MANIFEST.json",
]

FIELDS = [
    "canonical_model_id", "artifact_key", "artifact_kind", "model_family",
    "publication_label", "training_seed", "dataset_release", "foundation_role",
    "max_L", "correlation", "model_path", "model_sha256", "job_id",
    "computational_status", "provenance_status", "scientific_status",
    "publication_role", "source_registries", "source_rows", "notes",
]

def first(row, *names):
    for n in names:
        if row.get(n):
            return row[n]
    return ""

def artifact_key(row):
    # A path is the identity of the physical artifact.  The previous version
    # preferred SHA256 and therefore allowed a path-only alias to evade a
    # later hash-bearing row for the same file.
    path = first(row, "model_path")
    if path:
        return "path:" + path.removeprefix("env:")
    digest = first(row, "model_sha256", "compiled_sha256", "sha256")
    if digest:
        return "sha256:" + digest
    if path:
        return "path:" + path
    return "row:" + first(row, "canonical_model_id", "model_id", "canonical_id")

def normalize(row, source, line):
    path = first(row, "model_path")
    family = first(row, "model_family", "family", "lineage", "model_role")
    cid = first(row, "canonical_model_id", "canonical_id", "model_id")
    return {
        "artifact_key": artifact_key(row),
        "canonical_model_id": cid,
        "artifact_kind": "compiled_or_model_checkpoint",
        "model_family": family,
        "publication_label": first(row, "publication_label", "model_label"),
        "training_seed": first(row, "training_seed", "seed", "training_or_eval_status"),
        "dataset_release": first(row, "dataset_release"),
        "foundation_role": first(row, "foundation_role"),
        "max_L": "",
        "correlation": "",
        "model_path": path,
        "model_sha256": first(row, "model_sha256", "compiled_sha256", "sha256"),
        "job_id": first(row, "job_id"),
        "computational_status": first(row, "computational_status", "training_or_eval_status"),
        "provenance_status": first(row, "provenance_status", "lineage_status", "identity_status", "lineage"),
        "scientific_status": first(row, "scientific_status"),
        "publication_role": first(row, "publication_role", "model_role"),
        "source_registries": source,
        "source_rows": f"{source}:{line}",
        "notes": first(row, "notes", "publication_action", "evidence", "training_md_seed_note"),
    }

def merge(dst, item):
    for key, value in item.items():
        if key in ("source_registries", "source_rows", "notes"):
            if value and value not in dst[key].split(" || "):
                dst[key] = " || ".join(x for x in (dst[key], value) if x)
        elif not dst.get(key) and value:
            dst[key] = value

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    merged = {}
    raw_items = []
    source_rows = []
    for rel in REGISTRIES:
        path = ROOT / rel
        if not path.exists():
            source_rows.append([rel, "missing", "0", ""])
            continue
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        source_rows.append([rel, "read", str(len(rows)), hashlib.sha256(path.read_bytes()).hexdigest()])
        for line, row in enumerate(rows, 2):
            item = normalize(row, rel, line)
            raw_items.append(item)

    # Build connected components under the two independent exact identities:
    # normalized physical path OR non-empty SHA256.  This catches both the
    # original path-only alias bug and byte-identical copies at different paths.
    parent = list(range(len(raw_items)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    by_identity = {}
    for i, item in enumerate(raw_items):
        path = item["model_path"].removeprefix("env:")
        identities = []
        if path:
            identities.append("path:" + path)
        if item["model_sha256"]:
            identities.append("sha256:" + item["model_sha256"])
        for identity in identities:
            if identity in by_identity:
                union(i, by_identity[identity])
            else:
                by_identity[identity] = i
    components = {}
    for i, item in enumerate(raw_items):
        components.setdefault(find(i), []).append(item)
    for members in components.values():
        representative = members[0]
        path = representative["model_path"].removeprefix("env:")
        key = ("path:" + path) if path else ("sha256:" + representative["model_sha256"])
        base = dict(representative)
        base["artifact_key"] = key
        merged[key] = base
        for item in members[1:]:
            merge(merged[key], item)

    # Upgrade known architecture/protocol fields from the authoritative manifests.
    overrides = {
        "scratch_600k_grouped_v1": ("scratch_600k_grouped_v1", "2", "2"),
        "from_scratch_all_data_v1": ("from_scratch_all_data_v1", "2", "2"),
        "ft_600k_grouped_A2": ("ft_600k_grouped_A2", "1", "3"),
        "ft_multit_grouped_A2": ("ft_multit_grouped_A2", "1", "3"),
        "scratch_grouped": ("scratch_grouped", "1", "3"),
    }
    for item in merged.values():
        for token, (family, max_l, corr) in overrides.items():
            if token in (item["model_family"] or "") or token in (item["canonical_model_id"] or ""):
                item["max_L"], item["correlation"] = max_l, corr
                break
        if not item["canonical_model_id"]:
            digest = item["model_sha256"]
            item["canonical_model_id"] = "unresolved_" + (digest[:16] if digest else hashlib.sha256(item["artifact_key"].encode()).hexdigest()[:16])
        if item["training_seed"] in ("0", "NA"):
            item["training_seed"] = "NA"

    rows = sorted(merged.values(), key=lambda x: x["canonical_model_id"])
    with (OUT / "MODEL_REGISTRY_UNIFIED.tsv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    # Explicit alias map: every source identity that collapsed into a single
    # physical artifact remains inspectable and reproducible.
    with (OUT / "DUPLICATE_ARTIFACT_MAP.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["artifact_key", "canonical_model_id", "model_path", "model_sha256", "source_rows", "merge_rule"])
        for item in rows:
            aliases = item["source_rows"].split(" || ")
            if len(aliases) > 1:
                w.writerow([item["artifact_key"], item["canonical_model_id"], item["model_path"], item["model_sha256"], item["source_rows"], "same normalized physical path; SHA256 retained when available"])

    # Audit the precise failure mode that motivated this repair: old
    # hash-first identities that point to the same normalized physical path.
    by_path = {}
    for item in raw_items:
        path = item["model_path"].removeprefix("env:")
        if path:
            old_key = ("sha256:" + item["model_sha256"]) if item["model_sha256"] else ("path:" + path)
            by_path.setdefault(path, set()).add(old_key)
    with (OUT / "PATH_COLLISION_AUDIT.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["normalized_model_path", "pre_repair_identity_count", "pre_repair_identities", "canonical_model_id", "repair"])
        by_id = {x["model_path"].removeprefix("env:"): x["canonical_model_id"] for x in rows}
        for path, keys in sorted(by_path.items()):
            if len(keys) > 1:
                w.writerow([path, len(keys), " || ".join(sorted(keys)), by_id.get(path, ""), "merged path aliases; retained best available SHA256 and all provenance"])

    by_hash = {}
    for item in raw_items:
        digest = item["model_sha256"]
        if digest:
            by_hash.setdefault(digest, set()).add(item["model_path"].removeprefix("env:"))
    with (OUT / "HASH_COLLISION_AUDIT.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["model_sha256", "distinct_normalized_paths", "paths", "canonical_model_id", "repair"])
        for digest, paths in sorted(by_hash.items()):
            if len(paths) > 1:
                canonical = next((x["canonical_model_id"] for x in rows if x["model_sha256"] == digest), "")
                w.writerow([digest, len(paths), " || ".join(sorted(paths)), canonical, "merged byte-identical copies by SHA256"])

    with (OUT / "SOURCE_REGISTRY_AUDIT.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["source_registry", "status", "row_count", "sha256"])
        w.writerows(source_rows)

    with (OUT / "SOURCE_MANIFEST_AUDIT.tsv").open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["manifest", "status", "sha256", "campaign_id", "manifest_status", "summary"])
        for rel in MANIFESTS:
            p = ROOT / rel
            if not p.exists():
                w.writerow([rel, "missing", "", "", "", ""])
                continue
            data = json.loads(p.read_text())
            campaign = data.get("campaign_id", "")
            status = data.get("status", data.get("scientific_status", ""))
            summary = "candidate_count=" + str(data.get("candidate_count", "")) if "candidate_count" in data else ""
            if "model_matrix" in data:
                summary = (summary + "; " if summary else "") + "model_matrix=" + str(len(data["model_matrix"]))
            if "families" in data:
                summary = (summary + "; " if summary else "") + "families=" + str(len(data["families"]))
            w.writerow([rel, "read", hashlib.sha256(p.read_bytes()).hexdigest(), campaign, status, summary])

    meta = {
        "registry_version": "2026-10-17",
        "scope": "MLIP models relevant to the 639-candidate static screen, failure-attribution campaign, model matrix, and investigated historical comparators",
        "deduplication_key": "connected components under normalized exact model_path OR non-empty SHA256",
        "preservation_rule": "source registries and artifacts are retained; this registry is additive",
        "model_row_count": len(rows),
        "pre_dedup_source_row_count": sum(int(x[2]) for x in source_rows if x[1] == "read"),
        "path_alias_group_count": sum(1 for x in rows if len(x["source_rows"].split(" || ")) > 1),
        "pre_repair_path_collision_count": sum(1 for keys in by_path.values() if len(keys) > 1),
        "source_registry_count": len(REGISTRIES),
        "source_manifest_count": len(MANIFESTS),
        "excluded_scope": "task3 NEB-only campaign manifests are not model registries and remain in their original audit roots",
        "generated_by": "build_unified_registry.py",
    }
    (OUT / "REGISTRY_MANIFEST.json").write_text(json.dumps(meta, indent=2) + "\n")

if __name__ == "__main__":
    main()
