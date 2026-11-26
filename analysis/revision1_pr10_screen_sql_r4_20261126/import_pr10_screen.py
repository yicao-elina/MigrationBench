#!/usr/bin/env python3
"""Build an additive R4 SQLite provenance snapshot for PR10's 639-frame screen.

By default inputs are read from the pinned Git tree, not the possibly dirty
working copy. The importer does not access Rockfish or external payloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

SNAPSHOT_ID = "migrationbench_pr10_screen_r4_20261126"
CAPSULE = "data/raw_compact/revision1/from_scratch_failure_attribution_20261013"
FIGURE = "paper/revision1/figures/figS_from_scratch_failure_attribution"
DEFAULT_SOURCE_COMMIT = "19708d74d14103832e4f86b51b8b5d8875175cc3"
MANIFESTS = [
    f"{CAPSULE}/diagnostic_manifest.json",
    f"{CAPSULE}/dft_batch_selection_manifest.json",
    f"{CAPSULE}/figure_and_text_manifest.json",
    f"{CAPSULE}/SUBMISSION_RECORD.json",
    f"{CAPSULE}/full_cell_resource_gate.md",
    f"{CAPSULE}/README.md",
    f"{CAPSULE}/failure_attribution_results_discussion.md",
    "paper/revision1/figures/figS_from_scratch_failure_attribution.manifest.json",
    "paper/revision1/scripts/render_from_scratch_failure_attribution.py",
    "paper/revision1/appendix/from_scratch_mlip_candidate_selection_methods.tex",
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_bytes(repo: Path, ref: str, rel: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{ref}:{rel}"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout


def git_oid(repo: Path, ref: str, rel: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{ref}:{rel}"],
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout.strip()


def tracked_capsule_files(repo: Path, ref: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", ref, "--", CAPSULE],
        check=True, text=True, stdout=subprocess.PIPE,
    ).stdout
    return [p for p in out.splitlines() if Path(p).suffix.lower() in {".csv", ".tsv", ".json", ".md"}]


def read_tabular(raw: bytes, delimiter: str) -> tuple[list[str], list[dict[str, str]]]:
    stream = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8-sig", newline="")
    reader = csv.reader(stream, delimiter=delimiter)
    original_headers = next(reader, None)
    if not original_headers:
        raise ValueError("tabular source has no header")
    seen: dict[str, int] = {}
    headers = []
    for header in original_headers:
        seen[header] = seen.get(header, 0) + 1
        headers.append(header if seen[header] == 1 else f"{header}__duplicate_{seen[header]}")
    rows = []
    for values in reader:
        if len(values) != len(headers):
            raise ValueError(f"tabular row width {len(values)} does not match {len(headers)} columns")
        rows.append({k: (v if v is not None else "") for k, v in zip(headers, values)})
    return headers, rows


def jdump(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def integer(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def real(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def boolean(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value.lower() in {"true", "1", "yes"})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-ref", default=DEFAULT_SOURCE_COMMIT)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "snapshot.sqlite")
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = args.output.resolve()
    source_ref = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", args.source_ref], check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    created = "2026-11-25T00:00:00Z"  # fixed build metadata for deterministic rebuilds

    csv_paths = tracked_capsule_files(repo, source_ref)
    if not csv_paths:
        raise RuntimeError(f"No tracked capsule files at {source_ref}")
    all_paths = sorted(set(csv_paths + MANIFESTS + [
        f"{FIGURE}.manifest.json", f"{FIGURE}.pdf", f"{FIGURE}.png", f"{FIGURE}.svg"
    ]))
    file_data: dict[str, bytes] = {p: git_bytes(repo, source_ref, p) for p in all_paths}
    file_oids = {p: git_oid(repo, source_ref, p) for p in all_paths}

    if out.exists():
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(out)
    con.row_factory = sqlite3.Row
    schema = (Path(__file__).resolve().parent / "schema.sql").read_text()
    con.executescript(schema)
    con.execute("PRAGMA foreign_keys=ON")

    con.execute(
        "INSERT INTO release_snapshot VALUES (?,?,?,?,?,?,?)",
        (SNAPSHOT_ID, "MigrationBench", "PR10 from-scratch failure-attribution screen",
         source_ref, "r4", "639 canonical frames × 3 static MLIP checkpoints; no completed DFT claim", created),
    )
    source_ids: dict[str, int] = {}
    rows_by_path: dict[str, list[dict[str, str]]] = {}
    for rel in all_paths:
        data = file_data[rel]
        ext = Path(rel).suffix.lower()
        role = "CAMPAIGN_SOURCE" if rel.startswith(CAPSULE + "/") else "PAPER_EVIDENCE_ASSET"
        media = {".csv": "text/csv", ".tsv": "text/tab-separated-values", ".json": "application/json",
                 ".md": "text/markdown", ".py": "text/x-python", ".tex": "text/x-tex",
                 ".pdf": "application/pdf", ".png": "image/png", ".svg": "image/svg+xml"}.get(ext)
        cur = con.execute(
            "INSERT INTO source_file(snapshot_id,relative_path,sha256,byte_size,file_role,media_type,git_blob_oid) VALUES (?,?,?,?,?,?,?)",
            (SNAPSHOT_ID, rel, sha(data), len(data), role, media, file_oids[rel]),
        )
        source_ids[rel] = cur.lastrowid
        if ext in {".csv", ".tsv"}:
            _, rows = read_tabular(data, "\t" if ext == ".tsv" else ",")
            rows_by_path[rel] = rows
            con.executemany(
                "INSERT INTO source_row VALUES (?,?,?)",
                [(source_ids[rel], i + 2, jdump(row)) for i, row in enumerate(rows)],
            )

    # Register all source and figure assets under one path/hash inventory.
    asset_ids: dict[str, int] = {}
    for rel in all_paths:
        kind = "PAPER_FIGURE_OUTPUT" if rel.startswith(FIGURE + ".") and not rel.endswith(".manifest.json") else (
            "FIGURE_RENDER_MANIFEST" if rel == f"{FIGURE}.manifest.json" else "SOURCE_EVIDENCE")
        status = "DERIVED_FIGURE" if kind == "PAPER_FIGURE_OUTPUT" else "CHECKED_IN_EVIDENCE"
        cur = con.execute(
            "INSERT INTO artifact_asset(snapshot_id,logical_name,uri,sha256,byte_size,artifact_kind,scientific_status,source_file_id) VALUES (?,?,?,?,?,?,?,?)",
            (SNAPSHOT_ID, Path(rel).name, rel, sha(file_data[rel]), len(file_data[rel]), kind, status, source_ids[rel]),
        )
        asset_ids[rel] = cur.lastrowid

    model_path = f"{CAPSULE}/MODEL_REGISTRY.tsv"
    model_rows = rows_by_path[model_path]
    for i, r in enumerate(model_rows, start=2):
        con.execute(
            """INSERT INTO model_checkpoint VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (r["model_sha256"], r["canonical_model_id"], r["model_label"], r["model_family"],
             integer(r["training_seed"]), r["model_path"], r["identity_status"], r["training_md_seed_note"],
             source_ids[model_path], i),
        )

    mlip_manifest_path = f"{CAPSULE}/mlip_analysis_manifest.json"
    mlip_manifest = json.loads(file_data[mlip_manifest_path])
    figure_text_manifest = json.loads(file_data[f"{CAPSULE}/figure_and_text_manifest.json"])
    analysis_manifest_hash = figure_text_manifest["analysis_manifest_sha256"]
    protocol_id = "pr10_static_mlip_singlepoint_" + analysis_manifest_hash[:12]
    con.execute(
        "INSERT INTO evaluation_protocol VALUES (?,?,?,?,?, ?,?)",
        (protocol_id, "PR10 paired static MLIP single-point screen", "STATIC_SINGLE_POINT",
         "MANIFEST_REPORTED_VERIFIED", analysis_manifest_hash,
         "Exact protocol details are bounded by source manifests and per-row software_json; not DFT validation.",
         "NO_COMPLETED_DFT_RESULT"),
    )

    # Run-level lineage: source model is the checkpoint that generated the MD trajectory.
    run_path = f"{CAPSULE}/run_summary.csv"
    run_ids: dict[tuple[str, str], int] = {}
    for i, r in enumerate(rows_by_path[run_path], start=2):
        model_sha = r["model_sha256"]
        cur = con.execute(
            """INSERT INTO md_run(source_label,training_seed,md_velocity_seed,run_key,source_model_sha256,
               input_sha256,source_root,host,slurm_job_id,slurm_array_task_id,terminal,execution_status,
               stop_reason,onset_step,onset_time_ps,first_trigger,last_step,last_time_ps,max_force_eV_A,
               max_temperature_K,min_distance_A,source_row_file_id,source_row_number)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (r["source"], int(r["training_seed"]), int(r["md_seed"]), r["run"], model_sha, r["input_sha256"],
             r["root"], r["host"], r["slurm_job_id"], r["slurm_array_task_id"], boolean(r["terminal"]),
             r["status"], r["stop_reason"], integer(r["onset_step"]), real(r["onset_ps"]), r["trigger"],
             integer(r["last_step"]), real(r["last_time_ps"]), real(r["max_force_eV_A"]),
             real(r["max_temperature_K"]), real(r["min_distance_A"]), source_ids[run_path], i),
        )
        run_ids[(r["source"], r["training_seed"])] = cur.lastrowid

    # Candidate manifest supplies the canonical candidate IDs and source-frame link.
    candidate_path = f"{CAPSULE}/candidate_manifest_639.csv"
    category_rows = {r["candidate_id"]: r for r in rows_by_path[f"{CAPSULE}/candidate_categorization.csv"]}
    frame_ids: dict[tuple[int, int, str], int] = {}
    candidate_row_refs: dict[str, tuple[int, int]] = {}
    for i, r in enumerate(rows_by_path[candidate_path], start=2):
        run_id = run_ids[(r["source"], r["training_seed"])]
        step = int(r["step"])
        win = r["window"]
        frame_key = (run_id, step, win)
        if frame_key not in frame_ids:
            fcur = con.execute(
                """INSERT INTO trajectory_frame(run_id,step,time_ps,source_window,candidate_group,
                   source_snapshot_uri,source_snapshot_sha256,frame_sha256,source_row_file_id,source_row_number)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (run_id, step, real(r["time_ps"]), win, r["candidate_group"], r["source_snapshot"],
                 None, None, source_ids[candidate_path], i),
            )
            frame_ids[frame_key] = fcur.lastrowid
        cat = category_rows.get(r["candidate_id"], {})
        prepared_hash = r.get("local_extxyz_sha256", "")
        hash_status = "RECORDED_IN_SOURCE" if len(prepared_hash) == 64 else "UNAVAILABLE_IN_SOURCE_ROW"
        ccur = con.execute(
            """INSERT INTO candidate VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (r["candidate_id"], int(r["candidate_index"]), frame_ids[frame_key], r["candidate_group"],
             r.get("evaluation_scope") or "SCREENING_CANDIDATE", r["model_sha256"], r["prepared_extxyz"],
             prepared_hash if len(prepared_hash) == 64 else None, hash_status, r["status"], r["dft_status"],
             r.get("selection_role") or None, r.get("selection_reason") or None,
             real(cat.get("min_distance_A")),
             int(real(cat.get("min_distance_A")) is not None and real(cat.get("min_distance_A")) >= 0.8),
             source_ids[candidate_path], i),
        )
        candidate_row_refs[r["candidate_id"]] = (source_ids[candidate_path], i)
        con.execute("INSERT INTO candidate_alias VALUES (?,?,?,?,?)",
                    ("canonical", r["candidate_id"], r["candidate_id"], source_ids[candidate_path], i))

    mapping_path = f"{CAPSULE}/candidate_execution_mapping.csv"
    for i, r in enumerate(rows_by_path[mapping_path], start=2):
        cid = r["candidate_id_canonical"]
        con.execute("INSERT OR IGNORE INTO candidate_alias VALUES (?,?,?,?,?)",
                    ("execution", r["candidate_id_execution"], cid, source_ids[mapping_path], i))

    # Row-level static evaluations: evaluation checkpoint is joined by canonical
    # model ID, distinct from candidate.source_md_checkpoint_sha256.
    eval_path = f"{CAPSULE}/mlip_results/mlip_candidate_results_all.csv"
    env_ids: set[str] = set()
    eval_rows = rows_by_path[eval_path]
    for i, r in enumerate(eval_rows, start=2):
        canonical_id = r["candidate_id"]
        if canonical_id not in candidate_row_refs:
            raise RuntimeError(f"Evaluation references unknown canonical candidate {canonical_id}")
        model_id = r["model_id"]
        cp = con.execute("SELECT checkpoint_sha256 FROM model_checkpoint WHERE canonical_model_id=?", (model_id,)).fetchone()
        if not cp:
            raise RuntimeError(f"Evaluation references unknown checkpoint {model_id}")
        software = r.get("software_json", "")
        env_sha = sha(software.encode()) if software else None
        if env_sha and env_sha not in env_ids:
            con.execute("INSERT INTO software_environment VALUES (?,?,?)", (env_sha, software, "PER_EVALUATION_METADATA"))
            env_ids.add(env_sha)
        alias = r.get("execution_candidate_id") or None
        con.execute(
            """INSERT INTO model_evaluation(candidate_id,checkpoint_sha256,protocol_id,environment_sha256,
               execution_candidate_alias,input_uri,input_sha256,natoms,species_composition,pbc_json,cell_volume_A3,
               geometry_finite,min_interatomic_distance_A,evaluation_status,runtime_error,energy_eV,
               energy_per_atom_eV,forces_finite,max_force_eV_A,rms_force_eV_A,p95_force_eV_A,p99_force_eV_A,
               top_force_species,top_force_atom_index,scientific_status,source_row_file_id,source_row_number)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (canonical_id, cp[0], protocol_id, env_sha, alias, r.get("input_path"), r.get("input_sha256"),
             integer(r.get("natoms")), r.get("species"), r.get("pbc"), real(r.get("cell_volume_A3")),
             boolean(r.get("geometry_finite")), real(r.get("min_interatomic_distance_A")), r["evaluation_status"],
             r.get("runtime_error") or None, real(r.get("energy_eV")), real(r.get("energy_per_atom_eV")),
             boolean(r.get("forces_finite")), real(r.get("max_force_eV_A")), real(r.get("rms_force_eV_A")),
             real(r.get("p95_force_eV_A")), real(r.get("p99_force_eV_A")), r.get("top_force_species") or None,
             integer(r.get("top_force_atom_index")), "STATIC_MLIP_ONLY_NO_DFT_VALIDATION", source_ids[eval_path], i),
        )

    # Selection batches: initial coverage wave and the later 20-row MLIP priority set.
    dft_manifest_path = f"{CAPSULE}/dft_batch_selection_manifest.json"
    dft_manifest = json.loads(file_data[dft_manifest_path])
    con.execute("INSERT INTO selection_batch VALUES (?,?,?,?,?,?,?,?,?)",
                ("wave_01", "INITIAL_COVERAGE", "SELECTION_ONLY", 20, 20, None,
                 source_ids[dft_manifest_path], sha(file_data[dft_manifest_path]),
                 "Coverage selection is not DFT output; campaign source records PREPARED_ONLY."))
    wave_path = f"{CAPSULE}/wave_01_manifest.csv"
    for i, r in enumerate(rows_by_path[wave_path], start=2):
        cid = r["candidate_id"]
        if cid in candidate_row_refs:
            con.execute("INSERT INTO selection_member VALUES (?,?,?,?,?,?,?,?)",
                        ("wave_01", cid, integer(r.get("batch_order")), r.get("selection_role"),
                         r.get("selection_reason"), r.get("dft_status") or "NOT_REPORTED", source_ids[wave_path], i))
    con.execute("INSERT INTO selection_batch VALUES (?,?,?,?,?,?,?,?,?)",
                ("mlip_dft_priority_20", "MLIP_FORCE_DISAGREEMENT_PRIORITY", dft_manifest["status"],
                 dft_manifest["requested"], dft_manifest["n_selected"], int(bool(dft_manifest.get("no_dft_submitted"))),
                 source_ids[dft_manifest_path], sha(file_data[dft_manifest_path]), dft_manifest["boundary"]))
    select_path = f"{CAPSULE}/mlip_dft_selection.csv"
    for i, r in enumerate(rows_by_path[select_path], start=2):
        cid = r.get("candidate_id", "")
        if cid in candidate_row_refs:
            con.execute("INSERT INTO selection_member VALUES (?,?,?,?,?,?,?,?)",
                        ("mlip_dft_priority_20", cid, integer(r.get("batch_order")), r.get("selection_role"),
                         r.get("selection_basis"), r.get("dft_status") or "PREPARED_ONLY", source_ids[select_path], i))

    submission_path = f"{CAPSULE}/SUBMISSION_RECORD.json"
    resource_gate_path = f"{CAPSULE}/full_cell_resource_gate.md"
    con.execute("INSERT INTO campaign_status_evidence(scope,status_as_reported,time_scope,interpretation_boundary,source_file_id) VALUES (?,?,?,?,?)",
                ("QE_CANDIDATE_ARRAY", "SUBMITTED_PENDING", "2026-10-13 submission snapshot",
                 "Historical status string only; not a live scheduler query and not evidence of completed DFT.",
                 source_ids[submission_path]))
    con.execute("INSERT INTO campaign_status_evidence(scope,status_as_reported,time_scope,interpretation_boundary,source_file_id) VALUES (?,?,?,?,?)",
                ("FULL_CELL_QE_RETRIES", "CANCELED_AFTER_RESOURCE_PREFLIGHT", "Subsequent 2026-10-13 resource-gate record",
                 "Source report states no DFT energy or force passed the gate; no live job status inferred.",
                 source_ids[resource_gate_path]))

    # The analysis manifest referenced by the figure pack is not part of this
    # capsule; preserve its expected digest as an unresolved artifact reference.
    referenced_analysis_path = "analysis/from_scratch_failure_attribution_20261011/mlip_candidate_campaign_20261013/results/analysis_manifest.json"
    con.execute("INSERT INTO artifact_asset(snapshot_id,logical_name,uri,sha256,byte_size,artifact_kind,scientific_status,source_file_id) VALUES (?,?,?,?,?,?,?,NULL)",
                (SNAPSHOT_ID, "analysis_manifest.json", referenced_analysis_path, analysis_manifest_hash,
                 None, "REFERENCED_MANIFEST_NOT_IN_CAPSULE", "HASH_RECORDED_BUT_SOURCE_FILE_NOT_IMPORTED"))

    # Figure-level manifest plus explicit panel data links from the checked-in renderer.
    figure_manifest_path = f"{FIGURE}.manifest.json"
    fig_manifest = json.loads(file_data[figure_manifest_path])
    figure_id = "figS_from_scratch_failure_attribution"
    figure_file_id = source_ids[f"{FIGURE}.pdf"]
    # Ensure source files used by the renderer match the recorded figure manifest hashes.
    for name, expected_hash in fig_manifest["source_inputs"].items():
        rel = f"{CAPSULE}/mlip_results/{name}"
        if rel not in file_data or sha(file_data[rel]) != expected_hash:
            raise RuntimeError(f"Figure input hash mismatch for {rel}")
    con.execute("INSERT INTO figure VALUES (?,?,?,?,?,?,?,?)",
                (figure_id, figure_file_id, source_ids[figure_manifest_path], fig_manifest["status"],
                 fig_manifest["n_candidate_rows"], fig_manifest["n_successful_model_rows"], fig_manifest["n_models"],
                 fig_manifest["interpretation_boundary"]))
    panels = [
        ("a", "SCHEMATIC", "Sampling and ranking operator; conceptual schematic, not measured data.", None, "SCHEMATIC_ONLY"),
        ("b", "QUANTITATIVE_SUMMARY", "Candidate-level static maximum-force distributions by stage and checkpoint.", f"{CAPSULE}/mlip_results/mlip_candidate_results_all.csv", "STATIC_MLIP_EVALUATIONS"),
        ("c", "QUANTITATIVE_SUMMARY", "Paired force disagreement across registered checkpoints.", f"{CAPSULE}/mlip_results/candidate_model_spread.csv", "DERIVED_FROM_PAIRED_STATIC_MLIP"),
        ("d", "QUANTITATIVE_SUMMARY", "Stage-wise median maximum force by checkpoint.", f"{CAPSULE}/mlip_results/model_stage_summary.csv", "DERIVED_STATIC_MLIP_SUMMARY"),
    ]
    for panel, kind, desc, src, role in panels:
        con.execute("INSERT INTO figure_panel VALUES (?,?,?,?,?,?)",
                    (figure_id, panel, kind, desc, source_ids[src] if src else None, role))
    for fmt in ("pdf", "png", "svg"):
        rel = f"{FIGURE}.{fmt}"
        con.execute("INSERT INTO figure_output VALUES (?,?,?)", (figure_id, fmt, asset_ids[rel]))

    # Claims are retained with their source row and verbatim evidence refs.
    claims_path = f"{CAPSULE}/CLAIM_EVIDENCE_MATRIX.tsv"
    claims = rows_by_path[claims_path]
    basename_lookup: dict[str, list[str]] = {}
    for p in all_paths:
        basename_lookup.setdefault(Path(p).name, []).append(p)
    for i, r in enumerate(claims, start=2):
        con.execute("INSERT INTO claim VALUES (?,?,?,?,?,?,?)",
                    (r["claim_id"], r["claim"], r["status"], r["boundary_or_gap"],
                     "NOT_SPECIFIED_IN_CAPSULE", source_ids[claims_path], i))
        refs = [x.strip() for x in r["evidence"].split(";") if x.strip()]
        for ref in refs:
            normalized = ref.replace("\\", "/").removeprefix("./")
            resolved = normalized if normalized in source_ids else None
            status = "RESOLVED_EXACT_PATH" if resolved else None
            if not resolved:
                candidates = basename_lookup.get(Path(normalized).name, [])
                if len(candidates) == 1:
                    resolved = candidates[0]
                    status = "RESOLVED_UNIQUE_BASENAME"
                elif len(candidates) > 1:
                    status = "AMBIGUOUS_BASENAME"
                else:
                    status = "UNRESOLVED_OR_EXTERNAL"
            con.execute("INSERT INTO claim_evidence VALUES (?,?,?,?,?)",
                        (r["claim_id"], ref, source_ids[resolved] if resolved else None,
                         "CLAIM_SUPPORT_REFERENCE", status))
    panel_claims = {
        "C001": ["b", "c", "d"], "C003": ["b", "d"], "C004": ["c"], "C013": ["a", "b", "c", "d"]
    }
    for claim_id, pids in panel_claims.items():
        for pid in pids:
            con.execute("INSERT INTO claim_figure_panel VALUES (?,?,?,?)",
                        (claim_id, figure_id, pid, "PANEL_SUPPORT; verify exact claim wording against panel scope"))

    con.commit()
    fk = con.execute("PRAGMA foreign_key_check").fetchall()
    quick = con.execute("PRAGMA quick_check").fetchone()[0]
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in (
        "source_file", "source_row", "md_run", "trajectory_frame", "candidate", "candidate_alias",
        "model_checkpoint", "model_evaluation", "selection_member", "figure_panel", "claim")}
    expected = {"md_run": 9, "candidate": 639, "model_checkpoint": 3, "model_evaluation": 1917,
                "figure_panel": 4, "claim": len(claims)}
    failures = {k: (counts[k], v) for k, v in expected.items() if counts[k] != v}
    matrix = con.execute("SELECT COUNT(*) n, COUNT(DISTINCT candidate_id) nc, COUNT(DISTINCT checkpoint_sha256) nm, SUM(evaluation_status='success') ns FROM model_evaluation").fetchone()
    input_mismatch = con.execute("""SELECT COUNT(*) FROM model_evaluation e JOIN candidate c USING(candidate_id)
      WHERE c.prepared_structure_sha256 IS NOT NULL AND e.input_sha256 <> c.prepared_structure_sha256""").fetchone()[0]
    rows_per_candidate_bad = con.execute("SELECT COUNT(*) FROM (SELECT candidate_id FROM model_evaluation GROUP BY candidate_id HAVING COUNT(*)<>3)").fetchone()[0]
    claim_resolution_counts = {r[0]: r[1] for r in con.execute(
        "SELECT resolution_status,COUNT(*) FROM claim_evidence GROUP BY resolution_status")}
    if quick != "ok" or fk or failures or tuple(matrix) != (1917, 639, 3, 1917) or input_mismatch or rows_per_candidate_bad:
        raise RuntimeError(f"Integrity failure: quick_check={quick}, fk={len(fk)}, counts={failures}, matrix={tuple(matrix)}, input_mismatch={input_mismatch}, incomplete_candidates={rows_per_candidate_bad}")
    con.close()
    manifest = {
        "snapshot_id": SNAPSHOT_ID, "schema_version": "r4", "source_ref": args.source_ref,
        "source_commit": source_ref, "database": out.name, "database_sha256": sha(out.read_bytes()),
        "schema_sha256": sha((Path(__file__).resolve().parent / "schema.sql").read_bytes()),
        "importer_sha256": sha(Path(__file__).resolve().read_bytes()),
        "queries_sha256": sha((Path(__file__).resolve().parent / "queries.sql").read_bytes()),
        "integrity": {"quick_check": quick, "foreign_key_violations": len(fk)},
        "row_counts": counts, "expected_counts": expected,
        "scope_boundary": "Static MLIP screen and source MD metadata only; DFT remains unvalidated; no DUAL-X gain asserted.",
        "matrix_checks": {"rows": matrix["n"], "candidates": matrix["nc"], "checkpoints": matrix["nm"],
                          "successful_rows": matrix["ns"], "input_hash_mismatches": input_mismatch,
                          "candidates_missing_three_model_rows": rows_per_candidate_bad},
        "claim_evidence_resolution_counts": claim_resolution_counts,
        "source_mode": "git_show_from_pinned_ref", "external_payloads_verified": False,
    }
    (out.parent / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
