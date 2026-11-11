#!/usr/bin/env python3
"""Response Table R2 (tab:resp-controlled-arms): grouped route comparison + overlap audit.

Route block
  * FT-600K, Scratch-600K, FT-MultiT, Scratch-MultiT: train/test frame counts, seed count and
    test E/F RMSE (mean +/- sample SD) from the current-release parity summary.
  * Scratch-5% and MPA0: test RMSE per seed from the grouped A2 seed comparison; frame counts from
    the zero-force training audit (Scratch-5% train 89; grouped test 267).
  * G1 |dE|: pathway_1_4_abs_error_eV from the grouped A2 seed comparison.
Overlap block: the two SOAP/RMSD leakage audits over the 25 legacy NEB images.
Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import Evidence, fmt_int, fmt_num, fmt_pm, fmt_sci  # noqa: E402

A2 = "paper/response_to_reviewers/reviewer_a2_grouped_retraining/data"
SEEDS_CSV = f"{A2}/grouped_seed_comparison.csv"
OVERLAP = {"Full grouped 600~K": f"{A2}/overlap_full_leakage_audit.json",
           "Scratch-5\\% subset": f"{A2}/overlap_scratch5_leakage_audit.json"}
CURRENT = "paper/revision1/data_processed/current_release"
PARITY = f"{CURRENT}/parity_summary.csv"
FRAMES = f"{CURRENT}/zero_force_training_summary.json"

# (row label, parity_summary model or None, grouped_seed_comparison family or None,
#  zero-force datasets (train, test) used when the route is not in the parity summary)
ROUTES = [
    ("FT-600K", "FT-600K", "OMAT-FT", None),
    ("Scratch-600K", "Scratch-600K", "Scratch", None),
    ("Scratch-5\\%", None, "Scratch-5%", ("grouped_scratch5", "grouped_scratch5_test")),
    ("FT-MultiT", "FT-MultiT", None, None),
    ("Scratch-MultiT", "Scratch-MultiT", None, None),
    ("MPA0 (zero-shot)", None, "MPA0", (None, "grouped_FT600K_test")),
]
NOT_APPLICABLE = "---"
OVERLAP_HEADER = ["Training subset", "Train frames", "Exact matches$^a$", "Min. SOAP cosine",
                  "RMSD candidates", "Min. Cr-local RMSD"]
G1_DECIMALS, RMSE_DECIMALS, SOAP_DIGITS, RMSD_DECIMALS = 2, 0, 2, 2
CROSS_CHECK_TOLERANCE = 0.05  # grouped_seed_comparison stores RMSE to 0.1 meV


def half_up(value: float, decimals: int) -> str:
    """Round half away from zero (279 for 278.5), as the submitted table does."""
    quantum = Decimal(1).scaleb(-decimals)
    return str(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def pm(values: list[float], decimals: int) -> str:
    return f"${half_up(statistics.mean(values), decimals)}\\pm{half_up(statistics.stdev(values), decimals)}$"


def read(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def route_metrics(parity: list[dict], seeds: list[dict], frames: dict, route: tuple) -> dict:
    _, model, family, datasets = route
    if model is not None:
        test = [r for r in parity if r["model"] == model and r["split"] == "test"]
        train = {r["n_frames"] for r in parity if r["model"] == model and r["split"] == "train"}
        return dict(train=frames_cell(train), test={r["n_frames"] for r in test}, seeds=len(test),
                    energy=[float(r["energy_rmse_meV_per_atom"]) for r in test],
                    force=[float(r["force_rmse_meV_A"]) for r in test])
    rows = [r for r in seeds if r["family"] == family]
    train_set, test_set = datasets
    return dict(train=fmt_int(frames[train_set]) if train_set else NOT_APPLICABLE,
                test={str(frames[test_set])}, seeds=len(rows),
                energy=[float(r["energy_rmse_mev_atom"]) for r in rows],
                force=[float(r["force_rmse_mev_A"]) for r in rows])


def frames_cell(values: set) -> str:
    if len(values) != 1:
        raise ValueError(f"train frame count differs across seeds: {values}")
    return fmt_int(values.pop())


def rmse_cell(values: list[float]) -> str:
    return pm(values, RMSE_DECIMALS) if len(values) > 1 else half_up(values[0], RMSE_DECIMALS)


def g1_cell(seeds: list[dict], family: str | None) -> str:
    if family is None:
        return NOT_APPLICABLE
    errors = [float(r["pathway_1_4_abs_error_eV"]) for r in seeds if r["family"] == family]
    return fmt_pm(statistics.mean(errors), statistics.stdev(errors), G1_DECIMALS) if len(errors) > 1 \
        else fmt_num(errors[0], G1_DECIMALS)


def cross_check(parity: list[dict], seeds: list[dict]) -> list[str]:
    """Parity-summary RMSE per seed vs the grouped seed comparison for the routes in both."""
    problems = []
    for _, model, family, _ in ROUTES:
        if model is None or family is None:
            continue
        for row in (r for r in seeds if r["family"] == family):
            match = next(p for p in parity if p["model"] == model and p["split"] == "test" and p["seed"] == row["seed"])
            for parity_key, seed_key in (("energy_rmse_meV_per_atom", "energy_rmse_mev_atom"),
                                         ("force_rmse_meV_A", "force_rmse_mev_A")):
                if abs(float(match[parity_key]) - float(row[seed_key])) > CROSS_CHECK_TOLERANCE:
                    problems.append(f"{model} seed {row['seed']} {parity_key}: {match[parity_key]} vs {row[seed_key]}")
    return problems


def overlap_row(ev: Evidence, label: str, path: str, frames: dict) -> tuple[list[str], list[str]]:
    train = json.loads(ev.input(path).read_text())["summaries"]["train"]
    exact = train["exact_global_fingerprint_matches"] + train["exact_local_fingerprint_matches"]
    row = [label, fmt_int(train["total_frames"]), str(exact), fmt_sci(train["soap_cosine_distance"]["min"], SOAP_DIGITS),
           str(train["local_rmsd_comparable_frames"]),
           f"{fmt_num(train['cr_local_species_rmsd_A']['min'], RMSD_DECIMALS)}~\\AA"]
    return row, [] if train["total_frames"] in frames.values() else [f"{label}: {train['total_frames']} frames"]


SCRATCH5_FORCE_CELL = "row 3 col 6"
GAP_NOTE = ("grouped_seed_comparison.csv and accepted_models.tsv store the Scratch-5% per-seed test force RMSE "
            "to 0.1 meV/A only; their mean rounds below the submitted value. No higher-precision Scratch-5% "
            "evaluation exists in the repo (searched parity_summary.csv, parity_model_registry.tsv, "
            "MODEL_REGISTRY.tsv, sources/rockfish_models.tsv, analysis/**, shap per-frame tables). The per-seed "
            "MACE run logs are on Rockfish only (retrain_matrix_scratch16_20260927/scratch5_grouped_seed*/*_run-*.log).")


def explain_gaps(ev: Evidence, seeds: list[dict]) -> None:
    errors = ev.checks[-1]["details"]
    if any(SCRATCH5_FORCE_CELL in e for e in errors):
        values = [float(r["force_rmse_mev_A"]) for r in seeds if r["family"] == "Scratch-5%"]
        ev.check("DATA GAP: Scratch-5% test F RMSE not derivable at submitted precision", False,
                 {"per_seed": values, "mean": statistics.mean(values), "note": GAP_NOTE})


def main() -> int:
    ev = Evidence(__file__, "tab:resp-controlled-arms")
    parity = read(ev.input(PARITY))
    seeds = read(ev.input(SEEDS_CSV))
    frames = {f["dataset"]: f["frames"] for f in json.loads(ev.input(FRAMES).read_text())["files"]}

    rows, test_problems = [], []
    for route in ROUTES:
        metrics = route_metrics(parity, seeds, frames, route)
        if len(metrics["test"]) != 1:
            test_problems.append(f"{route[0]}: test frames {metrics['test']}")
        rows.append([route[0], f"{metrics['train']} ({metrics['seeds']})", fmt_int(metrics["test"].pop()),
                     g1_cell(seeds, route[2]), rmse_cell(metrics["energy"]), rmse_cell(metrics["force"])])
    rows.append(OVERLAP_HEADER)
    frame_problems = []
    for label, path in OVERLAP.items():
        row, problems = overlap_row(ev, label, path, frames)
        rows.append(row)
        frame_problems += problems
    ev.check("test frame count identical across seeds of each route", not test_problems, test_problems or "ok")
    ev.check("parity_summary test RMSE equals grouped_seed_comparison per seed (FT-600K, Scratch-600K)",
             not (problems := cross_check(parity, seeds)), problems or f"agree within {CROSS_CHECK_TOLERANCE}")
    ev.check("overlap-audit train frame counts appear in the zero-force training audit", not frame_problems,
             frame_problems or "ok")
    if not ev.compare_table(rows, "response_to_reviewers.tex"):
        explain_gaps(ev, seeds)
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
