#!/usr/bin/env python3
"""Render the Cr three-path fixed/self-relaxed 2x3 figure.

This figure is deliberately provenance-aware.  Current Foundation profiles and
current A2 FT self-relaxed profiles are kept separate from the legacy canonical
matrix used where the current Scratch three-path raw profiles are unavailable.
Non-converged current FT profiles are never plotted as accepted evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "three_path_r02"
SRC = WORK / "source_tables_remote"
FT = WORK / "ft_self_remote"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

PATHS = [
    ("in_gap", "in-gap diffusion", "cr_canonical_in_gap"),
    ("gap_to_ql", "gap→QL", "cr_canonical_gap_to_ql"),
    ("within_ql", "within-QL", "cr_canonical_within_ql"),
]
PATH_CLASS = {k: label for k, label, _ in PATHS}
PATH_ID_TO_KEY = {pid: key for key, _, pid in PATHS}

COLORS = {
    "foundation": "#CF4520",
    "scratch_legacy": "#FF9F1B",
    "ft_600k": "#002E71",
    "ft_multit": "#67ACE5",
    "dft": "#222222",
}
ORDER = ["foundation", "scratch_legacy", "ft_600k", "ft_multit"]
LABELS = {
    "foundation": "Foundation",
    "scratch_legacy": "Scratch (legacy From-scratch)",
    "ft_600k": "FT-600K",
    "ft_multit": "FT-MultiT",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def add_record(records: list[dict], *, path_key: str, observable: str, group: str,
               model_id: str, image_index: int, energy: float, source: str,
               status: str, publication_status: str, boundary: str) -> None:
    records.append({
        "path_key": path_key,
        "path_class": PATH_CLASS[path_key],
        "observable": observable,
        "group": group,
        "model_id": model_id,
        "image_index": int(image_index),
        "image_fraction": int(image_index) / 9.0,
        "relative_energy_eV": float(energy),
        "scientific_status": status,
        "publication_status": publication_status,
        "source": source,
        "boundary": boundary,
    })


def load_current_foundation() -> list[dict]:
    records: list[dict] = []
    self_path = SRC / "FOUNDATION_SELF_IMAGE_PROFILES.tsv"
    for r in read_tsv(self_path):
        if r["element"] != "Cr":
            continue
        key = r["path_class"]
        if key not in PATH_CLASS:
            continue
        add_record(
            records,
            path_key=key,
            observable="SELF_RELAXED_ENDPOINT_ONLY_NEB",
            group="foundation",
            model_id=r["model_id"],
            image_index=int(r["image_index"]),
            energy=float(r["rel_energy_eV"]),
            source=str(self_path),
            status=r["scientific_status"],
            publication_status=r["publication_status"],
            boundary="CURRENT_FOUNDATION_SELF_R02; endpoint-only MLIP NEB proxy",
        )
    return records


def load_legacy_scratch_and_ft() -> list[dict]:
    path = SRC / "LEGACY_TIER1_IMAGE_PROFILES.tsv"
    records: list[dict] = []
    for r in read_tsv(path):
        pid = r["path_id"]
        key = PATH_ID_TO_KEY.get(pid)
        if key is None:
            continue
        model = r["model_id"]
        if model.startswith("foundation_"):
            group = "foundation"
        elif model.startswith("from_scratch_"):
            group = "scratch_legacy"
        elif model.startswith("ft600k_"):
            group = "ft_600k"
        elif model.startswith("ftmultit_"):
            group = "ft_multit"
        else:
            continue
        obs = r["observable"]
        if obs not in {"FIXED_PATH_SINGLE_POINT", "SELF_RELAXED_ENDPOINT_ONLY_NEB"}:
            continue
        # Legacy FT self-relaxed profiles are not used as a substitute for the
        # current A2 FT self campaign.  Keep legacy FT only for the fixed row.
        if obs == "SELF_RELAXED_ENDPOINT_ONLY_NEB" and group in {"foundation", "ft_600k", "ft_multit"}:
            continue
        add_record(
            records,
            path_key=key,
            observable=obs,
            group=group,
            model_id=model,
            image_index=int(r["image_index"]),
            energy=float(r["relative_energy_eV"]),
            source=str(path),
            status="LEGACY_TIER1_PROFILE",
            publication_status="LEGACY_SUPPORTING_EVIDENCE",
            boundary="LEGACY_CANONICAL_MATRIX_20261019; model hashes predate current A2/Scratch Lmax1",
        )
    return records


def load_current_ft_self() -> list[dict]:
    records: list[dict] = []
    accepted: set[tuple[str, str]] = set()
    for collector in [FT / "r02" / "FT_SELF_R02_COLLECT.tsv", FT / "r03" / "FT_SELF_R03_RETRY_COLLECT.tsv"]:
        if not collector.exists():
            continue
        for row in read_tsv(collector):
            if row.get("scientific_status") == "CONVERGED" and row.get("publication_status") == "PRIMARY_EVIDENCE":
                accepted.add((row["system"], row["model_id"]))
    manifests = sorted(FT.rglob("endpoint_only_manifest.json"))
    for manifest in manifests:
        d = json.loads(manifest.read_text())
        engine = d.get("engine", {})
        model_id = manifest.parent.parent.name
        system = manifest.parent.parent.parent.name
        key = {"in_gap": "in_gap", "gap_to_ql": "gap_to_ql", "within_ql": "within_ql"}.get(system)
        if key is None:
            continue
        accepted_here = (system, model_id) in accepted
        if d.get("campaign_id") == "ft_interlayer_sliding_self_20261025_r03_retry":
            accepted_here = False
            status = "FAILURE_MODE"
            pub = "DIAGNOSTIC_ONLY"
            boundary = "CURRENT_A2_FT_SELF_R03_RETRY; non-converged residual/scientific gate"
        else:
            status = d.get("scientific_status", "UNKNOWN")
            pub = d.get("publication_status", "UNKNOWN")
            boundary = "CURRENT_A2_FT_SELF_R02; endpoint-only MLIP NEB proxy"
        if not accepted_here:
            continue
        group = "ft_600k" if model_id.startswith("ft_600k_") else "ft_multit"
        for p in engine.get("profile", []):
            add_record(
                records,
                path_key=key,
                observable="SELF_RELAXED_ENDPOINT_ONLY_NEB",
                group=group,
                model_id=model_id,
                image_index=int(p["image_index"]),
                energy=float(p["rel_energy_eV"]),
                source=str(manifest),
                status=status,
                publication_status=pub,
                boundary=boundary,
            )
    return records


def _dft_reference_sources() -> list[Path]:
    """QE neb.out files behind dft_reference(), hashed into the render manifest."""
    for parent in Path(__file__).resolve().parents:
        raw = parent / "data" / "raw_compact" / "cr81_dft_neb_reference_20261013" / "raw"
        if raw.is_dir():
            return sorted(raw.glob("2L-Cu__neb-*/neb.out"))
    raise FileNotFoundError("data/raw_compact/cr81_dft_neb_reference_20261013/raw not found")


def dft_reference() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """DFT-NEB reference profiles parsed from the committed QE neb.out files.

    Formerly hard-coded arrays copied from foundation_self_relaxed_4x3_R03.pdf;
    now read via src/migrationbench/dft_neb_reference.py (values identical to
    1e-6 eV, see tests/test_dft_neb_reference.py). QE status: not converged by
    the 0.03 eV/A force gate (COMPLETED_BUT_SCF_WARNING).
    """
    import importlib.util

    for parent in Path(__file__).resolve().parents:
        module_path = parent / "src" / "migrationbench" / "dft_neb_reference.py"
        if module_path.is_file():
            spec = importlib.util.spec_from_file_location("dft_neb_reference", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.profiles_for_plot()
    raise FileNotFoundError("src/migrationbench/dft_neb_reference.py not found above " + str(Path(__file__)))


def grouped(records: list[dict], observable: str, path_key: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    bins: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r["observable"] == observable and r["path_key"] == path_key:
            bins[r["group"]][r["image_index"]].append(r["relative_energy_eV"])
    out = {}
    for group, points in bins.items():
        xs = np.array(sorted(points), dtype=float) / 9.0
        vals = np.array([points[int(i)] for i in sorted(points)], dtype=float)
        out[group] = (xs, vals.mean(axis=1), vals.std(axis=1, ddof=1) if vals.shape[1] > 1 else np.zeros(len(xs)), vals.shape[1])
    return out


def write_tables(records: list[dict]) -> tuple[Path, Path]:
    summary = OUT / "cr_three_path_plotted_profiles_R02.tsv"
    with summary.open("w", newline="") as f:
        fields = ["observable", "path_key", "path_class", "group", "n_models", "n_profile_rows", "scientific_statuses", "publication_statuses", "source_boundary"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for obs in ["FIXED_PATH_SINGLE_POINT", "SELF_RELAXED_ENDPOINT_ONLY_NEB"]:
            for key, label, _ in PATHS:
                for group in ORDER:
                    rows = [r for r in records if r["observable"] == obs and r["path_key"] == key and r["group"] == group]
                    if not rows:
                        continue
                    w.writerow({
                        "observable": obs,
                        "path_key": key,
                        "path_class": label,
                        "group": group,
                        "n_models": len({r["model_id"] for r in rows}),
                        "n_profile_rows": len(rows),
                        "scientific_statuses": ";".join(sorted({r["scientific_status"] for r in rows})),
                        "publication_statuses": ";".join(sorted({r["publication_status"] for r in rows})),
                        "source_boundary": ";".join(sorted({r["boundary"] for r in rows})),
                    })
    coverage = OUT / "cr_three_path_coverage_audit_R02.tsv"
    with coverage.open("w", newline="") as f:
        fields = ["observable", "path_class", "group", "n_models", "status"]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for obs in ["FIXED_PATH_SINGLE_POINT", "SELF_RELAXED_ENDPOINT_ONLY_NEB"]:
            for key, label, _ in PATHS:
                for group in ORDER:
                    rows = [r for r in records if r["observable"] == obs and r["path_key"] == key and r["group"] == group]
                    if rows:
                        status = "PLOTTED"
                        n = len({r["model_id"] for r in rows})
                    else:
                        status = "MISSING_OR_EXCLUDED"
                        n = 0
                    w.writerow({"observable": obs, "path_class": label, "group": group, "n_models": n, "status": status})
    return summary, coverage


def draw_panel(ax, records: list[dict], obs: str, path_key: str, letter: str, dft: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    x_dft, y_dft = dft[path_key]
    ax.plot(x_dft, y_dft, color=COLORS["dft"], lw=1.55, ls=(0, (4, 2)), marker="D", ms=3.6, mfc="white", mec=COLORS["dft"], mew=0.8, zorder=5)
    data = grouped(records, obs, path_key)
    values = list(y_dft)
    for group in ORDER:
        if group not in data:
            continue
        x, y, sd, _ = data[group]
        c = COLORS[group]
        ax.fill_between(x, y - sd, y + sd, color=c, alpha=0.20, lw=0, zorder=2)
        ax.plot(x, y, color=c, lw=2.0, marker="o", ms=4.3, mfc="white", mec=c, mew=0.95, zorder=5)
        values.extend((y - sd).tolist()); values.extend((y + sd).tolist())
    lo, hi = min(values), max(values)
    pad = max(0.18, (hi - lo) * 0.12)
    ax.set_xlim(0, 1)
    ax.set_ylim(np.floor((lo - pad) * 2) / 2, np.ceil((hi + pad) * 2) / 2)
    ax.set_xticks([0, .25, .5, .75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_title(PATH_CLASS[path_key], fontsize=12.3, fontweight="bold", pad=8)
    ax.text(-0.14, 1.04, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    ax.set_xlabel("Reaction Coordinate", fontsize=10.8, fontweight="bold", labelpad=6)
    ax.set_ylabel("Relative Energy (eV)", fontsize=10.8, fontweight="bold", labelpad=6)
    ax.minorticks_on()
    ax.tick_params(which="major", direction="out", length=4.5, width=1.0)
    ax.tick_params(which="minor", direction="out", length=2.5, width=.8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.05); spine.set_color("#111111")
    ax.grid(False)


def main() -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 10.0, "pdf.fonttype": 42, "ps.fonttype": 42})
    records = load_current_foundation() + load_legacy_scratch_and_ft() + load_current_ft_self()
    summary, coverage = write_tables(records)
    dft = dft_reference()
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.5), sharex=False, sharey=False)
    for j, (key, _, _) in enumerate(PATHS):
        draw_panel(axes[0, j], records, "FIXED_PATH_SINGLE_POINT", key, chr(ord("a") + j), dft)
        draw_panel(axes[1, j], records, "SELF_RELAXED_ENDPOINT_ONLY_NEB", key, chr(ord("d") + j), dft)
    axes[0, 0].text(-0.27, 0.5, "Fixed path", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="center", fontsize=12.5, fontweight="bold")
    axes[1, 0].text(-0.27, 0.5, "Self-relaxed", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="center", fontsize=12.5, fontweight="bold")
    handles = [Line2D([0], [0], color=COLORS["dft"], lw=1.55, ls=(0, (4, 2)), marker="D", ms=3.6, mfc="white", mec=COLORS["dft"], label="DFT reference")]
    for group in ORDER:
        handles.append(Line2D([0], [0], color=COLORS[group], lw=2, marker="o", mfc="white", label=LABELS[group]))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .985), ncol=3, frameon=False, fontsize=8.7, handlelength=2.2, handletextpad=.45, columnspacing=1.1)
    fig.subplots_adjust(left=.105, right=.985, bottom=.105, top=.80, wspace=.33, hspace=.42)
    stem = OUT / "cr_three_path_fixed_vs_self_2x3_R02"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    source_paths = [
        SRC / "FOUNDATION_FIXED_IMAGE_PROFILES.tsv",
        SRC / "FOUNDATION_SELF_IMAGE_PROFILES.tsv",
        SRC / "LEGACY_TIER1_IMAGE_PROFILES.tsv",
        FT / "r02" / "FT_SELF_R02_COLLECT.tsv",
        FT / "r03" / "FT_SELF_R03_RETRY_COLLECT.tsv",
        *_dft_reference_sources(),
    ]
    manifest = {
        "figure": str(stem.with_suffix(".pdf")),
        "columns": [label for _, label, _ in PATHS],
        "rows": ["FIXED_PATH_SINGLE_POINT", "SELF_RELAXED_ENDPOINT_ONLY_NEB"],
        "campaign_boundary": "Endpoint-only MLIP NEB proxy for self-relaxed rows; not a DFT MEP barrier.",
        "dft_reference": "DFT-NEB profiles parsed from QE neb.out (data/raw_compact/cr81_dft_neb_reference_20261013) via src/migrationbench/dft_neb_reference.py; values are retained as reference only.",
        "current_ft_self": "R02 accepted 15/18; R03 retry 0/3 accepted. Non-converged R03 profiles excluded from plotted evidence.",
        "fixed_boundary": "All fixed-row model curves use the legacy canonical three-path matrix; the current single-model Foundation fixed diagnostic was retained in source staging but excluded from the main plot because its within-QL profile has a 43 eV finite-energy excursion.",
        "scratch_boundary": "Current Scratch Lmax1 three-path raw profiles were not found; legacy From-scratch canonical profiles are plotted as a single orange Scratch legacy line, not relabeled as current Scratch-600K/MultiT.",
        "sources": {str(p): sha256(p) for p in source_paths if p.exists()},
        "tables": {"plotted_profiles": str(summary), "coverage": str(coverage)},
        "colors": COLORS,
        "record_counts": {"all_loaded": len(records), "current_ft_self_accepted": len(load_current_ft_self())},
    }
    out_manifest = OUT / "cr_three_path_fixed_vs_self_2x3_R02.manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
