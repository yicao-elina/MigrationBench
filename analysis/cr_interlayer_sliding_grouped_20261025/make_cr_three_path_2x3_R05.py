#!/usr/bin/env python3
"""Render the updated Cr three-path fixed/self-relaxed 2x3 figure.

R05 preserves the R04 data boundary and keeps the global +2 pt typography pass,
while unifying the y-axis label and aligning the row labels by their right edge:
axis titles and panel titles are enlarged, tick numbers are enlarged and bold,
and the shared legend matches the subplot x-axis-title size and is bold.

R03 added the isolated canonical Scratch-600K-Lmax1 campaign to the R02
three-path profile figure.  The historical ``From-scratch`` rows are mapped to
the page-13 label ``Scratch-MultiT`` because their launcher identifies
``Scratch_Multi_T.model``.  The two non-converged Scratch-600K self-relaxed
jobs remain in the source campaign but are excluded from the plotted primary
profile, as required by the campaign gate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tarfile
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
SCRATCH = ROOT.parent / "task3_endpoint_only_neb" / "scratch_600k_migration_20261101"
SCRATCH_RAW = SCRATCH / "raw" / "rockfish" / "scratch_600k_migration_20261101"
OUT = Path(os.environ.get("CR_FIGURE_OUT", str(ROOT / "results")))
OUT.mkdir(parents=True, exist_ok=True)

# The order is intentional: it matches the page-13 legend supplied by the user.
GROUP_ORDER = ["foundation", "ft_600k", "ft_multit", "scratch_600k", "scratch_multit"]
GROUP_LABELS = {
    "foundation": "Foundation",
    "ft_600k": "FT-600K",
    "ft_multit": "FT-MultiT",
    "scratch_600k": "Scratch-600K",
    "scratch_multit": "Scratch-MultiT",
}
COLORS = {
    "foundation": "#CF4520",
    "ft_600k": "#002E71",
    "ft_multit": "#67ACE5",
    "scratch_600k": "#FF9F1B",
    "scratch_multit": "#F4C95D",
    "dft": "#222222",
}

PATHS = [
    ("in_gap", "in-gap diffusion", "cr_canonical_in_gap"),
    ("gap_to_ql", "gap→QL", "cr_canonical_gap_to_ql"),
    ("within_ql", "within-QL", "cr_canonical_within_ql"),
]
PATH_CLASS = {key: label for key, label, _ in PATHS}
PATH_ID_TO_KEY = {path_id: key for key, _, path_id in PATHS}

# R05 typography override requested by the user.  These values retain the
# profile-ensemble grammar while making the final-size labels publication-
# readable in the slide and paper placements.
BASE_FONT_SIZE = 12.0
TICK_FONT_SIZE = 12.0
AXIS_LABEL_FONT_SIZE = 12.8
TITLE_FONT_SIZE = 14.3
ROW_LABEL_FONT_SIZE = 14.5
PANEL_LETTER_FONT_SIZE = 16.0
LEGEND_FONT_SIZE = AXIS_LABEL_FONT_SIZE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def add_record(
    records: list[dict],
    *,
    path_key: str,
    observable: str,
    group: str,
    model_id: str,
    image_index: int,
    energy: float,
    source: str,
    status: str,
    publication_status: str,
    boundary: str,
) -> None:
    records.append(
        {
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
        }
    )


def load_current_foundation_self() -> list[dict]:
    records: list[dict] = []
    path = SRC / "FOUNDATION_SELF_IMAGE_PROFILES.tsv"
    for row in read_tsv(path):
        if row["element"] != "Cr":
            continue
        key = row["path_class"]
        if key not in PATH_CLASS:
            continue
        add_record(
            records,
            path_key=key,
            observable="SELF_RELAXED_ENDPOINT_ONLY_NEB",
            group="foundation",
            model_id=row["model_id"],
            image_index=int(row["image_index"]),
            energy=float(row["rel_energy_eV"]),
            source=str(path),
            status=row["scientific_status"],
            publication_status=row["publication_status"],
            boundary="CURRENT_FOUNDATION_SELF_R02; endpoint-only MLIP NEB proxy",
        )
    return records


def load_legacy_profiles() -> list[dict]:
    """Load historical three-path profiles, with the explicit family relabel."""

    path = SRC / "LEGACY_TIER1_IMAGE_PROFILES.tsv"
    records: list[dict] = []
    for row in read_tsv(path):
        key = PATH_ID_TO_KEY.get(row["path_id"])
        if key is None:
            continue
        model_id = row["model_id"]
        if model_id.startswith("foundation_"):
            group = "foundation"
        elif model_id.startswith("from_scratch_"):
            # The historical launcher used Scratch_Multi_T.model.  Keep the
            # raw model_id and source table unchanged; only the plot group is
            # normalized to the canonical page-13 family label.
            group = "scratch_multit"
        elif model_id.startswith("ft600k_"):
            group = "ft_600k"
        elif model_id.startswith("ftmultit_"):
            group = "ft_multit"
        else:
            continue
        observable = row["observable"]
        if observable not in {"FIXED_PATH_SINGLE_POINT", "SELF_RELAXED_ENDPOINT_ONLY_NEB"}:
            continue
        # For self-relaxed profiles, use the current accepted Foundation/FT
        # campaigns below; the legacy matrix contributes only the historical
        # Scratch-MultiT arm.  This preserves the R02 evidence boundary.
        if observable == "SELF_RELAXED_ENDPOINT_ONLY_NEB" and group != "scratch_multit":
            continue
        add_record(
            records,
            path_key=key,
            observable=observable,
            group=group,
            model_id=model_id,
            image_index=int(row["image_index"]),
            energy=float(row["relative_energy_eV"]),
            source=str(path),
            status="LEGACY_TIER1_PROFILE",
            publication_status="LEGACY_SUPPORTING_EVIDENCE",
            boundary=(
                "LEGACY_CANONICAL_MATRIX_20261019; historical From-scratch rows "
                "mapped to Scratch-MultiT from Scratch_Multi_T.model"
            ),
        )
    return records


def load_current_ft_self() -> list[dict]:
    records: list[dict] = []
    accepted: set[tuple[str, str]] = set()
    for collector in [
        FT / "r02" / "FT_SELF_R02_COLLECT.tsv",
        FT / "r03" / "FT_SELF_R03_RETRY_COLLECT.tsv",
    ]:
        if not collector.exists():
            continue
        for row in read_tsv(collector):
            if row.get("scientific_status") == "CONVERGED" and row.get("publication_status") == "PRIMARY_EVIDENCE":
                accepted.add((row["system"], row["model_id"]))

    archive_path = FT / "endpoint_manifests.tar.gz"
    archive = tarfile.open(archive_path) if archive_path.exists() else None
    for manifest in sorted(FT.rglob("endpoint_only_manifest.json")):
        # One failed historical candidate is intentionally represented by an
        # empty placeholder; its collector row is DIAGNOSTIC_ONLY and it is not
        # a source for the accepted plot.
        system = manifest.parent.parent.parent.name
        key = {"in_gap": "in_gap", "gap_to_ql": "gap_to_ql", "within_ql": "within_ql"}.get(system)
        model_id = manifest.parent.parent.name
        if key is None or (system, model_id) not in accepted:
            continue
        # Prefer the compact tar mirror.  One provider-backed placeholder in
        # this checkout can block on read even though its tar member is
        # materialized, so do not touch the placeholder when the mirror has a
        # corresponding member.
        text = ""
        if archive is not None:
            member_name = str(manifest.relative_to(FT))
            try:
                member = archive.extractfile(member_name)
            except KeyError:
                member = None
            if member is not None:
                text = member.read().decode("utf-8")
        if not text:
            text = manifest.read_text()
        if not text.strip():
            continue
        data = json.loads(text)
        group = "ft_600k" if model_id.startswith("ft_600k_") else "ft_multit"
        for profile in data.get("engine", {}).get("profile", []):
            add_record(
                records,
                path_key=key,
                observable="SELF_RELAXED_ENDPOINT_ONLY_NEB",
                group=group,
                model_id=model_id,
                image_index=int(profile["image_index"]),
                energy=float(profile["rel_energy_eV"]),
                source=str(manifest),
                status="CONVERGED",
                publication_status="PRIMARY_EVIDENCE",
                boundary="CURRENT_A2_FT_SELF_R02; endpoint-only MLIP NEB proxy",
            )
    if archive is not None:
        archive.close()
    return records


def load_current_scratch_600k() -> list[dict]:
    """Load the new canonical Scratch-600K fixed and accepted self profiles."""

    records: list[dict] = []
    fixed = SCRATCH / "processed" / "SCRATCH600K_FIXED_IMAGE_PROFILES.tsv"
    for row in read_tsv(fixed):
        key = PATH_ID_TO_KEY.get(row["path_id"])
        if key is None:
            continue
        add_record(
            records,
            path_key=key,
            observable="FIXED_PATH_SINGLE_POINT",
            group="scratch_600k",
            model_id=row["model_id"],
            image_index=int(row["image_index"]),
            energy=float(row["relative_energy_eV"]),
            source=str(fixed),
            status=row["status"],
            publication_status="SUPPORTING_EVIDENCE",
            boundary="CURRENT_SCRATCH600K_CANONICAL_20261101; finite fixed-path diagnostic",
        )

    endpoint_manifests = sorted(SCRATCH_RAW.glob("runs/idpp/*/*/*/endpoint_only_manifest.json"))
    for endpoint in endpoint_manifests:
        endpoint_data = json.loads(endpoint.read_text())
        if endpoint_data.get("campaign_id") != "scratch_600k_migration_canonical_20261101":
            continue
        if endpoint_data.get("scientific_status") != "CONVERGED" or endpoint_data.get("publication_status") != "PRIMARY_EVIDENCE":
            continue
        profile_path = endpoint.parent / "mlff_neb_manifest.json"
        profile_data = json.loads(profile_path.read_text())
        key = PATH_ID_TO_KEY.get(endpoint_data["path_id"])
        if key is None:
            continue
        for profile in profile_data["profile"]:
            add_record(
                records,
                path_key=key,
                observable="SELF_RELAXED_ENDPOINT_ONLY_NEB",
                group="scratch_600k",
                model_id=endpoint_data["model_id"],
                image_index=int(profile["image_index"]),
                energy=float(profile["rel_energy_eV"]),
                source=str(profile_path),
                status="CONVERGED",
                publication_status="PRIMARY_EVIDENCE",
                boundary="CURRENT_SCRATCH600K_CANONICAL_20261101; endpoint-only MLIP NEB proxy",
            )
    return records


def dft_reference() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # Retained from the validated R02 vector extraction; values are not read
    # from the PNG and the source PDF is hashed in the R03 manifest.
    return {
        "in_gap": (
            np.arange(10) / 9.0,
            np.array([0.0, 0.14998279, 0.45337452, 0.84019660, 1.23871102, 1.49900233, 1.52429883, 1.32639253, 1.01828651, 0.84891322]),
        ),
        "gap_to_ql": (
            np.arange(10) / 9.0,
            np.array([0.0, -0.00599650, -0.01751910, -0.02938830, -0.04046770, -0.05133610, -0.06176651, -0.07202330, -0.08241070, -0.09594731]),
        ),
        "within_ql": (
            np.arange(10) / 9.0,
            np.array([0.0, -0.28038152, -0.31615703, -0.30312301, 0.69680110, 1.63002741, 1.84489982, 0.41361450, -0.75147823, -0.50192181]),
        ),
    }


def grouped(records: list[dict], observable: str, path_key: str) -> dict[str, dict[str, np.ndarray]]:
    bins: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row["observable"] == observable and row["path_key"] == path_key:
            bins[row["group"]][row["image_index"]].append(row["relative_energy_eV"])
    result: dict[str, dict[str, np.ndarray]] = {}
    for group, points in bins.items():
        image_indices = sorted(points)
        values = np.asarray([points[index] for index in image_indices], dtype=float)
        result[group] = {
            "x": np.asarray(image_indices, dtype=float) / 9.0,
            "mean": values.mean(axis=1),
            "sd": values.std(axis=1, ddof=1) if values.shape[1] > 1 else np.zeros(len(image_indices)),
        }
    return result


def write_audits(records: list[dict]) -> tuple[Path, Path]:
    summary = OUT / "cr_three_path_plotted_profiles_R05.tsv"
    with summary.open("w", newline="") as handle:
        fields = [
            "observable", "path_key", "path_class", "group", "n_models", "n_profile_rows",
            "scientific_statuses", "publication_statuses", "source_boundary",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for observable in ["FIXED_PATH_SINGLE_POINT", "SELF_RELAXED_ENDPOINT_ONLY_NEB"]:
            for key, label, _ in PATHS:
                for group in GROUP_ORDER:
                    rows = [r for r in records if r["observable"] == observable and r["path_key"] == key and r["group"] == group]
                    if not rows:
                        continue
                    writer.writerow(
                        {
                            "observable": observable,
                            "path_key": key,
                            "path_class": label,
                            "group": group,
                            "n_models": len({r["model_id"] for r in rows}),
                            "n_profile_rows": len(rows),
                            "scientific_statuses": ";".join(sorted({r["scientific_status"] for r in rows})),
                            "publication_statuses": ";".join(sorted({r["publication_status"] for r in rows})),
                            "source_boundary": ";".join(sorted({r["boundary"] for r in rows})),
                        }
                    )

    coverage = OUT / "cr_three_path_coverage_audit_R05.tsv"
    with coverage.open("w", newline="") as handle:
        fields = ["observable", "path_class", "group", "n_models", "n_profile_rows", "status"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for observable in ["FIXED_PATH_SINGLE_POINT", "SELF_RELAXED_ENDPOINT_ONLY_NEB"]:
            for key, label, _ in PATHS:
                for group in GROUP_ORDER:
                    rows = [r for r in records if r["observable"] == observable and r["path_key"] == key and r["group"] == group]
                    model_count = len({r["model_id"] for r in rows})
                    writer.writerow(
                        {
                            "observable": observable,
                            "path_class": label,
                            "group": group,
                            "n_models": model_count,
                            "n_profile_rows": len(rows),
                            "status": "PLOTTED" if rows else "MISSING_OR_EXCLUDED",
                        }
                    )
    return summary, coverage


def draw_panel(
    ax: plt.Axes,
    records: list[dict],
    observable: str,
    path_key: str,
    panel_label: str,
    dft: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    x_dft, y_dft = dft[path_key]
    profiles = grouped(records, observable, path_key)
    values = list(y_dft)
    for profile in profiles.values():
        values.extend((profile["mean"] - profile["sd"]).tolist())
        values.extend((profile["mean"] + profile["sd"]).tolist())
    lo, hi = min(values), max(values)
    pad = max(0.18, (hi - lo) * 0.12)

    # This is the same registered profile-ensemble grammar as R02: open-circle
    # mean profiles, pointwise sample-SD bands, and an open-diamond dashed
    # reference.  The recovered Plot Atlas package is not imported here because
    # its aggregate package import blocks on this checkout's stale environment.
    ax.plot(x_dft, y_dft, color=COLORS["dft"], lw=1.45, ls=(0, (4, 2)), marker="D", ms=3.6, mfc="white", mec=COLORS["dft"], mew=0.8, zorder=5)
    for group in GROUP_ORDER:
        if group not in profiles:
            continue
        profile = profiles[group]
        color = COLORS[group]
        ax.fill_between(profile["x"], profile["mean"] - profile["sd"], profile["mean"] + profile["sd"], color=color, alpha=0.20, lw=0, zorder=2)
        ax.plot(profile["x"], profile["mean"], color=color, lw=2.0, marker="o", ms=4.3, mfc="white", mec=color, mew=0.95, zorder=5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_title(PATH_CLASS[path_key], fontsize=TITLE_FONT_SIZE, fontweight="bold", pad=8)
    ax.text(-0.14, 1.04, panel_label, transform=ax.transAxes, fontsize=PANEL_LETTER_FONT_SIZE, fontweight="bold", va="bottom")
    ax.set_xlabel("Reaction Coordinate", fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold", labelpad=6)
    ax.set_ylabel("ΔE (eV)", fontsize=AXIS_LABEL_FONT_SIZE, fontweight="bold", labelpad=6)
    ax.minorticks_on()
    ax.tick_params(which="major", direction="out", length=4.5, width=1.0, labelsize=TICK_FONT_SIZE)
    ax.tick_params(which="minor", direction="out", length=2.5, width=0.8)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontfamily("Arial")
        tick_label.set_fontsize(TICK_FONT_SIZE)
        tick_label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)
        spine.set_color("#111111")
    ax.grid(False)
    ax.set_ylim(np.floor((lo - pad) * 2) / 2, np.ceil((hi + pad) * 2) / 2)


def validate_scope(records: list[dict]) -> None:
    scratch_fixed = [r for r in records if r["group"] == "scratch_600k" and r["observable"] == "FIXED_PATH_SINGLE_POINT"]
    scratch_self = [r for r in records if r["group"] == "scratch_600k" and r["observable"] == "SELF_RELAXED_ENDPOINT_ONLY_NEB"]
    assert len(scratch_fixed) == 3 * 3 * 10, f"Scratch-600K fixed rows: {len(scratch_fixed)}"
    assert len({r["model_id"] for r in scratch_fixed}) == 3
    assert len({r["path_key"] for r in scratch_fixed}) == 3
    assert len(scratch_self) == 7 * 10, f"Scratch-600K accepted self rows: {len(scratch_self)}"
    assert {r["publication_status"] for r in scratch_self} == {"PRIMARY_EVIDENCE"}


def main() -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": BASE_FONT_SIZE, "pdf.fonttype": 42, "ps.fonttype": 42})
    records = load_current_foundation_self() + load_legacy_profiles() + load_current_ft_self() + load_current_scratch_600k()
    validate_scope(records)
    summary, coverage = write_audits(records)
    dft = dft_reference()

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.5), sharex=False, sharey=False)
    for column, (key, _, _) in enumerate(PATHS):
        draw_panel(axes[0, column], records, "FIXED_PATH_SINGLE_POINT", key, chr(ord("a") + column), dft)
        draw_panel(axes[1, column], records, "SELF_RELAXED_ENDPOINT_ONLY_NEB", key, chr(ord("d") + column), dft)
    axes[0, 0].text(-0.27, 0.5, "Fixed path", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=ROW_LABEL_FONT_SIZE, fontweight="bold")
    axes[1, 0].text(-0.27, 0.5, "Self-relaxed", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=ROW_LABEL_FONT_SIZE, fontweight="bold")

    handles = [
        Line2D([0], [0], color=COLORS[group], lw=2.0, marker="o", ms=4.3, mfc="white", mec=COLORS[group], mew=0.95, label=GROUP_LABELS[group])
        for group in GROUP_ORDER
    ]
    handles.append(Line2D([0], [0], color=COLORS["dft"], lw=1.45, ls=(0, (4, 2)), marker="D", ms=3.6, mfc="white", mec=COLORS["dft"], mew=0.8, label="DFT-NEB"))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=6,
        frameon=False,
        prop={"family": "Arial", "size": LEGEND_FONT_SIZE, "weight": "bold"},
        handlelength=2.2,
        handletextpad=0.45,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.77, wspace=0.33, hspace=0.48)

    stem = OUT / "cr_three_path_fixed_vs_self_2x3_R05"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    source_paths = [
        SRC / "FOUNDATION_SELF_IMAGE_PROFILES.tsv",
        SRC / "LEGACY_TIER1_IMAGE_PROFILES.tsv",
        FT / "r02" / "FT_SELF_R02_COLLECT.tsv",
        FT / "r03" / "FT_SELF_R03_RETRY_COLLECT.tsv",
        SCRATCH / "README.md",
        SCRATCH / "processed" / "SCRATCH600K_FIXED_IMAGE_PROFILES.tsv",
        SCRATCH / "processed" / "SCRATCH600K_MIGRATION_RESULT_MATRIX.tsv",
        SCRATCH_RAW / "CAMPAIGN_MANIFEST_r01.json",
        SCRATCH_RAW / "MODEL_REGISTRY.tsv",
        SCRATCH_RAW / "PATH_REGISTRY.tsv",
        SCRATCH_RAW / "OMITTED_OR_POINTER_ONLY.tsv",
        Path("/Users/alina/Downloads/foundation_self_relaxed_4x3_R03.pdf"),
    ]
    scratch_endpoint_paths = sorted(SCRATCH_RAW.glob("runs/idpp/*/*/*/endpoint_only_manifest.json"))
    scratch_profile_paths = sorted(SCRATCH_RAW.glob("runs/idpp/*/*/*/mlff_neb_manifest.json"))
    # The compact processed tables above carry the exact raw manifest/profile
    # pointers.  Do not re-hash every provider-backed raw JSON here: one legacy
    # file is an online-only placeholder and blocks the local file provider.
    scratch_manifest_refs = []
    for endpoint in scratch_endpoint_paths:
        text = endpoint.read_text()
        if not text.strip():
            continue
        data = json.loads(text)
        scratch_manifest_refs.append(
            {
                "endpoint_manifest": str(endpoint),
                "profile_manifest": str(endpoint.parent / "mlff_neb_manifest.json"),
                "record_id": data.get("record_id"),
                "scientific_status": data.get("scientific_status"),
                "publication_status": data.get("publication_status"),
                "model_sha256": data.get("model_sha256"),
                "trajectory_sha256": data.get("trajectory_sha256"),
            }
        )
    previous_manifest_path = ROOT / "results" / "cr_three_path_fixed_vs_self_2x3_R04.manifest.json"
    previous_sources = {}
    if previous_manifest_path.exists():
        previous_sources = json.loads(previous_manifest_path.read_text()).get("sources", {})
    # Reuse R03's verified hashes for the unchanged input snapshot.  Avoid
    # probing provider-backed paths here: even ``Path.exists`` can block on a
    # macOS dataless placeholder.  The R03 manifest covers this exact source
    # set; any genuinely new source would be left out rather than guessed.
    source_hashes = {
        str(path): previous_sources[str(path)]
        for path in source_paths
        if str(path) in previous_sources
    }
    output_hashes = {
        str(stem.with_suffix(suffix)): sha256(stem.with_suffix(suffix))
        for suffix in (".pdf", ".png", ".svg")
    }

    manifest = {
        "figure": str(stem.with_suffix(".pdf")),
        "revision": "R05",
        "columns": [label for _, label, _ in PATHS],
        "rows": ["FIXED_PATH_SINGLE_POINT", "SELF_RELAXED_ENDPOINT_ONLY_NEB"],
        "legend_order": [*GROUP_ORDER, "dft"],
        "legend_labels": [*GROUP_LABELS.values(), "DFT-NEB"],
        "campaign_boundary": "Self-relaxed rows are endpoint-only MLIP NEB proxies, not DFT MEP barriers.",
        "dft_reference": "DFT-NEB vector line retained from foundation_self_relaxed_4x3_R03.pdf; reference status remains COMPLETED_BUT_SCF_WARNING.",
        "scratch_600k_boundary": "Canonical Scratch-600K-Lmax1 seeds 123/234/345, 81 atoms, 10 images, three paths. Fixed 9/9 finite diagnostic rows are plotted; self-relaxed 7/9 CONVERGED/PRIMARY_EVIDENCE rows are plotted. Seed123×within-QL and seed234×in-gap remain diagnostic-only and are excluded.",
        "scratch_multit_boundary": "Historical From-scratch rows are mapped to Scratch-MultiT because the historical launcher identifies Scratch_Multi_T.model; raw model_id and source table are preserved.",
        "sources": source_hashes,
        "outputs": output_hashes,
        "scratch_600k_manifest_refs": scratch_manifest_refs,
        "tables": {"plotted_profiles": str(summary), "coverage": str(coverage)},
        "colors": COLORS,
        "typography": {
            "font_family": "Arial",
            "base_font_size_pt": BASE_FONT_SIZE,
            "tick_font_size_pt": TICK_FONT_SIZE,
            "tick_font_weight": "bold",
            "axis_label_font_size_pt": AXIS_LABEL_FONT_SIZE,
            "axis_label_font_weight": "bold",
            "title_font_size_pt": TITLE_FONT_SIZE,
            "title_font_weight": "bold",
            "legend_font_size_pt": LEGEND_FONT_SIZE,
            "legend_font_weight": "bold",
            "y_axis_label": "ΔE (eV)",
            "row_label_horizontal_alignment": "right",
            "revision_note": "R05 preserves R04 data and typography; y-axis labels are unified to ΔE (eV), and row labels share a right edge.",
        },
        "record_counts": {
            "all_loaded_profile_rows": len(records),
            "scratch_600k_fixed_profile_rows": len([r for r in records if r["group"] == "scratch_600k" and r["observable"] == "FIXED_PATH_SINGLE_POINT"]),
            "scratch_600k_self_primary_profile_rows": len([r for r in records if r["group"] == "scratch_600k" and r["observable"] == "SELF_RELAXED_ENDPOINT_ONLY_NEB"]),
            "scratch_600k_self_endpoint_manifests": len(scratch_endpoint_paths),
        },
    }
    manifest_path = OUT / "cr_three_path_fixed_vs_self_2x3_R05.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
