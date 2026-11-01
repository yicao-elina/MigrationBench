#!/usr/bin/env python3
"""Build clustered-bar replacements for the Figure 2 trajectory diagnostics.

Temperature and Cr-count are categorical summaries, so the paper-facing
grammar is mean +/- sample SD over the three seeds with every seed point
overlaid.  The complete frame-level points are also retained around every
bar, so the figure does not reduce a category to three black markers. The
original held-out-trajectory panel is rendered with the same grammar, but
missing model/trajectory combinations remain blank. These are fixed-split
reconstructions; no LOTO result is inferred.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

WORKTREE = Path(__file__).resolve().parents[4]
WILEY = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench")
DATA_ROOT = WILEY / "analysis/rmse_train_test_20261026"
SUMMARY_PATH = DATA_ROOT / "parity_summary.csv"
FRAME_ROOT = DATA_ROOT / "raw/rockfish_parity"
SOURCE_SUMMARY_PATH = SUMMARY_PATH
SOURCE_FRAME_ROOT = FRAME_ROOT
COLORS = {
    "FT-600K": "#002E71",
    "FT-MultiT": "#67ACE5",
    "Scratch-600K": "#FF9F1B",
    "Scratch-MultiT": "#F4C95D",
}
FAMILIES = ["Scratch-600K", "FT-600K", "Scratch-MultiT", "FT-MultiT"]
SEEDS = ["123", "234", "345"]
SPLITS = ["train", "test"]
FONT = "Arial"

WORKTREE = Path(__file__).resolve().parents[4]
OUT_DEFAULT = (
    WORKTREE
    / "docs/reproducibility/canva_figure_provenance/R03_typography18pt/figures/figure02_clustered_v5_full_points"
)

TEMP_ORDER = ["300K", "600K", "1200K", "NA"]
TEMP_LABELS = {"300K": "300 K", "600K": "600 K", "1200K": "1200 K", "NA": "isolated\nreference"}
CR_ORDER = ["Cr0", "Cr1", "Cr2", "Cr3", "Cr4", "isolated"]
CR_LABELS = {"Cr0": "No Cr", "Cr1": "1 Cr", "Cr2": "2 Cr", "Cr3": "3 Cr", "Cr4": "4 Cr", "isolated": "isolated\nreference"}

VECTOR_FIGURE_DIR = WORKTREE / "docs/reproducibility/canva_figure_provenance/R03_typography18pt/figures/figure02_trajectory_variants_v2"


def family_from_filename(name: str) -> str:
    if name.startswith("ft_600k"):
        return "FT-600K"
    if name.startswith("ft_multit"):
        return "FT-MultiT"
    if name.startswith("scratch_600k"):
        return "Scratch-600K"
    if name.startswith("scratch_multit"):
        return "Scratch-MultiT"
    raise ValueError(name)


def seed_from_filename(name: str) -> str:
    match = re.search(r"seed(\d+)", name)
    if not match:
        raise ValueError(name)
    return match.group(1)


def split_from_filename(name: str) -> str:
    return "test" if "_test__" in name else "train"


def cr_count(trajectory: str) -> str:
    if trajectory == "isolated_atoms_SG15":
        return "isolated"
    match = re.search(r"Cr(\d+)", trajectory)
    return f"Cr{match.group(1)}" if match else "Cr0"


def temperature(trajectory: str) -> str:
    match = re.search(r"(\d+)K", trajectory)
    return f"{match.group(1)}K" if match else "NA"


def house_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT, "Helvetica", "DejaVu Sans"],
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "axes.labelsize": 20,
            "axes.labelweight": "bold",
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
            "axes.linewidth": 1.10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def style_axis(ax) -> None:
    ax.tick_params(axis="both", which="major", direction="out", length=6, width=1.05)
    ax.tick_params(axis="both", which="minor", direction="out", length=3, width=0.8)
    ax.minorticks_on()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(fig, out_dir: Path, stem: str) -> dict[str, str]:
    paths = {
        "png": out_dir / f"{stem}.png",
        "pdf": out_dir / f"{stem}.pdf",
        "svg": out_dir / f"{stem}.svg",
    }
    for path in paths.values():
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {key: str(value) for key, value in paths.items()}


def friendly_trajectory(trajectory: str) -> str:
    if trajectory == "2L_o3_t3_600K_aimd_1":
        return "O3\nNo Cr · 600 K"
    if trajectory == "2L_tetra_Cr4_600K_aimd_1":
        return "Tetra\n4 Cr · 600 K"
    label = trajectory.replace("2L_", "").replace("_aimd_1", "").replace("_aimd", "")
    label = label.replace("_", " ")
    return label


def read_frame_records() -> list[dict[str, object]]:
    records = []
    for path in sorted(FRAME_ROOT.glob("*__per_frame.csv")):
        family = family_from_filename(path.name)
        seed = seed_from_filename(path.name)
        split = split_from_filename(path.name)
        with path.open(newline="", errors="replace") as handle:
            for row in csv.DictReader(handle):
                energy_ref = float(row["energy_ref_eV_per_atom"])
                energy_pred = float(row["energy_pred_eV_per_atom"])
                force_rmse = float(row["force_rmse_meV_A"])
                n_atoms = float(row["n_atoms"])
                trajectory = row["trajectory"]
                records.append(
                    {
                        "family": family,
                        "seed": seed,
                        "split": split,
                        "trajectory": trajectory,
                        "temperature": temperature(trajectory),
                        "cr": cr_count(trajectory),
                        "energy_error_meV": (energy_pred - energy_ref) * 1000.0,
                        "force_rmse_meV_A": force_rmse,
                        "n_atoms": n_atoms,
                    }
                )
    if not records:
        raise RuntimeError("No per-frame CSV records found")
    return records


VECTOR_COUNTS = {
    "temperature": {
        "Scratch-600K": {"train": {"600K": 1783}, "test": {"600K": 267}},
        "FT-600K": {"train": {"600K": 1783}, "test": {"600K": 267}},
        "Scratch-MultiT": {"train": {"300K": 823, "600K": 1594, "1200K": 239}, "test": {"600K": 767}},
        "FT-MultiT": {"train": {"300K": 823, "600K": 1594, "1200K": 239}, "test": {"600K": 767}},
    },
    "cr": {
        "Scratch-600K": {"train": {"Cr0": 376, "Cr1": 160, "Cr2": 306, "Cr3": 441, "Cr4": 500}, "test": {"Cr0": 267}},
        "FT-600K": {"train": {"Cr0": 376, "Cr1": 160, "Cr2": 306, "Cr3": 441, "Cr4": 500}, "test": {"Cr0": 267}},
        "Scratch-MultiT": {"train": {"Cr0": 615, "Cr1": 160, "Cr2": 1011, "Cr3": 441, "Cr4": 429}, "test": {"Cr0": 267, "Cr4": 500}},
        "FT-MultiT": {"train": {"Cr0": 615, "Cr1": 160, "Cr2": 1011, "Cr3": 441, "Cr4": 429}, "test": {"Cr0": 267, "Cr4": 500}},
    },
}


def _vector_axis(root: ET.Element, axis_id: str) -> ET.Element:
    namespace = "{http://www.w3.org/2000/svg}"
    for element in root.iter(namespace + "g"):
        if element.attrib.get("id") == axis_id:
            return element
    raise KeyError(axis_id)


def _vector_y_transform(axis: ET.Element) -> tuple[float, float]:
    exponents = []
    minor_ys = []
    namespace = "{http://www.w3.org/2000/svg}"
    for element in axis.iter(namespace + "g"):
        if not element.attrib.get("id", "").startswith("ytick_"):
            continue
        for use in element.iter(namespace + "use"):
            style = use.attrib.get("style", "")
            if "stroke-width: 0.8" in style:
                minor_ys.append(float(use.attrib["y"]))
            if "stroke-width: 1.35" not in style:
                continue
            text = " ".join(part.strip() for part in element.itertext() if part.strip())
            numbers = re.findall(r"[-−]?\d+", text)
            if numbers:
                exponents.append((float(use.attrib["y"]), float(numbers[-1].replace("−", "-"))))
    if len(exponents) == 1:
        y0, exponent0 = exponents[0]
        above = sorted((y for y in minor_ys if y < y0), reverse=True)
        below = sorted((y for y in minor_ys if y > y0))
        inferred = [(y0, exponent0)]
        inferred.extend((y, exponent0 + np.log10(multiplier)) for y, multiplier in zip(above, range(2, 10)))
        inferred.extend(
            (y, exponent0 - 1.0 + np.log10(multiplier))
            for y, multiplier in zip(below, range(9, 1, -1))
        )
        exponents = inferred
    if len(exponents) < 2:
        raise RuntimeError("Could not recover logarithmic y ticks from vector figure")
    ys, values = np.asarray(exponents, dtype=float).T
    slope, intercept = np.polyfit(ys, values, 1)
    return float(slope), float(intercept)


def _vector_x_centers(axis: ET.Element, expected: int) -> list[float]:
    namespace = "{http://www.w3.org/2000/svg}"
    centers = []
    for element in axis.iter(namespace + "g"):
        if not element.attrib.get("id", "").startswith("xtick_"):
            continue
        for use in element.iter(namespace + "use"):
            if "stroke-width: 1.35" in use.attrib.get("style", ""):
                centers.append(float(use.attrib["x"]))
                break
    centers = sorted(set(round(center, 6) for center in centers))
    if len(centers) != expected:
        raise RuntimeError(f"Expected {expected} categorical x centers, found {centers}")
    return centers


def read_frame_records_from_svg(svg_path: Path, grouping: str) -> list[dict[str, object]]:
    """Recover the exact plotted frame values from the prior vector source.

    This fallback is needed because several canonical per-frame CSVs are
    currently iCloud dataless placeholders. The earlier SVG contains every
    plotted frame as a transformed vector point. Point order is retained, so
    the three equal-seed chunks can be restored using the audited frame counts.
    """
    root = ET.parse(svg_path).getroot()
    namespace = "{http://www.w3.org/2000/svg}"
    color_to_family = {value.lower(): key for key, value in COLORS.items()}
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: {"energy": [], "force": []})
    for axis_number, family in enumerate(FAMILIES, start=1):
        for metric, axis_number_for_metric in (("energy", axis_number), ("force", axis_number + 4)):
            axis = _vector_axis(root, f"axes_{axis_number_for_metric}")
            slope, intercept = _vector_y_transform(axis)
            centers = _vector_x_centers(axis, len(TEMP_ORDER) if grouping == "temperature" else len(CR_ORDER))
            for use in axis.iter(namespace + "use"):
                style = use.attrib.get("style", "").lower()
                if "fill-opacity: 0.1" not in style:
                    continue
                color_match = re.search(r"fill:\s*(#[0-9a-f]+)", style)
                if not color_match or color_match.group(1) not in color_to_family:
                    continue
                x = float(use.attrib["x"])
                y = float(use.attrib["y"])
                category_index = int(np.argmin(np.abs(np.asarray(centers) - x)))
                category = (TEMP_ORDER if grouping == "temperature" else CR_ORDER)[category_index]
                split = "train" if x < centers[category_index] else "test"
                value = float(10.0 ** (slope * y + intercept))
                grouped[(family, category, split)][metric].append(value)

    records = []
    counts = VECTOR_COUNTS[grouping]
    for (family, category, split), metrics in sorted(grouped.items()):
        energy_values = metrics["energy"]
        force_values = metrics["force"]
        if len(energy_values) != len(force_values):
            raise RuntimeError(f"Vector energy/force counts differ for {family}/{category}/{split}")
        per_seed = counts.get(family, {}).get(split, {}).get(category, 0)
        if per_seed <= 0 or len(energy_values) != per_seed * len(SEEDS):
            raise RuntimeError(
                f"Unexpected vector count for {family}/{category}/{split}: "
                f"{len(energy_values)} (expected {per_seed * len(SEEDS)})"
            )
        for index, (energy, force) in enumerate(zip(energy_values, force_values)):
            seed = SEEDS[index // per_seed]
            trajectory = "unknown"
            n_atoms = 1.0
            if grouping == "cr" and split == "test" and category == "Cr0":
                trajectory, n_atoms = "2L_o3_t3_600K_aimd_1", 86.0
            elif grouping == "cr" and split == "test" and category == "Cr4":
                trajectory, n_atoms = "2L_tetra_Cr4_600K_aimd_1", 84.0
            records.append(
                {
                    "family": family,
                    "seed": seed,
                    "split": split,
                    "trajectory": trajectory,
                    "temperature": category if grouping == "temperature" else "unknown",
                    "cr": category if grouping == "cr" else "unknown",
                    "energy_error_meV": energy,
                    "force_rmse_meV_A": force,
                    "n_atoms": n_atoms,
                }
            )
    if not records:
        raise RuntimeError(f"No vector points recovered from {svg_path}")
    return records


def aggregate_by_group(
    records: list[dict[str, object]], group_key: str
) -> dict[str, dict[str, dict[tuple[str, str, str], float]]]:
    """Aggregate both metrics once per family/seed/split/category."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record[group_key]),
                str(record["family"]),
                str(record["seed"]),
                str(record["split"]),
            )
        ].append(record)
    output: dict[str, dict[str, dict[tuple[str, str, str], float]]] = {
        "energy": {},
        "force": {},
    }
    for (group_value, family, seed, split), values in grouped.items():
        key = (family, seed, split)
        energy_errors = np.asarray([float(value["energy_error_meV"]) for value in values], dtype=float)
        force_errors = np.asarray([float(value["force_rmse_meV_A"]) for value in values], dtype=float)
        weights = np.asarray([float(value["n_atoms"]) for value in values], dtype=float)
        output["energy"].setdefault(group_value, {})[key] = float(np.sqrt(np.mean(energy_errors * energy_errors)))
        output["force"].setdefault(group_value, {})[key] = float(
            np.sqrt(np.sum(weights * force_errors * force_errors) / np.sum(weights))
        )
    return output


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size != 3 or not np.isfinite(array).all():
        raise RuntimeError(f"Expected three finite seed values, found {array}")
    return float(np.mean(array)), float(np.std(array, ddof=1))


def frame_point_counts(
    records: list[dict[str, object]], group_key: str, order: list[str]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for family in FAMILIES:
        for split in SPLITS:
            for category in order:
                count = sum(
                    1
                    for record in records
                    if str(record["family"]) == family
                    and str(record["split"]) == split
                    and str(record[group_key]) == category
                )
                if count:
                    counts[f"{family}/{split}/{category}"] = count
    return counts


def split_handles() -> list[Patch]:
    return [
        Patch(facecolor="#777777", edgecolor="#111111", linewidth=0.7, alpha=0.30, hatch="///", label="train"),
        Patch(facecolor="#777777", edgecolor="#111111", linewidth=0.7, alpha=0.90, label="held-out test"),
    ]


def _raw_metric_value(record: dict[str, object], metric: str) -> float:
    """Return the positive frame-level quantity shown on the log y-axis."""
    if metric == "energy":
        return abs(float(record["energy_error_meV"]))
    return float(record["force_rmse_meV_A"])


def _raw_points_for(
    records: list[dict[str, object]],
    *,
    family: str,
    group_key: str,
    category: str,
    split: str,
    metric: str,
) -> np.ndarray:
    values = [
        _raw_metric_value(record, metric)
        for record in records
        if str(record["family"]) == family
        and str(record[group_key]) == category
        and str(record["split"]) == split
    ]
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array) & (array > 0.0)]


def _stable_seed(*parts: object) -> int:
    """Create deterministic jitter without relying on Python's randomized hash."""
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _asymmetric_sd(mean: float, sd: float, floor: float) -> np.ndarray:
    """Keep an SD whisker valid when a log axis has a positive lower bound."""
    lower = max(float(mean - sd), float(floor))
    upper = max(float(mean + sd), lower * (1.0 + 1.0e-9))
    return np.asarray([[float(mean - lower)], [float(upper - mean)]])


def _y_floor(raw_values: list[np.ndarray]) -> float:
    finite = np.concatenate([values for values in raw_values if values.size]) if any(values.size for values in raw_values) else np.asarray([])
    if finite.size == 0:
        return 1.0e-3
    return max(float(np.nanmin(finite)) * 0.65, 1.0e-4)


def category_bar_figure(
    aggregates: dict[str, dict[str, dict[tuple[str, str, str], float]]],
    frame_records: list[dict[str, object]],
    *,
    group_key: str,
    order: list[str],
    labels: dict[str, str],
    xlabel: str,
    title: str,
    stem: str,
    out_dir: Path,
) -> dict[str, str]:
    metrics = [("energy", "Energy RMSE (meV atom$^{-1}$)"), ("force", "Force RMSE (meV Å$^{-1}$)")]
    fig, axes = plt.subplots(2, 4, figsize=(24.5, 13.0), squeeze=False)
    for row_index, (metric, ylabel) in enumerate(metrics):
        for col, family in enumerate(FAMILIES):
            ax = axes[row_index, col]
            raw_by_position: dict[tuple[int, str], np.ndarray] = {}
            raw_values = []
            for category_index, category in enumerate(order):
                for split in SPLITS:
                    raw = _raw_points_for(
                        frame_records,
                        family=family,
                        group_key=group_key,
                        category=category,
                        split=split,
                        metric=metric,
                    )
                    raw_by_position[(category_index, split)] = raw
                    raw_values.append(raw)
            floor = _y_floor(raw_values)

            # Every finite frame-level value is retained. The small deterministic
            # horizontal jitter prevents thousands of coincident points from
            # becoming a single opaque vertical stroke while preserving the
            # category and train/test positions exactly.
            for category_index, category in enumerate(order):
                for split in SPLITS:
                    raw = raw_by_position[(category_index, split)]
                    if raw.size == 0:
                        continue
                    xpos = float(category_index) + (-0.18 if split == "train" else 0.18)
                    rng = np.random.default_rng(_stable_seed(metric, family, category, split))
                    jitter = rng.uniform(-0.125, 0.125, raw.size)
                    ax.scatter(
                        np.full(raw.size, xpos) + jitter,
                        raw,
                        s=6.0,
                        color=COLORS[family],
                        alpha=0.16 if split == "train" else 0.22,
                        edgecolors="none",
                        zorder=1,
                    )

            for category_index, category in enumerate(order):
                for split_index, split in enumerate(SPLITS):
                    values_by_seed = aggregates[metric].get(category, {})
                    values = [values_by_seed.get((family, seed, split), np.nan) for seed in SEEDS]
                    if not np.isfinite(values).all():
                        continue
                    mean, sd = mean_sd(values)
                    xpos = float(category_index) + (-0.18 if split == "train" else 0.18)
                    ax.bar(
                        xpos,
                        mean - floor,
                        bottom=floor,
                        width=0.31,
                        yerr=_asymmetric_sd(mean, sd, floor),
                        capsize=4.0,
                        color=COLORS[family],
                        alpha=0.30 if split == "train" else 0.90,
                        hatch="///" if split == "train" else None,
                        edgecolor="#111111",
                        linewidth=0.65,
                        error_kw={"ecolor": "#222222", "elinewidth": 1.25, "capthick": 1.25},
                        zorder=2,
                    )
                    jitter = np.linspace(-0.065, 0.065, len(values))
                    ax.scatter(
                        np.full(len(values), xpos) + jitter,
                        values,
                        s=38,
                        color="#222222",
                        edgecolors="white",
                        linewidths=0.4,
                        zorder=5,
                    )
            ax.set_xticks(range(len(order)), [labels[key] for key in order])
            ax.set_xlim(-0.55, len(order) - 0.35)
            if row_index == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(labelleft=False)
            ax.set_yscale("log")
            ax.set_ylim(bottom=floor * 0.9)
            ax.set_title(f"{chr(65 + col + row_index * 4)}  {family}", loc="left", color=COLORS[family], pad=10)
            style_axis(ax)
    fig.legend(handles=split_handles(), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, fontsize=18)
    fig.suptitle(title, y=1.015, fontsize=22, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.12, top=0.865, hspace=0.36, wspace=0.12)
    return save(fig, out_dir, stem)


def trajectory_bar_figure(
    aggregates: dict[str, dict[str, dict[tuple[str, str, str], float]]],
    frame_records: list[dict[str, object]],
    out_dir: Path,
) -> dict[str, str]:
    test_trajectories = ["2L_o3_t3_600K_aimd_1", "2L_tetra_Cr4_600K_aimd_1"]
    labels = {trajectory: friendly_trajectory(trajectory) for trajectory in test_trajectories}
    metrics = [("energy", "Energy RMSE (meV atom$^{-1}$)"), ("force", "Force RMSE (meV Å$^{-1}$)")]
    fig, axes = plt.subplots(2, 1, figsize=(17.5, 12.5), squeeze=False)
    for row_index, (metric, ylabel) in enumerate(metrics):
        ax = axes[row_index, 0]
        raw_by_position: dict[tuple[int, str], np.ndarray] = {}
        raw_values = []
        for trajectory_index, trajectory in enumerate(test_trajectories):
            for family_index, family in enumerate(FAMILIES):
                raw = _raw_points_for(
                    frame_records,
                    family=family,
                    group_key="trajectory",
                    category=trajectory,
                    split="test",
                    metric=metric,
                )
                raw_by_position[(trajectory_index, family)] = raw
                raw_values.append(raw)
        floor = _y_floor(raw_values)

        for trajectory_index, trajectory in enumerate(test_trajectories):
            values_by_seed = aggregates[metric].get(trajectory, {})
            for family_index, family in enumerate(FAMILIES):
                raw = raw_by_position[(trajectory_index, family)]
                if raw.size:
                    xpos = float(trajectory_index) + (family_index - 1.5) * 0.18
                    rng = np.random.default_rng(_stable_seed(metric, family, trajectory))
                    jitter = rng.uniform(-0.065, 0.065, raw.size)
                    ax.scatter(
                        np.full(raw.size, xpos) + jitter,
                        raw,
                        s=6.0,
                        color=COLORS[family],
                        alpha=0.22,
                        edgecolors="none",
                        zorder=1,
                    )
                values = [values_by_seed.get((family, seed, "test"), np.nan) for seed in SEEDS]
                if not np.isfinite(values).all():
                    continue
                mean, sd = mean_sd(values)
                xpos = float(trajectory_index) + (family_index - 1.5) * 0.18
                ax.bar(
                    xpos,
                    mean - floor,
                    bottom=floor,
                    width=0.15,
                    yerr=_asymmetric_sd(mean, sd, floor),
                    capsize=3.5,
                    color=COLORS[family],
                    alpha=0.90,
                    edgecolor="#111111",
                    linewidth=0.65,
                    error_kw={"ecolor": "#222222", "elinewidth": 1.15, "capthick": 1.15},
                    zorder=2,
                )
                jitter = np.linspace(-0.035, 0.035, len(values))
                ax.scatter(
                    np.full(len(values), xpos) + jitter,
                    values,
                    s=34,
                    color="#222222",
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=5,
                )
        ax.set_xticks(range(len(test_trajectories)), [labels[trajectory] for trajectory in test_trajectories])
        ax.set_ylabel(ylabel)
        if row_index == 1:
            ax.set_xlabel("held-out trajectory")
        else:
            ax.tick_params(labelbottom=False)
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor * 0.9)
        ax.set_title(f"{chr(65 + row_index)}  held-out trajectory summary", loc="left", pad=10)
        style_axis(ax)
    family_handles = [
        Line2D([], [], marker="s", linestyle="", color=COLORS[family], markersize=9, label=family)
        for family in FAMILIES
    ]
    fig.legend(handles=family_handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=4, fontsize=18)
    fig.suptitle("Figure 2 revision | held-out trajectory RMSE + full frame coverage", y=1.015, fontsize=22, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.86, hspace=0.36)
    return save(fig, out_dir, "variant03_by_heldout_trajectory_clustered")


def build(out_dir: Path, vector_dir: Path | None = None) -> dict[str, object]:
    house_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    if vector_dir is None:
        records = read_frame_records()
        temperature_records = records
        cr_records = records
        sources = [SOURCE_SUMMARY_PATH, *sorted(SOURCE_FRAME_ROOT.glob("*__per_frame.csv"))]
        source_mode = "per_frame_csv"
        force_aggregation = "n_atoms-weighted RMSE of per-frame force RMSE in meV A^-1"
        energy_aggregation = "RMSE of signed per-frame energy error in meV per atom"
    else:
        temperature_svg = vector_dir / "variant04_by_temperature_class.svg"
        cr_svg = vector_dir / "variant05_by_cr_count_class.svg"
        vector_manifest = vector_dir / "trajectory_variants_v2_manifest.json"
        temperature_records = read_frame_records_from_svg(temperature_svg, "temperature")
        cr_records = read_frame_records_from_svg(cr_svg, "cr")
        records = temperature_records
        sources = [temperature_svg, cr_svg, vector_manifest]
        source_mode = "recovered_from_prior_vector_point_sources"
        force_aggregation = "unweighted RMS of plotted per-frame force RMSE; n_atoms are not encoded in the source SVG"
        energy_aggregation = "RMSE of plotted per-frame absolute energy error; this equals signed-error RMSE after squaring"
    temperature_aggregates = aggregate_by_group(temperature_records, "temperature")
    cr_aggregates = aggregate_by_group(cr_records, "cr")
    trajectory_aggregates = aggregate_by_group(cr_records, "trajectory")
    outputs = {
        "temperature": category_bar_figure(
            temperature_aggregates,
            temperature_records,
            group_key="temperature",
            order=TEMP_ORDER,
            labels=TEMP_LABELS,
            xlabel="trajectory temperature class",
            title="Figure 2 revision | clustered RMSE + full frame coverage by temperature class",
            stem="variant04_by_temperature_class_clustered_full_points",
            out_dir=out_dir,
        ),
        "cr_count": category_bar_figure(
            cr_aggregates,
            cr_records,
            group_key="cr",
            order=CR_ORDER,
            labels=CR_LABELS,
            xlabel="trajectory Cr-count class",
            title="Figure 2 revision | clustered RMSE + full frame coverage by Cr-count class",
            stem="variant05_by_cr_count_class_clustered_full_points",
            out_dir=out_dir,
        ),
        "trajectory": trajectory_bar_figure(trajectory_aggregates, cr_records, out_dir),
    }
    coverage = defaultdict(int)
    for record in temperature_records:
        coverage[f"{record['family']}/{record['split']}/{record['temperature']}"] += 1
    frame_counts = {
        "temperature": frame_point_counts(temperature_records, "temperature", TEMP_ORDER),
        "cr_count": frame_point_counts(cr_records, "cr", CR_ORDER),
        "held_out_trajectory": frame_point_counts(
            [record for record in cr_records if str(record["split"]) == "test"],
            "trajectory",
            ["2L_o3_t3_600K_aimd_1", "2L_tetra_Cr4_600K_aimd_1"],
        ),
    }
    manifest = {
        "status": "RECONSTRUCTION",
        "purpose": "clustered-bar replacements for temperature, Cr-count, and held-out trajectory diagnostics",
        "grammar": {
            "bars": "mean across three seeds",
            "whisker": "sample SD across three seeds",
            "points": "all finite frame-level points jittered around each family/category/split position, plus three black seed-level category aggregates",
            "frame_point_count": "every point recovered from the readable per-frame CSVs or prior SVG point source",
            "train_style": "transparent hatched bar",
            "held_out_test_style": "opaque solid bar",
            "model_colors": "R04 palette",
            "energy_aggregation": energy_aggregation,
            "force_aggregation": force_aggregation,
        },
        "category_definitions": {"temperature": TEMP_LABELS, "cr_count": CR_LABELS},
        "data": {
            "summary_path": str(SUMMARY_PATH),
            "source_summary_path": str(SOURCE_SUMMARY_PATH),
            "read_frame_root": str(FRAME_ROOT),
            "source_frame_root": str(SOURCE_FRAME_ROOT),
            "per_frame_csv_count": len(list(SOURCE_FRAME_ROOT.glob("*__per_frame.csv"))),
            "source_mode": source_mode,
            "vector_recovery_note": (
                "Prior SVG point vectors were used because some canonical CSVs are iCloud dataless placeholders; "
                "the SVG stores all plotted frame points but not n_atoms or original seed IDs."
                if vector_dir is not None
                else None
            ),
            "families": FAMILIES,
            "seeds": SEEDS,
            "splits": SPLITS,
            "test_trajectory_identities": ["2L_o3_t3_600K_aimd_1", "2L_tetra_Cr4_600K_aimd_1"],
            "coverage_counts": dict(coverage),
            "frame_point_counts": frame_counts,
            "loto_status": "NOT_PERFORMED",
            "loto_requirement": "requires retraining/evaluation with trajectory-group holdout; fixed-split predictions are not LOTO",
        },
        "sources": [{"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in sources],
        "outputs": {
            name: {
                ext: {"path": path, "sha256": sha256(Path(path)), "size_bytes": Path(path).stat().st_size}
                for ext, path in files.items()
            }
            for name, files in outputs.items()
        },
    }
    manifest_path = out_dir / "figure02_clustered_v5_full_points_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    caption_path = out_dir / "figure02_clustered_v5_full_points_caption.md"
    caption_path.write_text(
        "# Figure 2 clustered-bar diagnostics with full frame coverage\n\n"
        "Temperature and Cr-count are categorical classes. Every finite frame-level value is plotted as a semi-transparent model-colored point at its corresponding train/test and category position; the point cloud is therefore the complete available frame coverage rather than three seed markers. The bar is the mean of the three seed-level category RMSE values, with sample SD whiskers and the three black seed points overlaid. Transparent hatched bars are train and opaque solid bars are held-out test. The held-out trajectory replacement uses only the two trajectory identities present in the fixed test package; missing model/trajectory combinations remain unfilled.\n\n"
        f"Energy aggregation: {energy_aggregation}. Force aggregation: {force_aggregation}. These are fixed-split reconstructions, not leave-one-trajectory-out evaluations.\n"
    )
    return {"outputs": outputs, "manifest": str(manifest_path), "caption": str(caption_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--frame-root", type=Path, default=None, help="optional local readable copy of per-frame CSVs")
    parser.add_argument("--vector-dir", type=Path, default=None, help="recover plotted points from the prior v2 SVG pair")
    args = parser.parse_args()
    global FRAME_ROOT
    if args.frame_root is not None:
        FRAME_ROOT = args.frame_root
    print(json.dumps(build(args.out_dir, args.vector_dir), indent=2))


if __name__ == "__main__":
    main()

