#!/usr/bin/env python3
"""Re-render the supplied Figure 2 variant and Slide 3 with the shared palette.

This is an additive wrapper around the existing reproducible figure builders.
The data and panel geometry are inherited from the source builders; only the
visual encoding is changed here:

* Foundation, Fine-tuning and Scratch use the publication strategy colors.
* 600 K is open and Multi-T is filled wherever a model marker/box is drawn.
* Seed identity remains a hatch in the RMSE boxplots.
* Parity density uses one shared light-blue to deep-blue count scale, not a strategy color.
"""

from __future__ import annotations

import hashlib
import json
import sys
from copy import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.patches import Patch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea


R03_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = R03_ROOT / "figures"
SOURCE_SCRIPTS = Path(
    "/Users/alina/.codex/worktrees/paper-figures-repro-release/"
    "docs/reproducibility/canva_figure_provenance/scripts"
)
sys.path.insert(0, str(SOURCE_SCRIPTS))

import _figure02_latest_layout as v12  # noqa: E402
import _figure02_latest_panels as v11  # noqa: E402
import _figure02_latest_seed as v9  # noqa: E402
import build_slide3_energy_force_train_test_separated as slide3  # noqa: E402


_REFERENCE_AXIS = v12._reference_axis
_SLIDE3_STYLE_AXIS = slide3.style_axis
_SLIDE3_ADD_VECTOR_COLORBAR = slide3.add_vector_colorbar
_AXES_HEXBIN = Axes.hexbin
_AXES_PLOT = Axes.plot
_AXES_SET_TITLE = Axes.set_title
_FIGURE_LEGEND = Figure.legend
_FIGURE_TEXT = Figure.text


PALETTE = {
    "Foundation": "#8FC7AB",
    "Fine-tuning": "#29A7B6",
    "Scratch": "#327FB1",
    "DFT": "#303840",
}


def darken(color: str, factor: float = 0.58) -> str:
    """Return a darker version of a palette color for outlines and lines."""
    rgb = np.asarray(mpl.colors.to_rgb(color), dtype=float) * factor
    return mpl.colors.to_hex(np.clip(rgb, 0.0, 1.0), keep_alpha=False)


STRATEGY_DARK = {strategy: darken(color) for strategy, color in PALETTE.items() if strategy != "DFT"}
PANEL_EDGE = "#1C5968"
SLIDE3_NEUTRAL_EDGE = "#6E7781"
SLIDE3_COUNT_COLORS = ("#E8F1F5", "#BDD7E3", "#80B5CB", "#4382A7", "#174A70")
SLIDE3_COUNT_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "migrationbench_count_blues", SLIDE3_COUNT_COLORS, N=256
)
FAMILY_COLORS = {
    "Scratch-600K": PALETTE["Scratch"],
    "FT-600K": PALETTE["Fine-tuning"],
    "Scratch-MultiT": PALETTE["Scratch"],
    "FT-MultiT": PALETTE["Fine-tuning"],
}
FAMILY_TEMPERATURE = {
    "Scratch-600K": "600 K",
    "FT-600K": "600 K",
    "Scratch-MultiT": "Multi-T",
    "FT-MultiT": "Multi-T",
}
# All three seeds use an explicit texture so seed 123 is not visually lost:
# horizontal lines = 123, forward slashes = 234, backslashes = 345.
SEED_HATCH = {"123": "---", "234": "///", "345": "\\\\\\"}

INPUT_PDFS = [
    Path("/Users/alina/Downloads/variant04_temperature_plus_scratch_ft_shared_legend_square (1).pdf"),
    Path("/Users/alina/Downloads/slide3_energy_force_train_test_separated (1).pdf"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def family_strategy(family: str) -> str:
    return "Fine-tuning" if family.startswith("FT-") else "Scratch"


def family_edge(family: str) -> str:
    return STRATEGY_DARK[family_strategy(family)]


def semantic_box_and_points(
    ax,
    raw: np.ndarray,
    xpos: float,
    *,
    color: str,
    seed: str,
    family: str,
    category: str,
    metric: str,
    box_width: float,
) -> None:
    """Draw an RMSE box cluster with strategy color + temperature fill."""
    if raw.size == 0:
        return
    logged = np.log10(raw)
    strategy_color = PALETTE[family_strategy(family)]
    edge_color = family_edge(family)
    open_temperature = FAMILY_TEMPERATURE[family] == "600 K"
    facecolor = "white" if open_temperature else strategy_color
    rng = np.random.default_rng(v9.base._stable_seed("semantic-seed-box", metric, family, category, seed))
    jitter = rng.uniform(-box_width * 0.43, box_width * 0.43, raw.size)
    ax.scatter(
        np.full(raw.size, xpos) + jitter,
        logged,
        s=5.0,
        facecolors=facecolor,
        edgecolors=edge_color,
        linewidths=0.45,
        alpha=0.42 if open_temperature else 0.28,
        zorder=1,
    )
    result = ax.boxplot(
        [logged],
        positions=[xpos],
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        whis=1.5,
        boxprops={
            "facecolor": facecolor,
            "edgecolor": edge_color,
            "linewidth": 1.85,
            "alpha": 0.92 if not open_temperature else 1.0,
        },
        medianprops={"color": edge_color, "linewidth": 2.2},
        whiskerprops={"color": edge_color, "linewidth": 1.85},
        capprops={"color": edge_color, "linewidth": 1.85},
    )
    for patch in result["boxes"]:
        patch.set_hatch(SEED_HATCH[seed])
        patch.set_zorder(4)
        # Hatch strokes are a separate semantic channel for the random seed.
        # Overlay them after the filled box so dense frame points and the
        # Multi-T face color cannot visually suppress the seed encoding.
        if SEED_HATCH[seed]:
            hatch_overlay = copy(patch)
            hatch_overlay.set_facecolor("none")
            hatch_overlay.set_edgecolor(edge_color)
            hatch_overlay.set_linewidth(1.95)
            hatch_overlay.set_alpha(1.0)
            hatch_overlay.set_zorder(12)
            ax.add_patch(hatch_overlay)


def draw_overview_axes(fig, axes, records: list[dict[str, object]], *, add_labels: bool = True) -> dict[str, object]:
    """Copy the canonical overview calculation with semantic fills."""
    metrics = [("energy", "Energy RMSE (meV atom$^{-1}$)"), ("force", "Force RMSE (meV Å$^{-1}$)")]
    comparisons = v9._comparison_pairs(v9.FAMILIES)
    comparison_manifest: dict[str, object] = {}
    for row_index, (metric, ylabel) in enumerate(metrics):
        ax = axes[row_index]
        raw_by_family = {family: v11._overview_raw(records, family, metric) for family in v9.FAMILIES}
        seed_values = {
            family: [v11._seed_aggregate(records, family, seed, metric) for seed in v9.SEEDS]
            for family in v9.FAMILIES
        }
        combined = np.concatenate([np.log10(raw) for raw in raw_by_family.values()])
        combined = np.concatenate([combined] + [np.log10(np.asarray(values)) for values in seed_values.values()])
        low = float(np.floor(np.nanmin(combined) - 0.18))
        raw_high = float(np.nanmax(combined))
        high = raw_high + 0.45 + v11.BRACKET_STEP * len(comparisons)
        for family in v9.FAMILIES:
            raw = raw_by_family[family]
            logged = np.log10(raw)
            xpos = v11.OVERVIEW_FAMILY_X[family]
            strategy_color = PALETTE[family_strategy(family)]
            edge_color = family_edge(family)
            open_temperature = FAMILY_TEMPERATURE[family] == "600 K"
            facecolor = "white" if open_temperature else strategy_color
            rng = np.random.default_rng(v9.base._stable_seed("semantic-overview", metric, family))
            jitter = rng.uniform(-v11.OVERVIEW_BOX_WIDTH * 0.42, v11.OVERVIEW_BOX_WIDTH * 0.42, raw.size)
            ax.scatter(
                np.full(raw.size, xpos) + jitter,
                logged,
                s=5.0,
                facecolors=facecolor,
                edgecolors=edge_color,
                linewidths=0.45,
                alpha=0.42 if open_temperature else 0.28,
                zorder=1,
            )
            box = ax.boxplot(
                [logged],
                positions=[xpos],
                widths=v11.OVERVIEW_BOX_WIDTH,
                patch_artist=True,
                showfliers=False,
                manage_ticks=False,
                whis=1.5,
                boxprops={
                    "facecolor": facecolor,
                    "edgecolor": edge_color,
                    "linewidth": 2.0,
                    "alpha": 0.92 if not open_temperature else 1.0,
                },
                medianprops={"color": edge_color, "linewidth": 2.7},
                whiskerprops={"color": edge_color, "linewidth": 2.0},
                capprops={"color": edge_color, "linewidth": 2.0},
            )
            for patch in box["boxes"]:
                patch.set_zorder(4)
            center, lower, upper = v9._seed_summary(seed_values[family])
            ax.errorbar(
                [xpos],
                [center],
                yerr=[[center - lower], [upper - center]],
                fmt="none",
                ecolor=edge_color,
                elinewidth=2.4,
                capsize=5.0,
                capthick=2.4,
                zorder=6,
            )
            ax.scatter([xpos], [center], s=48, facecolors=facecolor, edgecolors=edge_color, linewidths=1.8, zorder=7)

        comparison_manifest[metric] = {}
        for level, (family_a, family_b, description) in enumerate(comparisons):
            pvalue = v9._paired_p(seed_values[family_a], seed_values[family_b])
            comparison_manifest[metric][description] = {
                "p_value": pvalue,
                "marker": v9._sig_label(pvalue),
                "n_seed_pairs": 3,
            }
            if np.isfinite(pvalue):
                v9._draw_bracket(
                    ax,
                    v11.OVERVIEW_FAMILY_X[family_a],
                    v11.OVERVIEW_FAMILY_X[family_b],
                    raw_high + 0.18 + level * v11.BRACKET_STEP,
                    v9._sig_label(pvalue),
                )

        v9._set_log_exponent_axis(ax, (low, high))
        ax.set_xlim(-0.58, 1.98)
        ax.set_xticks([0.0, 1.4], ["Scratch", "FT"])
        if row_index == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("model lineage · 600K open / Multi-T filled", labelpad=20)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{chr(65 + row_index)}  {ylabel.split(' RMSE')[0]}", loc="left", pad=12)
        v11._square_axis(ax)
        for spine in ax.spines.values():
            spine.set_color(PANEL_EDGE)
            spine.set_linewidth(2.2)
    return comparison_manifest


def family_strategy_handles() -> list[Patch]:
    present = {family_strategy(family) for family in v9.FAMILIES}
    return [
        Patch(facecolor=PALETTE[strategy], edgecolor=STRATEGY_DARK[strategy], linewidth=1.0, label=strategy)
        for strategy in ("Foundation", "Fine-tuning", "Scratch")
        if strategy in present
    ]


def temperature_handles() -> list[Patch]:
    return [
        Patch(facecolor="white", edgecolor=PANEL_EDGE, linewidth=1.3, label="600 K (open)"),
        Patch(facecolor="#AAB2BA", edgecolor=PANEL_EDGE, linewidth=1.3, label="Multi-T (filled)"),
    ]


def seed_handles() -> list[Patch]:
    return [
        Patch(facecolor="white", edgecolor=PANEL_EDGE, linewidth=1.3, hatch=SEED_HATCH[seed], label=f"seed {seed}")
        for seed in v9.SEEDS
    ]


def colorized_reference_axis(ax) -> None:
    _REFERENCE_AXIS(ax)
    for spine in ax.spines.values():
        if spine.get_visible():
            spine.set_color(PANEL_EDGE)
            spine.set_linewidth(2.2)


def colorized_temperature_group_headers(ax, low: float, categories: list[str], *, show_top_labels: bool) -> None:
    """Use the shared dark palette edge for broken-axis group rules."""
    for category in categories:
        center = v12.TEMP_CENTERS[category]
        start = center - v12.GROUP_HALF_WIDTH
        end = center + v12.GROUP_HALF_WIDTH
        if show_top_labels:
            ax.plot(
                [start, end],
                [1.025, 1.025],
                transform=ax.get_xaxis_transform(),
                color=PANEL_EDGE,
                linewidth=4.2,
                solid_capstyle="butt",
                clip_on=False,
                zorder=10,
            )
            ax.text(
                center,
                1.055,
                v9.TEMP_LABELS[category],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=21,
                fontweight="bold",
                color=PANEL_EDGE,
                clip_on=False,
                zorder=10,
            )
        ax.plot(
            [start, end],
            [low, low],
            color=PANEL_EDGE,
            linewidth=2.2,
            solid_capstyle="butt",
            clip_on=False,
            zorder=9,
        )


def colorized_slide3_style_axis(ax) -> None:
    _SLIDE3_STYLE_AXIS(ax)
    for spine in ax.spines.values():
        spine.set_color(SLIDE3_NEUTRAL_EDGE)
        spine.set_linewidth(1.45)
    ax.tick_params(axis="both", colors=SLIDE3_NEUTRAL_EDGE)
    ax.xaxis.label.set_color(SLIDE3_NEUTRAL_EDGE)
    ax.yaxis.label.set_color(SLIDE3_NEUTRAL_EDGE)


def colorized_slide3_colorbar(fig, axes: np.ndarray, norm, label: str) -> None:
    """Add a vector colorbar using the same sequential blue count map."""
    top = axes[0, -1].get_position().y1
    bottom = axes[-1, -1].get_position().y0
    cax = fig.add_axes([0.925, bottom, 0.017, top - bottom])
    cax.set_yscale("log")
    cax.set_xlim(0.0, 1.0)
    cax.set_ylim(norm.vmin, norm.vmax)
    edges = np.geomspace(norm.vmin, norm.vmax, 72)
    for lower, upper in zip(edges[:-1], edges[1:]):
        cax.add_patch(
            mpl.patches.Rectangle(
                (0.0, lower),
                1.0,
                upper - lower,
                facecolor=SLIDE3_COUNT_CMAP(norm(np.sqrt(lower * upper))),
                edgecolor="none",
                transform=cax.transData,
            )
        )
    cax.xaxis.set_visible(False)
    cax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
    cax.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0))
    cax.yaxis.tick_right()
    cax.tick_params(
        axis="y",
        which="major",
        labelsize=slide3.COLORBAR_SIZE,
        width=1.0,
        length=5,
        pad=5,
        colors=SLIDE3_NEUTRAL_EDGE,
    )
    cax.set_ylabel(label, fontsize=slide3.COLORBAR_SIZE, fontweight="bold", labelpad=32, rotation=270)
    cax.yaxis.set_label_position("right")
    for tick in cax.get_yticklabels():
        tick.set_fontfamily(slide3.FONT)
        tick.set_fontsize(slide3.COLORBAR_SIZE)
        tick.set_fontweight("bold")
        tick.set_color(SLIDE3_NEUTRAL_EDGE)
    cax.yaxis.label.set_color(SLIDE3_NEUTRAL_EDGE)
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.25)
        spine.set_color(SLIDE3_NEUTRAL_EDGE)


def colorized_slide3_hexbin(self, *args, **kwargs):
    """Use one perceptually ordered blue count scale for every parity bin."""
    if kwargs.get("cmap") == "magma":
        kwargs["cmap"] = SLIDE3_COUNT_CMAP
    return _AXES_HEXBIN(self, *args, **kwargs)


def colorized_slide3_plot(self, *args, **kwargs):
    """Use the neutral DFT color for every y=x reference line."""
    if kwargs.get("color") == "#cf2e2e":
        kwargs["color"] = PALETTE["DFT"]
    return _AXES_PLOT(self, *args, **kwargs)


def colorized_slide3_title(self, label="", *args, **kwargs):
    """Render panel titles as neutral letter + colored strategy + neutral temperature."""
    if isinstance(label, str):
        family = next((name for name in slide3.FAMILIES if name in label), None)
        if family is not None:
            letter = label.split()[0]
            strategy = family_strategy(family)
            temperature = "600 K" if family.endswith("600K") else "Multi-T"
            glyph = "○" if temperature == "600 K" else "●"
            fontsize = kwargs.get("fontsize", slide3.PANEL_TITLE)
            fontweight = kwargs.get("fontweight", "bold")
            fontfamily = kwargs.get("fontfamily", slide3.FONT)
            neutral_props = FontProperties(family=fontfamily, size=fontsize, weight=fontweight)
            strategy_props = FontProperties(family=fontfamily, size=fontsize, weight=fontweight)
            children = [
                TextArea(letter, textprops={"color": PALETTE["DFT"], "fontproperties": neutral_props}),
                TextArea(
                    f"  {strategy}",
                    textprops={"color": PALETTE[strategy], "fontproperties": strategy_props},
                ),
                TextArea(
                    f" · {glyph} {temperature}",
                    textprops={"color": PALETTE["DFT"], "fontproperties": neutral_props},
                ),
            ]
            packed = HPacker(children=children, align="baseline", pad=0, sep=0)
            title_box = AnchoredOffsetbox(
                loc="lower left",
                child=packed,
                bbox_to_anchor=(0.0, 1.018),
                bbox_transform=self.transAxes,
                frameon=False,
                borderpad=0.0,
                pad=0.0,
            )
            self.add_artist(title_box)
            return _AXES_SET_TITLE(self, "", *args, **kwargs)
    return _AXES_SET_TITLE(self, label, *args, **kwargs)


def suppressed_slide3_legend(self, *args, **kwargs):
    """Train/test are already encoded by the separate rows; omit duplicate swatches."""
    return None


def suppressed_slide3_top_title(self, x, y, s, *args, **kwargs):
    """Remove the redundant global title while retaining shared axis labels."""
    if isinstance(s, str) and s.startswith("DFT reference vs MLIP prediction"):
        return None
    return _FIGURE_TEXT(self, x, y, s, *args, **kwargs)


def render_variant04() -> dict[str, str]:
    v9._house_style()
    mpl.rcParams.update({"hatch.linewidth": 1.85})
    v9.base.FRAME_ROOT = v9.FRAME_ROOT
    v9.COLORS.clear()
    v9.COLORS.update(FAMILY_COLORS)
    v12.TEMP_CENTERS = {category: index * 2.15 for index, category in enumerate(v9.TEMP_CATEGORIES)}
    v12.TEMP_FAMILY_OFFSETS = {"Scratch-MultiT": -0.53, "FT-MultiT": 0.53}
    v12.GROUP_HALF_WIDTH = 0.90
    v12._reference_axis = colorized_reference_axis
    v12._draw_temperature_group_headers = colorized_temperature_group_headers
    v9._box_and_points = semantic_box_and_points
    records = v12._physical_records(v12._records())

    fig = plt.figure(figsize=(16.0, 16.0))
    grid = fig.add_gridspec(2, 5, width_ratios=[1.0, 1.0, 1.0, 1.2, 1.2], wspace=0.24, hspace=0.30)
    temp_axes = np.asarray([fig.add_subplot(grid[0, :3]), fig.add_subplot(grid[1, :3])], dtype=object)
    overview_axes = np.asarray([fig.add_subplot(grid[0, 3:]), fig.add_subplot(grid[1, 3:])], dtype=object)
    v12._draw_temperature_strip(temp_axes[0], records, "energy", 0, show_xlabels=False, show_top_labels=True)
    v12._draw_temperature_strip(temp_axes[1], records, "force", 1, show_xlabels=True, show_top_labels=False)
    draw_overview_axes(fig, overview_axes, records, add_labels=False)
    overview_axes[1].set_xlabel("model lineage · 600K open / Multi-T filled", labelpad=18)
    fig.text(0.285, 0.985, "Temperature-resolved MultiT", ha="center", va="top", fontsize=23, fontweight="bold")
    fig.text(0.75, 0.985, "Overall Scratch vs FT", ha="center", va="top", fontsize=23, fontweight="bold")
    handles = family_strategy_handles() + temperature_handles() + seed_handles()
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=8,
        prop={"weight": "bold", "size": 18},
        columnspacing=0.85,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.115, top=0.84)
    stem = "variant04_temperature_plus_scratch_ft_shared_legend_square_palette_updated"
    paths = {ext: OUTPUT_DIR / f"{stem}.{ext}" for ext in ("png", "pdf", "svg")}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["svg"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {key: str(path) for key, path in paths.items()}


def render_slide3() -> tuple[dict[str, str], dict[str, object]]:
    """Render the existing 4x4 parity grid with unified blue density encoding."""
    for mapping in (slide3.base.COLORS, slide3.energy_source.COLORS, slide3.COLORS):
        mapping.clear()
        mapping.update(FAMILY_COLORS)
    slide3.base.setup_style()
    slide3.style_axis = colorized_slide3_style_axis
    slide3.add_vector_colorbar = colorized_slide3_colorbar
    typography_names = (
        "MIN_TEXT",
        "TITLE_SIZE",
        "PANEL_TITLE",
        "AXIS_LABEL",
        "TICK_SIZE",
        "ANNOTATION",
        "LEGEND_SIZE",
        "COLORBAR_SIZE",
    )
    original_typography = {name: getattr(slide3, name) for name in typography_names}
    for name, value in original_typography.items():
        setattr(slide3, name, value + 2)
    mpl.rcParams.update({"font.size": slide3.MIN_TEXT, "axes.labelsize": slide3.AXIS_LABEL, "axes.titlesize": slide3.PANEL_TITLE})
    Axes.hexbin = colorized_slide3_hexbin
    Axes.plot = colorized_slide3_plot
    Axes.set_title = colorized_slide3_title
    Figure.legend = suppressed_slide3_legend
    Figure.text = suppressed_slide3_top_title
    try:
        panels, counts, stats = slide3.load_panels()
        outputs, extents, norms, visible_counts = slide3.render(panels, counts, stats, out_root=R03_ROOT)
    finally:
        for name, value in original_typography.items():
            setattr(slide3, name, value)
        Axes.hexbin = _AXES_HEXBIN
        Axes.plot = _AXES_PLOT
        Axes.set_title = _AXES_SET_TITLE
        Figure.legend = _FIGURE_LEGEND
        Figure.text = _FIGURE_TEXT
    renamed: dict[str, str] = {}
    stem = "slide3_energy_force_train_test_separated_palette_updated"
    for extension, old_path in outputs.items():
        source = Path(old_path)
        target = OUTPUT_DIR / f"{stem}.{extension}"
        source.replace(target)
        renamed[extension] = str(target)
    return renamed, {
        "counts": counts,
        "visible_counts": visible_counts,
        "extents": extents,
        "count_norm": norms,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    variant_outputs = render_variant04()
    slide_outputs, slide_metadata = render_slide3()
    source_scripts = [
        SOURCE_SCRIPTS / "_figure02_latest_seed.py",
        SOURCE_SCRIPTS / "_figure02_latest_layout.py",
        SOURCE_SCRIPTS / "_figure02_latest_panels.py",
        SOURCE_SCRIPTS / "build_slide3_energy_force_train_test_separated.py",
        SOURCE_SCRIPTS / "build_slide3_nature_density.py",
        SOURCE_SCRIPTS / "build_slide3_energy_train_test_cluster_centered_heatmap.py",
    ]
    outputs = {**{f"variant04_{key}": value for key, value in variant_outputs.items()}, **{f"slide3_{key}": value for key, value in slide_outputs.items()}}
    manifest = {
        "render_revision": "palette_recolor_20261102_r03",
        "status": "RECONSTRUCTION",
        "input_pdfs": {str(path): sha256(path) for path in INPUT_PDFS if path.exists()},
        "source_scripts": {str(path): sha256(path) for path in source_scripts if path.exists()},
        "palette": PALETTE,
        "family_mapping": FAMILY_COLORS,
        "semantic_encoding": {
            "strategy": "Foundation=#8FC7AB, Fine-tuning=#29A7B6, Scratch=#327FB1",
            "temperature": "600 K=open/white; Multi-T=filled with strategy color",
            "seed": "RMSE seed 123/234/345 are hatch patterns; hatch strokes are overlaid at zorder=12 with linewidth=1.85",
            "parity_density": "all Slide3 energy and force hexbins use the shared sequential count gradient #E8F1F5 -> #BDD7E3 -> #80B5CB -> #4382A7 -> #174A70; energy and force retain independent count normalizations",
            "reference": "DFT/reference is #303840 where present in the source grammar",
            "slide3_titles": "panel titles use dark-gray letters and temperature glyphs (open circle=600 K, filled circle=Multi-T) with Scratch=#327FB1 and Fine-tuning=#29A7B6",
            "slide3_reference_line": "y=x reference lines use dashed DFT dark gray #303840",
            "slide3_axes": "Slide3 axes, ticks, panel borders, and count-colorbar borders use neutral gray #6E7781",
            "slide3_split_legend": "duplicate train/test swatch legend removed because train/test are encoded by separate rows and retained row labels",
            "slide3_typography": "all Slide3 visible text sizes increased by 2 pt; redundant global title removed",
            "outline_style": "all figure/panel borders and model-specific box lines use a darker version of the corresponding strategy color rather than pure black",
            "overview_box_lines": "right-side Energy/Force box borders, whiskers, caps, medians, and mean/error markers use darkened strategy colors with mean/error linewidth=2.4; overview points retain strategy colors",
        },
        "outputs": outputs,
        "output_hashes": {key: sha256(Path(value)) for key, value in outputs.items()},
        "slide3_metadata": slide_metadata,
    }
    manifest_path = R03_ROOT / "PALETTE_RECOLOR_R03_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"outputs": outputs, "manifest": str(manifest_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
