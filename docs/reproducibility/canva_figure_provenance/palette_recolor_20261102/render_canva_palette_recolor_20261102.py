#!/usr/bin/env python3
"""Render/recolor the locally catalogued Canva P01–P10 figure set.

This is intentionally additive. It reuses the existing provenance-aware
builders or their locally rendered vector outputs, applies the requested
semantic palette, and writes a manifest with source hashes and boundaries.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image


CAMPAIGN = Path(__file__).resolve().parent
FIGURES = CAMPAIGN / "figures"
RAW = CAMPAIGN / "raw_render"
WORKTREE = Path("/Users/alina/.codex/worktrees/canva-figure-provenance")
MAIN_REPO = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench")
DRAWIO_BIN = Path("/Applications/draw.io.app/Contents/MacOS/draw.io")

MODEL = {"Foundation": "#8FC7AB", "FT": "#29A7B6", "Scratch": "#327FB1"}
TASK = {
    "Static": "#73869A",
    "MD": "#5C9B7C",
    "In-gap": "#D28A32",
    "Gap-to-QL": "#CD6D62",
    "Within-QL": "#7D5C99",
    "Sliding": "#287E8A",
}
OPERATOR = {"fixed": "#F4BDC3", "self": "#E978A4"}
DFT = "#303840"
STATUS = {"failed": "#BD6472", "warning": "#78838D", "pending": "#A6AFB7"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_module_from_code(name: str, path: Path, code: str):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(code, str(path), "exec"), module.__dict__)
    return module


def configure_atomic(module) -> None:
    module.COLORS.update(
        {
            "Foundation": MODEL["Foundation"],
            "FT-600K": MODEL["FT"],
            "FT-MultiT": MODEL["FT"],
            "Scratch-600K": MODEL["Scratch"],
            "Scratch-MultiT": MODEL["Scratch"],
            "QE reference": DFT,
            "MACE-MP": MODEL["Foundation"],
            "MACE-OMAT-0": MODEL["Foundation"],
            "MACE-MH": MODEL["Foundation"],
        }
    )
    module.TASK_MAP_COLORS = {
        "L0 DFT Reference": TASK["Static"],
        "L1 Thermal Equilibrium": TASK["MD"],
        "L2 NEB Migration": TASK["In-gap"],
        "L3 Layer Sliding": TASK["Sliding"],
    }
    module.TASK_DISPLAY_COLORS = dict(module.TASK_MAP_COLORS)
    module.TASK_SUBLEVEL_COLORS = {
        "Static DFT": TASK["Static"],
        "DFT-AIMD": TASK["MD"],
        "MLIP-MD": TASK["MD"],
        "DFT Fixed": TASK["In-gap"],
        "MLIP Relaxed": OPERATOR["self"],
        "DFT Sliding": TASK["Sliding"],
        "MLIP Sliding": TASK["Sliding"],
    }
    module.ENSEMBLE_COLORS = {
        "From-scratch": MODEL["Scratch"],
        "FT-600K": MODEL["FT"],
        "FT-MultiT": MODEL["FT"],
        "Foundation": MODEL["Foundation"],
    }
    module.GROUPED_COLORS = {
        family: (MODEL["Foundation"] if family == "Foundation" else MODEL["FT"] if family.startswith("FT") else MODEL["Scratch"])
        for family in module.GROUPED_FAMILIES
    }


def copy_raw(source: Path, page: str) -> Path:
    destination = RAW / page / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def style_svg(text: str, replacements: dict[str, str], stroke_styles: dict[str, tuple[str, str]] | None = None) -> str:
    """Apply case-insensitive color changes and optional class line styles."""
    for old, new in replacements.items():
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    for old, (new, dash) in (stroke_styles or {}).items():
        pattern = re.compile(rf"stroke:\s*{re.escape(old)}", flags=re.IGNORECASE)
        text = pattern.sub(f"stroke: {new}; stroke-dasharray: {dash}", text)
    # Color the semantic title text without changing data geometry.
    title_rules = [
        (r"(in-gap diffusion)", TASK["In-gap"]),
        (r"(gap(?:→|-&gt;)QL|gap-to-QL)", TASK["Gap-to-QL"]),
        (r"(within-QL)", TASK["Within-QL"]),
        (r"(Sliding)", TASK["Sliding"]),
        (r"(fixed(?:-path)?|Fixed path)", OPERATOR["fixed"]),
        (r"(self-relaxed(?: path)?|Self-relaxed)", OPERATOR["self"]),
    ]
    for pattern, color in title_rules:
        text = re.sub(
            rf'(<text\b[^>]*style=")([^"].*?)("[^>]*>[^<]*{pattern}[^<]*</text>)',
            lambda match: f'{match.group(1)}{match.group(2)}; fill: {color}{match.group(3)}',
            text,
            flags=re.IGNORECASE,
        )
    return text


def render_png_from_svg(svg: Path, png: Path) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    magick = shutil.which("magick")
    if not magick:
        raise RuntimeError("ImageMagick 'magick' is required to rasterize recolored SVGs")
    raster_svg = svg
    svg_text = svg.read_text(encoding="utf-8")
    # Drawio emits a CSS light-dark() value that ImageMagick's SVG reader does
    # not understand. It is only a page background here, so normalize it in a
    # transient rasterization copy while preserving the authored SVG.
    sanitized = re.sub(r"light-dark\((?:[^()]|\([^()]*\))*\)", "white", svg_text)
    if sanitized != svg_text:
        raster_svg = svg.with_name(f"{svg.stem}.magick.svg")
        raster_svg.write_text(sanitized, encoding="utf-8")
    subprocess.run(
        [magick, "-background", "white", "-density", "120", str(raster_svg), "-alpha", "remove", "-alpha", "off", str(png)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def render_pdf_from_svg(svg: Path, pdf: Path) -> None:
    magick = shutil.which("magick")
    if not magick:
        return
    subprocess.run(
        [magick, "-background", "white", "-density", "180", str(svg), "-alpha", "remove", "-alpha", "off", str(pdf)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def recolor_png_exact(source_png: Path, destination_png: Path, replacements: dict[str, str]) -> None:
    """Recolor solid vector/raster pixels without rerasterizing dense point clouds."""
    image = Image.open(source_png).convert("RGBA")
    array = np.asarray(image).copy()
    for old, new in replacements.items():
        old_rgb = tuple(int(old[i : i + 2], 16) for i in (1, 3, 5))
        new_rgb = tuple(int(new[i : i + 2], 16) for i in (1, 3, 5))
        mask = np.all(array[:, :, :3] == old_rgb, axis=2)
        array[mask, :3] = new_rgb
    Image.fromarray(array, mode="RGBA").save(destination_png, optimize=True)


def materialize_svg(
    source_svg: Path,
    page: str,
    replacements: dict[str, str],
    stroke_styles=None,
    source_png: Path | None = None,
) -> dict[str, str]:
    copy_raw(source_svg, page)
    page_dir = FIGURES / page
    page_dir.mkdir(parents=True, exist_ok=True)
    destination_svg = page_dir / f"{page}_palette_updated.svg"
    destination_svg.write_text(style_svg(source_svg.read_text(encoding="utf-8"), replacements, stroke_styles), encoding="utf-8")
    destination_png = page_dir / f"{page}_palette_updated.png"
    if source_png is not None and source_png.exists() and source_png.stat().st_size > 0:
        recolor_png_exact(source_png, destination_png, replacements)
    else:
        render_png_from_svg(destination_svg, destination_png)
    return {"png": str(destination_png), "svg": str(destination_svg)}


def model_replacements() -> dict[str, str]:
    return {
        "#CF4520": MODEL["Foundation"],
        "#D5A238": MODEL["Foundation"],
        "#d5a238": MODEL["Foundation"],
        "#D85A45": MODEL["Foundation"],
        "#d85a45": MODEL["Foundation"],
        "#5D8B56": MODEL["Foundation"],
        "#5d8b56": MODEL["Foundation"],
        "#002E71": MODEL["FT"],
        "#67ACE5": MODEL["FT"],
        "#FF9F1B": MODEL["Scratch"],
        "#F4C95D": MODEL["Scratch"],
        "#222222": DFT,
    }


def p02_drawio() -> dict[str, str]:
    source = MAIN_REPO / "analysis/migrationbench_training_design_figure_20261024/make_editable_drawio.py"
    page = "CANVA-P02"
    page_dir = FIGURES / page
    page_dir.mkdir(parents=True, exist_ok=True)
    copy_raw(source, page)
    code = source.read_text(encoding="utf-8")
    for old, new in {
        "#173F5F": DFT,
        "#1F77B4": MODEL["FT"],
        "#F0A23B": MODEL["Scratch"],
        "#7B3F98": MODEL["Foundation"],
        "#176A8A": TASK["Sliding"],
        "#4A6070": TASK["Static"],
        "#FFF5E6": "#F2F5F6",
    }.items():
        code = re.sub(re.escape(old), new, code, flags=re.IGNORECASE)
    # The source was authored with a newer f-string parser. Keep the source
    # artifact unchanged, but use an equivalent expression for this runtime.
    code = code.replace(
        "edges.append(f'<mxCell {' '.join(attrs)}>{''.join(geometry)}</mxCell>')",
        'edges.append("<mxCell " + " ".join(attrs) + ">" + "".join(geometry) + "</mxCell>")',
    )
    drawio = page_dir / f"{page}_palette_updated.drawio"
    namespace = {"__file__": str(page_dir / "generated_p02.py"), "__name__": "__main__"}
    exec(compile(code, str(source), "exec"), namespace)
    generated = page_dir / "migrationbench_full_pipeline_two_distributions_three_models_editable.drawio"
    if generated.exists() and generated != drawio:
        generated.replace(drawio)
    if not DRAWIO_BIN.exists():
        raise FileNotFoundError(DRAWIO_BIN)
    svg = page_dir / f"{page}_palette_updated.svg"
    pdf = page_dir / f"{page}_palette_updated.pdf"
    png = page_dir / f"{page}_palette_updated.png"
    for fmt, output in (("svg", svg), ("pdf", pdf)):
        subprocess.run(
            [str(DRAWIO_BIN), "--headless", "--export", "--format", fmt, "--output", str(output), str(drawio)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    render_png_from_svg(svg, png)
    return {"png": str(png), "svg": str(svg), "pdf": str(pdf), "drawio": str(drawio)}


def p01_and_p05(base) -> tuple[dict[str, str], dict[str, str]]:
    configure_atomic(base)
    def fast_save_figure(fig, out: Path, stem: str, dpi: int = 400):
        out.mkdir(parents=True, exist_ok=True)
        path = out / stem
        fig.savefig(path.with_suffix(".png"), dpi=120, bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return {ext: str(path.with_suffix(f".{ext}")) for ext in ("png", "pdf", "svg")}
    base.save_figure = fast_save_figure
    out1 = CAMPAIGN / "builder_output" / "P01"
    out5 = CAMPAIGN / "builder_output" / "P05"
    assets1, _ = base.fig1(out1)
    assets5, _ = base.fig6(out5)
    p01_replacements = {
        **model_replacements(),
        "#8DA9BF": TASK["Static"],
        "#8da9bf": TASK["Static"],
        "#6E9E75": TASK["MD"],
        "#6e9e75": TASK["MD"],
        "#C4547A": TASK["In-gap"],
        "#c4547a": TASK["In-gap"],
        "#E68F1D": TASK["Sliding"],
        "#e68f1d": TASK["Sliding"],
        "#7658C7": TASK["MD"],
        "#2D7FB5": TASK["In-gap"],
        "#E8792F": TASK["Sliding"],
        "#9A6A00": OPERATOR["self"],
        "#D5A238": OPERATOR["self"],
    }
    p01 = materialize_svg(Path(assets1["svg"]), "CANVA-P01", p01_replacements, source_png=Path(assets1["png"]))
    p05 = materialize_svg(
        Path(assets5["svg"]),
        "CANVA-P05",
        {**model_replacements(), "#9A6A00": STATUS["warning"], "#9a6a00": STATUS["warning"], "#B23A48": STATUS["failed"]},
        {"#002E71": (MODEL["FT"], "7,3"), "#67ACE5": (MODEL["FT"], "2,2"), "#FF9F1B": (MODEL["Scratch"], "7,3"), "#F4C95D": (MODEL["Scratch"], "2,2")},
        source_png=Path(assets5["png"]),
    )
    return p01, p05


def p03_and_p04(base) -> tuple[dict[str, str], dict[str, str]]:
    configure_atomic(base)
    scripts = WORKTREE / "docs/reproducibility/canva_figure_provenance/scripts"
    blue_cmap = LinearSegmentedColormap.from_list("canva_blue_density", ["#E8F1F5", "#BDD7E3", "#80B5CB", "#4382A7", "#174A70"])
    mpl.colormaps.register(blue_cmap, name="canva_blue_density")
    parity_path = scripts / "build_slide3_energy_force_train_test_separated.py"
    parity_code = parity_path.read_text(encoding="utf-8")
    parity_code = parity_code.replace('cmap="magma"', 'cmap="canva_blue_density"')
    parity_code = parity_code.replace('mpl.colormaps["magma"]', 'mpl.colormaps["canva_blue_density"]')
    parity_code = parity_code.replace('dpi=320', 'dpi=140')
    parity = load_module_from_code("canva_slide3_palette", parity_path, parity_code)
    parity.COLORS = base.COLORS
    panels, counts, stats = parity.load_panels()
    out3 = CAMPAIGN / "builder_output" / "P03"
    assets3, _, _, _ = parity.render(panels, counts, stats, out_root=out3)
    p03 = materialize_svg(Path(assets3["svg"]), "CANVA-P03", {**model_replacements(), "#cf2e2e": DFT}, source_png=Path(assets3["png"]))

    frame = load_module("canva_frame_palette", scripts / "build_figure02_latest.py")
    frame.v9.base.COLORS = base.COLORS
    frame.v9.COLORS = base.COLORS
    out4 = CAMPAIGN / "builder_output" / "P04"
    built = frame.build(out4)
    overview_svg = Path(built["outputs"]["overview"]["svg"])
    p04 = materialize_svg(overview_svg, "CANVA-P04", model_replacements(), source_png=Path(built["outputs"]["overview"]["png"]))
    return p03, p04


def recolor_existing_pages() -> dict[str, dict[str, str]]:
    existing = {
        "CANVA-P06": MAIN_REPO / "analysis/cr_interlayer_sliding_grouped_20261025/results/cr_three_path_fixed_vs_self_2x3_R02.svg",
        "CANVA-P07": MAIN_REPO / "analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/archive/legacy_figure_products_20261024/figures/foundation_class_3x3_R03/foundation_model_class_3x3_R03.svg",
        "CANVA-P08": MAIN_REPO / "analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/archive/legacy_figure_products_20261024/figures/foundation_class_4x3_R03/foundation_model_class_4x3_R03.svg",
        "CANVA-P09": MAIN_REPO / "analysis/cr_interlayer_sliding_grouped_20261025/results/figures/plot_atlas_grouped_cr_r04/cr_interlayer_sliding_grouped_2x2_plot_atlas_r04.svg",
        "CANVA-P10": MAIN_REPO / "analysis/foundation_interlayer_sliding_10model_20261024/results/figures/plot_atlas_grouped_r04/interlayer_sliding_10model_grouped_2x2_plot_atlas_r04.svg",
    }
    outputs = {}
    for page, source in existing.items():
        if not source.exists():
            raise FileNotFoundError(source)
        replacements = model_replacements()
        if page == "CANVA-P06":
            styles = {
                "#002E71": (MODEL["FT"], "7,3"),
                "#67ACE5": (MODEL["FT"], "2,2"),
                "#FF9F1B": (MODEL["Scratch"], "7,3"),
            }
        elif page in {"CANVA-P07", "CANVA-P08", "CANVA-P10"}:
            styles = {
                "#D5A238": (MODEL["Foundation"], "7,3"),
                "#D85A45": (MODEL["Foundation"], "4,2"),
                "#5D8B56": (MODEL["Foundation"], "2,2"),
            }
        else:
            styles = {
                "#002E71": (MODEL["FT"], "7,3"),
                "#67ACE5": (MODEL["FT"], "2,2"),
                "#FF9F1B": (MODEL["Scratch"], "7,3"),
                "#F4C95D": (MODEL["Scratch"], "2,2"),
            }
        outputs[page] = materialize_svg(source, page, replacements, styles, source_png=source.with_suffix(".png"))
    return outputs


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    base_path = WORKTREE / "docs/reproducibility/canva_figure_provenance/scripts/build_atomic_migration_figures.py"
    base = load_module("build_atomic_migration_figures", base_path)
    manifest = {
        "campaign_id": "canva_palette_recolor_20261102",
        "historical_catalog_scope": "CANVA-P01 through CANVA-P10",
        "current_design": {"design_id": "DAHZXpLtN6U", "status": "REAUTH_REQUIRED", "page_count": None},
        "palette_file": str(CAMPAIGN / "PALETTE_STANDARD.json"),
        "pages": {},
    }

    p02 = p02_drawio()
    manifest["pages"]["CANVA-P02"] = {
        "title": "Do foundation models help to improve MLIP?",
        "status": "REPRODUCED_EDITABLE_DRAWIO",
        "source_scripts": [str(MAIN_REPO / "analysis/migrationbench_training_design_figure_20261024/make_editable_drawio.py")],
        "outputs": p02,
        "notes": "Editable Drawio source regenerated with Foundation/FT/Scratch semantic colors; workflow connectors remain subdued teal/gray.",
    }

    p01, p05 = p01_and_p05(base)
    manifest["pages"]["CANVA-P01"] = {
        "title": "MigrationBench configuration universe + model workflow",
        "status": "REPRODUCED_FROM_PACKAGED_DATA",
        "source_scripts": [str(base_path)],
        "outputs": p01,
        "notes": "Task colors are remapped to Static/MD/NEB/Sliding; split markers remain shape-coded.",
    }
    manifest["pages"]["CANVA-P05"] = {
        "title": "MD (1 ns)",
        "status": "REPRODUCED_FROM_PACKAGED_DATA",
        "source_scripts": [str(base_path)],
        "outputs": p05,
        "notes": "FT temperatures share teal and Scratch temperatures share blue; temperature distinction is retained by line/marker grammar.",
    }

    p03, p04 = p03_and_p04(base)
    manifest["pages"]["CANVA-P03"] = {
        "title": "RMSE / energy-force parity",
        "status": "LOCAL_RECONSTRUCTION",
        "source_scripts": [str(WORKTREE / "docs/reproducibility/canva_figure_provenance/scripts/build_slide3_energy_force_train_test_separated.py")],
        "outputs": p03,
        "notes": "Historical Canva PDF is pointer-only. Density is the requested light-blue to dark-blue gradient; model strategy is encoded in title color.",
    }
    manifest["pages"]["CANVA-P04"] = {
        "title": "Frame-level error distributions",
        "status": "LOCAL_RECONSTRUCTION",
        "source_scripts": [str(WORKTREE / "docs/reproducibility/canva_figure_provenance/scripts/build_figure02_latest.py")],
        "outputs": p04,
        "notes": "Historical Canva PDF is pointer-only. Frame-level train/test records remain auditable in the packaged data manifest.",
    }

    existing_page_sources = {
        "CANVA-P06": MAIN_REPO / "analysis/cr_interlayer_sliding_grouped_20261025/make_cr_three_path_2x3_R02.py",
        "CANVA-P07": MAIN_REPO / "analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/archive/legacy_figure_products_20261024/build_foundation_class_3x3_r01.py",
        "CANVA-P08": MAIN_REPO / "analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/archive/legacy_figure_products_20261024/build_foundation_class_4x3_r03.py",
        "CANVA-P09": MAIN_REPO / "analysis/cr_interlayer_sliding_grouped_20261025/scripts/build_cr_grouped_r04.py",
        "CANVA-P10": MAIN_REPO / "analysis/foundation_interlayer_sliding_10model_20261024/scripts/plot_interlayer_grouped_profile_atlas_r01.py",
    }
    existing_page_assets = {
        "CANVA-P06": MAIN_REPO / "analysis/cr_interlayer_sliding_grouped_20261025/results/cr_three_path_fixed_vs_self_2x3_R02.svg",
        "CANVA-P07": MAIN_REPO / "analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/archive/legacy_figure_products_20261024/figures/foundation_class_3x3_R03/foundation_model_class_3x3_R03.svg",
        "CANVA-P08": MAIN_REPO / "analysis/neb_cross_element_triptychs_20261024/foundation_self_relaxed_4x3_20261024/archive/legacy_figure_products_20261024/figures/foundation_class_4x3_R03/foundation_model_class_4x3_R03.svg",
        "CANVA-P09": MAIN_REPO / "analysis/cr_interlayer_sliding_grouped_20261025/results/figures/plot_atlas_grouped_cr_r04/cr_interlayer_sliding_grouped_2x2_plot_atlas_r04.svg",
        "CANVA-P10": MAIN_REPO / "analysis/foundation_interlayer_sliding_10model_20261024/results/figures/plot_atlas_grouped_r04/interlayer_sliding_10model_grouped_2x2_plot_atlas_r04.svg",
    }
    recolored = recolor_existing_pages()
    manifest["pages"].update({
        page: {
            "status": "PALETTE_RECOLOR_OF_LOCAL_RENDERED_ARTIFACT",
            "source_scripts": [str(existing_page_sources[page])],
            "source_assets": [str(existing_page_assets[page])],
            "outputs": outputs,
            "notes": "Existing local vector render was preserved and recolored; raw input boundary remains as recorded by the source package.",
        }
        for page, outputs in recolored.items()
    })

    for page_data in manifest["pages"].values():
        page_data["source_scripts"] = [
            {"path": path, "sha256": sha256(Path(path))}
            for path in page_data.get("source_scripts", [])
            if Path(path).exists()
        ]
        page_data["source_assets"] = [
            {"path": path, "sha256": sha256(Path(path))}
            for path in page_data.get("source_assets", [])
            if Path(path).exists()
        ]
        page_data["outputs"] = {
            ext: {"path": path, "sha256": sha256(Path(path)), "bytes": Path(path).stat().st_size}
            for ext, path in page_data["outputs"].items()
            if Path(path).exists()
        }
    (CAMPAIGN / "RENDER_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"campaign": manifest["campaign_id"], "pages": sorted(manifest["pages"]), "manifest": str(CAMPAIGN / "RENDER_MANIFEST.json")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
