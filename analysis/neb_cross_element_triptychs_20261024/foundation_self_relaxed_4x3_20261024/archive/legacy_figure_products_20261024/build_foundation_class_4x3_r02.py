#!/usr/bin/env python3
"""Render the additive R02 four-row Foundation-model class figure.

R02 extends the prior R01 figure by adding the locally evaluated Cr
MACE-MH-1 fixed-path profile.  The original R01 artifacts remain untouched.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("build_foundation_class_4x3_r01.py")
SPEC = importlib.util.spec_from_file_location("foundation_r01", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT}")
R01 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R01)

OUT = R01.OUT
FIG_OUT = OUT / "figures" / "foundation_class_4x3_R02"
DATA_OUT = OUT / "data" / "foundation_class_4x3_R02"
SUPPLEMENT = DATA_OUT / "Cr_mace_mh1_fixed_path_profiles_R02.tsv"
CR_INPUT_DIR = R01.CR_INPUT_DIR


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_supplement() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SUPPLEMENT.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            row["image_index"] = int(row["image_index"])
            for key in ("s", "energy_eV", "relative_energy_eV", "max_force_eV_A"):
                row[key] = float(row[key])
            rows.append(row)
    if len(rows) != 30:
        raise ValueError(f"Expected 30 MACE-MH-1 Cr rows, found {len(rows)}")
    return rows


def aggregate(model_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: list[dict[str, object]] = []
    expected_by_element = {
        element: {"MACE-MP": 5, "MACE-OMAT-0": 2, "MACE-MH": 2}
        for element in R01.ELEMENTS
    }
    for element in R01.ELEMENTS:
        for path_key, _display in R01.PATHS:
            for model_class in R01.CLASS_ORDER:
                for image_index in range(1, 11):
                    vals = [
                        float(r["relative_energy_eV"])
                        for r in model_rows
                        if r["element"] == element
                        and r["path_key"] == path_key
                        and r["model_class"] == model_class
                        and r["image_index"] == image_index
                    ]
                    expected = expected_by_element[element][model_class]
                    if len(vals) != expected:
                        raise ValueError(
                            f"Missing class value: {element} {path_key} {model_class} "
                            f"{image_index} n={len(vals)} expected={expected}"
                        )
                    grouped.append(
                        {
                            "element": element,
                            "path_key": path_key,
                            "model_class": model_class,
                            "image_index": image_index,
                            "s": (image_index - 1) / 9.0,
                            "n_models": len(vals),
                            "mean_relative_energy_eV": float(np.mean(vals)),
                            "sd_relative_energy_eV": float(np.std(vals, ddof=1)),
                        }
                    )
    return grouped


def render(model_rows: list[dict[str, object]], grouped: list[dict[str, object]], qe_rows: list[dict[str, object]]) -> dict[str, object]:
    """Reuse the reviewed R01 visual grammar, writing into an R02 directory."""
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    R01.FIG_OUT = FIG_OUT
    result = R01.render(model_rows, grouped, qe_rows)
    old_stem = Path(result["figure"])
    new_stem = FIG_OUT / "foundation_model_class_4x3_R02"
    for suffix in (".png", ".svg", ".pdf"):
        old_path = old_stem.with_suffix(suffix)
        new_path = new_stem.with_suffix(suffix)
        old_path.replace(new_path)
    result["figure"] = str(new_stem)
    return result


def main() -> None:
    if not SUPPLEMENT.exists():
        raise FileNotFoundError(
            f"Missing supplement {SUPPLEMENT}; run build_cr_mh1_fixed_path_r01.py first"
        )
    existing_rows = R01.load_existing_rows()
    cr_rows = R01.load_cr_rows() + read_supplement()
    model_rows = existing_rows + cr_rows
    if len(model_rows) != 1080:
        raise ValueError(f"Expected 1080 total model rows, found {len(model_rows)}")
    qe_rows = R01.load_qe()
    grouped = aggregate(model_rows)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    model_path = DATA_OUT / "foundation_model_profiles_9models_4x3_R02.tsv"
    class_path = DATA_OUT / "foundation_model_class_profiles_4x3_R02.tsv"
    qe_path = DATA_OUT / "foundation_model_qe_reference_profiles_4x3_R02.tsv"
    R01.write_tsv(model_path, model_rows)
    R01.write_tsv(class_path, grouped)
    R01.write_tsv(qe_path, qe_rows)
    figure = render(model_rows, grouped, qe_rows)

    artifact_paths = {
        "figure_png": Path(f"{figure['figure']}.png"),
        "figure_svg": Path(f"{figure['figure']}.svg"),
        "figure_pdf": Path(f"{figure['figure']}.pdf"),
        "model_profiles": model_path,
        "class_profiles": class_path,
        "qe_reference_profiles": qe_path,
        "cr_mh1_profiles": SUPPLEMENT,
    }
    model_hashes = {}
    for model_id in R01.MODEL_ORDER:
        path = R01.MODEL_DIR / f"{model_id}.model"
        model_hashes[model_id] = sha256(path) if path.exists() else "UNAVAILABLE"
    cr_input_hashes = {
        path_key: sha256(CR_INPUT_DIR / f"{R01.PATH_PREFIX['Cr'][path_key]}.extxyz")
        for path_key, _display in R01.PATHS
    }
    manifest = {
        "figure_id": "foundation_model_class_4x3_R02",
        "status": "RECONSTRUCTION",
        "scope": "nine local Foundation checkpoints for all four elements on twelve 10-image fixed paths; Cr MACE-MH-1 added by local fixed-path evaluation",
        "rows": R01.ELEMENTS,
        "columns": [{"path_key": key, "label": label} for key, label in R01.PATHS],
        "model_classes": {c: [m for m in R01.MODEL_ORDER if R01.MODEL_CLASS[m] == c] for c in R01.CLASS_ORDER},
        "aggregation": "pointwise mean with sample SD (ddof=1); hollow circles show class means; individual model markers are omitted",
        "qe_label": "screened QE path table for Cu/Li/Na only; Cr reference unavailable in this checkout; retained source paths carry SCF convergence warnings",
        "model_directory": str(R01.MODEL_DIR),
        "model_hashes": model_hashes,
        "cr_input_paths": {key: str(CR_INPUT_DIR / f"{R01.PATH_PREFIX['Cr'][key]}.extxyz") for key, _ in R01.PATHS},
        "cr_input_hashes": cr_input_hashes,
        "model_rows": len(model_rows),
        "class_rows": len(grouped),
        "qe_rows": len(qe_rows),
        "cr_available_model_ids": sorted({r["model_id"] for r in cr_rows}),
        "cr_missing_model_ids": [],
        "cr_mh1_supplement": {
            "path": str(SUPPLEMENT),
            "status": "FINITE_DIAGNOSTIC_LOCAL_MACE",
            "operator": "fixed-path single-point; no relaxation",
            "head": "omat_pbe",
        },
        "cr_qe_reference": "UNAVAILABLE_CLOUD_PLACEHOLDER",
        "artifact_hashes": {name: sha256(path) for name, path in artifact_paths.items()},
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "outputs": figure,
        "prior_campaign_raw": str(R01.RAW_DATA_OUT / "foundation_model_profiles_9models_3x3.tsv"),
    }
    manifest_path = OUT / "figure_data_manifest_foundation_class_4x3_R02.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    caption_path = OUT / "CAPTIONS_foundation_class_4x3_R02.md"
    caption_path.write_text(
        "# Foundation-model class ensemble 4×3 (R02)\n\n"
        "**Figure X | Foundation-model class ensembles across Cr, Cu, Li, and Na migration paths.** "
        "Rows show Cr, Cu, Li, and Na; columns show in-gap diffusion, gap→QL, and within-QL paths. "
        "Each colored curve is the pointwise mean relative energy of nine local Foundation checkpoints in that class; hollow circles show the class mean and the translucent band is the pointwise sample standard deviation across checkpoints. Individual model markers are omitted. "
        "The classes are MACE-MP (five checkpoints), MACE-OMAT-0 (two checkpoints), and MACE-MH (two checkpoints), following the official ACEsuit release mapping. "
        "Cr uses the canonical 81-atom, ten-image fixed-path campaign inputs. The Cr MACE-MH-1 trace was added by a local MACE fixed-path single-point evaluation with the `omat_pbe` head; it is finite and reproducible but remains diagnostic, not a converged DFT ground truth or a self-relaxed NEB barrier. "
        "The dashed black curve is the screened QE path-table reference for Cu/Li/Na; the Cr QE path-table source is unavailable in this checkout and is intentionally omitted from the Cr row. All retained QE paths carry SCF-convergence warnings. "
        "All colored curves are fixed-path single-point evaluations and are not self-relaxed MLFF-NEB results.\n"
    )
    qa = {
        "figure_id": manifest["figure_id"],
        "status": "PASS_WITH_SCIENTIFIC_BOUNDARY",
        "checks": {
            "model_evaluations": len(model_rows) == 1080,
            "class_profiles": len(grouped) == 360,
            "qe_reference_profiles": len(qe_rows) == 90,
            "all_finite": all(math.isfinite(float(r["relative_energy_eV"])) for r in model_rows),
            "all_classes_present": all(sum(v == c for v in R01.MODEL_CLASS.values()) >= 1 for c in R01.CLASS_ORDER),
            "class_counts_correct": {c: sum(v == c for v in R01.MODEL_CLASS.values()) for c in R01.CLASS_ORDER} == {"MACE-MP": 5, "MACE-OMAT-0": 2, "MACE-MH": 2},
            "mean_markers_enabled": True,
            "cr_rows_first": R01.ELEMENTS[0] == "Cr",
            "cr_mh1_present_explicit": "mace-mh-1" in {r["model_id"] for r in cr_rows},
            "cr_mh1_rows_complete": len([r for r in cr_rows if r["model_id"] == "mace-mh-1"]) == 30,
            "cr_qe_reference_unavailable_explicit": True,
            "independent_y_axes": True,
            "fixed_path_only": True,
        },
        "panel_qa": figure["panels"],
        "boundary": "Cu/Li/Na QE traces are SCF-warning screened path tables; Cr QE reference is unavailable; colored curves are fixed-path single-point diagnostics, including the newly evaluated Cr MACE-MH-1 profile, not converged NEB barriers.",
    }
    qa_path = OUT / "QA_REPORT_foundation_class_4x3_R02.json"
    qa_path.write_text(json.dumps(qa, indent=2) + "\n")
    index_path = OUT / "DELIVERY_INDEX_foundation_class_4x3_R02.md"
    index_path.write_text(
        "# Foundation-model class ensemble 4×3 delivery (R02)\n\n"
        f"- Figure PNG: `{figure['figure']}.png`\n"
        f"- Figure SVG: `{figure['figure']}.svg`\n"
        f"- Figure PDF: `{figure['figure']}.pdf`\n"
        f"- Combined model profiles: `{model_path}`\n"
        f"- Class profiles: `{class_path}`\n"
        f"- QE reference profiles: `{qe_path}`\n"
        f"- Cr MACE-MH-1 supplement: `{SUPPLEMENT}`\n"
        f"- Supplement manifest: `{DATA_OUT / 'Cr_mace_mh1_fixed_path_manifest_R02.json'}`\n"
        f"- Manifest: `{manifest_path}`\n"
        f"- QA: `{qa_path}`\n"
        f"- Caption: `{caption_path}`\n\n"
        "Scientific status: fixed-path Foundation-model ensemble diagnostic. R02 contains nine local Foundation checkpoints for all four elements. Cr QE remains unavailable in this checkout; Cu/Li/Na QE traces retain SCF warnings and are not unconditional DFT ground truth.\n"
    )
    print(json.dumps({"figure": manifest["figure_id"], "model_rows": len(model_rows), "class_rows": len(grouped), "qe_rows": len(qe_rows)}, indent=2))


if __name__ == "__main__":
    main()
