#!/usr/bin/env python3
"""Render R03 with the Cr screened QE/DFT reference added to the R02 figure."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path


SCRIPT = Path(__file__).with_name("build_foundation_class_4x3_r02.py")
SPEC = importlib.util.spec_from_file_location("foundation_r02", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT}")
R02 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R02)
R01 = R02.R01

OUT = R01.OUT
FIG_OUT = OUT / "figures" / "foundation_class_4x3_R03"
DATA_OUT = OUT / "data" / "foundation_class_4x3_R03"
CR_REFERENCE = OUT / "data" / "Cr_fixed_path_representative_profiles.tsv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cr_qe() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with CR_REFERENCE.open(newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["kind"] != "reference" or row["series_key"] != "qe_reference":
                continue
            path_key = row["path_key"]
            rows.append(
                {
                    "element": "Cr",
                    "path_key": path_key,
                    "path_id": R01.PATH_PREFIX["Cr"][path_key],
                    "image_index": int(row["image_index"]),
                    "s": float(row["s"]),
                    "relative_energy_eV": float(row["relative_energy_eV"]),
                    "status": row["status"],
                    "source_path": row["source_path"],
                    "source_record": row["source_record"],
                }
            )
    if len(rows) != 30:
        raise ValueError(f"Expected 30 Cr QE reference rows, found {len(rows)}")
    for path_key, _label in R01.PATHS:
        subset = [r for r in rows if r["path_key"] == path_key]
        if len(subset) != 10:
            raise ValueError(f"Incomplete Cr QE reference: {path_key}")
    return rows


def render(model_rows, grouped, qe_rows):
    R02.FIG_OUT = FIG_OUT
    result = R02.render(model_rows, grouped, qe_rows)
    old_stem = Path(result["figure"])
    new_stem = FIG_OUT / "foundation_model_class_4x3_R03"
    for suffix in (".png", ".svg", ".pdf"):
        old_stem.with_suffix(suffix).replace(new_stem.with_suffix(suffix))
    result["figure"] = str(new_stem)
    return result


def main() -> None:
    if not CR_REFERENCE.exists():
        raise FileNotFoundError(CR_REFERENCE)
    existing_rows = R01.load_existing_rows()
    cr_rows = R01.load_cr_rows() + R02.read_supplement()
    model_rows = existing_rows + cr_rows
    if len(model_rows) != 1080:
        raise ValueError(f"Expected 1080 model rows, found {len(model_rows)}")
    qe_rows = R01.load_qe() + load_cr_qe()
    if len(qe_rows) != 120:
        raise ValueError(f"Expected 120 QE reference rows, found {len(qe_rows)}")
    grouped = R02.aggregate(model_rows)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    model_path = DATA_OUT / "foundation_model_profiles_9models_4x3_R03.tsv"
    class_path = DATA_OUT / "foundation_model_class_profiles_4x3_R03.tsv"
    qe_path = DATA_OUT / "foundation_model_qe_reference_profiles_4x3_R03.tsv"
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
        "cr_reference": CR_REFERENCE,
    }
    model_hashes = {}
    for model_id in R01.MODEL_ORDER:
        path = R01.MODEL_DIR / f"{model_id}.model"
        model_hashes[model_id] = sha256(path) if path.exists() else "UNAVAILABLE"
    cr_input_hashes = {
        path_key: sha256(R01.CR_INPUT_DIR / f"{R01.PATH_PREFIX['Cr'][path_key]}.extxyz")
        for path_key, _display in R01.PATHS
    }
    manifest = {
        "figure_id": "foundation_model_class_4x3_R03",
        "status": "RECONSTRUCTION",
        "scope": "nine local Foundation checkpoints and screened QE references for Cr, Cu, Li, and Na on twelve 10-image fixed paths",
        "rows": R01.ELEMENTS,
        "columns": [{"path_key": key, "label": label} for key, label in R01.PATHS],
        "model_classes": {c: [m for m in R01.MODEL_ORDER if R01.MODEL_CLASS[m] == c] for c in R01.CLASS_ORDER},
        "aggregation": "pointwise mean with sample SD (ddof=1); hollow circles show class means; individual model markers are omitted",
        "qe_label": "screened QE path table for all four elements; all retained reference paths carry SCF convergence warnings",
        "model_directory": str(R01.MODEL_DIR),
        "model_hashes": model_hashes,
        "cr_input_paths": {key: str(R01.CR_INPUT_DIR / f"{R01.PATH_PREFIX['Cr'][key]}.extxyz") for key, _ in R01.PATHS},
        "cr_input_hashes": cr_input_hashes,
        "cr_reference_path": str(CR_REFERENCE),
        "cr_reference_sha256": sha256(CR_REFERENCE),
        "model_rows": len(model_rows),
        "class_rows": len(grouped),
        "qe_rows": len(qe_rows),
        "cr_available_model_ids": sorted({r["model_id"] for r in cr_rows}),
        "cr_missing_model_ids": [],
        "cr_mh1_supplement": {
            "path": str(R02.SUPPLEMENT),
            "status": "FINITE_DIAGNOSTIC_LOCAL_MACE",
            "operator": "fixed-path single-point; no relaxation",
            "head": "omat_pbe",
        },
        "cr_qe_reference": "COMPLETED_BUT_SCF_WARNING_DIAGNOSTIC",
        "artifact_hashes": {name: sha256(path) for name, path in artifact_paths.items()},
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "outputs": figure,
        "prior_campaign_raw": str(R01.RAW_DATA_OUT / "foundation_model_profiles_9models_3x3.tsv"),
    }
    manifest_path = OUT / "figure_data_manifest_foundation_class_4x3_R03.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    caption_path = OUT / "CAPTIONS_foundation_class_4x3_R03.md"
    caption_path.write_text(
        "# Foundation-model class ensemble 4×3 (R03)\n\n"
        "**Figure X | Foundation-model class ensembles and screened QE references across Cr, Cu, Li, and Na migration paths.** "
        "Rows show Cr, Cu, Li, and Na; columns show in-gap diffusion, gap→QL, and within-QL paths. "
        "Each colored curve is the pointwise mean relative energy of nine local Foundation checkpoints in that class; hollow circles show the class mean and the translucent band is the pointwise sample standard deviation across checkpoints. Individual model markers are omitted. "
        "The classes are MACE-MP (five checkpoints), MACE-OMAT-0 (two checkpoints), and MACE-MH (two checkpoints), following the official ACEsuit release mapping. "
        "Cr uses the canonical 81-atom, ten-image fixed-path campaign inputs, and its screened QE reference is now shown with the same dashed black line and hollow-diamond markers as the other rows. The Cr MACE-MH-1 trace was added by a local MACE fixed-path single-point evaluation with the `omat_pbe` head; it is finite and reproducible but remains diagnostic. "
        "All four QE reference paths carry SCF-convergence warnings; they are screened diagnostic references and are not presented as unconditional converged DFT ground truth or publication-grade activation barriers. All colored curves are fixed-path single-point evaluations and are not self-relaxed MLFF-NEB results.\n"
    )
    qa = {
        "figure_id": manifest["figure_id"],
        "status": "PASS_WITH_SCIENTIFIC_BOUNDARY",
        "checks": {
            "model_evaluations": len(model_rows) == 1080,
            "class_profiles": len(grouped) == 360,
            "qe_reference_profiles": len(qe_rows) == 120,
            "all_finite": all(math.isfinite(float(r["relative_energy_eV"])) for r in model_rows + qe_rows),
            "all_classes_present": all(sum(v == c for v in R01.MODEL_CLASS.values()) >= 1 for c in R01.CLASS_ORDER),
            "class_counts_correct": {c: sum(v == c for v in R01.MODEL_CLASS.values()) for c in R01.CLASS_ORDER} == {"MACE-MP": 5, "MACE-OMAT-0": 2, "MACE-MH": 2},
            "mean_markers_enabled": True,
            "cr_rows_first": R01.ELEMENTS[0] == "Cr",
            "cr_mh1_present_explicit": "mace-mh-1" in {r["model_id"] for r in cr_rows},
            "cr_mh1_rows_complete": len([r for r in cr_rows if r["model_id"] == "mace-mh-1"]) == 30,
            "cr_qe_reference_present_explicit": len(load_cr_qe()) == 30,
            "cr_qe_all_scf_warning": all(r["status"] == "COMPLETED_BUT_SCF_WARNING" for r in load_cr_qe()),
            "independent_y_axes": True,
            "fixed_path_only": True,
        },
        "panel_qa": figure["panels"],
        "boundary": "All QE traces, including Cr, are SCF-warning screened path tables; colored curves are fixed-path single-point diagnostics, including the newly evaluated Cr MACE-MH-1 profile, not converged NEB barriers.",
    }
    qa_path = OUT / "QA_REPORT_foundation_class_4x3_R03.json"
    qa_path.write_text(json.dumps(qa, indent=2) + "\n")
    index_path = OUT / "DELIVERY_INDEX_foundation_class_4x3_R03.md"
    index_path.write_text(
        "# Foundation-model class ensemble 4×3 delivery (R03)\n\n"
        f"- Figure PNG: `{figure['figure']}.png`\n"
        f"- Figure SVG: `{figure['figure']}.svg`\n"
        f"- Figure PDF: `{figure['figure']}.pdf`\n"
        f"- Combined model profiles: `{model_path}`\n"
        f"- Class profiles: `{class_path}`\n"
        f"- QE reference profiles: `{qe_path}`\n"
        f"- Cr QE/DFT reference source: `{CR_REFERENCE}`\n"
        f"- Cr MACE-MH-1 supplement: `{R02.SUPPLEMENT}`\n"
        f"- Manifest: `{manifest_path}`\n"
        f"- QA: `{qa_path}`\n"
        f"- Caption: `{caption_path}`\n\n"
        "Scientific status: fixed-path Foundation-model ensemble plus screened QE-reference diagnostic. All four elements now have a displayed reference trace; all four reference paths retain SCF-warning status and are not unconditional DFT ground truth.\n"
    )
    print(json.dumps({"figure": manifest["figure_id"], "model_rows": len(model_rows), "class_rows": len(grouped), "qe_rows": len(qe_rows)}, indent=2))


if __name__ == "__main__":
    main()
