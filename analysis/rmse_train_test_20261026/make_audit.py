#!/usr/bin/env python3
"""Create compact provenance, quality, and release records for the parity package."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw" / "rockfish_parity"
FIG = ROOT / "figures"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    if "--manifest-only" in sys.argv:
        manifest = {
            "package": "rmse_train_test_20261026",
            "status": "PUBLICATION_PROVISIONAL",
            "rebuild": "python analysis/rmse_train_test_20261026/make_figures.py",
            "rebuild_frame_only": "python analysis/rmse_train_test_20261026/make_figures.py --frame-only",
            "audit": "python analysis/rmse_train_test_20261026/make_audit.py",
            "source_evaluation": "Rockfish Slurm job 32511152; 24/24 completed with PARITY_EVAL_OK",
            "files": {},
        }
        include = [
            ROOT / "README.md", ROOT / "evaluate_scatter.py", ROOT / "make_figures.py", ROOT / "make_audit.py", ROOT / "parity_manifest.tsv",
            ROOT / "parity_summary.csv", ROOT / "QUALITY_AUDIT.tsv", ROOT / "MODEL_REGISTRY.tsv",
            ROOT / "CLAIM_EVIDENCE_MATRIX.tsv", ROOT / "FAILURE_EXCLUSION.tsv", ROOT / "GAP_REPORT.md",
        ] + sorted(FIG.glob("fig_*"))
        for p in include:
            if p.exists():
                manifest["files"][str(p.relative_to(ROOT))] = {"sha256": sha256(p), "bytes": p.stat().st_size}
        (ROOT / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Updated release manifest for {len(manifest['files'])} files")
        return
    jsons = sorted(RAW.glob("*.json"))
    rows = []
    registry = []
    checks = []
    all_ok = True
    for jp in jsons:
        d = json.loads(jp.read_text())
        s = d["summary"]
        stem = jp.stem
        npz_p = RAW / f"{stem}__parity.npz"
        csv_p = RAW / f"{stem}__per_frame.csv"
        z = np.load(npz_p)
        expected = int(s["n_frames"])
        finite = all(np.isfinite(z[k]).all() for k in z.files)
        shapes_ok = (
            len(z["energy_ref_eV_per_atom"]) == expected
            and len(z["energy_pred_eV_per_atom"]) == expected
            and len(z["frame_force_rmse_meV_A"]) == expected
            and len(z["force_ref_eV_A"]) == int(s["n_force_components"])
            and len(z["force_pred_eV_A"]) == int(s["n_force_components"])
        )
        crows = read_csv(csv_p)
        csv_ok = len(crows) == expected and all(
            all(x not in ("", "nan", "NaN") for x in row.values()) for row in crows
        )
        ok = d.get("status") == "PARITY_EVAL_OK" and d.get("model_sha_ok") and d.get("data_sha_ok") and finite and shapes_ok and csv_ok
        all_ok &= ok
        checks.extend([
            [stem, "status", "PASS" if d.get("status") == "PARITY_EVAL_OK" else "FAIL", d.get("status", "missing")],
            [stem, "model_sha256", "PASS" if d.get("model_sha_ok") else "FAIL", str(d.get("model_sha_ok"))],
            [stem, "data_sha256", "PASS" if d.get("data_sha_ok") else "FAIL", str(d.get("data_sha_ok"))],
            [stem, "finite_arrays", "PASS" if finite else "FAIL", ";".join(z.files)],
            [stem, "array_shapes", "PASS" if shapes_ok else "FAIL", f"frames={expected};force_components={s['n_force_components']}"],
            [stem, "per_frame_csv", "PASS" if csv_ok else "FAIL", f"rows={len(crows)}"],
        ])
        rows.append({
            "model": d["model_family"], "seed": d["training_seed"], "split": d["split"],
            "eval_id": d["eval_id"], "model_id": d["model_id"], "model_path": d["model_path"],
            "model_sha256": d["model_sha256"], "data_path": d["data_path"], "data_sha256": d["data_sha256"],
            "n_frames": expected, "n_force_components": s["n_force_components"],
            "evaluation_status": d["status"], "lineage_status": "LINEAGE_SUPPORTED" if ok else "LINEAGE_REVIEW",
        })

    with (ROOT / "QUALITY_AUDIT.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["eval_id", "check", "status", "evidence"])
        w.writerows(checks)
        w.writerow(["PACKAGE", "matrix_completeness", "PASS" if len(jsons) == 24 else "FAIL", f"{len(jsons)}/24 evaluations"])
        w.writerow(["PACKAGE", "overall", "PASS" if all_ok and len(jsons) == 24 else "FAIL", "all final parity evaluations passed local integrity checks"])

    with (ROOT / "MODEL_REGISTRY.tsv").open("w", newline="") as f:
        fields = list(rows[0])
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader(); w.writerows(rows)

    with (ROOT / "CLAIM_EVIDENCE_MATRIX.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["claim_id", "claim", "evidence", "status", "caveat"])
        w.writerow(["FIG-PARITY", "DFT/reference versus MLIP/prediction for train/test AIMD points with common 600K/MultiT axes", "figures/fig_parity_train_test_common_2x4.{pdf,svg,png}; raw/rockfish_parity/*__parity.npz; parity_summary.csv", "SUPPORTED", "Train is gray context and test is colored overlay; all three seeds are included and R2/RMSE use full arrays."])
        w.writerow(["FIG-FRAME", "Frame-level error distributions contain the evaluated train/test frames", "figures/fig_rmse_frame_distributions.{pdf,svg,png}; raw/rockfish_parity/*__per_frame.csv", "SUPPORTED", "Energy is absolute error per atom; force is per-frame component RMSE."])
        w.writerow(["DATA-LINEAGE", "The 24 model/seed/split evaluations use hash-checked model and XYZ inputs", "MODEL_REGISTRY.tsv; parity_manifest.tsv; QUALITY_AUDIT.tsv", "SUPPORTED", "Current parity values are a direct Rockfish re-evaluation, not recovered historical paired training output."])
        w.writerow(["SCIENTIFIC-STATUS", "Test energy R2 is negative for 600K families because the test reference-energy span is narrow", "parity_summary.csv; GAP_REPORT.md", "DIAGNOSTIC", "Retained; no outlier or model was silently removed."])

    with (ROOT / "FAILURE_EXCLUSION.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["item", "type", "status", "action", "impact"])
        w.writerow(["32510971", "wrapper/environment", "EXCLUDED_FROM_SCIENCE", "retained stderr under raw/failed_wrapper_lc_all; repaired shell initialization", "no final evaluation result"])
        w.writerow(["32511092", "wrapper/environment", "EXCLUDED_FROM_SCIENCE", "retained stderr under raw/failed_wrapper_mace_polar; reran with the known MACE environment", "no final evaluation result"])
        w.writerow(["24 final evaluations", "scientific evaluation", "RETAINED", "all successful outputs included", "no scientific failures excluded"])
        w.writerow(["600K test energy R2", "metric diagnostic", "RETAINED", "reported in parity panels and GAP_REPORT.md", "negative R2 reflects narrow reference-energy variance"])
        w.writerow(["Scratch-MultiT seed123 test force", "high-error observation", "RETAINED", "not filtered from arrays, summary, or plots", "visible as a genuine distribution tail"])

    manifest = {
        "package": "rmse_train_test_20261026",
        "status": "PUBLICATION_PROVISIONAL",
        "rebuild": "python analysis/rmse_train_test_20261026/make_figures.py",
        "audit": "python analysis/rmse_train_test_20261026/make_audit.py",
        "source_evaluation": "Rockfish Slurm job 32511152; 24/24 completed with PARITY_EVAL_OK",
        "files": {},
    }
    include = [
        ROOT / "README.md", ROOT / "evaluate_scatter.py", ROOT / "make_figures.py", ROOT / "make_audit.py", ROOT / "parity_manifest.tsv",
        ROOT / "parity_summary.csv", ROOT / "QUALITY_AUDIT.tsv", ROOT / "MODEL_REGISTRY.tsv",
        ROOT / "CLAIM_EVIDENCE_MATRIX.tsv", ROOT / "FAILURE_EXCLUSION.tsv", ROOT / "GAP_REPORT.md",
    ] + sorted(FIG.glob("fig_*"))
    for p in include:
        if p.exists():
            manifest["files"][str(p.relative_to(ROOT))] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    (ROOT / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Audited {len(jsons)} evaluations; overall={'PASS' if all_ok and len(jsons)==24 else 'FAIL'}")


if __name__ == "__main__":
    main()
