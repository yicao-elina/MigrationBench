"""Extract stage-aware QE-NEB geometry diagnostics and model disagreement.

The saved ``.path`` files are used only as read-only diagnostic snapshots.  The
script does not promote them to converged NEB references.  Stage labels are
explicitly geometric/path-position labels: endpoints (images 1 and 10), a
three-image transition-state window (maximum-energy image +/- 1),
near-endpoint images (2, 3, 8, 9), and any remaining interior images.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "remote_snapshot"
MANIFEST = ROOT / "dft_gate_manifest.csv"
MODEL_IMAGE = Path(os.environ.get("STRUCT_MODEL_IMAGE", str(ROOT / "exploratory_energy_audit/corrected_dft_foundation_models/corrected_dft_foundation_image_profiles.csv")))
OUT = Path(os.environ.get("STRUCT_OUT", str(ROOT / "exploratory_energy_audit/structural_diagnostics")))
OUT.mkdir(parents=True, exist_ok=True)
HARTREE_TO_EV = 27.211386245988
BOHR_TO_A = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_A = HARTREE_TO_EV / BOHR_TO_A


def pbc_delta(a: np.ndarray, b: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Minimum-image displacement from b to a for row-vector cell vectors."""
    frac = (a - b) @ np.linalg.inv(cell)
    frac -= np.rint(frac)
    return frac @ cell


def parse_path(path: Path, symbols: list[str], cell: np.ndarray) -> list[dict]:
    lines = path.read_text(errors="replace").splitlines()
    rows: list[dict] = []
    for i, line in enumerate(lines):
        if not line.strip().startswith("Image:"):
            continue
        image = int(line.split(":", 1)[1])
        energy = float(lines[i + 1].strip().split()[0]) * HARTREE_TO_EV
        coords, grads = [], []
        for j in range(len(symbols)):
            values = [float(x) for x in lines[i + 2 + j].split()[:6]]
            # sb2te3.path stores Cartesian coordinates in Bohr and gradients
            # in Hartree/Bohr.  Convert before combining with the manifest's
            # Angstrom cell and before reporting force-like descriptors.
            coords.append(np.asarray(values[:3], dtype=float) * BOHR_TO_A)
            grads.append(np.asarray(values[3:6], dtype=float) * HARTREE_PER_BOHR_TO_EV_PER_A)
        rows.append({"image_index": image, "path_energy_raw": energy,
                     "positions": np.asarray(coords, dtype=float),
                     "gradients": np.asarray(grads, dtype=float)})
    if len(rows) != 10:
        raise ValueError(f"{path}: expected 10 image blocks, found {len(rows)}")
    rows.sort(key=lambda x: x["image_index"])
    return rows


def stage_labels(energies: np.ndarray, image: int) -> tuple[str, str, int]:
    ts = int(np.argmax(energies) + 1)
    if image in (1, 10):
        return "endpoint", "endpoint", ts
    if abs(image - ts) <= 1:
        return "transition_state_window", "transition_state", ts
    if image in (2, 3, 8, 9):
        return "near_endpoint", "near_endpoint", ts
    return "interior", "near_endpoint_or_interior", ts


def extract() -> pd.DataFrame:
    gate = pd.read_csv(MANIFEST).set_index("path_id")
    rows = []
    for path_id, g in gate.iterrows():
        path = SNAPSHOT / g.remote_neb_dir.rsplit("/", 1)[-2] / ""  # unused guard
        # Use the local snapshot path directly; remote_neb_dir preserves provenance.
        rel = Path(g.remote_neb_dir).relative_to("/scratch4/pclancy3/joshitha/vdW-sratch/sb2te3")
        local = SNAPSHOT / rel / "sb2te3.path"
        symbols = str(g.species_order).split()
        cell = np.asarray(ast.literal_eval(g.cell_A), dtype=float)
        images = parse_path(local, symbols, cell)
        energies = np.asarray([x["path_energy_raw"] for x in images])
        e_rel = energies - energies[0]
        ref_pos = images[0]["positions"]
        dopant = len(symbols) - 1
        host = np.arange(len(symbols) - 1)
        for record, rel_energy in zip(images, e_rel):
            pos = record["positions"]
            d_host = np.asarray([np.linalg.norm(pbc_delta(pos[dopant], pos[j], cell)) for j in host])
            nearest = np.sort(d_host)
            host_delta = np.asarray([np.linalg.norm(pbc_delta(pos[j], ref_pos[j], cell)) for j in host])
            all_delta = np.asarray([np.linalg.norm(pbc_delta(pos[j], ref_pos[j], cell)) for j in range(len(symbols))])
            grad_norm = np.linalg.norm(record["gradients"], axis=1)
            stage, coarse, ts = stage_labels(energies, int(record["image_index"]))
            rows.append({
                "path_id": path_id,
                "system": g.system,
                "dopant": g.dopant,
                "image_index": int(record["image_index"]),
                "path_energy_raw": float(record["path_energy_raw"]),
                "qe_relative_energy_eV": float(rel_energy),
                "ts_image_index": ts,
                "stage": stage,
                "stage_coarse": coarse,
                "dopant_displacement_from_image1_A": float(np.linalg.norm(pbc_delta(pos[dopant], ref_pos[dopant], cell))),
                "dopant_nearest_host_A": float(nearest[0]),
                "dopant_nearest3_mean_A": float(nearest[:3].mean()),
                "dopant_coordination_n_within_4p3A": int(np.sum(d_host <= 4.3)),
                "dopant_coordination_n_within_4p6A": int(np.sum(d_host <= 4.6)),
                "nearest3_distance_std_A": float(nearest[:3].std()),
                "host_rms_displacement_from_image1_A": float(np.sqrt(np.mean(host_delta**2))),
                "all_atom_rms_displacement_from_image1_A": float(np.sqrt(np.mean(all_delta**2))),
                "max_gradient_norm_eV_per_A": float(grad_norm.max()),
                "rms_gradient_norm_eV_per_A": float(np.sqrt(np.mean(grad_norm**2))),
                "cell_volume_A3": float(abs(np.linalg.det(cell))),
                "cell_a_A": float(np.linalg.norm(cell[0])),
                "cell_b_A": float(np.linalg.norm(cell[1])),
                "cell_c_A": float(np.linalg.norm(cell[2])),
            })
    return pd.DataFrame(rows)


def add_model_disagreement(d: pd.DataFrame) -> pd.DataFrame:
    model = pd.read_csv(MODEL_IMAGE)
    keys = ["path_id", "image_index"]
    model_cols = [c for c in model.model_id.unique() if c != "qe_reference"]
    pivot = model.pivot_table(index=keys, columns="model_id", values="relative_energy_eV")
    values = pivot[model_cols]
    out = pd.DataFrame({
        "path_id": [x[0] for x in values.index], "image_index": [x[1] for x in values.index],
        "foundation_median_relative_energy_eV": values.median(axis=1).to_numpy(),
        "foundation_std_eV": values.std(axis=1, ddof=0).to_numpy(),
        "foundation_range_eV": (values.max(axis=1) - values.min(axis=1)).to_numpy(),
        "foundation_q25_q75_range_eV": (values.quantile(.75, axis=1) - values.quantile(.25, axis=1)).to_numpy(),
    })
    out["foundation_median_abs_error_to_qe_eV"] = np.abs(out.foundation_median_relative_energy_eV - model.groupby(keys).qe_relative_energy_eV.first().reindex(values.index).to_numpy())
    return d.merge(out, on=["path_id", "image_index"], how="left", validate="one_to_one")


def build_embedding(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    descriptor_cols = [
        "dopant_displacement_from_image1_A", "dopant_nearest_host_A", "dopant_nearest3_mean_A",
        "dopant_coordination_n_within_4p3A", "dopant_coordination_n_within_4p6A", "nearest3_distance_std_A",
        "host_rms_displacement_from_image1_A", "all_atom_rms_displacement_from_image1_A",
        "max_gradient_norm_eV_per_A", "rms_gradient_norm_eV_per_A", "qe_relative_energy_eV", "foundation_range_eV",
    ]
    x = StandardScaler().fit_transform(d[descriptor_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    pca = PCA(n_components=2, random_state=0).fit_transform(x)
    tsne = TSNE(n_components=2, perplexity=10, init="pca", learning_rate="auto", random_state=0).fit_transform(x)
    e = d[["path_id", "system", "dopant", "image_index", "stage", "stage_coarse"]].copy()
    e["pca1"], e["pca2"] = pca[:, 0], pca[:, 1]
    e["tsne1"], e["tsne2"] = tsne[:, 0], tsne[:, 1]
    return e, {"descriptor_columns": descriptor_cols, "n_frames": len(d), "pca_explained_variance_ratio": PCA(n_components=2, random_state=0).fit(x).explained_variance_ratio_.tolist(), "scope": "NEB frames only; training/test/AIMD raw structures are not included"}


def main() -> None:
    d = add_model_disagreement(extract()).sort_values(["path_id", "image_index"])
    d.to_csv(OUT / "qe_path_structure_stage_metrics.csv", index=False)
    stage_summary = d.groupby("stage", observed=False).agg(
        n_images=("image_index", "size"),
        median_qe_relative_energy_eV=("qe_relative_energy_eV", "median"),
        median_dopant_displacement_A=("dopant_displacement_from_image1_A", "median"),
        median_nearest_host_A=("dopant_nearest_host_A", "median"),
        median_coordination_4p3A=("dopant_coordination_n_within_4p3A", "median"),
        median_host_rms_displacement_A=("host_rms_displacement_from_image1_A", "median"),
        median_foundation_range_eV=("foundation_range_eV", "median"),
        max_foundation_range_eV=("foundation_range_eV", "max"),
    ).reset_index()
    stage_summary.to_csv(OUT / "qe_path_stage_summary.csv", index=False)
    embedding, emb_meta = build_embedding(d)
    embedding.to_csv(OUT / "neb_descriptor_embedding.csv", index=False)
    path_summary = d.groupby(["path_id", "dopant", "ts_image_index"], as_index=False).agg(
        qe_barrier_like_eV=("qe_relative_energy_eV", "max"),
        foundation_range_max_eV=("foundation_range_eV", "max"),
        max_gradient_norm_eV_per_A=("max_gradient_norm_eV_per_A", "max"),
        endpoint_dopant_nearest_host_A=("dopant_nearest_host_A", "first"),
        ts_dopant_nearest_host_A=("dopant_nearest_host_A", lambda x: float(x.iloc[int(np.argmax(d.loc[x.index, 'qe_relative_energy_eV']))])),
    )
    path_summary.to_csv(OUT / "qe_path_summary.csv", index=False)
    manifest = {
        "status": "diagnostic_stage_and_structure_analysis",
        "n_paths": int(d.path_id.nunique()), "n_images": int(len(d)),
        "stage_definition": {"endpoint": [1, 10], "transition_state_window": "argmax(QE path energy) +/- 1", "near_endpoint": [2, 3, 8, 9], "interior": "remaining non-endpoint, non-TS-window images"},
        "energy_units": "sb2te3.path energy values converted Hartree to eV with 27.211386245988; positions converted Bohr to Angstrom",
        "cell_handling": "fixed cell from QE manifest; minimum-image distances; no lattice relaxation descriptor because cell is fixed",
        "embedding": emb_meta,
        "training_distribution_status": "not_claimed: no authoritative raw training split was found in the current workspace; grouped test and available AIMD assets are separate follow-up inputs",
    }
    (OUT / "structural_diagnostics_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"paths": int(d.path_id.nunique()), "images": len(d), "stage_counts": d.stage.value_counts().to_dict(), "model_disagreement_paths": int(d.dropna(subset=['foundation_range_eV']).path_id.nunique()), "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
