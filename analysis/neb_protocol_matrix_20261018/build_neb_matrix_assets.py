#!/usr/bin/env python3
"""Build provenance-aware fixed/self-relaxed NEB matrix tables and figures.

This is a first-pass audit view.  Fixed-path and self-relaxed results are
intentionally kept as separate operators.  Missing cells remain missing and
legacy 81-atom Foundation self-NEB records are marked diagnostic rather than
silently promoted to validated rankings.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap

ATLAS = Path("/Users/alina/Documents/Codex/plot_library")
import sys
sys.path.insert(0, str(ATLAS / "src"))
from plot_atlas import apply_publication_style, bold_tick_labels, get_palette, panel_label, save_figure


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
DERIVED = ROOT / "derived"
FIGURES = ROOT / "figures"

FIXED_MODELS = [
    "foundation_omat_ft",
    "ft600k_seed123", "ft600k_seed234", "ft600k_seed345",
    "scratch_multit_seed123", "scratch_multit_seed234", "scratch_multit_seed345",
    "ft_multit_seed123", "ft_multit_seed234", "ft_multit_seed345",
]
SELF_MODELS = ["foundation_omat_ft", "ft600k_seed123", "scratch_multit_seed123", "ft_multit_seed123"]
PATHS = {61: ["1-2", "1-3", "1-4", "1-5", "1-7"], 81: ["neb_1", "neb_2", "neb_3", "neb_4", "neb_5"]}

MODEL_LABELS = {
    "foundation_omat_ft": "Foundation",
    "ft600k_seed123": "FT-600K (s123)",
    "ft600k_seed234": "FT-600K (s234)",
    "ft600k_seed345": "FT-600K (s345)",
    "scratch_multit_seed123": "Scratch-MultiT (s123)",
    "scratch_multit_seed234": "Scratch-MultiT (s234)",
    "scratch_multit_seed345": "Scratch-MultiT (s345)",
    "ft_multit_seed123": "FT-MultiT (s123)",
    "ft_multit_seed234": "FT-MultiT (s234)",
    "ft_multit_seed345": "FT-MultiT (s345)",
}

SHA_TO_MODEL = {
    # The self-NEB manifests preserve the full 64-character digest.  The
    # auxiliary specialist_models.tsv snapshot contains a truncated Foundation
    # digest, so keep the manifest-exact key here.
    "e618ad582b84239905b9c3b77ce6e9ce111b0ecd1533223a1a6aac7a696b8aa0": "foundation_omat_ft",
    "e618ad582b84239905b9c3b77ce6e9ce111b0ecd1533223a1a6aac7a696b8aa": "foundation_omat_ft",
    "6829d16e68bdd1238ee4226363f76ceac936d9e7325c9282283d72093e2bb88a": "ft600k_seed123",
    "0d83b39b8ce81a7b7a4a26e5a3dfe51645aee6ca73180d72da4007d701b70fcf": "scratch_multit_seed123",
    "f61082f629b1f03db12e9b2f34a525790979d831ada3c895cae48eed18cf7588": "ft_multit_seed123",
}

OLD_81_FOUNDATION = {
    f"neb_{n}": Path(
        f"data/published/cluster/mb_mlff_cpu_81_neb_{n}__s42_307635{16+n}"
        f"/mlff_81_neb_{n}_s42/mlff_neb_manifest.json"
    )
    for n in range(1, 6)
}


def jload(path: Path) -> dict:
    return json.loads(path.read_text())


def model_family(model_id: str) -> str:
    if model_id.startswith("foundation"):
        return "Foundation"
    if model_id.startswith("ft600k"):
        return "FT-600K"
    if model_id.startswith("scratch"):
        return "Scratch-MultiT"
    return "FT-MultiT"


def reference_map() -> dict[tuple[int, str], dict]:
    out = {}
    with (RAW / "dft_neb_barriers.csv").open() as handle:
        for row in csv.DictReader(handle):
            path = row["path"]
            if path.startswith("neb_"):
                cell = 81
            elif path in PATHS[61]:
                cell = 61
            else:
                continue
            out[(cell, path)] = {
                "reference_eV": float(row["barrier_eV"]),
                "reference_status": "provisional_candidate" if cell == 61 else "high_force_candidate",
                "reference_source": "data/processed/dft_neb_barriers.csv",
            }
    return out


def fixed_records(refs: dict) -> list[dict]:
    records = []
    for cell, subdir in ((61, "specialist_61"), (81, "specialist_81")):
        found = {}
        for path in sorted((RAW / "fixed" / subdir).glob("*.json")):
            d = jload(path)
            found[(d["model_id"], d["path_id"].replace("81-neb-", "neb_"))] = (path, d)
        for model_id in FIXED_MODELS:
            for path_id in PATHS[cell]:
                key = (model_id, path_id)
                ref = refs[(cell, path_id)]
                if key not in found:
                    records.append({
                        "protocol": "fixed_path", "cell_atoms": cell, "path_id": path_id,
                        "model_id": model_id, "model_label": MODEL_LABELS[model_id],
                        "family": model_family(model_id), "seed": model_id.split("seed")[-1],
                        **ref, "barrier_eV": np.nan, "barrier_error_eV": np.nan,
                        "force_max_eV_A": np.nan, "endpoint_delta_eV": np.nan,
                        "finite": False, "optimizer_converged": "",
                        "final_true_force_eV_A": np.nan, "final_neb_residual_eV_A": np.nan,
                        "source": "pending_gpu_retry_32199417",
                        "run_status": "PENDING_GPU_RETRY", "publication_status": "MISSING",
                        "failure_flag": "",
                    })
                    continue
                path, d = found[key]
                fmax = max(d.get("force_max_eV_per_A", []) or [np.nan])
                failure = "force_pathology" if np.isfinite(fmax) and fmax > 10 else ""
                records.append({
                    "protocol": "fixed_path", "cell_atoms": cell, "path_id": path_id,
                    "model_id": model_id, "model_label": MODEL_LABELS[model_id],
                    "family": model_family(model_id), "seed": model_id.split("seed")[-1],
                    **ref, "barrier_eV": d.get("barrier_eV", np.nan),
                    "barrier_error_eV": d.get("barrier_eV", np.nan) - ref["reference_eV"],
                    "force_max_eV_A": fmax, "endpoint_delta_eV": d.get("endpoint_delta_eV", np.nan),
                    "finite": bool(d.get("finite", False)), "optimizer_converged": "",
                    "final_true_force_eV_A": np.nan, "final_neb_residual_eV_A": np.nan,
                    "source": str(path), "run_status": "COMPLETE_FINITE" if d.get("finite") else "FAILED_NONFINITE",
                    "publication_status": "FAILURE_MODE" if failure else "PROVISIONAL_FIXED_PATH",
                    "failure_flag": failure,
                })
    return records


def self_records(refs: dict) -> list[dict]:
    found = {}
    self_roots = [RAW / "self_current", RAW / "self_gpu_retry"]
    for self_root in self_roots:
        for path in self_root.glob("**/mlff_neb_manifest.json"):
            d = jload(path)
            model_id = SHA_TO_MODEL.get(d.get("model_path_sha256"))
            if model_id is None:
                continue
            name = path.parent.name
            if name.startswith("mlff_61_"):
                cell, path_id = 61, name.removeprefix("mlff_61_").removesuffix("_s42")
            elif name.startswith("mlff_81_"):
                cell, path_id = 81, name.removeprefix("mlff_81_").removesuffix("_s42")
            else:
                continue
            source_tag = "GPU_RETRY_CANONICAL" if self_root.name == "self_gpu_retry" else "CURRENT_CANONICAL"
            found[(cell, path_id, model_id)] = (path, d, source_tag)
    # The existing 81-atom Foundation records have exact input/checkpoint identity,
    # but lack the new final convergence fields; retain them as diagnostic only.
    for path_id, path in OLD_81_FOUNDATION.items():
        if path.exists():
            d = jload(path)
            found[(81, path_id, "foundation_omat_ft")] = (path, d, "LEGACY_DIAGNOSTIC")

    records = []
    for cell in (61, 81):
        for model_id in SELF_MODELS:
            for path_id in PATHS[cell]:
                ref = refs[(cell, path_id)]
                key = (cell, path_id, model_id)
                if key not in found:
                    records.append({
                        "protocol": "self_relaxed", "cell_atoms": cell, "path_id": path_id,
                        "model_id": model_id, "model_label": MODEL_LABELS[model_id],
                        "family": model_family(model_id), "seed": "42",
                        **ref, "barrier_eV": np.nan, "barrier_error_eV": np.nan,
                        "force_max_eV_A": np.nan, "endpoint_delta_eV": np.nan,
                        "finite": False, "optimizer_converged": "",
                        "final_true_force_eV_A": np.nan, "final_neb_residual_eV_A": np.nan,
                        "optimizer_steps": np.nan, "source": "pending_gpu_retry_32199418",
                        "run_status": "PENDING_GPU_RETRY", "publication_status": "MISSING",
                        "failure_flag": "",
                    })
                    continue
                path, d, provenance = found[key]
                profile = d.get("profile") or []
                barrier = d.get("barrier_proxy_eV", np.nan)
                if not np.isfinite(barrier) and profile:
                    barrier = max(float(x.get("rel_energy_eV", np.nan)) for x in profile)
                converged = d.get("optimizer_converged")
                steps = d.get("optimizer_steps_completed", np.nan)
                true_force = d.get("final_max_internal_true_force_eV_A", np.nan)
                neb_resid = d.get("final_max_neb_residual_eV_A", np.nan)
                run_status = "COMPLETE_CONVERGED" if converged is True else "COMPLETE_NONCONVERGED"
                if provenance == "LEGACY_DIAGNOSTIC":
                    run_status = "LEGACY_DIAGNOSTIC"
                records.append({
                    "protocol": "self_relaxed", "cell_atoms": cell, "path_id": path_id,
                    "model_id": model_id, "model_label": MODEL_LABELS[model_id],
                    "family": model_family(model_id), "seed": "42", **ref,
                    "barrier_eV": barrier,
                    "barrier_error_eV": barrier - ref["reference_eV"] if np.isfinite(barrier) else np.nan,
                    "force_max_eV_A": np.nan, "endpoint_delta_eV": np.nan,
                    "finite": bool(profile), "optimizer_converged": converged if converged is not None else "",
                    "final_true_force_eV_A": true_force, "final_neb_residual_eV_A": neb_resid,
                    "optimizer_steps": steps, "source": str(path), "run_status": run_status,
                    "publication_status": "PROVENANCE_PARTIAL" if provenance == "LEGACY_DIAGNOSTIC" else "PROVISIONAL_SELF_RELAXED",
                    "failure_flag": "legacy_missing_convergence_fields" if provenance == "LEGACY_DIAGNOSTIC" else "",
                })
    return records


def write_outputs(df: pd.DataFrame) -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    df.to_csv(DERIVED / "neb_matrix_status.csv", index=False, float_format="%.8f")
    df.to_csv(DERIVED / "neb_matrix_status.tsv", index=False, sep="\t", float_format="%.8f")
    failure_counts = {
        "|".join(map(str, key if isinstance(key, tuple) else (key,))): int(value)
        for key, value in df.query("failure_flag != ''").groupby(["protocol", "failure_flag"]).size().items()
    }
    summary = {
        "fixed_path": df.query("protocol == 'fixed_path'").groupby("run_status").size().to_dict(),
        "self_relaxed": df.query("protocol == 'self_relaxed'").groupby("run_status").size().to_dict(),
        "failure_modes": failure_counts,
        "note": "Fixed-path and self-relaxed operators are not pooled into one ranking. DFT values are provisional/candidate references.",
    }
    (DERIVED / "neb_matrix_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")


def _cmap(name="descriptor_reduction"):
    p = get_palette(name)
    return LinearSegmentedColormap.from_list(f"{name}_seq", p["sequential"])


def heatmap_panel(ax, df, cell, protocol, vmin, vmax, *, show_y=True):
    paths = PATHS[cell]
    models = FIXED_MODELS if protocol == "fixed_path" else SELF_MODELS
    sub = df.query("protocol == @protocol and cell_atoms == @cell").copy()
    mat = np.full((len(models), len(paths)), np.nan)
    status = {}
    for i, model in enumerate(models):
        for j, path in enumerate(paths):
            row = sub[(sub.model_id == model) & (sub.path_id == path)].iloc[0]
            mat[i, j] = row.barrier_eV if np.isfinite(row.barrier_eV) else np.nan
            status[(i, j)] = row
    cmap = _cmap()
    cmap.set_bad("#F0F0F0")
    norm = LogNorm(vmin=vmin, vmax=vmax) if protocol == "fixed_path" else Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(paths)), paths)
    ax.set_yticks(range(len(models)), [MODEL_LABELS[m] for m in models] if show_y else [])
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlabel("Pathway")
    if show_y:
        ax.set_ylabel("MLIP route / seed")
    ax.set_title(f"{cell}-atom {'fixed-path' if protocol == 'fixed_path' else 'self-relaxed'}")
    for i in range(len(models)):
        for j in range(len(paths)):
            row = status[(i, j)]
            if not np.isfinite(row.barrier_eV):
                ax.add_patch(patches.Rectangle((j - .5, i - .5), 1, 1, facecolor="#F0F0F0", edgecolor="white", lw=.7))
                ax.text(j, i, "P", ha="center", va="center", color="#777777", fontsize=8, fontweight="bold")
                continue
            value = float(row.barrier_eV)
            label = f"{value:.2f}" if value < 100 else ">100"
            color = "white" if (protocol == "fixed_path" and value > 5) or (protocol == "self_relaxed" and value > (vmin + vmax) / 2) else "#202020"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=7)
            if row.failure_flag:
                ax.plot(j, i, marker="x", markersize=8, mew=1.2, color="#111111")
            if protocol == "self_relaxed":
                if row.run_status == "LEGACY_DIAGNOSTIC":
                    ax.text(j + .36, i - .34, "D", ha="right", va="top", fontsize=7, color="#333333", fontweight="bold")
                elif row.optimizer_converged is True:
                    ax.text(j + .43, i - .34, "C", ha="right", va="top", fontsize=7, color="#147A4B", fontweight="bold")
                elif row.run_status == "COMPLETE_NONCONVERGED":
                    ax.text(j + .43, i - .34, "!", ha="right", va="top", fontsize=7, color="#A33A2B", fontweight="bold")
    ax.set_xlim(-.5, len(paths)-.5)
    ax.set_ylim(len(models)-.5, -.5)
    for spine in ax.spines.values():
        spine.set_linewidth(.8)
    bold_tick_labels(ax)
    return im


def make_figures(df: pd.DataFrame) -> None:
    apply_publication_style(font_size=10)
    FIGURES.mkdir(parents=True, exist_ok=True)
    pending_fixed = int((df.protocol == "fixed_path").mul(df.run_status == "PENDING_GPU_RETRY").sum())
    pending_self = int((df.protocol == "self_relaxed").mul(df.run_status == "PENDING_GPU_RETRY").sum())
    # Fixed-path: common DFT geometry, model energy assignment. Log color scale keeps
    # severe high-distortion failures visible without hiding the low-barrier regime.
    fig, axes = plt.subplots(2, 1, figsize=(11.2, 7.8), constrained_layout=True, gridspec_kw={"height_ratios": [1.15, 1]})
    im = heatmap_panel(axes[0], df, 61, "fixed_path", .3, 100)
    heatmap_panel(axes[1], df, 81, "fixed_path", .3, 100)
    panel_label(axes[0], "a", x=-.075, y=1.06); panel_label(axes[1], "b", x=-.075, y=1.06)
    cbar = fig.colorbar(im, ax=axes, shrink=.86, pad=.018)
    cbar.set_label("Fixed-path barrier assigned by MLIP (eV; log scale)")
    cbar.ax.tick_params(labelsize=9)
    pending_note = "P = pending GPU retry; " if pending_fixed else ""
    fig.text(.01, -.008, pending_note + "x = max fixed-path force > 10 eV A-1. DFT candidates are kept in the table; their references are provisional/high-force candidates.", fontsize=8, color="#222222")
    save_figure(fig, FIGURES / "fixed_path_neb_matrix", dpi=320)
    fig.savefig(FIGURES / "fixed_path_neb_matrix.pdf", bbox_inches="tight", metadata={"Creator": "MigrationBench Plot Atlas", "Date": None})
    plt.close(fig)

    # Self-relaxed: MLIP-native barrier proxy and convergence diagnostics. Missing
    # GPU cells remain gray; D marks the legacy 81-atom Foundation diagnostic.
    vals = df[(df.protocol == "self_relaxed") & np.isfinite(df.barrier_eV)].barrier_eV
    vmax = max(1.0, float(vals.max()) if len(vals) else 1.0)
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 5.8), constrained_layout=True)
    im = heatmap_panel(axes[0], df, 61, "self_relaxed", 0, vmax)
    heatmap_panel(axes[1], df, 81, "self_relaxed", 0, vmax)
    panel_label(axes[0], "a", x=-.075, y=1.06); panel_label(axes[1], "b", x=-.075, y=1.06)
    cbar = fig.colorbar(im, ax=axes, shrink=.86, pad=.018)
    cbar.set_label("Self-relaxed MLIP barrier proxy (eV)")
    cbar.ax.tick_params(labelsize=9)
    pending_note = "; P = pending GPU retry" if pending_self else ""
    fig.text(.01, -.008, "C = optimizer converged; ! = finite but not converged; D = legacy 81-atom Foundation diagnostic" + pending_note + ". These are MLIP-native path-generation diagnostics, not DFT barriers.", fontsize=8, color="#222222")
    save_figure(fig, FIGURES / "self_relaxed_neb_matrix", dpi=320)
    fig.savefig(FIGURES / "self_relaxed_neb_matrix.pdf", bbox_inches="tight", metadata={"Creator": "MigrationBench Plot Atlas", "Date": None})
    plt.close(fig)


def main() -> None:
    refs = reference_map()
    records = fixed_records(refs) + self_records(refs)
    df = pd.DataFrame(records)
    write_outputs(df)
    make_figures(df)
    print(df.groupby(["protocol", "run_status"]).size().to_string())
    print(f"Wrote tables to {DERIVED}")
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
