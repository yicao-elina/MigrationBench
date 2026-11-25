"""SI Fig. S15: fixed-path energy-profile comparison (a) and self-relaxed NEB stability (b)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
FIXED_CSV = REPO / "paper/revision1/data_processed/cluster/neb_benchmark/unified_fixed_geometry_barriers_audit.csv"
SELF_NEB_CSV = REPO / "data/processed/neb_selfneb_scratch_per_step.csv"
OUT = REPO / "paper/revision1/figures/figS_neb_operators.pdf"

WITHIN_GAP_PATH = "1-4"
MODEL_ORDER = ["MACE Foundation", "MACE Scratch", "FT-600K", "FT-MultiT"]
MODEL_LABEL = {"MACE Foundation": "Foundation", "MACE Scratch": "Scratch",
               "FT-600K": "FT-600K", "FT-MultiT": "FT-MultiT"}
MODEL_COLOR = {"MACE Foundation": "#d62728", "MACE Scratch": "#e8792b",
               "FT-600K": "#08306b", "FT-MultiT": "#6baed6"}


def load_fixed_path_values() -> tuple[pd.Series, float]:
    table = pd.read_csv(FIXED_CSV)
    rows = table[table.path == WITHIN_GAP_PATH].set_index("model_name")
    return rows.loc[MODEL_ORDER, "mace_barrier_eV"], float(rows.dft_reference_barrier_eV.iloc[0])


def load_band_maximum() -> pd.Series:
    steps = pd.read_csv(SELF_NEB_CSV)
    interior = steps[steps.image > 0]
    band_max = interior.groupby("step").rel_to_image0_eV.max()
    return band_max.where(band_max > 0)


def main() -> None:
    fixed_values, qe_reference = load_fixed_path_values()
    band_max = load_band_maximum()

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.6, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})

    ax_a.bar([MODEL_LABEL[m] for m in MODEL_ORDER], fixed_values.values,
             color=[MODEL_COLOR[m] for m in MODEL_ORDER], edgecolor="black", linewidth=0.6)
    ax_a.axhline(qe_reference, color="black", linestyle="--", linewidth=1.0)
    ax_a.set_xlim(-0.6, 4.5)
    ax_a.text(3.55, qe_reference + 0.12, f"QE image-set\nreference\n{qe_reference:.3f} eV", ha="left", va="bottom", fontsize=7)
    for x, value in enumerate(fixed_values.values):
        ax_a.text(x, value + 0.06, f"{value:.2f}", ha="center", fontsize=8)
    ax_a.set_ylabel("Fixed-path energy difference (eV)")
    ax_a.set_title("(a) Fixed-path reference profile, within-gap path", fontsize=9)
    ax_a.tick_params(axis="x", labelsize=8)

    ax_b.semilogy(band_max.index, band_max.values, color=MODEL_COLOR["MACE Scratch"], linewidth=0.9)
    ax_b.set_xlabel("NEB optimizer step")
    ax_b.set_ylabel("Maximum image energy above endpoint (eV)")
    ax_b.set_title("(b) Self-relaxed NEB, Scratch model", fontsize=9)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
