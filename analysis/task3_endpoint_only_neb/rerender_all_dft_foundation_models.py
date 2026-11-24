"""Refresh the ten-path same-geometry Foundation-model figures.

The ten-path processed table is the source of truth for this historical/all-
paths view.  QE image residuals are joined from the shared NEB parser, while
the main energy profiles deliberately retain points and line segments only.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import analyze_corrected_dft_foundation_comparison as renderer
from neb_plot_style import extract_qe_neb_uncertainty


OUT = ROOT / "exploratory_energy_audit" / "all_dft_foundation_models"
IMAGE_TABLE = OUT / "corrected_dft_foundation_image_profiles.csv"


def main() -> None:
    data = pd.read_csv(IMAGE_TABLE)
    uncertainty = extract_qe_neb_uncertainty(
        sorted(data.path_id.unique()),
        remote_root=ROOT / "remote_snapshot",
        n_images=10,
        n_tail_tables=5,
    )
    data = data.drop(
        columns=[
            "qe_energy_std_eV", "qe_energy_range_eV", "qe_force_error_eV_A",
            "qe_last_iteration", "qe_n_complete_tables", "qe_uncertainty_status",
        ],
        errors="ignore",
    ).merge(
        uncertainty[
            [
                "path_id", "image_index", "qe_energy_std_eV", "qe_energy_range_eV",
                "qe_force_error_eV_A", "qe_last_iteration", "qe_n_complete_tables",
                "qe_uncertainty_status",
            ]
        ],
        on=["path_id", "image_index"],
        how="left",
        validate="many_to_one",
    )
    data.to_csv(IMAGE_TABLE, index=False)
    renderer.OUT = OUT
    barriers, summary = renderer.make_tables(data)
    renderer.render_profiles(data)
    renderer.render_barrier_comparison(barriers, summary)
    print({
        "output": str(OUT),
        "paths": int(data.path_id.nunique()),
        "models": int(data.model_id.nunique()),
        "rows": int(len(data)),
    })


if __name__ == "__main__":
    main()
