"""Create a deterministic 4-family x 3-seed x 3-route input fixture.

The fixture is for visual regression and explanation only.  It is explicitly
labelled SYNTHETIC_DEMONSTRATION and must never be used as a benchmark result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(20261031)
    families = ["Foundation", "FT-600K", "FT-MultiT", "Scratch-MultiT"]
    seeds = [123, 234, 345]
    paths = [
        ("cr_canonical_in_gap", "In-gap diffusion", np.array([0.00, 0.20, 0.48, 0.74, 0.92, 0.97, 0.80, 0.53, 0.24, 0.00])),
        ("cr_canonical_gap_to_ql", "Gap-to-QL", np.array([0.00, 0.09, 0.30, 0.57, 0.76, 0.88, 0.70, 0.46, 0.17, 0.00])),
        ("cr_canonical_within_ql", "Within-QL", np.array([0.00, 0.16, 0.40, 0.71, 1.00, 0.90, 0.64, 0.35, 0.13, 0.00])),
    ]
    amplitude = {"Foundation": 0.055, "FT-600K": 0.080, "FT-MultiT": 0.115, "Scratch-MultiT": 0.160}
    rows: list[dict[str, object]] = []
    for family_index, family in enumerate(families):
        for seed_index, seed in enumerate(seeds):
            for path_index, (path_id, path_label, dft) in enumerate(paths):
                shape = np.sin(np.linspace(0, np.pi, len(dft))) * (1.0 if family_index % 2 == 0 else -0.75)
                drift = np.linspace(-0.4, 0.55, len(dft)) * (family_index - 1.5) / 4
                error = amplitude[family] * (shape + drift) + (seed_index - 1) * 0.008
                error += rng.normal(0.0, amplitude[family] * 0.13, size=len(dft))
                model_id = family.lower().replace("-", "_") + f"_seed{seed}"
                for image_index, (reference, delta) in enumerate(zip(dft, error)):
                    rows.append(
                        {
                            "record_id": f"DEMO-{family_index + 1}-{seed}-{path_index + 1}-{image_index}",
                            "model_family": family,
                            "model_id": model_id,
                            "model_label": family,
                            "training_seed": seed,
                            "path_id": path_id,
                            "path_label": path_label,
                            "image_index": image_index,
                            "reaction_coordinate": image_index / (len(dft) - 1),
                            "model_relative_energy_eV": reference + delta,
                            "dft_relative_energy_eV": reference,
                            "model_sha256": f"synthetic-{family_index + 1}-{seed}",
                            "path_geometry_sha256": f"synthetic-fixed-path-{path_index + 1}",
                            "dft_path_geometry_sha256": f"synthetic-fixed-path-{path_index + 1}",
                            "dft_reference_id": f"synthetic-dft-{path_index + 1}",
                            "dft_reference_status": "SYNTHETIC_DEMONSTRATION",
                            "observable": "FIXED_PATH_SINGLE_POINT",
                            "scientific_status": "SYNTHETIC_DEMONSTRATION",
                            "publication_status": "DEMONSTRATION_ONLY",
                            "source_profile_artifact": "deterministic_fixture",
                            "dataset_status": "SYNTHETIC_DEMONSTRATION",
                        }
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
