"""Build the seed-paired fixed-path input from the surviving campaign tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MODEL_GROUPS = [
    ("scratch_multit", "Scratch-MultiT", "from_scratch_all_data_v1"),
    ("scratch_600k", "Scratch-600K", "scratch_600k_grouped_v1"),
    ("ft_600k", "FT-600K", "ft_600k_grouped_A2"),
    ("ft_multit", "FT-MultiT", "ft_multit_grouped_A2"),
]
SYSTEMS = [
    ("pristine", "Pristine sliding", "1.bulk"),
    ("cr_in_gap", "Cr-in-gap sliding", "2.cr-in-gap"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.campaign_root
    profiles = pd.read_csv(root / "results/tables/fixed_profiles.csv")
    metrics = pd.read_csv(root / "results/tables/fixed_metrics_per_model.csv")
    dft = pd.read_csv(root / "inputs/dft/combined_sb2te3_energy.csv")
    dft_columns = {"1.bulk": "1.bulk", "2.cr-in-gap": "2.cr-in-gap"}

    rows: list[dict[str, object]] = []
    for group_key, family_label, model_prefix in MODEL_GROUPS:
        group_profiles = profiles.loc[profiles["model_id"].astype(str).str.startswith(model_prefix)].copy()
        model_ids = sorted(group_profiles["model_id"].dropna().unique())
        if len(model_ids) != 3:
            raise ValueError(f"{group_key} expected 3 model seeds, found {model_ids}")
        for model_id in model_ids:
            model_rows = group_profiles.loc[group_profiles["model_id"] == model_id]
            model_meta = metrics.loc[metrics["model_id"] == model_id]
            if model_meta.empty:
                raise ValueError(f"missing metric metadata for {model_id}")
            model_sha = str(model_rows["model_sha256"].dropna().iloc[0])
            training_seed = int(model_rows["training_seed"].dropna().iloc[0])
            model_label = str(model_rows["model_label"].dropna().iloc[0]).split(" / seed", 1)[0]
            for path_id, path_label, dft_reference_id in SYSTEMS:
                path_rows = model_rows.loc[model_rows["system_id"] == path_id].sort_values("image_index")
                if len(path_rows) != 10 or path_rows["image_index"].tolist() != list(range(10)):
                    raise ValueError(f"{model_id}/{path_id} does not contain the expected 10 images")
                path_meta = model_meta.loc[model_meta["system_id"] == path_id]
                if len(path_meta) != 1:
                    raise ValueError(f"missing metric metadata for {model_id}/{path_id}")
                manifest_path = Path(str(path_meta["manifest_path"].iloc[0]))
                manifest = json.loads(manifest_path.read_text())
                if str(manifest["dft_reference_id"]) != dft_reference_id:
                    raise ValueError(f"DFT reference mismatch for {model_id}/{path_id}")
                geometry_hash = str(manifest["input_sha256"])
                dft_values = dft[dft_columns[dft_reference_id]].to_numpy(float)
                reaction_coordinate = dft["Reaction_Coord"].to_numpy(float)
                if len(dft_values) != 10:
                    raise ValueError("combined DFT reference must contain 10 images")
                for row in path_rows.itertuples(index=False):
                    image_index = int(row.image_index)
                    rows.append(
                        {
                            "record_id": f"SLIDE24-{group_key}-s{training_seed}-{path_id}-img{image_index:02d}",
                            "model_family": family_label,
                            "model_id": str(model_id),
                            "model_label": f"{model_label} / seed {training_seed}",
                            "training_seed": training_seed,
                            "path_id": path_id,
                            "path_label": path_label,
                            "image_index": image_index,
                            "reaction_coordinate": float(reaction_coordinate[image_index]),
                            "model_relative_energy_eV": float(row.relative_energy_eV),
                            "dft_relative_energy_eV": float(dft_values[image_index]),
                            "model_sha256": model_sha,
                            "path_geometry_sha256": geometry_hash,
                            "dft_path_geometry_sha256": geometry_hash,
                            "dft_reference_id": dft_reference_id,
                            "dft_reference_status": str(manifest["dft_reference_status"]),
                            "observable": str(manifest["observable"]),
                            "scientific_status": str(row.status),
                            "publication_status": str(row.publication_status),
                            "source_profile_artifact": str(Path(str(path_meta["profile_path"].iloc[0]))),
                            "dataset_status": "PROVENANCE_PARTIAL",
                        }
                    )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, sep="\t", index=False)
    print(f"wrote {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
