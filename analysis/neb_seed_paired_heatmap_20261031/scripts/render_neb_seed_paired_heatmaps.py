"""Render provenance-checked fixed-path NEB seed heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path

from migrationbench.neb_seed_paired_heatmap import render_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="TSV in NEB_FIXED_PATH_INPUT_SCHEMA.tsv form")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--baseline-family", help="Family used for seed-matched effect tables; defaults to input order")
    parser.add_argument("--expected-families", type=int, default=4)
    parser.add_argument("--expected-seeds-per-family", type=int, default=3)
    parser.add_argument("--expected-paths", type=int, default=3)
    parser.add_argument("--allow-synthetic-demo", action="store_true")
    args = parser.parse_args()
    render_bundle(
        args.input,
        args.output_dir,
        args.processed_dir,
        expected_families=args.expected_families,
        expected_seeds_per_family=args.expected_seeds_per_family,
        expected_paths=args.expected_paths,
        baseline_family=args.baseline_family,
        allow_synthetic_demo=args.allow_synthetic_demo,
    )


if __name__ == "__main__":
    main()
