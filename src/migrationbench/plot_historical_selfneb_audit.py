#!/usr/bin/env python3
"""Plot a uniformly rescored historical self-NEB audit report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(args.report.read_text())
    energies = report["rescored_image_energies_eV"]
    ys = [energy - energies[0] for energy in energies]
    xs = list(range(1, len(energies) + 1))

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.23)
    ax.axhline(0.0, color="#6B7280", linewidth=0.9, linestyle="--")
    ax.plot(xs, ys, color="#2A6F97", linewidth=2, marker="o", markersize=5)
    ax.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], s=70, facecolor="white", edgecolor="#C44536", linewidth=2, zorder=4, label="frozen endpoints")
    minimum = report["minimum_image_index_qe"]
    ax.scatter([minimum], [ys[minimum - 1]], s=75, color="#2A9D8F", zorder=5, label="path minimum")
    ax.set_xticks(xs)
    ax.set_xlabel("NEB image index")
    ax.set_ylabel("Uniformly rescored E - E(initial) (eV)")
    ax.set_title("Historical Fig. S3 Foundation band: diagnostic rescoring", loc="left")
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.7)
    ax.legend(frameon=False, loc="upper right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    note = (
        f"forward={report['forward_barrier_eV']:.3f} eV; "
        f"reverse={report['reverse_barrier_eV']:.3f} eV; "
        f"max-min span={report['energy_span_eV']:.3f} eV\n"
        "Historical optimization invalid: DFT-energy endpoints were mixed with MACE-energy internal images."
    )
    fig.text(0.10, 0.055, note, ha="left", va="bottom", fontsize=8.5, color="#374151")

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(args.output_stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
