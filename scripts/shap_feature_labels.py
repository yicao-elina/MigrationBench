"""Feature labels for the 390-dimensional DScribe SOAP layout used in Fig. 6."""

from __future__ import annotations


def _pair_labels(species: tuple[str, ...], n_max: int, l_max: int) -> list[str]:
    labels: list[str] = []
    for i, first in enumerate(species):
        for j in range(i, len(species)):
            second = species[j]
            if first == second:
                pairs = [(n1, n2) for n1 in range(1, n_max + 1)
                         for n2 in range(n1, n_max + 1)]
            else:
                pairs = [(n1, n2) for n1 in range(1, n_max + 1)
                         for n2 in range(1, n_max + 1)]
            for l in range(l_max + 1):
                labels.extend(f"{first}-{second}_n{n1}{n2}_l{l}" for n1, n2 in pairs)
    return labels


def soap_feature_names(species=("Cr", "Sb", "Te"), n_max=4, l_max=4) -> list[str]:
    labels = _pair_labels(tuple(sorted(species)), n_max, l_max)
    if len(labels) != 390:
        raise ValueError(f"unexpected SOAP feature count: {len(labels)}")
    return labels


def all_feature_names(species=("Cr", "Sb", "Te")) -> list[str]:
    return soap_feature_names(species) + [
        "n_atoms", "volume", "volume_per_atom", "cell_a", "cell_b", "cell_c",
        "angle_alpha", "angle_beta", "angle_gamma", "coord_mean", "coord_std",
        "dist_mean", "dist_std", "min_dist", "cr_fraction", "sb_fraction", "te_fraction",
    ]
