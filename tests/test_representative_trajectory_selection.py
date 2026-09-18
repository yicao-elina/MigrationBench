import copy
import math
import sys
from pathlib import Path

import pytest
from ase import Atoms

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "migrationbench"
sys.path.insert(0, str(SCRIPT_DIR))

from select_representative_trajectories import (  # noqa: E402
    displacement_signature,
    facility_coverage,
    greedy_select,
    local_signature,
    normalized_energy_curve,
    pair_signature,
    raw_pair_distances,
    topology_signature,
)
from repair_neb_images import min_pair_distance, repel_close_pairs  # noqa: E402


def transform_image(image, rotation, translation):
    result = copy.deepcopy(image)
    result["positions"] = [
        tuple(
            sum(rotation[row][column] * position[column] for column in range(3))
            + translation[row]
            for row in range(3)
        )
        for position in image["positions"]
    ]
    return result


def test_distance_descriptors_are_rigid_motion_invariant_without_pbc():
    initial = {
        "symbols": ["Cr", "Te", "Sb", "Te"],
        "positions": [(0.0, 0.0, 0.0), (2.5, 0.0, 0.0), (0.0, 2.8, 0.0), (0.0, 0.0, 3.1)],
        "cell": None,
    }
    moved = copy.deepcopy(initial)
    moved["positions"][0] = (0.4, 0.3, 0.2)
    rotation = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    translation = (4.2, -3.1, 1.7)
    transformed_initial = transform_image(initial, rotation, translation)
    transformed_moved = transform_image(moved, rotation, translation)

    assert pair_signature(moved) == pytest.approx(pair_signature(transformed_moved))
    assert local_signature(moved, 0, 3) == pytest.approx(local_signature(transformed_moved, 0, 3))
    assert topology_signature(moved, 0, [2.6, 3.2, 4.0]) == pytest.approx(
        topology_signature(transformed_moved, 0, [2.6, 3.2, 4.0])
    )
    assert displacement_signature(moved, initial, 0) == pytest.approx(
        displacement_signature(transformed_moved, transformed_initial, 0)
    )


def test_normalized_energy_curve_is_positive_affine_invariant():
    rows = [{"energy_eV": value} for value in (-10.0, -8.0, -9.0)]
    transformed = [{"energy_eV": 3.5 * row["energy_eV"] + 117.0} for row in rows]
    assert [row[0] for row in normalized_energy_curve(rows)] == pytest.approx(
        [row[0] for row in normalized_energy_curve(transformed)]
    )


def test_pair_distance_refuses_cross_system_comparison():
    curve = {name: [[0.0], [1.0]] for name in (
        "geometry_frechet", "local_environment", "displacement_field", "energy_profile", "bond_topology"
    )}
    with pytest.raises(ValueError, match="Cross-system"):
        raw_pair_distances(
            {"system_id": "a", "curves": curve},
            {"system_id": "b", "curves": curve},
        )


def test_facility_coverage_is_monotonic_and_bounded():
    ids = ["a", "b", "c"]
    similarity = {
        ("a", "a"): 1.0, ("b", "b"): 1.0, ("c", "c"): 1.0,
        ("a", "b"): 0.5, ("b", "a"): 0.5,
        ("a", "c"): 0.1, ("c", "a"): 0.1,
        ("b", "c"): 0.2, ("c", "b"): 0.2,
    }
    values = [facility_coverage(ids, selected, similarity) for selected in ([], ["a"], ["a", "b"], ids)]
    assert values == sorted(values)
    assert values[0] == 0.0
    assert values[-1] == 1.0


def test_greedy_selection_reaches_target_without_duplicate_paths():
    descriptors = [
        {"path_id": name, "quality_tiebreak_score": quality}
        for name, quality in (("a", 0.8), ("b", 0.5), ("c", 0.9))
    ]
    pairs = [
        {"path_a": "a", "path_b": "b", "similarity": math.exp(-0.2)},
        {"path_a": "a", "path_b": "c", "similarity": math.exp(-2.0)},
        {"path_a": "b", "path_b": "c", "similarity": math.exp(-1.8)},
    ]
    selected = greedy_select(descriptors, pairs, 0.90)
    assert selected[-1]["cumulative_coverage"] >= 0.90
    assert len({row["path_id"] for row in selected}) == len(selected)
    assert all(
        right["cumulative_coverage"] >= left["cumulative_coverage"]
        for left, right in zip(selected, selected[1:])
    )

    valid_only = greedy_select(descriptors, pairs, 0.90, allowed_centers={"a", "b"})
    assert {row["path_id"] for row in valid_only} <= {"a", "b"}
    assert valid_only[-1]["cumulative_coverage"] < 0.90


def test_migrant_only_repair_preserves_hosts_and_endpoints():
    endpoint_a = Atoms("CrSbTe", positions=[(0, 0, 0), (2.5, 0, 0), (0, 3, 0)], cell=[10, 10, 10], pbc=True)
    middle = endpoint_a.copy()
    middle.positions[0] = (1.1, 0, 0)
    endpoint_b = endpoint_a.copy()
    endpoint_b.positions[0] = (4.0, 0, 0)
    original_hosts = middle.positions[1:].copy()

    repaired, actions = repel_close_pairs(
        [endpoint_a, middle, endpoint_b], 1.8, 20, 0.1, migrant_index=0, policy="migrant-only"
    )

    assert repaired[0].positions == pytest.approx(endpoint_a.positions)
    assert repaired[-1].positions == pytest.approx(endpoint_b.positions)
    assert repaired[1].positions[1:] == pytest.approx(original_hosts)
    assert min_pair_distance(repaired[1])[0] >= 1.8 - 1.0e-8
    assert actions[0]["moved_atom_indices"] == [0]
    assert actions[0]["threshold_pass"] is True
