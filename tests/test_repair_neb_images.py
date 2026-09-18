import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "migrationbench"))

from repair_neb_images import (  # noqa: E402
    attach_cell_from_qe_template,
    resample_images_by_arc_length,
    unwrap_images,
)
from prepare_qe_extxyz_relax_ab import write_branch  # noqa: E402
from compare_endpoint_relaxation_speed import transform_shift_A  # noqa: E402
from monitor_mlff_neb_jobs import (  # noqa: E402
    PROVENANCE_CHECKS,
    classify as classify_mlff,
    expected_jobs,
    path_change_metrics,
)
from reconstruct_migrant_clearance_graph import fibonacci_directions, shortest_layered_path  # noqa: E402
from run_mlff_neb import ReferenceTetherCalculator  # noqa: E402
try:
    from run_mlff_neb import finalize_dual_history_manifest  # noqa: E402
except ImportError:
    finalize_dual_history_manifest = None
from compare_mlff_trust_region import comparison_decision as trust_region_decision  # noqa: E402


class ZeroCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}


def test_unwrap_preserves_minimum_image_displacement():
    cell = np.diag([10.0, 10.0, 10.0])
    images = [
        Atoms("He", positions=[[9.8, 1.0, 1.0]], cell=cell, pbc=True),
        Atoms("He", positions=[[0.2, 1.0, 1.0]], cell=cell, pbc=True),
        Atoms("He", positions=[[0.6, 1.0, 1.0]], cell=cell, pbc=True),
    ]
    output = unwrap_images(images)
    assert np.allclose([image.positions[0, 0] for image in output], [9.8, 10.2, 10.6])


def test_resampling_preserves_endpoints_and_avoids_lattice_jump():
    cell = np.diag([10.0, 10.0, 10.0])
    images = unwrap_images([
        Atoms("He2", positions=[[9.8, 1.0, 1.0], [5.0, 5.0, 5.0]], cell=cell, pbc=True),
        Atoms("He2", positions=[[0.2, 1.0, 1.0], [5.2, 5.0, 5.0]], cell=cell, pbc=True),
    ])
    output = resample_images_by_arc_length(images, 5)
    assert len(output) == 5
    assert np.allclose(output[0].positions, images[0].positions)
    assert np.allclose(output[-1].positions, images[-1].positions)
    steps = [np.linalg.norm(b.positions - a.positions, axis=1).max() for a, b in zip(output, output[1:])]
    assert max(steps) < 0.2


def test_plain_xyz_requires_explicit_cell_template(tmp_path):
    images = [Atoms("He", positions=[[0.0, 0.0, 0.0]])]
    template = tmp_path / "pw.in"
    template.write_text(
        "CELL_PARAMETERS angstrom\n2 0 0\n0 2 0\n0 0 2\n"
        "ATOMIC_POSITIONS angstrom\nHe 0 0 0\n"
    )
    attached = attach_cell_from_qe_template(images, template)
    assert attached[0].pbc.all()
    assert np.allclose(attached[0].cell.array, np.diag([2.0, 2.0, 2.0]))


def test_cell_template_atom_count_is_not_silently_relevant(tmp_path):
    image = Atoms("He2", positions=[[0, 0, 0], [1, 1, 1]])
    template = tmp_path / "pw.in"
    template.write_text(
        "CELL_PARAMETERS angstrom\n3 0 0\n0 3 0\n0 0 3\n"
        "ATOMIC_POSITIONS angstrom\nHe 0 0 0\nHe 1 1 1\n"
    )
    attached = attach_cell_from_qe_template([image], template)
    assert len(attached[0]) == 2
    assert attached[0].cell.rank == 3


def test_qe_ab_direct_manifest_is_identity(tmp_path):
    atoms = Atoms("HeCr", positions=[[0, 0, 0], [1, 1, 1]], cell=np.diag([3.0] * 3), pbc=True)
    common = {
        "path_id": "fixture",
        "source_image_index_qe": 2,
        "seed": 47,
        "resources": {"pw_max_seconds": 100},
        "transform": {
            "name": "hard_clearance",
            "cr_displacement_A": 0.4,
            "metrics": {"cr_displacement_A": 0.4},
        },
    }
    template = """&CONTROL
  prefix = 'old'
/
&SYSTEM
  nat = 2
  ntyp = 2
/
&ELECTRONS
/
ATOMIC_SPECIES
He 4.0 He.upf
Cr 52.0 Cr.upf
ATOMIC_POSITIONS angstrom
He 0 0 0
Cr 1 1 1
K_POINTS gamma
CELL_PARAMETERS angstrom
3 0 0
0 3 0
0 0 3
"""
    direct = write_branch(tmp_path, common, "direct", atoms, template, "direct", "direct_job")
    transformed = write_branch(tmp_path, common, "transformed", atoms, template, "changed", "changed_job")
    assert direct["transform"]["name"] == "identity"
    assert direct["transform"]["metrics"]["cr_displacement_A"] == 0.0
    assert transformed["transform"]["metrics"]["cr_displacement_A"] == 0.4
    assert "calculation = 'relax'" in Path(direct["relax_input"]).read_text()


def test_transform_shift_accepts_flat_and_nested_manifests():
    assert transform_shift_A({"transform": {"cr_displacement_A": 0.3}}) == 0.3
    assert transform_shift_A({"transform": {"metrics": {"cr_displacement_A": 0.4}}}) == 0.4
    with pytest.raises(ValueError, match="lacks Cr displacement"):
        transform_shift_A({"transform": {}})


def test_mlff_monitor_rejects_nonperiodic_completed_output():
    manifest = {
        "optimizer_converged": True,
        "final_max_internal_neb_force_eV_A": 0.05,
        "fmax_target_eV_A": 0.1,
    }
    valid = {name: True for name in PROVENANCE_CHECKS}
    valid.update({
        "periodic_output": True,
        "final_geometry": True,
        "endpoints_preserved": True,
        "no_large_energy_collapse": True,
    })
    assert classify_mlff("COMPLETED", manifest, valid) == "ready_for_mechanism_clustering"
    assert classify_mlff("COMPLETED", manifest, {**valid, "periodic_output": False}) == "invalid_nonperiodic_output"
    assert classify_mlff("COMPLETED", manifest, {**valid, "runtime_input_hashes": False}) == "invalid_provenance_or_history"
    assert classify_mlff("COMPLETED", manifest, {**valid, "no_large_energy_collapse": False}) == "terminal_basin_collapse"


def test_mlff_monitor_labels_restrained_convergence_for_review_only():
    valid = {name: True for name in PROVENANCE_CHECKS}
    valid.update({
        "periodic_output": True,
        "final_geometry": True,
        "endpoints_preserved": True,
        "no_large_energy_collapse": True,
    })
    manifest = {
        "optimizer_converged": False,
        "optimizer_converged_under_optimization_potential": True,
        "reference_tether": {"enabled": True},
        "final_max_internal_neb_force_eV_A": 0.05,
        "fmax_target_eV_A": 0.1,
    }
    assert classify_mlff("COMPLETED", manifest, valid) == "ready_for_dft_preconditioner_review"
    manifest["optimizer_converged_under_optimization_potential"] = False
    assert classify_mlff("COMPLETED", manifest, valid) == "terminal_restrained_unconverged"


def test_reference_tether_uses_minimum_image_energy_and_force():
    reference = Atoms("CrHe", positions=[[9.8, 1, 1], [5, 5, 5]], cell=np.diag([10.0] * 3), pbc=True)
    moved = reference.copy()
    moved.positions[0, 0] = 0.2
    moved.calc = ReferenceTetherCalculator(ZeroCalculator(), reference, [2.0, 0.0])
    assert moved.get_potential_energy() == pytest.approx(0.16)
    forces = moved.get_forces()
    assert forces[0, 0] == pytest.approx(-0.8)
    assert np.allclose(forces[1], 0.0)
    assert moved.calc.max_displacement_A == pytest.approx(0.4)


@pytest.mark.skipif(
    finalize_dual_history_manifest is None,
    reason="active Rockfish runner is frozen before dual-manifest helper",
)
def test_dual_history_manifest_declares_nonphysical_tether_semantics(tmp_path):
    extxyz = tmp_path / "history.extxyz"
    csv_path = tmp_path / "history.csv"
    extxyz.write_text("fixture\n")
    csv_path.write_text("iteration,image\n0,0\n0,1\n1,0\n1,1\n")
    payload, output = finalize_dual_history_manifest(
        tmp_path,
        extxyz,
        csv_path,
        n_images=2,
        optimizer_steps=1,
        tether={"enabled": True, "migrant_k_eV_A2": 4.0, "host_k_eV_A2": 0.5},
    )
    assert payload["complete"] is True
    assert payload["reported_barrier_source"] == "base_MACE_without_reference_tether"
    assert payload["field_semantics"]["tether_energy_eV"].startswith("nonphysical")
    assert json.loads(output.read_text())["row_count"] == 4


def test_trust_region_comparison_is_fail_closed():
    valid_row = {
        "classification": "ready_for_dft_preconditioner_review",
        "checks": {"dual_potential_history": True},
    }
    manifest = {"reference_tether": {"enabled": True}}
    historical = {
        "manifest_hash_match": True,
        "unique_input_row_match": True,
        "endpoints_accepted": False,
        "candidate_production_eligible": False,
    }
    assert trust_region_decision(True, valid_row, manifest, historical) == (
        "trust_region_diagnostic_ready_as_curvature_donor_requires_endpoint_remap"
    )
    production = {
        **historical,
        "endpoints_accepted": True,
        "candidate_production_eligible": True,
    }
    assert trust_region_decision(True, valid_row, manifest, production) == (
        "trust_region_candidate_ready_for_dft_ab_design"
    )
    assert trust_region_decision(False, valid_row, manifest, production) == (
        "invalid_physical_identity_mismatch"
    )
    assert trust_region_decision(
        True, {**valid_row, "checks": {}}, manifest, production
    ) == "invalid_incomplete_dual_potential_history"
    assert trust_region_decision(
        True, valid_row, {"reference_tether": {"enabled": False}}, production
    ) == "invalid_missing_reference_tether"


def test_mlff_monitor_path_change_metrics_are_periodic_and_endpoint_aware(tmp_path):
    from ase.io import write

    cell = np.diag([10.0, 10.0, 10.0])
    initial = [
        Atoms("CrHe", positions=[[9.8 + 0.1 * i, 1, 1], [5, 5, 5]], cell=cell, pbc=True)
        for i in range(4)
    ]
    final = [atoms.copy() for atoms in initial]
    final[1].positions[0, 0] += 10.2
    final[2].positions[0, 0] += 0.2
    input_path = tmp_path / "input.extxyz"
    output_path = tmp_path / "output.extxyz"
    write(input_path, initial)
    write(output_path, final)
    metrics = path_change_metrics(input_path, output_path)
    assert metrics["endpoint_max_displacement_A"] == pytest.approx(0.0)
    assert metrics["migrant_internal_coordinate_rms_A"] == pytest.approx(0.2)
    assert metrics["final_geometry"]["max_migrant_step_A"] < 0.5


def test_mlff_monitor_combines_paired_and_single_job_configs(tmp_path):
    model_hash = "a" * 64
    paired = tmp_path / "paired.json"
    paired.write_text(json.dumps({
        "model_path_sha256": model_hash,
        "jobs": [{"job_id": 1, "input_sha256": "b" * 64}],
    }))
    single = tmp_path / "single.json"
    single.write_text(json.dumps({
        "job_id": 2,
        "job_name": "clearance",
        "path_id": "fixture_clearance",
        "input_images_sha256": "c" * 64,
        "model_path_sha256": model_hash,
    }))
    result = expected_jobs([paired, single])
    assert result["1"]["input_sha256"] == "b" * 64
    assert result["2"]["input_sha256"] == "c" * 64
    assert result["2"]["model_path_sha256"] == model_hash


def test_mlff_monitor_inherits_reference_tether_protocol(tmp_path):
    config = tmp_path / "tether.json"
    config.write_text(json.dumps({
        "migrant_tether_k_eV_A2": 4.0,
        "host_tether_k_eV_A2": 0.5,
        "jobs": [{"job_id": 3, "input_sha256": "d" * 64}],
    }))
    row = expected_jobs([config])["3"]
    assert row["migrant_tether_k_eV_A2"] == 4.0
    assert row["host_tether_k_eV_A2"] == 0.5


def test_clearance_graph_shortest_path_is_deterministic_and_continuous():
    assert np.allclose(fibonacci_directions(16, 48), fibonacci_directions(16, 48))
    layers = [
        {"positions": np.array([[0.0, 0.0, 0.0]]), "reference_offsets_A": np.array([0.0])},
        {
            "positions": np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            "reference_offsets_A": np.array([0.1, 0.0]),
        },
        {"positions": np.array([[2.0, 0.0, 0.0]]), "reference_offsets_A": np.array([0.0])},
    ]
    selected, objective = shortest_layered_path(layers, np.diag([10.0] * 3), 1.5, 1.0, 1.0)
    assert selected == [0, 0, 0]
    assert objective == pytest.approx(2.01)
