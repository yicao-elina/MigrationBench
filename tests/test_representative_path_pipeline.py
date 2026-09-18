import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.io import write as ase_write


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "migrationbench"
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_neb_path_topology import energy_topology  # noqa: E402
from parse_qe_relax_output import RY_BOHR_TO_EV_A, last_complete_force_step, parse_relax_out, write_extxyz  # noqa: E402
from monitor_qe_relax_jobs import classify as classify_qe_relax, collect_lineage, sync_lineage_parents  # noqa: E402
from prepare_qe_scf_warmup import set_namelist_value, walltime_seconds  # noqa: E402
from analyze_qe_scf_scaling_probe import analyze as analyze_qe_scf_scaling  # noqa: E402
from expand_qe_scf_warmup_batch import accepted_images, parse_image_list, path_job_slug  # noqa: E402
from monitor_qe_scf_warmups import classify as classify_scf_warmup  # noqa: E402
from monitor_qe_scf_warmups import magnetic_observables  # noqa: E402
from continue_qe_scf_warmup import selected_parent  # noqa: E402
from prepare_qe_neb_from_scf_warmups import select_warmup_jobs  # noqa: E402
from parse_qe_neb_output import parse_neb_out  # noqa: E402
from monitor_qe_neb_jobs import classify as classify_neb_job  # noqa: E402
from continue_qe_neb import derive_name as derive_neb_restart_name, select_parent as select_neb_restart_parent  # noqa: E402
from continue_qe_relax import parent_seed, prepare as prepare_qe_relax_continuation, select_parent as select_qe_relax_parent  # noqa: E402
from parse_qe_neb_path_history import BOHR_TO_A, HARTREE_TO_EV, parse_path_file  # noqa: E402
from parse_qe_neb_full_history import RY_PER_BOHR_TO_EV_PER_A, RY_TO_EV, parse_pw_blocks  # noqa: E402
from build_neb_history_dataset import composition_formula  # noqa: E402
from build_mlff_history_dataset import build_run as build_mlff_run  # noqa: E402
from build_mlff_history_dataset import runtime_identity as mlff_runtime_identity  # noqa: E402
from build_mlff_history_dataset import sha256 as mlff_sha256  # noqa: E402
from compare_endpoint_relaxation_speed import (  # noqa: E402
    comparison_decision,
    display_number,
    energy_metrics,
    endpoint_repeat_record,
    first_force_threshold_step,
    first_sustained_force_threshold,
    relaxation_data,
    lineage_runtime_provenance,
    resolve_acceptance_paths,
)
from update_qe_exact_geometry_transform_ab import hours_since_status  # noqa: E402
from update_site_proposer_transform_ab import validate_pairing as validate_site_transform_pairing  # noqa: E402
from build_relaxation_pair_dataset import validate_rows as validate_relaxation_pair_rows  # noqa: E402
from plan_site_transform_ab_repeats import filtered_status as filter_site_transform_status  # noqa: E402
from submit_qe_scf_sensitivity_batch import parse_receipt as parse_sensitivity_receipt  # noqa: E402
from plan_site_transform_ab_repeats import filtered_endpoint_status as filter_site_transform_endpoint_status  # noqa: E402
from plan_site_transform_ab_repeats import paired_direct_batch  # noqa: E402
from build_historical_mlff_candidate_manifest import candidate as historical_mlff_candidate  # noqa: E402
from build_site_stability_proposer import dft_label_status, local_descriptor, pairwise_order_accuracy  # noqa: E402
from audit_qe_calculator_identity import identity as qe_calculator_identity  # noqa: E402
from prepare_qe_protocol_sensitivity import apply_spin_model, replace_kpoints  # noqa: E402
from analyze_qe_protocol_sensitivity import RY_TO_EV as SENSITIVITY_RY_TO_EV, evaluate_gate  # noqa: E402
from plan_next_qe_protocol_stage import build_kpoint_config  # noqa: E402
from select_migration_endpoints import annotate_sites, build_pairs, read_sites  # noqa: E402
from generate_nonlinear_neb_candidates import clearance_arc, endpoint_pbc_deviation, historical_ase_images, historical_warp, linear_images, path_rms_distance, perpendicular_directions, validate_production_endpoint_acceptance  # noqa: E402
from launch_mlff_candidate_batch import eligible_candidates  # noqa: E402
from compare_mlff_preconditioner_runs import first_step_below, trapezoid_auc  # noqa: E402
from append_workflow_event import load_json_arg  # noqa: E402
from qe_neb_from_images import pre_output_gate  # noqa: E402
from prepare_qe_neb_handoff_pair import prepare as prepare_qe_handoff_pair  # noqa: E402
from validate_qe_neb_handoff_pair import validate as validate_qe_handoff_pair  # noqa: E402
from compare_qe_neb_initialization_ab import comparison_decision as qe_neb_ab_decision  # noqa: E402
from compare_qe_neb_initialization_ab import path_distance as qe_neb_path_distance  # noqa: E402
from compare_qe_neb_initialization_ab import lineage_runtime_provenance as qe_neb_lineage_runtime  # noqa: E402
from compare_mlff_trust_region import comparison_decision as trust_region_decision  # noqa: E402
from compare_mlff_trust_region import stopping_budget_comparison  # noqa: E402
from materialize_endpoint_remap_candidates import materialize as materialize_endpoint_remap, validate_segment_acceptance  # noqa: E402
from prepare_site_proposer_dft_validation import inject_reference_cell, read_csv as read_site_proposer_csv, reference_cell_block, select_matchable_pairs  # noqa: E402
from compare_site_proposer_dft_validation import compare as compare_site_proposer_dft, exact_paired_binomial_pvalue, paired_yield_metrics  # noqa: E402
from plan_endpoint_discovery_progress import plan as plan_endpoint_discovery  # noqa: E402
from migrationbench_dataset import manuscript_gate  # noqa: E402
from capture_runtime_provenance import build_record, named_path  # noqa: E402
from prepare_qe_repeat_relaxations import repeat_identifiers, repeat_input, structure_sha256  # noqa: E402
from build_endpoint_pair_acceptance import evaluate_record, repeat_metrics  # noqa: E402
from assign_endpoint_basins import assign as assign_endpoint_basins  # noqa: E402
from prepare_qe_endpoint_relaxations import replace_positions as replace_endpoint_positions  # noqa: E402
from prepare_qe_endpoint_relaxations import transform_pw as transform_endpoint_pw  # noqa: E402
from prepare_qe_endpoint_relaxations import uses_multiframe_coordinates  # noqa: E402
from prepare_qe_endpoint_relaxations import may_use_numbered_qe_coordinates  # noqa: E402
from validate_migrationbench_dataset import trust_region_semantics_valid  # noqa: E402


class RepresentativePathPipelineTest(unittest.TestCase):
    def test_qe_scf_scaling_recommendation_requires_clean_1p5x_throughput(self):
        baseline = {
            "job_id": "base",
            "elapsed": "03:00:00",
            "parsed": {"total_scf_iterations_seen": 3},
        }
        probe = {
            "job_id": "probe",
            "elapsed": "03:00:00",
            "ntasks": 4,
            "classification": "clean_timeout_needs_restart",
            "parsed": {"scf_iteration_records": 6, "scf_converged": False},
        }
        result = analyze_qe_scf_scaling(baseline, probe)
        self.assertEqual(result["walltime_iteration_rate_ratio"], 2.0)
        self.assertTrue(result["recommend_4r4p_for_future_clean_continuations"])
        self.assertFalse(result["label_eligibility"])
        probe["classification"] = "terminal_needs_review"
        self.assertFalse(
            analyze_qe_scf_scaling(baseline, probe)[
                "recommend_4r4p_for_future_clean_continuations"
            ]
        )

    def test_endpoint_discovery_planner_requires_repeat_then_acceptance(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            batch = root / "batch.json"
            batch.write_text(json.dumps({"jobs": [
                {"branch_id": "a", "source_path_id": "site::a", "source_image_index_qe": 1},
                {"branch_id": "b", "source_path_id": "site::b", "source_image_index_qe": 1},
            ]}))
            first = root / "first.json"
            first.write_text(json.dumps({"jobs": [
                {"branch_id": "a", "job_id": "1", "path_id": "site::a", "image_index_qe": 1, "classification": "accepted_local_minimum"},
                {"branch_id": "b", "job_id": "2", "path_id": "site::b", "image_index_qe": 1, "classification": "relax_done_force_gate_failed"},
            ]}))
            self.assertEqual(plan_endpoint_discovery(batch, first)["stage"], "prepare_repeat_relaxations")
            repeat = root / "repeat.json"
            repeat.write_text(json.dumps({"jobs": [
                {"path_id": "site::a", "image_index_qe": 1, "classification": "accepted_local_minimum"},
            ]}))
            self.assertEqual(
                plan_endpoint_discovery(batch, first, repeat)["stage"],
                "build_repeat_acceptance",
            )
            acceptance = root / "acceptance.json"
            acceptance.write_text(json.dumps({
                "first_status_sha256": mlff_sha256(first),
                "endpoint_records": [{"path_id": "site::a", "image_index_qe": 1, "status": "accepted"}],
            }))
            ready = plan_endpoint_discovery(batch, first, repeat, acceptance)
            self.assertEqual(ready["stage"], "ready_for_basin_assignment_and_final_comparison")
            self.assertEqual(ready["actions"], ["assign_endpoint_basins", "run_repeat_validated_prospective_comparator"])

    def test_site_proposer_comparator_withholds_censored_yield(self):
        pairs = [
            {"proposal": {"accepted": True}, "control": {"accepted": True}},
            {"proposal": {"accepted": True}, "control": {"accepted": False}},
            {"proposal": {"accepted": False}, "control": {"accepted": True}},
            {"proposal": {"accepted": True}, "control": {"accepted": False}},
            {"proposal": {"accepted": False}, "control": {"accepted": False}},
            {"proposal": {"accepted": True}, "control": {"accepted": False}},
        ]
        censored = paired_yield_metrics(pairs, finalized=False)
        self.assertIsNone(censored["paired_yield_difference"])
        final = paired_yield_metrics(pairs, finalized=True)
        self.assertAlmostEqual(final["proposal_yield"], 4 / 6)
        self.assertAlmostEqual(final["control_yield"], 2 / 6)
        self.assertAlmostEqual(final["paired_yield_difference"], 2 / 6)
        self.assertAlmostEqual(final["exact_paired_pvalue"], exact_paired_binomial_pvalue(3, 1))

    def test_site_proposer_comparator_binds_runtime_and_input_hashes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            jobs, statuses = [], []
            for arm in ("proposal", "control"):
                local_dir = root / arm
                local_dir.mkdir()
                relax_input = local_dir / "relax.in"
                relax_input.write_text(f"{arm}\n")
                input_hash = mlff_sha256(relax_input)
                (local_dir / "runtime_provenance.json").write_text(json.dumps({
                    "environment": {"SLURM_JOB_ID": f"job_{arm}"},
                    "inputs": {"relax_input": {"exists": True, "sha256": input_hash}},
                    "binaries": {"pw.x": {"exists": True, "sha256": "pw-hash"}},
                }))
                branch_id = f"branch_{arm}"
                jobs.append({
                    "branch_id": branch_id,
                    "arm": arm,
                    "pair_index": 1,
                    "raw_site_id": f"site_{arm}",
                    "source_path_id": f"site::{arm}",
                    "source_image_index_qe": 1,
                    "relax_input_sha256": input_hash,
                })
                statuses.append({
                    "branch_id": branch_id,
                    "job_id": f"job_{arm}",
                    "classification": "accepted_local_minimum",
                    "local_dir": str(local_dir),
                    "parsed": {
                        "lineage_total_ionic_steps": 4,
                        "lineage_total_scf_iterations": 20,
                        "final_max_atom_force_eV_A": 0.02,
                        "final_energy_eV": -10.0,
                    },
                })
            batch_path = root / "batch.json"
            batch_path.write_text(json.dumps({"jobs": jobs, "pairs": [{"pair_index": 1}]}))
            status_path = root / "status.json"
            status_path.write_text(json.dumps({"jobs": statuses}))
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps({
                "counts": {
                    "inputs": 2,
                    "parse_errors": 0,
                    "calculator_identities": 1,
                    "incomplete_identities": 0,
                    "soc_enabled_inputs": 0,
                    "spin_polarized_inputs": 2,
                },
                "profiles": [{"calculator_identity": "calc"}],
            }))
            acceptance_path = root / "acceptance.json"
            acceptance_path.write_text(json.dumps({
                "first_status_sha256": mlff_sha256(status_path),
                "endpoint_records": [
                    {
                        "path_id": f"site::{arm}",
                        "image_index_qe": 1,
                        "parent_job_id": f"job_{arm}",
                        "repeat_job_id": f"repeat_{arm}",
                        "calculator_identity": "calc",
                        "status": "accepted",
                        "failed_checks": [],
                        "checks": {
                            "parent_runtime_provenance": True,
                            "repeat_runtime_provenance": True,
                            "calculator_identity_unchanged": True,
                        },
                    }
                    for arm in ("proposal", "control")
                ]
            }))
            _, censored = compare_site_proposer_dft(SimpleNamespace(
                batch_manifest=batch_path,
                status=status_path,
                endpoint_acceptance=None,
                calculator_audit=audit_path,
                out_dir=root / "comparison_censored",
            ))
            self.assertFalse(censored["formal_result_available"])
            self.assertEqual(censored["pairs"][0]["proposal"]["result"], "repeat_required")
            _, result = compare_site_proposer_dft(SimpleNamespace(
                batch_manifest=batch_path,
                status=status_path,
                endpoint_acceptance=acceptance_path,
                calculator_audit=audit_path,
                out_dir=root / "comparison",
            ))
            self.assertTrue(result["formal_result_available"])
            self.assertEqual(result["metrics"]["both_accepted"], 1)
            self.assertEqual(result["metrics"]["proposal_yield"], 1.0)
            rejected_payload = json.loads(acceptance_path.read_text())
            rejected_payload["endpoint_records"][0]["status"] = "rejected"
            rejected_payload["endpoint_records"][0]["failed_checks"] = ["repeat_force_gate"]
            rejected_path = root / "acceptance_rejected.json"
            rejected_path.write_text(json.dumps(rejected_payload))
            _, rejected = compare_site_proposer_dft(SimpleNamespace(
                batch_manifest=batch_path,
                status=status_path,
                endpoint_acceptance=rejected_path,
                calculator_audit=audit_path,
                out_dir=root / "comparison_rejected",
            ))
            self.assertTrue(rejected["formal_result_available"])
            self.assertEqual(rejected["metrics"]["control_only"], 1)
            self.assertEqual(
                rejected["pairs"][0]["proposal"]["result"],
                "rejected_repeat_validation",
            )

    def test_site_proposer_prospective_batch_is_independent_and_cell_complete(self):
        project_root = Path(__file__).resolve().parents[1]
        proposer_root = project_root / "tests" / "fixtures" / "site_stability_proposer" / "v1_mace_screening"
        rows = read_site_proposer_csv(proposer_root / "site_descriptors_and_oof_predictions.csv")
        pairs = select_matchable_pairs(rows, 6, 2.0)
        selected = [row for pair in pairs for row in pair[:2]]
        self.assertEqual(len({row["raw_site_id"] for row in selected}), 12)
        self.assertEqual(len({row["void_family_group"] for row in selected}), 12)
        self.assertTrue(all(row["dft_relax_status"] == "missing" for row in selected))
        self.assertTrue(all(match["match_level"] == 0 for _, _, match in pairs))
        self.assertTrue(all(match["descriptor_distance_standardized_l2"] <= 2.0 for _, _, match in pairs))
        source = project_root / "tests" / "fixtures" / "site_search_qe" / "qe_inputs" / "unique_119" / "relax.in"
        reference = project_root / "tests" / "fixtures" / "site_search_qe" / "2L_octo_Cr1.in"
        combined = inject_reference_cell(source.read_text(), reference_cell_block(reference))
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "relax.in"
            output.write_text(combined)
            identity = qe_calculator_identity(output, "prospective")
            self.assertTrue(identity["identity_complete"])
            self.assertIsNotNone(identity["cell_fingerprint"])
        with self.assertRaisesRegex(ValueError, "already contains"):
            inject_reference_cell(combined, reference_cell_block(reference))

    def test_trust_region_mlff_dataset_uses_base_potential_as_physical_label(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            standard_frames = []
            dual_frames = []
            for iteration in range(2):
                for image_index in range(4):
                    atoms = Atoms(
                        "CrHe",
                        positions=[[1.0 + 0.1 * image_index, 1, 1], [5, 5, 5]],
                        cell=np.diag([10.0] * 3),
                        pbc=True,
                    )
                    atoms.info.update({
                        "optimizer_iteration": iteration,
                        "image_index": image_index,
                        "energy_eV": 100.0 + image_index,
                        "neb_residual_eV_A": 0.1,
                    })
                    atoms.arrays["forces"] = np.zeros((2, 3))
                    if image_index not in {0, 3}:
                        atoms.arrays["neb_forces"] = np.zeros((2, 3))
                    standard_frames.append(atoms)

                    dual = atoms.copy()
                    dual.info.update({
                        "neb_image": image_index,
                        "base_energy_eV": -10.0 + image_index,
                        "restrained_energy_eV": -9.75 + image_index,
                        "tether_energy_eV": 0.25,
                    })
                    dual.arrays["base_forces"] = np.full((2, 3), 0.01)
                    dual.arrays["restrained_forces"] = np.full((2, 3), 0.02)
                    dual_frames.append(dual)
            standard_path = root / "mlff_neb_iteration_history.extxyz"
            dual_path = root / "mlff_neb_dual_potential_history.extxyz"
            dual_csv = root / "mlff_neb_dual_potential_history.csv"
            ase_write(standard_path, standard_frames)
            ase_write(dual_path, dual_frames)
            dual_csv.write_text("fixture\n")
            history = {
                "n_images": 4,
                "n_optimizer_iterations_including_initial": 2,
                "spring_constant_eV_A2": 0.1,
                "climb": False,
                "neb_method": "improvedtangent",
                "runtime_provenance": None,
                "outputs": {"extxyz": str(standard_path), "extxyz_sha256": mlff_sha256(standard_path)},
            }
            history_path = root / "mlff_neb_iteration_history_manifest.json"
            history_path.write_text(json.dumps(history))
            main = {
                "seed": 42,
                "calculator": "mace-foundation",
                "model_path": "/model",
                "model_path_sha256": "a" * 64,
                "source_candidate_manifest_sha256": "b" * 64,
                "run_role": "diagnostic_preconditioner",
                "reference_tether": {"enabled": True},
                "optimizer_converged": False,
                "optimizer_converged_under_optimization_potential": True,
                "final_max_neb_residual_eV_A": 0.05,
                "outputs": {
                    "log": str(root / "log"),
                    "dual_potential_history_extxyz": str(dual_path),
                    "dual_potential_history_extxyz_sha256": mlff_sha256(dual_path),
                    "dual_potential_history_csv": str(dual_csv),
                    "dual_potential_history_csv_sha256": mlff_sha256(dual_csv),
                },
            }
            main_path = root / "mlff_neb_manifest.json"
            main_path.write_text(json.dumps(main))
            _, calculations, iterations, path, derived, _ = build_mlff_run(
                main_path, history_path, "fixture", "trust", "2026-01-01T00:00:00Z"
            )
            self.assertEqual(calculations[0]["energy_eV"], -10.0)
            self.assertEqual(calculations[0]["forces_eV_A"][0][0], 0.01)
            self.assertEqual(iterations[0]["optimization_energy_eV"], -9.75)
            self.assertEqual(iterations[0]["optimization_atomic_forces_eV_A"][0][0], 0.02)
            self.assertEqual(iterations[0]["regularization_energy_eV"], 0.25)
            self.assertEqual(path["barrier_eV"], 3.0)
            self.assertEqual(derived["protocol"], "trust_region_mlff_neb_base_rescore")
            self.assertFalse(derived["manuscript_allowed"])

    def test_trust_region_dataset_semantics_keep_base_and_optimizer_potentials_separate(self):
        config = {"nsites": 2}
        calc = {"method": "trust_region_mlff_neb_base_rescore"}
        row = {
            "energy_eV": -10.0,
            "optimization_energy_eV": -9.75,
            "regularization_energy_eV": 0.25,
            "optimization_atomic_forces_eV_A": [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
            "metadata_json": json.dumps({
                "physical_energy_force_source": "base_MACE_without_reference_tether",
                "neb_force_potential": "base_plus_reference_tether",
            }),
        }
        self.assertTrue(trust_region_semantics_valid(row, calc, config))
        self.assertFalse(trust_region_semantics_valid({**row, "regularization_energy_eV": 0.5}, calc, config))

    def test_historical_xyz_endpoint_relax_input_is_fixed_cell_and_long_step(self):
        template = """&CONTROL
  prefix = 'old'
/
&SYSTEM
  nat = 2
/
&ELECTRONS
/
&IONS
  ion_dynamics = 'bfgs'
/
&CELL
  cell_dynamics = 'bfgs'
/
ATOMIC_SPECIES
Cr 51.996 Cr.UPF
Te 127.60 Te.UPF
ATOMIC_POSITIONS angstrom
Cr 0.0 0.0 0.0
Te 2.0 0.0 0.0
K_POINTS gamma
CELL_PARAMETERS angstrom
5 0 0
0 5 0
0 0 5
"""
        with_positions = replace_endpoint_positions(
            template, ["Cr", "Te"], [(0.2, 0.3, 0.4), (2.2, 0.0, 0.0)]
        )
        result = transform_endpoint_pw(with_positions, "ep_test_s42", 84600, 200)
        self.assertIn("calculation = 'relax'", result)
        self.assertIn("max_seconds = 84600", result)
        self.assertIn("nstep = 200", result)
        self.assertIn("prefix = 'ep_test_s42'", result)
        self.assertNotRegex(result, r"(?im)^\s*&CELL\b")
        self.assertIn("Cr  0.200000000000  0.300000000000  0.400000000000", result)

    def test_historical_multiframe_layout_never_treats_pw1_as_image1_coordinates(self):
        self.assertTrue(uses_multiframe_coordinates({"image_format": "multi_frame_xyz_with_qe_cell"}))
        self.assertTrue(uses_multiframe_coordinates({"image_format": "multi_frame_xyz"}))
        self.assertFalse(uses_multiframe_coordinates({"image_format": "qe_numbered_inputs"}))
        with tempfile.TemporaryDirectory() as raw:
            pw1 = Path(raw) / "pw_1.in"
            pw1.write_text("template only\n")
            self.assertFalse(
                may_use_numbered_qe_coordinates(
                    {"image_format": "multi_frame_xyz_with_qe_cell"}, pw1
                )
            )
            self.assertTrue(
                may_use_numbered_qe_coordinates({"image_format": "qe_numbered_inputs"}, pw1)
            )

    def test_runtime_provenance_hashes_inputs_and_records_missing_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.dat"
            source.write_text("traceable\n")
            record = build_record([("input", source)], ["definitely_missing_binary_xyz"], ["pytest"])
            self.assertTrue(record["inputs"]["input"]["exists"])
            self.assertEqual(record["inputs"]["input"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertFalse(record["binaries"]["definitely_missing_binary_xyz"]["exists"])
            self.assertIsNotNone(record["packages"]["pytest"]["distribution_version"])
            self.assertIsNotNone(record["packages"]["pytest"]["module_file_sha256"])

    def test_runtime_provenance_named_path_requires_name(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            named_path("missing_name_separator")

    def test_mlff_dataset_runtime_identity_is_hash_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime_provenance.json"
            runtime.write_text(json.dumps({
                "platform": "fixture",
                "python": {"version": "3.9"},
                "packages": {"ase": {"module_version": "3.25.0"}},
                "environment": {"LOADEDMODULES": "qe/7.3.1-cpu"},
                "binaries": {"neb.x": {"sha256": "a" * 64}},
            }))
            digest = hashlib.sha256(runtime.read_bytes()).hexdigest()
            history_path = root / "history.json"
            record = mlff_runtime_identity(history_path, {
                "runtime_provenance": {"uri": str(runtime), "sha256": digest}
            })
            self.assertEqual(record["sha256"], digest)
            self.assertTrue(record["environment_id"].startswith("ENV_"))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                mlff_runtime_identity(history_path, {
                    "runtime_provenance": {"uri": str(runtime), "sha256": "b" * 64}
                })

    def test_production_wrappers_capture_runtime_provenance(self):
        wrappers = (
            "submit_slurm_qe_neb.sh",
            "submit_slurm_qe_neb_restart_verified.sh",
            "submit_slurm_qe_neb_warmstarted.sh",
            "submit_slurm_qe_relax.sh",
            "submit_slurm_mlff_neb.sh",
            "submit_slurm_mlff_neb_cpu.sh",
            "submit_slurm_qe_scf_warmup.sh",
            "submit_slurm_qe_scf_warmup_restart.sh",
        )
        for name in wrappers:
            text = (SCRIPT_DIR / name).read_text()
            self.assertIn("capture_runtime_provenance.py", text, name)
            self.assertIn("runtime_provenance.json", text, name)

    def test_launchers_export_fixed_remote_code_root(self):
        launchers = (
            "continue_qe_neb.py",
            "continue_qe_relax.py",
            "launch_qe_neb_from_scf_warmups.py",
            "submit_qe_endpoint_relax_batch.py",
            "launch_mlff_candidate_batch.py",
            "continue_qe_scf_warmup.py",
            "expand_qe_scf_warmup_batch.py",
            "submit_qe_scf_sensitivity_batch.py",
        )
        for name in launchers:
            text = (SCRIPT_DIR / name).read_text()
            self.assertIn("MIGRATIONBENCH_CODE_ROOT", text, name)
            self.assertIn("remote_code_root", text, name)

    def test_sensitivity_submission_receipt_requires_numeric_job_id(self):
        self.assertEqual(parse_sensitivity_receipt("30865004|submitted_new_job"), ["30865004", "submitted_new_job"])
        with self.assertRaises(RuntimeError):
            parse_sensitivity_receipt("|submitted_new_job")

    def test_legacy_unmanifested_launchers_fail_closed(self):
        launchers = (
            SCRIPT_DIR / "submit_qe_neb_job.sh",
            SCRIPT_DIR / "submit_mlff_neb_cpu_job.sh",
            SCRIPT_DIR / "submit_mlff_neb_job.sh",
            SCRIPT_DIR / "restart_chain" / "submit_qe_neb_restart_job.sh",
            SCRIPT_DIR / "restart_chain" / "submit_1_7_long_qe_restart_chain.sh",
            SCRIPT_DIR / "restart_chain" / "continue_1_7_qe_restart_chain_template.sh",
        )
        for path in launchers:
            self.assertIn("exit 64", path.read_text().splitlines()[:12], str(path))

    def test_endpoint_basin_assignment_uses_complete_linkage(self):
        cell = [(10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)]
        records = []
        for index, x_value in enumerate((0.0, 0.2, 0.4)):
            records.append({
                "path_id": "fixture", "image_index_qe": index + 1,
                "parent_job_id": str(index), "repeat_job_id": str(index + 10),
                "accepted_structure": "fixture_{}".format(index),
                "accepted_structure_sha256": "{:064x}".format(index + 1),
                "calculator_identity": "calc", "symbols": ["Cr", "Te"],
                "positions": [(x_value, 0.0, 0.0), (5.0, 0.0, 0.0)],
                "cell": cell, "migrant_index": 0, "system_key": "Cr1Te1_cell",
            })
        assignments, comparisons = assign_endpoint_basins(records, {
            "maximum_migrant_distance_A": 0.25,
            "maximum_host_rmsd_A": 0.15,
            "maximum_local_signature_rms_A": 0.5,
            "local_signature_neighbors": 1,
        })
        self.assertEqual(len(comparisons), 3)
        self.assertEqual(len({row["basin_id"] for row in assignments}), 2)

    def test_qe_handoff_uses_forward_endpoint_acceptance_binding(self):
        import hashlib
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            initial, final, images = root / "initial.extxyz", root / "final.extxyz", root / "images.extxyz"
            initial.write_text("initial")
            final.write_text("final")
            images.write_text("images")
            file_hash = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            acceptance = root / "endpoint_pair_acceptance.json"
            acceptance.write_text(json.dumps({
                "status": "accepted",
                "initial_structure_sha256": file_hash(initial),
                "final_structure_sha256": file_hash(final),
            }))
            candidate = root / "nonlinear_candidate_manifest.json"
            candidate.write_text(json.dumps({
                "endpoint_status": "accepted_local_minima",
                "endpoint_acceptance_sha256": file_hash(acceptance),
                "initial_endpoint_source_sha256": file_hash(initial),
                "final_endpoint_source_sha256": file_hash(final),
                "candidates": [{
                    "branch_id": "linear_mic",
                    "images": images.name,
                    "images_sha256": file_hash(images),
                    "geometry_gate": {"status": "pass"},
                    "duplicate_of": None,
                    "production_eligible_before_mace": True,
                }],
            }))
            args = SimpleNamespace(
                run_role="production",
                initialization_role="direct_baseline",
                source_candidate_manifest=str(candidate),
                direct_baseline_images=str(images),
                endpoint_acceptance=str(acceptance),
                source_mlff_manifest=None,
                max_mlff_residual=0.2,
            )
            gate = pre_output_gate(args, images)
            self.assertEqual(gate["status"], "pass")
            self.assertTrue(gate["checks"]["candidate_endpoint_acceptance_binding"])

    def test_trust_region_qe_handoff_requires_bound_comparator_acceptance(self):
        import hashlib
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            initial = root / "initial.extxyz"
            final = root / "final.extxyz"
            direct = root / "direct.extxyz"
            transformed = root / "trust.extxyz"
            dual_extxyz = root / "mlff_neb_dual_potential_history.extxyz"
            dual_csv = root / "mlff_neb_dual_potential_history.csv"
            for path, content in (
                (initial, "initial"), (final, "final"), (direct, "direct"),
                (transformed, "trust"), (dual_extxyz, "dual-extxyz"),
                (dual_csv, "dual-csv"),
            ):
                path.write_text(content)
            file_hash = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            endpoint = root / "endpoint_pair_acceptance.json"
            endpoint.write_text(json.dumps({
                "status": "accepted",
                "initial_structure_sha256": file_hash(initial),
                "final_structure_sha256": file_hash(final),
            }))
            candidate = root / "nonlinear_candidate_manifest.json"
            candidate.write_text(json.dumps({
                "endpoint_status": "accepted_local_minima",
                "endpoint_acceptance_sha256": file_hash(endpoint),
                "initial_endpoint_source_sha256": file_hash(initial),
                "final_endpoint_source_sha256": file_hash(final),
                "candidates": [{
                    "branch_id": "historical_warp",
                    "images": direct.name,
                    "images_sha256": file_hash(direct),
                    "geometry_gate": {"status": "pass"},
                    "duplicate_of": None,
                    "production_eligible_before_mace": True,
                }],
            }))
            mlff = root / "mlff_neb_manifest.json"
            mlff.write_text(json.dumps({
                "source_candidate_manifest_sha256": file_hash(candidate),
                "input_images_sha256": file_hash(direct),
                "optimizer_converged": False,
                "optimizer_converged_under_optimization_potential": True,
                "final_max_neb_residual_eV_A": 0.1,
                "reference_tether": {"enabled": True},
                "outputs": {
                    "images_sha256": file_hash(transformed),
                    "dual_potential_history_extxyz": str(dual_extxyz),
                    "dual_potential_history_extxyz_sha256": file_hash(dual_extxyz),
                    "dual_potential_history_csv": str(dual_csv),
                    "dual_potential_history_csv_sha256": file_hash(dual_csv),
                },
            }))
            acceptance = root / "trust_region_comparison.json"
            acceptance.write_text(json.dumps({
                "decision": "trust_region_candidate_ready_for_dft_ab_design",
                "physical_identity_match": True,
                "qe_handoff_allowed": False,
                "provenance": {"trust_manifest_sha256": file_hash(mlff)},
            }))
            args = SimpleNamespace(
                run_role="production",
                initialization_role="trust_region_mlff_preconditioned",
                source_candidate_manifest=str(candidate),
                direct_baseline_images=str(direct),
                endpoint_acceptance=str(endpoint),
                source_mlff_manifest=str(mlff),
                mlff_preconditioner_acceptance=str(acceptance),
                max_mlff_residual=0.2,
            )
            gate = pre_output_gate(args, transformed)
            self.assertEqual(gate["status"], "pass")
            self.assertTrue(gate["checks"]["dual_history_extxyz"])
            self.assertTrue(gate["checks"]["trust_region_manifest_binding"])
            rejected = json.loads(acceptance.read_text())
            rejected["decision"] = "trust_region_candidate_rejected"
            acceptance.write_text(json.dumps(rejected))
            with self.assertRaisesRegex(ValueError, "trust_region_acceptance_decision"):
                pre_output_gate(args, transformed)

    def test_qe_handoff_schema_1_1_binds_inputs_and_handoff_manifests(self):
        root = Path(__file__).resolve().parents[1]
        candidate_dir = (
            root / "tests" / "fixtures" / "nonlinear_candidate_smoke"
            / "1-6_iter108_img2_to_img5_v2_s42"
        )
        with tempfile.TemporaryDirectory() as raw:
            raw_path = Path(raw)
            direct_images = candidate_dir / "historical_curvature_warp.extxyz"
            candidate_manifest = candidate_dir / "nonlinear_candidate_manifest.json"
            engine_template = raw_path / "qe_engine_template.in"
            engine_template.write_text("""&CONTROL
  calculation = 'scf'
  prefix = 'handoff_fixture'
  pseudo_dir = '/tmp'
  outdir = '/tmp'
/
&SYSTEM
  ibrav = 0
  nat = 61
  ntyp = 3
  ecutwfc = 50.0
  ecutrho = 200.0
  occupations = 'smearing'
  smearing = 'cold'
  degauss = 0.0146997236
  vdw_corr = 'DFT-D3'
/
&ELECTRONS
  conv_thr = 1.0d-6
/
ATOMIC_SPECIES
Cr 51.996 Cr_PBE_FR.SG15v1.2.UPF
Sb 121.760 Sb_PBE_FR.SG15v1.2.UPF
Te 127.600 Te_PBE_FR.SG15v1.2.UPF
K_POINTS gamma
""")
            mlff_manifest = raw_path / "mlff_neb_manifest.json"
            mlff_manifest.write_text(json.dumps({
                "input_images_sha256": mlff_sha256(direct_images),
                "source_candidate_manifest_sha256": mlff_sha256(candidate_manifest),
                "seed": 44,
                "calculator": "mace-foundation-fixture",
                "model_path_sha256": "a" * 64,
                "outputs": {"images_sha256": mlff_sha256(direct_images)},
            }))
            args = SimpleNamespace(
                direct_images=direct_images,
                mlff_images=direct_images,
                engine_template=engine_template,
                source_candidate_manifest=candidate_manifest,
                source_mlff_manifest=mlff_manifest,
                mlff_initialization_role="mlff_preconditioned",
                mlff_preconditioner_acceptance=None,
                out_dir=raw_path / "pair",
                path_id="1-6_segment_2_to_5",
                path_family="unverified_smoke",
                seed=44,
                run_role="diagnostic",
                endpoint_acceptance=None,
                calculator_acceptance=None,
                source_neb_output=None,
                max_mlff_residual=0.2,
                nstep_path=200,
                opt_scheme="broyden",
                ci_scheme="auto",
                ds=0.2,
                k_min=0.1,
                k_max=0.3,
                path_thr=0.03,
            )
            pair_path, pair = prepare_qe_handoff_pair(args)
            result = validate_qe_handoff_pair(pair_path)
            self.assertEqual(pair["schema_version"], "1.1")
            self.assertEqual(result["status"], "pass")
            self.assertTrue(all(row["checks"]["handoff_manifest_sha256"] for row in result["rows"]))

    def test_qe_neb_ab_comparison_rejects_prefix_and_mechanism_switch(self):
        converged = {
            "converged": True,
            "has_overflow": False,
            "activation_forward_eV": 0.8,
        }
        prefix = {**converged, "converged": False}
        same = {
            "cell_max_abs_delta_A": 0.0,
            "all_atom_rms_A": 0.05,
            "migrant_curve_rms_A": 0.10,
        }
        decision, eligible = qe_neb_ab_decision(
            prefix, converged, same, True, True, 0.15, 0.25, 0.05, True, True
        )
        self.assertEqual(decision, "pending_or_censored_not_both_converged")
        self.assertFalse(eligible)
        changed = {**same, "migrant_curve_rms_A": 0.4}
        decision, eligible = qe_neb_ab_decision(
            converged, converged, changed, True, True, 0.15, 0.25, 0.05, True, True
        )
        self.assertEqual(decision, "different_final_mechanism_no_speedup")
        self.assertFalse(eligible)
        decision, eligible = qe_neb_ab_decision(
            converged, converged, same, True, True, 0.15, 0.25, 0.05, True, True
        )
        self.assertEqual(decision, "comparable_same_mechanism_speedup_measurable")
        self.assertTrue(eligible)
        decision, eligible = qe_neb_ab_decision(
            converged, converged, same, True, True, 0.15, 0.25, 0.05, False, True
        )
        self.assertEqual(decision, "invalid_pair_manifest_input_binding")
        self.assertFalse(eligible)
        decision, eligible = qe_neb_ab_decision(
            converged, converged, same, True, True, 0.15, 0.25, 0.05, True, False
        )
        self.assertEqual(decision, "invalid_incomplete_qe_runtime_provenance")
        self.assertFalse(eligible)

    def test_qe_neb_runtime_gate_requires_neb_and_pw_for_every_segment(self):
        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw) / "mb_qe_fixture_s42_123"
            run_dir.mkdir()
            neb_input = run_dir / "neb.in"
            neb_input.write_text("fixture\n")
            runtime = {
                "environment": {"SLURM_JOB_ID": "123"},
                "inputs": {"neb_input": {"exists": True, "sha256": mlff_sha256(neb_input)}},
                "binaries": {"neb.x": {"exists": True, "sha256": "neb-hash"}},
            }
            (run_dir / "runtime_provenance.json").write_text(json.dumps(runtime))
            self.assertFalse(qe_neb_lineage_runtime([run_dir])["passed"])
            runtime["binaries"]["pw.x"] = {"exists": True, "sha256": "pw-hash"}
            (run_dir / "runtime_provenance.json").write_text(json.dumps(runtime))
            self.assertTrue(qe_neb_lineage_runtime([run_dir])["passed"])

    def test_qe_neb_final_path_distance_uses_periodic_minimum_image(self):
        left = {
            "symbols": ["Cr", "Te"],
            "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            "images": [
                [[9.9, 1.0, 1.0], [5.0, 5.0, 5.0]],
                [[0.1, 1.0, 1.0], [5.0, 5.0, 5.0]],
            ],
        }
        right = {
            **left,
            "images": [
                [[-0.1, 1.0, 1.0], [5.0, 5.0, 5.0]],
                [[10.1, 1.0, 1.0], [5.0, 5.0, 5.0]],
            ],
        }
        distances = qe_neb_path_distance(left, right)
        self.assertAlmostEqual(distances["all_atom_rms_A"], 0.0, places=8)
        self.assertAlmostEqual(distances["migrant_curve_max_A"], 0.0, places=8)

    def test_historical_trust_region_path_is_curvature_donor_not_qe_handoff(self):
        trust_row = {
            "classification": "ready_for_dft_preconditioner_review",
            "checks": {"dual_potential_history": True},
        }
        trust_manifest = {"reference_tether": {"enabled": True}}
        historical_candidate = {
            "manifest_hash_match": True,
            "unique_input_row_match": True,
            "endpoints_accepted": False,
            "candidate_production_eligible": False,
        }
        self.assertEqual(
            trust_region_decision(True, trust_row, trust_manifest, historical_candidate),
            "trust_region_diagnostic_ready_as_curvature_donor_requires_endpoint_remap",
        )
        production_candidate = {
            **historical_candidate,
            "endpoints_accepted": True,
            "candidate_production_eligible": True,
        }
        self.assertEqual(
            trust_region_decision(True, trust_row, trust_manifest, production_candidate),
            "trust_region_candidate_ready_for_dft_ab_design",
        )

    def test_trust_region_different_step_caps_are_comparable_only_after_natural_stops(self):
        baseline = {
            "steps_requested": 500,
            "optimizer_steps_completed": 33,
            "optimizer_converged": True,
        }
        trust = {
            "steps_requested": 150,
            "optimizer_steps_completed": 31,
            "optimizer_converged": False,
            "optimizer_converged_under_optimization_potential": True,
        }
        result = stopping_budget_comparison(baseline, trust)
        self.assertTrue(result["comparable_before_common_budget"])
        trust["optimizer_steps_completed"] = 150
        trust["optimizer_converged_under_optimization_potential"] = False
        result = stopping_budget_comparison(baseline, trust)
        self.assertFalse(result["comparable_before_common_budget"])
        self.assertEqual(
            trust_region_decision(True, {}, {}, {}, False),
            "invalid_stopping_budget_mismatch_or_censoring",
        )

    def test_production_nonlinear_candidate_requires_exact_endpoint_hashes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            initial, final = root / "initial.extxyz", root / "final.extxyz"
            initial.write_text("initial")
            final.write_text("final")
            import hashlib
            acceptance = root / "endpoint_pair_acceptance.json"
            acceptance.write_text(json.dumps({
                "status": "accepted",
                "initial_structure_sha256": hashlib.sha256(initial.read_bytes()).hexdigest(),
                "final_structure_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
            }))
            result = validate_production_endpoint_acceptance(
                "accepted_local_minima", acceptance, initial, final
            )
            self.assertEqual(result["status"], "accepted")
            final.write_text("changed")
            with self.assertRaisesRegex(ValueError, "final_structure_sha256"):
                validate_production_endpoint_acceptance(
                    "accepted_local_minima", acceptance, initial, final
                )
    def test_endpoint_repeat_acceptance_end_to_end_fixture(self):
        qe_input = """&CONTROL
  calculation = 'relax'
  restart_mode = 'from_scratch'
  prefix = 'endpoint'
  tprnfor = .true.
  max_seconds = 100
/
&SYSTEM
  nat = 2
  ntyp = 2
  ecutwfc = 50
  ecutrho = 200
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.005
/
&ELECTRONS
  conv_thr = 1.0d-6
/
ATOMIC_SPECIES
Cr 1 Cr.UPF
Te 1 Te.UPF
ATOMIC_POSITIONS angstrom
Cr 0 0 0
Te 2.1 0 0
K_POINTS gamma
CELL_PARAMETERS angstrom
10 0 0
0 10 0
0 0 10
"""
        output_template = """
! total energy = {energy} Ry
atom 1 type 1 force = 0.0001 0.0 0.0
atom 2 type 2 force = -0.0001 0.0 0.0
Total force = 0.0002
ATOMIC_POSITIONS (angstrom)
Cr {crx} 0 0
Te 2.1 0 0
bfgs converged in 1 scf cycles and 1 bfgs steps
End of BFGS Geometry Optimization
JOB DONE.
"""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            parent_dir, repeat_dir, accepted_dir = root / "parent", root / "repeat", root / "accepted"
            for directory in (parent_dir, repeat_dir, accepted_dir):
                directory.mkdir()
            (parent_dir / "relax.in").write_text(qe_input)
            parent_output = output_template.format(energy="-10.0000", crx="0.000")
            (parent_dir / "relax.out").write_text(parent_output)
            parent_step = last_complete_force_step(parse_relax_out(parent_dir / "relax.out"))
            parent_step["cell_A"] = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
            repeat_text = repeat_input(qe_input, "repeat", 84600, parent_step)
            (repeat_dir / "relax.in").write_text(repeat_text)
            (repeat_dir / "relax.out").write_text(
                output_template.format(energy="-10.0001", crx="0.000")
            )
            import hashlib
            repeat_input_hash = hashlib.sha256((repeat_dir / "relax.in").read_bytes()).hexdigest()
            parent_input_hash = hashlib.sha256((parent_dir / "relax.in").read_bytes()).hexdigest()
            for directory, job_id, input_hash in (
                (parent_dir, "100", parent_input_hash),
                (repeat_dir, "101", repeat_input_hash),
            ):
                (directory / "runtime_provenance.json").write_text(json.dumps({
                    "environment": {"SLURM_JOB_ID": job_id},
                    "inputs": {"relax_input": {"exists": True, "sha256": input_hash}},
                    "binaries": {"pw.x": {"exists": True, "sha256": "fixture-pw"}},
                }))
            (repeat_dir / "endpoint_relax_manifest.json").write_text(json.dumps({
                "parent_job_id": "100",
                "relax_input_sha256": repeat_input_hash,
            }))
            parent_job = {
                "job_id": "100", "path_id": "fixture", "image_index_qe": 1,
                "local_dir": str(parent_dir), "classification": "accepted_local_minimum",
            }
            repeat_job = {
                "job_id": "101", "path_id": "fixture", "image_index_qe": 1,
                "local_dir": str(repeat_dir), "classification": "accepted_local_minimum",
            }
            record = evaluate_record(parent_job, repeat_job, accepted_dir)
            self.assertEqual(record["status"], "accepted")
            self.assertEqual(record["failed_checks"], [])
            self.assertTrue(Path(record["accepted_structure"]).is_file())
            runtime = json.loads((parent_dir / "runtime_provenance.json").read_text())
            runtime["inputs"]["relax_input"]["sha256"] = "0" * 64
            (parent_dir / "runtime_provenance.json").write_text(json.dumps(runtime))
            rejected = evaluate_record(parent_job, repeat_job, accepted_dir)
            self.assertEqual(rejected["status"], "rejected")
            self.assertIn("parent_runtime_provenance", rejected["failed_checks"])

    def test_repeat_relax_input_is_idempotent_and_replaces_final_geometry(self):
        text = """&CONTROL
  calculation = 'relax'
  restart_mode = 'from_scratch'
  prefix = 'old'
  tprnfor = .true.
  max_seconds = 100
/
&SYSTEM
  nat = 2
  ntyp = 2
  ecutwfc = 50
  ecutrho = 200
/
&ELECTRONS
  conv_thr = 1.0d-6
/
ATOMIC_SPECIES
Cr 1 Cr.UPF
Te 1 Te.UPF
ATOMIC_POSITIONS angstrom
Cr 0 0 0
Te 2 0 0
K_POINTS gamma
CELL_PARAMETERS angstrom
10 0 0
0 10 0
0 0 10
"""
        step = {
            "symbols": ["Cr", "Te"],
            "positions_A": [[0.1, 0.2, 0.3], [2.1, 0.0, 0.0]],
            "cell_A": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        }
        transformed = repeat_input(text, "repeat", 84600, step)
        self.assertEqual(transformed.lower().count("calculation ="), 1)
        self.assertEqual(transformed.lower().count("max_seconds ="), 1)
        self.assertIn("Cr  0.100000000000  0.200000000000  0.300000000000", transformed)

    def test_repeat_relax_branch_tags_prevent_ab_job_collisions(self):
        direct = repeat_identifiers("site::unique_119", None, 1, 42, "direct")
        transformed = repeat_identifiers("site::unique_119", None, 1, 42, "geomvoid")
        self.assertNotEqual(direct["branch_id"], transformed["branch_id"])
        self.assertNotEqual(direct["job_name"], transformed["job_name"])
        self.assertNotEqual(direct["prefix"], transformed["prefix"])
        self.assertEqual(transformed["branch_tag"], "geomvoid")

    def test_repeat_relax_metrics_and_structure_hash_are_deterministic(self):
        parent = {
            "symbols": ["Cr", "Te"],
            "positions_A": [[9.98, 0, 0], [4, 0, 0]],
            "cell_A": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
            "energy_eV": -10.0,
            "max_atom_force_eV_A": 0.01,
        }
        repeat = {
            **parent,
            "positions_A": [[0.02, 0, 0], [4.01, 0, 0]],
            "energy_eV": -10.01,
        }
        metrics = repeat_metrics(parent, repeat)
        self.assertAlmostEqual(metrics["maximum_repeat_relax_displacement_A"], 0.04)
        self.assertAlmostEqual(metrics["repeat_relax_energy_drop_eV"], 0.01)
        self.assertEqual(structure_sha256(parent), structure_sha256(parent))

    def test_live_qe_parser_uses_last_complete_force_block(self):
        text = """
! total energy = -10.0 Ry
atom 1 type 1 force = 0.001 0.0 0.0
Total force = 0.001
ATOMIC_POSITIONS (angstrom)
Cr 0 0 0

! total energy = -10.1 Ry
ATOMIC_POSITIONS (angstrom)
Cr 0.1 0 0
"""
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "relax.out"
            output.write_text(text)
            parsed = parse_relax_out(output)
            self.assertTrue(parsed["trailing_incomplete_force_step"])
            self.assertEqual(parsed["final_force_ionic_step"], 0)
            self.assertAlmostEqual(
                parsed["final_max_atom_force_eV_A"], 0.001 * RY_BOHR_TO_EV_A
            )
            self.assertEqual(last_complete_force_step(parsed)["ionic_step"], 0)

    def test_qe_relax_clean_max_seconds_is_restartable(self):
        parsed = {
            "job_done": False,
            "bfgs_converged": False,
            "max_seconds_reached": True,
            "n_ionic_steps": 3,
            "final_force_ionic_step": 2,
            "scf_not_converged_count": 0,
        }
        self.assertEqual(
            classify_qe_relax(parsed, "not_in_squeue", "COMPLETED"),
            "clean_max_seconds_restartable",
        )
        parsed["scf_not_converged_count"] = 1
        self.assertEqual(
            classify_qe_relax(parsed, "not_in_squeue", "COMPLETED"),
            "partial_relax_needs_review_or_restart",
        )

    def test_qe_relax_max_ionic_steps_is_not_bfgs_convergence(self):
        text = """! total energy = -10.0 Ry
atom 1 type 1 force = 0.002 0.0 0.0
Total force = 0.002
ATOMIC_POSITIONS (angstrom)
Cr 0 0 0
The maximum number of steps has been reached.
End of BFGS Geometry Optimization
JOB DONE.
"""
        with tempfile.TemporaryDirectory() as raw:
            relax_input = Path(raw) / "relax.in"
            relax_input.write_text("""ATOMIC_POSITIONS angstrom
Cr -0.1 0 0
CELL_PARAMETERS angstrom
10 0 0
0 10 0
0 0 10
""")
            output = Path(raw) / "relax.out"
            output.write_text(text)
            parsed = parse_relax_out(output, relax_input)
            self.assertFalse(parsed["bfgs_converged"])
            self.assertTrue(parsed["geometry_optimization_ended"])
            self.assertTrue(parsed["max_ionic_steps_reached"])
            self.assertEqual(parsed["steps"][0]["positions_A"][0][0], -0.1)
            self.assertEqual(parsed["latest_geometry"]["positions_A"][0][0], 0.0)
            self.assertFalse(parsed["latest_geometry_has_evaluated_forces"])
            self.assertEqual(
                classify_qe_relax(parsed, "not_in_squeue", "COMPLETED"),
                "clean_max_ionic_steps_restartable",
            )

    def test_qe_relax_continuation_uses_last_complete_geometry(self):
        qe_input = """&CONTROL
  calculation = 'relax'
  restart_mode = 'from_scratch'
  prefix = 'parent'
  max_seconds = 100
/
&SYSTEM
  nat = 2
  ntyp = 2
  ecutwfc = 50
  ecutrho = 200
  occupations = 'smearing'
  smearing = 'gaussian'
  degauss = 0.005
/
&ELECTRONS
  conv_thr = 1.0d-6
/
ATOMIC_SPECIES
Cr 52 Cr.UPF
Te 128 Te.UPF
ATOMIC_POSITIONS angstrom
Cr 0 0 0
Te 2 0 0
K_POINTS gamma
CELL_PARAMETERS angstrom
10 0 0
0 10 0
0 0 10
"""
        qe_output = """iteration # 1
! total energy = -10.0 Ry
atom 1 type 1 force = 0.001 0.0 0.0
atom 2 type 2 force = -0.001 0.0 0.0
Total force = 0.002
ATOMIC_POSITIONS (angstrom)
Cr 0.25 0 0
Te 2.1 0 0
Maximum CPU time exceeded
"""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            parent_dir, out_dir = root / "parent", root / "continuation"
            parent_dir.mkdir()
            (parent_dir / "relax.in").write_text(qe_input)
            (parent_dir / "relax.out").write_text(qe_output)
            (parent_dir / "endpoint_relax_manifest.json").write_text("{}")
            status = root / "status.json"
            status.write_text('{"jobs": []}')
            parent = {
                "job_id": "100",
                "job_name": "mb_fixture_s47",
                "branch_id": "fixture_s47",
                "path_id": "fixture",
                "image_index_qe": 2,
                "classification": "clean_max_seconds_restartable",
                "remote_run_dir": "/scratch/fixture",
                "local_dir": str(parent_dir),
                "seed": 47,
            }
            args = SimpleNamespace(
                out_dir=out_dir,
                walltime="24:00:00",
                max_seconds=84600,
                nstep=200,
                ntasks=2,
                memory="160G",
                attempt=2,
                seed=47,
                relax_status=status,
                mode="continuation",
                disable_symmetry=False,
            )
            manifest, _ = prepare_qe_relax_continuation(args, parent)
            text = (out_dir / "relax.in").read_text()
            self.assertIn("Cr  0.250000000000", text)
            self.assertIn("max_seconds = 84600", text)
            self.assertIn("nstep = 200", text)
            self.assertEqual(manifest["resources"]["nstep"], 200)
            self.assertEqual(manifest["lineage_metrics_before_segment"]["ionic_steps"], 1)
            self.assertEqual(manifest["lineage_metrics_before_segment"]["scf_iterations"], 1)
            self.assertEqual(manifest["continuation_policy"]["geometry_source"], "parent_latest_printed_geometry")
            self.assertFalse(manifest["parent"]["latest_geometry_has_evaluated_forces"])
            self.assertFalse(manifest["continuation_policy"]["symmetry_release"])
            self.assertNotEqual(
                manifest["parent"]["latest_printed_geometry_sha256"],
                manifest["parent"]["last_complete_force_geometry_sha256"],
            )

    def test_qe_relax_continuation_seed_is_bound_to_parent_name(self):
        parent = {"job_name": "mb_endpoint_s42", "branch_id": "endpoint_s42"}
        self.assertEqual(parent_seed(parent), 42)
        with self.assertRaisesRegex(RuntimeError, "cannot resolve one seed"):
            parent_seed(parent, 47)

    def test_qe_relax_force_refinement_parent_gate_is_separate(self):
        status = {"jobs": [{"job_id": "1", "classification": "relax_done_force_gate_failed"}]}
        self.assertEqual(select_qe_relax_parent(status, "1", "force-refinement")["job_id"], "1")
        with self.assertRaisesRegex(RuntimeError, "not cleanly restartable"):
            select_qe_relax_parent(status, "1", "continuation")

    def test_qe_relax_lineage_accumulates_segments_and_curves(self):
        segment_output = """iteration # 1
! total energy = {energy} Ry
atom 1 type 1 force = 0.001 0.0 0.0
Total force = 0.001
ATOMIC_POSITIONS (angstrom)
Cr {x} 0 0
"""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            parent, child = root / "parent_100", root / "child_101"
            parent.mkdir(); child.mkdir()
            for directory, energy, x in ((parent, -10.0, 0.0), (child, -10.1, 0.1)):
                (directory / "relax.in").write_text("input")
                (directory / "relax.out").write_text(segment_output.format(energy=energy, x=x))
            (parent / "endpoint_relax_manifest.json").write_text("{}")
            (child / "endpoint_relax_manifest.json").write_text(json.dumps({
                "parent": {"remote_run_dir": "/scratch/parent_100"}
            }))
            segments = collect_lineage(child, root)
            self.assertEqual(len(segments), 2)
            self.assertEqual(sum(row["ionic_steps"] for row in segments), 2)
            (child / "relax_lineage.json").write_text(json.dumps({"segments": segments}))
            combined = relaxation_data({"local_dir": str(child)})
            self.assertEqual([row["ionic_step"] for row in combined["steps"]], [0, 1])
            self.assertEqual([row["segment_index"] for row in combined["steps"]], [1, 2])

    def test_qe_relax_monitor_refreshes_declared_parent_before_lineage_parse(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            child = root / "child_101"
            child.mkdir()
            (child / "endpoint_relax_manifest.json").write_text(json.dumps({
                "parent": {"remote_run_dir": "/scratch/parent_100"}
            }))
            with patch("monitor_qe_relax_jobs.run") as mocked_run:
                sync_lineage_parents(child, root, "rockfish")
            mocked_run.assert_called_once()
            command = mocked_run.call_args.args[0]
            self.assertEqual(command[0], "rsync")
            self.assertIn("rockfish:/scratch/parent_100/", command)

    def test_qe_relax_parser_does_not_overwrite_total_forces_with_d3_breakdown(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            relax_input = root / "relax.in"
            relax_input.write_text("""&SYSTEM\n  ibrav=0, nat=2, ntyp=1\n/\nATOMIC_SPECIES\nTe 127.6 Te.UPF\nATOMIC_POSITIONS angstrom\nTe 0 0 0\nTe 0 0 2\nCELL_PARAMETERS angstrom\n4 0 0\n0 4 0\n0 0 8\nK_POINTS gamma\n""")
            relax_out = root / "relax.out"
            relax_out.write_text("""! total energy = -10.0 Ry\nForces acting on atoms (cartesian axes, Ry/au):\natom 1 type 1 force = 0.00001 0.0 0.0\natom 2 type 1 force = -0.00001 0.0 0.0\nTotal force = 0.00002\nDFT-D3 dispersion contribution to forces:\natom 1 type 1 force = 0.003 0.0 0.0\natom 2 type 1 force = -0.003 0.0 0.0\nATOMIC_POSITIONS (angstrom)\nTe 0 0 0\nTe 0 0 2\nJOB DONE.\n""")
            parsed = parse_relax_out(relax_out, relax_input)
            self.assertAlmostEqual(
                parsed["final_max_atom_force_eV_A"], 0.00001 * RY_BOHR_TO_EV_A
            )

    def test_relaxation_report_formats_live_missing_force(self):
        self.assertEqual(display_number(None), "pending")

    def test_dataset_manuscript_gate_requires_four_acceptance_artifacts(self):
        args = SimpleNamespace(
            protocol="dft_neb",
            convergence_status="converged",
            path_acceptance=None,
            endpoint_acceptance=None,
            calculator_acceptance=None,
            runtime_provenance=None,
            pathway_id="fixture",
            calculator_identity="fixture_identity",
        )
        passed, reasons, hashes = manuscript_gate(args)
        self.assertFalse(passed)
        self.assertEqual(len(reasons), 4)
        self.assertTrue(all(value is None for value in hashes.values()))

    def test_qe_handoff_production_rejects_unverified_candidate(self):
        root = Path(__file__).resolve().parents[1]
        candidate_path = root / "tests/fixtures/nonlinear_candidate_smoke/1-6_iter108_img2_to_img5_v2_s42/nonlinear_candidate_manifest.json"
        direct_images = candidate_path.parent / "historical_curvature_warp.extxyz"
        with tempfile.TemporaryDirectory() as directory:
            acceptance_path = Path(directory) / "endpoint_acceptance.json"
            acceptance_path.write_text('{"status":"accepted"}')
            args = SimpleNamespace(
                run_role="production",
                initialization_role="direct_baseline",
                source_candidate_manifest=str(candidate_path),
                direct_baseline_images=str(direct_images),
                endpoint_acceptance=str(acceptance_path),
                source_mlff_manifest=None,
                max_mlff_residual=0.2,
            )
            with self.assertRaisesRegex(ValueError, "candidate_endpoint_status"):
                pre_output_gate(args, direct_images)

    def test_long_inline_workflow_json_is_not_treated_as_a_path(self):
        payload = '{"values": [' + ",".join("1" for _ in range(200)) + "]}"
        self.assertEqual(len(load_json_arg(payload, {})["values"]), 200)

    def test_task_relevant_residual_metrics(self):
        iterations = [
            {"optimizer_iteration": 0, "residual": 4.0},
            {"optimizer_iteration": 1, "residual": 2.0},
            {"optimizer_iteration": 3, "residual": 0.25},
        ]
        self.assertAlmostEqual(trapezoid_auc(iterations, "residual"), 5.25)
        self.assertEqual(first_step_below(iterations, "residual", 0.5), 3)
        self.assertIsNone(first_step_below(iterations, "residual", 0.1))

    def test_nonlinear_candidate_endpoints_are_exact_and_arc_is_not_linear(self):
        initial = {
            "symbols": ["Cr", "Te"],
            "positions": [(1.0, 1.0, 1.0), (5.0, 5.0, 5.0)],
            "cell": [(10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)],
        }
        final = {
            "symbols": ["Cr", "Te"],
            "positions": [(9.0, 1.0, 1.0), (5.0, 5.2, 5.0)],
            "cell": initial["cell"],
        }
        linear = linear_images(initial, final, 5)
        curved = clearance_arc(initial, final, 5, 0, 0.5, (0.0, 1.0, 0.0))
        self.assertEqual(curved[0]["positions"], initial["positions"])
        self.assertAlmostEqual(endpoint_pbc_deviation(curved[-1], final), 0.0)
        self.assertNotEqual(curved[2]["positions"][0], linear[2]["positions"][0])
        self.assertAlmostEqual(linear[-2]["positions"][0][0], -0.5)
        self.assertAlmostEqual(linear[-1]["positions"][0][0], -1.0)
        directions = perpendicular_directions((1.0, 1.0, 0.2), initial["cell"], 15.0)
        self.assertTrue(all(abs(sum(a * b for a, b in zip(left, right))) < math.cos(math.radians(15.0)) + 1e-12 for i, left in enumerate(directions) for right in directions[:i]))

    def test_historical_curvature_warp_maps_new_endpoints_exactly(self):
        cell = [(10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)]
        initial = {"symbols": ["Cr", "Te"], "positions": [(1.0, 1.0, 1.0), (5.0, 5.0, 5.0)], "cell": cell}
        final = {"symbols": ["Cr", "Te"], "positions": [(3.0, 1.0, 1.0), (5.0, 5.0, 5.0)], "cell": cell}
        history = [
            {"symbols": ["Cr", "Te"], "positions": [(2.0, 2.0, 1.0), (5.0, 5.0, 5.0)], "cell": cell},
            {"symbols": ["Cr", "Te"], "positions": [(3.0, 3.0, 1.0), (5.0, 5.0, 5.0)], "cell": cell},
            {"symbols": ["Cr", "Te"], "positions": [(4.0, 2.0, 1.0), (5.0, 5.0, 5.0)], "cell": cell},
        ]
        warped = historical_warp(initial, final, history, 5)
        self.assertEqual(warped[0]["positions"], initial["positions"])
        self.assertAlmostEqual(endpoint_pbc_deviation(warped[-1], final), 0.0)
        self.assertGreater(warped[2]["positions"][0][1], 1.5)
        self.assertAlmostEqual(path_rms_distance(warped, warped), 0.0)
        self.assertGreater(path_rms_distance(warped, linear_images(initial, final, 5)), 0.1)
        self.assertGreater(
            path_rms_distance(warped, linear_images(initial, final, 5), [0]),
            path_rms_distance(warped, linear_images(initial, final, 5)),
        )

    def test_multiframe_historical_curvature_donor_supports_explicit_segment(self):
        root = Path(__file__).resolve().parents[1]
        historical = root / "tests" / "fixtures" / "historical_neb" / "legacy_61" / "1-4"
        segment = historical_ase_images(
            historical / "sb2te3.xyz",
            historical / "pw_1.in",
            start_image_qe=1,
            end_image_qe=4,
        )
        self.assertEqual(len(segment), 4)
        self.assertEqual(segment[0]["symbols"].count("Cr"), 1)
        self.assertIsNotNone(segment[0]["cell"])

    def test_endpoint_remap_segment_requires_matching_accepted_image_records(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first = root / "first.extxyz"
            final = root / "final.extxyz"
            first.write_text("first")
            final.write_text("final")
            acceptance = root / "endpoint_pair_acceptance.json"
            acceptance.write_text(json.dumps({
                "status": "accepted",
                "initial_structure": str(first),
                "initial_structure_sha256": mlff_sha256(first),
                "final_structure": str(final),
                "final_structure_sha256": mlff_sha256(final),
                "endpoint_records": [
                    {"image_index_qe": 1, "status": "accepted", "calculator_identity": "calc"},
                    {"image_index_qe": 4, "status": "accepted", "calculator_identity": "calc"},
                ],
            }))
            evidence = validate_segment_acceptance(
                {"historical_start_image_qe": 1, "historical_end_image_qe": 4},
                acceptance,
            )
            self.assertEqual(evidence["calculator_identity"], "calc")
            rejected = json.loads(acceptance.read_text())
            rejected["endpoint_records"][1]["status"] = "rejected"
            acceptance.write_text(json.dumps(rejected))
            with self.assertRaisesRegex(ValueError, "not accepted"):
                validate_segment_acceptance(
                    {"historical_start_image_qe": 1, "historical_end_image_qe": 4},
                    acceptance,
                )

    def test_endpoint_remap_materializes_both_historical_segments_end_to_end(self):
        project_root = Path(__file__).resolve().parents[1]
        historical = project_root / "tests" / "fixtures" / "historical_neb" / "legacy_61" / "1-4"
        donor_images = historical / "sb2te3.xyz"
        donor_template = historical / "pw_1.in"
        historical_rows = historical_ase_images(donor_images, donor_template, 1, 5)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            structures = {}
            for image_index_qe in (1, 4, 5):
                row = historical_rows[image_index_qe - 1]
                atoms = Atoms(
                    row["symbols"],
                    positions=row["positions"],
                    cell=row["cell"],
                    pbc=True,
                )
                path = root / f"image_{image_index_qe}.extxyz"
                ase_write(path, atoms)
                structures[image_index_qe] = path

            segments = [
                ("1-4_image1_to_image4", 1, 4),
                ("1-4_image4_to_image5", 4, 5),
            ]
            acceptance_args = []
            for segment_id, initial_index, final_index in segments:
                acceptance = root / f"{segment_id}_acceptance.json"
                acceptance.write_text(json.dumps({
                    "status": "accepted",
                    "initial_structure": str(structures[initial_index]),
                    "initial_structure_sha256": mlff_sha256(structures[initial_index]),
                    "final_structure": str(structures[final_index]),
                    "final_structure_sha256": mlff_sha256(structures[final_index]),
                    "endpoint_records": [
                        {"image_index_qe": initial_index, "status": "accepted", "calculator_identity": "synthetic-qe-identity"},
                        {"image_index_qe": final_index, "status": "accepted", "calculator_identity": "synthetic-qe-identity"},
                    ],
                }))
                acceptance_args.append(f"{segment_id}={acceptance}")

            plan = root / "configs" / "endpoint_remap_plan.json"
            plan.parent.mkdir()
            plan.write_text(json.dumps({
                "seed": 42,
                "historical_curvature_donor": {
                    "images": str(donor_images),
                    "images_sha256": mlff_sha256(donor_images),
                    "qe_template": str(donor_template),
                    "qe_template_sha256": mlff_sha256(donor_template),
                },
                "segments": [
                    {
                        "segment_id": segment_id,
                        "historical_start_image_qe": initial_index,
                        "historical_end_image_qe": final_index,
                        "n_output_images": 7,
                    }
                    for segment_id, initial_index, final_index in segments
                ],
            }))
            output, result = materialize_endpoint_remap(SimpleNamespace(
                plan=plan,
                acceptance=acceptance_args,
                out_root=root / "materialized",
                generator=SCRIPT_DIR / "generate_nonlinear_neb_candidates.py",
            ))
            self.assertTrue(output.is_file())
            self.assertEqual(result["status"], "ready_for_new_mace_runs")
            self.assertEqual(len(result["segments"]), 2)
            for segment in result["segments"]:
                candidate = json.loads(Path(segment["candidate_manifest"]).read_text())
                self.assertEqual(candidate["endpoint_status"], "accepted_local_minima")
                self.assertEqual(candidate["seed"], 42)
                self.assertTrue(any(
                    row["production_eligible_before_mace"]
                    for row in candidate["candidates"]
                ))
                acceptance = json.loads(Path(segment["endpoint_acceptance"]).read_text())
                self.assertEqual(
                    candidate["initial_endpoint_source_sha256"],
                    acceptance["initial_structure_sha256"],
                )
                self.assertEqual(
                    candidate["final_endpoint_source_sha256"],
                    acceptance["final_structure_sha256"],
                )

    def test_mlff_candidate_launcher_separates_smoke_and_production_gates(self):
        row = {"branch_id": "curve", "geometry_gate": {"status": "pass"},
               "duplicate_of": None, "production_eligible_before_mace": False}
        manifest = {"endpoint_status": "unverified_smoke", "candidates": [row]}
        self.assertEqual(eligible_candidates(manifest, "smoke", {"curve"}, 2), [row])
        with self.assertRaises(RuntimeError):
            eligible_candidates(manifest, "smoke", {"curve"}, 3)
        with self.assertRaises(RuntimeError):
            eligible_candidates(manifest, "production", {"curve"}, 2)
        self.assertEqual(eligible_candidates(manifest, "diagnostic", {"curve"}, 200), [row])
        with self.assertRaises(RuntimeError):
            eligible_candidates(manifest, "diagnostic", {"curve"}, 501)
        manifest["endpoint_status"] = "accepted_local_minima"
        row["production_eligible_before_mace"] = True
        self.assertEqual(eligible_candidates(manifest, "production", {"curve"}, 500), [row])
    def test_qe_neb_restart_gate_and_attempt_name(self):
        row = {
            "job_id": "2", "classification": "clean_max_seconds_restartable",
            "restart_artifacts": {"restart_ready": True, "latest_path_iteration": 7},
        }
        self.assertEqual(select_neb_restart_parent({"jobs": [row]}, "2"), row)
        self.assertEqual(derive_neb_restart_name("mb_path_r3_s42", 4, 42), "mb_path_r4_s42")
        row["classification"] = "terminal_needs_review"
        with self.assertRaises(RuntimeError):
            select_neb_restart_parent({"jobs": [row]}, "2")

    def test_neb_convergence_uses_movable_images_and_complete_final_iteration(self):
        text = """
------------------------------ iteration  1 ------------------------------
activation energy (->) = 1.000000 eV
activation energy (<-) = 0.500000 eV
image energy (eV) error (eV/A) frozen
1 -10.0 9.0 T
2 -9.0 0.020 F
3 -9.5 8.0 T

------------------------------ iteration  2 ------------------------------
activation energy (->) = 1.010000 eV
activation energy (<-) = 0.510000 eV
image energy (eV) error (eV/A) frozen
1 -10.0 9.0 T
2 -9.0 0.019 F
3 -9.5 8.0 T

JOB DONE.
"""
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "neb.out"
            output.write_text(text)
            parsed = parse_neb_out(output)
            self.assertAlmostEqual(parsed["max_image_error_eV_A"], 0.019)
            self.assertAlmostEqual(parsed["max_image_error_all_eV_A"], 9.0)
            self.assertTrue(parsed["converged_by_default_gate"])
            output.write_text(text.replace("JOB DONE.", "------------------------------ iteration  3 ------------------------------\nJOB DONE."))
            self.assertFalse(parse_neb_out(output)["converged_by_default_gate"])

    def test_neb_clean_restart_classification_requires_time_guard_and_artifacts(self):
        parsed = {"has_overflow": False, "converged_by_default_gate": False, "job_done": True, "last_tcpu_seconds": 84000.0, "n_complete_iterations_parsed": 2}
        limits = {"max_seconds": 84600.0}
        restart = {"restart_ready": True}
        self.assertEqual(classify_neb_job(parsed, "not_in_squeue", "COMPLETED", limits, restart), "clean_max_seconds_restartable")
        restart["restart_ready"] = False
        self.assertEqual(classify_neb_job(parsed, "not_in_squeue", "COMPLETED", limits, restart), "terminal_needs_review")

    def test_neb_assembler_selects_latest_warmup_attempt_per_image(self):
        rows = [
            {"path_id": "p", "image_index_qe": 1, "job_id": "10", "attempt": 1},
            {"path_id": "p", "image_index_qe": 1, "job_id": "12", "attempt": 2},
            {"path_id": "p", "image_index_qe": 2, "job_id": "11", "attempt": 1},
        ]
        selected = select_warmup_jobs({"jobs": rows}, "p", 2)
        self.assertEqual(selected[1]["job_id"], "12")
        self.assertEqual(selected[2]["job_id"], "11")

    def test_scf_continuation_gate_requires_clean_timeout_and_restart_artifacts(self):
        row = {
            "job_id": "1", "classification": "clean_timeout_needs_restart", "queue_state": "COMPLETED",
            "restart_artifacts": {"charge_density": True, "xml": True},
        }
        self.assertEqual(selected_parent({"jobs": [row]}, "1"), row)
        row["queue_state"] = "RUNNING"
        with self.assertRaises(RuntimeError):
            selected_parent({"jobs": [row]}, "1")

    def test_scf_warmup_acceptance_requires_finite_energy_and_charge_xml(self):
        parsed = {"job_done": True, "scf_converged": True, "final_energy_Ry": -100.0}
        restart = {"charge_density": True, "xml": True, "wavefunction_files": 0}
        self.assertEqual(classify_scf_warmup("COMPLETED", parsed, restart), "accepted_scf_warmstart")
        parsed["final_energy_Ry"] = None
        self.assertEqual(classify_scf_warmup("COMPLETED", parsed, restart), "terminal_needs_review")

    def test_qe_magnetic_observables_preserve_scalar_and_noncollinear_site_vectors(self):
        text = """
 total magnetization       =     2.5000 Bohr mag/cell
 absolute magnetization    =     2.7000 Bohr mag/cell
 atom:    1    charge:  13.20    magn:  2.40
 atom:    2    charge:  14.70    magn:  0.10  -0.20   0.30
 total magnetization       =     2.6000 Bohr mag/cell
 atom:    1    charge:  13.25    magn:  2.50
 """
        parsed = magnetic_observables(text, ["Cr", "Sb"])
        self.assertAlmostEqual(parsed["total_magnetization_Bohr_magneton_cell"], 2.6)
        self.assertEqual(parsed["total_magnetization_components_Bohr_magneton_cell"], [2.6])
        self.assertAlmostEqual(parsed["absolute_magnetization_Bohr_magneton_cell"], 2.7)
        self.assertEqual(parsed["site_magnetic_moments_last"][0]["species"], "Cr")
        self.assertEqual(
            parsed["site_magnetic_moments_last"][1]["magnetization_components_Bohr_magneton"],
            [0.1, -0.2, 0.3],
        )

    def test_scf_warmup_batch_helpers_are_path_scoped_and_deterministic(self):
        status = {
            "jobs": [
                {"path_id": "81_neb_4", "image_index_qe": 1, "classification": "accepted_scf_warmstart"},
                {"path_id": "81_neb_4", "image_index_qe": 2, "classification": "scf_running"},
                {"path_id": "other", "image_index_qe": 2, "classification": "accepted_scf_warmstart"},
            ]
        }
        self.assertEqual(parse_image_list("1-2,4"), [1, 2, 4])
        self.assertEqual(accepted_images(status, "81_neb_4"), {1})
        self.assertEqual(path_job_slug("81_neb_4"), "81n4")
        self.assertEqual(walltime_seconds("1-00:00:00"), 86400)

    def test_qe_namelist_update_replaces_or_adds_only_requested_value(self):
        text = "&CONTROL\n  max_seconds = 100\n/\n&ELECTRONS\n  conv_thr = 1d-6\n/\n"
        updated = set_namelist_value(text, "CONTROL", "max_seconds", "200")
        updated = set_namelist_value(updated, "CONTROL", "disk_io", "'low'")
        self.assertIn("max_seconds = 200", updated)
        self.assertIn("disk_io = 'low'", updated)
        self.assertIn("conv_thr = 1d-6", updated)

    def test_relaxation_energy_drop_is_positive_when_energy_decreases(self):
        metrics = energy_metrics({"steps": [{"energy_eV": -10.0}, {"energy_eV": -10.4}]})
        self.assertAlmostEqual(metrics["relaxation_energy_drop_eV"], 0.4)

    def test_force_threshold_reports_first_reached_step(self):
        parsed = {"steps": [
            {"ionic_step": 0, "max_atom_force_eV_A": 0.8},
            {"ionic_step": 1, "max_atom_force_eV_A": 0.18},
            {"ionic_step": 2, "max_atom_force_eV_A": 0.09},
        ]}
        self.assertEqual(first_force_threshold_step(parsed, 0.2), 1)
        self.assertEqual(first_force_threshold_step(parsed, 0.1), 2)
        self.assertIsNone(first_force_threshold_step(parsed, 0.05))

    def test_sustained_force_threshold_rejects_transient_crossing(self):
        parsed = {"steps": [
            {"ionic_step": 0, "max_atom_force_eV_A": 0.8, "cumulative_scf_iterations": 8},
            {"ionic_step": 1, "max_atom_force_eV_A": 0.08, "cumulative_scf_iterations": 15},
            {"ionic_step": 2, "max_atom_force_eV_A": 0.25, "cumulative_scf_iterations": 24},
            {"ionic_step": 3, "max_atom_force_eV_A": 0.09, "cumulative_scf_iterations": 30},
            {"ionic_step": 4, "max_atom_force_eV_A": 0.04, "cumulative_scf_iterations": 35},
        ]}
        self.assertEqual(first_force_threshold_step(parsed, 0.1), 1)
        self.assertEqual(
            first_sustained_force_threshold(parsed, 0.1),
            {"ionic_step": 3, "cumulative_scf_iterations": 30},
        )
        self.assertEqual(
            first_sustained_force_threshold(parsed, 0.05),
            {"ionic_step": 4, "cumulative_scf_iterations": 35},
        )

    def test_relax_parser_attaches_cumulative_scf_work_to_each_evaluation(self):
        output = """iteration # 1
iteration # 2
! total energy = -10.0 Ry
atom 1 type 1 force = 0.1 0.0 0.0
Total force = 0.1
ATOMIC_POSITIONS (angstrom)
Cr 0 0 0
iteration # 1
! total energy = -10.1 Ry
atom 1 type 1 force = 0.01 0.0 0.0
Total force = 0.01
ATOMIC_POSITIONS (angstrom)
Cr 0.1 0 0
"""
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "relax.out"
            path.write_text(output)
            parsed = parse_relax_out(path)
        self.assertEqual(
            [step["cumulative_scf_iterations"] for step in parsed["steps"]],
            [2, 3],
        )

    def test_exact_ab_monitor_cadence_uses_newest_status(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            older = root / "older.json"
            newer = root / "newer.json"
            older.write_text(json.dumps({"created_at_utc": "2026-09-13T09:00:00+00:00"}))
            newer.write_text(json.dumps({"created_at_utc": "2026-09-13T11:00:00+00:00"}))
            age = hours_since_status(
                [older, newer], datetime.fromisoformat("2026-09-13T12:30:00+00:00")
            )
            self.assertAlmostEqual(age, 1.5)

    def test_historical_mlff_candidate_fails_closed_without_pbc_gate(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            images = root / "path.extxyz"
            images.write_text("fixture\n")
            repair = root / "path.manifest.json"
            repair.write_text(json.dumps({
                "output_images": str(images),
                "output_sha256": hashlib.sha256(images.read_bytes()).hexdigest(),
                "output_pbc": [False, False, False],
                "endpoints_pass": True,
                "all_intermediate_images_pass": True,
            }))
            with self.assertRaisesRegex(RuntimeError, "repair gate failed"):
                historical_mlff_candidate("fixture", repair, root, 42)

    def test_ab_comparison_never_finalizes_basin_or_speedup_from_running_prefix(self):
        calculator = {
            "calculator_identities_match": True,
            "calculator_identity_accepted_by_n24": True,
            "lineage_runtime_provenance_complete": True,
        }
        self.assertEqual(
            comparison_decision(False, "pending", calculator),
            "pending_not_converged",
        )
        calculator["calculator_identity_accepted_by_n24"] = False
        self.assertEqual(
            comparison_decision(True, "same", calculator),
            "same_basin_calculator_gate_pending",
        )
        calculator["calculator_identity_accepted_by_n24"] = True
        self.assertEqual(
            comparison_decision(True, "same", calculator),
            "comparable_same_basin_accepted_calculator",
        )
        calculator["lineage_runtime_provenance_complete"] = False
        self.assertEqual(
            comparison_decision(True, "same", calculator),
            "same_basin_runtime_provenance_pending",
        )
        calculator["lineage_runtime_provenance_complete"] = True
        calculator["calculator_identities_match"] = False
        self.assertEqual(
            comparison_decision(True, "same", calculator),
            "calculator_identity_mismatch",
        )

    def test_ab_speedup_runtime_gate_covers_every_lineage_segment(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            segment = root / "segment"
            segment.mkdir()
            relax_input = segment / "relax.in"
            relax_input.write_text("fixture\n")
            (segment / "endpoint_relax_manifest.json").write_text(json.dumps({
                "submission": {"job_id": "123"},
            }))
            (root / "relax_lineage.json").write_text(json.dumps({
                "segments": [{"local_dir": str(segment)}],
            }))
            job = {"local_dir": str(root), "job_id": "456"}
            self.assertFalse(lineage_runtime_provenance(job)["passed"])
            (segment / "runtime_provenance.json").write_text(json.dumps({
                "environment": {"SLURM_JOB_ID": "123"},
                "inputs": {"relax_input": {"exists": True, "sha256": mlff_sha256(relax_input)}},
                "binaries": {"pw.x": {"exists": True, "sha256": "pw-hash"}},
            }))
            runtime = lineage_runtime_provenance(job)
            self.assertTrue(runtime["passed"])
            self.assertEqual(runtime["segment_count"], 1)

    def test_ab_speedup_requires_hash_bound_repeat_acceptance(self):
        calculator = {
            "calculator_identities_match": True,
            "calculator_identity_accepted_by_n24": True,
            "lineage_runtime_provenance_complete": True,
        }
        self.assertEqual(
            comparison_decision(True, "pending", calculator, "pending"),
            "same_basin_repeat_acceptance_pending",
        )
        self.assertEqual(
            comparison_decision(True, "pending", calculator, "rejected"),
            "same_basin_repeat_acceptance_rejected",
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            structure = root / "accepted.extxyz"
            ase_write(
                structure,
                Atoms("CrTe", positions=[[0, 0, 0], [2.5, 0, 0]], cell=[10, 10, 10], pbc=True),
                format="extxyz",
            )
            acceptance = root / "endpoint_acceptance_batch.json"
            acceptance.write_text(json.dumps({"endpoint_records": [{
                "path_id": "site-a",
                "image_index_qe": 1,
                "parent_job_id": "123",
                "status": "accepted",
                "accepted_structure": str(structure),
                "accepted_structure_sha256": mlff_sha256(structure),
            }]}))
            result = endpoint_repeat_record(
                {"path_id": "site-a", "image_index_qe": 1, "job_id": "123"},
                acceptance,
            )
            self.assertEqual(result["state"], "accepted")
            self.assertEqual(len(result["accepted_step"]["positions_A"]), 2)
            payload = json.loads(acceptance.read_text())
            payload["endpoint_records"][0]["accepted_structure_sha256"] = "bad"
            acceptance.write_text(json.dumps(payload))
            self.assertEqual(
                endpoint_repeat_record(
                    {"path_id": "site-a", "image_index_qe": 1, "job_id": "123"},
                    acceptance,
                )["state"],
                "invalid_structure_provenance",
            )

    def test_ab_comparator_accepts_two_arm_specific_acceptance_artifacts(self):
        direct = Path("direct_acceptance.json")
        transformed = Path("transformed_acceptance.json")
        self.assertEqual(
            resolve_acceptance_paths(None, direct, transformed),
            (direct, transformed),
        )
        combined = Path("combined_acceptance.json")
        self.assertEqual(
            resolve_acceptance_paths(combined, None, None),
            (combined, combined),
        )
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            resolve_acceptance_paths(None, direct, None)
        with self.assertRaisesRegex(ValueError, "not both"):
            resolve_acceptance_paths(combined, direct, transformed)

    def test_site_transform_pairing_is_exact_and_hash_bound_to_direct_jobs(self):
        direct = {"jobs": [
            {"branch_id": f"direct-{index}", "job_id": str(100 + index)}
            for index in range(6)
        ]}
        transformed = {"jobs": [
            {"branch_id": f"transform-{index}", "job_id": str(200 + index)}
            for index in range(6)
        ]}
        manifest = {"jobs": [
            {
                "branch_id": f"transform-{index}",
                "source_arm": "proposal",
                "direct_baseline_branch_id": f"direct-{index}",
                "direct_baseline_job_id": str(100 + index),
                "source_path_id": f"site-{index}",
                "source_image_index_qe": 1,
            }
            for index in range(6)
        ]}
        self.assertEqual(
            len(validate_site_transform_pairing(direct, transformed, manifest)), 6
        )
        manifest["jobs"][0]["direct_baseline_job_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Direct baseline job binding mismatch"):
            validate_site_transform_pairing(direct, transformed, manifest)

    def test_relaxation_pair_dataset_fails_closed_on_speedup_and_leakage(self):
        row = {
            "relaxation_pair_id": "RP_fixture",
            "eligible_for_current_proposer_training": False,
            "formal_speedup": False,
            "comparison_status": "pending_not_converged",
            "basin_equivalence": "pending",
            "direct_repeat_acceptance": "pending",
            "transformed_repeat_acceptance": "pending",
            "runtime_provenance_complete": False,
            "calculator_identity_accepted": False,
            "speedup_metrics": {
                "ionic_step_speedup": None,
                "scf_iteration_speedup": None,
                "walltime_speedup": None,
            },
        }
        self.assertEqual(validate_relaxation_pair_rows([row]), [])
        invalid = dict(row)
        invalid["speedup_metrics"] = dict(row["speedup_metrics"])
        invalid["speedup_metrics"]["ionic_step_speedup"] = 1.2
        self.assertTrue(validate_relaxation_pair_rows([invalid]))
        invalid = dict(row)
        invalid["eligible_for_current_proposer_training"] = True
        self.assertTrue(validate_relaxation_pair_rows([invalid]))

        formal_without_acceptance_provenance = dict(row)
        formal_without_acceptance_provenance.update({
            "formal_speedup": True,
            "comparison_status": "comparable_same_basin_accepted_calculator",
            "basin_equivalence": "same",
            "direct_repeat_acceptance": "accepted",
            "transformed_repeat_acceptance": "accepted",
            "runtime_provenance_complete": True,
            "calculator_identity_accepted": True,
            "speedup_metrics": {
                "ionic_step_speedup": 1.1,
                "scf_iteration_speedup": 1.1,
                "walltime_speedup": 1.1,
            },
            "metadata_json": {},
        })
        failures = validate_relaxation_pair_rows(
            [formal_without_acceptance_provenance]
        )
        self.assertTrue(any("direct acceptance provenance" in item for item in failures))
        self.assertTrue(any("transformed acceptance provenance" in item for item in failures))

    def test_site_transform_repeat_plan_excludes_matched_controls(self):
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw) / "source.json"
            source.write_text("{}\n")
            direct_jobs = []
            transformed_jobs = []
            for index in range(6):
                direct_jobs.extend([
                    {"branch_id": f"proposal-{index}", "arm": "proposal"},
                    {"branch_id": f"control-{index}", "arm": "control"},
                ])
                transformed_jobs.append({
                    "direct_baseline_branch_id": f"proposal-{index}",
                })
            subset = paired_direct_batch(
                {"jobs": direct_jobs}, {"jobs": transformed_jobs}, source
            )
            self.assertEqual(len(subset["jobs"]), 6)
            self.assertEqual({row["arm"] for row in subset["jobs"]}, {"proposal"})
            status_path = Path(raw) / "status.json"
            status_path.write_text("{}\n")
            status = {"jobs": [
                {"branch_id": row["branch_id"], "classification": "pending"}
                for row in direct_jobs
            ]}
            filtered = filter_site_transform_status(
                status,
                {row["branch_id"] for row in subset["jobs"]},
                status_path,
            )
            self.assertEqual(len(filtered["jobs"]), 6)

            repeat_rows = [
                {
                    "path_id": f"site-{index}",
                    "image_index_qe": 1,
                    "classification": "accepted_local_minimum",
                }
                for index in range(12)
            ]
            repeat_path = Path(raw) / "repeat.json"
            repeat_path.write_text(json.dumps({"jobs": repeat_rows}) + "\n")
            repeat_view = filter_site_transform_endpoint_status(
                {"jobs": repeat_rows},
                {(f"site-{index}", 1) for index in range(6)},
                repeat_path,
            )
            self.assertEqual(len(repeat_view["jobs"]), 6)
            self.assertEqual(
                {row["path_id"] for row in repeat_view["jobs"]},
                {f"site-{index}" for index in range(6)},
            )
            self.assertEqual(
                repeat_view["source_status_sha256"], mlff_sha256(repeat_path)
            )

    def test_ab_basin_reference_uses_synced_direct_input_not_manifest_source(self):
        comparator = (Path(__file__).resolve().parents[1] / "scripts/migrationbench/compare_endpoint_relaxation_speed.py").read_text()
        self.assertIn('Path(direct_job["local_dir"]) / "relax.in"', comparator)
        self.assertNotIn('Path(transform.get("source_pw_input")', comparator)

    def test_local_site_descriptor_is_translation_rotation_permutation_invariant(self):
        import numpy as np
        cell = np.diag([10.0, 10.0, 10.0])
        image = {
            "symbols": ["Cr", "Te", "Sb", "Te"],
            "positions": [(1.0, 1.0, 1.0), (3.0, 1.0, 1.0), (1.0, 3.0, 1.0), (1.0, 1.0, 4.0)],
        }
        names, first = local_descriptor(image, cell)
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotated = {
            "symbols": ["Cr", "Te", "Te", "Sb"],
            "positions": [tuple(np.asarray(image["positions"][index]) @ rotation + 2.0) for index in (0, 3, 1, 2)],
        }
        _, second = local_descriptor(rotated, cell @ rotation)
        self.assertEqual(len(names), len(first))
        np.testing.assert_allclose(first, second, atol=1.0e-10)

    def test_pairwise_order_accuracy_ignores_near_ties(self):
        self.assertEqual(pairwise_order_accuracy([0.0, 1.0, 1.05], [0.0, 0.9, -5.0], 0.2), 0.5)

    def test_dft_site_label_rejects_output_older_than_input(self):
        import os
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            input_path, output_path = root / "relax.in", root / "relax.out"
            input_path.write_text("new input\n")
            output_path.write_text("JOB DONE.\n")
            os.utime(output_path, (1, 1))
            os.utime(input_path, (2, 2))
            status, _ = dft_label_status(input_path, output_path)
            self.assertEqual(status, "historical_output_unlinked_to_current_input")

    def test_qe_calculator_identity_requires_explicit_soc_flags(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "pw.in"
            path.write_text("""&CONTROL
 calculation='scf'
/
&SYSTEM
 ecutwfc=70, ecutrho=280, occupations='smearing', smearing='cold', degauss=0.005,
 noncolin=.true., lspinorb=.true., vdw_corr='DFT-D3'
/
&ELECTRONS
 conv_thr=1d-6
/
ATOMIC_SPECIES
Cr 51.9 Cr.upf
Te 127.6 Te.upf
K_POINTS gamma
CELL_PARAMETERS angstrom
10 0 0
0 10 0
0 0 10
ATOMIC_POSITIONS angstrom
Cr 0 0 0
Te 2 0 0
""")
            row = qe_calculator_identity(path, "test")
            self.assertTrue(row["soc_enabled"])
            self.assertEqual(row["composition"], "Cr1Te1")
            path.write_text(path.read_text().replace("lspinorb=.true.", "lspinorb=.false."))
            self.assertFalse(qe_calculator_identity(path, "test")["soc_enabled"])

    def test_protocol_input_spin_and_kpoint_transforms_are_explicit(self):
        text = "&SYSTEM\n nspin=2\n starting_magnetization(1)=0.5\n/\nK_POINTS gamma\nCELL_PARAMETERS angstrom\n1 0 0\n0 1 0\n0 0 1\n"
        soc = apply_spin_model(text, "soc", 0.5)
        self.assertNotIn("nspin", soc.lower())
        self.assertIn("noncolin = .true.", soc)
        self.assertIn("lspinorb = .true.", soc)
        automatic = replace_kpoints(soc, {"mode": "automatic", "grid_shift": [2, 2, 1, 0, 0, 0]})
        self.assertIn("K_POINTS AUTOMATIC\n2 2 1 0 0 0", automatic)
        self.assertAlmostEqual(SENSITIVITY_RY_TO_EV, 13.605693122994)

    def test_protocol_gate_distinguishes_pending_invalid_fail_and_pass(self):
        base = [
            {"profile_id": "ref", "both_scf_accepted": True,
             "delta_E_difference_from_reference_eV": 0.0, "job_classifications": ["accepted_scf_warmstart"] * 2},
            {"profile_id": "test", "both_scf_accepted": False,
             "delta_E_difference_from_reference_eV": None, "job_classifications": ["scf_running", "scf_running"]},
        ]
        self.assertEqual(evaluate_gate(base, "ref", 0.02)[0], "pending")
        base[1]["job_classifications"] = ["terminal_needs_review", "scf_running"]
        self.assertEqual(evaluate_gate(base, "ref", 0.02)[0], "invalid")
        base[1].update({"both_scf_accepted": True, "delta_E_difference_from_reference_eV": 0.03})
        self.assertEqual(evaluate_gate(base, "ref", 0.02)[0], "fail")
        base[1]["delta_E_difference_from_reference_eV"] = 0.01
        self.assertEqual(evaluate_gate(base, "ref", 0.02)[0], "pass")

    def test_next_protocol_stage_is_fail_closed_and_preserves_accepted_cutoff(self):
        source = {
            "path_id": "1-6", "source_run_dir": "/tmp/run", "source_iteration": 108,
            "image_indices": [2, 4], "geometry_roles": {"2": "low", "4": "high"},
            "pair_delta_E_tolerance_eV": 0.02, "seed": 42,
            "resources": {"walltime": "24:00:00", "max_seconds": 84600, "memory": "160G"},
            "profiles": [{"id": "hist", "job_slug": "h", "ecutwfc_Ry": 50,
                          "ecutrho_Ry": 200, "degauss_Ry": 0.01, "spin_model": "nonspin",
                          "kpoints": {"mode": "gamma"}, "ntasks": 2, "kpoint_pools": 1}],
        }
        with self.assertRaises(RuntimeError):
            build_kpoint_config({"gate": "pending", "reference_profile": "hist"}, source, Path("out"))
        config = build_kpoint_config(
            {"gate": "pass", "gate_reason": "within_tolerance", "reference_profile": "hist"},
            source, Path("out"),
        )
        self.assertEqual(config["profiles"][1]["ecutwfc_Ry"], 50)
        self.assertEqual(config["profiles"][1]["kpoints"]["grid_shift"], [2, 2, 1, 0, 0, 0])
        self.assertEqual(config["profiles"][1]["kpoint_pools"], 1)

    def test_composition_formula_is_derived_for_arbitrary_materials(self):
        self.assertEqual(composition_formula(["Te", "Cr", "Te", "Sb"]), "Cr1Sb1Te2")

    def test_pw_force_blocks_preserve_dft_energy_force_and_scf_work(self):
        text = """
!    total energy              =   -20.00000000 Ry
     convergence has been achieved in  12 iterations
     Forces acting on atoms (cartesian axes, Ry/au):
     atom    1 type  1   force =     0.01000000  0.00000000  0.00000000
     atom    2 type  2   force =    -0.01000000  0.00000000  0.00000000
"""
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "PW.out"
            path.write_text(text)
            blocks = parse_pw_blocks(path, 2)
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0]["scf_iterations"], 12)
            self.assertAlmostEqual(blocks[0]["energy_eV"], -20.0 * RY_TO_EV)
            self.assertAlmostEqual(blocks[0]["forces_eV_A"][0][0], 0.01 * RY_PER_BOHR_TO_EV_PER_A)

    def test_qe_path_restart_preserves_iteration_energy_geometry_and_gradient(self):
        text = """RESTART INFORMATION
       7
    1000
       0
NUMBER OF IMAGES
   1
APPLY CONSTANT BIAS
F
ENERGIES, POSITIONS AND GRADIENTS
Image:    1
      -2.000000000000
       1.000000000000 2.000000000000 3.000000000000 0.010000000000 0.000000000000 0.000000000000
       4.000000000000 5.000000000000 6.000000000000 0.000000000000 0.020000000000 0.000000000000
"""
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "test.path7"
            path.write_text(text)
            parsed = parse_path_file(path, 2)
            self.assertEqual(parsed["iteration"], 7)
            self.assertAlmostEqual(parsed["images"][0]["energy_eV"], -2.0 * HARTREE_TO_EV)
            self.assertAlmostEqual(parsed["images"][0]["atoms"][0]["position_A"][0], BOHR_TO_A)
            self.assertEqual(parsed["images"][0]["atoms"][0]["move_mask"], [-1, -1, -1])

    def test_hidden_basin_segmentation(self):
        rows = [
            {"energy_eV": 0.0, "error_eV_A": 0.0},
            {"energy_eV": -1.0, "error_eV_A": 0.1},
            {"energy_eV": 0.5, "error_eV_A": 0.2},
            {"energy_eV": 0.2, "error_eV_A": 0.1},
        ]
        config = {"energy_local_minimum_tolerance_eV": 0.02, "candidate_force_max_eV_A": 0.30}
        basins, segments, indices = energy_topology(rows, config)
        self.assertEqual(indices, [0, 1, 3])
        self.assertEqual(len(basins), 1)
        self.assertTrue(segments[0]["endpoint_dominated"])
        self.assertAlmostEqual(segments[1]["barrier_forward_eV"], 1.5)
        self.assertAlmostEqual(segments[1]["barrier_reverse_eV"], 0.3)

    def test_site_keys_and_pairs_are_system_scoped(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            paths = []
            for source, z0 in (("a", 0.0), ("b", 1.0)):
                path = root / f"{source}.csv"
                with path.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["unique_id", "cr_z", "mace_energy_eV"])
                    writer.writeheader()
                    writer.writerow({"unique_id": "unique_001", "cr_z": z0, "mace_energy_eV": -10.0})
                    writer.writerow({"unique_id": "unique_002", "cr_z": z0 + 0.5, "mace_energy_eV": -9.9})
                paths.append(str(path))
            sites, _ = read_sites(paths)
            sites = annotate_sites(sites)
            pairs = build_pairs(sites, 0.1, 2.0, 2.0)
            self.assertEqual(len({site["site_id"] for site in sites}), 4)
            self.assertEqual(len(pairs), 2)
            self.assertTrue(all(pair["start_site_id"].split("::")[0] == pair["end_site_id"].split("::")[0] for pair in pairs))

    def test_qe_relax_iterations_preserve_energy_and_forces(self):
        text = """
!    total energy              =     -10.00000000 Ry
     Forces acting on atoms (cartesian axes, Ry/au):
     atom    1 type  1   force =     0.00100000  0.00000000  0.00000000
     atom    2 type  2   force =    -0.00100000  0.00000000  0.00000000
     Total force =     0.002000
ATOMIC_POSITIONS (angstrom)
Cr 0.0 0.0 0.0
Te 2.0 0.0 0.0

     bfgs converged in 1 scf cycles and 1 bfgs steps
     End of BFGS Geometry Optimization
     JOB DONE.
"""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "relax.out"
            output.write_text(text)
            parsed = parse_relax_out(output)
            self.assertTrue(parsed["job_done"])
            self.assertTrue(parsed["bfgs_converged"])
            self.assertEqual(parsed["n_ionic_steps"], 1)
            self.assertEqual(parsed["total_scf_iterations_seen"], 0)
            self.assertAlmostEqual(parsed["final_max_atom_force_eV_A"], 0.001 * RY_BOHR_TO_EV_A)
            extxyz = root / "iterations.extxyz"
            write_extxyz(extxyz, parsed["steps"])
            self.assertIn("Properties=species:S:1:pos:R:3:forces:R:3", extxyz.read_text())


if __name__ == "__main__":
    unittest.main()
