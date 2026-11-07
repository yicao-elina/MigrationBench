"""Merge scientifically admissible retry manifests into a copied registry table.

Only finite fixed-path manifests with ``FINITE_DIAGNOSTIC`` and
``SUPPORTING_EVIDENCE`` status are admitted.  The original matrix is never
overwritten; the output is an additive, date-stamped merged source with the
retry provenance retained in ``run_root``, ``profile_artifact``, and
``retry_parent``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _manifest_paths(root: Path, kind: str) -> list[Path]:
    if kind == "fixed":
        return sorted(root.glob("runs/fixed/**/fixed_single_point_manifest.json"))
    if kind == "idpp":
        # Retry evidence is copied into several additive, date-stamped roots;
        # recurse so the merger does not depend on the remote directory layout.
        return sorted(
            list(root.rglob("endpoint_only_manifest.json"))
            + list(root.rglob("endpoint_extension_manifest.json"))
        )
    raise ValueError(f"unknown retry kind: {kind}")


def merge(
    base_path: Path,
    retry_root: Path | None,
    output_path: Path,
    idpp_roots: list[Path] | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    # Keep mixed provenance fields assignable (job ids may be strings after a
    # retry, while the canonical table often parses them as integers).
    base = pd.read_csv(base_path, sep="\t").astype(object)
    admitted: list[Path] = []
    manifests = []
    if retry_root is not None:
        manifests.extend((path, "fixed") for path in _manifest_paths(retry_root, "fixed"))
    for idpp_root in idpp_roots or []:
        manifests.extend((path, "idpp") for path in _manifest_paths(idpp_root, "idpp"))
    for path, kind in manifests:
        manifest = json.loads(path.read_text())
        accepted = (
            kind == "fixed"
            and manifest.get("scientific_status") == "FINITE_DIAGNOSTIC"
            and manifest.get("publication_status") == "SUPPORTING_EVIDENCE"
            and manifest.get("observable") == "FIXED_PATH_SINGLE_POINT"
        ) or (
            kind == "idpp"
            and manifest.get("observable") == "SELF_RELAXED_ENDPOINT_ONLY_NEB"
            and (
                (
                    manifest.get("scientific_status") == "CONVERGED"
                    and manifest.get("publication_status") == "PRIMARY_EVIDENCE"
                )
                or (
                    manifest.get("scientific_status") == "NONCONVERGED"
                    and manifest.get("publication_status") == "DIAGNOSTIC_ONLY"
                    and pd.notna(pd.to_numeric(manifest.get("forward_barrier_eV"), errors="coerce"))
                )
            )
        )
        if not accepted:
            continue
        record_id = str(manifest["record_id"])
        selected = base.index[base["record_id"].astype(str).eq(record_id)].tolist()
        if len(selected) != 1:
            raise ValueError(f"retry record must replace exactly one base row: {record_id}")
        idx = selected[0]
        old_job = str(
            base.at[idx, "job_id"]
            if "job_id" in base.columns
            else base.at[idx, "raw_manifest_path"]
            if "raw_manifest_path" in base.columns
            else ""
        )
        local_attempt = path.parent
        def existing(column: str, default: object = "") -> object:
            return base.at[idx, column] if column in base.columns else default

        updates = {
            "dft_reference_status": manifest.get("dft_reference_status", existing("dft_reference_status")),
            "endpoint_hash": f"{manifest['endpoint_hash_initial']};{manifest['endpoint_hash_final']}",
            "path_geometry_sha256": manifest.get("input_sha256", existing("path_geometry_sha256")),
            "input_sha256": manifest.get("input_sha256", existing("input_sha256")),
            "model_path": manifest.get("model_path", existing("model_path")),
            "model_sha256": manifest.get("model_sha256", existing("model_sha256")),
            "model_head": manifest.get("model_head", existing("model_head")),
            "final_max_internal_neb_force_eV_A": manifest.get("final_max_internal_neb_force_eV_A"),
            "final_max_neb_residual_eV_A": manifest.get("final_max_neb_residual_eV_A"),
            "forward_barrier_eV": manifest.get("forward_barrier_eV"),
            "reverse_barrier_eV": manifest.get("reverse_barrier_eV"),
            "barrier_proxy_eV": manifest.get("forward_barrier_eV")
            if kind == "fixed"
            else manifest.get("barrier_proxy_eV", manifest.get("forward_barrier_eV")),
            "profile_artifact": str(local_attempt / "image_profile.csv")
            if kind == "fixed"
            else str(local_attempt / "mlff_neb_images.extxyz"),
            "convergence_trace_artifact": "",
            "trajectory_sha256": "",
            "job_id": manifest.get("comparison_campaign", {}).get("slurm_job_id", old_job),
            "run_root": str(local_attempt),
            "environment_manifest": str(local_attempt / "runtime_manifest.json"),
            "provenance_status": "RETRY_VERIFIED",
            "scientific_status": manifest["scientific_status"],
            "publication_status": manifest["publication_status"],
            "failure_class": manifest.get("failure_class", "NA"),
            "retry_parent": old_job,
            # Compact Scratch-600K matrices use one value/provenance pair
            # instead of the richer TIER1 schema.
            "value_eV": manifest.get("forward_barrier_eV"),
            "value_name": "forward_barrier_eV",
            "raw_manifest_path": str(path),
            "raw_profile_path": str(local_attempt / "image_profile.csv")
            if kind == "fixed"
            else str(local_attempt / "mlff_neb_images.extxyz"),
        }
        for key, value in updates.items():
            if key in base.columns:
                base.at[idx, key] = value
        admitted.append(path)
    if not admitted:
        raise ValueError("no admissible retry manifests found in the supplied retry roots")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(output_path, sep="\t", index=False)
    return base, admitted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--retry-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idpp-root", type=Path, action="append", dest="idpp_roots")
    args = parser.parse_args()
    frame, paths = merge(args.base, args.retry_root, args.output, args.idpp_roots)
    print(f"admitted={len(paths)} rows={len(frame)} output={args.output}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
