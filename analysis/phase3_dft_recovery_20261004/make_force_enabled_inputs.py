#!/usr/bin/env python3
"""Derive force-printing QE inputs without changing geometry or DFT settings."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "qe_inputs_reconstructed"
OUTPUT_ROOT = ROOT / "qe_inputs_force"
CASES = [
    "normal_461800_initial",
    "bounded_multiT_462000_step464000",
    "runaway_foundation_461800_step462120",
    "runaway_scratch_461800_step462060",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive(text: str) -> str:
    if "tprnfor" in text.lower():
        return text
    lines = text.splitlines()
    control_end = next(i for i, line in enumerate(lines) if line.strip() == "/")
    lines.insert(control_end, "  tprnfor = .true.")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "package": "phase3_dft_recovery_20261004",
        "derivation": "canonical scf.in plus CONTROL tprnfor=.true.; geometry/cell/DFT settings unchanged",
        "cases": {},
    }
    for case in CASES:
        source_dir = SOURCE_ROOT / case
        output_dir = OUTPUT_ROOT / case
        output_dir.mkdir(parents=True, exist_ok=True)
        canonical = source_dir / "scf.in"
        force_input = output_dir / "scf.in"
        force_input.write_text(derive(canonical.read_text()))
        mapping = json.loads((source_dir / "mapping.json").read_text())
        metadata = dict(mapping)
        metadata.update(
            {
                "canonical_input_sha256": sha256(canonical),
                "input_sha256": sha256(force_input),
                "force_output_enabled": True,
                "derived_output_flag": "tprnfor = .true.",
                "geometry_source": str(canonical.relative_to(ROOT)),
                "scope": "RECONSTRUCTION / REPRODUCED_SCOPE_LIMITED",
            }
        )
        (output_dir / "mapping.json").write_text(json.dumps(metadata, indent=2) + "\n")
        manifest["cases"][case] = metadata
    (OUTPUT_ROOT / "reconstruction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(CASES)} force-enabled inputs under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
