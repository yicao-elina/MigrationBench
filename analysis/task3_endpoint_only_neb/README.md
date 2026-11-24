# Task 3 — endpoint-only MACE NEB preflight

This namespace is isolated from the existing campaigns. The collaborator DFT
tree remains read-only at
`/scratch4/pclancy3/joshitha/vdW-sratch/sb2te3` on Rockfish.

## Current gate status

The audit covers 10 NEB directories: four Cu, three Li, and three Na. All
declared 10-image paths have finite parsed energies/gradients in their saved
`.path` artifact and endpoint/path coordinate identity within 0.009–0.050 Å.
However, none has an explicit path-convergence marker. Nine completed logs
retain a final `SCF convergence NOT achieved on image ...` warning; one path has
no completed `neb*.out` selected. Therefore **0/10 pass the DFT gate** and no
formal MACE NEB was launched.

`JOB DONE` and the finite checkpoint are retained as observations only. They do
not promote a DFT path to a reference.

## Endpoint-only leakage boundary

Only `pw_1.in` and `pw_2.in` provide local endpoint inputs. The saved
`sb2te3.path`, `.xyz`, `.axsf`, and checkpoint artifacts are inspected only for
the DFT gate and identity audit. They are never used to initialize linear or
IDPP paths. See `endpoint_only_leakage_audit.csv` and
`endpoint_only_provenance.json`.

## Geometry preflight

Minimum-image linear interpolation was generated for all 10 endpoint pairs,
using the declared 10 images, fixed cell, PBC, species order, and dopant
identity. A bounded one-step IDPP smoke was run for the representative
`2L-Cu__neb-gap-diffuse` pair. All 11 generated preflight bands passed the
finite-geometry and 0.70 Å minimum-separation screen. This is not evidence of
MACE force stability or a barrier.

The figure `endpoint_initialization_smoke.{png,svg,pdf}` is a preflight
visualization only. It is deliberately not presented as a DFT-vs-MACE profile.

## Formal-stage outputs

The required MACE CSVs are present with explicit `blocked_dft_gate` rows or
empty image metrics. `formal_gate_smoke.json` records the planned fixed-cell
protocol and confirms that no formal command was executed. The failure/retry
ledger retains every path and the exact gate reason; no failure was dropped or
imputed.

## Reproduction

```bash
python build_endpoint_only_manifest.py --snapshot remote_snapshot
python run_interpolation_smoke.py --idpp-steps 1
python build_blocked_outputs.py
python plot_smoke.py
```

The snapshot was made from a single read-only SSH tar stream. `remote_sha256.tsv`
contains the Rockfish SHA256 records used in the manifest. The manifest and all
endpoint inputs carry source hashes.
