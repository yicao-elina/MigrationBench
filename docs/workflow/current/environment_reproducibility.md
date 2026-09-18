# Runtime Environment And Reproducibility

## Rockfish Reference Environment

The production reference was observed directly on Rockfish on 2026-09-13 after
`conda activate mace`:

| Component | Version |
|---|---|
| Python | 3.9.22 |
| ASE | 3.25.0 |
| MACE | 0.3.12 |
| NumPy | 2.0.2 |
| SciPy | 1.13.1 |
| pandas | 2.3.1 |
| matplotlib | 3.9.4 |
| scikit-learn | 1.6.1 |
| PyTorch | 2.7.0+cu118 |
| pytest | 8.4.2 |
| PyYAML | 6.0.2 |
| Quantum ESPRESSO module | `qe/7.3.1-cpu` |

The environment currently contains a real metadata drift: imported pandas
reports `2.3.1` from its package code, while `importlib.metadata` reports
distribution version `2.2.3`. Runtime records preserve both values plus the
module-file hash. The table above uses the imported implementation version.

`environment.yml` reproduces the Python-level dependencies. CUDA-enabled
PyTorch and the site-provided QE module remain platform-specific dependencies;
their exact runtime identity is recorded per job rather than pretending they
are portable Conda packages.

## Per-Job Runtime Identity

`capture_runtime_provenance.py` creates `runtime_provenance.json` before the
scientific executable starts. The record contains:

- Python implementation, version, and executable;
- distribution and imported-module versions of the analysis and MACE
  dependencies, including module paths and hashes;
- Conda, module, OpenMP, CUDA, host, and Slurm resource variables;
- resolved scientific executable paths and SHA-256 hashes;
- hashes and byte counts for every declared input and manifest;
- the provenance-writer's own hash.

The runtime record is immutable evidence for a run. It supplements, rather
than replaces, the calculator identity and scientific branch manifest. A result
must not be pooled with another result when either the scientific calculator
identity or a consequential runtime component differs without an explicit
sensitivity check.

## Local Test Note

The authoritative regression environment is Rockfish `mace`. Local environments
on the workstation are heterogeneous and are not silently treated as equivalent.
The current code passed all 70 tests in Rockfish Slurm job `30852079`, including
the fixed-code-root launcher contract and periodic path-repair smoke gates. The
workstation `ase` Conda environment passed all 53 dependency-compatible
`unittest` cases; local Python 3.14 lacks ASE. Rockfish remains authoritative
because it is the production scientific environment.

Every production launcher now exports `MIGRATIONBENCH_CODE_ROOT` explicitly.
This prevents a job submitted from an arbitrary login directory or automation
from resolving its provenance helper through a different `SLURM_SUBMIT_DIR`.
Runs submitted before this instrumentation require a later provenance-complete
continuation or repeat before they can pass the manuscript release gate.

Rockfish job `30852080` independently loaded `qe/7.3.1-cpu` and the MACE
environment, then verified hash-bound identities for Python, `pw.x`, and
`neb.x`. Initial SCF warmups and charge-density-based SCF continuations now use
the same runtime capture before invoking `pw.x`; a continuation additionally
hashes the copied parent charge density and XML schema.

The old direct submission scripts that accepted only an input path, without a
branch manifest and monitor gate, now fail closed with exit code 64. They remain
in the repository solely as historical records. New work must use the modern
candidate, warmup, relaxation, or NEB launchers.
