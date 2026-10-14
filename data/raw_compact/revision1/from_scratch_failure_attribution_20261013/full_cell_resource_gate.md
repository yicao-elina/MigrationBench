# DFT wave-1 resource gate — 2026-10-13

Wave 1 was frozen at 20 candidates and submitted as the Rockfish array
`31917081_[0-19%2]`. The initial array failed before QE for an Intel-MPI
launcher mismatch (`-bootstrap slurm`); that operational attempt is retained
under `operational_failures_31917081/`. The corrected launcher passed separate
2-task and 96-task MPI smoke tests. Two retry tasks (`31917088_[0-1%2]`) and a
corrected continuation (`31917097_[2-19%2]`) were then started.

The first two corrected QE tasks reached QE input/resource initialization, but
the resource estimate is not feasible for the requested two-node allocation:

- candidate: `dftcand_0012` and independently `dftcand_0071`;
- QE: 7.2, non-magnetic SCF, PBE + DFT-D3, 70/280 Ry, 2×2×2 k points;
- cell: 2050 atoms and 31,900 electrons;
- Kohn–Sham states: 19,140;
- estimated static dynamic RAM per process: >206.63 GB;
- estimated maximum dynamic RAM per process: >337.28 GB;
- estimated total dynamic RAM: >32,379.25 GB.

The retry and remaining array were cancelled after this preflight. No DFT
energy or force passed the scientific gate, no candidate is marked as a DFT
result, and no wave 2 was submitted.

This is a computational feasibility blocker, not evidence for or against the
force-error mechanism. Continuing requires an explicit method/resource choice:
either a resource-provisioned full-cell calculation, or a separately validated
gamma-only/reduced-cost or local-fragment protocol with its own method label
and calibration against a tractable reference. The current campaign must not
silently change any of those choices.
