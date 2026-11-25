# Historical grouped Foundation-vs-NEB audit: 81-atom branch (2026-10-03)

The five 81-atom fixed-path profiles are finite and have retained `JOB DONE` markers, but every QE output contains an explicit `SCF convergence NOT achieved` warning. They are therefore stress-test/exploratory references, not formal converged DFT reference answers.

| Path | Images in `.dat` | `.dat` barrier (eV) | NEB-output barrier (eV) | Difference (eV) | JOB DONE | SCF warning |
|---|---:|---:|---:|---:|---|---|
| neb_1 | 10 | 1.034728 | 1.034729 | 6.550e-08 | True | True |
| neb_2 | 10 | 1.561262 | 1.561262 | 1.377e-09 | True | True |
| neb_3 | 10 | 1.691053 | 1.691053 | -1.029e-08 | True | True |
| neb_4 | 10 | 2.063858 | 2.063858 | 2.476e-09 | True | True |
| neb_5 | 10 | 1.414868 | 1.414868 | 7.599e-08 | True | True |

Foundation profile audit: 50/50 JSON profiles found; 50/50 finite; 50/50 satisfy the stored relative-energy identity.

The old `SCF convergence warning` label is directionally correct, but it must propagate into every barrier comparison and figure caption: these are diagnostic stress-test references, not uniformly converged DFT barriers.
