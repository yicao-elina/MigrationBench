# Table-backed figure evidence packages

These are additive Plot Atlas renders created from analyses that already had
source tables/manifests but no complete figure package. They are not copied
over existing campaign outputs. Each directory contains:

`processed table -> Plot Atlas render script -> PNG/SVG/PDF -> SHA256 manifest -> caption/results/discussion`

| Package | Article/SI role | Scientific boundary |
|---|---|---|
| `MD-SENSITIVITY-48RUN` | Integrator/thermostat ablation and dynamic-stability mechanism | 48 complete short handoffs; not transport or DFT validation |
| `MACE-CANDIDATE-EVAL` | Candidate selection and evaluation coverage audit | Runtime coverage; not a model-accuracy ranking |
| `DFT-PHASE3-INPUT-AUDIT` | Reduced-cell DFT-validation boundary | MACE input-force audit; no accepted QE target |
| `DFT-SINGLE-POINT-GATE` | Independent DFT coverage/acceptance boundary | 0/20 accepted; no scientific DFT comparison |
| `FOUNDATION-PATH-GATE` | Foundation path validation gate | Gate-only; numerical profile panels remain blocked |
| `NEB-TOPOLOGY-AUDIT` | Historical path topology and hidden-basin audit | Reconstruction/historical; endpoint stability and barrier acceptance remain open |
| `DFT-RECOVERY-RECONSTRUCTION` | DFT recovery and MACE force-validation boundary | Reconstruction-only; source scope is explicitly limited |

The exact generator is
`../../scripts/render_table_backed_packages.py`, executed with the Plot Atlas
environment at `/Users/alina/Documents/Codex/plot_library/.venv/bin/python`.
