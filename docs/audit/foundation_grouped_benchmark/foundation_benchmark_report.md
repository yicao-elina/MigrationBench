# Foundation-model grouped benchmark: publication-ready audit

## Scope and acceptance

The benchmark used the frozen grouped test set (`267` structures; SHA256 `d49cf4b2ab49fc1e521d314c5828cf5dcaea6853e70cecb2727ffd0678bb27ab`) and the same five-image pathway inputs used by the grouped campaign. Ten foundation checkpoints were evaluated. Equilibrium jobs completed with finite energies and forces for all ten model IDs; the NEB matrix produced `50/50` finite model/path JSON profiles for paths `1-2`, `1-3`, `1-4`, `1-5`, and `1-7`. These are fixed five-image single-point profiles, so an optimizer convergence criterion is not applicable; the acceptance gate is finite energy/force output plus preserved image ordering and endpoint/profile metadata.

Absolute total energies have a model/reference zero-point offset. Therefore the reported equilibrium energy metric is the RMSE after fitting one global per-atom energy offset on the frozen test set; raw total-energy RMSE is retained as a diagnostic and is not interpreted as physical error. Forces are unshifted. MH-1 required the isolated MACE 0.3.16 runtime and `omat_pbe` head; its original MACE 0.3.12 deserialization failure is retained in the job log. It is finite after compatibility repair but remains excluded from the main model ranking pending confirmation of the intended head/reference.

## Main results

The formal raw energy-per-atom RMSE and raw total-energy RMSE are retained in `analysis/equilibrium_split_summary.csv`. The raw per-atom metric is approximately `2.19×10^3 eV/atom` and the raw total-energy metric is approximately `1.8–2.2×10^5 eV`, because the frozen QE reference carries a large absolute energy gauge relative to the foundation checkpoints. These raw values are not meaningful cross-family accuracy measures by themselves. The table below reports the explicitly labelled offset-corrected diagnostic alongside the unshifted force metric; the raw columns must remain in any release and must not be silently replaced.

The independently evaluated grouped validation split is also finite for all ten checkpoints. Offset-corrected validation energy RMSE spans `4.59–6.99 meV/atom`, and force RMSE spans `266.6–296.1 meV/Å`; these validation values are reported separately and are not pooled with the test results.

| model | offset-corrected E RMSE (meV/atom) | F RMSE (meV/Å) | path 1–4 barrier (eV) | status |
|---|---:|---:|---:|---|
| MP-0b3 medium | 21.56 | 1059.30 | 1.503 | usable, high force error |
| MPA0 medium | 17.07 | 646.56 | 1.791 | usable, path barrier high |
| OMAT-0 small | 15.83 | 537.43 | 1.544 | usable |
| OMAT-0 medium | 15.48 | 533.31 | 1.602 | main representative |
| MH-0 | 14.93 | 474.21 | 1.627 | appendix; lowest force RMSE |
| MH-1 | 15.28 | 588.38 | 1.841 | compatibility/head caveat |
| Agnesi medium | 19.90 | 947.76 | 1.467 | appendix |
| Agnesi stress small | 16.93 | 799.91 | 1.307 | appendix |
| Agnesi stress medium | 22.59 | 1025.74 | 1.489 | appendix |
| Agnesi stress large | 18.34 | 955.28 | 1.829 | appendix |

The grouped DFT reference for pathway 1–4 is `0.336050 eV`. All foundation models overestimate this barrier, with absolute errors of approximately `1.17–1.50 eV` among the four main representatives; none reproduces the grouped fine-tuned model-family barrier accuracy. In this completed benchmark MPA0 does not show a kinetic advantage: its path-1–4 error is larger than MP-0b3 and OMAT-0 medium. This is a model-transfer result, not evidence that the grouped `0.16 eV` claim transfers to foundation checkpoints.

## Reviewer-ready discussion

On the frozen provenance-grouped test set, foundation models produced finite predictions across all structures and NEB images, but their dynamical fidelity was substantially weaker than their offset-corrected equilibrium energy errors suggest. The representative foundation models had offset-corrected diagnostic energy RMSE values of 15.28–21.56 meV/atom after removal of a single global energy reference offset, while force RMSE values remained 533–1059 meV/Å. For the headline pathway 1–4, the DFT barrier is 0.336 eV, whereas the four main foundation models predicted 1.503–1.841 eV. Thus, low equilibrium energy RMSE alone does not imply transferable migration barriers under the grouped, trajectory-held-out evaluation.

The appendix models show the same qualitative separation between static and dynamical metrics. MH-0 has the lowest force RMSE among the evaluated checkpoints (474 meV/Å), but still predicts a 1.627 eV path-1–4 barrier. Agnesi stress/density variants span 16.93–22.59 meV/atom in offset-corrected energy RMSE and 800–1026 meV/Å in force RMSE, without a monotonic size trend in the barrier. MH-1 was successfully evaluated only after switching to an isolated MACE 0.3.16 runtime and explicitly selecting `omat_pbe`; because the intended head mapping must be confirmed before making a model-family claim, MH-1 is reported as a compatibility-sensitive appendix result rather than included in the main ranking.

These results support a conservative reviewer response: the grouped split is operational and leakage-audited, but the prior barrier claim is model-family-specific. The foundation benchmark does not support a general statement that the 0.16-eV barrier error survives model replacement. The publishable conclusion is instead that grouped trajectory separation exposes a clear distinction between equilibrium fit quality and migration-path fidelity. Foundation-versus-grouped comparisons must label the foundation energy column as offset-corrected diagnostic, while retaining the raw total-energy metric in the reproducibility tables.

## Reproducibility index

- Campaign root: `/scratch16/pclancy3/yi/foundation_grouped_benchmark_20261001`
- Local mirror: `foundation_grouped_benchmark_20261001/`
- Checkpoint manifest and SHA256: `models.tsv`
- Equilibrium raw outputs: `results/equilibrium/`
- NEB raw profiles: `results/neb/` (50 JSON profiles; a materialized summary copy is `analysis/neb_summary_complete.csv`)
- Tabulated results: `analysis/equilibrium_summary.csv`, `analysis/neb_summary.csv`
- Test/valid split results: `analysis/equilibrium_split_summary.csv`; valid job array `31483778`
- Special representative profiles: `analysis/special_profile_diagnostics.csv` (raw/offset-invariant barrier, image index, force profile, endpoint delta)
- Checkpoint preflight: `results/preflight_manifest.csv`; preflight array `31483607`
- Foundation-versus-grouped main comparison: `analysis/foundation_vs_grouped_comparison.csv`, `figures/foundation_vs_grouped_main_comparison.pdf`
- Figures: `figures/foundation_equilibrium_rmse.pdf`, `figures/foundation_vs_grouped_main_comparison.pdf`, `figures/foundation_appendix_all_path_barriers_complete.pdf`, `figures/path_1-4_energy_profiles.pdf`
- Failed compatibility evidence: Slurm `31481628_5` under MACE 0.3.12; repaired MH-1 run `31482442` under MACE 0.3.16.
