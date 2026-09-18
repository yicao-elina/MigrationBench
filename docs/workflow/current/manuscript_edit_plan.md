# Manuscript Edit Plan

## Text Edits

| Location | Current problem | Upstream | New target | Status |
|---|---|---|---|---|
| Abstract | Claims "generalizable standard" too strongly | N14 | Candidate framework demonstrated on Cr-doped Sb2Te3; broader validation as outlook | done-local draft |
| Introduction | Research question implies established generality | N14 | Phrase as benchmark question and scope statement | done-local draft |
| Introduction delta paragraph | Missing delta from arXiv:2509.00090 | N14 | Add exact novelty delta: MD transport, NEB stability, SHAP framework, sliding analysis | done-local draft |
| §3.1 transport | FT-600K naive-FT MD protocol invalid | N11 | Quarantine old value; insert corrected values once rerun finishes | partial |
| §3.2 NEB reference | 0.34 vs ~0.3 eV inconsistency | N2, N6 | Use one path id and one barrier definition; currently 1-4 candidate 0.336050 eV with caveat | partial |
| §3.2 fixed vs self-NEB | Foundation "unacceptable" vs "exceptional" conflict | N6, N8 | Explain that Fig. 3 fixed-path scoring and Fig. S3 self-relaxed NEB are different observables; recheck the `~0.7 eV` and `0.41 eV` values by provenance/rerun before retaining either qualitative claim | protocol distinction accepted; numeric audit pending |
| §3.2 deep penetration | "by chance" unsupported | N9, N6 | Replace with seed-stability decision rule and measured std | pending data |
| §3.2 FT-600K accuracy | 0.16 eV could be overlap/seed artifact | N7, N9 | Checkpoint-scoped SOAP/RMSD audit passed; retain only if grouped-seed reruns also pass | leakage passed; seed evidence pending |
| SI DFT Methods | Claims all calculations used 100/400 Ry, 4x4x1, and SOC; 377 audited inputs form 12 identities and none explicitly enables SOC | N24 | Replace universal claim with actual per-result calculator identities after cutoff, k-point, and spin decisions; explicit SOC is paused by user instruction | blocked on final protocol wording |
| §3.4 latent analysis | t-SNE/PHATE overinterpreted | N12 | "Consistent with" language; original-space caveat | done-local draft |
| §3.5 SHAP | Mechanistic claims about MACE from surrogate | N13 | Scope to surrogate sensitivity unless perturbation confirms | partial |
| Conclusion | Broad principles stated too strongly | N14, N8 | Hypotheses and future cross-system validation | done-local draft |
| Data/code availability | Repo stubs and missing SHAP code | N13 | State actual scripts, dataset schema, provenance manifest | partial |

## Figure And Table Edits

| Asset | Current problem | Upstream | Intended replacement | Status |
|---|---|---|---|---|
| Fig. 2 transport | Naive FT protocol asymmetry; uncertainty weak | N11 | Block-averaged transport with corrected FT-600K rerun or hatched quarantine | partial |
| Fig. 3 / NEB panel | Mixed fixed-path and self-NEB interpretation | N6, N8 | Side-by-side fixed-path, self-relaxed path, and DFT-refined NEB results | pending |
| Fig. S3 | Self-relaxed NEB is a different observable from Fig. 3 fixed-path scoring; recovered 82-atom Cr2 run also mixed DFT endpoint energies with MACE internal-image energies and does not losslessly reproduce 0.41 eV | N3, N6 | Reproduce the `0.41 eV` value from original provenance or replace with a correctly rerun, lossless self-NEB/protocol panel | protocol distinction accepted; numeric audit pending |
| SI DFT NEB table | Needs converged source/gate columns and calculator identity | N2, N6, N24 | Add path id, image count, barrier, movable-image max error, convergence status, source path, cutoffs, k points, smearing, spin/SOC, and pseudopotential hashes | partial |
| Table S1 | Single-seed and old split | N9 | Mean +/- std over grouped-split seeds | pending |
| Fig. 5 latent | Embedding-space silhouette invalid | N12 | Original-space and sensitivity panel | done-local draft |
| Fig. 6 SHAP | Surrogate R2/provenance incomplete | N13 | R2 table plus corrected feature labels; perturbation result if available | partial |
| New SI overlap figure | Reviewer A2 asks nearest-neighbor evidence | N7 | SOAP/RMSD empirical nearest-neighbor distance distributions with thresholds, split counts, and scope limitation | generated and validated |
| New SI workflow/provenance figure | Reviewer trust issue spans many points | N0-N8 | Data provenance DAG from raw logs to response claims | current deliverable |

## Figure Rebuild Rule

No figure should be manually patched from screenshots. Every revised visual should consume either:

- `Revision1/data_processed/*.csv/json`
- `Revision1/data_processed/cluster/<job>/...`
- explicit audited tables under `Revision1/tables/`

and should have the script path recorded in the caption note or SI provenance table.
