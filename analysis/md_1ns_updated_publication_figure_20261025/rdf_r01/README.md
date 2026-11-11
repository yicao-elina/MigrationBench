# RDF-R01 provenance and computation note

## Scope

Panel E of the updated six-panel MD figure is a Cr–Cr radial distribution function (RDF) computed from the raw extended-XYZ trajectories. The source set was resolved SQL-first and then completed from Rockfish when the registry did not contain the raw coordinates.

## SQL-first lookup

- Registry: `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/migrationbench_asset_registry_20261025/migrationbench_asset_registry.sqlite`
- The registry identifies the `md_1ns` task, the merged MD asset, the accepted FT 1 ns family boundary, the current Foundation job, and the Scratch failure-mode boundary.
- The registry does not contain the raw coordinate streams required for an RDF calculation, so the raw trajectory paths were resolved on Rockfish and recorded verbatim in `rdf_input_manifest.tsv`.

## Rockfish source and method

- Host alias: `rockfish`; shared calculation root: `/scratch16/pclancy3/yi/rdf_r01`
- Input trajectories are the exact paths in `rdf_input_manifest.tsv`.
- FT-600K and FT-MultiT: three seeds; frames below step 220000 were excluded as equilibration. The available segments were de-duplicated by the recorded `step` field.
- Foundation: current `Foundation-OMAT-zero-shot`, including the continuation/retry segments through step 1000000; kept as a provisional comparator in the figure.
- Scratch-600K and Scratch-MultiT: current smoke/failure-mode trajectories only; they are shown structurally in panel E but are not interpreted as 1 ns results.
- Cr atoms were selected from each frame. Pair distances use the triclinic cell in the extxyz `Lattice` field and the minimum-image convention. The histogram uses `dr=0.1 Å`, `rmax=12 Å`, and every twentieth eligible frame; panel E displays `r≤10 Å`.
- The RDF is normalized as the Cr–Cr neighbor count per central Cr divided by the shell volume and the mean Cr number density. Curves are run means; bands are sample standard deviations across available runs.
- Pressure stability uses the same raw headers: (P=-\mathrm{tr}(\sigma)/3), with the standard stress sign convention, converted from eV Å(^{-3}) to GPa using 160.21766208. Pressure curves use the production segment from step 220000 onward and are aggregated by run.

## Validation snapshot

All 13 run entries completed with `invalid_frames=0`. The stable FT runs each contribute 196 sampled frames from step 220000 through approximately step 1000000. Foundation contributes 196 sampled frames through step 1000000. Scratch contributes only 1–2 sampled frames per smoke trajectory because those files are short failure-mode traces.

## Reproduction files

- `compute_rdf.cpp`: compiled and run on Rockfish with `g++ -O3 -std=c++17`.
- `rdf_input_manifest.tsv`: SQL/Rockfish-resolved raw input paths and equilibration cutoffs.
- `rdf_runs.csv`: run-level RDF values and coverage metadata.
- `rdf_groups.csv`: group mean and run-to-run standard deviation used by the renderer.
- `rdf_meta.tsv`: frame counts, step coverage, and invalid-frame audit.
- `pressure_runs.csv`: sampled hydrostatic pressure from the same raw headers, used for panel B.
- `RDF_SHA256SUMS.txt`: hashes for the computed RDF tables and metadata.
