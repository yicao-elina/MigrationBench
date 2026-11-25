# Foundation self-relaxed NEB campaign: frozen acceptance criteria

Campaign ID: `foundation_self_relaxed_4x3_20261024`

This file is frozen before formal Rockfish execution. The attached PDF is used only as the visual reference; it does not override the user request or the scientific gates below.

## Scientific and data acceptance

1. **Scope and identity**
   - Exactly four canonical systems: Cr, Cu, Li, and Na.
   - Exactly three canonical path classes per system: `in_gap`, `gap_to_ql`, and `within_ql`.
   - Exactly the nine Foundation checkpoints in `MODEL_REGISTRY.tsv`; no MPA, fine-tuned, scratch, Blender, or legacy alias may enter the primary panel data.
   - Every model row carries the resolved Rockfish path, SHA-256, model head, checkpoint family, and execution environment.
   - Legacy raw stems are aliases only. Element and path identity must come from the canonical registry, never from a filename stem.

2. **Endpoint-only initialization**
   - Each formal run receives only one initial and one final endpoint file.
   - The runner must construct exactly 10 images with deterministic IDPP interpolation (`seed=42`); intermediate source frames are not passed to the NEB runner.
   - Endpoint atom order, symbols, cell, PBC, and endpoint hashes must match the canonical path registry.
   - Each completed trajectory must contain 10 ordered images and the same atom count and cell/PBC contract.

3. **Model and optimization gates**
   - Model loading must succeed under the recorded environment and requested head.
   - All energies, forces, and coordinates used for a profile must be finite.
   - `optimizer_converged=true`, `final_max_internal_neb_force_eV_A <= 0.05`, and `final_max_neb_residual_eV_A <= 0.05` are required for `CONVERGED` / `PRIMARY_EVIDENCE`.
   - Load failures, non-finite outputs, malformed trajectories, and non-converged runs remain in the failure ledger with their logs and are never silently dropped.
   - MLFF profiles are self-relaxed model observables / diagnostic preconditioners, not DFT ground truth. QE/DFT references, when shown, retain their existing SCF-warning boundary.

4. **Completeness**
   - The target matrix is 4 systems × 3 paths × 9 models = 108 row-level attempts.
   - A publication-ready complete campaign requires all 108 rows to pass the data gates. If a checkpoint remains unavailable or a run remains non-converged, the release is diagnostic/incomplete and the figure must show the status explicitly.
   - Class mean/SD traces may be plotted only from rows that passed the model/data gates; the denominator and excluded-row ledger must be reported.

5. **Provenance and reproducibility**
   - Record campaign, job, array-task, runner, wrapper, environment, endpoint, model, trajectory, and derived-profile hashes.
   - Preserve raw outputs, stderr/stdout, runtime provenance, scheduler records, failure reasons, and the final row-level audit.
   - Fixed-path single-point profiles and self-relaxed NEB profiles are separate observables and must not be merged under one label.

## Frozen visual acceptance

The latest attached reference defines the plotting grammar for the new product:

- One final composite with **4 rows × 3 columns**, rows ordered Cr, Cu, Li, Na; columns ordered `in-gap diffusion`, `gap→QL`, `within-QL`.
- Explicit inter-column gutters; independent complete y-axis and tick labels on every panel; shared x-axis name and shared y-axis name outside the grid.
- Arial first, Helvetica second, then a metrically compatible sans-serif fallback; no grid.
- Compact model/status legend; model colors remain class-consistent and are not reused for unrelated annotations.
- Curves are self-relaxed Foundation-model profiles only, normalized to the first image; markers/uncertainty semantics must be stated in the caption/manifest.
- Failure or incomplete status is explicit and cannot be hidden by interpolation, smoothing, or an empty panel.
- No Blender or molecular rendering layer.
- Export the composite and all 12 independent panels as PNG, SVG, and PDF. The composite is a layout artifact built from the independent panel recipe, not a separate plotting logic.
- Review the composite at full resolution and intended manuscript scale; check clipping, legend collisions, tick legibility, panel-letter order, and consistent axes/layout.

## Independent review gate

The execution agent is not the final certifier. An independent data-review subagent must audit the row-level registry, hashes, status accounting, geometry/topology, and derived profiles. An independent visual-review subagent must inspect the composite and all standalone panels against this contract. Each reviewer returns `PASS` or a numbered feedback list. Any `FAIL` creates a repair/re-review cycle; the release is accepted only after both reviewers return `PASS` and the final manifest records their report hashes.

