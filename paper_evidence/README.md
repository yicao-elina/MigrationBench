# Paper evidence workflow

This directory is the paper-facing control plane for MigrationBench evidence.
It separates three states that should not be mixed:

1. `analysis/.../exploratory_energy_audit/` contains reproducible candidate
   figures, tables, model predictions, raw snapshots, and audit manifests.
2. `paper_evidence/candidates/FIGURE_CANDIDATES.tsv` is the human-readable
   candidate gallery. A row is a proposal, not a publication decision.
3. `paper_evidence/releases/` contains only artifacts explicitly selected for a
   draft or release. A release is an Overleaf-friendly package with
   `figures/`, `tables/`, `notes/`, `tex/`, and `manifest/`.

## What to inspect in VSCode

Open the repository root and use these paths as the main navigation points:

- Candidate registry: `paper_evidence/candidates/FIGURE_CANDIDATES.tsv`
- Manual selection file: `paper_evidence/release_selection.tsv`
- Current Figure 1 candidate:
  `analysis/task3_endpoint_only_neb/exploratory_energy_audit/figure1_task_type_matrix_v3/`
- Current bidirectional gap-to-QL candidate:
  `analysis/task3_endpoint_only_neb/exploratory_energy_audit/gap2ql_bidirectional_20261016/`
- Current Cr versus Cu/Li/Na audit:
  `analysis/task3_endpoint_only_neb/exploratory_energy_audit/cr_vs_culinas_systematic_20261015/`
- Rockfish raw inputs, outputs, model evaluations, and provenance:
  `analysis/task3_endpoint_only_neb/exploratory_energy_audit/rockfish_cr_neb_20261013/`
- Claim-to-artifact template: `paper_evidence/templates/CLAIM_EVIDENCE_MATRIX.tsv`
- Editable paragraph template: `paper_evidence/templates/section_bidirectional_neb.tex`

The current candidates are intentionally marked `PROVENANCE_PARTIAL`. In
particular, the Rockfish Cr NEB traces are completed diagnostic paths with
retained SCF warnings; they are useful for audit and hypothesis generation, but
are not final converged DFT barriers until the scientific gate is closed.

## Candidate-to-release workflow

1. Inspect the candidate directory and its manifest, raw/source inputs, and
   processing script.
2. Edit only the `selected` column in `release_selection.tsv` (`yes` or `no`).
   Keep the candidate registry immutable except when adding a new version.
3. Build a release package:

   ```bash
   python paper_evidence/scripts/build_release_package.py \
     --release-id draft-20261016 \
     --selection paper_evidence/release_selection.tsv \
     --out paper_evidence/releases/draft-20261016
   ```

4. Review the generated `manifest/release_manifest.json`, copied tables,
   generated `tex/evidence_bundle.tex`, and status notes before adding anything
   to the manuscript.
5. Manually edit the generated TeX section if needed, but preserve the source
   candidate, its status, and the release manifest. Manual edits belong in the
   release package, not in the generated analysis directory.
6. Fill one row per paper claim in
   `paper_evidence/templates/CLAIM_EVIDENCE_MATRIX.tsv`, then copy the filled
   matrix into the release `manifest/` directory.

The release builder copies compact figures, tables, notes, and provenance. It
does not copy raw containers, full model checkpoints, node-local scratch, or
unbounded logs. Those remain in their source campaign and are referenced by
hash/path in the manifest.

## Overleaf handoff

Upload the contents of one release directory, preserving the relative layout:

```text
figures/*.pdf|*.png|*.svg
tables/*
notes/*
tex/evidence_bundle.tex
manifest/release_manifest.json
manifest/CLAIM_EVIDENCE_MATRIX.tsv
```

In the Overleaf manuscript, use:

```tex
\input{tex/evidence_bundle.tex}
```

The generated TeX uses paths relative to the manuscript root, so `figures/`
should remain beside `tex/`. Do not cite a moving branch or an untagged
candidate directory in the paper. Record the Git commit, release ID,
figure/table hashes, source-input hashes, and the Overleaf Git commit or
version in the release manifest.

## Git discipline

Use a review branch for paper-facing selection and keep raw campaign evidence
immutable. Commit code, manifests, compact processed tables, captions, and
selected release packages together. Before an Overleaf push: pull first,
resolve conflicts locally, compile, inspect the PDF, then push. Never overwrite
an accepted release; create `draft-...-v2` or a new release ID instead.
