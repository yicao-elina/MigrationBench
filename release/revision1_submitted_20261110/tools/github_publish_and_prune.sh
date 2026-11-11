#!/usr/bin/env bash
# Publish the Revision 1 release branch and prune superseded GitHub branches.
# Run from the repository root:  bash release/revision1_submitted_20261110/tools/github_publish_and_prune.sh
# Every deleted branch is first preserved as an annotated archive/* tag (created 2026-11-10).
set -euo pipefail

RELEASE=release/revision1-submitted-20261110
git push -u origin "$RELEASE"

# Remote-only branches with unique content: keep as tags, then delete the branch.
ARCHIVE_TAGS=(
  archive/remote-codex-dft-mace47-evidence-package-20261023
  archive/remote-codex-document-mlip-639-selection-provenance
  archive/remote-codex-multimodel-639-evidence-pr
  archive/remote-paper-cr-neb-evidence-workflow-20261016
  archive/remote-revert-16-codex-paper-figures-repro-release
)
git push origin "${ARCHIVE_TAGS[@]}"

PRUNE=(
  codex/dft-mace47-evidence-package-20261023
  codex/document-mlip-639-selection-provenance
  codex/multimodel-639-evidence-pr
  paper/cr-neb-evidence-workflow-20261016
  revert-16-codex/paper-figures-repro-release
  dev
  archive/main-before-paper-repro-2026-09-17
  chore/skipjack-production-environment
  fix/qe-mlff-monitoring-and-neb-controls
)
git push origin --delete "${PRUNE[@]}"
git fetch --prune origin
echo "Done. Next: open a PR from $RELEASE into main."
