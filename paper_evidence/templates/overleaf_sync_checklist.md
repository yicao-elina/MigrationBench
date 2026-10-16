# Overleaf sync checklist

- [ ] Choose a new release ID; do not overwrite an accepted release.
- [ ] Pull the current Overleaf Git state before copying or pushing files.
- [ ] Build the release package from `release_selection.tsv`.
- [ ] Review `manifest/release_manifest.json` and all SHA256 values.
- [ ] Review the claim-to-artifact matrix; no `UNVERIFIED` claim may be written
      as a final result.
- [ ] Confirm every figure has a caption, label, source data, processing script,
      and scientific status.
- [ ] Confirm `tex/evidence_bundle.tex` compiles with the manuscript root.
- [ ] Compile the full manuscript and inspect the PDF visually.
- [ ] Record the manuscript/Overleaf commit or version in the release manifest.
- [ ] Commit the release package and the manuscript-facing text together on a
      review branch.
- [ ] Only after review, merge or push to the working Overleaf branch.
