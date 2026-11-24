# Model inputs

The model binaries are intentionally not committed: the Foundation model and
the three compiled/uncompiled epoch-999 seed models occupy about 556 MB in this
working copy. The final package records their expected locations and SHA256
digests in:

- `../manifests/model_manifest.json`
- `../results/provenance/seed123_final_model_provenance.json`
- `../results/provenance/seed456_final_model_provenance.json`
- `../results/provenance/seed789_final_model_provenance.json`

Place the model files listed in those manifests in this directory before running
the local packaging or A100 profile scripts.
