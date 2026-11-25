# OpenMPI bootstrap repair — 2026-10-16

Jobs `32074582_[0-2]`, `32075474_[0-2]`, `32075144_[0-5]`, and `32075596`
failed before training or model auditing. Their stderr records the same
environment failure: `openmpi/4.1.6` could not be loaded while the default
`openmpi/3.1.6` module was active. No model, checkpoint, loss, validation, or
gate result from those launches is accepted.

The repair was tested in a Rockfish batch-equivalent shell. It restores the
default module collection, unloads `openmpi/3.1.6`, and then loads
`openmpi/4.1.6`. The repaired submissions are additive and preserve the failed
logs:

| scope | failed job | repaired job | script SHA256 |
|---|---|---|---|
| Multi-T scratch | `32074582_[0-2]` | `32114528_[0-2]` | `14c60303d8343344989ed7d590bbe5e2bf9e15c1e26eda373ba491244e399bca` |
| grouped 600K scratch | `32075474_[0-2]` | `32114529_[0-2]` | `566bfc6b02fb36e04fe2c44fe699e54a044a9f7ada2dc4a3d3e5b3f57da6207e` |
| grouped 5% FT | `32075144_[0-5]` | `32114530_[0-5]` | `d74ba9f06ef5c2da372a08fac1aaf06a615a63adc4b3a04f7566c93226ec69f3` |
| compiled-model gate | `32075596` | `32114531` | `6a6ca7b38fd4bd44572fa94d7e70c55ef9dfe20ee69f55c2e8f4fd2fbbb3eefe` |

At the submission check, all four repaired jobs were `PENDING`; no scientific
acceptance was inferred from submission or scheduler state.
