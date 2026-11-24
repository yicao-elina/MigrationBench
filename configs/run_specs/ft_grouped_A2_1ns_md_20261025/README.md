# FT grouped A2 exact-model 1 ns MD campaign

This is an additive campaign for the six exact `*_stagetwo_compiled.model`
files supplied by the user. The pre-existing trajectory audit found no
trajectory whose model hash matches any of those six files. Older grouped
20 ps artifacts and the older Multi-T 86-atom 1 ns artifact remain historical
evidence and are not used as continuation checkpoints.

The common input is the established 2050-atom, three-dimensional-PBC state at
absolute step 200,000 (200 ps), with finite momenta. Each exact model first
passes an isolated 20 ps smoke gate, then a dependent continuation advances the
same model to absolute step 1,000,000 (1 ns). The new model-generated segment
is therefore 800 ps beyond the common 200 ps state, consistent with the
existing MigrationBench 1 ns convention.

The continuation wrapper preserves the raw smoke and run directories,
records model/input hashes and the Langevin RNG state, and automatically
submits a dependent segment only after a walltime checkpoint. Scientific,
provenance, nonfinite, or geometry failures are not auto-restarted.
