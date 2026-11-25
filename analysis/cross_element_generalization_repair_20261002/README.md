# Isolated cross-element generalization / checkpoint-repair test

This namespace is independent of all historical model files and campaigns.
Original checkpoints are referenced read-only from `analysis/task2_cross_element/models/` and are never overwritten.

## Current diagnosis

- FT-600K is a TorchScript checkpoint whose embedded `atomic_numbers` are exactly `[24, 51, 52]` (`Cr, Sb, Te`). Adding Cu/Li/Na to a text list cannot repair it: the element embeddings and associated model parameters are absent. A real repair requires a new fine-tuning run initialized from the broad Foundation model, with the full target element roster and replay/anchor data for the retained elements.
- FT-MultiT currently fails in the local CPU deployment before element probing. Its failure is a CUDA/backend serialization/runtime issue, not yet an element-support result. The GPU smoke script below will load the original checkpoint on a compatible GPU node and record its actual roster before running probes.

## Safe repair strategy

1. Audit original checkpoint metadata and hashes.
2. Run FT-MultiT on a compatible CUDA environment without editing the checkpoint.
3. Build a new FT-600K broad-roster repair from Foundation + Cr/Sb/Te fine-tuning data + replay/anchor data for the elements whose transfer must be retained.
4. Save the repaired checkpoint under a new path and require the six-element smoke gate before any path evaluation.

The repair is deliberately not implemented as an in-place mutation of a TorchScript checkpoint. Such a mutation would fabricate missing element embeddings and would not be scientifically defensible.
