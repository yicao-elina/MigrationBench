# Foundation-model MD smoke campaign

This experiment runs the MACE foundation models on the Cr-doped Sb2Te3 input
at 600 K. The CI `--smoke` mode runs two models for 2,000 steps to validate the
end-to-end calculator, trajectory, plotting, and uncertainty-summary path.

The smoke output is a reproducibility check, not evidence for a 200 ps or 1 ns
scientific claim. Production runs require new manifests, longer equal-length
trajectories, model-specific burn-in, and the steady-state uncertainty audit.
