# FT interlayer self-relaxed Rockfish campaign

This is the parallel completion campaign for the Cr system's missing
FT-600K and FT-MultiT interlayer self-relaxed profiles.

- six existing FT checkpoints: three FT-600K and three FT-MultiT seeds;
- two fixed endpoint systems: bulk (80 atoms) and Cr-in-gap (81 atoms);
- ten images, ASE-IDPP initialization, fixed endpoints, CPU MACE evaluation;
- FIRE, 500 steps, `fmax=0.05 eV/A`, `dt=0.1`, `maxstep=0.2`, `k=0.1`, seed 42;
- observable: `ENDPOINT_ONLY_MLIP_NEB_PROXY`, explicitly not a DFT MEP.

The Slurm array is gated per row by checkpoint SHA256, endpoint atom count and
identity, runtime provenance, finite 10-image profile, optimizer convergence,
endpoint preservation, and iteration-history completeness.  The collector
must pass all 12 rows before these data can enter the grouped 2×2 figure.
