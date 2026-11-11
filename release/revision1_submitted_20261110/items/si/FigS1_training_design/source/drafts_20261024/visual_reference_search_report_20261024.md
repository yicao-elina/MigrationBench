# Independent visual-reference search report — 2026-10-24

Search agent: `01a170dc-1b78-7c81-9316-6fc50c2c4a76`.

## Primary reference hierarchy

The closest full scientific grammar is Behler's tutorial sequence: Cartesian coordinates → per-atom descriptor → element-specific atomic neural network → atomic energies → total energy. See [Behler 2015, Wiley](https://onlinelibrary.wiley.com/doi/full/10.1002/qua.24890) and the accessible [PDF](https://aiichironakano.github.io/cs653/Behler-NNpot-IJQC15.pdf).

For the current figure, the report recommends combining these visual idioms:

- **Coordinates:** OVITO-style Cartesian axis tripod plus colored particle markers. See [OVITO viewport documentation](https://docs.ovito.org/usage/viewports.html) and [particle visual elements](https://ovito.org/docs/current/reference/pipelines/visual_elements/particles.html).
- **Descriptors:** bracketed per-atom vectors/matrix-like rows, with explicit center-atom index and feature notation. DeepPot-SE's descriptor equations are a useful visual reference, but its exact internal construction must not be claimed as universal MACE behavior. See [Training Machine Learning Potentials for Reactive Systems](https://pmc.ncbi.nlm.nih.gov/articles/PMC10923003/).
- **Atomic-species neural network:** layered circular nodes, visible weighted connections, and atomic output `E_i`, following Behler–Parrinello and ANI-style diagrams. See [Behler & Parrinello 2007](https://doi.org/10.1103/PhysRevLett.98.146401) and [ANI-1](https://pmc.ncbi.nlm.nih.gov/articles/PMC5414547/).
- **Total energy:** explicit aggregation of atomic contributions, `E_\mathrm{tot}=\sum_i E_i`. The Behler 2015 reference and [NequIP Fig. 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC9068614/) provide the relevant visual grammar.
- **Message passing:** node = atom, edge = neighborhood relation, message/update = feature transformation, readout = permutation-invariant aggregation. See [MPNN](https://proceedings.mlr.press/v70/gilmer17a.html), [DimeNet](https://arxiv.org/abs/2003.03123), and [MACE](https://arxiv.org/abs/2206.07697). These are conceptual references only; no unsupported MACE internals should be added.

## Design consequence

The final source must retain the four original visual modules as real vector elements. The two new coordinate panels, route-specific training annotations, and three resulting model identities are additions to that scientific pipeline—not replacements for it. Generic placeholder boxes are reserved only for compact labeled route operations where no original scientific visual module exists.
