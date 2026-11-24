# Scratch-600K migration campaign R01

This compact source note records the campaign boundary used by R05.

- canonical model family: Scratch-600K-Lmax1;
- training seeds: 123, 234, 345;
- Cr system: 81 atoms and 10 images per path;
- paths: `in-gap`, `gap→QL`, and `within-QL`;
- fixed path: 9/9 finite;
- self-relaxed endpoint-only IDPP: 7/9 converged primary evidence;
- diagnostic-only self cases: seed123 × within-QL and seed234 × in-gap;
- jobs: smoke `32791406`, fixed `32791432`, self-relaxed `32791433`.

The full remote trajectory/iteration inventory remains pointer-only. Model,
path, endpoint, and source hashes are preserved in the neighboring campaign
manifest and `../provenance/POINTER_ONLY_INPUTS.tsv`.
