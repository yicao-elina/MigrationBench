# 15-model MD figure R05 change note

R05 changes only panel D of R04. D now shows three species-resolved coordination numbers, Cr–Sb, Cr–Te and Sb–Te, as a four-cluster bar chart. Each cluster is a model-class aggregation with mean ± sample standard deviation across its internal model points. Foundation contains the three Foundation checkpoints; FT-600K and FT-MultiT contain their three seeds; Scratch pools the three Scratch-600K and three Scratch-MultiT seeds into one six-point class.

The partial RDFs were recomputed from the same raw Rockfish extended-XYZ trajectories with triclinic minimum-image distances, `dr=0.05 Å`, `rmax=8 Å`, and every twentieth eligible frame. Pair-specific cutoffs are fixed once from the first minima of the pooled stable-reference partial RDFs: Cr–Sb 3.2 Å, Cr–Te 4.1 Å, and Sb–Te 4.2 Å. CN is reported as neighbors per center atom using the partner-species number density.

Panels A–C, E and F, the 15-model source matrix, status classification, and established model palette are unchanged. The 15-model matrix remains four target-reaching trajectories (three Foundation plus FT600K seed234) and eleven scientific-threshold failures. The partial-RDF audit has `invalid_frames=0` for all 15 models; the original MD15 series/MSD/VACF/RDF audit also remains finite with `invalid_frames=0`. The RDF 0–2 Å correction is retained in panel E.
