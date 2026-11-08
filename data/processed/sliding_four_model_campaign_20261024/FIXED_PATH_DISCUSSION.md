# Fig. 4b/c fixed-path discussion (Stage 1)

The Stage 1 campaign evaluates MACE energies and forces on the ten retained DFT/QE geometries for pristine and Cr-in-gap Sb2Te3. The observable is a fixed-path relative-energy diagnostic: each profile is referenced to its own image 0, and no MACE geometry optimization or self-relaxed NEB convergence is used. The retained DFT reference is the no-CI ten-image QE path and remains classified as `COMPLETED_BUT_SCF_WARNING`. FR pseudopotential filenames are present in the archived inputs, but explicit noncollinear/spin-orbit flags were not independently confirmed in the retained input/output; this is therefore not upgraded to a verified SOC statement.

## pristine Sb2Te3 bilayer sliding

The DFT reference reaches 1.654 eV at image 6 and has an image-9 endpoint drift of 0.014 eV relative to image 0.
Among the current checkpoint evaluations, the smallest ten-point RMSE is obtained by MPA-0 medium (0.546 eV); its fixed-path peak is 1.245 eV at image 9, with endpoint drift 1.245 eV and Spearman profile correlation 0.36.
The three Foundation entries are checkpoint variants rather than random-seed replicates and are reported individually: OMAT-0 medium: peak 1.372 eV, RMSE 0.565 eV; MPA-0 medium: peak 1.245 eV, RMSE 0.546 eV; MP-0b3 medium: peak 1.049 eV, RMSE 0.611 eV.
FT-600K gives a training-seed peak mean +/- SD of 1.156 +/- 0.345 eV and RMSE mean +/- SD of 0.719 +/- 0.100 eV (n=3).
FT-MultiT gives a training-seed peak mean +/- SD of 1.101 +/- 0.690 eV and RMSE mean +/- SD of 0.719 +/- 0.029 eV (n=3).
Scratch-600K gives a training-seed peak mean +/- SD of 1.204 +/- 0.631 eV and RMSE mean +/- SD of 0.816 +/- 0.064 eV (n=3).
Scratch-MultiT gives a training-seed peak mean +/- SD of 1.419 +/- 0.217 eV and RMSE mean +/- SD of 0.730 +/- 0.108 eV (n=3).

## Cr-in-gap Sb2Te3 bilayer sliding

The DFT reference reaches 2.571 eV at image 6 and has an image-9 endpoint drift of -0.020 eV relative to image 0.
Among the current checkpoint evaluations, the smallest ten-point RMSE is obtained by MPA-0 medium (0.972 eV); its fixed-path peak is 1.981 eV at image 9, with endpoint drift 1.981 eV and Spearman profile correlation 0.39.
The three Foundation entries are checkpoint variants rather than random-seed replicates and are reported individually: OMAT-0 medium: peak 1.914 eV, RMSE 0.973 eV; MPA-0 medium: peak 1.981 eV, RMSE 0.972 eV; MP-0b3 medium: peak 1.933 eV, RMSE 0.990 eV.
FT-600K gives a training-seed peak mean +/- SD of 2.158 +/- 0.253 eV and RMSE mean +/- SD of 1.205 +/- 0.071 eV (n=3).
FT-MultiT gives a training-seed peak mean +/- SD of 1.975 +/- 0.497 eV and RMSE mean +/- SD of 1.104 +/- 0.054 eV (n=3).
Scratch-600K gives a training-seed peak mean +/- SD of 1.335 +/- 1.453 eV and RMSE mean +/- SD of 1.787 +/- 0.518 eV (n=3).
Scratch-MultiT gives a training-seed peak mean +/- SD of 1.588 +/- 0.491 eV and RMSE mean +/- SD of 1.555 +/- 0.333 eV (n=3).

## Interpretation boundary

The updated b/c panels support comparison of relative-energy profile shape, peak location, endpoint periodicity, and training-seed spread on a common fixed geometry. They do not establish a validated migration barrier: the MACE paths are not optimized, the DFT reference retains its warning-bearing status, and the fixed-path operator is not interchangeable with a self-relaxed NEB observable. The two Scratch subfamilies remain separate because both use max_L=2 and differ in training-data lineage; they are not collapsed into one Scratch mean. Legacy `dc41a989...` Foundation results are retained only for the A0 pipeline self-check and are excluded from the current panel and current conclusions.

