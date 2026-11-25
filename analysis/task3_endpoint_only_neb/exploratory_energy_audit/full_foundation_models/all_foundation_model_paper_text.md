# Manuscript-ready figure text

## Figure caption

**Figure X | Energy-profile dispersion across the complete local MACE foundation-model set.** Relative single-point energies are shown along eight endpoint-only linear paths constructed from fresh QE endpoint inputs (Cu/Li/Na in the QLs, gap-diffuse, and gap2QL/gap2Ql variants; ten images per path). Each colored line denotes one of the nine models in `/Users/alina/Downloads/mace-foundation-models`; MH-0 and MH-1 were evaluated with the explicit `omat_pbe` head. (a–h) Individual path profiles. The companion spread panel ranks models by their root-mean-square deviation from the across-model median profile and summarizes the path-wise range of the maximum relative energy across models. The black QE trace uses the corrected `.path` Hartree-to-eV conversion but is not geometry matched to these fresh linear inputs.

## Results and discussion

The complete nine-model comparison shows that the apparent agreement of a two-model plot is not representative of the full foundation-model ensemble. Agreement is path dependent: the three gap-diffuse/gap2QL-type families remain relatively close, whereas the QLs paths expose very large model dependence. The path-wise median barrier-like maxima are 0.13–0.85 eV for the six gap-based paths but 16.53 eV (Li QLs) and 37.44 eV (Cu QLs) for the two direct QLs linear interpolations. These very large values are diagnostic of endpoint-only interpolation and likely short-range/overlap sensitivity, not evidence for physical migration barriers.

Across all 720 model–image evaluations, the closest profile to the ensemble median is Agnesi-M-dens (RMSE 0.151 eV). The most dispersed is Agnesi-S-dens (2.908 eV), followed by OMAT-0-M (1.122 eV), OMAT-0-S (1.095 eV), Agnesi-M (0.825 eV), and MH-0 (0.816 eV). Thus the models do not all differ equally: the spread is concentrated in particular model families and path classes. The result supports using the full model matrix as a screening/uncertainty diagnostic, but it does not identify a physically correct barrier or establish DFT accuracy. The controlled same-geometry nine-model comparison on the retained QE/NEB path frames is provided separately in `../corrected_dft_foundation_models/`.

## Scope and reproducibility

All nine local model files were evaluated successfully (9 × 80 = 720 frames). Energies are normalized independently within each path to the first image. The inputs are fresh endpoint-only linear interpolations generated from remote QE endpoint files; this is intentionally a model-disagreement experiment and must not be labeled as a converged NEB calculation.
