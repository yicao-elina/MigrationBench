# DFT–MACE endpoint-47 site-distribution and disagreement analysis

## Evidence scope

This package separates four quantities that must not be conflated:

1. geometric placement of the 47 immutable source structures;
2. relative DFT energies only within the 21 BFGS-final retry endpoints;
3. local-basin sensitivity from the existing four-model relaxation campaign;
4. twelve-member MACE static source-geometry disagreement.

It does not report a calibrated MACE-versus-DFT error. The final DFT XYZ
geometries are not locally available for same-geometry MACE force/energy
evaluation. Absolute QE and MACE total energies are therefore never
subtracted.

## Results

### The 47 sites cover three physically interpretable vertical regions

Using the two central Te surfaces in each immutable 81-atom source structure,
the candidate Cr locations distribute as 14 within the van der Waals gap, 14
at the gap–QL interface, and 19 within a QL (Fig. 1; Table 1). This is a
geometric classification, not a stable/metastable/unstable classification.
Each of the three pathway families has 24 endpoint memberships, but membership
is deliberately multi-label and the counts are not additive. In other words,
the route taxonomy records intended path context while the vertical class
records the actual source-site placement.

### The retry-21 energy window is heterogeneous, but not a stability landscape

Only the 21 BFGS-final retry endpoints enter the same-campaign relative-energy
panel (Fig. 2). Their window spans 0–4.732 eV. The within-QL subset is small
(n=3) and has a median ΔE of 0.0007 eV; interface sites (n=7) have a median
of 1.097 eV and extend to 4.732 eV; gap sites (n=11) have a median of
1.884 eV and extend to 3.767 eV. These descriptive differences are useful for
site coverage, but cannot establish thermodynamic stability, metastability, or
instability: that would require a matched landscape definition and, at
minimum, local-curvature or phonon/Hessian evidence. The three time-limit
records unique_077, unique_078, and unique_093 remain excluded from this
ranking.

### Existing four-model relaxation exposes localized basin sensitivity

The legacy four-model relaxation set contains 564 endpoint × deterministic
start × model records. It has 561 successful relaxations and retains the three
Scratch failures at unique_151. Median three-start energy spans are 6.88 meV
for FT-600K, 7.48 meV for FT-MultiT, 4.28 meV for the OMAT-derived legacy
comparator, and 12.10 meV for Scratch (Table 2). However, endpoint-specific
maxima reach 0.418, 0.912, 0.748, and 0.118 eV, respectively. The correct
interpretation is discrete local-basin sensitivity to the three deterministic
starts, not a reaction barrier and not a DFT error.

### Twelve-member source-geometry disagreement is associated with local geometry

The new hash-verified static ensemble completed 12 models × 47 source
geometries = 564 successful energy/force predictions (Fig. 4). The 12 members
are three distinct Foundation-family comparators and three training seeds in
each of FT-600K, FT-MultiT, and Scratch. Foundation-family members are treated
as distinct checkpoint/training-data comparators, not as random-seed
replicates.

The root-mean component standard deviation of the Cr force vector is strongly
associated with vertical and coordination descriptors:

- distance from the van der Waals gap center: Spearman ρ=0.564, BH q=4.33e-4;
- number of Sb neighbors within 3.2 Å: ρ=0.576, BH q=4.33e-4;
- number of Te neighbors within 3.2 Å: ρ=-0.510, BH q=1.67e-3;
- nearest-Te distance: ρ=-0.507, BH q=1.67e-3.

By contrast, none of the three multi-label pathway-family memberships explains
this force-disagreement metric after multiple-testing correction. The
within-model relative-energy rank dispersion also has no detected association
with the simple route or local-descriptor terms after correction. Thus the
supported finding is more specific than “gap sites are difficult”: disagreement
tracks a continuous vertical/local-coordination motif rather than the route
label alone.

## What this does and does not answer about error

The data support the following hypothesis: local coordination and departure
from the central gap are priority strata for a matched-geometry DFT error
test, because they produce larger MACE-model force disagreement. They do not
yet support the statement that MACE error is caused by, or even correlated
with, those features. Ensemble disagreement needs calibration against
independent DFT labels before it can be called an uncertainty estimate or used
as an error predictor.

The next decisive analysis is therefore:

1. transfer the 43 direct and 4 reconstructed final DFT XYZ structures with
   cell, PBC, atom-order, and provenance hashes;
2. evaluate all 12 checkpoint members on those exact DFT geometries;
3. compute force MAE/RMSE and protocol-compatible relative-energy errors;
4. calibrate model disagreement against observed error with a held-out,
   route-grouped split; and
5. add a reference-training descriptor archive before making a quantitative
   coverage or OOD claim.

## Related-work positioning and figure grammar

The figure language follows the useful parts of recent atomistic-ML work
without copying an external figure, layout, or data:

- MACE foundation models have materially different checkpoints and training
  corpora; their identity should be audited rather than collapsed into a
  generic “foundation” label. See the MACE foundation study:
  https://doi.org/10.1063/5.0297006
- Ensemble/readout variability is a model-based UQ signal only after
  calibration. Bilbrey et al. distinguish ensemble and quantile uncertainty
  and demonstrate their use with explicit training/held-out labels:
  https://doi.org/10.1038/s41524-025-01572-y
- Structural feature projections are helpful for hypothesis generation, but
  are not training-data coverage by themselves. Schwalbe-Koda et al. derive
  completeness/outlier metrics from a reference dataset of atom-centered
  features, which is not yet available for this campaign:
  https://doi.org/10.1038/s41467-025-59232-0

Accordingly, Fig. 4 uses a descriptor map, uncertainty-versus-structure
scatter, and model-rank heatmap as a transparent diagnostic grammar. It does
not label the map as OOD or use the ensemble standard deviation as a DFT error
bar.

## Figure captions

**Figure 1 | Geometric distribution of the 47 Cr endpoint source structures.**
Projected host lattice and Cr candidate sites, in-plane coverage, and the
vertical classification derived from the central Te surfaces. Colors indicate
within-gap, gap–QL interface, and within-QL source geometry; they do not
indicate thermodynamic stability. Route membership is multi-label.

**Figure 2 | Restricted DFT relative-energy distribution for the converged retry
subset.** Only 21 BFGS-final retry endpoints are ranked. The energy range is
defined within this one campaign subset; the three time-limit nonconverged
endpoints are excluded. The panels test geometric stratification descriptively,
not a stability classification.

**Figure 3 | Existing four-model local-basin sensitivity.** Three deterministic
starts per endpoint show model-internal energy-span variation and cross-model
final-Cr basin displacement. The Scratch failure at unique_151 is retained as
failure evidence. No panel subtracts MACE and QE total energies.

**Figure 4 | Twelve-member MACE source-geometry disagreement is associated with
local structural descriptors.** The model set comprises three distinct
Foundation-family comparators and three seeds in each fine-tuned/scratch
family. Cr-force disagreement is associated with distance from the gap center
and local Te/Sb coordination, whereas simple pathway membership is not
significant after BH correction. The descriptor projection is not training-data
coverage, and model disagreement is not a calibrated DFT error.

## Manuscript-ready English results text

Across the 47 immutable endpoint source structures, Cr sites occupied three
geometric regions defined relative to the central Te surfaces: 14 within the
van der Waals gap, 14 at the gap–QL interface, and 19 within a QL. In the
restricted set of 21 BFGS-final retry calculations, the same-campaign DFT
relative-energy window spanned 0–4.732 eV. We interpret this as heterogeneity
within a protocol-matched subset, rather than as a thermodynamic stability
ranking, because neither global-basin nor vibrational-stability evidence is
available.

To test whether model behavior is structured by local geometry, we evaluated
12 hash-verified MACE checkpoints on each immutable source geometry. The
root-mean component standard deviation of the Cr force vector increased with
distance from the van der Waals gap center (Spearman ρ=0.564, BH
q=4.33×10−4) and Sb coordination (ρ=0.576, q=4.33×10−4), while decreasing
with Te coordination (ρ=−0.510, q=1.67×10−3). In contrast, the pathway-family
labels did not explain this diagnostic after multiple-testing correction.
These results identify vertical placement and local coordination as candidate
strata for future error calibration, rather than demonstrating a direct
MACE-versus-DFT error mechanism.

The present dataset does not support an absolute MACE–DFT energy comparison:
the DFT final geometries are not locally available for same-geometry MACE
evaluation. We therefore retain model disagreement as a diagnostic, preserve
the Scratch failures, and reserve claims about prediction error, uncertainty
calibration, and training-data coverage for the matched-geometry benchmark.
