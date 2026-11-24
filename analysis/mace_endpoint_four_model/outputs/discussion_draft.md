# Four-model MACE endpoint relaxation: publication discussion draft

Dataset: 564 endpoint×variant×model records (47 endpoints, 3 deterministic starts, 4 models). Scientific success is read from each summary JSON and requires optimizer convergence plus the stated finite-force criterion. Absolute total energies are not compared across models.

## Results
- **FT-600K**: 141/141 scientific-successful relaxations; median final maximum force 0.0279 eV/Å and median FIRE steps among all records 53.
- **Foundation**: 141/141 scientific-successful relaxations; median final maximum force 0.0270 eV/Å and median FIRE steps among all records 25.
- **Scratch**: 138/141 scientific-successful relaxations; median final maximum force 0.0267 eV/Å and median FIRE steps among all records 53.
- **FT-MultiT**: 141/141 scientific-successful relaxations; median final maximum force 0.0270 eV/Å and median FIRE steps among all records 50.

- Relative to FT-600K, **Foundation** has median paired Cr displacement 0.225 Å (95th percentile 0.880 Å; fraction ≤0.10 Å: 15.6%).
- Relative to FT-600K, **Scratch** has median paired Cr displacement 0.284 Å (95th percentile 0.562 Å; fraction ≤0.10 Å: 9.2%).
- Relative to FT-600K, **FT-MultiT** has median paired Cr displacement 0.283 Å (95th percentile 0.487 Å; fraction ≤0.10 Å: 3.5%).

## Landscape interpretation
The landscape panels use E_rel(endpoint, variant, model) = E_final − min_variant(E_final) for each endpoint and model. This removes model-dependent energy offsets and makes the three deterministic starting geometries visible as a local, discrete relaxation landscape. The common color scale is descriptive within the plotted normalization; it must not be read as a cross-model absolute-energy comparison. It is not a replacement for a continuous reaction-coordinate scan or a converged NEB.

## Discussion
The central result is substantial model dependence in the final basin coordinates, despite uniformly small residual forces for successful runs. Relative to FT-600K, the median Cr displacement is 0.225–0.284 Å and only 3.5–15.6% of paired structures fall within 0.10 Å. Therefore the present data do not support the stronger claim that endpoint pre-relaxation is model-insensitive. Instead, they support a sensitivity-screening interpretation: Foundation, Scratch and FT-MultiT generate valid locally converged structures, but they do not generally select the same Cr basin as FT-600K. The most divergent endpoints should be retained as an explicit uncertainty set and adjudicated by DFT. The three Scratch failures at unique_151 are also scientifically informative and should remain visible in the failure table rather than being imputed or removed. Differences in FIRE steps or residual forces are secondary diagnostics of local curvature and optimizer conditioning, not direct measures of model accuracy.

## Recommended manuscript wording
Across 47 Cr@Sb2Te3 endpoints and three deterministic starting geometries, the four MACE models were compared using paired final geometries, convergence, residual forces, and model-internal relative energies. Because MACE energy zero points are model-dependent, cross-model absolute total energies were not compared. Instead, each endpoint was assigned a discrete relaxation landscape after subtracting the minimum energy among its three starting variants within each model. All models except three Scratch runs at unique_151 satisfied the relaxation gate, but the final Cr coordinates showed appreciable model dependence: the median displacement from FT-600K was 0.225–0.284 Å, with only 3.5–15.6% of paired structures within 0.10 Å. Thus, the multi-model ensemble is best used to identify basin-sensitive endpoints and define DFT-priority cases, rather than to claim a unique model-independent endpoint geometry.
