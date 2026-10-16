# Gap→QL bidirectional activation-energy and geometry audit

## Geometry finding

All four canonical inputs use the same 81-atom cell and atom ordering. The raw Cr `pw_1.in` and `pw_2.in` are exact Cu-coordinate substitutions at both endpoints: only the final dopant species changes from Cu to Cr. Li and Na are not coordinate-identical to Cu; their endpoint structures contain element-specific dopant and host relaxation. The saved NEB paths also move differently after iteration, so final image positions are not pointwise identical across dopants.

## Energy definition

For every ordered image profile, `forward = max(E_i − E_1)` and `reverse = max(E_i − E_10)`. These are two directional readouts of one profile. The Foundation models evaluate the fixed DFT images and do not constitute separate ML NEB optimizations.

## QE directional values

`endpoint_delta_eV = forward − reverse`; positive values mean the final endpoint is higher than the initial endpoint, and negative values mean the process is downhill in the forward direction.

| dopant   | path_class       |   forward_activation_eV |   reverse_activation_eV |   endpoint_delta_eV |
|:---------|:-----------------|------------------------:|------------------------:|--------------------:|
| Cr       | gap → QL         |                0.000000 |                0.095947 |           -0.095947 |
| Cu       | gap → QL         |                0.025851 |                0.000000 |            0.025851 |
| Li       | gap → QL         |                0.005263 |                0.000000 |            0.005263 |
| Na       | gap → QL         |                0.058431 |                0.000000 |            0.058431 |
| Cr       | in-gap diffusion |                1.524299 |                0.675386 |            0.848913 |
| Cu       | in-gap diffusion |                0.289683 |                0.299887 |           -0.010204 |
| Li       | in-gap diffusion |                0.580664 |                0.255346 |            0.325318 |
| Na       | in-gap diffusion |                0.874214 |                0.185276 |            0.688939 |
| Cr       | within-QL        |                1.844900 |                2.346822 |           -0.501922 |
| Cu       | within-QL        |                1.008121 |                0.791521 |            0.216600 |
| Li       | within-QL        |                1.969137 |                1.839863 |            0.129274 |
| Na       | within-QL        |                3.098166 |                2.823136 |            0.275030 |

## Foundation-model directional ranges

| dopant   | path_class       |   n_models |   forward_median_eV |   forward_min_eV |   forward_max_eV |   reverse_median_eV |   reverse_min_eV |   reverse_max_eV |
|:---------|:-----------------|-----------:|--------------------:|-----------------:|-----------------:|--------------------:|-----------------:|-----------------:|
| Cr       | gap → QL         |         10 |            0.029836 |         0.000000 |         0.072545 |            0.000000 |         0.000000 |         0.086292 |
| Cr       | in-gap diffusion |         10 |            1.167452 |         0.899060 |         1.549396 |            0.701506 |         0.467452 |         0.767725 |
| Cr       | within-QL        |         10 |           24.222283 |        21.176812 |        29.735416 |           24.726953 |        21.492634 |        30.049008 |
| Cu       | gap → QL         |         10 |            0.106621 |         0.076150 |         0.138884 |            0.000000 |         0.000000 |         0.000000 |
| Cu       | in-gap diffusion |         10 |            0.230956 |         0.104866 |         0.404403 |            0.457406 |         0.400927 |         0.518217 |
| Cu       | within-QL        |         10 |           24.032036 |        22.546811 |        37.702128 |           23.720049 |        22.262687 |        37.352449 |
| Li       | gap → QL         |         10 |            0.181115 |         0.107330 |         0.240876 |            0.000000 |         0.000000 |         0.000000 |
| Li       | in-gap diffusion |         10 |            0.393690 |         0.339034 |         0.453976 |            0.428503 |         0.341400 |         0.605088 |
| Li       | within-QL        |         10 |            1.935250 |         1.780952 |         2.256874 |            1.719743 |         1.620224 |         1.981069 |
| Na       | gap → QL         |         10 |            0.196429 |         0.123735 |         0.245517 |            0.000000 |         0.000000 |         0.000000 |
| Na       | in-gap diffusion |         10 |            0.695845 |         0.638353 |         0.744203 |            0.336409 |         0.268696 |         0.415034 |
| Na       | within-QL        |          9 |            3.442287 |         3.092656 |         3.649054 |            2.666136 |         2.298358 |         2.804342 |

## Interpretation

Cr gap→QL is downhill in the forward direction in the retained path table, whereas Cu, Li, and Na retain nonzero forward rises. This is consistent with a chemistry-dependent coordination/size response, but the present data do not isolate covalent radius from charge transfer, d-state bonding, and endpoint relaxation. The radius/coordination explanation is therefore a mechanistic hypothesis, not a demonstrated causal decomposition.

The full directional table and geometry audit are stored beside this report. Cr QE values remain `COMPLETED_BUT_SCF_WARNING`; they are diagnostic path-table references, not final converged barriers.
