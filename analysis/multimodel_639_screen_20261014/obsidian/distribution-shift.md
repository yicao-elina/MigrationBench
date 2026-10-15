# Distribution shift relative to DFT/NEB references

## Reference populations

The stored SOAP/PCA reference embedding contains 521 structures:

| reference | n | PCA1 median | PCA2 median |
|---|---:|---:|---:|
| training | 250 | 0.065 | 0.004 |
| validation | 122 | −0.272 | 0.152 |
| test | 124 | −0.009 | 0.092 |
| NEB | 25 | −8.776 | 213.866 |

The descriptor settings are retained in `../inputs/diagnostic_manifest.json`: Cr/Sb/Te SOAP, 5 Å cutoff, σ=0.5 Å, nmax=4, lmax=4, followed by 20-component PCA.

## Candidate location

- Candidate PCA1 range: 1.173–2.918
- Candidate PCA2 range: −4.774 to −2.908
- Median descriptor distance to training: 11.0–12.7 by candidate stage
- Median descriptor distance to NEB: 615.1–616.0 by candidate stage
- Outside training q95 fraction: 1.0 for every stage
- Outside training q99 fraction: 1.0 for every stage
- Candidate-versus-all-reference 2D JS divergence: 0.905 under the fixed audit binning

The candidate set is therefore strongly shifted relative to the existing DFT training/validation/test embedding and is also distinct from the NEB reference population. These are descriptor-space diagnostics, not calibrated uncertainty estimates and not proof that the structures are chemically invalid.
