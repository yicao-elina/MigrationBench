# 15-model 1 ns MD status and figure release

Date: 2026-10-27  
Campaign root: `/scratch16/pclancy3/yi/grouped_A2_ft_1ns_md_20261025/`  
Input: `MODEL_REGISTRY.tsv` plus Rockfish trajectory/ledger evidence; local SQL was used first for registry cross-checking.

## Scientific status

The intended matrix is complete at 15 model identities: three zero-shot MACE Foundation checkpoints, three FT-600K seeds, three FT-MultiT seeds, three Scratch-600K seeds, and three Scratch-MultiT seeds. All 15 are represented in the figure individually and are plotted only through the raw trajectory horizon that exists for that model.

The current evidence supports four target-reaching trajectories and eleven scientific-threshold failures:

| Model | Last step | Last time (ps) | Evidence classification |
|---|---:|---:|---|
| Foundation MPA-0 | 1,000,000 | 1,000 | target reached; provisional stability evidence |
| Foundation MP-0b3 | 1,000,000 | 1,000 | target reached; provisional stability evidence |
| Foundation OMAT-0 | 1,000,000 | 1,000 | target reached; provisional stability evidence |
| FT-600K seed 123 | 236,000 | 236 | scientific-threshold failure |
| FT-600K seed 234 | 1,000,000 | 1,000 | target reached; ledger `target_reached` |
| FT-600K seed 345 | 485,000 | 485 | scientific-threshold failure |
| FT-MultiT seed 123 | 237,200 | 237.2 | scientific-threshold failure |
| FT-MultiT seed 234 | 270,200 | 270.2 | scientific-threshold failure |
| FT-MultiT seed 345 | 286,000 | 286 | scientific-threshold failure |
| Scratch-600K seed 123 | 204,400 | 204.4 | scientific-threshold failure |
| Scratch-600K seed 234 | 200,800 | 200.8 | scientific-threshold failure |
| Scratch-600K seed 345 | 200,400 | 200.4 | scientific-threshold failure |
| Scratch-MultiT seed 123 | 201,200 | 201.2 | scientific-threshold failure |
| Scratch-MultiT seed 234 | 200,400 | 200.4 | scientific-threshold failure |
| Scratch-MultiT seed 345 | 200,400 | 200.4 | scientific-threshold failure |

This means “15/15 have terminal trajectory/status evidence,” not “15/15 are stable 1 ns runs.” The six Scratch curves are intentionally retained as short-horizon failure-mode evidence; they are not extended or imputed to 1 ns.

The refreshed analyzer produced 15/15 series, MSD, VACF, and RDF files. Every trajectory has `invalid_frames=0`; all plotted output columns are finite, and the core temperature/pressure metrics are finite for all 15 models.

## RDF correction

The old RDF panel used a logarithmic ordinate and discarded `g(r) <= 0` values. The 0–2 Å region therefore looked empty even though the raw Cr–Cr RDF contains valid zero bins. The refreshed panel retains those bins and uses a zero-inclusive symlog ordinate with an explicit `g(r)=0` baseline. No values were fabricated: an empty short-range Cr–Cr shell remains a physical zero.

## Analysis and plotting definitions

- Temperature: instantaneous kinetic temperature recomputed from raw momenta with ASE masses.
- Pressure: hydrostatic pressure `P = -trace(stress)/3`, converted from eV Å⁻³ to GPa.
- MSD: triclinic fractional unwrapping; panel C shows `MSD_xy`.
- VACF: normalized multi-origin velocity autocorrelation; each model keeps its observed lag range.
- RDF: Cr–Cr minimum-image distances, `r_max = 12 Å`, `dr = 0.1 Å`, every twentieth eligible frame; panel E displays 0–10 Å.
- Failed trajectories are plotted to their observed endpoint only. Clipped endpoints in A/B are marked rather than silently removed.

## Provenance boundary

The local SQLite registry currently labels the relevant MD campaigns `PARTIAL`, so it is not sufficient by itself to certify the current Rockfish state. The release uses the Rockfish registries, ledgers/status records, raw extxyz paths, and SHA256 values as the current evidence layer; the SQL result is retained as a cross-check and is not overwritten.

The release manifest records the input manifest, analyzer source, Rockfish summaries, status audit, and figure hashes.
