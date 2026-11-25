# FT-600K repair r01 MD smoke failure report

The training and GPU model gate passed, so the reference MD workflow was
started with the seed-specific compiled models. Both 20 ps smoke runs were
terminated by the existing scientific-threshold gate before the requested
smoke endpoint. Neither task submitted a continuation; no 1 ns trajectory
exists.

## Common protocol

- input: `2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz`
- input SHA256: `e2f72e149940af7fd9016d2e1d58545cbe4734f935d91d9e4dc84c2e7645e454`
- system: 2050 atoms, periodic in all three directions
- timestep: 1 fs
- target temperature: 600 K
- friction: 0.01
- absolute start step: 200000 (200.0 ps)
- requested smoke length: 20000 steps
- MD velocity seed: 1100
- smoke array: `32455770_[0-1]`, both on `gpu18`

## Results

| seed | state | last step | last time | temperature | max force | min distance | status |
|---:|---|---:|---:|---:|---:|---:|---|
| 234 | `FAILED`, exit `1:0`, actual `32455778` | 201800 | 201.8 ps | 17,605,136.6 K | 234.43 eV/Å | `nan` | `smoke_pass=false` |
| 345 | `FAILED`, exit `1:0`, actual `32455770` | 201200 | 201.2 ps | 2,174,618.1 K | 550.89 eV/Å | `nan` | `smoke_pass=false` |

Both failures are scientific instability, not scheduler or model-load errors.
The trajectories are retained under `remote_evidence/` and must not be used
for 1 ns diffusion, transport, or family-mean claims. They may be used only as
explicitly labelled failure evidence.
