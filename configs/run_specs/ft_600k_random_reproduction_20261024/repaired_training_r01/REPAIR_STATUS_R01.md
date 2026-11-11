# FT-600K repaired training r01

Status at audit time: training completed for both seeds and independent GPU
load/finite-probe gate `32455612` passed. The 20 ps MD smoke array
`32455770_[0-1]` is submitted; each successful smoke task will release its own
1 ns continuation.

## Template and repair

The working repair template is
`hpc_repairs/rockfish_20261019/submit_ft_600k_random_reproduction_repair_r01.slurm`
(SHA256 `f6851f843800ca8ec859f70d588c03208c551d44a0d17ff15f372c9f11bd63f9`).
It follows the successful training pattern recorded by
`TRAINING_COMPLETION_RECORD_32114528.json`, but fixes the failed 20261016
bootstrap by sourcing the verified Rockfish Conda installation at
`/data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh`.
It writes to the new, non-overwriting root
`/scratch16/pclancy3/yi/ft_600k_random_reproduction_20261019_repair_r01`.

## Scheduler evidence

| stage | job/array task | seed | state | elapsed | node |
|---|---|---:|---|---:|---|
| bootstrap/data/GPU preflight | `32211668` | — | `COMPLETED`, exit `0:0` | 00:00:19 | gpu10 |
| full training array task 0 | `32211669_0` → execution `32211672` | 234 | `COMPLETED`, exit `0:0` | 08:59:34 | gpu11 |
| full training array task 1 | `32211669_1` → execution `32211669` | 345 | `COMPLETED`, exit `0:0` | 09:00:08 | gpu12 |

Both preflight files record the historical random split and the expected data
hashes. Both training logs end with `Done` and `training_complete=...`.

## Model outputs

| seed | compiled model | size | SHA256 | test RMSE E | test RMSE F | relative F RMSE |
|---:|---|---:|---|---:|---:|---:|
| 234 | `...seed234_stagetwo_compiled.model` | 13,102,921 bytes | `87bcc1d07164918e0aad86c29ae64bd5ed9f57a5def3161c1a4f8f2df57cc4fd` | 0.5 meV/atom | 40.1 meV/Å | 8.03% |
| 345 | `...seed345_stagetwo_compiled.model` | 13,102,921 bytes | `a3abccc3bc589cb41a7d281dcefa1e86cb42ac07f9e9fb4905608d469a1e85a3` | 0.5 meV/atom | 38.9 meV/Å | 7.80% |

The training logs include non-fatal warnings about unavailable cuequivariance,
ASE key naming, and multihead fine-tuning metadata. The independent GPU model
gate passed despite these warnings, so both checkpoints are now verified for
loading and finite-probe use; trajectory-level scientific acceptance remains
separate.

## GPU model gate

Gate job `32455612` ran on `gpu03`, finished in 42 seconds with exit `0:0`,
and produced `model_gate.json` with `status: PASS`. Both compiled models
loaded on CUDA; train, validation, and test probes produced finite energy and
forces. Each probe contained Cr/Sb/Te (atomic numbers 24/51/52). The gate
hashes exactly match the two `model_hash.txt` files above.

## MD handoff

The smoke and continuation scripts are the new files under
`hpc_repairs/rockfish_20261024/`. The smoke array used the seed-specific
compiled model and MD velocity seed `1100`; each successful task would have
submitted its continuation to absolute step `1000000` with `afterok`. Smoke
array `32455770` ran both tasks on `gpu18`, but both failed the scientific
stability threshold near 201 ps. No continuation was submitted. The complete
failure evidence is in `MD_FAILURE_REPORT.md`; the failed trajectories are
retained and excluded from transport claims.
