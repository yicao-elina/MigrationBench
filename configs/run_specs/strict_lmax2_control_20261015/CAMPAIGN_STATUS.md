# Campaign status

Submitted to Rockfish as Slurm array `32055099_[0-1]` on 2026-10-15.

Remote root:

`/scratch16/pclancy3/yi/strict_lmax2_control_20261015`

Array mapping:

| array task | training seed | canonical intended model |
|---:|---:|---|
| 0 | 123 | `strict_lmax2_seed123` |
| 1 | 345 | `strict_lmax2_seed345` |

The array was submitted with a two-task concurrency cap. At submission, the
root was absent and no historical output directory was overwritten. The first
status check showed both tasks `PENDING` with no scheduler error.

## Frozen remote inputs

| input | SHA256 |
|---|---|
| `data/train.xyz` | `ce0a0188d624f0a3319e329bd5c87f75bdaf427e5ec96bcc4482cb8a0825d9e5` |
| `data/valid.xyz` | `93731a055fa39cbe73c469272ea7d26a0cc408cdd7daf87fdeea9703bee80540` |
| `data/test.xyz` | `2698ce412761c10ca6a3b9c014d7078e353bebb3fe70854846730bb8e6b4af14` |
| seed234 compiled template | `52fccd0994664618256ec5e986dc5e962050c845d3f05f3d70862ff064009af5` |

## Scientific status

Training is pending; no new model or 639-point result is publication-ready.
The next accepted state requires all gates in `ACCEPTANCE_MATRIX.tsv` to pass.

## Interim figure change

The current historical 12-model figure was re-rendered with the panel-b
vertical grey separators removed. The horizontal 100 eV Angstrom^-1 force gate
is retained. Interim outputs are:

- PNG SHA256: `f2c0bfba5c65638d545f9f6ff5ed62b7e3e58bed7f830aa805e4bb041f449f19`
- PDF SHA256: `b3676990a587b16b7cde2b24614023abef37f74ffc65c2cd50b3367cd8201a94`

These are an updated historical-matrix reconstruction; they will be replaced
by the strict-trio merged figure only after the new static evaluation passes.
