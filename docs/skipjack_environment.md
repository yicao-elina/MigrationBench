# Skipjack environment

The validated production path on Skipjack uses the ARCH-provided CUDA/PyTorch
module and a project-local Python package prefix. This keeps the PyTorch build
matched to the cluster CUDA stack and avoids downloading a second CUDA runtime.
The conda environment is optional for lightweight tooling; do not run MACE with
its Python interpreter because mixing it with the ARCH PyTorch module can hang
or load incompatible binary libraries.

```bash
module load gcc/9.3.0 pytorch/2.10.0-b200-py3.11
export MB_PYTHON_PKGS="$PWD/.python_pkgs"
python -m pip install --target "$MB_PYTHON_PKGS" --no-deps \
  -r requirements-skipjack.txt
export PYTHONPATH="$MB_PYTHON_PKGS:${PYTHONPATH:-}"
```

If a conda environment is wanted for repository-only tools, create it with
`conda env create -f environment-skipjack.yml`, but keep using the module's
`python` after loading PyTorch for MACE/MD jobs. The environment is therefore
reusable as a named project tool environment, while the validated MACE binary
runtime is the ARCH module stack plus `MB_PYTHON_PKGS`.

The tested runtime is Python 3.11.9, PyTorch 2.10.0+cu128, ASE 3.25.0, and
MACE-Torch 0.3.12. Verify it before submitting jobs:

```bash
python - <<'PY'
import sys, torch, ase, mace
from mace.calculators import MACECalculator
print(sys.version.split()[0])
print(torch.__version__)
print(ase.__version__)
print(mace.__version__)
print(MACECalculator.__name__)
PY
```

This is a reusable environment for MACE inference, ASE relaxation, NEB
preconditioning, and MD scripts that use the same calculator interface. It does
not make a model scientifically valid for a new system: the checkpoint SHA,
element coverage, spin/charge assumptions, timestep, ensemble, and acceptance
gates must still be recorded per campaign.

The runtime was validated on Skipjack with Slurm job `908659`: a two-step ASE
Langevin MACE smoke on the pilot `Cr@Sb` input completed with finite energies.
The same validation used the module stack above and reported Python 3.11.9,
PyTorch 2.10.0+cu128, ASE 3.25.0, and MACE-Torch 0.3.12.

For a generic workstation, use the repository `environment.yml`; it is kept
separate from the Skipjack module recipe because workstation CUDA/PyTorch
resolution is not the same as the ARCH module stack.
