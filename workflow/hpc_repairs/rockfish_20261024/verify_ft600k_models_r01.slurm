#!/bin/bash -l
# GPU load/finite-probe gate for the completed FT-600K repair models.
# This job is read-only with respect to model files and writes only verification
# evidence under the repair root.
#SBATCH --job-name=mb-ft600k-gate-r01
#SBATCH --account=pclancy3_gpu
#SBATCH --qos=qos_gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=/scratch16/pclancy3/yi/ft_600k_random_reproduction_20261019_repair_r01/gate-%j.out
#SBATCH --error=/scratch16/pclancy3/yi/ft_600k_random_reproduction_20261019_repair_r01/gate-%j.err

set -eo pipefail
export SLURM_CONF=/cm/shared/apps/slurm/var/etc/slurm/slurm.conf
export RF_CONDA_SITE="${RF_CONDA_SITE:-}"
export SINGULARITY_BIND="${SINGULARITY_BIND:-}"
source /etc/profile.d/modules.sh
set +u
source /data/apps/go.sh
source /data/apps/helpers/singularity.sh
module restore
module unload openmpi/3.1.6 >/dev/null 2>&1 || true
ml openmpi/4.1.6
source /data/apps/linux-centos8-cascadelake/gcc-9.3.0/anaconda3-2020.07-i7qavhiohb2uwqs4eqjeefzx3kp5jqdu/etc/profile.d/conda.sh
conda activate mace
set -u

ROOT=/scratch16/pclancy3/yi/ft_600k_random_reproduction_20261019_repair_r01
GATE_ROOT="$ROOT/model_gate_r01"
DATA_ROOT=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/data_600K
FOUNDATION=/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/MACE-matpes-pbe-omat-ft.model
MODEL234="$ROOT/models/ft_600k_random_reproduction_repair_r01_seed234/ft_600k_random_reproduction_repair_r01_seed234_stagetwo_compiled.model"
MODEL345="$ROOT/models/ft_600k_random_reproduction_repair_r01_seed345/ft_600k_random_reproduction_repair_r01_seed345_stagetwo_compiled.model"
mkdir -p "$GATE_ROOT"

for f in "$DATA_ROOT/train.xyz" "$DATA_ROOT/valid.xyz" "$DATA_ROOT/test.xyz" "$FOUNDATION" "$MODEL234" "$MODEL345"; do
  test -s "$f"
done
{
  echo "gate_job_id=${SLURM_JOB_ID:-unknown}"
  echo "host=$(hostname)"
  command -v python
  python --version
  python -c 'import torch, ase, mace; print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0)); print("ase", ase.__version__, "mace", getattr(mace, "__version__", "unknown"))'
  sha256sum "$DATA_ROOT"/train.xyz "$DATA_ROOT"/valid.xyz "$DATA_ROOT"/test.xyz "$FOUNDATION" "$MODEL234" "$MODEL345"
} | tee "$GATE_ROOT/preflight_gate.txt"

python - "$GATE_ROOT/model_gate.json" "$DATA_ROOT/train.xyz" "$DATA_ROOT/valid.xyz" "$DATA_ROOT/test.xyz" "$MODEL234" "$MODEL345" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

out_path, train_path, valid_path, test_path, *model_paths = sys.argv[1:]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


results = []
for model_path in model_paths:
    model_result = {
        "model_path": model_path,
        "model_sha256": sha256(model_path),
        "size_bytes": Path(model_path).stat().st_size,
        "probes": [],
    }
    try:
        calculator = MACECalculator(
            model_paths=[model_path], device="cuda", default_dtype="float32"
        )
        for split, path in (("train", train_path), ("valid", valid_path), ("test", test_path)):
            atoms = read(path, index=0)
            atoms.calc = calculator
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            numbers = sorted({int(z) for z in atoms.numbers})
            finite = bool(np.isfinite(energy) and np.isfinite(forces).all())
            if not finite:
                raise RuntimeError(f"non-finite output for {split}")
            model_result["probes"].append(
                {
                    "split": split,
                    "natoms": int(len(atoms)),
                    "species": numbers,
                    "energy_eV": energy,
                    "force_max_eV_per_A": float(np.max(np.abs(forces))),
                    "force_l2_eV_per_A": float(np.linalg.norm(forces)),
                    "finite": finite,
                }
            )
        model_result["status"] = "PASS"
    except Exception as exc:
        model_result["status"] = "FAIL"
        model_result["error"] = repr(exc)
    results.append(model_result)

payload = {
    "gate": "ft600k_repair_r01_gpu_load_finite_probe",
    "status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL",
    "input_sha256": {
        "train": sha256(train_path),
        "valid": sha256(valid_path),
        "test": sha256(test_path),
    },
    "models": results,
}
Path(out_path).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2))
if payload["status"] != "PASS":
    raise SystemExit(2)
PY

echo "MODEL_GATE_PASS"
