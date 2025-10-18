
# Machine Learning Force Field (MLFF) Training

This directory contains tools and configurations for training machine learning force fields using processed NEB data.

## Workflow Overview

```
Merged XYZ → Data Splitting → Isolated Atoms → Training/Validation/Test → MLFF Training
```

## Quick Start

```bash
# 1. Copy merged data from NEB processing
cp ../03_neb/merged.xyz .

# 2. Prepare isolated atom reference
# Edit isolated_atom.xyz with single atom energies

# 3. Split data into train/validation/test sets
python split_data.py merged.xyz

# 4. Submit training job
cd srun/
# Edit sbatch.slurm as needed
sbatch sbatch.slurm
```

## Files Description

### Input Files

- **`merged.xyz`** - Combined NEB trajectory data from `03_neb/`
- **`isolated_atom.xyz`** - Reference energies for isolated atoms (required for energy baseline)

### Scripts

#### `split_data.py`
Splits the merged dataset into training, validation, and test sets.

**Usage:**
```bash
python split_data.py <input_xyz> [--train_ratio 0.8] [--val_ratio 0.1] [--test_ratio 0.1]
```

**Default behavior:**
- 80% training data (includes isolated atoms)
- 10% validation data  
- 10% test data
- Ensures isolated atoms are in training set

**Output:**
- `train.xyz` - Training dataset with isolated atoms
- `valid.xyz` - Validation dataset
- `test.xyz` - Test dataset

### Training Configuration

#### `srun/sbatch.slurm`
SLURM job submission script for MLFF training.

**Key parameters to edit:**
```bash
#SBATCH --job-name=mlff_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Training command (example for MACE)
mace_run_train \
    --train_file="../train.xyz" \
    --valid_file="../valid.xyz" \
    --test_file="../test.xyz" \
    --config_type_weights='{"Default":1.0}' \
    --model="MACE" \
    --hidden_irreps="128x0e + 128x1o" \
    --r_max=5.0 \
    --batch_size=32 \
    --max_num_epochs=500 \
    --patience=50 \
    --energy_weight=1000.0 \
    --forces_weight=100.0 \
    --device=cuda
```

## Data Preparation Details

### 1. Isolated Atom Reference

Create `isolated_atom.xyz` with reference energies:
```
1
Lattice="20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=-X.XXXXX config_type=IsolatedAtom
Te  10.0  10.0  10.0  0.0  0.0  0.0
1
Lattice="20.0 0.0 0.0 0.0 20.0 0.0 0.0 0.0 20.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=-Y.YYYYY config_type=IsolatedAtom  
Sb  10.0  10.0  10.0  0.0  0.0  0.0
```

**Note:** Replace `-X.XXXXX` and `-Y.YYYYY` with actual DFT-calculated isolated atom energies.

### 2. Data Statistics

Check dataset statistics:
```bash
# Count structures
grep -c "Lattice" train.xyz
grep -c "Lattice" valid.xyz  
grep -c "Lattice" test.xyz

# Check energy range
python -c "
import ase.io
structures = ase.io.read('train.xyz', ':')
energies = [s.info['energy'] for s in structures]
print(f'Energy range: {min(energies):.3f} to {max(energies):.3f} eV')
print(f'Total structures: {len(structures)}')
"
```

## Training Parameters

### Recommended MACE Settings

For Sb₂Te₃ systems:
- **r_max**: 5.0-6.0 Å (capture nearest neighbors)
- **hidden_irreps**: "128x0e + 128x1o" or "256x0e + 128x1o"  
- **batch_size**: 16-32 (adjust based on GPU memory)
- **energy_weight**: 1000.0
- **forces_weight**: 100.0
- **max_num_epochs**: 500-1000
- **patience**: 50-100

### Hardware Requirements

- **GPU**: Recommended (CUDA-compatible)
- **Memory**: 16+ GB RAM
- **Storage**: 10+ GB free space
- **Time**: 4-24 hours depending on dataset size

## Monitoring Training

### Log Files
- `training.log` - Detailed training progress
- `results.txt` - Final model performance metrics
- `checkpoints/` - Model checkpoints during training

### Key Metrics to Monitor
- **Energy MAE** (Mean Absolute Error)
- **Force MAE** 
- **Training/Validation loss convergence
- **Learning rate schedule**

### Example Monitoring
```bash
# Watch training progress
tail -f training.log

# Check convergence
grep "Epoch" training.log | tail -20

# Final results
cat results.txt
```

## Model Validation

After training completion:

```bash
# Test model performance
mace_eval \
    --model="model.pth" \
    --test_file="test.xyz" \
    --output="evaluation_results.json"

# Generate plots (if available)
python plot_results.py evaluation_results.json
```

## Troubleshooting

**Common Issues:**

1. **CUDA out of memory**
   - Reduce `batch_size`
   - Use smaller `hidden_irreps`

2. **Poor convergence**
   - Check data quality and energy ranges
   - Adjust learning rate or patience
   - Ensure isolated atoms are included

3. **High force errors**
   - Increase `forces_weight`
   - Check force data quality
   - Consider force regularization

**Debugging:**
```bash
# Check data format
head -50 train.xyz

# Validate XYZ files
python -c "import ase.io; ase.io.read('train.xyz', ':')"

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## Output Files

Successful training produces:
- `model.pth` - Trained MLFF model
- `training_log.txt` - Complete training history
- `evaluation_results.json` - Model performance metrics
- `checkpoints/` - Intermediate model states

## Next Steps

After successful training:
1. Validate model on independent test systems
2. Use model for MD simulations or property predictions
3. Consider active learning for model improvement
