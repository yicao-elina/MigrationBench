
# AIMD Workflow Example

This directory contains a complete example of running AIMD simulations for Cr-doped Sb2Te3.

## Files Overview

### Input Files
- `input/structure.xyz` - Initial relaxed structure (75 atoms)
- `input/aimd.in.template` - Quantum Espresso AIMD input template
- `input/submit.slurm.template` - SLURM job submission template

### Scripts
- `scripts/prepare_aimd_input.py` - Generate QE input from template
- `scripts/monitor_aimd.py` - Monitor job progress
- `scripts/convert_to_xyz.py` - Convert QE output to XYZ trajectory

### Output Files (Examples)
- `output/aimd.out` - QE output file (first 1000 lines shown)
- `output/aimd_trajectory.xyz` - Converted trajectory (every 10th frame)
- `output/job_summary.json` - Job statistics and metadata

## Quick Start

### 1. Prepare Input
```bash
cd scripts/
python prepare_aimd_input.py \
    --structure ../input/structure.xyz \
    --template ../input/aimd.in.template \
    --temperature 600 \
    --timestep 0.5 \
    --nsteps 2000 \
    --output aimd_600K.in
```

### 2. Submit Job
```bash
# Modify SLURM template for your cluster
cp ../input/submit.slurm.template submit_aimd.slurm

# Edit partition, modules, paths in submit_aimd.slurm
sbatch submit_aimd.slurm
```

### 3. Monitor Progress
```bash
python monitor_aimd.py --job-id YOUR_JOB_ID
```

### 4. Process Output
```bash
python convert_to_xyz.py \
    --input aimd.out \
    --output aimd_trajectory.xyz \
    --skip 10  # Save every 10th frame
```

## Understanding the Output

### Energy Conservation
Good AIMD should show:
- Total energy drift < 0.001 eV/ps
- Temperature fluctuations ±5% of target
- No sudden energy jumps

### Trajectory Quality  
Check for:
- Reasonable atomic displacements
- No unphysical bond breaking
- Proper thermal motion at target temperature

## Troubleshooting

### Common Issues
1. **Job killed due to time limit**
   - Reduce `nsteps` or increase time in SLURM script
   
2. **SCF convergence problems**
   - Adjust `conv_thr` in input file
   - Check initial structure quality
   
3. **Memory issues**
   - Increase memory request in SLURM script
   - Reduce `ecutwfc` if necessary

### Getting Help
- Check QE documentation: https://www.quantum-espresso.org/
- Post issues with input files and error messages
