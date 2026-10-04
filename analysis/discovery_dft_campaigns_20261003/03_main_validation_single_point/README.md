# Discovery main DFT single-point validation

This is an isolated Discovery package for the main MigrationBench candidate
campaign.  It preserves the full 80-row candidate manifest and the 20
deterministic wave-1 QE inputs.  The remaining candidates stay pending until
the Discovery smoke gate passes; MD candidates with incomplete provenance are
not submitted.

Discovery-specific method choices are recorded explicitly:

- QE 7.5 H200 container: `/projects/pclancy3_lab/qe75-h200-v2.sif` (current Discovery default; the old 7.3.1 wrapper is historical only)
- launcher: `srun --export=ALL singularity exec --nv --cleanenv`, then the container's HPC-X/OpenMPI `mpirun` with `OPAL_PREFIX=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/comm_libs/12.8/hpcx/hpcx-2.22.1/ompi`
- single-point resources: `--ntasks-per-node=1 --cpus-per-task=11 --gres=gpu:1`, `OMP_NUM_THREADS=11`, `OMP_STACKSIZE=2G`
- pseudopotentials: `/home/ycao73/ONCVPseudoPack/SG15_v1.2/UPF/PBE/`
- non-magnetic QE input; no `nspin` or magnetization keys
- `calculation='scf'`, `ecutwfc=70`, `ecutrho=280`, DFT-D3, 2x2x2 k mesh
- The main wave-1 inputs use `degauss=0.0146997236`, matching the main
  campaign method. The earlier Discovery-reference smoke variant with
  `degauss=0.005` was discarded before submission.
- A separate sensitivity input is provided at
  `qe_inputs_sensitivity/degauss_0.05/cand_0001.scf.in`; it keeps the same
  geometry and all other settings as `cand_0001`, changing only `degauss` to
  `0.05`. It must be submitted only as a paired sensitivity calculation,
  never merged into the main-campaign results.
