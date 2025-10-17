from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
from ase.io import read, write
from ase.build import make_supercell
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import random
from mace.calculators import mace_mp, MACECalculator
import jhu_colors
def run_mace_md(input_file, model_paths='mace_mp',
                device='cpu',  # or 'cuda' for GPU
                nx=1, ny=1, nz=1, temperature=300,
                steps=1000, interval=50):
    """
    Run a 300K MD simulation using the MACE calculator and save final plot.

    Parameters:
    - input_file (str): Path to the input XYZ file.
    - model_path (str): Path to the trained MACE model.
    - nx, ny, nz (int): Supercell size in x, y, z directions.
    - temperature (float): Target temperature (K).
    - steps (int): Number of MD steps.
    - interval (int): Frame write and data sample interval.
    """

    # Load and expand the structure
    atoms = read(input_file)
    print(f"Loaded structure from {input_file}")
    atoms = make_supercell(atoms, np.diag([nx, ny, nz]))
    print(f"Supercell dimensions: {nx}x{ny}x{nz}")

    if model_paths == 'mace_mp':
        # Set up MACE calculator with dispersion
        calculator = mace_mp()
    else:
        # Set up MACE calculator with specified model path
        calculator = MACECalculator(model_path=model_paths, device=device)
        print(f"Using MACE model from {model_paths}")
    atoms.calc = calculator

    # Initialize velocities
    random.seed(701)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    Stationary(atoms)
    ZeroRotation(atoms)

    # MD integrator
    dyn = Langevin(atoms,
                   timestep=1.0 * units.fs,
                   temperature_K=temperature,
                   friction=0.01)

    # Output file setup
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{base_name}_maceMD_eval_{nx}x{ny}x{nz}_T{temperature}K.xyz"
    os.system(f"rm -rfv {output_file}")  # Remove old file if exists

    # Storage for analysis
    time_fs = []
    temperature_list = []
    energy_per_atom = []

    def write_frame():
        atoms.write(output_file, append=True)
        time_fs.append(dyn.get_time() / units.fs)
        temperature_list.append(atoms.get_temperature())
        energy_per_atom.append(atoms.get_potential_energy() / len(atoms))

    dyn.attach(write_frame, interval=interval)

    # Run MD
    print("Starting MD simulation...")
    t0 = time.time()
    dyn.run(steps)
    t1 = time.time()
    print(f"MD finished in {(t1 - t0)/60:.2f} minutes.")
    print(f"Trajectory saved to: {output_file}")

    # Save plots
    fig, ax = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    # font Arial
    plt.rc('font', family='Arial')
    time_ps = np.array(time_fs) / 1000.0
    ax[0].plot(time_ps, energy_per_atom, color="blue")
    ax[0].set_ylabel("E (eV/atom)")
    ax[0].grid(True)

    ax[1].plot(time_ps, temperature_list, color="red")
    ax[1].set_ylabel("T (K)")
    ax[1].set_xlabel("Time (ps)")
    ax[1].grid(True)

    plt.tight_layout()
    plot_file = f"{base_name}_md_plot.pdf"
    plt.savefig(plot_file)
    print(f"Plot saved as {plot_file}")

if __name__ == "__main__":
    # input_file = 'relax.out'  # Replace with your structure file
    # model_path = 'mace_mp'  # Path to your MACE model
    device = 'cuda'  # or 'cpu' for CPU
    input_file = '../2L_octo_Cr2_v2_relax.extxyz'
    model_path = '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_600K/finetuned_MACE_multihead0804_compiled.model'
    
    nx, ny, nz = 5, 5, 1       # Supercell size
    temperature = 600          # in Kelvin
    steps = 200000               # MD steps
    interval = 200              # Save every 50 steps

    run_mace_md(input_file, nx=nx, ny=ny, nz=nz,
                model_paths=model_path, device=device,
                temperature=temperature, steps=steps, interval=interval)
