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
import pandas as pd
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
    csv_file = f"{base_name}_maceMD_eval_{nx}x{ny}x{nz}_T{temperature}K_data.csv"
    os.system(f"rm -rfv {output_file}")  # Remove old file if exists
    os.system(f"rm -rfv {csv_file}")  # Remove old CSV file if exists

    # Storage for analysis
    time_fs = []
    temperature_list = []
    energy_per_atom = []
    total_energy = []
    max_force = []
    mean_force = []
    step_list = []

    def write_frame():
        # Get current temperature and add it to atoms.info
        current_temp = atoms.get_temperature()
        atoms.info['temperature'] = current_temp
        atoms.info['time_fs'] = dyn.get_time() / units.fs
        atoms.info['step'] = dyn.nsteps
        
        # Write frame with temperature info
        atoms.write(output_file, append=True)
        
        # Calculate forces and energies
        forces = atoms.get_forces()
        force_magnitudes = np.linalg.norm(forces, axis=1)
        
        # Store data for analysis
        step_list.append(dyn.nsteps)
        time_fs.append(dyn.get_time() / units.fs)
        temperature_list.append(current_temp)
        total_energy.append(atoms.get_potential_energy())
        energy_per_atom.append(atoms.get_potential_energy() / len(atoms))
        max_force.append(np.max(force_magnitudes))
        mean_force.append(np.mean(force_magnitudes))

    dyn.attach(write_frame, interval=interval)

    # Run MD
    print("Starting MD simulation...")
    t0 = time.time()
    dyn.run(steps)
    t1 = time.time()
    print(f"MD finished in {(t1 - t0)/60:.2f} minutes.")
    print(f"Trajectory saved to: {output_file}")

    # Save data to CSV
    df = pd.DataFrame({
        'step': step_list,
        'time_fs': time_fs,
        'temperature_K': temperature_list,
        'total_energy_eV': total_energy,
        'energy_per_atom_eV': energy_per_atom,
        'max_force_eV_A': max_force,
        'mean_force_eV_A': mean_force
    })
    df.to_csv(csv_file, index=False)
    print(f"Data saved to: {csv_file}")

    # Save plots with enhanced styling
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    # font Arial
    plt.rc('font', family='Arial')
    time_ps = np.array(time_fs) / 1000.0
    
    # Energy plot
    ax[0].plot(time_ps, energy_per_atom, color="blue", linewidth=1.5)
    ax[0].set_ylabel("E (eV/atom)", fontsize=12)
    ax[0].grid(True, alpha=0.3)
    ax[0].set_title(f"MD Simulation: {nx}x{ny}x{nz} supercell at {temperature}K", fontsize=14)
    
    # Temperature plot
    ax[1].plot(time_ps, temperature_list, color="red", linewidth=1.5)
    ax[1].axhline(y=temperature, color='black', linestyle='--', alpha=0.5, label=f'Target: {temperature}K')
    ax[1].set_ylabel("T (K)", fontsize=12)
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()
    
    # Force plot
    ax[2].plot(time_ps, max_force, color="green", linewidth=1.5, label='Max force')
    ax[2].plot(time_ps, mean_force, color="orange", linewidth=1.5, label='Mean force')
    ax[2].set_ylabel("Force (eV/Å)", fontsize=12)
    ax[2].set_xlabel("Time (ps)", fontsize=12)
    ax[2].grid(True, alpha=0.3)
    ax[2].legend()

    plt.tight_layout()
    plot_file = f"{base_name}_md_plot.pdf"
    plt.savefig(plot_file, dpi=300)
    print(f"Plot saved as {plot_file}")
    
    # Print summary statistics
    print("\n--- MD Summary Statistics ---")
    print(f"Average temperature: {np.mean(temperature_list):.2f} ± {np.std(temperature_list):.2f} K")
    print(f"Average energy/atom: {np.mean(energy_per_atom):.4f} ± {np.std(energy_per_atom):.4f} eV")
    print(f"Average max force: {np.mean(max_force):.4f} ± {np.std(max_force):.4f} eV/Å")

if __name__ == "__main__":
    # input_file = 'relax.out'  # Replace with your structure file
    # model_path = 'mace_mp'  # Path to your MACE model
    device = 'cuda'  # or 'cpu' for CPU
    input_file = '../../2L_octo_Cr2_v2_relax.extxyz'
    model_path = "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_Multi_T/finetuned_MACE_multihead0804.model"
    
    nx, ny, nz = 5, 5, 1       # Supercell size
    temperature = 600          # in Kelvin
    steps = 200000             # MD steps
    interval = 200             # Save every 200 steps

    run_mace_md(input_file, nx=nx, ny=ny, nz=nz,
                model_paths=model_path, device=device,
                temperature=temperature, steps=steps, interval=interval)