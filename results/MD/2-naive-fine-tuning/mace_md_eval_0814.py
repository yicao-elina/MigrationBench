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
import pandas as pd

def calculate_heat_flux(atoms, velocities, forces, stress_tensor):
    """
    Calculate a simplified instantaneous heat flux vector for thermal conductivity estimates.
    J = (1/V) * sum_i [ E_i * v_i ]  -  (σ · ⟨v⟩)        (heuristic virial contribution)
    where E_i uses kinetic energy only (1/2 m v^2). Pairwise potential contributions are omitted.
    """
    volume = atoms.get_volume()
    masses = atoms.get_masses()

    # Ensure velocities array exists and has the correct shape
    if velocities is None or len(velocities) != len(masses):
        return np.zeros(3)

    # Scalar kinetic energy per atom: (1/2) m |v|^2
    v2 = np.sum(velocities**2, axis=1)               # (N,)
    ke_scalar = 0.5 * masses * v2                    # (N,)

    # Kinetic contribution: sum_i [ E_i * v_i ]
    kinetic_flux = np.sum(ke_scalar[:, None] * velocities, axis=0)  # (3,)

    # Virial/stress contribution (very approximate)
    v_mean = np.mean(velocities, axis=0)             # (3,)
    potential_flux = -np.dot(stress_tensor[:3, :3], v_mean) * volume  # (3,)

    heat_flux = (kinetic_flux + potential_flux) / volume
    return heat_flux

def calculate_com_velocity(atoms):
    """Calculate center-of-mass velocity."""
    masses = atoms.get_masses()
    velocities = atoms.get_velocities()
    if velocities is None:
        return np.zeros(3)
    total_mass = np.sum(masses)
    com_velocity = np.sum(masses[:, np.newaxis] * velocities, axis=0) / total_mass
    return com_velocity

def normalize_device(device: str) -> str:
    if device is None:
        return "cpu"
    d = device.lower()
    if d == "cuda":
        return "cuda:0"
    return device

def run_mace_md(input_file, model_paths='mace_mp',
                device='cpu',  # or 'cuda' / 'cuda:0' for GPU
                nx=1, ny=1, nz=1, temperature=300.0,
                steps=1000, interval=50):
    """
    Run MD simulation using a MACE calculator and save comprehensive data for analysis.

    Parameters:
    - input_file (str): Path to the input (EX)XYZ file.
    - model_paths (str): Path to a trained MACE model (.model) or 'mace_mp'.
    - device (str): 'cpu', 'cuda', or 'cuda:0'.
    - nx, ny, nz (int): Supercell replication factors.
    - temperature (float): Target temperature (K).
    - steps (int): Number of MD steps.
    - interval (int): Sampling interval in MD steps.
    """

    # Load and expand the structure
    atoms = read(input_file)
    print(f"Loaded structure from {input_file}")
    atoms = make_supercell(atoms, np.diag([nx, ny, nz]))
    print(f"Supercell dimensions: {nx}x{ny}x{nz}")
    print(f"Total atoms: {len(atoms)}")

    # Set up MACE calculator
    if model_paths == 'mace_mp':
        calculator = mace_mp()
        print("Using mace_mp pretrained calculator.")
    else:
        dev = normalize_device(device)
        calculator = MACECalculator(model_path=model_paths, device=dev)
        print(f"Using MACE model from {model_paths} on device {dev}")
    atoms.calc = calculator

    # Initialize velocities reproducibly
    random.seed(701)
    np.random.seed(701)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)     # remove linear COM momentum
    ZeroRotation(atoms)   # remove angular momentum

    # MD integrator (Langevin; friction is in 1/fs)
    dyn = Langevin(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=temperature,
        friction=0.01
    )

    # Output file setup
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{base_name}_maceMD_eval_{nx}x{ny}x{nz}_T{int(temperature)}K.xyz"
    try:
        os.remove(output_file)
        print(f"Removed existing trajectory: {output_file}")
    except FileNotFoundError:
        pass

    # CSV file names for different properties
    thermo_csv    = f"{base_name}_thermodynamics_{nx}x{ny}x{nz}_T{int(temperature)}K.csv"
    positions_csv = f"{base_name}_positions_{nx}x{ny}x{nz}_T{int(temperature)}K.csv"
    velocities_csv= f"{base_name}_velocities_{nx}x{ny}x{nz}_T{int(temperature)}K.csv"
    forces_csv    = f"{base_name}_forces_{nx}x{ny}x{nz}_T{int(temperature)}K.csv"
    stress_csv    = f"{base_name}_stress_{nx}x{ny}x{nz}_T{int(temperature)}K.csv"
    heat_flux_csv = f"{base_name}_heat_flux_{nx}x{ny}x{nz}_T{int(temperature)}K.csv"

    # Initialize data storage
    thermo_data, positions_data, velocities_data, forces_data = [], [], [], []
    stress_data, heat_flux_data = [], []

    # Store initial positions for MSD calculation
    initial_positions = atoms.get_positions().copy()

    # Atom types for labeling
    atom_types = atoms.get_chemical_symbols()
    unique_types = sorted(set(atom_types))
    print(f"Atom types: {unique_types}")

    def write_frame():
        """Collect and write all data at each sampling interval."""
        # Append current configuration to trajectory
        write(output_file, atoms, append=True)

        # Time bookkeeping (ASE internal time divided by fs → femtoseconds)
        current_time = dyn.get_time() / units.fs

        # Current properties
        temp = atoms.get_temperature()
        pot_energy = atoms.get_potential_energy()
        kin_energy = atoms.get_kinetic_energy()
        total_energy = pot_energy + kin_energy
        pot_energy_per_atom = pot_energy / len(atoms)
        kin_energy_per_atom = kin_energy / len(atoms)
        volume = atoms.get_volume()

        # Kinematics and forces
        positions = atoms.get_positions()
        velocities = atoms.get_velocities()
        if velocities is None:
            # Should not happen after initialization; keep safe default
            velocities = np.zeros_like(positions)
        forces = atoms.get_forces()

        # Stress tensor (Voigt order: xx, yy, zz, yz, xz, xy) in eV/Å^3
        try:
            stress = atoms.get_stress()  # shape (6,)
        except Exception:
            stress = np.zeros(6)

        # Build full 3x3 stress tensor
        stress_tensor = np.zeros((3, 3))
        stress_tensor[0, 0] = stress[0]  # xx
        stress_tensor[1, 1] = stress[1]  # yy
        stress_tensor[2, 2] = stress[2]  # zz
        stress_tensor[1, 2] = stress_tensor[2, 1] = stress[3]  # yz
        stress_tensor[0, 2] = stress_tensor[2, 0] = stress[4]  # xz
        stress_tensor[0, 1] = stress_tensor[1, 0] = stress[5]  # xy

        # Heat flux (very approximate without pairwise force decomposition)
        heat_flux = calculate_heat_flux(atoms, velocities, forces, stress_tensor)

        # Center of mass velocity (to monitor momentum drift)
        com_velocity = calculate_com_velocity(atoms)

        # Store thermodynamic data
        thermo_data.append({
            'time_fs': current_time,
            'temperature_K': temp,
            'pot_energy_eV': pot_energy,
            'pot_energy_per_atom_eV': pot_energy_per_atom,
            'kin_energy_eV': kin_energy,
            'kin_energy_per_atom_eV': kin_energy_per_atom,
            'total_energy_eV': total_energy,
            'volume_A3': volume,
            'com_vx': com_velocity[0],
            'com_vy': com_velocity[1],
            'com_vz': com_velocity[2]
        })

        # Store positions (for RDF/MSD)
        disp = positions - initial_positions
        for i, (pos, dpos, atom_type) in enumerate(zip(positions, disp, atom_types)):
            positions_data.append({
                'time_fs': current_time,
                'atom_id': i,
                'atom_type': atom_type,
                'x': pos[0], 'y': pos[1], 'z': pos[2],
                'dx_from_init': dpos[0],
                'dy_from_init': dpos[1],
                'dz_from_init': dpos[2]
            })

        # Store velocities (for VACF/diffusion)
        for i, (vel, atom_type) in enumerate(zip(velocities, atom_types)):
            velocities_data.append({
                'time_fs': current_time,
                'atom_id': i,
                'atom_type': atom_type,
                'vx': vel[0], 'vy': vel[1], 'vz': vel[2]
            })

        # Store forces
        for i, (force, atom_type) in enumerate(zip(forces, atom_types)):
            forces_data.append({
                'time_fs': current_time,
                'atom_id': i,
                'atom_type': atom_type,
                'fx': force[0], 'fy': force[1], 'fz': force[2]
            })

        # Store stress (and simple pressure proxy)
        stress_data.append({
            'time_fs': current_time,
            'stress_xx': stress[0],
            'stress_yy': stress[1],
            'stress_zz': stress[2],
            'stress_yz': stress[3],
            'stress_xz': stress[4],
            'stress_xy': stress[5],
            'pressure_eV_per_A3': -np.mean(stress[:3])  # −(trace/3)
        })

        # Store heat flux
        heat_flux_data.append({
            'time_fs': current_time,
            'Jx': heat_flux[0],
            'Jy': heat_flux[1],
            'Jz': heat_flux[2],
            'J_magnitude': float(np.linalg.norm(heat_flux))
        })

    # Attach the data collection function
    dyn.attach(write_frame, interval=interval)

    # Run MD
    print("Starting MD simulation...")
    t0 = time.time()
    dyn.run(steps)
    t1 = time.time()
    print(f"MD finished in {(t1 - t0)/60:.2f} minutes.")
    print(f"Trajectory saved to: {output_file}")

    # Save all CSV files
    print("\nSaving analysis data to CSV files...")

    # Thermodynamics
    pd.DataFrame(thermo_data).to_csv(thermo_csv, index=False)
    print(f"Thermodynamics data saved to: {thermo_csv}")

    # Positions (can be large)
    if len(positions_data) < 1_000_000:
        pd.DataFrame(positions_data).to_csv(positions_csv, index=False)
        print(f"Positions data saved to: {positions_csv}")
    else:
        print(f"Positions data too large ({len(positions_data)} rows), skipping CSV save")

    # Velocities
    if len(velocities_data) < 1_000_000:
        pd.DataFrame(velocities_data).to_csv(velocities_csv, index=False)
        print(f"Velocities data saved to: {velocities_csv}")
    else:
        print(f"Velocities data too large ({len(velocities_data)} rows), skipping CSV save")

    # Forces
    if len(forces_data) < 1_000_000:
        pd.DataFrame(forces_data).to_csv(forces_csv, index=False)
        print(f"Forces data saved to: {forces_csv}")
    else:
        print(f"Forces data too large ({len(forces_data)} rows), skipping CSV save")

    # Stress
    pd.DataFrame(stress_data).to_csv(stress_csv, index=False)
    print(f"Stress data saved to: {stress_csv}")

    # Heat flux
    pd.DataFrame(heat_flux_data).to_csv(heat_flux_csv, index=False)
    print(f"Heat flux data saved to: {heat_flux_csv}")

    # Create summary plots
    plt.rc('font', family='Arial')
    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    thermo_df = pd.DataFrame(thermo_data)
    time_ps = thermo_df['time_fs'].values / 1000.0  # fs → ps

    # Energy plot
    ax[0].plot(time_ps, thermo_df['pot_energy_per_atom_eV'], label='Potential')
    ax[0].plot(time_ps, thermo_df['kin_energy_per_atom_eV'], label='Kinetic')
    ax[0].plot(time_ps,
               thermo_df['pot_energy_per_atom_eV'] + thermo_df['kin_energy_per_atom_eV'],
               label='Total', linestyle='--')
    ax[0].set_ylabel("Energy (eV/atom)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Temperature plot
    ax[1].plot(time_ps, thermo_df['temperature_K'])
    ax[1].axhline(y=temperature, linestyle='--', alpha=0.5)
    ax[1].set_ylabel("Temperature (K)")
    ax[1].grid(True, alpha=0.3)

    # Pressure (from stress) or Volume
    if len(stress_data) > 0 and 'pressure_eV_per_A3' in pd.DataFrame(stress_data).columns:
        stress_df = pd.DataFrame(stress_data)
        ax[2].plot(time_ps, stress_df['pressure_eV_per_A3'])
        ax[2].set_ylabel("Pressure (eV/Å³)")
    else:
        ax[2].plot(time_ps, thermo_df['volume_A3'])
        ax[2].set_ylabel("Volume (Å³)")

    ax[2].set_xlabel("Time (ps)")
    ax[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = f"{base_name}_md_analysis_{nx}x{ny}x{nz}_T{int(temperature)}K.pdf"
    plt.savefig(plot_file, dpi=150)
    print(f"\nAnalysis plot saved as: {plot_file}")

    # Print summary statistics
    print("\n--- Simulation Summary ---")
    print(f"Average temperature: {thermo_df['temperature_K'].mean():.1f} ± {thermo_df['temperature_K'].std():.1f} K")
    print(f"Average potential energy: {thermo_df['pot_energy_per_atom_eV'].mean():.4f} ± {thermo_df['pot_energy_per_atom_eV'].std():.4f} eV/atom")
    print(f"Average volume: {thermo_df['volume_A3'].mean():.1f} ± {thermo_df['volume_A3'].std():.1f} Å³")

    return {
        'thermo_csv': thermo_csv,
        'positions_csv': positions_csv if len(positions_data) < 1_000_000 else None,
        'velocities_csv': velocities_csv if len(velocities_data) < 1_000_000 else None,
        'forces_csv': forces_csv if len(forces_data) < 1_000_000 else None,
        'stress_csv': stress_csv,
        'heat_flux_csv': heat_flux_csv
    }

if __name__ == "__main__":
    device = 'cuda'  # or 'cpu' for CPU
    input_file = '../2L_octo_Cr2_v2_relax.extxyz'
    model_path = '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_600K/finetuned_MACE_multihead0804_compiled.model'

    nx, ny, nz = 5, 5, 1       # Supercell size
    temperature = 600          # K
    steps = 2000               # MD steps
    interval = 5            # sample every 200 steps (consistent with comment)

    csv_files = run_mace_md(
        input_file,
        nx=nx, ny=ny, nz=nz,
        model_paths=model_path,
        device=device,
        temperature=temperature,
        steps=steps,
        interval=interval
    )

    print("\n--- Generated CSV files for post-processing ---")
    for key, filepath in csv_files.items():
        if filepath and os.path.exists(filepath):
            print(f"{key}: {filepath}")
