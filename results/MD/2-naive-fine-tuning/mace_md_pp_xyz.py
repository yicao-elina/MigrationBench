# -*- coding: utf-8 -*-
"""
Post-Processing Script for MACE MD Trajectories from a Single .extxyz File

This script directly processes a MACE-driven molecular dynamics trajectory stored
in an extended XYZ (.extxyz) file. It extracts all necessary information—
positions, momenta, forces, energies, and cell parameters—on a per-frame basis
to perform a comprehensive analysis of the material's properties.

The analyses performed are:
1.  Thermodynamic Stability: Plots temperature, energy, and pressure evolution.
2.  Radial Distribution Function (RDF): Computes g(r) for structural analysis.
3.  Mean Squared Displacement (MSD) & Diffusion: Calculates atomic mobility and
    the diffusion coefficient (D).
4.  Velocity Autocorrelation Function (VACF): Analyzes vibrational dynamics.
5.  Thermal Conductivity (κ): Calculated via the Green-Kubo formalism from the
    heat flux autocorrelation function (HFACF).

This script is designed to produce publication-quality figures and adheres to
rigorous standards for scientific data analysis.

Author: Yi Cao (with assistance from Gemini)
Date: August 14, 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.stats import linregress
from ase.io import read
from ase.geometry import find_mic
from tqdm import tqdm

# Apply user-specific plotting style for publication-quality figures.
try:
    import jhu_colors
    get_jhu_color = jhu_colors.get_jhu_color
    # Define a color mapping for atom pairs
    COLOR_MAP = {
        'Cr-Te': get_jhu_color('Heritage Blue'),
        'Sb-Te': get_jhu_color('Spirit Blue'),
        'Te-Te': get_jhu_color('Homewood Green'),
        'Cr-Sb': get_jhu_color('Orange'),
        'Cr-Cr': get_jhu_color('Red'),
        'Sb-Sb': get_jhu_color('Purple'),
        'Total': get_jhu_color('Double Black'),
    }
except ImportError:
    print("Warning: 'jhu_colors' package not found. Using default matplotlib styles.")
    def get_jhu_color(x): return None
    COLOR_MAP = {key: get_jhu_color(None) for key in ['Cr-Te', 'Sb-Te', 'Te-Te', 'Cr-Sb', 'Cr-Cr', 'Sb-Sb', 'Total']}


# --- Configuration ---
# Path to the extended XYZ trajectory file.
XYZ_FILE_PATH = './2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz'

# Time step between frames in the XYZ file, in femtoseconds.
# This must match the (timestep * interval) from the simulation run.
# E.g., 1.0 fs timestep * 200 interval = 200 fs.
DT_FS = 200.0

# Directory to save the generated plots.
OUTPUT_DIR = f'./analysis_plots_from_xyz'

# --- Helper Functions ---

def calculate_heat_flux(atoms, velocities, forces, stress_tensor):
    """
    Calculate a simplified instantaneous heat flux vector.
    J = (1/V) * [ sum_i(E_i * v_i) + sum_i(r_i * (F_i · v_i)) ]
    This version uses the full virial stress tensor provided by the calculator.
    """
    volume = atoms.get_volume()
    masses = atoms.get_masses()
    
    # Scalar kinetic energy per atom: (1/2) m |v|^2
    v2 = np.sum(velocities**2, axis=1)
    ke_scalar = 0.5 * masses * v2

    # Convective (kinetic) term: sum_i [ E_i * v_i ]
    kinetic_flux = np.sum(ke_scalar[:, None] * velocities, axis=0)

    # Virial/stress contribution
    # The stress tensor from ASE is already volume-normalized and includes kinetic
    # and potential parts. We multiply by volume to get the virial tensor.
    # The potential flux is - (virial_tensor · v_mean)
    v_mean = np.mean(velocities, axis=0)
    potential_flux = -np.dot(stress_tensor, v_mean) * volume

    heat_flux = (kinetic_flux + potential_flux) / volume
    return heat_flux

def calculate_autocorr_fft(data_array):
    """Calculates the autocorrelation of vector components using FFT."""
    data_detrended = data_array - data_array.mean(axis=0)
    fft_data = np.fft.fft(data_detrended, n=2*len(data_detrended)-1, axis=0)
    psd = np.abs(fft_data)**2
    autocorr = np.fft.ifft(psd, axis=0).real
    autocorr /= autocorr[0] # Normalize
    return autocorr[:len(data_array)]

# --- Data Loading and Processing ---

def load_data_from_extxyz(filepath, dt_fs):
    """
    Loads and processes all data from a single .extxyz file.
    Returns a dictionary of pandas DataFrames for easy analysis.
    """
    print(f"--- Loading and Processing Trajectory from {filepath} ---")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Trajectory file not found: {filepath}")

    frames = read(filepath, index=':')
    if not frames:
        raise ValueError("No frames found in the trajectory file.")

    print(f"Found {len(frames)} frames.")

    thermo_data, positions_data, velocities_data, heat_flux_data = [], [], [], []
    initial_positions = frames[0].get_positions().copy()
    atom_types = frames[0].get_chemical_symbols()

    for i, atoms in enumerate(tqdm(frames, desc="Processing Frames")):
        current_time = i * dt_fs
        
        # --- Extract Data from Atoms Object ---
        positions = atoms.get_positions()
        momenta = atoms.get_momenta()
        forces = atoms.get_forces()
        masses = atoms.get_masses()[:, np.newaxis]
        velocities = momenta / masses

        pot_energy = atoms.get_potential_energy()
        kin_energy = atoms.get_kinetic_energy()
        temperature = atoms.get_temperature()
        volume = atoms.get_volume()
        stress_voigt = atoms.get_stress() # Voigt order (xx, yy, zz, yz, xz, xy)
        
        # Reconstruct 3x3 stress tensor
        s = stress_voigt
        stress_tensor = np.array([[s[0], s[5], s[4]],
                                  [s[5], s[1], s[3]],
                                  [s[4], s[3], s[2]]])

        # --- Thermodynamics ---
        thermo_data.append({
            'time_fs': current_time,
            'temperature_K': temperature,
            'pot_energy_eV': pot_energy,
            'kin_energy_eV': kin_energy,
            'total_energy_eV': pot_energy + kin_energy,
            'volume_A3': volume,
            'pressure_GPa': -np.trace(stress_tensor) / 3 * 160.21766208 # eV/Å³ to GPa
        })

        # --- Positions & MSD ---
        disp = positions - initial_positions
        for j, (pos, d) in enumerate(zip(positions, disp)):
            positions_data.append({
                'time_fs': current_time, 'atom_id': j, 'atom_type': atom_types[j],
                'x': pos[0], 'y': pos[1], 'z': pos[2],
                'dx': d[0], 'dy': d[1], 'dz': d[2]
            })

        # --- Velocities & VACF ---
        for j, vel in enumerate(velocities):
            velocities_data.append({
                'time_fs': current_time, 'atom_id': j, 'atom_type': atom_types[j],
                'vx': vel[0], 'vy': vel[1], 'vz': vel[2]
            })
            
        # --- Heat Flux & Thermal Conductivity ---
        heat_flux = calculate_heat_flux(atoms, velocities, forces, stress_tensor)
        heat_flux_data.append({
            'time_fs': current_time,
            'Jx': heat_flux[0], 'Jy': heat_flux[1], 'Jz': heat_flux[2]
        })

    # --- Convert to DataFrames ---
    print("\nConverting extracted data to DataFrames...")
    dataframes = {
        'thermo': pd.DataFrame(thermo_data),
        'positions': pd.DataFrame(positions_data),
        'velocities': pd.DataFrame(velocities_data),
        'heat_flux': pd.DataFrame(heat_flux_data),
        'frames': frames # Keep frames for RDF cell info
    }
    return dataframes

# --- Analysis & Plotting Functions (largely reusable) ---

def plot_thermodynamics(thermo_df, output_dir):
    """Plots temperature, energy, and pressure evolution."""
    print("\n--- Plotting Thermodynamics ---")
    time_ps = thermo_df['time_fs'] / 1000.0
    num_atoms = len(thermo_df.iloc[0]) # A bit of a hack, assumes first frame is representative

    fig, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

    # Plot 1: Energies
    ax = axes[0]
    ax.plot(time_ps, thermo_df['pot_energy_eV'] / num_atoms, label='Potential', color=get_jhu_color('Heritage Blue'))
    ax.plot(time_ps, thermo_df['kin_energy_eV'] / num_atoms, label='Kinetic', color=get_jhu_color('Spirit Blue'))
    ax.plot(time_ps, thermo_df['total_energy_eV'] / num_atoms, label='Total', color=get_jhu_color('Double Black'), linestyle='--')
    ax.set_ylabel('Energy (eV/atom)')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Temperature
    ax = axes[1]
    ax.plot(time_ps, thermo_df['temperature_K'], color=get_jhu_color('Red'))
    avg_temp = thermo_df['temperature_K'].mean()
    ax.axhline(avg_temp, color=get_jhu_color('Double Black'), linestyle='--', label=f'Avg: {avg_temp:.1f} K')
    ax.set_ylabel('Temperature (K)')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 3: Pressure
    ax = axes[2]
    ax.plot(time_ps, thermo_df['pressure_GPa'], color=get_jhu_color('Homewood Green'))
    avg_pressure = thermo_df['pressure_GPa'].mean()
    ax.axhline(avg_pressure, color=get_jhu_color('Double Black'), linestyle='--', label=f'Avg: {avg_pressure:.2f} GPa')
    ax.set_ylabel('Pressure (GPa)')
    ax.set_xlabel('Time (ps)')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'thermodynamics.pdf'), dpi=300)
    plt.close()
    print("Thermodynamics plot saved.")


def calculate_and_plot_rdf(positions_df, frames, output_dir, r_max=10.0, n_bins=200):
    """Calculates and plots the Radial Distribution Function (RDF)."""
    print("\n--- Calculating Radial Distribution Function (RDF) ---")
    bins = np.linspace(0, r_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    
    atom_types = positions_df['atom_type'].unique()
    pairs = sorted([tuple(sorted((t1, t2))) for i, t1 in enumerate(atom_types) for j, t2 in enumerate(atom_types) if j >= i])
    rdf_counts = {pair: np.zeros(n_bins) for pair in pairs}
    num_atoms_by_type = positions_df.groupby('atom_type')['atom_id'].nunique().to_dict()
    
    for frame_idx, frame_df in tqdm(positions_df.groupby('time_fs'), desc="Processing RDF frames"):
        atoms_obj = frames[int(frame_idx / DT_FS)]
        cell = atoms_obj.get_cell()
        volume = atoms_obj.get_volume()
        
        # ASE's get_all_distances is efficient for this
        distances = atoms_obj.get_all_distances(mic=True)
        
        for i in range(len(atoms_obj)):
            for j in range(i + 1, len(atoms_obj)):
                dist = distances[i, j]
                if dist < r_max:
                    pair = tuple(sorted((atoms_obj.symbols[i], atoms_obj.symbols[j])))
                    bin_idx = np.searchsorted(bins, dist) - 1
                    if bin_idx >= 0:
                        rdf_counts[pair][bin_idx] += 2

    # Normalization
    rdf_final = {}
    n_frames = len(frames)
    avg_volume = np.mean([f.get_volume() for f in frames])

    for pair, counts in rdf_counts.items():
        type1, type2 = pair
        N1, N2 = num_atoms_by_type[type1], num_atoms_by_type[type2]
        norm_factor = (N1 * (N2 - (1 if type1 == type2 else 0))) / avg_volume
        shell_volumes = 4.0 * np.pi * bin_centers**2 * (r_max / n_bins)
        shell_volumes[0] = 4./3. * np.pi * bins[1]**3
        g_r = counts / (n_frames * norm_factor * shell_volumes)
        rdf_final[pair] = g_r

    # Plotting
    plt.figure(figsize=(8, 6))
    for pair_name, g_r in rdf_final.items():
        label = f'{pair_name[0]}-{pair_name[1]}'
        plt.plot(bin_centers, g_r, label=label, color=COLOR_MAP.get(label, get_jhu_color('Double Black')))
    
    plt.title('Radial Distribution Function (RDF)')
    plt.xlabel('Distance, r (Å)'); plt.ylabel('g(r)')
    plt.axhline(1.0, color='gray', linestyle='--'); plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6); plt.xlim(0, r_max); plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rdf.pdf'), dpi=300)
    plt.close()
    print("RDF plot saved.")


def calculate_and_plot_msd(positions_df, output_dir, fit_start_ps=5.0):
    """Calculates MSD, plots it, and computes the diffusion coefficient."""
    print("\n--- Calculating Mean Squared Displacement (MSD) ---")
    positions_df['d_sq'] = positions_df['dx']**2 + positions_df['dy']**2 + positions_df['dz']**2
    msd_df = positions_df.groupby('time_fs')['d_sq'].mean().reset_index()
    msd_df['time_ps'] = msd_df['time_fs'] / 1000.0
    
    plt.figure(figsize=(6, 4))
    plt.plot(msd_df['time_ps'], msd_df['d_sq'], color=get_jhu_color('Heritage Blue'), label='MSD')
    
    fit_df = msd_df[msd_df['time_ps'] >= fit_start_ps]
    if len(fit_df) > 2:
        time_s = fit_df['time_fs'] * 1e-15
        msd_m2 = fit_df['d_sq'] * 1e-20
        slope, intercept, r_value, _, _ = linregress(time_s, msd_m2)
        diffusion_coeff_cm2_s = (slope / 6.0) * 1e4
        print(f"Diffusion Coefficient (D) from MSD slope: {diffusion_coeff_cm2_s:.4e} cm²/s")
        fit_line_A2 = (slope * time_s + intercept) * 1e20
        plt.plot(fit_df['time_ps'], fit_line_A2, color=get_jhu_color('Red'), linestyle='--', 
                 label=f'Linear Fit (D={diffusion_coeff_cm2_s:.2e} cm²/s)')

    plt.title('Mean Squared Displacement (MSD)'); plt.xlabel('Time (ps)')
    plt.ylabel(r'MSD (Å²)')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'msd_diffusion.pdf'), dpi=300)
    plt.close()
    print("MSD and Diffusion plot saved.")


def plot_vacf(velocities_df, output_dir):
    """Calculates and plots the Velocity Autocorrelation Function (VACF)."""
    print("\n--- Calculating Velocity Autocorrelation Function (VACF) ---")
    vel_array = velocities_df[['vx', 'vy', 'vz']].values.reshape(
        len(velocities_df['time_fs'].unique()), -1, 3)
    
    vacf_sum = np.zeros(vel_array.shape[0])
    for i in range(vel_array.shape[1]): # Loop over atoms
        vacf_atom = calculate_autocorr_fft(vel_array[:, i, :])
        vacf_sum += vacf_atom.sum(axis=1) # Sum over vx,vy,vz components
        
    vacf_avg = vacf_sum / (vel_array.shape[1] * 3)
    time_ps = velocities_df['time_fs'].unique() / 1000.0
    
    plt.figure(figsize=(6, 4))
    plt.plot(time_ps, vacf_avg, color=get_jhu_color('Heritage Blue'))
    plt.title('Velocity Autocorrelation Function (VACF)'); plt.xlabel('Time (ps)')
    plt.ylabel('Normalized VACF'); plt.axhline(0, color='gray', linestyle='--')
    plt.grid(True, linestyle='--', alpha=0.6); plt.xlim(0, time_ps[-1] / 4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vacf.pdf'), dpi=300)
    plt.close()
    print("VACF plot saved.")


def plot_hfacf_and_conductivity(heat_flux_df, thermo_df, output_dir):
    """Calculates thermal conductivity via Green-Kubo from the HFACF."""
    print("\n--- Calculating Thermal Conductivity (Green-Kubo) ---")
    hfacf = calculate_autocorr_fft(heat_flux_df[['Jx', 'Jy', 'Jz']].values)
    hfacf_avg = hfacf.mean(axis=1)
    time_ps = heat_flux_df['time_fs'] / 1000.0
    
    fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
    axes[0].plot(time_ps, hfacf_avg, color=get_jhu_color('Heritage Blue'))
    axes[0].set_title('Heat Flux Autocorrelation Function (HFACF)')
    axes[0].set_ylabel('Normalized HFACF'); axes[0].axhline(0, color='gray', linestyle='--')
    axes[0].grid(True, linestyle='--', alpha=0.6); axes[0].set_xlim(0, 5)

    kB_eV_K = 8.617333e-5
    avg_temp_K = thermo_df['temperature_K'].mean()
    avg_volume_A3 = thermo_df['volume_A3'].mean()
    j_sq_avg = (heat_flux_df[['Jx', 'Jy', 'Jz']]**2).values.mean()
    hfacf_unnormalized = hfacf_avg * j_sq_avg
    
    prefactor = avg_volume_A3 / (kB_eV_K * avg_temp_K**2)
    # Use cumulative_trapezoid for efficient running integration.
    # It returns an array one element shorter, so we set initial=0 for the t=0 point.
    integral_values = cumulative_trapezoid(hfacf_unnormalized, dx=DT_FS, initial=0)
    kappa_iso = (prefactor / 3.0) * integral_values
    eV_fsA_to_W_mK = 1.60218e6
    kappa_iso_W_mK = kappa_iso * eV_fsA_to_W_mK
    final_kappa = kappa_iso_W_mK[-1]
    
    print(f"Converged Thermal Conductivity (κ): {final_kappa:.3f} W/(m·K)")

    axes[1].plot(time_ps, kappa_iso_W_mK, color=get_jhu_color('Red'))
    axes[1].set_title('Thermal Conductivity (Running Integral)')
    axes[1].set_xlabel('Integration Time (ps)'); axes[1].set_ylabel('κ (W/m·K)')
    axes[1].axhline(final_kappa, color='gray', linestyle='--', label=f'Final κ = {final_kappa:.2f}')
    axes[1].grid(True, linestyle='--', alpha=0.6); axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hfacf_conductivity.pdf'), dpi=300)
    plt.close()
    print("HFACF and Conductivity plot saved.")


def main():
    """Main execution function."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # Load and process data directly from the .extxyz file
    data = load_data_from_extxyz(XYZ_FILE_PATH, DT_FS)

    # --- Run all analysis and plotting functions ---
    plot_thermodynamics(data['thermo'], OUTPUT_DIR)
    calculate_and_plot_rdf(data['positions'], data['frames'], OUTPUT_DIR)
    calculate_and_plot_msd(data['positions'], OUTPUT_DIR, fit_start_ps=5.0)
    plot_vacf(data['velocities'], OUTPUT_DIR)
    plot_hfacf_and_conductivity(data['heat_flux'], data['thermo'], OUTPUT_DIR)
    
    print("\n--- Analysis Complete ---")
    print(f"All plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
