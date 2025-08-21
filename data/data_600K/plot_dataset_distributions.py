import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ase.io import read
import jhu_colors
from jhu_colors import get_jhu_color

# --- ⚙️ Configuration ---
# Update with your final reference energies if they differ.
ISOLATED_ATOM_ENERGIES = {
    'Cr': -2270.518789079322, # SG15
    'Sb': -1908.633351763048, # SG15
    'Te': -2369.398259284361 # SG15
}

# Define the dataset files and their corresponding labels and colors
DATASET_FILES = {
    'Train':      {'path': 'train.xyz', 'color': get_jhu_color('Heritage Blue')},
    'Validation': {'path': 'valid.xyz', 'color': get_jhu_color('Spirit Blue')},
    'Test':       {'path': 'test.xyz',  'color': get_jhu_color('Red')}
}

# Directory to save the output plots
OUTPUT_DIR = "dataset_analysis_plots"

def load_and_process_data(filepath: str) -> (np.ndarray, np.ndarray):
    """
    Reads an XYZ file and calculates the per-atom energy and per-atom force
    magnitudes for all frames.

    Args:
        filepath: Path to the .xyz file.

    Returns:
        A tuple containing two numpy arrays:
        - energies_per_atom: The atomic energy for each frame.
        - all_force_magnitudes: A flattened array of all force magnitudes
                                from all atoms in all frames.
    """
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}. Skipping.")
        return np.array([]), np.array([])

    print(f"Processing '{filepath}'...")
    frames = read(filepath, index=":")
    if not isinstance(frames, list):
        frames = [frames]

    energies_per_atom = []
    all_force_magnitudes = []

    for atoms in frames:
        # Calculate E_atm per atom
        e_total = atoms.get_potential_energy()
        e0_sum = sum(ISOLATED_ATOM_ENERGIES.get(sym, 0) for sym in atoms.get_chemical_symbols())
        num_atoms = len(atoms)
        if num_atoms > 0:
            energies_per_atom.append((e_total - e0_sum) / num_atoms)

        # Calculate force magnitudes
        forces = atoms.get_forces()
        magnitudes = np.linalg.norm(forces, axis=1)
        all_force_magnitudes.extend(magnitudes)

    return np.array(energies_per_atom), np.array(all_force_magnitudes)

def plot_distributions():
    """
    Generates and saves the energy and force distribution plots.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_data = {}
    for label, info in DATASET_FILES.items():
        energies, forces = load_and_process_data(info['path'])
        all_data[label] = {'energies': energies, 'forces': forces}

    # --- 1. Plot Energy Distribution ---
    fig_e, ax_e = plt.subplots(figsize=(6, 4))
    for label, data in all_data.items():
        if data['energies'].size > 0:
            sns.kdeplot(data['energies'], ax=ax_e, color=DATASET_FILES[label]['color'],
                        label=label, fill=True, alpha=0.1, linewidth=2)
    
    ax_e.set_title('Distribution of Atomic Energy per Atom', fontsize=14)
    ax_e.set_xlabel('$E^{\\rm atm}$ (eV/atom)', fontsize=12)
    ax_e.set_ylabel('Density', fontsize=12)
    ax_e.legend(title='Dataset')
    ax_e.grid(axis='y', linestyle='--', alpha=0.7)
    # Set a sensible x-limit to focus on the main distribution 
    ax_e.set_xlim(-5, max(np.percentile(all_data['Train']['energies'], 90), 0.5))
    plt.tight_layout()
    energy_plot_path = os.path.join(OUTPUT_DIR, "energy_distribution_comparison.png")
    plt.savefig(energy_plot_path, dpi=300)
    print(f"\n✅ Energy distribution plot saved to: {energy_plot_path}")
    plt.close(fig_e)

    # --- 2. Plot Force Distribution ---
    fig_f, ax_f = plt.subplots(figsize=(6, 4))
    for label, data in all_data.items():
        if data['forces'].size > 0:
            sns.kdeplot(data['forces'], ax=ax_f, color=DATASET_FILES[label]['color'],
                        label=label, fill=True, alpha=0.1, linewidth=2)

    ax_f.set_title('Distribution of Atomic Force Magnitudes', fontsize=14)
    ax_f.set_xlabel('Force Magnitude (eV/Å)', fontsize=12)
    ax_f.set_ylabel('Density', fontsize=12)
    ax_f.legend(title='Dataset')
    ax_f.grid(axis='y', linestyle='--', alpha=0.7)
    # Set a sensible x-limit to focus on the main distribution
    ax_f.set_xlim(0, max(np.percentile(all_data['Train']['forces'], 90), 5))
    plt.tight_layout()
    force_plot_path = os.path.join(OUTPUT_DIR, "force_distribution_comparison.png")
    plt.savefig(force_plot_path, dpi=300)
    print(f"✅ Force distribution plot saved to: {force_plot_path}")
    plt.close(fig_f)

if __name__ == "__main__":
    plot_distributions()