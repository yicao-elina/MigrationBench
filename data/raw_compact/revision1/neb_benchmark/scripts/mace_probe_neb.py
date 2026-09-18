#!/usr/bin/env python3
"""
Compares the Minimum Energy Pathway (MEP) for an atomic migration process
as predicted by various MACE models against a ground-truth DFT calculation.

This script performs the following steps:
1.  Loads a ground-truth Nudged Elastic Band (NEB) trajectory, which includes
    the initial, final, and several intermediate images of an atomic migration.
2.  Loads the corresponding ground-truth energy profile from a DFT calculation.
3.  Iterates through a dictionary of pre-trained MACE models.
4.  For each MACE model, it calculates the potential energy for every image
    along the ground-truth NEB path.
5.  Normalizes all energy profiles (both DFT and MACE predictions) so the
    energy of the initial state is zero.
6.  Generates a publication-quality plot comparing the energy profiles,
    clearly showing the migration barrier predicted by each model versus DFT.
7.  Calculates and prints the error in the predicted activation energy (the
    barrier height) for each model, providing a quantitative metric for model
    accuracy on this critical task.

Dependencies:
    pip install numpy matplotlib ase torch mace-torch jhu_colors
"""

import numpy as np
import matplotlib.pyplot as plt
import torch  # <-- Make sure this import is here!
from ase.io import read
from mace.calculators import MACECalculator
import jhu_colors  # Assumes jhu_colors is installed and sets style
# --- CONFIGURATION ---

# Paths to the trained MACE models you want to compare.
# The keys will be used as labels in the plot legend.
MODEL_PATHS = {
    "MACE Scratch": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE_models_l1_0802/mace_l1_0802_compiled.model",
    "MACE Naive FT": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_600K/finetuned_MACE_multihead0804_compiled.model",
    # "MACE Multi-Head FT": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE_models_multihead/mace0806_run-123_epoch-0.pt",
    # Add your other models (Strategy D, E) here when they are ready.
}

# Path to the ground-truth XYZ file containing the NEB images.
# This file should contain all images (initial, intermediates, final) in order.
GROUND_TRUTH_XYZ_PATH = "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/1.vdw_corr_DFT_D3/1-4/sb2te3.xyz"

# Path to the ground-truth data file with reaction coordinates and energies.
# Assumes a text file where columns 0 and 1 are reaction coordinate and energy.
GROUND_TRUTH_DAT_PATH = "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.neb/1.vdw_corr_DFT_D3/1-4/sb2te3.dat"

# Output filename for the plot
OUTPUT_PLOT_FILENAME = "neb_migration_barrier_comparison.png"

# --- MAIN SCRIPT ---

def compare_neb_barriers():
    """
    Main function to load data, run predictions, and generate the comparison plot.
    """
    # Set device (CPU is fine for single-point energy calculations)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load ground-truth DFT data
    try:
        dft_data = np.loadtxt(GROUND_TRUTH_DAT_PATH)
        reaction_coords = dft_data[:, 0]
        dft_energies = dft_data[:, 1]
        # Normalize so the first point is 0
        dft_energies_normalized = dft_energies - dft_energies[0]
        dft_barrier = np.max(dft_energies_normalized)
        print(f"Loaded ground-truth DFT data. Barrier height: {dft_barrier:.3f} eV")
    except FileNotFoundError:
        print(f"Error: Ground-truth data file not found at '{GROUND_TRUTH_DAT_PATH}'")
        return
    except Exception as e:
        print(f"Error loading DFT data file: {e}")
        return

    # 2. Load the NEB images from the XYZ file
    try:
        neb_images = read(GROUND_TRUTH_XYZ_PATH, index=":")
        print(f"Loaded {len(neb_images)} NEB images from '{GROUND_TRUTH_XYZ_PATH}'")
        if len(neb_images) != len(reaction_coords):
            print("Warning: Number of images in XYZ file does not match number of points in DAT file.")
    except FileNotFoundError:
        print(f"Error: Ground-truth XYZ file not found at '{GROUND_TRUTH_XYZ_PATH}'")
        return
    except Exception as e:
        print(f"Error loading NEB images: {e}")
        return

    # 3. Initialize the plot
    fig, ax = plt.subplots(figsize=(8, 6)) # Larger figure for clarity

    # Plot the DFT ground truth first
    ax.plot(
        reaction_coords,
        dft_energies_normalized,
        label="DFT (Ground Truth)",
        color=jhu_colors.get_jhu_color('Double Black'),
        marker='o',
        markersize=8,
        linestyle='-',
        linewidth=2.5,
        zorder=10 # Ensure it's on top
    )

    # Dictionary to store results and errors
    results = {}
    colors = ['Heritage Blue', 'Spirit Blue', 'Red', 'Orange', 'Homewood Green']
    markers = ['s', '^', 'v', 'D', 'P']

    # 4. Loop through each MACE model to evaluate
    for i, (model_name, model_path) in enumerate(MODEL_PATHS.items()):
        print(f"\n--- Evaluating model: {model_name} ---")
        try:
            # Load MACE model as an ASE calculator with compatibility settings
            # Try with default settings first, then with reduced features if needed
            try:
                calc = MACECalculator(
                    model_path, 
                    device=device,
                    default_dtype='float64',  # Explicitly set dtype
                )
            except Exception as e1:
                print(f"Standard loading failed, trying with minimal features...")
                # Try with minimal features for older models
                calc = MACECalculator(
                    model_path, 
                    device=device,
                    default_dtype='float64',
                    compute_stress=False,
                    compute_virials=False,
                )
                
        except Exception as e:
            print(f"Error loading model '{model_name}' from '{model_path}': {e}")
            continue

        # Calculate energy for each image in the NEB path
        model_energies = []
        for j, image in enumerate(neb_images):
            try:
                # Clone the atoms object to avoid modifying the original
                atoms_copy = image.copy()
                atoms_copy.calc = calc
                energy = atoms_copy.get_potential_energy()
                model_energies.append(energy)
            except RuntimeError as e:
                if "compute_edge_forces" in str(e):
                    # If we get the edge_forces error, recreate calculator without advanced features
                    print(f"Recreating calculator for {model_name} without advanced features...")
                    try:
                        # Create a minimal calculator
                        import torch
                        model = torch.jit.load(model_path, map_location=device)
                        
                        # Create a wrapper calculator that only computes energy
                        from ase.calculators.calculator import Calculator
                        
                        class MinimalMACECalculator(Calculator):
                            implemented_properties = ['energy']
                            
                            def __init__(self, model, device='cpu'):
                                super().__init__()
                                self.model = model
                                self.device = device
                                
                            def calculate(self, atoms=None, properties=['energy'], system_changes=None):
                                super().calculate(atoms, properties, system_changes)
                                
                                # Prepare input data for MACE model
                                from mace.data import AtomicData, config_from_atoms
                                from mace.tools import torch_geometric
                                
                                config = config_from_atoms(atoms)
                                data = AtomicData.from_config(config, z_table={51: 0, 52: 1, 24: 2}, cutoff=5.0)
                                batch = torch_geometric.dataloader.Collater(follow_batch=None, exclude_keys=None)([data])
                                
                                # Run model with minimal arguments
                                with torch.no_grad():
                                    out = self.model(
                                        batch.to_dict(),
                                        training=False,
                                        compute_force=False,
                                        compute_virials=False,
                                        compute_stress=False,
                                    )
                                
                                self.results['energy'] = out['energy'].item()
                        
                        calc = MinimalMACECalculator(model, device)
                        atoms_copy = image.copy()
                        atoms_copy.calc = calc
                        energy = atoms_copy.get_potential_energy()
                        model_energies.append(energy)
                        
                    except Exception as e2:
                        print(f"Failed to compute energy for image {j} with minimal calculator: {e2}")
                        # Use a placeholder value
                        model_energies.append(np.nan)
                else:
                    print(f"Error computing energy for image {j}: {e}")
                    model_energies.append(np.nan)

        # Check if we got valid energies
        if np.any(np.isnan(model_energies)):
            print(f"Warning: Some energies could not be computed for {model_name}")
            continue

        # Normalize the predicted energies
        model_energies_normalized = np.array(model_energies) - model_energies[0]
        model_barrier = np.max(model_energies_normalized)
        barrier_error = abs(model_barrier - dft_barrier)

        results[model_name] = {
            "energies": model_energies_normalized,
            "barrier": model_barrier,
            "error_eV": barrier_error,
            "error_percent": (barrier_error / dft_barrier) * 100 if dft_barrier != 0 else 0
        }

        # Plot the model's predicted energy profile
        ax.plot(
            reaction_coords,
            model_energies_normalized,
            label=f"{model_name} (Error: {barrier_error:.3f} eV)",
            color=jhu_colors.get_jhu_color(colors[i % len(colors)]),
            marker=markers[i % len(markers)],
            linestyle='--',
            linewidth=2
        )

    # 5. Finalize and save the plot
    ax.set_xlabel("Reaction Coordinate", fontsize=14)
    ax.set_ylabel("Relative Energy (eV)", fontsize=14)
    ax.set_title("NEB Migration Barrier: DFT vs. MACE Models", fontsize=16, weight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_FILENAME, dpi=300)
    print(f"\nSaved comparison plot to '{OUTPUT_PLOT_FILENAME}'")

    # 6. Print a summary of the results
    print("\n--- Barrier Height Comparison Summary ---")
    print(f"{'Model Name':<25} | {'Predicted Barrier (eV)':<25} | {'Error (eV)':<15} | {'Error (%)':<15}")
    print("-" * 85)
    print(f"{'DFT (Ground Truth)':<25} | {dft_barrier:<25.3f} | {'-':<15} | {'-':<15}")
    for model_name, data in results.items():
        print(f"{model_name:<25} | {data['barrier']:<25.3f} | {data['error_eV']:<15.3f} | {data['error_percent']:<15.2f}")
if __name__ == "__main__":
    compare_neb_barriers()
