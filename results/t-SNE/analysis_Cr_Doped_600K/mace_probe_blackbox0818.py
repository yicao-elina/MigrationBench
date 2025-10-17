#!/usr/bin/env python3
"""
Black-box Latent Space Probe for MACE Models (Multi-System Version)

This script analyzes and compares the learned representations of MACE models
across multiple, user-defined systems or simulation conditions. It operates by
extracting observable outputs (energies, forces) and structural descriptors
from MD trajectories, then visualizes the resulting feature space to diagnose
differences between training paradigms.

Workflow for each defined system:
1.  Loads molecular dynamics trajectories for a set of specified models.
2.  Extracts a feature vector for each frame, including:
    - Predicted energies and atomic forces.
    - Radial distribution histograms for structural information.
    - SOAP descriptors for local chemical environments.
3.  Performs dimensionality reduction on the feature space using t-SNE.
4.  Quantifies the separation of model representations using silhouette scores.
5.  Generates and saves a set of comparison plots to a unique directory:
    - t-SNE embeddings (individual models and a cross-model overlay).
    - Bar plot of silhouette scores to compare clustering quality.

This modular design allows for systematic, reproducible benchmarking across
diverse materials science applications.

Dependencies:
    pip install numpy matplotlib scikit-learn ase dscribe jhu_colors
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, silhouette_samples, r2_score
from itertools import cycle

# Import JHU color standards for publication-quality figures.
try:
    import jhu_colors
    get_jhu_color = jhu_colors.get_jhu_color
except ImportError:
    print("Warning: 'jhu_colors' package not found. Using default matplotlib styles.")
    def get_jhu_color(x): return None

# ------------------------
# CONFIGURATION
# ------------------------
# Define all comparison sets here. The script will loop through each one.
COMPARISON_SETS = {
    "Cr_Doped_Sb2Te3_600K": {
        "output_dir": "analysis_Cr_Doped_600K",
        "models": {
    'Foundation': '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/0-foundation-mace_mp/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz',
    'Scratch': '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/1-from-scratch/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz',
    'FT - 600K': '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/2-naive-fine-tuning/temp/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz',
    'FT - Multi-T': '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/3-multi_T-fine-tuning/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz'
        }
    },
    # --- Example of a second comparison set ---
    # "Another_System_300K": {
    #     "output_dir": "analysis_Another_System_300K",
    #     "models": {
    #         "Model_A": "/path/to/your/model_A.xyz",
    #         "Model_B": "/path/to/your/model_B.xyz"
    #     }
    # }
}

# --- Global Parameters ---
FRAME_SAMPLING_RATE = 10  # Process every Nth frame for efficiency
CUTOFF = 5.0
SOAP_NMAX, SOAP_LMAX = 4, 4
PCA_DIM = 10
FINITE_DIFF_DELTA = 1e-3

# ------------------------
# UTILITY FUNCTIONS
# ------------------------
def parse_xyz(filepath, frame_step):
    """
    Robustly parses an XYZ file to extract energies, forces, and ASE Atoms objects.
    It first attempts to use ASE's built-in parser. If it fails to find an energy
    key, it falls back to manually parsing the header line of each frame.
    """
    print(f"  > Reading frames from {os.path.basename(filepath)}...")
    try:
        atoms_list = read(filepath, index=f"::{frame_step}")
        if not atoms_list:
            raise IOError("No frames could be read.")
    except Exception as e:
        print(f"Error reading file with ASE: {e}")
        return None, None, None

    energies, forces = [], []
    energy_keys_to_try = ['energy', 'MACE_energy', 'free_energy', 'potential_energy']
    found_key = None

    # --- Attempt 1: Use ASE's info dictionary ---
    for atoms in atoms_list:
        energy_found = False
        if found_key and found_key in atoms.info:
            energies.append(atoms.info[found_key])
            energy_found = True
        else:
            for key in energy_keys_to_try:
                if key in atoms.info:
                    energies.append(atoms.info[key])
                    found_key = key
                    energy_found = True
                    break
        if not energy_found:
            energies.append(None) # Mark as missing if not found

        forces.append(atoms.get_forces().flatten())

    # --- Attempt 2: Fallback to manual parsing if ASE failed ---
    if all(e is None for e in energies):
        print("  > ASE parsing failed to find energy. Falling back to manual parsing...")
        energies = [] # Reset energies list
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            frame_start_indices = [i for i, line in enumerate(lines) if line.strip().isdigit() and i + 1 < len(lines) and "Lattice" in lines[i+1]]
            sampled_indices = frame_start_indices[::frame_step]

            if len(sampled_indices) != len(atoms_list):
                 print(f"Warning: Mismatch in frame count between ASE ({len(atoms_list)}) and manual parser ({len(sampled_indices)}). Results may be inconsistent.")

            for i in sampled_indices:
                header_line = lines[i+1]
                energy_match = re.search(r"(?:energy|free_energy)\s*=\s*([-0-9.eE]+)", header_line)
                if energy_match:
                    energies.append(float(energy_match.group(1)))
                else:
                    raise ValueError(f"Manual parser could not find energy in header: {header_line.strip()}")
            found_key = "manual"
        except Exception as e:
            print(f"Error during manual parsing: {e}")
            return None, None, None

    if len(energies) != len(atoms_list):
        print(f"Error: Final energy count ({len(energies)}) does not match frame count ({len(atoms_list)}).")
        return None, None, None

    energies = np.array(energies, dtype=float)
    forces = np.array(forces)
    print(f"  > Successfully parsed {len(atoms_list)} frames (Energy source: '{found_key}')")
    return energies, forces, atoms_list

def neighbor_histograms(atoms_list, cutoff, bins=30):
    """Generates histograms of neighbor distances."""
    hists = []
    for at in atoms_list:
        _, _, d = neighbor_list('ijd', at, cutoff=cutoff)
        hist, _ = np.histogram(d, bins=bins, range=(0, cutoff))
        hists.append(hist)
    return np.array(hists)

def soap_descriptors(atoms_list, cutoff, nmax, lmax, species):
    """Generates SOAP descriptors for a list of ASE Atoms objects."""
    soap = SOAP(species=species, periodic=True, r_cut=cutoff, n_max=nmax, l_max=lmax, average="inner")
    return np.array([soap.create(at) for at in atoms_list])

def finite_diff_force_sensitivity(model_forces, atoms_list, delta):
    """Placeholder for force sensitivity calculation."""
    sens = []
    for at, base_f in zip(atoms_list, model_forces):
        idx = len(at) // 2
        f0 = base_f[idx*3:(idx+1)*3]
        # In a real scenario, one would re-run the model on a displaced structure.
        f1 = f0  # Placeholder assumes zero change
        sens.append(np.linalg.norm(f1 - f0))
    return np.array(sens).reshape(-1, 1)

# ------------------------
# ANALYSIS & PLOTTING
# ------------------------
def run_analysis_for_set(set_name, config):
    """
    Main function to run the entire analysis and plotting pipeline for a single
    comparison set.
    """
    output_dir = config["output_dir"]
    model_paths = config["models"]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    all_rows, all_labels = [], []
    for label, path in model_paths.items():
        print(f"\n--- Processing model '{label}' for set '{set_name}' ---")
        if not os.path.exists(path):
            print(f"Warning: File not found for '{label}'. Skipping.")
            continue
            
        E, F, atoms_list = parse_xyz(path, frame_step=FRAME_SAMPLING_RATE)
        if atoms_list is None or len(atoms_list) == 0:
            print(f"Warning: No data loaded for '{label}'. Skipping.")
            continue

        neigh = neighbor_histograms(atoms_list, CUTOFF)
        species = list(set(atoms_list[0].get_chemical_symbols()))
        soap = soap_descriptors(atoms_list, CUTOFF, SOAP_NMAX, SOAP_LMAX, species=species)

        n_samples = soap.shape[0]
        actual_pca_dim = min(PCA_DIM, n_samples, soap.shape[1])
        if actual_pca_dim > 0:
            soap_pca = PCA(n_components=actual_pca_dim).fit_transform(soap)
            if actual_pca_dim < PCA_DIM:
                padding = np.zeros((n_samples, PCA_DIM - actual_pca_dim))
                soap_pca = np.hstack([soap_pca, padding])
        else:
            soap_pca = np.zeros((n_samples, PCA_DIM))

        sens = finite_diff_force_sensitivity(F, atoms_list, FINITE_DIFF_DELTA)

        feats = np.hstack([E.reshape(-1, 1), F, neigh, soap_pca, sens])
        all_rows.append(feats)
        all_labels.extend([label] * len(feats))

    if not all_rows:
        print(f"\nNo data processed for set '{set_name}'. Aborting analysis for this set.")
        return

    features = np.vstack(all_rows)
    all_labels = np.array(all_labels)

    # Save features to CSV
    output_csv_path = os.path.join(output_dir, f"{set_name}_probe_features.csv")
    np.savetxt(output_csv_path, features, delimiter=",", fmt='%f')
    print(f"\nSaved combined features to {output_csv_path}")

    # --- Analysis ---
    if np.any(np.isnan(features)):
        print("Warning: NaN values found in features. Replacing with 0.")
        features = np.nan_to_num(features, nan=0.0)

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(features) - 1), random_state=42, n_iter=1000)
    tsne_emb = tsne.fit_transform(features)

    print("Calculating silhouette scores...")
    unique_labels = sorted(list(model_paths.keys()))
    sil_scores = {}
    if len(unique_labels) > 1:
        overall_score = silhouette_score(tsne_emb, all_labels)
        sil_scores['overall'] = overall_score
        print(f"  > Overall Silhouette Score: {overall_score:.3f}")
        
        all_sample_scores = silhouette_samples(tsne_emb, all_labels)
        for label in unique_labels:
            mask = all_labels == label
            if np.sum(mask) > 0:
                avg_score = np.mean(all_sample_scores[mask])
                sil_scores[label] = avg_score
                print(f"  > Avg. Silhouette for '{label}': {avg_score:.3f}")
    else:
        print("  > Only one model found, cannot calculate silhouette score.")

    # --- Plotting ---
    print("Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(8, 10), constrained_layout=True)
    axes = axes.flatten() # Flatten for easy indexing
    fig.suptitle(f"Latent Space Analysis: {set_name}", fontsize=16, weight='bold')

    # Define colors using jhu_colors
    color_palette = [
        get_jhu_color('Heritage Blue'),get_jhu_color('Spirit Blue'), get_jhu_color('Red'), get_jhu_color('Orange'), get_jhu_color('Homewood Green'),
        get_jhu_color('Purple')
    ]
    color_map = {label: color for label, color in zip(unique_labels, cycle(color_palette))}

    # Panel A: First model
    if len(unique_labels) > 0:
        label_a = unique_labels[0]
        mask_a = all_labels == label_a
        axes[0].scatter(tsne_emb[mask_a, 0], tsne_emb[mask_a, 1], c=color_map[label_a], alpha=0.7, s=24, edgecolors='none')
        axes[0].set_title(f"(a) t-SNE: {label_a}", fontsize=14)
    else:
        axes[0].set_title("(a) N/A", fontsize=14)
    axes[0].set_xlabel("t-SNE Dimension 1"); axes[0].set_ylabel("t-SNE Dimension 2")


    # Panel B: Second model (if it exists)
    if len(unique_labels) > 1:
        label_b = unique_labels[1]
        mask_b = all_labels == label_b
        axes[1].scatter(tsne_emb[mask_b, 0], tsne_emb[mask_b, 1], c=color_map[label_b], alpha=0.7, s=24, edgecolors='none')
        axes[1].set_title(f"(b) t-SNE: {label_b}", fontsize=14)
    else:
        axes[1].text(0.5, 0.5, "Single model", ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("(b) N/A", fontsize=14)
    axes[1].set_xlabel("t-SNE Dimension 1")

    # Panel C: Cross-model overlay
    for label in unique_labels:
        color = color_map.get(label)
        mask = all_labels == label
        axes[2].scatter(tsne_emb[mask, 0], tsne_emb[mask, 1], label=label, alpha=0.7, c=color, s=24, edgecolors='none')
    axes[2].legend(title="Model Type")
    axes[2].set_title("(c) Cross-Model Overlay", fontsize=14)
    axes[2].set_xlabel("t-SNE Dimension 1")

    # Panel D: Silhouette scores
    if len(unique_labels) > 1:
        model_scores = {k: v for k, v in sil_scores.items() if k != 'overall'}
        bar_colors = [color_map.get(key) for key in model_scores.keys()]
        axes[3].bar(model_scores.keys(), model_scores.values(), color=bar_colors)
        overall_val = sil_scores.get('overall', 0)
        axes[3].axhline(y=overall_val, color=get_jhu_color('Red'), linestyle='--',
                        label=f'Overall Avg: {overall_val:.3f}')
        axes[3].legend()

        # 🔧 Rotate tick labels if longer than 5 chars
        for label in axes[3].get_xticklabels():
            if len(label.get_text()) > 5:
                label.set_rotation(45)
                label.set_ha('right')

    else:
        axes[3].text(0.5, 0.5, 'Single model\nNo score', ha='center', va='center', transform=axes[3].transAxes)
    axes[3].set_title("(d) Silhouette Scores", fontsize=14)
    axes[3].set_ylabel("Avg. Silhouette Score")
    axes[3].set_ylim(min(list(sil_scores.values()) + [-0.1]) * 1.1 if sil_scores else -0.1, 1.0)


    for ax in axes:
        ax.grid(False)

    output_fig_path = os.path.join(output_dir, f"{set_name}_tsne_silhouette_panels.pdf")
    plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {output_fig_path}")

# ------------------------
# MAIN EXECUTION
# ------------------------
if __name__ == "__main__":
    for set_name, config in COMPARISON_SETS.items():
        print(f"\n{'='*20} Starting Analysis for Set: {set_name} {'='*20}")
        run_analysis_for_set(set_name, config)
    print(f"\n{'='*20} All Analyses Complete {'='*20}")
