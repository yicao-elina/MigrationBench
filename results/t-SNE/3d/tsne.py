#!/usr/bin/env python3
"""
Black-box Latent Space Probe for MACE Models (Multi-System Version) - 3D t-SNE

This script analyzes and compares the learned representations of MACE models
across multiple, user-defined systems or simulation conditions using 3D t-SNE
visualization with customizable markers for different data sources.

Enhanced Features:
- 3D t-SNE embedding for richer visualization
- Support for multiple XYZ files per model with custom labels
- Different marker shapes for different data sources
- Interactive 3D plots with rotation capabilities

Dependencies:
    pip install numpy matplotlib scikit-learn ase dscribe jhu_colors
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    # "Cr_Doped_Sb2Te3_600K": {
    #     "output_dir": "analysis_Cr_Doped_600K_3D",
    #     "data_sources": [
    #         {
    #             "label": "Foundation",
    #             "files": [
    #                 '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/0-foundation-mace_mp/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz'
    #             ],
    #             "marker": "o",  # circle
    #             "color": "Heritage Blue"
    #         },
    #         {
    #             "label": "Scratch",
    #             "files": [
    #                 '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/1-from-scratch/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz'
    #             ],
    #             "marker": "s",  # square
    #             "color": "Spirit Blue"
    #         },
    #         {
    #             "label": "FT - 600K",
    #             "files": [
    #                 '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/2-naive-fine-tuning/temp/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz'
    #             ],
    #             "marker": "^",  # triangle up
    #             "color": "Red"
    #         },
    #         {
    #             "label": "FT - Multi-T",
    #             "files": [
    #                 '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/3-multi_T-fine-tuning/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz'
    #             ],
    #             "marker": "D",  # diamond
    #             "color": "Orange"
    #         }
    #     ]
    # },
    # --- Example of a second comparison set with multiple files per source ---
    "Multi_File_Example": {
        "output_dir": "analysis_multi_file_example",
        "data_sources": [
            {
                "label": "High_Temp",
                "files": [
                    "/path/to/high_temp_1.xyz",
                    "/path/to/high_temp_2.xyz"
                ],
                "marker": "o",
                "color": "Red"
            },
            {
                "label": "Low_Temp", 
                "files": [
                    "/path/to/low_temp_1.xyz",
                    "/path/to/low_temp_2.xyz"
                ],
                "marker": "s",
                "color": "Blue"
            }
        ]
    }
}

# --- Global Parameters ---
FRAME_SAMPLING_RATE = 10  # Process every Nth frame for efficiency
CUTOFF = 5.0
SOAP_NMAX, SOAP_LMAX = 4, 4
PCA_DIM = 10
FINITE_DIFF_DELTA = 1e-3

# Available marker styles for automatic assignment
MARKER_STYLES = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '8']

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

def process_data_source(source_config):
    """Process all files for a single data source and return combined features."""
    label = source_config["label"]
    files = source_config["files"]
    
    all_features = []
    all_atoms = []
    
    for filepath in files:
        print(f"\n--- Processing file for '{label}': {os.path.basename(filepath)} ---")
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}. Skipping.")
            continue
            
        E, F, atoms_list = parse_xyz(filepath, frame_step=FRAME_SAMPLING_RATE)
        if atoms_list is None or len(atoms_list) == 0:
            print(f"Warning: No data loaded from {filepath}. Skipping.")
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
        
        all_features.append(feats)
        all_atoms.extend(atoms_list)
    
    if not all_features:
        return None, []
    
    combined_features = np.vstack(all_features)
    print(f"  > Combined {len(combined_features)} frames for '{label}'")
    return combined_features, all_atoms

# ------------------------
# ANALYSIS & PLOTTING
# ------------------------
def run_analysis_for_set(set_name, config):
    """
    Main function to run the entire analysis and plotting pipeline for a single
    comparison set with 3D t-SNE visualization.
    """
    output_dir = config["output_dir"]
    data_sources = config["data_sources"]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    all_features = []
    all_labels = []
    source_info = []  # Store source configuration for plotting
    
    for source_config in data_sources:
        label = source_config["label"]
        features, atoms_list = process_data_source(source_config)
        
        if features is not None:
            all_features.append(features)
            all_labels.extend([label] * len(features))
            source_info.append({
                'label': label,
                'marker': source_config.get('marker', 'o'),
                'color': source_config.get('color', 'blue'),
                'count': len(features)
            })

    if not all_features:
        print(f"\nNo data processed for set '{set_name}'. Aborting analysis for this set.")
        return

    features = np.vstack(all_features)
    all_labels = np.array(all_labels)

    # Save features to CSV
    output_csv_path = os.path.join(output_dir, f"{set_name}_probe_features.csv")
    np.savetxt(output_csv_path, features, delimiter=",", fmt='%f')
    print(f"\nSaved combined features to {output_csv_path}")

    # --- Analysis ---
    if np.any(np.isnan(features)):
        print("Warning: NaN values found in features. Replacing with 0.")
        features = np.nan_to_num(features, nan=0.0)

    print("Running 3D t-SNE...")
    tsne = TSNE(n_components=3, perplexity=min(30, len(features) - 1), random_state=42, n_iter=1000)
    tsne_emb = tsne.fit_transform(features)

    print("Calculating silhouette scores...")
    unique_labels = [info['label'] for info in source_info]
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
        print("  > Only one data source found, cannot calculate silhouette score.")

    # --- 3D Plotting ---
    print("Generating 3D plots...")
    
    # Create figure with subplots: 2x2 layout with 3D plots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"3D Latent Space Analysis: {set_name}", fontsize=16, weight='bold')

    # Define colors using jhu_colors
    color_palette = [
        get_jhu_color('Heritage Blue'), get_jhu_color('Spirit Blue'), 
        get_jhu_color('Red'), get_jhu_color('Orange'), 
        get_jhu_color('Homewood Green'), get_jhu_color('Purple')
    ]
    
    # Create color and marker mappings
    color_map = {}
    marker_map = {}
    for i, info in enumerate(source_info):
        # Use specified color or fall back to palette
        if info['color'] and get_jhu_color(info['color']):
            color_map[info['label']] = get_jhu_color(info['color'])
        else:
            color_map[info['label']] = color_palette[i % len(color_palette)]
        marker_map[info['label']] = info['marker']

    # Panel 1: First data source (3D)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    if len(source_info) > 0:
        label_1 = source_info[0]['label']
        mask_1 = all_labels == label_1
        ax1.scatter(tsne_emb[mask_1, 0], tsne_emb[mask_1, 1], tsne_emb[mask_1, 2], 
                   c=color_map[label_1], marker=marker_map[label_1], alpha=0.7, s=24)
        ax1.set_title(f"(a) 3D t-SNE: {label_1}", fontsize=12)
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2") 
    ax1.set_zlabel("t-SNE 3")

    # Panel 2: Second data source (3D)
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    if len(source_info) > 1:
        label_2 = source_info[1]['label']
        mask_2 = all_labels == label_2
        ax2.scatter(tsne_emb[mask_2, 0], tsne_emb[mask_2, 1], tsne_emb[mask_2, 2],
                   c=color_map[label_2], marker=marker_map[label_2], alpha=0.7, s=24)
        ax2.set_title(f"(b) 3D t-SNE: {label_2}", fontsize=12)
    else:
        ax2.text(0.5, 0.5, 0.5, "Single data source", ha='center', va='center', 
                transform=ax2.transAxes)
        ax2.set_title("(b) N/A", fontsize=12)
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.set_zlabel("t-SNE 3")

    # Panel 3: Cross-source overlay (3D)
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    for info in source_info:
        label = info['label']
        mask = all_labels == label
        ax3.scatter(tsne_emb[mask, 0], tsne_emb[mask, 1], tsne_emb[mask, 2],
                   label=label, alpha=0.7, c=color_map[label], 
                   marker=marker_map[label], s=24)
    ax3.legend(title="Data Source", loc='upper left', bbox_to_anchor=(0, 1))
    ax3.set_title("(c) Cross-Source Overlay", fontsize=12)
    ax3.set_xlabel("t-SNE 1")
    ax3.set_ylabel("t-SNE 2")
    ax3.set_zlabel("t-SNE 3")

    # Panel 4: Silhouette scores (2D bar plot)
    ax4 = fig.add_subplot(2, 2, 4)
    if len(unique_labels) > 1:
        model_scores = {k: v for k, v in sil_scores.items() if k != 'overall'}
        bar_colors = [color_map.get(key) for key in model_scores.keys()]
        bars = ax4.bar(model_scores.keys(), model_scores.values(), color=bar_colors)
        
        overall_val = sil_scores.get('overall', 0)
        ax4.axhline(y=overall_val, color=get_jhu_color('Red') or 'red', linestyle='--',
                   label=f'Overall Avg: {overall_val:.3f}')
        ax4.legend()

        # Rotate tick labels if longer than 5 chars
        for label in ax4.get_xticklabels():
            if len(label.get_text()) > 5:
                label.set_rotation(45)
                label.set_ha('right')
    else:
        ax4.text(0.5, 0.5, 'Single data source\nNo score', ha='center', va='center', 
                transform=ax4.transAxes)
    
    ax4.set_title("(d) Silhouette Scores", fontsize=12)
    ax4.set_ylabel("Avg. Silhouette Score")
    if sil_scores:
        ax4.set_ylim(min(list(sil_scores.values()) + [-0.1]) * 1.1, 1.0)
    else:
        ax4.set_ylim(-0.1, 1.0)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save the figure
    output_fig_path = os.path.join(output_dir, f"{set_name}_3D_tsne_analysis.pdf")
    plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D figure to {output_fig_path}")

    # --- Additional: Create a separate interactive 3D plot ---
    fig_3d = plt.figure(figsize=(12, 9))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    for info in source_info:
        label = info['label']
        mask = all_labels == label
        ax_3d.scatter(tsne_emb[mask, 0], tsne_emb[mask, 1], tsne_emb[mask, 2],
                     label=f"{label} (n={info['count']})", alpha=0.7, 
                     c=color_map[label], marker=marker_map[label], s=36)
    
    ax_3d.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax_3d.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax_3d.set_zlabel("t-SNE Dimension 3", fontsize=12)
    ax_3d.set_title(f"Interactive 3D t-SNE: {set_name}", fontsize=14, weight='bold')
    ax_3d.legend(title="Data Sources", loc='upper left', bbox_to_anchor=(0, 1))
    
    # Save interactive 3D plot
    output_3d_path = os.path.join(output_dir, f"{set_name}_interactive_3D_tsne.pdf")
    plt.savefig(output_3d_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved interactive 3D plot to {output_3d_path}")

    # --- Summary Statistics ---
    print(f"\n--- Summary for {set_name} ---")
    print(f"Total data points: {len(features)}")
    for info in source_info:
        print(f"  {info['label']}: {info['count']} points (marker: {info['marker']})")
    if sil_scores:
        print(f"Overall silhouette score: {sil_scores.get('overall', 'N/A'):.3f}")

# ------------------------
# MAIN EXECUTION
# ------------------------
if __name__ == "__main__":
    for set_name, config in COMPARISON_SETS.items():
        print(f"\n{'='*20} Starting 3D Analysis for Set: {set_name} {'='*20}")
        run_analysis_for_set(set_name, config)
    print(f"\n{'='*20} All 3D Analyses Complete {'='*20}")