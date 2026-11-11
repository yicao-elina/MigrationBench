#!/usr/bin/env python3
"""
Black-box Latent Space Probe for MACE Models (Multi-System Version)

This script analyzes and compares the learned representations of MACE models
by projecting high-dimensional feature vectors into 2D using both t-SNE
and PHATE to provide a robust, multi-faceted view of the latent space.

Workflow for each defined system:
1.  Loads molecular dynamics trajectories for a set of specified models.
2.  Extracts a feature vector for each frame, including energies, forces,
    and structural descriptors (RDF histograms, SOAP).
3.  Performs dimensionality reduction on the feature space using:
    - t-SNE: To emphasize local structure and cluster separation.
    - PHATE: To preserve global structure and trajectory-like progressions.
4.  Generates and saves a comprehensive 2x2 comparison plot:
    - (a) t-SNE overlay showing model representation separation.
    - (b) PHATE overlay showing preservation of the underlying data manifold.
    - (c) PHATE embedding colored by potential energy to link geometry to physics.
    - (d) Silhouette scores to quantify t-SNE cluster separation.

Dependencies:
    pip install numpy matplotlib scikit-learn ase dscribe jhu_colors phate
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
from sklearn.metrics import silhouette_score, silhouette_samples
from itertools import cycle

# --- NEW: Import PHATE ---
try:
    import phate
except ImportError:
    print("Error: 'phate' package not found. Please install with 'pip install --user phate'")
    phate = None

# Import JHU color standards
try:
    import jhu_colors
    get_jhu_color = jhu_colors.get_jhu_color
except ImportError:
    print("Warning: 'jhu_colors' package not found. Using default matplotlib styles.")
    def get_jhu_color(x): return None

# ------------------------
# CONFIGURATION
# ------------------------
COMPARISON_SETS = {
    "Cr_Doped_Sb2Te3_600K": {
        "output_dir": "analysis_Cr_Doped_600K_v2",
        "models": {
            'Foundation': '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/0-foundation-mace_mp/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz',
            'Scratch': '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/1-from-scratch/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz',
            'FT - 600K': '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/2-naive-fine-tuning/temp/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz',
            'FT - Multi-T': '/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/4.lmp/Rippling/2L_octo_Cr2_v2/0806/3-multi_T-fine-tuning/2L_octo_Cr2_v2_relax_maceMD_eval_5x5x1_T600K.xyz'
        }
    },
}

# --- Global Parameters ---
FRAME_SAMPLING_RATE = 10
CUTOFF = 5.0
SOAP_NMAX, SOAP_LMAX = 4, 4
PCA_DIM = 10
FINITE_DIFF_DELTA = 1e-3

# ------------------------
# UTILITY FUNCTIONS (ASSUMED UNCHANGED FROM ORIGINAL SCRIPT)
# ... (parse_xyz, neighbor_histograms, soap_descriptors, etc. are identical)
# ------------------------
def parse_xyz(filepath, frame_step):
    """Robustly parses an XYZ file to extract energies, forces, and ASE Atoms objects."""
    print(f"  > Reading frames from {os.path.basename(filepath)}...")
    try:
        atoms_list = read(filepath, index=f"::{frame_step}")
        if not atoms_list: raise IOError("No frames could be read.")
    except Exception as e:
        print(f"Error reading file with ASE: {e}"); return None, None, None
    energies, forces = [], []
    energy_keys_to_try = ['energy', 'MACE_energy', 'free_energy', 'potential_energy']
    found_key = None
    for atoms in atoms_list:
        energy_found = False
        if found_key and found_key in atoms.info:
            energies.append(atoms.info[found_key]); energy_found = True
        else:
            for key in energy_keys_to_try:
                if key in atoms.info:
                    energies.append(atoms.info[key]); found_key = key; energy_found = True; break
        if not energy_found: energies.append(None)
        forces.append(atoms.get_forces().flatten())
    if all(e is None for e in energies):
        print("  > ASE parsing failed to find energy. Falling back to manual parsing...")
        energies = []
        try:
            with open(filepath, 'r') as f: lines = f.readlines()
            frame_start_indices = [i for i, line in enumerate(lines) if line.strip().isdigit() and i + 1 < len(lines) and "Lattice" in lines[i+1]]
            sampled_indices = frame_start_indices[::frame_step]
            for i in sampled_indices:
                header_line = lines[i+1]
                energy_match = re.search(r"(?:energy|free_energy)\s*=\s*([-0-9.eE]+)", header_line)
                if energy_match: energies.append(float(energy_match.group(1)))
                else: raise ValueError(f"Manual parser could not find energy in header: {header_line.strip()}")
            found_key = "manual"
        except Exception as e:
            print(f"Error during manual parsing: {e}"); return None, None, None
    if len(energies) != len(atoms_list):
        print(f"Error: Final energy count ({len(energies)}) does not match frame count ({len(atoms_list)})."); return None, None, None
    energies = np.array(energies, dtype=float); forces = np.array(forces)
    print(f"  > Successfully parsed {len(atoms_list)} frames (Energy source: '{found_key}')")
    return energies, forces, atoms_list

def neighbor_histograms(atoms_list, cutoff, bins=30):
    hists = []
    for at in atoms_list:
        _, _, d = neighbor_list('ijd', at, cutoff=cutoff); hist, _ = np.histogram(d, bins=bins, range=(0, cutoff)); hists.append(hist)
    return np.array(hists)

def soap_descriptors(atoms_list, cutoff, nmax, lmax, species):
    soap = SOAP(species=species, periodic=True, r_cut=cutoff, n_max=nmax, l_max=lmax, average="inner")
    return np.array([soap.create(at) for at in atoms_list])

def finite_diff_force_sensitivity(model_forces, atoms_list, delta):
    return np.zeros(len(atoms_list)).reshape(-1, 1) # Placeholder

# ------------------------
# ANALYSIS & PLOTTING
# ------------------------
def run_analysis_for_set(set_name, config):
    output_dir = config["output_dir"]; model_paths = config["models"]
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")
    all_rows, all_labels = [], []
    for label, path in model_paths.items():
        print(f"\n--- Processing model '{label}' for set '{set_name}' ---")
        if not os.path.exists(path):
            print(f"Warning: File not found for '{label}'. Skipping."); continue
        E, F, atoms_list = parse_xyz(path, frame_step=FRAME_SAMPLING_RATE)
        if atoms_list is None or len(atoms_list) == 0:
            print(f"Warning: No data loaded for '{label}'. Skipping."); continue
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
        else: soap_pca = np.zeros((n_samples, PCA_DIM))
        sens = finite_diff_force_sensitivity(F, atoms_list, FINITE_DIFF_DELTA)
        feats = np.hstack([E.reshape(-1, 1), F, neigh, soap_pca, sens])
        all_rows.append(feats); all_labels.extend([label] * len(feats))
    if not all_rows:
        print(f"\nNo data processed for set '{set_name}'. Aborting."); return
    features = np.vstack(all_rows); all_labels = np.array(all_labels)
    output_csv_path = os.path.join(output_dir, f"{set_name}_probe_features.csv")
    np.savetxt(output_csv_path, features, delimiter=",", fmt='%f')
    print(f"\nSaved combined features to {output_csv_path}")
    if np.any(np.isnan(features)):
        print("Warning: NaN values found in features. Replacing with 0.")
        features = np.nan_to_num(features, nan=0.0)

    # --- Analysis: t-SNE and PHATE ---
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(features) - 1), random_state=42, n_iter=1000)
    tsne_emb = tsne.fit_transform(features)

    # --- NEW: Run PHATE ---
    if phate:
        print("Running PHATE...")
        phate_op = phate.PHATE(random_state=42, n_jobs=-1)
        phate_emb = phate_op.fit_transform(features)
    else:
        print("PHATE not available. Skipping PHATE analysis.")
        phate_emb = np.zeros_like(tsne_emb) # Placeholder for plotting

    print("Calculating silhouette scores...")
    unique_labels = sorted(list(model_paths.keys()))
    sil_scores = {}
    if len(unique_labels) > 1:
        overall_score = silhouette_score(tsne_emb, all_labels)
        sil_scores['overall'] = overall_score
        print(f"  > Overall Silhouette Score (from t-SNE): {overall_score:.3f}")
        all_sample_scores = silhouette_samples(tsne_emb, all_labels)
        for label in unique_labels:
            mask = all_labels == label
            if np.sum(mask) > 0:
                avg_score = np.mean(all_sample_scores[mask])
                sil_scores[label] = avg_score
                print(f"  > Avg. Silhouette for '{label}': {avg_score:.3f}")

    # --- Plotting: Revised 2x2 Layout ---
    print("Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(f"Representation Analysis: {set_name}", fontsize=18, weight='bold')
    color_palette = [get_jhu_color('Heritage Blue'), get_jhu_color('Spirit Blue'),  get_jhu_color('Red'), get_jhu_color('Orange')]
    color_map = {label: color for label, color in zip(unique_labels, cycle(color_palette))}
    
    # Panel A: t-SNE Cross-Model Overlay
    ax = axes[0, 0]
    for label in unique_labels:
        mask = all_labels == label
        ax.scatter(tsne_emb[mask, 0], tsne_emb[mask, 1], label=label, alpha=0.7, c=color_map.get(label), s=20, edgecolors='none')
    ax.legend(title="Model Type", fontsize=10)
    ax.set_title("(a) t-SNE Projection (Local Structure)", fontsize=14)
    ax.set_xlabel("t-SNE Dimension 1"); ax.set_ylabel("t-SNE Dimension 2")

    # Panel B: PHATE Cross-Model Overlay
    ax = axes[0, 1]
    if phate:
        for label in unique_labels:
            mask = all_labels == label
            ax.scatter(phate_emb[mask, 0], phate_emb[mask, 1], label=label, alpha=0.7, c=color_map.get(label), s=20, edgecolors='none')
        ax.legend(title="Model Type", fontsize=10)
    else:
        ax.text(0.5, 0.5, "PHATE not installed", ha='center', va='center', transform=ax.transAxes)
    ax.set_title("(b) PHATE Projection (Global Structure)", fontsize=14)
    ax.set_xlabel("PHATE Dimension 1"); ax.set_ylabel("PHATE Dimension 2")

    # Panel C: PHATE Embedding Colored by Energy
    ax = axes[1, 0]
    energies = features[:, 0] # Energy is the first column
    if phate:
        sc = ax.scatter(phate_emb[:, 0], phate_emb[:, 1], c=energies, cmap='jhu', s=10, alpha=0.8)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Potential Energy (eV)", fontsize=12)
    else:
        ax.text(0.5, 0.5, "PHATE not installed", ha='center', va='center', transform=ax.transAxes)
    ax.set_title("(c) PHATE Colored by System Energy", fontsize=14)
    ax.set_xlabel("PHATE Dimension 1"); ax.set_ylabel("PHATE Dimension 2")
    
    # Panel D: Silhouette Scores
    ax = axes[1, 1]
    if len(unique_labels) > 1:
        model_scores = {k: v for k, v in sil_scores.items() if k != 'overall'}
        bar_colors = [color_map.get(key) for key in model_scores.keys()]
        ax.bar(model_scores.keys(), model_scores.values(), color=bar_colors)
        overall_val = sil_scores.get('overall', 0)
        ax.axhline(y=overall_val, color=get_jhu_color('Red'), linestyle='--', label=f'Overall Avg: {overall_val:.3f}')
        ax.legend()
        ax.tick_params(axis='x', rotation=30, labelsize=10)
    else:
        ax.text(0.5, 0.5, 'Single model\nNo score', ha='center', va='center', transform=ax.transAxes)
    ax.set_title("(d) Silhouette Scores (from t-SNE)", fontsize=14)
    ax.set_ylabel("Avg. Silhouette Score")
    ax.set_ylim(min(list(sil_scores.values()) + [-0.1]) * 1.1 if sil_scores else -0.1, 1.0)
    
    output_fig_path = os.path.join(output_dir, f"{set_name}_representation_analysis.pdf")
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