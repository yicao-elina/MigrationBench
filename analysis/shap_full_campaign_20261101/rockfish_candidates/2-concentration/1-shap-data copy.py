#!/usr/bin/env python3
"""
SOAP Descriptor Analysis and Visualization

This script performs SOAP descriptor analysis on atomic trajectories from different
doping conditions, computes feature importance, and creates visualizations including
heatmaps showing differences between doping conditions.
"""

import os
import re
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

# Core scientific packages
from ase.io import read
from ase.neighborlist import neighbor_list
from dscribe.descriptors import SOAP
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# --- Configuration ---

# Data Paths for different doping conditions
DATA_PATHS = {
    "Bulk": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/2.aimd/3-2.2-layer_CrSb2Te3/1.bulk-aimd/aimd_merged_traj_bulk.xyz",
    "1/8 ML": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/2.aimd/3-2.2-layer_CrSb2Te3/converted_xyz/2L_octo_Cr1_v2_600K_aimd_1.xyz",
    "1/4 ML": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/2.aimd/3-2.2-layer_CrSb2Te3/converted_xyz/2L_octo_Cr2_600K_aimd_1.xyz",
    "3/8 ML": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/2.aimd/3-2.2-layer_CrSb2Te3/converted_xyz/2L_octo_Cr3_600K_aimd_1.xyz",
    "1/2 ML": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/2.aimd/3-2.2-layer_CrSb2Te3/converted_xyz/2L_tetra_Cr4_600K_aimd1.xyz",
    "3/4 ML": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/2.aimd/3-2.2-layer_CrSb2Te3/converted_xyz/2L_o3_t3_600K_aimd_1.xyz",
}

# Analysis Parameters
FRAME_SAMPLING_RATE = 5
CUTOFF = 5.0
SOAP_NMAX, SOAP_LMAX = 4, 4
N_NEIGHBOR_BINS = 30

# Output Configuration
OUTPUT_DIR = 'soap_analysis_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting style
plt.style.use('default')
sns.set_palette("husl")

# --- Classes and Functions ---

class SOAPFeatureExtractor:
    """
    SOAP descriptor extractor for structural analysis.
    """
    def __init__(self, cutoff=5.0, soap_nmax=4, soap_lmax=4, n_bins=30):
        self.cutoff = cutoff
        self.soap_nmax = soap_nmax
        self.soap_lmax = soap_lmax
        self.n_bins = n_bins
        self.soap_descriptor = None
        self.species = None
        
    def parse_xyz_robust(self, filepath, frame_step=1):
        """Parse XYZ file to extract ground-truth energies and atoms objects."""
        print(f"  > Reading frames from {os.path.basename(filepath)}...")
        try:
            atoms_list = read(filepath, index=f"::{frame_step}")
            if not atoms_list:
                raise IOError("No frames could be read.")
        except Exception as e:
            print(f"Error reading file with ASE: {e}")
            return None, None

        energies = []
        energy_keys_to_try = ['energy', 'free_energy', 'potential_energy']
        found_key = None

        for atoms in atoms_list:
            energy_found = False
            for key in energy_keys_to_try:
                if key in atoms.info:
                    energies.append(atoms.info[key])
                    found_key = key
                    energy_found = True
                    break
            if not energy_found:
                energies.append(None)

        # Fallback to manual parsing if ASE failed to extract energies
        if all(e is None for e in energies):
            print("  > ASE parsing failed to find energy. Falling back to manual parsing...")
            energies = []
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                frame_start_indices = [i for i, line in enumerate(lines) 
                                       if line.strip().isdigit() and i + 1 < len(lines) 
                                       and "Lattice" in lines[i+1]]
                sampled_indices = frame_start_indices[::frame_step]

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
                return None, None

        if len(energies) != len(atoms_list):
            print(f"Error: Mismatch in parsed counts. Frames: {len(atoms_list)}, Energies: {len(energies)}")
            return None, None

        energies = np.array(energies, dtype=float)
        print(f"  > Successfully parsed {len(atoms_list)} frames (Energy source: '{found_key}')")
        return energies, atoms_list

    def extract_neighbor_histograms(self, atoms_list):
        """Extract neighbor distance histograms."""
        print("  > Extracting neighbor distance histograms...")
        hists = []
        for atoms in atoms_list:
            try:
                _, _, d = neighbor_list('ijd', atoms, cutoff=self.cutoff)
                hist, _ = np.histogram(d, bins=self.n_bins, range=(0, self.cutoff))
                hists.append(hist)
            except Exception as e:
                print(f"    Warning: Failed to compute neighbors for one frame: {e}")
                hists.append(np.zeros(self.n_bins))
        return np.array(hists)

    def extract_soap_descriptors(self, atoms_list):
        """Extract SOAP descriptors."""
        print("  > Extracting SOAP descriptors...")
        try:
            # Get unique species from all frames to ensure consistency
            all_species = set()
            for atoms in atoms_list[:10]:  # Check first 10 frames
                all_species.update(atoms.get_chemical_symbols())
            
            # Filter to only expected elements for Cr-Sb2Te3 system
            expected_elements = {'Cr', 'Sb', 'Te'}
            self.species = sorted(list(all_species.intersection(expected_elements)))
            
            if not self.species:
                # Fallback if intersection is empty
                self.species = sorted(list(all_species))
            
            print(f"    Species found: {self.species}")
            
            self.soap_descriptor = SOAP(
                species=self.species, periodic=True, r_cut=self.cutoff, 
                n_max=self.soap_nmax, l_max=self.soap_lmax, average="inner"
            )
            
            soap_features = []
            for i, atoms in enumerate(atoms_list):
                try:
                    soap_vec = self.soap_descriptor.create(atoms)
                    soap_features.append(soap_vec)
                except Exception as e:
                    print(f"    Warning: SOAP failed for frame {i}: {e}")
                    if len(soap_features) > 0:
                        soap_features.append(np.zeros_like(soap_features[0]))
                    else:
                        estimated_size = len(self.species) * (self.soap_nmax ** 2) * (self.soap_lmax + 1)
                        soap_features.append(np.zeros(estimated_size))
            
            soap_array = np.array(soap_features)
            print(f"    SOAP shape: {soap_array.shape}")
            return soap_array
            
        except Exception as e:
            print(f"    Error in SOAP extraction: {e}")
            n_frames = len(atoms_list)
            estimated_size = 3 * (self.soap_nmax ** 2) * (self.soap_lmax + 1)  # Assume 3 species
            return np.zeros((n_frames, estimated_size))

    def get_soap_feature_names(self):
        """Generate physically meaningful names for SOAP features."""
        if self.species is None:
            return []
        
        feature_names = []
        # Generate names based on SOAP structure: species pairs + radial/angular components
        for i, species_i in enumerate(self.species):
            for j, species_j in enumerate(self.species):
                if j >= i:  # Only upper triangular due to symmetry
                    for n in range(self.soap_nmax):
                        for l in range(self.soap_lmax + 1):
                            pair_name = f"{species_i}-{species_j}" if i != j else f"{species_i}-{species_i}"
                            feature_names.append(f"SOAP_{pair_name}_n{n}_l{l}")
        
        return feature_names

    def extract_structural_features(self, atoms_list):
        """Extract basic structural features."""
        print("  > Extracting structural features...")
        structural_features = []
        
        for atoms in atoms_list:
            features = []
            features.append(len(atoms))
            features.append(atoms.get_volume())
            cell = atoms.get_cell()
            features.extend(cell.lengths())
            features.extend(cell.angles())
            
            try:
                i, j, d = neighbor_list('ijd', atoms, cutoff=self.cutoff)
                coord_numbers = np.bincount(i, minlength=len(atoms))
                features.extend([np.mean(coord_numbers), np.std(coord_numbers), 
                               np.max(coord_numbers), np.min(coord_numbers)])
                features.extend([np.mean(d), np.std(d), np.min(d), np.max(d)])
            except:
                features.extend([0.0] * 8)
            
            structural_features.append(features)
        
        return np.array(structural_features)

    def extract_all_features(self, atoms_list):
        """Extract all features."""
        print(f"Extracting comprehensive features...")
        
        neighbor_hists = self.extract_neighbor_histograms(atoms_list)
        soap_features = self.extract_soap_descriptors(atoms_list)
        
        # Store actual number of SOAP features for naming
        self._actual_soap_features = soap_features.shape[1]
        
        structural_features = self.extract_structural_features(atoms_list)
        
        # Combine all features
        all_features = np.hstack([neighbor_hists, soap_features, structural_features])
        
        print(f"  > Combined feature shape: {all_features.shape}")
        return all_features

    def get_feature_names(self):
        """Generate feature names for interpretability."""
        feature_names = []
        
        # Neighbor histogram names
        feature_names.extend([f'neighbor_hist_bin_{i}' for i in range(self.n_bins)])
        
        # SOAP feature names
        soap_names = self.get_soap_feature_names()
        # Adjust if there's a mismatch
        if hasattr(self, '_actual_soap_features') and len(soap_names) != self._actual_soap_features:
            print(f"    Adjusting SOAP names: expected {len(soap_names)}, got {self._actual_soap_features}")
            if len(soap_names) > self._actual_soap_features:
                soap_names = soap_names[:self._actual_soap_features]
            else:
                # Pad with generic names
                while len(soap_names) < self._actual_soap_features:
                    soap_names.append(f'SOAP_feat_{len(soap_names)}')
        
        feature_names.extend(soap_names)
        
        # Structural feature names
        feature_names.extend(['n_atoms', 'volume', 'cell_a', 'cell_b', 'cell_c', 
                             'angle_alpha', 'angle_beta', 'angle_gamma',
                             'coord_mean', 'coord_std', 'coord_max', 'coord_min', 
                             'dist_mean', 'dist_std', 'dist_min', 'dist_max'])
        
        return feature_names

class SOAPAnalyzer:
    """Main analyzer class for SOAP descriptors."""
    def __init__(self, data_paths):
        self.data_paths = data_paths
        self.feature_extractor = SOAPFeatureExtractor(
            cutoff=CUTOFF, soap_nmax=SOAP_NMAX, soap_lmax=SOAP_LMAX, n_bins=N_NEIGHBOR_BINS
        )
        
    def prepare_dataset(self, scenario, n_samples=300):
        """Prepare dataset for a given doping scenario."""
        print(f"\nPreparing dataset for {scenario}...")
        
        trajectory_file = self.data_paths[scenario]
        if not os.path.exists(trajectory_file):
            print(f"Trajectory file not found: {trajectory_file}")
            return None
        
        # Parse ground-truth DFT data from the XYZ file
        dft_energies, atoms_list = self.feature_extractor.parse_xyz_robust(
            trajectory_file, frame_step=FRAME_SAMPLING_RATE
        )
        
        if atoms_list is None: 
            return None
        
        if len(atoms_list) > n_samples:
            atoms_list = atoms_list[:n_samples]
            dft_energies = dft_energies[:n_samples]
        
        try:
            features = self.feature_extractor.extract_all_features(atoms_list)
            return {
                'features': features, 
                'energies': dft_energies,
                'atoms_list': atoms_list, 
                'feature_names': self.feature_extractor.get_feature_names(),
                'scenario': scenario
            }
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return None

def train_predictor_and_get_importance(dataset):
    """Train a predictor and get feature importance."""
    print(f"  Training predictor for {dataset['scenario']}...")
    
    X = dataset['features']
    y = dataset['energies']
    
    # Remove NaN values
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid_mask], y[valid_mask]
    
    if len(X) == 0:
        raise ValueError("No valid training data after removing NaN values")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"    Model performance - MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return {
        'feature_importance': model.feature_importances_,
        'feature_names': dataset['feature_names'],
        'mae': mae,
        'r2': r2,
        'scenario': dataset['scenario']
    }

def plot_feature_importance(results, top_n=20):
    """Plot feature importance for each doping condition."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, (scenario, result) in enumerate(results.items()):
        if result is None:
            continue
            
        # Get top features
        importance = result['feature_importance']
        feature_names = result['feature_names']
        
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1][:top_n]
        top_importance = importance[sorted_idx]
        top_names = [feature_names[idx] for idx in sorted_idx]
        
        # Truncate long names
        top_names = [name[:30] + '...' if len(name) > 30 else name for name in top_names]
        
        # Plot
        ax = axes[i]
        bars = ax.barh(range(len(top_importance)), top_importance)
        ax.set_yticks(range(len(top_importance)))
        ax.set_yticklabels(top_names, fontsize=8)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{scenario} (R² = {result["r2"]:.3f})')
        ax.invert_yaxis()
        
        # Color bars by feature type
        colors = []
        for name in top_names:
            if 'SOAP' in name:
                colors.append('skyblue')
            elif 'neighbor' in name:
                colors.append('lightcoral')
            else:
                colors.append('lightgreen')
        
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    # Remove empty subplot
    if len(results) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_by_doping.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def cluster_soap_by_radial_angular(results):
    """Cluster SOAP features by radial (n) and angular (l) quantum numbers."""
    # Get all SOAP features and their importance across scenarios
    all_soap_data = []
    soap_feature_names = []
    scenarios = []
    
    for scenario, result in results.items():
        if result is None:
            continue
            
        feature_names = result['feature_names']
        importance = result['feature_importance']
        
        # Extract SOAP features
        soap_importance = []
        soap_names = []
        
        for i, name in enumerate(feature_names):
            if 'SOAP' in name and is_valid_soap_feature(name):
                soap_importance.append(importance[i])
                soap_names.append(name)
        
        if soap_importance:
            all_soap_data.append(soap_importance)
            if not soap_feature_names:  # Store names from first scenario
                soap_feature_names = soap_names
            scenarios.append(scenario)
    
    if not all_soap_data or not soap_feature_names:
        return None, None, None
    
    # Group features by radial (n) and angular (l) quantum numbers
    radial_angular_groups = {}
    
    for i, name in enumerate(soap_feature_names):
        n_val, l_val = extract_n_l_values(name)
        group_key = f"n{n_val}_l{l_val}"
        
        if group_key not in radial_angular_groups:
            radial_angular_groups[group_key] = []
        radial_angular_groups[group_key].append(i)
    
    # Create grouped importance matrix
    n_groups = len(radial_angular_groups)
    grouped_importance = np.zeros((len(scenarios), n_groups))
    group_names = []
    
    # Sort groups by n and l values for logical ordering
    sorted_groups = sorted(radial_angular_groups.items(), 
                          key=lambda x: (int(x[0].split('_')[0][1:]), int(x[0].split('_')[1][1:])))
    
    for j, (group_name, indices) in enumerate(sorted_groups):
        group_names.append(group_name)
        # Sum importance for all atomic pairs with this n,l combination
        for i, scenario_data in enumerate(all_soap_data):
            grouped_importance[i, j] = np.sum([scenario_data[idx] for idx in indices])
    
    return grouped_importance, group_names, scenarios

def is_valid_soap_feature(feature_name):
    """Check if SOAP feature contains only valid elements (Cr, Sb, Te)."""
    valid_elements = {'Cr', 'Sb', 'Te'}
    
    # Extract atomic pair from feature name
    if 'SOAP_' in feature_name:
        parts = feature_name.split('_')
        if len(parts) >= 2:
            pair_part = parts[1]
            if '-' in pair_part:
                elements = pair_part.split('-')
                return all(elem in valid_elements for elem in elements)
            else:
                return pair_part in valid_elements
    
    return False

def extract_n_l_values(feature_name):
    """Extract n and l quantum numbers from SOAP feature name."""
    # Default values
    n_val, l_val = 0, 0
    
    parts = feature_name.split('_')
    for part in parts:
        if part.startswith('n') and len(part) > 1:
            try:
                n_val = int(part[1:])
            except ValueError:
                pass
        elif part.startswith('l') and len(part) > 1:
            try:
                l_val = int(part[1:])
            except ValueError:
                pass
    
    return n_val, l_val

def extract_atomic_pair(feature_name):
    """Robustly extract atomic pair from SOAP feature name, restricted to Cr, Sb, Te."""
    valid_elements = {'Cr', 'Sb', 'Te'}
    
    # Pattern 1: SOAP_Pair_n#_l# format
    if 'SOAP_' in feature_name:
        parts = feature_name.split('_')
        if len(parts) >= 2:
            pair_part = parts[1]
            if '-' in pair_part:
                elements = pair_part.split('-')
                if all(elem in valid_elements for elem in elements):
                    return pair_part
            elif pair_part in valid_elements:
                return f"{pair_part}-{pair_part}"
    
    # Pattern 2: Look for valid element symbols in the name
    found_elements = []
    for element in valid_elements:
        if element in feature_name:
            found_elements.append(element)
    
    if len(found_elements) >= 2:
        return f"{found_elements[0]}-{found_elements[1]}"
    elif len(found_elements) == 1:
        return f"{found_elements[0]}-{found_elements[0]}"
    
    # If no valid elements found, return None (will be filtered out)
    return None

def plot_soap_feature_heatmap(results):
    """Plot clustered heatmap of SOAP feature importance by radial/angular quantum numbers."""
    clustered_importance, cluster_names, scenarios = cluster_soap_by_radial_angular(results)
    
    if clustered_importance is None:
        print("No SOAP features found for clustered heatmap")
        return
    
    # Plot heatmap
    plt.figure(figsize=(14, 8))
    
    sns.heatmap(clustered_importance, 
                xticklabels=cluster_names,
                yticklabels=scenarios,
                annot=True, 
                fmt='.3f',
                cmap='viridis',
                cbar_kws={'label': 'Feature Importance (Grouped by n,l)'})
    
    plt.title('SOAP Feature Importance by Radial (n) and Angular (l) Quantum Numbers')
    plt.xlabel('Radial-Angular Combinations (n,l)')
    plt.ylabel('Doping Condition')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'soap_importance_heatmap_radial_angular.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_importance_differences(results, reference='1/8 ML'):
    """Plot differences in feature importance relative to reference condition."""
    if reference not in results or results[reference] is None:
        print(f"Reference condition '{reference}' not available")
        return
    
    ref_importance = results[reference]['feature_importance']
    ref_features = results[reference]['feature_names']
    
    # Calculate differences
    differences = {}
    for scenario, result in results.items():
        if result is None or scenario == reference:
            continue
            
        # Ensure same features (should be the case)
        if result['feature_names'] == ref_features:
            diff = result['feature_importance'] - ref_importance
            differences[scenario] = diff
    
    if not differences:
        print("No valid comparisons for difference plot")
        return
    
    # Create difference matrix for heatmap
    scenarios = list(differences.keys())
    n_features = len(ref_features)
    
    # Focus on top features from reference
    top_indices = np.argsort(ref_importance)[::-1][:30]  # Top 30 features
    
    diff_matrix = np.zeros((len(scenarios), len(top_indices)))
    
    for i, scenario in enumerate(scenarios):
        for j, feat_idx in enumerate(top_indices):
            diff_matrix[i, j] = differences[scenario][feat_idx]
    
    # Plot
    plt.figure(figsize=(15, 8))
    
    # Truncate feature names
    top_feature_names = [ref_features[idx] for idx in top_indices]
    display_names = [name[:25] + '...' if len(name) > 25 else name for name in top_feature_names]
    
    # Use diverging colormap centered at zero
    norm = TwoSlopeNorm(vmin=diff_matrix.min(), vcenter=0, vmax=diff_matrix.max())
    
    sns.heatmap(diff_matrix,
                xticklabels=display_names,
                yticklabels=scenarios,
                annot=False,
                cmap='RdBu_r',
                norm=norm,
                cbar_kws={'label': f'Importance Difference vs {reference}'})
    
    plt.title(f'Feature Importance Differences Relative to {reference}')
    plt.xlabel('Features (Top 30 from reference)')
    plt.ylabel('Doping Condition')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'importance_differences_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def plot_soap_pair_analysis(results):
    """Analyze and plot SOAP features by atomic pairs, filtered to valid elements only."""
    # Group SOAP features by atomic pairs
    pair_importance = {}
    
    for scenario, result in results.items():
        if result is None:
            continue
            
        feature_names = result['feature_names']
        importance = result['feature_importance']
        
        scenario_pairs = {}
        
        for i, name in enumerate(feature_names):
            if 'SOAP' in name:
                # Use robust pair extraction with filtering
                pair = extract_atomic_pair(name)
                if pair is not None:  # Only include valid pairs
                    if pair not in scenario_pairs:
                        scenario_pairs[pair] = []
                    scenario_pairs[pair].append(importance[i])
        
        # Sum importance for each pair
        for pair, importances in scenario_pairs.items():
            if pair not in pair_importance:
                pair_importance[pair] = {}
            pair_importance[pair][scenario] = np.sum(importances)
    
    if not pair_importance:
        print("No valid SOAP pair data found")
        return
    
    # Create bar plot
    pairs = sorted(list(pair_importance.keys()))
    scenarios = [s for s in results.keys() if results[s] is not None]
    
    print(f"Found valid atomic pairs: {pairs}")  # Debug info
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(pairs))
    width = 0.12  # Adjusted for 6 scenarios
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(scenarios)))
    
    for i, scenario in enumerate(scenarios):
        values = [pair_importance[pair].get(scenario, 0) for pair in pairs]
        ax.bar(x + i * width, values, width, label=scenario, color=colors[i])
    
    ax.set_xlabel('Atomic Pairs')
    ax.set_ylabel('Total SOAP Importance')
    ax.set_title('SOAP Feature Importance by Atomic Pairs (Cr-Sb₂Te₃ System)')
    ax.set_xticks(x + width * (len(scenarios) - 1) / 2)
    ax.set_xticklabels(pairs)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'soap_pairs_importance.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function."""
    print("Starting SOAP Descriptor Analysis...")
    print("="*60)
    
    analyzer = SOAPAnalyzer(DATA_PATHS)
    
    # Prepare datasets for all scenarios
    datasets = {}
    for scenario in DATA_PATHS.keys():
        print(f"\n{'='*20} SCENARIO: {scenario.upper()} {'='*20}")
        dataset = analyzer.prepare_dataset(scenario, n_samples=200)
        datasets[scenario] = dataset
    
    # Train predictors and get feature importance
    results = {}
    for scenario, dataset in datasets.items():
        if dataset is not None:
            try:
                result = train_predictor_and_get_importance(dataset)
                results[scenario] = result
            except Exception as e:
                print(f"Failed to analyze {scenario}: {e}")
                results[scenario] = None
        else:
            results[scenario] = None
    
    # Create visualizations
    print(f"\nCreating visual izations...")
    
    # 1. Feature importance by doping condition
    plot_feature_importance(results)
    
    # 2. Clustered SOAP feature heatmap by radial/angular quantum numbers
    plot_soap_feature_heatmap(results)
    
    # 3. Importance differences heatmap
    plot_importance_differences(results, reference='1/8 ML')
    
    # 4. SOAP pairs analysis (filtered to valid elements)
    plot_soap_pair_analysis(results)
    
    # Save results
    print(f"\nSaving analysis results...")
    results_file = os.path.join(OUTPUT_DIR, 'soap_analysis_results.npz')
    np.savez_compressed(results_file, results=results)
    
    print("\n" + "="*60)
    print("SOAP Descriptor Analysis Complete!")
    print(f"Results saved in '{OUTPUT_DIR}' directory")
    print("="*60)

if __name__ == "__main__":
    main()