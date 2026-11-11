#!/usr/bin/env python3
"""
MACE SHAP Analysis - MACE Model Comparison Version (Native MACE Evaluation)

This script performs SHAP analysis to understand differences between MACE models
by using their predictions as features. It uses the native MACE evaluation pipeline
for robust energy and force predictions, avoiding ASE Calculator wrapper issues.
"""

import os
import re
import numpy as np
import pandas as pd
import warnings
import traceback
from pathlib import Path

# Core scientific packages
from ase.io import read
from sklearn.ensemble import RandomForestRegressor
import shap
import torch
from tqdm import tqdm

# Import MACE components for native evaluation
from mace import data, tools
from mace.tools import torch_geometric, utils

warnings.filterwarnings('ignore')

# --- Configuration ---

# Model and Data Paths
MODEL_PATHS = {
    "MACE Scratch": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE_models_l1_0802/mace_l1_0802_compiled.model",
    "MACE Foundation": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/finetuned_MACE_compiled.model",
    "MACE FT - 600K": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_600K/finetuned_MACE_multihead0804_compiled.model",
    "MACE FT - Multi-T": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_Multi_T/finetuned_MACE_multihead0804.model",
}

DATA_PATHS = {
    "1/8 ML": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/2.aimd/3-2.2-layer_CrSb2Te3/converted_xyz/2L_octo_Cr1_v2_600K_aimd_1.xyz",
    "1/4 ML": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/2.aimd/3-2.2-layer_CrSb2Te3/converted_xyz/2L_octo_Cr2_600K_aimd_1.xyz",
    "3/8 ML": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/2.aimd/3-2.2-layer_CrSb2Te3/converted_xyz/2L_octo_Cr3_600K_aimd_1.xyz",
    "1/2 ML": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/2.aimd/3-2.2-layer_CrSb2Te3/converted_xyz/2L_tetra_Cr4_600K_aimd1.xyz",
    "3/4 ML": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/2.aimd/3-2.2-layer_CrSb2Te3/converted_xyz/2L_o3_t3_600K_aimd_1.xyz",
}

# Analysis Parameters
FRAME_SAMPLING_RATE = 5
BATCH_SIZE = 32  # For MACE inference

# Output Configuration
OUTPUT_DIR = 'shap_analysis_results'
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, 'mace_comparison_shap_results.npz')

# --- Classes and Functions ---

class NativeMACEPredictor:
    """
    Native MACE predictor using the robust evaluation pipeline
    """
    def __init__(self, device='cpu'):
        self.device = tools.torch_tools.init_device(device)
        tools.torch_tools.set_default_dtype("float64")
        self.models = {}
        self.model_names = []
        print(f"Initialized MACE predictor on device: {self.device}")
    
    def load_models(self, model_paths):
        """Load all MACE models using native approach"""
        print("Loading MACE models using native evaluation pipeline...")
        
        for model_name, model_path_str in model_paths.items():
            try:
                model_path = Path(model_path_str)
                print(f"  Loading {model_name} from {model_path.name}...")
                
                # Load model directly, handling both .pt and .model files
                if model_path.suffix == ".pt":
                    checkpoint = torch.load(model_path, map_location=self.device)
                    model = checkpoint.get('model', checkpoint)
                else:
                    model = torch.load(model_path, map_location=self.device)
                
                model.to(self.device)
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                
                self.models[model_name] = model
                self.model_names.append(model_name)
                print(f"  ✓ Successfully loaded {model_name}")
                
            except Exception as e:
                print(f"  ✗ Failed to load {model_name}: {e}")
                traceback.print_exc()
        
        print(f"Successfully loaded {len(self.models)} models")
    
    def predict_batch(self, atoms_list, model_name):
        """Predict energies and forces for a batch of structures using native MACE"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        
        try:
            # Prepare data in MACE's native format
            z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
            configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
            
            atomic_data_list = [
                data.AtomicData.from_config(
                    config, z_table=z_table, cutoff=float(model.r_max)
                )
                for config in configs
            ]
            
            # Create data loader
            data_loader = torch_geometric.dataloader.DataLoader(
                dataset=atomic_data_list,
                batch_size=min(BATCH_SIZE, len(atomic_data_list)),
                shuffle=False,
                drop_last=False,
            )
            
            # Run inference
            all_energies = []
            all_forces = []
            
            with torch.no_grad():
                for batch in data_loader:
                    batch = batch.to(self.device)
                    output = model(batch.to_dict())
                    
                    # Extract energies
                    energies = tools.torch_tools.to_numpy(output["energy"])
                    all_energies.extend(energies)
                    
                    # Extract forces
                    forces = tools.torch_tools.to_numpy(output["forces"])
                    
                    # Reshape forces to match expected format
                    batch_ptr = batch.ptr.cpu().numpy()
                    forces_per_structure = []
                    
                    for i in range(len(batch_ptr) - 1):
                        start_idx = batch_ptr[i]
                        end_idx = batch_ptr[i + 1]
                        structure_forces = forces[start_idx:end_idx]
                        forces_per_structure.append(structure_forces)
                    
                    all_forces.extend(forces_per_structure)
            
            return np.array(all_energies), all_forces
            
        except Exception as e:
            print(f"    Error in native MACE prediction for {model_name}: {e}")
            # Return NaN arrays as fallback
            n_structures = len(atoms_list)
            energies = np.full(n_structures, np.nan)
            forces = [np.full((len(atoms), 3), np.nan) for atoms in atoms_list]
            return energies, forces

class MACEModelFeatureExtractor:
    """
    Feature extractor that uses native MACE model predictions as features
    """
    def __init__(self, device='cpu'):
        self.predictor = NativeMACEPredictor(device)
        self.model_names = []
        
    def parse_xyz_robust(self, filepath, frame_step=1):
        """Robustly parse XYZ file to extract ground-truth energies, forces, and atoms objects."""
        print(f"  > Reading frames from {os.path.basename(filepath)}...")
        try:
            atoms_list = read(filepath, index=f"::{frame_step}")
            if not atoms_list:
                raise IOError("No frames could be read.")
        except Exception as e:
            print(f"Error reading file with ASE: {e}")
            return None, None, None

        energies, forces = [], []
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
            
            try:
                frame_forces = atoms.get_forces()
                forces.append(frame_forces.flatten())
            except (RuntimeError, IOError, AttributeError):
                forces.append(np.zeros(len(atoms) * 3))

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
                return None, None, None

        if len(energies) != len(atoms_list) or len(forces) != len(atoms_list):
            print(f"Error: Mismatch in parsed counts. Frames: {len(atoms_list)}, Energies: {len(energies)}, Forces: {len(forces)}")
            return None, None, None

        energies = np.array(energies, dtype=float)
        forces = np.array(forces)
        print(f"  > Successfully parsed {len(atoms_list)} frames (Ground-truth energy source: '{found_key}')")
        return energies, forces, atoms_list

    def extract_mace_predictions(self, model_paths, atoms_list, dft_energies, dft_forces):
        """Extract predictions from all MACE models using native evaluation"""
        print("  > Extracting MACE model predictions using native evaluation...")
        
        # Load all models
        self.predictor.load_models(model_paths)
        self.model_names = self.predictor.model_names
        
        if not self.model_names:
            raise RuntimeError("No MACE models were successfully loaded")
        
        model_predictions = {}
        
        # Get predictions from all models
        for model_name in self.model_names:
            print(f"    Getting predictions from {model_name}...")
            
            try:
                energies, forces = self.predictor.predict_batch(atoms_list, model_name)
                
                # Validate predictions
                valid_energy_mask = ~np.isnan(energies)
                n_valid = np.sum(valid_energy_mask)
                
                print(f"      Got {n_valid}/{len(energies)} valid energy predictions")
                
                model_predictions[model_name] = {
                    'energies': energies,
                    'forces': forces,
                    'valid_mask': valid_energy_mask
                }
                
            except Exception as e:
                print(f"      Error getting predictions from {model_name}: {e}")
                # Store NaN predictions for failed models
                n_structures = len(atoms_list)
                model_predictions[model_name] = {
                    'energies': np.full(n_structures, np.nan),
                    'forces': [np.full((len(atoms), 3), np.nan) for atoms in atoms_list],
                    'valid_mask': np.zeros(n_structures, dtype=bool)
                }
        
        return model_predictions

    def extract_model_comparison_features(self, model_predictions, dft_energies):
        """Extract features that highlight model differences and consensus"""
        print("  > Creating model comparison features...")
        
        n_frames = len(dft_energies)
        comparison_features = []
        
        for i in range(n_frames):
            frame_features = []
            
            # 1. Individual model predictions and errors
            energy_predictions = []
            force_magnitudes = []
            valid_predictions = []
            
            for model_name in self.model_names:
                pred_data = model_predictions[model_name]
                energy_pred = pred_data['energies'][i]
                forces = pred_data['forces'][i]
                is_valid = pred_data['valid_mask'][i]
                
                # Individual model energy prediction
                frame_features.append(energy_pred if is_valid else 0.0)
                
                # Individual model energy error from DFT
                if is_valid and not np.isnan(dft_energies[i]):
                    energy_error = energy_pred - dft_energies[i]
                    frame_features.append(energy_error)
                else:
                    frame_features.append(0.0)
                
                # Individual model force statistics
                if is_valid and not np.any(np.isnan(forces)):
                    if len(forces.shape) == 1:
                        forces = forces.reshape(-1, 3)
                    force_magnitudes_frame = np.linalg.norm(forces, axis=1)
                    mean_force_mag = np.mean(force_magnitudes_frame)
                    max_force_mag = np.max(force_magnitudes_frame)
                    
                    frame_features.extend([mean_force_mag, max_force_mag])
                    force_magnitudes.append(mean_force_mag)
                else:
                    frame_features.extend([0.0, 0.0])
                
                # Store valid predictions for consensus analysis
                if is_valid:
                    energy_predictions.append(energy_pred)
                    valid_predictions.append(model_name)
            
            # 2. Model consensus features
            if len(energy_predictions) > 1:
                # Energy consensus statistics
                energy_consensus = np.mean(energy_predictions)
                energy_disagreement = np.std(energy_predictions)
                energy_range = np.ptp(energy_predictions)
                
                frame_features.extend([
                    energy_consensus,
                    energy_disagreement, 
                    energy_range,
                    len(energy_predictions)  # Number of valid predictions
                ])
                
                # Consensus error from DFT
                if not np.isnan(dft_energies[i]):
                    consensus_error = energy_consensus - dft_energies[i]
                    frame_features.append(consensus_error)
                else:
                    frame_features.append(0.0)
                
                # Force consensus statistics
                if len(force_magnitudes) > 1:
                    force_consensus = np.mean(force_magnitudes)
                    force_disagreement = np.std(force_magnitudes)
                    frame_features.extend([force_consensus, force_disagreement])
                else:
                    frame_features.extend([0.0, 0.0])
                
                # 3. Model reliability indicators
                if not np.isnan(dft_energies[i]):
                    # Which model is closest to DFT?
                    dft_errors = [abs(pred - dft_energies[i]) for pred in energy_predictions]
                    if dft_errors:
                        best_pred_idx = np.argmin(dft_errors)
                        best_model = valid_predictions[best_pred_idx]
                        
                        # One-hot encoding for best model
                        for model_name in self.model_names:
                            frame_features.append(1.0 if model_name == best_model else 0.0)
                    else:
                        frame_features.extend([0.0] * len(self.model_names))
                else:
                    frame_features.extend([0.0] * len(self.model_names))
                
            else:
                # Fallback if insufficient valid predictions
                n_consensus_features = 5  # consensus, disagreement, range, n_valid, consensus_error
                n_force_consensus = 2     # force_consensus, force_disagreement
                n_reliability = len(self.model_names)  # best model indicators
                
                fallback_features = [0.0] * (n_consensus_features + n_force_consensus + n_reliability)
                frame_features.extend(fallback_features)
            
            comparison_features.append(frame_features)
        
        return np.array(comparison_features)

    def get_feature_names(self):
        """Generate feature names for MACE comparison features"""
        feature_names = []
        
        # Individual model features
        for model_name in self.model_names:
            safe_name = model_name.replace(" ", "_").replace("-", "_")
            feature_names.extend([
                f"{safe_name}_energy_pred",
                f"{safe_name}_energy_error",
                f"{safe_name}_force_mean",
                f"{safe_name}_force_max"
            ])
        
        # Consensus features
        feature_names.extend([
            "energy_consensus",
            "energy_disagreement",
            "energy_range",
            "n_valid_predictions",
            "consensus_error_from_dft",
            "force_consensus",
            "force_disagreement"
        ])
        
        # Best model indicators
        for model_name in self.model_names:
            safe_name = model_name.replace(" ", "_").replace("-", "_")
            feature_names.append(f"{safe_name}_is_best_model")
        
        return feature_names

    def get_feature_categories(self):
        """Return feature categorization for analysis"""
        categories = {
            'Individual_Model_Predictions': {
                'energy_predictions': 'Individual model energy predictions',
                'energy_errors': 'Individual model errors from DFT',
                'force_statistics': 'Individual model force statistics'
            },
            'Model_Consensus': {
                'consensus_statistics': 'Agreement/disagreement between models',
                'consensus_accuracy': 'How well consensus matches DFT',
                'prediction_reliability': 'Number of valid predictions'
            },
            'Model_Reliability': {
                'best_model_indicators': 'Which model performs best for each structure'
            }
        }
        return categories

class MACEModelAnalyzer:
    """Main analyzer class for comparing MACE models using native evaluation"""
    def __init__(self, model_paths, data_paths, device='cpu'):
        self.model_paths = model_paths
        self.data_paths = data_paths
        self.device = device
        self.feature_extractor = MACEModelFeatureExtractor(device)
    
    def prepare_dataset(self, scenario, n_samples=300):
        """Prepare dataset for model comparison analysis"""
        print(f"\nPreparing dataset for scenario: {scenario}...")
        
        trajectory_file = self.data_paths[scenario]
        if not os.path.exists(trajectory_file):
            print(f"Trajectory file not found: {trajectory_file}")
            return None
        
        # Parse ground-truth DFT data from the XYZ file
        dft_energies, dft_forces, atoms_list = self.feature_extractor.parse_xyz_robust(
            trajectory_file, frame_step=FRAME_SAMPLING_RATE
        )
        
        if atoms_list is None: return None
        
        if len(atoms_list) > n_samples:
            atoms_list = atoms_list[:n_samples]
            dft_energies = dft_energies[:n_samples]
            dft_forces = dft_forces[:n_samples]
        
        try:
            # Get predictions from all MACE models using native evaluation
            model_predictions = self.feature_extractor.extract_mace_predictions(
                self.model_paths, atoms_list, dft_energies, dft_forces
            )
            
            # Create comparison features
            features = self.feature_extractor.extract_model_comparison_features(
                model_predictions, dft_energies
            )
            
            return {
                'features': features,
                'energies': dft_energies,
                'forces': dft_forces,
                'atoms_list': atoms_list,
                'feature_names': self.feature_extractor.get_feature_names(),
                'feature_categories': self.feature_extractor.get_feature_categories(),
                'model_predictions': model_predictions
            }
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            traceback.print_exc()
            return None

class ModelComparisonPredictor:
    """ML model to predict DFT energy from MACE model comparison features"""
    def __init__(self, training_data):
        self.model = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, max_depth=8,
            min_samples_split=5, min_samples_leaf=2
        )
        
        X = training_data['features']
        y = training_data['energies']
        
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid_mask], y[valid_mask]
        
        if len(X) == 0:
            raise ValueError("No valid training data after removing NaN values")
        
        if len(X) < 10:
            print(f"    Warning: Very small dataset ({len(X)} samples), using simple model")
            self.model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
        
        self.model.fit(X, y)
        print(f"    Trained model comparison predictor on {len(X)} samples.")
        print(f"    R² score: {self.model.score(X, y):.4f}")
    
    def predict(self, X):
        return self.model.predict(X)

def perform_model_comparison_shap_analysis(analyzer, scenario, dataset, max_samples=200):
    """Perform SHAP analysis for model comparison"""
    print(f"  Performing SHAP analysis for scenario: {scenario}...")
    if dataset is None: return None
    
    try:
        predictor = ModelComparisonPredictor(dataset)
        X = dataset['features']
        
        if np.any(np.isnan(X)):
            print("    Warning: NaN values found in features. Replacing with 0.")
            X = np.nan_to_num(X, nan=0.0)
        
        n_samples = min(len(X), max_samples)
        X_sampled = X[:n_samples]
        
        background_size = min(50, max(5, n_samples // 4))
        test_size = min(100, n_samples - background_size)
        
        if test_size <= 0:
             print(f"    Warning: Not enough samples ({n_samples}) to perform SHAP analysis. Skipping.")
             return None

        background_data = X_sampled[:background_size]
        test_data = X_sampled[background_size:background_size + test_size]
        
        print(f"    Using {background_size} background samples and {test_size} test samples")
        
        explainer = shap.TreeExplainer(predictor.model)
        shap_values = explainer(test_data)
        
        feature_names = dataset['feature_names']
        
        num_shap_features = shap_values.values.shape[1]
        if len(feature_names) != num_shap_features:
            print(f"  Warning: Mismatch between feature names ({len(feature_names)}) and SHAP values ({num_shap_features}). Adjusting...")
            feature_names = feature_names[:num_shap_features]
            while len(feature_names) < num_shap_features:
                feature_names.append(f'unknown_feature_{len(feature_names)}')

        results = {
            'scenario': scenario,
            'shap_values': shap_values.values,
            'base_values': shap_values.base_values,
            'data': shap_values.data,
            'feature_importance': np.mean(np.abs(shap_values.values), axis=0),
            'feature_names': feature_names,
            'feature_categories': dataset['feature_categories'],
            'model_performance': {
                'r2_score': predictor.model.score(X_sampled, dataset['energies'][:len(X_sampled)])
            }
        }
        
        print(f"    ✓ Completed SHAP analysis for {scenario}")
        return results
        
    except Exception as e:
        print(f"    ✗ SHAP analysis failed for {scenario}: {e}")
        traceback.print_exc()
        return None

def main():
    """Main execution function for MACE model comparison analysis."""
    print("Starting MACE Model Comparison SHAP Analysis (Native Evaluation)...")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    analyzer = MACEModelAnalyzer(MODEL_PATHS, DATA_PATHS, device=device)
    
    print(f"Will analyze {len(MODEL_PATHS)} MACE models:")
    for model_name in MODEL_PATHS.keys():
        print(f"  - {model_name}")
        
    all_results = {}
    
    for scenario in DATA_PATHS.keys():
        print(f"\n{'='*20} SCENARIO: {scenario.upper()} {'='*20}")
        
        dataset = analyzer.prepare_dataset(scenario, n_samples=150)
        
        if dataset:
            result = perform_model_comparison_shap_analysis(analyzer, scenario, dataset, max_samples=100)
            all_results[scenario] = result
        else:
            all_results[scenario] = None
                
    print(f"\nSaving analysis results to '{OUTPUT_FILENAME}'...")
    try:
        np.savez_compressed(OUTPUT_FILENAME, results=all_results)
        print("✓ Successfully saved results.")
    except Exception as e:
        print(f"✗ Failed to save results: {e}")

    print("\n" + "="*60)
    print("MACE Model Comparison SHAP Analysis Complete!")
    print("This analysis reveals:")
    print("  - Which model predictions are most important for accuracy")
    print("  - How model consensus affects prediction quality")
    print("  - Which models perform best under different Cr concentrations")
    print("  - Where models disagree and why")
    print("="*60)

if __name__ == "__main__":
    main()