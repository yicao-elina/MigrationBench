#!/usr/bin/env python3
"""
Part 1: MACE Energy Prediction for NEB Pathway (v3 - Native MACE Evaluation)

This script calculates potential energies for NEB images by bypassing the ASE
Calculator wrapper and using the native MACE data loading and model evaluation
pipeline, as inspired by the official MACE `evaluate.py` script. This approach
is more robust against API version mismatches.

Additionally, it parses ground truth energies from the neb.out file and includes
them in the output for comparison.

Workflow:
1.  Defines paths to MACE models (.model and .pt files).
2.  Loads the sequence of atomic images from the ground-truth XYZ file.
3.  Parses ground truth energies from the neb.out file.
4.  Iterates through each MACE model path.
5.  Loads the model directly using torch.load, handling both checkpoint (.pt)
    and compiled (.model) formats.
6.  Converts all ASE Atoms objects into a list of MACE's native `AtomicData` format.
7.  Uses a `torch_geometric.DataLoader` to create batches.
8.  Calls `model(batch.to_dict())` directly to get the energy predictions.
9.  Saves the results including ground truth to a structured CSV file ('neb_predictions.csv').
"""
import pandas as pd
import torch
from ase.io import read
from tqdm import tqdm
from pathlib import Path
import re

# Import necessary components from MACE library
from mace import data, tools
from mace.tools import torch_geometric, utils

# --- CONFIGURATION ---

# Paths to the trained MACE models you want to compare.
MODEL_PATHS = {
    "MACE Scratch": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE_models_l1_0802/mace_l1_0802_compiled.model",
    "MACE Foundation": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-omat/finetuned_MACE_compiled.model",
    "MACE FT - 600K": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_600K/finetuned_MACE_multihead0804_compiled.model",
    "MACE FT - Multi-T": "/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/3.fine-tuning/2-layer/MACE-multihead_Multi_T/finetuned_MACE_multihead0804.model",
}

# Path to the ground-truth XYZ file containing the NEB images.
GROUND_TRUTH_XYZ_PATH = "1-7/sb2te3.xyz"
GROUND_TRUTH_ENERGYS_PATH = "1-7/neb.out"
# Output filename for the raw prediction data
OUTPUT_CSV_FILENAME = "neb_predictions.csv"

# --- HELPER FUNCTIONS ---

def parse_neb_energies(filename):
    """
    Parse neb.out file and extract the latest energy values.
    
    Args:
        filename (str): Path to the neb.out file
        
    Returns:
        dict: Dictionary mapping image index to energy value
    """
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Find all energy blocks
    blocks = []
    
    # Split by the header line
    header_pattern = r'image\s+energy \(eV\)\s+error \(eV/A\)\s+frozen'
    sections = re.split(header_pattern, content)
    
    for section in sections[1:]:  # Skip first part before any header
        current_block = []
        lines = section.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try to parse as data line
            parts = line.split()
            if len(parts) >= 4:
                try:
                    # Check if first element is a number
                    image_num = int(parts[0])
                    energy = float(parts[1])
                    error = float(parts[2])
                    frozen = parts[3]
                    
                    current_block.append({
                        'image': image_num,
                        'energy': energy,
                        'error': error,
                        'frozen': frozen
                    })
                except ValueError:
                    # Not a data line, might be end of block
                    if current_block:  # If we have data, save the block
                        break
        
        if current_block:
            blocks.append(current_block)
    
    if not blocks:
        print("Warning: No energy data found in neb.out file!")
        return {}
    
    # Get the latest block and convert to dictionary
    latest_block = blocks[-1]
    print(f"Found {len(blocks)} energy blocks in neb.out, using the latest one with {len(latest_block)} images")
    
    # Convert to dictionary with 0-based indexing to match image_index
    energy_dict = {entry['image'] - 1: entry['energy'] for entry in latest_block}
    
    return energy_dict

def predict_neb_energies_native():
    """
    Main function to run MACE predictions using the native evaluation pipeline.
    """
    # Set device and default dtype
    device = tools.torch_tools.init_device("cuda" if torch.cuda.is_available() else "cpu")
    tools.torch_tools.set_default_dtype("float64")
    print(f"Using device: {device}")

    # 1. Load the NEB images from the XYZ file
    try:
        neb_images = read(GROUND_TRUTH_XYZ_PATH, index=":")
        print(f"Loaded {len(neb_images)} NEB images from '{GROUND_TRUTH_XYZ_PATH}'")
    except FileNotFoundError:
        print(f"Error: Ground-truth XYZ file not found at '{GROUND_TRUTH_XYZ_PATH}'")
        return
    except Exception as e:
        print(f"Error loading NEB images: {e}")
        return

    # 2. Parse ground truth energies from neb.out
    try:
        ground_truth_energies = parse_neb_energies(GROUND_TRUTH_ENERGYS_PATH)
        print(f"Loaded ground truth energies for {len(ground_truth_energies)} images")
    except FileNotFoundError:
        print(f"Warning: Ground-truth energy file not found at '{GROUND_TRUTH_ENERGYS_PATH}'")
        ground_truth_energies = {}
    except Exception as e:
        print(f"Warning: Error loading ground truth energies: {e}")
        ground_truth_energies = {}

    # List to store final result dictionaries
    results_list = []

    # 3. Loop through each MACE model to evaluate
    for model_name, model_path_str in MODEL_PATHS.items():
        print(f"\n--- Evaluating model: {model_name} ---")
        model_path = Path(model_path_str)
        
        try:
            # Load model directly, handling both .pt and .model files
            if model_path.suffix == ".pt":
                print(f"Loading PyTorch checkpoint (.pt): {model_path.name}")
                checkpoint = torch.load(model_path, map_location=device)
                model = checkpoint.get('model', checkpoint) # Handle dict or raw model
            else:
                print(f"Loading TorchScript model (.model): {model_path.name}")
                model = torch.load(model_path, map_location=device)
            
            model.to(device)
            model.eval() # Set model to evaluation mode
            for param in model.parameters():
                param.requires_grad = False

        except Exception as e:
            print(f"Error loading model '{model_name}' from '{model_path}': {e}")
            continue

        # 4. Prepare data in MACE's native format
        z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
        configs = [data.config_from_atoms(atoms) for atoms in neb_images]
        
        atomic_data_list = [
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max)
            )
            for config in configs
        ]
        
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=atomic_data_list,
            batch_size=len(atomic_data_list), # Process all images in one batch
            shuffle=False,
            drop_last=False,
        )

        # 5. Run inference
        try:
            for batch in data_loader:
                batch = batch.to(device)
                # Call model directly with the batch dictionary
                output = model(batch.to_dict())
                energies = tools.torch_tools.to_numpy(output["energy"])
                
                for i, energy in enumerate(energies):
                    # Get ground truth energy if available
                    gt_energy = ground_truth_energies.get(i, float('nan'))
                    
                    results_list.append({
                        "model_name": model_name,
                        "image_index": i,
                        "predicted_energy_eV": energy,
                        "ground_truth_energy_eV": gt_energy,
                        "error_eV": energy - gt_energy if not pd.isna(gt_energy) else float('nan')
                    })
            print(f"Successfully calculated {len(energies)} energies for {model_name}.")

        except Exception as e:
            print(f"\nError during model inference for {model_name}: {e}")
            # Add NaNs for this model if inference fails
            for i in range(len(neb_images)):
                gt_energy = ground_truth_energies.get(i, float('nan'))
                results_list.append({
                    "model_name": model_name,
                    "image_index": i,
                    "predicted_energy_eV": float('nan'),
                    "ground_truth_energy_eV": gt_energy,
                    "error_eV": float('nan')
                })

    # 6. Convert results to a pandas DataFrame and save to CSV
    if not results_list:
        print("No results were generated. Exiting.")
        return

    df_results = pd.DataFrame(results_list)
    
    # Sort by model_name and image_index for better readability
    df_results = df_results.sort_values(['model_name', 'image_index'])
    
    # Save to CSV
    df_results.to_csv(OUTPUT_CSV_FILENAME, index=False)
    print(f"\nSuccessfully saved all predictions to '{OUTPUT_CSV_FILENAME}'")
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    for model_name in df_results['model_name'].unique():
        model_data = df_results[df_results['model_name'] == model_name]
        valid_errors = model_data['error_eV'].dropna()
        if len(valid_errors) > 0:
            mae = valid_errors.abs().mean()
            rmse = (valid_errors ** 2).mean() ** 0.5
            print(f"\n{model_name}:")
            print(f"  MAE: {mae:.6f} eV")
            print(f"  RMSE: {rmse:.6f} eV")
        else:
            print(f"\n{model_name}: No valid error data")

if __name__ == "__main__":
    predict_neb_energies_native()