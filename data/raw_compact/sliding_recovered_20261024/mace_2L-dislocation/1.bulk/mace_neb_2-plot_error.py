#!/usr/bin/env python3
"""
Part 2: Post-Processing and Plotting of NEB Results

This script reads pre-computed energy predictions and ground-truth data to
generate a comparison plot of NEB migration barriers.

Workflow:
1.  Loads the MACE model predictions and ground truth from the CSV file generated 
    by the prediction script.
2.  Processes the data:
    - Normalizes all energy profiles so the initial state (image 0) has an
      energy of zero.
    - Groups predictions by model name.
3.  Calculates the activation energy (barrier height) and the prediction error
    for each model.
4.  Generates a publication-quality plot comparing the energy profiles.
5.  Prints a summary table of the barrier height errors.

This script should be run AFTER the prediction script.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jhu_colors
from jhu_colors import get_jhu_color

# --- CONFIGURATION ---

# Path to the CSV file containing the MACE predictions and ground truth.
PREDICTIONS_CSV_PATH = "neb_predictions.csv"

# Output filename for the plot
OUTPUT_PLOT_FILENAME = "neb_migration_barrier_comparison_final.png"

# --- MAIN SCRIPT ---

def plot_neb_comparison():
    """
    Main function to load processed data and generate the comparison plot.
    """
    # 1. Load predictions and ground truth from CSV
    try:
        df_preds = pd.read_csv(PREDICTIONS_CSV_PATH)
        print(f"Loaded predictions for {df_preds['model_name'].nunique()} models from '{PREDICTIONS_CSV_PATH}'")
    except FileNotFoundError:
        print(f"Error: Predictions file not found at '{PREDICTIONS_CSV_PATH}'")
        print("Please run the prediction script first.")
        return

    # 2. Extract ground truth data (same for all models)
    # Get ground truth from the first model's data
    gt_data = df_preds[df_preds['model_name'] == df_preds['model_name'].unique()[0]].sort_values('image_index')
    
    # Check if ground truth data exists
    if 'ground_truth_energy_eV' not in df_preds.columns or gt_data['ground_truth_energy_eV'].isnull().all():
        print("Error: No ground truth energy data found in the CSV file.")
        return
    
    # Extract ground truth energies
    image_indices = gt_data['image_index'].values
    dft_energies = gt_data['ground_truth_energy_eV'].values
    
    # Create reaction coordinate (normalized to 0-1)
    num_images = len(image_indices)
    reaction_coords = np.linspace(0, 1, num_images)
    
    # Normalize ground truth energies
    dft_energies_normalized = dft_energies - dft_energies[0]
    dft_barrier = np.max(dft_energies_normalized)
    print(f"Loaded ground-truth DFT data. Barrier height: {dft_barrier:.3f} eV")

    # 3. Initialize the plot
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot the DFT ground truth first
    ax.plot(
        reaction_coords,
        dft_energies_normalized,
        label="DFT (Ground Truth)",
        color=get_jhu_color('Double Black'),
        marker='o',
        markersize=8,
        linestyle='-',
        linewidth=2.5,
        zorder=10
    )

    results = {}
    model_names = df_preds['model_name'].unique()
    colors = ['Heritage Blue', 'Spirit Blue', 'Red', 'Orange', 'Homewood Green']
    markers = ['s', '^', 'v', 'D', 'P']

    # 4. Process and plot data for each model
    for i, model_name in enumerate(model_names):
        model_df = df_preds[df_preds['model_name'] == model_name].sort_values('image_index')
        
        if model_df['predicted_energy_eV'].isnull().any():
            print(f"Warning: Skipping model '{model_name}' due to missing energy values.")
            continue

        pred_energies = model_df['predicted_energy_eV'].values
        
        # Normalize predicted energies
        pred_energies_normalized = pred_energies - pred_energies[0]
        model_barrier = np.max(pred_energies_normalized)
        barrier_error = abs(model_barrier - dft_barrier)

        # Calculate additional metrics
        mae = model_df['error_eV'].abs().mean()
        rmse = np.sqrt((model_df['error_eV'] ** 2).mean())

        results[model_name] = {
            "barrier": model_barrier,
            "error_eV": barrier_error,
            "mae": mae,
            "rmse": rmse,
        }

        # Plot the model's predicted energy profile
        ax.plot(
            reaction_coords,
            pred_energies_normalized,
            label=f"{model_name}",
            color=get_jhu_color(colors[i % len(colors)]),
            marker=markers[i % len(markers)],
            linestyle='--',
            linewidth=2,
            markersize=6
        )

    # 5. Finalize and save the plot
    ax.set_xlabel("Reaction Coordinate", fontsize=14)
    ax.set_ylabel("Relative Energy (eV)", fontsize=14)
    ax.set_title("NEB Migration Barrier: DFT vs. MACE Models", fontsize=16, weight='bold')
    # ax.set_ylim(-1.6, 4)  # Set y-axis limits for better visibility
    ax.legend(fontsize=10, loc='best')
    ax.grid(False)  # Disable grid for cleaner look
    
    # Set y-axis to start from 0
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_FILENAME, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot to '{OUTPUT_PLOT_FILENAME}'")

    # 6. Print a comprehensive summary of the results
    print("\n" + "="*90)
    print("BARRIER HEIGHT COMPARISON SUMMARY")
    print("="*90)
    print(f"{'Model Name':<25} | {'Predicted Barrier (eV)':<22} | {'Barrier Error (eV)':<18} | {'MAE (eV)':<10} | {'RMSE (eV)':<10}")
    print("-" * 90)
    print(f"{'DFT (Ground Truth)':<25} | {dft_barrier:<22.4f} | {'-':<18} | {'-':<10} | {'-':<10}")
    print("-" * 90)

    # save the results to a CSV file
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv("neb_barrier_results.csv", index_label="model_name")
    
    for model_name, data in results.items():
        print(f"{model_name:<25} | {data['barrier']:<22.4f} | {data['error_eV']:<18.4f} | {data['mae']:<10.4f} | {data['rmse']:<10.4f}")
    
    print("="*90)
    
    # 7. Create an additional plot showing the error along the pathway
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    
    for i, model_name in enumerate(model_names):
        model_df = df_preds[df_preds['model_name'] == model_name].sort_values('image_index')
        
        if model_df['error_eV'].isnull().any():
            continue
            
        errors = model_df['error_eV'].values
        
        ax2.plot(
            reaction_coords,
            errors,
            label=f"{model_name}",
            color=get_jhu_color(colors[i % len(colors)]),
            marker=markers[i % len(markers)],
            linestyle='-',
            linewidth=2,
            markersize=6
        )
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel("Reaction Coordinate", fontsize=14)
    ax2.set_ylabel("Prediction Error (eV)", fontsize=14)
    ax2.set_title("Energy Prediction Error Along NEB Path", fontsize=16, weight='bold')
    ax2.legend(fontsize=12, loc='best')
    ax2.grid(False)  # Disable grid for cleaner look
    
    plt.tight_layout()
    error_plot_filename = OUTPUT_PLOT_FILENAME.replace('.png', '_errors.png')
    plt.savefig(error_plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved error plot to '{error_plot_filename}'")

if __name__ == "__main__":
    plot_neb_comparison()