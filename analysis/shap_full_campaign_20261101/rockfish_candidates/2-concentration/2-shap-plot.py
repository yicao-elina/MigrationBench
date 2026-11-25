#!/usr/bin/env python3
"""
MACE Model SHAP Analysis - Part 2: Visualization

This script loads pre-computed SHAP analysis results and generates a suite of
publication-quality visualizations. It is designed to be run after
'run_shap_analysis.py' and uses the jhu_colors package for consistent,
professional aesthetics suitable for high-impact scientific communication.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import jhu_colors
from jhu_colors import get_jhu_color

# --- Configuration ---

# Input data file from the analysis script
# INPUT_FILE = 'shap_analysis_results/shap_results.npz'
INPUT_FILE = "soap_analysis_results1/soap_analysis_results.npz"
PLOT_OUTPUT_DIR = 'plots'

# These must match the analysis script to correctly label plots
MODEL_PATHS = {
    "MACE Scratch": "...",
    "MACE Foundation": "...",
    "MACE FT - 600K": "...",
    "MACE FT - Multi-T": "...",
}
DATA_PATHS = {
    "normal_diffusion": "...",
    "in_gap": "...",
    "deep_penetration": "..."
}
# Parameters needed for feature type mapping
N_NEIGHBOR_BINS = 30
PCA_DIM = 20

# Color mapping for models
MODEL_COLORS = {
    "MACE Scratch": get_jhu_color('Orange'), 
    "MACE Foundation": get_jhu_color('Red'),
    "MACE FT - 600K": get_jhu_color('Heritage Blue'),
    "MACE FT - Multi-T": get_jhu_color('Spirit Blue')
}


def create_feature_importance_heatmap(all_results, models, scenarios, output_dir):
    """Creates a feature importance heatmap across models and scenarios."""
    figsize_ratio = (10, 4)
    fig, axes = plt.subplots(
        len(scenarios), 1, 
        figsize=(figsize_ratio[0], figsize_ratio[1] * len(scenarios)),
        squeeze=False
    )
    axes = axes.flatten()

    for i, scenario in enumerate(scenarios):
        importance_data = []
        model_labels = [model for model in models if all_results[scenario].get(model) is not None]
        
        if not model_labels:
            axes[i].text(0.5, 0.5, f'No valid data for {scenario}', ha='center', va='center', transform=axes[i].transAxes)
            continue
            
        for model in model_labels:
            result = all_results[scenario][model]
            importance = result['feature_importance'][:50] # Top 50 features
            importance_data.append(importance)
        
        importance_matrix = np.array(importance_data)
        im = axes[i].imshow(importance_matrix, aspect='auto', cmap='jhu')
        
        axes[i].set_title(f'Feature Importance: {scenario.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Models', fontsize=12)
        axes[i].set_yticks(np.arange(len(model_labels)))
        axes[i].set_yticklabels(model_labels)
        axes[i].set_xlabel('Feature Index (Top 50)', fontsize=12)
        
        cbar = fig.colorbar(im, ax=axes[i])
        cbar.set_label('Mean |SHAP Value|', fontsize=12)
        
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    filepath = os.path.join(output_dir, 'fig_shap_heatmap.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved heatmap to {filepath}")

def create_feature_type_analysis(all_results, models, scenarios, output_dir):
    """Analyzes and plots importance aggregated by feature type."""
    feature_types = {
        'Energy': [0],
        'Force': list(range(1, 7)),
        'Neighbor': list(range(7, 7 + N_NEIGHBOR_BINS)),
        'SOAP': list(range(7 + N_NEIGHBOR_BINS, 7 + N_NEIGHBOR_BINS + PCA_DIM)),
        'Structural': list(range(7 + N_NEIGHBOR_BINS + PCA_DIM, 7 + N_NEIGHBOR_BINS + PCA_DIM + 16))
    }
    
    num_scenarios = len(scenarios)
    fig, axes = plt.subplots(1, num_scenarios, figsize=(6 * num_scenarios, 5), squeeze=False)
    axes = axes.flatten()
    
    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        model_labels = [model for model in models if all_results[scenario].get(model) is not None]
        
        if not model_labels:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{scenario.replace("_", " ").title()}', fontweight='bold')
            continue

        type_importance = {ftype: [] for ftype in feature_types.keys()}
        for model in model_labels:
            result = all_results[scenario][model]
            importance = result['feature_importance']
            for ftype, indices in feature_types.items():
                valid_indices = [idx for idx in indices if idx < len(importance)]
                avg_importance = np.mean(importance[valid_indices]) if valid_indices else 0
                type_importance[ftype].append(avg_importance)

        x = np.arange(len(feature_types))
        width = 0.8 / len(model_labels)
        
        for j, model in enumerate(model_labels):
            values = [type_importance[ftype][j] for ftype in feature_types.keys()]
            ax.bar(x + j*width - width*(len(model_labels)-1)/2, values, width, label=model, color=MODEL_COLORS.get(model))
        
        ax.set_title(f'{scenario.replace("_", " ").title()}', fontweight='bold')
        if i == 0:
            ax.set_ylabel('Mean |SHAP Value|', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(feature_types.keys(), rotation=45, ha='right')
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=len(models), fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    filepath = os.path.join(output_dir, 'fig_shap_feature_type_analysis.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved feature type analysis to {filepath}")


def create_individual_shap_plots(all_results, models, scenarios, output_dir):
    """Creates individual SHAP summary bar plots for each model."""
    for model in models:
        num_scenarios = len(scenarios)
        fig, axes = plt.subplots(1, num_scenarios, figsize=(6 * num_scenarios, 5), squeeze=False)
        axes = axes.flatten()
        
        has_data = False
        for i, scenario in enumerate(scenarios):
            result = all_results[scenario].get(model)
            
            if result:
                has_data = True
                importance = result['feature_importance']
                top_indices = np.argsort(importance)[-15:]
                top_importance = importance[top_indices]
                top_names = [result['feature_names'][j] for j in top_indices]
                
                axes[i].barh(np.arange(len(top_names)), top_importance, color=MODEL_COLORS.get(model, 'gray'))
                axes[i].set_yticks(np.arange(len(top_names)))
                axes[i].set_yticklabels(top_names, fontsize=9)
                axes[i].set_title(f'{scenario.replace("_", " ").title()}', fontweight='bold')
                if i == 0:
                    axes[i].set_xlabel('Mean |SHAP Value|', fontsize=12)
            else:
                axes[i].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{scenario.replace("_", " ").title()}', fontweight='bold')

        if has_data:
            fig.suptitle(f'Top 15 Features for {model}', fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            model_savename = model.replace(" ", "_").replace("-", "_")
            filepath = os.path.join(output_dir, f'fig_shap_summary_{model_savename}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved individual plot for {model} to {filepath}")
        
        plt.close(fig)

def main():
    """Main execution function for plotting."""
    print("Starting MACE SHAP Visualization...")
    print("="*60)
    
    # Create output directory for plots
    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Analysis results file not found at '{INPUT_FILE}'")
        print("Please run 'run_shap_analysis.py' first.")
        return
        
    # Load data
    print(f"Loading analysis data from '{INPUT_FILE}'...")
    with np.load(INPUT_FILE, allow_pickle=True) as data:
        all_results = data['results'].item()
    print("✓ Data loaded successfully.")
    
    # Get model and scenario names from the loaded data structure
    scenarios = list(all_results.keys())
    models = list(MODEL_PATHS.keys()) # Assumes models are consistent
    
    # Generate plots
    print("\nGenerating visualizations...")
    create_feature_importance_heatmap(all_results, models, scenarios, PLOT_OUTPUT_DIR)
    create_feature_type_analysis(all_results, models, scenarios, PLOT_OUTPUT_DIR)
    create_individual_shap_plots(all_results, models, scenarios, PLOT_OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("MACE SHAP Visualization Complete!")
    print(f"Plots saved in '{PLOT_OUTPUT_DIR}' directory.")
    print("="*60)

if __name__ == "__main__":
    main()