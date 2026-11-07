import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def set_plot_settings():
    """
    Set custom plot settings for matplotlib.
    """
    plt.rc('font', size=16)  # Default text sizes
    plt.rc('axes', titlesize=16)  # Axes title font size
    plt.rc('legend', fontsize=12)  # Legend font size
    plt.rc('xtick', labelsize=16)  # X-axis tick label font size
    plt.rc('ytick', labelsize=16)  # Y-axis tick label font size

    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False


def generate_nature_quality_plot(csv_file, columns_to_plot, labels):
    """
    Reads data from a CSV file and generates a Nature-quality plot.

    Args:
        csv_file (str): Path to the input CSV data file.
        columns_to_plot (list): A list of column names to plot on the y-axis.
        labels (list): The corresponding legend labels for the plotted columns.
    """
    try:
        data_df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ Error: The file '{csv_file}' was not found.")
        sys.exit(1)

    # --- Plotting Setup ---
    set_plot_settings()
    fig, ax = plt.subplots(figsize=(6, 4))

    # A color palette inspired by scientific journals
    colors = ['grey', 'brown','#0C5DA5', '#FF9500', '#8E44AD', '#1E8449', '#D35400', '#3498DB']

    x_axis_column = data_df.columns[0]
    x_data = data_df[x_axis_column]

    for i, (col, label) in enumerate(zip(columns_to_plot, labels)):
        if col not in data_df.columns:
            print(f"⚠️ Warning: Column '{col}' not found. It will be skipped.")
            continue

        y_data = data_df[col]
        color = colors[i % len(colors)]

        # --- Smooth the data using spline interpolation ---
        if len(x_data) > 3:
            x_smooth = np.linspace(x_data.min(), x_data.max(), 300)
            spline_func = make_interp_spline(x_data, y_data, k=3)
            y_smooth = spline_func(x_smooth)
            ax.plot(x_smooth, y_smooth, label=label, color=color, linewidth=2.5)
        else:
            ax.plot(x_data, y_data, marker='o', linestyle='-', label=label, color=color, linewidth=2.5)

        # Overlay the original data points
        ax.scatter(x_data, y_data, color=color, s=50, zorder=5, edgecolor='white', linewidth=1)

    # --- Fine-tuning for Publication Quality (User-specified settings) ---
    ax.set_title("Combined NEB Energy Profiles", fontsize=18, fontweight='bold', pad=10)

    # -- Axis, legend, grid
    plt.xlabel("Reaction Coordinate", fontweight='bold', labelpad=9, fontsize=16)
    plt.ylabel("Energy (eV)", fontweight='bold', labelpad=9, fontsize=16)
    plt.grid(False)
    plt.legend(fontsize=16, loc='best', frameon=False)

    # -- Axis ticks (Nature style)
    ax.tick_params(which='both', direction='in', top=True, right=True, length=6, width=1.2, labelsize=14)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Use tight_layout first, then adjust
    plt.tight_layout()
    # plt.subplots_adjust(left=0.18, bottom=0.16, right=0.95, top=0.93)

    # --- Save plot in high-resolution formats ---
    output_pdf = "energy_profile_plot.pdf"
    output_png = "energy_profile_plot.png"
    plt.savefig(output_pdf, dpi=300)
    plt.savefig(output_png, dpi=300)

    print(f"✅ Plot successfully saved as '{output_pdf}' and '{output_png}'")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a publication-quality plot from CSV data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("csv_file", type=str, help="Path to the input CSV file.")
    parser.add_argument(
        "--columns",
        nargs='+',
        required=True,
        help="Space-separated list of column names to plot on the y-axis."
    )
    parser.add_argument(
        "--labels",
        nargs='+',
        required=True,
        help="Space-separated list of labels for the legend (must match the order of columns)."
    )

    args = parser.parse_args()

    if len(args.columns) != len(args.labels):
        print("❌ Error: The number of --columns must match the number of --labels.")
        sys.exit(1)

    generate_nature_quality_plot(args.csv_file, args.columns, args.labels)