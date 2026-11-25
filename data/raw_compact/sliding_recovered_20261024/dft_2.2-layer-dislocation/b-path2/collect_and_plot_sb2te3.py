import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy.interpolate import make_interp_spline

# Constants
DATA_FILENAME = "sb2te3.dat"
CSV_OUTPUT = "combined_sb2te3_energy.csv"
PLOT_OUTPUT = "combined_sb2te3_energy_plot.pdf"
MIN_POINTS = 5  # Minimum points required to interpolate


def find_neb_files(root_dir, filename=DATA_FILENAME):
    """Find all sb2te3.dat files recursively from root_dir"""
    return sorted(glob(os.path.join(root_dir, f"**/{filename}"), recursive=True))


def read_reference_data(filepath):
    """Read the reference file to get reaction coordinates and energy"""
    data = np.loadtxt(filepath, usecols=(0, 1))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 0], data[:, 1]


def read_energy_column(filepath, expected_length):
    """Read only the energy (second) column and ensure length matches"""
    energy = np.loadtxt(filepath, usecols=(1,))
    if energy.ndim == 0:
        energy = np.array([energy])
    if len(energy) != expected_length:
        raise ValueError(f"Length mismatch: {filepath}")
    return energy


def collect_all_data(root_dir):
    """Collect all energy profiles, aligned by the reference reaction coordinates"""
    file_list = find_neb_files(root_dir)
    if not file_list:
        raise FileNotFoundError("No sb2te3.dat files found.")

    # Reference file sets the reaction coordinates
    ref_coords, ref_energy = read_reference_data(file_list[0])
    data = {'Reaction_Coord': ref_coords}
    labels = []

    # Add first entry
    label0 = os.path.relpath(os.path.dirname(file_list[0]), root_dir).replace(os.sep, '_')
    data[label0] = ref_energy
    labels.append(label0)

    # Process the rest
    for filepath in file_list[1:]:
        try:
            energy = read_energy_column(filepath, len(ref_coords))
            label = os.path.relpath(os.path.dirname(filepath), root_dir).replace(os.sep, '_')
            data[label] = energy
            labels.append(label)
        except Exception as e:
            print(f"Skipping {filepath}: {e}")

    df = pd.DataFrame(data)
    return df, labels

def set_plot_settings():
    """
    Set custom plot settings for matplotlib.
    """
    plt.rc('font', size=16)  # Default text sizes
    plt.rc('axes', titlesize=16)  # Axes title font size
    plt.rc('legend', fontsize=12)  # Legend font size
    plt.rc('xtick', labelsize=18)  # X-axis tick label font size
    plt.rc('ytick', labelsize=18)  # Y-axis tick label font size

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams["savefig.format"] = 'pdf'

def plot_smooth_profiles(df, labels, output_file):
    """Plot smoothed energy profiles"""
    x = df["Reaction_Coord"].values
    set_plot_settings()
    plt.figure(figsize=(6, 4))

    for label in labels:
        y = df[label].values
        if len(x) >= MIN_POINTS:
            # Spline interpolation
            x_smooth = np.linspace(x.min(), x.max(), 300)
            spline = make_interp_spline(x, y, k=3)
            y_smooth = spline(x_smooth)
            plt.plot(x_smooth, y_smooth, label=label)
            plt.plot(x, y, 'o', color='gray', markersize=4)
        else:
            # Not enough points for smoothing
            plt.plot(x, y, 'o-', label=label)

    plt.xlabel("Reaction Coordinate")
    plt.ylabel("Energy (eV)")
    plt.title("Combined NEB Energy Profiles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


def main():
    root_dir = os.getcwd()
    try:
        df, labels = collect_all_data(root_dir)
        df.to_csv(CSV_OUTPUT, index=False)
        print(f"✅ CSV saved: {CSV_OUTPUT}")
        plot_smooth_profiles(df, labels, PLOT_OUTPUT)
        print(f"✅ Plot saved: {PLOT_OUTPUT}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
