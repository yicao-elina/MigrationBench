from ase.io import read, write
import numpy as np
import os

def split_data(input_file, isolated_file, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    """
    Split an XYZ trajectory into training, validation, and test sets, including isolated atom configurations.

    Parameters:
    - input_file (str): The input .extxyz file.
    - isolated_file (str): The isolated atom .xyz file to be added to the training set.
    - train_ratio (float): Fraction of data for training.
    - valid_ratio (float): Fraction of data for validation.
    - test_ratio (float): Fraction of data for testing.
    """
    # Read all frames from the input file
    atoms_list = read(input_file, index=':')
    total_frames = len(atoms_list)
    print(f"Total frames from input: {total_frames}")

    # Shuffle indices for random sampling
    indices = np.arange(total_frames)
    np.random.shuffle(indices)

    # Calculate the number of frames for each set
    train_count = int(train_ratio * total_frames)
    valid_count = int(valid_ratio * total_frames)

    # Split indices
    train_indices = indices[:train_count]
    valid_indices = indices[train_count:train_count + valid_count]
    test_indices = indices[train_count + valid_count:]

    # Split data
    train_set = [atoms_list[i] for i in train_indices]
    valid_set = [atoms_list[i] for i in valid_indices]
    test_set = [atoms_list[i] for i in test_indices]

    # Read isolated atom data
    if os.path.exists(isolated_file):
        isolated_atoms = read(isolated_file, index=':')
        print(f"Adding {len(isolated_atoms)} isolated atom frames to the training set.")
        train_set.extend(isolated_atoms)
    else:
        print(f"Warning: Isolated atom file '{isolated_file}' not found. Proceeding without isolated data.")

    # Create output directories if they don't exist
    dat_dir = "data_neb"
    os.makedirs(dat_dir, exist_ok=True)

    # Write to separate files
    write(os.path.join(dat_dir, "train.xyz"), train_set, format="extxyz")
    write(os.path.join(dat_dir, "valid.xyz"), valid_set, format="extxyz")
    write(os.path.join(dat_dir, "test.xyz"), test_set, format="extxyz")

    print(f"Train set: {len(train_set)} frames")
    print(f"Validation set: {len(valid_set)} frames")
    print(f"Test set: {len(test_set)} frames")

if __name__ == "__main__":
    # Example usage: replace file names with actual file paths
    # /data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/2.aimd/3.Cr-Sb2Te3/1/650K-pbe/aimd_md.xyz
    # split_data("aimd_md.xyz", "isolated_atoms.xyz")
    # 2025-05-28
    split_data("merged_neb.xyz", "isolated_atoms_SG15.xyz")
