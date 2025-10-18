#!/usr/bin/env python3
"""
Split XYZ trajectory data into training, validation, and test sets for MLFF training.
Includes isolated atom configurations in the training set.
"""

from ase.io import read, write
import numpy as np
import os
import argparse
import sys
from pathlib import Path

def split_data(input_file, isolated_file, output_dir, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split an XYZ trajectory into training, validation, and test sets, including isolated atom configurations.

    Parameters:
    - input_file (str): The input .extxyz file.
    - isolated_file (str): The isolated atom .xyz file to be added to the training set.
    - output_dir (str): Directory to save the split datasets.
    - train_ratio (float): Fraction of data for training.
    - valid_ratio (float): Fraction of data for validation.
    - test_ratio (float): Fraction of data for testing.
    - seed (int): Random seed for reproducibility.
    """
    
    # Validate ratios
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + valid_ratio + test_ratio}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found")
    
    print(f"Reading trajectory data from: {input_file}")
    
    # Read all frames from the input file
    try:
        atoms_list = read(input_file, index=':')
        total_frames = len(atoms_list)
        print(f"Total frames from input: {total_frames}")
    except Exception as e:
        raise RuntimeError(f"Error reading input file '{input_file}': {e}")

    if total_frames == 0:
        raise ValueError("No frames found in input file")

    # Shuffle indices for random sampling
    indices = np.arange(total_frames)
    np.random.shuffle(indices)

    # Calculate the number of frames for each set
    train_count = int(train_ratio * total_frames)
    valid_count = int(valid_ratio * total_frames)
    test_count = total_frames - train_count - valid_count  # Ensure all frames are used

    print(f"Splitting data: {train_count} train, {valid_count} valid, {test_count} test")

    # Split indices
    train_indices = indices[:train_count]
    valid_indices = indices[train_count:train_count + valid_count]
    test_indices = indices[train_count + valid_count:]

    # Split data
    train_set = [atoms_list[i] for i in train_indices]
    valid_set = [atoms_list[i] for i in valid_indices]
    test_set = [atoms_list[i] for i in test_indices]

    # Read isolated atom data
    if isolated_file and os.path.exists(isolated_file):
        try:
            isolated_atoms = read(isolated_file, index=':')
            print(f"Adding {len(isolated_atoms)} isolated atom frames to the training set.")
            train_set.extend(isolated_atoms)
        except Exception as e:
            print(f"Warning: Error reading isolated atom file '{isolated_file}': {e}")
    elif isolated_file:
        print(f"Warning: Isolated atom file '{isolated_file}' not found. Proceeding without isolated data.")
    else:
        print("No isolated atom file specified.")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path.absolute()}")

    # Write to separate files
    train_file = output_path / "train.xyz"
    valid_file = output_path / "valid.xyz"
    test_file = output_path / "test.xyz"
    
    try:
        write(str(train_file), train_set, format="extxyz")
        write(str(valid_file), valid_set, format="extxyz")
        write(str(test_file), test_set, format="extxyz")
    except Exception as e:
        raise RuntimeError(f"Error writing output files: {e}")

    # Print summary
    print("\n" + "="*50)
    print("DATA SPLITTING COMPLETED")
    print("="*50)
    print(f"Train set: {len(train_set)} frames -> {train_file}")
    print(f"Validation set: {len(valid_set)} frames -> {valid_file}")
    print(f"Test set: {len(test_set)} frames -> {test_file}")
    print(f"Total frames processed: {len(train_set) + len(valid_set) + len(test_set)}")
    
    # Calculate actual ratios
    total_processed = len(train_set) + len(valid_set) + len(test_set)
    actual_train_ratio = len(train_set) / total_processed
    actual_valid_ratio = len(valid_set) / total_processed
    actual_test_ratio = len(test_set) / total_processed
    
    print(f"\nActual ratios:")
    print(f"  Train: {actual_train_ratio:.3f}")
    print(f"  Valid: {actual_valid_ratio:.3f}")
    print(f"  Test:  {actual_test_ratio:.3f}")

def main():
    parser = argparse.ArgumentParser(
        description='Split XYZ trajectory data into train/validation/test sets for MLFF training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default ratios
  python split_data.py merged_data.xyz
  
  # Specify all parameters
  python split_data.py merged_data.xyz --isolated isolated_atoms.xyz --output data_split --train 0.7 --valid 0.2 --test 0.1
  
  # Use different output directory
  python split_data.py trajectory.xyz -o /path/to/output/dir
  
  # Set random seed for reproducibility
  python split_data.py data.xyz --seed 123
        """
    )
    
    # Required arguments
    parser.add_argument('input_xyz', 
                       help='Input XYZ trajectory file (merged NEB data)')
    
    # Optional arguments
    parser.add_argument('--isolated', '-i', 
                       default='isolated_atoms.xyz',
                       help='Isolated atom XYZ file (default: isolated_atoms.xyz)')
    
    parser.add_argument('--output', '-o', 
                       default='data_split',
                       help='Output directory for split datasets (default: data_split)')
    
    parser.add_argument('--train', 
                       type=float, default=0.8,
                       help='Training set ratio (default: 0.8)')
    
    parser.add_argument('--valid', 
                       type=float, default=0.1,
                       help='Validation set ratio (default: 0.1)')
    
    parser.add_argument('--test', 
                       type=float, default=0.1,
                       help='Test set ratio (default: 0.1)')
    
    parser.add_argument('--seed', 
                       type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    parser.add_argument('--no-isolated', 
                       action='store_true',
                       help='Skip isolated atom data (ignore --isolated)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.train <= 0 or args.valid <= 0 or args.test <= 0:
        print("Error: All ratios must be positive")
        sys.exit(1)
    
    if abs(args.train + args.valid + args.test - 1.0) > 1e-6:
        print(f"Error: Ratios must sum to 1.0, got {args.train + args.valid + args.test}")
        sys.exit(1)
    
    # Handle isolated atom file
    isolated_file = None if args.no_isolated else args.isolated
    
    try:
        split_data(
            input_file=args.input_xyz,
            isolated_file=isolated_file,
            output_dir=args.output,
            train_ratio=args.train,
            valid_ratio=args.valid,
            test_ratio=args.test,
            seed=args.seed
        )
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()