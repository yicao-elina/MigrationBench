import os
import random
import argparse
from pathlib import Path
import shutil

def read_xyz_file(file_path):
    """Read XYZ file and return list of structures (each structure is a list of lines)"""
    structures = []
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist, skipping...")
        return structures
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if lines[i].strip() == '':
            i += 1
            continue
            
        try:
            # First line should be number of atoms
            num_atoms = int(lines[i].strip())
            
            # Get the complete structure (num_atoms + 2 lines: count + comment + atoms)
            structure = []
            structure.append(lines[i])  # Number of atoms
            
            if i + 1 < len(lines):
                structure.append(lines[i + 1])  # Comment line
            else:
                break
                
            # Add atom lines
            for j in range(num_atoms):
                if i + 2 + j < len(lines):
                    structure.append(lines[i + 2 + j])
                else:
                    break
            
            if len(structure) == num_atoms + 2:
                structures.append(structure)
            
            i += num_atoms + 2
            
        except ValueError:
            # Skip lines that don't start with a number
            i += 1
    
    return structures

def sample_structures(structures, ratio):
    """Sample structures based on the given ratio"""
    if ratio >= 1.0:
        return structures
    
    num_samples = max(1, int(len(structures) * ratio))
    return random.sample(structures, num_samples)

def write_xyz_structures(structures, output_path):
    """Write structures to XYZ file"""
    with open(output_path, 'w') as f:
        for structure in structures:
            for line in structure:
                f.write(line)

def merge_xyz_files(data_paths_and_ratios, output_folder):
    """
    Merge XYZ files from multiple folders with specified sampling ratios
    
    Args:
        data_paths_and_ratios: List of tuples (path, ratio)
        output_folder: Path to output folder
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    file_types = ['train.xyz', 'valid.xyz', 'test.xyz']
    
    for file_type in file_types:
        print(f"\nProcessing {file_type}...")
        all_structures = []
        
        for data_path, ratio in data_paths_and_ratios:
            file_path = os.path.join(data_path, file_type)
            print(f"  Reading from {file_path} with ratio {ratio}")
            
            structures = read_xyz_file(file_path)
            original_count = len(structures)
            
            if original_count == 0:
                print(f"    No structures found in {file_path}")
                continue
            
            sampled_structures = sample_structures(structures, ratio)
            sampled_count = len(sampled_structures)
            
            print(f"    Original: {original_count} structures, Sampled: {sampled_count} structures")
            all_structures.extend(sampled_structures)
        
        # Write merged file
        output_path = os.path.join(output_folder, file_type)
        write_xyz_structures(all_structures, output_path)
        print(f"  Merged {len(all_structures)} structures to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Merge XYZ files from multiple folders with sampling')
    parser.add_argument('--config', type=str, help='Path to config file with folder paths and ratios')
    parser.add_argument('--output', type=str, required=True, help='Output folder path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    if args.config:
        # Read from config file
        data_paths_and_ratios = []
        with open(args.config, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        path = parts[0]
                        ratio = float(parts[1])
                        data_paths_and_ratios.append((path, ratio))
    else:
        # Interactive input
        data_paths_and_ratios = []
        print("Enter data folder paths and sampling ratios (press Enter with empty path to finish):")
        
        while True:
            path = input("Data folder path: ").strip()
            if not path:
                break
            
            if not os.path.exists(path):
                print(f"Warning: Path {path} does not exist!")
                continue
            
            while True:
                try:
                    ratio = float(input(f"Sampling ratio for {path} (0.0-1.0): "))
                    if 0.0 <= ratio <= 1.0:
                        break
                    else:
                        print("Ratio must be between 0.0 and 1.0")
                except ValueError:
                    print("Please enter a valid number")
            
            data_paths_and_ratios.append((path, ratio))
    
    if not data_paths_and_ratios:
        print("No valid data paths provided!")
        return
    
    print(f"\nData paths and ratios:")
    for path, ratio in data_paths_and_ratios:
        print(f"  {path}: {ratio}")
    
    print(f"Output folder: {args.output}")
    
    # Perform the merge
    merge_xyz_files(data_paths_and_ratios, args.output)
    print(f"\nMerging completed! Results saved to {args.output}")

if __name__ == "__main__":
    main()