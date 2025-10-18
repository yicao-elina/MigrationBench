#!/usr/bin/env python3
"""
Script to convert NEB output and XYZ coordinates to extended XYZ format
"""

import re
import argparse
import os
from typing import List, Tuple, Dict

def parse_neb_output(neb_file: str) -> Dict:
    """
    Parse NEB output file to extract energies and other information
    """
    with open(neb_file, 'r') as f:
        content = f.read()
    
    # Find all iterations
    iterations = re.findall(r'-+ iteration (\d+) -+.*?image\s+energy \(eV\)\s+error \(eV/A\)\s+frozen\s+(.*?)(?=path length|$)', 
                           content, re.DOTALL)
    
    if not iterations:
        raise ValueError("No iterations found in NEB output file")
    
    # Get the last iteration
    last_iteration = iterations[-1]
    iteration_num = last_iteration[0]
    energy_section = last_iteration[1]
    
    # Parse energies and errors for each image
    energy_lines = re.findall(r'(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+[FT]', energy_section)
    
    energies = {}
    errors = {}
    
    for line in energy_lines:
        image_num = int(line[0])
        energy = float(line[1])
        error = float(line[2])
        energies[image_num] = energy
        errors[image_num] = error
    
    # Extract path information
    path_match = re.search(r'path length\s+=\s+([\d.]+)\s+bohr', content)
    inter_image_match = re.search(r'inter-image distance\s+=\s+([\d.]+)\s+bohr', content)
    
    path_length = float(path_match.group(1)) if path_match else 0.0
    inter_image_distance = float(inter_image_match.group(1)) if inter_image_match else 0.0
    
    # Extract activation energies
    act_energy_forward = re.search(r'activation energy \(->\)\s+=\s+([\d.]+)\s+eV', content)
    act_energy_backward = re.search(r'activation energy \(<-\)\s+=\s+([\d.]+)\s+eV', content)
    
    result = {
        'iteration': int(iteration_num),
        'energies': energies,
        'errors': errors,
        'path_length': path_length,
        'inter_image_distance': inter_image_distance,
        'activation_energy_forward': float(act_energy_forward.group(1)) if act_energy_forward else 0.0,
        'activation_energy_backward': float(act_energy_backward.group(1)) if act_energy_backward else 0.0
    }
    
    return result

def parse_xyz_file(xyz_file: str) -> List[Tuple[int, List[Tuple[str, float, float, float]]]]:
    """
    Parse XYZ file containing multiple images
    """
    with open(xyz_file, 'r') as f:
        lines = f.readlines()
    
    images = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        try:
            natoms = int(line)
        except ValueError:
            i += 1
            continue
        
        # Skip comment line
        i += 2
        
        atoms = []
        for j in range(natoms):
            if i + j < len(lines):
                parts = lines[i + j].split()
                if len(parts) >= 4:
                    element = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    atoms.append((element, x, y, z))
        
        if len(atoms) == natoms:
            images.append((natoms, atoms))
        
        i += natoms
    
    return images

def estimate_lattice_parameters(atoms: List[Tuple[str, float, float, float]]) -> str:
    """
    Estimate lattice parameters based on atomic positions
    This is a simple estimation - you may need to adjust based on your specific system
    """
    # Extract coordinates
    coords = [(x, y, z) for _, x, y, z in atoms]
    
    # Find approximate cell dimensions
    x_coords = [x for x, y, z in coords]
    y_coords = [y for x, y, z in coords]
    z_coords = [z for x, y, z in coords]
    
    # Simple estimation - you might want to use actual lattice parameters
    a = max(x_coords) - min(x_coords) + 2.0  # Add some buffer
    b = max(y_coords) - min(y_coords) + 2.0
    c = max(z_coords) - min(z_coords) + 2.0
    
    # For a simple orthorhombic cell
    return f'"{a:.6f} 0.0 0.0 0.0 {b:.6f} 0.0 0.0 0.0 {c:.6f}"'

def generate_fake_forces(natoms: int) -> List[Tuple[float, float, float]]:
    """
    Generate placeholder forces (all zeros)
    """
    return [(0.0, 0.0, 0.0) for _ in range(natoms)]

def write_extended_xyz(output_file: str, neb_data: Dict, xyz_data: List, 
                      lattice_params: str = None, include_forces: bool = True):
    """
    Write extended XYZ file with NEB data
    """
    with open(output_file, 'w') as f:
        for image_idx, (natoms, atoms) in enumerate(xyz_data):
            image_num = image_idx + 1
            
            # Get energy for this image
            energy_ev = neb_data['energies'].get(image_num, 0.0)
            error = neb_data['errors'].get(image_num, 0.0)
            
            # Convert eV to Hartree (approximately)
            energy_hartree = energy_ev * 0.036749322176  # eV to Hartree conversion
            
            # Estimate lattice if not provided
            if lattice_params is None:
                lattice = estimate_lattice_parameters(atoms)
            else:
                lattice = lattice_params
            
            # Generate forces (placeholder)
            forces = generate_fake_forces(natoms) if include_forces else None
            
            # Write header
            f.write(f"{natoms}\n")
            
            # Write extended XYZ comment line
            properties = "Properties=species:S:1:pos:R:3"
            if include_forces:
                properties += ":forces:R:3"
            
            comment = f'Lattice={lattice} {properties} '
            comment += f'energy={energy_hartree:.8f} '
            comment += f'neb_energy_eV={energy_ev:.6f} '
            comment += f'neb_error_eV_per_A={error:.6f} '
            comment += f'neb_iteration={neb_data["iteration"]} '
            comment += f'neb_image={image_num} '
            comment += 'pbc="T T T"'
            
            f.write(comment + '\n')
            
            # Write atoms
            for atom_idx, (element, x, y, z) in enumerate(atoms):
                line = f"{element:>2s} {x:15.8f} {y:15.8f} {z:15.8f}"
                
                if include_forces and forces:
                    fx, fy, fz = forces[atom_idx]
                    line += f" {fx:15.8f} {fy:15.8f} {fz:15.8f}"
                
                f.write(line + '\n')
            
            f.write('\n')  # Separate images

def main():
    parser = argparse.ArgumentParser(description='Convert NEB output and XYZ to extended XYZ format')
    parser.add_argument('--neb', default='neb.out', help='NEB output file (default: neb.out)')
    parser.add_argument('--xyz', default='sb2te3.xyz', help='XYZ coordinate file (default: sb2te3.xyz)')
    parser.add_argument('--output', default='extended_neb.xyz', help='Output extended XYZ file (default: extended_neb.xyz)')
    parser.add_argument('--lattice', help='Lattice parameters string (if not provided, will be estimated)')
    parser.add_argument('--no-forces', action='store_true', help='Do not include placeholder forces')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.neb):
        print(f"Error: NEB file '{args.neb}' not found")
        return
    
    if not os.path.exists(args.xyz):
        print(f"Error: XYZ file '{args.xyz}' not found")
        return
    
    try:
        # Parse NEB output
        print(f"Parsing NEB output from {args.neb}...")
        neb_data = parse_neb_output(args.neb)
        print(f"Found iteration {neb_data['iteration']} with {len(neb_data['energies'])} images")
        
        # Parse XYZ coordinates
        print(f"Parsing XYZ coordinates from {args.xyz}...")
        xyz_data = parse_xyz_file(args.xyz)
        print(f"Found {len(xyz_data)} coordinate sets")
        
        # Check consistency
        if len(xyz_data) != len(neb_data['energies']):
            print(f"Warning: Number of XYZ images ({len(xyz_data)}) != number of NEB energies ({len(neb_data['energies'])})")
        
        # Write extended XYZ
        print(f"Writing extended XYZ to {args.output}...")
        write_extended_xyz(args.output, neb_data, xyz_data, 
                          args.lattice, not args.no_forces)
        
        print("Conversion completed successfully!")
        
        # Print summary
        print("\nSummary:")
        print(f"  Iteration: {neb_data['iteration']}")
        print(f"  Images: {len(neb_data['energies'])}")
        print(f"  Activation energy (forward): {neb_data['activation_energy_forward']:.6f} eV")
        print(f"  Activation energy (backward): {neb_data['activation_energy_backward']:.6f} eV")
        print(f"  Path length: {neb_data['path_length']:.3f} bohr")
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()