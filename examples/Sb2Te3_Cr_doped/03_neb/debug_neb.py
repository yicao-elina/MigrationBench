#!/usr/bin/env python3
# test_fixed_parser.py

import re

def test_fixed_parser(neb_file):
    with open(neb_file, 'r') as f:
        content = f.read()
    
    # Find all iteration blocks
    iteration_blocks = re.findall(
        r'-+ iteration\s+(\d+)\s+-+(.*?)(?=-+ iteration|\Z)', 
        content, re.DOTALL
    )
    
    print(f"Found {len(iteration_blocks)} total iterations")
    
    # Find the last COMPLETE iteration
    for iter_num, iter_content in reversed(iteration_blocks):
        has_activation_forward = re.search(r'activation energy \(->\)\s*=\s*[\d.]+\s*eV', iter_content)
        has_activation_backward = re.search(r'activation energy \(<-\)\s*=\s*[\d.]+\s*eV', iter_content)
        has_energy_table = re.search(r'image\s+energy \(eV\)\s+error \(eV/A\)\s+frozen', iter_content)
        
        print(f"Iteration {iter_num}: forward={bool(has_activation_forward)}, backward={bool(has_activation_backward)}, table={bool(has_energy_table)}")
        
        if has_activation_forward and has_activation_backward and has_energy_table:
            print(f"\n=== USING ITERATION {iter_num} ===")
            
            # Extract activation energies
            forward_match = re.search(r'activation energy \(->\)\s*=\s*([\d.]+)\s*eV', iter_content)
            backward_match = re.search(r'activation energy \(<-\)\s*=\s*([\d.]+)\s*eV', iter_content)
            
            print(f"Activation forward: {forward_match.group(1) if forward_match else 'Not found'}")
            print(f"Activation backward: {backward_match.group(1) if backward_match else 'Not found'}")
            
            # Extract energy table
            energy_table = re.search(
                r'image\s+energy \(eV\)\s+error \(eV/A\)\s+frozen\s*\n(.*?)(?=\n\s*path length|\n\s*inter-image|\Z)',
                iter_content, re.DOTALL
            )
            
            if energy_table:
                energy_section = energy_table.group(1)
                energy_lines = re.findall(r'\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+[FT]', energy_section)
                print(f"Found {len(energy_lines)} energy entries:")
                for line in energy_lines:
                    print(f"  Image {line[0]}: {line[1]} eV, error {line[2]} eV/A")
            
            break

if __name__ == "__main__":
    test_fixed_parser("2D_neb/neb_1/neb.out")