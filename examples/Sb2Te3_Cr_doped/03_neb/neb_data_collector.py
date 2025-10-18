#!/usr/bin/env python3
"""
Advanced NEB Data Collection and Organization for MLFF Training
"""

import os
import yaml
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import re

def parse_neb_output(neb_file: str) -> Dict:
    """
    Parse NEB output file to extract energies and other information
    Enhanced to find the last COMPLETE iteration with all data
    """
    with open(neb_file, 'r') as f:
        content = f.read()
    
    # Find all iteration blocks
    iteration_blocks = re.findall(
        r'-+ iteration\s+(\d+)\s+-+(.*?)(?=-+ iteration|\Z)', 
        content, re.DOTALL
    )
    
    if not iteration_blocks:
        raise ValueError("No iterations found in NEB output file")
    
    print(f"Found {len(iteration_blocks)} total iterations")
    
    # Find the last COMPLETE iteration (has activation energies and energy table)
    last_complete_iteration = None
    last_complete_content = None
    
    # Search backwards through iterations to find the last complete one
    for iter_num, iter_content in reversed(iteration_blocks):
        # Check if this iteration has activation energies
        has_activation_forward = re.search(r'activation energy \(->\)\s*=\s*[\d.]+\s*eV', iter_content)
        has_activation_backward = re.search(r'activation energy \(<-\)\s*=\s*[\d.]+\s*eV', iter_content)
        has_energy_table = re.search(r'image\s+energy \(eV\)\s+error \(eV/A\)\s+frozen', iter_content)
        
        if has_activation_forward and has_activation_backward and has_energy_table:
            last_complete_iteration = int(iter_num)
            last_complete_content = iter_content
            print(f"Using last complete iteration: {last_complete_iteration}")
            break
    
    if last_complete_iteration is None:
        raise ValueError("No complete iteration found with activation energies and energy table")
    
    # Extract activation energies from the complete iteration
    act_energy_forward_match = re.search(
        r'activation energy \(->\)\s*=\s*([\d.]+)\s*eV', 
        last_complete_content
    )
    act_energy_backward_match = re.search(
        r'activation energy \(<-\)\s*=\s*([\d.]+)\s*eV', 
        last_complete_content
    )
    
    # Extract energy table - look for the table after activation energies
    energy_table_match = re.search(
        r'image\s+energy \(eV\)\s+error \(eV/A\)\s+frozen\s*\n(.*?)(?=\n\s*path length|\n\s*inter-image|\n\s*-+|\Z)',
        last_complete_content, re.DOTALL
    )
    
    energies = {}
    errors = {}
    
    if energy_table_match:
        energy_section = energy_table_match.group(1)
        print(f"Found energy table")
        
        # Parse each line: whitespace + image_num + energy + error + frozen_status
        energy_lines = re.findall(r'\s*(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+[FT]', energy_section)
        
        print(f"Found {len(energy_lines)} energy entries")
        
        for line in energy_lines:
            image_num = int(line[0])
            energy = float(line[1])
            error = float(line[2])
            energies[image_num] = energy
            errors[image_num] = error
        
        # Debug: print first few energies
        if energy_lines:
            print(f"Sample energies: {energy_lines[:3]}")
    
    # Extract path information
    path_match = re.search(r'path length\s*=\s*([\d.]+)\s*bohr', last_complete_content)
    inter_image_match = re.search(r'inter-image distance\s*=\s*([\d.]+)\s*bohr', last_complete_content)
    
    path_length = float(path_match.group(1)) if path_match else 0.0
    inter_image_distance = float(inter_image_match.group(1)) if inter_image_match else 0.0
    
    # If still no energies found, create dummy data
    if not energies:
        print(f"Warning: Could not parse energy table")
        # Count images from self-consistency lines
        image_matches = re.findall(r'self-consistency for image\s+(\d+)', last_complete_content)
        if image_matches:
            max_image = max(int(x) for x in image_matches)
            print(f"Creating dummy energies for {max_image} images")
            for i in range(1, max_image + 1):
                energies[i] = 0.0
                errors[i] = 0.0
    
    result = {
        'iteration': last_complete_iteration,
        'energies': energies,
        'errors': errors,
        'path_length': path_length,
        'inter_image_distance': inter_image_distance,
        'activation_energy_forward': float(act_energy_forward_match.group(1)) if act_energy_forward_match else 0.0,
        'activation_energy_backward': float(act_energy_backward_match.group(1)) if act_energy_backward_match else 0.0
    }
    
    # Debug output
    print(f"Final results:")
    print(f"  Iteration used: {result['iteration']}")
    print(f"  Activation forward: {result['activation_energy_forward']} eV")
    print(f"  Activation backward: {result['activation_energy_backward']} eV")
    print(f"  Path length: {result['path_length']} bohr")
    print(f"  Energies found: {len(result['energies'])}")
    if result['energies']:
        sample_energies = list(result['energies'].items())[:3]
        print(f"  Sample energies: {sample_energies}")
    
    return result

def parse_xyz_file(xyz_file: str) -> List[Tuple[int, List[Tuple[str, float, float, float]]]]:
    """Parse XYZ file containing multiple images"""
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
        
        i += 2  # Skip comment line
        
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

def parse_migration_energy_file(dat_file: str) -> Dict:
    """Parse migration energy data file"""
    reaction_coords = []
    migration_energies = []
    errors = []
    
    with open(dat_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 2:
                reaction_coord = float(parts[0])
                migration_energy = float(parts[1])
                error = float(parts[2]) if len(parts) > 2 else 0.0
                
                reaction_coords.append(reaction_coord)
                migration_energies.append(migration_energy)
                errors.append(error)
    
    if not reaction_coords:
        raise ValueError("No valid data found in migration energy file")
    
    return {
        'reaction_coords': np.array(reaction_coords),
        'migration_energies': np.array(migration_energies),
        'migration_errors': np.array(errors),
        'migration_barrier': max(migration_energies) - min(migration_energies) if migration_energies else 0.0
    }

def estimate_lattice_parameters(atoms: List[Tuple[str, float, float, float]]) -> str:
    """Estimate lattice parameters based on atomic positions"""
    coords = [(x, y, z) for _, x, y, z in atoms]
    
    x_coords = [x for x, y, z in coords]
    y_coords = [y for x, y, z in coords]
    z_coords = [z for x, y, z in coords]
    
    a = max(x_coords) - min(x_coords) + 2.0
    b = max(y_coords) - min(y_coords) + 2.0
    c = max(z_coords) - min(z_coords) + 2.0
    
    return f'"{a:.6f} 0.0 0.0 0.0 {b:.6f} 0.0 0.0 0.0 {c:.6f}"'

@dataclass
class PathMetadata:
    """Metadata for each NEB path"""
    original_path: str
    category: str
    subcategory: str
    name: str
    num_images: int
    num_atoms: int
    energy_range: Tuple[float, float]
    migration_barrier: Optional[float]
    activation_energy_forward: float
    activation_energy_backward: float
    has_migration_data: bool
    custom_metadata: Dict
    processing_timestamp: str
    validation_status: str

class NEBDataCollector:
    """Main class for collecting and organizing NEB data"""
    
    def __init__(self, config_file: str):
        self.setup_logging()  # Setup logging FIRST
        self.config = self.load_config(config_file)
        self.metadata_collection = []
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('neb_collection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_config(self, config_file: str) -> Dict:
        """Load YAML configuration file"""
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Handle flat structure (auto-convert to nested structure)
        self.convert_flat_to_nested_structure(config)
        return config
    
    def convert_flat_to_nested_structure(self, config: Dict):
        """Convert flat category structure to nested structure"""
        categories = config.get('categories', {})
        
        for category_name, category_data in categories.items():
            # Check if this is a flat structure (has 'paths' directly under category)
            if isinstance(category_data, dict) and 'paths' in category_data:
                # Convert to nested structure
                description = category_data.get('description', f"Auto-generated subcategory for {category_name}")
                paths = category_data.get('paths', [])
                
                # Create a default subcategory
                subcategory_name = 'default'
                
                # Replace the flat structure with nested structure
                categories[category_name] = {
                    subcategory_name: {
                        'description': description,
                        'paths': paths
                    }
                }
        
        self.logger.info("Converted flat category structure to nested structure")
    
    def validate_path(self, path_info: Dict, category: str, subcategory: str) -> bool:
        """Validate a single NEB path"""
        path = path_info['path']
        
        if not os.path.exists(path):
            self.logger.warning(f"Path does not exist: {path}")
            return False
        
        # Check required files
        required_files = self.config.get('validation', {}).get('required_files', [])
        for req_file in required_files:
            file_path = os.path.join(path, req_file)
            if not os.path.exists(file_path):
                self.logger.warning(f"Required file missing: {file_path}")
                return False
        
        return True
    
    def process_single_path(self, path_info: Dict, category: str, subcategory: str) -> Optional[PathMetadata]:
        """Process a single NEB calculation path"""
        path = path_info['path']
        name = path_info['name']
        custom_metadata = path_info.get('metadata', {})
        
        self.logger.info(f"Processing: {path}")
        
        try:
            # File paths
            neb_file = os.path.join(path, self.config['global_settings']['file_patterns']['neb_output'])
            xyz_file = os.path.join(path, self.config['global_settings']['file_patterns']['xyz_coords'])
            dat_file = os.path.join(path, self.config['global_settings']['file_patterns']['migration_data'])
            
            # Parse data
            neb_data = parse_neb_output(neb_file)
            xyz_data = parse_xyz_file(xyz_file)
            
            # Validate that we have some data
            if not neb_data['energies']:
                self.logger.warning(f"No energy data found for {path}")
            
            if not xyz_data:
                self.logger.warning(f"No XYZ data found for {path}")
                return None
            
            migration_data = None
            has_migration_data = False
            if os.path.exists(dat_file):
                try:
                    migration_data = parse_migration_energy_file(dat_file)
                    has_migration_data = True
                except Exception as e:
                    self.logger.warning(f"Could not parse migration data for {path}: {e}")
            
            # Create output directory structure
            output_base = Path(self.config['output_base_directory'])
            category_dir = output_base / category / subcategory
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            output_file = category_dir / f"{name}.xyz"
            
            # Write extended XYZ with enhanced metadata
            self.write_enhanced_xyz(
                str(output_file), neb_data, xyz_data, migration_data,
                category, subcategory, name, custom_metadata
            )
            
            # Calculate statistics
            energies = list(neb_data['energies'].values()) if neb_data['energies'] else [0.0]
            energy_range = (min(energies), max(energies))
            num_atoms = len(xyz_data[0][1]) if xyz_data else 0
            
            # Create metadata
            metadata = PathMetadata(
                original_path=path,
                category=category,
                subcategory=subcategory,
                name=name,
                num_images=len(xyz_data),
                num_atoms=num_atoms,
                energy_range=energy_range,
                migration_barrier=migration_data['migration_barrier'] if migration_data else None,
                activation_energy_forward=neb_data['activation_energy_forward'],
                activation_energy_backward=neb_data['activation_energy_backward'],
                has_migration_data=has_migration_data,
                custom_metadata=custom_metadata,
                processing_timestamp=datetime.now().isoformat(),
                validation_status="success"
            )
            
            self.logger.info(f"Successfully processed: {name} ({len(xyz_data)} images, {len(neb_data['energies'])} energies)")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error processing {path}: {e}")
            import traceback
            traceback.print_exc()
            return PathMetadata(
                original_path=path,
                category=category,
                subcategory=subcategory,
                name=name,
                num_images=0,
                num_atoms=0,
                energy_range=(0.0, 0.0),
                migration_barrier=None,
                activation_energy_forward=0.0,
                activation_energy_backward=0.0,
                has_migration_data=False,
                custom_metadata=custom_metadata,
                processing_timestamp=datetime.now().isoformat(),
                validation_status=f"failed: {str(e)}"
            )
    
    def write_enhanced_xyz(self, output_file: str, neb_data: Dict, xyz_data: List,
                        migration_data: Optional[Dict], category: str, subcategory: str,
                        name: str, custom_metadata: Dict):
        """Write enhanced XYZ with additional metadata for MLFF training"""
        
        total_images = len(xyz_data)
        lattice_params = self.config['global_settings'].get('lattice_params')
        
        with open(output_file, 'w') as f:
            for image_idx, (natoms, atoms) in enumerate(xyz_data):
                image_num = image_idx + 1
                
                # Get the total system energy directly from NEB output (in eV)
                system_energy_ev = neb_data['energies'].get(image_num, 0.0)
                neb_force_error = neb_data['errors'].get(image_num, 0.0)
                
                # Migration energy data
                migration_energy = 0.0
                migration_error = 0.0
                reaction_coord = 0.0
                
                if migration_data and len(migration_data['reaction_coords']) == total_images:
                    reaction_coord = migration_data['reaction_coords'][image_idx]
                    migration_energy = migration_data['migration_energies'][image_idx]
                    migration_error = migration_data.get('migration_errors', [0.0] * total_images)[image_idx]
                else:
                    reaction_coord = image_idx / (total_images - 1) if total_images > 1 else 0.0
                
                # Enhanced lattice parameters
                if lattice_params is None:
                    lattice = estimate_lattice_parameters(atoms)
                else:
                    lattice = f'"{lattice_params}"'
                
                # Generate placeholder forces (zeros for now)
                forces = [(0.0, 0.0, 0.0) for _ in range(natoms)]
                
                # Write header
                f.write(f"{natoms}\n")
                
                # Enhanced comment line with MLFF-relevant metadata
                properties = "Properties=species:S:1:pos:R:3:forces:R:3"
                
                comment_parts = [
                    f'Lattice={lattice}',
                    f'{properties}',
                    # Main energy field - total system energy in eV (what MLFF will use)
                    f'energy={system_energy_ev:.10f}',
                    # NEB-specific metadata
                    f'neb_force_error_eV_per_A={neb_force_error:.10f}',
                    f'reaction_coordinate={reaction_coord:.10f}',
                    f'migration_energy_eV={migration_energy:.10f}',
                    f'migration_error_eV={migration_error:.10f}',
                    f'activation_energy_forward_eV={neb_data["activation_energy_forward"]:.6f}',
                    f'activation_energy_backward_eV={neb_data["activation_energy_backward"]:.6f}',
                    # Metadata
                    f'neb_iteration={neb_data["iteration"]}',
                    f'neb_image={image_num}',
                    f'category="{category}"',
                    f'subcategory="{subcategory}"',
                    f'path_name="{name}"',
                    f'total_images={total_images}',
                    f'path_length_bohr={neb_data["path_length"]:.6f}',
                    f'inter_image_distance_bohr={neb_data["inter_image_distance"]:.6f}',
                ]
                
                # Add custom metadata
                for key, value in custom_metadata.items():
                    if isinstance(value, str):
                        comment_parts.append(f'{key}="{value}"')
                    else:
                        comment_parts.append(f'{key}={value}')
                
                comment_parts.append('pbc="T T T"')
                
                comment = ' '.join(comment_parts)
                f.write(comment + '\n')
                
                # Write atoms with forces
                for atom_idx, (element, x, y, z) in enumerate(atoms):
                    fx, fy, fz = forces[atom_idx]
                    line = f"{element:>2s} {x:15.8f} {y:15.8f} {z:15.8f} {fx:15.8f} {fy:15.8f} {fz:15.8f}"
                    f.write(line + '\n')

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        output_base = Path(self.config['output_base_directory'])
        
        # Statistics
        total_paths = len(self.metadata_collection)
        successful_paths = len([m for m in self.metadata_collection if m.validation_status == "success"])
        total_images = sum(m.num_images for m in self.metadata_collection if m.validation_status == "success")
        total_atoms = sum(m.num_atoms for m in self.metadata_collection if m.validation_status == "success")
        
        # Energy statistics
        all_energies = []
        for metadata in self.metadata_collection:
            if metadata.validation_status == "success" and metadata.energy_range[0] != metadata.energy_range[1]:
                all_energies.extend(metadata.energy_range)
        
        report = {
            'processing_summary': {
                'total_paths_attempted': total_paths,
                'successful_paths': successful_paths,
                'failed_paths': total_paths - successful_paths,
                'total_images': total_images,
                'total_atoms': total_atoms,
                'success_rate': f"{successful_paths/total_paths*100:.1f}%" if total_paths > 0 else "0%"
            },
            'energy_statistics': {
                'min_energy_eV': float(np.min(all_energies)) if all_energies else 0.0,
                'max_energy_eV': float(np.max(all_energies)) if all_energies else 0.0,
                'energy_range_eV': float(np.max(all_energies) - np.min(all_energies)) if all_energies else 0.0
            },
            'category_breakdown': {},
            'detailed_metadata': [asdict(m) for m in self.metadata_collection]
        }
        
        # Category breakdown
        for metadata in self.metadata_collection:
            cat = metadata.category
            subcat = metadata.subcategory
            if cat not in report['category_breakdown']:
                report['category_breakdown'][cat] = {}
            if subcat not in report['category_breakdown'][cat]:
                report['category_breakdown'][cat][subcat] = {
                    'paths': 0, 'images': 0, 'successful': 0
                }
            
            report['category_breakdown'][cat][subcat]['paths'] += 1
            if metadata.validation_status == "success":
                report['category_breakdown'][cat][subcat]['successful'] += 1
                report['category_breakdown'][cat][subcat]['images'] += metadata.num_images
        
        # Write reports
        with open(output_base / 'processing_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        with open(output_base / 'processing_report.yaml', 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        # Write simple summary
        with open(output_base / 'summary.txt', 'w') as f:
            f.write("NEB Data Collection Summary\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Total paths processed: {total_paths}\n")
            f.write(f"Successful: {successful_paths}\n")
            f.write(f"Failed: {total_paths - successful_paths}\n")
            f.write(f"Total images: {total_images}\n")
            f.write(f"Total atoms: {total_atoms}\n")
            if total_paths > 0:
                f.write(f"Success rate: {successful_paths/total_paths*100:.1f}%\n\n")
            
            if all_energies:
                f.write(f"Energy range: {np.min(all_energies):.3f} to {np.max(all_energies):.3f} eV\n")
                f.write(f"Energy span: {np.max(all_energies) - np.min(all_energies):.3f} eV\n\n")
            
            f.write("Category breakdown:\n")
            for cat, subcats in report['category_breakdown'].items():
                f.write(f"  {cat}:\n")
                for subcat, stats in subcats.items():
                    f.write(f"    {subcat}: {stats['successful']}/{stats['paths']} paths, {stats['images']} images\n")
        
        self.logger.info(f"Summary report written to {output_base}")
    
    def run(self):
        """Main execution method"""
        self.logger.info("Starting NEB data collection")
        
        # Process all categories
        for category, subcategories in self.config['categories'].items():
            self.logger.info(f"Processing category: {category}")
            
            for subcategory, subcat_data in subcategories.items():
                self.logger.info(f"  Processing subcategory: {subcategory}")
                
                # Handle both dict and string subcategory data
                if isinstance(subcat_data, dict):
                    paths = subcat_data.get('paths', [])
                else:
                    self.logger.warning(f"Unexpected subcategory data type for {subcategory}: {type(subcat_data)}")
                    continue
                
                for path_info in paths:
                    if self.validate_path(path_info, category, subcategory):
                        metadata = self.process_single_path(path_info, category, subcategory)
                        if metadata:
                            self.metadata_collection.append(metadata)
        
        # Generate summary report
        self.generate_summary_report()
        
        self.logger.info("NEB data collection completed")

def main():
    parser = argparse.ArgumentParser(description='Collect and organize NEB data for MLFF training')
    parser.add_argument('config', help='YAML configuration file')
    parser.add_argument('--dry-run', action='store_true', help='Validate configuration without processing')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found")
        return
    
    try:
        collector = NEBDataCollector(args.config)
        
        if args.dry_run:
            print("Configuration loaded successfully!")
            print(f"Found {len(collector.config['categories'])} categories")
            total_paths = sum(
                len(subcat_data.get('paths', []))
                for cat_data in collector.config['categories'].values()
                for subcat_data in cat_data.values()
                if isinstance(subcat_data, dict)
            )
            print(f"Total paths to process: {total_paths}")
            
            # Test parsing one file
            if args.debug:
                for category, subcategories in collector.config['categories'].items():
                    for subcategory, subcat_data in subcategories.items():
                        if isinstance(subcat_data, dict):
                            paths = subcat_data.get('paths', [])
                            if paths:
                                test_path = paths[0]['path']
                                neb_file = os.path.join(test_path, 'neb.out')
                                if os.path.exists(neb_file):
                                    print(f"\nTesting parsing of: {neb_file}")
                                    try:
                                        neb_data = parse_neb_output(neb_file)
                                        print(f"  Iteration: {neb_data['iteration']}")
                                        print(f"  Images found: {len(neb_data['energies'])}")
                                        print(f"  Activation energies: {neb_data['activation_energy_forward']:.3f} / {neb_data['activation_energy_backward']:.3f} eV")
                                        if neb_data['energies']:
                                            print(f"  Sample energ ies: {list(neb_data['energies'].items())[:3]}")
                                    except Exception as e:
                                        print(f"  Error: {e}")
                                        import traceback
                                        traceback.print_exc()
                                break
                        break
                    break
        else:
            collector.run()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()