
# NEB Data Processing for MLFF Training

This directory contains tools for collecting, organizing, and converting NEB (Nudged Elastic Band) calculation data into extended XYZ format suitable for machine learning force field (MLFF) training.

## Workflow Overview

```
Raw NEB Data → Collection → YAML Config → Extended XYZ → Merged Dataset
```

## Quick Start

```bash
# 1. Collect key files from NEB calculations
./collect_sb2te3.sh /path/to/neb/calculations

# 2. Generate YAML configuration
./generate_neb_yaml.sh 221_vdw_corr_DFT_D3/

# 3. Process NEB data to extended XYZ
python neb_data_collector.py neb_collection_config.yaml

# 4. Merge all XYZ files
cd mlff_training_data && find . -name "*.xyz" -exec cat {} \; > ../merged.xyz
```

## Scripts Description

### 1. `collect_sb2te3.sh`
Extracts essential files from NEB calculation directories.

**Usage:**
```bash
./collect_sb2te3.sh <source_neb_directory>
```

**Extracted files:**
- `neb.out` - NEB calculation output
- `sb2te3.xyz` - Atomic coordinates for all images
- `sb2te3.dat` - Migration energy data (if available)

### 2. `generate_neb_yaml.sh`
Automatically generates YAML configuration for data processing.

**Usage:**
```bash
./generate_neb_yaml.sh <base_directory> [output_config.yaml]
```

**Example:**
```bash
./generate_neb_yaml.sh 221_vdw_corr_DFT_D3/
# Creates: neb_collection_config.yaml
```

### 3. `neb_data_collector.py`
Main processing script that converts NEB data to extended XYZ format.

**Usage:**
```bash
# Test configuration
python neb_data_collector.py config.yaml --dry-run

# Process data
python neb_data_collector.py config.yaml
```

**Features:**
- Hierarchical data organization
- Rich metadata inclusion (energies, barriers, reaction coordinates)
- Automatic validation and error handling
- Comprehensive reporting

## Output Structure

```
mlff_training_data/
├── 221_vdw_corr_DFT_D3/
│   └── default/
│       ├── 1-2.xyz
│       ├── 1-3.xyz
│       └── ...
├── processing_report.json
├── processing_report.yaml
└── summary.txt
```

## Extended XYZ Format

Each structure includes comprehensive metadata:
```
61
Lattice="..." Properties=species:S:1:pos:R:3:forces:R:3 energy=-75762.367 neb_energy_eV=-133596.845 reaction_coordinate=0.0 migration_energy_eV=0.0 activation_energy_forward_eV=2.790 category="221_vdw_corr_DFT_D3" ...
Te    2.10710974    1.21654016    3.60958037    0.00000000    0.00000000    0.00000000
...
```

## Configuration File

The YAML configuration supports:
- Multiple categories and subcategories
- Custom metadata for each path
- Validation rules
- File pattern specifications

**Example structure:**
```yaml
categories:
  system_size:
    small_cell:
      description: "2x2x1 supercell systems"
      paths:
        - path: "calculations/path1"
          name: "system_path1"
          metadata:
            defect_type: "vacancy"
```

## Data Merging

After processing, merge all XYZ files:
```bash
cd mlff_training_data
find . -name "*.xyz" -exec cat {} \; > ../merged.xyz
```

## Troubleshooting

**Common issues:**
1. **Missing files**: Check that `neb.out` and `sb2te3.xyz` exist in each path
2. **Path errors**: Ensure relative paths in YAML are correct
3. **Permission errors**: Make scripts executable with `chmod +x script.sh`

**Validation:**
```bash
# Check configuration
python neb_data_collector.py config.yaml --dry-run

# Check output
head -20 merged.xyz
wc -l merged.xyz
```

## Next Steps

After generating `merged.xyz`, proceed to `../04_mlff_training/` for MLFF training preparation.

---
