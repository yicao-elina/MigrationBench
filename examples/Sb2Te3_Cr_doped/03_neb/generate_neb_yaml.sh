#!/bin/bash

# generate_neb_yaml.sh - Generate YAML config for NEB data collection

if [ $# -eq 0 ]; then
    echo "Usage: $0 <base_directory> [output_config.yaml]"
    echo "Example: $0 221_vdw_corr_DFT_D3/"
    exit 1
fi

BASE_DIR="$1"
OUTPUT_YAML="${2:-neb_collection_config.yaml}"

if [ ! -d "$BASE_DIR" ]; then
    echo "❌ Error: Directory '$BASE_DIR' does not exist"
    exit 1
fi

# Create YAML config
cat > "$OUTPUT_YAML" << EOF
# Auto-generated NEB collection config
output_base_directory: "mlff_training_data"

# Global settings
global_settings:
  include_forces: true
  file_patterns:
    neb_output: "neb.out"
    xyz_coords: "sb2te3.xyz"
    migration_data: "sb2te3.dat"

# Data organization structure
categories:
  $(basename "$BASE_DIR"):
    default:
      description: "Automatically collected NEB trajectories from $(basename "$BASE_DIR")"
      paths:
EOF

# Find all subdirectories with neb.out files
find "$BASE_DIR" -name "neb.out" -type f | while read -r neb_file; do
    # Get the directory containing the neb.out file
    neb_dir=$(dirname "$neb_file")
    
    # Get relative path from base directory
    rel_path=$(realpath --relative-to="$BASE_DIR" "$neb_dir")
    
    # Create a safe name (replace / with _)
    safe_name=$(echo "$rel_path" | sed 's/\//_/g')
    
    # Add to YAML
    cat >> "$OUTPUT_YAML" << EOF
        - path: "$BASE_DIR/$rel_path"
          name: "$safe_name"
          metadata: {}
EOF
done

# Add validation rules
cat >> "$OUTPUT_YAML" << EOF

# Data validation rules
validation:
  min_images_per_path: 3
  max_energy_per_atom: 10.0
  required_files: ["neb.out", "sb2te3.xyz"]
  optional_files: ["sb2te3.dat"]
EOF

echo "✅ YAML configuration written to: $OUTPUT_YAML"

# Show summary
echo ""
echo "📊 Summary:"
neb_count=$(find "$BASE_DIR" -name "neb.out" -type f | wc -l)
echo "   Found $neb_count NEB calculations"
echo "   Base directory: $BASE_DIR"
echo "   Output config: $OUTPUT_YAML"
echo ""
echo "🚀 Next steps:"
echo "   1. Review the config: cat $OUTPUT_YAML"
echo "   2. Test: python neb_data_collector.py $OUTPUT_YAML --dry-run"
echo "   3. Run: python neb_data_collector.py $OUTPUT_YAML"