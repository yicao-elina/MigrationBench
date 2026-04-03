#!/bin/bash
# =============================================
# Script: init_migrationbench.sh
# Purpose: Initialize MigrationBench project structure
# =============================================

set -e  # Exit on error

ROOT_DIR="MigrationBench"

# Helper function
create_dir() {
    mkdir -p "$1"
    echo "📁 Created directory: $1"
}

create_file() {
    if [ ! -f "$1" ]; then
        touch "$1"
        echo "📝 Created file: $1"
    fi
}

echo "🚀 Initializing MigrationBench directory structure..."
echo "------------------------------------------------------"

# ================================
# Core directory tree
# ================================
create_dir "$ROOT_DIR/migrationbench/core"
create_dir "$ROOT_DIR/migrationbench/workflows"
create_dir "$ROOT_DIR/migrationbench/models"
create_dir "$ROOT_DIR/migrationbench/analysis"
create_dir "$ROOT_DIR/migrationbench/visualization"
create_dir "$ROOT_DIR/migrationbench/io"
create_dir "$ROOT_DIR/migrationbench/utils"

# ================================
# Examples, templates, tests, docs
# ================================
create_dir "$ROOT_DIR/examples"
create_dir "$ROOT_DIR/templates/model_configs"
create_dir "$ROOT_DIR/templates/slurm_templates"
create_dir "$ROOT_DIR/tests/test_workflows"
create_dir "$ROOT_DIR/tests/test_models"
create_dir "$ROOT_DIR/tests/test_analysis"
create_dir "$ROOT_DIR/tests/fixtures"
create_dir "$ROOT_DIR/docs/source"
create_dir "$ROOT_DIR/docs/tutorials"
create_dir "$ROOT_DIR/docs/api"
create_dir "$ROOT_DIR/docs/examples"
create_dir "$ROOT_DIR/scripts"
create_dir "$ROOT_DIR/data/sample_structures"
create_dir "$ROOT_DIR/data/reference_results"
create_dir "$ROOT_DIR/data/test_data"

# ================================
# Python package initialization
# ================================
find "$ROOT_DIR/migrationbench" -type d -exec bash -c 'touch "$0/__init__.py"' {} \;

# ================================
# Core Python files
# ================================
for file in benchmark.py config.py exceptions.py; do
    create_file "$ROOT_DIR/migrationbench/core/$file"
done

for file in aimd.py neb.py mlff_training.py md_simulation.py base_workflow.py; do
    create_file "$ROOT_DIR/migrationbench/workflows/$file"
done

for file in base_model.py mace_model.py nequip_model.py schnet_model.py model_factory.py; do
    create_file "$ROOT_DIR/migrationbench/models/$file"
done

for file in energy_analysis.py trajectory_analysis.py latent_analysis.py convergence.py metrics.py; do
    create_file "$ROOT_DIR/migrationbench/analysis/$file"
done

for file in plots.py interactive.py reports.py; do
    create_file "$ROOT_DIR/migrationbench/visualization/$file"
done

for file in readers.py writers.py converters.py; do
    create_file "$ROOT_DIR/migrationbench/io/$file"
done

for file in file_utils.py math_utils.py validation.py logging_config.py; do
    create_file "$ROOT_DIR/migrationbench/utils/$file"
done

# ================================
# Example scripts
# ================================
for file in quick_start.py custom_model_integration.py batch_benchmarking.py advanced_analysis.py; do
    create_file "$ROOT_DIR/examples/$file"
done

# ================================
# Templates
# ================================
create_file "$ROOT_DIR/templates/benchmark_config.yaml"
create_file "$ROOT_DIR/templates/model_configs/mace_config.yaml"
create_file "$ROOT_DIR/templates/model_configs/nequip_config.yaml"
create_file "$ROOT_DIR/templates/model_configs/schnet_config.yaml"
create_file "$ROOT_DIR/templates/slurm_templates/aimd_job.slurm"
create_file "$ROOT_DIR/templates/slurm_templates/neb_job.slurm"
create_file "$ROOT_DIR/templates/slurm_templates/training_job.slurm"

# ================================
# Tests
# ================================
create_file "$ROOT_DIR/tests/__init__.py"
create_file "$ROOT_DIR/tests/test_benchmark.py"
create_file "$ROOT_DIR/tests/conftest.py"

# ================================
# Top-level project files
# ================================
for file in setup.py pyproject.toml requirements.txt environment.yml README.md CHANGELOG.md CONTRIBUTING.md LICENSE .gitignore; do
    create_file "$ROOT_DIR/$file"
done

echo "✅ MigrationBench structure successfully initialized!"
