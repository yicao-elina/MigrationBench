# 🚀 Quick Start Guide

Get up and running with MigrationBench in 15 minutes!

## Prerequisites

- Linux/macOS system
- Python 3.8+
- Access to HPC cluster (recommended)

## Installation

### Option 1: Conda Environment (Recommended)
```bash
# Clone repository
git clone https://github.com/yicao-elina/MigrationBench.git
cd MigrationBench

# Create conda environment
conda env create -f environment.yml
conda activate migrationbench

# Install additional tools
pip install -e .
```

### Option 2: Docker (Easiest)
```bash
docker pull migrationbench/complete:latest
docker run -v $(pwd):/workspace migrationbench/complete:latest
```

## Software Dependencies

### Required Software
1. **Quantum Espresso** (for AIMD/NEB)
   ```bash
   # See INSTALLATION.md for detailed instructions
   ./scripts/install_quantum_espresso.sh
   ```

2. **MACE** (for MLFF training)
   ```bash
   pip install mace-torch
   ```

3. **ASE** (for MD simulations)
   ```bash
   pip install ase
   ```

## Quick Example

### 1. Run the Complete Tutorial
```bash
cd tutorials/
jupyter notebook 00_Complete_Pipeline.ipynb
```

### 2. Test with Example Data
```bash
cd examples/Sb2Te3_Cr_doped/

# Check the workflow
python ../../tools/validation/installation_checker.py

# Run analysis on provided example data
cd 05_Analysis/scripts/
python prediction_error.py
```

### 3. Adapt to Your System
```bash
# Copy example structure
cp examples/Sb2Te3_Cr_doped/ examples/my_system/

# Modify input files
# - Replace structure.xyz with your system
# - Adjust templates for your HPC environment
# - Update configuration files

# Follow the tutorial notebooks step by step
```

## Next Steps

1. 📚 Read the [User Guide](docs/user_guide/workflow_overview.md)
2. 🔧 Follow [Installation Guide](INSTALLATION.md) for detailed setup
3. 📓 Work through [Tutorial Notebooks](tutorials/)
4. 🎯 Adapt [Examples](examples/) to your research
5. 💬 Join discussions in [Issues](https://github.com/yicao-elina/MigrationBench/issues)

## Getting Help

- 📖 Check [Troubleshooting Guide](docs/user_guide/troubleshooting.md)
- 💬 Open an [Issue](https://github.com/yicao-elina/MigrationBench/issues)
- 📧 Contact: ycao73@jh.edu

---
**Next: [Installation Guide](INSTALLATION.md) | [Tutorials](tutorials/)**
```

### **INSTALLATION.md**

```markdown
# 🛠️ Installation Guide

Complete installation instructions for all MigrationBench components.

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 18.04+, CentOS 7+) or macOS 10.15+
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 32GB recommended
- **Storage**: 50GB free space
- **Python**: 3.8 or higher

### Recommended Requirements
- **HPC Access**: SLURM-based cluster
- **GPU**: NVIDIA GPU with CUDA 11.0+ (for MACE training)
- **MPI**: OpenMPI or Intel MPI (for QE)

## Installation Methods

### Method 1: Conda Environment (Recommended)

#### Step 1: Install Miniconda
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

#### Step 2: Create Environment
```bash
git clone https://github.com/yicao-elina/MigrationBench.git
cd MigrationBench
conda env create -f environment.yml
conda activate migrationbench
```

#### Step 3: Install MigrationBench
```bash
pip install -e .
```

### Method 2: Docker Installation

#### Step 1: Install Docker
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io
sudo systemctl start docker
sudo systemctl enable docker
```

#### Step 2: Pull MigrationBench Image
```bash
# Complete environment with all software
docker pull migrationbench/complete:latest

# Python-only environment
docker pull migrationbench/python-only:latest
```

#### Step 3: Run Container
```bash
docker run -it -v $(pwd):/workspace migrationbench/complete:latest
```

### Method 3: Manual Installation

#### Step 1: Python Environment
```bash
python -m venv migrationbench-env
source migrationbench-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 2: Install Software Dependencies
See individual software installation sections below.

## Software Dependencies

### 1. Quantum Espresso Installation

#### Option A: Automated Script
```bash
./scripts/install_quantum_espresso.sh
```

#### Option B: Manual Installation
```bash
# Download QE
wget https://github.com/QEF/q-e/releases/download/qe-7.2/qe-7.2-ReleasePack.tar.gz
tar -xzf qe-7.2-ReleasePack.tar.gz
cd qe-7.2

# Configure and compile
./configure --prefix=$HOME/software/qe
make all
make install

# Add to PATH
echo 'export PATH="$HOME/software/qe/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### Option C: HPC Module
```bash
# If available on your cluster
module load QuantumESPRESSO/7.2-foss-2022a
```

### 2. MACE Installation

#### CPU Version
```bash
pip install mace-torch
```

#### GPU Version
```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install mace-torch
```

#### From Source (Development)
```bash
git clone https://github.com/ACEsuit/mace.git
cd mace
pip install -e .
```

### 3. Additional MLFF Packages (Optional)

#### NequIP
```bash
pip install nequip
```

#### SchNetPack
```bash
pip install schnetpack
```

## Validation

### Test Installation
```bash
python tools/validation/installation_checker.py
```

### Run Example
```bash
cd examples/Sb2Te3_Cr_doped/05_Analysis/scripts/
python prediction_error.py
```

## HPC-Specific Setup

### SLURM Configuration
```bash
# Copy and modify SLURM templates
cp templates/quantum_espresso/slurm_templates/qe_aimd.slurm.template my_qe_aimd.slurm

# Edit for your cluster:
# - Partition names
# - Module names  
# - Resource limits
# - File paths
```

### Module Loading
```bash
# Create module loading script
cat > load_modules.sh << EOF
#!/bin/bash
module load QuantumESPRESSO/7.2-foss-2022a
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
EOF

chmod +x load_modules.sh
```

## Troubleshooting

### Common Issues

#### 1. QE Compilation Errors
```bash
# Install development tools
sudo apt-get install build-essential gfortran libopenmpi-dev

# Or on CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install openmpi-devel
```

#### 2. MACE GPU Issues
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Permission Issues
```bash
# Fix file permissions
chmod +x scripts/*.sh
chmod +x tools/**/*.py
```

### Getting Help

1. Check [Troubleshooting Guide](docs/user_guide/troubleshooting.md)
2. Search [Issues](https://github.com/yicao-elina/MigrationBench/issues)
3. Open new issue with:
   - Operating system
   - Python version
   - Error messages
   - Installation method used

---
**Next: [Quick Start](QUICK_START.md) | [Tutorials](tutorials/)**


## 🔧 **Essential Tool Scripts**

## 📊 **Example Data Structure**

### **examples/Sb2Te3_Cr_doped/01_AIMD/README.md**

