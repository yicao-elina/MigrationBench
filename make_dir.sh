#!/bin/bash

# Create the main project directory
mkdir -p mlff-benchmark-cr-sb2te3

# Navigate to the project directory
cd mlff-benchmark-cr-sb2te3

# Create main directories
mkdir -p data/{00_raw,01_processed,02_benchmark}
mkdir -p notebooks
mkdir -p scripts
mkdir -p src/{analysis,models,utils}
mkdir -p paper/figures
mkdir -p results/{models,logs,figures_raw}

# Create .gitkeep files for empty directories
touch data/00_raw/.gitkeep
touch data/01_processed/.gitkeep
touch data/02_benchmark/.gitkeep
touch paper/figures/.gitkeep

# Create __init__.py files for Python packages
touch src/__init__.py
touch src/analysis/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py

# Create main project files
touch README.md
touch LICENSE
touch environment.yml
touch requirements.txt

# Create notebook files
touch notebooks/00_data_exploration.ipynb
touch notebooks/01_model_analysis.ipynb
touch notebooks/02_figure_generation.ipynb

# Create script files
touch scripts/download_data.py
touch scripts/process_data.py
touch scripts/train.py
touch scripts/evaluate.py

# Create paper files
touch paper/manuscript.tex
touch paper/references.bib

# Create .gitignore file
cat > .gitignore << 'EOF'
# Results directory (too large for git)
results/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Data files (if too large)
*.xyz
*.vasp
OUTCAR*
POSCAR*
*.traj

# Model checkpoints (if too large)
*.pt
*.pth
*.model
*.pkl

# Logs
*.log
*.out
*.err

# Temporary files
*.tmp
*.temp
EOF

# Initialize git repository
git init

echo "Project structure created successfully!"
echo "Directory structure:"
tree -a