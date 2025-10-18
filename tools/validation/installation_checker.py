#!/usr/bin/env python3
"""
Installation validation script for MigrationBench.

Checks all software dependencies and provides diagnostic information.
"""

import sys
import subprocess
import shutil
from pathlib import Path
import importlib

class InstallationChecker:
    def __init__(self):
        self.results = {}
        self.errors = []
        
    def check_python_packages(self):
        """Check Python package dependencies."""
        required_packages = [
            'numpy', 'scipy', 'matplotlib', 'pandas',
            'ase', 'pymatgen', 'sklearn', 'yaml'
        ]
        
        optional_packages = {
            'mace': 'MACE force field training',
            'nequip': 'NequIP force field training', 
            'torch': 'PyTorch for ML models',
            'nglview': '3D structure visualization',
            'plotly': 'Interactive plotting'
        }
        
        print("🐍 Checking Python packages...")
        
        # Check required packages
        for package in required_packages:
            try:
                mod = importlib.import_module(package)
                version = getattr(mod, '__version__', 'unknown')
                self.results[package] = {'status': 'OK', 'version': version}
                print(f"  ✅ {package}: {version}")
            except ImportError:
                self.results[package] = {'status': 'MISSING', 'version': None}
                self.errors.append(f"Missing required package: {package}")
                print(f"  ❌ {package}: NOT FOUND")
        
        # Check optional packages
        print("\n🔧 Checking optional packages...")
        for package, description in optional_packages.items():
            try:
                mod = importlib.import_module(package)
                version = getattr(mod, '__version__', 'unknown')
                print(f"  ✅ {package}: {version} ({description})")
            except ImportError:
                print(f"  ⚠️  {package}: NOT FOUND ({description})")
    
    def check_external_software(self):
        """Check external software installations."""
        software_list = {
            'pw.x': 'Quantum Espresso PWscf',
            'neb.x': 'Quantum Espresso NEB',
            'pp.x': 'Quantum Espresso PostProc',
            'mace_run_train': 'MACE training executable'
        }
        
        print("\n🔬 Checking external software...")
        
        for executable, description in software_list.items():
            path = shutil.which(executable)
            if path:
                # Try to get version
                try:
                    if 'pw.x' in executable:
                        result = subprocess.run([executable, '-version'], 
                                              capture_output=True, text=True, timeout=10)
                        version_info = result.stdout.split('\n')[0] if result.stdout else 'unknown'
                    else:
                        version_info = 'found'
                    
                    print(f"  ✅ {executable}: {path} ({ version_info})")
                    self.results[executable] = {'status': 'OK', 'path': path}
                except:
                    print(f"  ⚠️  {executable}: {path} (version check failed)")
                    self.results[executable] = {'status': 'PARTIAL', 'path': path}
            else:
                print(f"  ❌ {executable}: NOT FOUND ({description})")
                self.results[executable] = {'status': 'MISSING', 'path': None}
    
    def check_gpu_support(self):
        """Check GPU and CUDA support."""
        print("\n🖥️ Checking GPU support...")
        
        # Check NVIDIA GPU
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.split('\n')[8:10]  # GPU info lines
                print("  ✅ NVIDIA GPU detected:")
                for line in gpu_info:
                    if 'GeForce' in line or 'Tesla' in line or 'RTX' in line:
                        print(f"    {line.strip()}")
            else:
                print("  ❌ nvidia-smi not found or failed")
        except FileNotFoundError:
            print("  ❌ NVIDIA drivers not installed")
        
        # Check PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  ✅ PyTorch CUDA: {torch.version.cuda}")
                print(f"    Available GPUs: {torch.cuda.device_count()}")
            else:
                print("  ⚠️  PyTorch installed but CUDA not available")
        except ImportError:
            print("  ❌ PyTorch not installed")
    
    def check_file_structure(self):
        """Check repository file structure."""
        print("\n📁 Checking file structure...")
        
        required_dirs = [
            'examples', 'tutorials', 'templates', 'tools', 'docs'
        ]
        
        for dir_name in required_dirs:
            if Path(dir_name).exists():
                print(f"  ✅ {dir_name}/ directory found")
            else:
                print(f"  ❌ {dir_name}/ directory missing")
                self.errors.append(f"Missing directory: {dir_name}")
    
    def generate_report(self):
        """Generate installation report."""
        print("\n" + "="*60)
        print("📊 INSTALLATION REPORT")
        print("="*60)
        
        if not self.errors:
            print("🎉 All checks passed! MigrationBench is ready to use.")
        else:
            print("⚠️  Issues found:")
            for error in self.errors:
                print(f"  - {error}")
        
        print("\n💡 Next steps:")
        if not self.errors:
            print("  1. Try running: cd tutorials && jupyter notebook")
            print("  2. Or test with: cd examples/Sb2Te3_Cr_doped/05_Analysis/scripts/")
        else:
            print("  1. Fix the issues listed above")
            print("  2. See INSTALLATION.md for detailed instructions")
            print("  3. Run this checker again")
        
        print(f"\n📧 Need help? Open an issue at:")
        print(f"   https://github.com/yicao-elina/MigrationBench/issues")

def main():
    print("🔍 MigrationBench Installation Checker")
    print("="*50)
    
    checker = InstallationChecker()
    checker.check_python_packages()
    checker.check_external_software()
    checker.check_gpu_support()
    checker.check_file_structure()
    checker.generate_report()

if __name__ == "__main__":
    main()