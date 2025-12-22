#!/bin/bash

# Fix Environment Script for Orochi
# This script installs missing pip dependencies that aren't installed by conda

echo "========================================================================"
echo "  Orochi Environment Fix Script"
echo "========================================================================"
echo ""
echo "This script will install missing pip packages from temp/requirements.txt"
echo ""

# Check if we're in a conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠ Warning: No conda environment is active!"
    echo "  Please activate the mamba_biomed environment first:"
    echo "  conda activate mamba_biomed"
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to cancel..."
fi

# Check if we're in the right directory
if [ ! -f "./temp/requirements.txt" ]; then
    echo "✗ Error: temp/requirements.txt not found!"
    echo "  Please run this script from the repository root directory."
    exit 1
fi

echo "Installing pip dependencies from temp/requirements.txt..."
echo ""

# Install lpips specifically (the missing package)
echo "Installing lpips (missing package)..."
pip install lpips==0.1.4

# Install other potentially missing packages
echo ""
echo "Installing other key packages..."
pip install mamba-ssm==2.2.2 --no-deps
pip install causal-conv1d==1.4.0

# Install all requirements to ensure nothing is missing
echo ""
echo "Installing all requirements from temp/requirements.txt..."
pip install -r temp/requirements.txt

echo ""
echo "========================================================================"
echo "  Installation Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Run the test again: python quick_test.py"
echo "  2. If mamba_ssm still fails, run: python setup_environment.py"
echo ""
