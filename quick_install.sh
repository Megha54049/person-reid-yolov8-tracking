#!/bin/bash
# quick_install.sh - Quick installation of essential dependencies

echo "=========================================="
echo "Installing Essential Dependencies"
echo "=========================================="

# Install core packages
echo "Installing core Python packages..."
pip install --upgrade pip

# Install PyTorch (CPU version for faster install, can be upgraded later)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install essential packages
echo "Installing essential packages..."
pip install opencv-python
pip install ultralytics
pip install numpy
pip install scipy
pip install pillow
pip install pyyaml

echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo "Now run: python3 test_system.py"
