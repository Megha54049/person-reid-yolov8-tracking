#!/bin/bash
# install_all.sh - Complete dependency installation

echo "==========================================="
echo "Installing Person ReID System Dependencies"
echo "==========================================="

# Update system
sudo apt-get update
sudo apt-get install -y ffmpeg libsm6 libxext6 git wget unzip

# Install Python packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics==8.0.196
pip install opencv-python==4.8.1.78
pip install Pillow==10.0.0
pip install numpy==1.24.3
pip install scipy==1.11.4
pip install scikit-learn==1.3.0
pip install tqdm==4.66.1
pip install pyyaml==6.0.1
pip install pandas==2.1.3
pip install seaborn==0.12.2

# Install FAISS
pip install faiss-gpu==1.7.4

# Install BoT-SORT from source
echo "Installing BoT-SORT..."
git clone https://github.com/NirAharon/BoT-SORT.git
cd BoT-SORT
pip install -r requirements.txt
pip install -e .
cd ..

# Create directory structure
mkdir -p weights
mkdir -p logs
mkdir -p outputs
mkdir -p data/annotations

echo "==========================================="
echo "Downloading Pre-trained Models"
echo "==========================================="

# Download YOLOv8m weights
wget -P weights/ https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt

# Download TransReID weights
echo "Downloading TransReID weights..."
wget -P weights/ https://github.com/damo-cv/TransReID/releases/download/v1.0/transreid_market1501_msmt17.pth

echo "==========================================="
echo "Setting up Data Structure"
echo "==========================================="

# Check dataset
if [ -d "/home/meghaagrawal940/annotated_data" ]; then
    echo "Dataset found at /home/meghaagrawal940/annotated_data"
    echo "Total frames to process: 477"
else
    echo "Warning: Dataset directory not found"
    echo "Please ensure data is at: /home/meghaagrawal940/annotated_data"
fi

# Create symlink for easy access
ln -sf /home/meghaagrawal940/annotated_data ./data/annotated

echo "==========================================="
echo "Installation Complete!"
echo "==========================================="
echo "Next steps:"
echo "1. Run: chmod +x run_system_477frames.sh"
echo "2. Run: ./run_system_477frames.sh"
echo "==========================================="
