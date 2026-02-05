#!/bin/bash
# run_system_477frames.sh
# Complete execution script for 477 frames with buffer size 100

echo "================================================"
echo "Person ReID System - 477 Frames, Buffer 100"
echo "================================================"
echo "Start time: $(date)"
echo "================================================"

# Step 0: Check system
echo -e "\n[0/5] System Check..."
python3 -c "
import sys
print(f'Python: {sys.version}')
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('PyTorch not installed')
"

# Step 1: Install dependencies
echo -e "\n[1/5] Checking dependencies..."
if [ ! -f "weights/yolov8m.pt" ]; then
    echo "Missing model weights. Please run: ./install_all.sh"
    echo "Or download manually from ultralytics"
fi

# Step 2: Verify dataset
echo -e "\n[2/5] Verifying dataset..."
python3 -c "
import os
from pathlib import Path

data_dir = Path('/home/meghaagrawal940/annotated_data')
if not data_dir.exists():
    print('⚠️  Dataset directory not found!')
    print('Expected: /home/meghaagrawal940/annotated_data')
else:
    print(f'✓ Dataset directory found: {data_dir}')
"

# Step 3: Check for trained model (optional)
echo -e "\n[3/5] Checking for trained YOLOv8 model..."
if [ -f "runs/detect/yolov8_477frames/weights/best.pt" ]; then
    echo "✓ Using trained model"
else
    echo "⚠️  Trained model not found. Will use default YOLOv8m"
    echo "To train: python3 train_yolov8_477.py"
fi

# Step 4: Run Person ReID system
echo -e "\n[4/5] Running Person ReID System..."
echo "Configuration:"
echo "  - Buffer size: 100"
echo "  - Total frames: 477"
echo "  - Device: $(python3 -c 'import torch; print(\"GPU\" if torch.cuda.is_available() else \"CPU\")')"

python3 person_reid_full_477.py \
    --input "/home/meghaagrawal940/Input_file/2 staircase video.mp4" \
    --output "output_477frames_buffer100.mp4"

if [ $? -ne 0 ]; then
    echo "❌ ReID system failed!"
    exit 1
fi

# Step 5: Generate final report
echo -e "\n[5/5] Generating final report..."
python3 -c "
import json
import os

try:
    if os.path.exists('processing_statistics.json'):
        with open('processing_statistics.json', 'r') as f:
            stats = json.load(f)
        
        print('='*60)
        print('FINAL SYSTEM REPORT')
        print('='*60)
        print(f\"Total frames processed: {stats['video_processing']['total_frames']}\")
        print(f\"Processing time: {stats['video_processing']['processing_time']:.2f}s\")
        print(f\"Average FPS: {stats['video_processing']['average_fps']:.2f}\")
        print(f\"Global identities: {stats['video_processing']['global_identities']}\")
        print(f\"Buffer size: {stats['system_info']['buffer_size']}\")
        print('='*60)
    else:
        print('Statistics file not found.')
except Exception as e:
    print(f'Error reading statistics: {e}')
"

echo "================================================"
echo "Execution Complete!"
echo "End time: $(date)"
echo "================================================"
echo "Output Files:"
echo "  1. output_477frames_buffer100.mp4 - Processed video"
echo "  2. identity_memory.pkl - Global identity database"
echo "  3. processing_statistics.json - Detailed statistics"
echo "  4. system_summary.txt - System summary"
echo "================================================"
