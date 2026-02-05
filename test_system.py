#!/usr/bin/env python3
"""
Quick test to verify the complete system
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

def test_components():
    """Test all system components"""
    print("Testing system components...")
    
    tests = []
    
    # Test 1: PyTorch and CUDA
    try:
        print("\n1. Testing PyTorch...")
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        tests.append(("PyTorch", True))
    except ImportError as e:
        print(f"   ❌ Failed: {e}")
        tests.append(("PyTorch", False))
    
    # Test 2: Dataset
    try:
        print("\n2. Testing dataset...")
        data_dir = Path('/home/meghaagrawal940/annotated_data')
        if data_dir.exists():
            total = 0
            for split in ['train', 'val']:
                img_dir = data_dir / 'images' / split
                if img_dir.exists():
                    images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
                    total += len(images)
                    print(f"   {split}: {len(images)} images")
            
            print(f"   Total: {total} images")
            if total == 477:
                print("   ✅ 477 frames found")
                tests.append(("Dataset (477 frames)", True))
            elif total > 0:
                print(f"   ⚠️  {total} frames found (expected 477)")
                tests.append(("Dataset", True))
            else:
                print("   ❌ No images found")
                tests.append(("Dataset", False))
        else:
            print("   ❌ Dataset directory not found")
            tests.append(("Dataset", False))
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests.append(("Dataset", False))
    
    # Test 3: Model weights
    try:
        print("\n3. Testing model weights...")
        weights = {
            'YOLOv8': Path('weights/yolov8m.pt'),
            'Trained YOLO': Path('runs/detect/yolov8_477frames/weights/best.pt')
        }
        
        for name, path in weights.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"   ✅ {name}: {size_mb:.1f} MB")
                tests.append((f"{name} weights", True))
            else:
                print(f"   ⚠️  {name}: Not found")
                tests.append((f"{name} weights", False))
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests.append(("Model weights", False))
    
    # Test 4: Input video
    try:
        print("\n4. Testing input video...")
        video_path = '/home/meghaagrawal940/Input_file/2 staircase video.mp4'
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                cap.release()
                
                print(f"   ✅ Video found: {frames} frames, {width}x{height}, {fps} FPS")
                tests.append(("Input video", True))
            else:
                print("   ❌ Cannot open video")
                tests.append(("Input video", False))
        else:
            print(f"   ❌ Video not found: {video_path}")
            tests.append(("Input video", False))
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests.append(("Input video", False))
    
    # Test 5: Output directory
    try:
        print("\n5. Testing output directory...")
        os.makedirs('outputs', exist_ok=True)
        test_file = 'outputs/test_write.txt'
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print("   ✅ Output directory writable")
        tests.append(("Output directory", True))
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests.append(("Output directory", False))
    
    # Test 6: Required Python packages
    try:
        print("\n6. Testing required packages...")
        required_packages = [
            'torch',
            'cv2',
            'numpy',
            'PIL',
            'ultralytics',
            'scipy'
        ]
        
        missing = []
        for package in required_packages:
            try:
                if package == 'cv2':
                    import cv2
                elif package == 'PIL':
                    from PIL import Image
                else:
                    __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} - NOT INSTALLED")
                missing.append(package)
        
        if not missing:
            tests.append(("Required packages", True))
        else:
            print(f"   Missing packages: {', '.join(missing)}")
            tests.append(("Required packages", False))
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        tests.append(("Required packages", False))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    all_passed = True
    critical_failed = False
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            # Check if it's a critical failure
            if any(x in test_name.lower() for x in ['pytorch', 'packages', 'input video']):
                critical_failed = True
            all_passed = False
    
    print(f"{'='*60}")
    if all_passed:
        print("✅ ALL TESTS PASSED - System ready!")
        print("\nTo run the complete system:")
        print("  chmod +x run_system_477frames.sh")
        print("  ./run_system_477frames.sh")
        return True
    elif critical_failed:
        print("❌ CRITICAL TESTS FAILED")
        print("\nPlease install missing dependencies:")
        print("  pip install torch torchvision ultralytics opencv-python scipy pillow numpy")
        return False
    else:
        print("⚠️  SOME TESTS FAILED (non-critical)")
        print("\nSystem may still work, but some features might be limited.")
        print("To run the system:")
        print("  python3 person_reid_full_477.py")
        return True

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print("SYSTEM TEST - 477 Frames, Buffer 100")
    print(f"{'='*60}")
    
    success = test_components()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
