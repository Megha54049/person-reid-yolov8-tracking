#!/usr/bin/env python3
"""
Extract frames from video to create the complete dataset for YOLOv8 training.
"""

import cv2
import os
from pathlib import Path

def extract_frames_from_video(video_path, output_dir, num_frames=477):
    """Extract frames from video"""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Extracting: {num_frames} frames")
    
    # Extract frames
    frame_count = 0
    saved_count = 0
    
    while saved_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame
        frame_path = output_dir / f"frame_{frame_count:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        saved_count += 1
        
        if saved_count % 50 == 0:
            print(f"  Extracted {saved_count}/{num_frames} frames...")
        
        frame_count += 1
    
    cap.release()
    
    print(f"✅ Extracted {saved_count} frames to {output_dir}")
    return True

def main():
    # Paths
    video_path = "/home/meghaagrawal940/Input_file/2 staircase video.mp4"
    
    # Create dataset structure
    base_dir = Path("/home/meghaagrawal940/annotated_data/2_stairs_annotated")
    images_dir = base_dir / "data" / "images" / "train"
    
    # Extract frames
    print("Extracting frames from video...")
    extract_frames_from_video(video_path, images_dir, num_frames=477)
    
    # Verify
    image_files = list(images_dir.glob("*.png"))
    print(f"\n✅ Total images extracted: {len(image_files)}")
    
    # Check if labels exist
    labels_dir = base_dir / "labels" / "train"
    if labels_dir.exists():
        label_files = list(labels_dir.glob("*.txt"))
        print(f"✅ Total labels found: {len(label_files)}")
        
        if len(image_files) == len(label_files):
            print("✅ Images and labels match!")
        else:
            print(f"⚠️  Warning: Images ({len(image_files)}) and labels ({len(label_files)}) don't match")
    
    print("\n" + "="*60)
    print("Dataset ready for YOLOv8 training!")
    print("="*60)

if __name__ == "__main__":
    main()
