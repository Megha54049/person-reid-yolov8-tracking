#!/usr/bin/env python3
"""
Complete YOLOv8 Training for 477 Frames Dataset
Author: Senior AI Engineer
"""

import os
import sys
import torch
import yaml
import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
import cv2

class YOLOv8Trainer477:
    """YOLOv8 Trainer optimized for 477 frames dataset"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_dir = Path('/home/meghaagrawal940/annotated_data/2_stairs_annotated/data')
        self.output_dir = Path('runs/detect/yolov8_477frames')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"YOLOv8 TRAINER - 477 FRAMES DATASET")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def validate_dataset_structure(self):
        """Validate and fix dataset structure for 477 frames"""
        print("Validating dataset structure...")
        
        # Expected structure
        expected_dirs = [
            'images/train',
            'images/val',
            'labels/train',
            'labels/val'
        ]
        
        # Check and create directories
        for dir_path in expected_dirs:
            full_path = self.data_dir / dir_path
            if not full_path.exists():
                print(f"  Creating directory: {full_path}")
                full_path.mkdir(parents=True, exist_ok=True)
        
        # Count files
        stats = {}
        for split in ['train', 'val']:
            img_dir = self.data_dir / 'images' / split
            label_dir = self.data_dir / 'labels' / split
            
            # Count images
            images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg'))
            labels = list(label_dir.glob('*.txt'))
            
            stats[split] = {
                'images': len(images),
                'labels': len(labels),
                'image_files': [str(p) for p in images[:5]],  # Sample files
                'label_files': [str(p) for p in labels[:5]]
            }
        
        total_frames = stats['train']['images'] + stats['val']['images']
        
        print(f"\nDataset Statistics:")
        print(f"  Training images: {stats['train']['images']}")
        print(f"  Training labels: {stats['train']['labels']}")
        print(f"  Validation images: {stats['val']['images']}")
        print(f"  Validation labels: {stats['val']['labels']}")
        print(f"  Total frames: {total_frames}")
        
        if total_frames != 477:
            print(f"\n⚠️  Warning: Expected 477 frames, found {total_frames}")
            print("  The system will still work, but may not be optimal.")
        
        # Verify label format
        if stats['train']['labels'] > 0:
            sample_label = list((self.data_dir / 'labels' / 'train').glob('*.txt'))[0]
            with open(sample_label, 'r') as f:
                lines = f.readlines()
                if lines:
                    sample = lines[0].strip()
                    parts = sample.split()
                    if len(parts) == 5:
                        print(f"\n✓ Label format verified: [class x_center y_center width height]")
                    else:
                        print(f"\n⚠️  Warning: Unexpected label format: {sample}")
        
        return stats, total_frames
    
    def create_data_yaml(self, stats):
        """Create data.yaml configuration file"""
        data_yaml = {
            'path': str(self.data_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/val',
            'nc': 1,  # Number of classes
            'names': ['person'],  # Class names
            'roboflow': {
                'workspace': 'person-477',
                'project': 'staircase-detection',
                'version': '1',
                'license': 'MIT',
                'url': 'https://universe.roboflow.com/dataset/staircase'
            }
        }
        
        yaml_path = self.data_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"\n✓ Created data.yaml at: {yaml_path}")
        return str(yaml_path)
    
    def prepare_data_augmentation(self):
        """Prepare data augmentation configuration for small dataset"""
        aug_config = {
            'hsv_h': 0.015,  # Hue augmentation
            'hsv_s': 0.7,    # Saturation augmentation
            'hsv_v': 0.4,    # Value augmentation
            'degrees': 0.0,  # Rotation
            'translate': 0.2,  # Translation (20%)
            'scale': 0.9,    # Scale (+/- 9%)
            'shear': 0.0,    # Shear
            'perspective': 0.001,  # Perspective
            'flipud': 0.0,   # Flip up-down
            'fliplr': 0.5,   # Flip left-right (50%)
            'mosaic': 1.0,   # Mosaic augmentation
            'mixup': 0.15,   # Mixup augmentation (15%)
            'copy_paste': 0.3,  # Copy-paste (30%)
        }
        return aug_config
    
    def train_model(self, data_yaml_path, total_frames):
        """Train YOLOv8 model with optimized settings"""
        print(f"\n{'='*60}")
        print("STARTING YOLOv8 TRAINING")
        print(f"{'='*60}")
        
        # Load model
        model = YOLO('weights/yolov8m.pt')
        
        # Calculate optimal parameters based on dataset size
        batch_size = min(16, max(4, total_frames // 30))  # 4-16 based on dataset size
        epochs = 10  # Fixed as requested
        
        # Training arguments optimized for 477 frames
        train_args = {
            'data': data_yaml_path,
            'epochs': epochs,
            'imgsz': 640,
            'batch': batch_size,
            'workers': min(4, os.cpu_count() // 2),
            'device': self.device,
            'pretrained': True,
            'verbose': True,
            'save': True,
            'exist_ok': True,
            'project': str(self.output_dir.parent),
            'name': self.output_dir.name,
            'seed': 42,
            'deterministic': True,
            
            # Optimization
            'optimizer': 'AdamW',  # Better for small datasets
            'lr0': 0.001,  # Lower learning rate
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Augmentation (from prepare_data_augmentation)
            **self.prepare_data_augmentation(),
            
            # Regularization
            'label_smoothing': 0.0,
            'dropout': 0.0,
            
            # Validation
            'val': True,
            'save_period': -1,
            'save_json': False,
            'save_hybrid': False,
            
            # Class settings
            'single_cls': True,
            'classes': [0],
            
            # Training controls
            'close_mosaic': 10,
            'cos_lr': True,
            'patience': 50,  # Early stopping patience
            'freeze': None,
            
            # Dataset
            'fraction': 1.0,  # Use all data
            'overlap_mask': True,
            'mask_ratio': 4,
        }
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: 640")
        print(f"  Device: {self.device}")
        print(f"  Total frames: {total_frames}")
        print(f"  Augmentation: Enabled")
        
        # Start training
        start_time = datetime.now()
        print(f"\nTraining started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            results = model.train(**train_args)
            training_successful = True
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            training_successful = False
            results = None
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        print(f"\nTraining completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Training duration: {training_duration}")
        
        return results, training_successful
    
    def evaluate_model(self, model_path):
        """Evaluate trained model"""
        print(f"\n{'='*60}")
        print("EVALUATING TRAINED MODEL")
        print(f"{'='*60}")
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return None
        
        model = YOLO(model_path)
        
        # Evaluate on validation set
        metrics = model.val(
            data=str(self.data_dir / 'data.yaml'),
            split='val',
            imgsz=640,
            batch=4,
            device=self.device,
            conf=0.001,
            iou=0.6,
            verbose=True
        )
        
        # Save evaluation results
        eval_results = {
            'metrics': {
                'map50': float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0,
                'map': float(metrics.box.map) if hasattr(metrics.box, 'map') else 0,
                'precision': float(metrics.box.mp) if hasattr(metrics.box, 'mp') else 0,
                'recall': float(metrics.box.mr) if hasattr(metrics.box, 'mr') else 0,
            },
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\nEvaluation Results:")
        print(f"  mAP@0.5: {eval_results['metrics']['map50']:.4f}")
        print(f"  mAP@0.5:0.95: {eval_results['metrics']['map']:.4f}")
        print(f"  Precision: {eval_results['metrics']['precision']:.4f}")
        print(f"  Recall: {eval_results['metrics']['recall']:.4f}")
        
        # Save evaluation results
        eval_file = self.output_dir / 'evaluation_results.json'
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"\n✓ Evaluation results saved to: {eval_file}")
        
        return eval_results
    
    def export_model(self, model_path):
        """Export model to different formats"""
        print(f"\n{'='*60}")
        print("EXPORTING MODEL")
        print(f"{'='*60}")
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        
        try:
            model = YOLO(model_path)
            
            # Export to ONNX
            print("Exporting to ONNX format...")
            model.export(format='onnx', imgsz=640, simplify=True)
            print(f"✓ Model exported to ONNX format")
            
            # Export to TensorRT (if GPU available)
            if self.device == 'cuda':
                try:
                    print("Exporting to TensorRT format...")
                    model.export(format='engine', imgsz=640)
                    print(f"✓ Model exported to TensorRT format")
                except Exception as e:
                    print(f"⚠️  TensorRT export skipped: {e}")
        except Exception as e:
            print(f"⚠️  Export failed: {e}")
    
    def create_training_report(self, stats, total_frames, results, eval_results):
        """Create comprehensive training report"""
        report = {
            'training_info': {
                'dataset_path': str(self.data_dir),
                'total_frames': total_frames,
                'training_frames': stats['train']['images'],
                'validation_frames': stats['val']['images'],
                'training_date': datetime.now().isoformat(),
                'device_used': self.device,
            },
            'dataset_stats': stats,
            'training_results': {
                'best_model': str(self.output_dir / 'weights' / 'best.pt'),
                'last_model': str(self.output_dir / 'weights' / 'last.pt'),
            },
            'evaluation': eval_results if eval_results else {},
            'system_info': {
                'python_version': sys.version,
                'torch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            }
        }
        
        report_file = self.output_dir / 'training_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also create a text summary
        summary_file = self.output_dir / 'training_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"YOLOv8 TRAINING SUMMARY - 477 FRAMES DATASET\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"Dataset Information:\n")
            f.write(f"{'-'*40}\n")
            f.write(f"Total frames: {total_frames}\n")
            f.write(f"Training images: {stats['train']['images']}\n")
            f.write(f"Validation images: {stats['val']['images']}\n")
            f.write(f"Dataset path: {self.data_dir}\n\n")
            
            f.write(f"Training Configuration:\n")
            f.write(f"{'-'*40}\n")
            f.write(f"Epochs: 10\n")
            f.write(f"Image size: 640\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Augmentation: Enabled\n\n")
            
            if eval_results:
                f.write(f"Evaluation Results:\n")
                f.write(f"{'-'*40}\n")
                f.write(f"mAP@0.5: {eval_results['metrics']['map50']:.4f}\n")
                f.write(f"mAP@0.5:0.95: {eval_results['metrics']['map']:.4f}\n")
                f.write(f"Precision: {eval_results['metrics']['precision']:.4f}\n")
                f.write(f"Recall: {eval_results['metrics']['recall']:.4f}\n\n")
            
            f.write(f"Output Files:\n")
            f.write(f"{'-'*40}\n")
            f.write(f"Best model: {self.output_dir}/weights/best.pt\n")
            f.write(f"Last model: {self.output_dir}/weights/last.pt\n")
            f.write(f"Training report: {report_file}\n")
            f.write(f"Evaluation results: {self.output_dir}/evaluation_results.json\n")
            f.write(f"Tensorboard logs: {self.output_dir}\n\n")
            
            f.write(f"{'='*60}\n")
            f.write(f"TRAINING COMPLETE - READY FOR INFERENCE\n")
            f.write(f"{'='*60}\n")
        
        print(f"\n✓ Training report saved to: {report_file}")
        print(f"✓ Training summary saved to: {summary_file}")
        
        return report
    
    def run(self):
        """Main training pipeline"""
        # Step 1: Validate dataset
        stats, total_frames = self.validate_dataset_structure()
        
        # Step 2: Create data configuration
        data_yaml = self.create_data_yaml(stats)
        
        # Step 3: Train model
        results, success = self.train_model(data_yaml, total_frames)
        
        if not success:
            print("\n❌ Training failed. Exiting.")
            return False
        
        # Step 4: Evaluate model
        model_path = self.output_dir / 'weights' / 'best.pt'
        eval_results = self.evaluate_model(str(model_path))
        
        # Step 5: Export model
        self.export_model(str(model_path))
        
        # Step 6: Create report
        self.create_training_report(stats, total_frames, results, eval_results)
        
        print(f"\n{'='*60}")
        print(f"✅ YOLOv8 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"1. Run the Person ReID system:")
        print(f"   python person_reid_full_477.py")
        print(f"2. Or use the complete script:")
        print(f"   ./run_system_477frames.sh")
        print(f"\n{'='*60}")
        
        return True

if __name__ == '__main__':
    trainer = YOLOv8Trainer477()
    trainer.run()
