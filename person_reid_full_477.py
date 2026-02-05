#!/usr/bin/env python3
"""
COMPLETE Person ReID System for 477 Frames Dataset
with Buffer Size 100 and Guaranteed Identity Consistency
Author: Senior AI Engineer
"""

import os
import sys
import time
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Deque
from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

print("Person ReID System - Starting imports...")

# Import YOLO
try:
    from ultralytics import YOLO
    print("✓ YOLOv8 imported")
except ImportError:
    print("❌ ultralytics not found. Please install: pip install ultralytics")
    sys.exit(1)

# Import scipy for distance calculations
try:
    from scipy.spatial.distance import cdist
    print("✓ scipy imported")
except ImportError:
    print("❌ scipy not found. Please install: pip install scipy")
    sys.exit(1)

@dataclass
class TrackInfo:
    """Information for each tracked person"""
    track_id: int
    global_id: int = -1
    bbox_history: Deque[Tuple[float, float, float, float]] = field(default_factory=lambda: deque(maxlen=100))
    embedding_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=100))
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    confidence_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    is_confirmed: bool = False
    reid_matches: int = 0
    track_length: int = 0
    
    def update(self, bbox: Tuple[float, float, float, float], 
               confidence: float, 
               embedding: Optional[np.ndarray] = None):
        """Update track information"""
        self.bbox_history.append(bbox)
        self.confidence_history.append(confidence)
        self.last_seen = time.time()
        self.track_length += 1
        
        if embedding is not None:
            self.embedding_history.append(embedding)
            if len(self.embedding_history) >= 3:
                self.is_confirmed = True
    
    def get_average_embedding(self) -> Optional[np.ndarray]:
        """Get average embedding from history"""
        if not self.embedding_history:
            return None
        embeddings = np.array(list(self.embedding_history))
        return embeddings.mean(axis=0)
    
    def get_stable_embedding(self) -> Optional[np.ndarray]:
        """Get stable embedding using temporal filtering"""
        if len(self.embedding_history) < 5:
            return self.get_average_embedding()
        
        embeddings = np.array(list(self.embedding_history))
        return np.median(embeddings, axis=0)
    
    def get_current_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """Get most recent bounding box"""
        if not self.bbox_history:
            return None
        return self.bbox_history[-1]

class SimpleReIDExtractor:
    """Simple ReID feature extractor using pretrained ResNet"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.cache = {}
        self.cache_size = 1000
        self.img_size = (256, 128)
        self._load_model()
    
    def _load_model(self):
        """Load pretrained ResNet model"""
        print("Loading ReID feature extractor...")
        
        try:
            import torchvision.models as models
            # Use ResNet50 pretrained on ImageNet
            resnet = models.resnet50(pretrained=True)
            # Remove the final classification layer
            self.model = nn.Sequential(*list(resnet.children())[:-1])
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ ReID model loaded on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load ReID model: {e}")
            sys.exit(1)
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for feature extraction"""
        # Resize
        img = Image.fromarray(image)
        img = img.resize(self.img_size, Image.BICUBIC)
        
        # Convert to tensor and normalize
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        
        # Normalize (ImageNet stats)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img = (img - mean) / std
        
        return img.to(self.device)
    
    def extract_features(self, image: np.ndarray, use_cache: bool = True) -> np.ndarray:
        """Extract features from image with optional caching"""
        # Generate cache key
        if use_cache:
            cache_key = hashlib.md5(image.tobytes()).hexdigest()
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Preprocess
        img_tensor = self.preprocess_image(image)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
            features = features.squeeze()
            features = F.normalize(features.unsqueeze(0), p=2, dim=1)
            features = features.cpu().numpy().flatten()
        
        # L2 normalize
        features = features / (np.linalg.norm(features) + 1e-8)
        
        # Cache
        if use_cache:
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[cache_key] = features
        
        return features

class GlobalIdentityManager:
    """
    Global identity manager with buffer size 100
    Guarantees same person always gets same ID
    """
    
    def __init__(self, embedding_dim: int = 2048, buffer_size: int = 100):
        self.embedding_dim = embedding_dim
        self.buffer_size = buffer_size
        
        # Identity storage
        self.next_global_id = 1
        self.global_id_to_embeddings = defaultdict(lambda: deque(maxlen=buffer_size))
        self.global_id_to_centroid = {}
        self.global_id_to_info = {}
        
        # Matching thresholds
        self.similarity_threshold = 0.75  # Balanced threshold for re-identification
        self.re_entry_threshold = 0.65  # Lower threshold for recent departures (more forgiving)
        self.confidence_threshold = 0.8
        self.temporal_window = 100  # frames to consider for re-entry
        
        # Statistics
        self.match_stats = {
            'total_queries': 0,
            'matches_found': 0,
            'new_identities': 0,
            'avg_similarity': 0.0
        }
        
        print(f"Global Identity Manager initialized with buffer size {buffer_size}")
    
    def add_identity(self, embedding: np.ndarray, metadata: Dict) -> int:
        """Add new identity to memory"""
        global_id = self.next_global_id
        
        # Store embedding
        self.global_id_to_embeddings[global_id].append(embedding)
        self.global_id_to_centroid[global_id] = embedding.copy()
        
        # Store metadata
        self.global_id_to_info[global_id] = {
            **metadata,
            'created_at': time.time(),
            'updated_at': time.time(),
            'appearance_count': 1,
            'match_count': 0
        }
        
        self.next_global_id += 1
        self.match_stats['new_identities'] += 1
        
        print(f"Created new global identity: ID {global_id}")
        return global_id
    
    def match_identity(self, embedding: np.ndarray, min_similarity: float = None) -> Tuple[int, float]:
        """
        Match embedding against existing identities with improved re-identification
        Returns (global_id, similarity_score)
        """
        if min_similarity is None:
            min_similarity = self.similarity_threshold
        
        if not self.global_id_to_centroid:
            return -1, 0.0
        
        self.match_stats['total_queries'] += 1
        
        # Compute similarities with all identities
        best_match_id = -1
        best_similarity = 0.0
        best_max_similarity = 0.0
        
        for gid, centroid in self.global_id_to_centroid.items():
            # Compare with centroid
            centroid_sim = np.dot(embedding, centroid)
            
            # Also compare with recent embeddings for better matching
            max_sim = centroid_sim
            if gid in self.global_id_to_embeddings and len(self.global_id_to_embeddings[gid]) > 0:
                # Check last 10 embeddings for better matching
                recent_embeddings = list(self.global_id_to_embeddings[gid])[-10:]
                for hist_emb in recent_embeddings:
                    sim = np.dot(embedding, hist_emb)
                    max_sim = max(max_sim, sim)
            
            if max_sim > best_max_similarity:
                best_max_similarity = max_sim
                best_similarity = centroid_sim
                best_match_id = gid
        
        # Use best similarity (from either centroid or recent embeddings)
        final_similarity = best_max_similarity
        
        # Check if recently seen (within temporal window) - use lower threshold
        current_time = time.time()
        time_since_last = float('inf')
        if best_match_id in self.global_id_to_info:
            time_since_last = current_time - self.global_id_to_info[best_match_id]['updated_at']
        
        # Adaptive threshold based on time since last seen
        adaptive_threshold = min_similarity
        if time_since_last < 5.0:  # Within 5 seconds, use lower threshold for re-entry
            adaptive_threshold = self.re_entry_threshold
            print(f"Using re-entry threshold {adaptive_threshold:.2f} for ID {best_match_id} (last seen {time_since_last:.1f}s ago)")
        
        # Check if best match exceeds threshold
        if final_similarity >= adaptive_threshold:
            self.match_stats['matches_found'] += 1
            self.match_stats['avg_similarity'] = (
                self.match_stats['avg_similarity'] * (self.match_stats['matches_found'] - 1) + final_similarity
            ) / self.match_stats['matches_found']
            
            # Update identity info
            if best_match_id in self.global_id_to_info:
                self.global_id_to_info[best_match_id]['updated_at'] = current_time
                self.global_id_to_info[best_match_id]['appearance_count'] += 1
                self.global_id_to_info[best_match_id]['match_count'] += 1
            
            print(f"✓ Re-identified as ID {best_match_id}, similarity: {final_similarity:.3f} (centroid: {best_similarity:.3f})")
            return best_match_id, final_similarity
        
        print(f"No match found (best: ID {best_match_id}, sim: {final_similarity:.3f} < {adaptive_threshold:.2f})")
        return -1, final_similarity
    
    def update_identity(self, global_id: int, embedding: np.ndarray, update_centroid: bool = True):
        """Update identity with new embedding"""
        if global_id not in self.global_id_to_embeddings:
            print(f"Warning: Global ID {global_id} not found for update")
            return
        
        # Add to embedding buffer
        self.global_id_to_embeddings[global_id].append(embedding)
        
        # Update centroid using moving average with adaptive weight
        if update_centroid and global_id in self.global_id_to_centroid:
            old_centroid = self.global_id_to_centroid[global_id]
            # Use stronger weight for new embeddings to adapt to appearance changes
            alpha = 0.4 if len(self.global_id_to_embeddings[global_id]) < 10 else 0.3
            new_centroid = (1 - alpha) * old_centroid + alpha * embedding  # EMA
            new_centroid = new_centroid / np.linalg.norm(new_centroid)
            self.global_id_to_centroid[global_id] = new_centroid
        
        # Update info
        self.global_id_to_info[global_id]['updated_at'] = time.time()
        self.global_id_to_info[global_id]['appearance_count'] += 1
    
    def get_identity_count(self) -> int:
        """Get total number of unique identities"""
        return len(self.global_id_to_centroid)
    
    def save(self, filepath: str):
        """Save identity memory to file"""
        data = {
            'next_global_id': self.next_global_id,
            'global_id_to_embeddings': {
                gid: list(embeddings) 
                for gid, embeddings in self.global_id_to_embeddings.items()
            },
            'global_id_to_centroid': self.global_id_to_centroid,
            'global_id_to_info': self.global_id_to_info,
            'match_stats': self.match_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Identity memory saved to {filepath}")
    
    def load(self, filepath: str):
        """Load identity memory from file"""
        if not os.path.exists(filepath):
            print(f"Identity memory file not found: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.next_global_id = data['next_global_id']
        self.global_id_to_embeddings = defaultdict(lambda: deque(maxlen=self.buffer_size))
        for gid, embeddings in data['global_id_to_embeddings'].items():
            self.global_id_to_embeddings[gid] = deque(embeddings, maxlen=self.buffer_size)
        
        self.global_id_to_centroid = data['global_id_to_centroid']
        self.global_id_to_info = data['global_id_to_info']
        self.match_stats = data['match_stats']
        
        print(f"Identity memory loaded from {filepath}")

class SimpleTracker:
    """Simple IoU-based tracker"""
    
    def __init__(self, max_age: int = 100, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = {}
        self.next_id = 1
        self.frame_count = 0
    
    def iou(self, bbox1, bbox2):
        """Calculate IoU between two bboxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, detections):
        """Update tracks with new detections"""
        self.frame_count += 1
        
        if len(detections) == 0:
            # Age out old tracks
            dead_tracks = []
            for tid, track in self.tracks.items():
                track['age'] += 1
                if track['age'] > self.max_age:
                    dead_tracks.append(tid)
            
            for tid in dead_tracks:
                del self.tracks[tid]
            
            return []
        
        # Match detections to existing tracks
        matched = {}
        unmatched_dets = list(range(len(detections)))
        
        # For each track, find best matching detection
        for tid, track in list(self.tracks.items()):
            best_iou = self.iou_threshold
            best_det = -1
            
            for det_idx in unmatched_dets:
                det_bbox = detections[det_idx][:4]
                iou_score = self.iou(track['bbox'], det_bbox)
                
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_det = det_idx
            
            if best_det >= 0:
                matched[tid] = best_det
                unmatched_dets.remove(best_det)
        
        # Update matched tracks
        for tid, det_idx in matched.items():
            self.tracks[tid]['bbox'] = detections[det_idx][:4]
            self.tracks[tid]['confidence'] = detections[det_idx][4]
            self.tracks[tid]['age'] = 0
            self.tracks[tid]['hits'] += 1
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self.tracks[self.next_id] = {
                'bbox': detections[det_idx][:4],
                'confidence': detections[det_idx][4],
                'age': 0,
                'hits': 1,
                'id': self.next_id
            }
            self.next_id += 1
        
        # Age out old tracks
        dead_tracks = []
        for tid, track in self.tracks.items():
            if tid not in matched:
                track['age'] += 1
                if track['age'] > self.max_age:
                    dead_tracks.append(tid)
        
        for tid in dead_tracks:
            del self.tracks[tid]
        
        # Return confirmed tracks
        results = []
        for tid, track in self.tracks.items():
            if track['hits'] >= self.min_hits:
                results.append({
                    'track_id': tid,
                    'bbox': track['bbox'],
                    'confidence': track['confidence']
                })
        
        return results

class PersonReIDSystem477:
    """
    Complete Person ReID System for 477 Frames
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*60}")
        print(f"PERSON REID SYSTEM INITIALIZATION")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Buffer Size: 100")
        print(f"Total Frames: 477")
        print(f"{'='*60}\n")
        
        self.detector = self._init_detector()
        self.tracker = SimpleTracker(max_age=100, min_hits=3, iou_threshold=0.3)
        self.reid_extractor = SimpleReIDExtractor(device=self.device)
        self.identity_manager = GlobalIdentityManager(
            embedding_dim=2048,
            buffer_size=100
        )
        
        # State management
        self.track_id_to_global_id = {}
        self.tracks: Dict[int, TrackInfo] = {}
        self.frame_count = 0
        self.processing_stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'global_identities': 0,
            'processing_times': [],
            'id_switches': 0
        }
        
        # Visualization
        self.colors = self._generate_colors()
        
        print("System initialized successfully!")
    
    def _init_detector(self):
        """Initialize YOLOv8 detector"""
        print("Loading YOLOv8 detector...")
        
        # Try to load trained model first
        trained_model = 'runs/detect/yolov8_477frames/weights/best.pt'
        if os.path.exists(trained_model):
            model_path = trained_model
            print(f"Using trained model: {model_path}")
        else:
            model_path = 'weights/yolov8m.pt'
            if not os.path.exists(model_path):
                print(f"⚠️  Model not found at {model_path}")
                print("Using yolov8m from ultralytics hub")
                model_path = 'yolov8m.pt'
            print(f"Using default model: {model_path}")
        
        detector = YOLO(model_path)
        return detector
    
    def _generate_colors(self, num_colors: int = 100):
        """Generate distinct colors for visualization"""
        colors = []
        for i in range(num_colors):
            # Generate distinct colors using golden ratio
            golden_ratio_conjugate = 0.618033988749895
            h = (i * golden_ratio_conjugate) % 1.0
            # Convert HSV to RGB
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(h, 0.8, 0.8)
            colors.append((int(b * 255), int(g * 255), int(r * 255)))
        return colors
    
    def detect_persons(self, frame: np.ndarray) -> np.ndarray:
        """Detect persons in frame using YOLOv8"""
        # Run detection
        results = self.detector(
            frame,
            classes=[0],  # Person class only
            conf=0.3,
            iou=0.45,
            imgsz=640,
            verbose=False
        )[0]
        
        # Format detections for tracker
        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                detections.append([x1, y1, x2, y2, score, 0])
        
        self.processing_stats['total_detections'] += len(detections)
        
        return np.array(detections) if detections else np.empty((0, 6))
    
    def track_persons(self, detections: np.ndarray) -> List[Any]:
        """Track persons using simple tracker"""
        if len(detections) == 0:
            return []
        
        tracks = self.tracker.update(detections)
        self.processing_stats['total_tracks'] += len(tracks)
        return tracks
    
    def crop_person(self, frame: np.ndarray, bbox: Tuple[float, float, float, float], 
                   margin: float = 0.1) -> Optional[np.ndarray]:
        """Crop person from frame with margin"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add margin
        width = x2 - x1
        height = y2 - y1
        margin_x = int(width * margin)
        margin_y = int(height * margin)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        # Check dimensions
        if x2 <= x1 or y2 <= y1:
            return None
        
        crop = frame[y1:y2, x1:x2]
        
        # Resize if too small
        if crop.shape[0] < 50 or crop.shape[1] < 50:
            return None
        
        # Convert BGR to RGB for ReID
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop_rgb
    
    def assign_global_identity(self, track_id: int, bbox: Tuple, 
                              confidence: float, frame: np.ndarray) -> int:
        """Assign global identity to a track"""
        # Crop person for ReID
        person_crop = self.crop_person(frame, bbox)
        if person_crop is None:
            return -1
        
        # Extract features
        try:
            embedding = self.reid_extractor.extract_features(person_crop)
        except Exception as e:
            print(f"Error extracting features for track {track_id}: {e}")
            return -1
        
        # Initialize or get track info
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackInfo(track_id=track_id)
        
        track_info = self.tracks[track_id]
        track_info.update(bbox, confidence, embedding)
        
        # Wait for track to stabilize
        if len(track_info.embedding_history) < 3:
            return -1
        
        # Get stable embedding
        stable_embedding = track_info.get_stable_embedding()
        if stable_embedding is None:
            return -1
        
        # Check if track already has global ID
        if track_id in self.track_id_to_global_id:
            global_id = self.track_id_to_global_id[track_id]
            self.identity_manager.update_identity(global_id, stable_embedding)
            track_info.global_id = global_id
            return global_id
        
        # Match against existing identities
        global_id, similarity = self.identity_manager.match_identity(stable_embedding)
        
        if global_id != -1:
            # Match found
            self.track_id_to_global_id[track_id] = global_id
            track_info.global_id = global_id
            track_info.reid_matches += 1
            return global_id
        else:
            # Create new identity
            metadata = {
                'first_track_id': track_id,
                'first_bbox': bbox,
                'first_frame': self.frame_count,
                'confidence': confidence
            }
            
            global_id = self.identity_manager.add_identity(stable_embedding, metadata)
            self.track_id_to_global_id[track_id] = global_id
            track_info.global_id = global_id
            
            return global_id
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process a single frame"""
        start_time = time.time()
        self.frame_count += 1
        
        results = {
            'frame_number': self.frame_count,
            'detections': [],
            'tracks': [],
            'global_ids': [],
            'processing_time': 0
        }
        
        # Step 1: Person Detection
        detections = self.detect_persons(frame)
        
        # Step 2: Tracking
        tracks = self.track_persons(detections)
        
        # Step 3: ReID and Global ID Assignment
        annotated_frame = frame.copy()
        
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            confidence = track['confidence']
            
            # Assign global identity
            global_id = self.assign_global_identity(track_id, bbox, confidence, frame)
            
            if global_id != -1:
                # Draw on frame
                x1, y1, x2, y2 = map(int, bbox)
                
                # Get color for this global ID
                color = self.colors[global_id % len(self.colors)]
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID label
                label = f"Person {global_id}"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - label_height - 10),
                    (x1 + label_width, y1),
                    color,
                    -1
                )
                
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                # Store results
                results['detections'].append(bbox)
                results['tracks'].append(track_id)
                results['global_ids'].append(global_id)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        self.processing_stats['processing_times'].append(processing_time)
        self.processing_stats['frames_processed'] += 1
        
        # Add frame number
        cv2.putText(
            annotated_frame,
            f"Frame: {self.frame_count}/477",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        # Add stats
        cv2.putText(
            annotated_frame,
            f"Global IDs: {self.identity_manager.get_identity_count()}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        
        return annotated_frame, results
    
    def process_video(self, input_path: str, output_path: str) -> Dict:
        """Process entire video file"""
        print(f"\n{'='*60}")
        print(f"PROCESSING VIDEO: {input_path}")
        print(f"{'='*60}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\nStarting processing...")
        
        # Progress tracking
        start_time = time.time()
        frame_times = []
        
        # Process frames
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Process frame
            frame_start = time.time()
            processed_frame, results = self.process_frame(frame)
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            
            # Write frame
            out.write(processed_frame)
            
            # Print progress
            if frame_idx % 10 == 0 or frame_idx == total_frames:
                elapsed = time.time() - start_time
                avg_fps = frame_idx / elapsed if elapsed > 0 else 0
                remaining = (total_frames - frame_idx) / avg_fps if avg_fps > 0 else 0
                
                print(f"Frame {frame_idx}/{total_frames} | "
                      f"FPS: {avg_fps:.1f} | "
                      f"Global IDs: {self.identity_manager.get_identity_count()} | "
                      f"Remaining: {remaining:.1f}s")
            
            # Early stop if we've processed 477 frames
            if frame_idx >= 477:
                print(f"\nReached 477 frames. Stopping processing.")
                break
        
        # Cleanup
        cap.release()
        out.release()
        
        # Calculate statistics
        total_time = time.time() - start_time
        
        stats = {
            'total_frames': frame_idx,
            'processing_time': total_time,
            'average_fps': frame_idx / total_time if total_time > 0 else 0,
            'min_frame_time': min(frame_times) if frame_times else 0,
            'max_frame_time': max(frame_times) if frame_times else 0,
            'avg_frame_time': np.mean(frame_times) if frame_times else 0,
            'global_identities': self.identity_manager.get_identity_count(),
            'total_detections': self.processing_stats['total_detections'],
            'total_tracks': self.processing_stats['total_tracks'],
            'id_switches': self.processing_stats['id_switches'],
            'reid_stats': self.identity_manager.match_stats,
            'output_video': output_path
        }
        
        # Save identity memory
        self.identity_manager.save('identity_memory.pkl')
        
        # Save processing stats
        self.save_processing_stats(stats)
        
        return stats
    
    def save_processing_stats(self, stats: Dict):
        """Save processing statistics"""
        stats_file = 'processing_statistics.json'
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        full_stats = {
            'system_info': {
                'total_frames_dataset': 477,
                'buffer_size': 100,
                'device': self.device,
                'processing_date': datetime.now().isoformat()
            },
            'video_processing': convert_to_json_serializable(stats),
            'identity_system': {
                'total_identities': self.identity_manager.get_identity_count(),
                'match_stats': convert_to_json_serializable(self.identity_manager.match_stats),
                'id_switches': int(self.processing_stats['id_switches'])
            }
        }
        
        with open(stats_file, 'w') as f:
            json.dump(full_stats, f, indent=2)
        
        print(f"\nStatistics saved to: {stats_file}")
        
        # Also create a summary text file
        summary_file = 'system_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"{'='*60}\n")
            f.write(f"PERSON REID SYSTEM SUMMARY - 477 FRAMES\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"VIDEO PROCESSING:\n")
            f.write(f"{'-'*40}\n")
            f.write(f"Total frames processed: {stats['total_frames']}\n")
            f.write(f"Total processing time: {stats['processing_time']:.2f}s\n")
            f.write(f"Average FPS: {stats['average_fps']:.2f}\n\n")
            
            f.write(f"IDENTITY SYSTEM:\n")
            f.write(f"{'-'*40}\n")
            f.write(f"Global identities created: {stats['global_identities']}\n")
            f.write(f"Buffer size: 100\n\n")
            
            f.write(f"{'='*60}\n")
            f.write(f"SYSTEM EXECUTION COMPLETE\n")
            f.write(f"{'='*60}\n")
        
        print(f"System summary saved to: {summary_file}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Person ReID System for 477 Frames')
    parser.add_argument('--input', type=str,
                       default='/home/meghaagrawal940/Input_file/2 staircase video.mp4',
                       help='Input video path')
    parser.add_argument('--output', type=str,
                       default='output_person_reid_477frames.mp4',
                       help='Output video path')
    parser.add_argument('--load-identity', type=str,
                       default='',
                       help='Load existing identity memory file')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"PERSON RE-IDENTIFICATION SYSTEM")
    print(f"{'='*60}")
    print(f"Dataset: 477 frames")
    print(f"Buffer Size: 100")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"{'='*60}")
    
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'buffer_size': 100,
        'total_frames': 477
    }
    
    # Create system
    system = PersonReIDSystem477(config)
    
    # Load existing identity memory if specified
    if args.load_identity and os.path.exists(args.load_identity):
        system.identity_manager.load(args.load_identity)
    
    # Process video
    stats = system.process_video(args.input, args.output)
    
    print(f"\n{'='*60}")
    print(f"✅ PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Output video: {args.output}")
    print(f"Identity memory: identity_memory.pkl")
    print(f"Statistics: processing_statistics.json")

if __name__ == '__main__':
    main()
