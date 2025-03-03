"""
Video Swin Transformer Evaluation on Kinetics 600
-------------------------------------------------
Script to evaluate the Video Swin Transformer model against the Kinetics 600 test set.

This script supports multi-view testing and automatic mixed precision for faster evaluation.

Usage:
    python evaluate_kinetics.py --video_root /path/to/kinetics600 \
                               --annotation_file /path/to/test.json \
                               --pretrained_path /path/to/swin_base_patch244_window877_kinetics600_22k.pth \
                               --model_type base \
                               --output_file results.json

Optional arguments:
    --num_clips: Number of temporal clips to sample for each video (default: 10)
    --num_crops: Number of spatial crops for each clip (1 or 3, default: 3)
    --batch_size: Batch size for DataLoader (default: 16)
    --num_workers: Number of workers for DataLoader (default: 8)
    --amp: Use Automatic Mixed Precision (default: True)
    --classnames_file: Path to class names mapping file (JSON)
"""

import os
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset, DataLoader
import warnings
import sys
import glob
import traceback

try:
    import decord
except ImportError:
    warnings.warn("Decord library not found. Please install with 'pip install decord'")
    decord = None


# Debug: Print current directory
current_dir = os.path.abspath(os.path.dirname(__file__))
print(f"Current directory: {current_dir}")

project_dir = current_dir
while project_dir and not os.path.exists(os.path.join(project_dir, 'vst.py')):
    parent = os.path.dirname(project_dir)
    if parent == project_dir:  # Reached the root directory
        project_dir = None
        break
    project_dir = parent

if project_dir:
    print(f"Found project directory: {project_dir}")
    sys.path.insert(0, project_dir)
else:
    print("WARNING: Could not find the project directory containing vst.py")
    # Add multiple potential parent directories to cover different project structures
    sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))  # up two levels
    sys.path.insert(0, os.path.dirname(current_dir))  # up one level

# Now attempt the import
try:
    from vst import (
        video_swin_b,
        video_swin_s,
        video_swin_t,
        video_swin_l,
        get_device
    )
    print("Successfully imported vst module")
except ImportError:
    raise ImportError(
        "Video Swin Transformer implementation not found. "
        "Make sure the vst.py file is in your PYTHONPATH."
    )


# Helper function to load a model with specific configuration
def load_model(model_type, pretrained_path, num_classes=600, device=None):
    """
    Load a Video Swin Transformer model.

    Args:
        model_type (str): Model type ('tiny', 'small', 'base', or 'large')
        pretrained_path (str): Path to the pretrained weights file
        num_classes (int): Number of classes (default: 600 for Kinetics 600)
        device (torch.device): Device to place the model on

    Returns:
        model (nn.Module): Loaded model
    """
    if device is None:
        device = get_device()

    print(f"Loading {model_type} model from {pretrained_path} to {device}")

    if model_type.lower() == 'base' or model_type.lower() == 'b':
        model = video_swin_b(pretrained=pretrained_path, num_classes=num_classes, device=device)
    elif model_type.lower() == 'small' or model_type.lower() == 's':
        model = video_swin_s(pretrained=pretrained_path, num_classes=num_classes, device=device)
    elif model_type.lower() == 'tiny' or model_type.lower() == 't':
        model = video_swin_t(pretrained=pretrained_path, num_classes=num_classes, device=device)
    elif model_type.lower() == 'large' or model_type.lower() == 'l':
        model = video_swin_l(pretrained=pretrained_path, num_classes=num_classes, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    return model


# Evaluator class to manage evaluation process
class VideoSwinEvaluator:
    """
    Evaluator for Video Swin Transformer on Kinetics dataset.
    """

    def __init__(self, model, dataloader, use_amp=True, device=None):
        """
        Initialize the evaluator.

        Args:
            model (nn.Module): The Video Swin Transformer model
            dataloader (DataLoader): DataLoader for the test set
            use_amp (bool): Whether to use Automatic Mixed Precision
            device (torch.device): Device to use for evaluation
        """
        self.model = model
        self.dataloader = dataloader
        self.use_amp = use_amp and torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
        self.device = device if device is not None else get_device()

        # Move model to device if it's not already there
        if next(model.parameters()).device != self.device:
            self.model = self.model.to(self.device)

        # Set evaluation mode
        self.model.eval()

        # Initialize metrics
        self.total_videos = 0
        self.correct_top1 = 0
        self.correct_top5 = 0
        self.video_predictions = {}

    def run(self):
        start_time = time.time()
        valid_labels = True  # Assume labels are valid initially

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Evaluating")):
                if batch['video'].shape[0] == 0:
                    print(f"Skipping empty batch {batch_idx}")
                    continue

                videos = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)

                # Check on the first batch: if all labels are -1, assume they are missing
                if valid_labels and (labels == -1).all():
                    print("Warning: Ground-truth labels appear to be missing. Skipping accuracy metrics.")
                    valid_labels = False

                b, v, c, t, h, w = videos.shape
                videos = videos.reshape(-1, c, t, h, w)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(videos)
                else:
                    outputs = self.model(videos)

                outputs = outputs.reshape(b, v, -1)
                outputs = outputs.mean(dim=1)

                # Always store predictions
                self._store_predictions(outputs, labels, batch_idx)

                # Only update metrics if valid labels are present
                if valid_labels:
                    self._update_metrics(outputs, labels, batch_idx)

        metrics = self._calculate_metrics(valid_labels)
        metrics['eval_time'] = time.time() - start_time
        return metrics

    def _store_predictions(self, outputs, labels, batch_idx):
        # This helper stores predictions for each video (same as before)
        _, pred_indices = outputs.topk(5, dim=1)
        batch_size = labels.size(0)
        for i in range(batch_size):
            video_id = f"batch_{batch_idx}_{i}"
            self.video_predictions[video_id] = {
                'label': labels[i].item(),
                'pred_top5': pred_indices[i].cpu().numpy().tolist(),
                'confidence': F.softmax(outputs[i], dim=0)[pred_indices[i][0]].item()
            }

    def _calculate_metrics(self, valid_labels):
        if not valid_labels:
            # When no valid labels are present, set accuracy values to None or "N/A"
            metrics = {
                'top1_accuracy': None,
                'top5_accuracy': None,
                'total_videos': self.total_videos,
                'predictions': self.video_predictions
            }
            return metrics

        accuracy_top1 = self.correct_top1 / self.total_videos if self.total_videos > 0 else 0
        accuracy_top5 = self.correct_top5 / self.total_videos if self.total_videos > 0 else 0
        metrics = {
            'top1_accuracy': accuracy_top1,
            'top5_accuracy': accuracy_top5,
            'total_videos': self.total_videos,
            'predictions': self.video_predictions
        }
        return metrics

    def _update_metrics(self, outputs, labels, batch_idx):
        # Calculate top-1 and top-5 accuracy for the batch
        topk_result = self._calculate_topk(outputs, labels, (1, 5))
        _, pred_indices = outputs.topk(5, dim=1)
        batch_size = labels.size(0)
        self.total_videos += batch_size
        self.correct_top1 += topk_result[0].item() * batch_size
        self.correct_top5 += topk_result[1].item() * batch_size

        # Store predictions for each video in the batch
        for i in range(batch_size):
            video_id = f"batch_{batch_idx}_{i}"
            self.video_predictions[video_id] = {
                'label': labels[i].item(),
                'pred_top5': pred_indices[i].cpu().numpy().tolist(),
                'confidence': F.softmax(outputs[i], dim=0)[pred_indices[i][0]].item()
            }

    @staticmethod
    def _calculate_topk(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


# Function to load class names from file
def load_classnames(file_path=None):
    """
    Load class names from a file or return a sequential mapping if no file.

    Args:
        file_path (str): Path to the JSON file with class names

    Returns:
        dict: Dictionary mapping class indices to class names
    """
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                # Try several common formats
                data = json.load(f)

                # Format 1: {index: class_name}
                if all(k.isdigit() for k in data.keys()):
                    return {int(k): v for k, v in data.items()}

                # Format 2: {class_name: index}
                elif all(isinstance(v, int) for v in data.values()):
                    return {v: k for k, v in data.items()}

                # Format 3: [class_name, class_name, ...]
                elif isinstance(data, list) and all(isinstance(v, str) for v in data):
                    return {i: classname for i, classname in enumerate(data)}

                # Format 4: [{id: index, name: class_name}, ...]
                elif isinstance(data, list) and all(isinstance(v, dict) for v in data):
                    if all('id' in v and 'name' in v for v in data):
                        return {v['id']: v['name'] for v in data}

                # Format 5: {"id_to_name": {}, "name_to_id": {}}
                elif "id_to_name" in data:
                    return {int(k): v for k, v in data["id_to_name"].items()}

                # Default
                print(f"Unknown class names format in {file_path}, using sequential indices")
                return {i: f"class_{i}" for i in range(600)}

            except Exception as e:
                print(f"Error loading class names from {file_path}: {e}")
                return {i: f"class_{i}" for i in range(600)}
    else:
        # Generate sequential class names
        return {i: f"class_{i}" for i in range(600)}


class KineticsDataset(Dataset):
    """
    Kinetics dataset for video action recognition.
    Supports multi-view testing with multiple clips and crops.
    """

    def __init__(self, video_root, annotation_path, clip_len=16, frame_sample_rate=2,
                 crop_size=224, short_side_size=256, num_clips=10, num_crops=10, mode='test'):
        """
        Initialize the Kinetics dataset.

        Args:
            video_root (str): Root directory path for Kinetics videos
            annotation_path (str): Path to the annotation file (JSON format)
            clip_len (int): Number of frames in a clip
            frame_sample_rate (int): Sampling rate for frames
            crop_size (int): Size of the crop
            short_side_size (int): Size to which the shorter side is resized
            num_clips (int): Number of clips per video for multi-view testing
            num_crops (int): Number of spatial crops per clip (1 or 3)
            mode (str): 'train', 'val', or 'test'
        """
        self.video_root = video_root
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.num_clips = num_clips
        self.num_crops = num_crops
        self.mode = mode

        # Initialize label mapping
        self.label_mapping = {}

        # Load annotations
        self.videos = []
        self.load_annotations(annotation_path)

        print(f"Loaded {len(self.videos)} videos for {mode}")
        print(f"Num_clips: {num_clips}. Crops: {num_crops}")

        # Print a few examples
        if self.videos:
            print("\nFirst few video entries:")
            for i, (video_id, video_path, label) in enumerate(self.videos[:3]):
                print(f"  {i + 1}. ID: {video_id}, Path: {video_path}, Label: {label}")
        else:
            print("No videos were loaded!")

    def load_annotations(self, annotation_path):
        """
        Load annotations based on the file format.
        """
        print(f"\nLoading annotations from {annotation_path}")

        if annotation_path.endswith('.json'):
            self.load_json_annotations(annotation_path)
        elif annotation_path.endswith('.csv'):
            self.load_csv_annotations(annotation_path)
        else:
            raise ValueError(f"Unsupported annotation file format: {annotation_path}")

        print(f"Loaded {len(self.videos)} videos for {self.mode}")

        # If no videos were loaded, try to be flexible with file paths
        if len(self.videos) == 0:
            print("No videos loaded, attempting flexible path matching...")
            self.try_flexible_path_matching()

    def load_json_annotations(self, annotation_file):
        """
        Load annotations from a JSON file.
        """
        print(f"Loading JSON annotations from {annotation_file}")
        try:
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)

            # Print annotation structure
            if annotations:
                first_key = next(iter(annotations))
                print(f"First annotation key: {first_key}")
                print(f"First annotation structure: {annotations[first_key]}")

            # Create a label mapping for string labels
            unique_labels = set()

            # First pass: collect all unique labels
            for video_id, info in annotations.items():
                # Extract label
                label = None
                if 'annotations' in info and 'label' in info['annotations']:
                    label = info['annotations']['label']
                elif 'label' in info:
                    label = info['label']

                if label is not None:
                    unique_labels.add(label)

            # Create label mapping for string labels
            string_labels = [label for label in unique_labels if isinstance(label, str) and not str(label).isdigit()]
            if string_labels:
                self.label_mapping = {label: idx for idx, label in enumerate(sorted(string_labels))}
                print(f"Created label mapping with {len(self.label_mapping)} unique string labels")

            # Second pass: add videos with mapped labels
            for video_id, info in annotations.items():
                # Check for file_path
                video_path = info.get('file_path', None)

                # Extract label
                label = -1
                if 'annotations' in info and 'label' in info['annotations']:
                    label_value = info['annotations']['label']
                    if isinstance(label_value, str) and not label_value.isdigit():
                        label = self.label_mapping.get(label_value, -1)
                    else:
                        try:
                            label = int(label_value)
                        except (ValueError, TypeError):
                            label = -1
                elif 'label' in info:
                    label_value = info['label']
                    if isinstance(label_value, str) and not label_value.isdigit():
                        label = self.label_mapping.get(label_value, -1)
                    else:
                        try:
                            label = int(label_value)
                        except (ValueError, TypeError):
                            label = -1

                # Check if file exists
                if video_path and os.path.exists(video_path):
                    self.videos.append((video_id, video_path, label))
                elif video_path:
                    # Try relative path
                    rel_path = os.path.join(self.video_root, os.path.basename(video_path))
                    if os.path.exists(rel_path):
                        self.videos.append((video_id, rel_path, label))
                    else:
                        print(f"Video not found: {video_path}")

        except Exception as e:
            print(f"Error loading JSON annotations: {e}")
            print(traceback.format_exc())

    def load_csv_annotations(self, annotation_path):
        import csv
        with open(annotation_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)  # Read header

            video_id_col = 0
            label_col = -1  # Default to -1 if no label column exists
            if header:
                lower_header = [h.lower() for h in header]
                if 'youtube_id' in lower_header:
                    video_id_col = lower_header.index('youtube_id')
                if 'label' in lower_header:
                    label_col = lower_header.index('label')
                elif 'class_name' in lower_header:
                    label_col = lower_header.index('class_name')
                elif 'classname' in lower_header:
                    label_col = lower_header.index('classname')

            # Collect unique labels only if label_col is valid (>=0)
            unique_labels = set()
            if label_col >= 0:
                for row in reader:
                    if len(row) > label_col:
                        unique_labels.add(row[label_col])
                # Create label mapping for string labels
                string_labels = [label for label in unique_labels if isinstance(label, str) and not label.isdigit()]
                if string_labels:
                    self.label_mapping = {label: idx for idx, label in enumerate(sorted(string_labels))}

                # Reset file pointer and skip header again
                f.seek(0)
                next(reader, None)
            else:
                print("Warning: No label column found in CSV; all labels will be set to -1.")

            # Process rows
            for row in reader:
                if len(row) > video_id_col:
                    video_id = row[video_id_col]
                    if label_col >= 0 and len(row) > label_col:
                        label_str = row[label_col]
                        # Convert label to int if possible; otherwise, use mapping if available
                        if isinstance(label_str, str) and not label_str.isdigit():
                            label = self.label_mapping.get(label_str, -1)
                        else:
                            try:
                                label = int(label_str)
                            except (ValueError, TypeError):
                                label = -1
                    else:
                        label = -1  # Dummy value when label is missing

                    video_path = self.find_video_path(video_id)
                    if video_path:
                        self.videos.append((video_id, video_path, label))

    def find_video_path(self, video_id):
        """
        Find the path to a video file based on the video_id.
        """
        # Common patterns for Kinetics videos
        patterns = [
            # Pattern 1: {video_root}/{video_id}.mp4
            os.path.join(self.video_root, f"{video_id}.mp4"),

            # Pattern 2: {video_root}/{subset}/{video_id}.mp4
            os.path.join(self.video_root, self.mode, f"{video_id}.mp4"),

            # Pattern 3: {video_root}/{video_id[:2]}/{video_id}.mp4 (first two chars)
            os.path.join(self.video_root, video_id[:2] if len(video_id) >= 2 else "", f"{video_id}.mp4"),
        ]

        for pattern in patterns:
            if os.path.exists(pattern):
                return pattern

        # If not found with common patterns, try a deeper search but with limits
        for root, _, files in os.walk(self.video_root, topdown=True, followlinks=False):
            for file in files:
                if file.startswith(video_id) and file.endswith('.mp4'):
                    return os.path.join(root, file)

            # Limit how deep we search to avoid very long searches
            rel_path = os.path.relpath(root, self.video_root)
            if rel_path.count(os.sep) > 2:  # Don't go deeper than 2 levels
                break

        return None

    def try_flexible_path_matching(self):
        """Try to match videos even if paths don't exactly match."""
        print(f"Scanning {self.video_root} for videos...")

        # Find all video files
        video_files = glob.glob(os.path.join(self.video_root, "**", "*.mp4"), recursive=True)
        print(f"Found {len(video_files)} MP4 files in {self.video_root}")

        # Use each file as a dummy entry
        for i, file_path in enumerate(video_files):
            video_id = f"video_{i}"
            label = i % 600  # Dummy label
            self.videos.append((video_id, file_path, label))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        """
        Get an item from the dataset.
        """
        video_id, video_path, label = self.videos[index]

        try:
            # Check if file exists
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                return self.get_empty_sample(label)

            # Load video
            if decord is not None:
                video_data = self.load_video_decord(video_path)
            else:
                video_data = self.load_video_cv2(video_path)

            if video_data is None:
                print(f"Failed to load video data for {video_path}")
                return self.get_empty_sample(label)

            # Process the data
            result = self.process_data(video_data, label)
            return result

        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            traceback.print_exc()
            return self.get_empty_sample(label)

    def get_empty_sample(self, label):
        """Return an empty tensor for videos that can't be loaded."""
        # Make sure label is an integer
        try:
            label_int = int(label)
        except (ValueError, TypeError):
            label_int = 0

        if self.mode == 'test':
            num_views = self.num_clips * self.num_crops
            return {
                'video': torch.zeros(num_views, 3, self.clip_len, self.crop_size, self.crop_size),
                'label': torch.tensor(label_int, dtype=torch.long),
                'video_idx': -1
            }
        else:
            return {
                'video': torch.zeros(3, self.clip_len, self.crop_size, self.crop_size),
                'label': torch.tensor(label_int, dtype=torch.long),
                'video_idx': -1
            }

    def load_video_decord(self, video_path):
        """Load video using Decord library for efficient decoding."""
        try:
            container = decord.VideoReader(video_path)
            total_frames = len(container)

            if total_frames <= 0:
                return None

            # Calculate frame indices based on num_clips
            frame_indices = self.sample_frames_from_video(total_frames)

            # Load frames based on indices
            video_data = []
            for indices in frame_indices:
                frames = container.get_batch(indices).asnumpy()
                frames = [torch.from_numpy(frame) for frame in frames]
                video_data.append(torch.stack(frames, dim=0))

            return video_data

        except Exception as e:
            print(f"Error in Decord loading for {video_path}: {e}")
            return None

    def load_video_cv2(self, video_path):
        """Load video using OpenCV (slower but more compatible)."""
        try:
            cap = cv2.VideoCapture(video_path)

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames <= 0 or not cap.isOpened():
                cap.release()
                return None

            # Calculate frame indices for each clip
            frame_indices = self.sample_frames_from_video(total_frames)

            # Read frames for each clip
            video_data = []
            for indices in frame_indices:
                frames = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame)
                    frames.append(frame)

                if len(frames) == len(indices):
                    video_data.append(torch.stack(frames, dim=0))

            cap.release()
            return video_data if video_data else None

        except Exception as e:
            print(f"Error in OpenCV loading for {video_path}: {e}")
            return None

    def sample_frames_from_video(self, total_frames):
        """Sample frame indices from the video based on num_clips."""
        # Ensure we can sample enough frames
        if total_frames <= 0:
            return []

        if self.mode == 'test':
            # For testing: uniformly sample multiple clips
            frame_indices = []

            # Calculate effective video length considering sampling rate
            effective_length = total_frames // self.frame_sample_rate

            # If video is too short, adjust sampling rate
            if effective_length < self.clip_len:
                # Try to sample without skipping frames
                if total_frames < self.clip_len:
                    # Very short video: repeat frames
                    indices = np.arange(0, self.clip_len)
                    indices = indices % total_frames
                else:
                    # Sample without rate
                    indices = np.linspace(0, total_frames - 1, self.clip_len).astype(int)

                # Replicate the same clip for num_clips
                for _ in range(self.num_clips):
                    frame_indices.append(indices)
            else:
                # Normal case: sample evenly spaced clips
                tick = effective_length / float(self.num_clips)
                for i in range(self.num_clips):
                    start_idx = int(tick / 2.0 + tick * i)

                    # Sample clip_len frames with frame_sample_rate
                    clip_indices = []
                    for j in range(self.clip_len):
                        idx = start_idx + j * self.frame_sample_rate
                        idx = min(idx, total_frames - 1)
                        clip_indices.append(int(idx))

                    frame_indices.append(np.array(clip_indices))

            return frame_indices
        else:
            # For training/validation: random or center sampling
            # This implementation focuses on test mode
            center_idx = total_frames // 2
            indices = np.arange(
                center_idx - self.clip_len * self.frame_sample_rate // 2,
                center_idx + self.clip_len * self.frame_sample_rate // 2,
                self.frame_sample_rate
            )

            # Handle boundary cases
            indices = np.clip(indices, 0, total_frames - 1)

            # If indices are too few, repeat last frame or interpolate
            if len(indices) < self.clip_len:
                indices = np.array(
                    [indices[i % len(indices)] for i in range(self.clip_len)]
                )

            return [indices]

    def process_data(self, video_data, label):
        """Process the video clips for model input."""
        # Make sure label is an integer
        try:
            label_int = int(label)
        except (ValueError, TypeError):
            label_int = 0

        processed_clips = []

        for clip in video_data:
            # Convert to float and scale to [0, 1]
            clip = clip.float() / 255.0

            # Apply spatial augmentation (center crop, 3 crops, etc.)
            processed_views = self.apply_spatial_transforms(clip)
            processed_clips.extend(processed_views)

        # Stack all clips and views
        video_tensor = torch.stack(processed_clips)

        # Create sample dict
        sample = {
            'video': video_tensor,  # Shape: [num_views, C, T, H, W]
            'label': torch.tensor(label_int, dtype=torch.long),
            'video_idx': len(processed_clips)  # For identification
        }

        return sample

    def apply_spatial_transforms(self, clip):
        """Apply spatial transforms to get multiple views of the clip."""
        # Clip shape: [T, H, W, C]
        T, H, W, C = clip.shape

        # Convert to [T, C, H, W] for easier processing
        clip = clip.permute(0, 3, 1, 2)

        # Resize video keeping aspect ratio
        if H >= W:
            new_h, new_w = int(self.short_side_size * H / W), self.short_side_size
        else:
            new_h, new_w = self.short_side_size, int(self.short_side_size * W / H)

        # Apply resize using F.interpolate
        clip = F.interpolate(clip, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Apply crops based on num_crops
        if self.num_crops == 1:
            # Only center crop
            crops = [self.center_crop(clip, self.crop_size)]
        else:
            # Use 3 crops: center, top-left, bottom-right
            crops = [
                self.center_crop(clip, self.crop_size),
                self.corner_crop(clip, self.crop_size, 'top-left'),
                self.corner_crop(clip, self.crop_size, 'bottom-right')
            ]

        # Apply normalization: subtract mean and divide by std
        normalized_crops = []
        for crop in crops:
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            normalized = (crop - mean) / std

            # Convert to [C, T, H, W] format
            normalized = normalized.permute(1, 0, 2, 3)
            normalized_crops.append(normalized)

        return normalized_crops

    def center_crop(self, clip, crop_size):
        """Apply center crop to the clip."""
        # clip shape: [T, C, H, W]
        _, _, H, W = clip.shape

        # Calculate crop offsets
        h_offset = (H - crop_size) // 2
        w_offset = (W - crop_size) // 2

        # Apply crop
        cropped = clip[:, :, h_offset:h_offset + crop_size, w_offset:w_offset + crop_size]
        return cropped

    def corner_crop(self, clip, crop_size, position):
        """Apply corner crop to the clip."""
        # clip shape: [T, C, H, W]
        _, _, H, W = clip.shape

        # Calculate crop offsets
        if position == 'top-left':
            h_offset, w_offset = 0, 0
        elif position == 'top-right':
            h_offset, w_offset = 0, W - crop_size
        elif position == 'bottom-left':
            h_offset, w_offset = H - crop_size, 0
        elif position == 'bottom-right':
            h_offset, w_offset = H - crop_size, W - crop_size
        else:
            # Default to center crop if invalid position
            h_offset = (H - crop_size) // 2
            w_offset = (W - crop_size) // 2

        # Apply crop
        cropped = clip[:, :, h_offset:h_offset + crop_size, w_offset:w_offset + crop_size]
        return cropped


def collate_fn(batch):
    """
    Custom collate function to handle empty samples.
    """
    # Filter out empty samples
    batch = [item for item in batch if item['video_idx'] != -1]

    if not batch:
        # Return empty batch with same structure
        return {
            'video': torch.empty(0, 30, 3, 16, 224, 224),  # Typical shape for num_clips=10, num_crops=3
            'label': torch.empty(0, dtype=torch.long),
            'video_idx': torch.empty(0, dtype=torch.long)
        }

    # Get the number of views (clips * crops)
    num_views = batch[0]['video'].shape[0]

    # Stack videos along batch dimension
    videos = torch.stack([item['video'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return {
        'video': videos,
        'label': labels,
        'video_idx': torch.tensor([item['video_idx'] for item in batch])
    }


def debug_dataset(dataset):
    """Print debugging information about the dataset."""
    print("\n==== DATASET DEBUG INFO ====")
    print(f"Dataset class: {dataset.__class__.__name__}")
    print(f"Video root: {dataset.video_root}")
    print(f"Number of videos: {len(dataset.videos)}")

    if hasattr(dataset, 'videos') and dataset.videos:
        print("\nFirst few video entries:")
        for i, (video_id, video_path, label) in enumerate(dataset.videos[:3]):
            print(f"  {i + 1}. ID: {video_id}")
            print(f"     Path: {video_path}")
            print(f"     Exists: {os.path.exists(video_path)}")
            print(f"     Label: {label}")

        # Check label types
        label_types = set(type(label) for _, _, label in dataset.videos)
        print(f"\nLabel types in dataset: {label_types}")

        if str in label_types:
            print("WARNING: String labels detected! These should be converted to integers.")
            string_labels = [(i, label) for i, (_, _, label) in enumerate(dataset.videos) if isinstance(label, str)]
            print(f"First few string labels: {string_labels[:5]}")

    print("\nTrying to load the first video:")
    if len(dataset) > 0:
        try:
            sample = dataset[0]
            print(f"  Sample keys: {sample.keys()}")
            if 'video' in sample:
                print(f"  Video tensor shape: {sample['video'].shape}")
            if 'label' in sample:
                print(f"  Label: {sample['label']} (type: {type(sample['label'])})")
        except Exception as e:
            print(f"  Error loading first video: {e}")
            traceback.print_exc()
    else:
        print("  No videos to load!")

    print("============================\n")
    return dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Video Swin Transformer evaluation on Kinetics')

    # Required arguments
    parser.add_argument('--video_root', type=str, required=True,
                        help='Root directory of Kinetics video files')
    parser.add_argument('--annotation_file', type=str, required=True,
                        help='Path to annotation file (JSON or CSV)')
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='Path to pretrained model weights')

    # Model configuration
    parser.add_argument('--model_type', type=str, default='base',
                        choices=['tiny', 'small', 'base', 'large', 't', 's', 'b', 'l'],
                        help='Type of Video Swin Transformer model')
    parser.add_argument('--num_classes', type=int, default=600,
                        help='Number of classes in the dataset')
    parser.add_argument('--window_size', type=str, default='8,7,7',
                        help='Window size (temporal,height,width), e.g., "8,7,7"')

    # Dataset configuration
    parser.add_argument('--clip_len', type=int, default=16,
                        help='Number of frames in each clip')
    parser.add_argument('--frame_sample_rate', type=int, default=2,
                        help='Sampling rate for frames')
    parser.add_argument('--num_clips', type=int, default=30,
                        help='Number of clips to sample from each video')
    parser.add_argument('--num_crops', type=int, default=3, choices=[1, 3],
                        help='Number of crops per clip (1: center, 3: center + corners)')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='Size of spatial crops')
    parser.add_argument('--short_side_size', type=int, default=256,
                        help='Short side size for resize before cropping')

    # Evaluation configuration
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--amp', action='store_true',
                        help='Use Automatic Mixed Precision for inference')
    parser.add_argument('--no_amp', action='store_true',
                        help='Do not use Automatic Mixed Precision for inference')
    parser.add_argument('--output_file', type=str, default='kinetics_results.json',
                        help='Path to save evaluation results')
    parser.add_argument('--classnames_file', type=str, default=None,
                        help='Path to class names mapping file (JSON format)')

    # Additional options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=None,
                        help='Specific GPU to use (defaults to auto-select)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output')

    args = parser.parse_args()

    # Ensure AMP settings are consistent
    if args.no_amp:
        args.amp = False

    # Parse window size
    try:
        t, h, w = map(int, args.window_size.split(','))
        args.window_size = (t, h, w)
    except:
        print(f"Invalid window size format: {args.window_size}. Using default (8,7,7)")
        args.window_size = (8, 7, 7)

    return args


def main():
    """Main evaluation function."""
    # Parse command line arguments
    args = parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Set up device
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Print key settings
    print(f"\nEvaluation settings:")
    print(f"  Model type: {args.model_type}")
    print(f"  Window size: {args.window_size}")
    print(f"  Number of clips: {args.num_clips}")
    print(f"  Number of crops: {args.num_crops}")
    print(f"  Using AMP: {args.amp}")
    print(f"  Batch size: {args.batch_size}")

    # Load class names if file is provided
    class_names = load_classnames(args.classnames_file)
    print(f"Loaded {len(class_names)} class names")

    # Create dataset
    test_dataset = KineticsDataset(
        video_root=args.video_root,
        annotation_path=args.annotation_file,
        clip_len=args.clip_len,
        frame_sample_rate=args.frame_sample_rate,
        crop_size=args.crop_size,
        short_side_size=args.short_side_size,
        num_clips=args.num_clips,
        num_crops=args.num_crops,
        mode='test'
    )

    print(f"Dataset size: {len(test_dataset)} videos")

    # Debug dataset if enabled
    if args.debug:
        test_dataset = debug_dataset(test_dataset)

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # Load model
    model = load_model(
        model_type=args.model_type,
        pretrained_path=args.pretrained_path,
        num_classes=args.num_classes,
        device=device
    )

    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded with {num_params / 1e6:.2f} million parameters")

    # Create evaluator
    evaluator = VideoSwinEvaluator(
        model=model,
        dataloader=test_loader,
        use_amp=args.amp,
        device=device
    )

    # Run evaluation
    print("\nStarting evaluation...")
    metrics = evaluator.run()

    # Print results
    print("\nEvaluation results:")
    print(f"  Number of videos: {metrics['total_videos']}")
    print(f"  Top-1 accuracy: {metrics['top1_accuracy']:.4f}")
    print(f"  Top-5 accuracy: {metrics['top5_accuracy']:.4f}")
    print(f"  Evaluation time: {metrics['eval_time']:.2f} seconds")

    # Add class names to predictions for better interpretation
    for video_id, pred_info in metrics['predictions'].items():
        label_idx = pred_info['label']
        pred_top5 = pred_info['pred_top5']

        pred_info['label_name'] = class_names.get(label_idx, f"unknown_{label_idx}")
        pred_info['pred_top5_names'] = [class_names.get(idx, f"unknown_{idx}") for idx in pred_top5]

    # Save results to file
    output_data = {
        'config': vars(args),
        'metrics': {
            'top1_accuracy': float(metrics['top1_accuracy']),  # Convert tensor to float for JSON
            'top5_accuracy': float(metrics['top5_accuracy']),
            'total_videos': metrics['total_videos'],
            'eval_time': metrics['eval_time']
        },
        'predictions': metrics['predictions'],
        'class_names': class_names
    }

    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {args.output_file}")


if __name__ == '__main__':
    main()