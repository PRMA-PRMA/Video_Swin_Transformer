#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video Swin Transformer Inference Script
-------------------------------------
Run inference using trained VST models.
"""

import os
import argparse
import json
import time
from typing import List, Tuple, Dict, Optional, Union

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Import VST implementation
from vst import video_swin_t, video_swin_s, video_swin_b, video_swin_l, get_device
from vst_custom import create_custom_model

# Default ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(checkpoint_path: str, device: torch.device, num_classes: int = 400, model_name: str = 'tiny',
               use_custom_head: bool = False):
    """
    Load a trained VST model.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        num_classes: Number of classes
        model_name: Model size ('tiny', 'small', 'base', 'large')
        use_custom_head: Whether to use custom head model

    Returns:
        Loaded model
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    # Create model
    if use_custom_head:
        model = create_custom_model(
            model_name=model_name,
            num_classes=num_classes,
            device=device
        )
    else:
        if model_name == 'tiny':
            model = video_swin_t(num_classes=num_classes, device=device)
        elif model_name == 'small':
            model = video_swin_s(num_classes=num_classes, device=device)
        elif model_name == 'base':
            model = video_swin_b(num_classes=num_classes, device=device)
        elif model_name == 'large':
            model = video_swin_l(num_classes=num_classes, device=device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def load_labels(label_path: str) -> List[str]:
    """
    Load class labels.

    Args:
        label_path: Path to labels file

    Returns:
        List of class names
    """
    # Different formats supported
    if label_path.endswith('.json'):
        with open(label_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                if all(k.isdigit() for k in data.keys()):
                    # {0: 'class1', 1: 'class2', ...}
                    return [data[str(i)] for i in range(len(data))]
                else:
                    # {'class1': 0, 'class2': 1, ...}
                    labels = [''] * len(data)
                    for name, idx in data.items():
                        labels[int(idx)] = name
                    return labels
    elif label_path.endswith('.txt'):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            labels = []
            for line in lines:
                line = line.strip()
                if ' ' in line:
                    # Format: "index label" or "label index"
                    parts = line.split(' ', 1)
                    if parts[0].isdigit():
                        # "index label"
                        labels.append(parts[1])
                    else:
                        # "label index"
                        labels.append(parts[0])
                else:
                    # Just a label per line
                    labels.append(line)
            return labels
    else:
        # Unknown format
        print(f"Unsupported labels file format: {label_path}")
        return [f"Class {i}" for i in range(1000)]


def load_video_frames(video_path: str,
                      clip_length: int = 16,
                      frame_stride: int = 2) -> np.ndarray:
    """
    Load frames from a video file.

    Args:
        video_path: Path to video file
        clip_length: Number of frames to extract
        frame_stride: Stride between frames

    Returns:
        Numpy array of frames (T, H, W, C)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate sampling indices
    required_frames = clip_length * frame_stride

    if total_frames <= required_frames:
        # If video is too short, use all frames and loop if needed
        indices = [i % total_frames for i in range(0, required_frames, frame_stride)][:clip_length]
    else:
        # Center sampling
        start = max(0, (total_frames - required_frames) // 2)
        indices = list(range(start, start + required_frames, frame_stride))[:clip_length]

    # Load frames
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            print(f"Error reading frame {idx} from {video_path}")
            # Create blank frame
            if frames:
                frames.append(np.zeros_like(frames[0]))
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

    cap.release()

    # Stack frames
    return np.stack(frames)


def load_frame_directory(dir_path: str,
                         clip_length: int = 16,
                         frame_stride: int = 2) -> np.ndarray:
    """
    Load frames from a directory.

    Args:
        dir_path: Directory containing frame images
        clip_length: Number of frames to extract
        frame_stride: Stride between frames

    Returns:
        Numpy array of frames (T, H, W, C)
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    # Get frame files
    frame_files = sorted([f for f in os.listdir(dir_path)
                          if f.endswith(('.jpg', '.jpeg', '.png'))])

    # Calculate sampling indices
    total_frames = len(frame_files)
    required_frames = clip_length * frame_stride

    if total_frames <= required_frames:
        # If not enough frames, use all frames and loop if needed
        indices = [i % total_frames for i in range(0, required_frames, frame_stride)][:clip_length]
    else:
        # Center sampling
        start = max(0, (total_frames - required_frames) // 2)
        indices = list(range(start, start + required_frames, frame_stride))[:clip_length]

    # Load frames
    frames = []
    for idx in indices:
        if idx < len(frame_files):
            frame_path = os.path.join(dir_path, frame_files[idx])
            try:
                # Read image
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
            except Exception as e:
                print(f"Error reading frame {frame_path}: {e}")
                # Create blank frame
                if frames:
                    frames.append(np.zeros_like(frames[0]))
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

    # Ensure we have enough frames
    while len(frames) < clip_length:
        # Duplicate last frame if not enough
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

    # Stack frames
    return np.stack(frames)


def preprocess_frames(frames: np.ndarray, spatial_size: int = 224) -> torch.Tensor:
    """
    Preprocess frames for model input.

    Args:
        frames: Numpy array of frames (T, H, W, C)
        spatial_size: Size to resize frames to

    Returns:
        Tensor of processed frames (1, C, T, H, W)
    """
    # Convert to PIL images for transforms
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize(spatial_size + 32),
        transforms.CenterCrop(spatial_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    tensor_frames = [transform(img) for img in pil_frames]

    # Stack frames (T, C, H, W)
    clips = torch.stack(tensor_frames)

    # Reorder to (C, T, H, W) and add batch dimension
    clips = clips.permute(1, 0, 2, 3).unsqueeze(0)

    return clips


def get_predictions(model: torch.nn.Module,
                    clips: torch.Tensor,
                    labels: List[str],
                    topk: int = 5) -> List[Tuple[int, str, float]]:
    """
    Get model predictions for a clip.

    Args:
        model: Trained model
        clips: Processed clips tensor (B, C, T, H, W)
        labels: List of class names
        topk: Number of top predictions to return

    Returns:
        List of (class_id, class_name, score) tuples
    """
    with torch.no_grad():
        # Forward pass
        outputs = model(clips)

        # Apply softmax
        probs = F.softmax(outputs, dim=1)[0]

        # Get top-k predictions
        topk_probs, topk_idxs = torch.topk(probs, topk)

        # Create prediction list
        predictions = []
        for i in range(topk):
            idx = topk_idxs[i].item()
            prob = topk_probs[i].item()
            label = labels[idx] if idx < len(labels) else f"Class {idx}"
            predictions.append((idx, label, prob))

        return predictions


def visualize_predictions(frames: np.ndarray,
                          predictions: List[Tuple[int, str, float]],
                          output_path: Optional[str] = None):
    """
    Visualize predictions with a frame from the video.

    Args:
        frames: Numpy array of frames
        predictions: List of (class_id, class_name, score) tuples
        output_path: Path to save visualization, if None, show plot
    """
    # Create figure
    fig = plt.figure(figsize=(12, 5))

    # Show middle frame
    middle_idx = len(frames) // 2
    middle_frame = frames[middle_idx]

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(middle_frame)
    ax1.set_title('Input Video (Middle Frame)')
    ax1.axis('off')

    # Show predictions
    ax2 = fig.add_subplot(1, 2, 2)

    class_names = [pred[1] for pred in predictions]
    scores = [pred[2] for pred in predictions]

    y_pos = range(len(class_names))
    ax2.barh(y_pos, scores, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(class_names)
    ax2.set_xlabel('Probability')
    ax2.set_title('Top Predictions')

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def process_input(input_path: str,
                  clip_length: int = 16,
                  frame_stride: int = 2,
                  spatial_size: int = 224) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Process input video or frame directory.

    Args:
        input_path: Path to video file or directory of frames
        clip_length: Number of frames to extract
        frame_stride: Stride between frames
        spatial_size: Size to resize frames to

    Returns:
        Tuple of (original_frames, processed_clips)
    """
    # Check if input is a directory or file
    if os.path.isdir(input_path):
        # Load frames from directory
        frames = load_frame_directory(input_path, clip_length, frame_stride)
    else:
        # Load frames from video
        frames = load_video_frames(input_path, clip_length, frame_stride)

    # Preprocess for model
    clips = preprocess_frames(frames, spatial_size)

    return frames, clips


def main():
    parser = argparse.ArgumentParser(description='VST Inference')

    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='tiny',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=400,
                        help='Number of classes')
    parser.add_argument('--use-custom-head', action='store_true',
                        help='Use custom head model')

    # Input parameters
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input video or directory of frames')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to class labels file')
    parser.add_argument('--clip-length', type=int, default=16,
                        help='Number of frames per clip')
    parser.add_argument('--frame-stride', type=int, default=2,
                        help='Stride between sampled frames')
    parser.add_argument('--spatial-size', type=int, default=224,
                        help='Spatial size of frames')

    # Output parameters
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save visualization')
    parser.add_argument('--topk', type=int, default=5,
                        help='Number of top predictions to show')
    parser.add_argument('--json-output', type=str, default=None,
                        help='Path to save predictions as JSON')

    # Execution parameters
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run inference on')

    args = parser.parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    print(f"Running inference on {device}")

    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        num_classes=args.num_classes,
        model_name=args.model,
        use_custom_head=args.use_custom_head
    )

    # Load class labels
    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} class labels")

    # Process input
    print(f"Processing input: {args.input}")
    frames, clips = process_input(
        input_path=args.input,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        spatial_size=args.spatial_size
    )

    # Get predictions
    print("Running inference...")
    predictions = get_predictions(model, clips.to(device), labels, args.topk)

    # Print predictions
    print("\nTop predictions:")
    for i, (idx, name, score) in enumerate(predictions):
        print(f"{i + 1}. {name}: {score:.4f}")

    # Save predictions as JSON if requested
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump({
                'input': args.input,
                'predictions': [
                    {'class_id': idx, 'class_name': name, 'probability': score}
                    for idx, name, score in predictions
                ]
            }, f, indent=4)
        print(f"Saved predictions to {args.json_output}")

    # Visualize predictions
    if args.output or args.output is None:  # None means show plot
        visualize_predictions(frames, predictions, args.output)


if __name__ == '__main__':
    main()