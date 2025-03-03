"""
Example usage of Video Swin Transformer
--------------------------------------
This script demonstrates how to use the Video Swin Transformer for:
1. Basic classification
2. Feature extraction
3. Custom head adaptation
4. Video processing
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from typing import List, Tuple

# Import the Video Swin Transformer
from vst import video_swin_b, video_swin_t
from vst_custom import create_custom_model


def example_basic_classification():
    """Example 1: Basic classification with pretrained model."""
    print("=== Example 1: Basic Classification ===")

    # Create a model (specify the path to your pretrained weights)
    pretrained_path = "path/to/pretrained_weights.pth"  # Replace with your path

    # Skip using pretrained weights if file doesn't exist
    if not os.path.exists(pretrained_path):
        print(f"Pretrained weights not found at {pretrained_path}, using random initialization")
        pretrained_path = None

    # Create the model
    model = video_swin_b(pretrained=pretrained_path, num_classes=400)
    model.eval()  # Set to evaluation mode

    # Create dummy input
    video = torch.randn(1, 3, 16, 224, 224)

    # Forward pass
    with torch.no_grad():
        output = model(video)

    # Get top predictions
    probabilities = torch.nn.functional.softmax(output, dim=1)
    top_prob, top_class = torch.topk(probabilities, 5)

    print("Top 5 predictions:")
    for i in range(5):
        print(f"  Class {top_class[0][i].item()}: {top_prob[0][i].item() * 100:.2f}%")

    print()


def example_feature_extraction():
    """Example 2: Feature extraction for transfer learning."""
    print("=== Example 2: Feature Extraction ===")

    # Create a model in feature extraction mode
    model = create_custom_model(
        model_name="tiny",  # "tiny", "small", "base", or "large"
        feature_mode=True
    )
    model.eval()

    # Input video
    video = torch.randn(1, 3, 16, 224, 224)

    # Extract features from the final layer
    with torch.no_grad():
        features = model(video)

    print(f"Final layer features shape: {features.shape}")

    # Extract features from all layers
    with torch.no_grad():
        all_features = model.extract_features(video)

    print("Features from all layers:")
    for i, feat in enumerate(all_features):
        print(f"  Layer {i} shape: {feat.shape}")

    # Extract features from a specific layer with pooling
    with torch.no_grad():
        layer2_features = model.extract_features(video, layer_idx=2, pool=True)

    print(f"Pooled features from layer 2: {layer2_features.shape}")
    print()


def example_custom_head():
    """Example 3: Using a custom classification head."""
    print("=== Example 3: Custom Classification Head ===")

    # Create model with custom head configuration
    model = create_custom_model(
        model_name="tiny",
        num_classes=1000,
        dropout_rate=0.5,
        activation=nn.ReLU()
    )

    # Input video
    video = torch.randn(1, 3, 16, 224, 224)

    # Forward pass with the current head
    output = model(video)
    print(f"Output shape with 1000 classes: {output.shape}")

    # Reset the head for a different task (e.g., 10-class classification)
    model.reset_head(
        num_classes=10,
        dropout_rate=0.2,
        activation=nn.GELU()
    )

    # Forward pass with the new head
    output = model(video)
    print(f"Output shape with 10 classes: {output.shape}")

    # Toggle between classification and feature extraction
    model.set_feature_mode(True)
    features = model(video)
    print(f"Feature mode output shape: {features.shape}")

    # Back to classification mode
    model.set_feature_mode(False)
    output = model(video)
    print(f"Classification mode output shape: {output.shape}")
    print()


def load_video_frames(video_path: str, num_frames: int = 16,
                      height: int = 224, width: int = 224) -> torch.Tensor:
    """
    Load frames from a video file.

    Args:
        video_path: Path to the video file
        num_frames: Number of frames to extract
        height: Target frame height
        width: Target frame width

    Returns:
        Tensor of shape [C, T, H, W]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []

    # Calculate frame indices to sample
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))

        # Normalize pixel values to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames could be extracted from {video_path}")

    # Convert to tensor [T, H, W, C] -> [C, T, H, W]
    frames = np.stack(frames, axis=0)
    frames = np.transpose(frames, (3, 0, 1, 2))

    return torch.tensor(frames, dtype=torch.float32)


def create_dummy_video(output_path: str = "dummy_video.mp4",
                       num_frames: int = 16,
                       width: int = 224,
                       height: int = 224) -> str:
    """
    Create a dummy video file for testing.

    Args:
        output_path: Path to save the video
        num_frames: Number of frames
        width: Frame width
        height: Frame height

    Returns:
        Path to the created video
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    for i in range(num_frames):
        # Create a frame with a gradient and frame number
        frame = np.ones((height, width, 3), dtype=np.uint8) * (i * 15)

        # Add frame number text
        cv2.putText(frame, f"Frame {i + 1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"Created dummy video at {output_path}")
    return output_path


def example_video_processing():
    """Example 4: Processing a video."""
    print("=== Example 4: Video Processing ===")

    # Create a dummy video
    video_path = create_dummy_video()

    try:
        # Load the video
        video_tensor = load_video_frames(video_path)
        print(f"Loaded video tensor with shape {video_tensor.shape}")

        # Add batch dimension
        video_tensor = video_tensor.unsqueeze(0)  # [1, C, T, H, W]

        # Create a model for inference
        model = video_swin_t(num_classes=400)
        model.eval()

        # Forward pass
        with torch.no_grad():
            output = model(video_tensor)

        print(f"Model output shape: {output.shape}")

    finally:
        # Clean up
        if os.path.exists(video_path):
            os.remove(video_path)

    print()


if __name__ == "__main__":
    example_basic_classification()
    example_feature_extraction()
    example_custom_head()
    example_video_processing()