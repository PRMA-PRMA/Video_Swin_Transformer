# Video Swin Transformer

A PyTorch implementation of Video Swin Transformer that doesn't rely on mmcv or other problematic external dependencies.

## Overview

This repository contains a clean, self-contained implementation of the Video Swin Transformer architecture as described in ["Video Swin Transformer" (CVPR 2022)](https://arxiv.org/abs/2106.13230). The implementation supports loading pretrained weights from the original implementation and provides utilities for video understanding tasks.

Key features:
- No mmcv dependency
- Support for all model variants (Tiny, Small, Base, Large)
- Pretrained weight loading with automatic window size detection
- Feature extraction capabilities for transfer learning
- Customizable classification head
- Complete training and evaluation pipeline
- Support for popular video datasets (Kinetics, UCF101, HMDB51, Something-Something-V2)
- Inference and model export utilities

## Installation

```bash
git clone https://github.com/PRMA-PRMA/video-swin-transformer.git
cd video-swin-transformer
pip install -r requirements.txt
```

### Requirements

- Python 3.7+
- PyTorch 1.10+
- torchvision
- NumPy
- einops
- opencv-python (for video processing)
- matplotlib (for visualization)
- tqdm (for progress bars)
- tensorboard (for training visualization)
- av (optional, for better video loading)
- memory-profiler (optional, for memory usage testing)

## Basic Model Usage

### Basic Usage

```python
import torch
from vst import video_swin_t, video_swin_b

# Create a model (tiny or base)
model = video_swin_t(num_classes=400)  # Tiny model
# model = video_swin_b(num_classes=400)  # Base model

# For pretrained models, provide the path to the weights
model = video_swin_b(pretrained="path/to/swin_base_patch244_window877_kinetics600_22k.pth", num_classes=400)

# Forward pass
video = torch.randn(1, 3, 16, 224, 224)  # [batch, channels, frames, height, width]
output = model(video)
print(f"Output shape: {output.shape}")  # [1, 400]
```

### Advanced Usage: Feature Extraction

The custom head extension provides more flexibility for transfer learning or fusion tasks:

```python
from vst_custom import create_custom_model

# Create a model in feature extraction mode
model = create_custom_model(
    model_name="base",  # "tiny", "small", "base", or "large"
    pretrained="path/to/pretrained_weights.pth",
    feature_mode=True
)

# Extract features
video = torch.randn(1, 3, 16, 224, 224)
features = model(video)
print(f"Feature shape: {features.shape}")  # [1, C, T, H, W]

# Extract features from specific layers
features = model.extract_features(video, layer_idx=2)
all_layers = model.extract_features(video)  # List of features from all layers
```

### Custom Classification Head

You can create a model with a custom classification head or modify it for different tasks:

```python
from vst_custom import create_custom_model
import torch.nn as nn

# Create model with dropout in classification head
model = create_custom_model(
    model_name="base",
    pretrained="path/to/weights.pth",
    num_classes=1000,
    dropout_rate=0.5,
    activation=nn.ReLU()
)

# Reset the head for a new task
model.reset_head(
    num_classes=10,
    dropout_rate=0.2,
    activation=nn.GELU()
)

# Freeze backbone for fine-tuning
model.freeze_backbone(freeze=True)
```

## Training Pipeline

The repository includes a complete training pipeline for Video Swin Transformer models.

### Dataset Preparation

The training code expects video frames to be extracted and organized in the following structure:

```
data/
├── dataset_name/
│   ├── frames/
│   │   ├── class1/
│   │   │   ├── video1/
│   │   │   │   ├── frame000001.jpg
│   │   │   │   ├── frame000002.jpg
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   └── annotations/
│       ├── train.txt
│       ├── val.txt
│       └── classInd.txt
```

For frame extraction, use the provided `extract_frames.py` script:

```bash
python extract_frames.py --input /path/to/videos --output /path/to/dataset/frames --extract-classes
```

### Training

Train a VST model from scratch or fine-tune a pre-trained model:

```bash
# Basic training
python train.py \
    --model tiny \
    --dataset ucf101 \
    --data-dir /path/to/dataset \
    --num-classes 101 \
    --batch-size 16 \
    --epochs 50 \
    --output-dir output/vst_tiny_ucf101

# Fine-tuning
python train.py \
    --model base \
    --dataset kinetics \
    --data-dir /path/to/dataset \
    --num-classes 400 \
    --pretrained \
    --pretrained-path /path/to/pretrained.pth \
    --finetune \
    --freeze-backbone \
    --lr 5e-5 \
    --batch-size 8 \
    --epochs 30 \
    --output-dir output/vst_base_kinetics_ft

# Advanced training with custom configuration
python train.py --config configs/ucf101_tiny.yaml
```

### Evaluation

Evaluate a trained model on a test set:

```bash
python evaluate.py \
    --checkpoint output/vst_tiny_ucf101/model_best.pth \
    --dataset ucf101 \
    --data-dir /path/to/dataset \
    --batch-size 16
```

### Inference

Run inference on a video or directory of frames:

```bash
python inference.py \
    --checkpoint output/vst_tiny_ucf101/model_best.pth \
    --model tiny \
    --num-classes 101 \
    --input path/to/video.mp4 \
    --labels path/to/labels.json \
    --output prediction.png
```

### Model Export

Export a trained model to TorchScript or ONNX format for deployment:

```bash
python export.py \
    --checkpoint output/model_best.pth \
    --model tiny \
    --num-classes 101 \
    --format all \
    --output-dir exported_models
```

## Supported Datasets

The training pipeline supports the following datasets:
- Kinetics (400/600)
- UCF101
- HMDB51
- Something-Something-V2

Custom datasets can be implemented by extending the base dataset class in `utils/datasets.py`.

## Model Variants

| Model  | Params | Embed Dim | Depths          | Heads           |
|--------|--------|-----------|-----------------|-----------------|
| Tiny   | 28M    | 96        | [2, 2, 6, 2]    | [3, 6, 12, 24]  |
| Small  | 50M    | 96        | [2, 2, 18, 2]   | [3, 6, 12, 24]  |
| Base   | 88M    | 128       | [2, 2, 18, 2]   | [4, 8, 16, 32]  |
| Large  | 197M   | 192       | [2, 2, 18, 2]   | [6, 12, 24, 48] |

## Pretrained Models

This implementation supports loading official pretrained weights from the original implementation. The weights can be downloaded from:

- [Kinetics-400 pretrained models](https://github.com/SwinTransformer/Video-Swin-Transformer#main-results-on-kinetics-400)
- [Kinetics-600 pretrained models](https://github.com/SwinTransformer/Video-Swin-Transformer#main-results-on-kinetics-600)

## Window Size Configuration

The window size is automatically detected from the pretrained weights filename:

- Format: `swin_{size}_patch244_window{D}{H}{W}_{dataset}_{version}.pth`
- Example: `swin_base_patch244_window877_kinetics600_22k.pth` has window size (8, 7, 7)

## Running Tests

To test the implementation, run:

```bash
python test_vst.py
```

For memory profiling, run:

```bash
python -m memory_profiler test_vst.py
```

## Acknowledgments

This implementation is based on the [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) paper by Ze Liu et al.

## License

Apache License 2.0