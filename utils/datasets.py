"""
Video Dataset Implementations
----------------------------
Dataset implementations for training and evaluating Video Swin Transformer models.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

# Try to import video processing libraries
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import av

    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False

# Default normalization parameters (ImageNet stats)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VideoFrameDataset(Dataset):
    """
    Base class for video datasets that store frames as individual image files.
    """

    def __init__(self,
                 root_dir: str,
                 annotation_file: Optional[str] = None,
                 split: str = 'train',
                 clip_length: int = 16,
                 frame_stride: int = 2,
                 temporal_sampling: str = 'random',
                 spatial_sampling: str = 'random',
                 spatial_size: int = 224,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Initialize the dataset.

        Args:
            root_dir: Root directory containing video frames
            annotation_file: Path to annotation file
            split: 'train' or 'val'
            clip_length: Number of frames in a clip
            frame_stride: Stride between frames
            temporal_sampling: How to sample frames ('random', 'center', 'uniform')
            spatial_sampling: How to crop frames ('random', 'center')
            spatial_size: Size of the spatial crop
            transform: Transform to apply to the clip
            target_transform: Transform to apply to the target
        """
        self.root_dir = root_dir
        self.annotation_file = annotation_file
        self.split = split
        self.clip_length = clip_length
        self.frame_stride = frame_stride
        self.temporal_sampling = temporal_sampling
        self.spatial_sampling = spatial_sampling
        self.spatial_size = spatial_size
        self.transform = transform
        self.target_transform = target_transform

        # Load annotations
        self.samples = self._load_annotations()

        # Get class information
        self.class_names = sorted(list(set(sample['class_name'] for sample in self.samples)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

        # Create default transforms if none provided
        if self.transform is None:
            self.transform = self._get_default_transform()

        print(f"Loaded {len(self.samples)} samples from {len(self.class_names)} classes")

    def _load_annotations(self) -> List[Dict]:
        """
        Load annotations from file or directory structure.

        Returns:
            List of sample dictionaries
        """
        samples = []

        # Implement in subclasses
        return samples

    def _get_default_transform(self) -> Callable:
        """
        Get default transform based on split.

        Returns:
            Transform function
        """
        if self.split == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(self.spatial_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.spatial_size + 32),
                transforms.CenterCrop(self.spatial_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])

    def _sample_frames(self, frame_paths: List[str]) -> List[str]:
        """
        Sample frames from a video according to the sampling strategy.

        Args:
            frame_paths: List of all frame paths for a video

        Returns:
            List of sampled frame paths
        """
        num_frames = len(frame_paths)
        required_frames = self.clip_length * self.frame_stride

        # Handle videos with fewer frames than required
        if num_frames < required_frames:
            # Repeat frames if necessary
            indices = [min(i * self.frame_stride % num_frames, num_frames - 1)
                       for i in range(self.clip_length)]
        else:
            # Apply temporal sampling strategy
            if self.temporal_sampling == 'center':
                # Center sampling
                start = max(0, (num_frames - required_frames) // 2)
                indices = list(range(start, start + required_frames, self.frame_stride))
            elif self.temporal_sampling == 'uniform':
                # Uniform sampling
                indices = np.linspace(0, num_frames - 1, self.clip_length, dtype=int)
            elif self.temporal_sampling == 'random' and self.split == 'train':
                # Random sampling for training
                max_start = max(0, num_frames - required_frames)
                start = random.randint(0, max_start)
                indices = list(range(start, start + required_frames, self.frame_stride))
            else:
                # Default to center sampling
                start = max(0, (num_frames - required_frames) // 2)
                indices = list(range(start, start + required_frames, self.frame_stride))

            # Ensure we don't exceed the number of frames
            indices = [min(i, num_frames - 1) for i in indices[:self.clip_length]]

        # Return selected frame paths
        return [frame_paths[i] for i in indices]

    def _load_frame(self, frame_path: str) -> Image.Image:
        """
        Load a frame from disk.

        Args:
            frame_path: Path to frame image

        Returns:
            PIL Image
        """
        return Image.open(frame_path).convert('RGB')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a video clip and its label.

        Args:
            idx: Sample index

        Returns:
            Tuple of (clip, label)
        """
        sample = self.samples[idx]
        video_path = sample['video_path']
        class_name = sample['class_name']

        # Get frame paths
        if os.path.isdir(video_path):
            # Directory of frames
            frame_files = sorted([f for f in os.listdir(video_path)
                                  if f.endswith(('.jpg', '.jpeg', '.png'))])
            frame_paths = [os.path.join(video_path, f) for f in frame_files]
        else:
            # Handle single file paths or patterns
            # Subclasses may override this for more complex cases
            raise ValueError(f"Expected directory of frames, got file: {video_path}")

        # Sample frames
        sampled_frame_paths = self._sample_frames(frame_paths)

        # Load frames
        frames = []
        for path in sampled_frame_paths:
            try:
                img = self._load_frame(path)
                frames.append(img)
            except Exception as e:
                print(f"Error loading frame {path}: {e}")
                # Create a blank frame on error
                img = Image.new('RGB', (self.spatial_size, self.spatial_size), color='black')
                frames.append(img)

        # Apply transforms to each frame
        if self.transform:
            # For training, apply different random transformations to maintain consistency
            if self.split == 'train' and self.spatial_sampling == 'random':
                # Apply same random crop and flip to all frames
                i, j, h, w = transforms.RandomResizedCrop.get_params(
                    frames[0], scale=(0.08, 1.0), ratio=(0.75, 1.3333))
                flip = random.random() < 0.5

                transformed_frames = []
                for img in frames:
                    # Crop and resize
                    img = transforms.functional.resized_crop(
                        img, i, j, h, w, (self.spatial_size, self.spatial_size))

                    # Flip if needed
                    if flip:
                        img = transforms.functional.hflip(img)

                    # Color jitter (different for each frame)
                    if 'ColorJitter' in str(self.transform):
                        img = transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4)(img)

                    # Convert to tensor and normalize
                    img = transforms.functional.to_tensor(img)
                    img = transforms.functional.normalize(
                        img, mean=IMAGENET_MEAN, std=IMAGENET_STD)

                    transformed_frames.append(img)
            else:
                # Apply the same transform to all frames for validation
                transformed_frames = [self.transform(img) for img in frames]
        else:
            # Convert frames to tensors
            transformed_frames = [transforms.ToTensor()(img) for img in frames]

        # Stack frames to create clip (T, C, H, W)
        clip = torch.stack(transformed_frames)

        # Convert to (C, T, H, W) format expected by the model
        clip = clip.permute(1, 0, 2, 3)

        # Get label
        label = self.class_to_idx[class_name]
        if self.target_transform:
            label = self.target_transform(label)

        return clip, label


class UCF101Dataset(VideoFrameDataset):
    """Dataset for UCF101."""

    def _load_annotations(self) -> List[Dict]:
        """Load UCF101 annotations."""
        samples = []

        # Get class mapping if available
        class_file = None
        if self.annotation_file:
            class_file = os.path.join(os.path.dirname(self.annotation_file), 'classInd.txt')

        class_mapping = {}
        if class_file and os.path.exists(class_file):
            with open(class_file, 'r') as f:
                for line in f:
                    class_id, class_name = line.strip().split(' ')
                    class_mapping[class_name] = int(class_id) - 1  # 0-indexed

        # Load split file if provided
        if self.annotation_file and os.path.exists(self.annotation_file):
            with open(self.annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    video_path = parts[0]
                    class_name = video_path.split('/')[0]

                    # Extract video name without extension
                    video_name = os.path.splitext(os.path.basename(video_path))[0]

                    # Find the frames directory
                    frames_dir = os.path.join(self.root_dir, 'frames', class_name, video_name)
                    if not os.path.isdir(frames_dir):
                        # Try alternate naming
                        frames_dir = os.path.join(self.root_dir, 'frames', class_name, video_name.replace('.', '_'))

                    if os.path.isdir(frames_dir) and len(os.listdir(frames_dir)) > 0:
                        samples.append({
                            'video_path': frames_dir,
                            'class_name': class_name
                        })
        else:
            # If no annotation file, scan the directory structure
            frames_dir = os.path.join(self.root_dir, 'frames')
            if os.path.isdir(frames_dir):
                for class_name in sorted(os.listdir(frames_dir)):
                    class_dir = os.path.join(frames_dir, class_name)
                    if os.path.isdir(class_dir):
                        for video_name in sorted(os.listdir(class_dir)):
                            video_dir = os.path.join(class_dir, video_name)
                            if os.path.isdir(video_dir) and len(os.listdir(video_dir)) > 0:
                                samples.append({
                                    'video_path': video_dir,
                                    'class_name': class_name
                                })

        # Split into train/val if needed
        if self.split == 'train' or self.split == 'val':
            # If no explicit split file, use a random split
            if not self.annotation_file:
                # Group by class
                classes = {}
                for sample in samples:
                    class_name = sample['class_name']
                    if class_name not in classes:
                        classes[class_name] = []
                    classes[class_name].append(sample)

                # Split each class (80% train, 20% val)
                train_samples = []
                val_samples = []
                for class_name, class_samples in classes.items():
                    n_train = int(len(class_samples) * 0.8)
                    train_samples.extend(class_samples[:n_train])
                    val_samples.extend(class_samples[n_train:])

                # Return appropriate split
                if self.split == 'train':
                    samples = train_samples
                else:
                    samples = val_samples

        return samples


class HMDB51Dataset(VideoFrameDataset):
    """Dataset for HMDB51."""

    def _load_annotations(self) -> List[Dict]:
        """Load HMDB51 annotations."""
        samples = []

        # Get all classes from directory structure
        frames_dir = os.path.join(self.root_dir, 'frames')
        class_names = []
        if os.path.isdir(frames_dir):
            class_names = sorted([d for d in os.listdir(frames_dir)
                                  if os.path.isdir(os.path.join(frames_dir, d))])

        # Load split file if provided
        split_id = 1  # Default split
        if self.annotation_file and os.path.exists(self.annotation_file):
            # Parse split files (format: video_name.avi split_id)
            with open(self.annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    if len(parts) >= 2:
                        video_name = parts[0]  # video_name.avi
                        video_split = int(parts[1])  # 1: train, 2: test, 0: unused

                        # Skip if not in the required split
                        if (self.split == 'train' and video_split != 1) or \
                                (self.split == 'val' and video_split != 2):
                            continue

                        # Extract class name from annotation file name
                        file_basename = os.path.basename(self.annotation_file)
                        class_name = file_basename.split('_test_')[0]

                        # Find the frames directory
                        video_name_no_ext = os.path.splitext(video_name)[0]
                        frames_dir = os.path.join(self.root_dir, 'frames', class_name, video_name_no_ext)

                        if os.path.isdir(frames_dir) and len(os.listdir(frames_dir)) > 0:
                            samples.append({
                                'video_path': frames_dir,
                                'class_name': class_name
                            })
        else:
            # If no annotation file, scan the directory structure
            for class_name in class_names:
                class_dir = os.path.join(frames_dir, class_name)
                for video_name in sorted(os.listdir(class_dir)):
                    video_dir = os.path.join(class_dir, video_name)
                    if os.path.isdir(video_dir) and len(os.listdir(video_dir)) > 0:
                        samples.append({
                            'video_path': video_dir,
                            'class_name': class_name
                        })

            # Split into train/val if needed
            if self.split == 'train' or self.split == 'val':
                # Group by class
                classes = {}
                for sample in samples:
                    class_name = sample['class_name']
                    if class_name not in classes:
                        classes[class_name] = []
                    classes[class_name].append(sample)

                # Split each class (80% train, 20% val)
                train_samples = []
                val_samples = []
                for class_name, class_samples in classes.items():
                    n_train = int(len(class_samples) * 0.8)
                    train_samples.extend(class_samples[:n_train])
                    val_samples.extend(class_samples[n_train:])

                # Return appropriate split
                if self.split == 'train':
                    samples = train_samples
                else:
                    samples = val_samples

        return samples


class KineticsDataset(VideoFrameDataset):
    """Dataset for Kinetics."""

    def _load_annotations(self) -> List[Dict]:
        """Load Kinetics annotations."""
        samples = []

        # Load annotation file (CSV format)
        if self.annotation_file and os.path.exists(self.annotation_file):
            import csv
            with open(self.annotation_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Expected format: youtube_id, time_start, time_end, label, split
                    youtube_id = row.get('youtube_id', '')
                    time_start = row.get('time_start', '0')
                    time_end = row.get('time_end', '0')
                    class_name = row.get('label', '')
                    split = row.get('split', '')

                    # Skip if not in the required split
                    if split and self.split != split:
                        continue

                    # Create video identifier
                    video_id = f"{youtube_id}_{time_start}_{time_end}"

                    # Look for frames directory
                    frames_dir = os.path.join(self.root_dir, 'frames', class_name, video_id)
                    if os.path.isdir(frames_dir) and len(os.listdir(frames_dir)) > 0:
                        samples.append({
                            'video_path': frames_dir,
                            'class_name': class_name
                        })
        else:
            # If no annotation file, scan the directory structure
            frames_dir = os.path.join(self.root_dir, 'frames')
            if os.path.isdir(frames_dir):
                for class_name in sorted(os.listdir(frames_dir)):
                    class_dir = os.path.join(frames_dir, class_name)
                    if os.path.isdir(class_dir):
                        for video_id in sorted(os.listdir(class_dir)):
                            video_dir = os.path.join(class_dir, video_id)
                            if os.path.isdir(video_dir) and len(os.listdir(video_dir)) > 0:
                                samples.append({
                                    'video_path': video_dir,
                                    'class_name': class_name
                                })

            # Split into train/val if needed
            if self.split == 'train' or self.split == 'val':
                # Random split (90% train, 10% val)
                random.shuffle(samples)
                split_idx = int(len(samples) * 0.9)
                if self.split == 'train':
                    samples = samples[:split_idx]
                else:
                    samples = samples[split_idx:]

        return samples


class SSv2Dataset(VideoFrameDataset):
    """Dataset for Something-Something-V2."""

    def _load_annotations(self) -> List[Dict]:
        """Load Something-Something-V2 annotations."""
        samples = []

        # Load annotation file (JSON format)
        if self.annotation_file and os.path.exists(self.annotation_file):
            with open(self.annotation_file, 'r') as f:
                data = json.load(f)

            for item in data:
                # Expected format: {"id": "123", "template": "Doing something", "label": "1"}
                video_id = item.get('id', '')
                class_name = item.get('template', '')

                # Look for frames directory
                frames_dir = os.path.join(self.root_dir, 'frames', video_id)
                if os.path.isdir(frames_dir) and len(os.listdir(frames_dir)) > 0:
                    samples.append({
                        'video_path': frames_dir,
                        'class_name': class_name
                    })
        else:
            # If no annotation file, scan the directory structure
            frames_dir = os.path.join(self.root_dir, 'frames')
            if os.path.isdir(frames_dir):
                # Something-Something has a flat structure with no class directories
                # We'll need a label mapping
                label_file = os.path.join(self.root_dir, 'labels.json')
                label_map = {}

                if os.path.exists(label_file):
                    with open(label_file, 'r') as f:
                        label_data = json.load(f)
                        for item in label_data:
                            video_id = item.get('id', '')
                            class_name = item.get('template', '')
                            label_map[video_id] = class_name

                # Scan frame directories
                for video_id in sorted(os.listdir(frames_dir)):
                    video_dir = os.path.join(frames_dir, video_id)
                    if os.path.isdir(video_dir) and len(os.listdir(video_dir)) > 0:
                        class_name = label_map.get(video_id, 'unknown')
                        samples.append({
                            'video_path': video_dir,
                            'class_name': class_name
                        })

            # Split into train/val if needed
            if self.split == 'train' or self.split == 'val':
                # Random split (90% train, 10% val)
                random.shuffle(samples)
                split_idx = int(len(samples) * 0.9)
                if self.split == 'train':
                    samples = samples[:split_idx]
                else:
                    samples = samples[split_idx:]

        return samples


def get_video_datasets(dataset_name: str,
                       data_dir: str,
                       clip_length: int = 16,
                       frame_stride: int = 2,
                       spatial_size: int = 224) -> Tuple[Dataset, Dataset]:
    """
    Create train and validation datasets.

    Args:
        dataset_name: Name of the dataset ('kinetics', 'ucf101', 'hmdb51', 'ssv2')
        data_dir: Root directory for the dataset
        clip_length: Number of frames in a clip
        frame_stride: Stride between frames
        spatial_size: Size of the spatial crop

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Common transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(spatial_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(spatial_size + 32),
        transforms.CenterCrop(spatial_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Create appropriate dataset based on name
    if dataset_name.lower() == 'ucf101':
        train_file = os.path.join(data_dir, 'annotations', 'ucf101_01_train.txt')
        val_file = os.path.join(data_dir, 'annotations', 'ucf101_01_test.txt')

        train_dataset = UCF101Dataset(
            root_dir=data_dir,
            annotation_file=train_file if os.path.exists(train_file) else None,
            split='train',
            clip_length=clip_length,
            frame_stride=frame_stride,
            spatial_size=spatial_size,
            transform=train_transform
        )

        val_dataset = UCF101Dataset(
            root_dir=data_dir,
            annotation_file=val_file if os.path.exists(val_file) else None,
            split='val',
            clip_length=clip_length,
            frame_stride=frame_stride,
            spatial_size=spatial_size,
            transform=val_transform
        )

    elif dataset_name.lower() == 'hmdb51':
        # Find split files
        split_dir = os.path.join(data_dir, 'annotations', 'splits')
        train_files = []
        test_files = []

        if os.path.exists(split_dir):
            for f in os.listdir(split_dir):
                if f.endswith('_test_split1.txt'):
                    test_files.append(os.path.join(split_dir, f))
                elif f.endswith('_train_split1.txt'):
                    train_files.append(os.path.join(split_dir, f))

        train_dataset = HMDB51Dataset(
            root_dir=data_dir,
            annotation_file=train_files[0] if train_files else None,
            split='train',
            clip_length=clip_length,
            frame_stride=frame_stride,
            spatial_size=spatial_size,
            transform=train_transform
        )

        val_dataset = HMDB51Dataset(
            root_dir=data_dir,
            annotation_file=test_files[0] if test_files else None,
            split='val',
            clip_length=clip_length,
            frame_stride=frame_stride,
            spatial_size=spatial_size,
            transform=val_transform
        )

    elif dataset_name.lower() == 'kinetics':
        train_file = os.path.join(data_dir, 'annotations', 'train.csv')
        val_file = os.path.join(data_dir, 'annotations', 'val.csv')

        train_dataset = KineticsDataset(
            root_dir=data_dir,
            annotation_file=train_file if os.path.exists(train_file) else None,
            split='train',
            clip_length=clip_length,
            frame_stride=frame_stride,
            spatial_size=spatial_size,
            transform=train_transform
        )

        val_dataset = KineticsDataset(
            root_dir=data_dir,
            annotation_file=val_file if os.path.exists(val_file) else None,
            split='val',
            clip_length=clip_length,
            frame_stride=frame_stride,
            spatial_size=spatial_size,
            transform=val_transform
        )

    elif dataset_name.lower() in ['ssv2', 'something-something-v2']:
        train_file = os.path.join(data_dir, 'annotations', 'something-something-v2-train.json')
        val_file = os.path.join(data_dir, 'annotations', 'something-something-v2-validation.json')

        # Create special transforms for Something-Something
        # The dataset requires temporal consistency, so we need to be careful with augmentations
        ssv2_train_transform = transforms.Compose([
            transforms.Resize((spatial_size + 32, spatial_size + 32)),
            transforms.RandomCrop(spatial_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        train_dataset = SSv2Dataset(
            root_dir=data_dir,
            annotation_file=train_file if os.path.exists(train_file) else None,
            split='train',
            clip_length=clip_length,
            frame_stride=frame_stride,
            spatial_size=spatial_size,
            transform=ssv2_train_transform
        )

        val_dataset = SSv2Dataset(
            root_dir=data_dir,
            annotation_file=val_file if os.path.exists(val_file) else None,
            split='val',
            clip_length=clip_length,
            frame_stride=frame_stride,
            spatial_size=spatial_size,
            transform=val_transform
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_dataset, val_dataset


# Create a simple init file for the utils package
def create_init_file():
    """Create an __init__.py file for the utils package."""
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    init_file = os.path.join(utils_dir, '__init__.py')

    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write('"""Utils package for Video Swin Transformer."""\n')


if __name__ == '__main__':
    # Create __init__.py when this file is run directly
    create_init_file()