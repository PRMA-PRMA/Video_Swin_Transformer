"""
Standalone dataset debug script.
This will test loading your annotation file directly.
"""

import os
import sys
import json
import torch
import cv2
import glob
from tqdm import tqdm

# Try to import decord, but don't require it
try:
    import decord
except ImportError:
    print("Warning: decord is not installed, falling back to OpenCV")
    decord = None


class SimpleKineticsDataset:
    """Simplified version of KineticsDataset for debugging."""

    def __init__(self, video_root, annotation_path, clip_len=16):
        self.video_root = video_root
        self.clip_len = clip_len
        self.videos = []

        print(f"Loading annotations from: {annotation_path}")
        print(f"Video root directory: {video_root}")

        # Load annotations
        if annotation_path.endswith('.json'):
            self._load_json_annotations(annotation_path)
        else:
            raise ValueError(f"Unsupported annotation format: {annotation_path}")

        print(f"Loaded {len(self.videos)} videos")

        # Check if videos exist
        existing = 0
        for _, video_path, _ in self.videos:
            if os.path.exists(video_path):
                existing += 1

        print(f"Found {existing} existing video files out of {len(self.videos)}")

    def _load_json_annotations(self, annotation_path):
        """Load annotations from a JSON file."""
        try:
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)

            print(f"Loaded {len(annotations)} entries from annotation file")

            # Show structure of first entry
            if annotations:
                first_key = next(iter(annotations))
                print(f"First entry structure: {annotations[first_key]}")

            # Process each entry
            for video_id, info in annotations.items():
                # Extract file path
                if 'file_path' in info:
                    file_path = info['file_path']
                else:
                    print(f"Warning: No file_path in entry for {video_id}")
                    continue

                # Extract label
                label = -1
                if 'annotations' in info and 'label' in info['annotations']:
                    label = info['annotations']['label']
                elif 'label' in info:
                    label = info['label']

                # Add to videos list
                self.videos.append((video_id, file_path, label))

        except Exception as e:
            print(f"Error loading annotations: {e}")
            import traceback
            traceback.print_exc()

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        """Try to load a single video to test."""
        video_id, video_path, label = self.videos[index]

        print(f"\nTrying to load video {index}: {video_path}")

        # Check if file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return None

        # Try to load video frames
        try:
            if decord is not None:
                print("Using decord to load video")
                video_frames = self._load_video_decord(video_path)
            else:
                print("Using OpenCV to load video")
                video_frames = self._load_video_cv2(video_path)

            if video_frames is None:
                print("Failed to load video frames")
                return None

            print(f"Successfully loaded video with {len(video_frames)} frames")
            if video_frames:
                print(f"First frame shape: {video_frames[0].shape}")

            return {
                'video_id': video_id,
                'frames': video_frames,
                'label': label
            }

        except Exception as e:
            print(f"Error loading video: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_video_decord(self, video_path):
        """Load video using Decord."""
        try:
            container = decord.VideoReader(video_path)
            total_frames = len(container)

            if total_frames <= 0:
                print("No frames found in video")
                return None

            # Sample frames
            indices = self._sample_frames(total_frames)
            frames = container.get_batch(indices).asnumpy()
            return frames

        except Exception as e:
            print(f"Decord error: {e}")
            return None

    def _load_video_cv2(self, video_path):
        """Load video using OpenCV."""
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print("Failed to open video with OpenCV")
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames <= 0:
                print("No frames detected in video")
                cap.release()
                return None

            # Sample frames
            indices = self._sample_frames(total_frames)
            frames = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

            cap.release()

            if not frames:
                print("No frames were successfully read")
                return None

            return frames

        except Exception as e:
            print(f"OpenCV error: {e}")
            return None

    def _sample_frames(self, total_frames):
        """Sample frames from video."""
        if total_frames <= self.clip_len:
            # If video is too short, loop frames
            indices = list(range(total_frames))
            while len(indices) < self.clip_len:
                indices = indices + indices
            return indices[:self.clip_len]
        else:
            # Uniformly sample frames
            return list(range(0, total_frames, total_frames // self.clip_len))[:self.clip_len]


def main():
    """Main function."""
    # Set paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    annotation_file = os.path.join(script_dir, "../data", "kinetics600", "annotations", "test_fixed.json")
    video_dir = os.path.join(script_dir, "../data", "kinetics600", "videos", "extracted")

    # Check file and directory existence
    if not os.path.exists(annotation_file):
        print(f"Error: Annotation file not found: {annotation_file}")
        return

    if not os.path.exists(video_dir):
        print(f"Error: Video directory not found: {video_dir}")
        return

    # Create dataset
    dataset = SimpleKineticsDataset(video_dir, annotation_file)

    # Try to load first few videos
    for i in range(min(3, len(dataset))):
        result = dataset[i]
        if result:
            print(f"Successfully loaded video {i}")
        else:
            print(f"Failed to load video {i}")

    # Check for video files directly
    print("\nDirectly checking video files in directory...")
    video_files = glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)
    print(f"Found {len(video_files)} MP4 files")

    if video_files:
        print("\nTrying to load a random video file directly...")
        video_path = video_files[0]
        print(f"Testing file: {video_path}")

        if decord is not None:
            try:
                container = decord.VideoReader(video_path)
                print(f"Decord: Video has {len(container)} frames")
                if len(container) > 0:
                    print("Video is readable!")
            except Exception as e:
                print(f"Decord error: {e}")

        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"OpenCV: Video has {frame_count} frames")
                ret, frame = cap.read()
                if ret:
                    print(f"Successfully read first frame with shape {frame.shape}")
                cap.release()
            else:
                print("OpenCV: Failed to open video")
        except Exception as e:
            print(f"OpenCV error: {e}")


if __name__ == "__main__":
    main()