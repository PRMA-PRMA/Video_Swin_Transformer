#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video Frame Extraction Utility
-----------------------------
A utility script to extract frames from video files for training and evaluation
with the Video Swin Transformer.

This script extracts frames from videos and organizes them in a directory structure
suitable for using with the provided dataset classes.
"""

import os
import sys
import argparse
import json
import shutil
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
import glob

import numpy as np

# Check for required libraries
try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not found. Install it with 'pip install opencv-python'")

try:
    import av

    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False
    print("Warning: PyAV not found. Install it with 'pip install av'")

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not found. Progress bars will be disabled.")


def extract_frames_cv2(video_path: str,
                       output_dir: str,
                       fps: Optional[float] = None,
                       max_frames: Optional[int] = None,
                       frame_format: str = 'frame_{:06d}.jpg',
                       resize: Optional[Tuple[int, int]] = None,
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None,
                       quality: int = 95) -> int:
    """
    Extract frames from a video using OpenCV.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save the frames
        fps: Target FPS (if None, uses the original video FPS)
        max_frames: Maximum number of frames to extract
        frame_format: Format string for the frame filenames
        resize: Tuple of (width, height) to resize frames to
        start_time: Start time in seconds
        end_time: End time in seconds
        quality: JPEG quality (0-100)

    Returns:
        Number of frames extracted
    """
    if not HAS_CV2:
        print(f"Error: OpenCV is not available. Cannot extract frames from {video_path}")
        return 0

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    # Determine the frame step (for FPS control)
    if fps is not None and fps < video_fps:
        frame_step = max(1, round(video_fps / fps))
    else:
        frame_step = 1

    # Set start and end frame indices
    start_frame = 0
    if start_time is not None:
        start_frame = int(start_time * video_fps)

    end_frame = total_frames
    if end_time is not None:
        end_frame = min(total_frames, int(end_time * video_fps))
    elif max_frames is not None:
        end_frame = min(total_frames, start_frame + max_frames * frame_step)

    # Set the initial position
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Prepare frame extraction
    frame_count = 0
    saved_count = 0

    # Extract frames
    while frame_count < end_frame - start_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frames according to the frame step
        if frame_count % frame_step == 0:
            # Resize if needed
            if resize:
                frame = cv2.resize(frame, resize)

            # Save the frame
            frame_path = os.path.join(output_dir, frame_format.format(saved_count + 1))
            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            saved_count += 1

            # Check if we've reached the maximum number of frames
            if max_frames is not None and saved_count >= max_frames:
                break

        frame_count += 1

    # Release resources
    cap.release()

    return saved_count


def extract_frames_pyav(video_path: str,
                        output_dir: str,
                        fps: Optional[float] = None,
                        max_frames: Optional[int] = None,
                        frame_format: str = 'frame_{:06d}.jpg',
                        resize: Optional[Tuple[int, int]] = None,
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None,
                        quality: int = 95) -> int:
    """
    Extract frames from a video using PyAV.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save the frames
        fps: Target FPS (if None, uses the original video FPS)
        max_frames: Maximum number of frames to extract
        frame_format: Format string for the frame filenames
        resize: Tuple of (width, height) to resize frames to
        start_time: Start time in seconds
        end_time: End time in seconds
        quality: JPEG quality (0-100)

    Returns:
        Number of frames extracted
    """
    if not HAS_PYAV:
        print(f"Error: PyAV is not available. Cannot extract frames from {video_path}")
        return 0

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Open the video file
        container = av.open(video_path)

        # Get the video stream
        stream = container.streams.video[0]

        # Get video properties
        video_fps = float(stream.average_rate)
        duration = stream.duration * stream.time_base
        if duration <= 0 and container.duration > 0:
            duration = float(container.duration) / av.time_base

        # Determine the frame step (for FPS control)
        if fps is not None and fps < video_fps:
            frame_step = max(1, round(video_fps / fps))
        else:
            frame_step = 1

        # Set start and end positions
        start_offset = 0
        if start_time is not None:
            start_offset = int(start_time / stream.time_base)
            container.seek(start_offset, stream=stream)

        end_offset = None
        if end_time is not None and duration > 0:
            end_offset = int(end_time / stream.time_base)

        # Prepare frame extraction
        frame_count = 0
        saved_count = 0

        # Extract frames
        for frame in container.decode(video=0):
            # Skip frames until the start time
            if start_time is not None and frame.pts * frame.time_base < start_time:
                continue

            # Stop at end time if specified
            if end_time is not None and frame.pts * frame.time_base > end_time:
                break

            # Process frames according to the frame step
            if frame_count % frame_step == 0:
                # Convert to ndarray
                img = frame.to_ndarray(format='rgb24')

                # Resize if needed
                if resize:
                    img = cv2.resize(img, resize)

                # Convert RGB to BGR for OpenCV
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Save the frame
                frame_path = os.path.join(output_dir, frame_format.format(saved_count + 1))
                cv2.imwrite(frame_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                saved_count += 1

                # Check if we've reached the maximum number of frames
                if max_frames is not None and saved_count >= max_frames:
                    break

            frame_count += 1

        # Close the container
        container.close()

        return saved_count

    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return 0


def extract_frames(video_path: str,
                   output_dir: str,
                   decoder: str = 'pyav',
                   **kwargs) -> int:
    """
    Extract frames from a video using the specified decoder.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save the frames
        decoder: Decoder to use ('pyav' or 'opencv')
        **kwargs: Additional arguments for the extraction function

    Returns:
        Number of frames extracted
    """
    if decoder == 'pyav' and HAS_PYAV:
        return extract_frames_pyav(video_path, output_dir, **kwargs)
    elif decoder == 'opencv' or (decoder == 'pyav' and not HAS_PYAV):
        return extract_frames_cv2(video_path, output_dir, **kwargs)
    else:
        print(f"Error: No suitable decoder found for {video_path}")
        return 0


def process_video(video_path: str,
                  output_root: str,
                  class_name: Optional[str] = None,
                  video_id: Optional[str] = None,
                  decoder: str = 'pyav',
                  **kwargs) -> Tuple[str, int]:
    """
    Process a single video file.

    Args:
        video_path: Path to the video file
        output_root: Root directory for saving frames
        class_name: Class name for organizing frames
        video_id: Video ID for naming the output directory
        decoder: Decoder to use ('pyav' or 'opencv')
        **kwargs: Additional arguments for the extraction function

    Returns:
        Tuple of (video_path, number of frames extracted)
    """
    try:
        # Determine output directory structure
        if class_name:
            # Use class-based directory structure
            frames_dir = os.path.join(output_root, 'frames', class_name)
        else:
            # Use flat directory structure
            frames_dir = os.path.join(output_root, 'frames')

        # Create output directory
        os.makedirs(frames_dir, exist_ok=True)

        # Determine video ID
        if video_id is None:
            # Use basename without extension
            video_id = os.path.splitext(os.path.basename(video_path))[0]

        # Create video directory
        video_dir = os.path.join(frames_dir, video_id)

        # Skip if already exists and not forced to overwrite
        if os.path.exists(video_dir) and not kwargs.get('overwrite', False):
            existing_frames = len(glob.glob(os.path.join(video_dir, '*.jpg')))
            print(f"Skipping {video_path} - output directory already exists with {existing_frames} frames")
            return video_path, existing_frames

        # Extract frames
        num_frames = extract_frames(
            video_path=video_path,
            output_dir=video_dir,
            decoder=decoder,
            **kwargs
        )

        return video_path, num_frames

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return video_path, 0


def process_video_file(args, video_file: str, class_info: Optional[Dict] = None):
    """
    Process a single video file based on command line arguments.

    Args:
        args: Command line arguments
        video_file: Path to video file
        class_info: Dictionary mapping video files to class names

    Returns:
        Tuple of (video_file, number of frames extracted)
    """
    # Determine class name
    class_name = None
    video_id = None

    if class_info is not None:
        # Try to get class info from the provided mapping
        basename = os.path.basename(video_file)
        if basename in class_info:
            class_name = class_info[basename]
        elif video_file in class_info:
            class_name = class_info[video_file]

    # If no class info found, try to infer from directory structure
    if class_name is None and args.extract_classes:
        # Extract class from directory name
        parent_dir = os.path.basename(os.path.dirname(video_file))
        if parent_dir and parent_dir != os.path.basename(args.input):
            class_name = parent_dir

    # Determine video ID
    if args.video_id_format == 'hash':
        # Use hash of video path
        import hashlib
        video_id = hashlib.md5(video_file.encode()).hexdigest()
    else:
        # Use basename without extension
        video_id = os.path.splitext(os.path.basename(video_file))[0]

    # Process the video
    return process_video(
        video_path=video_file,
        output_root=args.output,
        class_name=class_name,
        video_id=video_id,
        decoder=args.decoder,
        fps=args.fps,
        max_frames=args.max_frames,
        frame_format=args.frame_format,
        resize=(args.width, args.height) if args.width and args.height else None,
        start_time=args.start_time,
        end_time=args.end_time,
        quality=args.quality,
        overwrite=args.overwrite
    )


def process_dataset(args) -> List[Tuple[str, int]]:
    """
    Process a dataset of videos based on command line arguments.

    Args:
        args: Command line arguments

    Returns:
        List of (video_path, number of frames extracted) tuples
    """
    # Load class info if available
    class_info = None
    if args.class_file:
        try:
            with open(args.class_file, 'r') as f:
                data = json.load(f)

                # Convert to a mapping of video file to class name
                if isinstance(data, dict):
                    # If it's a direct mapping
                    class_info = data
                elif isinstance(data, list):
                    # If it's a list of objects
                    if all(isinstance(item, dict) for item in data):
                        # Try to find common field names
                        if all('video' in item and 'class' in item for item in data):
                            class_info = {item['video']: item['class'] for item in data}
                        elif all('id' in item and 'class_name' in item for item in data):
                            class_info = {item['id']: item['class_name'] for item in data}
                        elif all('filename' in item and 'label' in item for item in data):
                            class_info = {item['filename']: item['label'] for item in data}
        except Exception as e:
            print(f"Error loading class file: {e}")

    # Find all video files
    video_files = []

    if os.path.isfile(args.input):
        # Single video file
        video_files = [args.input]
    else:
        # Directory of videos
        for extension in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']:
            video_files.extend(glob.glob(os.path.join(args.input, '**', extension), recursive=True))

    # Sort files for consistent processing
    video_files.sort()

    if not video_files:
        print(f"No video files found in {args.input}")
        return []

    print(f"Found {len(video_files)} video files to process")

    # Process videos
    results = []

    if args.num_workers > 1:
        # Parallel processing
        process_func = partial(process_video_file, args, class_info=class_info)

        with Pool(args.num_workers) as pool:
            if HAS_TQDM:
                results = list(tqdm(pool.imap(process_func, video_files), total=len(video_files)))
            else:
                results = pool.map(process_func, video_files)
    else:
        # Sequential processing
        for video_file in (tqdm(video_files) if HAS_TQDM else video_files):
            result = process_video_file(args, video_file, class_info)
            results.append(result)

            # Print progress
            if not HAS_TQDM:
                print(f"Processed {video_file} - extracted {result[1]} frames")

    return results


def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos for Video Swin Transformer')

    # Input/output options
    parser.add_argument('--input', type=str, required=True,
                        help='Input video file or directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for frames')
    parser.add_argument('--class-file', type=str, default=None,
                        help='JSON file mapping video files to class names')
    parser.add_argument('--extract-classes', action='store_true',
                        help='Extract class names from directory structure')

    # Frame extraction options
    parser.add_argument('--decoder', type=str, default='pyav',
                        choices=['pyav', 'opencv'],
                        help='Video decoder to use')
    parser.add_argument('--fps', type=float, default=None,
                        help='Target FPS (if None, uses the original video FPS)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum number of frames to extract per video')
    parser.add_argument('--frame-format', type=str, default='frame_{:06d}.jpg',
                        help='Format string for the frame filenames')
    parser.add_argument('--width', type=int, default=None,
                        help='Width to resize frames to')
    parser.add_argument('--height', type=int, default=None,
                        help='Height to resize frames to')
    parser.add_argument('--start-time', type=float, default=None,
                        help='Start time in seconds')
    parser.add_argument('--end-time', type=float, default=None,
                        help='End time in seconds')
    parser.add_argument('--quality', type=int, default=95,
                        help='JPEG quality (0-100)')
    parser.add_argument('--video-id-format', type=str, default='basename',
                        choices=['basename', 'hash'],
                        help='Format to use for video IDs')

    # Processing options
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of worker processes')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output directories')

    args = parser.parse_args()

    # Check for required libraries
    if not HAS_CV2 and not HAS_PYAV:
        print("Error: Either OpenCV or PyAV is required for frame extraction")
        sys.exit(1)

    # Validate options
    if args.width and not args.height:
        print("Error: If --width is specified, --height must also be specified")
        sys.exit(1)
    if args.height and not args.width:
        print("Error: If --height is specified, --width must also be specified")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Start timing
    start_time = time.time()

    # Process videos
    results = process_dataset(args)

    # Calculate statistics
    total_videos = len(results)
    successful_videos = sum(1 for _, frames in results if frames > 0)
    total_frames = sum(frames for _, frames in results)

    # Print results
    print(f"\nFrame Extraction Summary:")
    print(f"  Total videos processed: {total_videos}")
    print(f"  Successfully extracted frames from: {successful_videos} videos")
    print(f"  Failed extractions: {total_videos - successful_videos} videos")
    print(f"  Total frames extracted: {total_frames}")
    print(f"  Time taken: {time.time() - start_time:.2f} seconds")

    # Save results to JSON
    results_file = os.path.join(args.output, 'extraction_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'total_videos': total_videos,
            'successful_videos': successful_videos,
            'total_frames': total_frames,
            'time_taken': time.time() - start_time,
            'videos': {os.path.basename(path): frames for path, frames in results}
        }, f, indent=2)

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()