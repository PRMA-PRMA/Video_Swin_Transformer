"""
Prepare Kinetics 600 dataset for evaluation using Amazon S3
----------------------------------------------------------
This script downloads and organizes the Kinetics 600 dataset from the Amazon S3 bucket
for testing the Video Swin Transformer implementation.

Features:
1. Download only annotation files (small)
2. Download a subset of test videos (to save space)
3. Filter annotations to match available videos
4. FIXED: Uses numeric class indices for correct evaluation
"""

import os
import sys
import json
import csv
import argparse
import requests
from tqdm import tqdm
import subprocess
import hashlib
import shutil
import glob
import tarfile
import random

# Base URLs for Kinetics 600 on Amazon S3
S3_BASE_URL = "https://s3.amazonaws.com/kinetics"
ANNOTATIONS_PATH = "600/annotations"
TEST_VIDEOS_PATH = "600/test"

# Alternative URLs for Kinetics 600 class list
KINETICS_600_CLASSES_URLS = [
    "https://raw.githubusercontent.com/open-mmlab/mmaction2/master/tools/data/kinetics/kinetics600_categories.txt",
    "https://raw.githubusercontent.com/open-mmlab/mmaction2/main/tools/data/kinetics/kinetics600_categories.txt",
    "https://raw.githubusercontent.com/silence9/mmaciton2/master/tols/data/kinetics/kinetics600_categories.txt"
]


def download_file(url, output_path, chunk_size=8192, allow_fail=False):
    """
    Download a file from a URL with progress bar.

    Args:
        url (str): URL to download from
        output_path (str): Path to save the file
        chunk_size (int): Size of chunks to download
        allow_fail (bool): If True, return None on failure instead of raising exception

    Returns:
        str: Path to the downloaded file or None if failed and allow_fail=True
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if file already exists
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return output_path

    print(f"Downloading {url} to {output_path}")

    try:
        # Stream download with progress bar
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Will raise an exception for 404s, 500s

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f, tqdm(
                desc=os.path.basename(output_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size):
                size = f.write(chunk)
                bar.update(size)

        return output_path
    except Exception as e:
        if allow_fail:
            print(f"Error downloading {url}: {e}")
            # Delete partial file if it exists
            if os.path.exists(output_path):
                os.remove(output_path)
            return None
        else:
            raise e


def download_annotation(subset, output_dir):
    """
    Download annotation file for the specified subset.

    Args:
        subset (str): Dataset subset (train, val, test)
        output_dir (str): Directory to save annotations

    Returns:
        str: Path to the downloaded annotation file
    """
    annotation_url = f"{S3_BASE_URL}/{ANNOTATIONS_PATH}/{subset}.csv"
    output_path = os.path.join(output_dir, "annotations", f"{subset}.csv")

    return download_file(annotation_url, output_path)


def try_download_official_class_list(output_dir):
    """
    Try to download official Kinetics 600 class list from multiple URLs.

    Args:
        output_dir (str): Directory to save the class list

    Returns:
        str: Path to the downloaded class list file or None if all downloads failed
    """
    output_path = os.path.join(output_dir, "annotations", "kinetics600_categories.txt")

    print("Trying to download official Kinetics 600 class list...")

    for url in KINETICS_600_CLASSES_URLS:
        try:
            path = download_file(url, output_path, allow_fail=True)
            if path:
                print(f"Successfully downloaded class list from {url}")
                return path
        except:
            continue

    print("Failed to download official class list from any source.")
    return None


def create_numeric_class_mapping(annotations_file, output_dir):
    """
    Create a numeric class mapping from annotation file.
    Convert string class names to numeric IDs in increasing order.

    Args:
        annotations_file (str): Path to JSON annotation file
        output_dir (str): Directory to save the class mapping

    Returns:
        dict: Class mapping dictionary or None if failed
    """
    if not os.path.exists(annotations_file):
        print(f"Annotation file not found: {annotations_file}")
        return None

    try:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)

        # Extract class names from annotations
        class_names = set()
        for vid_id, info in annotations.items():
            if 'annotations' in info and 'label' in info['annotations']:
                label = info['annotations']['label']
                if isinstance(label, str):
                    class_names.add(label)

        if not class_names:
            print("No class names found in annotations.")
            return None

        # Sort class names and assign numeric IDs
        sorted_names = sorted(list(class_names))
        name_to_id = {name: i for i, name in enumerate(sorted_names)}
        id_to_name = {i: name for i, name in enumerate(sorted_names)}

        # Create mapping file
        mapping = {
            "id_to_name": id_to_name,
            "name_to_id": name_to_id
        }

        output_file = os.path.join(output_dir, "annotations", "class_names_numeric.json")
        with open(output_file, 'w') as f:
            json.dump(mapping, f, indent=2)

        print(f"Created numeric class mapping with {len(sorted_names)} classes: {output_file}")
        return mapping

    except Exception as e:
        print(f"Error creating numeric class mapping: {e}")
        return None


def parse_official_class_list(class_list_file):
    """
    Parse the official Kinetics 600 class list.

    Args:
        class_list_file (str): Path to the class list file

    Returns:
        dict: Mapping from class name to class index
    """
    if not class_list_file or not os.path.exists(class_list_file):
        return None

    class_to_idx = {}
    try:
        with open(class_list_file, 'r') as f:
            for idx, line in enumerate(f):
                class_name = line.strip()
                if class_name:
                    class_to_idx[class_name] = idx

        print(f"Loaded {len(class_to_idx)} official class names")
        return class_to_idx
    except Exception as e:
        print(f"Error parsing official class list: {e}")
        return None


def download_test_videos(output_dir, parts=None):
    """
    Download test video parts from S3.

    Args:
        output_dir (str): Directory to save videos
        parts (list, optional): Specific part numbers to download. If None, download all.

    Returns:
        list: Paths to downloaded video archives
    """
    # Create test directory
    test_dir = os.path.join(output_dir, "videos", "test")
    os.makedirs(test_dir, exist_ok=True)

    downloaded_files = []

    # If parts is None, attempt to download parts 0-59
    if parts is None:
        parts = range(60)  # Kinetics test videos are split into 60 parts (0-59)

    # Download each specified part
    for part in parts:
        part_url = f"{S3_BASE_URL}/{TEST_VIDEOS_PATH}/part_{part}.tar.gz"
        output_path = os.path.join(test_dir, f"part_{part}.tar.gz")

        try:
            file_path = download_file(part_url, output_path)
            downloaded_files.append(file_path)
        except requests.exceptions.HTTPError as e:
            print(f"Error downloading part_{part}.tar.gz: {e}")
            # Continue with other parts even if one fails
            continue

    return downloaded_files


def extract_video_archive(archive_path, output_dir, limit=None):
    """
    Extract videos from a tar.gz archive.

    Args:
        archive_path (str): Path to tar.gz archive
        output_dir (str): Directory to extract videos
        limit (int, optional): Maximum number of videos to extract

    Returns:
        list: Paths to extracted video files
    """
    extracted_files = []

    print(f"Extracting {archive_path} to {output_dir}")

    # Open the archive
    with tarfile.open(archive_path, 'r:gz') as tar:
        # Get all MP4 files in the archive
        members = [m for m in tar.getmembers() if m.name.endswith('.mp4')]

        # Limit number of videos if specified
        if limit is not None:
            members = members[:min(limit, len(members))]

        # Extract each file with progress bar
        for member in tqdm(members, desc="Extracting videos"):
            tar.extract(member, path=output_dir)
            extracted_files.append(os.path.join(output_dir, member.name))

    return extracted_files


def convert_csv_to_json(csv_file, output_file, class_to_idx=None):
    """
    Convert CSV annotations to JSON format with proper class indices.

    Args:
        csv_file (str): Path to CSV annotation file
        output_file (str): Path to output JSON file
        class_to_idx (dict, optional): Mapping from class name to index. If None, use alphabetical order.

    Returns:
        dict: Annotation dictionary
    """
    annotations = {}
    unique_class_names = set()

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header

        # Get column indices; only set label_col if the header contains 'label'
        youtube_id_col = header.index('youtube_id') if 'youtube_id' in header else 0
        time_start_col = header.index('time_start') if 'time_start' in header else 1
        time_end_col = header.index('time_end') if 'time_end' in header else 2
        label_col = header.index('label') if 'label' in header else None

        for row in reader:
            youtube_id = row[youtube_id_col]
            annotation = {
                'subset': os.path.basename(csv_file).split('.')[0],
                'time_start': float(row[time_start_col]) if time_start_col < len(row) else 0,
                'time_end': float(row[time_end_col]) if time_end_col < len(row) else 0,
            }

            # Only add label if a label column exists and the value is not empty
            if label_col is not None and label_col < len(row) and row[label_col]:
                label_name = row[label_col]
                unique_class_names.add(label_name)

                # If we have an official mapping, use it; otherwise, keep string label
                if class_to_idx is not None and label_name in class_to_idx:
                    label_id = class_to_idx[label_name]
                    annotation['annotations'] = {
                        'label': label_id,
                        'label_name': label_name  # Keep the name for reference
                    }
                else:
                    annotation['annotations'] = {'label': label_name}

            annotations[youtube_id] = annotation

    print(f"Converted {csv_file} to {output_file} with {len(annotations)} entries")
    print(f"Found {len(unique_class_names)} unique class names in the annotations")

    # Warning if using official mapping but some classes weren't found
    if class_to_idx is not None:
        missing_classes = [name for name in unique_class_names if name not in class_to_idx]
        if missing_classes:
            print(f"Warning: {len(missing_classes)} class names in annotations were not found in official mapping.")
            print(f"First few missing classes: {missing_classes[:5]}")

    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    return annotations


def create_classnames_file(csv_file, output_file, official_class_to_idx=None):
    """
    Create a class names mapping file.

    Args:
        csv_file (str): Path to CSV annotation file
        output_file (str): Path to output JSON file
        official_class_to_idx (dict, optional): Official mapping from class name to index.
            If provided, it will be used instead of creating a new mapping.

    Returns:
        dict: Class mapping dictionary
    """
    # If we have an official mapping, use it
    if official_class_to_idx is not None:
        class_mapping = {idx: name for name, idx in official_class_to_idx.items()}
        class_mapping_with_indices = {
            "id_to_name": class_mapping,
            "name_to_id": official_class_to_idx
        }

        with open(output_file, 'w') as f:
            json.dump(class_mapping_with_indices, f, indent=2)

        print(f"Created class names mapping with {len(class_mapping)} official classes: {output_file}")
        return class_mapping_with_indices

    # Otherwise, extract class names from CSV and create numeric IDs
    class_names = set()
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header

        if 'label' not in header:
            print("Warning: No label column found in CSV. Cannot create class names mapping from this file.")
            return {}

        label_col = header.index('label')

        for row in reader:
            if label_col < len(row) and row[label_col]:
                class_names.add(row[label_col])

    sorted_names = sorted(list(class_names))
    class_mapping = {i: name for i, name in enumerate(sorted_names)}
    class_mapping_with_indices = {
        "id_to_name": class_mapping,
        "name_to_id": {name: i for i, name in class_mapping.items()}
    }

    with open(output_file, 'w') as f:
        json.dump(class_mapping_with_indices, f, indent=2)

    print(f"Created class names mapping with {len(sorted_names)} classes (numeric indices): {output_file}")
    print("NOTE: This mapping uses numeric indices (0 to N-1) instead of alphabetical order.")
    return class_mapping_with_indices


def update_existing_annotations(json_file, class_to_idx, output_file=None):
    """
    Update existing annotations to use numeric class indices.

    Args:
        json_file (str): Path to JSON annotation file
        class_to_idx (dict): Mapping from class name to index
        output_file (str, optional): Path to save updated annotations (defaults to overwriting original)

    Returns:
        dict: Updated annotations
    """
    if output_file is None:
        output_file = json_file

    try:
        with open(json_file, 'r') as f:
            annotations = json.load(f)

        updated = False
        for video_id, info in annotations.items():
            if 'annotations' in info and 'label' in info['annotations']:
                label = info['annotations']['label']
                if isinstance(label, str) and label in class_to_idx:
                    label_name = label
                    label_id = class_to_idx[label]
                    info['annotations'] = {
                        'label': label_id,
                        'label_name': label_name
                    }
                    updated = True

        if updated:
            with open(output_file, 'w') as f:
                json.dump(annotations, f, indent=2)
            print(f"Updated annotations in {json_file} with numeric class indices")
        else:
            print(f"No updates needed for {json_file} (may already use numeric indices)")

        return annotations

    except Exception as e:
        print(f"Error updating annotations: {e}")
        return None


def check_videos_against_annotations(annotations, video_dir):
    """
    Check which videos from the annotations are actually available.

    Args:
        annotations (dict): Annotation dictionary
        video_dir (str): Directory containing videos

    Returns:
        tuple: (found_videos, missing_videos)
    """
    found_videos = {}
    missing_videos = []

    print("Checking videos against annotations...")

    for video_id, info in tqdm(annotations.items(), desc="Checking videos"):
        # Check if video exists (could be different extensions)
        video_found = False
        for ext in ['.mp4', '.avi', '.mkv', '.mov']:
            # Try different common patterns
            patterns = [
                os.path.join(video_dir, f"{video_id}{ext}"),
                os.path.join(video_dir, 'test', f"{video_id}{ext}"),
                os.path.join(video_dir, '**', f"*{video_id}*{ext}")
            ]

            for pattern in patterns:
                matching_files = glob.glob(pattern, recursive=True)
                if matching_files:
                    # Video found
                    info['file_path'] = matching_files[0]
                    found_videos[video_id] = info
                    video_found = True
                    break

            if video_found:
                break

        if not video_found:
            missing_videos.append(video_id)

    print(f"Found {len(found_videos)} videos, missing {len(missing_videos)} videos")
    return found_videos, missing_videos


def filter_annotations(annotations, video_dir, output_file):
    """
    Create a filtered annotation file with only available videos.

    Args:
        annotations (dict): Annotation dictionary
        video_dir (str): Directory containing videos
        output_file (str): Path to save filtered annotations

    Returns:
        dict: Filtered annotations
    """
    found_videos, _ = check_videos_against_annotations(annotations, video_dir)

    # Write filtered annotations to file
    with open(output_file, 'w') as f:
        json.dump(found_videos, f, indent=2)

    print(f"Saved filtered annotations with {len(found_videos)} entries to {output_file}")
    return found_videos


def create_dummy_annotations(num_videos, output_file, official_class_to_idx=None):
    """
    Create dummy annotations for testing when real videos aren't available.

    Args:
        num_videos (int): Number of dummy videos to include
        output_file (str): Path to save dummy annotations
        official_class_to_idx (dict, optional): Official mapping from class name to index

    Returns:
        dict: Dummy annotations
    """
    annotations = {}

    # Create or use class names
    if official_class_to_idx is not None:
        class_names = list(official_class_to_idx.keys())
        num_classes = len(class_names)
        using_official = True
    else:
        # Create fixed set of class names
        class_names = [f"action_{i}" for i in range(600)]
        num_classes = 600
        using_official = False

    for i in range(num_videos):
        video_id = f"dummy_{i:06d}"
        class_idx = random.randint(0, num_classes - 1)
        class_name = class_names[class_idx]

        annotations[video_id] = {
            'subset': 'test',
            'annotations': {
                'label': official_class_to_idx[class_name] if using_official else class_idx,
                'label_name': class_name
            },
            'dummy': True
        }

    # Write to file
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"Created dummy annotations with {num_videos} entries: {output_file}")
    print(f"Using {'official' if using_official else 'generated'} class indices.")
    return annotations


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare Kinetics 600 dataset from Amazon S3 for evaluation')

    parser.add_argument('--output_dir', type=str, default='data/kinetics600',
                        help='Directory to save datasets')

    # Annotation options
    parser.add_argument('--download_annotations', action='store_true',
                        help='Download Kinetics 600 annotations from S3')
    parser.add_argument('--subset', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Dataset subset to work with')
    parser.add_argument('--create_classnames', action='store_true',
                        help='Create class names mapping file')
    parser.add_argument('--use_numeric_indices', action='store_true',
                        help='Use numeric class indices (0 to N-1) instead of string labels')
    parser.add_argument('--official_class_list', type=str, default=None,
                        help='Path to official class list file (if available)')

    # Video download options
    parser.add_argument('--download_videos', action='store_true',
                        help='Download test videos from S3')
    parser.add_argument('--parts', type=int, nargs='+',
                        help='Specific part numbers to download (e.g., --parts 0 1 2)')
    parser.add_argument('--extract', action='store_true',
                        help='Extract videos from downloaded archives')
    parser.add_argument('--max_videos_per_part', type=int, default=None,
                        help='Maximum number of videos to extract per archive')

    # Annotation processing
    parser.add_argument('--filter_annotations', action='store_true',
                        help='Filter annotations to match available videos')
    parser.add_argument('--video_dir', type=str,
                        help='Directory containing videos (for filtering)')
    parser.add_argument('--update_existing', type=str, default=None,
                        help='Update existing annotation file with numeric class indices')

    # Dummy data for testing
    parser.add_argument('--create_dummy', action='store_true',
                        help='Create dummy annotations for testing')
    parser.add_argument('--num_dummy_videos', type=int, default=100,
                        help='Number of dummy videos to include')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'annotations'), exist_ok=True)

    # Get official class indices if requested
    official_class_to_idx = None
    if args.official_class_list and os.path.exists(args.official_class_list):
        print(f"Using provided official class list: {args.official_class_list}")
        official_class_to_idx = parse_official_class_list(args.official_class_list)
    else:
        # Try downloading official class list
        class_list_file = try_download_official_class_list(args.output_dir)
        if class_list_file and os.path.exists(class_list_file):
            official_class_to_idx = parse_official_class_list(class_list_file)

    # Store whether we need to create a numeric class mapping
    class_mapping_needed = args.use_numeric_indices

    # Download annotations
    annotations = None
    if args.download_annotations:
        csv_file = download_annotation(args.subset, args.output_dir)
        json_file = os.path.join(args.output_dir, 'annotations', f'{args.subset}.json')
        annotations = convert_csv_to_json(csv_file, json_file, official_class_to_idx)

        # Flag to create class mapping if we downloaded annotations
        class_mapping_needed = True

    # Create class mapping if requested or needed
    if args.create_classnames or class_mapping_needed:
        # Try to get class names from subset's CSV
        csv_file = os.path.join(args.output_dir, 'annotations', f'{args.subset}.csv')
        class_names_file = os.path.join(args.output_dir, 'annotations', 'class_names.json')

        if os.path.exists(csv_file):
            create_classnames_file(csv_file, class_names_file, official_class_to_idx)
        else:
            # If current subset's CSV doesn't exist, try train or val
            alternative_subsets = ['train', 'val'] if args.subset == 'test' else ['val', 'train']
            found = False

            for alt_subset in alternative_subsets:
                alt_csv = os.path.join(args.output_dir, 'annotations', f'{alt_subset}.csv')
                if os.path.exists(alt_csv):
                    print(f"Using {alt_subset} annotations for class names")
                    create_classnames_file(alt_csv, class_names_file, official_class_to_idx)
                    found = True
                    break

            if not found:
                print("Warning: Could not find any annotation CSVs for class mapping")

    # Update existing annotations with numeric indices if requested
    if args.update_existing:
        if not os.path.exists(args.update_existing):
            print(f"Error: Annotation file not found: {args.update_existing}")
        else:
            # Get class mapping
            class_names_file = os.path.join(args.output_dir, 'annotations', 'class_names.json')
            if os.path.exists(class_names_file):
                with open(class_names_file, 'r') as f:
                    class_mapping = json.load(f)
                    class_to_idx = class_mapping.get('name_to_id', {})

                if class_to_idx:
                    update_existing_annotations(args.update_existing, class_to_idx)
                else:
                    print("Error: Class mapping file exists but has no name_to_id mapping")
            else:
                print("Error: Class mapping file not found, cannot update annotations")

    # Download test videos
    if args.download_videos:
        downloaded_files = download_test_videos(args.output_dir, args.parts)

        # Extract videos if requested
        if args.extract and downloaded_files:
            videos_dir = os.path.join(args.output_dir, 'videos', 'extracted')
            os.makedirs(videos_dir, exist_ok=True)

            for archive in downloaded_files:
                extract_video_archive(archive, videos_dir, args.max_videos_per_part)

    # Filter annotations based on available videos
    if args.filter_annotations:
        if not args.video_dir:
            print("Error: --video_dir is required for filtering annotations")
            return

        # Load annotations if not already loaded
        if annotations is None:
            json_file = os.path.join(args.output_dir, 'annotations', f'{args.subset}.json')
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    annotations = json.load(f)
            else:
                print(f"Error: Annotation file not found: {json_file}")
                print("Please download annotations first with --download_annotations")
                return

        # Filter and save
        output_file = os.path.join(args.output_dir, 'annotations', f'{args.subset}_filtered.json')
        filter_annotations(annotations, args.video_dir, output_file)

    # Create dummy annotations if requested
    if args.create_dummy:
        dummy_file = os.path.join(args.output_dir, 'annotations', f'{args.subset}_dummy.json')
        create_dummy_annotations(args.num_dummy_videos, dummy_file, official_class_to_idx)

    # If no specific action was requested, show help
    if not any([
        args.download_annotations,
        args.download_videos,
        args.filter_annotations,
        args.create_dummy,
        args.extract,
        args.create_classnames,
        args.update_existing
    ]):
        print("No action specified. Please choose at least one action:")
        print("  --download_annotations  Download annotations from S3")
        print("  --download_videos       Download test videos (use --parts to specify which)")
        print("  --extract               Extract downloaded video archives")
        print("  --filter_annotations    Filter annotations to match available videos")
        print("  --create_dummy          Create dummy annotations for testing")
        print("  --create_classnames     Create class name mapping file")
        print("  --update_existing       Update existing annotations with numeric indices")
        print("\nFor more details, use --help")


if __name__ == '__main__':
    main()