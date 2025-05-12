"""
Simple script to verify dataset loading.
This checks if annotation files can be loaded and videos can be found.
"""

import os
import sys
import json
import glob
from tqdm import tqdm


def print_separator():
    print("\n" + "=" * 80 + "\n")


# Set up paths
current_dir = os.path.abspath(os.path.dirname(__file__))
annotation_file = os.path.join(current_dir, "../data", "kinetics600", "annotations", "test_fixed.json")
video_dir = os.path.join(current_dir, "../data", "kinetics600", "videos", "extracted")

print_separator()
print(f"Checking annotation file: {annotation_file}")

# Check if annotation file exists
if not os.path.exists(annotation_file):
    print(f"ERROR: Annotation file {annotation_file} does not exist!")
    json_files = glob.glob(os.path.join(current_dir, "../data", "kinetics600", "annotations", "*.json"))
    if json_files:
        print("Available annotation files:")
        for file in json_files:
            print(f"  - {file}")
    sys.exit(1)

# Load annotation file
try:
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    print(f"Successfully loaded annotations with {len(annotations)} entries")

    # Check if empty
    if len(annotations) == 0:
        print("WARNING: Annotation file is empty!")
except Exception as e:
    print(f"ERROR: Failed to load annotation file: {e}")
    sys.exit(1)

# Check for file_path property in annotations
has_file_path = all('file_path' in info for info in annotations.values())
print(f"Annotations have file_path property: {has_file_path}")

if not has_file_path:
    print("WARNING: Some annotations are missing 'file_path'. This is essential!")
    # Count missing file_paths
    missing = sum(1 for info in annotations.values() if 'file_path' not in info)
    print(f"  - {missing}/{len(annotations)} annotations are missing file_path")

print_separator()
print(f"Checking video directory: {video_dir}")

# Check if video directory exists
if not os.path.exists(video_dir):
    print(f"ERROR: Video directory {video_dir} does not exist!")
    sys.exit(1)

# Count video files
mp4_files = glob.glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)
print(f"Found {len(mp4_files)} MP4 files in video directory")

print_separator()
print("Checking if videos from annotations exist...")

# Check if videos from annotations exist
found = 0
missing = 0
for video_id, info in tqdm(annotations.items(), desc="Checking videos"):
    if 'file_path' in info:
        file_path = info['file_path']
        if os.path.exists(file_path):
            found += 1
        else:
            missing += 1
            if missing <= 5:  # Only show first 5 missing
                print(f"Missing: {file_path}")

print(f"Found {found}/{len(annotations)} videos referenced in annotations")
if missing > 0:
    print(f"Missing {missing}/{len(annotations)} videos referenced in annotations")

if found == 0:
    print("\nPossible issue: The absolute paths in the annotation file might not match your system.")
    print("Let's try a more flexible approach with relative paths:")

    # Get the video files relative to the video dir
    available_videos = set()
    for path in mp4_files:
        basename = os.path.basename(path)
        available_videos.add(basename)

    # Check if any video files match the basenames in annotations
    rel_found = 0
    for video_id, info in annotations.items():
        if 'file_path' in info:
            basename = os.path.basename(info['file_path'])
            if basename in available_videos:
                rel_found += 1

    print(f"Using relative paths (filenames only), found {rel_found}/{len(annotations)} videos")

    if rel_found > 0:
        print("\nRECOMMENDATION: The paths seem to be absolute but don't match your file system.")
        print(
            "Please regenerate the annotations using the debug_and_fix_annotations.py script from your current directory.")

print_separator()
print("RECOMMENDATION:")
if found > 0:
    print(f"✅ Everything looks good! {found} videos should be processed during evaluation.")
elif rel_found > 0:
    print("Run debug_and_fix_annotations.py again to recreate the annotation file with correct paths.")
else:
    print("1. Ensure the video files exist in the expected location")
    print("2. Regenerate the annotations with debug_and_fix_annotations.py")
    print("3. Check if there are additional issues with file permissions or path encoding")
print_separator()