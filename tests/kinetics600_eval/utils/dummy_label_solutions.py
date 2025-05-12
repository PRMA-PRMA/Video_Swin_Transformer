"""
Create dummy labels for Kinetics evaluation when true labels are missing.
This script examines the annotation file and fixes test set annotations
by either extracting class information from filenames or creating dummy labels.
"""

import os
import json
import glob
import argparse
import re
import random
from tqdm import tqdm


def create_dummy_labels(input_file, output_file=None, class_count=600):
    """
    Create dummy labels for Kinetics evaluation.

    Args:
        input_file (str): Path to the input annotation file
        output_file (str, optional): Path to save the fixed annotations
        class_count (int, optional): Number of classes in the dataset

    Returns:
        dict: Modified annotations with dummy labels
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '_eval_ready.json'

    print(f"Loading annotations from {input_file}")

    try:
        with open(input_file, 'r') as f:
            annotations = json.load(f)
    except Exception as e:
        print(f"Error loading annotation file: {e}")
        return None

    print(f"Loaded {len(annotations)} video entries")

    # Check what kind of labels we have
    has_proper_labels = False
    label_types = set()
    sample_labels = []

    for video_id, info in list(annotations.items())[:10]:  # Check first 10 entries
        label = None
        # Check in annotations
        if 'annotations' in info and 'label' in info['annotations']:
            label = info['annotations']['label']
        # Check direct label
        elif 'label' in info:
            label = info['label']

        if label is not None:
            label_types.add(type(label).__name__)
            sample_labels.append(str(label))

    print(f"Label types found: {label_types}")
    print(f"Sample labels: {sample_labels}")

    # If all labels are "test" or a single value, they're likely not real class labels
    if len(set(sample_labels)) <= 1 and "test" in sample_labels:
        print("WARNING: All labels appear to be 'test' - these are data split indicators, not class labels!")
        has_proper_labels = False
    elif "int" in label_types or all(l.isdigit() for l in sample_labels if isinstance(l, str)):
        print("Labels appear to be integers or numeric strings - likely real class labels")
        has_proper_labels = True

    if has_proper_labels:
        print("Annotations already have proper labels, just ensuring they're integers")
        # Just ensure all labels are integers
        for video_id, info in tqdm(annotations.items(), desc="Converting labels to integers"):
            if 'annotations' in info and 'label' in info['annotations']:
                label = info['annotations']['label']
                if isinstance(label, str) and label.isdigit():
                    info['annotations']['label'] = int(label)
            if 'label' in info:
                label = info['label']
                if isinstance(label, str) and label.isdigit():
                    info['label'] = int(label)
    else:
        print("Creating dummy labels for evaluation")

        # Try to extract class info from filenames, or create dummy labels
        for video_id, info in tqdm(annotations.items(), desc="Creating dummy labels"):
            # First try to extract from filename if available
            extracted_label = None
            if 'file_path' in info:
                filename = os.path.basename(info['file_path'])
                # Some datasets encode class ID in filename (e.g., "classID_videoID")
                match = re.search(r'^(\d+)_', filename)
                if match:
                    extracted_label = int(match.group(1))
                    if extracted_label >= class_count:
                        extracted_label = extracted_label % class_count

            # If extraction failed, assign a dummy label (sequential or random)
            if extracted_label is None:
                # Use video_id hash to get a consistent dummy label
                # This way the same video always gets the same label
                dummy_label = hash(video_id) % class_count

                # Update both label locations
                if 'annotations' in info:
                    info['annotations']['dummy_evaluation'] = True
                    info['annotations']['original_label'] = info['annotations'].get('label', 'unknown')
                    info['annotations']['label'] = dummy_label
                else:
                    info['annotations'] = {
                        'dummy_evaluation': True,
                        'label': dummy_label
                    }

                # Also update direct label
                if 'label' in info:
                    info['original_label'] = info['label']
                info['label'] = dummy_label
                info['dummy_evaluation'] = True
            else:
                # Use extracted label
                if 'annotations' in info:
                    info['annotations']['label'] = extracted_label
                    info['annotations']['extracted_from_filename'] = True
                else:
                    info['annotations'] = {
                        'label': extracted_label,
                        'extracted_from_filename': True
                    }
                info['label'] = extracted_label

    # Save modified annotations
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"Saved evaluation-ready annotations to {output_file}")
    print("\nIMPORTANT: If using dummy labels, accuracy metrics will be meaningless!")
    print("The script will still generate predictions that can be used for analysis.\n")

    return annotations


def main():
    parser = argparse.ArgumentParser(description="Create dummy labels for test set evaluation")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input annotation file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save modified annotation file")
    parser.add_argument("--class_count", type=int, default=600,
                        help="Number of classes in the dataset (default: 600)")

    args = parser.parse_args()
    create_dummy_labels(args.input, args.output, args.class_count)


if __name__ == "__main__":
    main()