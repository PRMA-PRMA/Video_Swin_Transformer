"""
Script to fix label format in annotation files.
This converts string labels to integers for compatibility with PyTorch models.
"""

import os
import json
import glob
import argparse
from tqdm import tqdm


def fix_annotation_labels(input_file, output_file=None):
    """
    Fix annotation file by ensuring all labels are integers.
    Creates a mapping from string labels to integers if needed.

    Args:
        input_file (str): Path to input annotation file (JSON)
        output_file (str, optional): Path to save fixed annotation file
                                      If None, will use input_file + '_fixed.json'

    Returns:
        dict: Fixed annotations
    """
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '_fixed.json'

    print(f"Loading annotations from {input_file}")

    # Load annotations
    try:
        with open(input_file, 'r') as f:
            annotations = json.load(f)
    except Exception as e:
        print(f"Error loading annotation file: {e}")
        return None

    print(f"Loaded {len(annotations)} entries")

    # Check if we need to fix labels
    needs_fixing = False
    unique_labels = set()

    # First pass: check if any labels are strings
    for video_id, info in annotations.items():
        label = None
        if 'annotations' in info and 'label' in info['annotations']:
            label = info['annotations']['label']
        elif 'label' in info:
            label = info['label']

        if label is not None:
            unique_labels.add(label)
            if isinstance(label, str) and not label.isdigit():
                needs_fixing = True

    if not needs_fixing:
        print("No string labels found, annotations already have integer labels")
        return annotations

    # Create label mapping
    print(f"Found {len(unique_labels)} unique labels, creating mapping")
    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    # Save label mapping
    mapping_file = os.path.join(os.path.dirname(output_file), "label_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"Saved label mapping to {mapping_file}")

    # Second pass: convert labels
    fixed_annotations = {}
    for video_id, info in tqdm(annotations.items(), desc="Fixing labels"):
        fixed_info = info.copy()

        # Convert label in annotations
        if 'annotations' in fixed_info and 'label' in fixed_info['annotations']:
            label_str = fixed_info['annotations']['label']
            if isinstance(label_str, str):
                fixed_info['annotations']['label'] = label_mapping.get(label_str, 0)
                # Keep original for reference
                fixed_info['annotations']['label_name'] = label_str

        # Convert direct label
        if 'label' in fixed_info:
            label_str = fixed_info['label']
            if isinstance(label_str, str):
                fixed_info['label'] = label_mapping.get(label_str, 0)
                fixed_info['label_name'] = label_str

        fixed_annotations[video_id] = fixed_info

    # Save fixed annotations
    with open(output_file, 'w') as f:
        json.dump(fixed_annotations, f, indent=2)

    print(f"Saved fixed annotations to {output_file}")
    return fixed_annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix annotation labels for compatibility")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input annotation file")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save fixed annotation file")

    args = parser.parse_args()
    fix_annotation_labels(args.input, args.output)