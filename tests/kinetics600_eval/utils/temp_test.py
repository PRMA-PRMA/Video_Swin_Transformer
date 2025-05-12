#!/usr/bin/env python
import os
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import glob

try:
    import decord
    from decord import VideoReader

    decord.bridge.set_bridge("torch")
except ImportError:
    decord = None
    print("Decord not found. Please install it with 'pip install decord'.")

# Import your Video Swin Transformer implementation from vst.py
try:
    from vst import video_swin_b, video_swin_s, video_swin_t, video_swin_l, get_device
except ImportError:
    raise ImportError("vst module not found. Make sure vst.py is in your PYTHONPATH.")


def load_classnames(classnames_file, num_classes=600):
    """Load class names mapping from a JSON file or return a default mapping.

    Args:
        classnames_file (str): Path to the JSON file containing class names
        num_classes (int): Number of classes in the model

    Returns:
        dict: Dictionary mapping class indices to class names
    """
    if classnames_file and os.path.exists(classnames_file):
        with open(classnames_file, 'r') as f:
            data = json.load(f)

            # Handle different JSON structures
            if 'id_to_name' in data:
                # Convert keys to integers
                classnames = {int(k): v for k, v in data['id_to_name'].items()}
            else:
                # Try direct mapping if no id_to_name
                try:
                    classnames = {int(k): v for k, v in data.items()}
                except Exception:
                    classnames = data

            print(f"Loaded {len(classnames)} class names from {classnames_file}")

            # Fill in any missing classes
            for i in range(num_classes):
                if i not in classnames:
                    classnames[i] = f"unknown_{i}"

            return classnames
    else:
        print(f"Warning: Classnames file not found at {classnames_file}")
        print(f"Using default class names (unknown_0, unknown_1, etc.)")
        return {i: f"unknown_{i}" for i in range(num_classes)}


def load_model(model_type, pretrained_path, num_classes=600, device=None):
    if device is None:
        device = get_device()
    print(f"Loading {model_type} model on device {device}")

    # Create model without pretrained weights first
    if model_type.lower() in ['base', 'b']:
        model = video_swin_b(pretrained=None, num_classes=num_classes, device=device)
    elif model_type.lower() in ['small', 's']:
        model = video_swin_s(pretrained=None, num_classes=num_classes, device=device)
    elif model_type.lower() in ['tiny', 't']:
        model = video_swin_t(pretrained=None, num_classes=num_classes, device=device)
    elif model_type.lower() in ['large', 'l']:
        model = video_swin_l(pretrained=None, num_classes=num_classes, device=device)
    else:
        raise ValueError("Unknown model type")

    model = model.to(device)
    model.eval()

    # Use the improved inflate_weights method to load pretrained weights
    print(f"Loading pretrained weights from {pretrained_path}")
    model.inflate_weights(pretrained_path)

    # Print the model's state dict keys for debugging
    expected_keys = list(model.state_dict().keys())
    print("\nFirst 10 expected keys in model's state dict:")
    for key in expected_keys[:10]:
        print("  ", key)
    print("====================================\n")

    return model


def preprocess_video(video_path, clip_len=16, crop_size=224, short_side_size=256):
    if decord is None:
        raise ImportError("Decord is not installed.")
    try:
        vr = VideoReader(video_path)
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return None
    total_frames = len(vr)
    if total_frames < clip_len:
        print(f"Video {video_path} too short (only {total_frames} frames).")
        return None
    # Uniformly sample clip_len frames
    indices = np.linspace(0, total_frames - 1, num=clip_len, dtype=int)
    try:
        frames = vr.get_batch(indices)  # [clip_len, H, W, C]
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return None
    frames = frames.float() / 255.0  # scale to [0, 1]
    # Permute to [clip_len, C, H, W]
    frames = frames.permute(0, 3, 1, 2)
    # Resize if needed so the shorter side equals short_side_size
    _, C, H, W = frames.shape
    if H < crop_size or W < crop_size:
        if H < W:
            new_H = short_side_size
            new_W = int(W * short_side_size / H)
        else:
            new_W = short_side_size
            new_H = int(H * short_side_size / W)
        frames = torch.nn.functional.interpolate(frames, size=(new_H, new_W), mode="bilinear", align_corners=False)
        _, C, H, W = frames.shape
    # Center crop
    h_start = (H - crop_size) // 2
    w_start = (W - crop_size) // 2
    frames = frames[:, :, h_start:h_start + crop_size, w_start:w_start + crop_size]
    # Permute to [C, T, H, W]
    frames = frames.permute(1, 0, 2, 3)
    # Add batch dimension: [1, C, T, H, W]
    return frames.unsqueeze(0)


def evaluate_on_dataset(model, video_folder, device, classnames, clip_len=16):
    """Evaluate model on a folder of videos.

    Args:
        model: The Video Swin Transformer model
        video_folder (str): Path to folder containing videos
        device (torch.device): Device to run evaluation on
        classnames (dict): Dictionary mapping class indices to names
        clip_len (int): Number of frames to sample from each video
    """
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    if not video_files:
        print("No .mp4 video files found in", video_folder)
        return

    print(f"Found {len(video_files)} videos. Evaluating each one:")

    # Store results for summary
    all_results = []

    for video_path in video_files:
        input_tensor = preprocess_video(video_path, clip_len=clip_len)
        if input_tensor is None:
            continue

        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            top5_prob, top5_idx = torch.topk(probs, 5)
            top1_idx = top5_idx[0, 0].item()
            top5_indices = top5_idx[0].tolist()
            top5_prob = top5_prob[0].tolist()

        # Map indices to class names
        top1_name = classnames.get(top1_idx, f"unknown_{top1_idx}")
        top5_names = [classnames.get(idx, f"unknown_{idx}") for idx in top5_indices]

        # Format confidences as percentages
        top5_conf_pct = [f"{conf * 100:.1f}%" for conf in top5_prob]

        # Store result
        result = {
            'video': os.path.basename(video_path),
            'top1_class': top1_name,
            'top1_index': top1_idx,
            'top1_confidence': top5_prob[0],
            'top5_classes': top5_names,
            'top5_indices': top5_indices,
            'top5_confidences': top5_prob
        }
        all_results.append(result)

        # Print formatted result
        print(f"\nVideo: {os.path.basename(video_path)}")
        print(f"  Top-1 Prediction: {top1_name} (confidence: {top5_conf_pct[0]})")
        print("  Top-5 Predictions:")
        for name, conf_pct, idx in zip(top5_names, top5_conf_pct, top5_indices):
            print(f"    {name} (index: {idx}, confidence: {conf_pct})")

    # Print a summary of the most confident predictions
    if all_results:
        print("\n=== EVALUATION SUMMARY ===")
        sorted_results = sorted(all_results, key=lambda x: x['top1_confidence'], reverse=True)

        print(f"Most confident predictions:")
        for i, result in enumerate(sorted_results[:3], 1):
            print(f"{i}. {result['video']}: {result['top1_class']} ({result['top1_confidence'] * 100:.1f}%)")

        # Get most commonly predicted classes
        prediction_counter = {}
        for result in all_results:
            cls = result['top1_class']
            if cls in prediction_counter:
                prediction_counter[cls] += 1
            else:
                prediction_counter[cls] = 1

        if prediction_counter:
            print("\nMost common predictions:")
            common_classes = sorted(prediction_counter.items(), key=lambda x: x[1], reverse=True)
            for cls, count in common_classes[:3]:
                print(f"  {cls}: {count} videos")

    return all_results


def main():
    '''
    parser = argparse.ArgumentParser(
        description="Standalone script for debugging weight loading and evaluating on a small controlled dataset."
    )
    parser.add_argument("--video_folder", type=str, required=True,
                        help="Folder containing video files (e.g., .mp4)")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to the pretrained checkpoint")
    parser.add_argument("--model_type", type=str, default="base",
                        help="Model type: base, small, tiny, or large")
    parser.add_argument("--num_classes", type=int, default=600,
                        help="Number of classes (default 600)")
    parser.add_argument("--clip_len", type=int, default=16,
                        help="Number of frames per clip (default 16)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (e.g., 'cuda:0'); defaults to cuda if available")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device is not None else \
        (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Using device: {device}\n")

    model = load_model(args.model_type, args.pretrained_path, args.num_classes, device)
    evaluate_on_dataset(model, args.video_folder, device, args.clip_len)
    '''

    model_type = "base"
    pretrained_path = r"/checkpoints/pretrained_vst/swin_base_patch244_window877_kinetics600_22k.pth"
    num_classes = 600

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    video_folder = r"C:\Users\MartinParkerR\PycharmProjects\Video_Swin_Transformer\tests\kinetics600_eval\data\kinetics600\videos\temp"
    classnames_file = r"/tests/kinetics600_eval/data/kinetics600/annotations/class_names.json"
    clip_len = 16

    classnames = load_classnames(classnames_file, num_classes)

    model = load_model(model_type, pretrained_path, num_classes, device)
    evaluate_on_dataset(model, video_folder, device, classnames, clip_len)

if __name__ == "__main__":
    main()
