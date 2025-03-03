#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video Swin Transformer Evaluation Script
--------------------------------------
Evaluate a trained VST model on a test set and produce detailed metrics.
"""

import os
import argparse
import json
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Import VST implementation
from vst import video_swin_t, video_swin_s, video_swin_b, video_swin_l, get_device
from vst_custom import create_custom_model

# Import utilities
from utils.datasets import get_video_datasets
from utils.metrics import AverageMeter, accuracy, topk_errors, calculate_confusion_matrix


def load_model(checkpoint_path: str, device: torch.device, num_classes: int = 400, model_name: str = 'tiny',
               use_custom_head: bool = False):
    """
    Load a trained VST model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        num_classes: Number of classes
        model_name: Model size ('tiny', 'small', 'base', 'large')
        use_custom_head: Whether to use custom head model

    Returns:
        Loaded model
    """
    print(f"Loading checkpoint from {checkpoint_path}")

    # Create model
    if use_custom_head:
        model = create_custom_model(
            model_name=model_name,
            num_classes=num_classes,
            device=device
        )
    else:
        if model_name == 'tiny':
            model = video_swin_t(num_classes=num_classes, device=device)
        elif model_name == 'small':
            model = video_swin_s(num_classes=num_classes, device=device)
        elif model_name == 'base':
            model = video_swin_b(num_classes=num_classes, device=device)
        elif model_name == 'large':
            model = video_swin_l(num_classes=num_classes, device=device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def load_labels(dataset_name: str, data_dir: str) -> List[str]:
    """
    Load class labels based on dataset name.

    Args:
        dataset_name: Name of the dataset
        data_dir: Dataset directory

    Returns:
        List of class names
    """
    if dataset_name.lower() == 'ucf101':
        class_file = os.path.join(data_dir, 'annotations', 'classInd.txt')
        if os.path.exists(class_file):
            with open(class_file, 'r') as f:
                classes = []
                for line in f:
                    class_id, class_name = line.strip().split(' ')
                    classes.append(class_name)
            return classes

    elif dataset_name.lower() == 'hmdb51':
        # HMDB51 classes are determined from directory names
        frames_dir = os.path.join(data_dir, 'frames')
        if os.path.exists(frames_dir):
            classes = sorted([d for d in os.listdir(frames_dir)
                              if os.path.isdir(os.path.join(frames_dir, d))])
            return classes

    elif dataset_name.lower() == 'kinetics':
        class_file = os.path.join(data_dir, 'annotations', 'labels.csv')
        if os.path.exists(class_file):
            import csv
            with open(class_file, 'r') as f:
                reader = csv.DictReader(f)
                classes = sorted(set(row['label'] for row in reader))
            return classes

    elif dataset_name.lower() in ['ssv2', 'something-something-v2']:
        class_file = os.path.join(data_dir, 'annotations', 'something-something-v2-labels.json')
        if os.path.exists(class_file):
            with open(class_file, 'r') as f:
                label_dict = json.load(f)
                # Format: {"id": "1", "name": "class_name"}
                classes = [item['name'] for item in label_dict]
            return classes

    # If no valid labels found, return numbered classes
    print("Warning: No valid class labels found. Using generic class names.")
    return [f"Class {i}" for i in range(101)]


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device,
                   class_names: List[str], args: argparse.Namespace) -> Dict:
    """
    Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        data_loader: Test data loader
        device: Device to run evaluation on
        class_names: List of class names
        args: Command line arguments

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    # Metrics
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    # Collect predictions and targets for detailed metrics
    all_preds = []
    all_targets = []

    # Create confusion matrix
    num_classes = len(class_names)
    conf_matrix = torch.zeros(num_classes, num_classes, device=device)

    # Create iterator with optional progress bar
    if TQDM_AVAILABLE and not args.no_progress:
        data_iter = tqdm(data_loader, desc="Evaluating")
    else:
        data_iter = data_loader

    # Evaluation loop
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_iter):
            # Move to device
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            if args.amp and device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            # Compute metrics
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = targets.size(0)
            top1_meter.update(acc1.item(), batch_size)
            top5_meter.update(acc5.item(), batch_size)

            # Update confusion matrix
            _, preds = torch.max(outputs, 1)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                conf_matrix[t.long(), p.long()] += 1

            # Collect for detailed metrics
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

            # Update progress bar
            if TQDM_AVAILABLE and not args.no_progress:
                data_iter.set_postfix({
                    'Top-1 Acc': f"{top1_meter.avg:.2f}%",
                    'Top-5 Acc': f"{top5_meter.avg:.2f}%"
                })

    # Concatenate predictions and targets
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # Compute per-class metrics
    conf_matrix_np = conf_matrix.cpu().numpy()
    per_class_acc = conf_matrix_np.diagonal() / conf_matrix_np.sum(axis=1)

    # Convert confusion matrix to numpy for scikit-learn
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score

        # Compute precision, recall, f1
        precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
        recall = recall_score(all_targets, all_preds, average=None, zero_division=0)
        f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

        # Compute macro averages
        macro_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        macro_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        # Create classification report
        report = classification_report(all_targets, all_preds, target_names=class_names,
                                       zero_division=0, output_dict=True)
    except ImportError:
        # Fallback metrics if scikit-learn is not available
        precision = per_class_acc.copy()
        recall = per_class_acc.copy()
        f1 = per_class_acc.copy()
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        report = {"accuracy": top1_meter.avg / 100}

    # Gather results
    results = {
        'top1_acc': top1_meter.avg,
        'top5_acc': top5_meter.avg,
        'per_class_acc': per_class_acc * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'macro_precision': macro_precision * 100,
        'macro_recall': macro_recall * 100,
        'macro_f1': macro_f1 * 100,
        'conf_matrix': conf_matrix_np,
        'report': report
    }

    return results


def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: List[str],
                          output_path: str, figsize: Tuple[int, int] = (12, 10)):
    """
    Plot and save confusion matrix.

    Args:
        conf_matrix: Confusion matrix as numpy array
        class_names: List of class names
        output_path: Path to save the plot
        figsize: Figure size
    """
    # Normalize matrix
    norm_conf_matrix = conf_matrix.astype('float') / (conf_matrix.sum(axis=1, keepdims=True) + 1e-6)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot with colors
    im = ax.imshow(norm_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Normalized Counts', rotation=-90, va="bottom")

    # Set labels
    if len(class_names) <= 30:
        # Show all class names
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
    else:
        # Too many classes, show limited subset
        num_visible = 30
        step = len(class_names) // num_visible
        visible_indices = np.arange(0, len(class_names), step)
        ax.set_xticks(visible_indices)
        ax.set_yticks(visible_indices)
        ax.set_xticklabels([class_names[i] for i in visible_indices], rotation=45, ha="right")
        ax.set_yticklabels([class_names[i] for i in visible_indices])

    # Loop over data dimensions and create text annotations
    if len(class_names) <= 20:
        # Only show annotations for smaller matrices
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(j, i, f"{norm_conf_matrix[i, j]:.2f}",
                               ha="center", va="center", color="black" if norm_conf_matrix[i, j] < 0.5 else "white")

    # Set titles and labels
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Normalized Confusion Matrix')

    # Adjust layout and save
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_per_class_metrics(metrics: Dict, class_names: List[str], output_path: str,
                           figsize: Tuple[int, int] = (14, 8)):
    """
    Plot and save per-class metrics.

    Args:
        metrics: Dictionary of evaluation metrics
        class_names: List of class names
        output_path: Path to save the plot
        figsize: Figure size
    """
    # Extract metrics
    per_class_acc = metrics['per_class_acc']
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = metrics['f1']

    # If too many classes, select top/bottom classes
    if len(class_names) > 20:
        # Sort by F1 score
        indices = np.argsort(f1)

        # Get 10 best and 10 worst classes
        worst_indices = indices[:10]
        best_indices = indices[-10:]
        selected_indices = np.concatenate([worst_indices, best_indices])

        # Filter data
        selected_class_names = [class_names[i] for i in selected_indices]
        selected_per_class_acc = per_class_acc[selected_indices]
        selected_precision = precision[selected_indices]
        selected_recall = recall[selected_indices]
        selected_f1 = f1[selected_indices]
    else:
        # Use all classes
        selected_class_names = class_names
        selected_per_class_acc = per_class_acc
        selected_precision = precision
        selected_recall = recall
        selected_f1 = f1

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set width of bars
    barWidth = 0.2

    # Set positions of bars on X axis
    r1 = np.arange(len(selected_class_names))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    # Create bars
    ax.bar(r1, selected_per_class_acc, width=barWidth, label='Accuracy', color='blue', alpha=0.7)
    ax.bar(r2, selected_precision, width=barWidth, label='Precision', color='green', alpha=0.7)
    ax.bar(r3, selected_recall, width=barWidth, label='Recall', color='red', alpha=0.7)
    ax.bar(r4, selected_f1, width=barWidth, label='F1', color='purple', alpha=0.7)

    # Add labels and legend
    ax.set_xlabel('Class', fontweight='bold')
    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks([r + barWidth * 1.5 for r in range(len(selected_class_names))])
    ax.set_xticklabels(selected_class_names, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.legend()

    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Add values on top of bars
    for i, bars in enumerate(zip(selected_per_class_acc, selected_precision, selected_recall, selected_f1)):
        for j, (r, val) in enumerate(zip([r1[i], r2[i], r3[i], r4[i]], bars)):
            if val > 10:  # Only show values above 10%
                ax.text(r, val + 2, f"{val:.0f}", ha='center', fontsize=8)

    # Add horizontal line for average metrics
    ax.axhline(y=metrics['top1_acc'], color='blue', linestyle='--', alpha=0.5,
               label=f"Avg Acc: {metrics['top1_acc']:.1f}%")

    # Add macro-average metrics to the legend
    macro_values = [
        f"Macro-Precision: {metrics['macro_precision']:.1f}%",
        f"Macro-Recall: {metrics['macro_recall']:.1f}%",
        f"Macro-F1: {metrics['macro_f1']:.1f}%"
    ]

    # Create custom legend for macro metrics
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='green', lw=2, linestyle='--'),
        Line2D([0], [0], color='red', lw=2, linestyle='--'),
        Line2D([0], [0], color='purple', lw=2, linestyle='--')
    ]

    ax.legend(custom_lines + ax.get_legend_handles_labels()[0],
              macro_values + ax.get_legend_handles_labels()[1],
              loc='upper right')

    # Adjust layout and save
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='VST Evaluation')

    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='tiny',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of classes (if None, determined from dataset)')
    parser.add_argument('--use-custom-head', action='store_true',
                        help='Use custom head model')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ucf101', 'hmdb51', 'kinetics', 'ssv2'],
                        help='Dataset name')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--clip-length', type=int, default=16,
                        help='Number of frames per clip')
    parser.add_argument('--frame-stride', type=int, default=2,
                        help='Stride between frames')
    parser.add_argument('--spatial-size', type=int, default=224,
                        help='Spatial size for frames')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')

    # Output parameters
    parser.add_argument('--output-dir', type=str, default='evaluation',
                        help='Directory to save evaluation outputs')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress bar')

    # Execution parameters
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run evaluation on')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')

    args = parser.parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    print(f"Running evaluation on {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get class names
    class_names = load_labels(args.dataset, args.data_dir)

    # Set num_classes if not specified
    if args.num_classes is None:
        args.num_classes = len(class_names)
        print(f"Using {args.num_classes} classes based on dataset")

    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        num_classes=args.num_classes,
        model_name=args.model,
        use_custom_head=args.use_custom_head
    )

    # Create dataset and dataloader
    _, val_dataset = get_video_datasets(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        spatial_size=args.spatial_size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print(f"Evaluating on {len(val_dataset)} samples from {args.dataset} dataset")

    # Evaluate model
    results = evaluate_model(model, val_loader, device, class_names, args)

    # Print results
    print("\nEvaluation Results:")
    print(f"Top-1 Accuracy: {results['top1_acc']:.2f}%")
    print(f"Top-5 Accuracy: {results['top5_acc']:.2f}%")
    print(f"Macro-Precision: {results['macro_precision']:.2f}%")
    print(f"Macro-Recall: {results['macro_recall']:.2f}%")
    print(f"Macro-F1: {results['macro_f1']:.2f}%")

    # Save detailed results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {
            'top1_acc': results['top1_acc'],
            'top5_acc': results['top5_acc'],
            'macro_precision': results['macro_precision'],
            'macro_recall': results['macro_recall'],
            'macro_f1': results['macro_f1'],
            'per_class_metrics': {
                class_names[i]: {
                    'accuracy': results['per_class_acc'][i],
                    'precision': results['precision'][i],
                    'recall': results['recall'][i],
                    'f1': results['f1'][i]
                } for i in range(len(class_names))
            }
        }
        json.dump(json_results, f, indent=4)

    print(f"Detailed results saved to {results_path}")

    # Plot confusion matrix
    conf_matrix_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['conf_matrix'], class_names, conf_matrix_path)

    # Plot per-class metrics
    metrics_path = os.path.join(args.output_dir, 'per_class_metrics.png')
    plot_per_class_metrics(results, class_names, metrics_path)


if __name__ == '__main__':
    main()