"""
Metrics and Utility Functions
----------------------------
Metrics and utilities for training and evaluating Video Swin Transformer models.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union


class AverageMeter:
    """
    Computes and stores the average and current value.
    Used for tracking metrics during training and validation.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output: Model output (B x C)
        target: Target labels (B)
        topk: Tuple of k values

    Returns:
        List of accuracies for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.

    Args:
        optimizer: Optimizer

    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def topk_errors(preds: torch.Tensor, labels: torch.Tensor, ks: Tuple[int, ...] = (1, 5)) -> List[float]:
    """
    Computes the top-k error rate for each k in ks.

    Args:
        preds: Predictions (N x C)
        labels: Ground-truth labels (N)
        ks: Tuple of k values

    Returns:
        List of error rates for each k
    """
    errors = []
    for k in ks:
        # Get top-k accuracy
        acc = accuracy(preds, labels, topk=(k,))[0]
        # Compute error rate
        err = 100.0 - acc
        errors.append(err.item())
    return errors


def calculate_confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Calculate confusion matrix.

    Args:
        preds: Predictions (N x C) or class indices (N)
        labels: Ground-truth labels (N)
        num_classes: Number of classes

    Returns:
        Confusion matrix (num_classes x num_classes)
    """
    with torch.no_grad():
        # Get predicted class indices if not already
        if preds.dim() > 1:
            preds = torch.argmax(preds, dim=1)

        # Initialize confusion matrix
        conf_matrix = torch.zeros(num_classes, num_classes, device=preds.device)

        # Accumulate counts
        for t, p in zip(labels.view(-1), preds.view(-1)):
            conf_matrix[t.long(), p.long()] += 1

        return conf_matrix


def calculate_metrics_from_confusion_matrix(conf_matrix: torch.Tensor) -> Dict[str, float]:
    """
    Calculate various metrics from a confusion matrix.

    Args:
        conf_matrix: Confusion matrix (num_classes x num_classes)

    Returns:
        Dictionary of metrics (accuracy, precision, recall, f1)
    """
    # Calculate metrics
    num_classes = conf_matrix.size(0)

    # Convert to numpy for easier manipulation
    if isinstance(conf_matrix, torch.Tensor):
        conf_matrix = conf_matrix.cpu().numpy()

    # Class-wise metrics
    tp = np.diag(conf_matrix)
    fp = np.sum(conf_matrix, axis=0) - tp
    fn = np.sum(conf_matrix, axis=1) - tp

    # Overall accuracy
    accuracy = np.sum(tp) / np.sum(conf_matrix)

    # Class-wise precision, recall, and F1
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    # Replace NaNs with 0
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1 = np.nan_to_num(f1)

    # Average metrics
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_f1 = np.mean(f1)

    # Return metrics
    metrics = {
        'accuracy': float(accuracy * 100),
        'mean_precision': float(mean_precision * 100),
        'mean_recall': float(mean_recall * 100),
        'mean_f1': float(mean_f1 * 100)
    }

    return metrics


def compute_video_level_accuracy(clip_probs: torch.Tensor, video_labels: torch.Tensor) -> Tuple[float, float]:
    """
    Compute video-level accuracy by averaging clip predictions.

    Args:
        clip_probs: Tensor of clip probabilities (num_videos x num_clips x num_classes)
        video_labels: Tensor of video labels (num_videos)

    Returns:
        Tuple of (top1_accuracy, top5_accuracy) as percentages
    """
    # Average clip probabilities
    video_probs = torch.mean(clip_probs, dim=1)

    # Compute top-1 and top-5 accuracy
    acc1, acc5 = accuracy(video_probs, video_labels, topk=(1, 5))

    return acc1.item(), acc5.item()