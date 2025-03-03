#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video Swin Transformer Training Script
-------------------------------------
Training script for Video Swin Transformer models using your existing implementation.
"""

import os
import time
import json
import argparse
import datetime
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Import your existing VST implementation
from vst import video_swin_t, video_swin_s, video_swin_b, video_swin_l, get_device
from vst_custom import create_custom_model

# Import dataset and utilities
from utils.datasets import get_video_datasets
from utils.metrics import AverageMeter, accuracy


def get_model(args):
    """
    Create a VST model based on command line arguments.

    Args:
        args: Command line arguments

    Returns:
        Model instance
    """
    print(f"Creating {args.model} Video Swin Transformer model for {args.num_classes} classes")

    # Set up activation function if specified
    activation = None
    if args.activation == 'relu':
        activation = nn.ReLU()
    elif args.activation == 'gelu':
        activation = nn.GELU()

    # Determine device
    device = torch.device(args.device) if args.device else get_device()

    # Create using predefined functions or custom model
    if args.use_custom_head:
        model = create_custom_model(
            model_name=args.model,
            pretrained=args.pretrained_path if args.pretrained else None,
            num_classes=args.num_classes,
            dropout_rate=args.dropout,
            activation=activation,
            feature_mode=False,
            device=device
        )

        # Freeze backbone if specified
        if args.freeze_backbone:
            model.freeze_backbone(True)
            print("Backbone frozen for fine-tuning")
    else:
        # Use standard model constructors
        if args.model == 'tiny':
            model = video_swin_t(
                pretrained=args.pretrained_path if args.pretrained else False,
                num_classes=args.num_classes,
                device=device
            )
        elif args.model == 'small':
            model = video_swin_s(
                pretrained=args.pretrained_path if args.pretrained else False,
                num_classes=args.num_classes,
                device=device
            )
        elif args.model == 'base':
            model = video_swin_b(
                pretrained=args.pretrained_path if args.pretrained else False,
                num_classes=args.num_classes,
                device=device
            )
        elif args.model == 'large':
            model = video_swin_l(
                pretrained=args.pretrained_path if args.pretrained else False,
                num_classes=args.num_classes,
                device=device
            )

    # Print model summary
    count_parameters(model)

    return model


def get_optimizer(model, args):
    """
    Create optimizer for training.

    Args:
        model: Model to optimize
        args: Command line arguments

    Returns:
        Optimizer
    """
    # Define parameter groups with different learning rates
    # Backbone gets lower learning rate when fine-tuning
    if args.freeze_backbone and hasattr(model, 'base_model'):
        params = [
            {'params': model.head.parameters() if hasattr(model, 'head') else model.custom_head.parameters()}
        ]
    elif args.finetune and hasattr(model, 'base_model'):
        params = [
            {'params': model.base_model.parameters(), 'lr': args.lr * 0.1},
            {'params': model.head.parameters() if hasattr(model, 'head') else model.custom_head.parameters()}
        ]
    else:
        params = model.parameters()

    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    return optimizer


def get_scheduler(optimizer, args):
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer
        args: Command line arguments

    Returns:
        Learning rate scheduler
    """
    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
    elif args.lr_scheduler == 'multistep':
        # Convert percentages to epochs
        if args.lr_milestones:
            milestones = [int(m * args.epochs) for m in args.lr_milestones]
        else:
            milestones = [int(args.epochs * 0.6), int(args.epochs * 0.8)]

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.lr_gamma
        )
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr
        )
    elif args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_gamma,
            patience=args.patience,
            threshold=1e-4,
            min_lr=args.min_lr,
            verbose=True
        )
    else:
        scheduler = None

    return scheduler


def get_criterion(args):
    """
    Get loss function.

    Args:
        args: Command line arguments

    Returns:
        Loss function
    """
    if args.label_smoothing > 0:
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        return nn.CrossEntropyLoss()


def count_parameters(model):
    """
    Count number of trainable parameters in the model.

    Args:
        model: Model
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, args):
    """
    Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch
        args: Command line arguments

    Returns:
        Dict containing training metrics
    """
    model.train()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    # Create gradient scaler for AMP
    scaler = GradScaler() if args.amp else None

    # Get iterator with optional progress bar
    if TQDM_AVAILABLE and not args.no_progress:
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
    else:
        train_iter = train_loader

    # Training loop
    start_time = time.time()
    for i, (inputs, targets) in enumerate(train_iter):
        # Move data to device
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with optional AMP
        if args.amp:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Backward pass with scaler
            scaler.scale(loss).backward()

            # Gradient clipping
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            # Update weights
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward/backward
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient clipping
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            # Update weights
            optimizer.step()

        # Compute accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = targets.size(0)

        # Update meters
        loss_meter.update(loss.item(), batch_size)
        acc1_meter.update(acc1.item(), batch_size)
        acc5_meter.update(acc5.item(), batch_size)

        # Update progress bar
        if TQDM_AVAILABLE and not args.no_progress:
            train_iter.set_postfix({
                'loss': f"{loss_meter.avg:.4f}",
                'acc@1': f"{acc1_meter.avg:.2f}%",
                'acc@5': f"{acc5_meter.avg:.2f}%",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        # Log to console
        elif i % args.print_freq == 0:
            print(
                f"Train: [{epoch + 1}/{args.epochs}][{i}/{len(train_loader)}] "
                f"Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                f"Acc@1: {acc1_meter.val:.3f}% ({acc1_meter.avg:.3f}%) "
                f"Acc@5: {acc5_meter.val:.3f}% ({acc5_meter.avg:.3f}%) "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    train_time = time.time() - start_time

    # Return metrics
    return {
        'loss': loss_meter.avg,
        'acc1': acc1_meter.avg,
        'acc5': acc5_meter.avg,
        'time': train_time
    }


def validate(model, val_loader, criterion, device, args):
    """
    Validate the model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        args: Command line arguments

    Returns:
        Dict containing validation metrics
    """
    model.eval()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    # Get iterator with optional progress bar
    if TQDM_AVAILABLE and not args.no_progress:
        val_iter = tqdm(val_loader, desc="Validation")
    else:
        val_iter = val_loader

    # Validation loop
    start_time = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_iter):
            # Move data to device
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Compute accuracy
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = targets.size(0)

            # Update meters
            loss_meter.update(loss.item(), batch_size)
            acc1_meter.update(acc1.item(), batch_size)
            acc5_meter.update(acc5.item(), batch_size)

            # Update progress bar
            if TQDM_AVAILABLE and not args.no_progress:
                val_iter.set_postfix({
                    'loss': f"{loss_meter.avg:.4f}",
                    'acc@1': f"{acc1_meter.avg:.2f}%",
                    'acc@5': f"{acc5_meter.avg:.2f}%"
                })

            # Log to console
            elif i % args.print_freq == 0:
                print(
                    f"Val: [{i}/{len(val_loader)}] "
                    f"Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                    f"Acc@1: {acc1_meter.val:.3f}% ({acc1_meter.avg:.3f}%) "
                    f"Acc@5: {acc5_meter.val:.3f}% ({acc5_meter.avg:.3f}%)"
                )

    val_time = time.time() - start_time

    # Log final results
    print(
        f"Validation Results: "
        f"Loss: {loss_meter.avg:.4f} "
        f"Acc@1: {acc1_meter.avg:.3f}% "
        f"Acc@5: {acc5_meter.avg:.3f}% "
        f"Time: {val_time:.2f}s"
    )

    # Return metrics
    return {
        'loss': loss_meter.avg,
        'acc1': acc1_meter.avg,
        'acc5': acc5_meter.avg,
        'time': val_time
    }


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth'):
    """
    Save training checkpoint.

    Args:
        state: Checkpoint state
        is_best: Whether this is the best model so far
        output_dir: Directory to save checkpoint to
        filename: Checkpoint filename
    """
    ckpt_path = os.path.join(output_dir, filename)
    torch.save(state, ckpt_path)
    if is_best:
        best_path = os.path.join(output_dir, 'model_best.pth')
        torch.save(state, best_path)
        print(f"New best model saved to {best_path}")


def load_checkpoint(model, ckpt_path, optimizer=None, scheduler=None):
    """
    Load checkpoint.

    Args:
        model: Model to load weights into
        ckpt_path: Path to checkpoint
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state

    Returns:
        Dict containing checkpoint info
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # Load model state
    model.load_state_dict(checkpoint['state_dict'])

    # Optionally load optimizer state
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Optionally load scheduler state
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    return checkpoint


def main():
    parser = argparse.ArgumentParser(description='Video Swin Transformer Training')

    # Model parameters
    parser.add_argument('--model', default='tiny', choices=['tiny', 'small', 'base', 'large'],
                        help='Model size')
    parser.add_argument('--num-classes', type=int, default=400,
                        help='Number of classes to classify')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained model')
    parser.add_argument('--pretrained-path', type=str, default=None,
                        help='Path to pretrained model weights')
    parser.add_argument('--use-custom-head', action='store_true',
                        help='Use VideoSwinTransformerWithCustomHead')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate for classifier head')
    parser.add_argument('--activation', default=None, choices=['relu', 'gelu'],
                        help='Activation function for classifier head')

    # Dataset parameters
    parser.add_argument('--dataset', default='kinetics', choices=['kinetics', 'ucf101', 'hmdb51', 'ssv2'],
                        help='Dataset to use')
    parser.add_argument('--data-dir', default='./data',
                        help='Path to dataset root directory')
    parser.add_argument('--clip-length', type=int, default=16,
                        help='Number of frames in each clip')
    parser.add_argument('--frame-stride', type=int, default=2,
                        help='Stride between sampled frames')
    parser.add_argument('--spatial-size', type=int, default=224,
                        help='Input frame size')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train for')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch (for resuming)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=None,
                        help='Batch size for validation (defaults to --batch-size)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', default=None,
                        help='Device to train on (default: cuda if available)')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision')

    # Optimization parameters
    parser.add_argument('--optimizer', default='adamw', choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6,
                        help='Minimum learning rate for scheduler')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    parser.add_argument('--nesterov', action='store_true',
                        help='Use Nesterov momentum for SGD')
    parser.add_argument('--clip-grad-norm', type=float, default=1.0,
                        help='Gradient norm clipping value')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing factor')

    # Learning rate scheduler
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=['step', 'multistep', 'cosine', 'plateau', 'none'],
                        help='LR scheduler to use')
    parser.add_argument('--lr-step-size', type=int, default=30,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Gamma for schedulers')
    parser.add_argument('--lr-milestones', type=float, nargs='+', default=None,
                        help='Milestones for MultiStepLR as fraction of epochs (e.g., 0.6 0.8)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for ReduceLROnPlateau scheduler')

    # Fine-tuning parameters
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune pretrained model')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone weights for fine-tuning')

    # Checkpoint and logging
    parser.add_argument('--output-dir', default='./output',
                        help='Path to save outputs')
    parser.add_argument('--log-dir', default=None,
                        help='TensorBoard log directory (default: output_dir/logs)')
    parser.add_argument('--resume', default='',
                        help='Resume from checkpoint')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model on validation set')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--save-freq', type=int, default=1,
                        help='Save checkpoint frequency (epochs)')
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress bar')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set validation batch size if not specified
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size

    # Set log dir if not specified
    if args.log_dir is None:
        args.log_dir = os.path.join(args.output_dir, 'logs')

    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(args.device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # Save command line arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Create TensorBoard writer
    writer = None
    if TENSORBOARD_AVAILABLE:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=os.path.join(args.log_dir, timestamp))
        print(f"TensorBoard logs will be saved to {args.log_dir}/{timestamp}")

    # Create model
    model = get_model(args)
    model = model.to(device)

    # Create optimizer, scheduler, and loss
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    criterion = get_criterion(args)

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = load_checkpoint(
            model=model,
            ckpt_path=args.resume,
            optimizer=optimizer,
            scheduler=scheduler
        )
        args.start_epoch = checkpoint['epoch'] + 1

    # Create datasets and loaders
    train_dataset, val_dataset = get_video_datasets(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        clip_length=args.clip_length,
        frame_stride=args.frame_stride,
        spatial_size=args.spatial_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Evaluation mode
    if args.evaluate:
        val_metrics = validate(model, val_loader, criterion, device, args)
        return

    # Training loop
    best_acc = 0.0
    print(f"Starting training for {args.epochs} epochs")
    for epoch in range(args.start_epoch, args.epochs):
        print(f"\n{'=' * 20} Epoch {epoch + 1}/{args.epochs} {'=' * 20}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, args)

        # Update learning rate
        if scheduler is not None:
            if args.lr_scheduler == 'plateau':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        # Log metrics
        if writer is not None:
            writer.add_scalar('train/loss', train_metrics['loss'], epoch)
            writer.add_scalar('train/acc1', train_metrics['acc1'], epoch)
            writer.add_scalar('train/acc5', train_metrics['acc5'], epoch)
            writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            writer.add_scalar('val/acc1', val_metrics['acc1'], epoch)
            writer.add_scalar('val/acc5', val_metrics['acc5'], epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Save checkpoint
        is_best = val_metrics['acc1'] > best_acc
        best_acc = max(val_metrics['acc1'], best_acc)

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'best_acc1': best_acc,
            'args': vars(args)
        }

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                state=state,
                is_best=is_best,
                output_dir=args.output_dir,
                filename=f'checkpoint_epoch_{epoch + 1}.pth'
            )

        # Always save latest
        save_checkpoint(
            state=state,
            is_best=is_best,
            output_dir=args.output_dir,
            filename='checkpoint_latest.pth'
        )

    print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")

    # Clean up
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()