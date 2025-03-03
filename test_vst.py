"""
Test script for Video Swin Transformer
--------------------------------------
This script contains tests to validate the functionality of the Video Swin Transformer implementation,
with automatic GPU acceleration when available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import gc
import argparse
from typing import Optional, Tuple, Dict, Any, List, Union
from PIL import Image
import cv2
import psutil

# Conditionally import memory_profiler
try:
    from memory_profiler import profile as memory_profile
except ImportError:
    # Create a no-op decorator if memory_profiler is not installed
    def memory_profile(func):
        return func

# Import the Video Swin Transformer implementation
from vst import (
    video_swin_t,
    video_swin_b,
    video_swin_s,
    video_swin_l,
    window_partition,
    window_reverse,
    compute_mask,
    VideoSwinTransformerWithCustomHead,
    video_swin_t_custom,
    video_swin_b_custom,
    get_device
)

# Default path to pretrained weights (will be overridden by command line arguments)
#PRETRAINED_PATH = None
PRETRAINED_PATH = r"C:\Users\MartinParkerR\PycharmProjects\Video_Swin_Transformer\checkpoints\pretrained_vst\swin_base_patch244_window877_kinetics600_22k.pth"


def get_device_stats() -> Dict[str, Any]:
    """
    Get detailed information about the current device (CPU/GPU).

    Returns:
        Dict[str, Any]: Dictionary with device information
    """
    stats = {
        "device": get_device(),
        "cpu_info": {
            "logical_cores": os.cpu_count(),
            "memory_available_gb": psutil.virtual_memory().available / (1024 ** 3)
        }
    }

    if torch.cuda.is_available():
        stats["cuda_info"] = {
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "memory_allocated_gb": torch.cuda.memory_allocated(0) / (1024 ** 3),
            "memory_reserved_gb": torch.cuda.memory_reserved(0) / (1024 ** 3),
            "max_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        }

    return stats


def print_device_info():
    """Print device information."""
    stats = get_device_stats()
    device = stats["device"]

    print(f"\n{'=' * 20} DEVICE INFORMATION {'=' * 20}")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        cuda_info = stats["cuda_info"]
        print(f"GPU: {cuda_info['device_name']}")
        print(f"CUDA Devices: {cuda_info['device_count']}")
        print(f"Current CUDA Device: {cuda_info['current_device']}")
        print(f"Total GPU Memory: {cuda_info['max_memory_gb']:.2f} GB")
        print(f"GPU Memory Usage: {cuda_info['memory_allocated_gb']:.2f} GB allocated, "
              f"{cuda_info['memory_reserved_gb']:.2f} GB reserved")

        # Check for mixed precision support
        if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            print("Mixed precision (AMP) is supported")
        else:
            print("Mixed precision (AMP) is not supported")
    else:
        print("Running on CPU")

    print(f"CPU cores: {stats['cpu_info']['logical_cores']}")
    print(f"Available system memory: {stats['cpu_info']['memory_available_gb']:.2f} GB")
    print('=' * 60 + '\n')


def free_memory():
    """Free up memory by clearing caches and running garbage collection."""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def select_model_by_weights(pretrained_path: Optional[str] = None, device: Optional[torch.device] = None, **kwargs):
    """
    Selects the appropriate model based on the pretrained weights filename.

    Args:
        pretrained_path (str, optional): Path to pretrained weights. Defaults to None.
        device (torch.device, optional): Device to place the model on. If None, will use CUDA if available.
        **kwargs: Additional arguments to pass to the model constructor.

    Returns:
        model: Selected model instance
    """
    # If no device specified, use the best available
    if device is None:
        device = get_device()

    if pretrained_path is None:
        print("Using Video Swin Tiny model (no pretrained weights)")
        return video_swin_t(device=device, **kwargs)

    if 'base' in pretrained_path.lower():
        print("Using Video Swin Base model")
        return video_swin_b(pretrained=pretrained_path, device=device, **kwargs)
    elif 'small' in pretrained_path.lower():
        print("Using Video Swin Small model")
        return video_swin_s(pretrained=pretrained_path, device=device, **kwargs)
    elif 'large' in pretrained_path.lower():
        print("Using Video Swin Large model")
        return video_swin_l(pretrained=pretrained_path, device=device, **kwargs)
    else:
        print("Using Video Swin Tiny model")
        return video_swin_t(pretrained=pretrained_path, device=device, **kwargs)


def test_forward_pass(device: Optional[torch.device] = None):
    """
    Test 1: Basic forward pass through the model.

    Args:
        device (torch.device, optional): Device to run the test on. If None, will use the best available.
    """
    if device is None:
        device = get_device()

    print("=== Test 1: Forward Pass Sanity Check ===")
    print(f"Running on device: {device}")

    # Create input tensor directly on the specified device
    video = torch.randn(2, 3, 16, 224, 224, device=device)
    model = select_model_by_weights(PRETRAINED_PATH, device=device, num_classes=400)

    # Measure forward pass time
    start_time = time.time()

    # Use mixed precision if available and on CUDA
    if device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
        with torch.amp.autocast('cuda'):
            output = model(video)
    else:
        output = model(video)

    elapsed_time = time.time() - start_time

    print(f"Input shape: {video.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Forward pass time: {elapsed_time * 1000:.2f} ms")
    print("")

    # Clean up to avoid memory issues
    del model, video, output
    free_memory()


def test_window_partition_reverse(device: Optional[torch.device] = None):
    """
    Test 2: Validate window partition and reverse operations.

    Args:
        device (torch.device, optional): Device to run the test on. If None, will use the best available.
    """
    if device is None:
        device = get_device()

    print("=== Test 2: Window Partition and Reverse ===")
    print(f"Running on device: {device}")

    B, D, H, W, C = 1, 4, 4, 4, 1
    x = torch.arange(B * D * H * W * C, device=device).reshape(B, D, H, W, C).float()
    window_size = (2, 2, 2)

    windows = window_partition(x, window_size)
    x_reconstructed = window_reverse(windows, window_size, B, D, H, W)

    if torch.allclose(x, x_reconstructed):
        print("Window partition and reverse test passed.")
    else:
        print("Window partition and reverse test FAILED.")
        max_diff = torch.max(torch.abs(x - x_reconstructed)).item()
        print(f"Maximum difference: {max_diff}")
    print("")

    # Clean up
    del x, windows, x_reconstructed
    free_memory()


def test_overfitting(device: Optional[torch.device] = None):
    """
    Test 3: Check if the model can overfit to a small dataset.

    Args:
        device (torch.device, optional): Device to run the test on. If None, will use the best available.
    """
    if device is None:
        device = get_device()

    print("=== Test 3: Dummy Overfitting Test ===")
    print(f"Running on device: {device}")

    # Create data directly on device
    video = torch.randn(2, 3, 16, 224, 224, device=device)
    labels = torch.randint(0, 400, (2,), device=device)

    model = select_model_by_weights(PRETRAINED_PATH, device=device, num_classes=400)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Initialize scaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' and hasattr(torch.cuda, 'amp') else None

    num_epochs = 20
    losses = []

    total_time = 0
    for epoch in range(num_epochs):
        start_time = time.time()

        optimizer.zero_grad()

        # Use mixed precision if available and on CUDA
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(video)
                loss = criterion(outputs, labels)

            # Scale the loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(video)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - start_time
        total_time += epoch_time

        losses.append(loss.item())
        print(f"Epoch {epoch + 1:2d}/{num_epochs} - Loss: {loss.item():.4f} - Time: {epoch_time * 1000:.2f} ms")

    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Average epoch time: {total_time * 1000 / num_epochs:.2f} ms")
    print("")

    # Plot loss curve
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('overfitting_loss.png')
        plt.close()
        print("Saved loss curve to 'overfitting_loss.png'")
    except Exception as e:
        print(f"Warning: Could not save loss curve: {e}")

    # Clean up
    del model, video, labels, optimizer, criterion, scaler
    free_memory()


def test_feature_shapes(device: Optional[torch.device] = None):
    """
    Test 4: Check intermediate feature shapes throughout the network.

    Args:
        device (torch.device, optional): Device to run the test on. If None, will use the best available.
    """
    if device is None:
        device = get_device()

    print("=== Test 4: Intermediate Feature Inspection ===")
    print(f"Running on device: {device}")

    model = select_model_by_weights(PRETRAINED_PATH, device=device, num_classes=400)
    video = torch.randn(2, 3, 16, 224, 224, device=device)
    feature_shapes = {}

    def hook_fn(module, input, output, name):
        feature_shapes[name] = output.shape

    hooks = []
    # Register a hook on the patch embedding module.
    hooks.append(model.patch_embed.register_forward_hook(
        lambda m, i, o: hook_fn(m, i, o, 'patch_embed')))

    # Register hooks for each basic layer in the transformer.
    for i, layer in enumerate(model.layers):
        hooks.append(layer.register_forward_hook(
            lambda m, i, o, idx=i: hook_fn(m, i, o, f'layer_{idx}')))

    # Forward pass with mixed precision if available and on CUDA
    if device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
        with torch.amp.autocast('cuda'):
            _ = model(video)
    else:
        _ = model(video)

    # Print feature shapes
    for name, shape in feature_shapes.items():
        print(f"{name} output shape: {shape}")

    # Remove hooks
    for hook in hooks:
        hook.remove()

    print("")

    # Clean up
    del model, video, feature_shapes
    free_memory()


def test_attention_mask(device: Optional[torch.device] = None):
    """
    Test 5: Validate the attention mask computation.

    Args:
        device (torch.device, optional): Device to run the test on. If None, will use the best available.
    """
    if device is None:
        device = get_device()

    print("=== Test 5: Attention Mask Verification ===")
    print(f"Running on device: {device}")

    D, H, W = 8, 8, 8  # Small dimensions for easier inspection
    window_size = (2, 4, 4)
    shift_size = (1, 2, 2)

    try:
        mask = compute_mask(D, H, W, window_size, shift_size, device)
        print(f"Attention mask shape: {mask.shape}")

        # For large masks, just print a subset
        if mask.numel() > 100:
            print("Attention mask values (subset):")
            print(mask[:5, :5])
        else:
            print(f"Attention mask values:\n {mask}")
    except Exception as e:
        print(f"Error computing attention mask: {e}")

    print("")

    # Clean up
    if 'mask' in locals():
        del mask
    free_memory()


def test_pretrained_loading(device: Optional[torch.device] = None):
    """
    Test 6: Check pretrained weight loading.

    Args:
        device (torch.device, optional): Device to run the test on. If None, will use the best available.
    """
    if device is None:
        device = get_device()

    print("=== Test 6: Pretrained Weight Loading Test ===")
    print(f"Running on device: {device}")

    if PRETRAINED_PATH is None:
        print("No pretrained path provided. Skipping pretrained weight loading test.")
    else:
        try:
            # Track loading time
            start_time = time.time()
            model = select_model_by_weights(PRETRAINED_PATH, device=device, num_classes=400)
            loading_time = time.time() - start_time

            # Test forward pass
            video = torch.randn(2, 3, 16, 224, 224, device=device)

            # Forward pass with mixed precision if available and on CUDA
            if device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
                with torch.amp.autocast('cuda'):
                    output = model(video)
            else:
                output = model(video)

            print("Pretrained model loaded successfully.")
            print(f"Loading time: {loading_time:.2f} seconds")
            print(f"Output shape: {output.shape}")

            # Clean up
            del model, video, output
            free_memory()
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            import traceback
            print(traceback.format_exc())

    print("")


def test_eval_mode_consistency(device: Optional[torch.device] = None):
    """
    Test 7: Check if model gives consistent outputs in eval mode.

    Args:
        device (torch.device, optional): Device to run the test on. If None, will use the best available.
    """
    if device is None:
        device = get_device()

    print("=== Test 7: Evaluation Mode Consistency Test ===")
    print(f"Running on device: {device}")

    # Set a fixed seed for reproducibility
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed(42)

    video = torch.randn(1, 3, 16, 224, 224, device=device)

    model = select_model_by_weights(PRETRAINED_PATH, device=device, num_classes=400)
    model.eval()

    with torch.no_grad():
        # First forward pass
        if device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
            with torch.amp.autocast('cuda'):
                out1 = model(video)
        else:
            out1 = model(video)

        # Second forward pass
        if device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
            with torch.amp.autocast('cuda'):
                out2 = model(video)
        else:
            out2 = model(video)

    if torch.allclose(out1, out2, atol=1e-5):
        print("Eval mode consistency test passed.")
    else:
        print("Eval mode consistency test FAILED.")
        max_diff = torch.max(torch.abs(out1 - out2)).item()
        print(f"Maximum difference: {max_diff}")

    print("")

    # Clean up
    del model, video, out1, out2
    free_memory()


def test_dynamic_input_shape(device: Optional[torch.device] = None):
    """
    Test 8: Verify the model can handle varying input dimensions.

    Args:
        device (torch.device, optional): Device to run the test on. If None, will use the best available.
    """
    if device is None:
        device = get_device()

    print("=== Test 8: Dynamic Input Shape Test ===")
    print(f"Running on device: {device}")

    # Testing with different temporal dimensions
    video_16 = torch.randn(2, 3, 16, 224, 224, device=device)
    video_8 = torch.randn(2, 3, 8, 224, 224, device=device)

    # Different spatial dimensions
    video_192 = torch.randn(2, 3, 16, 192, 192, device=device)

    model = select_model_by_weights(PRETRAINED_PATH, device=device, num_classes=400)
    model.eval()

    with torch.no_grad():
        # Use mixed precision if available and on CUDA
        if device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
            with torch.amp.autocast('cuda'):
                out1 = model(video_16)
                out2 = model(video_8)
                out3 = model(video_192)
        else:
            out1 = model(video_16)
            out2 = model(video_8)
            out3 = model(video_192)

    print(f"Output shape with 16 frames, 224x224: {out1.shape}")
    print(f"Output shape with 8 frames, 224x224:  {out2.shape}")
    print(f"Output shape with 16 frames, 192x192: {out3.shape}")
    print("")

    # Clean up
    del model, video_16, video_8, video_192, out1, out2, out3
    free_memory()


@memory_profile
def test_memory_profile(device: Optional[torch.device] = None):
    """
    Test 9: Profile memory usage during forward and backward pass.

    Args:
        device (torch.device, optional): Device to run the test on. If None, will use the best available.
    """
    if device is None:
        device = get_device()

    print("=== Test 9: Memory Usage Profiling ===")
    print(f"Running on device: {device}")

    # Record initial memory usage
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        initial_mem = psutil.Process().memory_info().rss / (1024 ** 2)

    print(f"Initial memory usage: {initial_mem:.2f} MB")

    video = torch.randn(2, 3, 16, 224, 224, device=device)
    model = select_model_by_weights(PRETRAINED_PATH, device=device, num_classes=400)

    # Forward pass
    print("Performing forward pass...")

    # Use mixed precision if available and on CUDA
    if device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
        with torch.amp.autocast('cuda'):
            output = model(video)
    else:
        output = model(video)

    # Record memory after forward pass
    if device.type == 'cuda':
        forward_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        forward_mem = psutil.Process().memory_info().rss / (1024 ** 2)
        peak_mem = forward_mem  # No peak tracking for CPU

    print(f"Memory after forward pass: {forward_mem:.2f} MB")
    print(f"Peak memory usage: {peak_mem:.2f} MB")

    # Backward pass (if training)
    if model.training:
        print("Performing backward pass...")

        if device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
            with torch.amp.autocast('cuda'):
                loss = output.sum()

            # Use scaler for mixed precision
            scaler = torch.amp.GradScaler('cuda')
            scaler.scale(loss).backward()
        else:
            loss = output.sum()
            loss.backward()

        # Record memory after backward pass
        if device.type == 'cuda':
            backward_mem = torch.cuda.memory_allocated() / (1024 ** 2)
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            backward_mem = psutil.Process().memory_info().rss / (1024 ** 2)
            peak_mem = backward_mem  # No peak tracking for CPU

        print(f"Memory after backward pass: {backward_mem:.2f} MB")
        print(f"Peak memory usage: {peak_mem:.2f} MB")

    print(f"Output shape: {output.shape}")
    print("")

    # Clean up
    del model, video, output
    if 'loss' in locals():
        del loss
    if 'scaler' in locals():
        del scaler
    free_memory()


def test_speed_benchmark(device: Optional[torch.device] = None, batch_sizes=None, use_amp=True):
    """
    Test 10: Benchmark forward and backward pass speed across different batch sizes.

    Args:
        device (torch.device, optional): Device to run the test on. If None, will use the best available.
        batch_sizes (list, optional): List of batch sizes to test. If None, defaults to [1, 2, 4].
        use_amp (bool, optional): Whether to use automatic mixed precision on CUDA devices. Defaults to True.
    """
    if device is None:
        device = get_device()

    if batch_sizes is None:
        # Default batch sizes - adjust based on memory constraints
        if device.type == 'cuda':
            batch_sizes = [1, 2, 4]
        else:
            batch_sizes = [1, 2]  # Smaller batches for CPU

    print("=== Test 10: Speed Benchmark ===")
    print(f"Running on device: {device}")
    print(f"Testing batch sizes: {batch_sizes}")

    # Check if mixed precision is available
    amp_available = device.type == 'cuda' and hasattr(torch.cuda, 'amp')

    if use_amp and amp_available:
        print("Using Automatic Mixed Precision (AMP)")
    else:
        use_amp = False
        if amp_available:
            print("AMP available but not used")
        else:
            print("AMP not available")

    # Create model
    model = select_model_by_weights(PRETRAINED_PATH, device=device, num_classes=400)

    results = []

    for batch_size in batch_sizes:
        print(f"\nBenchmarking with batch size {batch_size}:")

        # Create input with current batch size
        video = torch.randn(batch_size, 3, 16, 224, 224, device=device)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        _ = model(video)
                else:
                    _ = model(video)

        # Synchronize if using CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Forward pass timing
        num_iterations = 10
        forward_times = []

        for _ in range(num_iterations):
            # Synchronize before timing
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.time()

            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        _ = model(video)
                else:
                    _ = model(video)

            # Synchronize after timing
            if device.type == 'cuda':
                torch.cuda.synchronize()

            end = time.time()
            forward_times.append((end - start) * 1000)  # Convert to ms

        # Calculate statistics
        forward_time_mean = np.mean(forward_times)
        forward_time_std = np.std(forward_times)

        # Backward pass timing (if training)
        backward_times = []

        if model.training:
            for _ in range(num_iterations):
                # Create a fresh optimizer for each iteration
                optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
                optimizer.zero_grad()

                # Synchronize before timing
                if device.type == 'cuda':
                    torch.cuda.synchronize()

                start = time.time()

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(video)
                        loss = outputs.sum()

                    # Use scaler for mixed precision
                    scaler = torch.amp.GradScaler('cuda')
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(video)
                    loss = outputs.sum()
                    loss.backward()
                    optimizer.step()

                # Synchronize after timing
                if device.type == 'cuda':
                    torch.cuda.synchronize()

                end = time.time()
                backward_times.append((end - start) * 1000)  # Convert to ms

                # Clean up
                del optimizer
                if 'scaler' in locals():
                    del scaler

            # Calculate statistics
            backward_time_mean = np.mean(backward_times)
            backward_time_std = np.std(backward_times)

        # Print and store results
        print(f"Forward pass time (mean ± std over {num_iterations} runs): "
              f"{forward_time_mean:.2f} ± {forward_time_std:.2f} ms")

        if model.training and backward_times:
            print(f"Backward pass time (mean ± std over {num_iterations} runs): "
                  f"{backward_time_mean:.2f} ± {backward_time_std:.2f} ms")
            print(f"Total time per batch (forward + backward): "
                  f"{forward_time_mean + backward_time_mean:.2f} ms")

        result = {
            "batch_size": batch_size,
            "forward_time_ms": forward_time_mean,
            "forward_time_std_ms": forward_time_std
        }

        if model.training and backward_times:
            result.update({
                "backward_time_ms": backward_time_mean,
                "backward_time_std_ms": backward_time_std,
                "total_time_ms": forward_time_mean + backward_time_mean
            })

        results.append(result)

        # Clean up
        del video
        free_memory()

    print("\nSummary of benchmark results:")
    print("-----------------------------")
    for result in results:
        print(f"Batch size: {result['batch_size']}")
        print(f"  Forward pass: {result['forward_time_ms']:.2f} ± {result['forward_time_std_ms']:.2f} ms")
        if 'backward_time_ms' in result:
            print(f"  Backward pass: {result['backward_time_ms']:.2f} ± {result['backward_time_std_ms']:.2f} ms")
            print(f"  Total time: {result['total_time_ms']:.2f} ms")
        print("")

    # Try to plot the results
    try:
        batch_sizes = [r["batch_size"] for r in results]
        forward_times = [r["forward_time_ms"] for r in results]

        plt.figure(figsize=(10, 6))
        plt.bar(batch_sizes, forward_times, alpha=0.7)
        plt.xlabel("Batch Size")
        plt.ylabel("Time (ms)")
        plt.title(f"Forward Pass Time vs Batch Size on {device.type.upper()}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        if all('backward_time_ms' in r for r in results):
            backward_times = [r["backward_time_ms"] for r in results]
            plt.figure(figsize=(10, 6))
            plt.bar([str(b) + ' (fwd)' for b in batch_sizes], forward_times, alpha=0.7)
            plt.bar([str(b) + ' (bwd)' for b in batch_sizes], backward_times, alpha=0.7)
            plt.xlabel("Batch Size and Operation")
            plt.ylabel("Time (ms)")
            plt.title(f"Forward and Backward Pass Time vs Batch Size on {device.type.upper()}")
            plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('speed_benchmark.png')
        plt.close()
        print("Saved benchmark visualization to 'speed_benchmark.png'")
    except Exception as e:
        print(f"Warning: Could not create benchmark plot: {e}")

    # Clean up
    del model
    free_memory()


def test_attention_visualization(device: Optional[torch.device] = None):
    """
    Test 11: Visualize real attention maps from the model.

    Args:
        device (torch.device, optional): Device to run the test on. If None, will use the best available.
    """
    if device is None:
        device = get_device()

    print("=== Test 11: Real Attention Map Visualization ===")
    print(f"Running on device: {device}")

    # Create a sample input
    video = torch.randn(1, 3, 8, 224, 224, device=device)

    # Load model
    model = select_model_by_weights(PRETRAINED_PATH, device=device, num_classes=400)
    model.eval()

    # Dictionary to store attention maps
    attention_maps = {}

    # Define hook function to capture attention weights
    def attention_hook(module, input_tensors, output, name):
        """Hook to extract attention weights from WindowAttention3D modules."""
        # The attention softmax is applied inside the forward function of WindowAttention3D
        # We need to extract the attention weights after softmax but before they're applied to values

        # For WindowAttention3D, we can reconstruct the attention from the Q, K matrices
        # Get Q and K from the input
        if hasattr(module, 'qkv') and len(input_tensors) > 0:
            x = input_tensors[0]  # Input to the attention module
            B_, N, C = x.shape

            # Extract QKV projections
            qkv = module.qkv(x)
            qkv = qkv.reshape(B_, N, 3, module.num_heads, C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

            # Compute attention maps
            q = q * module.scale
            attn = (q @ k.transpose(-2, -1))  # B_, nH, N, N

            # Apply relative position bias if available
            if hasattr(module, 'relative_position_bias_table') and hasattr(module, 'relative_position_index'):
                try:
                    relative_position_bias = module.relative_position_bias_table[
                        module.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)
                    relative_position_bias = relative_position_bias.permute(2, 0, 1)  # nH, N, N
                    attn = attn + relative_position_bias.unsqueeze(0)
                except Exception as e:
                    print(f"Warning: Couldn't apply position bias: {e}")

            # Apply softmax to get attention weights
            attn = torch.nn.functional.softmax(attn, dim=-1)

            # Store attention map (first batch only, all heads)
            attention_maps[name] = attn[0].detach().cpu().numpy()  # nH, N, N

    # Register hooks for attention modules in the first layer
    hooks = []
    hook_idx = 0

    # Register hooks to capture attention in the first few blocks
    for layer_idx, layer in enumerate(model.layers):
        if layer_idx > 1:  # Only use first two layers to avoid too many visualizations
            continue

        for block_idx, block in enumerate(layer.blocks):
            if block_idx > 1:  # Only use first two blocks per layer
                continue

            if hasattr(block, 'attn'):
                hook_name = f"layer{layer_idx}_block{block_idx}"
                hooks.append(
                    block.attn.register_forward_hook(
                        lambda mod, inp, out, name=hook_name: attention_hook(mod, inp, out, name)
                    )
                )
                hook_idx += 1

    print(f"Registered {len(hooks)} attention hooks")

    # Forward pass to extract attention weights
    try:
        with torch.no_grad():
            # Use mixed precision if available
            if device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
                with torch.amp.autocast(device_type='cuda'):
                    _ = model(video)
            else:
                _ = model(video)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # If we captured attention maps, visualize them
        if attention_maps:
            print(f"Successfully captured {len(attention_maps)} attention maps")

            # Create grid of plots for each layer/block
            num_layers = len(attention_maps)
            num_heads_to_show = min(8, list(attention_maps.values())[0].shape[0])  # Show at most 8 heads

            # Create figure with subplots
            fig_rows = num_layers
            fig_cols = num_heads_to_show
            fig, axs = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 2.5, fig_rows * 2.5))

            # If only one row, wrap in list for indexing consistency
            if fig_rows == 1:
                axs = [axs]

            # Plot attention maps
            for i, (name, attn_map) in enumerate(attention_maps.items()):
                for j in range(num_heads_to_show):
                    # Get attention map for this head
                    head_map = attn_map[j]

                    # For better visualization, extract a central portion (if map is large)
                    map_size = head_map.shape[0]
                    if map_size > 32:
                        # Take a central portion
                        center = map_size // 2
                        size = min(32, map_size // 2)
                        head_map = head_map[center - size:center + size, center - size:center + size]

                    # Plot
                    ax = axs[i][j]
                    im = ax.imshow(head_map, cmap='viridis')
                    ax.set_title(f'{name}\nHead {j + 1}', fontsize=10)
                    ax.axis('off')

            # Add colorbar
            fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8)

            # Adjust layout
            plt.subplots_adjust(wspace=0.05, hspace=0.2)

            # Save figure
            plt.savefig('real_attention_maps.png', dpi=150, bbox_inches='tight')
            plt.close()

            print("Saved real attention maps to 'real_attention_maps.png'")
        else:
            print("No attention maps were captured.")

    except Exception as e:
        print(f"Error visualizing attention maps: {e}")
        import traceback
        print(traceback.format_exc())

    print("")

    # Clean up
    del model, video
    if 'attention_maps' in locals():
        del attention_maps
    free_memory()


def test_custom_head(device: Optional[torch.device] = None):
    """
    Test 12: Test model with custom classification head.

    Args:
        device (torch.device, optional): Device to run the test on. If None, will use the best available.
    """
    if device is None:
        device = get_device()

    print("=== Test 12: Custom Classification Head ===")
    print(f"Running on device: {device}")

    if PRETRAINED_PATH is None:
        print("No pretrained path provided. Using random initialization.")

    # Create model with custom head for classification
    model_type = 'base' if PRETRAINED_PATH and 'base' in PRETRAINED_PATH.lower() else 'tiny'

    if model_type == 'base':
        model = video_swin_b_custom(
            pretrained=PRETRAINED_PATH,
            num_classes=400,
            head_dropout=0.2,
            device=device
        )
    else:
        model = video_swin_t_custom(
            pretrained=PRETRAINED_PATH,
            num_classes=400,
            head_dropout=0.2,
            device=device
        )

    # Test forward pass
    video = torch.randn(2, 3, 16, 224, 224, device=device)

    # Use mixed precision if available and on CUDA
    if device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
        with torch.amp.autocast('cuda'):
            output = model(video)
    else:
        output = model(video)

    print(f"Classification output shape: {output.shape}")

    # Test feature extraction mode
    model.set_feature_mode(True)

    if device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
        with torch.amp.autocast('cuda'):
            features = model(video)
    else:
        features = model(video)

    print(f"Feature extraction output shape: {features.shape}")

    # Reset head with new parameters
    model.reset_head(num_classes=1000, dropout_rate=0.5)
    model.set_feature_mode(False)

    if device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
        with torch.amp.autocast('cuda'):
            new_output = model(video)
    else:
        new_output = model(video)

    print(f"Output shape after head reset: {new_output.shape}")
    print("")

    # Clean up
    del model, video, output, features, new_output
    free_memory()


def create_dummy_video(save_path: str = "dummy_video.mp4", frames: int = 16,
                       width: int = 224, height: int = 224):
    """
    Create a dummy video file for testing.

    Args:
        save_path (str): Path to save the video
        frames (int): Number of frames
        width (int): Frame width
        height (int): Frame height

    Returns:
        str: Path to the created video file
    """
    print(f"Creating dummy video with {frames} frames at {width}x{height}...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))

    for i in range(frames):
        # Create a gradient frame with frame number
        frame = np.ones((height, width, 3), dtype=np.uint8) * (i * 15)
        frame = frame.astype(np.uint8)

        # Add frame number text
        cv2.putText(frame, f"Frame {i + 1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)

    out.release()
    print(f"Dummy video created at {save_path}")
    return save_path


def load_video(video_path: str, num_frames: int = 16,
               height: int = 224, width: int = 224,
               device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Load a video file as a tensor.

    Args:
        video_path (str): Path to the video file
        num_frames (int): Number of frames to extract
        height (int): Target height
        width (int): Target width
        device (torch.device, optional): Device to place the tensor on. If None, will use CUDA if available.

    Returns:
        torch.Tensor: Video tensor of shape [C, T, H, W]
    """
    if device is None:
        device = get_device()

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"Loading video from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))

        # Normalize to [0, 1]
        frame = frame / 255.0
        frames.append(frame)

    cap.release()

    # Stack frames and convert to tensor
    if not frames:
        raise ValueError("No frames were extracted from the video")

    frames = np.stack(frames, axis=0)  # T, H, W, C
    frames = np.transpose(frames, (3, 0, 1, 2))  # C, T, H, W

    tensor = torch.tensor(frames, dtype=torch.float32, device=device)
    print(f"Video loaded with shape: {tensor.shape}")
    return tensor


def test_real_video(device: Optional[torch.device] = None):
    """
    Test 13: Test the model on a simple video file.

    Args:
        device (torch.device, optional): Device to run the test on. If None, will use the best available.
    """
    if device is None:
        device = get_device()

    print("=== Test 13: Real Video Test ===")
    print(f"Running on device: {device}")

    # Create a dummy video file
    try:
        video_path = create_dummy_video()

        # Load the video as a tensor
        video_tensor = load_video(video_path, device=device)

        # Add batch dimension
        video_tensor = video_tensor.unsqueeze(0)  # 1, C, T, H, W

        # Forward pass through model
        model = select_model_by_weights(PRETRAINED_PATH, device=device, num_classes=400)
        model.eval()

        print("Running inference on video...")
        with torch.no_grad():
            # Use mixed precision if available and on CUDA
            if device.type == 'cuda' and hasattr(torch.cuda, 'amp'):
                with torch.amp.autocast('cuda'):
                    start_time = time.time()
                    output = model(video_tensor)
                    inference_time = time.time() - start_time
            else:
                start_time = time.time()
                output = model(video_tensor)
                inference_time = time.time() - start_time

        # Get top predictions
        probabilities = F.softmax(output, dim=1)
        top_prob, top_class = torch.topk(probabilities, 5)

        print(f"Inference completed in {inference_time:.3f} seconds")
        print("Top 5 predictions:")
        for i in range(5):
            print(f"  Class {top_class[0][i].item()}: {top_prob[0][i].item() * 100:.2f}%")

        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"Removed temporary video file: {video_path}")

    except Exception as e:
        print(f"Error in real video test: {e}")
        import traceback
        print(traceback.format_exc())

    print("")

    # Clean up
    if 'model' in locals():
        del model
    if 'video_tensor' in locals():
        del video_tensor
    if 'output' in locals():
        del output
    free_memory()


def test_cpu_vs_gpu():
    """Test 14: Compare CPU vs GPU performance."""
    print("=== Test 14: CPU vs GPU Performance Comparison ===")

    if not torch.cuda.is_available():
        print("CUDA not available. Skipping CPU vs GPU comparison.")
        return

    # Define batch size and iterations
    batch_size = 1
    num_iterations = 5

    # Create input tensor (on CPU initially)
    video = torch.randn(batch_size, 3, 16, 224, 224)

    # Test on CPU
    cpu_device = torch.device('cpu')
    cpu_model = select_model_by_weights(PRETRAINED_PATH, device=cpu_device, num_classes=400)
    cpu_model.eval()

    print("Running CPU inference...")
    cpu_times = []
    with torch.no_grad():
        # Warmup
        _ = cpu_model(video)

        for _ in range(num_iterations):
            start_time = time.time()
            _ = cpu_model(video)
            cpu_times.append(time.time() - start_time)

    cpu_mean = np.mean(cpu_times)
    cpu_std = np.std(cpu_times)

    # Clean up CPU resources
    del cpu_model

    # Test on GPU
    gpu_device = torch.device('cuda')
    gpu_model = select_model_by_weights(PRETRAINED_PATH, device=gpu_device, num_classes=400)
    gpu_model.eval()

    # Move input to GPU
    gpu_video = video.to(gpu_device)

    print("Running GPU inference...")
    gpu_times = []
    gpu_times_amp = []

    with torch.no_grad():
        # Warmup
        _ = gpu_model(gpu_video)
        torch.cuda.synchronize()

        # Standard precision
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = gpu_model(gpu_video)
            torch.cuda.synchronize()
            gpu_times.append(time.time() - start_time)

        # Mixed precision
        if hasattr(torch.cuda, 'amp'):
            print("Running GPU inference with AMP...")
            # Warmup with AMP
            with torch.amp.autocast('cuda'):
                _ = gpu_model(gpu_video)
            torch.cuda.synchronize()

            for _ in range(num_iterations):
                torch.cuda.synchronize()
                start_time = time.time()
                with torch.amp.autocast('cuda'):
                    _ = gpu_model(gpu_video)
                torch.cuda.synchronize()
                gpu_times_amp.append(time.time() - start_time)

    gpu_mean = np.mean(gpu_times)
    gpu_std = np.std(gpu_times)

    if gpu_times_amp:
        gpu_mean_amp = np.mean(gpu_times_amp)
        gpu_std_amp = np.std(gpu_times_amp)

    # Print results
    print("\nPerformance Comparison (average over {} iterations):".format(num_iterations))
    print(f"CPU: {cpu_mean:.4f} ± {cpu_std:.4f} s")
    print(f"GPU: {gpu_mean:.4f} ± {gpu_std:.4f} s")
    if gpu_times_amp:
        print(f"GPU with AMP: {gpu_mean_amp:.4f} ± {gpu_std_amp:.4f} s")

    speedup = cpu_mean / gpu_mean
    print(f"GPU Speedup over CPU: {speedup:.2f}x")

    if gpu_times_amp:
        speedup_amp = cpu_mean / gpu_mean_amp
        speedup_over_gpu = gpu_mean / gpu_mean_amp
        print(f"GPU+AMP Speedup over CPU: {speedup_amp:.2f}x")
        print(f"AMP Speedup over standard GPU: {speedup_over_gpu:.2f}x")

    # Visualize
    try:
        labels = ['CPU', 'GPU']
        times = [cpu_mean, gpu_mean]
        errors = [cpu_std, gpu_std]

        if gpu_times_amp:
            labels.append('GPU+AMP')
            times.append(gpu_mean_amp)
            errors.append(gpu_std_amp)

        plt.figure(figsize=(10, 6))
        plt.bar(labels, times, yerr=errors, alpha=0.7)
        plt.ylabel('Time (s)')
        plt.title('Inference Time Comparison')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add actual times as text
        for i, (time_val, err) in enumerate(zip(times, errors)):
            plt.text(i, time_val + err + 0.01, f"{time_val:.4f} s",
                     ha='center', va='bottom', fontweight='bold')

        plt.savefig('cpu_vs_gpu_comparison.png')
        plt.close()
        print("Saved comparison visualization to 'cpu_vs_gpu_comparison.png'")
    except Exception as e:
        print(f"Warning: Could not create comparison plot: {e}")

    # Clean up
    del gpu_model, video, gpu_video
    free_memory()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Video Swin Transformer implementation")

    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights file')
    parser.add_argument('--test', type=int, default=0,
                        help='Run a specific test (0 for all tests)')
    parser.add_argument('--use-cpu', action='store_true',
                        help='Force using CPU even if CUDA is available')
    parser.add_argument('--disable-amp', action='store_true',
                        help='Disable Automatic Mixed Precision (AMP) even if available')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=None,
                        help='Batch sizes to use for benchmarking')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Set global pretrained path
    global PRETRAINED_PATH
    if args.pretrained is not None:
        PRETRAINED_PATH = args.pretrained

    # Determine device
    if args.use_cpu:
        device = torch.device('cpu')
        print("Forcing CPU use as requested.")
    else:
        device = get_device()

    # Print device info
    print_device_info()

    # Set AMP flag
    use_amp = not args.disable_amp

    # Run specific test or all tests
    if args.test == 0:
        # Run all tests
        test_forward_pass(device)
        test_window_partition_reverse(device)
        test_overfitting(device)
        test_feature_shapes(device)
        test_attention_mask(device)
        test_pretrained_loading(device)
        test_eval_mode_consistency(device)
        test_dynamic_input_shape(device)
        test_memory_profile(device)
        test_speed_benchmark(device, args.batch_sizes, use_amp)
        test_attention_visualization(device)
        test_custom_head(device)
        test_real_video(device)

        # Run CPU vs GPU comparison if both are available
        if torch.cuda.is_available() and not args.use_cpu:
            test_cpu_vs_gpu()
    else:
        # Run a specific test
        test_funcs = {
            1: test_forward_pass,
            2: test_window_partition_reverse,
            3: test_overfitting,
            4: test_feature_shapes,
            5: test_attention_mask,
            6: test_pretrained_loading,
            7: test_eval_mode_consistency,
            8: test_dynamic_input_shape,
            9: test_memory_profile,
            10: lambda d: test_speed_benchmark(d, args.batch_sizes, use_amp),
            11: test_attention_visualization,
            12: test_custom_head,
            13: test_real_video,
            14: test_cpu_vs_gpu if torch.cuda.is_available() and not args.use_cpu else lambda: print(
                "CUDA not available. Skipping CPU vs GPU comparison.")
        }

        if args.test in test_funcs:
            print(f"\nRunning test {args.test}...")
            test_funcs[args.test](device)
        else:
            print(f"Invalid test number: {args.test}. Valid range: 1-14.")


if __name__ == '__main__':
    main()

    """
    Example command line use: python test_vst.py --test 11
    """