#test_vst.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import the Video Swin Transformer implementation
from vst import (
    video_swin_t,
    video_swin_s,
    video_swin_l,
    video_swin_b,
    window_partition,
    window_reverse,
    compute_mask
)

# Set pretrained_path to your pretrained .pth file if available,
# or leave it as None to run without pretrained weights.
PRETRAINED_PATH = None
#PRETRAINED_PATH = r"C:\Users\MartinParkerR\PycharmProjects\Video_Swin_Transformer\checkpoints\pretrained_vst\swin_base_patch244_window877_kinetics600_22k.pth"
#PRETRAINED_PATH = r"C:\Users\MartinParkerR\PycharmProjects\Video_Swin_Transformer\checkpoints\pretrained_vst\swin_small_patch244_window877_kinetics400_1k.pth"
PRETRAINED_PATH = r"/checkpoints/pretrained_vst/swin_tiny_patch244_window877_kinetics400_1k.pth"

# Example: PRETRAINED_PATH = "path/to/swin_tiny_patch4_window7_224.pth"


def test_forward_pass():
    print("=== Test 1: Forward Pass Sanity Check ===")
    video = torch.randn(2, 3, 16, 224, 224)

    # Determine which model to use based on the pretrained weights path
    if PRETRAINED_PATH and 'base' in PRETRAINED_PATH.lower():
        print("Using Video Swin Base model")
        model = video_swin_b(pretrained=PRETRAINED_PATH, num_classes=400)
    elif PRETRAINED_PATH and 'small' in PRETRAINED_PATH.lower():
        print("Using Video Swin Small model")
        model = video_swin_s(pretrained=PRETRAINED_PATH, num_classes=400)
    elif PRETRAINED_PATH and 'large' in PRETRAINED_PATH.lower():
        print("Using Video Swin Large model")
        model = video_swin_l(pretrained=PRETRAINED_PATH, num_classes=400)
    else:
        print("Using Video Swin Tiny model")
        model = video_swin_t(pretrained=PRETRAINED_PATH, num_classes=400)

    output = model(video)
    print(f"Input shape:  {video.shape}")
    print(f"Output shape: {output.shape}")
    print()


def test_window_partition_reverse():
    print("=== Test 2: Window Partition and Reverse ===")
    B, D, H, W, C = 1, 4, 4, 4, 1
    x = torch.arange(B * D * H * W * C).reshape(B, D, H, W, C).float()
    window_size = (2, 2, 2)
    windows = window_partition(x, window_size)
    x_reconstructed = window_reverse(windows, window_size, B, D, H, W)
    if torch.allclose(x, x_reconstructed):
        print("Window partition and reverse test passed.")
    else:
        print("Window partition and reverse test FAILED.")
    print()


def test_overfitting():
    print("=== Test 3: Dummy Overfitting Test ===")
    video = torch.randn(2, 3, 16, 224, 224)
    labels = torch.randint(0, 400, (2,))

    # Choose the appropriate model based on the pretrained path
    if PRETRAINED_PATH and 'base' in PRETRAINED_PATH.lower():
        print("Using Video Swin Base model")
        model = video_swin_b(pretrained=PRETRAINED_PATH, num_classes=400)
    elif PRETRAINED_PATH and 'small' in PRETRAINED_PATH.lower():
        print("Using Video Swin Small model")
        model = video_swin_s(pretrained=PRETRAINED_PATH, num_classes=400)
    elif PRETRAINED_PATH and 'large' in PRETRAINED_PATH.lower():
        print("Using Video Swin Large model")
        model = video_swin_l(pretrained=PRETRAINED_PATH, num_classes=400)
    else:
        print("Using Video Swin Tiny model")
        model = video_swin_t(pretrained=PRETRAINED_PATH, num_classes=400)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 20
    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(video)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Epoch {epoch + 1:2d}/{num_epochs} - Loss: {loss.item():.4f}")
    print("Final loss:", losses[-1])
    print()


def test_feature_shapes():
    print("=== Test 4: Intermediate Feature Inspection ===")

    # Choose appropriate model
    if PRETRAINED_PATH and 'base' in PRETRAINED_PATH.lower():
        print("Using Video Swin Base model")
        model = video_swin_b(pretrained=PRETRAINED_PATH, num_classes=400)
    elif PRETRAINED_PATH and 'small' in PRETRAINED_PATH.lower():
        print("Using Video Swin Small model")
        model = video_swin_s(pretrained=PRETRAINED_PATH, num_classes=400)
    elif PRETRAINED_PATH and 'large' in PRETRAINED_PATH.lower():
        print("Using Video Swin Large model")
        model = video_swin_l(pretrained=PRETRAINED_PATH, num_classes=400)
    else:
        print("Using Video Swin Tiny model")
        model = video_swin_t(pretrained=PRETRAINED_PATH, num_classes=400)

    video = torch.randn(2, 3, 16, 224, 224)
    hooks = []

    def print_hook(module, input, output):
        print(f"{module.__class__.__name__} output shape: {output.shape}")

    # Register a hook on the patch embedding module.
    hooks.append(model.patch_embed.register_forward_hook(print_hook))
    # Register hooks for each basic layer in the transformer.
    for i, layer in enumerate(model.layers):
        hooks.append(layer.register_forward_hook(print_hook))

    _ = model(video)

    for hook in hooks:
        hook.remove()
    print()


def test_attention_mask():
    print("=== Test 5: Attention Mask Verification ===")
    D, H, W = 8, 8, 8  # Small dimensions for easier inspection
    window_size = (2, 4, 4)
    shift_size = (1, 2, 2)
    mask = compute_mask(D, H, W, window_size, shift_size, device='cpu')
    print("Attention mask shape:", mask.shape)
    print("Attention mask values:\n", mask)
    print()


def test_pretrained_loading():
    print("=== Test 6: Pretrained Weight Loading Test ===")
    if PRETRAINED_PATH is None:
        print("No pretrained path provided. Skipping pretrained weight loading test.")
    else:
        try:
            # Choose appropriate model
            if 'base' in PRETRAINED_PATH.lower():
                print("Using Video Swin Base model")
                model = video_swin_b(pretrained=PRETRAINED_PATH, num_classes=400)
            elif 'small' in PRETRAINED_PATH.lower():
                print("Using Video Swin Small model")
                model = video_swin_s(pretrained=PRETRAINED_PATH, num_classes=400)
            elif 'large' in PRETRAINED_PATH.lower():
                print("Using Video Swin Large model")
                model = video_swin_l(pretrained=PRETRAINED_PATH, num_classes=400)
            else:
                print("Using Video Swin Tiny model")
                model = video_swin_t(pretrained=PRETRAINED_PATH, num_classes=400)

            video = torch.randn(2, 3, 16, 224, 224)
            output = model(video)
            print("Pretrained model loaded successfully.")
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print("Error loading pretrained weights:", e)
    print()


def test_eval_mode_consistency():
    print("=== Test 7: Evaluation Mode Consistency Test ===")
    torch.manual_seed(42)
    video = torch.randn(1, 3, 16, 224, 224)

    # Choose appropriate model
    if PRETRAINED_PATH and 'base' in PRETRAINED_PATH.lower():
        print("Using Video Swin Base model")
        model = video_swin_b(pretrained=PRETRAINED_PATH, num_classes=400)
    elif PRETRAINED_PATH and 'small' in PRETRAINED_PATH.lower():
        print("Using Video Swin Small model")
        model = video_swin_s(pretrained=PRETRAINED_PATH, num_classes=400)
    elif PRETRAINED_PATH and 'large' in PRETRAINED_PATH.lower():
        print("Using Video Swin Large model")
        model = video_swin_l(pretrained=PRETRAINED_PATH, num_classes=400)
    else:
        print("Using Video Swin Tiny model")
        model = video_swin_t(pretrained=PRETRAINED_PATH, num_classes=400)

    model.eval()
    with torch.no_grad():
        out1 = model(video)
        out2 = model(video)
    if torch.allclose(out1, out2, atol=1e-5):
        print("Eval mode consistency test passed.")
    else:
        print("Eval mode consistency test FAILED.")
    print()


def test_torchscript_conversion():
    print("=== Test 8: TorchScript Conversion Test ===")
    video = torch.randn(2, 3, 16, 224, 224)

    # Choose appropriate model
    if PRETRAINED_PATH and 'base' in PRETRAINED_PATH.lower():
        print("Using Video Swin Base model")
        model = video_swin_b(pretrained=PRETRAINED_PATH, num_classes=400)
    elif PRETRAINED_PATH and 'small' in PRETRAINED_PATH.lower():
        print("Using Video Swin Small model")
        model = video_swin_s(pretrained=PRETRAINED_PATH, num_classes=400)
    elif PRETRAINED_PATH and 'large' in PRETRAINED_PATH.lower():
        print("Using Video Swin Large model")
        model = video_swin_l(pretrained=PRETRAINED_PATH, num_classes=400)
    else:
        print("Using Video Swin Tiny model")
        model = video_swin_t(pretrained=PRETRAINED_PATH, num_classes=400)

    try:
        scripted_model = torch.jit.script(model)
        scripted_output = scripted_model(video)
        print("TorchScript conversion succeeded.")
        print(f"Scripted model output shape: {scripted_output.shape}")
    except Exception as e:
        print("TorchScript conversion FAILED:", e)
    print()


def test_dynamic_input_shape():
    print("=== Test 9: Dynamic Input Shape Test ===")
    # Testing with different temporal dimensions
    video_16 = torch.randn(2, 3, 16, 224, 224)
    video_8 = torch.randn(2, 3, 8, 224, 224)

    # Choose appropriate model
    if PRETRAINED_PATH and 'base' in PRETRAINED_PATH.lower():
        print("Using Video Swin Base model")
        model = video_swin_b(pretrained=PRETRAINED_PATH, num_classes=400)
    elif PRETRAINED_PATH and 'small' in PRETRAINED_PATH.lower():
        print("Using Video Swin Small model")
        model = video_swin_s(pretrained=PRETRAINED_PATH, num_classes=400)
    elif PRETRAINED_PATH and 'large' in PRETRAINED_PATH.lower():
        print("Using Video Swin Large model")
        model = video_swin_l(pretrained=PRETRAINED_PATH, num_classes=400)
    else:
        print("Using Video Swin Tiny model")
        model = video_swin_t(pretrained=PRETRAINED_PATH, num_classes=400)

    out1 = model(video_16)
    out2 = model(video_8)
    print(f"Output shape with 16 frames: {out1.shape}")
    print(f"Output shape with 8 frames:  {out2.shape}")
    print()


if __name__ == '__main__':
    test_forward_pass()
    print("-" * 50)
    test_window_partition_reverse()
    print("-" * 50)
    test_overfitting()
    print("-" * 50)
    test_feature_shapes()
    print("-" * 50)
    test_attention_mask()
    print("-" * 50)
    test_pretrained_loading()
    print("-" * 50)
    test_eval_mode_consistency()
    print("-" * 50)
    test_torchscript_conversion()
    print("-" * 50)
    test_dynamic_input_shape()