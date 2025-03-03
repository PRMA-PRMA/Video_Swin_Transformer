#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video Swin Transformer Export Script
----------------------------------
Convert VST models to deployment formats (ONNX, TorchScript).
"""

import os
import argparse
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# Import VST implementation
from vst import video_swin_t, video_swin_s, video_swin_b, video_swin_l, get_device
from vst_custom import create_custom_model


def load_model(checkpoint_path: str, device: torch.device, num_classes: int = 400,
               model_name: str = 'tiny', use_custom_head: bool = False):
    """
    Load a trained VST model.

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


def export_to_torchscript(model: torch.nn.Module, output_path: str, example_input: torch.Tensor,
                          scripting: bool = True):
    """
    Export model to TorchScript format.

    Args:
        model: Model to export
        output_path: Path to save the TorchScript model
        example_input: Example input tensor
        scripting: Whether to use scripting (True) or tracing (False)
    """
    try:
        print(f"Exporting model to TorchScript format ({output_path})")
        model.eval()

        # TorchScript via scripting or tracing
        if scripting:
            # Create scriptable model
            # Some models may need modifications to be scriptable
            torchscript_model = torch.jit.script(model)
        else:
            # Create traced model
            with torch.no_grad():
                torchscript_model = torch.jit.trace(model, example_input)

                # Test traced model
                orig_output = model(example_input)
                traced_output = torchscript_model(example_input)
                assert torch.allclose(orig_output, traced_output, rtol=1e-3, atol=1e-3), \
                    "Traced model outputs differ from original model!"

        # Save TorchScript model
        torchscript_model.save(output_path)
        print(f"TorchScript model saved to {output_path}")

    except Exception as e:
        print(f"Error exporting to TorchScript: {e}")
        raise


def export_to_onnx(model: torch.nn.Module, output_path: str, example_input: torch.Tensor, opset_version: int = 12):
    """
    Export model to ONNX format.

    Args:
        model: Model to export
        output_path: Path to save the ONNX model
        example_input: Example input tensor
        opset_version: ONNX opset version
    """
    try:
        print(f"Exporting model to ONNX format ({output_path})")
        model.eval()

        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                model,
                example_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )

        # Verify exported model
        try:
            import onnx
            import onnxruntime as ort

            # Check ONNX model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)

            # Compare ONNX and PyTorch outputs
            ort_session = ort.InferenceSession(output_path)

            # Prepare inputs
            ort_inputs = {ort_session.get_inputs()[0].name: example_input.numpy()}

            # Run ONNX model
            ort_outputs = ort_session.run(None, ort_inputs)

            # Check outputs (ONNX vs PyTorch)
            pytorch_output = model(example_input).detach().numpy()

            # Check if outputs are close
            np.testing.assert_allclose(ort_outputs[0], pytorch_output, rtol=1e-3, atol=1e-3)
            print("ONNX model verified: PyTorch and ONNX outputs match")

        except ImportError:
            print("Warning: onnx or onnxruntime not installed. Cannot verify the exported model.")
        except Exception as e:
            print(f"Warning: ONNX verification failed: {e}")

        print(f"ONNX model saved to {output_path}")

    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        raise


def export_model_metadata(model: torch.nn.Module, input_shape: Tuple[int, ...],
                          output_path: str, labels: Optional[List[str]] = None,
                          model_name: str = "VST", model_config: Dict = None):
    """
    Export model metadata.

    Args:
        model: Model to export
        input_shape: Input tensor shape
        output_path: Path to save metadata
        labels: Optional list of class labels
        model_name: Name of the model
        model_config: Model configuration
    """
    try:
        # Get number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Get input/output shapes
        example_input = torch.randn(input_shape).to(next(model.parameters()).device)
        with torch.no_grad():
            example_output = model(example_input)

        # Model capabilities
        capabilities = {
            "supports_feature_extraction": hasattr(model, "extract_features"),
            "supports_custom_head": hasattr(model, "reset_head")
        }

        # Prepare metadata
        metadata = {
            "model_name": model_name,
            "parameters": {
                "total": num_params,
                "trainable": trainable_params
            },
            "input_shape": list(input_shape),
            "output_shape": list(example_output.shape),
            "capabilities": capabilities,
            "config": model_config or {}
        }

        # Add labels if provided
        if labels:
            metadata["labels"] = labels

        # Save metadata
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"Model metadata saved to {output_path}")

    except Exception as e:
        print(f"Error exporting model metadata: {e}")


def main():
    parser = argparse.ArgumentParser(description='VST Model Export')

    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='tiny',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=400,
                        help='Number of classes')
    parser.add_argument('--use-custom-head', action='store_true',
                        help='Use custom head model')

    # Input parameters
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for example input')
    parser.add_argument('--clip-length', type=int, default=16,
                        help='Number of frames per clip')
    parser.add_argument('--spatial-size', type=int, default=224,
                        help='Spatial size of frames')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to labels file (for metadata)')

    # Export parameters
    parser.add_argument('--output-dir', type=str, default='exported_models',
                        help='Directory to save exported models')
    parser.add_argument('--format', type=str, default='all',
                        choices=['torchscript', 'onnx', 'all'],
                        help='Export format')
    parser.add_argument('--onnx-opset', type=int, default=12,
                        help='ONNX opset version')
    parser.add_argument('--torchscript-scripting', action='store_true',
                        help='Use scripting instead of tracing for TorchScript')

    # Other parameters
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run export on')

    args = parser.parse_args()

    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        num_classes=args.num_classes,
        model_name=args.model,
        use_custom_head=args.use_custom_head
    )

    # Create example input
    example_input = torch.randn(
        args.batch_size, 3, args.clip_length, args.spatial_size, args.spatial_size,
        device=device
    )

    # Create model name for files
    model_type = "custom" if args.use_custom_head else "standard"
    model_name = f"vst_{args.model}_{model_type}"

    # Export to TorchScript
    if args.format in ['torchscript', 'all']:
        ts_path = os.path.join(args.output_dir, f"{model_name}.pt")
        export_to_torchscript(model, ts_path, example_input, args.torchscript_scripting)

    # Export to ONNX
    if args.format in ['onnx', 'all']:
        onnx_path = os.path.join(args.output_dir, f"{model_name}.onnx")
        export_to_onnx(model, onnx_path, example_input, args.onnx_opset)

    # Export model metadata
    labels = None
    if args.labels and os.path.exists(args.labels):
        with open(args.labels, 'r') as f:
            if args.labels.endswith('.json'):
                labels = json.load(f)
            else:
                labels = [line.strip() for line in f.readlines()]

    # Create config dictionary
    model_config = {
        "model_size": args.model,
        "num_classes": args.num_classes,
        "use_custom_head": args.use_custom_head,
        "clip_length": args.clip_length,
        "spatial_size": args.spatial_size
    }

    metadata_path = os.path.join(args.output_dir, f"{model_name}_metadata.json")
    export_model_metadata(
        model=model,
        input_shape=(args.batch_size, 3, args.clip_length, args.spatial_size, args.spatial_size),
        output_path=metadata_path,
        labels=labels,
        model_name=model_name,
        model_config=model_config
    )

    print(f"Model export complete! Files saved to {args.output_dir}")


if __name__ == '__main__':
    main()