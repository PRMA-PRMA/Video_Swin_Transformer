"""
Video Swin Transformer with Custom Head
--------------------------------------
This module extends the Video Swin Transformer with customizable classification heads
and feature extraction capabilities for downstream tasks.

Key Features:
- Feature extraction mode for transfer learning
- Customizable classification head with dropout and activation options
- Support for multi-layer feature extraction
- API for resetting the classification head for new tasks
- Automatic GPU acceleration when available
"""

import torch
import torch.nn as nn
import os
from typing import List, Optional, Tuple, Union
from einops import rearrange

from vst import VideoSwinTransformer, get_device


class VideoClassificationHead(nn.Module):
    """Classification head for Video Swin Transformer.

    Provides a flexible classification head with configurable dropout and activation.
    """

    def __init__(self,
                 in_features: int,
                 num_classes: int = 1000,
                 dropout_rate: float = 0.0,
                 activation: Optional[nn.Module] = None,
                 normalize: bool = False,
                 device: Optional[torch.device] = None):
        """
        Initialize the classification head.

        Args:
            in_features (int): Number of input features
            num_classes (int, optional): Number of classes. Defaults to 1000.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
            activation (nn.Module, optional): Activation function. Defaults to None.
            normalize (bool, optional): Whether to L2 normalize features before classification. Defaults to False.
            device (torch.device, optional): Device to place the head on. If None, will use CUDA if available.
        """
        super().__init__()

        # Set device
        self.device = device if device is not None else get_device()

        layers = []
        self.normalize = normalize

        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(in_features, num_classes))

        if activation is not None:
            layers.append(activation)

        self.head = nn.Sequential(*layers)

        # Move to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        # Ensure input is on the correct device
        x = x.to(self.device)

        if self.normalize:
            x = nn.functional.normalize(x, p=2, dim=1)
        return self.head(x)


class VideoSwinTransformerWithCustomHead(nn.Module):
    """
    Video Swin Transformer with customizable classification head and feature extraction capabilities.
    Extends the base VideoSwinTransformer to support various downstream tasks.
    """

    def __init__(self,
                 base_model: Optional[VideoSwinTransformer] = None,
                 model_args: Optional[dict] = None,
                 num_classes: int = 1000,
                 dropout_rate: float = 0.0,
                 activation: Optional[nn.Module] = None,
                 feature_mode: bool = False,
                 normalize_features: bool = False,
                 device: Optional[torch.device] = None):
        """
        Initialize VideoSwinTransformerWithCustomHead.

        Args:
            base_model (VideoSwinTransformer, optional): Pre-initialized base model. Defaults to None.
            model_args (dict, optional): Arguments to initialize a new base model if base_model is None. Defaults to None.
            num_classes (int, optional): Number of classes for classification head. Defaults to 1000.
            dropout_rate (float, optional): Dropout rate for classification head. Defaults to 0.0.
            activation (nn.Module, optional): Activation function for classification head. Defaults to None.
            feature_mode (bool, optional): If True, operate in feature extraction mode. Defaults to False.
            normalize_features (bool, optional): Whether to L2 normalize features before classification. Defaults to False.
            device (torch.device, optional): Device to place the model on. If None, will use CUDA if available.
        """
        super().__init__()

        # Set device
        self.device = device if device is not None else get_device()

        # Print device information
        if self.device.type == 'cuda':
            print(f"Using GPU: {torch.cuda.get_device_name(self.device)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.2f} GB")
        else:
            print("Using CPU. No GPU detected or CUDA is not available.")

        # Initialize base model
        if base_model is not None:
            self.base_model = base_model.to(self.device)
        elif model_args is not None:
            # Clone the args to avoid modifying the original
            args = model_args.copy()
            # Ensure no classification head in base model
            args['num_classes'] = 0
            # Add device to arguments
            args['device'] = self.device
            self.base_model = VideoSwinTransformer(**args)
        else:
            raise ValueError("Either base_model or model_args must be provided")

        self.feature_mode = feature_mode
        self.num_classes = num_classes
        self.normalize_features = normalize_features

        # Store model parameters
        self.embed_dim = self.base_model.embed_dim
        self.depths = self.base_model.num_layers
        self.feat_dim = int(self.embed_dim * 2 ** (self.depths - 1))

        # Create custom classification head
        if not feature_mode and num_classes > 0:
            self.head = VideoClassificationHead(
                in_features=self.feat_dim,
                num_classes=num_classes,
                dropout_rate=dropout_rate,
                activation=activation,
                normalize=normalize_features,
                device=self.device
            )
        else:
            self.head = nn.Identity().to(self.device)

        # Move the entire model to the device
        self.to(self.device)

    def forward(self, x: torch.Tensor,
                return_features: bool = False,
                return_all_features: bool = False,
                layer_indices: Optional[List[int]] = None,
                use_amp: bool = False) -> Union[torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    List[torch.Tensor]]:
        """
        Forward function with flexible feature extraction.

        Args:
            x (torch.Tensor): Input tensor
            return_features (bool, optional): If True, return both features and predictions. Defaults to False.
            return_all_features (bool, optional): If True, return features from all transformer layers. Defaults to False.
            layer_indices (list[int], optional): Indices of layers to extract features from. Defaults to None.
            use_amp (bool, optional): Whether to use automatic mixed precision. Defaults to False.

        Returns:
            Various output formats depending on the parameters:
            - Default: Class predictions tensor
            - With return_features=True: Tuple of (final_features, predictions)
            - With return_all_features=True or layer_indices: List of feature maps
            - In feature_mode: Features only, without classification head
        """
        # Ensure input is on the correct device
        x = x.to(self.device)

        # Use automatic mixed precision if requested and CUDA is available
        if use_amp and self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                return self._forward_impl(x, return_features, return_all_features, layer_indices)
        else:
            return self._forward_impl(x, return_features, return_all_features, layer_indices)

    def _forward_impl(self, x: torch.Tensor,
                      return_features: bool = False,
                      return_all_features: bool = False,
                      layer_indices: Optional[List[int]] = None) -> Union[torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    List[torch.Tensor]]:
        """
        Implementation of forward function.

        Args:
            x (torch.Tensor): Input tensor
            return_features (bool): If True, return both features and predictions.
            return_all_features (bool): If True, return features from all transformer layers.
            layer_indices (list[int], optional): Indices of layers to extract features from.

        Returns:
            Various output formats depending on the parameters.
        """
        # Get features from the base model
        if return_all_features or layer_indices is not None:
            features = []
            x = self.base_model.patch_embed(x)
            x = self.base_model.pos_drop(x)

            for i, layer in enumerate(self.base_model.layers):
                x = layer(x)
                if return_all_features or (layer_indices and i in layer_indices):
                    features.append(x)

            if self.feature_mode:
                return features
        else:
            # Use the standard forward_features method
            x = self.base_model.forward_features(x)
            if self.feature_mode:
                return x

        # Apply normalization and global pooling
        x_norm = x
        if not isinstance(x, list):
            x_norm = rearrange(x, 'n c d h w -> n d h w c')
            x_norm = self.base_model.norm(x_norm)
            x_norm = rearrange(x_norm, 'n d h w c -> n c d h w')

        # Global average pooling
        if isinstance(x_norm, list):
            pooled_features = [torch.mean(feat, dim=[2, 3, 4]) for feat in x_norm]
        else:
            pooled_features = torch.mean(x_norm, dim=[2, 3, 4])

        # Apply classification head
        if isinstance(pooled_features, list):
            preds = [self.head(feat) for feat in pooled_features]
        else:
            preds = self.head(pooled_features)

        # Return based on mode
        if return_features:
            if isinstance(x_norm, list):
                return x_norm, preds
            else:
                return x_norm, preds
        elif return_all_features or layer_indices is not None:
            return features, preds
        else:
            return preds

    def extract_features(self, x: torch.Tensor,
                         layer_idx: Optional[int] = None,
                         apply_norm: bool = True,
                         pool: bool = False,
                         use_amp: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Extract features from specific layers or all layers.

        Args:
            x (torch.Tensor): Input tensor
            layer_idx (int, optional): Layer index to extract features from.
                                      If None, returns features from all layers. Defaults to None.
            apply_norm (bool, optional): Whether to apply normalization. Defaults to True.
            pool (bool, optional): Whether to apply global average pooling. Defaults to False.
            use_amp (bool, optional): Whether to use automatic mixed precision. Defaults to False.

        Returns:
            torch.Tensor or list[torch.Tensor]: Features from specified layer(s)
        """
        # Ensure input is on the correct device
        x = x.to(self.device)

        # Use automatic mixed precision if requested and CUDA is available
        if use_amp and self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                return self._extract_features_impl(x, layer_idx, apply_norm, pool)
        else:
            return self._extract_features_impl(x, layer_idx, apply_norm, pool)

    def _extract_features_impl(self, x: torch.Tensor,
                               layer_idx: Optional[int] = None,
                               apply_norm: bool = True,
                               pool: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Implementation of extract_features function.

        Args:
            x (torch.Tensor): Input tensor
            layer_idx (int, optional): Layer index to extract features from.
            apply_norm (bool): Whether to apply normalization.
            pool (bool): Whether to apply global average pooling.

        Returns:
            torch.Tensor or list[torch.Tensor]: Features from specified layer(s)
        """
        x = self.base_model.patch_embed(x)
        x = self.base_model.pos_drop(x)

        if layer_idx is not None:
            # Extract features from a specific layer
            if layer_idx < 0 or layer_idx >= len(self.base_model.layers):
                raise ValueError(f"Layer index {layer_idx} is out of range (0-{len(self.base_model.layers) - 1})")

            for i in range(layer_idx + 1):
                x = self.base_model.layers[i](x)

            if apply_norm and i == len(self.base_model.layers) - 1:
                x = rearrange(x, 'n c d h w -> n d h w c')
                x = self.base_model.norm(x)
                x = rearrange(x, 'n d h w c -> n c d h w')

            if pool:
                x = torch.mean(x, dim=[2, 3, 4])

            return x
        else:
            # Extract features from all layers
            features = []
            for i, layer in enumerate(self.base_model.layers):
                x = layer(x)

                feat = x
                if apply_norm and i == len(self.base_model.layers) - 1:
                    feat = rearrange(feat, 'n c d h w -> n d h w c')
                    feat = self.base_model.norm(feat)
                    feat = rearrange(feat, 'n d h w c -> n c d h w')

                if pool:
                    feat = torch.mean(feat, dim=[2, 3, 4])

                features.append(feat)

            return features

    def set_feature_mode(self, feature_mode: bool = True):
        """
        Set the model to feature extraction mode.

        Args:
            feature_mode (bool, optional): Whether to operate in feature extraction mode. Defaults to True.
        """
        self.feature_mode = feature_mode

    def reset_head(self, num_classes: int, dropout_rate: float = 0.0,
                   activation: Optional[nn.Module] = None,
                   normalize: bool = False):
        """
        Reset the classification head with new parameters.

        Args:
            num_classes (int): Number of classes for the new head
            dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
            activation (nn.Module, optional): Activation function. Defaults to None.
            normalize (bool, optional): Whether to normalize features. Defaults to False.
        """
        self.head = VideoClassificationHead(
            in_features=self.feat_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            activation=activation,
            normalize=normalize,
            device=self.device
        )
        self.num_classes = num_classes
        self.normalize_features = normalize
        self.feature_mode = False

    def freeze_backbone(self, freeze: bool = True):
        """
        Freeze or unfreeze the backbone parameters.

        Args:
            freeze (bool, optional): Whether to freeze the backbone. Defaults to True.
        """
        for param in self.base_model.parameters():
            param.requires_grad = not freeze

    def get_feature_dims(self) -> List[int]:
        """
        Get the feature dimensions for each layer.

        Returns:
            List[int]: List of feature dimensions
        """
        return [int(self.embed_dim * 2 ** i) for i in range(self.depths)]

    def to_device(self, device: Optional[torch.device] = None):
        """
        Move model to the specified device.

        Args:
            device (torch.device, optional): Target device. If None, will use CUDA if available.
        """
        if device is None:
            device = get_device()

        self.device = device
        self.base_model = self.base_model.to(device)
        self.head = self.head.to(device)
        return self


def create_custom_model(model_name: str = 'tiny',
                        pretrained: Optional[str] = None,
                        feature_mode: bool = False,
                        num_classes: int = 1000,
                        dropout_rate: float = 0.0,
                        activation: Optional[nn.Module] = None,
                        device: Optional[torch.device] = None,
                        **kwargs) -> VideoSwinTransformerWithCustomHead:
    """
    Create a custom Video Swin Transformer model.

    Args:
        model_name (str, optional): Model size ('tiny', 'small', 'base', 'large'). Defaults to 'tiny'.
        pretrained (str, optional): Path to pretrained weights. Defaults to None.
        feature_mode (bool, optional): Whether to operate in feature extraction mode. Defaults to False.
        num_classes (int, optional): Number of classes for classification head. Defaults to 1000.
        dropout_rate (float, optional): Dropout rate for classification head. Defaults to 0.0.
        activation (nn.Module, optional): Activation function for classification head. Defaults to None.
        device (torch.device, optional): Device to place the model on. If None, will use CUDA if available.
        **kwargs: Additional arguments for the base model.

    Returns:
        VideoSwinTransformerWithCustomHead: Configured model
    """
    # Set device
    if device is None:
        device = get_device()

    # Default configurations
    configs = {
        'tiny': {
            'embed_dim': 96,
            'depths': (2, 2, 6, 2),
            'num_heads': (3, 6, 12, 24),
            'patch_size': (2, 4, 4),
            'drop_path_rate': 0.1,
        },
        'small': {
            'embed_dim': 96,
            'depths': (2, 2, 18, 2),
            'num_heads': (3, 6, 12, 24),
            'patch_size': (2, 4, 4),
            'drop_path_rate': 0.2,
        },
        'base': {
            'embed_dim': 128,
            'depths': (2, 2, 18, 2),
            'num_heads': (4, 8, 16, 32),
            'patch_size': (2, 4, 4),
            'drop_path_rate': 0.3,
        },
        'large': {
            'embed_dim': 192,
            'depths': (2, 2, 18, 2),
            'num_heads': (6, 12, 24, 48),
            'patch_size': (2, 4, 4),
            'drop_path_rate': 0.3,
        }
    }

    if model_name not in configs:
        raise ValueError(f"Unknown model name: {model_name}. Available options: {list(configs.keys())}")

    # Extract window size from pretrained path if available
    window_size = (2, 7, 7)
    if isinstance(pretrained, str) and 'window' in pretrained:
        try:
            window_info = pretrained.split('_window')[1].split('_')[0]
            if len(window_info) == 3:
                window_size = (int(window_info[0]), int(window_info[1]), int(window_info[2]))
                print(f"Parsed window size from filename: {window_size}")
        except Exception as e:
            print(f"Error parsing window size from pretrained path: {e}")

    # Create model config
    model_config = configs[model_name].copy()
    model_config['window_size'] = window_size
    model_config.update(kwargs)

    # Add device to model config
    model_config['device'] = device

    # Create model
    model = VideoSwinTransformerWithCustomHead(
        model_args=model_config,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        activation=activation,
        feature_mode=feature_mode,
        device=device
    )

    # Load pretrained weights if provided
    if pretrained:
        print(f"Loading pretrained weights from {pretrained}")
        model.base_model.inflate_weights(pretrained)

    return model


if __name__ == "__main__":
    # Example usage
    # Auto-detect device
    device = get_device()
    print(f"Using device: {device}")

    model = create_custom_model('tiny', feature_mode=True, device=device)
    print(f"Created model with feature dimensions: {model.get_feature_dims()}")

    # Example with input tensor
    x = torch.randn(2, 3, 16, 224, 224, device=device)

    # Forward pass with automatic mixed precision if CUDA is available
    use_amp = device.type == 'cuda'
    features = model(x, use_amp=use_amp)
    print(f"Feature shape: {features.shape}")

    # Switch to classification mode
    model.set_feature_mode(False)
    model.reset_head(num_classes=400, dropout_rate=0.2)
    preds = model(x, use_amp=use_amp)
    print(f"Prediction shape: {preds.shape}")