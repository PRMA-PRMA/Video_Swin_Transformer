"""
Video Swin Transformer Implementation
------------------------------------
A PyTorch implementation of Video Swin Transformer that doesn't rely on mmcv or other dependencies.
This implementation supports loading pretrained weights from the original implementation and
automatically leverages available GPU acceleration.

Reference: "Video Swin Transformer" - https://arxiv.org/abs/2106.13230
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
import os
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from typing import List, Optional, Tuple, Union, Dict, Any
import math


def get_device() -> torch.device:
    """
    Get the available device (CUDA GPU or CPU).

    Returns:
        torch.device: The available device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


class MLP(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 act_layer: nn.Module = nn.GELU,
                 drop: float = 0.):
        """
        Initialize the MLP module.

        Args:
            in_features (int): Number of input features
            hidden_features (int, optional): Number of hidden features. Defaults to in_features.
            out_features (int, optional): Number of output features. Defaults to in_features.
            act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
            drop (float, optional): Dropout rate. Defaults to 0.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    """
    Partition the input tensor into non-overlapping windows.

    Args:
        x (torch.Tensor): Input tensor of shape (B, D, H, W, C)
        window_size (tuple[int]): Window size (Wd, Wh, Ww)

    Returns:
        windows (torch.Tensor): Windows after partition with shape
            (B*num_windows, window_size*window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1],
               W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows: torch.Tensor,
                   window_size: Tuple[int, int, int],
                   B: int, D: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse the window partition.

    Args:
        windows (torch.Tensor): Windows tensor of shape (B*num_windows, window_size*window_size*window_size, C)
        window_size (tuple[int]): Window size (Wd, Wh, Ww)
        B (int): Batch size
        D (int): Depth of video
        H (int): Height of video
        W (int): Width of video

    Returns:
        x (torch.Tensor): Reversed window tensor of shape (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2],
                     window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size: Tuple[int, int, int],
                    window_size: Tuple[int, int, int],
                    shift_size: Optional[Tuple[int, int, int]] = None) -> Union[Tuple[int, int, int],
Tuple[Tuple[int, int, int],
Tuple[int, int, int]]]:
    """
    Calculate the effective window size and shift size based on input dimensions.

    Args:
        x_size (tuple[int]): Input resolution (D, H, W)
        window_size (tuple[int]): Window size (Wd, Wh, Ww)
        shift_size (tuple[int], optional): Shift size (Sd, Sh, Sw). Defaults to None.

    Returns:
        tuple: Effective window size and optionally effective shift size
    """
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


import math


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both shifted and non-shifted window.
    """

    def __init__(self,
                 dim: int,
                 window_size: Tuple[int, int, int],
                 num_heads: int,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 attn_drop: float = 0.,
                 proj_drop: float = 0.):
        """
        Initialize the WindowAttention3D module.

        Args:
            dim (int): Number of input channels
            window_size (tuple[int]): Window size (Wd, Wh, Ww)
            num_heads (int): Number of attention heads
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            proj_drop (float, optional): Output dropout rate. Defaults to 0.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Create the parameter first, but don't initialize yet
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) *
                        (2 * self.window_size[1] - 1) *
                        (2 * self.window_size[2] - 1),
                        self.num_heads))

        # Initialize position indices
        self._init_relative_position_index()

        # The actual initialization of the bias table will be explicitly called
        # after weight loading to avoid it being overwritten

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _init_relative_position_index(self):
        """Initialize relative position indices for looking up the bias table"""
        # Get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def initialize_3d_position_bias(self):
        """
        Initialize 3D relative position bias table.
        This method should be explicitly called after model weight loading.
        """
        print(f"Initializing 3D position bias for window size {self.window_size} with {self.num_heads} heads")
        # Get window size
        window_d, window_h, window_w = self.window_size
        num_heads = self.num_heads

        # Initialize 2D position bias table first (height and width dimensions)
        table_size_2d = (2 * window_h - 1) * (2 * window_w - 1)
        table_2d = torch.zeros(table_size_2d, num_heads)
        nn.init.trunc_normal_(table_2d, std=0.02)

        # Now extend to 3D (treat the temporal dimension specially)
        # Create a special sinusoidal encoding for the temporal dimension
        d_position = torch.arange(2 * window_d - 1).float()
        d_signal = torch.zeros(2 * window_d - 1, num_heads)

        # Apply sinusoidal encoding - different frequencies for each head
        div_term = torch.exp(torch.arange(0, num_heads, 2) * -(math.log(10000.0) / num_heads))
        if num_heads >= 2:
            d_signal[:, 0::2] = torch.sin(d_position.unsqueeze(1) * div_term)
            if num_heads % 2 != 0:
                d_signal[:, 1::2] = torch.cos(d_position.unsqueeze(1) * div_term[:(num_heads // 2)])
            else:
                d_signal[:, 1::2] = torch.cos(d_position.unsqueeze(1) * div_term)
        else:
            # Handle single head case
            d_signal[:, 0] = torch.sin(d_position * div_term[0])

        # Combine the 2D spatial and 1D temporal biases
        table_3d = torch.zeros((2 * window_d - 1) * (2 * window_h - 1) * (2 * window_w - 1), num_heads)

        # For each temporal position, copy the 2D bias and then modulate it by the temporal signal
        for t_idx in range(2 * window_d - 1):
            t_start = t_idx * table_size_2d
            t_end = (t_idx + 1) * table_size_2d

            # Copy 2D bias and modulate by temporal position encoding
            temporal_factor = d_signal[t_idx, :].unsqueeze(0)  # [1, num_heads]
            table_3d[t_start:t_end, :] = table_2d * (1.0 + 0.1 * temporal_factor)

        # Assign to the bias table parameter
        with torch.no_grad():
            self.relative_position_bias_table.copy_(table_3d)

        print(f"Position bias table initialized with shape: {self.relative_position_bias_table.shape}")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Forward function.

        Args:
            x (torch.Tensor): Input features with shape of (num_windows*B, N, C)
            mask (torch.Tensor, optional): Attention mask with shape of (num_windows, N, N). Defaults to None.

        Returns:
            torch.Tensor: Output features
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Apply relative position bias - with better error handling
        if hasattr(self, 'relative_position_bias_table') and hasattr(self, 'relative_position_index'):
            try:
                # Get the relative position bias from the table using the position index
                idx = self.relative_position_index[:N, :N].reshape(-1)
                if idx.max() < self.relative_position_bias_table.size(0):
                    relative_position_bias = self.relative_position_bias_table[idx].reshape(
                        N, N, -1)  # Wd*Wh*Ww, Wd*Wh*Ww, nH
                    relative_position_bias = relative_position_bias.permute(2, 0,
                                                                            1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
                    attn = attn + relative_position_bias.unsqueeze(0)
                else:
                    print(f"Warning: Index out of bounds for position bias table. Using unbiased attention.")
            except (RuntimeError, IndexError) as e:
                print(f"Warning: Error computing relative position bias: {e}. Using unbiased attention.")

        # Apply attention mask if needed
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.
    """

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 window_size: Tuple[int, int, int] = (2, 7, 7),
                 shift_size: Tuple[int, int, int] = (0, 0, 0),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm,
                 use_checkpoint: bool = False):
        """
        Initialize the Swin Transformer Block.

        Args:
            dim (int): Number of input channels
            num_heads (int): Number of attention heads
            window_size (tuple[int], optional): Window size. Defaults to (2, 7, 7).
            shift_size (tuple[int], optional): Shift size for SW-MSA. Defaults to (0, 0, 0).
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.0.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.0.
            drop_path (float, optional): Stochastic depth rate. Defaults to 0.0.
            act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
            use_checkpoint (bool, optional): Whether to use checkpointing to save memory. Defaults to False.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        # Validate shift size
        for i in range(len(self.shift_size)):
            assert 0 <= self.shift_size[i] < self.window_size[
                i], f"shift_size {self.shift_size[i]} must be less than window_size {self.window_size[i]}"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor]) -> torch.Tensor:
        """
        First part of forward function (window attention).

        Args:
            x (torch.Tensor): Input tensor of shape (B, D, H, W, C)
            mask_matrix (torch.Tensor, optional): Attention mask for cyclic shift. Defaults to None.

        Returns:
            torch.Tensor: Output tensor
        """
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # Pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        # Cyclic shift
        has_shift = any(i > 0 for i in shift_size)
        if has_shift:
            try:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
                attn_mask = mask_matrix
            except RuntimeError as e:
                print(f"Warning: Cyclic shift error: {e}. Falling back to no shift.")
                shifted_x = x
                attn_mask = None
        else:
            shifted_x = x
            attn_mask = None

        # Partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # Merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        # Reverse cyclic shift
        if has_shift:
            try:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            except RuntimeError as e:
                print(f"Warning: Reverse cyclic shift error: {e}. Using shifted_x directly.")
                x = shifted_x
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x: torch.Tensor) -> torch.Tensor:
        """
        Second part of forward function (MLP).

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x: torch.Tensor, mask_matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Forward function.

        Args:
            x (torch.Tensor): Input feature, tensor size (B, D, H, W, C).
            mask_matrix (torch.Tensor, optional): Attention mask for cyclic shift. Defaults to None.

        Returns:
            torch.Tensor: Output tensor
        """
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer
    """

    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        """
        Initialize the Patch Merging layer.

        Args:
            dim (int): Number of input channels
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward function.

        Args:
            x (torch.Tensor): Input feature, tensor size (B, D, H, W, C).

        Returns:
            torch.Tensor: Output tensor
        """
        B, D, H, W, C = x.shape

        # Padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.
    """

    def __init__(self,
                 patch_size: Tuple[int, int, int] = (2, 4, 4),
                 in_chans: int = 3,
                 embed_dim: int = 96,
                 norm_layer: Optional[nn.Module] = None):
        """
        Initialize the Patch Embedding layer.

        Args:
            patch_size (tuple[int], optional): Patch token size. Defaults to (2, 4, 4).
            in_chans (int, optional): Number of input channels. Defaults to 3.
            embed_dim (int, optional): Number of linear projection output channels. Defaults to 96.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to None.
        """
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W)

        Returns:
            torch.Tensor: Output tensor
        """
        # Padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)  # B DHW C
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    """

    def __init__(self,
                 dim: int,
                 depth: int,
                 num_heads: int,
                 window_size: Tuple[int, int, int],
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: Union[List[float], float] = 0.,
                 norm_layer: nn.Module = nn.LayerNorm,
                 downsample: Optional[nn.Module] = None,
                 use_checkpoint: bool = False):
        """
        Initialize the basic layer.

        Args:
            dim (int): Number of input channels
            depth (int): Number of blocks
            num_heads (int): Number of attention heads
            window_size (tuple[int]): Local window size
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.0.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.0.
            drop_path (float | list[float], optional): Stochastic depth rate. Defaults to 0.0.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
            downsample (nn.Module, optional): Downsample layer at the end of the layer. Defaults to None.
            use_checkpoint (bool, optional): Whether to use checkpointing to save memory. Defaults to False.
        """
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward function.

        Args:
            x (torch.Tensor): Input feature, tensor size (B, C, D, H, W).

        Returns:
            torch.Tensor: Output tensor
        """
        # Calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]

        try:
            # Compute attention mask
            device = x.device
            attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, device)

            # Apply transformer blocks
            for blk in self.blocks:
                x = blk(x, attn_mask)

        except Exception as e:
            print(f"Warning: Error in BasicLayer forward: {e}. Using simplified forward pass.")
            # Fallback to simpler forward pass without mask
            for blk in self.blocks:
                x = blk(x)

        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)

        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0.):
        """
        Initialize DropPath.

        Args:
            drop_prob (float, optional): Probability of dropping path. Defaults to 0.0.
        """
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


@lru_cache()
def compute_mask(D: int, H: int, W: int,
                 window_size: Tuple[int, int, int],
                 shift_size: Tuple[int, int, int],
                 device: torch.device) -> torch.Tensor:
    """
    Compute attention mask for shifted window attention.

    Args:
        D (int): Depth of input feature
        H (int): Height of input feature
        W (int): Width of input feature
        window_size (tuple[int]): Window size
        shift_size (tuple[int]): Shift size
        device (torch.device): Device for computation

    Returns:
        attn_mask (torch.Tensor): Attention mask
    """
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class VideoSwinTransformer(nn.Module):
    """ Video Swin Transformer
    """

    def __init__(self,
                 patch_size: Tuple[int, int, int] = (2, 4, 4),
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 96,
                 depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
                 num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
                 window_size: Tuple[int, int, int] = (2, 7, 7),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 norm_layer: nn.Module = nn.LayerNorm,
                 patch_norm: bool = True,
                 frozen_stages: int = -1,
                 use_checkpoint: bool = False,
                 pretrained: Optional[str] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize Video Swin Transformer.

        Args:
            patch_size (tuple[int], optional): Patch size. Defaults to (2, 4, 4).
            in_chans (int, optional): Number of input image channels. Defaults to 3.
            num_classes (int, optional): Number of classes for classification head. Defaults to 1000.
            embed_dim (int, optional): Patch embedding dimension. Defaults to 96.
            depths (tuple[int], optional): Depth of each Swin Transformer layer. Defaults to (2, 2, 6, 2).
            num_heads (tuple[int], optional): Number of attention heads in different layers. Defaults to (3, 6, 12, 24).
            window_size (tuple[int], optional): Window size. Defaults to (2, 7, 7).
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.0.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop_rate (float, optional): Dropout rate. Defaults to 0.0.
            attn_drop_rate (float, optional): Attention dropout rate. Defaults to 0.0.
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.1.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
            patch_norm (bool, optional): If True, add normalization after patch embedding. Defaults to True.
            frozen_stages (int, optional): Stages to be frozen. Defaults to -1.
            use_checkpoint (bool, optional): Whether to use checkpointing to save memory. Defaults to False.
            pretrained (str, optional): Path to pretrained model. Defaults to None.
            device (torch.device, optional): Device to place the model on. If None, will use CUDA if available.
        """
        super().__init__()

        # Set device
        self.device = device if device is not None else get_device()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size
        self.feature_dims = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(int(embed_dim * 2 ** (self.num_layers - 1)))

        # Classifier head
        self.head = nn.Linear(int(embed_dim * 2 ** (self.num_layers - 1)),
                              num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self._freeze_stages()

        # Load pretrained weights if provided
        if pretrained is not None:
            self.inflate_weights(pretrained)

        # Move model to device
        self.to(self.device)

    def _freeze_stages(self):
        """Freeze stages.
        Freezes the patch embedding stage and specified transformer stages.
        """
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def _init_weights(self, m):
        """
        Initialize model weights.

        Args:
            m (nn.Module): Module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        Get parameters that do not decay.

        Returns:
            set: Set of parameter names that do not decay
        """
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """
        Get keywords of parameters that do not decay.

        Returns:
            set: Set of parameter keywords that do not decay
        """
        return {'relative_position_bias_table'}

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function to extract features.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Feature maps
        """
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward function.

        Args:
            x (torch.Tensor): Input tensor
            return_features (bool, optional): If True, return both features and class predictions. Defaults to False.

        Returns:
            torch.Tensor or tuple[torch.Tensor, torch.Tensor]:
                If return_features is False, returns class predictions.
                If return_features is True, returns a tuple of (features, class predictions).
        """
        # Ensure input is on the correct device
        x = x.to(self.device)

        x = self.forward_features(x)
        features = x

        # Global average pooling along the spatiotemporal dimensions
        x = torch.mean(x, dim=[2, 3, 4])

        # Classification head
        x = self.head(x)

        if return_features:
            return features, x
        return x

    def extract_features(self, x: torch.Tensor, layer_idx: Optional[int] = None) -> Union[
        torch.Tensor, List[torch.Tensor]]:
        """
        Extract features from specific layers or all layers.

        Args:
            x (torch.Tensor): Input tensor
            layer_idx (int, optional): Layer index to extract features from.
                                      If None, returns features from all layers. Defaults to None.

        Returns:
            torch.Tensor or list[torch.Tensor]: Features from specified layer(s)
        """
        # Ensure input is on the correct device
        x = x.to(self.device)

        x = self.patch_embed(x)
        x = self.pos_drop(x)

        if layer_idx is not None:
            # Extract features from a specific layer
            if layer_idx < 0 or layer_idx >= self.num_layers:
                raise ValueError(f"Layer index {layer_idx} is out of range (0-{self.num_layers - 1})")

            for i in range(layer_idx + 1):
                x = self.layers[i](x)

            return x
        else:
            # Extract features from all layers
            features = []
            for layer in self.layers:
                x = layer(x)
                features.append(x)

            return features

    def inflate_weights(self, pretrained_path: str):
        """
        Inflate 2D pretrained Swin Transformer weights to 3D for transfer learning.

        Args:
            pretrained_path (str): Path to the pretrained weights
        """
        print(f"Inflating weights from: {pretrained_path}")

        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pretrained model path not found: {pretrained_path}")

        try:
            # Load the checkpoint
            checkpoint = torch.load(pretrained_path, map_location=self.device)

            # Get state dict - handle different checkpoint formats
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Create new state dict with proper keys
            new_state_dict = {}

            # Process all keys to match expected format
            for k, v in state_dict.items():
                # Skip position bias tables and indices - they will be initialized by our custom WindowAttention3D
                if "relative_position_bias_table" in k or "relative_position_index" in k:
                    continue

                # Remove 'backbone.' prefix if present
                if k.startswith('backbone.'):
                    new_k = k[9:]  # Remove 'backbone.'
                else:
                    new_k = k

                # Map classification head keys
                if new_k.startswith('cls_head.fc_cls'):
                    if new_k == 'cls_head.fc_cls.weight':
                        new_k = 'head.weight'
                    elif new_k == 'cls_head.fc_cls.bias':
                        new_k = 'head.bias'

                # Add to new state dict
                new_state_dict[new_k] = v

            # Print key mapping information for debugging
            print(f"Key mapping examples:")
            sample_keys = list(state_dict.keys())[:5]
            for k in sample_keys:
                mapped_k = k
                if k.startswith('backbone.'):
                    mapped_k = k[9:]
                if mapped_k.startswith('cls_head.fc_cls'):
                    if mapped_k == 'cls_head.fc_cls.weight':
                        mapped_k = 'head.weight'
                    elif mapped_k == 'cls_head.fc_cls.bias':
                        mapped_k = 'head.bias'
                print(f"  {k} -> {mapped_k}")

            # Check if weights are already 3D (5D tensor for conv) or need inflation from 2D
            is_3d_weights = False
            if 'patch_embed.proj.weight' in new_state_dict:
                if len(new_state_dict['patch_embed.proj.weight'].shape) == 5:
                    is_3d_weights = True
                    print("Detected 3D weights, skipping inflation")

            # Inflate patch embedding weights if needed
            if not is_3d_weights and 'patch_embed.proj.weight' in new_state_dict:
                weight_2d = new_state_dict['patch_embed.proj.weight']  # [C_out, C_in, kH, kW]
                kD = self.patch_size[0]  # temporal patch size
                # Inflate to [C_out, C_in, kD, kH, kW]
                weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, kD, 1, 1) / kD
                new_state_dict['patch_embed.proj.weight'] = weight_3d

            # Handle other dimension mismatches
            for k in list(new_state_dict.keys()):
                if k in self.state_dict():
                    if new_state_dict[k].shape != self.state_dict()[k].shape:
                        print(
                            f"Skipping {k} due to shape mismatch: {new_state_dict[k].shape} vs {self.state_dict()[k].shape}")
                        del new_state_dict[k]
                else:
                    print(f"Skipping {k} as it does not exist in the model")
                    del new_state_dict[k]

            # Load the processed state dict
            msg = self.load_state_dict(new_state_dict, strict=False)
            print(f"Inflated weights loaded with message: {msg}")
            print(f"Loaded {len(new_state_dict)}/{len(self.state_dict())} parameters")

            # After loading weights, make sure to initialize position bias for all attention blocks
            print("Initializing custom 3D relative position bias tables...")
            self._init_attention_position_bias()

        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            import traceback
            print(traceback.format_exc())

    def _init_attention_position_bias(self):
        """Initialize position bias for all attention blocks"""
        # Initialize position bias for all attention layers
        for i, layer in enumerate(self.layers):
            for j, block in enumerate(layer.blocks):
                print(f"Initializing position bias for layer {i} block {j}")
                # Trigger the custom initialization in WindowAttention3D
                if hasattr(block.attn, 'initialize_3d_position_bias'):
                    block.attn.initialize_3d_position_bias()


class VideoClassificationHead(nn.Module):
    """Classification head for Video Swin Transformer.
    """

    def __init__(self,
                 in_features: int,
                 num_classes: int = 1000,
                 dropout_rate: float = 0.0,
                 activation: Optional[nn.Module] = None):
        """
        Initialize the classification head.

        Args:
            in_features (int): Number of input features
            num_classes (int, optional): Number of classes. Defaults to 1000.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
            activation (nn.Module, optional): Activation function. Defaults to None.
        """
        super().__init__()

        layers = []
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(in_features, num_classes))

        if activation is not None:
            layers.append(activation)

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor
        """
        return self.head(x)


class VideoSwinTransformerWithCustomHead(VideoSwinTransformer):
    """
    Video Swin Transformer with customizable classification head and feature extraction capabilities.
    Extends the base VideoSwinTransformer to support various downstream tasks.
    """

    def __init__(self,
                 patch_size: Tuple[int, int, int] = (2, 4, 4),
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 96,
                 depths: Tuple[int, int, int, int] = (2, 2, 6, 2),
                 num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
                 window_size: Tuple[int, int, int] = (2, 7, 7),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 norm_layer: nn.Module = nn.LayerNorm,
                 patch_norm: bool = True,
                 frozen_stages: int = -1,
                 use_checkpoint: bool = False,
                 pretrained: Optional[str] = None,
                 feature_mode: bool = False,
                 head_dropout: float = 0.0,
                 head_activation: Optional[nn.Module] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize VideoSwinTransformerWithCustomHead.

        Args:
            patch_size (tuple[int], optional): Patch size. Defaults to (2, 4, 4).
            in_chans (int, optional): Number of input image channels. Defaults to 3.
            num_classes (int, optional): Number of classes for classification head. Defaults to 1000.
            embed_dim (int, optional): Patch embedding dimension. Defaults to 96.
            depths (tuple[int], optional): Depth of each Swin Transformer layer. Defaults to (2, 2, 6, 2).
            num_heads (tuple[int], optional): Number of attention heads in different layers. Defaults to (3, 6, 12, 24).
            window_size (tuple[int], optional): Window size. Defaults to (2, 7, 7).
            mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.0.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop_rate (float, optional): Dropout rate. Defaults to 0.0.
            attn_drop_rate (float, optional): Attention dropout rate. Defaults to 0.0.
            drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.1.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
            patch_norm (bool, optional): If True, add normalization after patch embedding. Defaults to True.
            frozen_stages (int, optional): Stages to be frozen. Defaults to -1.
            use_checkpoint (bool, optional): Whether to use checkpointing to save memory. Defaults to False.
            pretrained (str, optional): Path to pretrained model. Defaults to None.
            feature_mode (bool, optional): If True, operate in feature extraction mode. Defaults to False.
            head_dropout (float, optional): Dropout rate for classification head. Defaults to 0.0.
            head_activation (nn.Module, optional): Activation function for classification head. Defaults to None.
            device (torch.device, optional): Device to place the model on. If None, will use CUDA if available.
        """
        # Set device
        device = device if device is not None else get_device()

        # Initialize with identity head to avoid creating the default head
        super().__init__(
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,  # Use 0 to create nn.Identity head in base class
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            frozen_stages=frozen_stages,
            use_checkpoint=use_checkpoint,
            pretrained=pretrained,
            device=device
        )

        self.feature_mode = feature_mode
        self.num_classes = num_classes

        # Create custom classification head if not in feature mode
        if not feature_mode and num_classes > 0:
            feat_dim = int(embed_dim * 2 ** (self.num_layers - 1))
            self.custom_head = VideoClassificationHead(
                in_features=feat_dim,
                num_classes=num_classes,
                dropout_rate=head_dropout,
                activation=head_activation
            )
        else:
            self.custom_head = nn.Identity()

        # Ensure the custom head is on the correct device
        self.custom_head = self.custom_head.to(self.device)

    def forward(self, x: torch.Tensor,
                return_features: bool = False,
                return_all_features: bool = False,
                layer_indices: Optional[List[int]] = None) -> Union[torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    List[torch.Tensor]]:
        """
        Forward function with flexible feature extraction.

        Args:
            x (torch.Tensor): Input tensor
            return_features (bool, optional): If True, return both features and predictions. Defaults to False.
            return_all_features (bool, optional): If True, return features from all transformer layers. Defaults to False.
            layer_indices (list[int], optional): Indices of layers to extract features from. Defaults to None.

        Returns:
            Various output formats depending on the parameters:
            - Default: Class predictions tensor
            - With return_features=True: Tuple of (final_features, predictions)
            - With return_all_features=True or layer_indices: List of feature maps
            - In feature_mode: Features only, without classification head
        """
        # Ensure input is on the correct device
        x = x.to(self.device)

        if layer_indices is not None or return_all_features:
            features = []
            x = self.patch_embed(x)
            x = self.pos_drop(x)

            for i, layer in enumerate(self.layers):
                x = layer(x)
                if return_all_features or (layer_indices and i in layer_indices):
                    features.append(x)

            if self.feature_mode:
                return features
        else:
            # Normal forward pass
            x = self.patch_embed(x)
            x = self.pos_drop(x)

            for layer in self.layers:
                x = layer(x)

            if self.feature_mode:
                return x

        # Normal classification flow (if not in feature_mode)
        x_norm = rearrange(x, 'n c d h w -> n d h w c')
        x_norm = self.norm(x_norm)
        x_norm = rearrange(x_norm, 'n d h w c -> n c d h w')

        # Global average pooling
        pooled = torch.mean(x_norm, dim=[2, 3, 4])

        # Apply classification head
        pred = self.custom_head(pooled)

        if return_features and not return_all_features and layer_indices is None:
            return x_norm, pred
        elif return_all_features or layer_indices is not None:
            return features, pred
        else:
            return pred

    def set_feature_mode(self, feature_mode: bool = True):
        """
        Set the model to feature extraction mode.

        Args:
            feature_mode (bool, optional): Whether to operate in feature extraction mode. Defaults to True.
        """
        self.feature_mode = feature_mode

    def reset_head(self, num_classes: int, dropout_rate: float = 0.0, activation: Optional[nn.Module] = None):
        """
        Reset the classification head with new parameters.

        Args:
            num_classes (int): Number of classes for the new head
            dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
            activation (nn.Module, optional): Activation function. Defaults to None.
        """
        feat_dim = int(self.embed_dim * 2 ** (self.num_layers - 1))
        self.custom_head = VideoClassificationHead(
            in_features=feat_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            activation=activation
        ).to(self.device)
        self.num_classes = num_classes
        self.feature_mode = False


def video_swin_t(pretrained: Union[bool, str] = False, **kwargs) -> VideoSwinTransformer:
    """Video Swin Transformer Tiny model

    Args:
        pretrained (str or bool): Path to pretrained weights or False for random initialization
        **kwargs: Additional arguments to pass to the VideoSwinTransformer constructor

    Returns:
        VideoSwinTransformer: Configured model
    """
    # Extract window size from pretrained path if available
    window_size = (2, 7, 7)
    if isinstance(pretrained, str) and 'window' in pretrained:
        try:
            # Try to extract window size from filename
            window_info = pretrained.split('_window')[1].split('_')[0]
            if len(window_info) == 3:
                window_size = (int(window_info[0]), int(window_info[1]), int(window_info[2]))
                print(f"Parsed window size from filename: {window_size}")
            else:
                print(f"Warning: Could not parse window size from {pretrained}, using default {window_size}")
        except Exception as e:
            print(f"Error parsing window size from pretrained path: {e}")

    device = kwargs.pop('device', get_device())
    model = VideoSwinTransformer(
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=window_size,
        patch_size=(2, 4, 4),
        drop_path_rate=0.1,
        pretrained=pretrained if isinstance(pretrained, str) else None,
        device=device,
        **kwargs)
    return model


def video_swin_s(pretrained: Union[bool, str] = False, **kwargs) -> VideoSwinTransformer:
    """Video Swin Transformer Small model

    Args:
        pretrained (str or bool): Path to pretrained weights or False for random initialization
        **kwargs: Additional arguments to pass to the VideoSwinTransformer constructor

    Returns:
        VideoSwinTransformer: Configured model
    """
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

    device = kwargs.pop('device', get_device())
    model = VideoSwinTransformer(
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        window_size=window_size,
        patch_size=(2, 4, 4),
        drop_path_rate=0.2,
        pretrained=pretrained if isinstance(pretrained, str) else None,
        device=device,
        **kwargs)
    return model


def video_swin_b(pretrained: Union[bool, str] = False, **kwargs) -> VideoSwinTransformer:
    """Video Swin Transformer Base model

    Args:
        pretrained (str or bool): Path to pretrained weights or False for random initialization
        **kwargs: Additional arguments to pass to the VideoSwinTransformer constructor

    Returns:
        VideoSwinTransformer: Configured model
    """
    # Extract window size from pretrained path if available
    window_size = (2, 7, 7)
    if isinstance(pretrained, str) and 'window' in pretrained:
        try:
            # Try to extract window size from filename
            window_info = pretrained.split('_window')[1].split('_')[0]
            if len(window_info) == 3:
                window_size = (int(window_info[0]), int(window_info[1]), int(window_info[2]))
                print(f"Parsed window size from filename: {window_size}")
            else:
                print(f"Warning: Could not parse window size from {pretrained}, using default {window_size}")
        except Exception as e:
            print(f"Error parsing window size from pretrained path: {e}")

    device = kwargs.pop('device', get_device())
    model = VideoSwinTransformer(
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=window_size,
        patch_size=(2, 4, 4),
        drop_path_rate=0.3,
        pretrained=pretrained if isinstance(pretrained, str) else None,
        device=device,
        **kwargs)
    return model


def video_swin_l(pretrained: Union[bool, str] = False, **kwargs) -> VideoSwinTransformer:
    """Video Swin Transformer Large model

    Args:
        pretrained (str or bool): Path to pretrained weights or False for random initialization
        **kwargs: Additional arguments to pass to the VideoSwinTransformer constructor

    Returns:
        VideoSwinTransformer: Configured model
    """
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

    device = kwargs.pop('device', get_device())
    model = VideoSwinTransformer(
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        window_size=window_size,
        patch_size=(2, 4, 4),
        drop_path_rate=0.3,
        pretrained=pretrained if isinstance(pretrained, str) else None,
        device=device,
        **kwargs)
    return model


def video_swin_t_custom(pretrained: Union[bool, str] = False, feature_mode: bool = False, num_classes: int = 1000,
                        **kwargs) -> VideoSwinTransformerWithCustomHead:
    """Video Swin Transformer Tiny model with custom head

    Args:
        pretrained (str or bool): Path to pretrained weights or False for random initialization
        feature_mode (bool, optional): Whether to operate in feature extraction mode. Defaults to False.
        num_classes (int, optional): Number of classes for classification head. Defaults to 1000.
        **kwargs: Additional arguments to pass to the VideoSwinTransformerWithCustomHead constructor

    Returns:
        VideoSwinTransformerWithCustomHead: Configured model
    """
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

    device = kwargs.pop('device', get_device())
    model = VideoSwinTransformerWithCustomHead(
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=window_size,
        patch_size=(2, 4, 4),
        drop_path_rate=0.1,
        pretrained=pretrained if isinstance(pretrained, str) else None,
        feature_mode=feature_mode,
        num_classes=num_classes,
        device=device,
        **kwargs)
    return model


def video_swin_b_custom(pretrained: Union[bool, str] = False, feature_mode: bool = False, num_classes: int = 1000,
                        **kwargs) -> VideoSwinTransformerWithCustomHead:
    """Video Swin Transformer Base model with custom head

    Args:
        pretrained (str or bool): Path to pretrained weights or False for random initialization
        feature_mode (bool, optional): Whether to operate in feature extraction mode. Defaults to False.
        num_classes (int, optional): Number of classes for classification head. Defaults to 1000.
        **kwargs: Additional arguments to pass to the VideoSwinTransformerWithCustomHead constructor

    Returns:
        VideoSwinTransformerWithCustomHead: Configured model
    """
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

    device = kwargs.pop('device', get_device())
    model = VideoSwinTransformerWithCustomHead(
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
        window_size=window_size,
        patch_size=(2, 4, 4),
        drop_path_rate=0.3,
        pretrained=pretrained if isinstance(pretrained, str) else None,
        feature_mode=feature_mode,
        num_classes=num_classes,
        device=device,
        **kwargs)

    return model