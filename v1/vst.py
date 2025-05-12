#vst.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size*window_size*window_size, C)
        window_size (tuple[int]): Window size
        B (int): Batch size
        D (int): Depth of video
        H (int): Height of video
        W (int): Width of video

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """
    Args:
        x_size (tuple[int]): input resolution
        window_size (tuple[int]): window size
        shift_size (tuple[int], optional): shift size
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


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both shifted and non-shifted window.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
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

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww, Wd*Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

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

    def __init__(self, dim, num_heads, window_size=(2, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must be less than window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must be less than window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must be less than window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
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

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
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

    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
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
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
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

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    """
    Compute attention mask for shifted window attention.
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

    def __init__(self, patch_size=(2, 4, 4), in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=(2, 7, 7), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 frozen_stages=-1, use_checkpoint=False, pretrained=None):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
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

        # classifier head
        self.head = nn.Linear(int(embed_dim * 2 ** (self.num_layers - 1)),
                              num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self._freeze_stages()

        # Load pretrained weights if provided
        if pretrained is not None:
            self.inflate_weights(pretrained)

    def _freeze_stages(self):
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
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = rearrange(x, 'n c d h w -> n d h w c')
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')

        return x

    def forward(self, x):
        x = self.forward_features(x)

        # Global average pooling along the spatiotemporal dimensions
        x = torch.mean(x, dim=[2, 3, 4])

        x = self.head(x)
        return x

    def inflate_weights(self, pretrained_path):
        """
        Inflate 2D pretrained Swin Transformer weights to 3D for transfer learning.

        Args:
            pretrained_path (str): Path to the 2D pretrained weights
        """
        print(f"Inflating weights from: {pretrained_path}")

        # Load the checkpoint
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # Get state dict - handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Check if keys have 'backbone.' prefix
        has_backbone_prefix = any(k.startswith('backbone.') for k in state_dict.keys())

        # Create a new state dict without the backbone prefix
        if has_backbone_prefix:
            print("Removing 'backbone.' prefix from keys")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_k = k[9:]  # Remove 'backbone.'
                else:
                    new_k = k
                new_state_dict[new_k] = v
            state_dict = new_state_dict

        # Remove unnecessary keys
        keys_to_remove = []
        for k in list(state_dict.keys()):
            if "relative_position_index" in k or "attn_mask" in k:
                keys_to_remove.append(k)

        for k in keys_to_remove:
            del state_dict[k]

        # Check if weights are already 3D (5D tensor for conv) or need inflation from 2D
        is_3d_weights = False
        if 'patch_embed.proj.weight' in state_dict:
            if len(state_dict['patch_embed.proj.weight'].shape) == 5:
                is_3d_weights = True
                print("Detected 3D weights, skipping inflation")

        # Inflate patch embedding weights if needed
        if not is_3d_weights and 'patch_embed.proj.weight' in state_dict:
            weight_2d = state_dict['patch_embed.proj.weight']  # [C_out, C_in, kH, kW]
            kD = self.patch_size[0]  # temporal patch size
            # Inflate to [C_out, C_in, kD, kH, kW]
            weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, kD, 1, 1) / kD
            state_dict['patch_embed.proj.weight'] = weight_3d

        # Process relative position bias tables
        for k in list(state_dict.keys()):
            if 'relative_position_bias_table' in k:
                try:
                    # Only process if the key exists in the model
                    if k not in self.state_dict():
                        print(f"Skipping {k} as it does not exist in model")
                        del state_dict[k]
                        continue

                    pretrained_bias = state_dict[k]
                    current_bias = self.state_dict()[k]
                    L1, num_heads = pretrained_bias.shape
                    L2, _ = current_bias.shape

                    if L1 != L2:
                        # Reshape and interpolate if needed
                        try:
                            # Assume square window for 2D model
                            S1 = int(np.sqrt(L1))
                            bias_reshaped = pretrained_bias.transpose(0, 1).view(1, num_heads, S1, S1)

                            # Target size for spatial dimensions
                            window_h, window_w = self.window_size[1], self.window_size[2]
                            target_size = (2 * window_h - 1, 2 * window_w - 1)

                            # Interpolate with bicubic interpolation
                            bias_interpolated = F.interpolate(
                                bias_reshaped, size=target_size, mode='bicubic', align_corners=False
                            )
                            bias_interpolated = bias_interpolated.view(num_heads, -1).transpose(0, 1)

                            # Repeat along temporal dimension if needed
                            window_d = self.window_size[0]
                            state_dict[k] = bias_interpolated.repeat((2 * window_d - 1), 1)
                        except Exception as e:
                            print(f"Warning: Failed to interpolate position bias for {k}: {e}")
                            # Remove this key to avoid error during loading
                            del state_dict[k]
                    else:
                        # If shapes match, just use the pretrained bias
                        state_dict[k] = pretrained_bias
                except Exception as e:
                    print(f"Warning: Error processing {k}: {e}")
                    # Remove this key to avoid error during loading
                    if k in state_dict:
                        del state_dict[k]

        # Handle other dimension mismatches
        for k in list(state_dict.keys()):
            if k in self.state_dict():
                if state_dict[k].shape != self.state_dict()[k].shape:
                    print(f"Skipping {k} due to shape mismatch: {state_dict[k].shape} vs {self.state_dict()[k].shape}")
                    del state_dict[k]
            else:
                print(f"Skipping {k} as it does not exist in the model")
                del state_dict[k]

        # Load the processed state dict
        msg = self.load_state_dict(state_dict, strict=False)
        print(f"Inflated weights loaded with message: {msg}")
        print(f"Loaded {len(state_dict)}/{len(self.state_dict())} parameters")


def video_swin_t(pretrained=False, **kwargs):
    """Video Swin Transformer Tiny model

    Args:
        pretrained (str or bool): Path to pretrained weights or False for random initialization
    """
    # Extract window size from pretrained path if available
    window_size = (2, 7, 7)
    if isinstance(pretrained, str) and 'window' in pretrained:
        try:
            # Try to extract window size from filename
            # Format example: swin_tiny_patch244_window877_kinetics400_1k.pth
            window_info = pretrained.split('_window')[1].split('_')[0]
            if len(window_info) == 3:
                window_size = (int(window_info[0]), int(window_info[1]), int(window_info[2]))
                print(f"Parsed window size from filename: {window_size}")
            else:
                print(f"Warning: Could not parse window size from {pretrained}, using default {window_size}")
        except Exception as e:
            print(f"Error parsing window size from pretrained path: {e}")

    model = VideoSwinTransformer(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=window_size,
        patch_size=(2, 4, 4),
        drop_path_rate=0.1,
        pretrained=pretrained if isinstance(pretrained, str) else None,
        **kwargs)
    return model


def video_swin_s(pretrained=False, **kwargs):
    """Video Swin Transformer Small model

    Args:
        pretrained (str or bool): Path to pretrained weights or False for random initialization
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

    model = VideoSwinTransformer(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=window_size,
        patch_size=(2, 4, 4),
        drop_path_rate=0.2,
        pretrained=pretrained if isinstance(pretrained, str) else None,
        **kwargs)
    return model


def video_swin_b(pretrained=False, **kwargs):
    """Video Swin Transformer Base model

    Args:
        pretrained (str or bool): Path to pretrained weights or False for random initialization
    """
    # Extract window size from pretrained path if available
    window_size = (2, 7, 7)
    if isinstance(pretrained, str) and 'window' in pretrained:
        try:
            # Try to extract window size from filename
            # Format example: swin_base_patch244_window877_kinetics600_22k.pth
            window_info = pretrained.split('_window')[1].split('_')[0]
            if len(window_info) == 3:
                window_size = (int(window_info[0]), int(window_info[1]), int(window_info[2]))
                print(f"Parsed window size from filename: {window_size}")
            else:
                print(f"Warning: Could not parse window size from {pretrained}, using default {window_size}")
        except Exception as e:
            print(f"Error parsing window size from pretrained path: {e}")

    model = VideoSwinTransformer(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=window_size,
        patch_size=(2, 4, 4),
        drop_path_rate=0.3,
        pretrained=pretrained if isinstance(pretrained, str) else None,
        **kwargs)
    return model


def video_swin_l(pretrained=False, **kwargs):
    """Video Swin Transformer Large model

    Args:
        pretrained (str or bool): Path to pretrained weights or False for random initialization
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

    model = VideoSwinTransformer(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=window_size,
        patch_size=(2, 4, 4),
        drop_path_rate=0.3,
        pretrained=pretrained if isinstance(pretrained, str) else None,
        **kwargs)
    return model


# Example usage
if __name__ == "__main__":
    # Create a random video tensor of shape [batch_size, channels, frames, height, width]
    video = torch.randn(2, 3, 16, 224, 224)

    # Initialize the Video Swin Transformer model
    model = video_swin_t(num_classes=400)

    # Example with pretrained weights:
    # model = video_swin_t(pretrained="path/to/swin_tiny_patch4_window7_224.pth", num_classes=400)

    # Forward pass
    output = model(video)

    print(f"Input shape: {video.shape}")
    print(f"Output shape: {output.shape}")