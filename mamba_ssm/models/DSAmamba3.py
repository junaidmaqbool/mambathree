# Copyright (c) 2026, Dao AI Lab, Goombalab.
#
# DSAmamba3.py — VSSM-style image classifier with Mamba3 backbone.
#
# Data-loading is fully compatible with DSA-Mamba (HbImageDataset / ImageFolder).
# To use in train.py, replace the import line:
#
#   from mamba_ssm.models.DSAmamba3 import VSSM as dsamamba
#
# Requirements (install into your Kaggle notebook):
#   pip install einops timm causal-conv1d triton
#   pip install -e /path/to/mambathree   # this repo

import math
from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from timm.layers import DropPath, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, trunc_normal_

from mamba_ssm.modules.mamba3 import Mamba3


# ---------------------------------------------------------------------------
# PatchEmbed2D — image → flat patch tokens
# ---------------------------------------------------------------------------

class PatchEmbed2D(nn.Module):
    """Convert a (B, C, H, W) image into (B, H/P, W/P, embed_dim) patch tokens."""

    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[Callable] = None,
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) → (B, H/P, W/P, embed_dim)
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# PatchMerging2D — 2× spatial downsampling, 2× channel expansion
# ---------------------------------------------------------------------------

class PatchMerging2D(nn.Module):
    def __init__(self, dim: int, norm_layer: Callable = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        # Pad to even spatial dimensions if needed
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, 2C)
        return x


# ---------------------------------------------------------------------------
# Mamba3ImageBlock — Mamba3 applied over the flattened spatial sequence
# ---------------------------------------------------------------------------

class Mamba3ImageBlock(nn.Module):
    """
    Applies Mamba3 over the H×W spatial sequence of a (B, H, W, C) feature map.

    When ``bidirectional=True`` (default), runs a second Mamba3 over the
    reversed sequence and averages the two outputs — this gives the block
    awareness of both spatial directions without the full 4-directional scan
    of the original SS2D, while keeping memory/compute manageable on GPU.
    """

    def __init__(
        self,
        hidden_dim: int,
        drop_path: float = 0.0,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
        d_state: int = 64,
        headdim: int = 64,
        expand: int = 2,
        chunk_size: int = 64,
        rope_fraction: float = 0.5,
        bidirectional: bool = True,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.norm = norm_layer(hidden_dim)

        mamba3_kwargs = dict(
            d_model=hidden_dim,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=1,
            rope_fraction=rope_fraction,
            is_mimo=False,   # SISO — no TileLang dependency; works on all CUDA GPUs
            chunk_size=chunk_size,
            layer_idx=layer_idx,
        )

        self.mamba_fwd = Mamba3(**mamba3_kwargs)
        if bidirectional:
            self.mamba_bwd = Mamba3(**mamba3_kwargs)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        shortcut = x

        x_norm = self.norm(x)
        x_seq = x_norm.view(B, H * W, C)          # (B, L, C)

        y_fwd = self.mamba_fwd(x_seq)              # (B, L, C)

        if self.bidirectional:
            y_bwd = self.mamba_bwd(x_seq.flip(1)).flip(1)   # scan reversed, unflip output
            y = (y_fwd + y_bwd) * 0.5
        else:
            y = y_fwd

        y = y.view(B, H, W, C)
        return shortcut + self.drop_path(y)


# ---------------------------------------------------------------------------
# VSSLayer — a stack of Mamba3ImageBlocks with optional spatial downsampling
# ---------------------------------------------------------------------------

class VSSLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        drop_path=0.0,
        norm_layer: Callable = nn.LayerNorm,
        downsample=None,
        d_state: int = 64,
        headdim: int = 64,
        expand: int = 2,
        chunk_size: int = 64,
        rope_fraction: float = 0.5,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Mamba3ImageBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=partial(norm_layer, eps=1e-6),
                d_state=d_state,
                headdim=headdim,
                expand=expand,
                chunk_size=chunk_size,
                rope_fraction=rope_fraction,
                bidirectional=bidirectional,
                layer_idx=i,
            )
            for i in range(depth)
        ])
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# ---------------------------------------------------------------------------
# VSSM — top-level classification model
# ---------------------------------------------------------------------------

class VSSM(nn.Module):
    """
    DSA-Mamba3: hierarchical VSSM-style image classifier built on Mamba3.

    This is a drop-in replacement for ``model.DSAmamba.VSSM`` in train.py.
    Replace the import in train.py with:

        from mamba_ssm.models.DSAmamba3 import VSSM as dsamamba

    Default config targets Kaggle T4 GPU (16 GB VRAM, T4 × 1):
    - 4 encoder stages: dims [96, 192, 384, 768]
    - SISO Mamba3 (no TileLang / MIMO required)
    - Input: (B, 3, 224, 224)
    - PatchEmbed: 4×4 → (B, 56, 56, 96)
    - After 3 PatchMerging: final spatial size 7×7 (49 tokens) at 768 dim
    - Global average pool → linear classification head

    Args:
        patch_size: spatial patch size for the ConvEmbed stem (default 4).
        in_chans: number of input image channels (default 3).
        num_classes: number of output classes (default 2 for binary anemia task).
        depths: number of Mamba3ImageBlock layers per stage.
        dims: channel width at each stage; must have len(depths) elements.
        d_state: SSM state dimension (reduce to 32 to save memory on small GPUs).
        headdim: dimension of each SSM head.
        expand: channel expansion factor inside Mamba3.
        chunk_size: Triton kernel chunk size (64 recommended for SISO).
        rope_fraction: fraction of state dimensions covered by RoPE (0.5 or 1.0).
        drop_rate: spatial dropout on patch tokens.
        drop_path_rate: stochastic depth rate (drops entire block residuals).
        bidirectional: if True, adds a backward Mamba3 scan for 2D awareness.
    """

    def __init__(
        self,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 2,
        depths: List[int] = [2, 2, 4, 2],
        dims: List[int] = [96, 192, 384, 768],
        d_state: int = 64,
        headdim: int = 64,
        expand: int = 2,
        chunk_size: int = 64,
        rope_fraction: float = 0.5,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: Callable = nn.LayerNorm,
        patch_norm: bool = True,
        bidirectional: bool = True,
        # absorb unused kwargs forwarded from train.py
        **kwargs,
    ):
        super().__init__()
        assert len(depths) == len(dims), "depths and dims must have the same length"
        self.num_classes = num_classes
        self.num_layers = len(depths)

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=dims[0],
            norm_layer=norm_layer if patch_norm else None,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth schedule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i],
                depth=depths[i],
                drop_path=dpr[sum(depths[:i]): sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if i < self.num_layers - 1 else None,
                d_state=d_state,
                headdim=headdim,
                expand=expand,
                chunk_size=chunk_size,
                rope_fraction=rope_fraction,
                bidirectional=bidirectional,
            )
            self.layers.append(layer)

        self.norm = norm_layer(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # Mamba3 learnable parameters that should not be decayed
        return {"dt_bias", "D", "B_bias", "C_bias"}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.patch_embed(x)   # (B, H/P, W/P, dims[0])
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)          # (B, H', W', dims[-1])
        # Global average pool over spatial dims
        x = x.permute(0, 3, 1, 2)  # (B, dims[-1], H', W')
        x = self.avgpool(x)         # (B, dims[-1], 1, 1)
        x = torch.flatten(x, 1)     # (B, dims[-1])
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x

    # ------------------------------------------------------------------
    # Convenience: count parameters
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
