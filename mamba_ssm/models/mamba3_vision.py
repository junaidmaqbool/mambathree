# Copyright (c) 2026, Dao AI Lab, Goombalab.
#
# mamba3_vision.py — Simple flat Mamba3 image backbone for classification & regression.
#
# Much simpler than DSAmamba3.py:
#   • No hierarchy (no PatchMerging, no 4 stages)
#   • Single channel dimension throughout
#   • N identical Mamba3 blocks in sequence
#   • Global average-pool → linear head
#
# Supports both tasks by switching num_classes:
#   num_classes=2  → classification (CrossEntropyLoss)
#   num_classes=1  → regression / Hb estimation (HuberLoss)
#
# Usage:
#   from mamba_ssm.models.mamba3_vision import Mamba3Vision
#   model = Mamba3Vision(num_classes=2)   # or num_classes=1

import torch
import torch.nn as nn

try:
    from timm.layers import DropPath, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, trunc_normal_

from mamba_ssm.modules.mamba3 import Mamba3


# ---------------------------------------------------------------------------
# Mamba3Block  —  pre-norm residual wrapper around Mamba3
# ---------------------------------------------------------------------------

class Mamba3Block(nn.Module):
    """Pre-LayerNorm Mamba3 residual block.

    forward: x (B, L, C) → x + drop_path(Mamba3(LN(x)))
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 32,
        headdim: int = 64,
        expand: int = 2,
        chunk_size: int = 64,
        rope_fraction: float = 0.5,
        drop_path: float = 0.0,
        layer_idx: int = None,
    ):
        super().__init__()
        assert (expand * dim) % headdim == 0, (
            f"expand*dim ({expand*dim}) must be divisible by headdim ({headdim}). "
            f"Choose headdim that divides {expand*dim}."
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mamba = Mamba3(
            d_model=dim,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=1,
            rope_fraction=rope_fraction,
            is_mimo=False,      # SISO — pure Triton, no TileLang needed
            chunk_size=chunk_size,
            layer_idx=layer_idx,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.mamba(self.norm(x)))


# ---------------------------------------------------------------------------
# Mamba3Vision  —  full image model
# ---------------------------------------------------------------------------

class Mamba3Vision(nn.Module):
    """
    Flat Mamba3 image classifier / regressor.

    Pipeline:
        (B, C, H, W)
          → Conv2d patch embed   (B, dim, H/P, W/P)
          → flatten + transpose  (B, L, dim)   L = (H/P)*(W/P)
          → learnable pos embed
          → depth × Mamba3Block
          → LayerNorm
          → global average pool  (B, dim)
          → Linear head          (B, num_classes)

    Args:
        img_size       Image resolution (square).  Default 224.
        patch_size     Patch size (square).  16 → 196 tokens, 8 → 784 tokens.
        in_chans       Input channels.  3 for RGB, 1 for grayscale.
        num_classes    2 for binary classification, 1 for Hb regression.
        dim            Channel width throughout (single value, no stages).
        depth          Number of Mamba3Block layers.
        d_state        SSM state size.  Reduce to 16 if OOM.
        headdim        Head dimension; must divide expand*dim evenly.
        expand         Channel expansion inside Mamba3.
        chunk_size     Triton kernel chunk size (64 for SISO).
        rope_fraction  Fraction of state dims with RoPE (0.5 or 1.0).
        drop_path_rate Stochastic depth rate.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 2,
        dim: int = 256,
        depth: int = 6,
        d_state: int = 32,
        headdim: int = 64,
        expand: int = 2,
        chunk_size: int = 64,
        rope_fraction: float = 0.5,
        drop_path_rate: float = 0.1,
        **kwargs,          # absorb extra kwargs from config dicts
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── Patch embedding ───────────────────────────────────────────────
        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        )
        self.patch_embed = nn.Conv2d(
            in_chans, dim,
            kernel_size=patch_size, stride=patch_size,
        )
        num_patches = (img_size // patch_size) ** 2

        # Learnable position embeddings (standard ViT-style)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))

        # ── Mamba3 blocks ─────────────────────────────────────────────────
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Mamba3Block(
                dim=dim,
                d_state=d_state,
                headdim=headdim,
                expand=expand,
                chunk_size=chunk_size,
                rope_fraction=rope_fraction,
                drop_path=dpr[i],
                layer_idx=i,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()

        # ── Weight init ───────────────────────────────────────────────────
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    # ------------------------------------------------------------------

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        # pos_embed + Mamba3 special params — all excluded from weight decay
        return {'pos_embed', 'dt_bias', 'D', 'B_bias', 'C_bias'}

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.patch_embed(x)                  # (B, dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)         # (B, L, dim)
        x = x + self.pos_embed                   # add position info

        for blk in self.blocks:
            x = blk(x)                           # (B, L, dim)

        x = self.norm(x)
        x = x.mean(dim=1)                        # global avg pool → (B, dim)
        return self.head(x)                      # (B, num_classes)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
