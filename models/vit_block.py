# ============================================================
# models/vit_block.py
# ViT Encoder Block injected into YOLOv12 backbone via hook
# ============================================================

import torch
import torch.nn as nn


class ViTBlock(nn.Module):
    """
    Vision Transformer (ViT) encoder block.
    Injected into YOLOv12 backbone via forward hook.

    Architecture:
        - Multi-Head Self-Attention (MHSA)
        - Feed-Forward Network (FFN) with GELU
        - Layer Norm before each sub-layer (Pre-LN)
    """

    def __init__(self, dim: int = 256, heads: int = 4, mlp_ratio: int = 4):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, C, H, W] from YOLO backbone
        Returns:
            x: Attention-enhanced feature map [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Flatten spatial dims → token sequence
        tokens = x.flatten(2).transpose(1, 2)          # [B, HW, C]

        # Multi-Head Self-Attention with residual
        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out

        # Feed-Forward Network with residual
        tokens = tokens + self.mlp(self.norm2(tokens))

        # Reshape back to feature map
        x = tokens.transpose(1, 2).reshape(B, C, H, W)
        return x
