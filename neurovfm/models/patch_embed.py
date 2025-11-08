"""
Patch Embedding for 3D Medical Images

Converts 3D image patches (voxels) into embedding vectors using a linear projection.
"""

from functools import partial

import torch.nn as nn
from einops import rearrange
from torch import _assert
from torch.nn.modules.utils import _pair

try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None


class PatchEmbed(nn.Module):
    """
    Embeds 3D image patches into a latent space.
    
    Converts volumetric patches (voxels) into fixed-dimensional embeddings using
    a linear projection. Operates on tokenized patches of size (d, h, w) where
    typically d=4, h=16, w=16.
    
    Args:
        patch_hw_size (int): Height and width of each patch. Defaults to 16.
        patch_d_size (int): Depth of each patch. Defaults to 4.
        in_chans (int): Number of input channels. Defaults to 1.
        embed_dim (int): Embedding dimension. Defaults to 768.
        norm_layer (nn.Module, optional): Normalization layer to apply after embedding.
        bias (bool): Whether to include bias in linear projection. Defaults to True.
        fused_bias_fc (bool): Whether to use fused dense layer for efficiency.
                              Defaults to True.
    
    Example:
        >>> import torch
        >>> from neurovfm.models import PatchEmbed
        >>> 
        >>> # Create patch embedder
        >>> embedder = PatchEmbed(
        ...     patch_hw_size=16,
        ...     patch_d_size=4,
        ...     in_chans=1,
        ...     embed_dim=768
        ... )
        >>> 
        >>> # Tokenized input: [N, 1024] where 1024 = 1*4*16*16
        >>> tokens = torch.randn(100, 1024)
        >>> embeddings = embedder(tokens)  # [100, 768]
    
    Notes:
        - Input should be pre-tokenized (flattened patches)
        - Patch dimensions are (C, D, H, W) = (in_chans, patch_d_size, patch_hw_size, patch_hw_size)
        - Total input dimension is C * D * H * W
    """
    
    def __init__(
        self,
        patch_hw_size: int = 16,
        patch_d_size: int = 4,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: nn.Module = None,
        bias: bool = True,
        fused_bias_fc: bool = True,
    ):
        super().__init__()
        patch_hw_size = _pair(patch_hw_size)
        self.patch_hw_size = patch_hw_size
        self.patch_d_size = patch_d_size

        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")

        linear_cls = nn.Linear if not fused_bias_fc or not bias else FusedDense
        self.proj = linear_cls(
            in_chans * patch_hw_size[0] * patch_hw_size[1] * patch_d_size, 
            embed_dim, 
            bias=bias
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tokens of shape [N, C*D*H*W]
        
        Returns:
            torch.Tensor: Embeddings of shape [N, embed_dim]
        """
        x = self.proj(x)
        x = self.norm(x)
        return x

