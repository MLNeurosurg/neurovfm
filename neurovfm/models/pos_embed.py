"""
Positional Encoding Modules for Spatial Coordinates

Various positional encoding schemes for 2D and 3D spatial data, including
sinusoidal encodings and learnable Fourier features.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import math
from typing import List, Tuple, Optional, Union

import torch
from torch import nn
from torch.nn.init import trunc_normal_
import torch.nn.functional as F

from positional_encodings.torch_encodings import PositionalEncoding3D

class PositionalEncoding3DWrapper(nn.Module):
    """
    Wrapper for 3D sinusoidal positional encodings.
    
    Pre-computes and caches a lookup table of positional encodings for 3D coordinates.
    Optimized for medical imaging volumes with specific depth and spatial dimensions.
    
    Args:
        in_dim (int): Input feature dimension. Defaults to 256.
        d (int): Positional encoding dimension. Must be divisible by 3. Defaults to 30.
        d_size (int): Maximum depth dimension. Defaults to 128.
        hw_size (int): Maximum height/width dimension. Defaults to 192.
        pe_factor (float): Scaling factor for positional encodings. Defaults to 1.
        concat (bool): If True, concatenates PE to input. If False, returns PE only
                      or adds to input. Defaults to True.
    
    Example:
        >>> from neurovfm.models import PositionalEncoding3DWrapper
        >>> 
        >>> pe = PositionalEncoding3DWrapper(
        ...     in_dim=256,
        ...     d=30,
        ...     d_size=128,
        ...     hw_size=192
        ... )
        >>> features = torch.randn(2, 1000, 256)
        >>> coords = torch.randint(0, 128, (2, 1000, 3))  # (d, h, w) coordinates
        >>> output = pe(features, coords)
        >>> print(output.shape)  # [2, 1000, 286]
    
    Notes:
        - Pre-computes lookup table for d_size x hw_size x hw_size volume
        - Efficient GPU indexing via buffer registration
        - Coordinates should be in (depth, height, width) order
    """
    def __init__(self, in_dim=256, d=30, d_size=128, hw_size=192, pe_factor=1, concat=True):
        super().__init__()
        self.concat = concat
        if self.concat:
            self.num_out = in_dim+d
        else:
            self.num_out = in_dim
            d = in_dim
        assert d % 3 == 0
        p_enc = PositionalEncoding3D(d)
        self.d_size = d_size
        self.hw_size = hw_size
        
        # Pre-compute the table and register it as a buffer
        pe_table = p_enc(torch.zeros(1, d_size, hw_size, hw_size, d))[0].view(d_size*(hw_size**2), d)
        self.register_buffer('p_enc', pe_table, persistent=False)

        self.pe_factor = pe_factor
        self.d=d
    
    def forward(self, x, coords):
        """
        Forward pass to add or concatenate 3D positional encodings.
        
        Args:
            x (torch.Tensor): Input features [B, N, in_dim]
            coords (torch.Tensor): 3D spatial coordinates [B, N, 3] in (d, h, w) order
        
        Returns:
            torch.Tensor: Features with positional encoding [B, N, num_out]
        """
        # Assuming coords is of shape (batch_size, num_tokens, 3)
        batch_size, num_tokens, _ = coords.shape

        # Flatten 3D coordinates to 1D index
        tocatid = coords[:, :, 0] * (self.hw_size**2) + coords[:, :, 1] * self.hw_size + coords[:, :, 2]
        
        # Index directly on GPU
        tocat = self.p_enc[tocatid]
        
        # Scale and convert dtype for AMP compatibility
        tocat = (tocat * self.pe_factor).to(dtype=x.dtype)

        if self.concat:
            return torch.cat((x, tocat), dim=-1)
        else:
            return tocat
