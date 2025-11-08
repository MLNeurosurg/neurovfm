"""
Projection Heads and MLP Modules

Various projection heads for self-supervised learning and downstream tasks,
including DINO, iBOT, contrastive learning, and general-purpose MLPs.
"""

# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from typing import List, Optional


class CSyncBatchNorm(nn.SyncBatchNorm):
    """
    Centered Synchronized Batch Normalization.
    
    Applies synchronized batch normalization across devices but only centers
    (subtracts mean) without variance normalization when with_var=False.
    
    Args:
        with_var (bool): Whether to include variance normalization. Defaults to False.
        *args, **kwargs: Arguments passed to nn.SyncBatchNorm
    """
    def __init__(self, *args, with_var=False, **kwargs):
        super(CSyncBatchNorm, self).__init__(*args, **kwargs)
        self.with_var = with_var

    def forward(self, x):
        # center norm
        self.training = False
        if not self.with_var:
            self.running_var = torch.ones_like(self.running_var)
        normed_x = super(CSyncBatchNorm, self).forward(x)
        # update center
        self.training = True
        _ = super(CSyncBatchNorm, self).forward(x)
        return normed_x

class CustomSequential(nn.Sequential):
    """
    Custom Sequential module with automatic dimension permutation for batch norm.
    
    Automatically handles dimension permutation for batch normalization layers
    when input has more than 2 dimensions.
    """
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)

    def forward(self, input):
        for module in self:
            dim = len(input.shape)
            if isinstance(module, self.bn_types) and dim > 2:
                perm = list(range(dim - 1)); perm.insert(1, dim - 1)
                inv_perm = list(range(dim)) + [1]; inv_perm.pop(1)
                input = module(input.permute(*perm)).permute(*inv_perm)
            else:
                input = module(input)
        return input

class MLP(nn.Module):
    """
    Multi-layer perceptron with flexible architecture.
    
    General-purpose MLP with configurable hidden layers, normalization, and activation.
    
    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        hidden_dims (List[int]): List of hidden layer dimensions
        norm (str, optional): Normalization type ('bn', 'syncbn', 'csyncbn', 'ln')
        act (str): Activation function ('relu', 'gelu'). Defaults to 'gelu'.
    
    Example:
        >>> from neurovfm.models import MLP
        >>> 
        >>> # Create 3-layer MLP
        >>> mlp = MLP(
        ...     in_dim=768,
        ...     out_dim=2,
        ...     hidden_dims=[512, 256],
        ...     norm='ln',
        ...     act='gelu'
        ... )
        >>> 
        >>> # Forward pass
        >>> features = torch.randn(32, 768)
        >>> logits = mlp(features)  # [32, 2]
    
    Notes:
        - Each hidden layer: Linear → [Norm] → Activation
        - Final layer: Linear (no norm/activation)
        - Supports flexible depth via hidden_dims list
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: List[int], norm: str = None, act: str = 'gelu'):
        super().__init__()
        self.out_dim = out_dim
        
        act_layer = self._build_act(act)
        layers = []
        
        # Build layers using dimensions from hidden_dims list
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if norm:
                layers.append(self._build_norm(norm, hidden_dim))
            layers.append(act_layer)
            prev_dim = hidden_dim
            
        # Final output layer
        layers.append(nn.Linear(prev_dim, out_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize linear layer weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _build_norm(self, norm, hidden_dim):
        """Build normalization layer."""
        if norm == 'bn':
            return nn.BatchNorm1d(hidden_dim)
        elif norm == 'syncbn':
            return nn.SyncBatchNorm(hidden_dim)
        elif norm == 'csyncbn':
            return CSyncBatchNorm(hidden_dim)
        elif norm == 'ln':
            return nn.LayerNorm(hidden_dim)
        else:
            raise ValueError(f"unknown norm type {norm}")
    
    def _build_act(self, act):
        """Build activation layer."""
        if act == 'relu':
            return nn.ReLU()
        elif act == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"unknown act type {act}")
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input features [B, in_dim]
        
        Returns:
            torch.Tensor: Output features [B, out_dim]
        """
        return self.mlp(x)