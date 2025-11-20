"""
Multiple Instance Learning (MIL) Pooling Modules

Aggregates variable-length sequences of patch embeddings into fixed-size representations
using attention-based pooling mechanisms.

Classes:
    - AggregateThenClassify: Attention-based MIL with gated attention
    - ClassifyThenAggregate: MIL combining attention pooling with patch-level predictions
"""

from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

try:
    import torch_scatter
except ImportError:
    torch_scatter = None

try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None

# Import MLP from projector
from neurovfm.models.projector import MLP


def pad_ragged(
    data: torch.Tensor, 
    cu_seqlens: torch.Tensor, 
    batch_first: bool = True, 
    padding_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a ragged tensor based on cumulative sequence lengths.
    
    Args:
        data (torch.Tensor): Ragged tensor of shape [total_tokens, ...]
        cu_seqlens (torch.Tensor): Cumulative sequence lengths [batch_size + 1]
        batch_first (bool): If True, output shape is [B, max_len, ...]. Defaults to True.
        padding_value (float): Value to use for padding. Defaults to 0.0.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - padded: Padded tensor [B, max_len, ...] or [max_len, B, ...]
            - mask: Boolean mask [B, max_len] indicating valid elements
    """
    sequences = [data[cu_seqlens[i]:cu_seqlens[i+1]] for i in range(len(cu_seqlens) - 1)]
    padded = torch.nn.utils.rnn.pad_sequence(
        sequences, 
        batch_first=batch_first, 
        padding_value=padding_value
    )

    # Create mask indicating valid (non-padded) elements
    seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    max_len = padded.shape[1]  # Get max sequence length from padded tensor
    indices = torch.arange(max_len, device=data.device)
    
    if batch_first:
        indices = indices.unsqueeze(0)  # Shape [1, max_len]
        mask = indices < seq_lengths.unsqueeze(1)  # Shape [B, max_len]
    else:
        indices = indices.unsqueeze(1)  # Shape [max_len, 1]
        mask = indices < seq_lengths.unsqueeze(0)  # Shape [max_len, B]

    return padded, mask


class AggregateThenClassify(nn.Module):
    """
    Attention-Based Multiple Instance Learning (AB-MIL).
    
    Aggregates a variable-length bag of instances into a single representation using
    learned attention weights. Supports optional gating mechanism.
    
    Args:
        dim (int): Input feature dimension
        hidden_dim (int, optional): Hidden dimension for attention computation.
                                    Defaults to dim.
        W_out (int): Number of attention heads/outputs. Defaults to 1.
        drop_rate (float): Dropout rate. Defaults to 0.0.
        init_std (float): Standard deviation for weight initialization. Defaults to 0.02.
        use_gating (bool): Whether to use gating mechanism. Defaults to True.
        use_norm (bool): Whether to apply normalization before attention. Defaults to True.
        norm_layer: Normalization layer constructor. Defaults to LayerNorm.
    
    Example:
        >>> import torch
        >>> from neurovfm.models import AggregateThenClassify
        >>> 
        >>> # Create Aggregate-Then-Classify pooling
        >>> mil = AggregateThenClassify(dim=768, hidden_dim=512, W_out=1)
        >>> 
        >>> # Variable-length batch with 3 sequences
        >>> features = torch.randn(50, 768)  # 50 total tokens
        >>> cu_seqlens = torch.tensor([0, 20, 35, 50])  # 3 sequences
        >>> 
        >>> # Aggregate to fixed size
        >>> output = mil(features, cu_seqlens=cu_seqlens)  # [3, 768]
    
    Notes:
        - Implements the attention mechanism from Ilse et al. (ICML 2018)
        - Gating improves attention quality by element-wise modulation
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        W_out: int = 1,
        drop_rate: float = 0.0,
        init_std: float = 0.02,
        use_gating: bool = True,
        use_norm: bool = True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_features = self.dim = dim
        if hidden_dim is None:
            hidden_dim = dim

        if FusedDense is None:
            raise ImportError("FusedDense from flash_attn is required for MIL modules")

        self.attention_V = FusedDense(dim, hidden_dim, bias=True)
        if use_gating:
            self.gating_V = FusedDense(dim, hidden_dim, bias=True)
        else:
            self.gating_V = None

        self.W = FusedDense(hidden_dim, W_out, bias=True)
        
        self.dropout = nn.Dropout(p=drop_rate)
        if use_norm:
            self.norm_attn = norm_layer(dim)
        else:
            self.norm_attn = None
        
        self.init_std = init_std
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self, 
        media: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        return_attn_probs: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            media (torch.Tensor): Input features [N, dim] where N is total tokens
            cu_seqlens (torch.Tensor): Cumulative sequence lengths [B+1]
            max_seqlen (int, optional): Maximum sequence length (unused, for API compatibility)
            return_attn_probs (bool): Whether to return attention weights. Defaults to False.
        
        Returns:
            torch.Tensor: Aggregated features [B, dim] if W_out==1, else [B, W_out, dim]
            OR Tuple[torch.Tensor, torch.Tensor]: (output, attention_weights) if return_attn_probs=True
        """
        # Apply pre-normalization
        if self.norm_attn is not None:
            media_attn = self.norm_attn(media)
        else:
            media_attn = media

        media_attn = self.dropout(media_attn)
        
        # Attention mechanism
        attention_features = torch.tanh(self.attention_V(media_attn))
        if self.gating_V is not None:
            gating_features = torch.sigmoid(self.gating_V(media_attn))
            score_features = attention_features * gating_features
        else:
            score_features = attention_features
        
        scores = self.W(score_features)  # [N, W_out]
        
        if torch_scatter is None:
            raise ImportError("torch_scatter is required for softmax attention in MIL")
        
        if scores.size(1) > 1:
            # Multi-head attention
            num_seqs = len(cu_seqlens) - 1
            instance_ids = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            for i in range(num_seqs):
                instance_ids[cu_seqlens[i]:cu_seqlens[i+1]] = i

            class_outputs = []
            attention_weights = torch.zeros_like(scores)  # [N, num_classes]
            
            for c in range(scores.size(1)):
                class_scores = scores[:, c]  # [N]
                
                # Compute softmax with numerical stability
                max_scores = torch_scatter.segment_csr(
                    src=class_scores,
                    indptr=cu_seqlens.long(),
                    reduce='max'
                )  # [B]
                
                class_scores = class_scores - max_scores[instance_ids]  # [N]
                exp_scores = torch.exp(class_scores)  # [N]
                
                normalizers = torch_scatter.segment_csr(
                    src=exp_scores,
                    indptr=cu_seqlens.long(),
                    reduce='sum'
                )  # [B]
                
                class_attention = exp_scores / normalizers[instance_ids]  # [N]
                attention_weights[:, c] = class_attention
                
                # Aggregate
                class_output = torch_scatter.segment_csr(
                    src=class_attention.unsqueeze(-1) * media_attn,
                    indptr=cu_seqlens.long(),
                    reduce='sum'
                )  # [B, dim]
                
                class_outputs.append(class_output)
            
            output = torch.stack(class_outputs, dim=1)  # [B, C, dim]

        else:
            # Single attention head
            scores = scores.squeeze(-1)
            
            # Compute softmax with numerical stability
            max_scores = torch_scatter.segment_csr(
                src=scores,
                indptr=cu_seqlens.long(),
                reduce='max'
            )  # [B]
            
            num_seqs = len(cu_seqlens) - 1
            instance_ids = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
            for i in range(num_seqs):
                instance_ids[cu_seqlens[i]:cu_seqlens[i+1]] = i
                
            scores = scores - max_scores[instance_ids]  # [N]
            exp_scores = torch.exp(scores)  # [N]
            
            normalizers = torch_scatter.segment_csr(
                src=exp_scores,
                indptr=cu_seqlens.long(),
                reduce='sum'
            )  # [B]
            
            attention_weights = (exp_scores / normalizers[instance_ids]).unsqueeze(-1)  # [N, 1]
        
            # Aggregate
            output = torch_scatter.segment_csr(
                src=attention_weights * media_attn,
                indptr=cu_seqlens.long(),
                reduce='sum'
            )  # [B, dim]

        if return_attn_probs:
            return output, attention_weights
        else:
            return output


class ClassifyThenAggregate(nn.Module):
    """
    Classify-Then-Aggregate Multiple Instance Learning.
    
    Combines attention-weighted pooling with patch-level predictions. The final
    bag-level prediction is the attention-weighted sum of patch predictions.
    
    Args:
        dim (int): Input feature dimension
        hidden_dim (int, optional): Hidden dimension for attention computation.
                                    Defaults to dim.
        W_out (int): Number of output classes. Defaults to 1.
        mlp_hidden_dims (List[int], optional): Hidden dimensions for prediction MLP.
        mlp_out_dim (int, optional): Output dimension for MLP. Defaults to W_out.
        drop_rate (float): Dropout rate. Defaults to 0.0.
        init_std (float): Standard deviation for weight initialization. Defaults to 0.02.
        use_gating (bool): Whether to use gating mechanism. Defaults to True.
        use_norm (bool): Whether to apply normalization. Defaults to False.
        use_output_bias_scale (bool): Whether to apply learnable bias/scale to output.
                                      Defaults to True.
        norm_layer: Normalization layer constructor. Defaults to LayerNorm.
    
    Example:
        >>> import torch
        >>> from neurovfm.models import ClassifyThenAggregate
        >>> 
        >>> # Create Classify-Then-Aggregate MIL for classification
        >>> mil = ClassifyThenAggregate(
        ...     dim=768,
        ...     W_out=2,
        ...     mlp_hidden_dims=[512, 256]
        ... )
        >>> 
        >>> # Variable-length batch
        >>> features = torch.randn(50, 768)
        >>> cu_seqlens = torch.tensor([0, 20, 35, 50])
        >>> 
        >>> # Get class logits
        >>> logits = mil(features, cu_seqlens=cu_seqlens)  # [3, 2]
    
    Notes:
        - Attention and prediction are computed separately
        - Final output is \\sum_i α_i * h_i where α_i are attention weights
        - More flexible than standard AB-MIL for classification tasks
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        W_out: int = 1,
        mlp_hidden_dims: Optional[List[int]] = None,
        mlp_out_dim: Optional[int] = None,
        drop_rate: float = 0.0,
        init_std: float = 0.02,
        use_gating: bool = True,
        use_norm: bool = False,
        use_output_bias_scale: bool = True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_features = self.dim = dim
        if hidden_dim is None:
            hidden_dim = dim

        if FusedDense is None:
            raise ImportError("FusedDense from flash_attn is required for MIL modules")

        self.attention_V = FusedDense(dim, hidden_dim, bias=True)
        if use_gating:
            self.gating_V = FusedDense(dim, hidden_dim, bias=True)
        else:
            self.gating_V = None

        self.W = FusedDense(hidden_dim, W_out, bias=True)
        self.W_out = W_out
        
        self.dropout = nn.Dropout(p=drop_rate)
        if use_norm:
            self.norm_attn = norm_layer(dim)
            self.norm_mlp = norm_layer(dim)
        else:
            self.norm_attn = None
            self.norm_mlp = None

        if mlp_out_dim is None:
            mlp_out_dim = W_out
        
        self.mlp = MLP(in_dim=dim, out_dim=mlp_out_dim, hidden_dims=mlp_hidden_dims)
        
        self.use_output_bias_scale = use_output_bias_scale
        if use_output_bias_scale:
            self.output_bias = nn.Parameter(torch.zeros(W_out))
            self.output_scale = nn.Parameter(torch.ones(W_out))
        
        self.init_std = init_std
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        media: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            media (torch.Tensor): Input features [N, dim]
            cu_seqlens (torch.Tensor): Cumulative sequence lengths [B+1]
            max_seqlen (int, optional): Maximum sequence length (unused)
            return_logits (bool): Whether to return attention weights and patch logits.
                                  Defaults to False.
        
        Returns:
            torch.Tensor: Bag-level predictions [B, W_out]
            OR Tuple[torch.Tensor, ...]: (output, attention_weights, patch_logits)
                                         if return_logits=True
        """
        # Apply pre-normalization
        if self.norm_attn is not None:
            media_attn = self.norm_attn(media)
        else:
            media_attn = media

        if self.norm_mlp is not None:
            media_mlp = self.norm_mlp(media)
        else:
            media_mlp = media

        media_attn = self.dropout(media_attn)
        media_mlp = self.dropout(media_mlp)

        # Compute attention scores
        attention_features = torch.tanh(self.attention_V(media_attn))
        if self.gating_V is not None:
            gating_features = torch.sigmoid(self.gating_V(media_attn))
            score_features = attention_features * gating_features
        else:
            score_features = attention_features
        scores = self.W(score_features)  # [N, W_out]

        # Compute patch-level predictions
        patch_logits = self.mlp(media_mlp)  # [N, W_out]

        if torch_scatter is None:
            raise ImportError("torch_scatter is required for softmax attention in MIL")
        
        # Create instance IDs
        num_seqs = len(cu_seqlens) - 1
        instance_ids = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)
        for i in range(num_seqs):
            instance_ids[cu_seqlens[i]:cu_seqlens[i+1]] = i

        class_outputs = []
        attention_weights = torch.zeros_like(scores)  # [N, W_out]
        
        for c in range(self.W_out):
            class_scores = scores[:, c]  # [N]
            
            # Compute softmax with numerical stability
            max_scores = torch_scatter.segment_csr(
                src=class_scores,
                indptr=cu_seqlens.long(),
                reduce='max'
            )  # [B]
            
            class_scores = class_scores - max_scores[instance_ids]  # [N]
            exp_scores = torch.exp(class_scores)  # [N]
            
            normalizers = torch_scatter.segment_csr(
                src=exp_scores,
                indptr=cu_seqlens.long(),
                reduce='sum'
            )  # [B]
            
            class_attention = exp_scores / normalizers[instance_ids]  # [N]
            attention_weights[:, c] = class_attention
            
            # Apply attention to patch logits
            if patch_logits.size(1) > 1:
                class_output = torch_scatter.segment_csr(
                    src=class_attention * patch_logits[:, c],
                    indptr=cu_seqlens.long(),
                    reduce='sum'
                )  # [B]
            else:
                class_output = torch_scatter.segment_csr(
                    src=class_attention * patch_logits.squeeze(-1),
                    indptr=cu_seqlens.long(),
                    reduce='sum'
                )  # [B]
            
            class_outputs.append(class_output)
        
        output = torch.stack(class_outputs, dim=1)  # [B, W_out]
            
        # Apply final bias/scale
        if self.use_output_bias_scale:
            output = output * self.output_scale + self.output_bias

        if return_logits:
            return output, attention_weights, patch_logits
        else:
            return output

