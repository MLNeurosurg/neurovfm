# Copyright (c) 2022, Tri Dao.
# Inspired by / adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import math
import re
from collections import OrderedDict
from copy import deepcopy
from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import trunc_normal_
from torchvision.ops import StochasticDepth
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func
from flash_attn.layers.patch_embed import PatchEmbed

try:
    from flash_attn.ops.triton.layer_norm import layer_norm_fn, RMSNorm
except ImportError:
    layer_norm_fn, RMSNorm = None, None

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear, FusedDense, RowParallelLinear
except ImportError:
    FusedDense, ColumnParallelLinear, RowParallelLinear = None, None, None

from flash_attn.modules.mlp import FusedMLP
try:
    from flash_attn.ops.triton.layer_norm import layer_norm_fn
except ImportError:
    layer_norm_fn = None

MLP_CHECKPOINT_LVL = 2


# --- Helper functions for padding/unpadding packed sequences ---
def pad_packed(data, cu_seqlens, max_seqlen, batch_first=True, padding_value=0.0):
    """Pads a packed tensor based on cumulative sequence lengths."""
    sequences = [data[cu_seqlens[i]:cu_seqlens[i+1]] for i in range(len(cu_seqlens) - 1)]
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)

    # Ensure padded sequence has length max_seqlen
    B, N, D = padded.shape
    if N < max_seqlen:
        padding_size = max_seqlen - N
        pad_shape = (B, padding_size, D) if batch_first else (padding_size, B, D)
        paddings = torch.full(pad_shape, padding_value, device=data.device, dtype=data.dtype)
        dim_to_pad = 1 if batch_first else 0
        padded = torch.cat([padded, paddings], dim=dim_to_pad)
    elif N > max_seqlen:
         # This shouldn't happen if max_seqlen is correct
         padded = padded[:, :max_seqlen, :] if batch_first else padded[:max_seqlen, :, :]


    # Create mask indicating valid (non-padded) elements
    seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    indices = torch.arange(max_seqlen, device=data.device)
    if batch_first:
        mask = indices.unsqueeze(0) < seq_lengths.unsqueeze(1) # Shape [B, max_seqlen]
    else:
        # This implementation assumes batch_first=True for simplicity
        raise NotImplementedError("pad_packed currently requires batch_first=True")

    return padded, mask # padded: (B, max_seqlen, D), mask: (B, max_seqlen)

def unpad_packed(padded_data, mask):
    """Unpads a tensor using a boolean mask, assuming batch_first=True."""
    # padded_data: (B, max_len, D), mask: (B, max_len)
    B, N, D = padded_data.shape
    unpadded_data = padded_data[mask] # Select elements where mask is True
    return unpadded_data # (total_tokens, D)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
    ):
        super().__init__()
        factory_kwargs = {'device': 'cpu', 'dtype': torch.bfloat16}

        self.dim_head = dim_head
        self.num_heads = num_heads
        inner_dim = dim_head * num_heads

        self.scale = qk_scale or dim_head ** -0.5
        self.qkv = FusedDense(dim, inner_dim*3, bias=qkv_bias, **factory_kwargs)
        self.proj = FusedDense(inner_dim, dim, bias=False, **factory_kwargs)

        self.attn_drop = attn_drop

    def forward(self, x, cu_seqlens=None, max_seqlen=None, use_flash_attn=True, return_attn_weights=False):
        """
        Forward pass for SelfAttention.

        Args:
            x (torch.Tensor): Input tensor. Shape (total_tokens, dim) if cu_seqlens is provided,
                              otherwise (batch_size, seq_len, dim).
            cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths for packed sequences.
            max_seqlen (int, optional): Maximum sequence length for packed sequences.
            use_flash_attn (bool): If True, use FlashAttention. Otherwise, use standard PyTorch attention.
            return_attn_weights (bool): If True and use_flash_attn is False, return attention weights.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor or None: Attention weights if return_attn_weights is True and use_flash_attn is False,
                                  otherwise None.
        """
        attn_weights_to_return = None # Initialize

        if use_flash_attn:
            if return_attn_weights:
                warnings.warn("Cannot return attention weights when use_flash_attn is True.")
            if cu_seqlens is None:
                # FlashAttention expects packed format. Need to pack if input is standard B, N, D
                # This requires knowing the original sequence lengths if padding exists.
                # Assuming no padding if cu_seqlens is None for now.
                 B, N, D = x.shape
                 cu_seqlens = torch.arange(0, (B + 1) * N, step=N, dtype=torch.int32, device=x.device)
                 max_seqlen = N
                 x = x.reshape(B * N, D) # Flatten to (total_tokens, dim)

            qkv = self.qkv(x).reshape(x.size(0), 3, self.num_heads, self.dim_head)
            # Note: flash_attn_varlen_qkvpacked_func implicitly handles unpadding internally for its output format
            x_out = flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_seqlen,
                dropout_p=self.attn_drop if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False
            ).flatten(start_dim=1) # Output shape (total_tokens, inner_dim)
            x_out = self.proj(x_out) # Output shape (total_tokens, dim)

        else: # Use standard PyTorch attention
            is_packed_input = cu_seqlens is not None
            if is_packed_input:
                if max_seqlen is None:
                    raise ValueError("max_seqlen must be provided when cu_seqlens is used with standard attention.")
                # Pad the input
                x_padded, attn_mask = pad_packed(x, cu_seqlens, max_seqlen, batch_first=True) # x_padded: (B, N, D), attn_mask: (B, N)
                B, N, D = x_padded.shape
            else:
                # Input is already in (B, N, D) format
                x_padded = x
                B, N, D = x_padded.shape
                # Create a dummy mask (all True) if no padding is involved
                attn_mask = torch.ones(B, N, dtype=torch.bool, device=x_padded.device)

            C = self.num_heads * self.dim_head

            qkv = self.qkv(x_padded) # (B, N, 3 * C)
            qkv = qkv.reshape(B, N, 3, self.num_heads, self.dim_head).permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, dim_head)
            q, k, v = qkv.unbind(0) # (B, num_heads, N, dim_head)

            # Calculate attention scores
            # (B, H, N, d) @ (B, H, d, N) -> (B, H, N, N)
            attn = (q @ k.transpose(-2, -1)) * self.scale

            # Apply the attention mask
            # attn_mask shape (B, N) -> (B, 1, 1, N) to broadcast over heads and query sequence length
            mask_value = torch.finfo(attn.dtype).min # Use min value of the dtype for masking
            # mask_value = -torch.inf # Alternative if using float32/64
            attn_mask_expanded = attn_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, N)
            attn = attn.masked_fill(~attn_mask_expanded, mask_value)

            # Softmax and Dropout
            attn = attn.softmax(dim=-1)
            if return_attn_weights:
                attn_weights_to_return = attn # Store weights after softmax, before dropout

            attn = F.dropout(attn, p=self.attn_drop, training=self.training)

            # Apply attention to values
            # (B, H, N, N) @ (B, H, N, d) -> (B, H, N, d)
            y = attn @ v
            y = y.transpose(1, 2).reshape(B, N, C) # (B, N, C)

            # Final projection
            x_out_padded = self.proj(y) # (B, N, dim)

            # Unpad if the original input was packed
            if is_packed_input:
                x_out = unpad_packed(x_out_padded, attn_mask) # (total_tokens, dim)
            else:
                x_out = x_out_padded # (B, N, dim)

        return x_out, attn_weights_to_return


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
    ):
        super().__init__()
        factory_kwargs = {'device': 'cpu', 'dtype': torch.bfloat16}

        self.dim_head = dim_head
        self.num_heads = num_heads
        inner_dim = dim_head * num_heads

        self.scale = qk_scale or dim_head ** -0.5
        self.q = FusedDense(dim, inner_dim, bias=qkv_bias, **factory_kwargs)
        self.kv = FusedDense(dim, inner_dim*2, bias=qkv_bias, **factory_kwargs)
        self.proj = FusedDense(inner_dim, dim, bias=False, **factory_kwargs)

        self.attn_drop = attn_drop

    def forward(self, x, cu_seqlens=None, max_seqlen=None, k=None, cu_seqlens_k=None, max_seqlen_k=None):
        if k is None:
            q = self.q(x[cu_seqlens[:-1]]).reshape(len(cu_seqlens)-1, self.num_heads, self.dim_head)
            kv = self.kv(x).reshape(x.size(0), 2, self.num_heads, self.dim_head)

            cu_seqlens_q = torch.arange(q.size(0)+1, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
            max_seqlen_q = 1

            q = flash_attn_varlen_kvpacked_func(
                q, kv, 
                cu_seqlens_q=cu_seqlens_q, 
                cu_seqlens_k=cu_seqlens, 
                max_seqlen_q=max_seqlen_q, 
                max_seqlen_k=max_seqlen, 
                dropout_p=self.attn_drop, 
                softmax_scale=self.scale
            ).flatten(start_dim=1)

        else:
            q = self.q(x).reshape(x.size(0), self.num_heads, self.dim_head)
            kv = self.kv(k).reshape(k.size(0), 2, self.num_heads, self.dim_head)

            q = flash_attn_varlen_kvpacked_func(
                q, kv, 
                cu_seqlens_q=cu_seqlens, 
                cu_seqlens_k=cu_seqlens_k, 
                max_seqlen_q=max_seqlen, 
                max_seqlen_k=max_seqlen_k, 
                dropout_p=self.attn_drop, 
                softmax_scale=self.scale
            ).flatten(start_dim=1)

        q = self.proj(q)
        return q


class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_head,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        dropout_layer=nn.Dropout,
        resid_dropout1=0.0,
        resid_dropout2=0.0,
        drop_path1=0.0,
        drop_path2=0.0,
        use_cross_attn=False):
        super().__init__()

        if use_cross_attn:
            self.mixer = CrossAttention(dim, dim_head, num_heads, qkv_bias, qk_scale, attn_drop)
        else:
            self.mixer = SelfAttention(dim, dim_head, num_heads, qkv_bias, qk_scale, attn_drop)
        self.use_cross_attn = use_cross_attn

        self.drop1 = dropout_layer(resid_dropout1)
        self.drop2 = dropout_layer(resid_dropout2)
        self.drop_path1 = StochasticDepth(drop_path1, mode="row")
        self.drop_path2 = StochasticDepth(drop_path2, mode="row")
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FusedMLP(in_features=dim, 
                            hidden_features=mlp_hidden_dim, 
                            checkpoint_lvl=MLP_CHECKPOINT_LVL, 
                            return_residual=False)

    def forward(self,
                x,
                cu_seqlens=None,
                max_seqlen=None,
                residual=None,
                use_flash_attn=True,
                return_attn_weights=False):

        if self.drop_path1.p == 0 or not self.training:
            rowscale1 = None
        else:
            rowscale1 = self.drop_path1(
                torch.ones(
                    x.shape[:-1],
                    device=x.device,
                    dtype=x.dtype,
                )
            )

        x, residual = layer_norm_fn(
            x,
            self.norm1.weight,
            self.norm1.bias,
            residual=residual,
            eps=self.norm1.eps,
            dropout_p=self.drop1.p if self.training else 0.0,
            rowscale=rowscale1,
            prenorm=True,
            residual_in_fp32=True,
            is_rms_norm=isinstance(self.norm1, RMSNorm),
        )

        x, attn_weights = self.mixer(x, cu_seqlens, max_seqlen, use_flash_attn, return_attn_weights)

        if self.use_cross_attn:
            residual = residual[cu_seqlens[:-1]]

        if self.drop_path2.p == 0 or not self.training:
            rowscale2 = None
        else:
            rowscale2 = self.drop_path2(
                torch.ones(
                    x.shape[:-1],
                    device=x.device,
                    dtype=x.dtype,
                )
            )

        x, residual = layer_norm_fn(
            x,
            self.norm2.weight,
            self.norm2.bias,
            residual=residual,
            eps=self.norm2.eps,
            dropout_p=self.drop2.p if self.training else 0.0,
            rowscale=rowscale2,
            prenorm=True,
            residual_in_fp32=True,
            is_rms_norm=isinstance(self.norm2, RMSNorm),
        )
        x = self.mlp(x)
        return x, residual, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        prefix_len: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        act_layer: nn.Module = nn.GELU,
        init_std: float = 0.02,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        self.num_heads = num_heads
        self.num_prefix_tokens = prefix_len

        if self.num_prefix_tokens > 0:
            raise NotImplementedError("Prefix tokens not supported yet")
            # self.cls_token = nn.Parameter(torch.zeros(1, self.num_prefix_tokens, embed_dim))
        else:
            self.cls_token = None

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.blocks = nn.ModuleList([
            Block(embed_dim, 
                  embed_dim // num_heads, 
                  num_heads, 
                  mlp_ratio=mlp_ratio, 
                  qkv_bias=qkv_bias, 
                  qk_scale=None,
                  attn_drop=attn_drop_rate, 
                  drop_path1=dpr[i-1] if i > 0 else 0.0,
                  drop_path2=dpr[i],
                  norm_layer=norm_layer,
                  act_layer=act_layer,
                  use_cross_attn=False)
            for i in range(depth)
        ])

        self.dropout = nn.Dropout(p=drop_rate)
        self.drop_path = StochasticDepth(p=dpr[-1], mode="row")
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.mixer.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, cu_seqlens=None, max_seqlen=None, use_flash_attn=True, return_attn_weights=False):
        hidden_states = x
        if self.cls_token is not None:
            raise NotImplementedError("Prefix tokens not supported yet")
        
        # -- fwd prop
        residual = None
        all_attn_weights = [] if return_attn_weights else None # Initialize correctly

        for i, blk in enumerate(self.blocks):
            # --- Ensure Block.forward accepts these flags and returns weights ---
            # Expected return from blk: (output_hidden_states, output_residual, block_attn_weights)
            # block_attn_weights will be None if return_attn_weights is False or use_flash_attn is True
            block_output = blk(hidden_states,
                               cu_seqlens=cu_seqlens,
                               max_seqlen=max_seqlen,
                               residual=residual,
                               # Pass flags down to the block:
                               use_flash_attn=use_flash_attn,
                               return_attn_weights=return_attn_weights)

            # --- Unpack block output ---
            hidden_states = block_output[0]
            residual = block_output[1]
            block_attn_weights = block_output[2] # This should be the third element returned by Block.forward

            if return_attn_weights and block_attn_weights is not None:
                 all_attn_weights.append(block_attn_weights) # Append the weights from the block
            # --- End modifications for block call ---

        # -- drop, add, norm
        if self.drop_path.p == 0 or not self.training:
            rowscale = None
        else:
            rowscale = self.drop_path(
                torch.ones(
                    hidden_states.shape[:-1],
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            )
        
        if self.norm is not None:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                eps=self.norm.eps,
                dropout_p=self.dropout.p if self.training else 0.0,
                rowscale=rowscale,
                prenorm=False,
                is_rms_norm=isinstance(self.norm, RMSNorm),
            )
        
        if return_attn_weights:
            return hidden_states, all_attn_weights
        else:
            return hidden_states


# Import for VisionPredictor (from user's codebase)
try:
    from waldo.models.perceiver import PerceiverBlock
except ImportError:
    PerceiverBlock = None

from neurovfm.models.pos_embed import PositionalEncoding3DWrapper
from neurovfm.models.patch_embed import PatchEmbed


class VisionPredictor(nn.Module):
    """
    Transformer-based predictor for masked token prediction.
    
    Predicts masked tokens given context tokens using transformer blocks.
    Supports optional cross-attention with text embeddings via Perceiver blocks.
    
    Args:
        vision_encoder_dim (int): Dimension of vision encoder outputs
        dim (int): Internal embedding dimension
        depth (int): Number of transformer blocks
        dim_head (int): Dimension of each attention head
        num_heads (int): Number of attention heads
        prefix_len (int): Number of prefix tokens (e.g., CLS tokens)
        text_encoder_dim (int, optional): Dimension of text encoder outputs
        use_perceiver_block (bool): Whether to use Perceiver blocks for cross-attention.
                                    Defaults to False.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
        qkv_bias (bool): Whether to include bias in QKV projection. Defaults to True.
        qk_scale (float, optional): Scale factor for QK attention
        drop_rate (float): Dropout rate. Defaults to 0.0.
        attn_drop_rate (float): Attention dropout rate. Defaults to 0.0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.0.
        norm_layer: Normalization layer constructor. Defaults to LayerNorm.
        act_layer: Activation layer constructor. Defaults to GELU.
        use_flash_attn (bool): Whether to use flash attention. Defaults to True.
        init_std (float): Standard deviation for weight initialization. Defaults to 0.02.
        pos_emb_cf (Dict, optional): Position embedding configuration
        use_mask_tokens (bool): Whether to use learnable mask tokens. Defaults to True.
        num_mask_tokens (int): Number of different mask token embeddings. Defaults to 1.
        zero_init_mask_tokens (bool): Whether to initialize mask tokens to zero.
                                      Defaults to True.
    
    Example:
        >>> import torch
        >>> from neurovfm.models import VisionPredictor
        >>> 
        >>> # Create predictor
        >>> predictor = VisionPredictor(
        ...     vision_encoder_dim=768,
        ...     dim=512,
        ...     depth=4,
        ...     dim_head=64,
        ...     num_heads=8,
        ...     prefix_len=0
        ... )
        >>> 
        >>> # Predict masked tokens
        >>> encoder_out = torch.randn(100, 768)  # Context tokens
        >>> coords = torch.randn(150, 3)  # 3D coordinates
        >>> info_cls_ctxt = (torch.arange(100), torch.tensor([0, 100]), 100)
        >>> info_tgt = (torch.arange(100, 150), torch.tensor([0, 50]), 50)
        >>> 
        >>> predictions, mask = predictor(
        ...     encoder_out, coords, info_cls_ctxt, info_tgt
        ... )
    
    Notes:
        - Implements masked prediction similar to MAE/JEPA architectures
        - Can incorporate text conditioning via Perceiver blocks
        - Supports multiple learnable mask token embeddings for diversity
    """
    
    def __init__(
        self,
        vision_encoder_dim: int,
        dim: int,
        depth: int,
        dim_head: int,
        num_heads: int,
        prefix_len: int,
        text_encoder_dim: Optional[int] = None,
        use_perceiver_block: bool = False,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        act_layer: nn.Module = nn.GELU,
        use_flash_attn: bool = True,
        init_std: float = 0.02,
        pos_emb_cf: Dict = None,
        use_mask_tokens: bool = True,
        num_mask_tokens: int = 1,
        zero_init_mask_tokens: bool = True,
    ):
        super().__init__()
        self.num_features = self.dim = dim
        self.num_prefix_tokens = prefix_len

        if FusedDense is None:
            raise ImportError("FusedDense from flash_attn is required")

        # Projection layers
        if pos_emb_cf is not None:
            if pos_emb_cf["params"]["concat"]:
                self.from_vision_encoder = FusedDense(vision_encoder_dim+pos_emb_cf["params"]["d"], dim, bias=True)
            else:
                self.from_vision_encoder = FusedDense(vision_encoder_dim, dim, bias=True)
        else:
            self.from_vision_encoder = FusedDense(vision_encoder_dim, dim, bias=True)

        self.to_vision_encoder = FusedDense(dim, vision_encoder_dim, bias=True)
        if text_encoder_dim is not None:
            self.from_text_encoder = FusedDense(text_encoder_dim, dim, bias=True)
        else:
            self.from_text_encoder = None

        # Mask tokens
        self.mask_tokens = None
        self.num_mask_tokens = 0
        if use_mask_tokens:
            self.num_mask_tokens = num_mask_tokens
            self.mask_tokens = nn.ParameterList([
                    nn.Parameter(torch.zeros(1, vision_encoder_dim))
                    for i in range(num_mask_tokens)
                ])

        assert depth > 0, "Depth must be greater than 0"
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        assert self.from_text_encoder is not None or not use_perceiver_block, "Perceiver block is only supported with text"
        
        if use_perceiver_block and PerceiverBlock is None:
            raise ImportError("PerceiverBlock not available. Install waldo or disable use_perceiver_block")
        
        self.blocks = nn.ModuleList([
            Block(dim,
                  dim_head,
                  num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  attn_drop=attn_drop_rate,
                  drop_path1=dpr[i-1] if i > 0 else 0.0,
                  drop_path2=dpr[i],
                  norm_layer=norm_layer,
                  act_layer=act_layer,
                  use_cross_attn=False)
            for i in range(depth)
        ])

        self.dropout = nn.Dropout(p=drop_rate)
        self.drop_path = StochasticDepth(p=dpr[-1], mode="row")
        self.norm = norm_layer(dim)

        # Initialize mask tokens
        if not zero_init_mask_tokens:
            for mt in self.mask_tokens:
                trunc_normal_(mt, std=init_std)
        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

        # Position embedding
        if pos_emb_cf is not None:
            if pos_emb_cf["which"] == "pe3d":
                self.pos_embed = PositionalEncoding3DWrapper(**pos_emb_cf["params"])
            else:
                raise ValueError(f"Positional embedding {pos_emb_cf.get('which', None)} not supported")
        else:
            self.pos_embed = None

    def fix_init_weight(self):
        """Rescale weights for better initialization."""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for block_id, block in enumerate(self.blocks):
            rescale(block.mixer.proj.weight.data, block_id + 1)
            rescale(block.mlp.fc2.weight.data, block_id + 1)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        vision_encoder_out: torch.Tensor,
        coords: torch.Tensor,
        info_cls_ctxt: Tuple[torch.Tensor, torch.Tensor, int],
        info_tgt: Tuple[torch.Tensor, torch.Tensor, int],
        info_cls_ctxt_tgt: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None,
        text_encoder_out: Optional[torch.Tensor] = None,
        info_media: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None,
        mask_index: int = 0,
        use_flash_attn: bool = True,
        return_attn_weights: bool = False,
        return_latents: bool = False
    ):
        """
        Forward pass for masked prediction.
        
        Args:
            vision_encoder_out (torch.Tensor): Encoder output features [N_context, vision_encoder_dim]
            coords (torch.Tensor): 3D coordinates for all tokens [N_total, 3]
            info_cls_ctxt (Tuple): Context tokens info (masks, cu_seqlens, max_seqlen)
            info_tgt (Tuple): Target (masked) tokens info (masks, cu_seqlens, max_seqlen)
            info_cls_ctxt_tgt (Tuple, optional): Combined context+target info
            text_encoder_out (torch.Tensor, optional): Text embeddings [N_text, text_encoder_dim]
            info_media (Tuple, optional): Text tokens info (masks, cu_seqlens, max_seqlen)
            mask_index (int): Which mask token to use (for multi-mask). Defaults to 0.
            use_flash_attn (bool): Whether to use flash attention. Defaults to True.
            return_attn_weights (bool): Whether to return attention weights. Defaults to False.
            return_latents (bool): Whether to return latent features. Defaults to False.
        
        Returns:
            Tuple: (predictions, target_mask) where predictions are in vision encoder space
                   Optional: attention weights or latents if requested
        """
        masks_cls_ctxt, masks_tgt = info_cls_ctxt[0], info_tgt[0]
        assert (masks_cls_ctxt is not None) and (masks_tgt is not None), 'Cannot run predictor without mask indices'

        if info_cls_ctxt_tgt is not None and info_cls_ctxt_tgt[0] is not None:
            masks_cls_ctxt_tgt = info_cls_ctxt_tgt[0]
            coords = coords[masks_cls_ctxt_tgt]

        # Add positional embedding to vision encoder output
        if self.pos_embed is not None:
            if self.pos_embed.concat:
                vision_encoder_out = self.pos_embed(vision_encoder_out.unsqueeze(0), coords[masks_cls_ctxt].unsqueeze(0)).squeeze(0)
            else:
                pos_embed = self.pos_embed(vision_encoder_out.unsqueeze(0), coords[masks_cls_ctxt].unsqueeze(0)).squeeze(0)
                vision_encoder_out = vision_encoder_out + pos_embed

        # Project to predictor embed dim
        latents = self.from_vision_encoder(vision_encoder_out)
        N_cls_ctxt, D = latents.shape

        # Initialize masked pred tokens and add pos embed
        if self.mask_tokens is None:
            raise NotImplementedError()
        else:
            mask_index = mask_index % self.num_mask_tokens
            pred_tokens = self.mask_tokens[mask_index]
            pred_tokens = pred_tokens.repeat(masks_tgt.size(0), 1)

            if self.pos_embed is not None:
                if self.pos_embed.concat:
                    pred_tokens = self.pos_embed(pred_tokens.unsqueeze(0), coords[masks_tgt].unsqueeze(0)).squeeze(0)
                else:
                    pred_tokens += self.pos_embed(pred_tokens.unsqueeze(0), coords[masks_tgt].unsqueeze(0)).squeeze(0)

            pred_tokens = self.from_vision_encoder(pred_tokens)

        if (isinstance(self.blocks[0], Block) and info_media is None) or isinstance(self.blocks[0], PerceiverBlock):
            # Compute info for new latents
            cls_ctxt_sizes = info_cls_ctxt[1][1:] - info_cls_ctxt[1][:-1]
            tgt_sizes = info_tgt[1][1:] - info_tgt[1][:-1]
            batch_sizes = cls_ctxt_sizes + tgt_sizes
            cu_seqlens_q = torch.cat([torch.tensor([0], device=batch_sizes.device), batch_sizes.cumsum(0)])
            cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32)
            max_seqlen_q = batch_sizes.max().item()

            # Concat cls+ctxt (from encoder) and masked target tokens
            latents = torch.cat([
                torch.cat([latents_slice, pred_slice])
                for latents_slice, pred_slice in zip(
                    torch.split(latents, cls_ctxt_sizes.tolist()),
                    torch.split(pred_tokens, tgt_sizes.tolist())
                )
            ])
            toselect = []
            for cls_ctxt_size, tgt_size in zip(cls_ctxt_sizes, tgt_sizes):
                toselect.extend([0]*cls_ctxt_size + [1]*tgt_size)
            toselect = torch.tensor(toselect, dtype=torch.bool).to(latents.device)

            if self.from_text_encoder is not None:
                # Project latents to dim and reorder
                media = self.from_text_encoder(text_encoder_out)
        
                # Compute info for latents + media
                media_sizes = info_media[1][1:] - info_media[1][:-1]
                latents_sizes = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                batch_sizes = media_sizes + latents_sizes

                cu_seqlens_k = torch.cat([torch.tensor([0], device=batch_sizes.device), batch_sizes.cumsum(0)])
                cu_seqlens_k = cu_seqlens_k.to(dtype=torch.int32)
                max_seqlen_k = batch_sizes.max().item()
        elif (isinstance(self.blocks[0], Block) and info_media is not None):
            assert self.from_text_encoder is not None, "Info media passed without specifying text encoder"
            
            # Project latents to dim and reorder
            media = self.from_text_encoder(text_encoder_out)

            # Compute info for new latents
            media_sizes = info_media[1][1:] - info_media[1][:-1]
            cls_ctxt_sizes = info_cls_ctxt[1][1:] - info_cls_ctxt[1][:-1]
            tgt_sizes = info_tgt[1][1:] - info_tgt[1][:-1]
            batch_sizes = media_sizes + cls_ctxt_sizes + tgt_sizes
            cu_seqlens_q = torch.cat([torch.tensor([0], device=batch_sizes.device), batch_sizes.cumsum(0)])
            cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32)
            max_seqlen_q = batch_sizes.max().item()

            # Concat media, cls+ctxt (from encoder) and masked target tokens
            latents = torch.cat([
                torch.cat([media_slice, latents_slice, pred_slice])
                for media_slice, latents_slice, pred_slice in zip(
                    torch.split(media, media_sizes.tolist()),
                    torch.split(latents, cls_ctxt_sizes.tolist()),
                    torch.split(pred_tokens, tgt_sizes.tolist())
                )
            ])
            toselect = []
            for media_size, cls_ctxt_size, tgt_size in zip(media_sizes, cls_ctxt_sizes, tgt_sizes):
                toselect.extend([0]*media_size + [0]*cls_ctxt_size + [1]*tgt_size)
            toselect = torch.tensor(toselect, dtype=torch.bool).to(latents.device)
        else:
            raise ValueError("Invalid block type and media combination")

        # Forward propagation
        residual = None
        all_attn_weights = [] if return_attn_weights else None

        for block in self.blocks:
            block_output = block(latents,
                                    cu_seqlens=cu_seqlens_q,
                                    max_seqlen=max_seqlen_q,
                                    residual=residual,
                                    use_flash_attn=use_flash_attn,
                                    return_attn_weights=return_attn_weights)
            
            latents = block_output[0]
            residual = block_output[1]
            block_attn_weights = block_output[2]

            if return_attn_weights and block_attn_weights is not None:
                all_attn_weights.append(block_attn_weights)

        # Drop, add, norm
        if self.drop_path.p == 0 or not self.training:
            rowscale = None
        else:
            rowscale = self.drop_path(
                torch.ones(
                    latents.shape[:-1],
                    device=latents.device,
                    dtype=latents.dtype,
                )
            )
        
        if self.norm is not None:
            latents = layer_norm_fn(
                latents,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                eps=self.norm.eps,
                dropout_p=self.dropout.p if self.training else 0.0,
                rowscale=rowscale,
                prenorm=False,
                is_rms_norm=isinstance(self.norm, RMSNorm),
            )

        if return_latents:
            return latents, toselect

        if return_attn_weights:
            return self.to_vision_encoder(latents), toselect, all_attn_weights
        else:
            return self.to_vision_encoder(latents), toselect


class VisionTransformer(TransformerEncoder):
    """
    Vision Transformer (ViT) for 3D medical imaging.
    
    Extends the base TransformerEncoder with support for volumetric patch embeddings
    and 3D positional encodings. Designed for processing tokenized 3D medical images
    where each token represents a 3D patch (e.g., 4x16x16 voxels).
    
    Args:
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer blocks
        num_heads (int): Number of attention heads
        prefix_len (int): Number of prefix tokens (e.g., CLS)
        token_dim (int): Dimension of input tokens. Defaults to 259 (256 + 3 for coords).
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
        qkv_bias (bool): Whether to include bias in QKV projection. Defaults to True.
        drop_rate (float): Dropout rate. Defaults to 0.0.
        attn_drop_rate (float): Attention dropout rate. Defaults to 0.0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.0.
        embed_layer_cf (Dict): Embedding layer configuration with keys:
                               - 'which': 'voxel' or 'linear'
                               - 'params': parameters for embedding layer
        norm_layer: Normalization layer constructor. Defaults to LayerNorm.
        act_layer: Activation layer constructor. Defaults to GELU.
        init_std (float): Standard deviation for weight initialization. Defaults to 0.02.
        pos_emb_cf (Dict, optional): Position embedding configuration with keys:
                                     - 'which': 'pe3d'
                                     - 'params': parameters for positional encoding
    
    Example:
        >>> import torch
        >>> from neurovfm.models import VisionTransformer
        >>> 
        >>> # Create ViT model
        >>> model = VisionTransformer(
        ...     embed_dim=768,
        ...     depth=12,
        ...     num_heads=12,
        ...     prefix_len=0,
        ...     token_dim=1024,  # 1*4*16*16
        ...     embed_layer_cf={
        ...         'which': 'voxel',
        ...         'params': {
        ...             'patch_hw_size': 16,
        ...             'patch_d_size': 4,
        ...             'in_chans': 1,
        ...             'embed_dim': 738
        ...         }
        ...     }
        ... )
        >>> 
        >>> # Forward pass
        >>> tokens = torch.randn(100, 1024)  # [N, token_dim]
        >>> coords = torch.randn(100, 3)     # [N, 3] for 3D positions
        >>> cu_seqlens = torch.tensor([0, 50, 100], dtype=torch.int32)
        >>> 
        >>> features = model(tokens, coords, cu_seqlens=cu_seqlens, max_seqlen=50)
    
    Notes:
        - Supports both 'voxel' (PatchEmbed) and 'linear' embedding layers
        - Uses 3D positional encodings
        - Implements gradient checkpointing during training for memory efficiency
        - Compatible with Flash Attention for efficient attention computation
    """
    
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        prefix_len: int,
        token_dim: int = 259,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer_cf: Dict = None,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        act_layer: nn.Module = nn.GELU,
        init_std: float = 0.02,
        pos_emb_cf: Dict = None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            prefix_len=prefix_len,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        
        if FusedDense is None:
            raise ImportError("FusedDense from flash_attn is required")
        
        # Token embedding layer
        if embed_layer_cf["which"] == "voxel":
            self.token_embed = PatchEmbed(**embed_layer_cf["params"])
        else:
            if pos_emb_cf is not None:
                if pos_emb_cf["params"]["concat"]:
                    self.token_embed = FusedDense(token_dim+pos_emb_cf["params"]["d"], embed_dim, bias=True)
                else:
                    self.token_embed = FusedDense(token_dim, embed_dim, bias=True)
            else:
                self.token_embed = FusedDense(token_dim, embed_dim, bias=True)

        self.apply(self._init_weights)
        self.fix_init_weight()

        # Positional embedding
        if pos_emb_cf is not None:
            if pos_emb_cf["which"] == "pe3d":
                self.pos_embed = PositionalEncoding3DWrapper(**pos_emb_cf["params"])
            else:
                raise ValueError(f"Positional embedding {pos_emb_cf.get('which', None)} not supported")
        else:
            self.pos_embed = None
        
    @torch.jit.ignore
    def no_weight_decay(self):
        """Returns parameter names that should not use weight decay."""
        return {"pos_embed", "cls_token"}

    def forward_features(
        self, 
        x: torch.Tensor,
        coords: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        masks_enc: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        use_flash_attn: bool = True,
        return_attn_weights: bool = False
    ):
        """
        Extract features from input tokens.
        
        Args:
            x (torch.Tensor): Input tokens [N, token_dim]
            coords (torch.Tensor): 3D coordinates [N, 3]
            masks (torch.Tensor, optional): Mask for selecting tokens
            masks_enc (torch.Tensor, optional): Additional encoder mask
            cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths [B+1]
            max_seqlen (int, optional): Maximum sequence length
            use_flash_attn (bool): Whether to use flash attention. Defaults to True.
            return_attn_weights (bool): Whether to return attention weights. Defaults to False.
        
        Returns:
            torch.Tensor: Output features [N, embed_dim]
            OR Tuple[torch.Tensor, List]: (features, attention_weights) if return_attn_weights=True
        """
        if isinstance(self.token_embed, PatchEmbed):
            # Apply masks if provided
            if masks is not None:
                x = x[masks]
                coords = coords[masks]
            if masks_enc is not None:
                x = x[masks_enc]
                coords = coords[masks_enc]

            # Dropout
            x = self.dropout(x)

            # Project x to embed dim
            if self.pos_embed is not None:
                if self.pos_embed.concat:
                    hidden_states = self.token_embed(x).reshape(-1, self.embed_dim-self.pos_embed.d)
                else:
                    hidden_states = self.token_embed(x).reshape(-1, self.embed_dim)
            else:
                hidden_states = self.token_embed(x).reshape(-1, self.embed_dim)

            # Add positional embedding to x
            if self.pos_embed is not None:
                if self.pos_embed.concat:
                    hidden_states = self.pos_embed(hidden_states.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
                else:
                    pos_embed = self.pos_embed(hidden_states.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
                    hidden_states = hidden_states + pos_embed

            N, D = hidden_states.shape
        else:
            # Apply masks if provided
            if masks is not None:
                x = x[masks]
                coords = coords[masks]
            if masks_enc is not None:
                x = x[masks_enc]
                coords = coords[masks_enc]

            # Dropout
            x = self.dropout(x)
            
            # Add positional embedding to x
            if self.pos_embed is not None:
                if self.pos_embed.concat:
                    x = self.pos_embed(x.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
                else:
                    pos_embed = self.pos_embed(x.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
                    x = x + pos_embed

            # Project x to embed dim
            hidden_states = self.token_embed(x)
            N, D = hidden_states.shape

        # Forward propagation through blocks
        residual = None
        all_attn_weights = [] if return_attn_weights else None

        for i, blk in enumerate(self.blocks):
            if self.training:
                block_output = checkpoint(
                    blk,
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    residual,
                    use_flash_attn,
                    return_attn_weights,
                    use_reentrant=False)
            else:
                block_output = blk(hidden_states, 
                                   cu_seqlens=cu_seqlens, 
                                   max_seqlen=max_seqlen, 
                                   residual=residual,
                                   use_flash_attn=use_flash_attn,
                                   return_attn_weights=return_attn_weights)
                
            hidden_states = block_output[0]
            residual = block_output[1]
            block_attn_weights = block_output[2]

            if return_attn_weights and block_attn_weights is not None:
                 all_attn_weights.append(block_attn_weights) 
    
        # Drop, add, norm
        if self.drop_path.p == 0 or not self.training:
            rowscale = None
        else:
            rowscale = self.drop_path(
                torch.ones(
                    hidden_states.shape[:-1],
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            )
        
        if self.norm is not None:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                eps=self.norm.eps,
                dropout_p=self.dropout.p if self.training else 0.0,
                rowscale=rowscale,
                prenorm=False,
                is_rms_norm=isinstance(self.norm, RMSNorm),
            )
        
        if return_attn_weights:
            return hidden_states, all_attn_weights
        else:
            return hidden_states

    def forward_penultimate_features(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        masks_enc: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        use_flash_attn: bool = True,
        return_attn_weights: bool = False
    ):
        """
        Extract penultimate layer features (before final layer norm).
        
        Useful for certain training objectives that require unnormalized features.
        
        Args:
            Same as forward_features
        
        Returns:
            torch.Tensor: Penultimate features [N, embed_dim]
            OR Tuple[torch.Tensor, List]: (features, attention_weights) if return_attn_weights=True
        """
        if isinstance(self.token_embed, PatchEmbed):
            # Apply masks if provided
            if masks is not None:
                x = x[masks]
                coords = coords[masks]
            if masks_enc is not None:
                x = x[masks_enc]
                coords = coords[masks_enc]

            # Dropout
            x = self.dropout(x)

            # Project x to embed dim
            if self.pos_embed is not None:
                if self.pos_embed.concat:
                    hidden_states = self.token_embed(x).reshape(-1, self.embed_dim-self.pos_embed.d)
                else:
                    hidden_states = self.token_embed(x).reshape(-1, self.embed_dim)
            else:
                hidden_states = self.token_embed(x).reshape(-1, self.embed_dim)

            # Add positional embedding to x
            if self.pos_embed is not None:
                if self.pos_embed.concat:
                    hidden_states = self.pos_embed(hidden_states.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
                else:
                    pos_embed = self.pos_embed(hidden_states.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
                    hidden_states = hidden_states + pos_embed

            N, D = hidden_states.shape
        else:
            # Apply masks if provided
            if masks is not None:
                x = x[masks]
                coords = coords[masks]
            if masks_enc is not None:
                x = x[masks_enc]
                coords = coords[masks_enc]

            # Dropout
            x = self.dropout(x)
            
            # Add positional embedding to x
            if self.pos_embed is not None:
                if self.pos_embed.concat:
                    x = self.pos_embed(x.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
                else:
                    pos_embed = self.pos_embed(x.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
                    x = x + pos_embed

            # Project x to embed dim
            hidden_states = self.token_embed(x)
            N, D = hidden_states.shape

        # Forward propagation (skip last block)
        residual = None
        all_attn_weights = [] if return_attn_weights else None

        for i, blk in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                continue
            
            if self.training:
                block_output = checkpoint(
                    blk,
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    residual,
                    use_flash_attn,
                    return_attn_weights,
                    use_reentrant=False)
            else:
                block_output = blk(hidden_states, 
                                   cu_seqlens=cu_seqlens, 
                                   max_seqlen=max_seqlen, 
                                   residual=residual,
                                   use_flash_attn=use_flash_attn,
                                   return_attn_weights=return_attn_weights)
                
            hidden_states = block_output[0]
            residual = block_output[1]
            block_attn_weights = block_output[2]

            if return_attn_weights and block_attn_weights is not None:
                 all_attn_weights.append(block_attn_weights) 

        # Only return prenorm tokens (no final layer norm)
        hidden_states = hidden_states + residual
        
        if return_attn_weights:
            return hidden_states, all_attn_weights
        else:
            return hidden_states

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        masks_enc: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        use_flash_attn: bool = True,
        return_attn_weights: bool = False
    ):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tokens [N, token_dim]
            coords (torch.Tensor): 3D coordinates [N, 3]
            masks (torch.Tensor, optional): Mask for selecting tokens
            masks_enc (torch.Tensor, optional): Additional encoder mask
            cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths [B+1]
            max_seqlen (int, optional): Maximum sequence length
            use_flash_attn (bool): Whether to use flash attention. Defaults to True.
            return_attn_weights (bool): Whether to return attention weights. Defaults to False.
        
        Returns:
            torch.Tensor: Output features [N, embed_dim]
            OR Tuple[torch.Tensor, List]: (features, attention_weights) if return_attn_weights=True
        """
        return self.forward_features(x, coords, masks, masks_enc, cu_seqlens, max_seqlen, use_flash_attn, return_attn_weights)


def get_vit_backbone(which: str = "vit_base", params: Dict = {}) -> VisionTransformer:
    """
    Create a VisionTransformer with predefined architecture.
    
    Args:
        which (str): Model size - one of ['vit_tiny', 'vit_small', 'vit_base', 
                     'vit_large', 'vit_huge', 'vit']. Defaults to 'vit_base'.
        params (Dict): Additional parameters to pass to VisionTransformer.
    
    Returns:
        VisionTransformer: Configured model
    
    Raises:
        ValueError: If model size is not recognized
    
    Example:
        >>> from neurovfm.models import get_vit_backbone
        >>> 
        >>> # Create a base ViT model
        >>> model = get_vit_backbone('vit_base')
        >>> 
        >>> # Create a custom variant with specific params
        >>> model = get_vit_backbone('vit_large', params={
        ...     'token_dim': 1024,
        ...     'drop_rate': 0.1
        ... })
    """
    if which == "vit_tiny":
        return VisionTransformer(
            embed_dim=192,
            depth=12,
            num_heads=3,
            prefix_len=0,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **params
        )
    elif which == "vit_small":
        return VisionTransformer(
            embed_dim=384,
            depth=12,
            num_heads=6,
            prefix_len=0,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **params
        )
    elif which == "vit_base":
        return VisionTransformer(
            embed_dim=768,
            depth=12,
            num_heads=12,
            prefix_len=0,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **params
        )
    elif which == "vit_large":
        return VisionTransformer(
            embed_dim=1024,
            depth=24,
            num_heads=16,
            prefix_len=0,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **params
        )
    elif which == "vit_huge":
        return VisionTransformer(
            embed_dim=1280,
            depth=32,
            num_heads=16,
            prefix_len=0,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **params
        )
    elif which == "vit":
        return VisionTransformer(**params)
    else:
        raise ValueError(
            "ViT backbone name must be in [vit_tiny, vit_small, vit_base, vit_large, vit_huge, vit]"
        )