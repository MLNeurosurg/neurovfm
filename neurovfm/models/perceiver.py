import torch
import torch.nn as nn

try:
    from torch.distributed.fsdp.fully_sharded_data_parallel import checkpoint as fsdp_checkpoint
except ImportError:
    from torch.utils.checkpoint import checkpoint as fsdp_checkpoint


class PerceiverAttention(nn.Module):
    """
    Single cross-attention layer for Perceiver Resampler.
    Based on: https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # pre-norm
        self.norm_queries = nn.LayerNorm(dim)
        self.norm_visual = nn.LayerNorm(dim)

        # projection layers for Q, K, V
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        # ffn
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(dim * 4, dim, bias=False),
        )
        
        # store dropout probability for scaled_dot_product_attention
        self.attn_dropout_p = dropout


    def forward(self, queries: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, num_queries, dim)
            visual_features: (B, num_visual_tokens, dim)
        Returns:
            (B, num_queries, dim)
        """
        B, num_queries, _ = queries.shape
        _B, num_visual_tokens, _ = visual_features.shape


        # --- cross-attention ---
        residual = queries
        
        # pre-norm
        queries_norm = self.norm_queries(queries)
        visual_features_norm = self.norm_visual(visual_features)

        # project to Q, K, V
        q = self.to_q(queries_norm)
        k, v = self.to_kv(visual_features_norm).chunk(2, dim=-1)

        # reshape for multi-head attention
        q = q.view(B, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, num_visual_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, num_visual_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout_p if self.training else 0.0
        )

        # reshape back and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, num_queries, -1)
        attn_output = self.to_out(attn_output)

        queries = residual + attn_output

        # --- ffn ---
        queries = queries + self.ffn(queries)
        
        return queries


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler for compressing visual features into a fixed number of tokens.
    Based on: https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
    """
    
    def __init__(
        self,
        dim: int,
        num_queries: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # learnable query tokens
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * (dim ** -0.5))
        
        # cross-attention + ffn layers
        self.layers = nn.ModuleList([
            PerceiverAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
            ) for _ in range(num_layers)
        ])
        
        # final layer norm
        self.final_norm = nn.LayerNorm(dim)


    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (B, num_visual_tokens, dim) - features from visual encoder
        
        Returns:
            torch.Tensor: (B, num_queries, dim) - compressed visual representation
        """
        B = visual_features.shape[0]
        
        # expand learnable queries for batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, dim)
        
        # apply cross-attention layers
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                queries = fsdp_checkpoint(layer, queries, visual_features, use_reentrant=False)
            else:
                queries = layer(queries, visual_features)
        
        # final normalization
        queries = self.final_norm(queries)
        
        return queries.to(torch.bfloat16)
