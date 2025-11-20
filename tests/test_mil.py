import torch
import pytest

from neurovfm.models import (
    AggregateThenClassify,
    ClassifyThenAggregate,
)
import neurovfm.models.mil as mil_mod


class _LinearDense(torch.nn.Module):
    """CPU-friendly stand-in for flash_attn FusedDense used in MIL modules."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, **_):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _ScatterStub:
    """Minimal torch_scatter.segment_csr implementation for tests."""

    @staticmethod
    def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
        """
        Args:
            src: Tensor of shape [N, ...]
            indptr: 1D tensor of shape [B+1] with cumulative indices
            reduce: 'sum' or 'max'
        Returns:
            Tensor of shape [B, ...]
        """
        B = indptr.numel() - 1
        out_shape = (B,) + tuple(src.shape[1:])
        out = src.new_zeros(out_shape)
        for b in range(B):
            start = int(indptr[b].item())
            end = int(indptr[b + 1].item())
            seg = src[start:end]
            if seg.numel() == 0:
                continue
            if reduce == "sum":
                out[b] = seg.sum(dim=0)
            elif reduce == "max":
                out[b] = seg.max(dim=0).values
            else:
                raise NotImplementedError(f"reduce={reduce} not supported in stub")
        return out


@pytest.fixture(autouse=True)
def _patch_mil_dependencies(monkeypatch):
    """
    Automatically patch MIL module to avoid GPU-only flash_attn / torch_scatter.

    Replaces:
      - FusedDense -> plain nn.Linear
      - torch_scatter.segment_csr -> simple CPU implementation
    """
    monkeypatch.setattr(mil_mod, "FusedDense", _LinearDense, raising=True)
    monkeypatch.setattr(mil_mod, "torch_scatter", _ScatterStub, raising=True)
    yield


def test_pad_ragged_basic():
    """pad_ragged correctly pads and builds mask for variable-length sequences."""
    data = torch.arange(1, 8).float().unsqueeze(-1)  # [7, 1]
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int64)  # two sequences: 3 and 4 tokens

    padded, mask = mil_mod.pad_ragged(data, cu_seqlens, batch_first=True)

    assert padded.shape == (2, 4, 1)
    assert mask.shape == (2, 4)
    # First sequence has 3 valid tokens
    assert mask[0].tolist() == [True, True, True, False]
    # Second sequence has 4 valid tokens
    assert mask[1].tolist() == [True, True, True, True]


def test_abmil_single_head_attention_sums_to_one():
    """AggregateThenClassify single-head attention weights sum to 1 per bag."""
    dim = 4
    mil = AggregateThenClassify(dim=dim, hidden_dim=dim, W_out=1, use_gating=True, use_norm=True)

    # Two bags with lengths 3 and 2
    lengths = [3, 2]
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int64)
    media = torch.randn(sum(lengths), dim)

    output, attn = mil(media, cu_seqlens=cu_seqlens, return_attn_probs=True)

    assert output.shape == (2, dim)
    assert attn.shape == (sum(lengths), 1)

    # Within each bag, attention weights should form a probability distribution
    for b in range(2):
        s, e = cu_seqlens[b].item(), cu_seqlens[b + 1].item()
        bag_weights = attn[s:e].squeeze(-1)
        assert torch.all(bag_weights >= 0)
        assert torch.allclose(bag_weights.sum(), torch.tensor(1.0), atol=1e-5)


def test_abmil_multi_head_shapes():
    """AggregateThenClassify with multiple heads returns [B, C, dim] and per-token attention [N, C]."""
    dim = 8
    num_heads = 3
    mil = AggregateThenClassify(dim=dim, hidden_dim=dim, W_out=num_heads, use_gating=False, use_norm=False)

    lengths = [2, 4]
    cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int64)
    media = torch.randn(sum(lengths), dim)

    output, attn = mil(media, cu_seqlens=cu_seqlens, return_attn_probs=True)

    assert output.shape == (2, num_heads, dim)
    assert attn.shape == (sum(lengths), num_heads)

    # Each head's attention distribution per bag should sum to 1
    for b in range(2):
        s, e = cu_seqlens[b].item(), cu_seqlens[b + 1].item()
        bag_weights = attn[s:e]  # [len_bag, num_heads]
        assert torch.all(bag_weights >= 0)
        assert torch.allclose(
            bag_weights.sum(dim=0), torch.ones(num_heads), atol=1e-5
        )


def test_additivemil_outputs_match_attention_weighted_patch_logits():
    """ClassifyThenAggregate bag outputs equal attention-weighted sum of patch logits."""
    dim = 6
    num_classes = 2
    mil = ClassifyThenAggregate(
        dim=dim,
        hidden_dim=dim,
        W_out=num_classes,
        mlp_hidden_dims=[4],
        use_gating=True,
        use_norm=True,
        use_output_bias_scale=False,
    )

    lengths = [3, 2]
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int64)
    media = torch.randn(sum(lengths), dim)

    output, attn, patch_logits = mil(
        media, cu_seqlens=cu_seqlens, return_logits=True
    )

    assert output.shape == (2, num_classes)
    assert attn.shape == (sum(lengths), num_classes)
    assert patch_logits.shape == (sum(lengths), num_classes)

    # Manually recompute bag predictions from attention weights and patch logits
    for b in range(2):
        s, e = cu_seqlens[b].item(), cu_seqlens[b + 1].item()
        for c in range(num_classes):
            weights_bc = attn[s:e, c]
            logits_bc = patch_logits[s:e, c]
            manual = (weights_bc * logits_bc).sum()
            assert torch.allclose(manual, output[b, c], atol=1e-5)



