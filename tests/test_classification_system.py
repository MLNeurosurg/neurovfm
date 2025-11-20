import torch
import pytest

import neurovfm.systems.classification as cls_mod
import neurovfm.models.mil as mil_mod
from neurovfm.models import (
    AggregateThenClassify,
    ClassifyThenAggregate,
)


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


class _DummyBackbone(torch.nn.Module):
    """Tiny vision backbone returning linear features; replaces heavy ViT in tests."""

    def __init__(self, embed_dim: int = 16, in_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = torch.nn.Linear(in_dim, embed_dim)

    def forward(self, x: torch.Tensor, *_, **__) -> torch.Tensor:
        return self.proj(x)


@pytest.fixture(autouse=True)
def _patch_mil_and_backbone(monkeypatch):
    """
    Make classification system CPU-friendly for tests:
      - MIL modules use Linear instead of FusedDense and a stubbed torch_scatter
      - get_vit_backbone returns a tiny dummy backbone
    """
    monkeypatch.setattr(mil_mod, "FusedDense", _LinearDense, raising=True)
    monkeypatch.setattr(mil_mod, "torch_scatter", _ScatterStub, raising=True)

    def _fake_backbone(**_):
        return _DummyBackbone(embed_dim=16, in_dim=8)

    monkeypatch.setattr(cls_mod, "get_vit_backbone", _fake_backbone, raising=True)
    yield


def test_vision_classifier_abmil_configuration():
    """VisionClassifier builds AggregateThenClassify pooler and projection head correctly."""
    vision_backbone_cf = {"which": "vit_base", "params": {}}
    pooler_cf = {
        "which": "abmil",
        "params": {
            "hidden_dim": 8,
            "W_out": 1,
            "use_gating": True,
            "use_norm": True,
        },
    }
    proj_params = {"out_dim": 3, "hidden_dims": [4]}

    model = cls_mod.VisionClassifier(
        vision_backbone_cf=vision_backbone_cf,
        pooler_cf=pooler_cf,
        proj_params=proj_params,
    )

    # Backbone is frozen and in eval mode
    assert all(not p.requires_grad for p in model.bb.parameters())
    assert not model.bb.training

    # Pooler type and dimensions
    assert isinstance(model.pooler, AggregateThenClassify)
    assert model.pooler.num_features == model.bb.embed_dim

    # Projection head uses MIL output dim as input
    assert model.proj.out_dim == proj_params["out_dim"]


def test_vision_classifier_additive_mil_configuration():
    """VisionClassifier builds ClassifyThenAggregate and omits extra projector."""
    vision_backbone_cf = {"which": "vit_base", "params": {}}
    pooler_cf = {
        "which": "addmil",
        "params": {
            "hidden_dim": 8,
            "W_out": 2,
            "mlp_hidden_dims": [4],
        },
    }
    proj_params = {"out_dim": 2, "hidden_dims": [4]}

    model = cls_mod.VisionClassifier(
        vision_backbone_cf=vision_backbone_cf,
        pooler_cf=pooler_cf,
        proj_params=proj_params,
    )

    assert isinstance(model.pooler, ClassifyThenAggregate)
    # For ClassifyThenAggregate, VisionClassifier should not add an extra projection MLP
    assert model.proj is None


def test_vision_classifier_avgpool_behaviour():
    """Avgpool pooler reduces tokens to bag-level means and feeds into MLP."""
    vision_backbone_cf = {"which": "vit_base", "params": {}}
    pooler_cf = {"which": "avgpool", "params": {}}
    proj_params = {"out_dim": 1, "hidden_dims": [8]}

    model = cls_mod.VisionClassifier(
        vision_backbone_cf=vision_backbone_cf,
        pooler_cf=pooler_cf,
        proj_params=proj_params,
    )

    # Pooler is a callable that uses torch_scatter.segment_csr under the hood
    assert callable(model.pooler)
    assert model.proj.out_dim == 1

    # Run a tiny forward: 2 bags with lengths 3 and 2
    lengths = [3, 2]
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int64)
    # Dummy backbone expects feature dim 8
    tokens = torch.randn(sum(lengths), 8)
    feats = model.bb(tokens)
    pooled = model.pooler(feats, cu_seqlens, max_seqlen=max(lengths))

    assert pooled.shape == (2, model.bb.embed_dim)
    logits = model.proj(pooled)
    assert logits.shape == (2, 1)


def test_compute_balanced_accuracy_binary():
    """compute_balanced_accuracy produces expected result for simple stats."""
    system = cls_mod.VisionClassificationSystem(
        model_hyperparams={
            "vision_backbone_cf": {"which": "vit_base", "params": {}},
            "pooler_cf": {"which": "avgpool", "params": {}},
            "proj_params": {"out_dim": 1, "hidden_dims": [8]},
        },
        loss_cf={"which": "bce"},
        opt_cf=None,
        schd_cf=None,
        normalization_stats_list=None,
        training_params={"wts": torch.tensor([1.0, 1.0])},
    )

    # stats = [TP, FP, TN, FN, support]
    stats = torch.tensor([80.0, 20.0, 90.0, 10.0, 0.0])
    bal_acc = system.compute_balanced_accuracy(stats)

    # Sensitivity = 80 / (80 + 10) = 0.888..., Specificity = 90 / (90 + 20) = 0.818...
    expected = ((80.0 / 90.0) + (90.0 / 110.0)) / 2.0
    assert torch.allclose(bal_acc, torch.tensor(expected), atol=1e-6)



