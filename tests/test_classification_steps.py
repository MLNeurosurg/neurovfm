import torch

import neurovfm.systems.classification as cls_mod
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
    """Tiny vision backbone returning linear features, replaces heavy ViT in tests."""

    def __init__(self, embed_dim: int = 16, in_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = torch.nn.Linear(in_dim, embed_dim)

    def forward(self, x: torch.Tensor, coords: torch.Tensor, *_, **__) -> torch.Tensor:
        del coords
        return self.proj(x)


def _patch_deps(monkeypatch):
    monkeypatch.setattr(mil_mod, "FusedDense", _LinearDense, raising=True)
    monkeypatch.setattr(mil_mod, "torch_scatter", _ScatterStub, raising=True)

    def _fake_backbone(**_):
        return _DummyBackbone(embed_dim=16, in_dim=8)

    monkeypatch.setattr(cls_mod, "get_vit_backbone", _fake_backbone, raising=True)


def _build_system_bce(monkeypatch) -> cls_mod.VisionClassificationSystem:
    _patch_deps(monkeypatch)
    model_hyperparams = {
        "vision_backbone_cf": {"which": "vit_base", "params": {}},
        "pooler_cf": {"which": "avgpool", "params": {}},
        "proj_params": {"out_dim": 1, "hidden_dims": [8]},
    }
    return cls_mod.VisionClassificationSystem(
        model_hyperparams=model_hyperparams,
        loss_cf={"which": "bce"},
        opt_cf=None,
        schd_cf=None,
        normalization_stats_list=None,
        training_params={"wts": torch.tensor([1.0, 1.0])},
    )


def _make_batch(num_series: int = 2, tokens_per_series: int = 3):
    """Construct a minimal batch compatible with VisionClassificationSystem."""
    total_tokens = num_series * tokens_per_series
    img = torch.randn(total_tokens, 8)  # backbone in_dim=8
    coords = torch.zeros(total_tokens, 3, dtype=torch.long)

    # Series-level cu_seqlens
    lengths = torch.tensor([tokens_per_series] * num_series, dtype=torch.int32)
    series_cu = torch.zeros(num_series + 1, dtype=torch.int32)
    series_cu[1:] = lengths.cumsum(0)

    # Study-level: assume one series per study
    study_cu = torch.arange(0, num_series + 1, dtype=torch.int32)

    batch = {
        "img": img,
        "coords": coords,
        "label": torch.tensor([0.0, 1.0])[:num_series],
        "size": torch.ones(num_series, 3, dtype=torch.int64),
        "series_masks_indices": torch.tensor([], dtype=torch.long),
        "series_cu_seqlens": series_cu,
        "series_max_len": int(tokens_per_series),
        "study_cu_seqlens": study_cu,
        "study_max_len": int(study_cu[-1].item()),
        "mode": ["ct"] * num_series,
        "path": ["vol_BrainWindow.nii.gz"] * num_series,
    }
    return batch


def test_training_step_bce_binary(monkeypatch):
    """training_step runs end-to-end for a tiny BCE setup and returns scalar loss."""
    system = _build_system_bce(monkeypatch)
    batch = _make_batch(num_series=2, tokens_per_series=3)

    loss = system.training_step(batch, batch_idx=0)
    assert loss.shape == ()


def test_validation_step_ce_multiclass(monkeypatch):
    """validation_step runs for a small CE/multiclass configuration."""
    _patch_deps(monkeypatch)
    num_classes = 3
    wts = torch.ones(num_classes)

    model_hyperparams = {
        "vision_backbone_cf": {"which": "vit_base", "params": {}},
        "pooler_cf": {"which": "avgpool", "params": {}},
        "proj_params": {"out_dim": num_classes, "hidden_dims": [8]},
    }
    system = cls_mod.VisionClassificationSystem(
        model_hyperparams=model_hyperparams,
        loss_cf={"which": "ce"},
        opt_cf=None,
        schd_cf=None,
        normalization_stats_list=None,
        training_params={"wts": wts},
    )

    batch = _make_batch(num_series=2, tokens_per_series=2)
    # Replace labels with integer class indices
    batch["label"] = torch.tensor([0, 2])

    # Should not raise; metrics and loss are handled internally
    system.validation_step(batch, batch_idx=0)


def test_on_train_epoch_end_does_not_crash(monkeypatch):
    """After a training_step, on_train_epoch_end should compute and reset metrics."""
    system = _build_system_bce(monkeypatch)
    batch = _make_batch(num_series=2, tokens_per_series=3)
    _ = system.training_step(batch, batch_idx=0)

    # Should run metric aggregation and logging without raising
    system.on_train_epoch_end()



