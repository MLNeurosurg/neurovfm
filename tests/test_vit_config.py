import torch

import neurovfm.models.vit as vit_mod


class _LinearDense(torch.nn.Linear):
    """CPU-friendly stand-in for flash_attn FusedDense used in ViT modules.

    Subclasses ``nn.Linear`` so it exposes ``weight``/``bias`` attributes,
    matching what the ViT code expects when it rescales layer weights.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, **_):
        super().__init__(in_features, out_features, bias=bias)


def _patch_fused_dense(monkeypatch):
    # Replace FusedDense with a simple Linear layer for config-level tests
    monkeypatch.setattr(vit_mod, "FusedDense", _LinearDense, raising=True)


def test_get_vit_backbone_known_configs(monkeypatch):
    """get_vit_backbone constructs VisionTransformer variants with sane hyperparams."""
    _patch_fused_dense(monkeypatch)

    for name in ["vit_tiny", "vit_small", "vit_base"]:
        model = vit_mod.get_vit_backbone(
            which=name, 
            params={
                "embed_layer_cf": {
                    "which": "voxel", 
                    "params": {
                        "patch_hw_size": 16, 
                        "patch_d_size": 4, 
                        "in_chans": 1, 
                        "embed_dim": 738, 
                        "bias": True, 
                        "fused_bias_fc": True
                    }
                }
            }
        ) 
        assert isinstance(model, vit_mod.VisionTransformer)
        assert model.embed_dim > 0
        assert len(model.blocks) > 0
        assert model.embed_dim % model.num_heads == 0


def test_get_vit_backbone_invalid_name_raises(monkeypatch):
    """Unknown backbone names should raise a clear ValueError."""
    _patch_fused_dense(monkeypatch)
    try:
        vit_mod.get_vit_backbone(which="vit_invalid_size", params={})
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown backbone name")



