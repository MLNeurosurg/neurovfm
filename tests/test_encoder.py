import torch
import pytest

from neurovfm.models.vit import VisionTransformer
from neurovfm.pipelines.encoder import EncoderPipeline
from neurovfm.systems.utils import NormalizationModule


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="VisionTransformer currently requires GPU + FlashAttention kernels",
)
def test_vision_transformer_forward_shape():
    """Smoketest: full VisionTransformer forward pass produces expected shape."""
    model = VisionTransformer(
        embed_dim=768,
        depth=2,  # keep small for test speed
        num_heads=8,
        prefix_len=0,
        token_dim=1024,
        embed_layer_cf={
            "which": "voxel",
            "params": {
                "patch_hw_size": 16,
                "patch_d_size": 4,
                "in_chans": 1,
                "embed_dim": 738,
            },
        },
        pos_emb_cf={
            "which": "pe3d",
            "params": {
                "in_dim": 738,
                "d": 30,
                "d_size": 128,
                "hw_size": 192,
                "pe_factor": 1,
                "concat": True,
            },
        },
    )

    device = torch.device("cuda")
    model = model.to(device)

    tokens = torch.randn(100, 1024, device=device)  # [N, token_dim]
    coords = torch.zeros(100, 3, device=device, dtype=torch.long)  # [N, 3] for 3D positions
    cu_seqlens = torch.tensor([0, 50, 100], dtype=torch.int32, device=device)

    # Use standard attention for test stability; BF16 autocast to match factory_kwargs
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
        features = model(
            tokens,
            coords,
            cu_seqlens=cu_seqlens,
            max_seqlen=50,
            use_flash_attn=True,
        )
    assert features.shape == (100, 768)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="VisionTransformer currently requires GPU + FlashAttention kernels",
)
def test_vision_transformer_packed_sequences():
    """VisionTransformer handles packed variable-length sequences on GPU."""
    model = VisionTransformer(
        embed_dim=64,
        depth=2,
        num_heads=8,
        prefix_len=0,
        token_dim=16,
        embed_layer_cf={"which": "linear", "params": {}},
        pos_emb_cf=None,
    )

    device = torch.device("cuda")
    model = model.to(device)

    # Two sequences of different lengths (5 and 7 tokens)
    n1, n2 = 5, 7
    tokens = torch.randn(n1 + n2, 16, device=device)
    coords = torch.zeros(n1 + n2, 3, device=device)
    cu_seqlens = torch.tensor([0, n1, n1 + n2], dtype=torch.int32, device=device)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16), torch.no_grad():
        out = model(
            tokens,
            coords,
            cu_seqlens=cu_seqlens,
            max_seqlen=max(n1, n2),
            use_flash_attn=False,
        )

    assert out.shape == (n1 + n2, 64)


def test_normalization_module_routing_and_normalization():
    """NormalizationModule routes by modality/window and normalizes per-series."""
    norm = NormalizationModule()

    # Three series, two tokens each
    img = torch.ones(6, 1024) * 255.0
    modes = ["mri", "ct", "ct"]
    paths = [
        "subj_mri.nii.gz",
        "subj_CT_BrainWindow.nii.gz",
        "subj_CT_BoneWindow.nii.gz",
    ]
    cu_seqlens = torch.tensor([0, 2, 4, 6], dtype=torch.int32)

    out = norm.normalize(img.clone(), modes, paths, cu_seqlens, sizes=None)

    assert out.shape == img.shape
    # Different window types should result in different normalized values
    assert not torch.allclose(out[0:2], out[2:4])
    assert not torch.allclose(out[2:4], out[4:6])


def test_normalization_module_invalid_custom_stats_shape():
    """NormalizationModule rejects malformed custom_stats_list."""
    bad_stats = [[0.1], [1.0]]  # not 2x4
    with pytest.raises(ValueError):
        NormalizationModule(custom_stats_list=bad_stats)


class DummyEncoderModel(torch.nn.Module):
    """Minimal stand-in for VisionTransformer for pipeline tests."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        self.last_kwargs = None

    def forward(
        self,
        tokens,
        coords,
        masks=None,
        cu_seqlens=None,
        max_seqlen=None,
        use_flash_attn=False,
    ):
        self.last_kwargs = dict(
            masks=masks,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            use_flash_attn=use_flash_attn,
        )
        return torch.ones(tokens.shape[0], self.out_dim, device=tokens.device)


def _make_fake_batch(n_tokens: int = 10, n_series: int = 2):
    tokens = torch.randn(n_tokens, 1024)
    coords = torch.zeros(n_tokens, 3, dtype=torch.long)
    per_series = torch.tensor(
        [n_tokens // n_series] * n_series, dtype=torch.int32
    )
    cu_seqlens = torch.zeros(n_series + 1, dtype=torch.int32)
    cu_seqlens[1:] = per_series.cumsum(0)

    batch = {
        "img": tokens,
        "coords": coords,
        "series_cu_seqlens": cu_seqlens,
        "series_max_len": int(per_series.max()),
        "study_cu_seqlens": torch.tensor([0, n_series], dtype=torch.int32),
        "study_max_len": n_series,
        "mode": ["ct"] * n_series,
        "path": ["series_BrainWindow.nii.gz"] * n_series,
        "size": [(4, 16, 16)] * n_series,
        "series_masks_indices": torch.tensor([]),  # background removed
    }
    return batch


def test_encoder_pipeline_embed_basic_cpu():
    """EncoderPipeline runs end-to-end on CPU and preserves token count."""
    model = DummyEncoderModel(out_dim=256)
    norm_stats = [
        NormalizationModule.DEFAULT_MEANS_VALUES,
        NormalizationModule.DEFAULT_STDS_VALUES,
    ]
    encoder = EncoderPipeline(
        model=model,
        normalization_stats=norm_stats,
        device="cpu",
    )

    batch = _make_fake_batch()
    with torch.no_grad():
        embs = encoder.embed(batch, use_amp=False)

    assert embs.shape[0] == batch["img"].shape[0]
    assert embs.shape[1] == 256
    # Model parameters should be frozen
    assert all(not p.requires_grad for p in encoder.model.parameters())


def test_encoder_pipeline_passes_masks_to_model():
    """EncoderPipeline correctly forwards non-empty series_masks_indices as masks."""
    model = DummyEncoderModel(out_dim=64)
    norm_stats = [
        NormalizationModule.DEFAULT_MEANS_VALUES,
        NormalizationModule.DEFAULT_STDS_VALUES,
    ]
    encoder = EncoderPipeline(
        model=model,
        normalization_stats=norm_stats,
        device="cpu",
    )

    batch = _make_fake_batch()
    # Select a subset of tokens to simulate foreground mask
    mask_indices = torch.tensor([1, 3, 5], dtype=torch.long)
    batch["series_masks_indices"] = mask_indices

    with torch.no_grad():
        embs = encoder.embed(batch, use_amp=False)

    assert embs.shape[0] == batch["img"].shape[0]
    assert model.last_kwargs is not None
    assert torch.equal(model.last_kwargs["masks"], mask_indices.to(encoder.device))
