import torch

from neurovfm.pipelines.encoder import EncoderPipeline
from neurovfm.systems.utils import NormalizationModule


class DummyEncoderModel(torch.nn.Module):
    """Minimal stand-in for VisionTransformer in encoder pipeline tests."""

    def __init__(self, out_dim: int = 64):
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
    series_cu = torch.zeros(n_series + 1, dtype=torch.int32)
    series_cu[1:] = per_series.cumsum(0)

    batch = {
        "img": tokens,
        "coords": coords,
        "series_cu_seqlens": series_cu,
        "series_max_len": int(per_series.max()),
        "study_cu_seqlens": torch.tensor([0, n_series], dtype=torch.int32),
        "study_max_len": n_series,
        "mode": ["ct"] * n_series,
        "path": ["series_BrainWindow.nii.gz"] * n_series,
        "size": [(4, 16, 16)] * n_series,
        "series_masks_indices": torch.tensor([]),  # background removed
    }
    return batch


def _build_encoder(out_dim: int = 64, device: str = "cpu") -> EncoderPipeline:
    model = DummyEncoderModel(out_dim=out_dim)
    norm_stats = [
        NormalizationModule.DEFAULT_MEANS_VALUES,
        NormalizationModule.DEFAULT_STDS_VALUES,
    ]
    return EncoderPipeline(
        model=model,
        normalization_stats=norm_stats,
        device=device,
    )


def test_encoder_pipeline_embed_no_masks_cpu():
    """EncoderPipeline runs end-to-end on CPU with empty masks."""
    encoder = _build_encoder(out_dim=32, device="cpu")
    batch = _make_fake_batch()

    with torch.no_grad():
        embs = encoder.embed(batch, use_amp=False)

    assert embs.shape[0] == batch["img"].shape[0]
    assert embs.shape[1] == 32
    # Underlying model sees masks=None
    assert encoder.model.last_kwargs["masks"] is None


def test_encoder_pipeline_embed_with_masks_forwards_indices():
    """Non-empty series_masks_indices are forwarded as masks to the model."""
    encoder = _build_encoder(out_dim=16, device="cpu")
    batch = _make_fake_batch()
    mask_indices = torch.tensor([0, 2, 4], dtype=torch.long)
    batch["series_masks_indices"] = mask_indices

    with torch.no_grad():
        embs = encoder.embed(batch, use_amp=False)

    assert embs.shape[0] == batch["img"].shape[0]
    assert encoder.model.last_kwargs is not None
    assert torch.equal(
        encoder.model.last_kwargs["masks"],
        mask_indices.to(torch.device(encoder.device)),
    )


def test_encoder_pipeline_amp_true_cpu():
    """use_amp=True path executes without error on CPU."""
    encoder = _build_encoder(out_dim=8, device="cpu")
    batch = _make_fake_batch()

    with torch.no_grad():
        embs = encoder.embed(batch, use_amp=True)

    assert embs.shape == (batch["img"].shape[0], 8)


def test_encoder_pipeline_call_alias():
    """__call__ delegates to embed."""
    encoder = _build_encoder(out_dim=10, device="cpu")
    batch = _make_fake_batch()

    with torch.no_grad():
        out_call = encoder(batch)
        out_embed = encoder.embed(batch)

    assert torch.allclose(out_call, out_embed)



