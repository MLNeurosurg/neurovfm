import torch

from neurovfm.models.pos_embed import PositionalEncoding3DWrapper
from neurovfm.models.projector import MLP, CustomSequential, CSyncBatchNorm


def test_positional_encoding_concat_and_additive():
    """PositionalEncoding3DWrapper returns correct shapes for concat=True/False."""
    B, N, in_dim = 2, 5, 6
    d = 6  # divisible by 3
    d_size, hw_size = 4, 4

    x = torch.randn(B, N, in_dim)
    coords = torch.zeros(B, N, 3, dtype=torch.long)

    # Concat mode
    pe_concat = PositionalEncoding3DWrapper(
        in_dim=in_dim, d=d, d_size=d_size, hw_size=hw_size, concat=True
    )
    out_concat = pe_concat(x, coords)
    assert out_concat.shape == (B, N, in_dim + d)

    # Additive mode
    pe_add = PositionalEncoding3DWrapper(
        in_dim=in_dim, d=d, d_size=d_size, hw_size=hw_size, concat=False
    )
    out_add = pe_add(x, coords)
    assert out_add.shape == (B, N, in_dim)


def test_mlp_forward_shapes():
    """MLP preserves batch dimension and maps in_dim->out_dim."""
    mlp = MLP(in_dim=10, out_dim=3, hidden_dims=[8, 6], norm="ln", act="gelu")
    x = torch.randn(4, 10)

    y = mlp(x)
    assert y.shape == (4, 3)


def test_custom_sequential_batchnorm_permutation():
    """CustomSequential correctly permutes dims around batchnorm for >2D inputs."""
    seq = CustomSequential(
        torch.nn.BatchNorm1d(4),
        torch.nn.ReLU(),
    )

    x = torch.randn(2, 3, 4)  # [B, T, C]
    y = seq(x)
    assert y.shape == x.shape


def test_csync_batchnorm_single_device():
    """CSyncBatchNorm runs on a single device and preserves shape."""
    bn = CSyncBatchNorm(4, with_var=False)
    x = torch.randn(2, 4)

    y = bn(x)
    assert y.shape == x.shape



