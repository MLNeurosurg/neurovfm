import numpy as np
import torch
import pytest

from pathlib import Path

from neurovfm.pipelines.preprocessor import StudyPreprocessor
import neurovfm.pipelines.preprocessor as preproc_mod
import neurovfm.data.io as io_mod
import neurovfm.data.preprocess as pp_mod


@pytest.fixture(autouse=True)
def _patch_io_and_preprocess(monkeypatch):
    """Stub out disk IO and heavy preprocessing for StudyPreprocessor tests."""

    class _DummyImage:
        pass

    def _fake_load_image(path: str, preprocess: bool = True):
        return _DummyImage()

    def _fake_prepare_for_inference(img, mode: str):
        # For CT, StudyPreprocessor expects a list of three window arrays
        if mode.lower() == "ct":
            img_arrs = [
                np.ones((4, 8, 8), dtype=np.float32),
                np.full((4, 8, 8), 2.0, dtype=np.float32),
                np.full((4, 8, 8), 3.0, dtype=np.float32),
            ]
        else:
            img_arrs = [np.ones((4, 8, 8), dtype=np.float32)]
        background_mask = np.zeros_like(img_arrs[0], dtype=bool)
        view = None
        return img_arrs, background_mask, view

    def _fake_tokenize_volume(img_arr, background_mask, patch_size, remove_background):
        # Produce a tiny, deterministic token representation for testing
        tokens = np.arange(6, dtype=np.float32).reshape(6, 1)
        coords = np.stack([np.zeros(6), np.arange(6), np.zeros(6)], axis=1).astype(
            np.int64
        )
        filtered = None
        return tokens, coords, filtered

    # Patch low-level IO / preprocessing modules
    monkeypatch.setattr(io_mod, "load_image", _fake_load_image, raising=True)
    monkeypatch.setattr(pp_mod, "prepare_for_inference", _fake_prepare_for_inference, raising=True)
    monkeypatch.setattr(pp_mod, "tokenize_volume", _fake_tokenize_volume, raising=True)

    # Also patch the bindings used inside StudyPreprocessor so tests never hit disk
    monkeypatch.setattr(preproc_mod, "load_image", _fake_load_image, raising=True)
    monkeypatch.setattr(preproc_mod, "prepare_for_inference", _fake_prepare_for_inference, raising=True)
    monkeypatch.setattr(preproc_mod, "tokenize_volume", _fake_tokenize_volume, raising=True)
    yield


def test_study_preprocessor_ct_multiple_windows():
    """CT modality: each volume produces three windowed series with correct metadata."""
    preproc = StudyPreprocessor()

    # Use a list of paths to avoid filesystem globbing
    paths = ["ct_vol1.nii.gz", "ct_vol2.nii.gz"]
    batch = preproc.load_study(paths, modality="ct")

    # Each CT volume -> 3 windows -> 6 series total
    assert batch["img"].ndim == 2
    assert batch["coords"].shape[1] == 3
    assert len(batch["mode"]) == 6
    assert all(m == "ct" for m in batch["mode"])
    assert len(batch["path"]) == 6
    assert all("Window" in p for p in batch["path"])
    assert len(batch["size"]) == 6

    # series_cu_seqlens and study_cu_seqlens are consistent
    series_cu = batch["series_cu_seqlens"]
    assert series_cu.dtype == torch.int32
    assert series_cu[-1].item() == batch["img"].shape[0]

    study_cu = batch["study_cu_seqlens"]
    assert study_cu.tolist() == [0, 6]  # one study with 6 series


def test_study_preprocessor_mri_single_volume():
    """MRI modality: single array path and metadata layout."""
    preproc = StudyPreprocessor()

    path = "mri_vol1.nii.gz"
    batch = preproc.load_study([path], modality="mri")

    assert batch["img"].ndim == 2
    assert batch["coords"].shape[1] == 3
    assert len(batch["mode"]) == 1
    assert batch["mode"][0] == "mri"
    assert len(batch["path"]) == 1
    assert batch["path"][0].endswith(path)
    assert len(batch["size"]) == 1

    # Single series / single study
    assert batch["series_cu_seqlens"].tolist() == [
        0,
        batch["img"].shape[0],
    ]
    assert batch["study_cu_seqlens"].tolist() == [0, 1]


def test_study_preprocessor_raises_on_no_images():
    """Empty image list raises a clear ValueError."""
    preproc = StudyPreprocessor()
    with pytest.raises(ValueError):
        preproc.load_study([], modality="ct")



