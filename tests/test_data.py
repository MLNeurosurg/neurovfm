import os
from pathlib import Path

import numpy as np
import pytest
import torch

from neurovfm.data.cache import CacheManager


def test_cache_init_without_metadata_raises(tmp_path):
	empty_dir = tmp_path / "no_metadata"
	empty_dir.mkdir()
	with pytest.raises(FileNotFoundError):
		_ = CacheManager(empty_dir)


def test_build_and_load_mri(mri_dataset_dir, mock_preprocess_mri):
	cache_mgr = CacheManager(mri_dataset_dir)
	cache_mgr.build_cache(num_workers=1, force=True)

	# Files are written
	processed = mri_dataset_dir / "processed" / "study_001"
	assert (processed / "T1.pt").exists()
	assert (processed / "T1_mask.pt").exists()
	assert (processed / "T1_metadata.json").exists()

	# Loading returns normalized float32 in [0,1] and uint8 mask
	data = cache_mgr.load_image("study_001", "T1")
	assert isinstance(data, dict)
	img = data["data"]
	mask = data["mask"]
	assert isinstance(img, torch.Tensor) and img.dtype == torch.float32
	assert isinstance(mask, torch.Tensor) and mask.dtype == torch.uint8
	assert img.ndim == 3 and mask.shape == img.shape
	assert img.min().item() >= 0.0 and img.max().item() <= 1.0

	stats = cache_mgr.get_cache_stats()
	assert stats["total_images"] == 1
	assert stats["cached_images"] == 1


def test_build_and_load_ct(ct_dataset_dir, mock_preprocess_ct):
	cache_mgr = CacheManager(ct_dataset_dir)
	cache_mgr.build_cache(num_workers=1, force=True)

	processed = ct_dataset_dir / "processed" / "study_ct_001"
	# Three windows plus mask and metadata
	for w in ("brain", "blood", "bone"):
		assert (processed / f"head_{w}.pt").exists()
	assert (processed / "head_mask.pt").exists()
	assert (processed / "head_metadata.json").exists()

	# Load all windows
	data_list = cache_mgr.load_image("study_ct_001", "head")
	assert isinstance(data_list, list) and len(data_list) == 3
	windows = [d["window"] for d in data_list]
	assert set(windows) == {"brain", "blood", "bone"}
	for d in data_list:
		img = d["data"]
		mask = d["mask"]
		assert isinstance(img, torch.Tensor) and img.dtype == torch.float32
		assert img.ndim == 3
		assert mask.dtype == torch.uint8 and mask.shape == img.shape
		assert img.min().item() >= 0.0 and img.max().item() <= 1.0

	# Load a single window
	only_bone = cache_mgr.load_image("study_ct_001", "head", window="bone")
	assert isinstance(only_bone, dict)
	assert only_bone["window"] == "bone"

	stats = cache_mgr.get_cache_stats()
	assert stats["total_images"] == 1
	assert stats["cached_images"] == 1