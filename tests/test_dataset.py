import numpy as np
import pytest
import torch

from neurovfm.data.cache import CacheManager
from neurovfm.datasets import ImageDataset, StudyAwareBatchSampler
from neurovfm.data.metadata import DatasetMetadata


class _DummyImage:
	"""
	Simple stand-in for a SimpleITK image providing only GetSpacing().
	"""
	def __init__(self, spacing=(1.0, 1.0, 1.0)):
		self._spacing = spacing
	def GetSpacing(self):
		return self._spacing


def _monkeypatch_preprocess(monkeypatch, mode: str):
	"""
	Patch neurovfm.data.cache.load_image and prepare_for_inference
	to return tokenization-friendly volumes (D divisible by 4, H/W by 16).
	"""
	import neurovfm.data.cache as cache_mod

	def fake_load_image(path, preprocess=True):
		return _DummyImage(spacing=(1.0, 1.0, 4.0))

	def fake_prepare_for_inference(img, mode: str, **kwargs):
		assert mode == mode
		# Shape: (D=16, H=32, W=32) → tokens: (4, 2, 2) = 16 tokens
		if mode == "mri":
			arr = np.linspace(0, 1, num=16 * 32 * 32, dtype=np.float32).reshape(16, 32, 32)
			background_mask = np.zeros_like(arr, dtype=bool)
			view = 0
			return [arr], background_mask, view
		elif mode == "ct":
			brain = np.full((16, 32, 32), 0.25, dtype=np.float32)
			blood = np.full((16, 32, 32), 0.50, dtype=np.float32)
			bone = np.full((16, 32, 32), 0.75, dtype=np.float32)
			background_mask = np.zeros_like(brain, dtype=bool)
			view = 1
			return [brain, blood, bone], background_mask, view
		else:
			raise AssertionError("Unsupported mode")

	monkeypatch.setattr(cache_mod, "load_image", fake_load_image)
	monkeypatch.setattr(cache_mod, "prepare_for_inference", fake_prepare_for_inference)


def test_image_dataset_mri_tokenized_shapes(mri_dataset_dir, monkeypatch):
	# Arrange: build cache with tokenizable shapes
	_monkeypatch_preprocess(monkeypatch, mode="mri")
	cache_mgr = CacheManager(mri_dataset_dir)
	cache_mgr.build_cache(num_workers=1, force=True)

	# Act
	ds = ImageDataset(mri_dataset_dir, use_cache=True, random_crop=False, augment=False, tokenize=True)
	sample = ds[0]

	# Assert
	assert isinstance(sample, dict)
	img = sample["img"]
	coords = sample["coords"]
	filt = sample["filtered"]
	size = sample["size"]
	assert img.ndim == 2 and img.shape[1] == 1024  # [N, 1024]
	assert coords.ndim == 2 and coords.shape[1] == 3
	assert filt.ndim == 1 and filt.shape[0] == img.shape[0]
	assert tuple(size.tolist()) == (4, 2, 2)  # D//4=16//4, H//16=32//16, W//16=32//16


def test_image_dataset_ct_windows_and_shapes(ct_dataset_dir, monkeypatch):
	# Arrange: build cache with tokenizable shapes
	_monkeypatch_preprocess(monkeypatch, mode="ct")
	cache_mgr = CacheManager(ct_dataset_dir)
	cache_mgr.build_cache(num_workers=1, force=True)

	# Act
	ds = ImageDataset(ct_dataset_dir, use_cache=True, random_crop=False, augment=False, tokenize=True)

	# Draw multiple samples to see different windows
	windows_seen = set()
	for _ in range(6):
		sample = ds[0]
		windows_seen.add(sample["window"])
		img = sample["img"]
		size = sample["size"]
		assert img.ndim == 2 and img.shape[1] == 1024
		assert tuple(size.tolist()) == (4, 2, 2)

	# Assert we only see valid CT windows and at least one appeared
	assert windows_seen.issubset({"brain", "blood", "bone"})
	assert len(windows_seen) >= 1


def test_study_aware_batch_sampler_groups_by_study(tmp_path):
	# Arrange: construct metadata with uneven images per study
	data_dir = tmp_path / "dataset"
	(data_dir / "raw").mkdir(parents=True, exist_ok=True)
	(data_dir / "processed").mkdir(parents=True, exist_ok=True)

	md = DatasetMetadata()
	md.add_study("study_A", "mri", {"img1": {"filename": "a1.nii.gz"}, "img2": {"filename": "a2.nii.gz"}})
	md.add_study("study_B", "ct", {"img1": {"filename": "b1.dcm"}})
	md.save(data_dir / "metadata.json")

	# Use cache=False and no raw fallback to avoid I/O; sampler doesn't fetch data
	ds = ImageDataset(
		data_dir,
		use_cache=False,
		fallback_to_raw=False,
		random_crop=False,
		augment=False,
		tokenize=True,
	)

	# Act
	sampler = StudyAwareBatchSampler(dataset=ds, batch_size=2, shuffle=False, seed=42)
	batches = list(iter(sampler))

	# Assert: each batch contains indices from a single study only
	def idx_to_study(i):
		study_name, _, _ = ds.image_index[i]
		return study_name

	for batch in batches:
		studies_in_batch = {idx_to_study(i) for i in batch}
		assert len(studies_in_batch) == 1
		assert len(batch) <= 2


