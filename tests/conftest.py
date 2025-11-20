import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pytest
import torch

from neurovfm.data.metadata import DatasetMetadata


@pytest.fixture
def create_dataset_dir(tmp_path) -> Path:
	"""
	Create a dataset directory structure with raw/ and processed/ and return its path.
	"""
	data_dir = tmp_path / "dataset"
	(data_dir / "raw").mkdir(parents=True, exist_ok=True)
	(data_dir / "processed").mkdir(parents=True, exist_ok=True)
	return data_dir


def _write_metadata(
	data_dir: Path,
	study_name: str,
	mode: str,
	images: Dict[str, Dict[str, object]]
) -> None:
	metadata = DatasetMetadata()
	metadata.add_study(study_name, mode, images)
	metadata.save(data_dir / "metadata.json")


@pytest.fixture
def mri_dataset_dir(create_dataset_dir) -> Path:
	"""
	Temporary MRI dataset with one study/image and metadata.json.
	"""
	data_dir = create_dataset_dir
	study_dir = data_dir / "raw" / "study_001"
	study_dir.mkdir(parents=True, exist_ok=True)
	# Placeholder file to mimic presence of raw image (not actually read due to mocks)
	placeholder = study_dir / "T1.nii.gz"
	placeholder.write_bytes(b"dummy")
	_write_metadata(
		data_dir,
		"study_001",
		"mri",
		{"T1": {"filename": placeholder.name, "processed": False}},
	)
	return data_dir


@pytest.fixture
def ct_dataset_dir(create_dataset_dir) -> Path:
	"""
	Temporary CT dataset with one study/image and metadata.json.
	"""
	data_dir = create_dataset_dir
	study_dir = data_dir / "raw" / "study_ct_001"
	study_dir.mkdir(parents=True, exist_ok=True)
	placeholder = study_dir / "head.dcm"
	placeholder.write_bytes(b"dummy")
	_write_metadata(
		data_dir,
		"study_ct_001",
		"ct",
		{"head": {"filename": placeholder.name, "processed": False}},
	)
	return data_dir


class _DummyImage:
	def __init__(self, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
		self._spacing = spacing
	def GetSpacing(self) -> Tuple[float, float, float]:
		return self._spacing


@pytest.fixture
def mock_preprocess_mri(monkeypatch):
	"""
	Mock neurovfm.data.cache.load_image and prepare_for_inference for MRI.
	"""
	# Import inside to ensure correct module target for monkeypatch
	import neurovfm.data.cache as cache_mod

	def fake_load_image(path, preprocess=True):
		return _DummyImage(spacing=(0.8, 0.8, 1.5))

	# Produce a single normalized array in [0,1], plus background mask (True=background in original)
	def fake_prepare_for_inference(img, mode: str):
		assert mode == "mri"
		arr = np.linspace(0, 1, num=1024, dtype=np.float32).reshape(16, 8, 8)
		background_mask = np.zeros_like(arr, dtype=bool)  # all foreground -> after invert becomes ones for background
		view = 0
		return [arr], background_mask, view

	monkeypatch.setattr(cache_mod, "load_image", fake_load_image)
	monkeypatch.setattr(cache_mod, "prepare_for_inference", fake_prepare_for_inference)


@pytest.fixture
def mock_preprocess_ct(monkeypatch):
	"""
	Mock neurovfm.data.cache.load_image and prepare_for_inference for CT with 3 windows.
	"""
	import neurovfm.data.cache as cache_mod

	def fake_load_image(path, preprocess=True):
		return _DummyImage(spacing=(0.7, 0.7, 3.0))

	def fake_prepare_for_inference(img, mode: str):
		assert mode == "ct"
		# Three different simple arrays for brain/blood/bone windows
		brain = np.full((10, 6, 6), 0.25, dtype=np.float32)
		blood = np.full((10, 6, 6), 0.5, dtype=np.float32)
		bone = np.full((10, 6, 6), 0.75, dtype=np.float32)
		background_mask = np.zeros_like(brain, dtype=bool)
		view = 1
		return [brain, blood, bone], background_mask, view

	monkeypatch.setattr(cache_mod, "load_image", fake_load_image)
	monkeypatch.setattr(cache_mod, "prepare_for_inference", fake_prepare_for_inference)


