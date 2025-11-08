"""
Medical Image Loading and Preprocessing

load_image: Load any medical image (NIfTI/DICOM)
prepare_for_inference: Convert to model-ready arrays
DatasetMetadata: Manage dataset metadata
CacheManager: Build preprocessing cache
"""

from .io import load_image
from .preprocess import prepare_for_inference
from .metadata import DatasetMetadata
from .cache import CacheManager

__all__ = ['load_image', 'prepare_for_inference', 'DatasetMetadata', 'CacheManager']