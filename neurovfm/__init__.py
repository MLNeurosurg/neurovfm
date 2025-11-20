"""
NeuroVFM: Vision Foundation Model for Medical Imaging

A comprehensive framework for self-supervised pretraining and supervised fine-tuning
of vision transformers on 3D medical imaging data (MRI, CT).
"""

__version__ = "0.1.0"

# Data loading and preprocessing
from .data import (
    DatasetMetadata,
    CacheManager,
    load_image,
)

# Datasets and data modules
from .datasets import (
    ImageDataset,
    StudyAwareBatchSampler,
    MultiViewCollator,
    ImageDataModule,
)

# Models
from .models import (
    VisionTransformer,
    VisionPredictor,
    get_vit_backbone,
    PatchEmbed,
    PositionalEncoding3DWrapper,
    MLP,
)

# Training systems
from .systems import (
    VisionPretrainingSystem,
    VisionClassificationSystem,
)

# Inference pipelines
from .pipelines import (
    load_encoder,
    load_diagnostic_head,
    EncoderPipeline,
    DiagnosticHead,
    StudyPreprocessor,
)

__all__ = [
    # Data
    "DatasetMetadata",
    "CacheManager",
    "load_image",
    # Datasets
    "ImageDataset",
    "StudyAwareBatchSampler",
    "MultiViewCollator",
    "ImageDataModule",
    # Models
    "VisionTransformer",
    "VisionPredictor",
    "get_vit_backbone",
    "PatchEmbed",
    "PositionalEncoding3DWrapper",
    "MLP",
    # Systems
    "VisionPretrainingSystem",
    "VisionClassificationSystem",
    # Pipelines
    "load_encoder",
    "load_diagnostic_head",
    "EncoderPipeline",
    "DiagnosticHead",
    "StudyPreprocessor",
]

