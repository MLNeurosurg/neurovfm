"""
PyTorch Lightning DataModule for Medical Imaging

Handles data loading, preprocessing, and batching for training and validation.
Supports both image-level and study-level batching strategies.
"""

import logging
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from typing import Optional

from neurovfm.data.metadata import DatasetMetadata
from neurovfm.data.cache import CacheManager
from neurovfm.datasets.dataset import ImageDataset, StudyAwareBatchSampler
from neurovfm.datasets.collators import MultiViewCollator, MultiBlockCollator


class ImageDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for medical imaging datasets.
    
    Handles:
    - Dataset setup from raw data or preprocessed cache
    - Dataloader creation with appropriate samplers and collators
    - Deterministic training via seeded workers
    - Study-aware batching for multi-series studies
    - Study-level labels for classification tasks
    
    Args:
        config (OmegaConf): Configuration object with structure:
            data:
                data_dir: str, path to dataset root (contains metadata.json and cache/)
                use_cache: bool, whether to use preprocessed cache
                fallback_to_raw: bool, whether to load from raw if cache miss
                study_labels: str (optional), path to CSV or JSON file with study-level labels
                    CSV format (recommended for multilabel):
                        study_id,label_a,label_b,label_c
                        study_001,1,0,1
                        study_002,0,1,0
                    JSON format (for single label):
                        {"study_001": 0, "study_002": 1, ...}
                dataset:
                    train:
                        params:
                            mode_filter: str (optional), 'ct' or 'mri'
                            ct_window_probs: List[float] (optional), [brain, blood, bone]
                            random_crop: bool, apply random cropping
                            augment: bool, apply spatial augmentation
                            max_crop_size: Tuple[int, int, int], max crop size in tokens
                            ... (other ImageDataset params)
                    val:
                        params: ... (similar to train)
                loader:
                    train:
                        batch_size: int
                        num_workers: int or 'auto'
                        use_study_sampler: bool, use StudyAwareBatchSampler
                        collate_fn:
                            remove_background: bool
                            patch_drop_rate: float
                        ... (other DataLoader params)
                    val: ... (similar structure)
            infra:
                seed: int
    
    Example:
        >>> config = OmegaConf.load('config.yaml')
        >>> dm = ImageDataModule(config)
        >>> dm.setup(stage='fit')
        >>> train_loader = dm.train_dataloader()
    """
    
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.seed = config.infra.seed
        
        self.train_dataset = None
        self.val_dataset = None
        
        # Track current chunk for memmap-based datasets (future feature)
        self._train_chunk_idx = 0
        self._val_chunk_idx = 0

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training or validation.
        
        Args:
            stage (str, optional): 'fit' for train/val, 'test' for testing
        """
        if stage == "fit" or stage is None:
            # Get data directory (should contain metadata.json and cache/)
            data_dir = self.config.data.data_dir
            
            # Verify metadata exists
            metadata_file = Path(data_dir) / 'metadata.json'
            if not metadata_file.exists():
                raise FileNotFoundError(
                    f"Metadata file not found: {metadata_file}\n"
                    "Please create metadata using DatasetMetadata.from_directory()"
                )
            
            # Get common dataset params
            common_params = {
                'data_dir': data_dir,
                'use_cache': self.config.data.get('use_cache', True),
                'fallback_to_raw': self.config.data.get('fallback_to_raw', True),
            }
            
            # Get study labels path if provided
            study_labels_path = self.config.data.get('study_labels', None)
            
            # Training dataset
            train_params = self.config.data.dataset.train.get('params', {})
            if study_labels_path:
                train_params['study_labels'] = study_labels_path
            
            self.train_dataset = ImageDataset(
                **common_params,
                **train_params
            )
            logging.info(f"Train dataset: {len(self.train_dataset)} images")
            
            # Validation dataset
            val_params = self.config.data.dataset.val.get('params', {})
            if study_labels_path:
                val_params['study_labels'] = study_labels_path
            
            self.val_dataset = ImageDataset(
                **common_params,
                **val_params
            )
            logging.info(f"Val dataset: {len(self.val_dataset)} images")

    @staticmethod
    def get_seed_worker_and_generator(seed: int):
        """
        Create worker init function and generator for deterministic training.
        
        Reference: https://pytorch.org/docs/stable/notes/randomness.html
        
        Args:
            seed (int): Random seed
        
        Returns:
            Dict with 'worker_init_fn' and 'generator' keys
        """
        def seed_worker(_):
            np.random.seed(seed)
            random.seed(seed)

        g = torch.Generator()
        g.manual_seed(seed)
        return {"worker_init_fn": seed_worker, "generator": g}

    def train_dataloader(self):
        """
        Create training dataloader.
        
        Returns:
            DataLoader for training
        """
        loader_params = OmegaConf.to_container(self.config.data.loader.train)
        
        # Setup collator
        if 'collate_fn' in loader_params:
            collate_params = loader_params.pop('collate_fn')
            # Use JEPA-style MultiBlockCollator for pretraining system,
            # otherwise default to MultiViewCollator.
            if getattr(getattr(self.config, "system", None), "which", "") == "VisionPretrainingSystem":
                # Default JEPA mask configuration (mirrors _MultiBlockCollator defaults)
                mask_cfg = {
                    "hw_pred_mask_scale": [
                        {"mri": [0.7, 0.7], "ct": [0.75, 0.75]},
                        {"mri": [0.25, 0.25], "ct": [0.2, 0.2]},
                    ],
                    "d_pred_mask_scale": (1.0, 1.0),
                    "enc_mask_scale": {"mri": 0.25, "ct": 0.2},
                    "drop_rate": collate_params.get("patch_drop_rate", 0.0),
                    "aspect_ratio": [(0.3, 3.0)],
                    "npred": [1],
                    "max_depth_scale": 1.0,
                    "remove_background": collate_params.get("remove_background", True),
                    "min_filt_ratio": collate_params.get("min_filt_ratio", 0.0),
                    "switch_enc_pred": collate_params.get("switch_enc_pred", [False]),
                }
                apply_masks_internally = collate_params.get("apply_masks_internally", False)
                collate_fn = MultiBlockCollator(
                    cfgs_mask=[mask_cfg],
                    apply_masks_internally=apply_masks_internally,
                )
            else:
                collate_fn = MultiViewCollator(**collate_params)
            loader_params['collate_fn'] = collate_fn
        
        # Setup study-aware sampler if requested
        use_study_sampler = loader_params.pop('use_study_sampler', False)
        if use_study_sampler:
            batch_size = loader_params.pop('batch_size')
            drop_last = loader_params.pop('drop_last', True)
            loader_params.pop('shuffle', None)  # Incompatible with batch_sampler
            
            sampler = StudyAwareBatchSampler(
                dataset=self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                seed=self.seed + self._train_chunk_idx,
                drop_last=drop_last
            )
            loader_params['batch_sampler'] = sampler
            self._train_chunk_idx += 1
        
        # Setup deterministic workers
        loader_params.update(self.get_seed_worker_and_generator(self.seed + self._train_chunk_idx))
        
        # Handle 'auto' num_workers
        if loader_params.get("num_workers") == "auto":
            loader_params["num_workers"] = min(8, torch.multiprocessing.cpu_count() // 2)
        
        logging.info(f"Train dataloader params: {loader_params}")
        return DataLoader(self.train_dataset, **loader_params)

    def val_dataloader(self):
        """
        Create validation dataloader.
        
        Returns:
            DataLoader for validation
        """
        loader_params = OmegaConf.to_container(self.config.data.loader.val)
        
        # Setup collator
        if 'collate_fn' in loader_params:
            collate_params = loader_params.pop('collate_fn')
            if getattr(getattr(self.config, "system", None), "which", "") == "VisionPretrainingSystem":
                mask_cfg = {
                    "hw_pred_mask_scale": [
                        {"mri": [0.7, 0.7], "ct": [0.75, 0.75]},
                        {"mri": [0.25, 0.25], "ct": [0.2, 0.2]},
                    ],
                    "d_pred_mask_scale": (1.0, 1.0),
                    "enc_mask_scale": {"mri": 0.25, "ct": 0.2},
                    "drop_rate": collate_params.get("patch_drop_rate", 0.0),
                    "aspect_ratio": [(0.3, 3.0)],
                    "npred": [1],
                    "max_depth_scale": 1.0,
                    "remove_background": collate_params.get("remove_background", True),
                    "min_filt_ratio": collate_params.get("min_filt_ratio", 0.0),
                    "switch_enc_pred": collate_params.get("switch_enc_pred", [False]),
                }
                apply_masks_internally = collate_params.get("apply_masks_internally", False)
                collate_fn = MultiBlockCollator(
                    cfgs_mask=[mask_cfg],
                    apply_masks_internally=apply_masks_internally,
                )
            else:
                collate_fn = MultiViewCollator(**collate_params)
            loader_params['collate_fn'] = collate_fn
        
        # Setup study-aware sampler if requested
        use_study_sampler = loader_params.pop('use_study_sampler', False)
        if use_study_sampler:
            batch_size = loader_params.pop('batch_size')
            drop_last = loader_params.pop('drop_last', False)
            loader_params.pop('shuffle', None)
            
            sampler = StudyAwareBatchSampler(
                dataset=self.val_dataset,
                batch_size=batch_size,
                shuffle=False,  # No shuffling for validation
                seed=self.seed + self._val_chunk_idx,
                drop_last=drop_last
            )
            loader_params['batch_sampler'] = sampler
            self._val_chunk_idx += 1
        
        # Setup deterministic workers
        loader_params.update(self.get_seed_worker_and_generator(self.seed + self._val_chunk_idx))
        
        # Handle 'auto' num_workers
        if loader_params.get("num_workers") == "auto":
            loader_params["num_workers"] = min(8, torch.multiprocessing.cpu_count() // 2)
        
        logging.info(f"Val dataloader params: {loader_params}")
        return DataLoader(self.val_dataset, **loader_params)

