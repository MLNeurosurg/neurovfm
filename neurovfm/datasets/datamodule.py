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
from typing import Optional, Dict, Any, List, Union
import json

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


from neurovfm.data.text import process_text

class VisionInstructionDataModule(pl.LightningDataModule):
    """
    Decorator datamodule for visual instruction tuning.

    Wraps an existing ImageDataModule and augments its collate_fn with text processing based on study conversations.    
    Text tensors are left-padded to the max length in the batch (pad_token_id for input_ids, 0 for attention_mask, -100 for labels).
    """

    def __init__(
        self,
        base_dm: ImageDataModule,
        tokenizer,
        system_prompt: str,
        max_seq_len: int,
        placeholder_token_id: int,
        pad_token_id: int, # tokenizer.pad_token_id
    ):
        super().__init__()
        self.base_dm = base_dm
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.max_seq_len = max_seq_len
        self.placeholder_token_id = placeholder_token_id
        self.pad_token_id = pad_token_id

    def setup(self, stage: Optional[str] = None):
        self.base_dm.setup(stage)

    def _build_text_collate_fn(self, base_collate):

        def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
            # collect conversations from raw batch before vision collation
            conv_map: Dict[str, List[Dict[str, str]]] = {}
            for sample in batch:
                study_id = sample.get("study") or sample.get("study_id")
                if study_id is None:
                    raise ValueError("Sample missing study identifier needed for text collation.")
                conv = sample.get("conversation", [])
                if len(conv) == 0:
                    raise KeyError(f"No conversation data for study_id: {study_id}. Please check the study_conversations file.")
                conv_map[study_id] = conv

            # base collate function for vision data
            vision_batch = base_collate(batch)

            study_ids = vision_batch.get("study")
            if study_ids is None:
                raise ValueError("vision_batch missing 'study' for text collation.")
            paths = vision_batch.get("path")
            if paths is None:
                raise ValueError("vision_batch missing 'path' for text collation.")

            input_ids_list, attn_mask_list, labels_list = [], [], []

            # tokenize text for each study
            for i, study_id in enumerate(study_ids):
                if study_id not in conv_map:
                    raise KeyError(f"No conversation data for study_id: {study_id}. Please check the study_conversations file.")

                n_images = len(paths[i])
                conversation = conv_map[study_id]

                processed = process_text(
                    conversation=conversation,
                    tokenizer=self.tokenizer,
                    max_seq_len=self.max_seq_len,
                    system_prompt=self.system_prompt,
                    image_placeholder_token_id=self.placeholder_token_id,
                    n_images=n_images,
                )

                input_ids_list.append(processed["input_ids"])
                attn_mask_list.append(processed["attention_mask"])
                labels_list.append(processed["labels"])

            # left pad text tensors to max length in batch
            # input_ids: pad_token_id
            # attention_mask: 0
            # labels: -100
            max_len = max(len(ids) for ids in input_ids_list)

            def _left_pad(seq, pad_value):
                pad_amount = max_len - len(seq)
                if pad_amount <= 0:
                    return torch.tensor(seq, dtype=torch.long)
                return torch.cat(
                    [
                        torch.full((pad_amount,), pad_value, dtype=torch.long),
                        torch.tensor(seq, dtype=torch.long),
                    ],
                    dim=0,
                )

            vision_batch["input_ids"] = torch.stack(
                [_left_pad(seq, self.pad_token_id) for seq in input_ids_list], dim=0
            )
            vision_batch["attention_mask"] = torch.stack(
                [_left_pad(seq, 0) for seq in attn_mask_list], dim=0
            )
            vision_batch["labels"] = torch.stack(
                [_left_pad(seq, -100) for seq in labels_list], dim=0
            )

            return vision_batch

        return _collate

    def _wrap_loader(self, dl: DataLoader) -> DataLoader:
        dl.collate_fn = self._build_text_collate_fn(dl.collate_fn)
        return dl

    def train_dataloader(self):
        return self._wrap_loader(self.base_dm.train_dataloader())

    def val_dataloader(self):
        return self._wrap_loader(self.base_dm.val_dataloader())

