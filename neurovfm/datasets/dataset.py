"""
PyTorch Dataset and Sampler for Medical Images

ImageDataset: Loads individual images with optional augmentation
StudyAwareBatchSampler: Groups images from same study into batches
"""

import json
import torch
import numpy as np
import pandas as pd
import math
import logging
from torch.utils.data import Dataset, Sampler
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Iterator
from einops import rearrange

from neurovfm.data.metadata import DatasetMetadata
from neurovfm.data.cache import CacheManager
from neurovfm.data.io import load_image
from neurovfm.data.preprocess import prepare_for_inference


class ImageDataset(Dataset):
    """
    PyTorch Dataset for medical images with augmentation and tokenization.
    
    Features:
    - CT window sampling (brain/blood/bone)
    - Random cropping with foreground detection
    - Spatial augmentation (flips, permutations)
    - Tokenization to NxD format (4x16x16 patches)
    - Label handling
    - Segmentation support
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        use_cache: bool = True,
        fallback_to_raw: bool = True,
        mode_filter: Optional[str] = None,
        transform: Optional[callable] = None,
        ct_window_probs: Optional[List[float]] = None,
        random_crop: bool = False,
        augment: bool = False,
        max_crop_size: Tuple[int, int, int] = (20, 20, 20),
        tokenize: bool = True,
        use_original_labels: bool = True,
        class_to_idx: Optional[Dict] = None,
        study_labels: Optional[Dict[str, Union[int, float, List]]] = None,
    ):
        """
        Initialize ImageDataset.
        
        Args:
            data_dir: Root dataset directory
            use_cache: Load from processed cache
            fallback_to_raw: Load from raw if cache miss
            mode_filter: Filter to 'ct' or 'mri' images
            transform: Optional transform applied after tokenization
            ct_window_probs: Sampling probabilities [brain, blood, bone]
            random_crop: Apply random cropping with foreground detection
            augment: Apply spatial augmentation (flips, permutations)
            max_crop_size: Maximum crop size (D, H, W) in tokens
            tokenize: Return tokenized format (N, 1024) vs raw (C, D, H, W)
            use_original_labels: Use original labels vs mapped indices
            class_to_idx: Mapping from class names to indices
            study_labels: Path to CSV file, DataFrame, or dict with study-level labels.
                         All images from same study will share the same label.
                         
                         CSV/DataFrame format (recommended for multilabel):
                         - First column: study_id
                         - Remaining columns: binary labels (0 or 1)
                         Example CSV:
                             study_id,label_a,label_b,label_c
                             study_001,1,0,1
                             study_002,0,1,0
                         
                         JSON/dict format (for single label):
                         {"study_001": 0, "study_002": 1, ...}
        """
        self.data_dir = Path(data_dir)
        self.use_cache = use_cache
        self.fallback_to_raw = fallback_to_raw
        self.mode_filter = mode_filter.lower() if mode_filter else None
        self.transform = transform
        
        # CT window sampling
        if ct_window_probs is None:
            self.ct_window_probs = [0.7, 0.15, 0.15]
        else:
            assert len(ct_window_probs) == 3, "ct_window_probs must have 3 values"
            assert abs(sum(ct_window_probs) - 1.0) < 1e-6, "ct_window_probs must sum to 1"
            self.ct_window_probs = ct_window_probs
        
        self.ct_windows = ['brain', 'blood', 'bone']
        
        # Augmentation settings
        self.random_crop = random_crop
        self.augment = augment
        self.max_crop_size = max_crop_size
        self.tokenize = tokenize
        
        # Label settings
        self.use_original_labels = use_original_labels
        self.class_to_idx_ = class_to_idx or {}
        
        # Load study-level labels from file, DataFrame, or dict
        self.study_labels = {}
        self.label_columns = None  # Track label column names for multilabel
        
        if study_labels is None:
            pass  # Empty labels (for pretraining)
        elif isinstance(study_labels, pd.DataFrame):
            # DataFrame with study_id column and label columns
            self._load_labels_from_dataframe(study_labels)
        elif isinstance(study_labels, (str, Path)):
            label_path = Path(study_labels)
            if not label_path.exists():
                logging.warning(f"Study labels file not found: {label_path}, using empty labels")
            elif label_path.suffix == '.csv':
                # Load CSV as DataFrame
                df = pd.read_csv(label_path)
                self._load_labels_from_dataframe(df)
                logging.info(f"Loaded study labels from CSV {label_path}: {len(self.study_labels)} studies")
            elif label_path.suffix == '.json':
                # Load JSON as dict (single label per study)
                with open(label_path, 'r') as f:
                    self.study_labels = json.load(f)
                logging.info(f"Loaded study labels from JSON {label_path}: {len(self.study_labels)} studies")
            else:
                raise ValueError(f"Unsupported label file format: {label_path.suffix}. Use .csv or .json")
        elif isinstance(study_labels, dict):
            self.study_labels = study_labels
        else:
            raise ValueError(f"study_labels must be a path, DataFrame, or dict, got {type(study_labels)}")
        
        # Load metadata
        metadata_file = self.data_dir / 'metadata.json'
        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}\n"
                "Please create metadata using DatasetMetadata.from_directory()"
            )
        
        self.metadata = DatasetMetadata.from_file(metadata_file)
        
        # Initialize cache manager
        if self.use_cache:
            self.cache_mgr = CacheManager(self.data_dir)
        else:
            self.cache_mgr = None
        
        # Build index
        self.image_index = self._build_index()
    
    def _load_labels_from_dataframe(self, df: pd.DataFrame):
        """
        Load labels from a DataFrame.
        
        Expected format:
        - First column: study_id (or 'study_id', 'study', 'study_name')
        - Remaining columns: binary labels (0 or 1 for multilabel)
        
        Args:
            df (pd.DataFrame): DataFrame with study IDs and labels
        """
        # Find study ID column
        possible_id_cols = ['study_id', 'study', 'study_name']
        id_col = None
        
        if df.columns[0] in possible_id_cols or df.columns[0].lower() in possible_id_cols:
            id_col = df.columns[0]
        else:
            # Assume first column is study ID
            id_col = df.columns[0]
            logging.info(f"Using '{id_col}' as study ID column")
        
        # Get label columns (all columns except study ID)
        label_cols = [col for col in df.columns if col != id_col]
        
        if len(label_cols) == 0:
            raise ValueError(f"No label columns found in DataFrame. Expected at least one label column after '{id_col}'")
        
        self.label_columns = label_cols
        
        # Build study_labels dict
        for _, row in df.iterrows():
            study_id = str(row[id_col])
            
            if len(label_cols) == 1:
                # Single label (can be binary, multiclass, or regression)
                self.study_labels[study_id] = row[label_cols[0]]
            else:
                # Multilabel: return as list of binary values
                labels = [int(row[col]) for col in label_cols]
                self.study_labels[study_id] = labels
        
        logging.info(f"Loaded labels for {len(self.study_labels)} studies with {len(label_cols)} label(s): {label_cols}")
    
    def _build_index(self) -> List[Tuple[str, str, str]]:
        """Build index of all images: (study_name, image_name, mode)"""
        index = []
        
        for study_name, study_info in self.metadata.get_all_studies().items():
            mode = study_info['mode']
            
            if self.mode_filter and mode != self.mode_filter:
                continue
            
            for image_name in study_info['images'].keys():
                index.append((study_name, image_name, mode))
        
        return index
    
    def __len__(self) -> int:
        return len(self.image_index)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single image with optional augmentation and tokenization.
        
        Returns:
            dict with keys:
                - img: Tensor [N, D] where D=1024 for single channel
                - coords: Tensor [N, 3] with (d, h, w) coordinates
                - filtered: Tensor [N] binary mask (1=background, 0=foreground)
                - size: Tensor [3] with (depth, height, width) in tokens
                - path: str, full path to image
                - label: int or str
                - study: str
                - mode: str ('ct' or 'mri')
                - window: str or None
        """
        study_name, image_name, mode = self.image_index[idx]
        
        # Sample CT window
        sampled_window = None
        if mode == 'ct':
            sampled_window = np.random.choice(self.ct_windows, p=self.ct_window_probs)
        
        # Load data
        if self.use_cache and self.cache_mgr:
            data = self._load_from_cache(study_name, image_name, mode, sampled_window)
        else:
            data = None
        
        if data is None and self.fallback_to_raw:
            data = self._load_from_raw(study_name, image_name, mode, sampled_window)
        
        if data is None:
            return self._return_bad_sample()
        
        # Extract data
        img = data['data']  # [D, H, W] float32 [0, 1]
        mask = data['mask']  # [D, H, W] uint8, 1=bg, 0=fg
        view = data['view']
        
        # Normalize types: downstream uses torch.from_numpy(...), so ensure numpy arrays here
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
                  
        # Get study-level label (all images from same study share this label)
        if study_name in self.study_labels:
            label = self.study_labels[study_name]
        else:
            # Default to numeric label 0 if no label provided (for pretraining)
            label = 0
        
        # Determine token dimensions
        # For images at 1x1x4mm with 16x16x4 patches: tokens = D//4, H//16, W//16
        D, H, W = img.shape
        inst_depth = D // 4
        inst_height = H // 16
        inst_width = W // 16
        
        # Apply random crop if enabled
        if self.random_crop:
            inst_depth, inst_height, inst_width, toselect, filt = self._apply_random_crop(
                img, mask, inst_depth, inst_height, inst_width
            )
        else:
            # Use all tokens
            n_tokens = inst_depth * inst_height * inst_width
            toselect = np.arange(n_tokens)
            
            # Tokenize mask
            mask_tokens = rearrange(
                torch.from_numpy(mask),
                "(d p1) (h p2) (w p3) -> (d h w) (p1 p2 p3)",
                d=inst_depth, h=inst_height, w=inst_width,
                p1=4, p2=16, p3=16
            )
            filt = mask_tokens.sum(dim=1) > 0  # Any background pixel -> background token
        
        # Tokenize image
        img_tokens = rearrange(
            torch.from_numpy(img).unsqueeze(0),  # Add channel dim
            "c (d p1) (h p2) (w p3) -> (d h w) (c p1 p2 p3)",
            d=inst_depth, h=inst_height, w=inst_width,
            p1=4, p2=16, p3=16
        ).squeeze(1)  # Remove channel dim for single channel: [N, 1024]
        
        # Select cropped region
        img_tokens = img_tokens[toselect]
        filt = filt[toselect]
        
        # Apply spatial augmentation
        if self.augment:
            img_tokens, filt = self._apply_spatial_aug(
                img_tokens, filt, inst_depth, inst_height, inst_width
            )
        
        # Generate coordinates
        coords = self._generate_coordinates(inst_depth, inst_height, inst_width)
        
        # Build output
        output = {
            'img': img_tokens.float(),  # [N, 1024]
            'coords': coords,  # [N, 3]
            'filtered': filt.to(torch.uint8),  # [N]
            'size': torch.tensor([inst_depth, inst_height, inst_width], dtype=torch.int32),
            'path': f"{study_name}/{image_name}",
            'label': self._process_label(label),
            'study': study_name,
            'mode': mode,
            'window': sampled_window
        }
        
        return output
    
    def _apply_random_crop(
        self, img: np.ndarray, mask: np.ndarray,
        inst_depth: int, inst_height: int, inst_width: int
    ) -> Tuple[int, int, int, np.ndarray, torch.Tensor]:
        """Apply random cropping with foreground detection."""
        MAX_N_ATTEMPTS = 5
        
        # Tokenize mask for crop selection
        mask_tokens = rearrange(
            torch.from_numpy(mask),
            "(d p1) (h p2) (w p3) -> d h w (p1 p2 p3)",
            d=inst_depth, h=inst_height, w=inst_width,
            p1=4, p2=16, p3=16
        )
        filt_init = (mask_tokens.sum(dim=-1) > 0).to(torch.uint8)  # [D, H, W]
        
        # Find suitable crop region
        front, top, left, crop_d, crop_h, crop_w = self._find_suitable_volume(
            (inst_depth, inst_height, inst_width),
            filt_init,
            self.max_crop_size[0],
            self.max_crop_size[1],
            self.max_crop_size[2],
            MAX_N_ATTEMPTS
        )
        
        # Create selection mask
        toselect = np.zeros((inst_depth, inst_height, inst_width), dtype=bool)
        toselect[front:front+crop_d, top:top+crop_h, left:left+crop_w] = True
        toselect = np.where(toselect.flatten())[0]
        
        # Get filtered tokens for crop
        filt = filt_init[front:front+crop_d, top:top+crop_h, left:left+crop_w].flatten()
        
        return crop_d, crop_h, crop_w, toselect, filt
    
    def _find_suitable_volume(
        self, size_init: Tuple[int, int, int], filt_init: torch.Tensor,
        max_depth: int, max_height: int, max_width: int, max_attempts: int
    ) -> Tuple[int, int, int, int, int, int]:
        """Find a random crop region with sufficient foreground content."""
        init_depth, init_height, init_width = size_init
        
        for attempt in range(max_attempts):
            # Sample crop size
            inst_depth = np.random.randint(1, min(max_depth, init_depth) + 1)
            inst_height = np.random.randint(1, min(max_height, init_height) + 1)
            inst_width = np.random.randint(1, min(max_width, init_width) + 1)
            
            # Sample position
            front = np.random.randint(0, init_depth - inst_depth + 1)
            top = np.random.randint(0, init_height - inst_height + 1)
            left = np.random.randint(0, init_width - inst_width + 1)
            
            # Check foreground content
            crop_filt = filt_init[front:front+inst_depth, top:top+inst_height, left:left+inst_width]
            fg_ratio = 1.0 - (crop_filt.sum().item() / crop_filt.numel())
            
            if fg_ratio > 0.1:  # At least 10% foreground
                return front, top, left, inst_depth, inst_height, inst_width
        
        # Fallback: use full volume
        return 0, 0, 0, init_depth, init_height, init_width
    
    def _apply_spatial_aug(
        self, img_tokens: torch.Tensor, filt: torch.Tensor,
        inst_depth: int, inst_height: int, inst_width: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply spatial augmentation (flips and H/W permutation)."""
        # Reshape to 3D
        img_3d = rearrange(
            img_tokens,
            "(d h w) (p1 p2 p3) -> (d p1) (h p2) (w p3)",
            d=inst_depth, h=inst_height, w=inst_width,
            p1=4, p2=16, p3=16
        )
        filt_3d = rearrange(
            filt, "(d h w) -> d h w",
            d=inst_depth, h=inst_height, w=inst_width
        )
        
        # Random H/W permutation
        permute_hw = torch.rand(1).item() < 0.5
        if permute_hw:
            img_3d = img_3d.permute(0, 2, 1)
            filt_3d = filt_3d.permute(0, 2, 1)
            inst_height, inst_width = inst_width, inst_height
        
        # Random flips
        flip_axes = [a for a in range(3) if torch.rand(1).item() < 0.5]
        if flip_axes:
            img_3d = torch.flip(img_3d, dims=flip_axes)
            filt_3d = torch.flip(filt_3d, dims=flip_axes)
        
        # Reshape back to tokens
        img_tokens = rearrange(
            img_3d,
            "(d p1) (h p2) (w p3) -> (d h w) (p1 p2 p3)",
            d=inst_depth, h=inst_height, w=inst_width,
            p1=4, p2=16, p3=16
        )
        filt = rearrange(
            filt_3d, "d h w -> (d h w)",
            d=inst_depth, h=inst_height, w=inst_width
        )
        
        return img_tokens, filt
    
    def _generate_coordinates(
        self, depth: int, height: int, width: int
    ) -> torch.Tensor:
        """Generate 3D coordinates for tokens."""
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(depth),
                torch.arange(height),
                torch.arange(width),
                indexing='ij'
            ),
            dim=-1
        ).reshape(-1, 3).to(torch.int32)
        return coords
    
    def _process_label(self, label):
        """Process label based on settings."""
        if self.use_original_labels:
            return label
        
        if self.class_to_idx_ and isinstance(label, str):
            if label in self.class_to_idx_:
                return self.class_to_idx_[label]
        
        return label
    
    def _load_from_cache(
        self, study_name: str, image_name: str,
        mode: str, window: Optional[str] = None
    ) -> Optional[Dict]:
        """Load image from cache."""
        try:
            if mode == 'ct':
                return self.cache_mgr.load_image(study_name, image_name, window=window)
            else:
                return self.cache_mgr.load_image(study_name, image_name)
        except Exception as e:
            logging.warning(f"Cache load error for {study_name}/{image_name}: {e}")
            return None
    
    def _load_from_raw(
        self, study_name: str, image_name: str,
        mode: str, window: Optional[str] = None
    ) -> Optional[Dict]:
        """Load and preprocess image from raw data."""
        try:
            study_info = self.metadata.get_study(study_name)
            if study_info is None:
                return None
            
            image_info = study_info['images'].get(image_name)
            if image_info is None:
                return None
            
            raw_study_dir = self.data_dir / 'raw' / study_name
            raw_path = raw_study_dir / image_info['filename']
            
            img_sitk = load_image(raw_path, preprocess=True)
            if img_sitk is None:
                return None
            
            result = prepare_for_inference(img_sitk, mode=mode)
            if result is None:
                return None
            
            img_arrs, background_mask, view = result
            mask_binary = (~background_mask).astype(np.uint8)
            
            if mode == 'ct':
                window_idx = self.ct_windows.index(window)
                return {
                    'data': img_arrs[window_idx],
                    'mask': mask_binary,
                        'view': view,
                        'mode': mode,
                    'window': window
                    }
            else:
                return {
                    'data': img_arrs[0],
                    'mask': mask_binary,
                    'view': view,
                    'mode': mode,
                    'window': None
                }
        
        except Exception as e:
            logging.warning(f"Raw load error for {study_name}/{image_name}: {e}")
            return None
    
    def _return_bad_sample(self) -> Dict:
        """Return a dummy sample for error cases."""
        return {
            'img': torch.zeros((1, 1024), dtype=torch.float32),
            'coords': torch.zeros((1, 3), dtype=torch.int32),
            'filtered': torch.ones((1,), dtype=torch.uint8),
            'size': torch.tensor([1, 1, 1], dtype=torch.int32),
            'path': 'error/error',
            'label': -1,
            'study': 'error',
            'mode': 'error',
            'window': None
        }


class StudyAwareBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that groups images from the same study.
    
    Ensures all images from a study are in the same batch and on the same GPU.
    Distributes studies across GPUs in DDP, not individual images.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None
    ):
        """
        Initialize StudyAwareBatchSampler.
        
        Args:
            dataset: ImageDataset instance
            batch_size: Maximum number of images per batch
            shuffle: Shuffle studies
            seed: Random seed
            drop_last: Drop last incomplete batch
            num_replicas: Number of processes (for DDP)
            rank: Process rank (for DDP)
        """
        if num_replicas is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                num_replicas = 1
            else:
                num_replicas = torch.distributed.get_world_size()
        
        if rank is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                rank = 0
            else:
                rank = torch.distributed.get_rank()
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        
        # Build study_to_indices mapping
        self.study_to_indices = {}
        for idx, (study_name, image_name, mode) in enumerate(dataset.image_index):
            if study_name not in self.study_to_indices:
                self.study_to_indices[study_name] = []
            self.study_to_indices[study_name].append(idx)
        
        self.study_ids = list(self.study_to_indices.keys())
        self.num_studies = len(self.study_ids)
        
        if self.num_studies == 0:
            self.num_samples_per_replica = 0
            self.total_size = 0
            return
        
        self.num_samples_per_replica = math.ceil(self.num_studies / self.num_replicas)
        self.total_size = self.num_samples_per_replica * self.num_replicas
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches."""
        if self.num_studies == 0:
            return iter([])
        
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Shuffle or sequential studies
        if self.shuffle:
            study_indices = torch.randperm(self.num_studies, generator=g).tolist()
        else:
            study_indices = list(range(self.num_studies))
        
        # Pad to make evenly divisible by num_replicas
        padding_size = self.total_size - self.num_studies
        if padding_size > 0:
            study_indices += study_indices[:padding_size]
        
        # Subsample for this rank
        study_indices = study_indices[self.rank:self.total_size:self.num_replicas]
        assert len(study_indices) == self.num_samples_per_replica
        
        my_study_ids = [self.study_ids[i] for i in study_indices]
        
        # Build batches
        batch = []
        for study_id in my_study_ids:
            image_indices = list(self.study_to_indices[study_id])
            
            # Check if adding this study exceeds batch size
            if len(batch) + len(image_indices) > self.batch_size and len(batch) > 0:
                yield batch
                batch = []
            
            # If study itself is larger than batch size, yield it alone
            if len(image_indices) > self.batch_size:
                if len(batch) > 0:
                    yield batch
                yield image_indices
                batch = []
            else:
                batch.extend(image_indices)
        
        # Yield final batch
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self) -> int:
        """Approximate number of batches."""
        if self.num_studies == 0:
            return 0
        
        # Estimate average images per study
        total_images = len(self.dataset)
        avg_images_per_study = total_images / self.num_studies
        
        # Estimated images for this replica
        total_estimated_images = self.num_samples_per_replica * avg_images_per_study
        
        num_batches = total_estimated_images / self.batch_size
        return math.floor(num_batches) if self.drop_last else math.ceil(num_batches)
    
    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling."""
        self.epoch = epoch

