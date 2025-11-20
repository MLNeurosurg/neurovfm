"""
Study Preprocessor for NeuroVFM

Loads and preprocesses medical imaging studies for inference.
"""

import torch
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

from neurovfm.data.io import load_image
from neurovfm.data.preprocess import prepare_for_inference, tokenize_volume


class StudyPreprocessor:
    """
    Preprocessor for loading and batching medical imaging studies.
    
    Handles loading NIfTI/DICOM files, preprocessing, and batching for encoder inference.
    
    The preprocessing pipeline:
    1. Load image (NIfTI/DICOM)
    2. Reorient to RPI
    3. Resample to target spacing (1x1x4mm by default)
    4. Transpose acquisition axis to first dimension
    5. Center crop to make dimensions divisible by patch_size
    6. Apply modality-specific normalization (CT windowing / MRI scaling)
    7. Tokenize into patches
    8. Remove background tokens (optional)
    
    Args:
        patch_size (tuple): Patch size for tokenization. Defaults to (4, 16, 16).
        target_spacing (tuple): Target spacing for resampling. Defaults to (1.0, 1.0, 4.0).
        remove_background (bool): If True, physically remove background tokens before inference.
            If False, keep all tokens and return mask indices. Defaults to True for simplicity.
    
    Example:
        >>> preproc = StudyPreprocessor()
        >>> batch = preproc.load_study("/path/to/study/", modality="ct")
        >>> embs = encoder.embed(batch)
    """
    
    def __init__(
        self,
        patch_size: tuple = (4, 16, 16),
        target_spacing: tuple = (1.0, 1.0, 4.0),
        remove_background: bool = True,
    ):
        self.patch_size = patch_size
        self.target_spacing = target_spacing
        self.remove_background = remove_background
    
    def load_study(
        self,
        study_path: Union[str, Path, List[str], List[Path]],
        modality: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Load and preprocess a study (one or more volumes).
        
        Preprocessing steps:
        1. Load image with automatic format detection
        2. Reorient to RPI orientation
        3. Resample to target spacing (1x1x4mm)
        4. Transpose acquisition axis to first dimension
        5. Center crop to dimensions divisible by patch_size
        6. Apply modality-specific normalization
        7. Tokenize into patches
        
        Args:
            study_path (Union[str, Path, List]): Path to study directory, or list of image paths
            modality (str): Modality type ("ct" or "mri")
        
        Returns:
            Dict containing:
                - img: Tokens [N_total, patch_features] (e.g., 1024)
                - coords: Coordinates [N_total, 3]
                - series_cu_seqlens: Cumulative sequence lengths
                - series_max_len: Maximum sequence length
                - study_cu_seqlens: Study-level cumulative lengths
                - study_max_len: Study max length
                - mode: List of modalities
                - path: List of file paths
                - size: List of volume sizes (after preprocessing)
        
        Example:
            >>> batch = preproc.load_study("/path/to/study/", modality="ct")
            >>> batch = preproc.load_study(["/path/vol1.nii.gz", "/path/vol2.nii.gz"], modality="mri")
        """
        # Convert to list of paths
        if isinstance(study_path, (str, Path)):
            study_path = Path(study_path)
            if study_path.is_dir():
                # Load all NIfTI/DICOM files in directory
                image_paths = sorted(list(study_path.glob("*.nii.gz")) + 
                                    list(study_path.glob("*.nii")) +
                                    list(study_path.glob("*.dcm")))
            else:
                # Single file
                image_paths = [study_path]
        else:
            image_paths = [Path(p) for p in study_path]
        
        if len(image_paths) == 0:
            raise ValueError(f"No valid image files found in {study_path}")
        
        logging.info(f"Loading {len(image_paths)} volumes from study")
        
        # Load and tokenize each volume
        all_tokens = []
        all_coords = []
        series_lengths = []
        mode_list = []
        path_list = []
        size_list = []
        
        for img_path in image_paths:
            # Load and preprocess image (reorient, resample, crop to patch multiples)
            img_sitk = load_image(str(img_path), preprocess=True)
            if img_sitk is None:
                logging.warning(f"Failed to load {img_path}, skipping")
                continue
            
            # Convert to array and apply modality-specific preprocessing
            # This also transposes acquisition axis to first dimension
            result = prepare_for_inference(img_sitk, mode=modality)
            if result is None:
                logging.warning(f"Failed to prepare {img_path}, skipping")
                continue
            
            img_arrs, background_mask, view = result
            
            # For CT, we'll use all three windows and stack them
            # For MRI, we have a single array
            if modality.lower() == 'ct':
                # Process each CT window separately
                for window_idx, img_arr in enumerate(img_arrs):
                    # Tokenize with background filtering
                    tokens, coords, filtered = tokenize_volume(
                        img_arr,
                        background_mask,
                        patch_size=self.patch_size,
                        remove_background=self.remove_background
                    )
                    
                    # Convert to tensors
                    tokens_tensor = torch.from_numpy(tokens).float()
                    coords_tensor = torch.from_numpy(coords).long()
                    
                    all_tokens.append(tokens_tensor)
                    all_coords.append(coords_tensor)
                    series_lengths.append(len(tokens_tensor))
                    mode_list.append(modality)
                    if window_idx == 0:
                        path_list.append(f"{img_path.stem}_BrainWindow")
                    elif window_idx == 1:
                        path_list.append(f"{img_path.stem}_BloodWindow")
                    else:
                        path_list.append(f"{img_path.stem}_BoneWindow")
                    size_list.append(img_arr.shape)
            else:
                # MRI: single array
                img_arr = img_arrs[0]
                
                # Tokenize with background filtering
                tokens, coords, filtered = tokenize_volume(
                    img_arr,
                    background_mask,
                    patch_size=self.patch_size,
                    remove_background=self.remove_background
                )
                
                # Convert to tensors
                tokens_tensor = torch.from_numpy(tokens).float()
                coords_tensor = torch.from_numpy(coords).long()
                
                all_tokens.append(tokens_tensor)
                all_coords.append(coords_tensor)
                series_lengths.append(len(tokens_tensor))
                mode_list.append(modality)
                path_list.append(str(img_path))
                size_list.append(img_arr.shape)
        
        if len(all_tokens) == 0:
            raise ValueError(f"No valid volumes loaded from {study_path}")
        
        # Concatenate all series
        img = torch.cat(all_tokens, dim=0)  # [N_total, patch_features]
        coords = torch.cat(all_coords, dim=0)  # [N_total, 3]
        
        # Build cumulative sequence lengths for series
        series_cu_seqlens = torch.zeros(len(series_lengths) + 1, dtype=torch.int32)
        series_cu_seqlens[1:] = torch.tensor(series_lengths, dtype=torch.int32).cumsum(0)
        series_max_len = max(series_lengths)
        
        # Build study-level cumulative lengths (all series belong to one study)
        study_cu_seqlens = torch.tensor([0, len(series_lengths)], dtype=torch.int32)
        study_max_len = len(series_lengths)
        
        batch = {
            "img": img,
            "coords": coords,
            "series_masks_indices": torch.tensor([]),  # Empty if remove_background=True
            "series_cu_seqlens": series_cu_seqlens,
            "series_max_len": series_max_len,
            "study_cu_seqlens": study_cu_seqlens,
            "study_max_len": study_max_len,
            "mode": mode_list,
            "path": path_list,
            "size": size_list,
        }
        
        logging.info(f"Loaded study: {len(series_lengths)} series, {len(img)} total tokens (background {'removed' if self.remove_background else 'kept'})")
        return batch
    
    def __call__(
        self,
        study_path: Union[str, Path, List[str], List[Path]],
        modality: str,
    ) -> Dict[str, torch.Tensor]:
        """Alias for load_study()"""
        return self.load_study(study_path, modality)

