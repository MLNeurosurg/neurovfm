"""
Preprocessing Cache Management

This module provides utilities for building and managing preprocessed data caches.
It handles conversion from raw medical images to preprocessed PyTorch tensors,
with special handling for CT (multiple windows) and MRI (normalized) data.

Example Usage:
    >>> from neurovfm.data.cache import CacheManager
    >>> 
    >>> # Build preprocessing cache for entire dataset
    >>> cache_mgr = CacheManager('/path/to/dataset')
    >>> cache_mgr.build_cache(num_workers=8, force=False)
    >>> 
    >>> # Load preprocessed data
    >>> data = cache_mgr.load_image('study_001', 'T1')
    >>> image = data['data']  # [D, H, W] tensor
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp

from .io import load_image
from .preprocess import prepare_for_inference
from .metadata import DatasetMetadata


class CacheManager:
    """
    Manages preprocessing cache for medical image datasets.
    
    Handles conversion of raw medical images to preprocessed PyTorch tensors,
    storage, and retrieval of cached data.
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize CacheManager.
        
        Args:
            data_dir (str or Path): Root dataset directory containing
                raw/ and processed/ subdirectories and metadata.json.
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.metadata_file = self.data_dir / 'metadata.json'
        
        # Load or create metadata
        if self.metadata_file.exists():
            self.metadata = DatasetMetadata.from_file(self.metadata_file)
        else:
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_file}\n"
                "Please create metadata first using DatasetMetadata.from_directory()"
            )
        
        # Create processed directory if it doesn't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def build_cache(self, num_workers: int = 1, force: bool = False, 
                    study_names: Optional[List[str]] = None):
        """
        Build preprocessing cache for the dataset.
        
        Processes raw images and saves preprocessed tensors to the cache.
        Can be parallelized across multiple workers.
        
        Args:
            num_workers (int): Number of parallel workers. Defaults to 1.
            force (bool): If True, reprocess even if cache exists. Defaults to False.
            study_names (list, optional): List of specific studies to process.
                If None, processes all studies.
        
        Example:
            >>> cache_mgr = CacheManager('/path/to/dataset')
            >>> cache_mgr.build_cache(num_workers=8, force=False)
        """
        # Get studies to process
        if study_names is None:
            studies = list(self.metadata.get_all_studies().keys())
        else:
            studies = study_names
        
        print(f"Building cache for {len(studies)} studies...")
        
        # Collect all images to process
        tasks = []
        for study_name in studies:
            study_info = self.metadata.get_study(study_name)
            if study_info is None:
                print(f"Warning: Study '{study_name}' not found in metadata")
                continue
            
            mode = study_info['mode']
            
            for image_name, image_info in study_info['images'].items():
                # Skip if already processed and not forcing
                if not force and image_info.get('processed', False):
                    if self._cache_exists(study_name, image_name, mode):
                        continue
                
                tasks.append((study_name, image_name, mode, image_info))
        
        if not tasks:
            print("All images already cached!")
            return
        
        print(f"Processing {len(tasks)} images...")
        
        # Process with multiprocessing if num_workers > 1
        if num_workers > 1:
            with mp.Pool(num_workers) as pool:
                results = list(tqdm(
                    pool.imap(self._process_image_wrapper, tasks),
                    total=len(tasks),
                    desc="Preprocessing"
                ))
        else:
            results = [
                self._process_image_wrapper(task)
                for task in tqdm(tasks, desc="Preprocessing")
            ]
        
        # Count successes
        successful = sum(1 for r in results if r is not None)
        print(f"Successfully processed {successful}/{len(tasks)} images")
        
        # Save updated metadata
        self.metadata.save(self.metadata_file)
    
    def _process_image_wrapper(self, task: Tuple) -> Optional[bool]:
        """
        Wrapper for processing a single image (for multiprocessing).
        
        Args:
            task (tuple): (study_name, image_name, mode, image_info)
        
        Returns:
            bool or None: True if successful, None if failed.
        """
        study_name, image_name, mode, image_info = task
        try:
            return self._process_image(study_name, image_name, mode, image_info)
        except Exception as e:
            print(f"Error processing {study_name}/{image_name}: {e}")
            return None
    
    def _process_image(self, study_name: str, image_name: str, 
                      mode: str, image_info: Dict) -> bool:
        """
        Process a single image and save to cache.
        
        Args:
            study_name (str): Study identifier.
            image_name (str): Image identifier.
            mode (str): Modality ('ct' or 'mri').
            image_info (dict): Image metadata.
        
        Returns:
            bool: True if successful.
        """
        # Construct path to raw image
        raw_study_dir = self.raw_dir / study_name
        raw_path = raw_study_dir / image_info['filename']
        
        # Load and preprocess
        img_sitk = load_image(raw_path, preprocess=True)
        if img_sitk is None:
            print(f"Failed to load {raw_path}")
            return False
        
        result = prepare_for_inference(img_sitk, mode=mode)
        if result is None:
            print(f"Failed to preprocess {raw_path}")
            return False
        
        img_arrs, background_mask, view = result
        
        # Create output directory
        processed_study_dir = self.processed_dir / study_name
        processed_study_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        metadata_dict = {
            'source_file': str(raw_path.relative_to(self.data_dir)),
            'mode': mode,
            'shape': list(img_arrs[0].shape),
            'spacing': list(img_sitk.GetSpacing()),
            'view': int(view),
            'processed_date': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # Convert mask: True=foreground -> 1=background, 0=foreground
        mask_binary = (~background_mask).astype(np.uint8)
        
        # Save based on modality
        if mode == 'ct':
            # Save three windowed versions as uint8
            windows = ['brain', 'blood', 'bone']
            for window_name, arr in zip(windows, img_arrs):
                output_path = processed_study_dir / f"{image_name}_{window_name}.pt"
                
                # Convert [0,1] float to [0,255] uint8
                arr_uint8 = (arr * 255).astype(np.uint8)
                
                data_dict = {
                    'data': torch.from_numpy(arr_uint8),
                    'view': view,
                    'shape': arr.shape,
                    'spacing': tuple(img_sitk.GetSpacing()),
                    'mode': mode,
                    'window': window_name
                }
                
                torch.save(data_dict, output_path)
            
            # Save mask once (shared across all windows)
            mask_path = processed_study_dir / f"{image_name}_mask.pt"
            torch.save(torch.from_numpy(mask_binary), mask_path)
            
            # Save metadata once for all windows
            metadata_dict['windows'] = windows
            metadata_path = processed_study_dir / f"{image_name}_metadata.json"
            
        else:  # MRI
            # Save single normalized version as uint8
            output_path = processed_study_dir / f"{image_name}.pt"
            
            # Convert [0,1] float to [0,255] uint8
            arr_uint8 = (img_arrs[0] * 255).astype(np.uint8)
            
            data_dict = {
                'data': torch.from_numpy(arr_uint8),
                'view': view,
                'shape': img_arrs[0].shape,
                'spacing': tuple(img_sitk.GetSpacing()),
                'mode': mode,
                'window': None
            }
            
            torch.save(data_dict, output_path)
            
            # Save mask
            mask_path = processed_study_dir / f"{image_name}_mask.pt"
            torch.save(torch.from_numpy(mask_binary), mask_path)
            
            # Save metadata
            metadata_path = processed_study_dir / f"{image_name}_metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        # Mark as processed in dataset metadata
        self.metadata.mark_processed(study_name, image_name)
        
        return True
    
    def _cache_exists(self, study_name: str, image_name: str, mode: str) -> bool:
        """
        Check if cached data exists for an image.
        
        Args:
            study_name (str): Study identifier.
            image_name (str): Image identifier.
            mode (str): Modality ('ct' or 'mri').
        
        Returns:
            bool: True if cache exists (including mask file).
        """
        processed_study_dir = self.processed_dir / study_name
        mask_path = processed_study_dir / f"{image_name}_mask.pt"
        
        if mode == 'ct':
            # Check for all three windows + mask
            windows = ['brain', 'blood', 'bone']
            return mask_path.exists() and all(
                (processed_study_dir / f"{image_name}_{w}.pt").exists()
                for w in windows
            )
        else:
            # Check for single file + mask
            return mask_path.exists() and (processed_study_dir / f"{image_name}.pt").exists()
    
    def load_image(self, study_name: str, image_name: str, 
                   window: Optional[str] = None) -> Optional[Union[Dict, List[Dict]]]:
        """
        Load preprocessed image from cache.
        
        Args:
            study_name (str): Study identifier.
            image_name (str): Image identifier.
            window (str, optional): For CT, specify window ('brain', 'blood', 'bone').
                If None and image is CT, loads all windows.
        
        Returns:
            dict or list or None: Preprocessed data dictionary with:
                - 'data': Tensor [D, H, W] float32 in [0, 1]
                - 'mask': Tensor [D, H, W] uint8, 1=background, 0=foreground
                - 'view': int
                - 'mode': str
                - 'window': str or None
            For CT with window=None, returns list of dicts (one per window).
        """
        processed_study_dir = self.processed_dir / study_name
        
        # Get mode from metadata
        study_info = self.metadata.get_study(study_name)
        if study_info is None:
            return None
        
        mode = study_info['mode']
        
        # Load mask (shared for all windows in CT)
        mask_path = processed_study_dir / f"{image_name}_mask.pt"
        if not mask_path.exists():
            return None
        mask = torch.load(mask_path)
        
        if mode == 'ct':
            if window is not None:
                # Load specific window
                cache_path = processed_study_dir / f"{image_name}_{window}.pt"
                if not cache_path.exists():
                    return None
                data = torch.load(cache_path)
                # Convert uint8 [0,255] back to float32 [0,1]
                data['data'] = data['data'].float() / 255.0
                data['mask'] = mask
                return data
            else:
                # Load all windows
                windows = ['brain', 'blood', 'bone']
                data_list = []
                for w in windows:
                    cache_path = processed_study_dir / f"{image_name}_{w}.pt"
                    if not cache_path.exists():
                        return None
                    data = torch.load(cache_path)
                    # Convert uint8 [0,255] back to float32 [0,1]
                    data['data'] = data['data'].float() / 255.0
                    data['mask'] = mask
                    data_list.append(data)
                return data_list
        else:
            # Load MRI
            cache_path = processed_study_dir / f"{image_name}.pt"
            if not cache_path.exists():
                return None
            data = torch.load(cache_path)
            # Convert uint8 [0,255] back to float32 [0,1]
            data['data'] = data['data'].float() / 255.0
            data['mask'] = mask
            return data
    
    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the preprocessing cache.
        
        Returns:
            dict: Cache statistics including total images, cached images, etc.
        """
        total_images = 0
        cached_images = 0
        
        for study_name, study_info in self.metadata.get_all_studies().items():
            mode = study_info['mode']
            for image_name in study_info['images']:
                total_images += 1
                if self._cache_exists(study_name, image_name, mode):
                    cached_images += 1
        
        return {
            'total_studies': len(self.metadata),
            'total_images': total_images,
            'cached_images': cached_images,
            'cache_coverage': cached_images / total_images if total_images > 0 else 0.0
        }

