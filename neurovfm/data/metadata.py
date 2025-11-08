"""
Metadata Management for Medical Image Datasets

This module provides utilities for creating, loading, and managing metadata
for medical image datasets. Metadata tracks study information, modalities,
sequences, and processing status.

Example Usage:
    >>> from neurovfm.data.metadata import DatasetMetadata
    >>> 
    >>> # Create metadata from raw data directory
    >>> metadata = DatasetMetadata.from_directory('/path/to/dataset')
    >>> 
    >>> # Get study information
    >>> study_info = metadata.get_study('study_001')
    >>> print(f"Mode: {study_info['mode']}")
    >>> 
    >>> # Save metadata
    >>> metadata.save('/path/to/dataset/metadata.json')
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime


class DatasetMetadata:
    """
    Manages metadata for medical image datasets.
    
    Tracks study-level information including modalities (CT/MRI), image files,
    sequences, and processing status.
    """
    
    def __init__(self, metadata_dict: Optional[Dict] = None):
        """
        Initialize DatasetMetadata.
        
        Args:
            metadata_dict (dict, optional): Existing metadata dictionary.
                If None, creates empty metadata.
        """
        if metadata_dict is None:
            self.data = {
                'dataset_name': 'NeuroVFM Dataset',
                'version': '1.0',
                'created': datetime.now().isoformat(),
                'num_studies': 0,
                'studies': {}
            }
        else:
            self.data = metadata_dict
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path]):
        """
        Load metadata from JSON file.
        
        Args:
            filepath (str or Path): Path to metadata JSON file.
        
        Returns:
            DatasetMetadata: Loaded metadata object.
        
        Raises:
            FileNotFoundError: If metadata file doesn't exist.
        """
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            metadata_dict = json.load(f)
        return cls(metadata_dict)
    
    @classmethod
    def from_directory(cls, data_dir: Union[str, Path], mode_mapping: Dict[str, str]):
        """
        Create metadata by scanning a raw data directory.
        
        Args:
            data_dir (str or Path): Root dataset directory containing raw/ subdirectory.
            mode_mapping (dict): Mapping of study names to modalities ('mri' or 'ct').
                Example: {'study_001': 'mri', 'study_002': 'ct'}
        
        Returns:
            DatasetMetadata: Metadata object with scanned studies.
        
        Example:
            >>> mode_mapping = {'study_001': 'mri', 'study_002': 'ct'}
            >>> metadata = DatasetMetadata.from_directory('/data', mode_mapping)
        """
        data_dir = Path(data_dir)
        raw_dir = data_dir / 'raw'
        
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
        
        metadata = cls()
        metadata.data['dataset_name'] = data_dir.name
        
        # Scan each study directory
        study_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
        
        for study_dir in study_dirs:
            study_name = study_dir.name
            
            # Get modality from mapping (required)
            if study_name not in mode_mapping:
                print(f"Warning: Study '{study_name}' not in mode_mapping, skipping")
                continue
            
            mode = mode_mapping[study_name].lower()
            if mode not in ['ct', 'mri']:
                print(f"Warning: Invalid mode '{mode}' for study '{study_name}', skipping")
                continue
            
            # Scan for image files
            images = cls._scan_images(study_dir)
            
            if images:
                metadata.add_study(study_name, mode, images)
        
        return metadata
    
    @staticmethod
    def _scan_images(study_dir: Path) -> Dict[str, Dict]:
        """
        Scan study directory for image files.
        
        Args:
            study_dir (Path): Study directory to scan.
        
        Returns:
            dict: Dictionary mapping image names (from filenames) to metadata.
        """
        images = {}
        
        # NIfTI files
        for nifti_file in study_dir.glob('*.nii*'):
            name = nifti_file.stem
            if name.endswith('.nii'):
                name = Path(name).stem  # Handle .nii.gz
            images[name] = {'filename': nifti_file.name, 'processed': False}
        
        # DICOM directories
        for item in study_dir.iterdir():
            if item.is_dir() and any(f.suffix.lower() in ['.dcm', '.dicom'] for f in item.rglob('*')):
                images[item.name] = {'filename': item.name, 'processed': False}
        
        # Single DICOM files
        for dcm_file in study_dir.glob('*.dcm'):
            images[dcm_file.stem] = {'filename': dcm_file.name, 'processed': False}
        
        return images
    
    def add_study(self, study_name: str, mode: str, images: Dict[str, Dict]):
        """
        Add a study to the metadata.
        
        Args:
            study_name (str): Unique study identifier.
            mode (str): Modality ('ct' or 'mri').
            images (dict): Dictionary of image information.
        """
        self.data['studies'][study_name] = {
            'mode': mode.lower(),
            'images': images
        }
        self.data['num_studies'] = len(self.data['studies'])
    
    def get_study(self, study_name: str) -> Optional[Dict]:
        """
        Get metadata for a specific study.
        
        Args:
            study_name (str): Study identifier.
        
        Returns:
            dict or None: Study metadata, or None if not found.
        """
        return self.data['studies'].get(study_name)
    
    def get_all_studies(self) -> Dict[str, Dict]:
        """
        Get metadata for all studies.
        
        Returns:
            dict: Dictionary of all study metadata.
        """
        return self.data['studies']
    
    def get_studies_by_mode(self, mode: str) -> List[str]:
        """
        Get list of study names for a specific modality.
        
        Args:
            mode (str): Modality ('ct' or 'mri').
        
        Returns:
            list: List of study names matching the modality.
        """
        return [
            name for name, info in self.data['studies'].items()
            if info['mode'] == mode.lower()
        ]
    
    def mark_processed(self, study_name: str, image_name: str):
        """
        Mark an image as processed.
        
        Args:
            study_name (str): Study identifier.
            image_name (str): Image identifier.
        """
        if study_name in self.data['studies']:
            if image_name in self.data['studies'][study_name]['images']:
                self.data['studies'][study_name]['images'][image_name]['processed'] = True
    
    def save(self, filepath: Union[str, Path]):
        """
        Save metadata to JSON file.
        
        Args:
            filepath (str or Path): Path to save metadata.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def __len__(self) -> int:
        """Return number of studies."""
        return self.data['num_studies']
    
    def __repr__(self) -> str:
        """String representation."""
        return f"DatasetMetadata(name='{self.data['dataset_name']}', studies={len(self)})"

