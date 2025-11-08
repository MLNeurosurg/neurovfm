"""
Medical Image I/O Module

This module provides a unified interface for loading and preprocessing medical images
from various formats (NIfTI, DICOM) with automatic format detection.

The main entry point is the `load_image()` function, which automatically detects
the image format based on file extension or directory structure, loads the image,
and applies standardized preprocessing suitable for model inference.

Example Usage:
    >>> from neurovfm.io import load_image
    >>> 
    >>> # Load NIfTI file
    >>> img = load_image('/path/to/scan.nii.gz')
    >>> 
    >>> # Load DICOM series from directory
    >>> img = load_image('/path/to/dicom_series/')
    >>> 
    >>> # Load single DICOM file
    >>> img = load_image('/path/to/scan.dcm')
    >>> 
    >>> # Convert to numpy array
    >>> if img is not None:
    ...     import SimpleITK as sitk
    ...     arr = sitk.GetArrayFromImage(img)
    ...     print(f"Shape: {arr.shape}")
"""

from pathlib import Path
from typing import Optional, Union
import SimpleITK as sitk

from .utils import load_nifti_file, load_dicom_file, preprocess_image


def load_image(path, preprocess=True):
    """
    Load and preprocess a medical image with automatic format detection.
    
    This is the main entry point for loading medical images. It automatically
    detects whether the input is NIfTI or DICOM format and applies the
    appropriate loading method, followed by optional standardized preprocessing.
    
    Format Detection Logic:
    - Files ending in .nii or .nii.gz → NIfTI
    - Files ending in .dcm or .dicom → DICOM
    - Directories → Assumed to contain DICOM series
    - Other extensions → Attempts generic SimpleITK loading (may fail)
    
    Preprocessing Pipeline (if preprocess=True):
    1. Reorients to standard RPI orientation
    2. Determines slice dimension based on spacing/size heuristics
    3. Resamples to anisotropic resolution (1x1x4mm)
    4. Crops to dimensions divisible by 16 (in-plane) and 4 (through-plane)
    
    Args:
        path (str or Path): Path to medical image file or directory containing
            DICOM series. Supported formats:
            - NIfTI: .nii, .nii.gz
            - DICOM: .dcm, .dicom, or directory with DICOM series
        preprocess (bool, optional): Whether to apply standardized preprocessing
            (reorientation, resampling, cropping). Defaults to True.
    
    Returns:
        SimpleITK.Image or None: Loaded (and optionally preprocessed) 3D image,
            or None if loading fails or image format is unsupported.
    
    Raises:
        FileNotFoundError: If the specified path doesn't exist.
    
    Examples:
        >>> # Load and preprocess a NIfTI file
        >>> img = load_image('brain_scan.nii.gz')
        >>> 
        >>> # Load DICOM without preprocessing
        >>> img = load_image('dicom_series/', preprocess=False)
        >>> 
        >>> # Load and convert to numpy
        >>> img = load_image('scan.nii.gz')
        >>> if img is not None:
        ...     arr = sitk.GetArrayFromImage(img)
        ...     print(f"Loaded volume: {arr.shape}, dtype: {arr.dtype}")
    
    Notes:
        - Multi-component images (e.g., RGB) are not supported and will return None
        - DICOM series should be in a single directory with consistent orientation
        - Preprocessing uses BSpline interpolation for high-quality resampling
        - Target spacing: 1x1x4mm (in-plane × through-plane)
    """
    path = Path(path)
    
    # Check if path exists
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    
    # Determine format and load image
    img_sitk = None
    
    if path.is_dir():
        # Directory: assume DICOM series
        print(f"Loading DICOM series from directory: {path}")
        img_sitk = load_dicom_file(path)
    else:
        # File: detect format by extension
        suffix = path.suffix.lower()
        
        if suffix in ['.nii', '.gz']:
            # Handle .nii.gz
            if path.name.endswith('.nii.gz'):
                print(f"Loading NIfTI file: {path}")
                img_sitk = load_nifti_file(path)
            elif suffix == '.nii':
                print(f"Loading NIfTI file: {path}")
                img_sitk = load_nifti_file(path)
            else:
                # .gz but not .nii.gz, might be compressed NIfTI
                print(f"Loading file (assuming NIfTI): {path}")
                img_sitk = load_nifti_file(path)
        
        elif suffix in ['.dcm', '.dicom']:
            print(f"Loading DICOM file: {path}")
            img_sitk = load_dicom_file(path)
        
        else:
            # Unknown format: try generic loading
            print(f"Warning: Unknown format '{suffix}'. Attempting generic load: {path}")
            try:
                img_sitk = sitk.ReadImage(str(path))
                if img_sitk.GetNumberOfComponentsPerPixel() > 1:
                    print(f"Warning: Multi-component image detected. Skipping {path}")
                    return None
            except Exception as e:
                print(f"Error loading file {path}: {e}")
                return None
    
    # Check if loading was successful
    if img_sitk is None:
        print(f"Failed to load image from: {path}")
        return None
    
    # Apply preprocessing if requested
    if preprocess:
        print(f"Applying standardized preprocessing...")
        try:
            img_sitk = preprocess_image(img_sitk)
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            return None
    
    print(f"Successfully loaded image: shape={img_sitk.GetSize()}, spacing={img_sitk.GetSpacing()}")
    return img_sitk

