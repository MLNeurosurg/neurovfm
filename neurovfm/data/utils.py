"""
Shared Utility Functions for Medical Image Processing

This module provides helper functions for medical image preprocessing including:
- Format-specific loading (NIfTI, DICOM)
- Anatomical reorientation
- Physical spacing computation
- Standardized preprocessing pipeline

These utilities support the main image loading interface in io.py.
"""

import numpy as np
from pathlib import Path
import SimpleITK as sitk


def reorient(img_sitk, tgt='RPI'):
    """
    Reorient a medical image to a target anatomical orientation.
    
    This function standardizes the anatomical orientation of medical images,
    which is crucial for consistent processing across different acquisition
    protocols and scanner manufacturers.
    
    Standard orientations:
        - 'RPI': Right-Posterior-Inferior (recommended for Python/NumPy)
        - 'LPS': Left-Posterior-Superior (common in 3D Slicer)
    
    Args:
        img_sitk (SimpleITK.Image): Input SimpleITK image of shape [x, y, z].
        tgt (str, optional): Target orientation string. Each character represents
            an anatomical direction:
            - First char: L (Left) or R (Right)
            - Second char: P (Posterior) or A (Anterior)  
            - Third char: I (Inferior) or S (Superior)
            Defaults to 'RPI'.
    
    Returns:
        SimpleITK.Image: Reoriented image with the same physical space but
            potentially transposed dimensions.
    
    Example:
        >>> img = sitk.ReadImage('scan.nii.gz')
        >>> img_rpi = reorient(img, tgt='RPI')
    """
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation(tgt)
    return orienter.Execute(img_sitk)


def compute_spacing(img_sitk):
    """
    Extract the physical spacing (resolution) from a SimpleITK image.
    
    The spacing represents the physical distance between adjacent voxels
    in millimeters for each dimension.
    
    Args:
        img_sitk (SimpleITK.Image): Input SimpleITK image.
    
    Returns:
        numpy.ndarray: Array of shape (3,) containing spacing values [x, y, z]
            in millimeters for each spatial dimension.
    
    Example:
        >>> img = sitk.ReadImage('scan.nii.gz')
        >>> spacing = compute_spacing(img)
        >>> print(f"Voxel spacing: {spacing} mm")
        Voxel spacing: [0.5 0.5 3.0] mm
    """
    spacing_sitk = img_sitk.GetSpacing()
    return np.array(spacing_sitk, dtype=float)


def load_nifti_file(fpath):
    """
    Load a NIfTI file using SimpleITK.
    
    Args:
        fpath (str or Path): Path to NIfTI file (.nii or .nii.gz).
    
    Returns:
        SimpleITK.Image or None: Loaded image, or None if loading fails
            or image has multiple components (e.g., RGB).
    """
    try:
        img_sitk = sitk.ReadImage(str(fpath))
        
        # Validate image format - reject multi-component images (e.g., RGB)
        if img_sitk.GetNumberOfComponentsPerPixel() > 1:
            print(f"Warning: Multi-component image detected. Skipping {fpath}")
            return None
        
        return img_sitk
    except Exception as e:
        print(f"Error loading NIfTI file {fpath}: {e}")
        return None


def load_dicom_file(path):
    """
    Load DICOM data (single file or series from directory) using SimpleITK.
    
    Args:
        path (str or Path): Path to either:
            - A directory containing a DICOM series (multiple .dcm files)
            - A single DICOM file
    
    Returns:
        SimpleITK.Image or None: Loaded image, or None if loading fails,
            no DICOM files found, or image has multiple components.
    """
    path = Path(path)
    
    try:
        # Read DICOM series or single file
        if path.is_dir():
            # Read DICOM series from directory
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(path))
            
            if len(dicom_names) == 0:
                print(f"Warning: No DICOM files found in {path}")
                return None
            
            reader.SetFileNames(dicom_names)
            img_sitk = reader.Execute()
        else:
            # Read single DICOM file
            img_sitk = sitk.ReadImage(str(path))
        
        # Validate image format - reject multi-component images (e.g., RGB)
        if img_sitk.GetNumberOfComponentsPerPixel() > 1:
            print(f"Warning: Multi-component image detected. Skipping {path}")
            return None
        
        return img_sitk
    except Exception as e:
        print(f"Error loading DICOM file/series {path}: {e}")
        return None


def preprocess_image(img_sitk):
    """
    Apply standardized preprocessing to a medical image.
    
    This function performs the complete preprocessing pipeline:
    1. Reorients to standard RPI orientation
    2. Determines slice dimension based on spacing/size heuristics
    3. Resamples to anisotropic resolution (1x1x4mm)
    4. Crops to dimensions divisible by 16 (in-plane) and 4 (through-plane)
    
    The slice dimension (z_dim) is automatically detected using the following logic:
    - If all spacings are equal: defaults to dimension 0
    - If sizes are all different: uses dimension with largest spacing
    - Otherwise: uses dimension with unique size (typically slice direction)
    
    Target spacing after resampling:
    - In-plane dimensions: 1.0 mm
    - Through-plane (slice) dimension: 4.0 mm
    
    Args:
        img_sitk (SimpleITK.Image): Input medical image (NIfTI or DICOM).
    
    Returns:
        SimpleITK.Image: Preprocessed 3D image with standardized spacing
            and dimensions suitable for model inference.
    
    Notes:
        - Cropping ensures dimensions are compatible with patch-based processing
        - BSpline interpolation is used for high-quality resampling
    """
    # Reorient to standard RPI coordinate system
    # RPI = Right-Posterior-Inferior (standard for medical image processing)
    img_sitk = reorient(img_sitk, tgt='RPI')  # (x, y, z)
    
    # Extract physical spacing information
    spacing_sitk = compute_spacing(img_sitk)  # -> np.array([x, y, z])
    
    original_spacing = spacing_sitk
    original_size = img_sitk.GetSize()
    
    # Determine slice dimension (z_dim) using heuristics
    # This identifies which dimension represents the slice/through-plane axis
    if len(set(list(original_spacing))) == 1:
        # All spacings equal - default to dimension 2
        z_dim = 2
    else:
        counts = np.bincount(original_size)
        if (counts == 1).all():
            # All sizes are unique - use dimension with largest spacing
            z_dim = np.argmax(original_spacing)
        else:
            # Use dimension with unique size (typically slice direction)
            z_dim = np.where(original_size == np.where(counts == 1)[0][0])[0][0]
    
    # Define target anisotropic spacing: 1x1x4 mm
    # In-plane: 1mm × 1mm for high resolution
    # Through-plane: 4mm for typical slice thickness
    target_spacing = [1, 1, 1]
    target_spacing[z_dim] = 4
    
    # Calculate new size based on spacing ratio
    new_size = [
        int(original_size[i] * (original_spacing[i] / target_spacing[i])) 
        for i in range(len(original_size))
    ]
    
    # Configure resampler with BSpline interpolation for smooth results
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img_sitk.GetOrigin())
    resampler.SetOutputDirection(img_sitk.GetDirection())
    resampler.SetInterpolator(sitk.sitkBSpline)
    
    # Execute resampling
    resized_img_sitk = resampler.Execute(img_sitk)
    
    # Crop to dimensions divisible by patch sizes
    # In-plane: divisible by 16 for patch-based processing
    # Through-plane: divisible by 4 for slice-based processing
    start_index, crop_size = [], []
    for idx in range(3):
        if idx == z_dim:
            # Slice dimension: make divisible by 4
            start_index.append((new_size[idx] % 4) // 2)
            crop_size.append((new_size[idx] // 4) * 4)
        else:
            # In-plane dimensions: make divisible by 16
            start_index.append((new_size[idx] % 16) // 2)
            crop_size.append((new_size[idx] // 16) * 16)
    
    # Extract cropped region (centered crop)
    resized_img_sitk = sitk.Extract(resized_img_sitk, crop_size, start_index)
    
    return resized_img_sitk