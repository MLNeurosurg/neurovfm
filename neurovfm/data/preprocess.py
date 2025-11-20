"""
Array Preprocessing Module for Model Inference

This module provides functions to convert loaded SimpleITK images into
properly formatted numpy arrays ready for model inference. It handles
modality-specific preprocessing (CT windowing, MRI normalization) and
background mask generation.

The main entry point is `prepare_for_inference()` which takes a preprocessed
SimpleITK image and returns model-ready arrays.

Example Usage:
    >>> from neurovfm.data.io import load_image
    >>> from neurovfm.data.preprocess import prepare_for_inference
    >>> 
    >>> # Load and preprocess image
    >>> img = load_image('scan.nii.gz')
    >>> 
    >>> # Prepare for model inference
    >>> img_arrs, background_mask, view = prepare_for_inference(img, mode='mri')
    >>> 
    >>> # img_arrs is a list of arrays ready for the model
    >>> for arr in img_arrs:
    ...     print(f"Array shape: {arr.shape}")
"""

import numpy as np
import SimpleITK as sitk
from typing import List, Tuple, Optional


def get_background_mask_ct(img_arr):
    """
    Generate a background mask for CT images.
    
    Identifies background regions in CT images based on intensity thresholds.
    Background voxels are typically air or regions outside the patient.
    
    Args:
        img_arr (numpy.ndarray): CT image array of shape [D, H, W].
    
    Returns:
        numpy.ndarray: Binary mask of shape [D, H, W] where True indicates
            foreground (patient tissue) and False indicates background.
    
    Notes:
        - CT values are in Hounsfield Units (HU)
        - Air is approximately -1000 HU
        - This function identifies regions likely to be background
    """
    # CT-specific background detection
    # Typically, background in CT is very low intensity (air ~ -1000 HU)
    mask = img_arr > -500  # Threshold for separating air from tissue
    return mask


def get_background_mask_mri(img_arr):
    """
    Generate a background mask for MRI images.
    
    Identifies background regions in MRI images based on intensity statistics.
    Background in MRI is typically low intensity noise outside the anatomy.
    
    Args:
        img_arr (numpy.ndarray): MRI image array of shape [D, H, W].
    
    Returns:
        numpy.ndarray: Binary mask of shape [D, H, W] where True indicates
            foreground (anatomy) and False indicates background.
    
    Notes:
        - MRI background is typically near-zero intensity
        - Uses percentile-based thresholding to separate signal from noise
    """
    # MRI-specific background detection
    # Background is typically low intensity
    threshold = np.percentile(img_arr, 10)
    mask = img_arr > threshold
    return mask


def clip_by_window(img_arr, window_width, window_level):
    """
    Apply CT windowing to enhance specific tissue contrast.
    
    CT windowing (also called window/level adjustment) enhances visualization
    of specific tissue types by mapping a range of Hounsfield Units to the
    full display range [0, 1].
    
    Args:
        img_arr (numpy.ndarray): CT image array in Hounsfield Units.
        window_width (int or float): Width of the window (range of HU values).
        window_level (int or float): Center of the window (middle HU value).
    
    Returns:
        numpy.ndarray: Windowed image normalized to [0, 1].
    
    Example:
        >>> # Brain window (good for brain tissue)
        >>> brain = clip_by_window(ct_array, window_width=80, window_level=40)
        >>> 
        >>> # Bone window (good for skeletal structures)
        >>> bone = clip_by_window(ct_array, window_width=2800, window_level=600)
    
    Notes:
        Common CT windows:
        - Brain: W=80, L=40
        - Subdural: W=200, L=80 (blood)
        - Bone: W=2800, L=600
        - Lung: W=1500, L=-600
        - Abdomen: W=400, L=50
    """
    # Calculate window boundaries
    lower = window_level - window_width / 2
    upper = window_level + window_width / 2
    
    # Clip and normalize to [0, 1]
    windowed = np.clip(img_arr, lower, upper)
    windowed = (windowed - lower) / (upper - lower + 1e-8)
    
    return windowed


def transpose_to_dhw(img_arr, z_dim):
    """
    Transpose image array so the slice dimension (resampled 4mm axis) is first.
    
    After resampling, the slice dimension (with 4mm spacing) needs to be
    the first dimension [D, H, W] for proper model processing.
    
    Args:
        img_arr (numpy.ndarray): Image array from SimpleITK (z, y, x) order.
        z_dim (int): Which dimension (0, 1, or 2) is the slice dimension
            in the original SimpleITK coordinate system.
    
    Returns:
        tuple: (transposed_array, view_indicator)
            - transposed_array: Array with shape [D, H, W] where D is slice dim
            - view_indicator: Integer (0, 1, or 2) indicating original slice axis
    
    Notes:
        - SimpleITK.GetArrayFromImage returns array in (z, y, x) order
        - We need to transpose so the 4mm-spaced dimension is first
    """
    # SimpleITK returns array in (z, y, x) order
    # z_dim indicates which axis in (x, y, z) space is the slice dimension
    # We need to map this to the array indexing
    
    # Convert z_dim from (x,y,z) to array (z,y,x) indexing
    view = 2 - z_dim
    
    # Transpose to put slice dimension first
    if view == 0:
        # Already in correct order [D, H, W]
        return img_arr, view
    elif view == 1:
        # Need to swap: (z, D, x) -> (D, z, x)
        return np.transpose(img_arr, (1, 0, 2)), view
    elif view == 2:
        # Need to swap: (z, y, D) -> (D, y, z)
        return np.transpose(img_arr, (2, 1, 0)), view
    else:
        # Default: no transpose
        return img_arr, view


def prepare_for_inference(img_sitk, mode, z_dim=None):
    """
    Convert a preprocessed SimpleITK image to model-ready numpy arrays.
    
    This function handles the complete array-level preprocessing pipeline:
    1. Converts SimpleITK image to numpy array
    2. Transposes so slice dimension (4mm spacing) is first [D, H, W]
    3. Applies modality-specific preprocessing:
       - CT: Generates brain/blood/bone windowed views + background mask
       - MRI: Applies percentile clipping and normalization + background mask
    4. Validates minimum dimensions (D>=4, H>=16, W>=16)
    
    Args:
        img_sitk (SimpleITK.Image): Preprocessed SimpleITK image (after
            reorientation, resampling, and cropping from io.load_image).
        mode (str): Imaging modality, either 'ct' or 'mri' (case-insensitive).
        z_dim (int, optional): Which dimension (0, 1, or 2) was the slice
            dimension in the original image coordinate system. If None,
            attempts to infer from spacing (dimension with largest spacing).
    
    Returns:
        tuple: (img_arrs, background_mask, view) or None if validation fails
            - img_arrs (list of numpy.ndarray): List of preprocessed arrays.
              For CT: 3 arrays (brain, blood, bone windows)
              For MRI: 1 array (normalized)
              Each array has shape [D, H, W] with values in [0, 1]
            - background_mask (numpy.ndarray): Binary mask [D, H, W] indicating
              foreground regions (True = tissue, False = background)
            - view (int): Indicator of which axis was the slice dimension (0, 1, or 2)
    
    Returns:
        None: If image dimensions are too small (D<4, H<16, or W<16)
    
    Raises:
        ValueError: If mode is not 'ct' or 'mri'
    
    Example:
        >>> from neurovfm.data.io import load_image
        >>> from neurovfm.data.preprocess import prepare_for_inference
        >>> 
        >>> # Load CT scan
        >>> img = load_image('ct_scan.nii.gz')
        >>> img_arrs, mask, view = prepare_for_inference(img, mode='ct')
        >>> 
        >>> # img_arrs contains 3 windowed views
        >>> print(f"Brain window: {img_arrs[0].shape}")
        >>> print(f"Blood window: {img_arrs[1].shape}")
        >>> print(f"Bone window: {img_arrs[2].shape}")
        >>> 
        >>> # Load MRI scan
        >>> img = load_image('mri_scan.nii.gz')
        >>> img_arrs, mask, view = prepare_for_inference(img, mode='mri')
        >>> print(f"Normalized MRI: {img_arrs[0].shape}")
    
    Notes:
        - Input image should already be preprocessed (reoriented, resampled to 1x1x4mm)
        - CT windows used: Brain (W=80, L=40), Blood (W=200, L=80), Bone (W=2800, L=600)
        - MRI normalization: 0.5th-99.5th percentile clipping followed by min-max scaling
        - Background masks help the model focus on relevant anatomy
    """
    mode = mode.lower()
    if mode not in ['ct', 'mri']:
        raise ValueError(f"Mode must be 'ct' or 'mri', got: {mode}")
    
    # Infer z_dim from spacing if not provided
    if z_dim is None:
        spacing = img_sitk.GetSpacing()
        if np.unique(spacing).shape[0] == 1:
            z_dim = 2 # assume axial
        else:
            z_dim = np.argmax(spacing)  # Dimension with largest spacing (4mm)
    
    # Convert SimpleITK image to numpy array
    # GetArrayFromImage returns (z, y, x) order
    img_arr = sitk.GetArrayFromImage(img_sitk)
    
    # Transpose to [D, H, W] where D is the slice dimension
    img_arr, view = transpose_to_dhw(img_arr, z_dim)
    D, H, W = img_arr.shape
    
    # Validate minimum dimensions
    if D < 4 or H < 16 or W < 16:
        print(f"Warning: Image too small (D={D}, H={H}, W={W}). Minimum: D>=4, H>=16, W>=16")
        return None
    
    # Ensure contiguous array in float64
    img_arr = img_arr.astype(np.float64).copy()
    
    # Apply modality-specific preprocessing
    if mode == 'ct':
        # CT: Generate windowed views and background mask
        background_mask = get_background_mask_ct(img_arr.copy())
        
        # Generate three standard CT windows: brain, blood (subdural), bone
        img_arrs = [
            clip_by_window(img_arr, window_width=80, window_level=40),      # Brain
            clip_by_window(img_arr, window_width=200, window_level=80),     # Blood
            clip_by_window(img_arr, window_width=2800, window_level=600)    # Bone
        ]
    
    elif mode == 'mri':
        # MRI: Percentile clipping and normalization
        background_mask = get_background_mask_mri(img_arr.copy())
        
        # Clip to 0.5th-99.5th percentile to remove outliers
        img_arr = np.clip(
            img_arr,
            np.percentile(img_arr, 0.5),
            np.percentile(img_arr, 99.5)
        )
        
        # Min-max normalization to [0, 1]
        img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8)
        
        img_arrs = [img_arr]
    
    return img_arrs, background_mask, view


def tokenize_volume(
    img_arr: np.ndarray,
    mask_arr: np.ndarray,
    patch_size: Tuple[int, int, int] = (4, 16, 16),
    remove_background: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tokenize a 3D volume into patches for model inference.
    
    Converts a 3D image volume [D, H, W] into a sequence of flattened patches.
    Each patch is flattened into a 1D vector, creating a tokens × features format.
    Also generates a background/foreground mask for each token.
    
    Args:
        img_arr (numpy.ndarray): Image array of shape [D, H, W] with values in [0, 1].
        mask_arr (numpy.ndarray): Binary mask of shape [D, H, W] (True = foreground).
        patch_size (tuple): Size of patches (depth, height, width). 
            Defaults to (4, 16, 16) for 1024-D tokens.
        remove_background (bool): If True, physically remove background tokens.
            If False, return all tokens with a filtered mask. Defaults to False.
    
    Returns:
        tuple: (tokens, coords, filtered)
            - tokens (numpy.ndarray): Tokenized patches [N, patch_features] or [N_fg, patch_features]
            - coords (numpy.ndarray): 3D coordinates [N, 3] or [N_fg, 3] for each token
            - filtered (numpy.ndarray): Background mask [N] where 1=background, 0=foreground
                                        (empty if remove_background=True)
    
    Example:
        >>> img_arr = np.random.rand(64, 192, 192)  # D, H, W
        >>> mask_arr = np.ones((64, 192, 192), dtype=bool)
        >>> tokens, coords, filtered = tokenize_volume(img_arr, mask_arr)
        >>> print(tokens.shape)  # (16*12*12, 4*16*16) = (2304, 1024)
        >>> print(coords.shape)  # (2304, 3)
        >>> print(filtered.shape)  # (2304,) - 0 for foreground, 1 for background
    
    Notes:
        - Image dimensions must be divisible by patch_size
        - Patch features = product of patch_size elements (e.g., 4*16*16=1024)
        - Coordinates represent the (d, h, w) index of each patch in the grid
        - A token is marked as background if ANY pixel in its patch is background
    """
    import torch
    from einops import rearrange
    
    D, H, W = img_arr.shape
    p1, p2, p3 = patch_size
    
    # Validate dimensions are divisible by patch size
    assert D % p1 == 0, f"Depth {D} not divisible by patch size {p1}"
    assert H % p2 == 0, f"Height {H} not divisible by patch size {p2}"
    assert W % p3 == 0, f"Width {W} not divisible by patch size {p3}"
    
    # Calculate number of patches per dimension
    n_patches_d = D // p1
    n_patches_h = H // p2
    n_patches_w = W // p3
    
    # Tokenize image: [D, H, W] -> [N, patch_features]
    tokens_torch = rearrange(
        torch.from_numpy(img_arr).unsqueeze(0),  # Add channel dim [1, D, H, W]
        "c (d p1) (h p2) (w p3) -> (d h w) (c p1 p2 p3)",
        d=n_patches_d, h=n_patches_h, w=n_patches_w,
        p1=p1, p2=p2, p3=p3
    ).squeeze(1)  # Remove channel dim: [N, 1024]
    
    # Tokenize mask: a patch is background if ANY pixel in it is background
    # mask_arr is True for foreground, False for background
    mask_tokens = rearrange(
        torch.from_numpy(mask_arr),
        "(d p1) (h p2) (w p3) -> (d h w) (p1 p2 p3)",
        d=n_patches_d, h=n_patches_h, w=n_patches_w,
        p1=p1, p2=p2, p3=p3
    )
    # filtered: 1 if any pixel in patch is background (i.e., not all pixels are foreground)
    # 0 if all pixels in patch are foreground
    filtered = (~(mask_tokens.all(dim=1))).to(torch.uint8)  # [N]
    
    # Generate 3D coordinates for each token
    coords_d, coords_h, coords_w = np.meshgrid(
        np.arange(n_patches_d),
        np.arange(n_patches_h),
        np.arange(n_patches_w),
        indexing='ij'
    )
    coords_torch = torch.from_numpy(
        np.stack([coords_d.flatten(), coords_h.flatten(), coords_w.flatten()], axis=1)
    ).long()
    
    if remove_background:
        # Physically remove background tokens
        fg_mask = ~filtered.bool()
        tokens = tokens_torch[fg_mask].numpy()
        coords = coords_torch[fg_mask].numpy()
        filtered_out = np.array([], dtype=np.uint8)  # Empty since we removed them
    else:
        # Keep all tokens, return filtered mask
        tokens = tokens_torch.numpy()
        coords = coords_torch.numpy()
        filtered_out = filtered.numpy()
    
    return tokens, coords, filtered_out

