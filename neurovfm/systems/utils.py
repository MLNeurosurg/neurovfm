"""
Utility modules for PyTorch Lightning systems.

Contains shared components like normalization modules used across
pretraining and downstream tasks.
"""

import logging
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizationModule(nn.Module):
    """
    Token-level normalization for medical imaging.

    Expects pre-tokenized rows with Features == 1024 (4*16*16).
    Applies: (divide by 255) -> mean/std normalize
    
    Normalization is modality and window-specific:
    - MRI: index 0
    - CT Brain Window: index 1
    - CT Blood Window: index 2
    - CT Bone Window: index 3
    
    Args:
        custom_stats_list (List[List[float]], optional): [[means], [stds]] for [mri, brain, blood, bone]
    """
    
    DEFAULT_MEANS_VALUES = [0.3141, 0.4139, 0.3184, 0.2719]  # mri, brain, blood, bone
    DEFAULT_STDS_VALUES  = [0.2623, 0.4059, 0.3605, 0.1875]

    def __init__(
        self,
        custom_stats_list: Optional[List[List[float]]] = None,
    ):
        super().__init__()
        
        # Setup normalization stats
        if custom_stats_list is not None:
            if not (len(custom_stats_list) == 2 and all(len(sub) == 4 for sub in custom_stats_list)):
                raise ValueError("custom_stats_list must be a 2x4 list of floats.")
            means_to_register = torch.tensor([float(x) for x in custom_stats_list[0]], dtype=torch.float32)
            stds_to_register  = torch.tensor([float(x) for x in custom_stats_list[1]], dtype=torch.float32)
            if (torch.isnan(means_to_register).any() or torch.isinf(means_to_register).any() or
                torch.isnan(stds_to_register).any() or torch.isinf(stds_to_register).any()):
                raise ValueError("Custom normalization stats must not contain NaN/Inf.")
            if (stds_to_register <= 0).any():
                raise ValueError("Standard deviations must be positive.")
        else:
            means_to_register = torch.tensor(self.DEFAULT_MEANS_VALUES, dtype=torch.float32)
            stds_to_register  = torch.tensor(self.DEFAULT_STDS_VALUES, dtype=torch.float32)

        self.register_buffer("means", means_to_register)
        self.register_buffer("stds",  stds_to_register)

    def get_normalization_params(self, modes: List[str], paths: List[str], device: torch.device):
        """
        Determine normalization parameters based on modality and window type.
        
        Args:
            modes (List[str]): List of modalities ('mri' or 'ct')
            paths (List[str]): List of file paths (used to determine CT window type)
            device (torch.device): Device for tensors
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (means, stds) for each sample in batch
        """
        batch_means = []
        batch_stds = []
        
        for mode, path in zip(modes, paths):
            if mode == "mri":
                idx = 0
            elif mode == "ct":
                # Determine CT window type from path
                if "BrainWindow" in path:
                    idx = 1
                elif "BloodWindow" in path:
                    idx = 2
                elif "BoneWindow" in path:
                    idx = 3
                else:
                    raise ValueError(f"Unknown CT window type: {path}")
            else:
                raise ValueError(f"Unknown modality: {mode}")
            
            batch_means.append(self.means[idx])
            batch_stds.append(self.stds[idx])
        
        return torch.stack(batch_means).to(device), torch.stack(batch_stds).to(device)

    def normalize(
        self, 
        img: torch.Tensor, 
        modes: List[str], 
        paths: List[str],
        cu_seqlens: torch.Tensor,
        sizes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Normalize input image tokens.
        
        Args:
            img (torch.Tensor): Input tokens (N, D) where D=1024 for 4x16x16 patches
            modes (List[str]): Modalities for each sequence in batch
            paths (List[str]): File paths for each sequence
            cu_seqlens (torch.Tensor): Cumulative sequence lengths (B+1)
            sizes (torch.Tensor, optional): Sizes for each sequence (B, 3) for D, H, W
        
        Returns:
            torch.Tensor: Normalized tokens
        """
        B = len(modes)
        device = img.device

        if img.dtype == torch.uint8:
            # 1) Divide by 255 (uint8 -> float)
            img = img / 255.0
        
        # 2) Mean/std normalization
        batch_means, batch_stds = self.get_normalization_params(modes, paths, device)
        
        for b in range(B):
            start_idx = cu_seqlens[b]
            end_idx = cu_seqlens[b + 1]
            
            img[start_idx:end_idx] = (img[start_idx:end_idx] - batch_means[b]) / batch_stds[b]
        
        return img

