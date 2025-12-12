"""
Collation Functions for Medical Image Batches

Efficient batch collation for variable-length medical imaging sequences with
support for:
- Background filtering
- Random cropping
- Patch dropout
- Study-aware batching
"""

import math
import random
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from itertools import groupby
from operator import itemgetter
from multiprocessing import Value


class MultiViewCollator:
    """
    Efficient collator for series-based datasets with background filtering.
    
    Produces per-series cumulative sequence lengths for efficient attention computation.
    Compatible with both ImageDataset (single series) and StudyAwareBatchSampler
    (multiple series per study).
    
    Args:
        remove_background (bool): Whether to filter background tokens. Defaults to True.
        patch_drop_rate (float): Probability of dropping each foreground patch. Defaults to 0.0.
        apply_masks_internally (bool): If True, applies masks to tensors in collator.
                                       If False, returns mask indices. Defaults to False.
    
    Example:
        >>> collator = MultiViewCollator(remove_background=True, patch_drop_rate=0.1)
        >>> batch = collator(batch_list)
        >>> img = batch['img']  # [N, 1024] tokenized images
        >>> series_cu_seqlens = batch['series_cu_seqlens']  # [B+1] cumulative lengths
    """
    
    def __init__(
        self,
        remove_background: bool = True,
        patch_drop_rate: float = 0.0,
        apply_masks_internally: bool = False,
    ):
        self.remove_background = remove_background
        self.patch_drop_rate = patch_drop_rate
        self.apply_masks_internally = apply_masks_internally

    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of series/images.
        
        Args:
            batch (List[Dict]): List of samples from dataset, each with keys:
                - 'img': [N, 1024] tokenized image
                - 'coords': [N, 3] token coordinates
                - 'filtered': [N] background mask
                - 'size': [3] volume dimensions (depth, height, width)
                - 'label': scalar or array
                - 'study': study identifier
                - 'path': file path
                - 'mode': 'mri' or 'ct'
        
        Returns:
            Dict with keys:
                - 'img': [N_total, 1024] concatenated images
                - 'coords': [N_total, 3] concatenated coordinates
                - 'filtered': [N_total] concatenated masks
                - 'study': List[str] study identifiers
                - 'label': [B] labels
                - 'size': [B, 3] sizes
                - 'path': List[str] paths
                - 'mode': List[str] modalities
                - 'series_masks_indices': [N_kept] indices of kept tokens (if not applied)
                - 'series_cu_seqlens': [num_series+1] cumulative sequence lengths
                - 'series_max_len': int, max series length
                - 'study_cu_seqlens': [B+1] cumulative study lengths
                - 'study_max_len': int, max study length
        """
        # Filter invalid samples
        batch = [item for item in batch if item is not None and item.get("img") is not None]
        
        # Handle empty batch (DDP edge case)
        if not batch:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return self._empty_batch(device)

        # Re-group flat batch by study_id
        study_groups = [list(group) for _, group in groupby(batch, key=itemgetter('study'))]

        # Prepare data
        study_imgs, study_coords, study_filters, study_sizes, study_labels = [], [], [], [], []
        series_paths, series_modes, study_ids = [], [], []
        
        for study_series in study_groups:
            study_imgs.append(torch.cat([s["img"] for s in study_series]))
            study_coords.append(torch.cat([s["coords"] for s in study_series]))
            study_filters.append(torch.cat([s["filtered"] for s in study_series]))
            study_sizes.append(torch.stack([s["size"] for s in study_series]))
            study_labels.append(study_series[0]['label'])
            study_ids.append(study_series[0]['study'])
            
            series_paths.extend([s["path"] for s in study_series])
            series_modes.extend([s["mode"] for s in study_series])

        B = len(study_labels)
        device = study_imgs[0].device

        # Concatenate into single tensors
        collated_batch = {
            'img': torch.cat(study_imgs, dim=0),
            'coords': torch.cat(study_coords, dim=0),
            'filtered': torch.cat(study_filters, dim=0),
            'path': series_paths,
            'mode': series_modes,
            'size': torch.cat(study_sizes, dim=0),
            'study': study_ids,
        }
        
        # Handle labels
        if isinstance(study_labels[0], (list, tuple)):
            collated_batch["label"] = torch.stack([torch.tensor(l) for l in study_labels]).to(dtype=torch.float, device=device)
        else:
            collated_batch["label"] = torch.tensor(study_labels, dtype=torch.float, device=device)

        if self.remove_background:
            collated_masks_indices, series_seqlens, study_seqlens = [], [], []
            global_token_offset = 0

            for study_idx in range(B):
                study_series_sizes = study_sizes[study_idx]
                study_total_tokens = sum(s.prod().item() for s in study_series_sizes)
                study_filtered_mask = collated_batch['filtered'][global_token_offset : global_token_offset + study_total_tokens]
                study_kept_tokens, series_token_offset = 0, 0

                for series_idx in range(len(study_series_sizes)):
                    series_total_tokens = study_series_sizes[series_idx].prod().item()
                    series_filt = study_filtered_mask[series_token_offset : series_token_offset + series_total_tokens]
                    not_filt = ~series_filt.to(torch.bool)
                    
                    if self.patch_drop_rate > 0:
                        keep_probs = torch.full_like(not_filt, 1.0 - self.patch_drop_rate, dtype=torch.float)
                        final_keep_mask_1d = not_filt & torch.bernoulli(keep_probs).to(torch.bool)
                    else:
                        final_keep_mask_1d = not_filt

                    relative_kept_indices = torch.where(final_keep_mask_1d)[0]
                    series_kept_tokens = len(relative_kept_indices)

                    series_seqlens.append(series_kept_tokens)
                    collated_masks_indices.append(global_token_offset + series_token_offset + relative_kept_indices)

                    study_kept_tokens += series_kept_tokens
                    series_token_offset += series_total_tokens
                
                study_seqlens.append(study_kept_tokens)
                global_token_offset += study_total_tokens

            # Construct final mask tensors
            final_indices = torch.cat(collated_masks_indices) if collated_masks_indices else torch.tensor([], dtype=torch.long, device=device)

            series_seqlens_tensor = torch.tensor(series_seqlens, device=device, dtype=torch.int32)
            cu_seqlens = torch.cat([torch.tensor([0], device=device, dtype=torch.int32), torch.cumsum(series_seqlens_tensor, dim=0)]).to(torch.int32)
            max_serie_len = series_seqlens_tensor.max().item() if len(series_seqlens) > 0 else 0
            
            study_seqlens_tensor = torch.tensor(study_seqlens, device=device, dtype=torch.int32)
            cu_study_seqlens = torch.cat([torch.tensor([0], device=device, dtype=torch.int32), torch.cumsum(study_seqlens_tensor, dim=0)]).to(torch.int32)
            max_study_len = study_seqlens_tensor.max().item() if len(study_seqlens) > 0 else 0
            
            if self.apply_masks_internally:
                collated_batch["img"] = collated_batch["img"][final_indices]
                collated_batch["coords"] = collated_batch["coords"][final_indices]
                collated_batch["filtered"] = collated_batch["filtered"][final_indices]
                collated_batch['series_masks_indices'] = torch.tensor([], dtype=torch.long, device=device)
            else:
                collated_batch['series_masks_indices'] = final_indices

            collated_batch['series_cu_seqlens'] = cu_seqlens
            collated_batch['series_max_len'] = max_serie_len
            collated_batch['study_cu_seqlens'] = cu_study_seqlens
            collated_batch['study_max_len'] = max_study_len
        else:
            raise NotImplementedError("remove_background=False not implemented")

        return collated_batch
    
    def _empty_batch(self, device: torch.device) -> Dict:
        """Create empty batch for DDP edge cases."""
        return {
            "img": torch.tensor([], device=device),
            "coords": torch.tensor([], device=device),
            "label": torch.tensor([], device=device),
            "filtered": torch.tensor([], device=device),
            "series_masks_indices": torch.tensor([], dtype=torch.long, device=device),
            "series_cu_seqlens": torch.tensor([0], dtype=torch.int32, device=device),
            "series_max_len": 0,
            "study_cu_seqlens": torch.tensor([0], dtype=torch.int32, device=device),
            "study_max_len": 0,
        }


# =================================================================================================
# JEPA-style Mask Generation Collator (Vol-JEPA Pretraining)
# =================================================================================================

def serie_collate_fn(batch):
    """
    Basic collation function for series/image data.
    
    Args:
        batch (List[Dict]): List of samples from dataset
    
    Returns:
        Dict: Collated batch with concatenated tensors
    """
    if not any([isinstance(data["study"], list) for data in batch]):
        if "img" in batch[0]:
            out = {
                "img": torch.cat([data["img"] for data in batch if data["img"] is not None], dim=0),
                "coords": torch.cat([data["coords"] for data in batch if data["coords"] is not None], dim=0),
                "filtered": torch.cat([data["filtered"] for data in batch if data["filtered"] is not None], dim=0),
                "size": torch.stack([data["size"] for data in batch if data["size"] is not None], dim=0),
                "path": [data["path"] for data in batch if data["path"] is not None],
                "label": [data["label"] for data in batch if data["label"] is not None],
                "study": [data["study"] for data in batch if data["study"] is not None],
                "mode": [data["mode"] for data in batch if data["mode"] is not None],
            }
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    
    if isinstance(out["label"][0], int):
        out["label"] = torch.tensor(out["label"]).half()
    else:
        out["label"] = torch.vstack([torch.tensor(l) for l in out["label"]]).half()
    return out


class MultiViewBaseCollator:
    """
    Base collator class with iteration counter for mask generation.
    
    Args:
        patch_size (Tuple[int, int, int]): Patch dimensions (D, H, W). Defaults to (4, 16, 16).
        min_keep (int): Minimum number of tokens to keep. Defaults to 4.
    """
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (4, 16, 16),
        min_keep: int = 4,
    ):
        super(MultiViewBaseCollator, self).__init__()
        self.patch_size = patch_size
        self.min_keep = min_keep
        self._itr_counter = Value('i', -1)  # Shared across worker processes

    def step(self):
        """Increment and return iteration counter (thread-safe)."""
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch: Dict):
        raise NotImplementedError()


class _MultiBlockCollator(MultiViewBaseCollator):
    """
    Internal mask generator for JEPA-style pretraining.
    
    Generates context (encoder) and target (predictor) masks for masked prediction tasks.
    Supports modality-specific masking strategies and background filtering.
    
    Args:
        hw_pred_mask_scale (List[Dict[str, Tuple[float, float]]]): HW mask scales per modality
        d_pred_mask_scale (Tuple[float, float]): Depth mask scale range
        enc_mask_scale (Dict[str, float]): Encoder mask scale per modality
        drop_rate (float): Token dropout rate. Defaults to 0.0.
        aspect_ratio (List[Tuple[float, float]]): Aspect ratio ranges for blocks
        npred (List[int]): Number of predicted blocks
        max_depth_scale (float): Maximum depth scaling. Defaults to 1.0.
        remove_background (bool): Whether to filter background. Defaults to False.
        min_filt_ratio (float): Minimum filter ratio. Defaults to 0.0.
        switch_enc_pred (Tuple[bool]): Whether to swap encoder/predictor masks
        init_ratio (float): Initial mask overlap ratio. Defaults to 0.95.
    """
    def __init__(
        self,
        hw_pred_mask_scale: List[Dict[str, Tuple[float, float]]] = [{"mri": [0.7, 0.7], "ct": [0.75, 0.75]}, {"mri": [0.25, 0.25], "ct": [0.2, 0.2]}],
        d_pred_mask_scale: Tuple[float, float] = (1., 1.),
        enc_mask_scale: Dict[str, float] = {"mri": 0.25, "ct": 0.2},
        drop_rate: float = 0.,
        aspect_ratio: List[Tuple[float, float]] = [(0.3, 3.0)],
        max_depth_scale: float = 1.,
        npred: List[int] = [1],
        remove_background: bool = False,
        min_filt_ratio: float = 0.,
        switch_enc_pred: Tuple[bool] = [False],
        init_ratio: float = 0.95,
        **kwargs
    ):
        super(_MultiBlockCollator, self).__init__(**kwargs)

        self.d_pred_mask_scale = d_pred_mask_scale
        self.enc_mask_scale = enc_mask_scale
        self.drop_rate = drop_rate
        self.max_depth_scale = max_depth_scale
        self.remove_background = remove_background
        self.min_filt_ratio = min_filt_ratio
        self.init_ratio = init_ratio

        self.cfgs = [{
            "hw_pred_mask_scale": e1,
            "aspect_ratio": e2,
            "npred": e3,
            "switch_enc_pred": e4,
        } for e1, e2, e3, e4 in zip(hw_pred_mask_scale, aspect_ratio, npred, switch_enc_pred)]

        self.min_keep = 4

    def _sample_block_size(
        self, 
        generator, 
        d_scale,
        hw_scale,
        aspect_ratio_scale,
        depth: int = None,
    ):
        """Sample block dimensions for masking."""
        if depth is not None:
            d = depth
        else:
            d = self.depth

        # Sample spatial block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = hw_scale
        hw_scale = min_s + _rand * (max_s - min_s)
        hw_num_keep = int(self.height * self.width * hw_scale)

        # Sample block aspect-ratio
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # Compute block height and width
        h = int(round(math.sqrt(hw_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(hw_num_keep / aspect_ratio)))
        h = max(1, min(h, self.height))
        w = max(1, min(w, self.width))

        return (d, h, w)

    def _sample_block_mask(self, b_size, mask_filt=None):
        """Sample a block mask with optional background filtering."""
        d, h, w = b_size

        mask = None
        mask_is_not_ok = True
        attempts_until_decrease_ratio = 0
        _ratio = self.init_ratio

        while mask_is_not_ok:
            top = torch.randint(0, self.height - h + 1, (1,))
            left = torch.randint(0, self.width - w + 1, (1,))
    
            mask = torch.ones((self.depth, self.height, self.width), dtype=torch.int32)
            front = 0
                
            if (mask_filt is None) or (_ratio < 0.1) or (not mask_filt[front:front+d, top:top+h, left:left+w].sum().item() < mask_filt.sum().item() * _ratio):
                mask[front:front+d, top:top+h, left:left+w] = 0
                mask_is_not_ok = False

            attempts_until_decrease_ratio += 1
            if attempts_until_decrease_ratio >= 5:
                attempts_until_decrease_ratio = 0
                _ratio -= 0.05

        return mask

    def _generate_masks(self, p_size, filt_vol, filt_ratio, mode, MIN_ENC_FRAC, MAX_ENC_FRAC, MAX_N_ATTEMPTS):
        """Generate encoder and predictor masks with retry logic."""
        num_attempts = 0
        while num_attempts < MAX_N_ATTEMPTS:
            # Create mask tensors
            mask_e = torch.ones((self.depth, self.height, self.width), dtype=torch.int32)
            mask_p_init = torch.nonzero(mask_e.flatten()).squeeze()
            
            # Create filter mask if needed
            if filt_ratio >= self.min_filt_ratio and self.remove_background:
                mask_filt = ~(filt_vol)
            else:
                mask_filt = None
            
            # Build context mask
            for _ in range(self.npred):
                block_mask = self._sample_block_mask(p_size, mask_filt=mask_filt)
                mask_e *= block_mask
                del block_mask

            filtered_idx = torch.where(filt_vol.flatten())[0]

            if self.switch_enc_pred:
                mask_e = (~mask_e.bool()).int()

            # Drop along depth axis to reach context mask scale
            mask_e_tmp = torch.nonzero(mask_e.flatten())
            curr_frac = mask_e_tmp[~torch.isin(mask_e_tmp, filtered_idx)].numel() / (mask_filt.sum().item() if mask_filt is not None else (self.depth * self.width * self.height))
            if curr_frac > self.enc_mask_scale[mode]:
                d_drop_scale = 1 - (self.enc_mask_scale[mode] / curr_frac)

                n_dad = math.floor(d_drop_scale * self.depth)
                offset_dad = torch.randint(0, self.depth - n_dad + 1, (1,))
                mask_e[offset_dad:offset_dad+n_dad] = 0
            del mask_e_tmp

            # Process masks and drop % of the context mask
            mask_e_flat = mask_e.flatten() 
            mask_e_flat = mask_e_flat * torch.bernoulli(torch.ones(mask_e_flat.shape) * (1-self.drop_rate))

            mask_p = torch.where(mask_e_flat == 0)[0]
            mask_e = torch.where(mask_e_flat == 1)[0]
            del mask_e_flat
            
            # Filter background
            if filt_ratio >= self.min_filt_ratio and self.remove_background:
                mask_p_init = mask_p_init[~torch.isin(mask_p_init, filtered_idx)]
                mask_p = mask_p[~torch.isin(mask_p, filtered_idx)]
                mask_e = mask_e[~torch.isin(mask_e, filtered_idx)]
            del filtered_idx
            
            # Check if valid
            if (mask_e.numel() == 0 or mask_p.numel() == 0 or mask_p_init.numel() == 0 or
                ((mask_e.numel() / mask_p_init.numel() > MAX_ENC_FRAC) or 
                 (mask_e.numel() / mask_p_init.numel() < MIN_ENC_FRAC))):
                num_attempts += 1
                continue
            
            # Swap if needed
            if mask_e.numel() / mask_p_init.numel() > MAX_ENC_FRAC:
                mask_e, mask_p = mask_p, mask_e
            
            return mask_p, mask_e, mask_p_init
        
        # If we failed after max attempts, return None
        return None, None, None

    def __call__(self, B, sizes, filtered, modes, viz: bool=False):
        """
        Generate masks for a batch.
        
        Args:
            B (int): Batch size
            sizes (torch.Tensor): Volume sizes (B, 3)
            filtered (torch.Tensor): Background mask (concatenated)
            modes (List[str]): Modalities for each sample
            viz (bool): Visualization mode. Defaults to False.
        
        Returns:
            Tuple: (pred_masks, pred_init_masks, enc_masks) each as lists of length B
        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        MIN_ENC_FRAC, MAX_ENC_FRAC = 0.05, 0.5
        MAX_N_ATTEMPTS = 5
        
        with torch.no_grad():
            collated_masks_pred = []
            collated_masks_enc = []
            collated_masks_pred_init = []

            offset = 0
            for b_idx in range(B):
                cfg_idx = torch.randint(len(self.cfgs), (1,), generator=g).item()
                
                self.hw_pred_mask_scale = self.cfgs[cfg_idx]["hw_pred_mask_scale"][modes[b_idx]]
                self.aspect_ratio = self.cfgs[cfg_idx]["aspect_ratio"]
                self.npred = self.cfgs[cfg_idx]["npred"]
                self.switch_enc_pred = self.cfgs[cfg_idx]["switch_enc_pred"]

                original_size = sizes[b_idx].tolist()
                seqlen = sizes[b_idx].prod().item()
                
                # Reshape filtered once per batch
                filt_vol_full = filtered[offset:offset+seqlen].reshape(original_size[0], 
                                                                    original_size[1], 
                                                                    original_size[2])
                
                self.depth, self.height, self.width = original_size
                filt_vol = filt_vol_full
                filt_ratio = 1 - (filt_vol.sum().item() / seqlen)

                # Sample block size
                hw_pred_mask_rescale = [v*filt_ratio for v in self.hw_pred_mask_scale]
                p_size = self._sample_block_size(
                    generator=g, d_scale=self.d_pred_mask_scale, hw_scale=hw_pred_mask_rescale,
                    aspect_ratio_scale=self.aspect_ratio, depth=self.depth
                )

                # Generate masks with retry
                mask_p, mask_e, mask_p_init = self._generate_masks(
                    p_size, filt_vol, filt_ratio, modes[b_idx], MIN_ENC_FRAC, MAX_ENC_FRAC, MAX_N_ATTEMPTS
                )
                
                if mask_p is not None and mask_e is not None and mask_p_init is not None:
                    # Map to original volume efficiently
                    original_vol = torch.arange(seqlen).reshape(original_size)
                    region = original_vol.flatten()
                    
                    # Map masks
                    mask_p = region[mask_p]
                    mask_p_init = region[mask_p_init]
                    mask_e = region[mask_e]

                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)
                    collated_masks_pred_init.append(mask_p_init)
                    
                    del original_vol, region
                else:
                    # Default masks (fallback)
                    collated_masks_pred.append(torch.tensor([0]))
                    collated_masks_enc.append(torch.tensor([0]))
                    collated_masks_pred_init.append(torch.tensor([0]))
                
                # Update offset for next batch
                offset += seqlen
                del filt_vol, filt_vol_full
                
            # Return results
            return (
                collated_masks_pred,
                collated_masks_pred_init,
                collated_masks_enc,
            )


class MultiBlockCollator:
    """
    JEPA-style collator with multi-block mask generation for Vol-JEPA pretraining.
    
    Wraps _MultiBlockCollator to provide a complete pretraining collator that generates
    encoder/context and predictor/target masks for self-supervised learning.
    
    Args:
        cfgs_mask (List[Dict[str, Any]]): List of mask generator configurations (only 1 supported)
        apply_masks_internally (bool): If True, applies masks in collator. Defaults to False.
        **kwargs: Additional arguments passed to _MultiBlockCollator
    
    Example:
        >>> mask_cfg = {
        ...     "hw_pred_mask_scale": [{"mri": [0.7, 0.7], "ct": [0.75, 0.75]}],
        ...     "d_pred_mask_scale": (1., 1.),
        ...     "enc_mask_scale": {"mri": 0.25, "ct": 0.2},
        ...     "drop_rate": 0.0,
        ...     "aspect_ratio": [(0.3, 3.0)],
        ...     "npred": [1],
        ...     "remove_background": True,
        ... }
        >>> collator = MultiBlockCollator(cfgs_mask=[mask_cfg])
        >>> udata, pred, pred_init, enc = collator(batch)
    """
    def __init__(
        self, 
        cfgs_mask: List[Dict[str, Any]],
        apply_masks_internally: bool = False,
        **kwargs
    ):
        super(MultiBlockCollator, self).__init__(**kwargs)
        assert len(cfgs_mask) == 1, "Only one mask generator is supported"
        self.mask_generators = []
        for m in cfgs_mask:
            mask_generator = _MultiBlockCollator(
                hw_pred_mask_scale=m["hw_pred_mask_scale"],
                d_pred_mask_scale=m["d_pred_mask_scale"],
                enc_mask_scale=m["enc_mask_scale"],
                drop_rate=m["drop_rate"],
                aspect_ratio=m["aspect_ratio"],
                npred=m["npred"],
                max_depth_scale=m.get("max_depth_scale", 1.),
                remove_background=m.get("remove_background", False),
                min_filt_ratio=m.get("min_filt_ratio", 0.),
                switch_enc_pred=m.get("switch_enc_pred", False)
            )
            self.mask_generators.append(mask_generator)

        self.apply_masks_internally = apply_masks_internally

    def step(self):
        """Step all mask generators."""
        for mask_generator in self.mask_generators:
            mask_generator.step()

    def __call__(self, batch):
        """
        Collate batch and generate masks.
        
        Args:
            batch (List[Dict]): List of samples from dataset
        
        Returns:
            Tuple: (collated_batch, pred_masks, pred_init_masks, enc_masks)
                - collated_batch (Dict): Basic collated data
                - pred_masks (List[Tuple]): [(indices, cu_seqlens, max_seqlen)] for predictor targets
                - pred_init_masks (List[Tuple]): [(indices, cu_seqlens, max_seqlen)] for full context
                - enc_masks (List[Tuple]): [(indices, cu_seqlens, max_seqlen)] for encoder context
        """
        with torch.no_grad():
            collated_batch = serie_collate_fn(batch)
            B = len(collated_batch["label"])
            
            # Initialize empty lists
            collated_masks_pred, collated_masks_pred_init, collated_masks_enc = [], [], []
            
            # Process mask generators
            for mask_generator in self.mask_generators:
                out_tuple = mask_generator(B, collated_batch["size"], collated_batch["filtered"], collated_batch["mode"], viz=False)
                collated_masks_pred.append(out_tuple[0])
                collated_masks_pred_init.append(out_tuple[1])
                collated_masks_enc.append(out_tuple[2])
            
            # Bucketize masks (map to pred_init space)
            bucketized_pred = [
                [torch.bucketize(collated_masks_pred[g_idx][b_idx], collated_masks_pred_init[g_idx][b_idx]) 
                 for b_idx in range(len(collated_masks_pred[g_idx]))]
                for g_idx in range(len(self.mask_generators))
            ]
            
            bucketized_enc = [
                [torch.bucketize(collated_masks_enc[g_idx][b_idx], collated_masks_pred_init[g_idx][b_idx]) 
                 for b_idx in range(len(collated_masks_enc[g_idx]))]
                for g_idx in range(len(self.mask_generators))
            ]
            
            # Create result dict
            _dict = {
                "pred": [[[], [0], 0] for _ in range(len(self.mask_generators))],
                "pred_init": [[[], [0], 0] for _ in range(len(self.mask_generators))],
                "enc": [[[], [0], 0] for _ in range(len(self.mask_generators))],
            }
            
            # Process each generator and batch
            for g_idx in range(len(self.mask_generators)):
                total_offset = 0
                adj_offset = 0
                
                for b_idx in range(len(collated_masks_pred[g_idx])):
                    # Process pred masks
                    pred_seqlen = bucketized_pred[g_idx][b_idx].size(0)
                    _dict["pred"][g_idx][0].append(bucketized_pred[g_idx][b_idx] + adj_offset)
                    _dict["pred"][g_idx][1].append(_dict["pred"][g_idx][1][-1]+pred_seqlen)
                    _dict["pred"][g_idx][2] = max(_dict["pred"][g_idx][2], pred_seqlen)
                    
                    # Process pred_init masks
                    pred_init_seqlen = collated_masks_pred_init[g_idx][b_idx].size(0)
                    _dict["pred_init"][g_idx][0].append(collated_masks_pred_init[g_idx][b_idx] + total_offset)
                    _dict["pred_init"][g_idx][1].append(_dict["pred_init"][g_idx][1][-1]+pred_init_seqlen)
                    _dict["pred_init"][g_idx][2] = max(_dict["pred_init"][g_idx][2], pred_init_seqlen)
                    
                    # Process enc masks
                    enc_seqlen = bucketized_enc[g_idx][b_idx].size(0)
                    _dict["enc"][g_idx][0].append(bucketized_enc[g_idx][b_idx] + adj_offset)
                    _dict["enc"][g_idx][1].append(_dict["enc"][g_idx][1][-1]+enc_seqlen)
                    _dict["enc"][g_idx][2] = max(_dict["enc"][g_idx][2], enc_seqlen)
                    
                    # Update offsets
                    seqlen = collated_batch["size"][b_idx].prod().item()
                    total_offset += seqlen
                    adj_offset += pred_init_seqlen
                
                # Concatenate tensors
                for k in ["pred", "pred_init", "enc"]:
                    if _dict[k][g_idx][0]:
                        _dict[k][g_idx][0] = torch.cat(_dict[k][g_idx][0])
                        _dict[k][g_idx][1] = torch.tensor(_dict[k][g_idx][1], dtype=torch.int32)

            if self.apply_masks_internally:
                collated_batch["img"] = collated_batch["img"][_dict["pred_init"][0][0]]
                collated_batch["coords"] = collated_batch["coords"][_dict["pred_init"][0][0]]
                collated_batch["filtered"] = collated_batch["filtered"][_dict["pred_init"][0][0]]
                _dict["pred_init"][0][0] = None
            
            # Create result tuple
            result = (
                collated_batch,
                _dict["pred"],
                _dict["pred_init"],
                _dict["enc"],
            )
            
            return result

