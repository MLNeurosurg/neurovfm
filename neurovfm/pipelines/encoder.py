"""
Encoder Pipeline for NeuroVFM

Loads pretrained VisionTransformer encoder and generates token-level embeddings.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from neurovfm.models import get_vit_backbone
from neurovfm.systems.utils import NormalizationModule
from neurovfm.pipelines.preprocessor import StudyPreprocessor


class EncoderPipeline:
    """
    Encoder pipeline for generating token-level embeddings from medical images.
    
    Args:
        model (nn.Module): VisionTransformer encoder
        normalization_stats (List[List[float]]): Normalization statistics [[means], [stds]]
        device (str): Device to run inference on. Defaults to "cuda" if available.
    
    Example:
        >>> encoder, preproc = load_encoder("mlinslab/neurovfm-encoder")
        >>> vols = preproc.load_study("/path/to/study/", modality="ct")
        >>> embs = encoder.embed(vols)  # Token-level embeddings
    """
    
    def __init__(
        self,
        model: nn.Module,
        normalization_stats: List[List[float]],
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.norm_module = NormalizationModule(
            custom_stats_list=normalization_stats
        ).to(self.device)
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
    
    @torch.inference_mode()
    def embed(
        self,
        batch: Dict[str, torch.Tensor],
        use_amp: bool = True,
    ) -> torch.Tensor:
        """
        Generate token-level embeddings for a batch of volumes.
        
        Args:
            batch (Dict): Batch dictionary from StudyPreprocessor containing:
                - img: Tokens [N_total, 1024]
                - coords: Coordinates [N_total, 3]
                - series_masks_indices: Optional mask indices
                - series_cu_seqlens: Cumulative sequence lengths for series
                - series_max_len: Maximum sequence length
                - mode: List of modalities
                - path: List of file paths
            use_amp (bool): Use automatic mixed precision. Defaults to True.
        
        Returns:
            torch.Tensor: Token-level embeddings [N_total, D]
        """
        # Move to device
        tokens = batch["img"].to(self.device)
        coords = batch["coords"].to(self.device)
        series_cu_seqlens = batch["series_cu_seqlens"].to(self.device)
        series_max_len = batch["series_max_len"]
        
        # Handle masks
        # If preprocessing removed background tokens, series_masks_indices will be empty
        # Otherwise, it contains indices of foreground tokens to keep
        if batch.get("series_masks_indices") is not None and batch["series_masks_indices"].numel() > 0:
            masks = batch["series_masks_indices"].to(self.device)
        else:
            masks = None  # Background already removed
        
        # Normalize tokens
        tokens = self.norm_module.normalize(
            tokens,
            batch["mode"],
            batch["path"],
            cu_seqlens=series_cu_seqlens,
            sizes=batch.get("size")
        )
        
        # Forward pass
        amp_dtype = torch.bfloat16 if use_amp else torch.float32
        with torch.amp.autocast(device_type=self.device, dtype=amp_dtype):
            embs = self.model(
                tokens,
                coords,
                masks=masks,
                cu_seqlens=series_cu_seqlens,
                max_seqlen=series_max_len,
                use_flash_attn=False  # Disable for inference
            )
        
        return embs
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Alias for embed()"""
        return self.embed(batch)


def load_encoder(
    model_name_or_path: str,
    device: Optional[str] = None,
) -> Tuple[EncoderPipeline, StudyPreprocessor]:
    """
    Load pretrained encoder from HuggingFace Hub.
    
    Args:
        model_name_or_path (str): HuggingFace model ID (e.g., "mlinslab/neurovfm-encoder")
                                   or local path to checkpoint
        device (str, optional): Device to load model on
    
    Returns:
        Tuple[EncoderPipeline, StudyPreprocessor]: Encoder pipeline and preprocessor
    
    Example:
        >>> encoder, preproc = load_encoder("mlinslab/neurovfm-encoder")
        >>> vols = preproc.load_study("/path/to/study/", modality="ct")
        >>> embs = encoder.embed(vols)
    """
    from huggingface_hub import hf_hub_download
    import json
    
    logging.info(f"Loading encoder from {model_name_or_path}")
    
    # Download config and weights from HF Hub
    if Path(model_name_or_path).exists():
        # Local path
        config_path = Path(model_name_or_path) / "config.json"
        weights_path = Path(model_name_or_path) / "pytorch_model.bin"
    else:
        # HuggingFace Hub
        config_path = hf_hub_download(
            repo_id=model_name_or_path,
            filename="config.json"
        )
        weights_path = hf_hub_download(
            repo_id=model_name_or_path,
            filename="pytorch_model.bin"
        )
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Build model
    model = get_vit_backbone(**config)
    
    # Load weights
    state_dict = torch.load(weights_path, map_location="cpu")
    
    # Remove any prefixes (e.g., "model.", "student.vision_encoder.")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # Handle potential prefixes
    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     # Remove common prefixes
    #     k = k.replace("model.", "")
    #     k = k.replace("student.vision_encoder.", "")
    #     k = k.replace("vision_encoder.", "")
    #     new_state_dict[k] = v
    
    model.load_state_dict(state_dict, strict=False)
    logging.info(f"Loaded encoder weights from {weights_path}")
    
    # Get normalization stats from config
    norm_stats = config.get("normalization_stats", [
        [0.3141, 0.4139, 0.3184, 0.2719],  # means: [mri, brain, blood, bone]
        [0.2623, 0.4059, 0.3605, 0.1875]   # stds
    ])
    
    # Create encoder pipeline
    encoder = EncoderPipeline(
        model=model,
        normalization_stats=norm_stats,
        device=device
    )
    
    # Create preprocessor
    preprocessor = StudyPreprocessor()
    
    logging.info("Encoder pipeline ready")
    return encoder, preprocessor

