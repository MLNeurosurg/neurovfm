"""
Diagnostic Head Pipeline for NeuroVFM

Loads trained diagnostic head (pooler + classifier) for multi-label classification.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

from neurovfm.models import ABMIL, AdditiveMIL


class DiagnosticHead:
    """
    Diagnostic head for multi-label classification on token-level embeddings.
    
    Pools token embeddings to study-level and applies classification head.
    
    Args:
        pooler (nn.Module): MIL pooling module (ABMIL or AdditiveMIL)
        classifier (nn.Module): Classification head (projection to label space)
        label_names (List[str]): List of label names
        device (str): Device to run inference on
        threshold (float): Prediction threshold for binary classification. Defaults to 0.5.
    
    Example:
        >>> dx_head = load_diagnostic_head("mlinslab/neurovfm-ctbleed-classifier")
        >>> encoder, preproc = load_encoder("mlinslab/neurovfm-encoder")
        >>> vols = preproc.load_study("/path/to/study/", modality="ct")
        >>> embs = encoder.embed(vols)
        >>> preds = dx_head.predict(embs, vols)  # [(label, prob, pred), ...]
    """
    
    def __init__(
        self,
        pooler: nn.Module,
        classifier: nn.Module,
        label_names: List[str],
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pooler = pooler.to(self.device).eval()
        self.classifier = classifier.to(self.device).eval()
        self.label_names = label_names
        self.threshold = threshold
        
        # Freeze models
        for param in self.pooler.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
    
    @torch.inference_mode()
    def predict(
        self,
        embeddings: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        use_amp: bool = True,
        return_probs: bool = True,
    ) -> List[Tuple[str, float, int]]:
        """
        Generate predictions from token-level embeddings.
        
        Args:
            embeddings (torch.Tensor): Token-level embeddings [N_total, D] from encoder
            batch (Dict): Batch dictionary containing:
                - study_cu_seqlens: Cumulative sequence lengths for studies
                - study_max_len: Maximum study sequence length
            use_amp (bool): Use automatic mixed precision. Defaults to True.
            return_probs (bool): Return probabilities in addition to predictions. Defaults to True.
        
        Returns:
            List[Tuple[str, float, int]]: List of (label_name, probability, prediction) tuples
        
        Example:
            >>> preds = dx_head.predict(embs, batch)
            >>> # [('hemorrhage', 0.92, 1), ('fracture', 0.13, 0), ('mass', 0.78, 1)]
        """
        # Move to device
        embeddings = embeddings.to(self.device)
        study_cu_seqlens = batch["study_cu_seqlens"].to(self.device)
        study_max_len = batch["study_max_len"]
        
        amp_dtype = torch.bfloat16 if use_amp else torch.float32
        with torch.amp.autocast(device_type=self.device, dtype=amp_dtype):
            # Pool embeddings to study-level
            pooled = self.pooler(
                embeddings,
                cu_seqlens=study_cu_seqlens,
                max_seqlen=study_max_len
            )
            
            # Get batch size (number of studies)
            num_studies = len(study_cu_seqlens) - 1
            
            # Reshape for classifier [B, D]
            pooled = pooled.view(num_studies, -1)
            
            # Apply classifier
            logits = self.classifier(pooled).squeeze(-1)  # [B, num_labels]
            
            # Apply sigmoid for multi-label
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).long()
        
        # Convert to CPU for output
        probs_cpu = probs.detach().cpu().float()
        preds_cpu = preds.detach().cpu()
        
        # Format as list of tuples: [(label, prob, pred), ...]
        # Assumes single study in batch for simplicity
        if num_studies == 1:
            results = [
                (self.label_names[i], probs_cpu[0, i].item(), preds_cpu[0, i].item())
                for i in range(len(self.label_names))
            ]
        else:
            # Multiple studies: return list of lists
            results = [
                [
                    (self.label_names[i], probs_cpu[j, i].item(), preds_cpu[j, i].item())
                    for i in range(len(self.label_names))
                ]
                for j in range(num_studies)
            ]
        
        return results
    
    def __call__(
        self,
        embeddings: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> List[Tuple[str, float, int]]:
        """Alias for predict()"""
        return self.predict(embeddings, batch)


def load_diagnostic_head(
    model_name_or_path: str,
    device: Optional[str] = None,
) -> DiagnosticHead:
    """
    Load pretrained diagnostic head from HuggingFace Hub.
    
    Args:
        model_name_or_path (str): HuggingFace model ID (e.g., "mlinslab/neurovfm-ctbleed-classifier")
                                   or local path to checkpoint
        device (str, optional): Device to load model on
    
    Returns:
        DiagnosticHead: Diagnostic head pipeline
    
    Example:
        >>> dx_head = load_diagnostic_head("mlinslab/neurovfm-ctbleed-classifier")
        >>> encoder, preproc = load_encoder("mlinslab/neurovfm-encoder")
        >>> vols = preproc.load_study("/path/to/study/", modality="ct")
        >>> embs = encoder.embed(vols)
        >>> preds = dx_head.predict(embs, vols)
    """
    from huggingface_hub import hf_hub_download
    import json
    
    logging.info(f"Loading diagnostic head from {model_name_or_path}")
    
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
    
    # Build pooler
    pooler_type = config["pooler"]["type"]
    pooler_params = config["pooler"]["params"]
    
    if pooler_type == "abmil":
        pooler = ABMIL(**pooler_params)
    elif pooler_type == "additive_mil":
        pooler = AdditiveMIL(**pooler_params)
    else:
        raise ValueError(f"Unknown pooler type: {pooler_type}")
    
    # Build classifier (simple linear projection)
    classifier_params = config["classifier"]
    classifier = nn.Linear(
        classifier_params["in_features"],
        classifier_params["num_labels"]
    )
    
    # Load weights
    state_dict = torch.load(weights_path, map_location="cpu")
    
    # Handle Lightning checkpoint format
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # Split into pooler and classifier weights
    pooler_state = {}
    classifier_state = {}
    
    for k, v in state_dict.items():
        # Remove common prefixes
        k = k.replace("model.", "")
        k = k.replace("classifier.", "")
        
        if "pooler" in k:
            new_k = k.replace("pooler.", "")
            pooler_state[new_k] = v
        elif "proj" in k or "head" in k or "fc" in k:
            new_k = k.replace("proj.", "").replace("head.", "").replace("fc.", "")
            classifier_state[new_k] = v
    
    pooler.load_state_dict(pooler_state, strict=False)
    classifier.load_state_dict(classifier_state, strict=False)
    
    logging.info(f"Loaded diagnostic head weights from {weights_path}")
    
    # Get label names and threshold from config
    label_names = config.get("label_names", [f"label_{i}" for i in range(classifier_params["num_labels"])])
    threshold = config.get("threshold", 0.5)
    
    # Create diagnostic head
    dx_head = DiagnosticHead(
        pooler=pooler,
        classifier=classifier,
        label_names=label_names,
        device=device,
        threshold=threshold
    )
    
    logging.info("Diagnostic head pipeline ready")
    return dx_head

