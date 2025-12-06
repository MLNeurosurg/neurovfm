"""
Report Generation Pipeline for NeuroVFM

Loads a trained NeuroVFM model and provides a high-level interface for
generating radiology reports from medical imaging studies.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import json
import os

from huggingface_hub import hf_hub_download, snapshot_download

from neurovfm.models.vlm import VisionLanguageModel
from neurovfm.pipelines.preprocessor import StudyPreprocessor
from neurovfm.data.text import process_text
from neurovfm.systems.utils import NormalizationModule


class FindingsGenerationPipeline:
    """
    Pipeline for generating radiological findings from medical images using NeuroVFM.
    
    Args:
        model (VisionLanguageModel): The loaded multimodal LLM.
        preprocessor (StudyPreprocessor): The study preprocessor.
        normalization_module (NormalizationModule): Module for normalizing image tokens.
        device (str): Device to run inference on. Defaults to "cuda" if available.
    """
    
    def __init__(
        self,
        model: VisionLanguageModel,
        preprocessor: StudyPreprocessor,
        normalization_module: NormalizationModule,
        device: Optional[str] = None,
    ):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.preprocessor = preprocessor
        self.norm_module = normalization_module.to(self.device)
        self.tokenizer = self.model.language_model.tokenizer

        self.SYS_PROMPT = "You are an expert neuro-radiologist AI assistant. Analyze the provided neuroimaging study and answer the user's request. /no_think"
        self.USER_PROMPT_START = "Generate a concise report of the key positive findings for this study."
        self.USER_PROMPT_END = "\nFormat your response as JSON with the following keys: 'exam_type' and 'findings'."
        self.GENERATE_KWARGS = {
            'max_new_tokens': 512,
            'do_sample': False,
            'num_beams': 4,
            'length_penalty': 1.0,
            'repetition_penalty': 1.2,
            'no_repeat_ngram_size': 4,
            'min_new_tokens': 3,
            'generation_schema': 'shortreport'
        }

        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
    @torch.inference_mode()
    def generate(
        self,
        batch: Dict[str, torch.Tensor],
        clinical_context: str = None,
        **kwargs
        ) -> str:
        """
        Generate a report for a given study.
        
        Args:
            batch (Dict): Batch dictionary from StudyPreprocessor containing:
                - img: Tokens [N_total, 1024]
                - coords: Coordinates [N_total, 3]
                - series_masks_indices: Optional mask indices
                - series_cu_seqlens: Cumulative sequence lengths for series
                - series_max_len: Maximum sequence length
                - mode: List of modalities
                - path: List of file paths
            clinical_context (str): Optional clinical context to inform the report generation.
        
        Returns:
            str: The generated report text.
        """

        # 1. Prepare visual input
        
        # Move relevant keys to device
        vision_batch = batch.copy()
        vision_batch["img"] = vision_batch["img"].to(self.device)
        vision_batch["coords"] = vision_batch["coords"].to(self.device)
        vision_batch["series_cu_seqlens"] = vision_batch["series_cu_seqlens"].to(self.device)
        vision_batch["study_cu_seqlens"] = vision_batch["study_cu_seqlens"].to(self.device)
        if vision_batch.get("series_masks_indices") is not None and vision_batch["series_masks_indices"].numel() > 0:
            vision_batch["series_masks_indices"] = vision_batch["series_masks_indices"].to(self.device)
        else:
            vision_batch["series_masks_indices"] = None

        # Normalize
        vision_batch["img"] = self.norm_module.normalize(
            vision_batch["img"],
            vision_batch["mode"],
            vision_batch["path"],
            cu_seqlens=vision_batch["series_cu_seqlens"],
            sizes=vision_batch.get("size")
        )

        # 2. Prepare text input

        # Construct user prompt
        prompt = self.USER_PROMPT_START
        if clinical_context:
            prompt += f"\nConsider the patient's clinical indication in your analysis: {clinical_context}"
        prompt += self.USER_PROMPT_END

        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ]

        # Tokenize
        processed_text = process_text(
            conversation=conversation,
            tokenizer=self.tokenizer,
            max_seq_len=2048,
            system_prompt=self.SYS_PROMPT, 
            image_placeholder_token_id=self.model.language_model.image_placeholder_token_id,
            n_images=len(vision_batch["series_cu_seqlens"])-1,
        )
        input_ids = torch.tensor([processed_text['input_ids']], device=self.device)
        attention_mask = torch.tensor([processed_text['attention_mask']], device=self.device)
        labels = torch.full_like(input_ids, -100) # irrelevant for generation

        # Prepare inputs for generation (left-padding, extract prompt)
        input_ids, attention_mask, _, _ = self.model._prepare_inputs_for_generation(
            input_ids, 
            attention_mask, 
            labels
        )

        # 3. Generate

        with torch.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
            gen_ids = self.model.generate(
                vision_batch=vision_batch,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self.GENERATE_KWARGS
            )
        decoded_text = self.tokenizer.decode(gen_ids.squeeze(0), skip_special_tokens=True).strip()

        # Parse JSON
        try:
            report_json = json.loads(decoded_text)
            exam_type = report_json["exam_type"] # str
            findings = report_json["findings"] # list of str
            
            report_lines = ["Detected Study Type:", f"{exam_type}", "", "Findings:"]
            if isinstance(findings, list) and findings:
                for idx, finding in enumerate(findings, 1):
                    report_lines.append(f"{idx}. {finding}")
            elif isinstance(findings, str) and findings:
                report_lines.append(f"1. {findings}")
            else:
                report_lines.append("The study is unremarkable.")
            report = "\n".join(report_lines)

            return report
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse JSON from generated text: {decoded_text}")
            return decoded_text
        
    def __call__(self, batch: Dict[str, torch.Tensor], clinical_context: str = None, **kwargs):
        """Alias for generate"""
        return self.generate(batch, clinical_context, **kwargs)


def load_vlm(
    model_name_or_path: str,
    device: Optional[str] = None,
) -> Tuple[FindingsGenerationPipeline, StudyPreprocessor]:
    """
    Load pretrained NeuroVFM findings generation model from a local path or HF Hub.
    
    Expected model files:
    - config.json: Model configuration
    - vision_connector.pt: Vision connector weights
    - vision_encoder.pt: Vision encoder weights
    - language_model/: Language model weights (HF model)

    Args:
        model_name_or_path (str): Path to folder or HF Hub ID.
        device (str): Device.

    Returns:
        pipeline, preprocessor
    """
    logging.info(f"Loading findings generation model from {model_name_or_path}")

    # Download config and weights from HF Hub
    if os.path.exists(model_name_or_path):
        local_dir = Path(model_name_or_path)
    else:
        local_dir = Path(snapshot_download(repo_id=model_name_or_path))

    config_path = local_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {local_dir}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load Normalization Stats
    norm_stats = config.get("normalization_stats", [
        [0.3141, 0.4139, 0.3184, 0.2719],  # means: [mri, brain, blood, bone]
        [0.2623, 0.4059, 0.3605, 0.1875]   # stds
    ])
    norm_module = NormalizationModule(custom_stats_list=norm_stats)

    # Instantiate model
    lm_path = local_dir / "language_model"
    if lm_path.exists():
        config['language_model_cf']['model_name_or_path'] = str(lm_path)
    
    model = VisionLanguageModel(
        vision_encoder_cf=config['vision_encoder_cf'],
        vision_connector_cf=config['vision_connector_cf'],
        language_model_cf=config['language_model_cf'],
        use_gradient_checkpointing=False    
    )

    # Load encoder weights
    enc_path = local_dir / "vision_encoder.pt"
    if enc_path.exists():
        logging.info(f"Loading encoder weights from {enc_path}")
        enc_state = torch.load(enc_path, map_location="cpu")
        # Handle 'state_dict' wrapper if present
        if "state_dict" in enc_state: enc_state = enc_state["state_dict"]
        model.vision_encoder.load_state_dict(enc_state, strict=False)
    else:
        logging.warning("vision_encoder.pt not found. Vision encoder initialized randomly.")

    # Load connector weights
    conn_path = local_dir / "vision_connector.pt"
    if conn_path.exists():
        logging.info(f"Loading connector weights from {conn_path}")
        conn_state = torch.load(conn_path, map_location="cpu")
        if "state_dict" in conn_state: conn_state = conn_state["state_dict"]
        model.vision_connector.load_state_dict(conn_state)
    else:
        logging.warning("vision_connector.pt not found. Connector initialized randomly.")
        
    # Create pipeline
    preprocessor = StudyPreprocessor()
    
    generator = FindingsGenerationPipeline(
        model=model, 
        preprocessor=preprocessor, 
        normalization_module=norm_module,
        device=device
    )
    
    logging.info("Findings generation pipeline ready.")
    return generator, preprocessor