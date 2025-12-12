"""
Training Script for Vision-Language Model SFT

This script trains VisionLanguageModel with supervised fine-tuning.
Uses the ImageDataModule + VisionInstructionDataModule stack and FSDP.
"""

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp import ShardingStrategy, StateDictType, BackwardPrefetch, MixedPrecision

from neurovfm.train.train import setup_logging, get_num_it_per_train_ep, setup_directories, setup_checkpoints, setup_loggers
from neurovfm.datasets import ImageDataModule, VisionInstructionDataModule
from neurovfm.systems import VisionInstructionTuningSystem
from neurovfm.models import VisionLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train VLM with supervised fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def load_model_with_full_pretrained_weights(model_name_or_path: str):
    """
    Load model config and weights from local path or Hugging Face Hub with full VLM pretrained weights.

    Args:
        model_name_or_path (str): Path to local directory or Hugging Face Hub ID.

    Returns:
        model (VisionLanguageModel): Model.
    """
    from huggingface_hub import snapshot_download

    # Download config and weights
    if os.path.exists(model_name_or_path):
        # Local path
        local_dir = Path(model_name_or_path)
    else:
        # HuggingFace Hub
        local_dir = Path(snapshot_download(repo_id=model_name_or_path))

    config_path = local_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {local_dir}")
    with open(config_path, "r") as f:
        config = json.load(f)

    lm_path = local_dir / "language_model"
    if lm_path.exists():
        config["language_model_cf"]["model_name_or_path"] = str(lm_path)

    model = VisionLanguageModel(
        vision_encoder_cf=config["vision_encoder_cf"],
        vision_connector_cf=config["vision_connector_cf"],
        language_model_cf=config["language_model_cf"],
        use_gradient_checkpointing=True,
    )

    enc_path = local_dir / "vision_encoder.pt"
    if enc_path.exists():
        enc_state = torch.load(enc_path, map_location="cpu")
        if "state_dict" in enc_state:
            enc_state = enc_state["state_dict"]
        model.vision_encoder.load_state_dict(enc_state, strict=False)

    conn_path = local_dir / "vision_connector.pt"
    if conn_path.exists():
        conn_state = torch.load(conn_path, map_location="cpu")
        if "state_dict" in conn_state:
            conn_state = conn_state["state_dict"]
        model.vision_connector.load_state_dict(conn_state, strict=False)

    logging.info(f"Loaded full VLM weights from {local_dir}")

    return model


def load_model_with_backbone_pretrained_weights(backbone_model_name_or_path: str, vision_connector_cf: dict, language_model_cf: dict):
    """
    Load model config and weights from local path or Hugging Face Hub with just visual backbone pretrained weights.

    Args:
        backbone_model_name_or_path (str): Path to local directory or Hugging Face Hub ID for the backbone model.
        vision_connector_cf (dict): Vision connector config.
        language_model_cf (dict): Language model config.

    Returns:
        model (VisionLanguageModel): Model.
    """
    from huggingface_hub import hf_hub_download

    # Download config and weights
    if Path(backbone_model_name_or_path).exists():
        # Local path
        config_path = Path(backbone_model_name_or_path) / "config.json"
        weights_path = Path(backbone_model_name_or_path) / "pytorch_model.bin"
    else:
        # HuggingFace Hub
        config_path = hf_hub_download(
            repo_id=backbone_model_name_or_path,
            filename="config.json"
        )
        weights_path = hf_hub_download(
            repo_id=backbone_model_name_or_path,
            filename="pytorch_model.bin"
        )
    
    # Load config
    with open(config_path, 'r') as f:
        bb_config = json.load(f)
    
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

    # Initialize model
    model = VisionLanguageModel(
        vision_encoder_cf=bb_config,
        vision_connector_cf=vision_connector_cf,
        language_model_cf=language_model_cf,
        use_gradient_checkpointing=True,
    )

    # Load backbone weights
    model.vision_encoder.load_state_dict(state_dict, strict=False)
    logging.info(f"Initialized new VLM with backbone weights from {weights_path}")

    return model


def save_vlm_checkpoint(model: VisionLanguageModel, config: dict, output_dir: str):
    """
    Saves the model components in the following format:
    - vision_encoder.pt
    - vision_connector.pt
    - language_model/ (HuggingFace format)
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Saving standardized checkpoint to {output_dir}...")

    # 1. Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # 2. Save Connector (weights only, no prefix)
    connector_state = model.vision_connector.state_dict()
    torch.save({"state_dict": connector_state}, os.path.join(output_dir, "vision_connector.pt"))
    
    # 3. Save Vision Encoder (weights only, no prefix)
    encoder_state = model.vision_encoder.state_dict()
    torch.save({"state_dict": encoder_state}, os.path.join(output_dir, "vision_encoder.pt"))
    
    # 4. Save LLM (using save_pretrained)
    # We must call save_pretrained on the inner HF model, not the wrapper
    llm_output_dir = os.path.join(output_dir, "language_model")
    model.language_model.llm.save_pretrained(llm_output_dir, safe_serialization=True)
    model.language_model.tokenizer.save_pretrained(llm_output_dir)
    
    logging.info("Checkpoint saved.")


def get_decoder_cls(llm):
    """
    Get the decoder layer class for the LLM. Necessary for FSDP auto-wrapping.

    Args:
        llm (nn.Module): LLM model.

    Returns:
        decoder_cls (type): Decoder layer class.
    """
    
    name = llm.__class__.__name__
    try:
        if "Qwen3" in name:
            from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
            return Qwen3DecoderLayer
        if "Qwen2" in name:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
            return Qwen2DecoderLayer
        if "Llama" in name:
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer
            return LlamaDecoderLayer
    except Exception:
        pass
    for m in llm.modules():
        if "DecoderLayer" in m.__class__.__name__:
            return m.__class__
    raise ValueError(f"Could not determine decoder layer class for {name}")


def setup_fsdp_strategy(model: VisionLanguageModel, fsdp_cf: Dict[str, Any]):
    """
    Setup the FSDP strategy for the model.

    Args:
        model (VisionLanguageModel): Model.
        fsdp_cf (Dict[str, Any]): FSDP configuration.

    Returns:
        fsdp_strategy (FSDPStrategy): FSDP strategy.
    """
    
    decoder_cls = get_decoder_cls(model.language_model.llm)

    def custom_auto_wrap_policy(module, recurse, nonwrapped_numel):
        if isinstance(module, decoder_cls):
            return True
        if isinstance(module, type(model.vision_encoder)):
            return False # Do not wrap the frozen vision encoder
        return recurse

    mixed_precision_cfg = fsdp_cf.get("mixed_precision", {})
    mixed_precision = MixedPrecision(
        param_dtype=getattr(torch, mixed_precision_cfg.get("param_dtype", "bfloat16")),
        reduce_dtype=getattr(torch, mixed_precision_cfg.get("reduce_dtype", "bfloat16")),
        buffer_dtype=getattr(torch, mixed_precision_cfg.get("buffer_dtype", "bfloat16")),
    )

    return FSDPStrategy(
        auto_wrap_policy=custom_auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=getattr(ShardingStrategy, fsdp_cf.get("sharding_strategy", "FULL_SHARD")),
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        backward_prefetch=getattr(BackwardPrefetch, fsdp_cf.get("backward_prefetch", "BACKWARD_PRE")),
        forward_prefetch=fsdp_cf.get("forward_prefetch", False),
        use_orig_params=fsdp_cf.get("use_orig_params", True),
        limit_all_gathers=fsdp_cf.get("limit_all_gathers", True),
        cpu_offload=None,
    )


def train(config: OmegaConf):
    """
    Main training function.
    
    Args:
        config (OmegaConf): Configuration object
    """
    # Setup directories
    exp_root, model_dir = setup_directories(config)

    # Ensure study_conversations are passed into dataset params
    study_convs = config.data.get("study_conversations", None)
    if study_convs is not None:
        config.data.dataset.train.params["study_conversations"] = study_convs
        config.data.dataset.val.params["study_conversations"] = study_convs

    # Setup base data module
    base_dm = ImageDataModule(config)
    base_dm.setup(stage="fit")

    # Compute iteration and batch-size stats for optimizer/scheduler
    train_stats = get_num_it_per_train_ep(len(base_dm.train_dataset), config)
    training_params = {
        **train_stats,
        "num_ep_total": config.training.trainer_params.max_epochs,
    }

    # Initialize model
    if hasattr(config.training, 'load_pretrained_full') and config.training.load_pretrained_full:
        # Initialize VLM with full pretrained weights
        model_name_or_path = config.training.load_pretrained_full.model_name_or_path
        model = load_model_with_full_pretrained_weights(model_name_or_path).to(torch.bfloat16)
    elif hasattr(config.training, 'load_pretrained_backbone') and config.training.load_pretrained_backbone:
        # Iniitalize new VLM with pretrained backbone
        backbone_model_name_or_path = config.training.load_pretrained_backbone.model_name_or_path
        model = load_model_with_backbone_pretrained_weights(
            backbone_model_name_or_path, 
            config.system.params.vision_connector_cf, 
            config.system.params.language_model_cf
        ).to(torch.bfloat16)
    else:
        # Initialize VLM from scratch
        model = VisionLanguageModel(
            vision_encoder_cf=config.system.params.vision_encoder_cf,
            vision_connector_cf=config.system.params.vision_connector_cf,
            language_model_cf=config.system.params.language_model_cf,
            use_gradient_checkpointing=True,
        ).to(torch.bfloat16)
        
    # Instantiate system
    system = VisionInstructionTuningSystem(
        stage=config.system.params.stage,
        model=model,
        opt_cf=config.system.params.get("opt_cf", None),
        schd_cf=config.system.params.get("schd_cf", None),
        training_params=training_params,
    )

    # Wrap data module for text processing support
    lm_cf = config.system.params.get("language_model_cf", {})
    system_prompt = lm_cf.get("system_prompt", "You are an expert neuro-radiologist AI assistant. Analyze the provided neuroimaging study and answer the user's request. /no_think")
    max_seq_len = lm_cf.get("max_seq_len", 2048)
    tok = system.model.language_model.tokenizer
    placeholder_id = system.model.language_model.image_placeholder_token_id
    pad_token_id = tok.pad_token_id or tok.eos_token_id

    wrapped_dm = VisionInstructionDataModule(
        base_dm=base_dm,
        tokenizer=tok,
        system_prompt=system_prompt,
        max_seq_len=max_seq_len,
        placeholder_token_id=placeholder_id,
        pad_token_id=pad_token_id,
    )

    # Setup callbacks
    callbacks = setup_checkpoints(config, model_dir)
    callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step", log_momentum=False))

    if config.infra.get("log_gpu", False):
        callbacks.append(pl.callbacks.DeviceStatsMonitor(cpu_stats=True))

    # Setup loggers
    loggers = setup_loggers(config, exp_root)

    # Setup FSDP strategy
    fsdp_strategy = setup_fsdp_strategy(system.model, config.system.params.get("fsdp_cf", {}))

    # Create trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy=fsdp_strategy,
        devices=config.infra.get("num_gpus", 1),
        num_nodes=config.infra.get("num_nodes", 1),
        sync_batchnorm=True,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=exp_root,
        log_every_n_steps=config.infra.get("log_every_n_steps", 50),
        **OmegaConf.to_container(config.training.trainer_params),
    )

    # Train
    trainer.fit(system, datamodule=wrapped_dm)

    # Save checkpoint
    save_vlm_checkpoint(system.model, OmegaConf.to_container(config.system.params, resolve=True), f"{model_dir}/final_ckpt")

    logging.info("Training complete!")
    return system, wrapped_dm, exp_root


def main():
    args = parse_args()
    setup_logging(debug=args.debug)

    config = OmegaConf.load(args.config)
    logging.info(f"Loaded config from {args.config}")
    logging.info(OmegaConf.to_yaml(config))

    train(config)


if __name__ == "__main__":
    main()

