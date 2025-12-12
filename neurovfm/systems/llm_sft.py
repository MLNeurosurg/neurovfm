"""
Vision-Language Model Supervised Fine-Tuning System

Implements visual instruction tuning for VisionLanguageModel.
Supports two training stages:
- s1_pretrain: train connector only, keep llm frozen, vision encoder frozen.
- s2_finetune: train connector + full llm, vision encoder frozen.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch

from neurovfm.models.vlm import VisionLanguageModel
from neurovfm.systems.utils import NormalizationModule
from neurovfm.optim import get_optimizer_scheduler


class VisionInstructionTuningSystem(pl.LightningModule):
    """
    Lightning module wrapping the vision-language model for SFT.
    """

    def __init__(
        self,
        stage: str,
        model: VisionLanguageModel,
        opt_cf: Optional[Dict[str, Any]] = None,
        schd_cf: Optional[Dict[str, Any]] = None,
        training_params: Optional[Dict[str, Any]] = None,
        normalization_stats_list: Optional[List[List[float]]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.stage = stage
        self.model = model

        self.opt_cf_ = opt_cf
        self.schd_cf_ = schd_cf
        self.training_params_ = training_params or {}

        # normalization module for vision tokens
        self.norm_module = NormalizationModule(
            custom_stats_list=normalization_stats_list
        ).to(self.device)

        self._configure_freezing()
        self._log_trainable_params()

    def _configure_freezing(self):
        """
        set requires_grad flags based on stage.
        """
        # vision encoder is always frozen
        if hasattr(self.model, "vision_encoder"):
            for p in self.model.vision_encoder.parameters():
                p.requires_grad = False

        if self.stage == "s1_pretrain":
            # connector trainable, llm frozen
            if hasattr(self.model, "vision_connector"):
                for p in self.model.vision_connector.parameters():
                    p.requires_grad = True
            if hasattr(self.model, "language_model"):
                for p in self.model.language_model.parameters():
                    p.requires_grad = False
        elif self.stage == "s2_finetune":
            # connector + llm trainable
            if hasattr(self.model, "vision_connector"):
                for p in self.model.vision_connector.parameters():
                    p.requires_grad = True
            if hasattr(self.model, "language_model"):
                for p in self.model.language_model.parameters():
                    p.requires_grad = True
        else:
            raise ValueError(f"unknown stage: {self.stage}")

    def _log_trainable_params(self):
        """
        Log parameter counts by component.
        """

        total_params = 0
        trainable_params = 0
        
        component_params = {}
        if self.model.language_model is not None:
            lm_params = sum(p.numel() for p in self.model.language_model.parameters())
            component_params['language_model'] = lm_params
        if self.model.vision_encoder is not None:
            ve_params = sum(p.numel() for p in self.model.vision_encoder.parameters())
            component_params['vision_encoder'] = ve_params
        if self.model.vision_connector is not None:
            vc_params = sum(p.numel() for p in self.model.vision_connector.parameters())
            component_params['vision_connector'] = vc_params
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logging.info("-" * 60)
        logging.info("Parameter Summary (Stage: {self.stage}):")
        logging.info(f"  Total parameters: {total_params:,}")
        for component, count in component_params.items():
            logging.info(f"    - {component}: {count:,}")
        logging.info(f"  Trainable parameters: {trainable_params:,}")
        logging.info(f"  Frozen parameters: {total_params - trainable_params:,}")
        logging.info(f"  Trainable percentage: {100 * trainable_params / total_params:.1f}%")
        logging.info("-" * 60)

    def forward(self, batch: Dict[str, Any]) -> Any:
        if "vision_batch" not in batch:
            raise ValueError("Vision data not found in batch.")
        if any(k not in batch for k in ["input_ids", "attention_mask", "labels"]):
            raise ValueError("Tokenized text (input_ids, attention_mask, labels) is required in batch.")

        # normalize vision tokens
        vision_batch = batch["vision_batch"]
        vision_batch["img"] = self.norm_module.normalize(
            img=vision_batch["img"],
            modes=vision_batch["mode"],
            paths=vision_batch["path"],
            cu_seqlens=vision_batch["series_cu_seqlens"],
            sizes=vision_batch.get("size"),
        )
        batch["vision_batch"] = vision_batch

        return self.model(
            vision_batch=batch["vision_batch"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels"),
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int):

        outputs = self.forward(batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):

        outputs = self.forward(batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Optimizer or (optimizer, scheduler) configuration
        """
        if not self.training_params_:
            return None

        opt, sch = get_optimizer_scheduler(
            self.model,
            opt_cf=self.opt_cf_,
            schd_cf=self.schd_cf_,
            **self.training_params_
        )

        if sch:
            lr_scheduler_config = {
                "scheduler": sch,
                "interval": "step",
                "frequency": 1,
                "name": "lr"
            }
            return [opt], lr_scheduler_config
        else:
            return [opt]

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:

        clip_val = gradient_clip_val or self.gradient_clip_val
        
        if clip_val is not None and clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
