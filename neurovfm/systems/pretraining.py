"""
Self-Supervised Vision Pretraining System (Vol-JEPA)

Implements masked prediction pretraining using a student-teacher EMA framework
with vision transformer backbone. Supports JEPA-style self-supervised learning.
"""

import logging
import copy
import gc
from typing import Dict, Optional, List, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
import torchmetrics

from neurovfm.models import VisionTransformer, VisionPredictor, get_vit_backbone
from neurovfm.systems.utils import NormalizationModule
from neurovfm.optim import get_optimizer_scheduler


class VisionNetwork(nn.Module):
    """
    Student-Teacher network for Vol-JEPA pretraining.
    
    Args:
        vision_backbone_cf (Dict): Configuration for vision encoder backbone
        predictor_cf (Dict): Configuration for predictor module
    """
    def __init__(
        self,
        vision_backbone_cf: Dict,
        predictor_cf: Dict,
    ):
        super().__init__()
        
        # Student encoder + predictor
        self.student = nn.ModuleDict({
            'vision_encoder': get_vit_backbone(**vision_backbone_cf),
        })
        
        # Predictor (maps student context to teacher target)
        self.predictor = VisionPredictor(**predictor_cf)
        
        # Teacher encoder (EMA of student)
        self.teacher = nn.ModuleDict({
            'vision_encoder': get_vit_backbone(**vision_backbone_cf),
        })
        
        # Initialize teacher as copy of student
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False


class VisionPretrainingSystem(pl.LightningModule):
    """
    PyTorch Lightning System for self-supervised Vision Transformer pretraining (Vol-JEPA).

    Implements masked prediction with student-teacher EMA framework for medical imaging.

    Args:
        model_hyperparams (Dict): Configuration for VisionNetwork
        ema_beta (List[float]): EMA momentum range [start, end]. Defaults to [0.994, 1.0]
        opt_cf (Dict): Optimizer configuration
        schd_cf (Dict): Learning rate scheduler configuration
        normalization_stats_list (List[List[float]], optional): Custom normalization stats
        training_params (Dict, optional): Training parameters (num_it_per_ep, num_ep_total, effective_batch_size)
        num_mask_generators (int): Number of mask generators. Defaults to 1.

    Example:
        >>> model_config = {
        ...     'vision_backbone_cf': {...},
        ...     'predictor_cf': {...},
        ... }
        >>> system = VisionPretrainingSystem(
        ...     model_hyperparams=model_config,
        ...     ema_beta=[0.994, 1.0],
        ...     opt_cf=opt_config,
        ...     schd_cf=schd_config,
        ...     training_params=training_params
        ... )
    """
    def __init__(
        self, 
        model_hyperparams: Dict,
        ema_beta: List[float] = [0.994, 1.0],
        opt_cf: Optional[Dict] = None,
        schd_cf: Optional[Dict] = None,
        normalization_stats_list: Optional[List[List[float]]] = None,
        training_params: Optional[Dict] = None,
        num_mask_generators: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.opt_cf_ = opt_cf
        self.schd_cf_ = schd_cf
        self.training_params_ = training_params

        # Initialize model
        self.model = VisionNetwork(**model_hyperparams)

        # Setup EMA momentum scheduler
        if training_params:
            ipe = training_params["num_it_per_ep"]
            num_epochs = training_params["num_ep_total"]
            ipe_scale = schd_cf["params"]["ipe_scale"]
            total_steps = int(ipe * num_epochs * ipe_scale)
            
            self.momentum_scheduler = (
                ema_beta[0] + i * (ema_beta[1] - ema_beta[0]) / (total_steps - 1)
                for i in range(total_steps)
            )
            self.beta = next(self.momentum_scheduler)

        # Setup loss (Smooth L1 for JEPA)
        self.criterion = nn.SmoothL1Loss()

        # Setup metrics
        if training_params:
            self.train_loss = nn.ModuleList([
                nn.ModuleDict({k: torchmetrics.MeanMetric() for k in ["mim", "total"]}) 
                for _ in range(num_mask_generators + 1)
            ])
            self.val_loss = nn.ModuleList([
                nn.ModuleDict({k: torchmetrics.MeanMetric() for k in ["mim", "total"]}) 
                for _ in range(num_mask_generators + 1)
            ])
        else:
            self.train_loss = self.val_loss = None

        # Setup normalization
        self.normalization_module = NormalizationModule(
            custom_stats_list=normalization_stats_list
        )

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]):
        """Resume EMA momentum scheduler from checkpoint."""
        if self.training_params_:
            global_step = checkpoint['global_step']
            for _ in range(global_step):
                self.beta = next(self.momentum_scheduler)

    def forward_teacher(
        self, 
        img: torch.Tensor, 
        coords: torch.Tensor, 
        pred_init: List[Tuple], 
        pred: List[Tuple]
    ) -> Tuple[List[torch.Tensor], List]:
        """
        Forward pass through teacher encoder to generate targets.
        
        Args:
            img (torch.Tensor): Normalized image tokens (N, D)
            coords (torch.Tensor): Token coordinates (N, 3)
            pred_init (List[Tuple]): Full volume masks for teacher [(indices, cu_seqlens, max_seqlen)]
            pred (List[Tuple]): Target region masks [(indices, cu_seqlens, max_seqlen)]
        
        Returns:
            Tuple: (target_features, empty_list) for compatibility
        """
        hs = []

        for i in range(len(pred_init)):
            with torch.no_grad():
                # Forward entire image through teacher
                h = self.model.teacher.vision_encoder(
                    img,
                    coords,
                    masks=pred_init[i][0],
                    masks_enc=None,
                    cu_seqlens=pred_init[i][1],
                    max_seqlen=pred_init[i][2]
                )
                # Normalize and keep only target tokens
                h = F.layer_norm(h, (h.size(-1),))  # [N_total, D]
                h_target = h[pred[i][0]]
                hs.append(h_target)
                del h

        return hs, []

    def forward_student(
        self, 
        img: torch.Tensor, 
        coords: torch.Tensor, 
        enc: List[Tuple], 
        pred: List[Tuple], 
        enc_pred: List[Tuple]
    ) -> Tuple[List[torch.Tensor], List]:
        """
        Forward pass through student encoder and predictor.
        
        Args:
            img (torch.Tensor): Normalized image tokens (N, D)
            coords (torch.Tensor): Token coordinates (N, 3)
            enc (List[Tuple]): Context masks for student encoder [(indices, cu_seqlens, max_seqlen)]
            pred (List[Tuple]): Target region masks [(indices, cu_seqlens, max_seqlen)]
            enc_pred (List[Tuple]): Full context masks [(indices, cu_seqlens, max_seqlen)]
        
        Returns:
            Tuple: (predictions, empty_list) for compatibility
        """
        # Encode context tokens
        zs = [
            self.model.student.vision_encoder(
                img, coords, 
                masks=enc_pred[i][0], 
                masks_enc=enc[i][0], 
                cu_seqlens=enc[i][1], 
                max_seqlen=enc[i][2]
            )
            for i in range(len(enc))
        ]

        # Predict target tokens
        predictions = []
        for i in range(len(pred)):
            pred_tokens, toselect = self.model.predictor(
                zs[i],
                coords,
                enc[i],
                pred[i],
                enc_pred[i],
            )
            # Select only target tokens; wrap in list for loss_fn interface
            predictions.append([pred_tokens[toselect]])
            del pred_tokens, toselect

        del zs
        return predictions, []

    def loss_fn(
        self, 
        zs: List[List[torch.Tensor]], 
        hs: List[torch.Tensor]
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Compute JEPA loss between student predictions and teacher targets.
        
        Args:
            zs (List[List[torch.Tensor]]): Student predictions
            hs (List[torch.Tensor]): Teacher targets
        
        Returns:
            Dict: Loss dictionary with 'mim' and 'total' keys
        """
        loss_mim = [
            self.criterion(zs[i][0], hs[i])
            for i in range(len(zs))
        ]

        return dict(mim=loss_mim, total=loss_mim)

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step for Vol-JEPA pretraining.
        
        Args:
            batch (Tuple): (collated_batch, pred_masks, pred_init_masks, enc_masks)
            batch_idx (int): Batch index
        
        Returns:
            torch.Tensor: Total loss for optimization
        """
        udata, pred, pred_init, enc = batch
        img, coords = udata["img"], udata["coords"]
        
        # Normalize images
        if pred_init[0][0] is not None:
            img = self.normalization_module.normalize(
                img, 
                udata["mode"], 
                udata["path"],
                cu_seqlens=torch.cat([
                    torch.tensor([0], device=img.device), 
                    torch.cumsum(udata["size"].prod(dim=1), dim=0)
                ]).to(torch.int32)
            )
        else:
            img = self.normalization_module.normalize(
                img, 
                udata["mode"], 
                udata["path"],
                cu_seqlens=pred_init[0][1]
            )
        
        bsz = len(udata["label"])

        # Log EMA momentum
        self.log(
            "train/teacher_beta",
            self.beta,
            on_step=True,
            on_epoch=True,
            batch_size=bsz,
            rank_zero_only=True
        )

        # Forward pass
        hs, _ = self.forward_teacher(img, coords, pred_init, pred)
        zs, _ = self.forward_student(img, coords, enc, pred, pred_init)

        # Compute loss
        loss_dict = self.loss_fn(zs, hs)
        del hs, zs

        # Log per-generator losses
        for g_idx in range(len(loss_dict["total"])):
            for k, v in loss_dict.items():
                self.log(
                    f"train/jepa_{g_idx}_{k}",
                    v[g_idx].detach().item(),
                    on_step=True,
                    on_epoch=False,
                    batch_size=bsz,
                    rank_zero_only=True
                )
                self.train_loss[g_idx][k].update(v[g_idx].detach().item(), weight=bsz)

        # Log average loss over generators
        for k, v in loss_dict.items():
            loss = sum(v) / len(v)
            self.log(
                f"train/jepa_{k}",
                loss.detach().item(),
                on_step=True,
                on_epoch=False,
                batch_size=bsz,
                rank_zero_only=True
            )
            self.train_loss[len(v)][k].update(loss.detach().item(), weight=bsz)

        return loss

    def validation_step(self, batch: Tuple, batch_idx: int):
        """
        Validation step for Vol-JEPA pretraining.
        
        Args:
            batch (Tuple): (collated_batch, pred_masks, pred_init_masks, enc_masks)
            batch_idx (int): Batch index
        """
        udata, pred, pred_init, enc = batch
        img, coords = udata["img"], udata["coords"]

        # Normalize images
        if pred_init[0][0] is not None:
            img = self.normalization_module.normalize(
                img, 
                udata["mode"], 
                udata["path"],
                cu_seqlens=torch.cat([
                    torch.tensor([0], device=img.device), 
                    torch.cumsum(udata["size"].prod(dim=1), dim=0)
                ]).to(torch.int32)
            )
        else:
            img = self.normalization_module.normalize(
                img, 
                udata["mode"], 
                udata["path"],
                cu_seqlens=pred_init[0][1]
            )
        
        bsz = len(udata["label"])

        # Forward pass
        hs, _ = self.forward_teacher(img, coords, pred_init, pred)
        zs, _ = self.forward_student(img, coords, enc, pred, pred_init)

        # Compute loss
        loss_dict = self.loss_fn(zs, hs)
        del hs, zs

        # Log per-generator losses
        for g_idx in range(len(loss_dict["total"])):
            for k, v in loss_dict.items():
                self.log(
                    f"val/jepa_{g_idx}_{k}",
                    v[g_idx].detach().item(),
                    on_step=True,
                    on_epoch=False,
                    batch_size=bsz,
                    rank_zero_only=True
                )
                self.val_loss[g_idx][k].update(v[g_idx].detach().item(), weight=bsz)

        # Log average loss over generators
        for k, v in loss_dict.items():
            loss = sum(v) / len(v)
            self.log(
                f"val/jepa_{k}",
                loss.detach().item(),
                on_step=True,
                on_epoch=False,
                batch_size=bsz,
                rank_zero_only=True
            )
            self.val_loss[len(v)][k].update(loss.detach().item(), weight=bsz)

    def on_train_epoch_end(self):
        """Log epoch-level training metrics and cleanup."""
        for idx, train_loss_dict in enumerate(self.train_loss):
            for k, train_loss_module in train_loss_dict.items():
                train_loss = train_loss_module.compute()
                name_str = (f"train/jepa_{idx}_{k}_epoch" if idx != len(self.train_loss) - 1 
                           else f"train/jepa_{k}_epoch")
                self.log(
                    name_str,
                    train_loss,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=True
                )
                logging.info(f"{name_str}: {train_loss}")
                train_loss_module.reset()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()

    def on_validation_epoch_end(self):
        """Log epoch-level validation metrics and cleanup."""
        for idx, val_loss_dict in enumerate(self.val_loss):
            for k, val_loss_module in val_loss_dict.items():
                val_loss = val_loss_module.compute()
                name_str = (f"val/jepa_{idx}_{k}_epoch" if idx != len(self.val_loss) - 1 
                           else f"val/jepa_{k}_epoch")
                self.log(
                    name_str,
                    val_loss,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=True
                )
                logging.info(f"{name_str}: {val_loss}")
                val_loss_module.reset()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()

    def on_before_zero_grad(self, optimizer):
        """Update teacher with EMA before optimizer zero_grad."""
        with torch.no_grad():
            m = self.beta
            for param_s, param_t in zip(
                self.model.student.vision_encoder.parameters(), 
                self.model.teacher.vision_encoder.parameters()
            ):
                param_t.data.mul_(m).add_(param_s.detach().data, alpha=1 - m)

        self.beta = next(self.momentum_scheduler)

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
