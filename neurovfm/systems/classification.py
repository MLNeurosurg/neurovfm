"""
Supervised Vision Classification System

Implements downstream classification tasks using pretrained vision encoders.
Supports MIL aggregation for study-level predictions.
"""

import logging
import gc
from typing import Dict, Optional, List, Tuple, Any
import torch
import torch.nn as nn
import torch.distributed as dist
import pytorch_lightning as pl
import torchmetrics
import torch_scatter

from neurovfm.models import (
    get_vit_backbone,
    AggregateThenClassify,
    ClassifyThenAggregate,
)
from neurovfm.models.projector import MLP
from neurovfm.systems.utils import NormalizationModule
from neurovfm.optim import get_optimizer_scheduler


class VisionClassifier(nn.Module):
    """
    Classifier network with pretrained vision encoder and MIL pooling.
    
    Args:
        vision_backbone_cf (Dict): Configuration for vision encoder backbone
        pooler_cf (Dict): Configuration for MIL pooler ('abmil', 'addmil', 'avgpool')
        proj_params (Dict): Configuration for projection head (output_dim, hidden_dims)
    """
    def __init__(
        self,
        vision_backbone_cf: Dict,
        pooler_cf: Dict,
        proj_params: Dict,
    ):
        super().__init__()

        # Frozen pretrained encoder
        self.bb = get_vit_backbone(**vision_backbone_cf)
        for param in self.bb.parameters():
            param.requires_grad = False
        self.bb.eval()

        # MIL Pooler
        which = pooler_cf["which"]
        if which in {"abmil", "aggregate_then_classify"}:
            self.pooler = AggregateThenClassify(
                dim=self.bb.embed_dim,
                **pooler_cf["params"],
            )
            self.proj = MLP(
                in_dim=self.pooler.num_features,
                out_dim=proj_params.get("out_dim", 1),
                hidden_dims=proj_params.get("hidden_dims", [self.pooler.num_features // 2]),
            )
        elif which in {"addmil", "classify_then_aggregate"}:
            self.pooler = ClassifyThenAggregate(
                dim=self.bb.embed_dim,
                **pooler_cf["params"],
            )
            # Classify-Then-Aggregate outputs logits directly
            self.proj = None
        elif pooler_cf["which"] == "avgpool":
            self.pooler = lambda x, cu_seqlens, max_seqlen: torch_scatter.segment_csr(
                src=x, indptr=cu_seqlens.long(), reduce='mean'
            )
            self.proj = MLP(
                in_dim=self.bb.embed_dim,
                out_dim=proj_params.get("out_dim", 1),
                hidden_dims=proj_params.get("hidden_dims", [self.bb.embed_dim // 2]),
            )
        else:
            raise ValueError(f"Pooler {pooler_cf['which']} not supported")


class VisionClassificationSystem(pl.LightningModule):
    """
    PyTorch Lightning System for supervised vision classification.

    Supports binary/multilabel/multiclass classification with pretrained frozen encoders.

    Args:
        model_hyperparams (Dict): Configuration for VisionClassifier
        loss_cf (Dict): Loss function configuration ('bce', 'ce', 'mse', 'mae')
        opt_cf (Dict): Optimizer configuration
        schd_cf (Dict): Learning rate scheduler configuration
        normalization_stats_list (List[List[float]], optional): Custom normalization stats
        training_params (Dict, optional): Training parameters (num_it_per_ep, num_ep_total, wts, etc.)

    Example:
        >>> model_config = {
        ...     'vision_backbone_cf': {...},
        ...     'pooler_cf': {'which': 'abmil', 'params': {...}},
        ...     'proj_params': {'out_dim': 5, 'hidden_dims': [256]}
        ... }
        >>> loss_config = {'which': 'bce'}
        >>> system = VisionClassificationSystem(
        ...     model_hyperparams=model_config,
        ...     loss_cf=loss_config,
        ...     opt_cf=opt_config,
        ...     schd_cf=schd_config,
        ...     training_params=training_params
        ... )
    """
    def __init__(
        self, 
        model_hyperparams: Dict,
        loss_cf: Optional[Dict] = None,
        opt_cf: Optional[Dict] = None,
        schd_cf: Optional[Dict] = None,
        normalization_stats_list: Optional[List[List[float]]] = None,
        training_params: Optional[Dict] = None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.loss_cf_ = loss_cf
        self.opt_cf_ = opt_cf
        self.schd_cf_ = schd_cf
        self.training_params_ = training_params

        # Extract class weights before removing from training_params
        wts = training_params["wts"]
        training_params.pop("wts")

        # Initialize model
        self.model = VisionClassifier(**model_hyperparams)

        # Setup loss function
        if loss_cf["which"] == "bce":
            if wts.ndim == 2:  # Multilabel
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=wts[:, 1] / wts[:, 0])
            else:  # Binary
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=wts[1] / wts[0])
        elif loss_cf["which"] == "ce":
            self.criterion = nn.CrossEntropyLoss(weight=wts)
        elif loss_cf["which"] in ["mse", "rmse"]:
            self.criterion = nn.MSELoss()
        elif loss_cf["which"] == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Loss {loss_cf['which']} not supported")

        # Setup metrics
        if training_params is not None:
            self.train_loss = torchmetrics.MeanMetric()
            self.val_loss = torchmetrics.MeanMetric()

            if loss_cf["which"] == "bce":
                if wts.ndim == 2:  # Multilabel
                    num_labels = wts.shape[0]
                    self.train_acc_stats = torchmetrics.classification.MultilabelStatScores(
                        num_labels=num_labels, threshold=0.0, average=None
                    )
                    self.val_acc_stats = torchmetrics.classification.MultilabelStatScores(
                        num_labels=num_labels, threshold=0.0, average=None
                    )
                    self.train_auc_stats = torchmetrics.AUROC(
                        task="multilabel", num_labels=num_labels, average=None
                    )
                    self.val_auc_stats = torchmetrics.AUROC(
                        task="multilabel", num_labels=num_labels, average=None
                    )
                else:  # Binary (1D weights)
                    self.train_acc_stats = torchmetrics.classification.BinaryStatScores(threshold=0.0)
                    self.val_acc_stats = torchmetrics.classification.BinaryStatScores(threshold=0.0)
                    self.train_auc_stats = torchmetrics.AUROC(task="binary")
                    self.val_auc_stats = torchmetrics.AUROC(task="binary")
            elif loss_cf["which"] == "ce":
                self.train_stats = torchmetrics.classification.MulticlassAccuracy(
                    num_classes=len(wts), average="macro"
                )
                self.val_stats = torchmetrics.classification.MulticlassAccuracy(
                    num_classes=len(wts), average="macro"
                )
                self.num_classes = len(wts)
            
            self.test_output = []
        else:
            self.train_loss = self.val_loss = self.test_output = None

        # Setup normalization
        self.normalization_module = NormalizationModule(
            custom_stats_list=normalization_stats_list
        )

    def compute_balanced_accuracy(self, stats: torch.Tensor) -> torch.Tensor:
        """Compute balanced accuracy from stat scores (TP, FP, TN, FN)."""
        tp, fp, tn, fn, _ = stats
        sensitivity = tp / (tp + fn + 1e-10)
        specificity = tn / (tn + fp + 1e-10)
        balanced_acc = (sensitivity + specificity) / 2
        return balanced_acc

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """
        Training step for classification.
        
        Args:
            batch (Dict): Collated batch from ImageDataset with background filtering
            batch_idx (int): Batch index
        
        Returns:
            torch.Tensor: Loss for optimization
        """
        # Extract data
        img, coords, labels = batch["img"], batch["coords"], batch["label"]
        sizes = batch["size"]
        
        # Create masks for encoder
        enc_pred = (
            batch["series_masks_indices"] if batch["series_masks_indices"].numel() > 0 else None,
            batch["series_cu_seqlens"],
            batch["series_max_len"]
        )
        study_masks = (
            None,
            batch["study_cu_seqlens"],
            batch["study_max_len"]
        )

        # Normalize images
        if enc_pred[0] is not None:
            img = self.normalization_module.normalize(
                img, 
                batch["mode"], 
                batch["path"],
                cu_seqlens=torch.cat([
                    torch.tensor([0], device=img.device), 
                    torch.cumsum(sizes.prod(dim=1), dim=0)
                ]),
                sizes=sizes
            )
        else:
            img = self.normalization_module.normalize(
                img, 
                batch["mode"], 
                batch["path"],
                cu_seqlens=batch["series_cu_seqlens"],
                sizes=sizes
            )

        bsz = len(labels)

        # Forward pass through frozen encoder
        with torch.no_grad():
            embs = self.model.bb(
                img, coords, 
                masks=enc_pred[0], 
                cu_seqlens=enc_pred[1], 
                max_seqlen=enc_pred[2]
            )

        # Pool embeddings at study level
        if study_masks is not None:
            study_cu_seqlens = study_masks[1]
            study_max_seqlen = study_masks[2]
        else:
            study_cu_seqlens = enc_pred[1]
            study_max_seqlen = enc_pred[2]

        # Apply pooler and projection
        if self.model.proj is not None:
            pooled = self.model.pooler(embs, cu_seqlens=study_cu_seqlens, max_seqlen=study_max_seqlen)
            if pooled.size(0) != bsz:
                pooled = pooled.view(bsz, -1, pooled.size(-1))
            logits = self.model.proj(pooled).squeeze(-1)
        else:
            logits = self.model.pooler(embs, cu_seqlens=study_cu_seqlens, max_seqlen=study_max_seqlen)
            logits = logits.squeeze(-1)

        # Compute loss
        if self.loss_cf_["which"] == "bce":
            if logits.ndim > 1:  # Multilabel
                pred = logits > 0.
                self.train_acc_stats.update(pred, labels.int())
                self.train_auc_stats.update(logits, labels.int())
                loss = self.criterion(logits, labels)
            else:  # Binary
                pred = logits > 0.
                self.train_acc_stats.update(pred, labels.int())
                self.train_auc_stats.update(logits, labels.int())
                loss = self.criterion(logits, labels)
        elif self.loss_cf_["which"] == "ce":
            pred = logits.argmax(dim=-1)
            self.train_stats.update(pred, labels.long())
            loss = self.criterion(logits, labels.long())
        else:
            loss = self.criterion(logits, labels)

        # Log metrics
        self.train_loss.update(loss.detach().item(), weight=bsz)
        self.log(
            "train/loss",
            loss.detach().item(),
            on_step=True,
            on_epoch=False,
            batch_size=bsz,
            rank_zero_only=True
        )

        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        """
        Validation step for classification.
        
        Args:
            batch (Dict): Collated batch from ImageDataset
            batch_idx (int): Batch index
        """
        # Extract data
        img, coords, labels = batch["img"], batch["coords"], batch["label"]
        sizes = batch["size"]
        
        # Create masks
        enc_pred = (
            batch["series_masks_indices"] if batch["series_masks_indices"].numel() > 0 else None,
            batch["series_cu_seqlens"],
            batch["series_max_len"]
        )
        study_masks = (
            None,
            batch["study_cu_seqlens"],
            batch["study_max_len"]
        )

        # Normalize images
        if enc_pred[0] is not None:
            img = self.normalization_module.normalize(
                img, 
                batch["mode"], 
                batch["path"],
                cu_seqlens=torch.cat([
                    torch.tensor([0], device=img.device), 
                    torch.cumsum(sizes.prod(dim=1), dim=0)
                ]),
                sizes=sizes
            )
        else:
            img = self.normalization_module.normalize(
                img, 
                batch["mode"], 
                batch["path"],
                cu_seqlens=batch["series_cu_seqlens"],
                sizes=sizes
            )

        bsz = len(labels)

        # Forward pass
        with torch.no_grad():
            embs = self.model.bb(
                img, coords, 
                masks=enc_pred[0], 
                cu_seqlens=enc_pred[1], 
                max_seqlen=enc_pred[2]
            )

        # Pool embeddings
        if study_masks is not None:
            study_cu_seqlens = study_masks[1]
            study_max_seqlen = study_masks[2]
        else:
            study_cu_seqlens = enc_pred[1]
            study_max_seqlen = enc_pred[2]

        if self.model.proj is not None:
            pooled = self.model.pooler(embs, cu_seqlens=study_cu_seqlens, max_seqlen=study_max_seqlen)
            if pooled.size(0) != bsz:
                pooled = pooled.view(bsz, -1, pooled.size(-1))
            logits = self.model.proj(pooled).squeeze(-1)
        else:
            logits = self.model.pooler(embs, cu_seqlens=study_cu_seqlens, max_seqlen=study_max_seqlen)
            logits = logits.squeeze(-1)

        # Compute loss and metrics
        if self.loss_cf_["which"] == "bce":
            if logits.ndim > 1:  # Multilabel
                pred = logits > 0.
                self.val_acc_stats.update(pred, labels.int())
                self.val_auc_stats.update(logits, labels.int())
                loss = self.criterion(logits, labels)
            else:  # Binary
                pred = logits > 0.
                self.val_acc_stats.update(pred, labels.int())
                self.val_auc_stats.update(logits, labels.int())
                loss = self.criterion(logits, labels)
        elif self.loss_cf_["which"] == "ce":
            pred = logits.argmax(dim=-1)
            self.val_stats.update(pred, labels.long())
            loss = self.criterion(logits, labels.long())
        else:
            loss = self.criterion(logits, labels)

        # Log metrics
        self.val_loss.update(loss.detach().item(), weight=bsz)
        self.log(
            "val/loss",
            loss.detach().item(),
            on_step=True,
            on_epoch=False,
            batch_size=bsz,
            rank_zero_only=True
        )

    def on_train_epoch_end(self):
        """Log epoch-level training metrics."""
        train_loss = self.train_loss.compute()
        self.log(
            "train/loss_epoch",
            train_loss,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True
        )
        logging.info(f"train/loss_epoch: {train_loss}")
        self.train_loss.reset()

        # Log classification metrics
        if self.loss_cf_["which"] == "bce":
            stats = self.train_acc_stats.compute()
            if stats.ndim > 1:  # Multilabel
                for i in range(stats.shape[0]):
                    bal_acc = self.compute_balanced_accuracy(stats[i])
                    self.log(f"train/bal_acc_label_{i}_epoch", bal_acc, sync_dist=True)
                auc = self.train_auc_stats.compute()
                for i in range(len(auc)):
                    self.log(f"train/auc_label_{i}_epoch", auc[i], sync_dist=True)
            else:  # Binary
                bal_acc = self.compute_balanced_accuracy(stats)
                self.log("train/bal_acc_epoch", bal_acc, sync_dist=True)
                auc = self.train_auc_stats.compute()
                self.log("train/auc_epoch", auc, sync_dist=True)
            
            self.train_acc_stats.reset()
            self.train_auc_stats.reset()
        elif self.loss_cf_["which"] == "ce":
            acc = self.train_stats.compute()
            self.log("train/acc_epoch", acc, sync_dist=True)
            logging.info(f"train/acc_epoch: {acc}")
            self.train_stats.reset()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()

    def on_validation_epoch_end(self):
        """Log epoch-level validation metrics."""
        val_loss = self.val_loss.compute()
        self.log(
            "val/loss_epoch",
            val_loss,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=True
        )
        logging.info(f"val/loss_epoch: {val_loss}")
        self.val_loss.reset()

        # Log classification metrics
        if self.loss_cf_["which"] == "bce":
            stats = self.val_acc_stats.compute()
            if stats.ndim > 1:  # Multilabel
                for i in range(stats.shape[0]):
                    bal_acc = self.compute_balanced_accuracy(stats[i])
                    self.log(f"val/bal_acc_label_{i}_epoch", bal_acc, sync_dist=True)
                auc = self.val_auc_stats.compute()
                for i in range(len(auc)):
                    self.log(f"val/auc_label_{i}_epoch", auc[i], sync_dist=True)
            else:  # Binary
                bal_acc = self.compute_balanced_accuracy(stats)
                self.log("val/bal_acc_epoch", bal_acc, sync_dist=True)
                auc = self.val_auc_stats.compute()
                self.log("val/auc_epoch", auc, sync_dist=True)
            
            self.val_acc_stats.reset()
            self.val_auc_stats.reset()
        elif self.loss_cf_["which"] == "ce":
            acc = self.val_stats.compute()
            self.log("val/acc_epoch", acc, sync_dist=True)
            logging.info(f"val/acc_epoch: {acc}")
            self.val_stats.reset()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.barrier()

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
            normbias_nowd=True,  # Norm and bias should not have weight decay
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
