"""
Training Script for Vision Models

Supports both self-supervised pretraining and supervised classification.
Uses PyTorch Lightning for distributed training with DDP.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import logging
import argparse
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pathlib import Path

from neurovfm.datasets import ImageDataModule
from neurovfm.systems import VisionPretrainingSystem, VisionClassificationSystem


# Map system names to classes
SYSTEMS = {
    "VisionPretrainingSystem": VisionPretrainingSystem,
    "VisionClassificationSystem": VisionClassificationSystem,
}


def get_num_it_per_train_ep(train_len: int, cf: OmegaConf) -> dict:
    """Calculate iterations per epoch and effective batch size.

    Args:
        train_len: length of the training set
        cf: global config

    Returns:
        num_it_per_ep: number of iteration in each epoch
    """
    if torch.cuda.is_available():
        # Use configured data-parallel world size if available
        world_size = cf.infra.get("num_gpus", 1) * cf.infra.get("num_nodes", 1)
    else:
        world_size = 1

    train_loader_cf = cf.data.loader.train
    batch_size = train_loader_cf.batch_size
    drop_last = train_loader_cf.get("drop_last", True)
    accumulate = cf.training.trainer_params.get("accumulate_grad_batches", 1)

    effective_batch_size = batch_size * world_size * accumulate

    num_it_per_ep = train_len // effective_batch_size

    if not drop_last:
        num_it_per_ep += int((train_len % effective_batch_size) > 0)

    return {
        "num_it_per_ep": num_it_per_ep,
        "effective_batch_size": effective_batch_size
    }
    

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Vision Models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def setup_logging(debug: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def instantiate_system(config: OmegaConf, training_params: dict):
    """
    Instantiate Lightning system from config.
    
    Args:
        config (OmegaConf): Configuration object
        training_params (dict): Training parameters
    
    Returns:
        pl.LightningModule: Instantiated system
    """
    system_name = config.system.which
    system_params = OmegaConf.to_container(config.system.params)
    
    if system_name not in SYSTEMS:
        raise ValueError(f"Unknown system: {system_name}. Available: {list(SYSTEMS.keys())}")
    
    return SYSTEMS[system_name](training_params=training_params, **system_params)


def instantiate_system_from_checkpoint(config: OmegaConf, checkpoint_path: str, training_params: dict = None):
    """
    Instantiate Lightning system from checkpoint.
    
    Args:
        config (OmegaConf): Configuration object
        checkpoint_path (str): Path to checkpoint file
        training_params (dict, optional): Training parameters
    
    Returns:
        pl.LightningModule: Instantiated system
    """
    system_name = config.system.which
    system_params = OmegaConf.to_container(config.system.params)
    
    if system_name not in SYSTEMS:
        raise ValueError(f"Unknown system: {system_name}")
    
    return SYSTEMS[system_name].load_from_checkpoint(
        checkpoint_path,
        training_params=training_params,
        **system_params
    )


def setup_directories(config: OmegaConf):
    """
    Setup experiment directories.
    
    Args:
        config (OmegaConf): Configuration object
    
    Returns:
        Tuple[Path, Path]: (experiment_root, model_dir)
    """
    exp_root = Path(config.infra.exp_root)
    exp_root.mkdir(parents=True, exist_ok=True)
    
    model_dir = exp_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Experiment root: {exp_root}")
    logging.info(f"Model directory: {model_dir}")
    
    return exp_root, model_dir


def setup_checkpoints(config: OmegaConf, model_dir: Path):
    """
    Setup checkpoint callbacks.
    
    Args:
        config (OmegaConf): Configuration object
        model_dir (Path): Directory to save checkpoints
    
    Returns:
        List[pl.callbacks.Callback]: Checkpoint callbacks
    """
    callbacks = []
    
    # Best model checkpoint
    if hasattr(config.training, 'monitor_metric'):
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=model_dir,
                filename="best-{epoch:02d}-{" + config.training.monitor_metric + ":.4f}",
                monitor=config.training.monitor_metric,
                mode=config.training.get('monitor_mode', 'min'),
                save_top_k=1,
                save_last=True,
            )
        )
    
    # Periodic checkpoint
    if hasattr(config.training, 'checkpoint_every_n_epochs'):
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=model_dir,
                filename="epoch-{epoch:02d}",
                every_n_epochs=config.training.checkpoint_every_n_epochs,
                save_top_k=-1,
            )
        )
    
    return callbacks


def setup_loggers(config: OmegaConf, exp_root: Path):
    """
    Setup loggers.
    
    Args:
        config (OmegaConf): Configuration object
        exp_root (Path): Experiment root directory
    
    Returns:
        List[pl.loggers.Logger]: Logger instances
    """
    loggers = [
        pl.loggers.TensorBoardLogger(save_dir=exp_root, name="tb"),
        pl.loggers.CSVLogger(save_dir=exp_root, name="csv")
    ]
    
    if hasattr(config.infra, 'wandb_project') and config.infra.wandb_project:
        import wandb
        loggers.append(
            pl.loggers.WandbLogger(
                save_dir=exp_root,
                project=config.infra.wandb_project,
                name=config.infra.get('comment', 'experiment')
            )
        )
        logging.info(f"Using wandb project {config.infra.wandb_project}/{config.infra.get('comment', 'experiment')}")
    else:
        logging.info("Not using wandb")
    
    return loggers


def train(config: OmegaConf):
    """
    Main training function.
    
    Args:
        config (OmegaConf): Configuration object
    """
    # Setup directories
    exp_root, model_dir = setup_directories(config)
    
    # Setup data
    dm = ImageDataModule(config)
    dm.setup(stage="fit")
    
    # Compute iteration and batch-size stats for optimizer/scheduler
    train_stats = get_num_it_per_train_ep(len(dm.train_dataset), config)
    training_params = {
        **train_stats,
        "num_ep_total": config.training.trainer_params.max_epochs,
    }
    
    if "Classification" in config.system.which:
        # Extract class weights from dataset (if available)
        if hasattr(dm.train_dataset, 'class_weights'):
            training_params["wts"] = dm.train_dataset.class_weights
        else:
            # Default balanced weights
            num_classes = config.system.params.get('num_classes', 2)
            training_params["wts"] = torch.ones(num_classes)
    
    logging.info(f"Training params: {training_params}")
    
    # Instantiate system
    if hasattr(config.training, 'resume_checkpoint') and config.training.resume_checkpoint:
        system = instantiate_system_from_checkpoint(
            config,
            config.training.resume_checkpoint,
            training_params
        )
        logging.info(f"Resuming from checkpoint: {config.training.resume_checkpoint}")
    else:
        system = instantiate_system(config, training_params)
        logging.info("Training from scratch")
    
    # Load pretrained backbone if specified
    if hasattr(config.training, 'load_backbone') and config.training.load_backbone:
        ckpt_path = config.training.load_backbone.ckpt_path
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        
        if hasattr(config.training.load_backbone, 'remove_prefix') and config.training.load_backbone.remove_prefix:
            prefix = config.training.load_backbone.remove_prefix
            state_dict = {
                k.removeprefix(prefix): ckpt_dict["state_dict"][k]
                for k in ckpt_dict["state_dict"]
                if prefix in k
            }
        else:
            state_dict = ckpt_dict["state_dict"]
        
        system.model.bb.load_state_dict(state_dict, strict=False)
        for param in system.model.bb.parameters():
            param.requires_grad = False
        system.model.bb.eval()
        logging.info(f"Loaded backbone from {ckpt_path}")
    
    # Setup callbacks
    callbacks = setup_checkpoints(config, model_dir)
    callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval="step", log_momentum=False))
    
    if config.infra.get("log_gpu", False):
        callbacks.append(pl.callbacks.DeviceStatsMonitor(cpu_stats=True))
    
    # Setup loggers
    loggers = setup_loggers(config, exp_root)
    
    # Create trainer
    use_gpu = torch.cuda.is_available()
    
    if use_gpu and config.infra.get("num_gpus", 1) > 1:
        strategy = pl.strategies.DDPStrategy(
            find_unused_parameters=False,
            static_graph=True
        )
    else:
        strategy = "auto"
    
    trainer = pl.Trainer(
        accelerator="cuda" if use_gpu else "cpu",
        strategy=strategy,
        devices=config.infra.get("num_gpus", 1) if use_gpu else "auto",
        num_nodes=config.infra.get("num_nodes", 1),
        sync_batchnorm=True if use_gpu else False,
        callbacks=callbacks,
        default_root_dir=exp_root,
        logger=loggers,
        log_every_n_steps=config.infra.get("log_every_n_steps", 250),
        **OmegaConf.to_container(config.training.trainer_params)
    )
    
    # Train
    trainer.fit(system, datamodule=dm)
    
    logging.info("Training complete!")
    return system, dm, exp_root


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(debug=args.debug)
    
    # Load config
    config = OmegaConf.load(args.config)
    logging.info(f"Loaded config from {args.config}")
    logging.info(OmegaConf.to_yaml(config))
    
    # Train
    train(config)


if __name__ == "__main__":
    main()

