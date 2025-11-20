# Training Guide

This guide explains how to train vision models with NeuroVFM, covering both self-supervised pretraining and supervised classification.

## Quick Start

### 1. Prepare Your Data

Organize your data by study:
```
/path/to/raw_data/
├── study_001/
│   ├── scan_001.nii.gz
│   ├── scan_002.nii.gz
│   └── ...
├── study_002/
│   ├── scan_001.dcm
│   ├── scan_002.dcm
│   └── ...
└── ...
```

### 2. Create Mode Mapping

Specify modality for each study in your config:
```yaml
data:
  mode_mapping:
    study_001: mri
    study_002: ct
    study_003: mri
    # ... add all studies
```

**Important**: For CT, the window type (brain/blood/bone) is automatically detected from the filename or path. Ensure your CT files contain one of these keywords:
- `brain` or `BrainWindow` → CT Brain Window
- `blood` or `BloodWindow` → CT Blood Window
- `bone` or `BoneWindow` → CT Bone Window

### 3. Self-Supervised Pretraining

Train a vision transformer backbone using masked prediction:

```bash
cd neurovfm/train
python train.py -c config/pretraining_example.yaml
```

Key config sections:
```yaml
system:
  which: VisionPretrainingSystem
  params:
    vision_backbone_cf:
      which: vit_base  # vit_tiny, vit_small, vit_base, vit_large
    loss_cf:
      which: jepa
```

### 4. Supervised Classification

Fine-tune the pretrained backbone for classification:

**Step 1: Create study labels CSV**
```csv
study_id,hemorrhage,fracture,mass
patient_001,1,0,1
patient_002,0,1,0
patient_003,1,1,0
```

**Step 2: Update config with labels path**
```yaml
data:
  data_dir: /path/to/data
  study_labels: /path/to/study_labels.csv  # Study-level labels
  # ... rest of data config

system:
  which: VisionClassificationSystem
  params:
    model_hyperparams:
      pooler_cf:
        which: classify_then_aggregate  # or aggregate_then_classify, both AB-MIL-based
      proj_params:
        out_dim: 3  # Number of labels (for multilabel)
    loss_cf:
      which: bce  # bce (binary/multilabel), ce (multiclass)

training:
  load_backbone:
    ckpt_path: /path/to/pretraining_checkpoint.ckpt
    remove_prefix: student_encoder.
```

**Step 3: Train**
```bash
cd neurovfm/train
python train.py -c config/classification_example.yaml
```

**Note**: All images from the same study automatically receive the same label. The model uses MIL pooling to aggregate series-level features into a single study-level prediction.

## Architecture Overview

### Data Pipeline

```
Raw Data (DICOM/NIfTI)
    ↓
[DatasetMetadata] → Scan files, detect studies
    ↓
[CacheManager] → Preprocess & cache as .pt files
    ↓
[ImageDataset] → Load, augment, tokenize
    ↓
[StudyAwareBatchSampler] → Group by study (optional)
    ↓
[MultiViewCollator] → Batch, filter background
    ↓
[DataLoader] → Mini-batches
```

### Model Pipeline

**Pretraining (Self-Supervised)**:
```
Tokenized Images [N, 1024]
    ↓
[VisionTransformer] → Encode context patches
    ↓
[VisionPredictor] → Predict masked patches
    ↓
[JEPA Loss] → Compare with teacher
```

**Classification (Supervised)**:
```
Tokenized Images [N, 1024]
    ↓
[VisionTransformer] (frozen) → Extract features
    ↓
[AggregateThenClassify/ClassifyThenAggregate] → Pool to study-level
    ↓
[MLP] → Classification head
    ↓
[BCE/CE Loss] → Compare with labels
```

## Key Features

### 1. Modality-Specific Normalization

The system automatically applies different normalization for:
- **MRI**: Mean/std normalization
- **CT Brain Window**: Specific mean/std for brain window
- **CT Blood Window**: Specific mean/std for blood window
- **CT Bone Window**: Specific mean/std for bone window

Window type is detected from the file path (must contain `brain`, `blood`, or `bone` keywords).

### 2. Data Augmentation

**Spatial** (applied in `ImageDataset`):
- Random cropping with foreground detection
- Random flips (left-right, anterior-posterior, superior-inferior)
- Random axis permutations

**Intensity** (applied in `NormalizationModule`):
- Gamma correction
- Brightness/contrast adjustment
- Gaussian noise (post-normalization)

Enable with:
```yaml
dataset:
  params:
    augment: true  # Spatial augmentations

system:
  params:
    enable_aug: true  # Intensity augmentations
```

### 3. CT Window Sampling

For training with CT images, each scan has 3 windows (brain/blood/bone). Control sampling probability:

```yaml
dataset:
  params:
    ct_window_probs: [0.7, 0.15, 0.15]  # brain, blood, bone
```

**ImageDataset**: Samples one window per scan (probabilistic)
**StudyAwareBatchSampler**: Loads all three windows for study-level tasks

### 4. Background Filtering

Remove background tokens to focus on anatomy:

```yaml
loader:
  collate_fn:
    remove_background: true
    patch_drop_rate: 0.1  # Optional: drop 10% of foreground patches
```

### 5. Efficient Storage

Preprocessed cache stores:
- Images as **uint8** [0-255] for 4x space savings
- Background masks as separate binary files
- CT windows as individual files

## Configuration Reference

### Data Configuration

```yaml
data:
  raw_data_dir: str              # Path to raw DICOM/NIfTI files
  cache_dir: str                 # Path to save preprocessed .pt files
  mode_mapping: Dict[str, str]   # study_name -> modality
  
  dataset:
    train/val:
      studies: List[str]         # Study names to include
      params:
        random_crop: bool        # Enable random cropping
        max_crop_size: [D,H,W]   # Max crop size in tokens
        augment: bool            # Enable spatial augmentations
        tokenize: bool           # Return tokenized format (vs raw)
        ct_window_probs: [f,f,f] # Sampling probs for brain/blood/bone
  
  loader:
    train/val:
      batch_size: int
      num_workers: int
      use_study_sampler: bool    # Use StudyAwareBatchSampler
      collate_fn:
        remove_background: bool
        patch_drop_rate: float
```

### System Configuration

**Pretraining**:
```yaml
system:
  which: VisionPretrainingSystem
  params:
    vision_backbone_cf:
      which: vit_base            # vit_tiny/small/base/large/huge
    predictor_cf:
      dim: int                   # Predictor dimension
      depth: int                 # Number of layers
    ema_beta: [0.994, 1.0]       # EMA momentum range [start, end]
    opt_cf:
      which: adamw               # sgd, adam, adamw
      params:
        lr: float
        weight_decay: float
    schd_cf:
      which: cos_linear_warmup   # step_lr, cos_warm_restart, cos_linear_warmup
      params:
        num_warmup_steps: float
        num_cycles: float
        ipe_scale: float
```

**Classification**:
```yaml
system:
  which: VisionClassificationSystem
  params:
    model_hyperparams:
      pooler_cf:
        which: classify_then_aggregate     # or aggregate_then_classify
      proj_params:
        out_dim: int             # Number of output classes
    loss_cf:
      which: bce                 # bce, ce, mse, mae, rmse
    enable_aug: bool             # Enable intensity augmentations
```

### Training Configuration

```yaml
training:
  trainer_params:
    max_epochs: int
    gradient_clip_val: float
    precision: str               # bf16-mixed, 16-mixed, 32
  
  monitor_metric: str            # Metric to monitor
  monitor_mode: str              # min or max
  checkpoint_every_n_epochs: int
  
  load_backbone:                 # Load pretrained backbone
    ckpt_path: str
    remove_prefix: str           # Prefix to remove from keys
  
  resume_checkpoint: str         # Resume training from checkpoint
```

## Multi-GPU Training

The training script automatically detects available GPUs and uses DDP:

```yaml
infra:
  num_gpus: 4
  num_nodes: 1
```

For SLURM clusters:
```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

cd neurovfm/train && srun python train.py -c config/pretraining_example.yaml
```

## Monitoring

Logs are saved to:
- **TensorBoard**: `exp_root/tb/`
- **CSV**: `exp_root/csv/`
- **Wandb** (optional): Set `infra.wandb_project`

View TensorBoard:
```bash
tensorboard --logdir /path/to/exp_root/tb
```

## Checkpointing

Checkpoints are saved to `exp_root/models/`:
- `best-{metric}.ckpt`: Best model based on monitor_metric
- `last.ckpt`: Most recent checkpoint
- `epoch-XX.ckpt`: Periodic checkpoints

Resume training:
```yaml
training:
  resume_checkpoint: /path/to/checkpoint.ckpt
```

Load for fine-tuning:
```yaml
training:
  load_backbone:
    ckpt_path: /path/to/pretrained.ckpt
    remove_prefix: student_encoder.
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size
2. Enable gradient accumulation
3. Use smaller model (vit_small instead of vit_base)
4. Reduce max_crop_size
5. Enable mixed precision (`precision: bf16-mixed`)

### Slow Training

1. Increase num_workers
2. Preprocess and cache data first
3. Use study_sampler for better GPU utilization
4. Enable background filtering to reduce tokens

### Poor Performance

1. Check normalization stats are correct
2. Verify mode_mapping is accurate
3. Enable augmentations
4. Increase training epochs
5. Adjust learning rate

## Example Workflows

### Workflow 1: Pretrain from Scratch

```bash
# 1. Preprocess data
cd neurovfm/train && python train.py -c config/pretraining_example.yaml

# 2. Monitor with tensorboard
tensorboard --logdir /path/to/exp_root/tb

# 3. Training will save checkpoints to exp_root/models/
```

### Workflow 2: Classification with Frozen Backbone

```bash
# 1. Create classification config with load_backbone pointing to pretrained checkpoint
# 2. Run training
cd neurovfm/train && python train.py -c config/classification_example.yaml
```

### Workflow 3: Multi-stage Pretraining

```bash
# 1. Pretrain on large unlabeled dataset
cd neurovfm/train && python train.py -c config/pretrain_stage1.yaml

# 2. Continue pretraining on domain-specific data
# Set resume_checkpoint in config
python train.py -c config/pretrain_stage2.yaml
```

## Advanced Topics

### Custom Model Architectures

Modify `vision_backbone_cf` for custom architectures:
```yaml
vision_backbone_cf:
  which: vit  # Use "vit" for fully custom
  params:
    embed_dim: 1024
    depth: 24
    num_heads: 16
    # ... custom params
```

### Custom Loss Functions

Extend `neurovfm.losses` and update the system to use your loss.

### Multi-modal Learning

Combine MRI and CT by:
1. Setting appropriate mode_mapping
2. Enabling study_sampler to mix modalities in batches
3. The model automatically handles different normalizations

