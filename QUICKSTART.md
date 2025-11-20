# NeuroVFM Quick Start Guide

Get started with NeuroVFM in 5 minutes.

## Installation

```bash
cd /path/to/neurovfm
pip install -e .
```

## Inference (Using Pretrained Models)

### Basic Usage

```python
from neurovfm.pipelines import load_encoder, load_diagnostic_head

# Load pretrained models from HuggingFace
encoder, preprocessor = load_encoder("mlinslab/neurovfm-encoder")
dx_head = load_diagnostic_head("mlinslab/neurovfm-dx-ct")

# Load and preprocess study
batch = preprocessor.load_study("/path/to/ct/study/", modality="ct")

# Generate embeddings and predictions
embeddings = encoder.embed(batch)
predictions = dx_head.predict(embeddings, batch)

# Print results
for label, prob, pred in predictions:
    print(f"{label}: {prob:.3f} ({'positive' if pred else 'negative'})")
```

### Load from Local Checkpoint

```python
# Load from local path instead of HuggingFace
encoder, preprocessor = load_encoder("/path/to/local/checkpoint/")
dx_head = load_diagnostic_head("/path/to/local/classifier/")
```

See [Inference API](INFERENCE_API.md) and [examples/inference_example.py](examples/inference_example.py) for more details.

---

## Training Your Own Models

### Minimal Training Example

#### 1. Prepare Data Structure

```bash
/path/to/data/
├── study_001/
│   ├── scan1.nii.gz
│   └── scan2.nii.gz
├── study_002/
│   └── scan1.dcm
└── ...
```

#### 2. Create Minimal Config

`config.yaml`:
```yaml
infra:
  exp_root: ./experiments/test_run
  seed: 42
  num_gpus: 1
  num_nodes: 1

data:
  raw_data_dir: /path/to/data
  cache_dir: ./cache
  
  mode_mapping:
    study_001: mri
    study_002: ct
  
  dataset:
    train:
      studies: [study_001]
      params:
        tokenize: true
        ct_window_probs: [0.7, 0.15, 0.15]
    val:
      studies: [study_002]
      params:
        tokenize: true
        ct_window_probs: [0.7, 0.15, 0.15]
  
  loader:
    train:
      batch_size: 2
      num_workers: 4
      shuffle: true
      use_study_sampler: false
      collate_fn:
        remove_background: true
    val:
      batch_size: 2
      num_workers: 4
      shuffle: false
      use_study_sampler: false
      collate_fn:
        remove_background: true

system:
  which: VisionClassificationSystem
  params:
    model_hyperparams:
      vision_backbone_cf:
        which: vit_small  # Small model for testing
        params:
          token_dim: 1024
      pooler_cf:
        which: avgpool
      proj_params:
        out_dim: 1
    loss_cf:
      which: bce
    opt_cf:
      which: adamw
      params:
        lr: 1e-4
    schd_cf:
      which: cosine
      params:
        ipe_scale: 1.0
    enable_aug: false

training:
  trainer_params:
    max_epochs: 5
    precision: 32
  monitor_metric: val/loss_epoch
  monitor_mode: min
```

#### 3. Run Training

```bash
python -m neurovfm.train.train --config config.yaml
```

## What Happens

1. **Data Loading**: Scans `raw_data_dir`, creates metadata
2. **Preprocessing**: Converts to standard format, caches to `cache_dir`
3. **Training**: Loads batches, trains model, saves checkpoints
4. **Monitoring**: Logs to TensorBoard in `exp_root/tb/`

## View Results

```bash
tensorboard --logdir ./experiments/test_run/tb
```

## Next Steps

### Scale Up

1. **Add more data**:
```yaml
dataset:
  train:
    studies: [study_001, study_002, ..., study_100]
```

2. **Use larger model**:
```yaml
vision_backbone_cf:
  which: vit_base  # or vit_large
```

3. **Enable augmentations**:
```yaml
dataset:
  params:
    augment: true
system:
  params:
    enable_aug: true
```

4. **Multi-GPU training**:
```yaml
infra:
  num_gpus: 4
```

### Self-Supervised Pretraining

Replace `system.which` with:
```yaml
system:
  which: VisionPretrainingSystem
  params:
    vision_backbone_cf:
      which: vit_base
    predictor_cf:
      dim: 768
      depth: 12
      dim_head: 64
      num_heads: 12
      prefix_len: 0
    ema_beta: [0.994, 1.0]
    loss_cf:
      which: jepa
```

Then fine-tune:
```yaml
training:
  load_backbone:
    ckpt_path: ./experiments/pretrain/models/best.ckpt
    remove_prefix: student_encoder.
```

## Common Issues

### OOM Error
- Reduce `batch_size`
- Use `vit_small` instead of `vit_base`
- Enable mixed precision: `precision: bf16-mixed`

### Slow Loading
- Preprocessing happens on first run (cached after)
- Increase `num_workers`
- Use SSD for `cache_dir`

### Poor Performance
- Check `mode_mapping` is correct
- Enable augmentations
- Increase `max_epochs`

## Full Documentation

- **Training Guide**: See `TRAINING.md`
- **Data Pipeline**: See `neurovfm/data/README.md`
- **API Reference**: See docstrings in code
- **Examples**: See `neurovfm/train/config/` directory

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                 Training Pipeline                    │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Data: ImageDataset + ImageDataModule              │
│  • Loads from cache or raw files                    │
│  • Applies augmentations                            │
│  • Tokenizes to [N, 1024] patches                   │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Collation: MultiViewCollator                       │
│  • Batches multiple images/studies                  │
│  • Filters background                               │
│  • Tracks sequence lengths                          │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Normalization: NormalizationModule                 │
│  • Modality-specific (MRI/CT brain/blood/bone)      │
│  • Optional intensity augmentations                 │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Model: VisionTransformer                           │
│  • Encodes patches to features                      │
│  • Flash attention for efficiency                   │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Task Head                                          │
│  • Pretraining: VisionPredictor → Predict masked    │
│  • Classification: AB-MIL → Classify                │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  Loss & Optimization                                │
│  • JEPA (pretraining) or BCE/CE (supervised)  │
│  • AdamW + Cosine LR schedule                       │
└─────────────────────────────────────────────────────┘
```
