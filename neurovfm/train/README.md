# Training

NeuroVFM supports three levels of training, each building on the previous:

1. **Encoder pretraining** -- self-supervised Vol-JEPA on unlabeled neuroimaging volumes
2. **Diagnostic head training** -- MIL classification on a frozen pretrained encoder
3. **Findings LLM training** -- supervised fine-tuning (SFT) of a vision-language model on a frozen encoder

All training uses PyTorch Lightning with DDP (levels 1-2) or FSDP (level 3). Configs live in `train/config/`.

## Quick Start

```bash
# Level 1: Pretrain encoder
python -m neurovfm.train.train --config train/config/pretrain.yaml

# Level 2: Train diagnostic heads (frozen encoder)
python -m neurovfm.train.train --config train/config/mil.yaml

# Level 3: Train findings LLM via SFT (frozen encoder)
python -m neurovfm.train.train_llm --config train/config/sft.yaml
```

## Data Preparation

All training levels expect a data root with the following layout:

```
/path/to/data_root/
  ├── metadata.json
  └── raw/
      ├── study_001/
      │   ├── volume1.nii.gz
      │   └── volume2.nii.gz
      ├── study_002/
      └── ...
```

### 1. Create metadata

Generate `metadata.json` from your data directory. Each study must be mapped to a modality (`mri` or `ct`):

```python
from pathlib import Path
from neurovfm.data import DatasetMetadata

data_root = Path("/path/to/data_root")
raw_dir = data_root / "raw"

mode_mapping = {d.name: "mri" for d in raw_dir.iterdir() if d.is_dir()}

metadata = DatasetMetadata.from_directory(data_root, mode_mapping)
metadata.save(data_root / "metadata.json")
```

See `examples/create_metadata.py` for a complete example.

### 2. Cache dataset (recommended)

Pre-tokenize volumes into `.pt` files for faster training:

```python
from neurovfm.data import CacheManager

cache = CacheManager("/path/to/data_root")
cache.build_cache(num_workers=8)
```

See `examples/cache_dataset.py` for a complete example. Once cached, set `use_cache: true` in your config.

## Directory Structure

```
train/
├── config/
│   ├── pretrain.yaml   # Vol-JEPA self-supervised pretraining
│   ├── mil.yaml        # MIL diagnostic head training
│   └── sft.yaml        # Vision-language model SFT
└── README.md
```

Training scripts themselves live in the `neurovfm` package:

```
neurovfm/train/
├── train.py            # Entry point for pretrain + MIL (Levels 1-2)
└── train_llm.py        # Entry point for SFT (Level 3)
```

## Level 1: Encoder Pretraining

Self-supervised pretraining of the 3D ViT encoder using Vol-JEPA (Joint Embedding Predictive Architecture). No labels required.

**Config:** `train/config/pretrain.yaml`

**Key config sections:**
- `system.which: VisionPretrainingSystem` -- selects the Vol-JEPA training loop
- `system.params.model_hyperparams.vision_backbone_cf` -- ViT architecture (default: ViT-B)
- `system.params.ema_beta` -- EMA momentum schedule for the target encoder
- `data.loader.train.collate_fn.patch_drop_rate` -- patch masking ratio for JEPA

**Run:**
```bash
python -m neurovfm.train.train --config train/config/pretrain.yaml
```

**Multi-GPU:**
Set `infra.num_gpus` and `infra.num_nodes` in the config. DDP is used automatically when `num_gpus > 1`.

## Level 2: Diagnostic Head Training

Trains a Classify-Then-Aggregate MIL pooler on top of a **frozen** pretrained encoder for study-level multi-label classification.

**Config:** `train/config/mil.yaml`

**Prerequisites:**
- A pretrained encoder checkpoint from Level 1
- A CSV with study-level binary labels (see `examples/data/mil_study_labels.csv` for format)

**Key config sections:**
- `system.which: VisionClassificationSystem` -- selects the classification training loop
- `system.params.model_hyperparams.pooler_cf` -- MIL pooler architecture
- `data.study_labels` -- path to your study labels CSV
- `training.load_backbone` -- (uncomment) path to pretrained encoder checkpoint and prefix to strip

**To use a pretrained encoder**, uncomment the `load_backbone` block at the bottom of the config:
```yaml
training:
  load_backbone:
    ckpt_path: /path/to/pretrained_checkpoint.ckpt
    remove_prefix: model.student.vision_encoder.
```

**Run:**
```bash
python -m neurovfm.train.train --config train/config/mil.yaml
```

## Level 3: Findings LLM Training (SFT)

Supervised fine-tuning of a multimodal LLM (e.g., Qwen3-8B) that generates radiology findings from neuroimaging volumes. The vision encoder is frozen; only the vision connector and language model are trained.

**Config:** `train/config/sft.yaml`

**Prerequisites:**
- A pretrained encoder (from Level 1, or use the released `mlinslab/neurovfm-encoder`)
- A JSON file mapping studies to conversations, e.g.:
  ```json
  {
    "study_001": [
      {"role": "user", "content": "Describe the findings."},
      {"role": "assistant", "content": "1. Normal brain parenchyma. 2. No acute intracranial abnormality."}
    ]
  }
  ```

**Key config sections:**
- `system.which: VisionInstructionTuningSystem` -- selects the SFT training loop
- `system.params.language_model_cf` -- language model (default: `Qwen/Qwen3-8B`)
- `system.params.vision_connector_cf` -- perceiver connector architecture
- `data.study_conversations` -- path to your conversations JSON

**To use a pretrained encoder or full VLM checkpoint**, uncomment the relevant block in the config:
```yaml
system:
  params:
    # Option A: Load just the encoder backbone
    load_pretrained_backbone:
      model_name_or_path: mlinslab/neurovfm-encoder

    # Option B: Load a full pretrained VLM (encoder + connector + LLM)
    load_pretrained_full:
      model_name_or_path: mlinslab/neurovfm-llm
```

**Run:**
```bash
python -m neurovfm.train.train_llm --config train/config/sft.yaml
```

SFT uses FSDP for multi-GPU training. Set `infra.num_gpus` (default: 8) in the config.

## Config Reference

All configs share a common structure:

| Section | Description |
|---------|-------------|
| `infra` | Experiment directory, seed, GPU count, logging |
| `data` | Data root, cache settings, dataset/loader params |
| `system` | Training system class and model/optimizer/scheduler params |
| `training` | Trainer params (epochs, precision, grad clip), checkpointing |

### Logging

- **TensorBoard** and **CSV** loggers are always enabled.
- **W&B**: Uncomment `infra.wandb_project` in any config to enable.

### Checkpointing

- Best model checkpoint is saved based on `training.monitor_metric`.
- Periodic checkpoints saved every `training.checkpoint_every_n_epochs` epochs.
- All checkpoints are saved to `{infra.exp_root}/models/`.

### Resuming Training

For Levels 1-2, uncomment the `resume_checkpoint` field:
```yaml
training:
  resume_checkpoint: /path/to/checkpoint.ckpt
```
