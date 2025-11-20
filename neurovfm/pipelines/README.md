# Inference Pipelines

High-level APIs for loading models and running inference on neuroimaging studies.

## Overview

The pipelines module provides three main components:

1. **EncoderPipeline**: Loads pretrained Vision Transformer encoder and generates token-level embeddings
2. **DiagnosticHead**: Pools token embeddings and applies classification head for diagnosis
3. **StudyPreprocessor**: Loads and preprocesses neuroimaging studies (NIfTI/DICOM)

## Quick Start

```python
from neurovfm.pipelines import load_encoder, load_diagnostic_head

# Load models
encoder, preprocessor = load_encoder("mlinslab/neurovfm-encoder")
dx_head = load_diagnostic_head("mlinslab/neurovfm-dx-ct")

# Load study
batch = preprocessor.load_study("/path/to/ct/study/", modality="ct")

# Generate embeddings and predictions
embeddings = encoder.embed(batch)  # Token-level embeddings
predictions = dx_head.predict(embeddings, batch)  # [(label, prob, pred), ...]

# Print results
for label, prob, pred in predictions:
    print(f"{label}: {prob:.3f} ({'positive' if pred else 'negative'})")
```

## Architecture Flow

```
Neuroimages (NIfTI/DICOM)
    ↓
[StudyPreprocessor] → Tokenized Volumes
    ↓
[EncoderPipeline] → Token-Level Embeddings [N, D]
    ↓
[DiagnosticHead]
    ├─ Pooler (AB-MIL) → Study-Level Embedding [1, D]
    └─ Classifier → Predictions [(label, prob, pred), ...]
```

## API Reference

### `load_encoder(model_name_or_path, device=None)`

Load pretrained encoder from HuggingFace Hub or local checkpoint.

**Args:**
- `model_name_or_path` (str): HuggingFace model ID (e.g., "mlinslab/neurovfm-encoder") or local path
- `device` (str, optional): Device to load model on

**Returns:**
- `Tuple[EncoderPipeline, StudyPreprocessor]`

### `EncoderPipeline.embed(batch, use_amp=True)`

Generate token-level embeddings for a batch of volumes.

**Args:**
- `batch` (Dict): Batch from StudyPreprocessor
- `use_amp` (bool): Use automatic mixed precision

**Returns:**
- `torch.Tensor`: Token-level embeddings [N_total, D]

### `load_diagnostic_head(model_name_or_path, device=None)`

Load pretrained diagnostic head from HuggingFace Hub or local checkpoint.

**Args:**
- `model_name_or_path` (str): HuggingFace model ID or local path
- `device` (str, optional): Device to load model on

**Returns:**
- `DiagnosticHead`

### `DiagnosticHead.predict(embeddings, batch, use_amp=True)`

Generate predictions from token-level embeddings.

**Args:**
- `embeddings` (torch.Tensor): Token-level embeddings [N_total, D]
- `batch` (Dict): Batch containing study-level metadata
- `use_amp` (bool): Use automatic mixed precision

**Returns:**
- `List[Tuple[str, float, int]]`: List of (label_name, probability, prediction)

### `StudyPreprocessor.load_study(study_path, modality)`

Load and preprocess a study (one or more volumes).

**Args:**
- `study_path` (Union[str, Path, List]): Path to study directory or list of image paths
- `modality` (str): Modality type ("ct" or "mri")

**Returns:**
- `Dict`: Batch dictionary with tokens, coords, cu_seqlens, etc.

**Background Filtering:**
By default (`remove_background=True`), background tokens are physically removed during preprocessing. This improves inference efficiency and memory usage. The encoder receives only foreground tokens and `series_masks_indices` is empty.

Alternatively, with `remove_background=False`, all tokens are kept and the encoder is applied to both foreground and background tokens. In the current high-level inference pipeline, `series_masks_indices` is not populated by `StudyPreprocessor`; index-based masking is instead used in the training data pipeline (see `MultiViewCollator` in `neurovfm.datasets`).

## Examples

### Basic Inference

```python
from neurovfm.pipelines import load_encoder, load_diagnostic_head

# Load models
encoder, preprocessor = load_encoder("mlinslab/neurovfm-encoder")
dx_head = load_diagnostic_head("mlinslab/neurovfm-dx-ct")

# Run inference
batch = preprocessor.load_study("/path/to/study/", modality="ct")
embeddings = encoder.embed(batch)
predictions = dx_head.predict(embeddings, batch)

print(predictions)
# [('hemorrhage', 0.924, 1), ('fracture', 0.132, 0), ('mass', 0.781, 1)]
```

### Batch Processing

```python
study_dirs = ["/data/patient_001/", "/data/patient_002/"]

all_predictions = []
for study_dir in study_dirs:
    batch = preprocessor.load_study(study_dir, modality="ct")
    embeddings = encoder.embed(batch)
    preds = dx_head.predict(embeddings, batch)
    all_predictions.append(preds)
```

### Load Multiple Images

```python
# Load specific images as a single study
image_paths = [
    "/path/to/scan1.nii.gz",
    "/path/to/scan2.nii.gz",
]
batch = preprocessor.load_study(image_paths, modality="mri")
embeddings = encoder.embed(batch)
```

### Extract Embeddings Only

```python
# Get embeddings without classification
encoder, preprocessor = load_encoder("mlinslab/neurovfm-encoder")
batch = preprocessor.load_study("/path/to/study/", modality="ct")
embeddings = encoder.embed(batch)  # [N_total, D]
```

### Local Checkpoints

```python
# Load from local path instead of HuggingFace
encoder, preprocessor = load_encoder("/path/to/local/checkpoint/")
dx_head = load_diagnostic_head("/path/to/local/classifier/")
```

## Preprocessing

The preprocessing pipeline follows these steps:
1. **Load**: Automatic format detection (NIfTI/DICOM)
2. **Reorient**: Standardize to RPI anatomical orientation
3. **Resample**: To target spacing (default: 1×1×4mm)
4. **Transpose**: Move acquisition axis to first dimension
5. **Center Crop**: To dimensions divisible by patch size (4×16×16)
6. **Normalize**: Modality-specific (CT windowing / MRI scaling)
7. **Tokenize**: Into patches (default: 4×16×16 → 1024-D tokens)

**Key Point**: There is NO fixed output size. Dimensions are determined by the original image size after resampling, then center-cropped to the nearest multiple of the patch size. This preserves the original anatomical extent while ensuring patch alignment.

Customize preprocessing:
```python
preprocessor = StudyPreprocessor(
    patch_size=(4, 16, 16),  # Patch size for tokenization
    target_spacing=(1.0, 1.0, 4.0),  # Resampling target (in-plane, through-plane)
    remove_background=True,  # Remove background tokens (default: True)
)
```

### Background Token Filtering

Neuroimages contain significant background (air, outside patient) that provides no useful information. The preprocessing pipeline automatically detects and handles background:

1. **Background Detection**: 
   - **CT**: HU threshold
   - **MRI**: Percentile-based intensity thresholding

2. **Token-Level Filtering**:
   - After tokenization, each patch is marked as background if ANY pixel in it is background
   - This ensures no information loss at patch boundaries

3. **Removal Options**:
   - **`remove_background=True` (default)**: Physically remove background tokens before encoder; encoder processes only foreground tokens
     - Pros: Faster inference, lower memory usage
     - Cons: None for inference
   - **`remove_background=False`**: Keep all tokens; encoder processes both foreground and background tokens (no mask indices are passed in this pipelines API)
     - Pros: Preserves full spatial structure and raw background context
     - Cons: Slower, uses more memory

For inference, **always use `remove_background=True`** (default) for optimal performance.

## Supported Modalities

- **CT**: Automatic windowing (brain, blood, bone)
- **MRI**: Intensity normalization

Specify modality when loading:
```python
batch = preprocessor.load_study("/path/to/study/", modality="ct")
batch = preprocessor.load_study("/path/to/study/", modality="mri")
```

## Performance

- **GPU**: Recommended for production use
- **AMP**: Automatic mixed precision enabled by default (bfloat16)
- **Batch Size**: Single study per batch (multiple series within study)

## Troubleshooting

**Error: "No module named 'huggingface_hub'"**
```bash
pip install huggingface_hub
```

**Error: "CUDA out of memory"**
- Reduce number of series in study
- Use CPU inference: `load_encoder(..., device="cpu")`
- Enable mixed precision (default)

**Error: "No valid image files found"**
- Ensure study directory contains .nii.gz, .nii, or .dcm files
- Or pass explicit list of image paths

## See Also

- [Training Guide](../../TRAINING.md)
- [Quick Start](../../QUICKSTART.md)
- [Data Preprocessing](../data/README.md)

