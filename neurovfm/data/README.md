# Neuroimaging Data Pipeline

Load and preprocess neuroimages (NIfTI, DICOM) for inference and training.

## Quick Start

### Inference (No Setup)

```python
from neurovfm.data import load_image, prepare_for_inference

# Load any image
img = load_image('scan.nii.gz')  # or .dcm or dicom_dir/

# Prepare for model
img_arrs, mask, view = prepare_for_inference(img, mode='mri')  # or mode='ct'

# Use with model
output = model(img_arrs[0])
```

### Training (With Cache)

**Step 1: Setup metadata**

```python
from neurovfm.data import DatasetMetadata

# Specify CT vs MRI for each study
mode_mapping = {
    'patient_001': 'mri',
    'patient_002': 'ct',
    # ... add all studies
}

# Create metadata
metadata = DatasetMetadata.from_directory('/data', mode_mapping)
metadata.save('/data/metadata.json')
```

**Step 2: Build cache (optional, speeds up training)**

```python
from neurovfm.data import CacheManager

cache = CacheManager('/data')
cache.build_cache(num_workers=8)
```

**Step 3: Train**

```python
from neurovfm.datasets import ImageDataset
from torch.utils.data import DataLoader

dataset = ImageDataset('/data', use_cache=True)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in loader:
    img = batch['img']  # [B, N, 1024] tokenized patches
    # Train model on token sequences
```

## Directory Structure

```
/data/
├── raw/                    # Your original images
│   ├── patient_001/
│   │   └── T1.nii.gz
│   └── patient_002/
│       └── ct_scan/        # DICOM series
├── processed/              # Auto-generated cache
│   ├── patient_001/
│   │   ├── T1.pt           # uint8 image data
│   │   └── T1_mask.pt      # uint8 mask
│   └── patient_002/
│       ├── ct_scan_brain.pt
│       ├── ct_scan_blood.pt
│       ├── ct_scan_bone.pt
│       └── ct_scan_mask.pt # Shared mask for all windows
└── metadata.json           # Study info (you create this)
```

## Data Format

### Image Tensors

**Token format** (N, D) where D=1024 (4×16×16 patches):
- **N**: Number of tokens (depth×height×width in token space)
- **D**: 1024 = 4 (slice) × 16 × 16 (in-plane) patch size

**Volume format** (when working directly with arrays from `load_image` / `prepare_for_inference` or `CacheManager.load_image`):
- **MRI**: `[1, D, H, W]` - single channel, normalized [0,1]
- **CT**: `[1, D, H, W]` - a single CT window view, normalized [0,1]

**Mask**: `[N]` - uint8, 1=background, 0=foreground (per token)

### Dimensions
- Token space: D//4, H//16, W//16
- Original: D = ~30-150 slices (4mm), H, W = divisible by 16 (1mm)

## Preprocessing Pipeline

**Image-level:**
1. Load (auto-detect NIfTI/DICOM)
2. Reorient to RPI
3. Detect slice dimension
4. Resample to 1×1×4mm
5. Crop to valid dimensions

**Array-level:**
- **CT**: Generate 3 windows (brain W=80/L=40, blood W=200/L=80, bone W=2800/L=600)
- **MRI**: Clip 0.5-99.5 percentile, min-max normalize

## API

### `load_image(path, preprocess=True)`
Load any medical image with automatic format detection.

### `prepare_for_inference(img_sitk, mode, z_dim=None)`
Convert SimpleITK image to model-ready arrays.
- **mode**: 'ct' or 'mri'  
- **Returns**: (img_arrs, mask, view) or None

### `DatasetMetadata`
Manage dataset metadata.
- `from_directory(data_dir, mode_mapping)`: Scan raw/ directory
- `save(filepath)`: Save to JSON

### `CacheManager`  
Build and manage preprocessing cache.
- `build_cache(num_workers, force)`: Process all images
- `load_image(study, image)`: Load cached data
- `get_cache_stats()`: Check progress

## Datasets

```python
from neurovfm.datasets import ImageDataset, StudyAwareBatchSampler
from torch.utils.data import DataLoader
```

### `ImageDataset`
Loads individual images with augmentation and tokenization.

```python
# Basic usage (returns tokenized format)
dataset = ImageDataset(
    data_dir='/data',
    use_cache=True,
    ct_window_probs=[0.7, 0.15, 0.15],  # brain/blood/bone sampling
    random_crop=False,                   # Random cropping
    augment=False,                       # Flips/permutations
    tokenize=True                        # Return (N, 1024) format
)

# Image-level training
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch in loader:
    img = batch['img']        # [B, N, 1024] tokenized
    coords = batch['coords']  # [B, N, 3] coordinates
    filt = batch['filtered']  # [B, N] background mask
    # Train model
```

**Output format:**
```python
{
    'img': [N, 1024],          # Tokenized image (4x16x16 patches)
    'coords': [N, 3],          # Token coordinates (d, h, w)
    'filtered': [N],           # Background mask (1=bg, 0=fg)
    'size': [3],               # Volume size in tokens (D, H, W)
    'path': str,               # Image path
    'label': int/str,          # Label
    'study': str,              # Study name
    'mode': str,               # 'ct' or 'mri'
    'window': str              # CT window or None
}
```

### `StudyAwareBatchSampler`
Groups images from same study into batches.

```python
# Study-level training (all images from study in same batch)
dataset = ImageDataset('/data', use_cache=True)
sampler = StudyAwareBatchSampler(
    dataset=dataset,
    batch_size=8,
    shuffle=True,
    seed=42
)

loader = DataLoader(dataset, batch_sampler=sampler)

for batch in loader:
    # All images in batch are from same study
    # Batch size varies based on images per study
    img = batch['img']        # [B, N, 1024]
    study = batch['study']    # All same study
    # Train multi-view model
```

**Benefits:**
- ✅ All study images on same GPU (important for multi-view learning)
- ✅ No manual batching logic needed
- ✅ DDP-aware (distributes studies, not images)

## Performance

| Method | Speed | Use Case |
|--------|-------|----------|
| Cached | ~0.01s/image | Training |
| On-the-fly | ~1-5s/image | Inference |

## Cache Storage Format

- **Images**: Stored as uint8 [0-255] for space efficiency (~4x smaller than float32)
- **Masks**: Stored as uint8 binary (1=background, 0=foreground)
- **Loading**: Auto-converted to float32 [0,1] when loaded

## Tips

✅ For **inference**: Just use `load_image()` + `prepare_for_inference()`  
✅ For **training**: Build cache once, train many times  
✅ Use **mode_mapping** to explicitly specify CT vs MRI (required)  
✅ **Masks**: 1=background, 0=foreground (inverted from typical conventions)  
✅ **CT**: One mask shared across all three windows  

## Example: Complete Training Setup

```python
# 1. Create metadata (once)
from neurovfm.data import DatasetMetadata

mode_mapping = {'study_001': 'mri', 'study_002': 'ct'}
metadata = DatasetMetadata.from_directory('/data', mode_mapping)
metadata.save('/data/metadata.json')

# 2. Build cache (once)
from neurovfm.data import CacheManager

cache = CacheManager('/data')
cache.build_cache(num_workers=8)

# 3A. Image-level training
from neurovfm.datasets import ImageDataset
from torch.utils.data import DataLoader

dataset = ImageDataset(
    '/data',
    use_cache=True,
    random_crop=True,
    augment=True
)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

for epoch in range(num_epochs):
    for batch in loader:
        img = batch['img']  # [4, N, 1024]
        outputs = model(img, batch['coords'], batch['filtered'])
        # ... training loop

# 3B. Study-level training (multi-view)
from neurovfm.datasets import ImageDataset, StudyAwareBatchSampler

dataset = ImageDataset('/data', use_cache=True)
sampler = StudyAwareBatchSampler(dataset, batch_size=8, shuffle=True)
loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # For DDP shuffling
    for batch in loader:
        # All images from same study
        outputs = model(batch['img'], batch['coords'])
        # ... training loop
```

## Study-Level Labels (Classification)

For supervised classification, provide study-level labels via CSV:

```csv
study_id,label_a,label_b,label_c
patient_001,1,0,1
patient_002,0,1,0
patient_003,1,1,0
```

**Usage:**

```python
from neurovfm.datasets import ImageDataset

# Binary classification (single label)
dataset = ImageDataset(
    '/data',
    study_labels='/data/labels.csv',  # CSV with study_id,label columns
    use_cache=True
)

# All images from same study get same label
batch = dataset[0]
print(batch['label'])  # Returns label for this study
print(batch['study'])  # Returns study name
```

**Supported formats:**
- **CSV** (recommended): First column = study_id, rest = binary labels
- **JSON**: `{"study_001": 0, "study_002": 1}` for single labels

**Label types:**
- Binary: Single 0/1 per study
- Multilabel: Multiple 0/1 columns per study
- Multiclass: Single integer (0, 1, 2, ...) per study

See `neurovfm/datasets/` for DataModule integration.
