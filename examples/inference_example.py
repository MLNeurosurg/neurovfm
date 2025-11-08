"""
Example: Running Inference with NeuroVFM

This example demonstrates how to:
1. Load a pretrained encoder
2. Load a diagnostic head
3. Run inference on a medical imaging study
"""

import torch
from neurovfm.pipelines import load_encoder, load_diagnostic_head

# ============================================================================
# Example 1: Basic Inference
# ============================================================================

# Load encoder and preprocessor
encoder, preprocessor = load_encoder("mlinslab/neurovfm-encoder")

# Load diagnostic head
dx_head = load_diagnostic_head("mlinslab/neurovfm-ctbleed-classifier")

# Load and preprocess a study (CT scan)
study_path = "/path/to/ct/study/"
batch = preprocessor.load_study(study_path, modality="ct")

# Generate token-level embeddings
embeddings = encoder.embed(batch)  # [N_total, D]

# Run diagnostic prediction (pooling + classification)
predictions = dx_head.predict(embeddings, batch)

# Print results
print("Diagnostic Predictions:")
for label, prob, pred in predictions:
    print(f"  {label}: {prob:.3f} ({'positive' if pred else 'negative'})")

# Output:
#   hemorrhage: 0.924 (positive)
#   fracture: 0.132 (negative)
#   mass: 0.781 (positive)


# ============================================================================
# Example 2: Batch Processing Multiple Studies
# ============================================================================

from pathlib import Path

study_dirs = [
    "/data/studies/patient_001/",
    "/data/studies/patient_002/",
    "/data/studies/patient_003/",
]

all_predictions = []
for study_dir in study_dirs:
    # Load study
    batch = preprocessor.load_study(study_dir, modality="ct")
    
    # Encode
    embeddings = encoder.embed(batch)
    
    # Predict
    preds = dx_head.predict(embeddings, batch)
    all_predictions.append((Path(study_dir).name, preds))

# Save results
import json
results = {
    study: [{"label": label, "prob": prob, "pred": pred} for label, prob, pred in preds]
    for study, preds in all_predictions
}
with open("predictions.json", "w") as f:
    json.dump(results, f, indent=2)


# ============================================================================
# Example 3: Using Custom Checkpoint (Local)
# ============================================================================

# Load from local checkpoint instead of HuggingFace
encoder, preprocessor = load_encoder("/path/to/local/checkpoint/")

# Load multiple images as a single study
image_paths = [
    "/path/to/scan1.nii.gz",
    "/path/to/scan2.nii.gz",
]
batch = preprocessor.load_study(image_paths, modality="mri")

embeddings = encoder.embed(batch)


# ============================================================================
# Example 4: Embeddings for Downstream Tasks
# ============================================================================

# Extract embeddings without classification
encoder, preprocessor = load_encoder("mlinslab/neurovfm-encoder")

batch = preprocessor.load_study("/path/to/study/", modality="ct")
embeddings = encoder.embed(batch)  # [N_total, D]

# Use embeddings for:
# - Similarity search
# - Clustering
# - Custom downstream models
# - Visualization (t-SNE, UMAP)

print(f"Extracted {len(embeddings)} token-level embeddings with dimension {embeddings.shape[1]}")


# ============================================================================
# Example 5: Custom Preprocessing Parameters
# ============================================================================

from neurovfm.pipelines.preprocessor import StudyPreprocessor

# Create preprocessor with custom parameters
preprocessor = StudyPreprocessor(
    patch_size=(4, 16, 16),  # Patch size for tokenization
    target_spacing=(1.0, 1.0, 4.0),  # Resampling target (mm)
)

# Load study - dimensions will be determined by original image size
# after resampling, then cropped to nearest multiple of patch_size
batch = preprocessor.load_study("/path/to/study/", modality="ct")

print(f"Study dimensions: {batch['size']}")  # Shows actual processed sizes
print(f"Total tokens: {len(batch['img'])}")  # Varies based on image size


# ============================================================================
# Example 6: Custom Normalization Stats
# ============================================================================

from neurovfm.pipelines.encoder import EncoderPipeline
from neurovfm.models import get_vit_backbone

# Build model manually
model = get_vit_backbone(
    patch_size=(4, 16, 16),
    embed_dim=768,
    depth=12,
    num_heads=12,
)

# Load weights
state_dict = torch.load("/path/to/weights.pth")
model.load_state_dict(state_dict)

# Create encoder with custom normalization
custom_stats = [
    [0.32, 0.41, 0.32, 0.27],  # Custom means [mri, brain, blood, bone]
    [0.26, 0.41, 0.36, 0.19],  # Custom stds
]
encoder = EncoderPipeline(
    model=model,
    normalization_stats=custom_stats,
    device="cuda"
)

# Use as normal
preprocessor = StudyPreprocessor()
batch = preprocessor.load_study("/path/to/study/", modality="ct")
embeddings = encoder.embed(batch)

