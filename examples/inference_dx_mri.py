import torch
from neurovfm.pipelines import load_encoder, load_diagnostic_head

# 1. Load encoder + preprocessor from your HF repo
encoder, preproc = load_encoder(
    "mlinslab/neurovfm-encoder",   # your repo ID
    device="cuda" if torch.cuda.is_available() else "cpu",
)
dx_head = load_diagnostic_head(
    "mlinslab/neurovfm-dx-mri",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# 2. Preprocess a study (directory or single file)
batch = preproc.load_study("/path/to/study/", modality="mri")

# 3. Get token-level representations
embs = encoder.embed(batch)  # shape: [N_tokens, D]
preds = dx_head(embs, batch)