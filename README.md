# NeuroVFM

## Health system–scale learning achieves generalist neuroimaging models

[Preprint](https://neurovfm.mlins.org) / [Interactive Demo](https://neurovfm.mlins.org) / [Models](https://huggingface.co/collections/mlinslab/neurovfm) / [MLiNS Lab](https://mlins.org)

**NeuroVFM** is a health system-scale, volumetric foundation model for multimodal neuroimaging, trained with self-supervision on **5.24M** MRI/CT volumes (**567k** studies) spanning **20+ years** of routine clinical care at Michigan Medicine. 

The NeuroVFM stack includes:

- **3D ViT encoder**, general-purpose representations for *any* clinical neuroimage (T1, T2, FLAIR, DWI, CT, etc.)
- **Study-level diagnostic heads**, covering **74 MRI**/**82 CT** expert-defined diagnoses for *any* neuroimaging study
- **Findings LLM**, generate preliminary findings given *any* neuroimaging study plus clinical context

> **Research use only.** Not a medical device. Do not use for clinical decision-making.

## Why NeuroVFM?

- **Health system-scale learning**: 
- **Single backbone, many tasks**: 
- **Robust to modality, protocol, vendor, and site effects**:
- **Flexible findings generation**:
- **Open weights and tooling**: 
