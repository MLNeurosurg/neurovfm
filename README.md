# NeuroVFM

## Health system–scale learning achieves generalist neuroimaging models

[Preprint](https://neurovfm.mlins.org) / [Interactive Demo](https://neurovfm.mlins.org) / [Models](https://huggingface.co/collections/mlinslab/neurovfm) / [MLiNS Lab](https://mlins.org)

**NeuroVFM** is a health system-scale, volumetric foundation model for multimodal neuroimaging, trained with self-supervision on **5.24M** MRI/CT volumes (**567k** studies) spanning **20+ years** of routine clinical care at Michigan Medicine. 

The NeuroVFM stack includes:

- **3D ViT encoder**, general-purpose representations for *any* clinical neuroimage (T1, T2, FLAIR, DWI, CT, etc.)
- **Study-level diagnostic heads**, covering **74 MRI**/**82 CT** expert-defined diagnoses for *any* neuroimaging study
- **Findings LLM**, generate preliminary findings given *any* neuroimaging study plus clinical context

> **Research use only.** Not a medical device. Do not use for clinical decision-making.

## 🔎 TL;DR (what NeuroVFM gives you)

```python
from neurovfm import load_encoder, load_diagnostic_head, load_llm

encoder, preproc = load_encoder("neurovfm-encoder")
dx_head = load_diagnostic_head("neurovfm-dx-mri")
findings_llm = load_llm("neurovfm-llm")

vol = preproc.load_study("/path/to/study/")         # study directory with 1+ DICOM/NIfTI files
emb = encoder.embed(vol)                            # series-wise embeddings
dx = dx_head.predict_proba(emb, top_k=3)            # top-3 diagnoses
report = findings_llm.generate_findings(
    emb,
    clinical_context="72F with acute right-sided weakness and aphasia",
)
```

```console
Top-3 MRI diagnoses
1. acute_ischemic_stroke               p=0.94
2. small_vessel_ischemic_disease       p=0.73
3. cerebral_atrophy                    p=0.67

Generated findings (excerpt)
Acute left MCA territory infarct with diffusion restriction in the left frontal,
insular, and parietal cortex with corresponding ADC hypointensity. No hemorrhagic
transformation. Chronic small vessel ischemic changes are present...
```