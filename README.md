# NeuroVFM

## Health system learning achieves generalist neuroimaging models

[**Preprint**](https://arxiv.org/abs/2511.18640) / [**Interactive Demo**](https://neurovfm.mlins.org) / [**Models**](https://huggingface.co/collections/mlinslab/neurovfm) / [**MLiNS Lab**](https://mlins.org)

**NeuroVFM** is a health system–scale, volumetric foundation model for multimodal neuroimaging, trained with self-supervision on **5.24M** MRI/CT volumes (**567k** studies) spanning **20+ years** of routine clinical care at Michigan Medicine. 

The NeuroVFM stack includes:

- **3D ViT encoder**, general-purpose representations for *any* clinical neuroimage (T1, T2, FLAIR, DWI, CT, etc.)
- **Study-level diagnostic heads**, covering **74 MRI**/**82 CT** expert-defined diagnoses for *any* neuroimaging study
- **Findings LLM**, generates preliminary findings given *any* neuroimaging study plus clinical context
- **Reasoning API**, pass outputs to a frontier reasoning model for higher-level tasks (e.g., triage)

> **Research use only.** Not a medical device. Do not use for clinical decision-making.

## 🔎 TL;DR (what NeuroVFM gives you)

Study-wise (multi-sequence) embeddings with diagnostic predictions:

```python
from neurovfm import load_encoder, load_diagnostic_head

encoder, preproc = load_encoder("mlinslab/neurovfm-encoder")
dx_head = load_diagnostic_head("mlinslab/neurovfm-dx-ct")

vols = preproc.load_study("/path/to/study/")             # study directory with 1+ DICOM/NIfTI files   
embs = encoder.embed(vols)                               # series-wise embeddings   

dx = dx_head.predict_proba(embs, top_k=3)
```

```console
>>> dx
1. aneurysmal_subarachnoid_hemorrhage       p=0.96
2. mass_effect                              p=0.91
3. obstructive_hydrocephalus                p=0.74
```

Radiological findings generation:

```python
from neurovfm import load_vlm, interpret_findings

generator, preproc = load_vlm("mlinslab/neurovfm-llm")

vols = preproc.load_study("/path/to/study/")

# clinical_context = "LOC and nausea."                  # optional clinical context
clinical_context = None

findings = generator.generate(vols, clinical_context)

# optional: pass findings to external frontier LLM to interpret (e.g. clinical triage)
api_key = "..." # requires API key (e.g., OpenAI) set in your environment
intepretation = interpret_findings(findings, clinical_context, api_key)

```

```console
>>> findings
Findings:
1. Acute subarachnoid hemorrhage centered in the basal cisterns with extension into
   the perimesencephalic cisterns and along the tentorial incisura. 
2. Hemorrhage and clot abut and partially efface the cerebral aqueduct with prominence 
   of the temporal horns. 
3. Mild mass effect on the midbrain.

>>> interpretation
Acuity Level: 
URGENT

Rationale: 
The pattern of hemorrhage in the basal and perimesencephalic cisterns with 
clot impinging on the cerebral aqueduct and early ventricular enlargement is
highly concerning for evolving obstructive hydrocephalus and brainstem compromise.
```