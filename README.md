# NeuroVFM

## Health system learning achieves generalist neuroimaging models

[Preprint](https://neurovfm.mlins.org) / [Interactive Demo](https://neurovfm.mlins.org) / [Models](https://huggingface.co/collections/mlinslab/neurovfm) / [MLiNS Lab](https://mlins.org)

**NeuroVFM** is a health system–scale, volumetric foundation model for multimodal neuroimaging, trained with self-supervision on **5.24M** MRI/CT volumes (**567k** studies) spanning **20+ years** of routine clinical care at Michigan Medicine. 

The NeuroVFM stack includes:

- **3D ViT encoder**, general-purpose representations for *any* clinical neuroimage (T1, T2, FLAIR, DWI, CT, etc.)
- **Study-level diagnostic heads**, covering **74 MRI**/**82 CT** expert-defined diagnoses for *any* neuroimaging study
- **Findings LLM**, generates preliminary findings given *any* neuroimaging study plus clinical context
- **Reasoning API**, pass outputs to a frontier reasoning model for higher-level tasks (e.g., triage)

> **Research use only.** Not a medical device. Do not use for clinical decision-making.

## 🔎 TL;DR (what NeuroVFM gives you)

```python
from neurovfm import load_encoder, load_diagnostic_head, load_llm, interpret_output
# requires an API key (e.g., OpenAI) set in your environment for external LLM calls

encoder, preproc = load_encoder("neurovfm-encoder")
dx_head = load_diagnostic_head("neurovfm-dx-ct")
llm = load_llm("neurovfm-llm")

vols = preproc.load_study("/path/to/study/")             # study directory with 1+ DICOM/NIfTI files   
embs = encoder.embed(vols)                               # series-wise embeddings   

dx = dx_head.predict_proba(embs, top_k=3)     
report = llm.generate_findings(
    embs,
    clinical_context="32M with sudden, severe 'worst headache of life'.",
)
interpretation = interpret_output(                 
    report=report,
    diagnoses=dx,
    model="gpt-5-thinking",
    action="triage",                              
)
```

```console
>>> dx
1. aneurysmal_subarachnoid_hemorrhage       p=0.96
2. mass_effect                              p=0.91
3. obstructive_hydrocephalus                p=0.74

>>> report
Acute subarachnoid hemorrhage centered in the basal cisterns with extension into
the perimesencephalic cisterns and along the tentorial incisura. Hemorrhage and
clot abut and partially efface the cerebral aqueduct with prominence of the
temporal horns. Mild mass effect on the midbrain.

>>> interpretation
Triage: URGENT
Rationale: The pattern of hemorrhage in the basal and perimesencephalic cisterns
with clot impinging on the cerebral aqueduct and early ventricular enlargement is
highly concerning for evolving obstructive hydrocephalus and brainstem compromise.
```