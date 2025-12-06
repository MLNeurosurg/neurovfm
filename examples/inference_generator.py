import torch
from neurovfm.pipelines import load_vlm, interpret_findings

# 1. Load VLM pipeline + preprocessor
generator, preproc = load_vlm(
    "mlinslab/neurovfm-llm",
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# 2. Preprocess a study (directory or single file)
batch = preproc.load_study("/path/to/study/", modality="mri")  # or "ct"

# clinical_context = "Left temporal lobe lesion." # Optional clinical context to inform generation
clinical_context = None

# 3. Generate findings
print("\nGenerating findings...")
report = generator(
    batch, 
    clinical_context=clinical_context
)

print("\n" + "="*40)
print("GENERATED FINDINGS")
print("="*40)
print(report)
print("="*40)

# 4. (Optional) Interpret findings using external LLM
print("\nInterpreting findings...")

# Include API key here
# Alternatively, set OPENAI_API_KEY environment variable
api_key = "..."

triage_output = interpret_findings(
    findings=report,
    clinical_context=clinical_context,
    api_key=api_key,
    system_prompt_path="neurovfm/pipelines/resources/triage_acuity_prompt.txt"
)
print("\n" + "="*40)
print("TRIAGE INTERPRETATION")
print("="*40)
print(triage_output)
print("="*40)