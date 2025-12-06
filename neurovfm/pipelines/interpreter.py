import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from openai import OpenAI  # type: ignore[import]
except ImportError:
    OpenAI = None


API_DEFAULT_KWARGS = {
    "model": "gpt-5-2025-08-07",
    "max_output_tokens": 5120,
    "text": {"verbosity": "low"},
    "reasoning": {"effort": "medium"},
}

DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parent / "resources" / "triage_acuity_prompt.txt"
)

def interpret_findings(
    findings: str,
    clinical_context: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt_path: Optional[str] = None,
    api_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Interpret a NeuroVFM-generated findings string via an OpenAI model.

    Args:
        findings: The text of the radiological findings.
        clinical_context: Optional clinical indication text.
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
        system_prompt_path: Path to instruction prompt (default: neurovfm/pipelines/resources/triage_acuity_prompt.txt)
        api_kwargs: Overrides for the OpenAI request (merged on top of defaults).

    Returns:
        A formatted multi-line string summarizing the triage interpretation.
    """
    if OpenAI is None:
        raise ImportError(
            "The 'openai' library is required for interpret_findings. "
            "Please install it via 'pip install openai'."
        )

    prompt_path = Path(system_prompt_path) if system_prompt_path else DEFAULT_PROMPT_PATH
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"System prompt file not found at {prompt_path}."
        )

    system_prompt = prompt_path.read_text(encoding="utf-8")
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    merged_kwargs = {**API_DEFAULT_KWARGS, **(api_kwargs or {})}

    # Format input
    user_content = ""
    if clinical_context:
        user_content += f"CLINICAL INDICATION:\n{clinical_context}\n\n"
    user_content += f"FINDINGS:\n{findings}"

    # Call API
    try:
        response = client.responses.create(
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            **merged_kwargs,
        )
    except Exception as exc:
        logging.error("OpenAI API call failed: %s", exc)
        return f"Triage interpretation failed ({exc})."

    raw_text = getattr(response, "output_text", None)
    if raw_text is None:
        # Fallback: traverse output list manually
        text_segments = []
        for item in getattr(response, "output", []):
            if isinstance(item, dict):
                for block in item.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        text_segments.append(block.get("text", ""))
        raw_text = "\n".join(text_segments).strip()
    content = raw_text or ""

    # Parse JSON
    try:
        if "```" in content:
            content = content.split("```")[1].strip()
        parsed = json.loads(content)
        assessment = parsed.get("triage_assessment", "<missing>")
        level = parsed.get("triage_level", "<missing>")
        report_lines = ["Acuity Level:", f"{level.upper()}", "", "Rationale:", f"{assessment}"]
        return "\n".join(report_lines)
    except json.JSONDecodeError:
        logging.error("Failed to parse JSON from LLM response: %s", content)
        return f"Could not parse JSON from LLM response.\nRaw content:\n{content}"