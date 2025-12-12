"""
Text Processing for NeuroVFM-LLM

This module provides utilities for text processing including:
- Text processing for visual instruction tuning
- Task classes for visual instruction tuning objectives
"""

from typing import Dict, List
import torch
import random
import json
import logging


def process_text(
    conversation: List[Dict[str, str]],
    tokenizer,
    max_seq_len: int,
    system_prompt: str,
    image_placeholder_token_id: int,
    n_images: int = 1,
    ) -> Dict:
    """
    Processes text_data into a format suitable for visual instruction tuning.
    Handles tokenization, chat templating, and label creation.
    NOTE: Assumes Qwen2/3 chat template. 
    """
    prepared_inputs = {}

    vision_placeholders_str = tokenizer.decode([image_placeholder_token_id])
    system_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"

    input_ids = []
    labels = []

    def _append_segment(text, is_target=False):
        nonlocal input_ids, labels
        token_ids = tokenizer(text, add_special_tokens=False).input_ids
        segment_labels = token_ids if is_target else [-100] * len(token_ids)
        input_ids.extend(token_ids)
        labels.extend(segment_labels)

    _append_segment(system_text)
    
    for i, turn in enumerate(conversation):
        role = 'user' if turn["role"] == 'user' else 'assistant'
        content = turn["content"]

        if role == 'user':

            if i == 0:
                img_content = f"<|vision_start|>{vision_placeholders_str}<|vision_end|>" * n_images
                content = img_content + content

            turn_text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            _append_segment(turn_text, is_target=False)
        elif role == 'assistant':
            prompt_text = f"<|im_start|>{role}\n"
            response_text = f"{content}<|im_end|>\n"
            _append_segment(prompt_text, is_target=False)
            _append_segment(response_text, is_target=True)

    # truncate and create attention mask
    final_input_ids = input_ids[:max_seq_len]
    final_labels = labels[:max_seq_len]
    attention_mask = [1] * len(final_input_ids)

    # recover the truncated text string corresponding to the input_ids
    truncated_text = tokenizer.decode(final_input_ids, skip_special_tokens=False)

    prepared_inputs.update({
        'input_ids': final_input_ids,
        'attention_mask': attention_mask,
        'labels': final_labels,
        'raw_text': truncated_text
    })
        
    return prepared_inputs