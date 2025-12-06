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


class BaseTask:
    """Base class for all tasks."""
    def __init__(self, study_id: str, serie_names: List[str]):
        self.study_id = study_id
        self.serie_names = serie_names

    def get_conversation(self) -> List[Dict[str, str]]:
        # generates a conversation list for the task
        # each element is a dict with "role" ('user' or 'assistant') and "content"

        raise NotImplementedError("Subclasses must implement get_conversation.")

    def get_metadata(self) -> Dict[str, str]:
        # returns a dict of metadata associated with the task instance.

        return {'task_type': self.__class__.__name__}


class AlignmentTask(BaseTask):
    """
    Task for Stage 1 alignment. 
    Answer: str, all captions appended, separated by spaces, random shuffle
    Prompt: empty
    """
    def __init__(self, study_id: str, serie_names: List[str], caption: List[str], examtype: str):
        super().__init__(study_id, serie_names)
        self.caption = caption
        self.examtype = examtype

    def get_conversation(self) -> List[Dict[str, str]]:

        # create a single string with all the captions appended, separated by spaces, random shuffle
        shuffled_caption = random.sample(self.caption, len(self.caption))
        shuffled_caption = [self.examtype] + shuffled_caption
        shuffled_caption = [caption.rstrip('.') for caption in shuffled_caption]
        caption_str = ". ".join(shuffled_caption)
            
        return [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": caption_str}
        ]

    def get_metadata(self) -> Dict[str, str]:
        metadata = super().get_metadata()
        metadata['caption'] = self.caption
        metadata['examtype'] = self.examtype
        return metadata
        

class ShortreportTask(BaseTask):
    """
    Task for Stage 2 fine-tuning. 
    Answer: JSON with: 
        - 'exam_type': str, the type of medical exam
        - 'findings': list of strings, each representing a single finding (from shortreports)
    Prompt: +/- clinical indication
    """
    def __init__(self, study_id: str, serie_names: List[str], findings: List[str], examtype: str, indication: str, indication_dropout_prob: float = 0.2, is_train: bool = True, include_indication: bool = True):
        super().__init__(study_id, serie_names)

        self.findings = findings
        self.examtype = examtype
        self.indication = indication
        self.indication_dropout_prob = indication_dropout_prob
        self.is_train = is_train
        self.include_indication = include_indication
        
        self.prompt_templates = [
            "Generate a concise report of the key positive findings for this study.",
            "Provide a list of the most important positive observations.",
            "What are the principal positive findings in this study? Please list them.",
            "Report the significant positive findings of this exam as a list.",
            "Create a list detailing the salient radiological findings.",
            "Review this neuroimaging study and produce a list of key positive observations.",
            "Please create a list of the primary radiological findings.",
            "Analyze this study and provide a list of key positive findings.",
        ]

    def get_conversation(self) -> List[Dict[str, str]]:

        if self.is_train:
            prompt = random.choice(self.prompt_templates)
            if self.include_indication and random.random() > self.indication_dropout_prob:
                prompt += f"\nConsider the patient's clinical indication in your analysis: {self.indication}"
        else:
            prompt = self.prompt_templates[0]
            if self.include_indication:
                prompt += f"\nConsider the patient's clinical indication in your analysis: {self.indication}"

        prompt += "\nFormat your response as JSON with the following keys: 'exam_type' and 'findings'."

        answer = json.dumps({"exam_type": self.examtype, "findings": self.findings})

        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer}
        ]

    def get_metadata(self) -> Dict[str, str]:
        metadata = super().get_metadata()
        metadata['findings'] = self.findings
        metadata['examtype'] = self.examtype
        metadata['indication'] = self.indication
        metadata['indication_dropout_prob'] = self.indication_dropout_prob
        metadata['is_train'] = self.is_train
        return metadata


def prepare_task_data(
    study_to_series_dict: Dict[str, List[str]],
    data_sources: Dict[str, Dict],
    task_name: str,
    is_train: bool,
    indication_dropout_prob: float = 0.2,
    include_indication: bool = True,
    ) -> Dict[str, BaseTask]:
    """
    Factory function to prepare a dictionary of task objects for a given dataset split.
    Filters out studies that do not have the required data for the specified task.
    """
    tasks = {}
    logging.info(f"Preparing data for task: '{task_name}'...")
    study_ids = list(study_to_series_dict.keys())

    if task_name == "alignment":
        assert "shortreport" in data_sources, "shortreport data source is required for alignment task"
        assert "examtype" in data_sources, "examtype data source is required for alignment task"
        caption_data = data_sources['shortreport']
        examtype_data = data_sources['examtype']

        for study_id in study_ids:
            if study_id in caption_data and study_id in examtype_data:
                tasks[study_id] = AlignmentTask(
                    study_id=study_id,
                    serie_names=study_to_series_dict[study_id],
                    caption=caption_data[study_id],
                    examtype=examtype_data[study_id]
                )
    
    elif task_name == "shortreport":
        assert "shortreport" in data_sources, "shortreport data source is required for shortreport task"
        assert "examtype" in data_sources, "examtype data source is required for shortreport task"
        assert "indication" in data_sources, "indication data source is required for shortreport task"

        shortreport_data = data_sources['shortreport']
        examtype_data = data_sources['examtype']
        indication_data = data_sources['indication']

        for study_id in study_ids:
            if all(study_id in d for d in [shortreport_data, indication_data, examtype_data]):
                tasks[study_id] = ShortreportTask(
                    study_id=study_id,
                    serie_names=study_to_series_dict[study_id],
                    findings=shortreport_data[study_id],
                    examtype=examtype_data[study_id],
                    indication=indication_data[study_id],
                    indication_dropout_prob=indication_dropout_prob,
                    is_train=is_train,
                    include_indication=include_indication,
                )
    else:
        raise ValueError(f"Unknown task_name: '{task_name}'")

    logging.info(f"Successfully created {len(tasks)} task objects for {len(study_ids)} requested studies.")
    return tasks
