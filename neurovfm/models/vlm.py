"""
Vision Language Model for 3D Medical Imaging

Implements a LLaVA-style multimodal model for radiology report generation.
Combines a pretrained vision encoder with a language model via a perceiver-based connector.

Generation follows a JSON schema via outlines/pydantic.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import outlines
import outlines.caching as cache
import torch
import torch.nn as nn
from outlines.processors.structured import JSONLogitsProcessor
from peft import LoraConfig, TaskType, get_peft_model
from pydantic import BaseModel, Field
from timm.layers.mlp import Mlp
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from neurovfm.models.perceiver import PerceiverResampler
from neurovfm.models.vit import get_vit_backbone


class ShortReport(BaseModel):
    # JSON schema for report generation
    exam_type: str = Field(..., description="The type of imaging study.")
    findings: List[str] = Field(..., description="A list of key radiological findings.")


class LanguageModel(nn.Module):
    """
    Wrapper for a pretrained LLM.
    """

    def __init__(
            self, 
            model_name_or_path: str, 
            use_gradient_checkpointing: bool = False,
            lora_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.lora_params = lora_params

        # init LLM and tokenizer
        self.llm: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side='left',
        )
        self.image_placeholder_token_id = self.tokenizer.convert_tokens_to_ids('<|image_pad|>') # [hardcoded] image padding token used by Qwen2/3 tokenizer

        if self.use_gradient_checkpointing:
            self.enable_gradient_checkpointing()

        if self.lora_params:
            self.enable_lora(self.lora_params)

        # init structured JSON generator
        # monkey-patch tokenizer attributes required for outlines structured generation compatibility
        self.tokenizer.vocabulary = self.tokenizer.get_vocab()
        self.tokenizer.special_tokens = self.tokenizer.all_special_tokens
        self.tokenizer.convert_token_to_string = lambda token: self.tokenizer.decode([self.tokenizer.vocabulary[token]])

        cache.disable_cache() # disable caching to prevent OverflowError with large vocabularies


    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        if hasattr(self.llm, 'gradient_checkpointing_enable'):
            self.llm.gradient_checkpointing_enable()


    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get token embeddings from input_ids.
        """
        embeddings = self.llm.get_input_embeddings()(input_ids)
        return embeddings


    @property
    def hidden_size(self) -> int:
        """
        Returns the hidden size of the LLM.
        """
        return self.llm.config.hidden_size

    @property
    def device(self) -> torch.device:
        return self.llm.device


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the LLM.
        """
        # ensure consistent dtypes
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)
        
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=True,
            **kwargs
        )

        return outputs


    @torch.inference_mode()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            inputs_embeds (torch.Tensor): Combined visual and prompt embeddings.
                                          Shape: (batch_size, seq_len, embed_dim)
            attention_mask (Optional[torch.Tensor]): Attention mask for inputs_embeds.
                                                     Shape: (batch_size, seq_len)
            **generate_kwargs: Additional arguments for Hugging Face `generate` method
                               (e.g., max_length, num_beams, do_sample, temperature).
        
        Returns:
            torch.Tensor: Generated token IDs.
        """
        # ensure consistent dtype
        inputs_embeds = inputs_embeds.to(torch.bfloat16)
        
        # pop custom kwarg to avoid passing it to model.generate
        schema_str = generate_kwargs.pop('generation_schema', None)
        logits_processor = None
        if schema_str:
            if schema_str == 'shortreport':
                schema = ShortReport
            else:
                raise ValueError(f"Invalid schema: {schema_str}")
            json_logits_processor = JSONLogitsProcessor(
                schema=schema,
                tokenizer=self.tokenizer,
                tensor_library_name="torch"
            )
            logits_processor = [json_logits_processor]
            # for structured generation, sampling must be disabled
            generate_kwargs['do_sample'] = False

        # create a final configuration dictionary by merging model defaults with user kwargs
        # User-provided kwargs will override the defaults.
        final_config_dict = {
            **self.llm.generation_config.to_dict(),
            **generate_kwargs
        }

        # if sampling is disabled (either explicitly or by using num_beams > 1), remove all sampling-related parameters to avoid warnings
        if not final_config_dict.get('do_sample', False):
            final_config_dict.pop('temperature', None)
            final_config_dict.pop('top_p', None)
            final_config_dict.pop('top_k', None)
            final_config_dict.pop('min_p', None)
            final_config_dict.pop('typical_p', None)

        # create the final GenerationConfig object from the clean dictionary
        generation_config = GenerationConfig.from_dict(final_config_dict)
        self.llm.generation_config = generation_config

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            logits_processor=logits_processor,
            generation_config=generation_config
        )

        return outputs


    def enable_lora(self, lora_config: Optional[Dict[str, Any]] = None):
        lora_config['task_type'] = TaskType.CAUSAL_LM
        peft_config = LoraConfig(**lora_config)
        self.llm = get_peft_model(self.llm, peft_config)


class VisionEncoder(nn.Module):
    """
    Wrapper for 3D volumetric vision transformer encoder.
    
    Wraps the pretrained VisionTransformer backbone with optional gradient
    checkpointing. Outputs per-study or per-batch embeddings.
    
    Note: This module expects pre-normalized inputs. Normalization should be
    handled by the pipeline or training system before calling this encoder.
    
    Args:
        backbone_cf (Dict): Configuration for get_vit_backbone (which, params)
        checkpoint_path (str, optional): Path to pretrained checkpoint
        use_gradient_checkpointing (bool): Enable gradient checkpointing for memory efficiency
        freeze (bool): Freeze encoder weights. Defaults to True.
    
    Example:
        >>> encoder_cf = {
        ...     'backbone_cf': {'which': 'vit_base', 'params': {...}},
        ...     'checkpoint_path': '/path/to/checkpoint.ckpt',
        ...     'freeze': True
        ... }
        >>> encoder = VisionEncoder(**encoder_cf)
        >>> embs, coords = encoder(batch)  # batch should have normalized 'img' tokens
    """
    
    def __init__(
        self,
        backbone_cf: Dict[str, Any],
        use_gradient_checkpointing: bool = False,
        freeze: bool = True,
    ):
        super().__init__()
        
        # initialize backbone
        self.encoder = get_vit_backbone(**backbone_cf)
        self.embed_dim = self.encoder.embed_dim
        
        # gradient checkpointing
        self.use_gradient_checkpointing = use_gradient_checkpointing
        if self.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # freeze encoder if specified
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.encoder, 'set_grad_checkpointing'):
            self.encoder.set_grad_checkpointing(True)
        elif hasattr(self.encoder, 'blocks'):
            for block in self.encoder.blocks:
                block.grad_checkpointing = True
    
    def forward(
        self, 
        batch: Dict[str, Any], 
        return_list: bool = True
    ) -> Union[Tuple[List[torch.Tensor], List[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through vision encoder.
        
        Note: Expects pre-normalized 'img' tokens.
        
        Args:
            batch (Dict): Batch dictionary with keys:
                - img: (total_seqlen, D) normalized image tokens
                - coords: (total_seqlen, 3) tensor of 3d coordinates
                - series_masks_indices: mask indices for foreground tokens
                - series_cu_seqlens: (n_series+1,) cumulative sequence lengths
                - series_max_len: int, maximum sequence length
                - study_cu_seqlens: (n_studies+1,) study cumulative sequence lengths
            return_list (bool): If True, return list of tensors per study.
                               If False, return single concatenated tensor.
        
        Returns:
            If return_list=True:
                Tuple[List[Tensor], List[Tensor]]: Per-study embeddings and coordinates
            If return_list=False:
                Tuple[Tensor, Tensor]: Concatenated embeddings and coordinates
        """        
        
        # forward pass through encoder
        # output shape: (total_tokens_in_batch, embed_dim)
        emb = self.encoder.forward(
            batch["img"],
            batch["coords"],
            masks=batch["series_masks_indices"],
            cu_seqlens=batch["series_cu_seqlens"],
            max_seqlen=batch["series_max_len"]
        )
        
        if return_list:
            # split embedding/coordinate tensors by study
            study_embeddings = []
            study_coords_list = []
            study_cu_seqlens = batch["study_cu_seqlens"]
            
            for study_idx in range(len(study_cu_seqlens) - 1):
                start_idx = study_cu_seqlens[study_idx]
                end_idx = study_cu_seqlens[study_idx + 1]
                
                study_emb = emb[start_idx:end_idx]
                study_embeddings.append(study_emb.to(torch.bfloat16))
                
                study_coords = batch["coords"][start_idx:end_idx]
                study_coords_list.append(study_coords)
            
            return study_embeddings, study_coords_list
        else:
            return emb.to(torch.bfloat16), batch["coords"]


class VisionConnector(nn.Module):
    """
    Connector module that projects visual tokens to the LLM embedding space using a perceiver resampler.
    
    Operates on a per-series (scan) basis, compressing each series to a fixed number of tokens before projection.
    
    Args:
        visual_embed_dim (int): Dimension of visual encoder embeddings
        llm_embed_dim (int): Dimension of LLM embeddings
        perceiver_cfg (Dict): Configuration for PerceiverResampler
        mlp_hidden_dim (int, optional): Hidden dimension for projection MLP
        mlp_drop (float): Dropout rate for MLP. Defaults to 0.0.
        mlp_act_layer (nn.Module): Activation layer for MLP. Defaults to nn.GELU.
    
    Example:
        >>> connector = VisionConnector(
        ...     visual_embed_dim=768,
        ...     llm_embed_dim=4096,
        ...     perceiver_cfg={'num_queries': 64, 'num_layers': 6, 'num_heads': 8}
        ... )
        >>> projected, lengths = connector(visual_tokens, series_cu_seqlens)
    """
    
    def __init__(
        self,
        visual_embed_dim: int,
        llm_embed_dim: int,
        perceiver_cfg: Dict[str, Any],
        mlp_hidden_dim: Optional[int] = None,
        mlp_drop: float = 0.0,
        mlp_act_layer: nn.Module = nn.GELU,
    ):
        super().__init__()
        
        # build perceiver with visual embed dim
        perceiver_cfg = dict(perceiver_cfg)
        perceiver_cfg["dim"] = visual_embed_dim
        self.perceiver = PerceiverResampler(**perceiver_cfg)
        
        # build projection mlp
        self.in_features = visual_embed_dim
        self.hidden = mlp_hidden_dim or llm_embed_dim
        self.out_features = llm_embed_dim
        self.mlp = Mlp(
            in_features=self.in_features,
            hidden_features=self.hidden,
            out_features=self.out_features,
            drop=mlp_drop,
            act_layer=mlp_act_layer,
        )
        
        self.num_queries = self.perceiver.num_queries
        self.output_is_list = True
    
    def forward(
        self, 
        visual_tokens: List[torch.Tensor], 
        serie_cu_seqlens: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[List[int]]]:
        """
        Compress and project visual tokens.
        
        Args:
            visual_tokens (List[Tensor]): List of per-study visual embeddings
                Each tensor has shape (N_tokens_for_study, D_vis)
            serie_cu_seqlens (List[Tensor]): List of per-study cumulative sequence lengths
                Each tensor has shape (N_series + 1,)
        
        Returns:
            Tuple[List[Tensor], List[List[int]]]:
                - List of projected tokens per study (N_series * num_queries, D_llm)
                - List of token counts per series for each study
        """
        projected_tokens_per_study: List[torch.Tensor] = []
        serie_lengths_per_study: List[List[int]] = []
        
        for study_tokens, cu_seqlens in zip(visual_tokens, serie_cu_seqlens):
            # study_tokens: (N_total_tokens_for_study, D_vis)
            # cu_seqlens: (N_series + 1,) tensor with cumulative lengths
            
            if study_tokens.numel() == 0:
                # create placeholder for empty studies (corrupted data)
                num_series = len(cu_seqlens) - 1
                if num_series > 0:
                    total_tokens = num_series * self.num_queries
                    placeholder_tokens = torch.zeros(
                        total_tokens, self.out_features,
                        device=self.perceiver.queries.device,
                        dtype=self.perceiver.queries.dtype
                    )
                    placeholder_lengths = [self.num_queries] * num_series
                    projected_tokens_per_study.append(placeholder_tokens)
                    serie_lengths_per_study.append(placeholder_lengths)
                else:
                    projected_tokens_per_study.append(
                        torch.empty(0, self.out_features,
                                    device=self.perceiver.queries.device,
                                    dtype=self.perceiver.queries.dtype)
                    )
                    serie_lengths_per_study.append([])
                continue
            
            compressed_per_serie = []
            num_series = len(cu_seqlens) - 1
            
            # split tokens into series based on cu_seqlens
            for i in range(num_series):
                start_idx = cu_seqlens[i]
                end_idx = cu_seqlens[i + 1]
                serie_tokens = study_tokens[start_idx:end_idx]
                
                if serie_tokens.numel() == 0:
                    # placeholder for empty series
                    placeholder = torch.zeros(
                        self.num_queries,
                        self.in_features,
                        device=self.perceiver.queries.device,
                        dtype=self.perceiver.queries.dtype,
                    )
                    compressed_per_serie.append(placeholder)
                else:
                    # perceiver expects (B, N, D), so unsqueeze
                    comp = self.perceiver(serie_tokens.unsqueeze(0)).squeeze(0)
                    compressed_per_serie.append(comp)
            
            # concatenate compressed tokens from all series
            if not compressed_per_serie:
                projected_tokens_per_study.append(
                    torch.empty(0, self.out_features,
                                device=study_tokens.device,
                                dtype=study_tokens.dtype)
                )
                serie_lengths_per_study.append([])
                continue
            
            all_serie_compressed = torch.cat(compressed_per_serie, dim=0)
            
            # project to llm dimension
            projected = self.mlp(all_serie_compressed)
            projected_tokens_per_study.append(projected)
            serie_lengths_per_study.append([self.num_queries] * num_series)
        
        return projected_tokens_per_study, serie_lengths_per_study


class VisionLanguageModel(nn.Module):
    """
    Multimodal LLM for neuroimaging report generation.
    
    Combines a pretrained vision encoder, perceiver-based connector, and a language model in a LLaVA-style architecture for visual instruction tuning.
    
    Args:
        vision_encoder_cf (Dict): Configuration for VisionEncoder
        vision_connector_cf (Dict): Configuration for VisionConnector
        language_model_cf (Dict): Configuration for LanguageModel
        use_gradient_checkpointing (bool): Enable gradient checkpointing
    
    Example:
        >>> model = NeuroLlavaModel(
        ...     vision_encoder_cf={...},
        ...     vision_connector_cf={...},
        ...     language_model_cf={...}
        ... )
        >>> outputs = model(vision_batch, input_ids, attention_mask, labels)
    """
    
    def __init__(
        self,
        vision_encoder_cf: Dict[str, Any],
        vision_connector_cf: Dict[str, Any],
        language_model_cf: Dict[str, Any],
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        
        # initialize vision encoder
        self.vision_encoder = VisionEncoder(
            backbone_cf=vision_encoder_cf,
            use_gradient_checkpointing=use_gradient_checkpointing,
            freeze=True,
        )
        self.visual_embed_dim = self.vision_encoder.embed_dim
        
        # initialize language model
        self.language_model = LanguageModel(
            model_name_or_path=language_model_cf['model_name_or_path'],
            use_gradient_checkpointing=use_gradient_checkpointing,
            lora_params=language_model_cf.get('lora_params', None), # optional LoRA parameters; if None, finetunes entire LLM
        )
        llm_dim = self.language_model.hidden_size
        
        # initialize vision connector
        cfg = vision_connector_cf.get('serie_perceiver', vision_connector_cf)
        self.vision_connector = VisionConnector(
            visual_embed_dim=self.visual_embed_dim,
            llm_embed_dim=llm_dim,
            perceiver_cfg=cfg.get('perceiver_cfg', cfg),
            mlp_hidden_dim=vision_connector_cf.get('mlp_hidden_dim'),
            mlp_drop=vision_connector_cf.get('mlp_drop', 0.0),
        )
        
    
    def forward(
        self,
        vision_batch: Optional[Dict[str, Any]],
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        generation_mode: bool = False,
        **generate_kwargs,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass for training or generation.
        
        Args:
            vision_batch (Dict): Batch from StudyPreprocessor
            input_ids (Tensor): Tokenized input ids (B, S)
            attention_mask (Tensor): Attention mask (B, S)
            labels (Tensor, optional): Labels for training (B, S)
            generation_mode (bool): If True, perform autoregressive generation
            **generate_kwargs: Additional generation arguments
        
        Returns:
            CausalLMOutputWithPast for training, or generated ids for inference
        """
        if generation_mode:
            return self.generate(
                vision_batch=vision_batch,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
        
        # 1. get projected visual embeddings
        vision_output = self._forward_vision(vision_batch)
        
        # 2. get text embeddings
        text_embeds = self.language_model.get_input_embeddings(input_ids)
        
        # 3. splice visual and text embeddings
        inputs_embeds, attention_mask, labels = self._splice_vision_and_text(
            vision_output=vision_output,
            text_embeds=text_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        # 4. forward through language model
        return self._forward_sft(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
    

    def generate(
        self,
        vision_batch: Optional[Dict[str, Any]],
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        **generate_kwargs,
    ):
        """
        Autoregressive text generation.
        
        Args:
            vision_batch (Dict): Batch from StudyPreprocessor
            input_ids (Tensor): Tokenized prompt ids
            attention_mask (Tensor): Attention mask for prompt
            **generate_kwargs: HuggingFace generate arguments
        
        Returns:
            Generated token ids or dict with hidden states if requested
        """
        generate_with_hidden_states = generate_kwargs.pop('generate_with_hidden_states', False)
        
        # 1. get projected visual embeddings
        vision_output = self._forward_vision(vision_batch)
        
        # 2. get text embeddings
        text_embeds = self.language_model.get_input_embeddings(input_ids)
        
        # 3. splice visual and text embeddings
        inputs_embeds, attention_mask, _ = self._splice_vision_and_text(
            vision_output=vision_output,
            text_embeds=text_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
        )
        
        # 4. generate
        return self._forward_generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generate_with_hidden_states=generate_with_hidden_states,
            **generate_kwargs,
        )
    

    def _splice_vision_and_text(
        self,
        vision_output,
        text_embeds,
        input_ids,
        attention_mask,
        labels=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Splice visual embeddings into text at placeholder locations.
        
        Args:
            vision_output: Output from _forward_vision
            text_embeds (Tensor): Text embeddings (B, S_text, D)
            input_ids (Tensor): Input ids (B, S_text)
            attention_mask (Tensor): Attention mask (B, S_text)
            labels (Tensor, optional): Labels (B, S_text)
        
        Returns:
            Tuple of (inputs_embeds, attention_mask, labels), all left-padded
        """
        final_embeds, final_attention_mask = [], []
        final_labels = [] if labels is not None else None

        # handle multiple placeholders for per-serie vision connectors
        projected_visual_tokens, projected_visual_serie_lengths = vision_output
        
        for i in range(text_embeds.shape[0]):
            study_visual_tokens = projected_visual_tokens[i]
            serie_lengths = projected_visual_serie_lengths[i]
            visual_token_chunks = list(torch.split(study_visual_tokens, serie_lengths, dim=0))

            placeholder_indices = (input_ids[i] == self.language_model.image_placeholder_token_id).nonzero(as_tuple=True)[0]

            if len(placeholder_indices) < len(visual_token_chunks):
                logging.warning(f"Found {len(placeholder_indices)} placeholders but {len(visual_token_chunks)} visual chunks for sample {i}. Truncating visual chunks.")
                visual_token_chunks = visual_token_chunks[:len(placeholder_indices)]
            elif len(placeholder_indices) > len(visual_token_chunks):
                raise ValueError(
                    f"Found {len(placeholder_indices)} placeholders but only "
                    f"{len(visual_token_chunks)} visual chunks for sample {i}."
                )
            
            spliced_embeds_parts, spliced_mask_parts = [], []
            spliced_labels_parts = [] if labels is not None else None
            last_text_idx = 0

            for j, placeholder_idx in enumerate(placeholder_indices):
                spliced_embeds_parts.append(text_embeds[i, last_text_idx:placeholder_idx])
                spliced_mask_parts.append(attention_mask[i, last_text_idx:placeholder_idx])
                if labels is not None:
                    spliced_labels_parts.append(labels[i, last_text_idx:placeholder_idx])

                visual_chunk = visual_token_chunks[j]
                spliced_embeds_parts.append(visual_chunk)
                spliced_mask_parts.append(torch.ones(visual_chunk.shape[0], dtype=torch.long, device=text_embeds.device))
                if labels is not None:
                    spliced_labels_parts.append(torch.full((visual_chunk.shape[0],), -100, dtype=torch.long, device=text_embeds.device))

                last_text_idx = placeholder_idx + 1

            spliced_embeds_parts.append(text_embeds[i, last_text_idx:])
            spliced_mask_parts.append(attention_mask[i, last_text_idx:])
            if labels is not None:
                spliced_labels_parts.append(labels[i, last_text_idx:])
            
            final_embeds.append(torch.cat(spliced_embeds_parts, dim=0))
            final_attention_mask.append(torch.cat(spliced_mask_parts, dim=0))
            if final_labels is not None:
                final_labels.append(torch.cat(spliced_labels_parts, dim=0))

        # left pad all sequences to the same length
        batch_size = len(final_embeds)
        max_len = max(len(s) for s in final_embeds)
        embed_dim = final_embeds[0].shape[-1]
        device, dtype = final_embeds[0].device, final_embeds[0].dtype

        inputs_embeds = torch.zeros(batch_size, max_len, embed_dim, device=device, dtype=dtype)
        attention_mask_padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(final_embeds):
            seq_len = len(seq)
            inputs_embeds[i, -seq_len:] = seq
            attention_mask_padded[i, -seq_len:] = final_attention_mask[i]

        labels_padded = None
        if final_labels is not None:
            labels_padded = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
            for i, seq in enumerate(final_labels):
                labels_padded[i, -len(seq):] = seq
        
        return inputs_embeds, attention_mask_padded, labels_padded
    

    def _forward_sft(
        self,
        inputs_embeds,
        attention_mask,
        labels,
    ):
        """Forward pass for supervised fine-tuning."""
        if labels is None:
            raise ValueError("For training, 'labels' must be provided.")
        
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
    

    def _forward_generate(
        self,
        inputs_embeds,
        attention_mask,
        generate_with_hidden_states=False,
        **generate_kwargs
    ):
        """Forward pass for generation."""
        if not generate_with_hidden_states:
            return self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generate_kwargs
            )
        else:
            # return hidden states along with generated ids
            assert inputs_embeds.shape[0] == 1, "batch size must be 1 for hidden state generation"
            
            generate_kwargs.update({
                'output_hidden_states': True,
                'return_dict_in_generate': True,
                'output_scores': True
            })
            
            outputs = self.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generate_kwargs
            )
            
            generated_ids = outputs.sequences
            sequences_scores = outputs.sequences_scores if 'sequences_scores' in outputs else None
            
            # process hidden states
            generated_token_hidden_states = []
            if 'hidden_states' in outputs:
                for t in range(len(outputs.hidden_states)):
                    step_hidden_states = outputs.hidden_states[t]
                    last_layer_hidden_state = step_hidden_states[-1]
                    token_hidden_state = last_layer_hidden_state[:, -1, :]
                    generated_token_hidden_states.append(token_hidden_state.unsqueeze(1))
            
            if generated_token_hidden_states:
                final_hidden_states = torch.cat(generated_token_hidden_states, dim=1)
            else:
                final_hidden_states = torch.empty(
                    inputs_embeds.shape[0], 0, inputs_embeds.shape[2],
                    device=inputs_embeds.device, dtype=inputs_embeds.dtype
                )
            
            # reshape outputs by prompt
            batch_size = inputs_embeds.shape[0]
            num_beams = generate_kwargs.get('num_beams', 1)
            num_generated_tokens = final_hidden_states.shape[1]
            
            reshaped_ids = generated_ids.view(batch_size, num_beams, -1)
            reshaped_hidden_states = final_hidden_states.view(
                batch_size, num_beams, num_generated_tokens, -1
            )
            reshaped_scores = sequences_scores.view(batch_size, num_beams) if sequences_scores is not None else None
            
            # extract top results and remove padding
            top_generated_ids = reshaped_ids[0, 0, :]
            top_hidden_states = reshaped_hidden_states[0, 0, :, :]
            pad_indices = (top_generated_ids == self.language_model.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            
            if len(pad_indices) > 0:
                first_pad_idx = pad_indices[0].item()
                top_generated_ids = top_generated_ids[:first_pad_idx]
                top_hidden_states = top_hidden_states[:first_pad_idx]
            
            top_score = reshaped_scores[0, 0] if reshaped_scores is not None else None
            
            return {
                "all_generated_ids": reshaped_ids,
                "all_hidden_states": reshaped_hidden_states,
                "all_scores": reshaped_scores,
                "top_generated_ids": top_generated_ids,
                "top_hidden_states": top_hidden_states,
                "top_score": top_score,
            }
    

    def _forward_vision(
        self,
        vision_batch: Optional[Dict[str, Any]],
    ) -> Union[torch.Tensor, List[torch.Tensor], Tuple[List[torch.Tensor], List[List[int]]]]:
        """
        Forward pass through vision encoder and connector.
        
        Args:
            vision_batch (Dict): Batch from StudyPreprocessor
        
        Returns:
            Projected visual tokens (format depends on connector strategy)
        """
        visual_tokens, visual_coords = self.vision_encoder(vision_batch)

        # need to create a series_cu_seqlens_list, where each tensor in the list is the cumulative lengths of the series for a given study
        # example:
        # study_cu_seqlens = [0, 1000, 2000, 3000]
        # series_cu_seqlens = [0, 250, 1000, 1250, 1500, 2000, 2500, 3000]
        # series_cu_seqlens_list = [[0, 250, 1000], [0, 250, 500, 1000], [0, 500, 1000]]
        study_cu_seqlens = vision_batch['study_cu_seqlens']       
        series_cu_seqlens_flat = vision_batch['series_cu_seqlens']
        
        batch_size = len(study_cu_seqlens) - 1
        series_cu_seqlens_list = []
        start_idx_in_series_flat = 0

        for i in range(batch_size):
            study_start_offset = study_cu_seqlens[i]
            study_end_offset = study_cu_seqlens[i+1]
            
            # find the index of the study's end offset in the flat series tensor
            end_idx_tensor = (series_cu_seqlens_flat == study_end_offset).nonzero(as_tuple=True)[0]
            if end_idx_tensor.numel() == 0:
                raise ValueError(f"Study boundary {study_end_offset} not found in series_cu_seqlens.")
            if len(end_idx_tensor) > 1:
                raise ValueError(f"Multiple indices found for study boundary {study_end_offset} in series_cu_seqlens: {end_idx_tensor}\nstudy_cu_seqlens: {study_cu_seqlens}\nseries_cu_seqlens_flat: {series_cu_seqlens_flat}")
            end_idx_in_series_flat = end_idx_tensor.item()

            # slice the cumulative lengths for the current study
            study_series_cu_seqlens = series_cu_seqlens_flat[start_idx_in_series_flat : end_idx_in_series_flat + 1]
            
            # make the lengths relative to the start of the study
            study_series_cu_seqlens_relative = study_series_cu_seqlens - study_start_offset
            series_cu_seqlens_list.append(study_series_cu_seqlens_relative)

            # the start for the next study is the end of the current one
            start_idx_in_series_flat = end_idx_in_series_flat

        return self.vision_connector(visual_tokens, series_cu_seqlens_list)

    
    def _prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask,
        labels,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        Extract prompt from tokenized conversation for generation.
        
        Args:
            input_ids (Tensor): Full tokenized conversation (B, S)
            attention_mask (Tensor): Attention mask (B, S)
            labels (Tensor): Labels with -100 for prompt tokens (B, S)
        
        Returns:
            Tuple of (input_ids, attention_mask, labels, decoded_texts)
        """
        # extract prompt parts from each sample in the batch
        # since data is left-padded, we can directly slice from start to first non -100 label
        gen_input_ids_list = []
        max_prompt_len = 0
        device = input_ids.device
        
        for i in range(len(input_ids)):
            # find the start of the response (first non -100 label)
            labels_i = labels[i]
            first_target_idx_tensor = (labels_i != -100).nonzero(as_tuple=True)[0]
            
            if len(first_target_idx_tensor) > 0:
                first_target_idx = first_target_idx_tensor[0]
                # the prompt is everything up to the start of the response
                prompt_ids = input_ids[i][:first_target_idx]
            else:
                # if no target, the whole sequence is the prompt (can happen with truncation)
                prompt_ids = input_ids[i]

            gen_input_ids_list.append(prompt_ids)
            if len(prompt_ids) > max_prompt_len:
                max_prompt_len = len(prompt_ids)

        # left pad all input_ids to max_prompt_len
        padded_gen_input_ids_list = []
        for prompt_ids in gen_input_ids_list:
            pad_len = max_prompt_len - len(prompt_ids)
            padded = torch.cat([
                torch.full((pad_len,), self.language_model.tokenizer.pad_token_id, dtype=torch.long, device=device),
                prompt_ids
            ])
            padded_gen_input_ids_list.append(padded)

        # decode back to raw text (for debugging)
        generation_texts = [self.language_model.tokenizer.decode(padded, skip_special_tokens=True).strip() for padded in padded_gen_input_ids_list]

        # return the original batch with the updated values
        input_ids_new = torch.stack(padded_gen_input_ids_list)
        attention_mask_new = torch.ones_like(input_ids_new).to(device)
        labels_new = None
        
        return input_ids_new, attention_mask_new, labels_new, generation_texts
    

    def set_training_stage(self, stage: str):
        """
        Configure trainable components for different training stages.
        
        Args:
            stage (str): One of 'pretrain' or 'finetune'
                - pretrain: Train connector only, freeze encoder and LLM
                - finetune: Train connector and LLM (with optional LoRA)
        """
        # vision encoder is always frozen
        if self.vision_encoder is not None:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        if stage == "pretrain":
            # stage 1: train projection components only
            for param in self.vision_connector.parameters():
                param.requires_grad = True
            
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        elif stage == "finetune":
            # stage 2: train connector + language model
            for param in self.vision_connector.parameters():
                param.requires_grad = True
            
            if self.language_model.lora_params:
                # lora is configured in language model __init__
                pass
            else:
                for param in self.language_model.parameters():
                    param.requires_grad = True
        
        else:
            raise ValueError(f"Unknown training stage: {stage}")
    

    def log_trainable_parameters(self):
        """Log parameter counts by component."""
        total_params = 0
        trainable_params = 0
        
        component_params = {}
        if self.language_model is not None:
            lm_params = sum(p.numel() for p in self.language_model.parameters())
            component_params['language_model'] = lm_params
        if self.vision_encoder is not None:
            ve_params = sum(p.numel() for p in self.vision_encoder.parameters())
            component_params['vision_encoder'] = ve_params
        if self.vision_connector is not None:
            vc_params = sum(p.numel() for p in self.vision_connector.parameters())
            component_params['vision_connector'] = vc_params
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logging.info("-" * 60)
        logging.info("NeuroLlavaModel Parameter Summary:")
        logging.info(f"  Total parameters: {total_params:,}")
        for component, count in component_params.items():
            logging.info(f"    - {component}: {count:,}")
        logging.info(f"  Trainable parameters: {trainable_params:,}")
        logging.info(f"  Frozen parameters: {total_params - trainable_params:,}")
        logging.info(f"  Trainable percentage: {100 * trainable_params / total_params:.1f}%")
        logging.info("-" * 60)
