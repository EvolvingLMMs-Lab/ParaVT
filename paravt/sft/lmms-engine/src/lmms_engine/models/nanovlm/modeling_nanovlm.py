from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel

from .configuration_nanovlm import NanovlmConfig


def _build_activation(act_name: str) -> nn.Module:
    if act_name in {"gelu", "gelu_new", "gelu_fast"}:
        return nn.GELU()
    if act_name in {"silu", "swish"}:
        return nn.SiLU()
    if act_name == "relu":
        return nn.ReLU()
    act = ACT2FN.get(act_name)
    return act if isinstance(act, nn.Module) else nn.GELU()


class NanovlmForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = NanovlmConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: NanovlmConfig, **kwargs):
        super().__init__(config)
        attn_implementation = kwargs.pop("attn_implementation", None)
        kwargs.pop("torch_dtype", None)

        # transformers-style init: build modules from config only.
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config,
            attn_implementation=attn_implementation,
        )
        # SigLIP vision tower keeps its native attention path.
        self.vision_model = AutoModel.from_config(config.vision_config)

        llm_hidden_size = getattr(self.language_model.config, "hidden_size", None)
        if llm_hidden_size is None:
            raise ValueError("Unable to infer language model hidden size from config.")

        self.multi_modal_projector = self._build_projector(
            config.vision_feature_dim,
            llm_hidden_size,
            config.projector_hidden_size,
            config.projector_num_layers,
            config.projector_hidden_act,
        )

        self.post_init()

    def _build_projector(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int],
        num_layers: int,
        act_name: str,
    ) -> nn.Module:
        if num_layers <= 1:
            return nn.Linear(in_dim, out_dim)
        hidden_dim = hidden_dim or out_dim
        activation = _build_activation(act_name)
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, out_dim),
        )

    def _encode_images(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        image_outputs = self.vision_model(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        image_features = image_outputs.last_hidden_state

        if image_features.dim() == 3:
            image_features = image_features.reshape(-1, image_features.shape[-1])
        return self.multi_modal_projector(image_features)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.language_model.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None):
        return self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of=pad_to_multiple_of)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            if input_ids is None:
                raise ValueError("input_ids must be provided when pixel_values is not None.")
            if self.config.image_token_id is None:
                raise ValueError("image_token_id must be set in NanovlmConfig when using images.")

            image_embeds = self._encode_images(
                pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
            )
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            image_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        return self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        pixel_values=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_attention_mask": kwargs.get("pixel_attention_mask", None),
            "spatial_shapes": kwargs.get("spatial_shapes", None),
            **kwargs,
        }
