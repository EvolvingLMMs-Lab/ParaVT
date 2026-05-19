from lmms_engine.mapping_func import register_model

from .configuration_nanovlm import NanovlmConfig
from .modeling_nanovlm import NanovlmForConditionalGeneration

register_model(
    "nanovlm",
    NanovlmConfig,
    NanovlmForConditionalGeneration,
    model_general_type="image_text_to_text",
)

__all__ = [
    "NanovlmConfig",
    "NanovlmForConditionalGeneration",
]
