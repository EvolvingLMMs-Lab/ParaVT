from typing import Any, Dict, Optional, Union

from transformers import AutoConfig, PretrainedConfig


class NanovlmConfig(PretrainedConfig):
    model_type = "nanovlm"
    sub_configs = {
        "vision_config": PretrainedConfig,
        "text_config": PretrainedConfig,
    }

    def __init__(
        self,
        vision_model_name: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        vision_config: Optional[Dict[str, Any]] = None,
        text_config: Optional[Dict[str, Any]] = None,
        image_token_id: Optional[int] = None,
        projector_hidden_size: Optional[int] = None,
        projector_num_layers: int = 2,
        projector_hidden_act: str = "gelu",
        vision_feature_dim: Optional[int] = None,
        image_token_count: int = 256,
        **kwargs,
    ):
        self.text_config = self._resolve_sub_config(
            config=text_config,
            model_name=llm_model_name,
            default_model_name="Qwen/Qwen3-0.6B",
        )
        self.vision_config = self._resolve_sub_config(
            config=vision_config,
            model_name=vision_model_name,
            default_model_name="google/siglip2-base-patch16-naflex",
        )

        # Keep legacy fields for BC with old checkpoints and scripts.
        self.vision_model_name = vision_model_name
        self.llm_model_name = llm_model_name
        self.image_token_id = image_token_id
        self.projector_hidden_size = projector_hidden_size
        self.projector_num_layers = int(projector_num_layers)
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_dim = int(vision_feature_dim or getattr(self.vision_config, "hidden_size", 1152))
        self.image_token_count = int(image_token_count)
        self.vocab_size = getattr(self.text_config, "vocab_size", None)
        super().__init__(**kwargs)

    @staticmethod
    def _resolve_sub_config(
        config: Optional[Union[PretrainedConfig, Dict[str, Any]]],
        model_name: Optional[str],
        default_model_name: str,
    ) -> PretrainedConfig:
        if config is not None:
            if isinstance(config, PretrainedConfig):
                return config
            if not isinstance(config, dict):
                raise TypeError(f"Unsupported sub-config type: {type(config)}")

            model_type = config.get("model_type")
            if model_type is None:
                raise ValueError("Nested config dict must contain `model_type`.")

            config_kwargs = {k: v for k, v in config.items() if k != "model_type"}
            return AutoConfig.for_model(model_type, **config_kwargs)

        return AutoConfig.from_pretrained(model_name or default_model_name)
