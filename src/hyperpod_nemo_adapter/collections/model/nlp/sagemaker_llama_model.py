# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import transformers
from omegaconf import OmegaConf
from packaging import version as pversion
from peft import LoraConfig
from transformers import LlamaConfig

from hyperpod_nemo_adapter.collections.model import SageMakerNLPBaseModel
from hyperpod_nemo_adapter.constants import CONFIG_MAPPING_HF_TO_RECIPE_ALIASES
from hyperpod_nemo_adapter.utils.config_utils import get_hf_config_from_name_or_path
from hyperpod_nemo_adapter.utils.general_utils import can_use_multimodal
from hyperpod_nemo_adapter.utils.log_utils import Logger

TF_VERSION = pversion.parse(transformers.__version__)

if TF_VERSION >= pversion.parse("4.51.1"):
    from transformers import Llama4Config, Llama4ForConditionalGeneration

if can_use_multimodal():
    from transformers import MllamaConfig

_logger = Logger().get_logger()


class SageMakerLlamaModel(SageMakerNLPBaseModel):
    """
    Lightning Model class for Llama
    """

    predefined_model = True

    def set_config_mapping_hf_to_recipe_aliases(self):
        config_map = CONFIG_MAPPING_HF_TO_RECIPE_ALIASES
        if OmegaConf.select(self._cfg, "rope_scaling.rope_type") == "llama3":
            if pversion.parse(transformers.__version__) < pversion.parse("4.44.2"):
                _logger.warning(
                    f"Rope scaling type 'llama3' is only supported for transformers >= 4.44.2, the current version is {pversion.parse(transformers.__version__)}"
                )
            else:
                config_map["rope_scaling"] = ["rope_scaling"]
        self._config_mapping_hf_to_recipe_aliases = config_map

    def get_model_config(self):
        """
        Get model config for Llama
        """
        configurable_dict = self._get_model_configurable_dict()
        multi_modal_enabled = self._cfg.get("multi_modal", None)
        if self._cfg.get("hf_model_name_or_path", None) is not None:
            model_config = get_hf_config_from_name_or_path(self._cfg)
            assert isinstance(model_config, LlamaConfig) or (
                multi_modal_enabled and isinstance(model_config, MllamaConfig)
            ), f"model_type is set to llama but hf_model_name_or_path is not the same model, getting {type(model_config)}"
            # Update the config based on user's input
            model_config.update(configurable_dict)
        else:
            model_config = LlamaConfig(
                **configurable_dict,
                hidden_act="silu",
                use_cache=False,
            )
        return model_config


class SageMakerLlama4Model(SageMakerNLPBaseModel):
    """
    Lightning Model class for Llama
    """

    predefined_model = True

    def get_model_config(self):
        """
        Get model config for Llama
        """
        configurable_dict = self._get_model_configurable_dict()
        assert self._cfg.get("hf_model_name_or_path", None) is not None
        model_config = get_hf_config_from_name_or_path(self._cfg)
        assert isinstance(model_config, Llama4Config)
        # Update the config based on user's input
        for k, v in configurable_dict.items():
            if hasattr(model_config.text_config, k):
                setattr(model_config.text_config, k, v)
        return model_config

    def _build_model_from_pretrain(self, model_cfg, torch_dtype=None, quantization_config=None):
        path = self._cfg.hf_model_name_or_path
        _logger.info("Loading pretrained weights from %s.", path)
        use_flash_attn = self._cfg.use_flash_attention
        attn = "flash_attention_2"
        # TODO add support later for flash att
        # ValueError: MllamaForCausalLM does not support Flash Attention 2.0 yet
        access_token = self._cfg.get("hf_access_token", None)
        if TF_VERSION < pversion.parse("4.37.1") or not use_flash_attn:
            return Llama4ForConditionalGeneration.from_pretrained(
                path,
                config=model_cfg,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                token=access_token,
            )
        return Llama4ForConditionalGeneration.from_pretrained(
            path,
            attn_implementation=attn,
            config=model_cfg,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            token=access_token,
        )

    def get_lora_config(self):
        lora_config = LoraConfig(
            target_modules="language_model.model.layers.*self_attn.(q_proj|k_proj|v_proj|o_proj)",
            # Alpha parameter for LoRA scaling
            lora_alpha=self._cfg.peft.alpha,
            # Dropout probability for LoRA layers
            lora_dropout=self._cfg.peft.dropout,
            # LoRA attention dimension
            r=self._cfg.peft.rank,
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode=False,
        )
        return lora_config
